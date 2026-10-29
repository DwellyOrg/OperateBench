"""Selected-six actual SDK wires; synthetic reference decisions, never model evidence."""

import json

import httpx
import pytest

from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import SELECTED_SIX, SELECTED_SIX_PROFILE


class SelectedMock(httpx.MockTransport):
    def __init__(self, provider, model, flow, fault=None):
        self.provider, self.model, self.fault = provider, model, fault
        self.native = SDKMockTransport(
            "openai"
            if provider == "xai"
            else "mistral"
            if provider == "river_http"
            else provider,
            flow,
            model=model,
        )
        self.calls = []
        super().__init__(self.handle)

    def handle(self, request):
        body = json.loads(request.content)
        self.calls.append(body)
        normalized = dict(body)
        if self.provider in ("openai", "xai") and isinstance(body["input"], list):
            last = body["input"][-1]
            normalized["input"] = last.get("output", last.get("content"))
        elif self.provider == "anthropic" and isinstance(
            body["messages"][-1]["content"], list
        ):
            normalized["messages"] = [
                {
                    "role": "user",
                    "content": body["messages"][-1]["content"][-1]["content"],
                }
            ]
        raw = self.native.handle(httpx.Request("POST", str(request.url), json=normalized))
        result = json.loads(raw.content)
        if self.provider in ("openai", "xai"):
            result["usage"] = {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            }
            result["output"].insert(
                0,
                {
                    "type": "reasoning",
                    "id": "rs_offline",
                    "summary": [],
                    "encrypted_content": "synthetic-encrypted",
                },
            )
        elif self.provider == "anthropic":
            result["usage"].update(
                input_tokens=11,
                output_tokens=7,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
            )
        elif self.provider == "river_http":
            result["usage"] = {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
            result["choices"][0]["message"]["reasoning_content"] = "synthetic reasoning"
        if self.fault in ("args", "badargs"):
            args = (
                "{broken"
                if self.fault == "badargs"
                else json.dumps(
                    {
                        "action_type": "unknown",
                        "payload": {
                            "null": None,
                            "nested": [True, 3, {"s": '{"json":1}'}],
                            "literal": "null",
                        },
                    }
                )
            )
            if self.provider in ("openai", "xai"):
                result["output"][-1].update(name="act", arguments=args)
            elif self.provider == "anthropic":
                result["content"][-1].update(
                    name="act",
                    input=json.loads(args)
                    if self.fault == "args"
                    else {"action_type": None},
                )
            else:
                result["choices"][0]["message"]["tool_calls"][0]["function"].update(
                    name="act", arguments=args
                )
        if self.provider == "river_http" and self.fault == "reasoning_type":
            result["choices"][0]["message"]["reasoning_content"] = 123
        if self.provider == "river_http" and self.fault == "unknown_usage":
            result["usage"]["extra_billable_tokens"] = 99
        if self.fault == "usage":
            result.pop("usage")
        if self.fault == "model":
            result["model"] = "wrong"
        if self.fault == "429":
            return httpx.Response(
                429,
                json={"error": {"message": "synthetic rate limit"}},
                headers={"Retry-After": "1"},
            )
        if self.fault == "json":
            return httpx.Response(200, content=b"{broken")
        return httpx.Response(
            200,
            content=json.dumps(result).encode(),
            headers={"content-type": "application/json"},
        )


class Network:
    def __init__(self, fault=None):
        self.transports = []
        self.fault = fault

    def http(self, slot):
        mock = SelectedMock(slot["provider"], slot["model"], slot["flow"], self.fault)
        self.transports.append(mock)
        return mock


def fixture(tmp_path, provider, model, fault=None):
    from tools.three_flow_admission import VARIABLES
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_campaign import assignments
    from tools.three_flow_live_factory import live_factory
    from tools.three_flow_profiles import ENDPOINTS
    from tools.three_flow_runtime import Trial, provider_identity

    policy = {
        "mode": "three-flow-optional-limits-v1",
        "owner_accepted": True,
        "owner_evidence_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "historical_liability_manifest_sha256": "c" * 64,
    }
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="selected-test",
        cap=None,
        limit_policy=policy,
        roster_profile=SELECTED_SIX_PROFILE,
        create=True,
    )
    slot = next(
        s
        for s in assignments("selected-test", SELECTED_SIX_PROFILE)
        if s["model"] == model and s["flow"] == "commerce"
    )
    p = {
        "endpoint": ENDPOINTS[provider],
        "context_max": 1050000,
        "input_max": None,
        "output_max": 128000 if provider in ("openai", "anthropic") else None,
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "network_timeout_seconds": None,
        "bounds_policy": {"mode": "diagnostic"},
    }
    if provider == "river_http":
        p["owner_output_cap_tokens"] = 30000
    admission = {
        "profiles": {model: p},
        "mode_profile": "off-or-minimum-v1",
        "transport_policy": "paced-safe-errors-v1",
        "limit_policy": policy,
    }
    # Mock wires have no provider quota; production pacer stays unchanged.
    import tools.three_flow_http as http_module
    from tools.three_flow_errors import ProviderPacer

    original_pacer = http_module.MISTRAL_PACER
    http_module.MISTRAL_PACER = ProviderPacer(
        clock=lambda: 0.0, sleep=lambda _delay: None
    )
    network = Network(fault)
    t = live_factory(
        admission, {VARIABLES[provider]: "synthetic-key-not-secret"}, network=network
    )(slot, budget)
    http_module.MISTRAL_PACER = original_pacer
    trial = Trial(
        "selected-test",
        slot["trial_id"],
        slot["flow"],
        slot["scenario"],
        model,
        t.max_output_tokens,
        provider_identity(t),
    )
    return budget, t, trial, network


@pytest.mark.parametrize("provider,model", SELECTED_SIX)
def test_actual_sdk_multiturn(tmp_path, provider, model):
    from tools.three_flow_runtime import replay_trial, run_trial

    budget, t, trial, network = fixture(tmp_path, provider, model)
    try:
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(network.transports[0].calls) > 1
        assert all(r["state"] == "settled" for r in budget.records.values())
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        calls = network.transports[0].calls
        if provider in ("openai", "xai"):
            assert all(x["strict"] is False for x in calls[0]["tools"])
            assert all(isinstance(c["input"], str) for c in calls)
            assert (
                "synthetic-encrypted"
                in record["provider_turns"][0]["wire"]["response_utf8"]
            )
        else:
            assert all(
                len(c["messages"]) == (1 if provider == "anthropic" else 2) for c in calls
            )
        assert t.settings["no_native_continuation"] is True
        if provider == "anthropic":
            assert all(c["tool_choice"] == {"type": "any"} for c in calls)
        if provider == "river_http":
            assert all(x["max_tokens"] <= 30000 for x in calls)
            assert (
                "synthetic reasoning"
                in record["provider_turns"][0]["wire"]["response_utf8"]
            )
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize("provider,model", SELECTED_SIX)
@pytest.mark.parametrize("fault", ["usage", "model", "429", "json"])
def test_fault_retains_unknown(tmp_path, provider, model, fault):
    from operatebench.agents.transport import ProviderFailure
    from tools.three_flow_runtime import run_trial

    budget, t, trial, network = fixture(tmp_path, provider, model, fault)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, t, output=tmp_path / "trial")
        assert len(network.transports[0].calls) == 1
        assert all(r["state"] == "unknown" for r in budget.records.values())
    finally:
        t.close()
        budget.close()


def test_exact_census():
    from tools.three_flow_campaign import CONFIGURATIONS, assignments

    slots = assignments("census", SELECTED_SIX_PROFILE)
    assert len(slots) == 264
    for _p, m in SELECTED_SIX:
        assert {(s["flow"], s["scenario"]) for s in slots if s["model"] == m} == set(
            CONFIGURATIONS
        )


@pytest.mark.parametrize("provider,model", SELECTED_SIX)
def test_predispatch_context_exhaustion(tmp_path, provider, model):
    from tools.three_flow_runtime import run_trial

    budget, t, trial, network = fixture(tmp_path, provider, model)
    t.guard.profile["context_max"] = 1
    try:
        from operatebench.agents.transport import ProviderFailure

        with pytest.raises(ProviderFailure):
            run_trial(trial, t, output=tmp_path / "trial")
        assert not network.transports[0].calls
        assert not budget.records
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize("provider,model", SELECTED_SIX)
@pytest.mark.parametrize("fault", ["args", "badargs"])
def test_arguments_preserved_or_scored_parse_fault(tmp_path, provider, model, fault):
    from tools.three_flow_runtime import run_trial

    budget, t, trial, _network = fixture(tmp_path, provider, model, fault)
    responses = []
    dispatch = t._dispatch

    def capture(payload):
        usage, response = dispatch(payload)
        responses.append(response)
        return usage, response

    t._dispatch = capture
    try:
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert not record["evaluation"]["reliable"]
        assert budget.totals()["unknown"] == 0
        if fault == "args":
            payload = responses[0].tool_calls[0].arguments["payload"]
            assert payload["null"] is None
            assert payload["nested"][:2] == [True, 3]
            assert isinstance(payload["nested"][2]["s"], str)
            assert payload["literal"] == "null"
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "reasoning",
        "usage",
        "nested_usage",
        "content",
        "call_id",
        "function",
        "message",
        "choice",
        "tool_container",
        "typed_disagrees",
    ],
)
def test_parent_river_witnesses(mutation):
    from openai.types.chat import ChatCompletion

    from operatebench.providers.faults import AdapterProviderError
    from operatebench.providers.wire import WireResponse
    from tools.three_flow_selected import river_response

    body = {
        "id": "chat_offline",
        "object": "chat.completion",
        "created": 1,
        "model": "GLM-5.3",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "reasoning",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "act",
                                "arguments": json.dumps(
                                    {"payload": {"nullable": None, "s": '{"x":1}'}}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    # First exercise the valid raw body and actual SDK type.
    valid = WireResponse(
        json.dumps(body),
        lambda: ChatCompletion.model_validate(body),
        kind="River witness",
    )
    assert river_response(valid, "GLM-5.3")[0].input_tokens == 11
    message = body["choices"][0]["message"]
    typed = ChatCompletion.model_validate(body)
    if mutation == "reasoning":
        message["reasoning_content"] = 123
    elif mutation == "usage":
        body["usage"]["extra_billable_tokens"] = 99
    elif mutation == "nested_usage":
        body["usage"]["prompt_tokens_details"]["unexplained_tokens"] = 1
    elif mutation == "content":
        message["content"] = {"bad": 1}
    elif mutation == "call_id":
        message["tool_calls"][0]["id"] = False
    elif mutation == "function":
        message["tool_calls"][0]["function"] = []
    elif mutation == "message":
        body["choices"][0]["message"] = []
    elif mutation == "choice":
        body["choices"][0] = []
    elif mutation == "tool_container":
        message["tool_calls"] = {}
    else:
        typed.usage.prompt_tokens = 12
    wire = WireResponse(json.dumps(body), lambda: typed, kind="River invalid witness")
    with pytest.raises(AdapterProviderError):
        river_response(wire, "GLM-5.3")


@pytest.mark.parametrize("fault", ["reasoning_type", "unknown_usage"])
def test_parent_witness_actual_send_retains_partial_unknown(tmp_path, fault):
    from operatebench.agents.transport import ProviderFailure
    from tools.three_flow_runtime import run_trial

    budget, transport, trial, network = fixture(tmp_path, "river_http", "GLM-5.3", fault)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(network.transports[0].calls) == 1
        assert all(row["state"] == "unknown" for row in budget.records.values())
        assert list((tmp_path / "trial").iterdir())
    finally:
        transport.close()
        budget.close()
