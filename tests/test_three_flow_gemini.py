"""Native Developer API evidence; every HTTP operation is an offline mock."""

import gzip
import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_gemini import MODELS, GeminiCampaignTransport, GeminiMockTransport
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def fixture(tmp_path, model, handler=None, *, flow="commerce", scenario="UK_NORMAL"):
    from tools.three_flow_gemini import ROSTER_PROFILE, offline_limit_policy

    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=None,
        limit_policy=offline_limit_policy(),
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    profile = MODELS[model]
    guard = SharedGuard(
        budget=budget,
        cell="trial",
        policy=LifecyclePricingPolicy(
            "offline-standard-estimate",
            Decimal(profile.input_rate),
            Decimal(profile.output_rate),
        ),
        max_output_tokens=65536,
    )
    mock = GeminiMockTransport(model, flow)
    transport = GeminiCampaignTransport(
        model=model,
        inner=mock if handler is None else httpx.MockTransport(handler(mock)),
        guard=guard,
        max_output_tokens=65536,
    )
    trial = Trial(
        "offline",
        "trial",
        flow,
        scenario,
        model,
        65536,
        provider_identity(transport),
    )
    return budget, mock, transport, trial


@pytest.mark.parametrize("model", ("gemini-4-argon", "gemini-3.8-flash"))
def test_native_roundtrip(tmp_path, model):
    budget, mock, transport, trial = fixture(tmp_path, model)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert mock.calls
        first = mock.calls[0]
        assert first["generationConfig"] == {
            "maxOutputTokens": 65536,
            "thinkingConfig": {"thinkingLevel": "low"},
        }
        assert len(first["tools"][0]["functionDeclarations"]) == 6
        assert all(t["output_tokens"] == 3 for t in record["provider_turns"])
        assert all(t["thinking_tokens"] == 2 for t in record["provider_turns"])
        assert (
            mock.calls[1]["contents"][1]["parts"][0]["thoughtSignature"]
            == "offline-signature"
        )
        assert "functionResponse" in mock.calls[1]["contents"][2]["parts"][0]
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        wire = record["provider_turns"][0]["wire"]
        assert json.loads(wire["request_utf8"]) == first
        assert "key" not in json.loads(wire["request_utf8"])
    finally:
        transport.close()
        budget.close()


def response_body(model="gemini-4-argon", args=None):
    return {
        "modelVersion": model,
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "act",
                                "args": args
                                if args is not None
                                else {
                                    "action_type": "example",
                                    "payload": {
                                        "text": '{"x":1}',
                                        "nested": [None, {"key": "value"}],
                                    },
                                },
                            },
                            "thoughtSignature": "signed",
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 7,
            "totalTokenCount": 20,
            "cachedContentTokenCount": 2,
        },
    }


def test_live_factory_requires_explicit_admission_without_network(tmp_path):
    budget, mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    try:
        with pytest.raises(ValueError, match="admission"):
            GeminiCampaignTransport.live(
                model=trial.model,
                api_key="synthetic-key-not-real",
                guard=offline.guard,
                max_output_tokens=65536,
                inner=httpx.MockTransport(mock.handle),
            )
        assert not mock.calls
        assert not offline.wire.captures
        assert budget.records == {}
    finally:
        offline.close()
        budget.close()


def test_live_default_direct_transport_is_explicit_and_unproxied(tmp_path, monkeypatch):
    from dataclasses import replace

    import tools.three_flow_gemini as gemini

    budget, mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    options = []

    def direct(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(mock.handle)

    monkeypatch.setattr(gemini.httpx, "HTTPTransport", direct)
    try:
        live = GeminiCampaignTransport.live(
            model=trial.model,
            api_key="synthetic-key-not-real",
            guard=offline.guard,
            max_output_tokens=65536,
            admitted=True,
        )
        try:
            assert options == [{"trust_env": False, "retries": 0}]
            assert live.http.follow_redirects is False
            assert live.http._trust_env is False
            assert live.settings["evidence_origin"] == "PROVIDER_CANDIDATE"
            assert "forbidden" not in live.settings["quota_admission"]
            assert live.settings["credential_source"] == "explicit header only"
            assert mock.calls == []
            direct_trial = replace(trial, provider_binding=provider_identity(live))
            record = run_trial(direct_trial, live, output=tmp_path / "direct-synthetic")
            assert record["evidence_origin"] == "PROVIDER_CANDIDATE"
            assert (
                replay_trial(
                    direct_trial, record, expected_record_digest=record["record_digest"]
                )["provider_calls"]
                == 0
            )
        finally:
            live.close()
    finally:
        offline.close()
        budget.close()


def test_live_mock_gzip_exact_history_usage_and_replay(tmp_path, capsys):
    key = "synthetic-key-not-real"
    budget, mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    requests = []

    def handle(request):
        assert str(request.url) == MODELS[trial.model].endpoint
        assert request.method == "POST"
        assert request.headers["x-goog-api-key"] == key
        assert "key" not in str(request.url).lower()
        assert key not in request.content.decode()
        assert request.extensions["timeout"] == {
            "connect": 10,
            "read": None,
            "write": 10,
            "pool": 10,
        }
        requests.append(json.loads(request.content))
        body = mock.handle(request).content
        return httpx.Response(
            200, content=gzip.compress(body), headers={"Content-Encoding": "gzip"}
        )

    live = GeminiCampaignTransport.live(
        model=trial.model,
        api_key=key,
        guard=offline.guard,
        max_output_tokens=65536,
        admitted=True,
        inner=httpx.MockTransport(handle),
    )
    try:
        record = run_trial(trial, live, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(requests) == len(mock.calls) == len(record["provider_turns"])
        assert len(requests) > 1
        assert (
            requests[1]["contents"][1]["parts"][0]["thoughtSignature"]
            == "offline-signature"
        )
        assert "functionResponse" in requests[1]["contents"][2]["parts"][0]
        for request, turn in zip(requests, record["provider_turns"], strict=True):
            assert json.loads(turn["wire"]["request_utf8"]) == request
            assert turn["usage_metadata"] == {
                **json.loads(turn["wire"]["response_utf8"])["usageMetadata"],
                "toolUsePromptTokenCount": 0,
            }
            assert turn["input_tokens"] == 1
            assert turn["output_tokens"] == 3
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert key not in repr(live)
        assert key not in json.dumps(record)
        assert key not in json.dumps(live.settings)
        assert key not in json.dumps(live.wire.captures)
        assert key not in (tmp_path / "campaign-budget.jsonl").read_text()
        assert all(
            key.encode() not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
        )
        assert key not in capsys.readouterr().out
    finally:
        live.close()
        offline.close()
        budget.close()


@pytest.mark.parametrize("status", (200, 400, 429, 302))
@pytest.mark.parametrize("escaped", (False, True))
def test_live_reflected_key_never_reaches_durable_response(
    tmp_path, capsys, status, escaped
):
    from operatebench.agents.transport import ProviderFailure

    key = "synthetic-key-not-real"
    budget, _mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    calls = []

    def handle(request):
        calls.append(request)
        body = json.dumps({"error": {"nested": [key]}})
        if escaped:
            body = body.replace("-", "\\u002d")
        return httpx.Response(
            status,
            content=body.encode(),
            headers={"Retry-After": key, "Location": "https://invalid.example/"},
        )

    live = GeminiCampaignTransport.live(
        model=trial.model,
        api_key=key,
        guard=offline.guard,
        max_output_tokens=65536,
        admitted=True,
        inner=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, live, output=tmp_path / "trial")
        assert key not in str(caught.value)
        assert len(calls) == 1
        assert live.stopped
        assert live.last_turn is None
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        assert len(live.wire.captures) == 1
        assert "response_utf8" not in live.wire.captures[0]
        assert "response_sha256" not in live.wire.captures[0]
        events = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert [
            event["kind"] for event in events if event["kind"].startswith("wire_")
        ] == (["wire_request"] if status == 200 else ["wire_request", "wire_response"])
        assert key not in json.dumps(events)
        assert all(
            key.encode() not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
        )
        assert key not in json.dumps(live.wire.captures)
        assert key not in capsys.readouterr().out
    finally:
        live.close()
        offline.close()
        budget.close()


def test_live_invalid_context_cannot_reach_wire(tmp_path, monkeypatch):
    import tools.three_flow_gemini as gemini
    from operatebench.agents.transport import ProviderFailure

    budget, _mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    calls = []
    live = GeminiCampaignTransport.live(
        model=trial.model,
        api_key="synthetic-key-not-real",
        guard=offline.guard,
        max_output_tokens=65536,
        admitted=True,
        inner=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200)
        ),
    )
    monkeypatch.setattr(
        gemini, "request_input_token_bound", lambda *_: live.profile.context
    )
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, live, output=tmp_path / "trial")
        assert calls == []
        assert live.wire.captures == []
        assert budget.records == {}
    finally:
        live.close()
        offline.close()
        budget.close()


@pytest.mark.parametrize(
    "malformed",
    (
        b'{"error":"synthetic\\u002dkey\\u002dnot\\u002dreal"',
        b'{"error":"synthetic\\u002dkey\\u002dnot\\u002dreal","error":"ordinary"}',
    ),
)
def test_live_uncapturable_success_never_reaches_observer(tmp_path, malformed):
    from operatebench.agents.transport import ProviderFailure

    budget, _mock, offline, trial = fixture(tmp_path, "gemini-4-argon")
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=malformed)

    live = GeminiCampaignTransport.live(
        model=trial.model,
        api_key="synthetic-key-not-real",
        guard=offline.guard,
        max_output_tokens=65536,
        admitted=True,
        inner=httpx.MockTransport(handle),
    )
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, live, output=tmp_path / "trial")
        assert len(calls) == 1
        assert live.stopped
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        assert "response_utf8" not in live.wire.captures[0]
        events = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert [
            event["kind"] for event in events if event["kind"].startswith("wire_")
        ] == ["wire_request"]
        assert all(
            malformed not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
        )
    finally:
        live.close()
        offline.close()
        budget.close()


def test_tool_use_prompt_tokens_complete_usage_equation():
    from operatebench.providers.faults import AdapterProviderError
    from tools.three_flow_gemini import parse_response

    body = response_body()
    body["usageMetadata"].update(toolUsePromptTokenCount=2, totalTokenCount=22)
    response, counts, _ = parse_response(json.dumps(body), "gemini-4-argon")
    assert response.output_tokens == 10
    assert counts["toolUsePromptTokenCount"] == 2
    assert counts["totalTokenCount"] == 22

    body["usageMetadata"]["totalTokenCount"] = 20
    with pytest.raises(AdapterProviderError, match="categories inconsistent"):
        parse_response(json.dumps(body), "gemini-4-argon")

    for invalid in (True, -1, "2", 1000000000000000000000000000):
        body["usageMetadata"].update(toolUsePromptTokenCount=invalid, totalTokenCount=22)
        with pytest.raises(AdapterProviderError, match="usage count invalid"):
            parse_response(json.dumps(body), "gemini-4-argon")


def test_tool_use_prompt_tokens_settle_wire_projection_and_replay(tmp_path):
    import copy

    from operatebench.agents.transport import content_digest

    def handler(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            body["usageMetadata"].update(toolUsePromptTokenCount=2, totalTokenCount=6)
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) == len(record["provider_turns"])
        for turn in record["provider_turns"]:
            assert turn["input_tokens"] == 3
            assert turn["output_tokens"] == 3
            assert turn["usage_metadata"]["toolUsePromptTokenCount"] == 2
            assert json.loads(turn["wire"]["response_utf8"])["usageMetadata"] == {
                "promptTokenCount": 1,
                "toolUsePromptTokenCount": 2,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 6,
                "cachedContentTokenCount": 0,
            }
            assert Decimal(turn["measured_usd"]) == (
                Decimal(3) * Decimal(MODELS[trial.model].input_rate)
                + Decimal(3) * Decimal(MODELS[trial.model].output_rate)
            ) / Decimal(1000000)
        assert budget.cap is None
        assert budget.cell_accounting("trial")["full_unknown_usd"] == "0"
        replay_trial(trial, record, expected_record_digest=record["record_digest"])
        for field, value in (("input_tokens", 1), ("measured_usd", "0")):
            changed = copy.deepcopy(record)
            changed["provider_turns"][0][field] = value
            changed["record_digest"] = content_digest(
                {key: val for key, val in changed.items() if key != "record_digest"}
            )
            with pytest.raises(ValueError, match="Gemini"):
                replay_trial(
                    trial, changed, expected_record_digest=changed["record_digest"]
                )
    finally:
        transport.close()
        budget.close()


def test_tool_use_prompt_tokens_over_input_bound_remain_unknown(tmp_path):
    from operatebench.agents.transport import ProviderFailure

    def handler(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            body["usageMetadata"].update(
                toolUsePromptTokenCount=1048576, totalTokenCount=1048580
            )
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
        assert transport.last_turn is None
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
    finally:
        transport.close()
        budget.close()


def test_exact_schema_and_no_loss_arguments():
    from operatebench.agents.lifecycle_contract import outcome_tools
    from tools.three_flow_gemini import declarations, parse_response

    projected = declarations()
    assert len(projected) == 6
    assert [d["parametersJsonSchema"] for d in projected] == [
        t["parameters"] for t in outcome_tools()
    ]
    body = response_body()
    response, usage, content = parse_response(json.dumps(body), "gemini-4-argon")
    assert (
        response.tool_calls[0].arguments
        == body["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
    )
    assert response.output_tokens == 10
    assert usage["cachedContentTokenCount"] == 2
    assert content == body["candidates"][0]["content"]
    raw = json.dumps(body).replace(
        '"action_type": "example"', '"action_type": "first", "action_type": "second"'
    )
    with pytest.raises(Exception, match="more than once"):
        parse_response(raw, "gemini-4-argon")


@pytest.mark.parametrize(
    "field,value",
    [
        ("thought", "true"),
        ("thought", 1),
        ("functionCall.id", None),
        ("functionCall.id", 7),
    ],
)
def test_native_response_rejects_wrong_thought_and_call_id_types(field, value):
    from operatebench.providers.faults import AdapterProviderError
    from tools.three_flow_gemini import parse_response

    body = response_body()
    part = body["candidates"][0]["content"]["parts"][0]
    if field == "thought":
        part["thought"] = value
    else:
        part["functionCall"]["id"] = value
    with pytest.raises(AdapterProviderError, match="Gemini invalid"):
        parse_response(json.dumps(body), "gemini-4-argon")


def test_native_response_preserves_valid_thought_call_id_signature_and_args():
    from tools.three_flow_gemini import parse_response

    body = response_body()
    part = body["candidates"][0]["content"]["parts"][0]
    part["thought"] = False
    part["functionCall"]["id"] = "call-1"
    response, _, content = parse_response(json.dumps(body), "gemini-4-argon")
    assert content["parts"][0] == part
    assert response.tool_calls[0].arguments == part["functionCall"]["args"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.pop("usageMetadata"),
        lambda b: b.update(modelVersion="gemini-flash-latest"),
        lambda b: b["usageMetadata"].update(thoughtsTokenCount=True),
        lambda b: b["usageMetadata"].update(totalTokenCount=999),
        lambda b: b["usageMetadata"].update(cachedContentTokenCount=11),
        lambda b: b["candidates"][0].update(finishReason="SAFETY"),
        lambda b: b["candidates"][0]["content"]["parts"][0].pop("thoughtSignature"),
        lambda b: b["candidates"][0]["content"]["parts"][0]["functionCall"].update(
            args='{"x":1}'
        ),
    ],
)
def test_native_response_negatives(mutation):
    from operatebench.providers.faults import AdapterProviderError
    from tools.three_flow_gemini import parse_response

    body = response_body()
    mutation(body)
    with pytest.raises(AdapterProviderError):
        parse_response(json.dumps(body), "gemini-4-argon")


@pytest.mark.parametrize("status", [400, 401, 403, 429, 503])
def test_http_failure_retains_liability_without_retry(tmp_path, status):
    from operatebench.agents.transport import ProviderFailure

    def handler(mock):
        def handle(request):
            mock.calls.append(json.loads(request.content))
            return httpx.Response(status, json={"error": "synthetic-secret-reflection"})

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        assert "synthetic-secret-reflection" not in json.dumps(transport.wire.captures)
        assert transport.wire.captures[0]["response_projection"]["http_status"] == status
    finally:
        transport.close()
        budget.close()


def test_distinct_series_and_gemini_null_cap(tmp_path):
    from tools.three_flow_campaign import assignments
    from tools.three_flow_gemini import LIMIT_PROFILE, ROSTER_PROFILE
    from tools.three_flow_profiles import registry

    slots = assignments("gemini-only", ROSTER_PROFILE)
    assert len(slots) == len({s["trial_id"] for s in slots}) == 88
    assert len({s["series_id"] for s in slots}) == 2
    for model in MODELS:
        selected = [s for s in slots if s["model"] == model]
        assert len(selected) == len({(s["flow"], s["scenario"]) for s in selected}) == 44
    rows = registry(roster_profile=ROSTER_PROFILE, mode_profile="gemini-low-v1")
    assert {r["model"] for r in rows} == set(MODELS)
    assert all(r["request_settings"]["thinking"] == "low" for r in rows)
    policy = {
        "mode": LIMIT_PROFILE,
        "owner_accepted": True,
        "owner_evidence_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "historical_liability_manifest_sha256": "c" * 64,
    }
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="gemini-only",
        cap=None,
        limit_policy=policy,
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    try:
        assert budget.cap is None
        assert budget.controller_profile == LIMIT_PROFILE
    finally:
        budget.close()


@pytest.mark.parametrize(
    "change",
    [
        "thinking",
        "output",
        "model",
        "request_bytes",
        "response_bytes",
        "usage",
        "history",
        "endpoint",
        "method",
        "reported_output_over_cap",
    ],
)
def test_replay_rejects_native_provenance_mutation(tmp_path, change):
    import copy

    from operatebench.agents.transport import content_digest

    budget, _mock, transport, trial = fixture(tmp_path, "gemini-4-argon")
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        changed = copy.deepcopy(record)
        wire = changed["provider_turns"][1]["wire"]
        payload = json.loads(wire["request_utf8"])
        if change == "thinking":
            payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] = "minimal"
        elif change == "output":
            payload["generationConfig"]["maxOutputTokens"] = 65537
        elif change == "model":
            body = json.loads(wire["response_utf8"])
            body["modelVersion"] = "wrong-model"
            wire["response_utf8"] = json.dumps(body)
        elif change == "request_bytes":
            wire["request_utf8"] += " "
        elif change == "response_bytes":
            wire["response_utf8"] += " "
        elif change == "usage":
            changed["provider_turns"][1]["thinking_tokens"] = 999
        elif change == "history":
            payload["contents"][1]["parts"][0]["thoughtSignature"] = "changed"
        elif change == "endpoint":
            wire["request_url"] = MODELS["gemini-3.8-flash"].endpoint
        elif change == "method":
            wire["request_method"] = "GET"
        elif change == "reported_output_over_cap":
            import hashlib

            body = json.loads(wire["response_utf8"])
            body["usageMetadata"].update(
                candidatesTokenCount=65537, totalTokenCount=65540
            )
            wire["response_utf8"] = json.dumps(body)
            wire["response_sha256"] = hashlib.sha256(
                wire["response_utf8"].encode()
            ).hexdigest()
            turn = changed["provider_turns"][1]
            turn["output_tokens"] = 65539
            turn["usage_metadata"] = body["usageMetadata"]
            turn["measured_usd"] = str(
                (
                    Decimal(1) * Decimal(MODELS[trial.model].input_rate)
                    + Decimal(65539) * Decimal(MODELS[trial.model].output_rate)
                )
                / Decimal(1000000)
            )
        if change in ("thinking", "output", "history"):
            wire["request_utf8"] = json.dumps(payload)
            import hashlib

            wire["request_sha256"] = hashlib.sha256(
                wire["request_utf8"].encode()
            ).hexdigest()
        changed["record_digest"] = content_digest(
            {k: v for k, v in changed.items() if k != "record_digest"}
        )
        with pytest.raises(ValueError, match="Gemini"):
            replay_trial(trial, changed, expected_record_digest=changed["record_digest"])
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": "synthetic-not-a-key"},
        {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        },
        {"model": "gemini-flash-latest"},
        {"inner": httpx.HTTPTransport()},
        {"max_output_tokens": 65537},
    ],
)
def test_constructor_rejects_unauthorized_capabilities(tmp_path, kwargs):
    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon")
    try:
        options = {
            "model": trial.model,
            "inner": mock,
            "guard": transport.guard,
            "max_output_tokens": 65536,
        }
        options.update(kwargs)
        with pytest.raises(ValueError):
            GeminiCampaignTransport(**options)
        assert not mock.calls
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", MODELS)
def test_all_88_native_offline_assignments(tmp_path, model):
    from tools.three_flow_campaign import assignments, run_offline
    from tools.three_flow_gemini import (
        ROSTER_PROFILE,
        offline_factory,
        offline_limit_policy,
    )

    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="acceptance-" + model,
        cap=None,
        limit_policy=offline_limit_policy(),
        roster_profile=ROSTER_PROFILE,
        create=True,
    )
    rows = run_offline(
        tmp_path,
        campaign_id="acceptance-" + model,
        providers=("gemini",),
        roster_profile=ROSTER_PROFILE,
        campaign_budget=budget,
        transport_factory=offline_factory,
        workers=4,
        slots=[
            s
            for s in assignments("acceptance-" + model, ROSTER_PROFILE)
            if s["model"] == model
        ],
    )
    counts = {
        "model": model,
        "assigned": 44,
        **{
            k: sum(r["classification"] == k for r in rows)
            for k in ("scored", "excluded", "aborted")
        },
    }
    evidence = tmp_path / "offline-counts.json"
    evidence.write_text(json.dumps(counts, indent=2) + "\n")
    evidence.chmod(0o600)
    assert len(rows) == len({r["trial_id"] for r in rows}) == 44
    assert len({(r["flow"], r["scenario"]) for r in rows}) == 44
    assert len({r["series_id"] for r in rows}) == 1
    assert counts == {
        "model": model,
        "assigned": 44,
        "scored": 44,
        "excluded": 0,
        "aborted": 0,
    }
    assert all(r["evidence_origin"] == "SDK_MOCK_NOT_LLM" for r in rows)
    assert all(r["eligible_for_live_results"] is False for r in rows)
    assert (
        json.loads((tmp_path / "scorecard.json").read_text())["global_budget"]["cap"]
        is None
    )


def test_gemini_preflight_and_profile_admission_are_non_authorizing():
    import copy

    from tools.three_flow_admission import proposal, validate, validate_profile
    from tools.three_flow_campaign import preflight
    from tools.three_flow_profiles import GEMINI_ROSTER_PROFILE

    description = proposal("a" * 40, GEMINI_ROSTER_PROFILE)
    assert description["cap_usd"] is None
    assert description["admitted"] is False
    assert description["credentials_accessed"] is False
    assert len(description["slots"]) == 88
    assert preflight(GEMINI_ROSTER_PROFILE)["global_cap_usd"] is None
    candidate = MODELS["gemini-4-argon"].settings(65536)
    validate_profile(
        candidate, "gemini", model="gemini-4-argon", roster_profile=GEMINI_ROSTER_PROFILE
    )
    bad = copy.deepcopy(candidate)
    bad["generationConfig"]["thinkingConfig"]["thinkingLevel"] = "off"
    with pytest.raises(ValueError):
        validate_profile(
            bad, "gemini", model="gemini-4-argon", roster_profile=GEMINI_ROSTER_PROFILE
        )
    with pytest.raises(ValueError, match="closed Gemini admission schema"):
        validate({"roster_profile": GEMINI_ROSTER_PROFILE}, "a" * 40, synthetic=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.pop("usageMetadata"),
        lambda b: b["usageMetadata"].update(toolUsePromptTokenCount=2, totalTokenCount=4),
        lambda b: b.update(modelVersion="gemini-flash-latest"),
        lambda b: b["candidates"][0].update(finishReason="SAFETY"),
    ],
)
def test_invalid_native_wire_excluded_with_unknown_liability(tmp_path, mutation):
    from operatebench.agents.transport import ProviderFailure

    def handler(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            mutation(body)
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        assert transport.stopped
    finally:
        transport.close()
        budget.close()


def test_null_cap_cannot_reuse_historical_or_meta_authority(tmp_path):
    from tools.three_flow_gemini import (
        LIMIT_PROFILE,
        ROSTER_PROFILE,
        offline_limit_policy,
    )

    tmp_path.chmod(0o700)
    old = CampaignBudget(tmp_path, campaign_id="old", cap=Decimal("1000"), create=True)
    old.close()
    with pytest.raises(ValueError):
        CampaignBudget(
            tmp_path,
            campaign_id="old",
            cap=None,
            limit_policy=offline_limit_policy(),
            roster_profile=ROSTER_PROFILE,
        )
    policy = offline_limit_policy()
    assert policy["mode"] == LIMIT_PROFILE
    policy["mode"] = "three-flow-meta-uncapped-v1"
    with pytest.raises(ValueError):
        CampaignBudget(
            tmp_path,
            campaign_id="gemini",
            cap=None,
            limit_policy=policy,
            roster_profile=ROSTER_PROFILE,
        )


def test_mock_connection_loss_is_excluded_without_retry(tmp_path):
    from operatebench.agents.transport import ProviderFailure

    def handler(mock):
        def handle(request):
            mock.calls.append(json.loads(request.content))
            raise httpx.ConnectError("synthetic connection loss")

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
        assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
    finally:
        transport.close()
        budget.close()


def test_native_wire_records_model_in_endpoint(tmp_path):
    budget, _mock, transport, trial = fixture(
        tmp_path,
        "gemini-4-argon",
        lambda mock: lambda request: httpx.Response(200, json=response_body()),
    )
    try:
        payload = {"generationConfig": transport.settings["generationConfig"]}
        reservation = transport.guard.authorize(input_tokens_upper_bound=10000)
        transport.wire.expected_payload = payload
        transport.wire.handle_request(
            httpx.Request("POST", transport.profile.endpoint, json=payload)
        )
        assert transport.wire.captures[0]["request_url"] == MODELS[trial.model].endpoint
        assert transport.wire.captures[0]["request_method"] == "POST"
        transport.guard.forfeit(reservation)
    finally:
        transport.close()
        budget.close()


def test_long_frozen_case_preserves_history_without_duplicate_observations(tmp_path):
    from dataclasses import replace

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon")
    trial = replace(trial, scenario="DE_WITHHOLDING")
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) > 25
        assert all(len(c["contents"][-1]["parts"]) == 1 for c in mock.calls)
        assert "functionResponse" in mock.calls[-1]["contents"][-1]["parts"][0]
        assert record["provider_turns"][-1]["input_bound"] + 65536 <= 1048576
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", MODELS)
def test_frozen_sc_reopen_context_blocker(tmp_path, model):
    """Previously RED for both models; retain the exact frozen regression."""
    budget, mock, transport, trial = fixture(
        tmp_path, model, flow="compliance", scenario="SC_reopen"
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) >= 40
        turn = record["provider_turns"][39]
        assert turn["input_bound"] == 985399
        assert turn["requested_output_bound"] == 63177
        assert mock.calls[39]["generationConfig"]["maxOutputTokens"] == 63177
        assert record["request_output_bounds"][39] == 63177
        replay_trial(trial, record, expected_record_digest=record["record_digest"])
        import copy
        import hashlib

        from operatebench.agents.transport import content_digest

        for mutation in ("cap", "policy", "usage"):
            changed = copy.deepcopy(record)
            selected = changed["provider_turns"][39]
            if mutation == "cap":
                selected["requested_output_bound"] = 65536
                changed["request_output_bounds"][39] = 65536
            elif mutation == "policy":
                selected["output_reservation_policy"] = "unversioned"
            else:
                wire = selected["wire"]
                body = json.loads(wire["response_utf8"])
                body["usageMetadata"].update(
                    candidatesTokenCount=63176, totalTokenCount=63179
                )
                wire["response_utf8"] = json.dumps(body)
                wire["response_sha256"] = hashlib.sha256(
                    wire["response_utf8"].encode()
                ).hexdigest()
                selected["output_tokens"] = 63178
                selected["usage_metadata"] = body["usageMetadata"]
                selected["measured_usd"] = str(
                    (
                        Decimal(MODELS[model].input_rate)
                        + Decimal(63178) * Decimal(MODELS[model].output_rate)
                    )
                    / Decimal(1000000)
                )
            changed["record_digest"] = content_digest(
                {k: v for k, v in changed.items() if k != "record_digest"}
            )
            with pytest.raises(ValueError, match="Gemini"):
                replay_trial(
                    trial, changed, expected_record_digest=changed["record_digest"]
                )
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("remaining", [65537, 65536, 65535, 1, 0, -1])
def test_dynamic_cap_actual_wire_and_guard(tmp_path, monkeypatch, remaining):
    import tools.three_flow_gemini as gemini
    from operatebench.agents.model import AGENT_TOOL_NAMES, MODEL_PROTOCOL_VERSION
    from operatebench.agents.transport import ModelRequest, content_digest
    from operatebench.providers.cost import (
        CostCapExceededError,
        CostReservationBreachedError,
    )

    budget, _mock, transport, trial = fixture(tmp_path, "gemini-4-argon")
    prompt = {"observation": {}}
    request = ModelRequest(
        model=trial.model,
        protocol_version=MODEL_PROTOCOL_VERSION,
        max_output_tokens=65536,
        tool_names=AGENT_TOOL_NAMES,
        observation_digest_sha256=content_digest({}),
        prompt_digest_sha256=content_digest(prompt),
        invocation_index=0,
        turn_index=0,
        prompt=prompt,
    )
    monkeypatch.setattr(gemini, "check_model_request", lambda *a, **k: None)
    monkeypatch.setattr(
        gemini, "request_input_token_bound", lambda *a: 1048576 - remaining
    )
    transport.wire.inner = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body())
    )
    try:
        if remaining <= 0:
            with pytest.raises(CostCapExceededError, match="before dispatch"):
                transport.send(request)
            assert not transport.wire.captures
            assert transport.guard.ticket is None
            assert not budget.records
        elif remaining == 1:
            with pytest.raises(CostReservationBreachedError):
                transport.send(transport.prepare_request(request))
            assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        else:
            transport.send(transport.prepare_request(request))
            cap = min(65536, remaining)
            wire = transport.last_turn["wire"]
            assert (
                json.loads(wire["request_utf8"])["generationConfig"]["maxOutputTokens"]
                == cap
            )
            assert transport.last_turn["requested_output_bound"] == cap
            assert transport.guard._max_output_tokens == cap
        assert budget.cap is None
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("remaining", [0, -1])
def test_no_output_room_is_excluded_before_provider_rpc(tmp_path, monkeypatch, remaining):
    import tools.three_flow_gemini as gemini
    from operatebench.agents.transport import FAULT_BUDGET, ProviderFailure

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon")
    monkeypatch.setattr(
        gemini,
        "request_input_token_bound",
        lambda *a: transport.profile.context - remaining,
    )
    try:
        with pytest.raises(ProviderFailure) as caught:
            run_trial(trial, transport, output=tmp_path / "trial")
        assert caught.value.fault == FAULT_BUDGET
        rows = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert rows[-1]["classification"] == "excluded"
        assert rows[-1]["exclusion_code"] == FAULT_BUDGET
        assert mock.calls == []
        assert transport.wire.captures == []
        assert transport.guard.ticket is None
        assert budget.records == {}
        assert budget.cell_accounting("trial")["full_unknown_usd"] == "0"
        assert not (tmp_path / "trial" / "record.json").exists()
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("reported", [63177, 63178])
def test_reduced_ceiling_usage_at_boundary(tmp_path, monkeypatch, reported):
    import tools.three_flow_gemini as gemini
    from operatebench.agents.transport import ProviderFailure

    def handler(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            body["usageMetadata"].update(
                candidatesTokenCount=reported - 2, totalTokenCount=reported + 1
            )
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, "gemini-4-argon", handler)
    monkeypatch.setattr(gemini, "request_input_token_bound", lambda *a: 985399)
    try:
        if reported == 63178:
            with pytest.raises(ProviderFailure):
                run_trial(trial, transport, output=tmp_path / "trial")
            assert len(mock.calls) == 1
            assert transport.stopped
            assert Decimal(budget.cell_accounting("trial")["full_unknown_usd"]) > 0
        else:
            record = run_trial(trial, transport, output=tmp_path / "trial")
            assert all(t["output_tokens"] == reported for t in record["provider_turns"])
            assert all(o == 63177 for o in record["request_output_bounds"])
            replay_trial(trial, record, expected_record_digest=record["record_digest"])
        assert mock.calls[0]["generationConfig"]["maxOutputTokens"] == 63177
        assert transport.guard._max_output_tokens == 63177
        assert all(r["ob"] == 63177 for r in budget.records.values())
        assert budget.cap is None
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("maximum", [1, 63177, 65535, 65537])
def test_profile_rejects_static_task_ceiling(maximum):
    for profile in MODELS.values():
        with pytest.raises(ValueError, match="physical output maximum"):
            profile.settings(maximum)
        assert profile.settings(65536)["physical_output_maximum"] == 65536
