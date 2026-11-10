"""Fable52 finite roster contract; all responses are synthetic SDK wires."""

import json

import pytest

from tests.test_selected_six_http import fixture
from tools.three_flow_profiles import registry, selected_roster

PROFILE = "anthropic-fable52-same44-v1"
MODEL = "claude-fable-5-2"


def test_fable52_finite_roster_and_mapping():
    from tools.three_flow_campaign import CONFIGURATIONS, assignments, preflight

    assert selected_roster(PROFILE) == (("anthropic", MODEL),)
    rows = registry(roster_profile=PROFILE, mode_profile="off-or-minimum-v1")
    assert rows[0]["request_settings"]["thinking"] == {"type": "adaptive"}
    assert rows[0]["request_settings"]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert rows[0]["mapping"] == "three-flow-anthropic-fable52-stateless-v1"
    assert rows[0]["request_settings"]["no_native_continuation"] is True
    slots = assignments("fable52", PROFILE)
    assert len(slots) == 44
    assert {(s["flow"], s["scenario"]) for s in slots} == set(CONFIGURATIONS)
    assert preflight(PROFILE)["global_cap_usd"] is None


@pytest.mark.parametrize("transport_policy", [None, "legacy-v1"])
def test_fable52_accepts_legacy_transport_policy(transport_policy):
    kwargs = {} if transport_policy is None else {"transport_policy": transport_policy}
    assert registry(
        roster_profile=PROFILE, mode_profile="off-or-minimum-v1", **kwargs
    ) == registry(
        roster_profile=PROFILE,
        mode_profile="off-or-minimum-v1",
        transport_policy="legacy-v1",
    )


def test_fable52_rejects_paced_transport_policy():
    with pytest.raises(ValueError, match="November requires legacy-v1 transport policy"):
        registry(
            roster_profile=PROFILE,
            mode_profile="off-or-minimum-v1",
            transport_policy="paced-safe-errors-v1",
        )


def assert_projection(call, request):
    from operatebench.agents import anthropic_messages as aa

    assert call["system"] == aa.request_instructions(request)
    assert call["messages"] == [
        {
            "role": "user",
            "content": json.dumps(
                dict(request.prompt),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }
    ]
    assert call["tools"] == aa.anthropic_outcome_tools()


def test_fable_sdk_transcript_and_accounting(tmp_path, monkeypatch):
    import httpx

    from tools.three_flow_mock import SDKMockTransport
    from tools.three_flow_runtime import replay_trial, run_trial

    original = SDKMockTransport.handle

    def thinking_response(self, request):
        response = original(self, request)
        body = json.loads(response.content)
        body["content"].insert(
            0,
            {
                "type": "thinking",
                "thinking": "synthetic progress",
                "signature": "offline-signature",
            },
        )
        return httpx.Response(200, json=body)

    monkeypatch.setattr(SDKMockTransport, "handle", thinking_response)
    budget, t, trial, network = fixture(
        tmp_path, "anthropic", MODEL, roster_profile=PROFILE
    )
    requests = []
    original_send = t.send

    def capture(request):
        requests.append(request)
        return original_send(request)

    monkeypatch.setattr(t, "send", capture)
    try:
        assert network.transports[0].calls == []
        assert t.http.timeout.connect == t.settings["connection_timeout_seconds"] == 10
        assert t.http.timeout.pool == t.settings["pool_timeout_seconds"] == 10
        assert (
            t.http.timeout.read is t.http.timeout.write is t.settings["timeout"] is None
        )
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert t.settings["native_continuation"] is False
        assert t.settings["no_native_continuation"] is True
        calls = network.transports[0].calls
        assert len(calls) > 1
        for i, call in enumerate(calls):
            assert call["thinking"] == {"type": "adaptive"}
            assert call["output_config"] == {"effort": "low"}
            assert call["tool_choice"] == {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }
            assert not {"temperature", "top_p", "top_k"} & call.keys()
            assert_projection(call, requests[i])
            raw = json.loads(record["provider_turns"][i]["wire"]["response_utf8"])
            assert {"thinking", "redacted_thinking"} <= {
                block["type"] for block in raw["content"]
            }
        assert len({r.request_digest_sha256 for r in requests}) == len(requests)
        for mutation in ("freeze_digest", "accumulate"):
            bad = json.loads(json.dumps(calls[1]))
            if mutation == "freeze_digest":
                bad["system"] = calls[0]["system"]
            else:
                bad["messages"] = calls[0]["messages"] + bad["messages"]
            with pytest.raises(AssertionError):
                assert_projection(bad, requests[1])
        assert all(r["state"] == "settled" for r in budget.records.values())
        assert budget.cap is None
        assert budget.totals()["remaining"] is None
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert (
            t.request_mapping
            == registry(roster_profile=PROFILE, mode_profile="off-or-minimum-v1")[0][
                "mapping"
            ]
        )
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize("fault", ["multiple", "json", "usage", "model"])
def test_fable_invalid_transport(tmp_path, fault):
    from operatebench.agents.transport import ProviderFailure
    from tools.three_flow_runtime import run_trial

    budget, t, trial, network = fixture(
        tmp_path, "anthropic", MODEL, fault, roster_profile=PROFILE
    )
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, t, output=tmp_path / "trial")
        assert len(network.transports[0].calls) == 1
        assert all(r["state"] == "unknown" for r in budget.records.values())
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize(
    "fault", ["args", "badargs", "invalid_payload", "refusal", "no_action"]
)
def test_fable_quality_failure_no_salvage(tmp_path, fault):
    from tools.three_flow_runtime import run_trial

    budget, t, trial, network = fixture(
        tmp_path, "anthropic", MODEL, fault, roster_profile=PROFILE
    )
    try:
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert not record["evaluation"]["reliable"]
        assert budget.totals()["unknown"] == 0
        assert network.transports[0].calls
    finally:
        t.close()
        budget.close()


def test_fable52_published_profile_closed():
    from tests.test_selected_six_bounds import mistral_profile
    from tools.three_flow_admission import digest, validate_profile
    from tools.three_flow_profiles import ENDPOINTS

    p = mistral_profile()
    p.update(
        endpoint=ENDPOINTS["anthropic"],
        context_max=1000000,
        output_max=128000,
        input_usd_per_million="16",
        output_usd_per_million="40",
        pricing_policy="fable52-standard-global-token-envelope-v1",
    )
    p["tariff_evidence"]["category_rates"] = {
        "input": "8",
        "cache_read": "0.20",
        "cache_write": "16",
        "output": "40",
    }
    p["rate_evidence_sha256"] = digest(p["tariff_evidence"])
    validate_profile(p, "anthropic", model=MODEL, roster_profile=PROFILE)
    validate_profile(
        dict(p, execution_policy="legacy-v1"),
        "anthropic",
        model=MODEL,
        roster_profile=PROFILE,
    )
    for field, value in [
        ("output_max", 4096),
        ("network_timeout_seconds", 30),
        ("input_usd_per_million", "2"),
    ]:
        with pytest.raises(ValueError):
            validate_profile(
                dict(p, **{field: value}),
                "anthropic",
                model=MODEL,
                roster_profile=PROFILE,
            )
    with pytest.raises(ValueError):
        validate_profile(
            p, "anthropic", model=MODEL, roster_profile="selected-six-http-v1"
        )


@pytest.mark.parametrize("transport_policy", [None, "paced-safe-errors-v1"])
def test_fable52_fresh_source_bound_admission_and_one_shot(tmp_path, transport_policy):
    from tests.test_selected_six_bounds import mistral_profile
    from tests.test_selected_six_http import Network
    from tests.test_three_flow_live import cli, write
    from tests.test_three_flow_live import fixture as admission_fixture
    from tests.test_three_flow_optional_limits import consent
    from tools import three_flow_live as live
    from tools.three_flow_admission import digest, validate
    from tools.three_flow_campaign import assignments

    sha = live.source_identity()
    path, a, _ = admission_fixture(tmp_path, "anthropic")
    slot = next(
        s for s in assignments(a["campaign_id"], PROFILE) if s["flow"] == "commerce"
    )
    p = mistral_profile()
    p.update(
        endpoint="https://api.anthropic.com/v1/messages",
        context_max=1000000,
        output_max=128000,
        input_usd_per_million="16",
        output_usd_per_million="40",
        pricing_policy="fable52-standard-global-token-envelope-v1",
    )
    p["tariff_evidence"]["category_rates"] = {
        "input": "8",
        "cache_read": "0.20",
        "cache_write": "16",
        "output": "40",
    }
    p["rate_evidence_sha256"] = digest(p["tariff_evidence"])
    a.update(
        roster_profile=PROFILE,
        mode_profile="off-or-minimum-v1",
        registry=registry(roster_profile=PROFILE, mode_profile="off-or-minimum-v1"),
        selected=[slot["trial_id"]],
        pending=[slot["trial_id"]],
        profiles={MODEL: p},
        cap_usd=None,
        limit_policy={**consent(), "source_sha": sha},
    )
    if transport_policy is not None:
        a["transport_policy"] = transport_policy
        with pytest.raises(
            ValueError, match="November requires legacy-v1 transport policy"
        ):
            validate(a, sha, synthetic=True)
        write(path, a)
        network = Network()
        assert cli(path, a, network) == 2
        assert network.transports == []
        return
    assert validate(a, sha, synthetic=True) == a
    with pytest.raises(ValueError):
        validate(a, "e" * 40, synthetic=True)
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 0
    count = len(network.transports)
    assert cli(path, a, network) == 2
    assert len(network.transports) == count
