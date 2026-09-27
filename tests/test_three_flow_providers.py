"""Network-blocked SDK-wire tests for the successor HTTP transports."""

import copy
import json
import socket
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.playback import PlaybackError
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure, content_digest
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import MODELS, HTTPCampaignTransport
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def fixture(
    tmp_path,
    provider,
    *,
    handler=None,
    model=None,
    flow="commerce",
    scenario="UK_NORMAL",
    mode_profile="legacy-v1",
    transport_policy="legacy-v1",
):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="test", cap=Decimal("1000"), create=True
    )
    maximum = 256000 if provider == "mistral" else 128000
    guard = SharedGuard(
        budget=budget,
        cell="trial",
        policy=LifecyclePricingPolicy(
            "offline-synthetic-not-approved", Decimal("1"), Decimal("2")
        ),
        max_output_tokens=maximum,
    )
    model = MODELS[provider] if model is None else model
    mock = SDKMockTransport(provider, flow, model=model)
    transport = HTTPCampaignTransport(
        provider=provider,
        model=model,
        api_key="offline-not-a-credential",
        inner=mock if handler is None else httpx.MockTransport(handler(mock)),
        guard=guard,
        max_output_tokens=maximum,
        transport_policy=transport_policy,
        **({"mode_profile": mode_profile} if mode_profile != "legacy-v1" else {}),
    )
    trial = Trial(
        "test",
        "trial",
        flow,
        scenario,
        model,
        maximum,
        provider_identity(transport),
    )
    return budget, mock, transport, trial


@pytest.mark.parametrize("provider", MODELS)
def test_provider_sdk_roundtrip(tmp_path, provider):
    budget, mock, transport, trial = fixture(tmp_path, provider)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        count = len(mock.calls)
        assert count > 0
        assert record["evaluation"]["reliable"]
        assert len(record["provider_turns"]) == count
        assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(mock.calls) == count
        if provider == "anthropic":
            assert all(
                c["thinking"] == {"type": "adaptive"}
                and c["tool_choice"] == {"type": "auto"}
                for c in mock.calls
            )
            assert all(t["output_tokens"] == 2 for t in record["provider_turns"])
        if provider == "mistral":
            assert all(0 < c["max_tokens"] < 256000 for c in mock.calls)
        assert budget.totals()["unknown"] == 0
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("flow", ["commerce", "compliance", "maintenance"])
def test_null_root_metadata_sdk_roundtrip_replays_without_provider(
    tmp_path, provider, flow
):
    def wrapper(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            key = "access_programs" if provider == "openai" else "diagnostics"
            assert key not in body  # The baseline fixture remains the old contract.
            body[key] = None
            return httpx.Response(200, json=body)

        return handle

    from tools.three_flow_campaign import assignments

    scenario = next(s["scenario"] for s in assignments("test") if s["flow"] == flow)
    budget, mock, transport, trial = fixture(
        tmp_path, provider, handler=wrapper, flow=flow, scenario=scenario
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        calls = len(mock.calls)
        assert calls > 0
        assert record["evaluation"]["reliable"]
        assert all(t["input_tokens"] == 1 for t in record["provider_turns"])
        assert all(
            t["output_tokens"] == (1 if provider == "openai" else 2)
            for t in record["provider_turns"]
        )
        assert budget.totals()["unknown"] == 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(mock.calls) == calls
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.parametrize(
    "fault", ["usage_bool", "usage_missing", "wrong_model", "wire", "network", "status"]
)
def test_provider_faults_hold_liability(tmp_path, provider, fault):
    calls = []

    def wrapper(mock):
        def handle(request):
            calls.append(1)
            if fault == "network":
                raise httpx.ReadError("offline", request=request)
            if fault == "status":
                return httpx.Response(
                    503, json={"error": {"message": "offline", "type": "api_error"}}
                )
            if fault == "wire":
                return httpx.Response(200, text="{bad")
            response = mock.handle(request)
            body = json.loads(response.content)
            if fault == "usage_bool":
                body["usage"][
                    "prompt_tokens" if provider == "mistral" else "input_tokens"
                ] = True
            elif fault == "usage_missing":
                body.pop("usage")
            elif fault == "wrong_model":
                body["model"] = "wrong"
            return httpx.Response(200, json=body)

        return handle

    budget, _, transport, trial = fixture(tmp_path, provider, handler=wrapper)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert calls == [1]
        assert budget.totals()["unknown"] > 0
        assert budget.totals()["pending"] == 0
        assert not (tmp_path / "trial" / "record.json").exists()
        rows = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert rows[-1]["classification"] == "excluded"
        assert any(r["kind"] == "wire_request" for r in rows)
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    "field",
    [
        "settings",
        "model",
        "flow",
        "scenario",
        "source",
        "grade",
        "decisions",
        "origin",
        "request",
    ],
)
def test_coherently_resealed_changes_rejected(tmp_path, field):
    budget, _, transport, trial = fixture(tmp_path, "openai")
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
    finally:
        transport.close()
        budget.close()
    altered = copy.deepcopy(record)
    if field == "settings":
        altered["provider"]["settings"]["max_output_tokens"] = 8
    elif field in ("model", "flow", "scenario"):
        altered["trial"][field] = "other"
    elif field == "source":
        altered["binding"]["campaign_runtime_digest"] = "0" * 64
    elif field == "grade":
        altered["evaluation"]["reliable"] = False
    elif field == "decisions":
        altered["decisions"].pop()
    elif field == "origin":
        altered["evidence_origin"] = "PROVIDER_CANDIDATE"
    else:
        altered["request_digests"][0] = "0" * 64
    altered["record_digest"] = content_digest(
        {k: v for k, v in altered.items() if k != "record_digest"}
    )
    with pytest.raises(ValueError):
        replay_trial(trial, altered, expected_record_digest=record["record_digest"])
    # Even if an attacker supplies their resealed outer digest, independent
    # assignment/settings/source binding and actual replay still reject these.
    with pytest.raises((ValueError, PlaybackError)):
        replay_trial(trial, altered, expected_record_digest=altered["record_digest"])
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )


def test_settings_refused_before_dispatch(tmp_path):
    budget, mock, transport, trial = fixture(tmp_path, "openai")
    try:
        wrong = replace(trial, provider_binding={})
        with pytest.raises(ValueError, match="settings"):
            run_trial(wrong, transport, output=tmp_path / "trial")
        assert not mock.calls
        assert budget.totals()["pending"] == 0
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
def test_model_malformed_arguments_are_scored_not_provider_fault(tmp_path, provider):
    def wrapper(mock):
        def handle(request):
            response = mock.handle(request)
            body = json.loads(response.content)
            # Syntactically valid provider envelope, invalid agent tool name.
            if provider == "openai":
                body["output"][0]["name"] = "unknown_agent_tool"
            elif provider == "anthropic":
                body["content"][-1]["name"] = "unknown_agent_tool"
            else:
                body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = (
                    "unknown_agent_tool"
                )
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(tmp_path, provider, handler=wrapper)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) > 0
        assert not record["evaluation"]["reliable"]
        assert budget.totals()["unknown"] == 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("provider", MODELS)
def test_sdk_cannot_read_provider_environment(tmp_path, provider, monkeypatch):
    import os

    original = type(os.environ).__getitem__

    def guarded(self, key):
        if key.startswith(("OPENAI_", "ANTHROPIC_", "MISTRAL_")):
            raise AssertionError("provider environment read forbidden")
        return original(self, key)

    monkeypatch.setattr(type(os.environ), "__getitem__", guarded)
    budget, mock, transport, trial = fixture(tmp_path, provider)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert mock.calls
        assert record["evaluation"]["reliable"]
    finally:
        transport.close()
        budget.close()
