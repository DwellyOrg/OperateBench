"""Offline native Gemini transport and replay for the new finite roster."""

import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_gemini import (
    LATE_NOVEMBER_MODELS,
    MODELS,
    GeminiCampaignTransport,
    GeminiMockTransport,
)
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def fixture(tmp_path, model, fault=None):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    p = LATE_NOVEMBER_MODELS[model]
    guard = SharedGuard(
        budget=budget,
        cell="trial",
        max_output_tokens=p.output,
        policy=LifecyclePricingPolicy(
            "synthetic", Decimal(p.input_rate), Decimal(p.output_rate)
        ),
    )
    mock = GeminiMockTransport(model, "commerce")

    def response(request):
        result = mock.handle(request)
        body = json.loads(result.content)
        if fault == "signature":
            del body["candidates"][0]["content"]["parts"][0]["thoughtSignature"]
        elif fault == "usage":
            del body["usageMetadata"]["totalTokenCount"]
        elif fault == "model":
            body["modelVersion"] = "unexpected-model"
        return httpx.Response(200, json=body)

    transport = GeminiCampaignTransport(
        model=model,
        inner=httpx.MockTransport(response),
        guard=guard,
        max_output_tokens=p.output,
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    trial = Trial(
        "offline",
        "trial",
        "commerce",
        "UK_NORMAL",
        model,
        p.output,
        provider_identity(transport),
    )
    return budget, mock, transport, trial


@pytest.mark.parametrize("model", tuple(LATE_NOVEMBER_MODELS))
def test_native_record_replay_and_history(tmp_path, model):
    budget, mock, transport, trial = fixture(tmp_path, model)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) > 1
        assert all(
            c["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
            for c in mock.calls
        )
        assert (
            mock.calls[1]["contents"][1]["parts"][0]["thoughtSignature"]
            == "offline-signature"
        )
        assert all(
            t["output_tokens"] == 3 and t["thinking_tokens"] == 2
            for t in record["provider_turns"]
        )
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert all(r["state"] == "settled" for r in budget.records.values())
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", tuple(LATE_NOVEMBER_MODELS))
@pytest.mark.parametrize("fault", ("signature", "usage", "model"))
def test_native_invalid_wire_forfeits(tmp_path, model, fault):
    budget, mock, transport, trial = fixture(tmp_path, model, fault)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
        assert all(r["state"] == "unknown" for r in budget.records.values())
    finally:
        transport.close()
        budget.close()


def test_old_roster_remains_isolated():
    assert set(MODELS) == {"gemini-4-argon", "gemini-3.8-flash"}
    assert (
        MODELS["gemini-4-argon"].settings(65536)["pricing_basis"]["verified_on"]
        == "2026-10-14"
    )
    with pytest.raises(ValueError):
        from tools.three_flow_gemini import ROSTER_PROFILE, gemini_profile

        gemini_profile("gemini-4-neon", ROSTER_PROFILE)


def test_operator_executes_selected_gemini_models(tmp_path):
    from tools.three_flow_campaign import assignments, run_campaign
    from tools.three_flow_gemini import offline_factory

    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    budget = CampaignBudget(
        root,
        campaign_id="offline-operator",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    slots = [
        next(
            s
            for s in assignments("offline-operator", LATE_NOVEMBER_PROFILE)
            if s["model"] == model and s["flow"] == "maintenance"
        )
        for model in LATE_NOVEMBER_MODELS
    ]
    rows = run_campaign(
        root,
        campaign_id="offline-operator",
        slots=slots,
        workers=2,
        campaign_budget=budget,
        transport_factory=offline_factory,
        evidence_origin="SDK_MOCK_NOT_LLM",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    assert len(rows) == 2
    assert all(row["classification"] == "scored" for row in rows)
