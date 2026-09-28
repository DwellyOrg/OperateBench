"""Reference-driven actual SDK mocks: mechanics, NOT real model performance."""

import json
import socket
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.playback import decision_record
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.core.protocol import AgentObservation
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_openai import OpenAICampaignTransport
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial

CASES = (
    [("maintenance", s) for s in ("V1", "V2", "V3")]
    + [("commerce", s) for s in COMMERCE_COMMANDS.reference_scenarios]
    + [("compliance", s) for s in COMPLIANCE_COMMANDS.reference_scenarios]
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def scripted_handler(flow, calls, negative=False):
    if flow == "maintenance":
        reference = build_agent("complete_early" if negative else "reference")
    else:
        commands = COMMERCE_COMMANDS if flow == "commerce" else COMPLIANCE_COMMANDS
        reference = commands.factories.build_agent(
            "skip_reads" if negative else "reference"
        )
    reference.begin_episode({})

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        public = json.loads(body["input"])["observation"]
        decision = decision_record(reference.decide(AgentObservation(**public)))
        name = decision.pop("kind").lower()
        if name == "ask":
            decision["wait"].pop("kind")
        return httpx.Response(
            200,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_offline",
                        "call_id": "call_offline",
                        "name": name,
                        "arguments": json.dumps(decision),
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    return handler


def setup_transport(tmp_path, trial, handler):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id=trial.campaign_id, cap=Decimal("1000"), create=True
    )
    guard = SharedGuard(
        budget=budget,
        cell=trial.trial_id,
        policy=LifecyclePricingPolicy(
            "synthetic-test-not-price-admission", Decimal("1"), Decimal("2")
        ),
        max_output_tokens=trial.max_output_tokens,
    )
    transport = OpenAICampaignTransport(
        api_key="offline-not-a-credential",
        inner=httpx.MockTransport(handler),
        guard=guard,
        max_output_tokens=trial.max_output_tokens,
    )
    return budget, transport


@pytest.mark.parametrize(("flow", "scenario"), CASES)
def test_real_sdk_record_and_zero_provider_replay(tmp_path, flow, scenario):
    calls = []
    trial = Trial("campaign-a", "trial-a", flow, scenario, "gpt-6-astra", 128000)
    budget, transport = setup_transport(tmp_path, trial, scripted_handler(flow, calls))
    trial = replace(trial, provider_binding=provider_identity(transport))
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial-a")
        assert budget.totals()["known"] > 0
        assert budget.totals()["pending"] == 0
        assert budget.totals()["unknown"] == 0
    finally:
        transport.close()
        budget.close()
    count = len(calls)
    assert count > 0
    assert record["evaluation"]["reliable"]
    result = replay_trial(trial, record, expected_record_digest=record["record_digest"])
    assert result["provider_calls"] == 0
    assert result["decisions_consumed"] == count
    assert len(calls) == count
    assert all(body["max_output_tokens"] == 128000 for body in calls)
    assert all("temperature" not in body and "reasoning" not in body for body in calls)
    assert all(
        body["tool_choice"] == "required" and body["store"] is False for body in calls
    )


@pytest.mark.parametrize(("flow", "scenario"), CASES)
def test_matched_negative_control_real_sdk_replay(tmp_path, flow, scenario):
    calls = []
    trial = Trial("campaign-a", "negative-a", flow, scenario, "gpt-6-astra", 128000)
    budget, transport = setup_transport(
        tmp_path, trial, scripted_handler(flow, calls, negative=True)
    )
    trial = replace(trial, provider_binding=provider_identity(transport))
    try:
        record = run_trial(trial, transport, output=tmp_path / "negative-a")
        assert calls
        assert not record["evaluation"]["reliable"]
        if flow == "maintenance":
            assert not record["evaluation"]["legitimate_completion"]
            assert "action_validity" in record["evaluation"]["failed_dimensions"]
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
