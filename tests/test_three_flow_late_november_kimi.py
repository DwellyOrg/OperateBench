"""Kimi Pro mock SDK wire and native reasoning continuation."""

import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tests.test_selected_six_http import SelectedMock
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_kimi_pro_preserves_reasoning_in_sdk_history(tmp_path):
    model = "moonshotai/Kimi-K3.1-Pro"
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    p = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=p,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
    )
    mock = SelectedMock("river_http", model, "commerce")
    t = HTTPCampaignTransport(
        provider="river_http",
        model=model,
        api_key="offline-key",
        inner=mock,
        guard=guard,
        max_output_tokens=30000,
        context_tokens=1048576,
        mode_profile="off-or-minimum-v1",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    t.settings.update(bounds_policy=p["bounds_policy"], context_request_bound=1048576)
    trial = Trial(
        "offline", "trial", "commerce", "UK_NORMAL", model, 30000, provider_identity(t)
    )
    try:
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(mock.calls) > 1
        assert t.settings["reasoning_effort"] == "none"
        assert t.settings["reasoning_mode"] == "OFF"
        assert all(
            c["reasoning_effort"] == "none" and c["max_tokens"] <= 30000
            for c in mock.calls
        )
        assert mock.calls[1]["messages"][-2]["reasoning_content"] == "synthetic reasoning"
        assert mock.calls[1]["messages"][-1]["role"] == "tool"
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        t.close()
        budget.close()


class OutputUsageMock(SelectedMock):
    def __init__(self, model, output):
        super().__init__("river_http", model, "commerce")
        self.output = output

    def handle(self, request):
        response = super().handle(request)
        body = json.loads(response.content)
        output = self.output if len(self.calls) == 1 else 7
        body["usage"]["completion_tokens"] = output
        body["usage"]["total_tokens"] = 11 + output
        return httpx.Response(200, json=body)


@pytest.mark.parametrize("reported", [30000, 30001, 30003])
def test_kimi_sdk_output_ceiling_and_restart(tmp_path, reported):
    model = "moonshotai/Kimi-K3.1-Pro"
    tmp_path.chmod(0o700)
    kwargs = {
        "campaign_id": "offline-ceiling",
        "cap": Decimal("1000"),
        "roster_profile": LATE_NOVEMBER_PROFILE,
    }
    budget = CampaignBudget(tmp_path, create=True, **kwargs)
    profile = {
        "endpoint": "https://api.river.ai/v1/chat/completions",
        "input_max": None,
        "output_max": 1048575,
        "context_max": 1048576,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=profile,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("2"), Decimal("10")),
    )
    mock = OutputUsageMock(model, reported)
    transport = HTTPCampaignTransport(
        provider="river_http",
        model=model,
        api_key="offline-key",
        inner=mock,
        guard=guard,
        max_output_tokens=30000,
        context_tokens=1048576,
        mode_profile="off-or-minimum-v1",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    transport.settings.update(
        bounds_policy=profile["bounds_policy"], context_request_bound=1048576
    )
    trial = Trial(
        "offline",
        "trial",
        "commerce",
        "UK_NORMAL",
        model,
        30000,
        provider_identity(transport),
    )
    try:
        if reported == 30000:
            record = run_trial(trial, transport, output=tmp_path / "trial")
            assert record["evaluation"]["reliable"]
            assert (
                replay_trial(
                    trial, record, expected_record_digest=record["record_digest"]
                )["provider_calls"]
                == 0
            )
            assert budget.totals()["known"] > 0
        else:
            with pytest.raises(ProviderFailure) as failure:
                run_trial(trial, transport, output=tmp_path / "trial")
            assert failure.value.fault == "provider_budget"
            assert len(mock.calls) == 1
            rows = [
                json.loads(line)
                for line in (tmp_path / "trial" / "partial.ndjson")
                .read_text()
                .splitlines()
            ]
            assert not any(row["kind"] == "decision" for row in rows)
            assert not (tmp_path / "trial" / "record.json").exists()
            assert rows[-1]["classification"] == "excluded"
            assert rows[-1]["exclusion_code"] == "provider_budget"
            wires = [row for row in rows if row["kind"] == "wire_response"]
            assert len(wires) == 1
            assert (
                json.loads(wires[0]["response_utf8"])["usage"]["completion_tokens"]
                == reported
            )
            assert budget.totals()["unknown"] > 0
            assert budget.totals()["known"] == 0
        assert all(call["max_tokens"] <= 30000 for call in mock.calls)
        assert mock.calls[0]["max_tokens"] == 30000
    finally:
        transport.close()
        budget.close()
    recovered = CampaignBudget(tmp_path, **kwargs)
    try:
        if reported > 30000:
            assert recovered.totals()["unknown"] > 0
            assert recovered.totals()["known"] == 0
            assert len(recovered.records) == 1
            assert next(iter(recovered.records.values()))["state"] == "unknown"
        else:
            assert recovered.totals()["known"] > 0
    finally:
        recovered.close()
