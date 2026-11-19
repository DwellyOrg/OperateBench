"""Offline Ember Responses route through the installed OpenAI SDK."""

import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_meta import profile
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_meta import validate_ember_profile
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("fault", [None, "model", "usage", "extra"])
def test_ember_sdk_record_and_replay(tmp_path, fault):
    p = profile("muse-wonder-1.0")
    p.update(
        output_max=524288, input_usd_per_million="0.30", output_usd_per_million="1.50"
    )
    validate_ember_profile(p, "meta", "muse-ember-1.0")
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    guard = AdmittedGuard(
        profile=p,
        budget=budget,
        cell="trial",
        max_output_tokens=524288,
        policy=LifecyclePricingPolicy("synthetic", Decimal("0.30"), Decimal("1.50")),
    )
    mock = SDKMockTransport("openai", "commerce", model="muse-ember-1.0")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        response = mock.handle(request)
        body = json.loads(response.content)
        body.update(store=False, parallel_tool_calls=False, tool_choice="auto", tools=[])
        body["usage"].update(
            input_tokens_details={"cached_tokens": 1},
            output_tokens_details={"reasoning_tokens": 1},
        )
        if fault == "usage":
            body["usage"]["output_tokens"] = True
        elif fault == "model":
            body["model"] = "muse-ember-1.0-contributor"
        elif fault == "extra":
            body["unexpected"] = True
        return httpx.Response(200, json=body)

    transport = HTTPCampaignTransport(
        provider="meta",
        model="muse-ember-1.0",
        api_key="offline-key",
        inner=httpx.MockTransport(handle),
        guard=guard,
        max_output_tokens=524288,
        context_tokens=1048576,
        mode_profile="off-or-minimum-v1",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    transport.settings.update(
        bounds_policy=p["bounds_policy"], context_request_bound=p["context_max"]
    )
    trial = Trial(
        "offline",
        "trial",
        "commerce",
        "UK_NORMAL",
        "muse-ember-1.0",
        524288,
        provider_identity(transport),
    )
    try:
        if fault:
            with pytest.raises(ProviderFailure):
                run_trial(trial, transport, output=tmp_path / "trial")
            assert len(requests) == 1
            assert all(r["state"] == "unknown" for r in budget.records.values())
            return
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert len(requests) > 1
        assert all(r["reasoning"] == {"effort": "minimal"} for r in requests)
        assert all(
            r["model"] == "muse-ember-1.0" and r["max_output_tokens"] <= 524288
            for r in requests
        )
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    ("input_tokens", "cached", "output", "expected"),
    [(200000, 50000, 100, "0.01525"), (200001, 50000, 100, "0.0457503")],
)
def test_ember_exact_tiers_and_restart(tmp_path, input_tokens, cached, output, expected):
    tmp_path.chmod(0o700)
    kwargs = {
        "campaign_id": "tariff",
        "cap": Decimal("1000"),
        "roster_profile": LATE_NOVEMBER_PROFILE,
    }
    budget = CampaignBudget(tmp_path, create=True, **kwargs)
    policy = LifecyclePricingPolicy("ember", Decimal("0.30"), Decimal("1.50"))
    ident = budget.reserve("cell", input_bound=200001, output_bound=100, policy=policy)
    budget.dispatch("cell", {"max_output_tokens": 100})
    assert budget.settle_ember(
        ident, input_tokens=input_tokens, output_tokens=output, cached_tokens=cached
    ) == Decimal(expected)
    budget.close()
    replay = CampaignBudget(tmp_path, **kwargs)
    try:
        assert replay.totals()["known"] == Decimal(expected)
        assert replay.records[ident]["cached"] == cached
    finally:
        replay.close()
