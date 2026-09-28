"""Latest roster is opt-in; actual SDK wires, never real provider claims."""

import copy
import json
import socket

import httpx
import pytest

from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_providers import fixture
from tools.three_flow_admission import (
    GUARANTEES,
    LATEST_HTTP_ENVELOPES,
    RESOURCE_POLICY,
    _validate,
    registry,
    runtime_identity,
    validate_latest_http_profile,
)
from tools.three_flow_campaign import (
    CONFIGURATIONS,
    ROSTER,
    assignments,
    preflight,
    selected_roster,
)
from tools.three_flow_http import ENDPOINTS, LATEST_HTTP_MODELS, minimum_effort
from tools.three_flow_runtime import replay_trial, run_trial

PROFILE = "latest-http-v1"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("provider,model", LATEST_HTTP_MODELS)
def test_latest_actual_sdk_business_roundtrip(tmp_path, provider, model):
    budget, mock, transport, trial = fixture(
        tmp_path, provider, model=model, mode_profile="off-or-minimum-v1"
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert mock.calls and all(c["model"] == model for c in mock.calls)
        assert transport.settings["endpoint"] == ENDPOINTS[provider]
        assert transport.settings["max_retries"] == 0
        row = next(
            r
            for r in registry(roster_profile=PROFILE, mode_profile="off-or-minimum-v1")
            if r["model"] == model
        )
        assert row["mapping"] == transport.request_mapping
        for key, value in row["request_settings"].items():
            assert transport.settings[key] == value
        if provider == "openai":
            assert all(
                c["reasoning"] == {"effort": minimum_effort(model)} for c in mock.calls
            )
            assert all(t["strict"] is False for t in mock.calls[0]["tools"])
        if provider == "anthropic":
            assert all(
                c["thinking"] == {"type": "adaptive"}
                and c["output_config"] == {"effort": "low"}
                for c in mock.calls
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


@pytest.mark.parametrize("provider,model", LATEST_HTTP_MODELS)
def test_wrong_response_model_is_rejected(tmp_path, provider, model):
    def wrapper(mock):
        def handle(request):
            body = json.loads(mock.handle(request).content)
            body["model"] = "wrong-model"
            return httpx.Response(200, json=body)

        return handle

    budget, mock, transport, trial = fixture(
        tmp_path, provider, model=model, handler=wrapper, mode_profile="off-or-minimum-v1"
    )
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(mock.calls) == 1
    finally:
        transport.close()
        budget.close()


def admission():
    model = "gpt-6-luna"
    slot = next(s for s in assignments("latest", PROFILE) if s["model"] == model)
    profile = {
        "endpoint": ENDPOINTS["openai"],
        "input_max": 262144,
        "output_max": 128000,
        "context_max": 1050000,
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "network_timeout_seconds": 17,
        "rate_evidence_sha256": "a" * 64,
        "bounds_evidence_sha256": "b" * 64,
        "guarantees": dict.fromkeys(GUARANTEES, True),
        "pricing_policy": "conservative-all-tier-envelope-v1",
    }
    return {
        "schema": "three-flow-admission-v1",
        "approved": False,
        "mode": "synthetic",
        "source_sha": "c" * 40,
        "runtime": runtime_identity(),
        "registry": registry(roster_profile=PROFILE, mode_profile="off-or-minimum-v1"),
        "roster_profile": PROFILE,
        "mode_profile": "off-or-minimum-v1",
        "campaign_id": "latest",
        "attempt_id": "first",
        "campaign_root": "/synthetic/campaign",
        "custody_root": "/synthetic/custody",
        "credential_file": "/synthetic/not-read",
        "cap_usd": "1000",
        "resource_policy": RESOURCE_POLICY,
        "selected": [slot["trial_id"]],
        "profiles": {model: profile},
        "assets": None,
        "operation": "initial",
        "original_claim_sha256": None,
        "journal_sha256": None,
        "pending": [slot["trial_id"]],
        "corrections": {},
        "parent_gates": True,
        "no_intervening_spend": True,
    }


def test_latest_factory_campaign_and_grouped_reports(tmp_path):
    from tools.three_flow_admission import VARIABLES
    from tools.three_flow_campaign import run_campaign
    from tools.three_flow_live_factory import live_factory
    from tools.three_flow_mock import SDKMockTransport

    a = admission()
    template = a["profiles"]["gpt-6-luna"]
    slots = [
        s
        for s in assignments("latest", PROFILE)
        if s["provider"] != "river"
        and s["flow"] == "commerce"
        and s["scenario"] == "UK_NORMAL"
    ]
    for slot in slots:
        model = slot["model"]
        context, i, o = LATEST_HTTP_ENVELOPES.get(model, (1000000, "1", "2"))
        a["profiles"][model] = dict(
            template,
            endpoint=ENDPOINTS[slot["provider"]],
            context_max=context,
            input_usd_per_million=i,
            output_usd_per_million=o,
        )

    class Network:
        def http(self, slot):
            return SDKMockTransport(slot["provider"], slot["flow"], model=slot["model"])

    factory = live_factory(
        a, dict.fromkeys(VARIABLES.values(), "dummy-not-a-credential"), network=Network()
    )
    root = tmp_path / "campaign"
    rows = run_campaign(
        root,
        campaign_id="latest",
        slots=slots,
        roster_profile=PROFILE,
        transport_factory=factory,
    )
    assert len(rows) == len(LATEST_HTTP_MODELS)
    assert all(r["classification"] == "scored" for r in rows)
    manifest = json.loads((root / "assignment.json").read_text())
    assert manifest["roster_profile"] == PROFILE
    assert len(manifest["all_initial_slots"]) == len(selected_roster(PROFILE)) * len(
        CONFIGURATIONS
    )
    report = json.loads((root / "scorecard.json").read_text())
    assert {r["model"] for r in report["rows"]} == {m for _, m in LATEST_HTTP_MODELS}
    assert len(list(root.glob("model-*-commerce/scorecard.json"))) == len(
        LATEST_HTTP_MODELS
    )


def test_latest_admission_binding_and_legacy_roster():
    a = admission()
    assert _validate(a, "c" * 40, synthetic=True, historical=False) is a
    for mutate in (
        lambda x: x.update(source_sha="d" * 40),
        lambda x: x["registry"][-1].update(model="wrong-model"),
        lambda x: x["profiles"]["gpt-6-luna"].update(input_usd_per_million="0.1"),
        lambda x: x.update(roster_profile="wrong-roster"),
        lambda x: x.pop("roster_profile"),
    ):
        bad = copy.deepcopy(a)
        mutate(bad)
        with pytest.raises(ValueError):
            _validate(bad, "c" * 40, synthetic=True, historical=False)
    assert len(ROSTER) == 14
    assert len(assignments("old")) == 616
    assert tuple((r["provider"], r["model"]) for r in registry()) == ROSTER
    latest = selected_roster(PROFILE)
    assert ("river", "Qwen/Qwen3.5-122B-A10B-FP8") in ROSTER
    assert ("river", "Qwen/Qwen3.5-122B-A10B-FP8") not in latest
    assert ("anthropic", "claude-fable-5-1") in latest
    assert preflight(PROFILE)["initial_assignments"] == len(latest) * len(CONFIGURATIONS)


@pytest.mark.parametrize("model", LATEST_HTTP_ENVELOPES)
def test_known_rates_cannot_be_lowered(model):
    context, input_rate, output_rate = LATEST_HTTP_ENVELOPES[model]
    p = {
        "context_max": context,
        "output_max": 128000,
        "input_usd_per_million": input_rate,
        "output_usd_per_million": output_rate,
    }
    validate_latest_http_profile(p, model)
    for key in ("input_usd_per_million", "output_usd_per_million"):
        with pytest.raises(ValueError):
            validate_latest_http_profile(dict(p, **{key: "0.001"}), model)
    validate_latest_http_profile(
        dict(p, input_usd_per_million="1000", output_usd_per_million="1000"), model
    )
