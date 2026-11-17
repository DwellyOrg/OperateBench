"""Opt-in OpenAI62 qualification; official SDK with synthetic reference wires only."""

import pytest

from tests.test_selected_six_bounds import mistral_profile
from tests.test_selected_six_http import fixture
from tests.test_three_flow_live import no_network

__all__ = ["no_network"]
from tools.three_flow_admission import digest, validate_profile
from tools.three_flow_campaign import CONFIGURATIONS, assignments, preflight, reports
from tools.three_flow_profiles import (
    OPENAI62_MODELS,
    OPENAI62_PROFILE,
    registry,
    selected_roster,
    selection_for,
)


def profile(model):
    p = mistral_profile()
    rates = (
        ("16", "0.8", "40", "60")
        if model == "gpt-6.2-astra"
        else ("4", "0.2", "10", "15")
    )
    p.update(
        endpoint="https://api.openai.com/v1/responses",
        context_max=1050000,
        input_max=922000,
        output_max=128000,
        input_usd_per_million=rates[2],
        output_usd_per_million=rates[3],
        pricing_policy="openai62-standard-token-envelope-v1",
    )
    p["tariff_evidence"].update(
        source="https://developers.openai.com/api/docs/pricing.md",
        category_rates=dict(
            zip(("input", "cache_read", "cache_write", "output"), rates, strict=True)
        ),
    )
    p["bounds_policy"]["source"] = (
        f"https://developers.openai.com/api/docs/models/{model}.md"
    )
    p["rate_evidence_sha256"] = digest(p["tariff_evidence"])
    return p


def test_complete_product_and_legacy():
    assert selected_roster(OPENAI62_PROFILE) == OPENAI62_MODELS
    slots = assignments("offline62", OPENAI62_PROFILE)
    assert len(slots) == len({s["trial_id"] for s in slots}) == 88
    for _, model in OPENAI62_MODELS:
        lane = [s for s in slots if s["model"] == model]
        assert {(s["flow"], s["scenario"]) for s in lane} == set(CONFIGURATIONS)
        assert [
            sum(s["flow"] == f for s in lane)
            for f in ("maintenance", "commerce", "compliance")
        ] == [3, 16, 25]
        assert all(s["repetition"] == 1 and s["rerun_of"] is None for s in lane)
        assert selection_for("openai", model, "legacy-v1") is None
    assert preflight(OPENAI62_PROFILE)["global_cap_usd"] is None
    assert not set(OPENAI62_MODELS) & set(selected_roster())
    assert not set(OPENAI62_MODELS) & set(selected_roster("latest-http-v1"))
    with pytest.raises(ValueError):
        selected_roster("openai62")
    with pytest.raises(ValueError):
        registry(roster_profile=OPENAI62_PROFILE)
    with pytest.raises(ValueError):
        registry(
            roster_profile=OPENAI62_PROFILE,
            mode_profile="off-or-minimum-v1",
            transport_policy="paced-safe-errors-v1",
        )
    assert selection_for("openai", "gpt-6.2-unknown", "off-or-minimum-v1") is None


@pytest.mark.parametrize("_,model", OPENAI62_MODELS)
def test_profile_closed(_, model):
    p = profile(model)
    validate_profile(p, "openai", model=model, roster_profile=OPENAI62_PROFILE)
    for field, value in (
        ("input_max", 921999),
        ("output_max", 4096),
        ("context_max", 1000000),
        ("network_timeout_seconds", 60),
        ("input_usd_per_million", "8"),
        ("pricing_policy", "selected-six-published-token-envelope-v1"),
        ("execution_policy", "legacy-v1"),
        ("extra", True),
    ):
        with pytest.raises(ValueError):
            validate_profile(
                dict(p, **{field: value}),
                "openai",
                model=model,
                roster_profile=OPENAI62_PROFILE,
            )
    p["tariff_evidence"]["source"] = "https://example.com/pricing"
    p["rate_evidence_sha256"] = digest(p["tariff_evidence"])
    with pytest.raises(ValueError):
        validate_profile(p, "openai", model=model, roster_profile=OPENAI62_PROFILE)
    with pytest.raises(ValueError):
        validate_profile(
            profile(model), "openai", model=model, roster_profile="selected-six-http-v1"
        )


@pytest.mark.parametrize("_,model", OPENAI62_MODELS)
@pytest.mark.parametrize("flow", ["maintenance", "commerce", "compliance"])
def test_sdk_engine_and_zero_api_replay(tmp_path, _, model, flow):
    from tests.test_selected_six_http import SelectedMock
    from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial

    budget, t, _, network = fixture(
        tmp_path, "openai", model, roster_profile=OPENAI62_PROFILE
    )
    t.guard.profile.update(profile(model))
    slot = next(
        s
        for s in assignments("selected-test", OPENAI62_PROFILE)
        if s["model"] == model and s["flow"] == flow
    )
    mock = network.transports[0]
    mock.native = SelectedMock("openai", model, flow).native
    trial = Trial(
        "selected-test",
        slot["trial_id"],
        flow,
        slot["scenario"],
        model,
        128000,
        provider_identity(t),
    )
    try:
        assert t.guard.output_bound(922000) == 128000
        with pytest.raises(ValueError):
            t.guard.output_bound(922001)
        record = run_trial(trial, t, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert t.client.max_retries == 0
        assert t.wire.endpoint == "https://api.openai.com/v1/responses"
        for call in mock.calls:
            assert call["model"] == model
            assert call["reasoning"] == {"effort": "low"}
            assert call["service_tier"] == "default"
            assert call["max_output_tokens"] == 128000
            assert all(tool["strict"] is False for tool in call["tools"])
            assert not {"background", "previous_response_id", "temperature"} & call.keys()
        count = len(mock.calls)
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(mock.calls) == count
        assert all(r["state"] == "settled" for r in budget.records.values())
        assert t.settings["service_tier"] == "default"
    finally:
        t.close()
        budget.close()


@pytest.mark.parametrize("_,model", OPENAI62_MODELS)
def test_wrong_model_excluded(tmp_path, _, model):
    from operatebench.agents.transport import ProviderFailure
    from tools.three_flow_runtime import run_trial

    budget, t, trial, network = fixture(
        tmp_path, "openai", model, "model", roster_profile=OPENAI62_PROFILE
    )
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, t, output=tmp_path / "trial")
        assert len(network.transports[0].calls) == 1
        assert all(r["state"] == "unknown" for r in budget.records.values())
    finally:
        t.close()
        budget.close()


def test_all_grouped_reports(tmp_path):
    slots = assignments("offline62", OPENAI62_PROFILE)
    # Technical exclusions still require complete grouped reporting.
    rows = [
        dict(
            s,
            classification="technical_exclusion",
            reason="synthetic",
            estimated_usd="0",
            unknown_liability_usd="0",
        )
        for s in slots
    ]
    reports(tmp_path, rows, {})
    assert len(list(tmp_path.glob("model-*/scorecard.md"))) == 6


def test_full88_source_bound_operator_mock(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from tests.test_selected_six_http import Network
    from tests.test_three_flow_live import cli, write
    from tests.test_three_flow_live import fixture as admission_fixture
    from tests.test_three_flow_optional_limits import consent
    from tools import three_flow_live as live
    from tools.three_flow_admission import validate

    # Synthetic authority only: this test must also run in an uncommitted checkout.
    sha = "b" * 40
    monkeypatch.setattr(live, "source_identity", lambda expected=None: sha)
    path, a, _ = admission_fixture(tmp_path)
    slots = assignments(a["campaign_id"], OPENAI62_PROFILE)
    a.update(
        roster_profile=OPENAI62_PROFILE,
        mode_profile="off-or-minimum-v1",
        registry=registry(
            roster_profile=OPENAI62_PROFILE, mode_profile="off-or-minimum-v1"
        ),
        selected=[s["trial_id"] for s in slots],
        pending=[s["trial_id"] for s in slots],
        profiles={model: profile(model) for _, model in OPENAI62_MODELS},
        cap_usd=None,
        limit_policy={**consent(), "source_sha": sha},
    )
    assert validate(a, sha, synthetic=True) == a
    for field, value in (
        ("source_sha", "e" * 40),
        ("roster_profile", "selected-six-http-v1"),
        ("selected", ["initial-999-00"]),
    ):
        with pytest.raises(ValueError):
            validate(dict(a, **{field: value}), sha, synthetic=True)
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 0
    assert len(network.transports) == 88
    root = Path(a["campaign_root"])
    rows = json.loads((root / "scorecard.json").read_text())["rows"]
    assert len(rows) == 88
    assert all(r["evidence_origin"] == "SDK_MOCK_NOT_LLM" for r in rows)
    assert len(list(root.glob("model-*/scorecard.json"))) == 6
    calls = sum(len(t.calls) for t in network.transports)
    assert cli(path, a, network) == 2
    assert sum(len(t.calls) for t in network.transports) == calls
