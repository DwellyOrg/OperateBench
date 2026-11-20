"""The new cohort is descriptive only until its routes are qualified."""

import pytest

from tools.three_flow_campaign import CONFIGURATIONS, assignments, preflight
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE, registry, selected_roster


def test_census_and_closed_dispatch():
    roster = selected_roster(LATE_NOVEMBER_PROFILE)
    assert roster == (
        ("gemini", "gemini-4-hydrogen"),
        ("gemini", "gemini-4-neon"),
        ("meta", "muse-ember-1.0"),
        ("river", "ricursiveAI/Ripple-65B-Preview-NVFP4"),
        ("river_http", "moonshotai/Kimi-K3.1-Pro"),
    )
    slots = assignments("offline", LATE_NOVEMBER_PROFILE)
    assert len(slots) == len({s["trial_id"] for s in slots}) == 220
    for provider, model in roster:
        assert {
            (s["flow"], s["scenario"])
            for s in slots
            if (s["provider"], s["model"]) == (provider, model)
        } == set(CONFIGURATIONS)
    report = preflight(LATE_NOVEMBER_PROFILE)
    assert report["initial_assignments"] == 220
    assert report["global_cap_usd"] is None
    assert report["dispatch_status"] == "external admission required"
    assert not report["admitted"] and not report["authority_created"]
    with pytest.raises(ValueError, match="exact native mode"):
        registry(roster_profile=LATE_NOVEMBER_PROFILE)
    rows = registry(roster_profile=LATE_NOVEMBER_PROFILE, mode_profile="late-november-v1")
    assert len(rows) == 5 and all(row["admitted"] is False for row in rows)


def test_new_roster_budget_exact_cap_and_restart(tmp_path):
    from decimal import Decimal

    from operatebench.providers.cost import CostCapExceededError
    from tests.test_three_flow_budget import policy
    from tools.three_flow_budget import CampaignBudget

    tmp_path.chmod(0o700)
    args = {
        "campaign_id": "offline",
        "cap": Decimal("3.25"),
        "roster_profile": LATE_NOVEMBER_PROFILE,
    }
    budget = CampaignBudget(tmp_path, create=True, **args)
    attempt = budget.reserve("cell", input_bound=0, output_bound=3, policy=policy())
    budget.close()
    budget = CampaignBudget(tmp_path, **args)
    try:
        assert budget.records[attempt]["state"] == "unknown"
        assert budget.totals()["remaining"] == Decimal("0.25")
        with pytest.raises(CostCapExceededError):
            budget.reserve("second", input_bound=0, output_bound=1, policy=policy())
    finally:
        budget.close()
    with pytest.raises(ValueError, match="foreign campaign or cap"):
        CampaignBudget(tmp_path, **{**args, "cap": Decimal("4")})


def test_new_roster_null_cap_needs_source_bound_consent(tmp_path):
    from tools.three_flow_budget import OPTIONAL_PROFILE, CampaignBudget

    tmp_path.chmod(0o700)
    args = {
        "campaign_id": "offline",
        "cap": None,
        "roster_profile": LATE_NOVEMBER_PROFILE,
    }
    with pytest.raises(ValueError, match="optional limits"):
        CampaignBudget(tmp_path, create=True, **args)
    consent = {
        "mode": OPTIONAL_PROFILE,
        "owner_accepted": True,
        "owner_evidence_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "historical_liability_manifest_sha256": "c" * 64,
    }
    budget = CampaignBudget(tmp_path, create=True, limit_policy=consent, **args)
    try:
        assert budget.totals()["remaining"] is None
    finally:
        budget.close()


def test_candidate_contracts_and_native_ripple_parser():
    from tests.ripple_parser_fixture import parse_ripple_content_blocks
    from tools.three_flow_late_november import CANDIDATES, META_SHORT_TIER

    assert set(CANDIDATES) == {
        model for _, model in selected_roster(LATE_NOVEMBER_PROFILE)
    }
    assert CANDIDATES["muse-ember-1.0"].reasoning == "minimal"
    assert CANDIDATES["muse-ember-1.0"].output == 524288
    assert META_SHORT_TIER["input_tokens_at_most"] == 200000
    ripple = CANDIDATES["ricursiveAI/Ripple-65B-Preview-NVFP4"]
    assert (ripple.context, ripple.output, ripple.reasoning) == (
        262144,
        30000,
        "provider-default",
    )
    parts, calls = parse_ripple_content_blocks(
        '<think>private</think>\n<tool_call>{"name":"act","arguments":{"x":1}}</tool_call>'
    )
    assert parts[0]["type"] == "thinking"
    assert calls[0]["function"]["name"] == "act"
    _, bad = parse_ripple_content_blocks(
        '<tool_call>{"name":"act","arguments":{},"extra":1}</tool_call>'
    )
    assert "error" in bad[0]
