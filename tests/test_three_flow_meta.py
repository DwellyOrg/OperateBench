"""Offline Meta selection/accounting proof; never launches provider work."""

from collections import Counter
from decimal import Decimal

import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.providers.cost import CostCapExceededError
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import assignments
from tools.three_flow_meta import campaign_transport, validate_meta_profile
from tools.three_flow_profiles import (
    ENDPOINTS,
    META_ENVELOPES,
    META_MODELS,
    META_ROSTER_PROFILE,
    ROSTER,
    registry,
    selected_roster,
)


def profile(model):
    context, ir, out = META_ENVELOPES[model]
    return {
        "endpoint": ENDPOINTS["meta"],
        "context_max": context,
        "input_max": None,
        "output_max": None,
        "bounds_policy": {"mode": "published-context-diagnostic-v1"},
        "input_usd_per_million": ir,
        "output_usd_per_million": out,
    }


def test_roster_exact_88_and_history_unchanged():
    slots = assignments("meta-fresh-series", META_ROSTER_PROFILE)
    assert len(slots) == len({s["trial_id"] for s in slots}) == 88
    assert selected_roster() == ROSTER
    assert len(assignments("historical")) == 616
    for model in META_MODELS:
        subset = [s for s in slots if s["model"] == model]
        assert Counter(s["flow"] for s in subset) == {
            "maintenance": 3,
            "commerce": 16,
            "compliance": 25,
        }
        assert all(s["repetition"] == 1 and s["rerun_of"] is None for s in subset)
    assert {s["campaign_id"] for s in slots} == {"meta-fresh-series"}


def test_public_synthetic_case_parity():
    # Freeze the public source literals independently of assignments/CONFIGURATIONS.
    # Maintenance V1-V3; commerce __init__.py and compliance spec.py reference cases.
    expected = [
        ("maintenance", "V1"),
        ("maintenance", "V2"),
        ("maintenance", "V3"),
        ("commerce", "UK_NORMAL"),
        ("commerce", "UK_LABEL_UNKNOWN"),
        ("commerce", "UK_MISSING_DISCLOSURE"),
        ("commerce", "UK_PAYMENT_FAILED"),
        ("commerce", "DE_NORMAL"),
        ("commerce", "DE_WITHHOLDING"),
        ("commerce", "DE_MISSING_DISCLOSURE"),
        ("commerce", "DE_FAULT_REVIEW"),
        ("commerce", "CA_POLICY"),
        ("commerce", "CA_NO_POLICY"),
        ("commerce", "CA_UNKNOWN_DISPLAY"),
        ("commerce", "CA_FAULT_REVIEW"),
        ("commerce", "AU_MAJOR"),
        ("commerce", "AU_MINOR"),
        ("commerce", "AU_CHANGE_OF_MIND"),
        ("commerce", "AU_UNCERTAIN_FAULT"),
        ("compliance", "EN_normal"),
        ("compliance", "EN_remediation"),
        ("compliance", "EN_access"),
        ("compliance", "EN_reopen"),
        ("compliance", "EN_qualification"),
        ("compliance", "SC_normal"),
        ("compliance", "SC_remediation"),
        ("compliance", "SC_access"),
        ("compliance", "SC_reopen"),
        ("compliance", "SC_qualification"),
        ("compliance", "AU_normal"),
        ("compliance", "AU_remediation"),
        ("compliance", "AU_access"),
        ("compliance", "AU_reopen"),
        ("compliance", "AU_qualification"),
        ("compliance", "NZ_normal"),
        ("compliance", "NZ_remediation"),
        ("compliance", "NZ_access"),
        ("compliance", "NZ_reopen"),
        ("compliance", "NZ_qualification"),
        ("compliance", "SC_checklist"),
        ("compliance", "AU_gas_expired"),
        ("compliance", "NZ_assessor_only"),
        ("compliance", "EN_wrong_property"),
        ("compliance", "EN_no_response"),
    ]
    assert len(expected) == 44
    for model in META_MODELS:
        assert [
            (s["flow"], s["scenario"])
            for s in assignments("fresh", META_ROSTER_PROFILE)
            if s["model"] == model
        ] == expected


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "muse-wonder-1.0"),
        ("meta", "muse-spark-1.0"),
        ("meta", "muse-wonder-full"),
        ("meta", "muse-wonder-1.0-contributor"),
        ("meta", "muse-spark-1.3-contributor"),
        ("meta", "glimmer"),
    ],
)
def test_refuse_wrong_provider_and_nonstandard(provider, model):
    with pytest.raises(ValueError):
        validate_meta_profile(profile(META_MODELS[0]), provider, model)


@pytest.mark.parametrize("model", META_MODELS)
def test_exact_prices_unknown_output_and_settings(model):
    p = profile(model)
    validate_meta_profile(p, "meta", model)
    for key, value in [
        ("output_max", 128000),
        ("input_usd_per_million", "8"),
        ("context_max", 1000000),
    ]:
        bad = {**p, key: value}
        with pytest.raises(ValueError):
            validate_meta_profile(bad, "meta", model)
    rows = registry(roster_profile=META_ROSTER_PROFILE, mode_profile="off-or-minimum-v1")
    assert {r["model"] for r in rows} == set(META_MODELS)
    assert all(r["request_settings"]["reasoning"] == {"effort": "minimal"} for r in rows)
    assert all(
        r["request_settings"]["native_reasoning_continuation"] is False for r in rows
    )
    with pytest.raises(ValueError):
        registry(roster_profile=META_ROSTER_PROFILE)


def test_two_models_diagnostics_and_restart_share_cap(tmp_path):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="meta-new", cap=Decimal("1000"), create=True
    )
    policies = {
        m: LifecyclePricingPolicy("meta-offline-" + m, Decimal(ir), Decimal(out))
        for m, (_, ir, out) in META_ENVELOPES.items()
    }
    try:
        first = budget.reserve(
            META_MODELS[0],
            input_bound=0,
            output_bound=30000000,
            policy=policies[META_MODELS[0]],
        )  # $600
        budget.forfeit(first)
        budget.reserve(
            META_MODELS[1],
            input_bound=0,
            output_bound=80000000,
            policy=policies[META_MODELS[1]],
        )  # $340
        with pytest.raises(CostCapExceededError):
            budget.reserve(
                "diagnostic",
                input_bound=0,
                output_bound=4000000,
                policy=policies[META_MODELS[0]],
            )
        assert budget.exposure == 940
    finally:
        budget.close()
    recovered = CampaignBudget(tmp_path, campaign_id="meta-new", cap=Decimal("1000"))
    try:
        assert recovered.totals()["unknown"] == 940
        with pytest.raises(CostCapExceededError):
            recovered.reserve(
                "retry",
                input_bound=0,
                output_bound=4000000,
                policy=policies[META_MODELS[0]],
            )
    finally:
        recovered.close()
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, campaign_id="foreign-series", cap=Decimal("1000"))


def test_launch_is_explicitly_blocked_pending_cross_lane_bridge():
    with pytest.raises(ValueError, match="integration is not ready"):
        campaign_transport()


def test_admission_requires_shared_cap_and_new_authority():
    from tests.test_three_flow_latest_http import admission
    from tools.three_flow_admission import GUARANTEES, validate

    a = admission()
    a["roster_profile"] = META_ROSTER_PROFILE
    a["registry"] = registry(
        roster_profile=META_ROSTER_PROFILE, mode_profile="off-or-minimum-v1"
    )
    slots = assignments(a["campaign_id"], META_ROSTER_PROFILE)
    a["selected"] = a["pending"] = [s["trial_id"] for s in slots]
    a["profiles"] = {}
    for model in META_MODELS:
        p = profile(model)
        p.update(
            network_timeout_seconds=17,
            rate_evidence_sha256="a" * 64,
            bounds_evidence_sha256="b" * 64,
            pricing_policy="conservative-all-tier-envelope-v1",
            guarantees={
                **dict.fromkeys(GUARANTEES, True),
                "provider_bounds_verified": False,
            },
        )
        p["bounds_policy"].update(
            source="https://dev.meta.ai/docs",
            scope="endpoint",
            interpretation="published-integer",
        )
        a["profiles"][model] = p
    validate(a, a["source_sha"], synthetic=True)
    for bad in [
        {**a, "cap_usd": "2000"},
        {**a, "cap_usd": None},
        {**a, "limit_policy": {}},
        {**a, "original_claim_sha256": "a" * 64},
    ]:
        with pytest.raises(ValueError):
            validate(bad, a["source_sha"], synthetic=True)


def test_unready_launch_refuses_before_credentials_or_consumption(tmp_path, monkeypatch):
    from tools import three_flow_admission, three_flow_live

    path = tmp_path / "offline-admission.json"
    path.write_text("{}")
    monkeypatch.setattr(three_flow_live, "read_private", lambda _: b"{}")
    monkeypatch.setattr(
        three_flow_admission,
        "validate",
        lambda *args, **kwargs: {"roster_profile": META_ROSTER_PROFILE},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("custody and credentials must remain untouched")

    monkeypatch.setattr(three_flow_live, "private_directory", forbidden)
    monkeypatch.setattr(three_flow_live, "private_fd", forbidden)
    monkeypatch.setattr(three_flow_live, "publish", forbidden)
    with pytest.raises(ValueError, match="integration is not ready"):
        three_flow_live.consume(path, "c" * 40)
    assert list(tmp_path.iterdir()) == [path]
