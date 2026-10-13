"""Offline Meta selection/accounting proof; never launches provider work."""

from collections import Counter
from decimal import Decimal

import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.providers.cost import CostCapExceededError
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import assignments, preflight
from tools.three_flow_meta import validate_meta_profile
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
        "bounds_policy": {
            "mode": "published-context-diagnostic-v1",
            "source": "https://dev.meta.ai/docs",
            "scope": "endpoint",
            "interpretation": "published-integer",
        },
        "network_timeout_seconds": None,
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
        ("network_timeout_seconds", 17),
        ("input_max", 100000),
        ("output_usd_per_million", "1"),
        ("endpoint", "https://api.openai.com/v1/responses"),
        ("execution_policy", "optional-execution-v1"),
        ("owner_output_cap_tokens", 4096),
        ("bounds_policy", {**p["bounds_policy"], "source": "https://example.com"}),
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


def test_two_models_retries_and_restart_share_cap(tmp_path):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="meta-new",
        cap=Decimal("990"),
        roster_profile=META_ROSTER_PROFILE,
        create=True,
    )
    with pytest.raises(FileExistsError):
        CampaignBudget(
            tmp_path,
            campaign_id=META_MODELS[1],
            cap=Decimal("990"),
            roster_profile=META_ROSTER_PROFILE,
            create=True,
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
                "campaign-retry",
                input_bound=0,
                output_bound=3000000,
                policy=policies[META_MODELS[0]],
            )
        assert budget.exposure == 940
    finally:
        budget.close()
    recovered = CampaignBudget(
        tmp_path,
        campaign_id="meta-new",
        cap=Decimal("990"),
        roster_profile=META_ROSTER_PROFILE,
    )
    try:
        assert recovered.totals()["unknown"] == 940
        with pytest.raises(CostCapExceededError):
            recovered.reserve(
                "retry",
                input_bound=0,
                output_bound=3000000,
                policy=policies[META_MODELS[0]],
            )
    finally:
        recovered.close()
    with pytest.raises(ValueError):
        CampaignBudget(
            tmp_path,
            campaign_id="foreign-series",
            cap=Decimal("990"),
            roster_profile=META_ROSTER_PROFILE,
        )


def test_admission_requires_shared_cap_and_new_authority(monkeypatch):
    from tests.test_three_flow_latest_http import admission
    from tools.three_flow_admission import GUARANTEES, validate

    a = admission()
    a["cap_usd"] = "990"
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
            network_timeout_seconds=None,
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
    from tools import three_flow_admission

    calls = []
    original = three_flow_admission.validate_meta_profile

    def checked(p, provider, model):
        calls.append(model)
        original(p, provider, model)

    monkeypatch.setattr(three_flow_admission, "validate_meta_profile", checked)
    validate(a, a["source_sha"], synthetic=True)
    assert calls == [s["model"] for s in slots]
    calls.clear()
    first_model = slots[0]["model"]
    invalid = {
        **a,
        "profiles": {
            **a["profiles"],
            first_model: {**a["profiles"][first_model], "network_timeout_seconds": 17},
        },
    }
    with pytest.raises(ValueError, match="Meta requires exact tariff/context"):
        validate(invalid, a["source_sha"], synthetic=True)
    assert calls == [first_model]
    for bad in [
        {**a, "cap_usd": "1000"},
        {**a, "cap_usd": "989"},
        {**a, "cap_usd": 990},
        {**a, "cap_usd": "2000"},
        {**a, "cap_usd": None},
        {**a, "limit_policy": {}},
        {**a, "original_claim_sha256": "a" * 64},
    ]:
        with pytest.raises(ValueError):
            validate(bad, a["source_sha"], synthetic=True)


def test_null_timeout_requires_exact_meta_roster():
    from tests.test_three_flow_latest_http import admission
    from tools.three_flow_admission import GUARANTEES, validate_profile

    meta = dict(
        profile(META_MODELS[0]),
        rate_evidence_sha256="a" * 64,
        bounds_evidence_sha256="b" * 64,
        pricing_policy="conservative-all-tier-envelope-v1",
        guarantees={**dict.fromkeys(GUARANTEES, True), "provider_bounds_verified": False},
    )
    validate_profile(
        meta, "meta", model=META_MODELS[0], roster_profile=META_ROSTER_PROFILE
    )
    for model, roster in [
        (META_MODELS[0], "legacy-v1"),
        ("muse-wonder-1.0-contributor", META_ROSTER_PROFILE),
    ]:
        with pytest.raises(ValueError):
            validate_profile(meta, "meta", model=model, roster_profile=roster)

    p = next(iter(admission()["profiles"].values()))
    p["network_timeout_seconds"] = None
    with pytest.raises(ValueError, match="network timeout"):
        validate_profile(p, "openai")


def test_invalid_meta_refused_before_custody(tmp_path, monkeypatch):
    from tests.test_three_flow_latest_http import admission
    from tools import three_flow_live

    a = admission()
    a["cap_usd"] = "990"
    a["roster_profile"] = META_ROSTER_PROFILE
    path = tmp_path / "invalid.json"
    path.write_text("{}")
    monkeypatch.setattr(
        three_flow_live, "read_private", lambda _: __import__("json").dumps(a).encode()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid admission reached custody")

    monkeypatch.setattr(three_flow_live, "private_directory", forbidden)
    with pytest.raises(ValueError):
        three_flow_live.consume(path, a["source_sha"], network=object())


def test_meta_preflight_preserves_other_profiles():
    from tools.three_flow_profiles import LATEST_ROSTER_PROFILE

    meta = preflight(META_ROSTER_PROFILE)
    assert meta["global_cap_usd"] == "990"
    assert meta["diagnostic_ceiling_usd"] == "10"
    assert meta["initial_assignments"] == 88
    for roster in ("legacy-v1", LATEST_ROSTER_PROFILE):
        report = preflight(roster)
        assert report["global_cap_usd"] == "1000"
        assert "diagnostic_ceiling_usd" not in report


@pytest.mark.parametrize("cap", [None, Decimal("1000"), Decimal("989")])
def test_meta_budget_rejects_other_caps_before_journal(tmp_path, cap):
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError, match="shared hard USD 990"):
        CampaignBudget(
            tmp_path,
            campaign_id="meta-new",
            cap=cap,
            roster_profile=META_ROSTER_PROFILE,
            create=True,
        )
    assert not (tmp_path / "campaign-budget.jsonl").exists()
