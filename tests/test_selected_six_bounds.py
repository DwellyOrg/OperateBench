"""Closed selected-six numeric bounds and Mistral request output enforcement."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tools.three_flow_admission import GUARANTEES, digest, validate_profile
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_profiles import ENDPOINTS, SELECTED_SIX_PROFILE
from tools.three_flow_selected import ACCOUNTING_POLICY, BOUNDS, PRICING_POLICY


def mistral_profile() -> dict[str, Any]:
    evidence = {
        "category_rates": {
            "input": "1.36", "cache_read": "0.14", "cache_write": None, "output": "4.18"
        },
        "source": "https://mistral.ai/pricing",
        "sha256": "a" * 64,
        "interpretation": (
            "published-category-maxima; estimated reported-token cost; "
            "unknown additional fees"
        ),
    }
    return {
        "endpoint": ENDPOINTS["mistral"],
        "context_max": 1048576,
        "input_max": None,
        "output_max": 262144,
        "input_usd_per_million": "1.36",
        "output_usd_per_million": "4.18",
        "network_timeout_seconds": None,
        "rate_evidence_sha256": digest(evidence),
        "bounds_evidence_sha256": "b" * 64,
        "guarantees": dict.fromkeys(GUARANTEES, False),
        "pricing_policy": PRICING_POLICY,
        "execution_policy": "three-flow-optional-limits-v1",
        "bounds_policy": {
            "mode": "published-context-diagnostic-v1",
            "source": "https://api.mistral.ai/v1/models/mistral-large-4",
            "scope": "endpoint",
            "interpretation": "published-integer",
        },
        "accounting_policy": dict(ACCOUNTING_POLICY),
        "tariff_evidence": evidence,
    }


def test_selected_six_exact_bounds() -> None:
    assert BOUNDS == {
        "gpt-6.1-terra": (1050000, 922000, 128000),
        "claude-haiku-5-5": (1000000, None, 128000),
        "mistral-large-4": (1048576, None, 262144),
        "grok-4.7": (500000, None, None),
        "GLM-5.3": (1048576, None, 128000),
        "moonshotai/Kimi-K3": (1048576, None, 32768),
    }
    p = mistral_profile()
    validate_profile(
        p, "mistral", model="mistral-large-4", roster_profile=SELECTED_SIX_PROFILE
    )
    assert all(value is False for value in p["guarantees"].values())


@pytest.mark.parametrize(
    "context,input_max,output",
    [
        (1000000, None, None),
        (1000000, None, 262144),
        (1048576, None, None),
        (1048575, None, 262144),
        (1048577, None, 262144),
        (1048576, 1048576, 262144),
        (1048576, None, 262143),
        (1048576, None, 262145),
        (1048576, None, 982766),
        (1048576, None, 30000),
    ],
)
def test_mistral_bounds_remain_closed(
    context: int, input_max: int | None, output: int | None
) -> None:
    p = mistral_profile()
    p.update(context_max=context, input_max=input_max, output_max=output)
    with pytest.raises(ValueError, match="published route/bounds differ"):
        validate_profile(
            p, "mistral", model="mistral-large-4", roster_profile=SELECTED_SIX_PROFILE
        )


def test_mistral_output_and_context_enforced() -> None:
    guard = object.__new__(AdmittedGuard)
    guard.profile = mistral_profile()
    guard.budget = cast(Any, SimpleNamespace(totals=lambda: {"remaining": None}))
    assert guard.output_bound(100) == 262144
    assert guard.output_bound(1048576 - 17) == 17
    guard.check_context(1048576 - 262144, 262144)
    with pytest.raises(ValueError, match="context/output"):
        guard.check_context(100, 262145)
    with pytest.raises(ValueError, match="context/output"):
        guard.check_context(1048576 - 16, 17)
