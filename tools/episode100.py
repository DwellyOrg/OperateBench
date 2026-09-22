"""Distinct candidate profile: at most 100 response decisions per whole episode.

RETRIEVE is a decision, not a free inner loop. An HTTP failure ends the episode;
there are no retries. The ledger's provider-budget taxonomy is retained, with
an explicit decision-limit cause in the controller outcome. This is not the
historical fixed-canary spec or authorization.
"""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from math import ceil
from types import ModuleType
from typing import Any

from operatebench.agents.pricing import LifecyclePricingPolicy

PROFILE = "synthetic-demo-episode100-v1"
AGENT_ID = "synthetic-demo-model-episode100-v1"
MAX_DECISIONS = 100


def load_profile_spec(module: ModuleType) -> Any:
    original = module.load_spec(module.FIXTURE)
    module.check_spec_identity(original)
    # 101 lets the episode-wide recorder refuse decision 101 *before* the
    # business runner could synthesize a scored invocation-limit failure.
    spec = replace(
        original, policy=replace(original.policy, max_turns_per_invocation=101)
    )
    return replace(spec, spec_digest_sha256=spec.compute_digest())


def controls_for(module: ModuleType, policy: LifecyclePricingPolicy, cap: Decimal) -> Any:
    # A redundant finite bound required by the legacy ledger schema, derived
    # from this NEW cap rather than any old 50-call native-input estimate.
    # Each admitted reservation <= aggregate cap; at most 100 reservations.
    minimum_rate = min(
        Fraction(policy.input_usd_per_mtok), Fraction(policy.output_usd_per_mtok)
    )
    tokens = ceil(MAX_DECISIONS * Fraction(cap) * 1_000_000 / minimum_rate)
    return replace(
        module.controls_for(policy, cap=cap),
        max_provider_calls=100,
        token_hard_cap=tokens,
        wall_clock_deadline_seconds=100 * module.TURN_DEADLINE_SECONDS,
    )
