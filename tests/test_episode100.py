from decimal import Decimal

from operatebench.agents.pricing import LifecyclePricingPolicy
from tools import diagnose_aggregate_budget as runner
from tools import episode100 as profile


def test_explicit_profile_not_historical_fifty_call_identity():
    m = runner.module("haiku45")
    spec = profile.load_profile_spec(m)
    spec.verify_identity()
    assert spec.spec_digest_sha256 != m.load_spec(m.FIXTURE).spec_digest_sha256
    assert spec.policy.max_turns_per_invocation == 101
    policy = LifecyclePricingPolicy(
        policy_id=m.FIXED_CANARY_POLICY_ID,
        input_usd_per_mtok=m.FIXED_CANARY_INPUT_USD_PER_MTOK,
        output_usd_per_mtok=m.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
        rate_source=m.FIXED_CANARY_RATE_SOURCE,
    )
    controls = profile.controls_for(m, policy, Decimal("1"))
    assert controls.max_provider_calls == 100
    assert controls.token_hard_cap != m.TOKEN_HARD_CAP
    assert profile.AGENT_ID != m.CANARY_AGENT_ID
