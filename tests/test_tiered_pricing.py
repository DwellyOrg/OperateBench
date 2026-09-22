"""Two providers price a long prompt differently, and the table has to say so.

xAI and OpenAI both publish a **tiered** standard rate for the models this
matrix pins: below a stated input-token threshold a request is priced at one
pair of rates, and at or above it the *whole request* — input and output — is
priced at a higher pair. The two vendors do not even agree on which side of the
threshold is which: xAI's page states that all tokens use the long rates once
the threshold is reached, and OpenAI's states ``≤272K input tokens`` against
``>272K input tokens``. Inclusivity is therefore a per-policy fact, not a
convention, and a build that guessed it would misprice exactly one token's worth
of requests at the boundary — in the direction that under-states spend.

The Mistral table is untiered and keeps its payload bytes exactly. Anthropic is
also untiered, but its current golden moved separately when the official source
withdrew an announced price-window change. The two tiered policies move their
version *and* their digest, because a run priced under the flat reading of the
same rates is not comparable with one priced under this.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from boundarybench.budget import (
    CostControls,
    RunCostGuard,
    check_stored_pricing_block,
    shipped_pricing_block,
)
from boundarybench.pricing import (
    PROVIDER_PRICING_POLICIES,
    LongContextTier,
    ModelPrice,
    PricingError,
    digest_of_policy_payload,
    measured_cost_usd,
    price_for,
    pricing_policy_digest,
    pricing_policy_payload,
)

#: The corrected current Anthropic digest, independently pinned here as well as
#: in ``test_provider_pricing_matrix``. Historical evidence is not rewritten.
ANTHROPIC_POLICY_DIGEST = (
    "f0b78a71619841f0c3e7d21a094258d24093b28d02744b5d3fb8ce6b7138561a"
)
#: The Mistral digest before tiering existed. Mistral publishes no tier for this
#: model, so its bytes must be untouched by the mechanism that prices one.
MISTRAL_POLICY_DIGEST = "c1b8eff206edb6ee06dc40f60f7d77e26d84a9b565c7a0462865930de9c76bbc"
#: The two digests that *must* move: a flat reading of these rates and a tiered
#: one are different tables, and a run recorded under the first must not
#: validate under the second.
SUPERSEDED_OPENAI_DIGEST = (
    "7c75433a5424fe997becc12c9904105bf1734025f2795b1fb55c3de9b2b8be7a"
)
SUPERSEDED_XAI_DIGEST = "ff2fadc8cb74d643a8842e250402b65051dce0029ea0901667bde9801be66d1b"


def _xai() -> ModelPrice:
    return price_for(provider="xai", model="grok-4.5")


def _openai() -> ModelPrice:
    return price_for(provider="openai", model="gpt-5.6-luna")


# -- the tier definitions themselves ------------------------------------------


def test_the_two_tiered_models_state_their_threshold_and_its_inclusivity() -> None:
    """The three numbers a tier is, and the one boolean that decides a boundary."""
    xai = _xai().long_context_tier
    assert isinstance(xai, LongContextTier)
    assert xai.input_token_threshold == 200_000
    # xAI: "all tokens in a request use the long rates once the threshold is
    # reached" — so a prompt of exactly 200,000 is already a long-context one.
    assert xai.applies_at_threshold is True
    assert xai.input_usd_per_million_tokens == Decimal("4")
    assert xai.output_usd_per_million_tokens == Decimal("12")

    openai = _openai().long_context_tier
    assert isinstance(openai, LongContextTier)
    assert openai.input_token_threshold == 272_000
    # OpenAI: "≤272K input tokens" against ">272K input tokens" — so exactly
    # 272,000 is still the base tier and the long one starts one token later.
    assert openai.applies_at_threshold is False
    assert openai.input_usd_per_million_tokens == Decimal("0.40")
    assert openai.output_usd_per_million_tokens == Decimal("1.80")


def test_the_untiered_policies_state_no_tier_at_all() -> None:
    """Absent, not null-on-every-price: a field is a claim, and these make none."""
    for provider, model in (
        ("anthropic", "claude-sonnet-5"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("mistral", "mistral-small-2603"),
    ):
        price = price_for(provider=provider, model=model)
        assert price.long_context_tier is None
        assert "long_context_tier" not in price.as_dict()


# -- the boundaries -----------------------------------------------------------


@pytest.mark.parametrize(
    ("input_tokens", "expected"),
    [
        # One token below the threshold: base rates, USD 2 / 1M input.
        (199_999, "0.399998"),
        # Exactly at it: xAI's page says the long rates already apply, so this
        # is USD 4 / 1M and not USD 2 / 1M. The parent's reproduction priced it
        # at 0.40 — half of what the provider charges.
        (200_000, "0.800000"),
        (200_001, "0.800004"),
    ],
)
def test_xai_selects_its_tier_at_the_exact_threshold(
    input_tokens: int, expected: str
) -> None:
    assert _xai().cost(input_tokens=input_tokens, output_tokens=0) == Decimal(expected)


@pytest.mark.parametrize(
    ("input_tokens", "expected"),
    [
        (271_999, "0.0543998"),
        # "≤272K input tokens" is the base tier, so the boundary token itself is
        # still cheap.
        (272_000, "0.0544000"),
        # And one more token moves the whole request to the long rates.
        (272_001, "0.1088004"),
    ],
)
def test_openai_selects_its_tier_one_token_after_the_threshold(
    input_tokens: int, expected: str
) -> None:
    assert _openai().cost(input_tokens=input_tokens, output_tokens=0) == Decimal(expected)


def test_the_long_tier_prices_the_output_tokens_too() -> None:
    """The published rule covers the completion as well as the prompt.

    A build that applied the long rate to the input and the base rate to the
    output would under-state every long-context turn by the difference on the
    completion — silently, and in the direction a cost cap cannot survive.
    """
    # 200,000 input at USD 4/M is 0.8; 1,000 output at USD 12/M is 0.012.
    assert _xai().cost(input_tokens=200_000, output_tokens=1_000) == Decimal("0.812000")
    # Below the threshold the same completion costs USD 6/M: 0.006.
    assert _xai().cost(input_tokens=199_999, output_tokens=1_000) == Decimal("0.405998")
    # 272,001 input at USD 0.40/M plus 1,000 output at USD 1.80/M.
    assert _openai().cost(input_tokens=272_001, output_tokens=1_000) == Decimal(
        "0.1106004"
    )
    assert _openai().cost(input_tokens=272_000, output_tokens=1_000) == Decimal(
        "0.0556000"
    )


def test_no_tier_means_one_rate_at_any_prompt_length() -> None:
    """An untiered model is not quietly charged a long-context premium.

    The refused shortcut, stated as a test: "charge the high tier for every
    request" would be safe against the cap and false about every other model.
    """
    mistral = price_for(provider="mistral", model="mistral-small-2603")
    assert mistral.cost(input_tokens=1_000_000, output_tokens=0) == Decimal("0.15")
    assert mistral.cost(input_tokens=10_000_000, output_tokens=0) == Decimal("1.50")
    sonnet = price_for(provider="anthropic", model="claude-sonnet-5")
    assert sonnet.cost(input_tokens=1_000_000, output_tokens=0) == Decimal("2")
    assert sonnet.cost(input_tokens=100_000_000, output_tokens=0) == Decimal("200")


def test_there_is_no_context_ceiling_standing_in_for_a_tier() -> None:
    """A prompt far past the threshold prices, rather than being refused.

    The other refused shortcut. Imposing an arbitrary maximum prompt length
    would turn a pricing question into a capability limit this build has no
    grounds to state.
    """
    assert _xai().cost(input_tokens=4_000_000, output_tokens=0) == Decimal("16.000000")
    assert _openai().cost(input_tokens=4_000_000, output_tokens=0) == Decimal("1.6000000")


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        # Not an integer, or a bool masquerading as one.
        {"input_token_threshold": 200_000.0},
        {"input_token_threshold": True},
        {"input_token_threshold": "200000"},
        # A threshold of zero or below prices nothing at the base rates, which
        # makes the base rates a number the table states and never uses.
        {"input_token_threshold": 0},
        {"input_token_threshold": -1},
        # Inclusivity is a decision, and an exact bool: ``1`` is not one.
        {"applies_at_threshold": 1},
        {"applies_at_threshold": "yes"},
        # Rates are exact Decimals, never binary floats or strings.
        {"input_usd_per_million_tokens": 4.0},
        {"output_usd_per_million_tokens": "12"},
        {"input_usd_per_million_tokens": Decimal("-1")},
        {"output_usd_per_million_tokens": Decimal("NaN")},
    ],
)
def test_a_tier_definition_that_is_not_one_is_refused(kwargs: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "input_token_threshold": 200_000,
        "applies_at_threshold": True,
        "input_usd_per_million_tokens": Decimal("4"),
        "output_usd_per_million_tokens": Decimal("12"),
        **kwargs,
    }
    with pytest.raises(PricingError):
        LongContextTier(**fields)


def test_a_long_tier_cheaper_than_the_base_tier_is_refused() -> None:
    """A "long" tier that costs less makes the conservative bound unsound.

    Every reservation in this build is computed from an *upper* bound on the
    input tokens. That bound is only conservative if a larger prompt cannot be
    cheaper, which is exactly what a descending tier would make possible. The
    comparison needs both halves, so it belongs on the price rather than on the
    tier.
    """
    for long_input, long_output in (("1", "12"), ("4", "5")):
        with pytest.raises(PricingError):
            ModelPrice(
                provider="xai",
                model="grok-4.5",
                input_usd_per_million_tokens=Decimal("2"),
                output_usd_per_million_tokens=Decimal("6"),
                effective_through=None,
                note="a fixture, not a reviewed price",
                long_context_tier=LongContextTier(
                    input_token_threshold=200_000,
                    applies_at_threshold=True,
                    input_usd_per_million_tokens=Decimal(long_input),
                    output_usd_per_million_tokens=Decimal(long_output),
                ),
            )


def test_a_tier_that_is_not_a_tier_object_is_refused() -> None:
    """A mapping that looks like one is not one: the tier is hashed as a claim."""
    with pytest.raises(PricingError):
        ModelPrice(
            provider="xai",
            model="grok-4.5",
            input_usd_per_million_tokens=Decimal("2"),
            output_usd_per_million_tokens=Decimal("6"),
            effective_through=None,
            note="a fixture, not a reviewed price",
            long_context_tier={"input_token_threshold": 200_000},  # type: ignore[arg-type]
        )


def test_a_cost_still_refuses_a_count_that_is_not_one() -> None:
    """Tiering does not loosen what a token count may be."""
    for tokens in (True, -1, 1.0, "5", None):
        with pytest.raises(PricingError):
            _xai().cost(input_tokens=tokens, output_tokens=0)  # type: ignore[arg-type]
        with pytest.raises(PricingError):
            _xai().cost(input_tokens=0, output_tokens=tokens)  # type: ignore[arg-type]


# -- the policy payloads ------------------------------------------------------


def test_the_untiered_policies_match_their_current_independent_goldens() -> None:
    """Current policy bytes are pinned; Mistral's remain unchanged by tiering."""
    assert pricing_policy_digest("anthropic") == ANTHROPIC_POLICY_DIGEST
    assert pricing_policy_digest("mistral") == MISTRAL_POLICY_DIGEST
    assert pricing_policy_payload()["policy_digest"] == ANTHROPIC_POLICY_DIGEST
    for provider in ("anthropic", "mistral"):
        payload = pricing_policy_payload(provider)
        for block in payload["models"].values():
            assert set(block) == {
                "input_usd_per_million_tokens",
                "output_usd_per_million_tokens",
                "effective_through",
                "note",
            }


def test_the_tiered_policies_moved_their_version_and_their_digest() -> None:
    """A run priced flat and a run priced tiered are two different tables."""
    assert pricing_policy_digest("openai") != SUPERSEDED_OPENAI_DIGEST
    assert pricing_policy_digest("xai") != SUPERSEDED_XAI_DIGEST
    assert PROVIDER_PRICING_POLICIES["openai"].policy_version == (
        "openai-tiered-2026-08-11"
    )
    assert PROVIDER_PRICING_POLICIES["xai"].policy_version == "xai-tiered-2026-08-11"


def test_a_tiered_payload_states_the_threshold_inclusivity_and_long_rates() -> None:
    """Recorded explicitly, because a reader cannot re-derive a vendor's rule.

    Everything a later reader needs in order to recompute any cost this run
    reported is inside the payload the run hashed: which count selects the tier,
    which side of the threshold it selects on, and both long rates.
    """
    block = pricing_policy_payload("xai")["models"]["grok-4.5"]["long_context_tier"]
    assert block == {
        "selected_by": "input_tokens",
        "input_token_threshold": 200_000,
        "applies_at_threshold": True,
        "input_usd_per_million_tokens": "4",
        "output_usd_per_million_tokens": "12",
    }
    block = pricing_policy_payload("openai")["models"]["gpt-5.6-luna"][
        "long_context_tier"
    ]
    assert block == {
        "selected_by": "input_tokens",
        "input_token_threshold": 272_000,
        "applies_at_threshold": False,
        "input_usd_per_million_tokens": "0.40",
        "output_usd_per_million_tokens": "1.80",
    }
    # Rates are strings for the same reason every other rate here is: a JSON
    # number is a binary float to most readers.
    assert all(
        isinstance(block[key], str)
        for key in ("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    )


def test_the_xai_policy_cites_the_page_that_states_the_tier() -> None:
    """The provenance of a tier is the page the tier was read from."""
    policy = PROVIDER_PRICING_POLICIES["xai"]
    assert policy.source == "https://docs.x.ai/developers/pricing"
    assert policy.snapshot_limitation is not None
    assert PROVIDER_PRICING_POLICIES["openai"].source == (
        "https://developers.openai.com/api/docs/pricing"
    )


@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_tampering_with_a_tier_moves_the_policy_digest(provider: str) -> None:
    """A digest that did not cover the tier would authorise the wrong rate."""
    payload = pricing_policy_payload(provider)
    assert payload["policy_digest"] == digest_of_policy_payload(payload)
    for field, value in (
        ("input_token_threshold", 9_000_000),
        ("applies_at_threshold", None),
        ("input_usd_per_million_tokens", "0"),
        ("output_usd_per_million_tokens", "0"),
    ):
        tampered = json.loads(json.dumps(payload))
        for block in tampered["models"].values():
            block["long_context_tier"][field] = value
        assert digest_of_policy_payload(tampered) != payload["policy_digest"]


# -- the stored pricing block -------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model"),
    [("openai", "gpt-5.6-luna"), ("xai", "grok-4.5")],
)
def test_a_stored_block_for_a_tiered_model_carries_its_tier(
    provider: str, model: str
) -> None:
    stored = shipped_pricing_block(price_for(provider=provider, model=model))
    assert "long_context_tier" in stored
    assert check_stored_pricing_block(stored, provider=provider, model=model)


@pytest.mark.parametrize(
    ("provider", "model"),
    [("anthropic", "claude-sonnet-5"), ("mistral", "mistral-small-2603")],
)
def test_a_stored_block_for_an_untiered_model_states_no_tier(
    provider: str, model: str
) -> None:
    """The Anthropic block a run on disk recorded keeps exactly its old keys."""
    stored = shipped_pricing_block(price_for(provider=provider, model=model))
    assert "long_context_tier" not in stored
    assert check_stored_pricing_block(stored, provider=provider, model=model)


def test_a_stored_block_that_drops_the_tier_is_refused() -> None:
    """Deleting the tier is the cheapest way to halve a recorded cost."""
    from boundarybench.budget import BudgetError

    stored = shipped_pricing_block(_xai())
    del stored["long_context_tier"]
    with pytest.raises(BudgetError):
        check_stored_pricing_block(stored, provider="xai", model="grok-4.5")


def test_a_stored_block_whose_tier_was_edited_is_refused() -> None:
    """Every field of it, and the JSON type as well as the value."""
    from boundarybench.budget import BudgetError

    for field, value in (
        ("input_token_threshold", 9_000_000),
        ("input_token_threshold", "200000"),
        ("applies_at_threshold", False),
        ("input_usd_per_million_tokens", "2"),
        ("output_usd_per_million_tokens", "6"),
        ("selected_by", "output_tokens"),
    ):
        stored = shipped_pricing_block(_xai())
        stored["long_context_tier"] = {**stored["long_context_tier"], field: value}
        with pytest.raises(BudgetError):
            check_stored_pricing_block(stored, provider="xai", model="grok-4.5")


def test_a_stored_tier_on_an_untiered_model_is_refused() -> None:
    """A tier nobody published is not a price this build reviewed."""
    from boundarybench.budget import BudgetError

    stored = shipped_pricing_block(
        price_for(provider="mistral", model="mistral-small-2603")
    )
    stored["long_context_tier"] = {
        "selected_by": "input_tokens",
        "input_token_threshold": 1,
        "applies_at_threshold": True,
        "input_usd_per_million_tokens": "99",
        "output_usd_per_million_tokens": "99",
    }
    with pytest.raises(BudgetError):
        check_stored_pricing_block(stored, provider="mistral", model="mistral-small-2603")


# -- the cost guard, the reservation and the re-derivation --------------------


def _guard(provider: str, model: str) -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("1000"),
            max_episodes=36,
            price=price_for(provider=provider, model=model),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )


def test_a_reservation_selects_its_tier_from_the_bound_it_authorises() -> None:
    """The conservative bound decides the tier, so the reservation is conservative.

    The bound over-states the input count, and the long rates are never cheaper,
    so a reservation computed from the bound is never below the cost of the
    request it authorises. That is what makes it a bound.
    """
    guard = _guard("xai", "grok-4.5")
    below = guard.reservation_for(input_tokens_upper_bound=199_999)
    at = guard.reservation_for(input_tokens_upper_bound=200_000)
    # 199,999 input at USD 2/M plus the pinned 1,024-token output ceiling at
    # USD 6/M, against 200,000 at USD 4/M plus the same ceiling at USD 12/M.
    assert below == Decimal("0.406142")
    assert at == Decimal("0.812288")
    assert at > below


def test_a_measured_settlement_selects_the_same_tier_as_the_reservation() -> None:
    """One rule, applied at authorisation and at settlement alike."""
    guard = _guard("openai", "gpt-5.6-luna")
    reservation = guard.authorize(input_tokens_upper_bound=300_000)
    measured = guard.settle(reservation, input_tokens=272_001, output_tokens=10)
    assert measured == Decimal("0.1088184")
    assert guard.measured_usd == measured
    # And the same counts one token lower are on the base tier.
    guard = _guard("openai", "gpt-5.6-luna")
    reservation = guard.authorize(input_tokens_upper_bound=300_000)
    assert guard.settle(reservation, input_tokens=272_000, output_tokens=10) == Decimal(
        "0.0544120"
    )


def test_measured_cost_re_derives_the_same_tier_from_stored_counts() -> None:
    """Replay reads the counts off the row and must reach the same number.

    The ledger re-derives a row's cost from its own token counts at the run's
    pinned policy and refuses a row whose stated cost does not follow. A replay
    that selected a different tier from the same counts would refuse every
    long-context row this build wrote.
    """
    price = _xai()
    for input_tokens, expected in ((199_999, "0.405998"), (200_000, "0.812000")):
        assert measured_cost_usd(
            price, input_tokens=input_tokens, output_tokens=1_000
        ) == Decimal(expected)
    assert measured_cost_usd(price, input_tokens=None, output_tokens=1_000) is None


def test_a_long_context_settlement_can_breach_a_reservation_it_outgrew() -> None:
    """The tier is not a way to hide a breach.

    A reservation taken on the base tier and settled on the long one is exactly
    the case a reservation breach exists to report: the request was made, the
    tokens were consumed, and the measured cost is banked rather than clipped.
    """
    from boundarybench.budget import CostReservationBreachedError

    guard = _guard("xai", "grok-4.5")
    reservation = guard.authorize(input_tokens_upper_bound=100)
    with pytest.raises(CostReservationBreachedError) as caught:
        guard.settle(reservation, input_tokens=200_000, output_tokens=0)
    assert caught.value.measured_usd == Decimal("0.800000")
    assert guard.measured_usd == Decimal("0.800000")
    assert guard.reservation_breached is True
