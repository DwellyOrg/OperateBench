"""The numbers a price, a token count and a cap are allowed to be.

Three separate defects lived in the gap between "a Decimal" and "a Decimal this
build can price a run with":

* ``ModelPrice`` validated its rates only on the tiered path, so an untiered
  price accepted a negative, a float, a NaN or an infinity. ``-1``/``-1``
  produced a negative reservation and a negative cost, and a negative cost
  *raised* the guard's remaining budget — a cost control that pays the run for
  spending.
* Token counts and tier thresholds were unbounded, so a count large enough
  silently rounded in the default 28-digit decimal context — the one thing this
  module's Decimal arithmetic exists to prevent — and a threshold of ``10**5000``
  failed as a bare ``ValueError`` from the integer-to-string limit while a
  policy payload was being built.
* A cap was checked for finiteness and nothing else, so ``1e1000000`` was
  accepted at parse time and surfaced later as a raw ``decimal.Overflow`` from
  inside the guard, at the moment a request was being authorised.

The domain is now stated once, bounded on both ends, checked on every path, and
raised about by name. Every accepted multiplication and addition is exact.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any

import pytest

from boundarybench.budget import (
    BudgetError,
    CostControls,
    RunCostGuard,
    parse_cost_cap,
    usd_text,
)
from boundarybench.pricing import (
    ANTHROPIC_PRICING_POLICY,
    MAX_EXACT_TOKEN_COUNT,
    MAX_USD_AMOUNT,
    MAX_USD_DECIMAL_PLACES,
    PROVIDER_PRICING_POLICIES,
    LongContextTier,
    ModelPrice,
    PricingError,
    price_for,
)

#: The independently pinned digest of the corrected current Anthropic policy.
#: Historical run/evidence bytes retain their own superseded policy identity.
ANTHROPIC_POLICY_DIGEST = (
    "f0b78a71619841f0c3e7d21a094258d24093b28d02744b5d3fb8ce6b7138561a"
)

NOT_A_DECIMAL: tuple[Any, ...] = (
    "2",
    2,
    2.0,
    True,
    None,
    Decimal("NaN"),
    Decimal("Infinity"),
    Decimal("-Infinity"),
    Decimal("-1"),
    Decimal("-0.000001"),
)


def _price(**changes: Any) -> ModelPrice:
    """One untiered fixture price. Not a reviewed rate."""
    fields: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "input_usd_per_million_tokens": Decimal("2"),
        "output_usd_per_million_tokens": Decimal("10"),
        "effective_through": None,
        "note": "a fixture, not a reviewed price",
    }
    fields.update(changes)
    return ModelPrice(**fields)


# -- 1. the constructor the merged base published -----------------------------


def test_the_legacy_keyword_constructor_still_builds_a_price() -> None:
    """The five fields callers wrote before a price could name its provider.

    ``provider`` is optional and comes after them, so code written against the
    single-provider build keeps working unchanged rather than failing on a
    required argument it has never heard of.
    """
    price = ModelPrice(
        model="claude-sonnet-5",
        input_usd_per_million_tokens=Decimal("2"),
        output_usd_per_million_tokens=Decimal("10"),
        effective_through="2026-08-31",
        note="introductory standard-tier price",
    )

    assert price.model == "claude-sonnet-5"
    assert price.input_usd_per_million_tokens == Decimal("2")
    assert price.effective_through == "2026-08-31"
    assert price.long_context_tier is None
    assert price.cost(input_tokens=1_000_000, output_tokens=0) == Decimal("2")


def test_the_legacy_positional_constructor_still_builds_a_price() -> None:
    """Field *order* is part of a public dataclass's constructor."""
    price = ModelPrice(
        "claude-haiku-4-5-20251001",
        Decimal("1"),
        Decimal("5"),
        None,
        "standard-tier price",
    )

    assert price.model == "claude-haiku-4-5-20251001"
    assert price.output_usd_per_million_tokens == Decimal("5")
    assert price.effective_through is None
    assert price.note == "standard-tier price"


def test_every_shipped_entry_still_states_and_validates_its_provider() -> None:
    for provider, policy in PROVIDER_PRICING_POLICIES.items():
        assert policy.provider == provider
        for model, price in policy.models.items():
            assert price.provider == provider
            assert price.model == model


# -- 2. rates, on every path --------------------------------------------------


@pytest.mark.parametrize("value", NOT_A_DECIMAL)
def test_an_untiered_price_refuses_a_base_input_rate_that_is_not_one(
    value: Any,
) -> None:
    """The path that returned early: no tier, so nothing was checked."""
    with pytest.raises(PricingError):
        _price(input_usd_per_million_tokens=value)


@pytest.mark.parametrize("value", NOT_A_DECIMAL)
def test_an_untiered_price_refuses_a_base_output_rate_that_is_not_one(
    value: Any,
) -> None:
    with pytest.raises(PricingError):
        _price(output_usd_per_million_tokens=value)


@pytest.mark.parametrize("value", NOT_A_DECIMAL)
def test_a_tiered_price_refuses_the_same_base_rates(value: Any) -> None:
    tier = LongContextTier(
        input_token_threshold=200_000,
        applies_at_threshold=True,
        input_usd_per_million_tokens=Decimal("400"),
        output_usd_per_million_tokens=Decimal("1200"),
    )
    with pytest.raises(PricingError):
        _price(input_usd_per_million_tokens=value, long_context_tier=tier)


@pytest.mark.parametrize("value", NOT_A_DECIMAL)
def test_a_tier_refuses_a_rate_that_is_not_one(value: Any) -> None:
    with pytest.raises(PricingError):
        LongContextTier(
            input_token_threshold=200_000,
            applies_at_threshold=True,
            input_usd_per_million_tokens=value,
            output_usd_per_million_tokens=Decimal("12"),
        )


def test_a_rate_beyond_the_stated_usd_domain_is_refused() -> None:
    """Bounded above, so no accepted product can overflow or round."""
    with pytest.raises(PricingError):
        _price(input_usd_per_million_tokens=MAX_USD_AMOUNT * 10)
    with pytest.raises(PricingError):
        _price(input_usd_per_million_tokens=Decimal("1e1000000"))


def test_a_rate_finer_than_the_stated_scale_is_refused() -> None:
    """A bound on the fraction as well as the magnitude: both bound the digits."""
    too_fine = Decimal(1).scaleb(-(MAX_USD_DECIMAL_PLACES + 1))
    with pytest.raises(PricingError):
        _price(input_usd_per_million_tokens=too_fine)
    assert _price(input_usd_per_million_tokens=Decimal(1).scaleb(-MAX_USD_DECIMAL_PLACES))


@pytest.mark.parametrize("value", (b"anthropic", 7, None, True))
def test_a_price_refuses_identity_fields_that_are_not_strings(value: Any) -> None:
    for field in ("provider", "model", "note"):
        with pytest.raises(PricingError):
            _price(**{field: value})


@pytest.mark.parametrize("value", (b"2026-08-31", 20260831, True))
def test_a_price_refuses_an_end_date_that_is_not_a_date_string(value: Any) -> None:
    """``None`` is the one non-string it accepts, and it means no announced end."""
    with pytest.raises(PricingError):
        _price(effective_through=value)


def test_a_price_accepts_no_announced_end_date() -> None:
    assert _price(effective_through=None).effective_through is None


# -- 3. the exact token bound -------------------------------------------------


def test_the_exact_token_bound_is_the_interoperable_one() -> None:
    """Two to the fifty-third, minus one: every integer below it round-trips.

    It is the largest integer a JSON number is exact for in every reader this
    build's artefacts have to survive, and it is orders of magnitude above any
    published context window, so nothing real is refused by it.
    """
    assert MAX_EXACT_TOKEN_COUNT == 2**53 - 1
    assert MAX_EXACT_TOKEN_COUNT == 9_007_199_254_740_991


def test_a_cost_at_the_exact_upper_bound_is_exact() -> None:
    """The bound is usable, and its arithmetic does not round.

    In the default 28-digit context this product loses its tail; the accepted
    domain is priced in a context wide enough that it cannot.
    """
    price = _price(
        input_usd_per_million_tokens=Decimal("0.123456789012"),
        output_usd_per_million_tokens=Decimal("0.123456789012"),
    )
    cost = price.cost(
        input_tokens=MAX_EXACT_TOKEN_COUNT, output_tokens=MAX_EXACT_TOKEN_COUNT
    )

    with localcontext() as context:
        context.prec = 200
        expected = (
            Decimal(2 * MAX_EXACT_TOKEN_COUNT) * Decimal("0.123456789012")
        ).scaleb(-6)
    assert cost == expected
    # Exact to the last digit, not merely close: the integer arithmetic below is
    # the same computation with no decimal point in it at all.
    assert cost * 10**18 == 2 * MAX_EXACT_TOKEN_COUNT * 123456789012


@pytest.mark.parametrize(
    "tokens", (MAX_EXACT_TOKEN_COUNT + 1, 10**30, -1, True, 1.0, "5", None)
)
def test_a_token_count_outside_the_domain_is_refused(tokens: Any) -> None:
    price = _price()
    with pytest.raises(PricingError):
        price.cost(input_tokens=tokens, output_tokens=0)
    with pytest.raises(PricingError):
        price.cost(input_tokens=0, output_tokens=tokens)


def test_zero_tokens_cost_zero() -> None:
    assert _price().cost(input_tokens=0, output_tokens=0) == Decimal(0)


# -- 4. thresholds ------------------------------------------------------------


@pytest.mark.parametrize(
    "threshold",
    (
        0,
        -1,
        True,
        1.0,
        MAX_EXACT_TOKEN_COUNT + 1,
        # Given an id of its own, because rendering an integer this long is
        # itself refused by the interpreter — which is the point of including
        # it, and the reason a refusal must not interpolate it.
        pytest.param(10**5000, id="five-thousand-digits"),
    ),
)
def test_a_tier_threshold_outside_the_domain_is_refused(threshold: Any) -> None:
    """Including the one whose refusal cannot quote the value it refused."""
    with pytest.raises(PricingError):
        LongContextTier(
            input_token_threshold=threshold,
            applies_at_threshold=True,
            input_usd_per_million_tokens=Decimal("4"),
            output_usd_per_million_tokens=Decimal("12"),
        )


def test_a_threshold_at_the_exact_upper_bound_serialises() -> None:
    """A payload can be built for anything the domain accepts."""
    tier = LongContextTier(
        input_token_threshold=MAX_EXACT_TOKEN_COUNT,
        applies_at_threshold=False,
        input_usd_per_million_tokens=Decimal("4"),
        output_usd_per_million_tokens=Decimal("12"),
    )
    block = _price(long_context_tier=tier).as_dict()

    assert block["long_context_tier"]["input_token_threshold"] == MAX_EXACT_TOKEN_COUNT
    assert block["long_context_tier"]["input_usd_per_million_tokens"] == "4"


# -- 5. tier boundaries and the shipped tables --------------------------------


def test_the_openai_tier_begins_one_token_past_its_threshold() -> None:
    price = price_for(provider="openai", model="gpt-5.6-luna")

    assert price.rates_for(272_000) == (Decimal("0.20"), Decimal("1.20"))
    assert price.rates_for(272_001) == (Decimal("0.40"), Decimal("1.80"))


def test_the_xai_tier_begins_at_its_threshold() -> None:
    price = price_for(provider="xai", model="grok-4.5")

    assert price.rates_for(199_999) == (Decimal("2"), Decimal("6"))
    assert price.rates_for(200_000) == (Decimal("4"), Decimal("12"))


def test_the_anthropic_policy_digest_is_the_one_every_run_recorded() -> None:
    assert ANTHROPIC_PRICING_POLICY.digest() == ANTHROPIC_POLICY_DIGEST


def test_every_shipped_policy_payload_still_builds() -> None:
    for provider, policy in PROVIDER_PRICING_POLICIES.items():
        payload = policy.payload()
        assert payload["provider"] == provider
        assert len(payload["policy_digest"]) == 64


# -- 6. caps ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ("1e1000000", "-1e1000000", "1e999999999", "NaN", "Infinity", "-Infinity")
)
def test_a_cap_outside_the_stated_domain_is_refused_at_parse_time(text: str) -> None:
    """Refused where it is read, not where it later overflows.

    ``1e1000000`` is a finite Decimal, so the old finiteness check passed it and
    the failure surfaced as a raw ``decimal.Overflow`` from inside the guard —
    at the moment a provider request was being authorised, in a type nothing
    catches.
    """
    with pytest.raises(BudgetError):
        parse_cost_cap(text)


def test_a_cap_outside_the_stated_domain_is_refused_on_construction() -> None:
    with pytest.raises(BudgetError):
        CostControls(
            max_cost_usd=Decimal("1e1000000"),
            max_episodes=1,
            price=price_for(provider="anthropic", model="claude-sonnet-5"),
        )


def test_a_cap_at_the_top_of_the_domain_is_accepted_and_reportable() -> None:
    controls = CostControls(
        max_cost_usd=MAX_USD_AMOUNT,
        max_episodes=1,
        price=price_for(provider="anthropic", model="claude-sonnet-5"),
    )
    guard = RunCostGuard(
        controls=controls, max_output_tokens=1024, max_attempts_per_turn=3
    )

    assert guard.remaining_usd == MAX_USD_AMOUNT
    assert guard.as_dict()["remaining_usd"] == usd_text(MAX_USD_AMOUNT)
    assert guard.as_dict()["max_cost_usd"] == usd_text(MAX_USD_AMOUNT)


# -- 7. the cost control, under the ordinary cap ------------------------------


def _guard(cap: str = "20") -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal(cap),
            max_episodes=12,
            price=price_for(provider="anthropic", model="claude-sonnet-5"),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )


def test_a_twenty_dollar_cap_authorises_settles_and_decreases() -> None:
    """The ordinary path, stated as arithmetic rather than as a range."""
    guard = _guard()
    reservation = guard.authorize(input_tokens_upper_bound=10_000)

    assert reservation == Decimal("0.03024")
    measured = guard.settle(reservation, input_tokens=137, output_tokens=29)
    assert measured == Decimal("0.000564")
    assert guard.measured_usd == Decimal("0.000564")
    assert guard.remaining_usd == Decimal("20") - Decimal("0.000564")


def test_no_settlement_can_increase_the_remaining_budget() -> None:
    """What the negative rates produced: a run paid for spending.

    The rates that made it possible cannot be constructed at all now, so this
    states the property from the other end — every settlement moves the
    remaining budget down or leaves it where it was.
    """
    guard = _guard()
    before = guard.remaining_usd
    reservation = guard.authorize(input_tokens_upper_bound=10_000)
    guard.settle(reservation, input_tokens=137, output_tokens=29)

    assert guard.remaining_usd < before
    assert guard.measured_usd > 0


def test_a_negative_rate_can_never_reach_a_guard() -> None:
    with pytest.raises(PricingError):
        _price(
            input_usd_per_million_tokens=Decimal("-1"),
            output_usd_per_million_tokens=Decimal("-1"),
        )
