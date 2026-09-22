"""What a Lifecycle run may cost, and what the guard says it did cost.

The pricing policy is the only place in the Lifecycle Track that turns tokens
into money, and the cost guard is the only place that decides whether a request
is authorised. Both are asked adversarially here:

**The rates are exact or they are refused.** A binary float, a decimal string
that is not one, a negative rate and a non-finite rate are all things somebody
can type; none of them is a rate a run can be priced at, so each is a named
refusal rather than a number that quietly becomes an amount.

**A refusal changes nothing.** The two ceilings — the cost cap and the token
ceiling — are checked at their boundary from both sides, and the run's recorded
exposure is asserted *after* a refusal as well as after an authorisation. A
guard that counted a refused request would report a run as having committed
budget to a request that was never dispatched.

**Measured and forfeited are never one number.** A settlement with no counts
behind it forfeits the whole reservation and is not spend; a response that cost
more than its reservation is banked at its measured value and stops the run.

Nothing here opens a socket or reads a credential.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    RATE_SOURCES,
    TOKENS_PER_RATE_UNIT,
    GuardEvent,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
    PricingPolicyError,
    TokenCapExceededError,
)
from operatebench.providers.cost import (
    CostCapExceededError,
    CostReservationBreachedError,
    usd_text,
)

INPUT_RATE = Decimal("1.25")
OUTPUT_RATE = Decimal("10.00")
MAX_OUTPUT_TOKENS = 4096


def policy(**overrides: Any) -> LifecyclePricingPolicy:
    fields: dict[str, Any] = {
        "policy_id": "operator_pinned_v1",
        "input_usd_per_mtok": INPUT_RATE,
        "output_usd_per_mtok": OUTPUT_RATE,
        "rate_source": RATE_SOURCE_OPERATOR,
    }
    fields.update(overrides)
    return LifecyclePricingPolicy(**fields)


def guard(**overrides: Any) -> LifecycleCostGuard:
    fields: dict[str, Any] = {
        "policy": policy(),
        "cap_usd": Decimal("1.00"),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    fields.update(overrides)
    return LifecycleCostGuard(**fields)


# -- a rate is exact, or it is not a rate --------------------------------------


@pytest.mark.parametrize(
    "rate",
    [
        1.25,
        0.0000001,
        None,
        b"1.25",
        [Decimal("1.25")],
    ],
)
def test_a_rate_that_is_not_an_exact_decimal_is_refused(rate: Any) -> None:
    """A binary float is not the rate anybody agreed to, and neither is a list."""
    with pytest.raises(PricingPolicyError):
        policy(input_usd_per_mtok=rate)


@pytest.mark.parametrize("rate", ["1,25", "one dollar", "", "1.2.3"])
def test_a_decimal_string_that_is_not_a_decimal_is_refused(rate: str) -> None:
    with pytest.raises(PricingPolicyError):
        policy(output_usd_per_mtok=rate)


@pytest.mark.parametrize("rate", ["-0.01", "NaN", "Infinity", "-Infinity"])
def test_a_negative_or_non_finite_rate_is_refused(rate: str) -> None:
    """A run cannot be priced at a rate that is not a finite, non-negative number."""
    with pytest.raises(PricingPolicyError):
        policy(input_usd_per_mtok=rate)


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_a_policy_with_no_name_is_refused(name: str) -> None:
    """An amount is attributed to a table, so the table is named."""
    with pytest.raises(PricingPolicyError):
        policy(policy_id=name)


def test_a_rate_source_outside_this_builds_vocabulary_is_refused() -> None:
    """This build makes no claim about any vendor's published prices."""
    with pytest.raises(PricingPolicyError):
        policy(rate_source="openai_published_list_prices")
    assert RATE_SOURCES == (RATE_SOURCE_OPERATOR,)


def test_a_rate_typed_as_a_string_and_as_a_decimal_price_identically() -> None:
    """The stored form is the exact decimal, however the operator typed it."""
    typed = policy(input_usd_per_mtok="1.25", output_usd_per_mtok="10.00")
    assert typed.digest_sha256 == policy().digest_sha256
    assert typed.price(input_tokens=1000, output_tokens=1000) == policy().price(
        input_tokens=1000, output_tokens=1000
    )


def test_changing_a_rate_changes_the_digest_the_ledger_carries() -> None:
    assert policy(output_usd_per_mtok="10.01").digest_sha256 != policy().digest_sha256


# -- token counts are counts ---------------------------------------------------


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (True, 10),
        (10, False),
        (-1, 10),
        (10, -1),
        (1.0, 10),
        ("10", 10),
        (None, 10),
    ],
)
def test_pricing_a_call_refuses_anything_that_is_not_a_token_count(
    input_tokens: Any, output_tokens: Any
) -> None:
    """``bool`` is an ``int`` subclass, so the type test is exact rather than kind."""
    with pytest.raises(PricingPolicyError):
        policy().price(input_tokens=input_tokens, output_tokens=output_tokens)


def test_a_call_of_no_tokens_costs_nothing_and_is_not_a_refusal() -> None:
    assert policy().price(input_tokens=0, output_tokens=0) == Decimal(0)


def test_a_price_is_the_two_rates_over_the_stated_rate_unit() -> None:
    """Derived independently of the module: rates per million, added, not rounded."""
    priced = policy().price(input_tokens=4211, output_tokens=118)
    expected = (Decimal(4211) * INPUT_RATE + Decimal(118) * OUTPUT_RATE) / Decimal(
        TOKENS_PER_RATE_UNIT
    )
    assert priced == expected
    assert usd_text(priced) == "0.00644375"


@pytest.mark.parametrize("calls", [0, -1, True, 1.0, "3", None])
def test_a_worst_case_over_a_call_count_that_is_not_one_is_refused(calls: Any) -> None:
    with pytest.raises(PricingPolicyError):
        policy().worst_case_usd(
            calls=calls, input_token_bound=1000, max_output_tokens=MAX_OUTPUT_TOKENS
        )


def test_the_worst_case_scales_with_the_call_count_it_is_asked_for() -> None:
    one = policy().worst_case_usd(
        calls=1, input_token_bound=14_706, max_output_tokens=MAX_OUTPUT_TOKENS
    )
    many = policy().worst_case_usd(
        calls=46, input_token_bound=14_706, max_output_tokens=MAX_OUTPUT_TOKENS
    )
    assert many == one * 46


# -- what a guard will accept as its own configuration -------------------------


def test_a_guard_enforces_a_stated_policy_and_not_a_look_alike() -> None:
    class LooksLikeAPolicy:
        def price(self, *, input_tokens: int, output_tokens: int) -> Decimal:
            return Decimal(0)

    with pytest.raises(PricingPolicyError):
        LifecycleCostGuard(
            policy=LooksLikeAPolicy(),  # type: ignore[arg-type]
            cap_usd=Decimal("1.00"),
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )


@pytest.mark.parametrize("cap", ["0", "-1", Decimal(0), Decimal("-0.01")])
def test_a_cap_of_zero_or_less_authorises_nothing_and_is_refused(cap: Any) -> None:
    """Refused rather than read as 'no limit', which is the dangerous reading."""
    with pytest.raises(PricingPolicyError):
        guard(cap_usd=cap)


@pytest.mark.parametrize("ceiling", [0, -1, True, 1.5, "4096", None])
def test_an_output_ceiling_that_is_not_a_positive_count_is_refused(
    ceiling: Any,
) -> None:
    with pytest.raises(PricingPolicyError):
        guard(max_output_tokens=ceiling)


@pytest.mark.parametrize("ceiling", [0, -1, True, 1.5, "1000"])
def test_a_token_ceiling_that_is_not_a_positive_count_is_refused(ceiling: Any) -> None:
    """Absent is expressible; malformed is not."""
    with pytest.raises(PricingPolicyError):
        guard(token_hard_cap=ceiling)


def test_a_guard_reports_the_policy_and_cap_it_was_built_with() -> None:
    stated = policy()
    built = LifecycleCostGuard(
        policy=stated, cap_usd="2.50", max_output_tokens=MAX_OUTPUT_TOKENS
    )
    assert built.policy is stated
    assert built.cap_usd == Decimal("2.50")
    assert built.token_hard_cap is None
    assert built.reserved_tokens == 0


# -- the two ceilings, at their boundaries -------------------------------------


def test_a_request_that_exactly_reaches_the_token_ceiling_is_authorised() -> None:
    """The ceiling is a bound on what is committed, not on what is asked for."""
    built = guard(token_hard_cap=5_000 + MAX_OUTPUT_TOKENS)
    built.authorize(input_tokens_upper_bound=5_000)
    assert built.reserved_tokens == 5_000 + MAX_OUTPUT_TOKENS


def test_one_token_past_the_ceiling_is_refused_and_commits_nothing() -> None:
    built = guard(token_hard_cap=5_000 + MAX_OUTPUT_TOKENS)
    with pytest.raises(TokenCapExceededError):
        built.authorize(input_tokens_upper_bound=5_001)
    assert built.reserved_tokens == 0
    assert built.reserved_usd == Decimal(0)
    assert built.committed_usd == Decimal(0)


def test_a_token_refusal_leaves_an_earlier_authorisation_exactly_where_it_was() -> None:
    built = guard(token_hard_cap=2 * (5_000 + MAX_OUTPUT_TOKENS))
    first = built.authorize(input_tokens_upper_bound=5_000)
    before_tokens, before_usd = built.reserved_tokens, built.reserved_usd
    with pytest.raises(TokenCapExceededError):
        built.authorize(input_tokens_upper_bound=5_001)
    assert (built.reserved_tokens, built.reserved_usd) == (before_tokens, before_usd)
    assert built.committed_usd == first


def test_a_request_whose_reservation_exactly_reaches_the_cost_cap_is_authorised() -> None:
    """The cap is compared against what would be committed, inclusively."""
    exact = policy().price(input_tokens=8_000, output_tokens=MAX_OUTPUT_TOKENS)
    built = guard(cap_usd=exact)
    assert built.authorize(input_tokens_upper_bound=8_000) == exact
    assert built.committed_usd == exact


def test_a_reservation_one_step_past_the_cost_cap_is_refused_before_dispatch() -> None:
    exact = policy().price(input_tokens=8_000, output_tokens=MAX_OUTPUT_TOKENS)
    built = guard(cap_usd=exact)
    with pytest.raises(CostCapExceededError):
        built.authorize(input_tokens_upper_bound=8_001)
    assert built.reserved_usd == Decimal(0)
    assert built.reserved_tokens == 0
    assert built.take_events() == ()


# -- how a reservation is resolved ---------------------------------------------


def test_cancellation_releases_a_never_dispatched_lifecycle_reservation() -> None:
    built = guard(token_hard_cap=10_000)
    reservation = built.authorize(input_tokens_upper_bound=5_000)

    built.cancel(reservation, input_tokens_upper_bound=5_000)

    assert built.committed_usd == Decimal(0)
    assert built.measured_usd == Decimal(0)
    assert built.forfeited_usd == Decimal(0)
    assert built.reserved_tokens == 0
    assert built.take_events() == ()
    with pytest.raises(PricingPolicyError):
        built.cancel(reservation, input_tokens_upper_bound=5_000)


def test_cancellation_mismatch_fails_closed_without_changing_lifecycle_totals() -> None:
    built = guard(token_hard_cap=10_000)
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    before = built.as_dict(), built.reserved_tokens, built.take_events()

    with pytest.raises(PricingPolicyError):
        built.cancel(reservation + Decimal("0.000000001"), input_tokens_upper_bound=5_000)

    assert (built.as_dict(), built.reserved_tokens, built.take_events()) == before
    built.cancel(reservation, input_tokens_upper_bound=5_000)


def test_equal_priced_cancellation_is_bound_to_the_just_authorised_token_exposure() -> (
    None
):
    output_tokens = 100
    first_bound, second_bound = 200, 400
    token_cap = first_bound + second_bound + 2 * output_tokens
    built = LifecycleCostGuard(
        policy=policy(input_usd_per_mtok=Decimal(0)),
        cap_usd=Decimal("1.00"),
        max_output_tokens=output_tokens,
        token_hard_cap=token_cap,
    )
    first = built.authorize(input_tokens_upper_bound=first_bound)
    second = built.authorize(input_tokens_upper_bound=second_bound)
    assert first == second
    before = built.committed_usd, built.reserved_tokens, built.reserved_usd

    with pytest.raises(PricingPolicyError):
        built.cancel(second, input_tokens_upper_bound=first_bound)

    assert (built.committed_usd, built.reserved_tokens, built.reserved_usd) == before
    built.cancel(second, input_tokens_upper_bound=second_bound)
    assert built.committed_usd == first
    assert built.reserved_tokens == first_bound + output_tokens
    assert built.reserved_usd == first + second
    # The exact token capacity released by cancellation can be authorised again.
    assert built.authorize(input_tokens_upper_bound=second_bound) == second
    assert built.reserved_tokens == token_cap
    assert built.reserved_usd == first + second + second


def test_equal_outstanding_reservations_resolve_one_at_a_time() -> None:
    built = guard()
    first = built.authorize(input_tokens_upper_bound=5_000)
    second = built.authorize(input_tokens_upper_bound=5_000)
    assert first == second

    built.forfeit(first)
    measured = built.settle(second, input_tokens=100, output_tokens=10)

    assert built.forfeited_usd == first
    assert built.measured_usd == measured
    assert built.committed_usd == first + (measured or Decimal(0))
    assert len(built.take_events()) == 2
    before = built.as_dict(), built.reserved_tokens
    with pytest.raises(PricingPolicyError):
        built.forfeit(first)
    assert (built.as_dict(), built.reserved_tokens) == before
    assert built.take_events() == ()


@pytest.mark.parametrize("resolution", ["settle", "forfeit"])
def test_mismatched_resolution_fails_before_lifecycle_totals_mutate(
    resolution: str,
) -> None:
    built = guard(token_hard_cap=10_000)
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    before = built.as_dict(), built.reserved_tokens, built.take_events()
    wrong = reservation + Decimal("0.000000001")

    with pytest.raises(PricingPolicyError):
        if resolution == "settle":
            built.settle(wrong, input_tokens=100, output_tokens=10)
        else:
            built.forfeit(wrong)

    assert (built.as_dict(), built.reserved_tokens, built.take_events()) == before
    built.cancel(reservation, input_tokens_upper_bound=5_000)


@pytest.mark.parametrize("resolution", ["settle", "forfeit"])
def test_lifecycle_reservation_cannot_be_resolved_twice(resolution: str) -> None:
    built = guard(token_hard_cap=10_000)
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    if resolution == "settle":
        built.settle(reservation, input_tokens=100, output_tokens=10)
    else:
        built.forfeit(reservation)
    before = built.as_dict(), built.reserved_tokens
    assert len(built.take_events()) == 1

    with pytest.raises(PricingPolicyError):
        if resolution == "settle":
            built.settle(reservation, input_tokens=100, output_tokens=10)
        else:
            built.forfeit(reservation)

    assert (built.as_dict(), built.reserved_tokens) == before
    assert built.take_events() == ()


def test_a_settlement_with_no_counts_forfeits_the_whole_reservation() -> None:
    """Exposure, not spend: the two totals are never added together."""
    built = guard()
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    assert built.settle(reservation, input_tokens=None, output_tokens=None) is None
    assert built.forfeited_usd == reservation
    assert built.measured_usd == Decimal(0)
    assert built.committed_usd == reservation


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"), [(None, 118), (4211, None), (None, None)]
)
def test_a_half_reported_usage_forfeits_rather_than_prices_what_it_has(
    input_tokens: Any, output_tokens: Any
) -> None:
    built = guard()
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    assert (
        built.settle(reservation, input_tokens=input_tokens, output_tokens=output_tokens)
        is None
    )
    assert built.measured_usd == Decimal(0)
    assert built.forfeited_usd == reservation


def test_a_measured_settlement_banks_the_measured_amount_and_releases_the_rest() -> None:
    built = guard()
    reservation = built.authorize(input_tokens_upper_bound=14_706)
    measured = built.settle(reservation, input_tokens=4211, output_tokens=118)
    assert measured == policy().price(input_tokens=4211, output_tokens=118)
    assert built.measured_usd == measured
    assert built.forfeited_usd == Decimal(0)
    # The unspent part of the reservation is released, so the run's committed
    # budget is what it measured rather than what it authorised.
    assert built.committed_usd == measured
    assert built.reserved_usd == reservation


def test_a_response_costing_more_than_its_reservation_stops_the_run_but_banks_it() -> (
    None
):
    """Hiding real spend is the failure that matters, so the amount is kept."""
    built = guard()
    reservation = built.authorize(input_tokens_upper_bound=10)
    with pytest.raises(CostReservationBreachedError) as raised:
        built.settle(reservation, input_tokens=1_000_000, output_tokens=1_000_000)
    measured = policy().price(input_tokens=1_000_000, output_tokens=1_000_000)
    assert raised.value.measured_usd == measured
    assert raised.value.reservation_usd == reservation
    assert raised.value.cap_usd == built.cap_usd
    assert built.measured_usd == measured
    assert built.forfeited_usd == Decimal(0)


def test_a_breached_reservation_is_still_reported_on_the_event_queue() -> None:
    """The recorder reads the queue, and a breach is exactly what it must record."""
    built = guard()
    reservation = built.authorize(input_tokens_upper_bound=10)
    with pytest.raises(CostReservationBreachedError):
        built.settle(reservation, input_tokens=1_000_000, output_tokens=1_000_000)
    events = built.take_events()
    assert len(events) == 1
    assert events[0].measured_usd is not None
    assert events[0].forfeited_usd is None


# -- the event queue -----------------------------------------------------------


def test_the_event_queue_drains_in_the_order_the_reservations_resolved() -> None:
    """Taking them in order is what keeps one-attempt-per-call from being assumed."""
    built = guard()
    first = built.authorize(input_tokens_upper_bound=5_000)
    built.settle(first, input_tokens=None, output_tokens=None)
    second = built.authorize(input_tokens_upper_bound=5_000)
    built.settle(second, input_tokens=100, output_tokens=10)
    events = built.take_events()
    assert [(e.measured_usd is None, e.forfeited_usd is None) for e in events] == [
        (True, False),
        (False, True),
    ]
    assert built.take_events() == ()


def test_the_last_event_is_the_most_recent_resolution_and_survives_a_take() -> None:
    """``last_event`` is read per row; draining the queue must not blank it."""
    built = guard()
    assert built.last_event() is None
    reservation = built.authorize(input_tokens_upper_bound=5_000)
    built.settle(reservation, input_tokens=None, output_tokens=None)
    built.take_events()
    last = built.last_event()
    assert isinstance(last, GuardEvent)
    assert last.forfeited_usd == reservation


def test_a_guard_with_nothing_resolved_yet_has_no_events_to_offer() -> None:
    assert guard().take_events() == ()


# -- what a run reports --------------------------------------------------------


def test_the_three_totals_are_reported_apart_as_exact_decimal_strings() -> None:
    built = guard()
    forfeited = built.authorize(input_tokens_upper_bound=5_000)
    built.settle(forfeited, input_tokens=None, output_tokens=None)
    measured_reservation = built.authorize(input_tokens_upper_bound=14_706)
    built.settle(measured_reservation, input_tokens=4211, output_tokens=118)
    reported = built.as_dict()
    assert reported["cap_usd"] == usd_text(built.cap_usd)
    assert reported["measured_cost_usd"] == usd_text(built.measured_usd)
    assert reported["forfeited_reservation_usd"] == usd_text(built.forfeited_usd)
    assert reported["reserved_usd"] == usd_text(forfeited + measured_reservation)
    assert reported["pricing_digest_sha256"] == policy().digest_sha256
    # Nothing here adds the measured cost and the forfeited reservation together
    # and calls the result what the run cost.
    assert reported["measured_cost_usd"] != reported["reserved_usd"]


def test_every_reported_amount_is_a_string_and_never_a_json_number() -> None:
    reported = guard().as_dict()
    assert all(
        isinstance(reported[name], str)
        for name in (
            "cap_usd",
            "measured_cost_usd",
            "forfeited_reservation_usd",
            "reserved_usd",
        )
    )
