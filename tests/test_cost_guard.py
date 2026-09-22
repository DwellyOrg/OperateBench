"""The fail-closed cost guard.

The guard answers one question before every provider request: *could this
request, including the retries the adapter is allowed to make, take the run past
the budget its operator authorised?* If it could, the request is not made.

Two quantities are tracked and never conflated. **Measured** cost is computed
from usage the provider actually reported. **Exposure** is the conservative
upper bound this build reserved for an attempt that returned no usage — an
attempt that may or may not have been billed, which local evidence cannot
settle. Exposure is deliberately an over-statement: the guard's job is to never
under-state what a run might have cost.

Nothing here contacts a provider.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from boundarybench.budget import (
    REQUEST_OVERHEAD_TOKEN_ALLOWANCE,
    BudgetError,
    CostCapExceededError,
    CostControls,
    RunCostGuard,
    conservative_input_token_bound,
    usd_text,
)
from boundarybench.pricing import price_for

SONNET = "claude-sonnet-5"


def controls(cap: str = "100", episodes: int | None = 72) -> CostControls:
    return CostControls(
        max_cost_usd=Decimal(cap),
        max_episodes=episodes,
        price=price_for(provider="anthropic", model=SONNET),
    )


def guard(cap: str = "100", **kwargs: object) -> RunCostGuard:
    return RunCostGuard(
        controls=controls(cap),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
        **kwargs,  # type: ignore[arg-type]
    )


# -- what the controls may be ------------------------------------------------


def test_a_cost_cap_without_a_price_is_refused() -> None:
    """A cap that cannot be measured against anything is not a cap."""
    with pytest.raises(BudgetError):
        CostControls(max_cost_usd=Decimal("100"), max_episodes=72, price=None)


@pytest.mark.parametrize("cap", ["0", "-1"])
def test_a_non_positive_cost_cap_is_refused(cap: str) -> None:
    with pytest.raises(BudgetError):
        CostControls(
            max_cost_usd=Decimal(cap),
            max_episodes=72,
            price=price_for(provider="anthropic", model=SONNET),
        )


@pytest.mark.parametrize("episodes", [0, -3, True])
def test_a_non_positive_episode_limit_is_refused(episodes: object) -> None:
    with pytest.raises(BudgetError):
        CostControls(
            max_cost_usd=None,
            max_episodes=episodes,  # type: ignore[arg-type]
            price=None,
        )


def test_controls_round_trip_through_their_stored_form_exactly() -> None:
    stored = controls("100.50").as_dict()

    assert stored["max_cost_usd"] == "100.50"
    assert stored["max_episodes"] == 72
    assert stored["pricing"]["policy_version"]
    assert stored["pricing"]["model"] == SONNET
    restored = CostControls.from_stored(stored, provider="anthropic", model=SONNET)
    assert restored == controls("100.50")


# -- the reservation ---------------------------------------------------------


def test_the_reservation_covers_the_whole_turn_including_its_retries() -> None:
    """One authorisation buys one attempt; the turn's bound is every attempt.

    The adapter may retry a transport fault up to its bounded limit, so the
    budget a turn can consume is the per-attempt bound times that limit. The
    guard is asked before each attempt, so each attempt is separately refused
    when what is left cannot pay for it.
    """
    active = guard()
    price = price_for(provider="anthropic", model=SONNET)
    expected = price.cost(input_tokens=1000, output_tokens=1024)

    assert active.reservation_for(input_tokens_upper_bound=1000) == expected
    assert active.turn_upper_bound_usd(input_tokens_upper_bound=1000) == expected * 3


def test_an_authorised_request_reserves_before_it_is_made() -> None:
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=1000)

    assert reservation > 0
    # Reserved, not spent: nothing is measured until usage comes back.
    assert active.measured_usd == Decimal(0)
    assert active.remaining_usd == Decimal("100") - reservation


def test_a_second_authorisation_before_the_first_is_resolved_is_refused() -> None:
    """One in-flight request at a time; anything else double-spends the budget."""
    active = guard()
    active.authorize(input_tokens_upper_bound=1000)
    with pytest.raises(BudgetError):
        active.authorize(input_tokens_upper_bound=1000)


def test_cancellation_releases_a_never_dispatched_boundary_reservation() -> None:
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=1000)

    active.cancel(reservation, input_tokens_upper_bound=1000)

    assert active.committed_usd == Decimal(0)
    assert active.measured_usd == Decimal(0)
    assert active.exposure_usd == Decimal(0)
    assert active.episode_measured_usd == Decimal(0)
    assert active.episode_exposure_usd == Decimal(0)
    with pytest.raises(BudgetError):
        active.cancel(reservation, input_tokens_upper_bound=1000)


def test_cancellation_mismatch_fails_closed_without_changing_boundary_totals() -> None:
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=1000)
    before = (
        active.committed_usd,
        active.measured_usd,
        active.exposure_usd,
        active.episode_measured_usd,
        active.episode_exposure_usd,
    )

    with pytest.raises(BudgetError):
        active.cancel(reservation + Decimal("0.000000001"), input_tokens_upper_bound=1000)

    assert (
        active.committed_usd,
        active.measured_usd,
        active.exposure_usd,
        active.episode_measured_usd,
        active.episode_exposure_usd,
    ) == before
    active.cancel(reservation, input_tokens_upper_bound=1000)


def test_cancellation_token_exposure_mismatch_fails_before_totals_change() -> None:
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=1000)
    before = (
        active.committed_usd,
        active.measured_usd,
        active.exposure_usd,
        active.episode_measured_usd,
        active.episode_exposure_usd,
    )

    with pytest.raises(BudgetError):
        active.cancel(reservation, input_tokens_upper_bound=1001)

    assert (
        active.committed_usd,
        active.measured_usd,
        active.exposure_usd,
        active.episode_measured_usd,
        active.episode_exposure_usd,
    ) == before
    active.cancel(reservation, input_tokens_upper_bound=1000)


# -- the boundary ------------------------------------------------------------


def _cap_for(input_bound: int, output_tokens: int = 1024) -> Decimal:
    price = price_for(provider="anthropic", model=SONNET)
    return price.cost(input_tokens=input_bound, output_tokens=output_tokens)


def test_a_request_whose_bound_exactly_equals_the_remaining_budget_is_allowed() -> None:
    """Equality is not excess. Refusing it would stop a run that fits."""
    exact = _cap_for(1000)
    active = RunCostGuard(
        controls=CostControls(
            max_cost_usd=exact,
            max_episodes=None,
            price=price_for(provider="anthropic", model=SONNET),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=1,
    )

    assert active.authorize(input_tokens_upper_bound=1000) == exact
    assert active.remaining_usd == Decimal(0)


def test_a_request_one_unit_over_the_remaining_budget_is_refused() -> None:
    exact = _cap_for(1000)
    active = RunCostGuard(
        controls=CostControls(
            max_cost_usd=exact - Decimal("0.000001"),
            max_episodes=None,
            price=price_for(provider="anthropic", model=SONNET),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=1,
    )

    with pytest.raises(CostCapExceededError) as refused:
        active.authorize(input_tokens_upper_bound=1000)
    assert "cost" in str(refused.value).lower()


def test_a_refused_request_leaves_the_guard_usable_and_unchanged() -> None:
    """The refusal is a decision, not a corruption: nothing is reserved by it."""
    exact = _cap_for(1000)
    active = RunCostGuard(
        controls=CostControls(
            max_cost_usd=exact,
            max_episodes=None,
            price=price_for(provider="anthropic", model=SONNET),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=1,
    )
    with pytest.raises(CostCapExceededError):
        active.authorize(input_tokens_upper_bound=2000)

    assert active.remaining_usd == exact
    assert active.authorize(input_tokens_upper_bound=1000) == exact


# -- settling and forfeiting -------------------------------------------------


def test_a_measured_response_settles_at_what_it_actually_cost() -> None:
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=100_000)
    measured = active.settle(reservation, input_tokens=137, output_tokens=29)

    assert measured == Decimal("0.000564")
    assert active.measured_usd == Decimal("0.000564")
    # The reservation is released in full: only the measurement is kept.
    assert active.exposure_usd == Decimal(0)
    assert active.remaining_usd == Decimal("100") - Decimal("0.000564")


def test_an_attempt_with_no_usage_keeps_its_conservative_bound_as_exposure() -> None:
    """Unmeasured is not free, and this build cannot prove it was not billed."""
    active = guard()
    reservation = active.authorize(input_tokens_upper_bound=1000)
    active.forfeit(reservation)

    assert active.measured_usd == Decimal(0)
    assert active.exposure_usd == reservation
    assert active.remaining_usd == Decimal("100") - reservation
    # And the two are never added into one number that pretends to be measured.
    assert active.as_dict()["measured_usd"] == "0"
    assert active.as_dict()["exposure_usd"] == usd_text(reservation)


def test_retried_attempts_accumulate_exposure_and_shrink_what_is_left() -> None:
    active = guard()
    for _ in range(3):
        active.forfeit(active.authorize(input_tokens_upper_bound=1000))

    assert active.exposure_usd == _cap_for(1000) * 3
    assert active.remaining_usd == Decimal("100") - _cap_for(1000) * 3


def test_exposure_can_exhaust_the_budget_and_the_next_request_is_refused() -> None:
    """Fail closed on exposure, not only on what was measured."""
    tiny = _cap_for(1000) * 2
    active = RunCostGuard(
        controls=CostControls(
            max_cost_usd=tiny,
            max_episodes=None,
            price=price_for(provider="anthropic", model=SONNET),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=1,
    )
    active.forfeit(active.authorize(input_tokens_upper_bound=1000))
    active.forfeit(active.authorize(input_tokens_upper_bound=1000))

    with pytest.raises(CostCapExceededError):
        active.authorize(input_tokens_upper_bound=1000)


def test_settling_or_forfeiting_without_a_reservation_is_refused() -> None:
    active = guard()
    with pytest.raises(BudgetError):
        active.settle(Decimal("1"), input_tokens=1, output_tokens=1)
    with pytest.raises(BudgetError):
        active.forfeit(Decimal("1"))


def test_settling_a_reservation_that_is_not_the_outstanding_one_is_refused() -> None:
    active = guard()
    active.authorize(input_tokens_upper_bound=1000)
    with pytest.raises(BudgetError):
        active.settle(Decimal("99"), input_tokens=1, output_tokens=1)


# -- per-episode accounting --------------------------------------------------


def test_each_episode_reports_only_its_own_exposure() -> None:
    active = guard()
    active.begin_episode()
    active.forfeit(active.authorize(input_tokens_upper_bound=1000))
    first = active.episode_exposure_usd

    active.begin_episode()
    second = active.episode_exposure_usd

    assert first == _cap_for(1000)
    assert second == Decimal(0)
    # The run-level total keeps both, so a per-episode reset cannot lose spend.
    assert active.exposure_usd == _cap_for(1000)


def test_each_episode_reports_only_its_own_measured_cost() -> None:
    active = guard()
    active.begin_episode()
    active.settle(
        active.authorize(input_tokens_upper_bound=100_000),
        input_tokens=137,
        output_tokens=29,
    )
    assert active.episode_measured_usd == Decimal("0.000564")

    active.begin_episode()
    assert active.episode_measured_usd == Decimal(0)
    assert active.measured_usd == Decimal("0.000564")


# -- resume ------------------------------------------------------------------


def test_a_resumed_guard_starts_from_the_spend_the_ledger_already_holds() -> None:
    active = guard()
    active.seed(measured_usd=Decimal("40"), exposure_usd=Decimal("55"))

    assert active.measured_usd == Decimal("40")
    assert active.exposure_usd == Decimal("55")
    assert active.remaining_usd == Decimal("5")


def test_a_resumed_guard_refuses_a_request_the_remaining_budget_cannot_cover() -> None:
    active = guard()
    active.seed(measured_usd=Decimal("99.999999"), exposure_usd=Decimal("0"))

    with pytest.raises(CostCapExceededError):
        active.authorize(input_tokens_upper_bound=1000)


def test_seeding_twice_is_refused_so_prior_spend_cannot_be_overwritten() -> None:
    active = guard()
    active.seed(measured_usd=Decimal("40"), exposure_usd=Decimal("0"))
    with pytest.raises(BudgetError):
        active.seed(measured_usd=Decimal("0"), exposure_usd=Decimal("0"))


def test_seeding_a_negative_prior_spend_is_refused() -> None:
    """A negative seed is the one edit that would hand a run free budget."""
    active = guard()
    with pytest.raises(BudgetError):
        active.seed(measured_usd=Decimal("-1"), exposure_usd=Decimal("0"))


def test_a_seed_beyond_the_cap_leaves_no_budget_and_refuses_everything() -> None:
    active = guard()
    active.seed(measured_usd=Decimal("120"), exposure_usd=Decimal("0"))

    assert active.remaining_usd == Decimal(0)
    with pytest.raises(CostCapExceededError):
        active.authorize(input_tokens_upper_bound=1)


# -- the request-size bound --------------------------------------------------


def test_the_input_bound_is_at_least_the_serialised_request_and_an_allowance() -> None:
    """A token encodes at least one UTF-8 byte, so bytes bound tokens from above.

    The allowance on top covers what the body does not show: the provider wraps
    tools and system content in its own framing, which this build cannot see and
    must not assume is free.
    """
    assert conservative_input_token_bound(0) == REQUEST_OVERHEAD_TOKEN_ALLOWANCE
    assert conservative_input_token_bound(4096) == 4096 + REQUEST_OVERHEAD_TOKEN_ALLOWANCE


def test_a_negative_request_size_is_refused_rather_than_clamped() -> None:
    with pytest.raises(BudgetError):
        conservative_input_token_bound(-1)
