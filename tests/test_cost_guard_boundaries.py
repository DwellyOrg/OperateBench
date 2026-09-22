"""Strict reading, tampering and boundary conditions in the cost guard.

:mod:`boundarybench.budget` is the only thing standing between a plan and real
money, so the interesting cases are not the ones where everything is well
formed. They are the ones where a stored manifest has been edited, where a
provider reports a token count that is not a token count, and where the guard is
asked to do two things at once. Each of those has exactly one safe answer:
refuse, or keep the conservative bound. Nothing here may resolve to "assume it
was free".

Nothing in this module contacts a provider or reads a credential.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import pytest

from boundarybench.budget import (
    BudgetError,
    CostControls,
    RunCostGuard,
    parse_cost_cap,
)
from boundarybench.pricing import (
    PRICING_POLICY_VERSION,
    price_for,
    pricing_policy_payload,
)

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"


def _controls(cap: str = "100", episodes: int | None = 72) -> CostControls:
    return CostControls(
        max_cost_usd=Decimal(cap),
        max_episodes=episodes,
        price=price_for(provider="anthropic", model=SONNET),
    )


def _stored(**overrides: Any) -> dict[str, Any]:
    """A well-formed stored block, so each test edits exactly one thing."""
    block = _controls().as_dict()
    block.update(overrides)
    return block


def _guard(cap: str = "100") -> RunCostGuard:
    return RunCostGuard(
        controls=_controls(cap),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )


# -- what an operator may type -----------------------------------------------


@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity", "-inf"])
def test_a_cap_that_is_not_a_finite_amount_is_refused(value: str) -> None:
    """``Decimal`` parses these; none of them is an amount of money.

    ``Infinity`` is the dangerous one: it parses, it compares greater than every
    reservation, and a run carrying it would authorise every request forever
    while appearing to have a cap.
    """
    with pytest.raises(BudgetError) as refused:
        parse_cost_cap(value)
    assert "finite" in str(refused.value)


def test_a_cap_that_is_not_a_decimal_is_refused_rather_than_converted() -> None:
    """A binary float is not the amount that was authorised, so it is not a cap."""
    with pytest.raises(BudgetError):
        CostControls(
            max_cost_usd=100.10,  # type: ignore[arg-type]
            max_episodes=72,
            price=price_for(provider="anthropic", model=SONNET),
        )
    with pytest.raises(BudgetError):
        CostControls(
            max_cost_usd="100.10",  # type: ignore[arg-type]
            max_episodes=72,
            price=price_for(provider="anthropic", model=SONNET),
        )


# -- reading a stored block strictly -----------------------------------------


def test_a_stored_block_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(
            ["max_cost_usd", "100"],  # type: ignore[arg-type]
            provider="anthropic",
            model=SONNET,
        )
    assert "JSON object" in str(refused.value)


def test_an_unknown_field_in_a_stored_block_is_refused_not_ignored() -> None:
    """An unrecognised limit is a limit this build cannot enforce."""
    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(
            _stored(max_cost_eur="90"), provider="anthropic", model=SONNET
        )
    assert "max_cost_eur" in str(refused.value)


@pytest.mark.parametrize("field", ["max_cost_usd", "max_episodes", "pricing"])
def test_a_missing_field_is_refused_rather_than_defaulted(field: str) -> None:
    """A deleted cap must not read as "no limit"; the absent field is named."""
    block = _stored()
    del block[field]

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(block, provider="anthropic", model=SONNET)
    assert field in str(refused.value)


def test_a_stored_block_with_every_limit_null_reads_back_as_an_uncapped_run() -> None:
    """The absence has to be recorded explicitly, and it round-trips as absence."""
    controls = CostControls.from_stored(
        {"max_cost_usd": None, "max_episodes": None, "pricing": None},
        provider="anthropic",
        model=SONNET,
    )

    assert controls.max_cost_usd is None
    assert controls.max_episodes is None
    assert controls.price is None
    assert controls.enforces_cost is False
    assert controls.as_dict() == {
        "max_cost_usd": None,
        "max_episodes": None,
        "pricing": None,
    }


@pytest.mark.parametrize("episodes", ["36", 36.0, True])
def test_a_stored_episode_limit_that_is_not_an_integer_is_refused(
    episodes: object,
) -> None:
    """``True`` included: ``bool`` is an ``int``, and it is not one episode."""
    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(
            _stored(max_episodes=episodes), provider="anthropic", model=SONNET
        )
    assert "max_episodes" in str(refused.value)


def test_a_stored_pricing_block_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(
            _stored(pricing="claude-sonnet-5 at USD 2/10"),
            provider="anthropic",
            model=SONNET,
        )
    assert "pricing" in str(refused.value)


def test_stored_pricing_for_another_model_is_refused() -> None:
    """It prices something this run did not execute, however valid it looks."""
    block = _stored()
    pricing = dict(block["pricing"])
    pricing["model"] = HAIKU
    block["pricing"] = pricing

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(block, provider="anthropic", model=SONNET)
    assert "model" in str(refused.value)


def test_a_stored_price_for_a_model_this_build_cannot_price_is_refused() -> None:
    """Consistent with itself and still unusable: the shipped table is authority."""
    block = _stored()
    pricing = dict(block["pricing"])
    pricing["model"] = "claude-opus-5"
    block["pricing"] = pricing

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(block, provider="anthropic", model="claude-opus-5")
    assert "claude-opus-5" in str(refused.value)


def test_a_policy_version_this_build_does_not_ship_is_refused_not_repriced() -> None:
    """Costs are only comparable within one price table."""
    block = _stored()
    pricing = dict(block["pricing"])
    pricing["policy_version"] = "anthropic-standard-2025-01-01"
    block["pricing"] = pricing

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(block, provider="anthropic", model=SONNET)
    message = str(refused.value)
    assert "anthropic-standard-2025-01-01" in message
    assert PRICING_POLICY_VERSION in message


def test_prices_edited_under_an_unchanged_version_are_refused() -> None:
    """The tamper the version check alone would miss.

    An editor who changes the rates and recomputes the pricing digest over them,
    leaving the version string alone, produces a block that is internally
    consistent and describes a price table this build has never shipped. The
    recorded digest is compared against the shipped one, so it is refused.
    """
    block = _stored()
    pricing = dict(block["pricing"])
    pricing["input_usd_per_million_tokens"] = "0.0001"
    pricing["policy_digest"] = "0" * 64
    block["pricing"] = pricing

    with pytest.raises(BudgetError) as refused:
        CostControls.from_stored(block, provider="anthropic", model=SONNET)
    assert "digest" in str(refused.value)


def test_an_untouched_stored_block_still_carries_this_build_s_policy() -> None:
    """The negative control: the checks above refuse edits, not every manifest."""
    stored: Mapping[str, Any] = _stored()
    controls = CostControls.from_stored(stored, provider="anthropic", model=SONNET)
    policy = pricing_policy_payload()

    assert controls.max_cost_usd == Decimal("100")
    assert controls.price is not None
    assert controls.price.model == SONNET
    assert stored["pricing"]["policy_digest"] == policy["policy_digest"]
    # And the price is the shipped one, not whatever the file said.
    assert controls.price is price_for(provider="anthropic", model=SONNET)


# -- what a guard may be built on --------------------------------------------


def test_no_guard_is_built_for_a_run_that_has_no_budget_to_enforce() -> None:
    """A guard over an uncapped run would appear to enforce something."""
    with pytest.raises(BudgetError):
        RunCostGuard(
            controls=CostControls(max_cost_usd=None, max_episodes=72, price=None),
            max_output_tokens=1024,
            max_attempts_per_turn=3,
        )


@pytest.mark.parametrize("tokens", [0, -1, 1024.0, True])
def test_a_guard_refuses_an_output_ceiling_that_cannot_bound_a_request(
    tokens: object,
) -> None:
    """Every reservation is computed from this number; zero would reserve nothing."""
    with pytest.raises(BudgetError) as refused:
        RunCostGuard(
            controls=_controls(),
            max_output_tokens=tokens,  # type: ignore[arg-type]
            max_attempts_per_turn=3,
        )
    assert "max_output_tokens" in str(refused.value)


@pytest.mark.parametrize("attempts", [0, -1, 3.0, True])
def test_a_guard_refuses_an_attempt_ceiling_that_cannot_bound_a_turn(
    attempts: object,
) -> None:
    with pytest.raises(BudgetError) as refused:
        RunCostGuard(
            controls=_controls(),
            max_output_tokens=1024,
            max_attempts_per_turn=attempts,  # type: ignore[arg-type]
        )
    assert "max_attempts_per_turn" in str(refused.value)


# -- ordering: one request at a time -----------------------------------------


def test_an_episode_cannot_begin_while_a_request_is_still_outstanding() -> None:
    """Beginning would zero the episode counters and strand the reservation.

    The reservation is still held afterwards, which is the part that matters: a
    guard that lost it would have released budget for a request whose outcome it
    never learned.
    """
    guard = _guard()
    reservation = guard.authorize(input_tokens_upper_bound=1000)

    with pytest.raises(BudgetError) as refused:
        guard.begin_episode()
    assert "outstanding" in str(refused.value)

    assert guard.remaining_usd == Decimal("100") - reservation
    # And the guard is still usable: the outstanding request can still resolve.
    guard.forfeit(reservation)
    assert guard.exposure_usd == reservation


# -- resolving a request whose usage cannot be read --------------------------


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (None, None),
        (137, None),
        (None, 29),
    ],
)
def test_a_response_with_half_a_usage_report_is_unmeasured_not_free(
    input_tokens: int | None, output_tokens: int | None
) -> None:
    """Pricing the reported half at zero for the other would understate spend."""
    guard = _guard()
    reservation = guard.authorize(input_tokens_upper_bound=1000)

    measured = guard.settle(
        reservation, input_tokens=input_tokens, output_tokens=output_tokens
    )

    assert measured is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd == reservation
    assert guard.episode_exposure_usd == reservation
    assert guard.remaining_usd == Decimal("100") - reservation


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (-1, 29),
        (137, -1),
        (True, 29),
    ],
)
def test_a_token_count_the_price_policy_refuses_keeps_its_reservation(
    input_tokens: int, output_tokens: int
) -> None:
    """The one direction that breaches a cap is dropping a reservation.

    A provider that reports a negative or non-integer count has told this build
    nothing usable. The guard must not raise here and must not release the
    budget it was holding: it converts the reservation to exposure, exactly as
    for a response that reported no usage at all. (The runner separately refuses
    the malformed measurement; that is a different job from the cap.)
    """
    guard = _guard()
    reservation = guard.authorize(input_tokens_upper_bound=1000)

    measured = guard.settle(
        reservation, input_tokens=input_tokens, output_tokens=output_tokens
    )

    assert measured is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd == reservation
    # The guard is resolved, so the next request can be authorised normally.
    assert guard.authorize(input_tokens_upper_bound=1000) == reservation
