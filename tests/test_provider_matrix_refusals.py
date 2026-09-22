"""What the matrix plan and its approvals refuse before any money is committed.

The plan is the artefact an operator reads and approves, and the authorisation
pass is the last thing between that approval and a run that spends against it.
Every refusal here is a way an approval could end up spent on an experiment
nobody approved: a cap that is not an exact amount, a split that leaves a cell
uncapped, a model this build cannot price and therefore cannot cap, an approval
for a cell the plan does not contain, and a cell whose configuration identity
was never computed.

Nothing here contacts a provider, reads a credential or creates a run directory.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    MatrixAuthorization,
    MatrixError,
    MatrixModel,
    ModelAllocation,
    PlannedModel,
    build_matrix_plan,
    check_authorizations,
    proportional_allocations,
)

GLOBAL_CAP = Decimal("50.00")
CONFIGURATION_ID = "c" * 64


def _plan(tmp_path: Path, **overrides: object):
    kwargs: dict = {
        "output_root": tmp_path / "matrix",
        "total_cost_usd": GLOBAL_CAP,
        "allocations": proportional_allocations(GLOBAL_CAP),
    }
    kwargs.update(overrides)
    return build_matrix_plan(**kwargs)


def _authorisation(entry: PlannedModel) -> MatrixAuthorization:
    return MatrixAuthorization(
        provider=entry.provider,
        model=entry.model,
        configuration_id=CONFIGURATION_ID,
        episode_cap=entry.episode_cap,
        cost_allocation_usd=entry.cost_allocation_usd,
        output_dir=entry.output_dir,
        live_enabled=True,
    )


# -- the cap has to be an exact amount ----------------------------------------


@pytest.mark.parametrize(
    "cap,message",
    [
        (50.0, "Decimal"),
        (Decimal("0"), "positive"),
        (Decimal("-1.00"), "positive"),
    ],
    ids=["binary float", "zero", "negative"],
)
def test_a_global_cap_that_is_not_an_exact_positive_amount_is_refused(
    cap: object, message: str, tmp_path: Path
) -> None:
    """A cap is money, and money is not a binary float.

    ``50.0`` is not ``50.00``: a cap held as a binary float would be compared
    against exact decimal spend, and the comparison that stops a run would then
    depend on a representation error nobody approved. A cap of zero or less
    authorises nothing, and a plan built under one would describe a run it does
    not permit.

    Refused in both places a cap enters — the split and the plan — because either
    is reachable on its own.
    """
    with pytest.raises(MatrixError) as split:
        proportional_allocations(cap)  # type: ignore[arg-type]
    assert message in str(split.value)

    with pytest.raises(MatrixError) as plan:
        _plan(tmp_path, total_cost_usd=cap, allocations=())
    assert message in str(plan.value)


@pytest.mark.parametrize(
    "amount,message",
    [(1.0, "Decimal"), (Decimal("-0.01"), "negative")],
    ids=["binary float", "negative"],
)
def test_a_per_model_allocation_that_is_not_an_exact_amount_is_refused(
    amount: object, message: str
) -> None:
    """One cell's share is checked on its own terms, not only in the sum.

    A negative share would let one cell's cap fund another's, which is the one
    thing the split exists to prevent: there is no borrowing between models.
    """
    with pytest.raises(MatrixError) as raised:
        ModelAllocation(provider="openai", model="gpt-5.6-luna", cost_usd=amount)  # type: ignore[arg-type]
    assert message in str(raised.value)


def test_a_cell_with_no_allocation_at_all_is_refused(tmp_path: Path) -> None:
    """A missing share is not a share of zero: it is a cell with no cap.

    Dropping one allocation and reducing the cap to match would otherwise
    produce a plan that sums correctly and leaves one model running under no
    enforced cap at all.
    """
    allocations = proportional_allocations(GLOBAL_CAP)
    kept = allocations[:-1]
    dropped = allocations[-1]
    with pytest.raises(MatrixError) as raised:
        _plan(
            tmp_path,
            total_cost_usd=GLOBAL_CAP - dropped.cost_usd,
            allocations=kept,
        )
    assert "do not cover" in str(raised.value)
    assert dropped.model in str(raised.value)


def test_a_model_this_build_cannot_price_cannot_be_capped(tmp_path: Path) -> None:
    """A cap that cannot be enforced against measured spend is not a control.

    The guard prices every turn from the run's pinned rates, so a cell whose
    model has no reviewed price could hold a dollar allocation and never check a
    single request against it. Refused at planning time, before anything runs.
    """
    unpriced = MatrixModel(
        provider="openai", model="gpt-not-a-priced-model", slug="openai-unpriced"
    )
    with pytest.raises(MatrixError) as raised:
        _plan(
            tmp_path,
            models=(unpriced,),
            total_cost_usd=GLOBAL_CAP,
            allocations=(
                ModelAllocation(
                    provider=unpriced.provider,
                    model=unpriced.model,
                    cost_usd=GLOBAL_CAP,
                ),
            ),
        )
    assert "no reviewed price" in str(raised.value)


# -- an approval is for one plan, and for all of it ----------------------------


def test_an_approval_for_a_cell_this_plan_does_not_contain_is_refused(
    tmp_path: Path,
) -> None:
    """What a stale approval looks like: one for an experiment this plan is not.

    Acting on it would spend a previous audit's authority on this tree, which is
    the failure an approval is supposed to make impossible rather than the one it
    causes.
    """
    plan = _plan(tmp_path)
    approvals = [_authorisation(entry) for entry in plan.models]
    approvals.append(
        MatrixAuthorization(
            provider="openai",
            model="gpt-from-a-previous-plan",
            configuration_id=CONFIGURATION_ID,
            episode_cap=EPISODES_PER_MODEL,
            cost_allocation_usd=Decimal("1.00"),
            output_dir=tmp_path / "stale",
            live_enabled=True,
        )
    )
    identities = {
        (entry.provider, entry.model): CONFIGURATION_ID for entry in plan.models
    }
    with pytest.raises(MatrixError) as raised:
        check_authorizations(plan, approvals, configuration_ids=identities)
    assert "does not contain" in str(raised.value)


def test_a_cell_with_no_approval_at_all_is_refused(tmp_path: Path) -> None:
    """One model's approval is not another's, so a missing one is not covered."""
    plan = _plan(tmp_path)
    approvals = [_authorisation(entry) for entry in plan.models[1:]]
    identities = {
        (entry.provider, entry.model): CONFIGURATION_ID for entry in plan.models
    }
    with pytest.raises(MatrixError) as raised:
        check_authorizations(plan, approvals, configuration_ids=identities)
    assert "no authorisation" in str(raised.value)
    assert plan.models[0].model in str(raised.value)


def test_a_cell_whose_configuration_identity_was_never_computed_is_refused(
    tmp_path: Path,
) -> None:
    """With no identity there is nothing to check the approval against.

    An absent identity is not a mismatch and must not be treated as one: it means
    the offline preflight that computes what this run would execute under has not
    been run for that cell, so the approval cannot be bound to anything.
    """
    plan = _plan(tmp_path)
    approvals = [_authorisation(entry) for entry in plan.models]
    identities = {
        (entry.provider, entry.model): CONFIGURATION_ID for entry in plan.models[1:]
    }
    with pytest.raises(MatrixError) as raised:
        check_authorizations(plan, approvals, configuration_ids=identities)
    assert "no configuration identity" in str(raised.value)
    assert "preflight" in str(raised.value)


def test_the_shipped_matrix_still_plans_and_authorises_cleanly(tmp_path: Path) -> None:
    """The refusals above must not have made the approved plan unapprovable."""
    plan = _plan(tmp_path)
    assert len(plan.models) == len(MATRIX_MODELS)
    check_authorizations(
        plan,
        [_authorisation(entry) for entry in plan.models],
        configuration_ids={
            (entry.provider, entry.model): CONFIGURATION_ID for entry in plan.models
        },
    )
