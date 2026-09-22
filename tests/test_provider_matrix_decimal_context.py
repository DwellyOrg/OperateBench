"""The matrix split is the same money whatever context the caller is in.

``decimal`` arithmetic reads its precision, its rounding mode and its traps from
*process-wide* state that any caller — a library, a notebook, a test that
forgot to restore what it changed — can set. Every other monetary computation
in this build already knows that and runs inside
:func:`~boundarybench.pricing.exact_decimal_context`; the matrix split did not,
so the amount five models were approved for depended on what the ambient
context happened to be when the plan was built.

Under the default context it produced the right answer, which is what made it
worth a module of its own: the failure is invisible until the process it runs
in is not the process the split was designed in. At ``prec=2`` with
``ROUND_UP`` the same cap split five ways came back as ``1.5E+2`` and ``2E+1``
— amounts that are not cents, do not sum to the cap, and would have been
written into an approval artefact as the numbers an operator agreed to.

So the whole computation — the weight sum, the multiply and divide, the floor,
the share sum, the remainder and the final adjustment — belongs inside the
exact context, and so does the check that re-adds the parts. These tests state
that as a property over hostile contexts rather than as a spot check: the
allocations are byte-identical to the baseline, they sum to the cap exactly,
and the ambient context is left exactly as it was found.

Nothing here changes global decimal state. Every context is entered with
``localcontext`` and every test asserts the ambient one afterwards.
"""

from __future__ import annotations

import decimal
from decimal import (
    ROUND_05UP,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_UP,
    Context,
    Decimal,
    Inexact,
    Rounded,
    localcontext,
)
from pathlib import Path

import pytest

from boundarybench.budget import usd_text
from boundarybench.providermatrix import (
    MATRIX_MODELS,
    ModelAllocation,
    build_matrix_plan,
    proportional_allocations,
    worst_case_unit_cost,
)

#: The cap the shipped matrix is planned against, and the one the parent repro
#: split five ways.
GLOBAL_CAP = Decimal("400")

#: Contexts a caller can leave behind, each hostile in a different way: too few
#: digits to hold a cent, a rounding mode that inflates, one that deflates, one
#: that is neither, and a context that raises on the very conditions exact
#: arithmetic over this domain must never produce.
HOSTILE_CONTEXTS = {
    "prec-2-round-up": Context(prec=2, rounding=ROUND_UP),
    "prec-2-round-down": Context(prec=2, rounding=ROUND_DOWN),
    "prec-1-round-ceiling": Context(prec=1, rounding=ROUND_CEILING),
    "prec-3-round-floor": Context(prec=3, rounding=ROUND_FLOOR),
    "prec-5-round-05up": Context(prec=5, rounding=ROUND_05UP),
    "prec-28-trapping": Context(prec=28, traps=[Inexact, Rounded]),
    "prec-9-trapping": Context(prec=9, traps=[Inexact, Rounded]),
}


def _texts(allocations: tuple[ModelAllocation, ...]) -> tuple[str, ...]:
    """The exact rendering of each share, in matrix order.

    Compared as text rather than as numbers because ``Decimal("1.5E+2")`` and
    ``Decimal("150")`` are equal and are not the same amount to write into an
    artefact: one states cents and one does not, and the artefact carries the
    string.
    """
    return tuple(usd_text(allocation.cost_usd) for allocation in allocations)


def _tuples(allocations: tuple[ModelAllocation, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (allocation.provider, allocation.model, str(allocation.cost_usd))
        for allocation in allocations
    )


@pytest.fixture(autouse=True)
def _restore_ambient_context():
    """Every test here leaves the process's decimal context exactly as it was."""
    before = decimal.getcontext().copy()
    yield
    after = decimal.getcontext()
    assert after.prec == before.prec
    assert after.rounding == before.rounding
    assert after.traps == before.traps


def test_the_baseline_split_is_the_one_the_approval_names() -> None:
    """The shipped answer, stated once so every other test can compare to it."""
    allocations = proportional_allocations(GLOBAL_CAP)
    assert _texts(allocations) == ("85.25", "170.54", "19.89", "113.67", "10.65")
    assert (
        sum((allocation.cost_usd for allocation in allocations), Decimal(0)) == GLOBAL_CAP
    )


@pytest.mark.parametrize("name", sorted(HOSTILE_CONTEXTS))
def test_an_ambient_context_cannot_change_what_the_matrix_is_approved(
    name: str,
) -> None:
    """The same plan, to the digit, from inside a context that cannot hold it.

    Not merely "close": identical strings. An allocation that rounded to the
    same value in a different scale is a different artefact, and an artefact is
    what an approval is.
    """
    baseline = proportional_allocations(GLOBAL_CAP)
    with localcontext(HOSTILE_CONTEXTS[name]):
        under_pressure = proportional_allocations(GLOBAL_CAP)
    assert _tuples(under_pressure) == _tuples(baseline)
    assert _texts(under_pressure) == _texts(baseline)


@pytest.mark.parametrize("name", sorted(HOSTILE_CONTEXTS))
def test_the_parts_still_equal_the_whole_under_any_context(name: str) -> None:
    """The invariant the split exists to hold, checked where it used to break.

    A short split silently under-spends an approval and an over-split spends
    more than was approved. Under ``prec=2`` the pre-fix split did both at once:
    the shares no longer summed to the cap at all.
    """
    with localcontext(HOSTILE_CONTEXTS[name]):
        allocations = proportional_allocations(GLOBAL_CAP)
    # Re-added outside the hostile context: what is asserted is that the parts
    # the caller was handed are the exact ones, not that a one-digit context
    # can add them up.
    total = sum((entry.cost_usd for entry in allocations), Decimal(0))
    assert total == GLOBAL_CAP
    assert usd_text(total) == usd_text(GLOBAL_CAP)


@pytest.mark.parametrize("name", sorted(HOSTILE_CONTEXTS))
def test_the_plan_that_validates_the_split_agrees_under_any_context(
    name: str, tmp_path: Path
) -> None:
    """``build_matrix_plan`` re-adds the shares, and that sum is money too.

    It is the check that refuses a split which does not equal its cap, so a
    context that rounds it turns the validator into the thing that rejects a
    correct plan — or, in the other direction, accepts an incorrect one.
    """
    baseline = build_matrix_plan(
        output_root=tmp_path,
        total_cost_usd=GLOBAL_CAP,
        allocations=proportional_allocations(GLOBAL_CAP),
        environ={},
    )
    with localcontext(HOSTILE_CONTEXTS[name]):
        under_pressure = build_matrix_plan(
            output_root=tmp_path,
            total_cost_usd=GLOBAL_CAP,
            allocations=proportional_allocations(GLOBAL_CAP),
            environ={},
        )
    assert under_pressure.as_dict() == baseline.as_dict()


@pytest.mark.parametrize("name", sorted(HOSTILE_CONTEXTS))
def test_the_weights_the_split_is_shaped_by_do_not_move_either(name: str) -> None:
    """Each cell's worst-case unit cost is priced in the exact context already.

    Asserted here because it is the input the split is proportional to: a
    weight that moved with the ambient context would move every share even if
    the division that used it were exact.
    """
    baseline = [
        worst_case_unit_cost(entry.provider, entry.model, output_tokens=32_768)
        for entry in MATRIX_MODELS
    ]
    with localcontext(HOSTILE_CONTEXTS[name]):
        under_pressure = [
            worst_case_unit_cost(entry.provider, entry.model, output_tokens=32_768)
            for entry in MATRIX_MODELS
        ]
    assert [str(value) for value in under_pressure] == [str(value) for value in baseline]


def test_the_call_does_not_leak_its_own_context_back_to_the_caller() -> None:
    """The exact context is entered and left, not installed.

    A fix that widened the process context instead of scoping one would make
    every later computation in the caller's process silently exact too — which
    hides the next instance of this defect rather than fixing it.
    """
    with localcontext(Context(prec=2, rounding=ROUND_UP)) as ctx:
        proportional_allocations(GLOBAL_CAP)
        assert decimal.getcontext() is ctx
        assert decimal.getcontext().prec == 2
        assert decimal.getcontext().rounding == ROUND_UP
        # And the caller's own arithmetic is still the caller's: this build
        # scoped its exactness, it did not impose it.
        assert str(Decimal("1") / Decimal("3")) == "0.34"


def test_the_split_is_unchanged_under_the_ordinary_context() -> None:
    """Deliberate non-movement.

    Restoring promised exact arithmetic must not move the answer anybody has
    already approved. The default context is the one every existing plan was
    built in, and the amounts under it are the same amounts as before.
    """
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        allocations = proportional_allocations(GLOBAL_CAP)
    assert _texts(allocations) == ("85.25", "170.54", "19.89", "113.67", "10.65")
    assert len(allocations) == len(MATRIX_MODELS)
    assert [entry.key for entry in allocations] == [
        (entry.provider, entry.model) for entry in MATRIX_MODELS
    ]
