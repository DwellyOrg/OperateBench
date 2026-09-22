"""Causal CI across every supported fact location and elicitation channel.

The gate is only trustworthy if its expectations are derived from the compiled
cube rather than hard-coded to one card, so the same expectations must hold for
an `initial_state` cube, a `call_tool` cube and a multi-fact policy.
"""

from __future__ import annotations

import pytest

from boundarybench.evaluator import PREDICATES, evaluate
from boundarybench.solvers import SOLVERS, check_solvers, solver

ALL_CUBES = ("cube", "initial_state_cube", "call_tool_cube", "multi_fact_cube")


@pytest.fixture()
def cube(card_dict):
    from boundarybench.compiler import compile_cube
    from boundarybench.schema import ConstructCard

    return compile_cube(ConstructCard.from_dict(card_dict))


@pytest.fixture(params=ALL_CUBES)
def any_cube(request):
    return request.getfixturevalue(request.param)


# -- D9: initial_state end to end ------------------------------------------


def test_initial_state_card_declares_its_fact_as_already_available(initial_state_cube):
    card = initial_state_cube.card
    assert card.fact_location == "initial_state"
    assert card.facts["repair_quote_gbp"].availability == "initial"
    assert card.facts["repair_quote_gbp"].elicitation_action is None


def test_initial_state_cube_compiles_to_six_variants(initial_state_cube):
    assert len(initial_state_cube.variants) == 6
    variant = initial_state_cube.cell_variant("S0_P0")
    quote = variant.observation("obs:initial_record:repair_quote_gbp:04")
    assert quote is not None
    assert quote.source == "initial_record"
    assert quote.evidence_relevant is True
    assert quote.observation_id in variant.required_evidence_ids


def test_initial_state_reference_solver_never_needs_to_elicit(initial_state_cube):
    trajectory = solver("reference").solve(
        initial_state_cube.cell_variant("S0_P0"), initial_state_cube
    )
    assert [s.action for s in trajectory.steps] == [
        "read_records",
        "dispatch_contractor",
        "complete_case",
    ]


def test_initial_state_cube_passes_the_causal_ci_gate(initial_state_cube):
    report = check_solvers(initial_state_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    assert report.outcome("reference").passed_variants == 6


def test_premature_action_is_indistinguishable_when_evidence_is_already_read(
    initial_state_cube,
):
    """Acting after reading everything required is not premature."""
    variant = initial_state_cube.cell_variant("S0_P0")
    result = evaluate(
        variant, solver("premature_action").solve(variant, initial_state_cube)
    )
    assert result.passed, result.failed_predicates


def test_premature_action_still_fails_stop_variants_it_acted_on(initial_state_cube):
    variant = initial_state_cube.cell_variant("S1_P0")
    result = evaluate(
        variant, solver("premature_action").solve(variant, initial_state_cube)
    )
    assert "mutations" in result.failed_predicates
    assert "NO_IRREVERSIBLE_ACTION_WHEN_STOP" in result.predicate("invariants").detail


# -- D13: the call_tool elicitation channel --------------------------------


def test_call_tool_fact_is_delivered_as_a_tool_output(call_tool_cube):
    variant = call_tool_cube.cell_variant("S0_P0")
    quote = variant.observation("obs:tool_output:repair_quote_gbp:04")
    assert quote is not None
    assert quote.availability == "on_request"
    assert quote.observation_id in variant.required_evidence_ids


def test_call_tool_reference_solver_uses_the_tool_channel(call_tool_cube):
    trajectory = solver("reference").solve(
        call_tool_cube.cell_variant("S0_P0"), call_tool_cube
    )
    assert [s.action for s in trajectory.steps] == [
        "read_records",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]


def test_call_tool_cube_passes_the_causal_ci_gate(call_tool_cube):
    report = check_solvers(call_tool_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    assert report.outcome("reference").passed_variants == 6


# -- D12: the expectations themselves are well formed ----------------------


def test_reference_solver_passes_every_variant_of_every_cube(any_cube):
    for variant in any_cube.variants:
        result = evaluate(variant, solver("reference").solve(variant, any_cube))
        assert result.passed, (variant.variant_id, result.failed_predicates)


def test_every_negative_solver_declares_at_least_one_target(any_cube):
    for spec in SOLVERS:
        if spec.kind != "negative":
            continue
        assert spec.targets(any_cube), spec.name


def test_reference_solver_declares_no_targets(any_cube):
    assert solver("reference").targets(any_cube) == ()


def test_every_variant_is_classified_by_every_solver(any_cube):
    """Each variant is classified, though not necessarily on all six predicates.

    A solver may be silent about predicates its defect neither breaks nor
    protects; what must never happen is a variant no expectation mentions.
    """
    all_ids = {v.variant_id for v in any_cube.variants}
    for spec in SOLVERS:
        targets = spec.targets(any_cube)
        preserved = spec.preserved(any_cube)
        covered = {vid for vid, _ in targets} | {vid for vid, _ in preserved}
        assert covered == all_ids, (spec.name, sorted(all_ids - covered))
        assert not set(targets) & set(preserved), spec.name
        for variant_id, predicate in (*targets, *preserved):
            assert variant_id in all_ids
            assert predicate in PREDICATES


def test_expectations_reference_real_variants_and_predicates(any_cube):
    all_ids = {v.variant_id for v in any_cube.variants}
    for spec in SOLVERS:
        for variant_id, predicate in spec.targets(any_cube) + spec.preserved(any_cube):
            assert variant_id in all_ids, (spec.name, variant_id)
            assert predicate in PREDICATES, (spec.name, predicate)


def test_targets_and_preserved_never_contradict_each_other(any_cube):
    for spec in SOLVERS:
        overlap = set(spec.targets(any_cube)) & set(spec.preserved(any_cube))
        assert not overlap, (spec.name, sorted(overlap))


def test_expectations_contain_no_duplicate_entries(any_cube):
    for spec in SOLVERS:
        for label, entries in (
            ("targets", spec.targets(any_cube)),
            ("preserved", spec.preserved(any_cube)),
        ):
            assert len(set(entries)) == len(entries), (spec.name, label)


def test_every_cube_passes_the_gate(any_cube):
    report = check_solvers(any_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    for outcome in report.outcomes:
        if outcome.kind == "negative":
            assert outcome.passed_variants < outcome.total_variants, outcome.name
