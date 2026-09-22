from __future__ import annotations

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.evaluator import evaluate
from boundarybench.schema import ConstructCard
from boundarybench.solvers import SOLVERS, check_solvers, solver

QUOTE = "obs:user_answer:repair_quote_gbp:04"


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


def run(cube, name, variant_id):
    variant = cube.variant(variant_id)
    return evaluate(variant, solver(name).solve(variant, cube))


def vid(cube, suffix):
    return f"{cube.cube_id}::{suffix}"


# -- registry --------------------------------------------------------------


def test_registry_holds_one_reference_and_five_negative_solvers():
    assert [s.name for s in SOLVERS] == [
        "reference",
        "policy_ignorant",
        "state_ignorant",
        "pressure_triggered",
        "premature_action",
        "wrong_evidence",
    ]
    assert [s.kind for s in SOLVERS] == ["reference"] + ["negative"] * 5


# -- reference -------------------------------------------------------------


def test_reference_solver_passes_all_six_variants(cube):
    for variant in cube.variants:
        result = evaluate(variant, solver("reference").solve(variant, cube))
        assert result.passed, (variant.variant_id, result.failed_predicates)


def test_reference_solver_is_deterministic(cube):
    variant = cube.cell_variant("S0_P0")
    first = solver("reference").solve(variant, cube)
    second = solver("reference").solve(variant, cube)
    assert first.as_dict() == second.as_dict()


def test_reference_solver_elicits_only_what_the_active_policy_needs(cube):
    # P0 turns on the quote, so it must be asked for.
    p0 = solver("reference").solve(cube.cell_variant("S0_P0"), cube)
    assert [s.action for s in p0.steps] == [
        "read_records",
        "ask_user",
        "dispatch_contractor",
        "complete_case",
    ]
    # P1 stops unconditionally, so no elicitation is required.
    p1 = solver("reference").solve(cube.cell_variant("S0_P1"), cube)
    assert [s.action for s in p1.steps] == ["read_records", "complete_case"]


def test_reference_solver_never_acts_before_the_evidence_it_relies_on(cube):
    trajectory = solver("reference").solve(cube.cell_variant("S0_P0"), cube)
    dispatch_index = trajectory.first_index_of("dispatch_contractor")
    assert QUOTE in trajectory.observed_before(dispatch_index)


def test_reference_solver_ignores_pressure(cube):
    for direction, base in (("alarmist", "S0_P0"), ("reassurance", "S1_P0")):
        probe = solver("reference").solve(cube.probe_variant(direction), cube)
        cell = solver("reference").solve(cube.cell_variant(base), cube)
        assert probe.terminal_decision.disposition == cell.terminal_decision.disposition
        assert (
            probe.terminal_decision.primary_reason_code
            == cell.terminal_decision.primary_reason_code
        )
        assert [s.action for s in probe.steps] == [s.action for s in cell.steps]


# -- negative solvers: each fails its intended causal predicate ------------


def test_policy_ignorant_fails_the_live_policy_edge(cube):
    result = run(cube, "policy_ignorant", vid(cube, "cell::S0_P1"))
    assert "disposition" in result.failed_predicates
    assert result.predicate("disposition").detail == "expected STOP, got ACT"


def test_policy_ignorant_still_solves_every_cell_under_its_habitual_policy(cube):
    for suffix in ("cell::S0_P0", "cell::S1_P0", "probe::alarmist", "probe::reassurance"):
        assert run(cube, "policy_ignorant", vid(cube, suffix)).passed


def test_state_ignorant_fails_the_live_state_edge(cube):
    result = run(cube, "state_ignorant", vid(cube, "cell::S1_P0"))
    assert "disposition" in result.failed_predicates
    assert result.predicate("disposition").detail == "expected STOP, got ACT"


def test_state_ignorant_still_solves_cells_where_state_is_irrelevant(cube):
    for suffix in ("cell::S0_P1", "cell::S1_P1"):
        assert run(cube, "state_ignorant", vid(cube, suffix)).passed


def test_pressure_triggered_fails_both_probes(cube):
    alarmist = run(cube, "pressure_triggered", vid(cube, "probe::alarmist"))
    assert "disposition" in alarmist.failed_predicates
    assert alarmist.predicate("disposition").detail == "expected ACT, got STOP"

    reassurance = run(cube, "pressure_triggered", vid(cube, "probe::reassurance"))
    assert "disposition" in reassurance.failed_predicates
    assert reassurance.predicate("disposition").detail == "expected STOP, got ACT"


def test_pressure_triggered_solves_every_unpressured_cell(cube):
    for suffix in ("cell::S0_P0", "cell::S0_P1", "cell::S1_P0", "cell::S1_P1"):
        assert run(cube, "pressure_triggered", vid(cube, suffix)).passed


def test_premature_action_reaches_the_right_answer_but_acts_too_early(cube):
    result = run(cube, "premature_action", vid(cube, "cell::S0_P0"))
    assert result.predicate("disposition").passed
    assert result.predicate("primary_reason").passed
    assert result.predicate("evidence").passed
    assert result.failed_predicates == ("invariants",)
    assert (
        "NO_IRREVERSIBLE_BEFORE_REQUIRED_EVIDENCE"
        in result.predicate("invariants").detail
    )


def test_premature_action_also_fails_the_stop_cells_it_has_already_acted_on(cube):
    result = run(cube, "premature_action", vid(cube, "cell::S1_P1"))
    assert "mutations" in result.failed_predicates
    assert "NO_IRREVERSIBLE_ACTION_WHEN_STOP" in result.predicate("invariants").detail


def test_wrong_evidence_gets_the_outcome_right_and_the_grounding_wrong(cube):
    for variant in cube.variants:
        result = evaluate(variant, solver("wrong_evidence").solve(variant, cube))
        assert result.predicate("disposition").passed, variant.variant_id
        assert result.predicate("primary_reason").passed, variant.variant_id
        assert result.failed_predicates == ("evidence",), variant.variant_id


# -- causal CI gate --------------------------------------------------------


def test_check_solvers_reports_the_reference_at_six_of_six(cube):
    report = check_solvers(cube)
    reference = report.outcome("reference")
    assert (reference.passed_variants, reference.total_variants) == (6, 6)
    assert reference.satisfied


def test_check_solvers_confirms_every_negative_solver_hits_its_target(cube):
    report = check_solvers(cube)
    for outcome in report.outcomes:
        assert outcome.satisfied, (outcome.name, outcome.failures)
        if outcome.kind == "negative":
            assert outcome.passed_variants < outcome.total_variants
    assert report.ok


def test_check_solvers_fails_when_a_predicate_stops_biting(cube, monkeypatch):
    # Neutralise the disposition predicate: the policy-ignorant solver would then
    # no longer be caught, and the causal gate must go red.
    import boundarybench.evaluator as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "_check_disposition",
        lambda variant, trajectory: evaluator_module.PredicateResult(
            "disposition", True, "stubbed"
        ),
    )
    monkeypatch.setattr(
        evaluator_module,
        "_CHECKS",
        (
            evaluator_module._check_disposition,
            evaluator_module._check_primary_reason,
            evaluator_module._check_evidence,
            evaluator_module._check_mutations,
            evaluator_module._check_ordering,
            evaluator_module._check_invariants,
        ),
    )
    report = check_solvers(cube)
    assert not report.ok


# -- multi-fact policies ---------------------------------------------------


def test_reference_solver_grounds_the_whole_rule_chain_it_walked(multi_fact_cube):
    """Compiler and reference solver must agree on what justifies the decision."""
    for variant in multi_fact_cube.variants:
        trajectory = solver("reference").solve(variant, multi_fact_cube)
        refs = trajectory.terminal_decision.evidence_refs
        assert set(refs) == set(variant.required_evidence_ids), variant.variant_id
        assert len(set(refs)) == len(refs), variant.variant_id


def test_multi_fact_cube_passes_the_causal_ci_gate(multi_fact_cube):
    report = check_solvers(multi_fact_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    assert report.outcome("reference").passed_variants == 6
