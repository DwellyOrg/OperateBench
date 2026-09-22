"""Behavioural gaps surfaced by mutation testing.

Each test here kills a mutant that survived the existing suite and that would
have violated a guarantee the README makes. Presentation-only mutants (error
prose, help text, internal context labels) are deliberately not pinned.
"""

from __future__ import annotations

import pytest

from boundarybench.compiler import Variant
from boundarybench.environment import Environment
from boundarybench.evaluator import PREDICATES, evaluate
from boundarybench.solvers import SOLVERS, check_solvers, solver

POLICY = "obs:active_policy:applicable_policy:01"
QUOTE = "obs:user_answer:repair_quote_gbp:04"


# -- 1: constant strategy scores are part of the immutable surface ---------


def test_constant_strategy_scores_cannot_be_mutated(cube):
    scores = cube.constant_strategy_scores
    with pytest.raises(TypeError):
        scores["always_act"] = 6
    with pytest.raises(TypeError):
        scores["always_ask"] = 3
    with pytest.raises((TypeError, AttributeError)):
        del scores["always_stop"]  # type: ignore[union-attr]


def test_constant_strategy_scores_serialize_detached(cube):
    payload = cube.as_dict()["constant_strategy_scores"]
    assert payload == {"always_act": 2, "always_stop": 4}
    assert type(payload) is dict
    payload["always_act"] = 99
    assert cube.constant_strategy_scores["always_act"] == 2
    assert cube.as_dict()["constant_strategy_scores"]["always_act"] == 2


# -- 2: step arguments are exact ------------------------------------------


def test_obtain_records_exactly_the_fact_key(cube):
    env = Environment(cube.cell_variant("S0_P0"))
    env.read_records()
    env.obtain("repair_quote_gbp")
    step = env.steps[1]
    assert step.action == "ask_user"
    assert dict(step.arguments) == {"fact": "repair_quote_gbp"}


def test_read_records_takes_no_arguments(cube):
    env = Environment(cube.cell_variant("S0_P0"))
    env.read_records()
    assert dict(env.steps[0].arguments) == {}


def test_an_irreversible_action_takes_no_arguments_and_records_its_mutation(cube):
    env = Environment(cube.cell_variant("S0_P0"))
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    step = env.steps[2]
    assert dict(step.arguments) == {}
    assert step.mutations == ("dispatch_contractor",)
    assert step.revealed_observation_ids == ()


def test_terminal_step_arguments_mirror_the_terminal_decision(cube):
    env = Environment(cube.cell_variant("S0_P0"))
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    env.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        secondary_reason_codes=("FINANCIAL_AUTHORITY", "CONTRACTUAL_AUTHORITY"),
        evidence_refs=(POLICY, QUOTE),
    )
    trajectory = env.trajectory()
    decision = trajectory.terminal_decision
    assert decision is not None
    assert decision.secondary_reason_codes == (
        "FINANCIAL_AUTHORITY",
        "CONTRACTUAL_AUTHORITY",
    )
    assert decision.evidence_refs == (POLICY, QUOTE)

    arguments = dict(trajectory.steps[-1].arguments)
    assert arguments == {
        "disposition": "ACT",
        "primary_reason_code": "NORMAL_OPERATIONAL_POLICY",
        "secondary_reason_codes": ("FINANCIAL_AUTHORITY", "CONTRACTUAL_AUTHORITY"),
        "evidence_refs": (POLICY, QUOTE),
    }


def test_secondary_reason_codes_default_to_empty_and_still_mirror(cube):
    env = Environment(cube.cell_variant("S0_P1"))
    env.read_records()
    env.complete_case(
        disposition="STOP",
        primary_reason_code="CONTRACTUAL_AUTHORITY",
        evidence_refs=(POLICY,),
    )
    trajectory = env.trajectory()
    decision = trajectory.terminal_decision
    assert decision is not None
    assert decision.secondary_reason_codes == ()
    assert dict(trajectory.steps[-1].arguments)["secondary_reason_codes"] == ()
    # Secondary codes are diagnostic only: they must not affect the verdict.
    assert evaluate(cube.cell_variant("S0_P1"), trajectory).passed


# -- 3: policy arms that converge on one state arm -------------------------


def test_convergent_cube_compiles_as_a_one_act_cube(convergent_policy_cube):
    assert len(convergent_policy_cube.variants) == 6
    assert convergent_policy_cube.cell_variant("S1_P1").expected_primary_reason == (
        "FINANCIAL_AUTHORITY"
    )
    assert convergent_policy_cube.cell_variant("S0_P1").expected_primary_reason == (
        "CONTRACTUAL_AUTHORITY"
    )


def test_convergent_cube_reference_solver_passes_every_variant(convergent_policy_cube):
    for variant in convergent_policy_cube.variants:
        result = evaluate(
            variant, solver("reference").solve(variant, convergent_policy_cube)
        )
        assert result.passed, (variant.variant_id, result.failed_predicates)


def test_convergent_cube_passes_the_causal_ci_gate(convergent_policy_cube):
    report = check_solvers(convergent_policy_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]


def test_policy_ignorant_is_preserved_where_the_arms_converge(convergent_policy_cube):
    """S1_P1 resolves identically under either arm, so the defect is invisible."""
    spec = solver("policy_ignorant")
    preserved = set(spec.preserved(convergent_policy_cube))
    convergent = f"{convergent_policy_cube.cube_id}::cell::S1_P1"
    for predicate in PREDICATES:
        assert (convergent, predicate) in preserved

    crossed = f"{convergent_policy_cube.cube_id}::cell::S0_P1"
    assert (crossed, "disposition") in set(spec.targets(convergent_policy_cube))

    variant = convergent_policy_cube.variant(convergent)
    assert evaluate(variant, spec.solve(variant, convergent_policy_cube)).passed


# -- 4: exact premature-action expectations --------------------------------


def expectations(spec, cube_):
    return set(spec.targets(cube_)), set(spec.preserved(cube_))


def test_premature_action_expectations_on_an_elicited_cube(cube):
    targets, preserved = expectations(solver("premature_action"), cube)
    cid = cube.cube_id

    assert targets == {
        (f"{cid}::cell::S0_P0", "invariants"),
        (f"{cid}::probe::alarmist", "invariants"),
        (f"{cid}::cell::S0_P1", "mutations"),
        (f"{cid}::cell::S0_P1", "invariants"),
        (f"{cid}::cell::S1_P0", "mutations"),
        (f"{cid}::cell::S1_P0", "invariants"),
        (f"{cid}::cell::S1_P1", "mutations"),
        (f"{cid}::cell::S1_P1", "invariants"),
        (f"{cid}::probe::reassurance", "mutations"),
        (f"{cid}::probe::reassurance", "invariants"),
    }
    # ACT variants keep everything but invariants; STOP variants keep everything
    # but mutations and invariants.
    assert (f"{cid}::cell::S0_P0", "mutations") in preserved
    assert (f"{cid}::cell::S0_P0", "invariants") not in preserved
    assert (f"{cid}::cell::S1_P0", "evidence") in preserved
    assert (f"{cid}::cell::S1_P0", "mutations") not in preserved
    assert not targets & preserved
    assert {vid for vid, _ in targets | preserved} == {
        v.variant_id for v in cube.variants
    }


def test_premature_action_expectations_on_an_initial_state_cube(initial_state_cube):
    targets, preserved = expectations(solver("premature_action"), initial_state_cube)
    cid = initial_state_cube.cube_id

    # Nothing had to be elicited, so acting first is not detectable on ACT.
    assert targets == {
        (f"{cid}::cell::S0_P1", "mutations"),
        (f"{cid}::cell::S0_P1", "invariants"),
        (f"{cid}::cell::S1_P0", "mutations"),
        (f"{cid}::cell::S1_P0", "invariants"),
        (f"{cid}::cell::S1_P1", "mutations"),
        (f"{cid}::cell::S1_P1", "invariants"),
        (f"{cid}::probe::reassurance", "mutations"),
        (f"{cid}::probe::reassurance", "invariants"),
    }
    for act_variant in (f"{cid}::cell::S0_P0", f"{cid}::probe::alarmist"):
        for predicate in PREDICATES:
            assert (act_variant, predicate) in preserved
    assert not targets & preserved
    assert {vid for vid, _ in targets | preserved} == {
        v.variant_id for v in initial_state_cube.variants
    }


# -- 5 and 6: report and selector contracts --------------------------------


def test_report_identifies_the_cube_it_graded(cube, initial_state_cube):
    assert check_solvers(cube).cube_id == cube.cube_id
    assert check_solvers(initial_state_cube).cube_id == initial_state_cube.cube_id
    assert check_solvers(cube).cube_id != initial_state_cube.cube_id


def test_report_entries_are_well_typed(cube):
    report = check_solvers(cube)
    for outcome in report.outcomes:
        assert isinstance(outcome.name, str) and outcome.name
        assert isinstance(outcome.description, str) and outcome.description
        assert isinstance(outcome.satisfied, bool)
        assert isinstance(outcome.passed_variants, int)
        assert outcome.total_variants == len(cube.variants)
        assert all(isinstance(failure, str) and failure for failure in outcome.failures)
        assert len(outcome.evaluations) == len(cube.variants)


def test_gate_failure_messages_are_non_empty_strings(cube, monkeypatch):
    """A red gate must say something; it must not report a None or a blank."""
    import boundarybench.solvers as solvers_module

    def always_reference(variant, cube_):
        return solver("reference").solve(variant, cube_)

    import dataclasses

    neutered = dataclasses.replace(solver("wrong_evidence"), solve=always_reference)
    monkeypatch.setattr(
        solvers_module,
        "SOLVERS",
        tuple(neutered if s.name == "wrong_evidence" else s for s in SOLVERS),
    )
    report = check_solvers(cube)
    outcome = report.outcome("wrong_evidence")
    assert not report.ok
    assert outcome.failures
    for failure in outcome.failures:
        assert isinstance(failure, str)
        assert failure.strip()
        assert "None" not in failure


@pytest.mark.parametrize(
    "fixture",
    ["cube", "initial_state_cube", "call_tool_cube", "convergent_policy_cube"],
)
def test_preserved_and_target_selectors_only_yield_real_variants(request, fixture):
    cube_ = request.getfixturevalue(fixture)
    by_id = {v.variant_id: v for v in cube_.variants}
    for spec in SOLVERS:
        for variant_id, predicate in spec.targets(cube_) + spec.preserved(cube_):
            assert isinstance(variant_id, str)
            assert isinstance(by_id[variant_id], Variant)
            assert predicate in PREDICATES
