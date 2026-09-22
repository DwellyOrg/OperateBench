"""Remaining error and edge branches across the slice.

These are reachable failure paths rather than coverage filler: each one is a way
a card, a trajectory or the gate itself can be wrong.
"""

from __future__ import annotations

import dataclasses

import pytest

from boundarybench.cli import main
from boundarybench.compiler import CompileError, attempted_rules, compile_cube
from boundarybench.environment import Environment, EnvironmentError
from boundarybench.evaluator import evaluate
from boundarybench.schema import ConstructCard, Rule
from boundarybench.solvers import (
    SOLVERS,
    SolverError,
    _first_divergence,
    _required_observations,
    check_solvers,
    solver,
)
from boundarybench.trajectory import Observation, Step, TerminalDecision
from tests.conftest import EXAMPLE_CARD, initial_state_card, mutate

POLICY = "obs:active_policy:applicable_policy:01"
QUOTE = "obs:user_answer:repair_quote_gbp:04"


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


@pytest.fixture()
def honest(cube):
    variant = cube.cell_variant("S0_P0")
    env = Environment(variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    env.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        evidence_refs=(POLICY, QUOTE),
    )
    return env.trajectory()


def ordering_detail(variant, trajectory) -> str:
    result = evaluate(variant, trajectory)
    assert "ordering" in result.failed_predicates, result.failed_predicates
    return result.predicate("ordering").detail


# -- evaluator: remaining integrity branches -------------------------------


def test_a_non_revealing_action_may_not_reveal_observations(cube, honest):
    steps = (
        *honest.steps[:2],
        dataclasses.replace(honest.steps[2], revealed_observation_ids=(POLICY,)),
        *honest.steps[3:],
    )
    forged = dataclasses.replace(honest, steps=steps)
    assert "may not reveal" in ordering_detail(cube.cell_variant("S0_P0"), forged)


def test_a_terminal_decision_without_a_terminal_step_fails(cube, honest):
    forged = dataclasses.replace(honest, steps=honest.steps[:-1])
    detail = ordering_detail(cube.cell_variant("S0_P0"), forged)
    assert "no terminal step" in detail


def test_a_terminal_step_without_a_decision_fails(cube, honest):
    forged = dataclasses.replace(honest, terminal_decision=None)
    detail = ordering_detail(cube.cell_variant("S0_P0"), forged)
    assert "no terminal decision" in detail


def test_non_contiguous_step_indices_fail(cube, honest):
    steps = honest.steps[:1] + tuple(
        dataclasses.replace(step, index=step.index + 10) for step in honest.steps[1:]
    )
    forged = dataclasses.replace(honest, steps=steps)
    assert "contiguous" in ordering_detail(cube.cell_variant("S0_P0"), forged)


def test_actions_after_the_terminal_call_fail(cube, honest):
    steps = (
        *honest.steps,
        Step(
            index=len(honest.steps) + 1,
            action="read_records",
            revealed_observation_ids=honest.steps[0].revealed_observation_ids,
        ),
    )
    forged = dataclasses.replace(honest, steps=steps)
    assert "after the terminal call" in ordering_detail(
        cube.cell_variant("S0_P0"), forged
    )


# -- environment ------------------------------------------------------------


def test_obtaining_an_initial_fact_before_reading_records_is_refused():
    initial_cube = compile_cube(ConstructCard.from_dict(initial_state_card()))
    env = Environment(initial_cube.cell_variant("S0_P0"))
    with pytest.raises(EnvironmentError, match="records have not been read"):
        env.obtain("repair_quote_gbp")


def test_an_unknown_observation_source_is_refused():
    with pytest.raises(ValueError, match="unknown observation source"):
        Observation(
            observation_id="obs:rumour:x:01",
            source="rumour",
            key="x",
            fact={},
            availability="initial",
            evidence_relevant=False,
        )


# -- solvers ----------------------------------------------------------------


def test_presented_rules_require_the_policy_to_have_been_observed(cube):
    from boundarybench.solvers import _presented_rules

    env = Environment(cube.cell_variant("S0_P0"))
    with pytest.raises(SolverError, match="active policy"):
        _presented_rules(env)


def test_required_evidence_naming_no_observation_is_a_structural_error(cube):
    variant = dataclasses.replace(
        cube.cell_variant("S0_P0"),
        required_evidence_ids=("obs:user_answer:ghost:99",),
    )
    with pytest.raises(SolverError, match="does not name an observation"):
        _required_observations(variant)


@pytest.mark.parametrize(
    ("expected", "actual", "divergence"),
    [
        (("ACT", "R", frozenset()), ("STOP", "R", frozenset()), "disposition"),
        (("ACT", "R", frozenset()), ("ACT", "Q", frozenset()), "primary_reason"),
        (("ACT", "R", frozenset({"a"})), ("ACT", "R", frozenset()), "evidence"),
        (("ACT", "R", frozenset({"a"})), ("ACT", "R", frozenset({"a"})), None),
    ],
)
def test_first_divergence_reports_the_highest_order_difference(
    expected, actual, divergence
):
    assert _first_divergence(expected, actual) == divergence


def test_attempted_rules_returns_the_whole_arm_when_nothing_matches(cube):
    arm = cube.card.policy_axis["P0"]
    walked = attempted_rules(arm, {})
    assert walked == arm.rules


def test_check_solvers_reports_a_preserved_expectation_that_broke(cube, monkeypatch):
    """A negative solver that damages something it should not must be caught."""

    def sabotage(variant, cube_):
        trajectory = solver("reference").solve(variant, cube_)
        return dataclasses.replace(
            trajectory,
            terminal_decision=TerminalDecision(
                disposition="STOP",
                primary_reason_code="CONTRACTUAL_AUTHORITY",
                evidence_refs=(),
            ),
        )

    import boundarybench.solvers as solvers_module

    sabotaged = dataclasses.replace(solver("wrong_evidence"), solve=sabotage)
    monkeypatch.setattr(
        solvers_module,
        "SOLVERS",
        tuple(sabotaged if s.name == "wrong_evidence" else s for s in SOLVERS),
    )
    report = check_solvers(cube)
    outcome = report.outcome("wrong_evidence")
    assert not outcome.satisfied
    assert any("expected to pass" in failure for failure in outcome.failures)
    assert not report.ok


def test_every_solver_spec_carries_a_description():
    for spec in SOLVERS:
        assert spec.description.strip()
        assert spec.kind in {"reference", "negative"}


# -- compiler ---------------------------------------------------------------


def test_rejects_a_cube_without_a_preserve_state_edge():
    """Both state edges live: the state axis never holds a decision still."""
    card = mutate()
    card["policy_axis"]["P1"]["rules"] = [
        {
            "id": "R_P1_LOW_VALUE_STOP",
            "when": [{"fact": "repair_quote_gbp", "op": "lte", "value": 250}],
            "decision": "STOP",
            "primary_reason": "CONTRACTUAL_AUTHORITY",
        },
        {
            "id": "R_P1_HIGH_VALUE_ACT",
            "when": [{"fact": "repair_quote_gbp", "op": "gt", "value": 250}],
            "decision": "ACT",
            "primary_reason": "NORMAL_OPERATIONAL_POLICY",
        },
    ]
    card["disposition_table"]["S1_P1"] = "ACT"
    with pytest.raises(CompileError, match="preserve state edge"):
        compile_cube(ConstructCard.from_dict(card))


def test_rejects_an_initial_state_cube_whose_facts_still_need_eliciting():
    card = mutate(fact_location="initial_state")
    with pytest.raises(CompileError, match="initial_state"):
        compile_cube(ConstructCard.from_dict(card))


def test_a_rule_reconstructed_from_a_presented_policy_round_trips(cube):
    presented = cube.cell_variant("S0_P0").observations[0].fact["rules"][0]
    rule = Rule.from_dict(presented, "presented_policy")
    assert rule.id == "R_P0_ABOVE_LIMIT"
    assert rule.referenced_facts == ("repair_quote_gbp",)


# -- CLI --------------------------------------------------------------------


def test_check_solvers_prints_failures_and_exits_one(capsys, monkeypatch):
    import boundarybench.cli as cli_module

    def failing_report(cube):
        real = check_solvers(cube)
        broken = dataclasses.replace(
            real.outcomes[1], satisfied=False, failures=("deliberate sabotage",)
        )
        return dataclasses.replace(
            real, outcomes=(real.outcomes[0], broken, *real.outcomes[2:])
        )

    monkeypatch.setattr(cli_module, "check_solvers", failing_report)
    assert main(["check-solvers", str(EXAMPLE_CARD)]) == 1
    out = capsys.readouterr().out
    assert "deliberate sabotage" in out
    assert "causal CI: FAILED" in out


def test_check_solvers_json_also_exits_one_when_the_gate_is_red(capsys, monkeypatch):
    import boundarybench.cli as cli_module

    def failing_report(cube):
        real = check_solvers(cube)
        broken = dataclasses.replace(real.outcomes[1], satisfied=False)
        return dataclasses.replace(
            real, outcomes=(real.outcomes[0], broken, *real.outcomes[2:])
        )

    monkeypatch.setattr(cli_module, "check_solvers", failing_report)
    assert main(["check-solvers", str(EXAMPLE_CARD), "--json"]) == 1
