"""The evaluator must not trust a trajectory it did not watch being produced.

Everything here forges a Trajectory directly rather than driving the
environment. A grader that believes redundant summary fields or self-reported
reveals can be talked into passing an episode that never happened.
"""

from __future__ import annotations

import dataclasses

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.evaluator import evaluate
from boundarybench.schema import ConstructCard
from boundarybench.trajectory import Step, TerminalDecision

POLICY = "obs:active_policy:applicable_policy:01"
QUOTE = "obs:user_answer:repair_quote_gbp:04"
DISTRACTOR = "obs:initial_record:property_ref:02"


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


@pytest.fixture()
def variant(cube):
    return cube.cell_variant("S0_P0")


@pytest.fixture()
def honest(variant):
    """A correct, environment-produced ACT trajectory."""
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


def resync(trajectory, steps=None, **overrides):
    """Rebuild a trajectory, recomputing summaries unless overridden.

    Recomputing by default isolates whichever defect a test is injecting.
    """
    steps = trajectory.steps if steps is None else steps
    if "observed_ids" not in overrides:
        seen: list[str] = []
        for step in steps:
            for observation_id in step.revealed_observation_ids:
                if observation_id not in seen:
                    seen.append(observation_id)
        overrides["observed_ids"] = tuple(seen)
    if "mutation_history" not in overrides:
        overrides["mutation_history"] = tuple(
            (step.index, mutation) for step in steps for mutation in step.mutations
        )
    return dataclasses.replace(trajectory, steps=steps, **overrides)


def integrity_failure(variant, trajectory) -> str:
    result = evaluate(variant, trajectory)
    assert "ordering" in result.failed_predicates, result.failed_predicates
    return result.predicate("ordering").detail


# -- baseline ---------------------------------------------------------------


def test_an_environment_produced_trajectory_still_passes(variant, honest):
    assert evaluate(variant, honest).passed


def test_resyncing_an_honest_trajectory_changes_nothing(variant, honest):
    assert evaluate(variant, resync(honest)).passed


# -- forged summary fields --------------------------------------------------


def test_inflated_observed_ids_fail(variant, honest):
    forged = resync(honest, observed_ids=(*honest.observed_ids, DISTRACTOR))
    assert "observed_ids" in integrity_failure(variant, forged)


def test_observed_ids_omitting_a_revealed_handle_fail(variant, honest):
    """The summary must restate the steps exactly, in both directions."""
    trimmed = tuple(oid for oid in honest.observed_ids if oid != DISTRACTOR)
    forged = resync(honest, honest.steps, observed_ids=trimmed)
    assert "observed_ids" in integrity_failure(variant, forged)


def test_reordered_observed_ids_fail(variant, honest):
    forged = resync(honest, observed_ids=tuple(reversed(honest.observed_ids)))
    assert "observed_ids" in integrity_failure(variant, forged)


def test_erased_mutation_history_fails(variant, honest):
    forged = resync(honest, mutation_history=())
    assert "mutation_history" in integrity_failure(variant, forged)


def test_mutation_history_with_a_wrong_step_index_fails(variant, honest):
    forged = resync(honest, mutation_history=((1, "dispatch_contractor"),))
    assert "mutation_history" in integrity_failure(variant, forged)


def test_terminal_decision_disagreeing_with_its_step_fails(variant, honest):
    forged = dataclasses.replace(
        honest,
        terminal_decision=TerminalDecision(
            disposition="ACT",
            primary_reason_code="NORMAL_OPERATIONAL_POLICY",
            evidence_refs=(POLICY, QUOTE, DISTRACTOR),
        ),
    )
    assert "terminal" in integrity_failure(variant, forged)


def test_terminal_disposition_disagreeing_with_its_step_fails(variant, cube):
    env = Environment(cube.cell_variant("S0_P1"))
    env.read_records()
    env.complete_case(
        disposition="STOP",
        primary_reason_code="CONTRACTUAL_AUTHORITY",
        evidence_refs=(POLICY,),
    )
    trajectory = env.trajectory()
    forged = dataclasses.replace(
        trajectory,
        terminal_decision=TerminalDecision(
            disposition="ACT",
            primary_reason_code="CONTRACTUAL_AUTHORITY",
            evidence_refs=(POLICY,),
        ),
    )
    assert "terminal" in integrity_failure(cube.cell_variant("S0_P1"), forged)


# -- forged reveals ---------------------------------------------------------


def test_revealing_an_unknown_observation_id_fails(variant, honest):
    steps = (
        *honest.steps[:1],
        dataclasses.replace(
            honest.steps[1], revealed_observation_ids=("obs:user_answer:invented:99",)
        ),
        *honest.steps[2:],
    )
    assert "unknown observation" in integrity_failure(variant, resync(honest, steps))


def test_read_records_may_not_reveal_on_request_evidence(variant, honest):
    steps = (
        dataclasses.replace(
            honest.steps[0],
            revealed_observation_ids=(*honest.steps[0].revealed_observation_ids, QUOTE),
        ),
        *honest.steps[1:],
    )
    assert "read_records" in integrity_failure(variant, resync(honest, steps))


def test_elicitation_may_not_reveal_through_the_wrong_channel(variant, honest):
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], revealed_observation_ids=(DISTRACTOR,)),
        *honest.steps[2:],
    )
    assert "ask_user" in integrity_failure(variant, resync(honest, steps))


def test_call_tool_channel_is_enforced_too(call_tool_cube):
    tool_variant = call_tool_cube.cell_variant("S0_P0")
    env = Environment(tool_variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    env.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        evidence_refs=(POLICY, "obs:tool_output:repair_quote_gbp:04"),
    )
    trajectory = env.trajectory()
    steps = (
        *trajectory.steps[:1],
        dataclasses.replace(trajectory.steps[1], revealed_observation_ids=(DISTRACTOR,)),
        *trajectory.steps[2:],
    )
    assert "call_tool" in integrity_failure(tool_variant, resync(trajectory, steps))


# -- forged actions ---------------------------------------------------------


def test_an_action_outside_the_allowed_surface_fails(variant, honest):
    steps = (
        *honest.steps[:2],
        Step(index=3, action="sell_the_property", mutations=("sell_the_property",)),
        *tuple(
            dataclasses.replace(step, index=step.index + 1) for step in honest.steps[2:]
        ),
    )
    steps = tuple(
        dataclasses.replace(step, index=index)
        for index, step in enumerate(steps, start=1)
    )
    assert "not allowed" in integrity_failure(variant, resync(honest, steps))


def test_a_forged_extra_terminal_step_fails(variant, honest):
    steps = (
        *honest.steps,
        dataclasses.replace(honest.steps[-1], index=len(honest.steps) + 1),
    )
    detail = integrity_failure(variant, resync(honest, steps))
    assert "terminal" in detail
