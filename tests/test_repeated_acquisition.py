"""A repeated acquisition is one turn, one step, and no second reveal.

Reveal-at-most-once was implemented as record-at-most-once: a model that asked
for a fact it already held got the same handle back and *no step*, while the
turn it spent was counted like any other. The episode's own summary then
disagreed with its own evidence — twelve completed provider turns against nine
recorded steps — and the ledger refused the row, correctly: a turn with no step
is a model action that left no audit trail.

These tests fix the other half of the contract. The handle stays single and the
observation stays revealed exactly once; the *action* is recorded every time it
is dispatched, with nothing revealed. Turns and steps then agree, and neither
the evaluator nor the replay will accept a repeat that claims a second reveal or
a first acquisition that claims none.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    ScriptedTestAdapter,
    TurnRequest,
    identity_for_test_double,
)
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment, query_affordances, replay_trajectory
from boundarybench.evaluator import evaluate
from boundarybench.ledger import (
    OUTCOME_LIMIT_EXHAUSTED,
    LedgerError,
    append_episode,
    read_ledger,
)
from boundarybench.runmanifest import (
    RUNNER_CONTRACT_VERSION,
    RunLimits,
    build_run_manifest,
)
from boundarybench.runner import build_episode_record, compiled_variants, run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from boundarybench.trajectory import Step, Trajectory
from tests.conftest import SUITE_MANIFEST, minimal_card

#: The one query the scripted episode asks for, over and over.
QUERY_KEY = "repair_quote_gbp"
ASK = "ask_user"

MAX_TURNS = 12
LIMITS = RunLimits(max_turns=MAX_TURNS, max_messages=64, episode_timeout_seconds=30.0)

POLICY = "obs:active_policy:applicable_policy:01"
QUOTE = "obs:user_answer:repair_quote_gbp:04"


# -- the live loop, end to end ----------------------------------------------


def _manifest(**overrides: Any) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        provider="test-double",
        model="scripted-test-double",
        implementation="scripted_test_double",
        adapter_version=ADAPTER_CONTRACT_VERSION,
        adapter_settings={},
        trials=1,
        limits=LIMITS,
        **overrides,
    )


def _asking_episode(manifest: Any, variants: dict[str, Variant]) -> tuple[Any, Variant]:
    """The first planned episode whose variant offers the repeated query."""
    for entry in manifest.episode_plan:
        variant = variants[entry.variant_id]
        if QUERY_KEY in dict(query_affordances(variant)).get(ASK, ()):
            return entry, variant
    raise AssertionError("the spike suite offers no ask_user query to repeat")


def _repeating_script(request: TurnRequest) -> AdapterCall:
    """Read the record once, then ask the same question until the turns run out.

    Deliberately never terminal: the episode ends on the turn limit, which is
    the one failure whose spec allows no daylight at all between turns and
    steps.
    """
    if not request.transcript:
        return AdapterCall("read_records", {})
    return AdapterCall(ASK, {"fact": QUERY_KEY})


def _episode(manifest: Any, variants: dict[str, Variant]) -> tuple[Any, Any, Variant]:
    entry, variant = _asking_episode(manifest, variants)
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=_repeating_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=LIMITS,
    )
    return entry, result, variant


def test_a_repeated_query_episode_spends_one_turn_per_recorded_step(
    tmp_path: Path,
) -> None:
    """The reproduction, scripted: twelve turns that ask one question."""
    manifest = _manifest()
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    entry, result, variant = _episode(manifest, variants)

    assert result.outcome == OUTCOME_LIMIT_EXHAUSTED
    assert result.error_class == "max_turns"
    assert result.turns_used == MAX_TURNS
    assert result.messages_used == 2 * MAX_TURNS
    steps = result.trajectory.steps
    assert len(steps) == MAX_TURNS

    # One read, then eleven asks: the first reveals, the rest reveal nothing.
    assert steps[0].action == "read_records"
    asks = steps[1:]
    assert [step.action for step in asks] == [ASK] * (MAX_TURNS - 1)
    assert all(dict(step.arguments) == {"fact": QUERY_KEY} for step in asks)
    assert len(asks[0].revealed_observation_ids) == 1
    assert all(step.revealed_observation_ids == () for step in asks[1:])
    assert all(step.mutations == () for step in asks)

    handle = asks[0].revealed_observation_ids[0]
    assert result.trajectory.observed_ids.count(handle) == 1

    # And the row this episode writes reads back through the production reader.
    path = tmp_path / "episodes.jsonl"
    append_episode(path, build_episode_record(manifest, entry, variant, result))
    rows = read_ledger(path, manifest, variants)
    assert len(rows) == 1
    assert rows[0].turns_used == len(rows[0].trajectory["steps"]) == MAX_TURNS
    assert rows[0].messages_used == 2 * MAX_TURNS


def test_a_row_with_a_repeated_step_edited_out_is_still_refused(
    tmp_path: Path,
) -> None:
    """The counters are what catch it, and they are not relaxed.

    Dropping a zero-reveal repeat leaves a trajectory that is internally
    consistent — it revealed nothing, mutated nothing and is summarised by
    nothing — so the only thing standing between the ledger and a lost model
    action is the rule that a turn-limited episode records a step per turn.
    """
    manifest = _manifest()
    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    entry, result, variant = _episode(manifest, variants)
    record = build_episode_record(manifest, entry, variant, result)

    trajectory = result.trajectory.as_dict()
    dropped = trajectory["steps"].pop()
    assert dropped["action"] == ASK
    assert dropped["revealed_observation_ids"] == []

    path = tmp_path / "episodes.jsonl"
    append_episode(path, dataclasses.replace(record, trajectory=trajectory))

    with pytest.raises(LedgerError) as refused:
        read_ledger(path, manifest, variants)

    assert "recorded step" in str(refused.value)


# -- grading and replay ------------------------------------------------------


@pytest.fixture()
def variant() -> Variant:
    return compile_cube(
        ConstructCard.from_dict(copy.deepcopy(minimal_card()))
    ).cell_variant("S0_P0")


def _repeated_trajectory(variant: Variant) -> Trajectory:
    """A genuine winning episode that happens to ask the same question twice."""
    environment = Environment(variant)
    environment.read_records()
    environment.obtain(QUERY_KEY)
    environment.obtain(QUERY_KEY)
    environment.act("dispatch_contractor")
    environment.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        evidence_refs=(POLICY, QUOTE),
    )
    return environment.trajectory()


#: Where the repeat sits in :func:`_repeated_trajectory`, as a position.
REPEAT = 2


def _asserted_shape(trajectory: Trajectory) -> Trajectory:
    """Every test below is about the repeat, so every test proves it is there."""
    assert [step.action for step in trajectory.steps] == [
        "read_records",
        ASK,
        ASK,
        "dispatch_contractor",
        "complete_case",
    ]
    assert trajectory.steps[REPEAT - 1].revealed_observation_ids == (QUOTE,)
    assert trajectory.steps[REPEAT].revealed_observation_ids == ()
    assert trajectory.observed_ids.count(QUOTE) == 1
    return trajectory


def _with_step(trajectory: Trajectory, index: int, **changes: Any) -> Trajectory:
    steps = list(trajectory.steps)
    steps[index] = dataclasses.replace(steps[index], **changes)
    return dataclasses.replace(trajectory, steps=tuple(steps))


def test_a_repeated_zero_reveal_acquisition_grades_as_the_episode_it_was(
    variant: Variant,
) -> None:
    trajectory = _asserted_shape(_repeated_trajectory(variant))

    evaluation = evaluate(variant, trajectory)
    assert evaluation.passed, evaluation.failed_predicates


def test_a_repeat_that_claims_a_second_reveal_is_refused(variant: Variant) -> None:
    forged = _with_step(
        _asserted_shape(_repeated_trajectory(variant)),
        REPEAT,
        revealed_observation_ids=(QUOTE,),
    )

    evaluation = evaluate(variant, forged)
    assert not evaluation.passed
    assert evaluation.predicate("ordering").detail.endswith("which was already revealed")


def test_a_first_acquisition_that_reveals_nothing_is_refused(variant: Variant) -> None:
    """A zero-reveal step is legal *because* the handle was already held."""
    environment = Environment(variant)
    environment.read_records()
    trajectory = environment.trajectory()
    fabricated = dataclasses.replace(
        trajectory,
        steps=(
            *trajectory.steps,
            Step(index=2, action=ASK, arguments={"fact": QUERY_KEY}),
        ),
    )

    evaluation = evaluate(variant, fabricated)
    assert not evaluation.passed
    assert "has not been revealed" in evaluation.predicate("ordering").detail


def test_replay_reproduces_a_repeat_and_disagrees_with_a_forged_one(
    variant: Variant,
) -> None:
    trajectory = _asserted_shape(_repeated_trajectory(variant))

    assert replay_trajectory(variant, trajectory).as_dict() == trajectory.as_dict()

    forged = _with_step(trajectory, REPEAT, revealed_observation_ids=(QUOTE,))
    assert replay_trajectory(variant, forged).as_dict() != forged.as_dict()


# -- run identity ------------------------------------------------------------


def test_the_runner_contract_version_moved_with_the_recording_semantics() -> None:
    assert RUNNER_CONTRACT_VERSION == "0.9.0"


def test_a_run_configured_before_the_repeat_was_recorded_cannot_be_resumed(
    tmp_path: Path,
) -> None:
    """Synthetic on purpose, and local on purpose.

    Both manifests are built here from this build's own settings; the older one
    is rehashed so that it is refused for its contract version rather than for a
    configuration id that never followed from its own payload.
    """
    from boundarybench.runmanifest import (
        RunManifestMismatchError,
        configuration_digest,
        execution_digest_of,
        open_run_session,
    )

    current = _manifest()
    older = dataclasses.replace(
        current, runner_contract_version="0.8.0", configuration_id=""
    )
    identity = configuration_digest(older.configuration_payload())
    older = dataclasses.replace(
        older,
        configuration_id=identity,
        execution_digest=execution_digest_of(
            configuration_id=identity,
            execution_id=current.execution_id,
            created_at_utc=current.created_at_utc,
        ),
    )
    assert older.configuration_id != current.configuration_id

    root = tmp_path / "synthetic-unrecorded-repeat"
    with open_run_session(root, older) as session:
        assert session.resumed is False

    with (
        pytest.raises(RunManifestMismatchError) as refused,
        open_run_session(root, current),
    ):
        pass

    assert older.configuration_id in str(refused.value)
    assert current.configuration_id in str(refused.value)
