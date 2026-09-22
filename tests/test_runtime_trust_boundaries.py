"""Adversarial checks for runtime trust boundaries.

Each case supplies a self-asserted fact that the run must independently derive:
a digest travelling with its object, an unconstrained counter, a released lock
handle, or a checksum over its own value. The run refuses any disagreement with
the evidence that actually determines the fact.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    AdapterUsage,
    ScriptedTestAdapter,
    TurnDeadline,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.evaluator import evaluate
from boundarybench.ledger import (
    OUTCOME_TIMEOUT,
    LedgerError,
    append_episode,
    read_ledger,
)
from boundarybench.runmanifest import (
    RunLimits,
    RunLockError,
    build_run_manifest,
    canonical_json,
    open_run_session,
)
from boundarybench.runner import (
    MESSAGES_PER_TURN,
    RunProvenanceError,
    build_episode_record,
    compiled_variants,
    execute_run,
    run_episode,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from boundarybench.trajectory import Step, TerminalDecision, Trajectory
from tests.conftest import SUITE_MANIFEST
from tests.test_runtime_execution_integrity import _completed_run, _rewrite_first_row

#: A canonical version-4 UUID, so a test can state the execution a record names
#: rather than observe whatever the platform generator produced.
_EXECUTION_ID = "1f0d7a3c-aaaa-4a1b-9c2d-000000000001"


def _at(second: int) -> datetime:
    """A stated instant, so a recorded timestamp is an assertion not an artefact."""
    return datetime(2026, 8, 8, 12, 0, second, 500000, tzinfo=UTC)


def _manifest(trials: int = 1, **overrides: object):
    fields: dict[str, object] = {
        "suite": validate_suite(SUITE_MANIFEST),
        "scaffold": load_scaffold(STANDARD_SCAFFOLD),
        "provider": "test-double",
        "model": "scripted-test-double",
        "implementation": "scripted_test_double",
        "adapter_version": ADAPTER_CONTRACT_VERSION,
        "adapter_settings": {},
        "trials": trials,
        "limits": RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    }
    fields.update(overrides)
    return build_run_manifest(**fields)  # type: ignore[arg-type]


def _fake_adapter(**overrides: object) -> ScriptedTestAdapter:
    fields: dict[str, object] = {
        "script": policy_following_script,
        "identity": identity_for_test_double(),
        "settings": {},
    }
    fields.update(overrides)
    return ScriptedTestAdapter(**fields)  # type: ignore[arg-type]


# -- 5. the run manifest states only identity-bound facts ---------------------


def test_the_wall_clock_is_execution_evidence_and_never_configuration_identity(
    tmp_path: Path,
) -> None:
    """Where the timestamp lives, stated as an exact partition of the file.

    The file is exactly two things. The configuration is everything that
    changes execution or grading, and the creation instant is provably not in
    it: the same plan built at two different instants is the same
    ``configuration_id``. The execution block is the instant, the execution id
    and the configuration they belong to, hashed as one unit, and the loose
    top-level ``created_at`` and its private checksum are both gone.
    """
    early = _manifest(now=lambda: _at(1), execution_id_factory=lambda: _EXECUTION_ID)
    late = _manifest(now=lambda: _at(58))

    with open_run_session(tmp_path / "run", early) as session:
        payload = json.loads(session.paths.manifest_path.read_text(encoding="utf-8"))

    # The wall clock is not configuration identity: two instants, one id.
    assert early.created_at_utc != late.created_at_utc
    assert early.configuration_id == late.configuration_id
    assert "created_at_utc" not in early.configuration_payload()
    assert "execution_id" not in early.configuration_payload()

    # And the stored file is that partition, exactly.
    assert set(payload) == set(early.configuration_payload()) | {
        "configuration_id",
        "execution",
    }
    assert "created_at" not in payload
    assert "manifest_integrity_digest" not in payload
    assert not hasattr(early, "manifest_integrity_digest")

    # The execution block is the timestamp's only home, and it is bound to the
    # configuration it ran and to the execution that ran it.
    assert payload["execution"] == {
        "configuration_id": early.configuration_id,
        "execution_id": _EXECUTION_ID,
        "created_at_utc": "2026-08-08T12:00:01.500000Z",
        "execution_digest": early.execution_digest,
    }
    assert early.created_at_utc == "2026-08-08T12:00:01.500000Z"


def test_build_run_manifest_refuses_a_loose_wall_clock_argument() -> None:
    """The clock is injected as a *clock*, never as a value to be recorded.

    ``now`` is a callable this build calls and then renders itself, through the
    one encoder that refuses a naive datetime. A ``created_at`` string argument
    would be a timestamp the caller chose and the build merely stored, which is
    not evidence of anything.
    """
    with pytest.raises(TypeError):
        _manifest(created_at="2099-01-01T00:00:00+00:00")


# -- 1. a released lock handle is spent ---------------------------------------


def test_a_released_run_session_cannot_be_used_to_append(tmp_path: Path) -> None:
    """The lock is the permission, so the handle must expire with it.

    ``assert_held`` only tested whether the descriptor field was negative, and
    nothing ever made it negative: closing the descriptor on the way out of the
    context manager left the ``RunSession`` looking exactly as valid as it had
    been inside. A caller holding the stale handle could therefore run every
    episode and append every row with no lock held at all, which is the precise
    condition the lock exists to prevent.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)

    with open_run_session(root, manifest) as session:
        pass

    with pytest.raises(RunLockError) as excinfo:
        session.assert_held()
    assert "released" in str(excinfo.value)

    with pytest.raises(RunLockError):
        execute_run(
            suite=suite,
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=_fake_adapter(),
            session=session,
        )
    assert not session.paths.ledger_path.exists()


# -- 2. provenance is recomputed, never self-asserted -------------------------


def test_an_altered_scaffold_prompt_cannot_execute_under_its_old_digest(
    tmp_path: Path,
) -> None:
    """A digest that travels with the object it describes proves nothing.

    ``Scaffold`` is frozen, but ``dataclasses.replace`` rebuilds it with a new
    ``system_prompt`` and the *old* ``content_digest``. Comparing the stored
    field against the manifest then agrees, and twelve episodes execute under a
    prompt the manifest never pinned.
    """
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    unpinned = dataclasses.replace(scaffold, system_prompt="UNPINNED PROMPT")
    assert unpinned.content_digest == manifest.scaffold_content_digest

    with open_run_session(tmp_path / "run", manifest) as session:
        with pytest.raises(RunProvenanceError) as excinfo:
            execute_run(
                suite=suite,
                scaffold=unpinned,
                adapter=_fake_adapter(),
                session=session,
            )
        assert "scaffold" in str(excinfo.value)
        assert not session.paths.ledger_path.exists()


def test_an_altered_expected_disposition_cannot_grade_under_the_old_suite_digest(
    tmp_path: Path,
) -> None:
    """The same attack one level down, against the grading contract itself.

    A variant rebuilt with a different ``expected_disposition`` and its original
    ``content_digest`` executed and *graded* under the run's suite identity: the
    row's digest matched, so nothing objected to a verdict computed against a
    grading contract the suite never contained.
    """
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    cube = suite.cubes[0]
    original = cube.cube.variants[0]
    flipped = "STOP" if original.expected_disposition == "ACT" else "ACT"
    tampered = dataclasses.replace(original, expected_disposition=flipped)
    assert tampered.content_digest == original.content_digest

    rebuilt = dataclasses.replace(
        suite,
        cubes=(
            dataclasses.replace(
                cube,
                cube=dataclasses.replace(
                    cube.cube, variants=(tampered, *cube.cube.variants[1:])
                ),
            ),
            *suite.cubes[1:],
        ),
    )
    assert rebuilt.content_digest == manifest.suite_content_digest

    with open_run_session(tmp_path / "run", manifest) as session:
        with pytest.raises(RunProvenanceError) as excinfo:
            execute_run(
                suite=rebuilt,
                scaffold=load_scaffold(STANDARD_SCAFFOLD),
                adapter=_fake_adapter(),
                session=session,
            )
        assert original.variant_id in str(excinfo.value)
        assert not session.paths.ledger_path.exists()


# -- 6. an answer that arrived too late is a timeout --------------------------


class _OverrunAdapter:
    """An adapter whose call crosses the deadline before it answers.

    ``usage_fails`` is the second half of the attack: a provider whose usage
    endpoint raises *after* an overrun. The provider fault is real, but it is
    downstream of a call that had already spent time the episode did not have.
    """

    def __init__(
        self, clock: list[float], *, overrun: float, usage_fails: bool = False
    ) -> None:
        self._clock = clock
        self._overrun = overrun
        self._usage_fails = usage_fails

    @property
    def identity(self) -> Any:
        return identity_for_test_double()

    @property
    def settings(self) -> dict[str, Any]:
        return {}

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        self._clock[0] += self._overrun
        return AdapterCall("read_records", {})

    def last_usage(self) -> AdapterUsage:
        if self._usage_fails:
            raise RuntimeError("usage failed after deadline")
        return AdapterUsage()


def _timed_episode(adapter: Any, *, timeout: float, clock: list[float]) -> Any:
    suite = validate_suite(SUITE_MANIFEST)
    variant = suite.cubes[0].cube.variants[0]
    return run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=adapter,
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=timeout),
        clock=lambda: clock[0],
    )


def test_an_overrun_call_is_a_timeout_even_when_usage_collection_then_fails() -> None:
    """Whether the answer arrived in time is decided before anything else.

    An adapter blocks past the budget and then raises from its usage endpoint.
    The provider fault is downstream of a call that already spent time the
    episode did not have, so timeout classification takes precedence.
    """
    clock = [0.0]
    result = _timed_episode(
        _OverrunAdapter(clock, overrun=5.0, usage_fails=True), timeout=1.0, clock=clock
    )

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"


def test_a_call_that_returns_exactly_on_the_deadline_is_a_timeout() -> None:
    """``cancelled()`` reports cancellation at ``remaining() <= 0``.

    The post-return boundary is inclusive: a call landing exactly on the deadline
    is a timeout, not ``limit_exhausted``/``max_turns``.
    """
    clock = [0.0]
    result = _timed_episode(_OverrunAdapter(clock, overrun=1.0), timeout=1.0, clock=clock)

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"
    # The round trip happened and is counted; only its answer is discarded.
    assert result.turns_used == 1
    assert len(result.trajectory.steps) == 0


# -- 7. a parser that runs out of stack fails by name -------------------------


def test_a_deeply_nested_scaffold_fails_without_a_traceback(tmp_path: Path) -> None:
    """``json.loads`` raises ``RecursionError``, which is not a ``ValueError``.

    Every other malformed-artefact path is a named domain failure, so the CLI
    reports it as one line on stderr. Nesting deeply enough to exhaust the
    parser's stack escaped all of them and reached the operator as a traceback.
    """
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "boundarybench.cli",
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            "fake-scripted",
            "--output-dir",
            str(tmp_path / "run"),
            "--scaffold",
            str(deep),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert "nest" in completed.stderr.lower()


# -- 3. a trajectory is only evidence if the environment would produce it -----


def _trajectory_from_raw(raw: Mapping[str, Any]) -> Trajectory:
    """Rebuild a typed ``Trajectory`` from a raw ledger trajectory payload.

    Used only so a test can recompute the real evaluator's verdict against a
    tampered trajectory, exactly as an attacker forging a self-consistent row
    would have to.
    """
    terminal_raw = raw["terminal_decision"]
    return Trajectory(
        variant_id=raw["variant_id"],
        steps=tuple(
            Step(
                index=step["index"],
                action=step["action"],
                arguments=step["arguments"],
                revealed_observation_ids=tuple(step["revealed_observation_ids"]),
                mutations=tuple(step["mutations"]),
            )
            for step in raw["steps"]
        ),
        terminal_decision=(
            None
            if terminal_raw is None
            else TerminalDecision(
                disposition=terminal_raw["disposition"],
                primary_reason_code=terminal_raw["primary_reason_code"],
                secondary_reason_codes=tuple(terminal_raw["secondary_reason_codes"]),
                evidence_refs=tuple(terminal_raw["evidence_refs"]),
            )
        ),
        observed_ids=tuple(raw["observed_ids"]),
        mutation_history=tuple((item[0], item[1]) for item in raw["mutation_history"]),
    )


def test_impossible_step_index_is_refused_even_with_recomputed_verdict(
    tmp_path: Path,
) -> None:
    """Reconstructing a trajectory's shape is not the same as replaying it.

    ``_trajectory`` rebuilds the typed object and demands it re-encode to
    exactly what was stored, and ``_bind_evaluation`` regrades it against the
    reconstruction — but neither checks that a *real* environment could ever
    have produced these steps. The environment numbers steps by when they
    actually happen; a stored step whose ``index`` was rewritten to a value the
    environment never assigned reconstructs and regrades exactly like a genuine
    one, evaluator verdict included, because the evaluator does not know where
    indices come from either. Only replaying the recorded actions through a
    fresh environment and demanding the result match exactly catches it.
    """
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)
    ledger = root / "episodes.jsonl"

    def mutate(row: dict[str, Any]) -> None:
        row["trajectory"]["steps"][0]["index"] = 99
        variant = variants[row["variant_id"]]
        trajectory = _trajectory_from_raw(row["trajectory"])
        recomputed = evaluate(variant, trajectory)
        row["evaluation"] = json.loads(canonical_json(recomputed.as_dict()))
        # The recomputed verdict really is failing: the attack is not merely
        # "make the evaluator pass", it is "make replay unnecessary".
        assert row["evaluation"]["passed"] is False

    _rewrite_first_row(ledger, mutate)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "replay" in str(excinfo.value)


# -- 4. a row's counters must follow from the trajectory that produced them ---


def test_a_row_with_forged_counters_is_refused(tmp_path: Path) -> None:
    """A type-correct counter that no trace binds to the trajectory is a free lie.

    The adversarial row rewrites a genuine trajectory's counters to values that
    are type-correct but impossible for its steps. Counter derivation must refuse
    the row.
    """
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)
    ledger = root / "episodes.jsonl"

    def mutate(row: dict[str, Any]) -> None:
        assert len(row["trajectory"]["steps"]) == 4
        row["turns_used"] = 999
        row["messages_used"] = 0

    _rewrite_first_row(ledger, mutate)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "turns_used" in str(excinfo.value) or "messages_used" in str(excinfo.value)


def test_a_failed_episode_may_spend_exactly_one_turn_more_than_its_recorded_steps(
    tmp_path: Path,
) -> None:
    """The narrow allowance: one unfinished turn, and never more than one.

    An adapter call that overruns the deadline spends a turn and leaves no step
    behind, because the environment never saw a validated action from it. That
    is the one situation in which ``turns_used`` may exceed the trajectory's
    step count, and only by exactly one — the turn that ended the episode.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    variant = suite.cubes[0].cube.variants[0]
    entry = manifest.episode_plan[0]
    clock = [0.0]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_OverrunAdapter(clock, overrun=1.0),
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=1.0),
        clock=lambda: clock[0],
    )
    assert result.turns_used == 1
    assert len(result.trajectory.steps) == 0

    with open_run_session(root, manifest) as session:
        append_episode(
            session.paths.ledger_path,
            build_episode_record(manifest, entry, variant, result),
        )

    variants = compiled_variants(suite)
    (record,) = read_ledger(root / "episodes.jsonl", manifest, variants)
    assert record.turns_used == 1
    # Immutable evidence: an empty JSON array reads back as an empty tuple.
    assert record.trajectory["steps"] == ()


def test_a_failed_row_whose_turns_exceed_its_steps_by_more_than_one_is_refused(
    tmp_path: Path,
) -> None:
    """More than one turn beyond the step count is not the late-answer case.

    Exactly one extra turn is the unfinished-call allowance. Two or more is not
    a shape a real episode can produce, no matter which failure it recorded.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    variant = suite.cubes[0].cube.variants[0]
    entry = manifest.episode_plan[0]
    clock = [0.0]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_OverrunAdapter(clock, overrun=1.0),
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=1.0),
        clock=lambda: clock[0],
    )
    record = build_episode_record(manifest, entry, variant, result)
    forged_turns = record.turns_used + 1
    forged = dataclasses.replace(
        record, turns_used=forged_turns, messages_used=forged_turns * MESSAGES_PER_TURN
    )

    with open_run_session(root, manifest) as session:
        append_episode(session.paths.ledger_path, forged)

    variants = compiled_variants(suite)
    with pytest.raises(LedgerError) as excinfo:
        read_ledger(root / "episodes.jsonl", manifest, variants)

    assert "turns_used" in str(excinfo.value)


# -- 8. every record is UTF-8, whatever the machine's locale says -------------


#: Run in a child process under an ASCII locale. Every file this build writes or
#: reads is exercised: the run manifest, the ledger and the scaffold. Kept as a
#: script rather than a fixture because the encoding a bare ``open()`` picks is
#: decided by the interpreter's environment at start-up, which cannot be changed
#: from inside the test process that is already running.
_ASCII_LOCALE_PROBE = """
import dataclasses
import locale
import sys
from pathlib import Path

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    ScriptedTestAdapter,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.ledger import append_episode, read_ledger
from boundarybench.runmanifest import (
    RunLimits,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import build_episode_record, compiled_variants, run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.scaffolds import scaffold_payload, write_scaffold

print("encoding", locale.getencoding())

TEXT = "R\\u00e9pondez en fran\\u00e7ais \\u2615"
root = Path(sys.argv[1])
suite = validate_suite(Path(sys.argv[2]))
scaffold = load_scaffold(STANDARD_SCAFFOLD)

manifest = build_run_manifest(
    suite=suite,
    scaffold=scaffold,
    provider="test-double",
    model="scripted-test-double",
    implementation="scripted_test_double",
    adapter_version=ADAPTER_CONTRACT_VERSION,
    adapter_settings={"system_prefix": TEXT},
    trials=1,
    limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
)

with open_run_session(root / "run", manifest) as session:
    stored = load_run_manifest(session.paths.manifest_path)
    assert stored.run_id == manifest.run_id, "manifest did not survive the round trip"
    assert stored.adapter_settings["system_prefix"] == TEXT, stored.adapter_settings

    entry = manifest.episode_plan[0]
    variants = compiled_variants(suite)
    times = iter([0.0])
    result = run_episode(
        variant=variants[entry.variant_id],
        scaffold=scaffold,
        adapter=ScriptedTestAdapter(
            script=policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=manifest.limits,
        clock=lambda: next(times, 100.0),
    )
    genuine = build_episode_record(manifest, entry, variants[entry.variant_id], result)
    assert genuine.failure_event is not None, "the episode recorded no cause"
    # Both halves, because the row's detail is now the one its cause states: a
    # non-ASCII failure detail has to survive the round trip in both places.
    record = dataclasses.replace(
        genuine,
        error_detail=TEXT,
        failure_event={**genuine.failure_event, "detail": TEXT},
    )
    append_episode(session.paths.ledger_path, record)
    assert TEXT.encode("utf-8") in session.paths.ledger_path.read_bytes()
    (row,) = read_ledger(session.paths.ledger_path, manifest, variants)
    assert row.error_detail == TEXT, row.error_detail

payload = scaffold_payload()
payload["system_prompt"] = TEXT + " Lisez le dossier, puis d\\u00e9cidez."
assert load_scaffold(
    write_scaffold(root / "scaffold.json", payload)
).system_prompt == payload["system_prompt"]

print("ok")
"""


def test_every_record_this_build_writes_is_utf8_whatever_the_locale_says(
    tmp_path: Path,
) -> None:
    """The encoding of a run's records is a property of the build, not the host.

    Left to the platform default, a manifest written on a UTF-8 developer
    machine is unreadable on an operator's POSIX-locale box, and a ledger row
    carrying an accented failure detail cannot be appended there at all —
    ``UnicodeEncodeError`` escaping mid-append, after the episode has run. The
    encoding is therefore stated at every read and every write, which only a
    process started under a non-UTF-8 locale can observe.
    """
    script = tmp_path / "ascii_locale_probe.py"
    script.write_text(_ASCII_LOCALE_PROBE, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), str(SUITE_MANIFEST)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path.cwd()), str(Path.cwd() / "src")]),
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONUTF8": "0",
        },
    )

    encoding = completed.stdout.partition("\n")[0].removeprefix("encoding ").strip()
    if not encoding or "utf" in encoding.lower():
        pytest.skip(f"this interpreter defaults to {encoding!r}, so it observes nothing")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.rstrip().endswith("ok")
