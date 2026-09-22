"""A whole run: the ledger it writes, and what resuming it does and does not do."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    ScriptedTestAdapter,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.ledger import OUTCOME_SUCCESS, read_ledger
from boundarybench.runmanifest import RunLimits, build_run_manifest, open_run_session
from boundarybench.runner import compiled_variants, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST


def _pieces(trials: int = 2) -> tuple[Any, Any, Any, Any]:
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider="test-double",
        model="scripted-test-double",
        implementation="scripted_test_double",
        adapter_version=ADAPTER_CONTRACT_VERSION,
        adapter_settings={},
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )
    adapter = ScriptedTestAdapter(
        script=policy_following_script, identity=identity_for_test_double(), settings={}
    )
    return suite, scaffold, manifest, adapter


def _execute(root: Path, trials: int = 2) -> Any:
    suite, scaffold, manifest, adapter = _pieces(trials)
    with open_run_session(root, manifest) as session:
        return execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=adapter,
            session=session,
        )


def test_a_complete_fake_run_executes_every_planned_episode(tmp_path: Path) -> None:
    report = _execute(tmp_path / "run")

    suite = validate_suite(SUITE_MANIFEST)
    planned = sum(len(cube.cube.variants) for cube in suite.cubes) * 2
    assert report.planned_count == planned == 24
    assert report.completed_count == planned
    assert report.executed_count == planned
    assert report.resumed_count == 0
    assert report.succeeded_count == planned
    assert report.failed_count == 0
    assert report.outcome_counts == {OUTCOME_SUCCESS: planned}

    lines = report.paths.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == planned
    records = read_ledger(
        report.paths.ledger_path, report.manifest, compiled_variants(suite)
    )
    assert [record.episode_id for record in records] == list(
        report.manifest.planned_episode_ids
    )
    # Every row is bound to the artefacts it was produced against.
    for record in records:
        assert record.suite_content_digest == suite.content_digest
        assert record.evaluation is not None
        assert record.evaluation["passed"] is True
        assert record.trajectory["terminal_decision"] is not None


def test_rerunning_a_complete_run_changes_no_bytes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    first = _execute(root)
    before = first.paths.ledger_path.read_bytes()
    manifest_before = first.paths.manifest_path.read_bytes()

    second = _execute(root)

    assert second.paths.ledger_path.read_bytes() == before
    assert second.paths.manifest_path.read_bytes() == manifest_before
    assert second.executed_count == 0
    assert second.resumed_count == 24
    assert second.completed_count == 24
    assert second.configuration_id == first.configuration_id
    # The stored execution, not a second one: nothing was re-executed, so there
    # is nothing here for a new execution id to name.
    assert second.execution_id == first.execution_id


#: The three fields an episode measures about itself. They are the only fields a
#: re-executed episode is *expected* to differ in, which is what makes comparing
#: everything else a real check rather than a vacuous one.
_EPISODE_TIMING_FIELDS = ("started_at_utc", "completed_at_utc", "elapsed_seconds")


def test_an_interrupted_run_appends_only_the_missing_episodes(
    tmp_path: Path,
) -> None:
    """The prefix already on disk is evidence: it is never re-executed.

    The prefix is compared as *bytes*, because that is the claim: those five
    rows were not touched, re-graded or rewritten. The nineteen appended rows
    cannot be compared as bytes against the earlier complete run, and requiring
    that they were was a real defect in this test rather than a strictness to
    preserve. Each episode now records when it ran, and the second execution of
    an episode genuinely ran at a different instant and took a different number
    of microseconds; a byte comparison would only pass if those timestamps were
    fabricated from the plan instead of measured.

    So the timing fields are excluded by name and *everything else* is required
    to be identical — and the timestamps are then checked on their own terms:
    inside the episode, ordered, and under this directory's one execution.
    """
    root = tmp_path / "run"
    complete = _execute(root)
    full_lines = complete.paths.ledger_path.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    prefix = "".join(full_lines[:5])
    complete.paths.ledger_path.write_bytes(prefix.encode("utf-8"))

    resumed = _execute(root)

    assert resumed.resumed_count == 5
    assert resumed.executed_count == 19
    assert resumed.completed_count == 24
    assert resumed.execution_id == complete.execution_id

    raw = resumed.paths.ledger_path.read_bytes()
    assert raw.startswith(prefix.encode("utf-8"))
    lines = raw.decode("utf-8").splitlines(keepends=True)
    assert len(lines) == 24
    assert lines[:5] == full_lines[:5]

    for before_line, after_line in zip(full_lines[5:], lines[5:], strict=True):
        before, after = json.loads(before_line), json.loads(after_line)
        assert after["execution_id"] == complete.execution_id
        assert after["started_at_utc"] <= after["completed_at_utc"]
        assert after["elapsed_seconds"] >= 0
        for field in _EPISODE_TIMING_FIELDS:
            del before[field]
            del after[field]
        assert after == before

    # And the whole file still reads back as this run's evidence.
    records = read_ledger(
        resumed.paths.ledger_path,
        resumed.manifest,
        compiled_variants(validate_suite(SUITE_MANIFEST)),
    )
    assert [record.episode_id for record in records] == list(
        resumed.manifest.planned_episode_ids
    )


def test_the_run_s_clock_is_the_clock_every_episode_is_timed_against(
    tmp_path: Path,
) -> None:
    """A run-level seam that stops at the run level times nothing.

    The clock is injected so a timeout is reproducible in CI. If ``execute_run``
    keeps it for itself and lets each episode fall back to the wall clock, every
    timeout test that goes through a whole run silently measures real elapsed
    time instead — and passes, because a fake episode never takes 30 seconds.
    """
    suite, scaffold, manifest, adapter = _pieces(trials=1)

    class RunawayClock:
        """Every reading is a hundred seconds after the last one."""

        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            value = self.now
            self.now += 100.0
            return value

    with open_run_session(tmp_path / "run", manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=adapter,
            session=session,
            clock=RunawayClock(),
        )

    assert report.executed_count == report.planned_count == 12
    assert report.succeeded_count == 0
    assert report.outcome_counts == {"timeout": 12}
    assert report.error_class_counts == {"episode_timeout": 12}
    assert {record.turns_used for record in report.records} == {0}


def test_report_states_its_scope_and_reports_no_model_score(tmp_path: Path) -> None:
    payload = _execute(tmp_path / "run", trials=1).as_dict()

    assert payload["status"] == "SYNTHETIC_INFRASTRUCTURE_SMOKE"
    assert "not a model benchmark result" in payload["scope"]
    assert "no ranking" in payload["scope"]
    assert set(payload["counts"]) == {
        "planned",
        "completed",
        "executed",
        "resumed",
        "succeeded",
        "failed",
        "verdict_passed",
        "verdict_failed",
        "verdict_absent",
    }
    # Execution outcomes and evaluator verdicts are reported apart, and said so.
    assert "execution outcome" in payload["counts_note"]
    assert "neither is a score" in payload["counts_note"]
    # Nothing in the report is a score: only execution outcomes are counted.
    assert "score" not in payload
    assert "accuracy" not in payload
    assert payload["retry_policy"] == {"retries": 0, "policy": "none"}
