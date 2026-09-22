"""Adversarial checks for runtime trust boundaries.

Each case lets something outside the run attempt to decide what the run records.
The invariant is refusal rather than repair whenever external input disagrees
with derived evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    ScriptedTestAdapter,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.ledger import LedgerError, read_ledger, sample_status_for
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    RunLockError,
    RunManifestError,
    RunManifestMismatchError,
    RunPathError,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
    resolve_run_paths,
    write_run_manifest,
)
from boundarybench.runner import (
    MESSAGES_PER_TURN,
    RunProvenanceError,
    compiled_variants,
    execute_run,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST


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


# -- unsafe paths ------------------------------------------------------------


def test_a_precreated_temporary_manifest_symlink_cannot_capture_the_write(
    tmp_path: Path,
) -> None:
    """The scratch file is created exclusively, so a squatted name is not followed.

    A predictable ``.run_manifest.json.tmp`` is pre-created as a symlink. The
    exclusive scratch-file creation must not follow it or overwrite its target.
    """
    root = tmp_path / "run"
    root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite me\n", encoding="utf-8")
    (root / f".{MANIFEST_FILENAME}.tmp").symlink_to(victim)

    manifest = _manifest()
    write_run_manifest(resolve_run_paths(root), manifest)

    assert victim.read_text(encoding="utf-8") == "do not overwrite me\n"
    assert (root / MANIFEST_FILENAME).exists()


def test_a_symlinked_ancestor_of_the_run_directory_is_refused(tmp_path: Path) -> None:
    """Checking only the final component lets a parent link redirect the whole run."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(RunPathError) as excinfo:
        resolve_run_paths(link / "run")

    assert "symlink" in str(excinfo.value)
    assert not (elsewhere / "run").exists()


# -- one writer at a time ----------------------------------------------------


def test_a_second_writer_cannot_open_a_run_another_writer_holds(
    tmp_path: Path,
) -> None:
    """The lock spans manifest open, ledger validation and every append.

    Without it two runners snapshot the same ``done`` set and both append the
    same episodes. ``flock`` is used rather than a sentinel file so the lock is
    released by the kernel when a crashed writer's descriptors close, and a
    stale run directory never needs manual repair.
    """
    manifest = _manifest()
    root = tmp_path / "run"

    with open_run_session(root, manifest) as first:
        assert first.paths.manifest_path.exists()
        with (
            pytest.raises(RunLockError) as excinfo,
            open_run_session(root, manifest),
        ):
            pass  # pragma: no cover - the lock must refuse this

    assert "already running" in str(excinfo.value)
    # Released on exit: the same directory opens cleanly afterwards.
    with open_run_session(root, manifest) as third:
        assert third.resumed is True


# -- provenance ---------------------------------------------------------------


def _fake_adapter(**overrides: object) -> ScriptedTestAdapter:
    fields: dict[str, object] = {
        "script": policy_following_script,
        "identity": identity_for_test_double(),
        "settings": {},
    }
    fields.update(overrides)
    return ScriptedTestAdapter(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "identity": identity_for_test_double(
                    implementation="UNRECORDED_OTHER_IMPL"
                )
            },
            "adapter",
        ),
        ({"settings": {"temperature": "0.9"}}, "adapter_settings"),
    ],
)
def test_an_adapter_that_is_not_the_recorded_one_fails_before_execution(
    tmp_path: Path, overrides: dict[str, object], expected: str
) -> None:
    """Rows must describe the adapter that ran, not the manifest's claim about it.

    The adversarial adapter disagrees with the manifest's implementation or
    settings. Execution is refused before a row can persist the wrong identity.
    """
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)

    with open_run_session(tmp_path / "run", manifest) as session:
        with pytest.raises(RunProvenanceError) as excinfo:
            execute_run(
                suite=suite,
                scaffold=scaffold,
                adapter=_fake_adapter(**overrides),
                session=session,
            )
        assert expected in str(excinfo.value)
        # Nothing executed, so nothing was recorded.
        assert not session.paths.ledger_path.exists()


# -- forged ledger evidence ---------------------------------------------------


def _completed_run(root: Path, trials: int = 1) -> tuple[object, dict[str, object]]:
    """Execute the shipped runner and return its manifest and compiled variants."""
    manifest = _manifest(trials=trials)
    suite = validate_suite(SUITE_MANIFEST)
    with open_run_session(root, manifest) as session:
        execute_run(
            suite=suite,
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=_fake_adapter(),
            session=session,
        )
    return manifest, compiled_variants(suite)


def _rewrite_first_row(path: Path, mutate: Any) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    mutate(row)
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        # Three concrete forgeries.
        (
            lambda row: row["trajectory"].update(unexpected="x"),
            "unknown field(s)",
        ),
        (lambda row: row.update(evaluation={"forged": True}), "evaluation"),
        (lambda row: row.update(variant_content_digest="0" * 64), "variant_content"),
        # Nested trajectory shapes.
        (lambda row: row["trajectory"].update(steps="not-a-list"), "'steps' must be"),
        (
            lambda row: row["trajectory"].update(terminal_decision={"bad": 1}),
            "terminal_decision",
        ),
        (
            lambda row: row["trajectory"].update(mutation_history=[["one"]]),
            "mutation_history",
        ),
        (
            lambda row: row["trajectory"]["steps"][0].update(index=True),
            "'index' must be",
        ),
        (
            lambda row: row["trajectory"]["steps"][0].update(extra=1),
            "unknown field(s)",
        ),
        (
            lambda row: row["trajectory"]["steps"][0].update(mutations="drop"),
            "'mutations' must be",
        ),
        (
            lambda row: row["trajectory"].update(observed_ids=[1]),
            "observed_ids",
        ),
        # Predicate-level forgery: pass a verdict the evaluator did not give.
        (
            lambda row: row["evaluation"].update(passed="yes"),
            "evaluation",
        ),
        (
            lambda row: row["evaluation"]["predicates"][0].update(detail="fabricated"),
            "evaluation",
        ),
        # Usage.
        (lambda row: row["usage"].update(input_tokens=True), "input_tokens"),
        (lambda row: row["usage"].update(output_tokens=-1), "output_tokens"),
        (lambda row: row["usage"].update(cost_usd="free"), "cost_usd"),
        (lambda row: row["usage"].update(latency_seconds=-0.5), "latency_seconds"),
        # Outcome/evaluation/error consistency.
        (lambda row: row.update(evaluation=None), "evaluation"),
        # ``sample_status`` is carried through with the outcome in both of these:
        # it is derived from the outcome, so leaving it behind would stop the row
        # at that derivation instead of at the invariant each case is about.
        (
            lambda row: row.update(
                outcome="timeout",
                sample_status=sample_status_for("timeout"),
                error_class=None,
            ),
            "error_class",
        ),
        # Relabelling a genuine success as a timeout fails on the stronger
        # invariant: a failing outcome must name the failure event that
        # produced it, and a success carries none to rename.
        (
            lambda row: row.update(
                outcome="timeout",
                sample_status=sample_status_for("timeout"),
                error_class="episode_timeout",
                error_detail="cut off",
            ),
            "must carry a 'failure_event'",
        ),
        # And the derivation itself: a relabel that does *not* carry it through
        # is refused for saying two things about one episode.
        (
            lambda row: row.update(outcome="timeout"),
            "is never stated independently",
        ),
    ],
)
def test_forged_ledger_rows_are_refused(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    """A row is only evidence if it reconstructs and regrades to what it claims.

    The adversarial rows add unknown trajectory keys, replace the evaluation with
    ``{"forged": true}``, or zero the variant digest. Reading rebuilds the typed
    trajectory, binds the digest to the planned variant and recomputes the
    verdict.
    """
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)
    ledger = root / "episodes.jsonl"
    _rewrite_first_row(ledger, mutate)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert expected in str(excinfo.value)


def test_a_ledger_that_is_not_a_prefix_of_the_plan_is_refused(tmp_path: Path) -> None:
    """Resume appends a suffix, so the rows on disk must be a prefix in order."""
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)
    ledger = root / "episodes.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    # Keep only the fifth episode: a gap, not a prefix.
    ledger.write_text(lines[4], encoding="utf-8")

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "prefix" in str(excinfo.value)


# -- a manifest that is strict about its own types ----------------------------


def _stored(tmp_path: Path, mutate: Any) -> Path:
    """Write a manifest, edit it, and hand back the path to load."""
    root = tmp_path / "run"
    with open_run_session(root, _manifest()) as session:
        path = session.paths.manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        # `False == 0` in Python; schema comparison must still distinguish them.
        (
            lambda m: m["retry_policy"].update(retries=False),
            "retry_policy",
        ),
        (lambda m: m["retry_policy"].update(retries=0.0), "retry_policy"),
        (lambda m: m["retry_policy"].update(policy=None), "retry_policy"),
        (lambda m: m["retry_policy"].update(extra=1), "retry_policy"),
        # Digest-shaped fields were only checked for being non-empty strings.
        (lambda m: m["suite"].update(suite_content_digest="short"), "SHA-256"),
        (lambda m: m["scaffold"].update(content_digest="ABC" + "d" * 61), "SHA-256"),
        # Numeric boundaries, including bool-as-int.
        (lambda m: m.update(trials=True), "'trials'"),
        (lambda m: m["limits"].update(max_turns=True), "'max_turns'"),
        (lambda m: m["limits"].update(max_messages=True), "'max_messages'"),
        (
            lambda m: m["limits"].update(episode_timeout_seconds=True),
            "episode_timeout_seconds",
        ),
        (
            lambda m: m["episode_plan"][0].update(trial_index=True),
            "'trial_index'",
        ),
    ],
)
def test_stored_manifests_are_type_strict(
    tmp_path: Path, mutate: Any, expected: str
) -> None:
    with pytest.raises(RunManifestError) as excinfo:
        load_run_manifest(_stored(tmp_path, mutate))

    assert expected in str(excinfo.value)


def test_a_manifest_field_added_after_the_fact_is_refused(tmp_path: Path) -> None:
    """The stored manifest holds exactly two things and admits no third.

    The adversarial field is a loose wall-clock ``created_at`` with no place in
    the stored schema. Creation time belongs only in the ``execution`` block,
    bound to the execution id and configuration id as one hashed unit.
    """
    path = _stored(tmp_path, lambda m: m.update(created_at="2099-01-01T00:00:00+00:00"))

    with pytest.raises(RunManifestError) as excinfo:
        load_run_manifest(path)

    assert "unknown field(s)" in str(excinfo.value)


def test_editing_any_stored_field_breaks_the_recorded_configuration_id(
    tmp_path: Path,
) -> None:
    """One rehash covers the configuration, because the configuration is identity."""
    path = _stored(tmp_path, lambda m: m.update(trials=2))

    with pytest.raises(RunManifestMismatchError) as excinfo:
        load_run_manifest(path)

    assert "does not hash to its recorded configuration_id" in str(excinfo.value)


def test_identical_plans_started_at_different_times_keep_one_configuration_id() -> None:
    """The whole point of separating configuration identity from the clock.

    And of separating it from execution identity: the two manifests are the same
    plan and are *not* the same run, which is why the second assertion is here
    rather than only the first.
    """
    first, second = _manifest(), _manifest()

    assert first.configuration_id == second.configuration_id
    assert first.execution_id != second.execution_id


def test_reordered_ledger_rows_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)
    ledger = root / "episodes.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    ledger.write_text("".join([lines[1], lines[0], *lines[2:]]), encoding="utf-8")

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(ledger, manifest, variants)

    assert "prefix" in str(excinfo.value)


# -- what the summary actually claims -----------------------------------------


def test_the_summary_separates_execution_outcomes_from_evaluator_verdicts(
    tmp_path: Path,
) -> None:
    """Execution outcomes and evaluator verdicts remain separate counts.

    An agent that reaches a terminal decision the evaluator fails is a completed
    episode and a wrong answer, so verdict counts are reported alongside outcome
    counts and never folded together.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)

    def always_stop(request: Any) -> Any:
        """Terminate immediately: the loop completes, the verdict does not."""
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall(
            "complete_case",
            {
                "disposition": "STOP",
                "primary_reason_code": "NOT_THE_EXPECTED_REASON",
                "secondary_reason_codes": [],
                "evidence_refs": [],
            },
        )

    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
            adapter=_fake_adapter(script=always_stop),
            session=session,
        )

    payload = report.as_dict()
    counts = payload["counts"]
    assert counts["completed"] == counts["planned"] == 12
    # Every episode ran to a terminal decision...
    assert counts["succeeded"] == 12
    assert counts["failed"] == 0
    # ...and every verdict is a fail. The two are reported separately.
    assert counts["verdict_passed"] == 0
    assert counts["verdict_failed"] == 12
    assert counts["verdict_absent"] == 0
    assert "execution outcome" in payload["counts_note"]


# -- interruption --------------------------------------------------------------


def test_an_interruption_after_three_appends_resumes_only_the_missing_suffix(
    tmp_path: Path,
) -> None:
    """A real interruption, not a hand-truncated file.

    The first run is stopped by raising out of the adapter once three episodes
    have durably landed. The second run must re-execute exactly the missing
    nine, leave the first three bytes untouched, and never call the adapter for
    an episode that already has a row.
    """
    root = tmp_path / "run"
    manifest = _manifest()
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    ledger = root / "episodes.jsonl"

    class Interrupted(KeyboardInterrupt):
        """A real interruption: what ^C or a SIGINT does mid-run.

        Deliberately not an ordinary exception — the runner *classifies* those
        as adapter failures and records a row, which is correct behaviour and
        the opposite of an interruption. This has to escape the loop entirely,
        leaving only what was already durable on disk.
        """

    def instrumented(seen: list[str], stop_after: int | None) -> Any:
        def script(request: Any) -> Any:
            if not request.transcript:
                # One entry per episode, recorded at its first turn.
                if stop_after is not None and len(seen) == stop_after:
                    raise Interrupted("power cut")
                seen.append(request.system_prompt[:0] or str(len(seen)))
            return policy_following_script(request)

        return script

    first_calls: list[str] = []
    with pytest.raises(Interrupted), open_run_session(root, manifest) as session:
        execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_fake_adapter(script=instrumented(first_calls, 3)),
            session=session,
        )

    assert len(first_calls) == 3
    prefix = ledger.read_bytes()
    assert prefix.count(b"\n") == 3

    second_calls: list[str] = []
    with open_run_session(root, manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=_fake_adapter(script=instrumented(second_calls, None)),
            session=session,
        )

    # Only the missing suffix ran, and the durable prefix is byte-identical.
    assert len(second_calls) == 9
    assert report.resumed_count == 3
    assert report.executed_count == 9
    assert report.completed_count == 12
    assert ledger.read_bytes().startswith(prefix)
    executed_ids = [record.episode_id for record in report.records]
    assert executed_ids == list(manifest.planned_episode_ids)


def test_a_row_records_the_messages_the_episode_spent(tmp_path: Path) -> None:
    """Turns alone do not describe the budget an episode consumed."""
    root = tmp_path / "run"
    manifest, variants = _completed_run(root)

    records = read_ledger(root / "episodes.jsonl", manifest, variants)

    for record in records:
        assert record.messages_used == record.turns_used * MESSAGES_PER_TURN
