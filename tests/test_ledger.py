"""The append-only episode ledger and what it refuses on resume."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    ScriptedTestAdapter,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.ledger import (
    KIND_UNKNOWN_ACTION,
    OUTCOME_LIMIT_EXHAUSTED,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    PHASE_RESPONSE_VALIDATION,
    EpisodeRecord,
    LedgerCorruptionError,
    LedgerError,
    LedgerIOError,
    append_episode,
    classify_failure,
    read_ledger,
    sample_status_for,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest
from boundarybench.runner import build_episode_record, compiled_variants, run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST


def _manifest(trials: int = 1):
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        provider="test-double",
        model="scripted-test-double",
        implementation="scripted_test_double",
        adapter_version=ADAPTER_CONTRACT_VERSION,
        adapter_settings={},
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _variants() -> dict[str, Any]:
    return compiled_variants(validate_suite(SUITE_MANIFEST))


def _with_overrides(record: EpisodeRecord, overrides: dict[str, Any]) -> EpisodeRecord:
    """Rebuild a genuine row with fields replaced, keeping the derived ones honest.

    ``sample_status`` follows from the outcome, so a test that relabels the
    outcome and leaves it behind builds a row that is refused by the
    *derivation* check before it reaches the invariant the test was written to
    probe. Re-deriving it here keeps each negative test pointed at its own
    subject, and models the stronger forgery: an editor who also rewrites every
    field that follows from the one they changed.
    """
    fields = {field: getattr(record, field) for field in record.__dataclass_fields__}
    fields.update(overrides)
    if "outcome" in overrides and "sample_status" not in overrides:
        fields["sample_status"] = sample_status_for(fields["outcome"])
    return EpisodeRecord(**fields)


def _record(manifest: Any, index: int = 0, **overrides: Any) -> EpisodeRecord:
    """Build a *genuine* row: a real episode, really graded.

    The fixture executes the loop rather than hand-writing a plausible-looking
    trajectory. A hand-written one would be rejected now — which is the point,
    but it would also mean every negative test below passed for the wrong
    reason, proving only that the fixture was invalid.
    """
    entry = manifest.episode_plan[index]
    variant = _variants()[entry.variant_id]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )
    record = build_episode_record(manifest, entry, variant, result)
    if not overrides:
        return record
    return _with_overrides(record, overrides)


def test_appending_writes_one_canonical_line_and_leaves_earlier_lines_alone(
    tmp_path: Path,
) -> None:
    manifest = _manifest(trials=2)
    path = tmp_path / "episodes.jsonl"

    append_episode(path, _record(manifest, 0))
    first_bytes = path.read_bytes()
    append_episode(path, _record(manifest, 1))

    raw = path.read_bytes()
    assert raw.startswith(first_bytes)
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 2
    # Canonical: one object per line, sorted keys, no incidental whitespace.
    assert lines[0] == json.dumps(
        json.loads(lines[0]), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    records = read_ledger(path, manifest, _variants())
    assert [record.episode_id for record in records] == [
        manifest.episode_plan[0].episode_id,
        manifest.episode_plan[1].episode_id,
    ]
    assert records[0].outcome == "success"
    assert records[0].evaluation is not None
    assert records[0].evaluation["variant_id"] == manifest.episode_plan[0].variant_id
    assert records[0].evaluation["passed"] is True


def test_missing_ledger_reads_as_no_episodes(tmp_path: Path) -> None:
    manifest = _manifest()
    assert read_ledger(tmp_path / "episodes.jsonl", manifest, _variants()) == ()


def _write_lines(path: Path, *payloads: Any, trailing_newline: bool = True) -> None:
    body = "".join(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        for payload in payloads
    )
    if not trailing_newline:
        body = body[:-1]
    path.write_text(body, encoding="utf-8")


def test_truncated_final_line_is_refused(tmp_path: Path) -> None:
    """A half-written row is evidence of a crash, never something to repair."""
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))
    path.write_bytes(path.read_bytes()[:-12])

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "does not end with a newline" in str(excinfo.value)


def test_malformed_json_line_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"run_id": broken}\n', encoding="utf-8")

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "line 1" in str(excinfo.value)


def test_duplicate_episode_row_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))
    append_episode(path, _record(manifest))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "duplicate episode" in str(excinfo.value)


def test_unplanned_episode_row_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, episode_id="not_in_the_plan#t001"))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "is not in this run's episode plan" in str(excinfo.value)


def test_row_from_another_configuration_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, configuration_id="c" * 64))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "belongs to configuration" in str(excinfo.value)


def test_row_from_another_execution_of_this_configuration_is_refused(
    tmp_path: Path,
) -> None:
    """The configuration is shared by every run of the plan; the execution is not.

    A row copied in from a second output directory carries this manifest's
    configuration id exactly — that is what deterministic identity means — so
    the configuration check alone would read it as evidence about this run.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(
        path, _record(manifest, execution_id="1f0d7a3c-cccc-4a1b-9c2d-000000000001")
    )

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "was written by execution" in str(excinfo.value)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row.pop("outcome"), "missing required field(s) ['outcome']"),
        (lambda row: row.update(extra=1), "unknown field(s) ['extra']"),
        # bool is an int subclass, so a boolean must not pass as a trial index.
        (lambda row: row.update(trial_index=True), "'trial_index' must be"),
        (lambda row: row.update(turns_used="3"), "'turns_used' must be"),
        (lambda row: row.update(outcome="triumph"), "unknown outcome 'triumph'"),
        (lambda row: row.update(trajectory=[]), "'trajectory' must be a JSON object"),
        (lambda row: row.update(usage=None), "'usage' must be a JSON object"),
        (lambda row: row.update(error_class=7), "'error_class' must be a string or null"),
        (
            lambda row: row.update(suite_content_digest="d" * 64),
            "suite_content_digest",
        ),
        (
            lambda row: row.update(scaffold_content_digest="d" * 64),
            "scaffold_content_digest",
        ),
        (lambda row: row.update(schema_version=8), "unsupported schema_version 8"),
        (
            lambda row: row.update(contract_versions={"package": "x"}),
            "contract_versions",
        ),
        (lambda row: row.update(variant_id="who_knows"), "variant_id"),
        (
            lambda row: row.update(configuration_id=7),
            "'configuration_id' must be a non-empty string",
        ),
        (
            lambda row: row.update(execution_id=7),
            "'execution_id' must be a non-empty string",
        ),
        # A row from the schema that had one id for both is refused as unknown
        # rather than read as either of the two that replaced it.
        (lambda row: row.update(run_id=row["configuration_id"]), "unknown field(s)"),
        (
            lambda row: row.update(sample_status="infrastructure_failure_ineligible"),
            "is never stated independently",
        ),
        (
            lambda row: row.update(started_at_utc="yesterday"),
            "started_at_utc must be an RFC 3339 UTC timestamp",
        ),
        (
            lambda row: row.update(suite_content_digest="not-a-digest"),
            "must be a lowercase 64-character SHA-256 hex digest",
        ),
        (
            lambda row: row.update(variant_content_digest="not-a-digest"),
            "'variant_content_digest' must be a lowercase 64-character",
        ),
        (lambda row: row.update(evaluation=[]), "'evaluation' must be a JSON object"),
        (lambda row: row.update(trial_index=2), "does not match the planned trial"),
        (
            lambda row: row.update(adapter={**row["adapter"], "provider": "openai"}),
            "does not match this run's adapter",
        ),
        # Inside the trajectory. Reconstruction reads every nested field, so each
        # one is a place a hand-edited row can disagree with the typed object the
        # evaluator has to be re-run against.
        (
            lambda row: row["trajectory"]["steps"].__setitem__(0, "read_records"),
            "steps[0] must be a JSON object",
        ),
        (
            lambda row: row["trajectory"]["steps"][0].update(arguments=[]),
            "'arguments' must be a JSON object",
        ),
        (
            lambda row: row["trajectory"].update(terminal_decision=1),
            "'terminal_decision' must be a JSON object or null",
        ),
        (
            lambda row: row["trajectory"].update(variant_id="another::cell::S0_P0"),
            "the trajectory belongs to variant",
        ),
        # Inside the verdict. The whole evaluation is recomputed, but its shape is
        # checked first: a predicate that is not a named pass/fail with a detail
        # is not something the recomputed verdict can be compared against.
        (
            lambda row: row["evaluation"]["predicates"].__setitem__(0, "passed"),
            "predicates[0] must be a JSON object",
        ),
        (
            lambda row: row["evaluation"]["predicates"][0].update(passed="yes"),
            "'passed' must be a boolean",
        ),
        (
            lambda row: row["evaluation"]["predicates"][0].update(detail=0),
            "'detail' must be a string",
        ),
    ],
)
def test_malformed_rows_are_refused(tmp_path: Path, mutate: Any, expected: str) -> None:
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    row = _record(manifest).as_dict()
    mutate(row)
    _write_lines(path, row)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert expected in str(excinfo.value)


@pytest.mark.parametrize("contract", ["package", "runner", "evaluator", "adapter"])
def test_row_graded_under_other_contract_versions_is_refused(
    tmp_path: Path, contract: str
) -> None:
    """A row is only evidence about the contracts that produced it.

    The evaluator contract names the grading rules, and the runner contract names
    how the loop counted turns and classified outcomes. A row written under a
    different one is a different measurement, so resuming onto it would silently
    mix two definitions of the same run.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    record = _record(manifest)
    append_episode(
        path,
        _record(
            manifest,
            contract_versions={**record.contract_versions, contract: "9.9.9-other"},
        ),
    )

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "contract_versions" in str(excinfo.value)


def test_a_terminally_failed_row_keeps_its_error_classification(tmp_path: Path) -> None:
    """A failure is recorded once, in full: outcome, class, detail and cause.

    The row is the only evidence that the episode was attempted, so a resume has
    to be able to tell "this failed and will not be retried" from "this has not
    run yet" without re-executing it.

    The failure is produced rather than asserted. A hand-written classification
    would round-trip whatever it was given, which proves nothing now that the
    summary is derived from the recorded cause: the row has to be one this
    build actually writes before its survival means anything.
    """
    manifest = _manifest()
    entry = manifest.episode_plan[0]
    variant = _variants()[entry.variant_id]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=30.0),
    )
    path = tmp_path / "episodes.jsonl"
    append_episode(path, build_episode_record(manifest, entry, variant, result))

    (record,) = read_ledger(path, manifest, _variants())

    assert record.succeeded is False
    assert record.outcome == OUTCOME_LIMIT_EXHAUSTED
    assert record.error_class == "max_turns"
    assert record.error_detail == (
        "episode reached its limit of 1 turn(s) without a terminal decision"
    )
    assert record.evaluation is None
    assert record.failure_event is not None
    assert record.failure_event["phase"] == "turn_start"
    assert record.failure_event["kind"] == "max_turns"
    assert record.failure_event["detail"] == record.error_detail


def test_an_empty_ledger_file_reads_as_no_episodes(tmp_path: Path) -> None:
    """A created-but-unwritten file is a run that has not got anywhere yet."""
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(b"")

    assert read_ledger(path, _manifest(), _variants()) == ()


def _unterminated_record(manifest: Any, **overrides: Any) -> EpisodeRecord:
    """A *genuine* row for an episode that never reached a decision.

    Produced by capping the turn budget at one rather than by editing a
    successful row, so the trajectory it carries is one the environment really
    produced and replays to itself. Forgeries below need that: a hand-built
    non-terminal trajectory would be refused by the replay check first, and the
    test would pass without ever reaching the classification it is about.
    """
    entry = manifest.episode_plan[0]
    variant = _variants()[entry.variant_id]
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=RunLimits(max_turns=1, max_messages=64, episode_timeout_seconds=30.0),
    )
    record = build_episode_record(manifest, entry, variant, result)
    if not overrides:
        return record
    return _with_overrides(record, overrides)


def test_a_failure_classification_must_name_the_call_it_refused() -> None:
    """Some faults are *about* a call, and cannot be classified without it.

    ``unknown_action`` and ``malformed_arguments`` say the adapter asked for
    something the scaffold does not offer. A row recording one without recording
    what was attempted would name a fault and withhold its only evidence, so the
    classifier refuses to produce one rather than writing a row that says less
    than it claims.
    """
    with pytest.raises(ValueError) as excinfo:
        classify_failure(
            phase=PHASE_RESPONSE_VALIDATION,
            kind=KIND_UNKNOWN_ACTION,
            detail="adapter called 'escalate', which the scaffold does not declare",
        )

    assert "requires the attempted call" in str(excinfo.value)


def test_a_record_whose_text_cannot_be_encoded_is_not_appended(tmp_path: Path) -> None:
    """The row is proven encodable before the file is touched at all.

    Error detail is the one field that carries text this build did not author: it
    comes from whatever the adapter or environment raised. A lone surrogate in it
    is a valid ``str`` that UTF-8 cannot encode, so the append has to fail as a
    named ledger error before the descriptor is opened — otherwise the ledger
    gains a partial line and the run's own reader rejects the file afterwards.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, 0))
    intact = path.read_bytes()

    with pytest.raises(LedgerError) as excinfo:
        append_episode(
            path,
            _unterminated_record(manifest, error_detail="provider said \ud800"),
        )

    assert "cannot be written to the ledger" in str(excinfo.value)
    # Nothing was appended, so the rows already committed are still readable.
    assert path.read_bytes() == intact
    assert len(read_ledger(path, manifest, _variants())) == 1


def test_a_record_carrying_an_unrenderable_integer_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A count no reader could carry is a named refusal, not a traceback.

    The provider integrations refuse a token count past
    :data:`~boundarybench.pricing.MAX_EXACT_TOKEN_COUNT` where they read it, so
    nothing should ever put one in a row. This is the layer under that: Python
    will not convert an integer of more than 4,300 digits to text, so a value
    that reached here anyway would raise ``ValueError`` from inside the JSON
    encoder — a type no ledger caller catches, at the point a durable row was
    being written.

    The append is where that has to be classified. What the row must not do is
    take the file down with it: the descriptor is never opened, the rows already
    committed stay readable, and the message says which row failed rather than
    what it contained.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, 0))
    intact = path.read_bytes()

    unrenderable = 10**5000
    with pytest.raises(LedgerError) as excinfo:
        append_episode(
            path,
            _record(manifest, 0, usage={"input_tokens": unrenderable}),
        )

    assert "cannot be written to the ledger" in str(excinfo.value)
    assert path.read_bytes() == intact
    assert len(read_ledger(path, manifest, _variants())) == 1


def test_a_trajectory_the_environment_would_refuse_is_not_evidence(
    tmp_path: Path,
) -> None:
    """Replay dispatches every recorded action, so an impossible one is caught.

    The row is internally consistent: the extra step has the right shape, the
    right index and reconstructs to exactly the bytes stored. What it does not
    have is a way to have happened — the case was already closed by the step
    before it, and the real environment refuses any action after that. Only
    replaying finds this; reconstruction and regrading both accept it.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    row = _record(manifest).as_dict()
    steps = row["trajectory"]["steps"]
    steps.append(
        {
            "index": len(steps) + 1,
            "action": "read_records",
            "arguments": {},
            "revealed_observation_ids": [],
            "mutations": [],
        }
    )
    _write_lines(path, row)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "could not be replayed through the real environment" in str(excinfo.value)


def test_a_success_without_a_terminal_decision_is_refused(tmp_path: Path) -> None:
    """``success`` is a claim about the shape of the episode, not just a label.

    Relabelling an episode that ran out of turns as a success was the cheapest
    forgery available: the trajectory replays, the counters add up and the error
    fields are simply cleared. Binding the outcome to the presence of a terminal
    decision is what closes it.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    row = _unterminated_record(manifest).as_dict()
    assert row["trajectory"]["terminal_decision"] is None
    row.update(
        outcome=OUTCOME_SUCCESS,
        # Relabelled with the eligibility that follows from the new outcome, so
        # this is the *complete* forgery rather than one that trips the derived
        # sample-status check before the terminal-decision binding is reached.
        sample_status=sample_status_for(OUTCOME_SUCCESS),
        error_class=None,
        error_detail=None,
        failure_event=None,
    )
    _write_lines(path, row)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "no terminal decision" in str(excinfo.value)


def test_a_failed_row_may_not_carry_a_verdict(tmp_path: Path) -> None:
    """Only an episode that reached a decision is graded, so only one is graded.

    A verdict attached to a failure is either invented or copied from another
    episode. Either way a report counting it would be counting a pass the
    evaluator never gave for a trajectory that never finished.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    row = _unterminated_record(manifest).as_dict()
    row["evaluation"] = _record(manifest).as_dict()["evaluation"]
    _write_lines(path, row)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "'evaluation' must be null for outcome" in str(excinfo.value)


def test_a_row_whose_variant_the_suite_does_not_hold_is_refused(tmp_path: Path) -> None:
    """A row can only be checked against the compiled variant it names.

    The variant is where the digest, the replay environment and the grading rules
    all come from. Reading a row without it would fall back to checking the
    digest for shape alone, which a well-formed digest of nothing in particular
    passes — so the row is refused instead of accepted unchecked.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, {})

    assert "is not in the compiled suite this run is bound to" in str(excinfo.value)


def test_a_verdict_that_cannot_be_recomputed_is_a_named_ledger_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator raising while regrading is corruption, not a crash.

    Every stored verdict is checked by running the evaluator again. If that
    raises, the row cannot be confirmed *or* refuted, and the read has to say so
    by name — an evaluator exception escaping ``read_ledger`` would reach the
    operator as a traceback in the middle of a resume.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise ArithmeticError("predicate weighting overflowed")

    monkeypatch.setattr("boundarybench.ledger.evaluate", _explode)

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "cannot be graded against variant" in str(excinfo.value)
    assert "ArithmeticError: predicate weighting overflowed" in str(excinfo.value)


def test_a_row_too_deep_to_parse_is_refused_by_name(tmp_path: Path) -> None:
    """``RecursionError`` is not a ``ValueError``, so it needs its own handler.

    A row nested past the parser's stack is corruption like any other truncated
    or malformed line, and is reported the same way. Without this it escaped
    every handler between the parser and the CLI and printed a traceback.
    """
    path = tmp_path / "episodes.jsonl"
    depth = 100_000
    path.write_text("[" * depth + "]" * depth + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, _manifest(), _variants())

    assert "nested too deeply" in str(excinfo.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"run_id": "a", "run_id": "b"}\n', "duplicate JSON key 'run_id'"),
        ('{"turns_used": NaN}\n', "not a finite JSON value"),
        ("[]\n", "each row must be a JSON object"),
        ("\n", "is blank; every row is one episode"),
    ],
)
def test_unparseable_ledger_rows_are_refused(
    tmp_path: Path, body: str, expected: str
) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, _manifest(), _variants())

    assert expected in str(excinfo.value)


def test_a_ledger_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    """Bytes that are not text are corruption, not a row to interpret loosely."""
    path = tmp_path / "episodes.jsonl"
    path.write_bytes(b'{"run_id": "\xff\xfe"}\n')

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, _manifest(), _variants())

    assert "not valid UTF-8" in str(excinfo.value)


# -- the file the ledger is, and is not --------------------------------------


def test_a_ledger_swapped_for_a_symlink_is_neither_written_nor_read_through(
    tmp_path: Path,
) -> None:
    """A ledger is a regular file written in place, at both ends.

    Validating the path and then opening it by name leaves a window: whatever
    the path pointed at when it was checked, it can point somewhere else by the
    time it is opened. Both the append and the read open the descriptor with
    ``O_NOFOLLOW``, so a link left in the ledger's place is refused rather than
    followed — and the file it aimed at is untouched.
    """
    manifest = _manifest()
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"do not append to me\n")
    path = tmp_path / "episodes.jsonl"
    path.symlink_to(victim)

    with pytest.raises(LedgerIOError) as append_error:
        append_episode(path, _record(manifest))
    with pytest.raises(LedgerIOError) as read_error:
        read_ledger(path, manifest, _variants())

    assert "never written through a link" in str(append_error.value)
    assert "never read through a link" in str(read_error.value)
    assert victim.read_bytes() == b"do not append to me\n"


def test_a_new_ledger_is_created_private_to_its_owner(tmp_path: Path) -> None:
    """The run's evidence is created 0600, not at whatever the umask allows.

    The mode is stated on the create, which is the only moment it can be set
    without a window in which the file exists and is world-readable. The
    process umask normally masks the difference away, so the umask is cleared
    here — otherwise a mode of 0777 and a mode of 0600 look identical on disk.
    """
    path = tmp_path / "episodes.jsonl"
    previous = os.umask(0)
    try:
        append_episode(path, _record(_manifest()))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_appended_row_is_on_disk_before_it_is_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durability barrier is the contract, and it has an order.

    ``append_episode`` returns only once the row survives a crash, because the
    caller treats a returned append as an episode that never needs running
    again. Syncing a descriptor whose bytes are still in a userspace buffer
    would return from that promise having written nothing.
    """
    path = tmp_path / "episodes.jsonl"
    record = _record(_manifest())
    seen: list[bytes] = []
    real_fsync = os.fsync

    def spy(descriptor: int) -> None:
        seen.append(path.read_bytes())
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", spy)

    append_episode(path, record)

    assert seen, "the append returned without ever syncing"
    assert seen[-1].endswith(b"\n")
    assert json.loads(seen[-1]) == record.as_dict()


# -- counters, provenance and the fields a row hands back --------------------


def test_a_success_that_spends_one_more_turn_than_it_records_is_refused(
    tmp_path: Path,
) -> None:
    """A completed episode ends by dispatching a step, not by failing after one.

    A *failed* episode may spend one turn more than it has steps — the turn that
    ended it. A success has no such turn, so the one-turn allowance must not
    reach it: with it, a row could claim a turn (and the two messages that buy
    it) that left no trace in the trajectory the row is graded on.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    genuine = _record(manifest)
    steps = len(genuine.trajectory["steps"])
    append_episode(
        path,
        _record(manifest, turns_used=steps + 1, messages_used=2 * (steps + 1)),
    )

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert genuine.outcome == OUTCOME_SUCCESS
    assert "exactly one turn per recorded step" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"error_class": "protocol_error", "error_detail": None},
        {"error_class": None, "error_detail": "the provider hiccupped"},
    ],
)
def test_a_success_carrying_either_error_field_is_refused(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """Half a classification on a success is still a contradiction.

    Requiring *both* fields to be present before objecting would accept a row
    that says it succeeded and names an error class, which is the shape a
    partially rewritten failure takes.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, **overrides))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "carries no 'error_class' or 'error_detail'" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"error_class": "max_turns", "error_detail": None},
        {"error_class": None, "error_detail": "ran out of turns"},
    ],
)
def test_a_failure_omitting_either_error_field_is_refused(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """A failure names both halves of its classification or it names nothing.

    The class says what kind of fault it was and the detail says what happened;
    a row with one and not the other cannot be acted on by a resume, and
    demanding that *both* be missing before objecting lets it through.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(
        path,
        _record(manifest, outcome=OUTCOME_LIMIT_EXHAUSTED, evaluation=None, **overrides),
    )

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "must name its 'error_class' and 'error_detail'" in str(excinfo.value)


def test_a_parsed_record_hands_back_exactly_the_row_it_accepted(
    tmp_path: Path,
) -> None:
    """Accepting evidence and returning it are one promise, not two.

    Everything downstream — the resume decision, the counts, the report — reads
    the parsed record rather than the file. A field that validates and is then
    dropped or blanked on the way out would make the ledger say one thing and
    the run report another, with nothing to reconcile them.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))
    persisted = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    (record,) = read_ledger(path, manifest, _variants())

    assert record.as_dict() == persisted
    # Named individually as well: an identical projection would also satisfy the
    # comparison above if both sides lost the same field.
    assert record.schema_version == 7
    assert record.configuration_id == manifest.configuration_id
    assert record.execution_id == manifest.execution_id
    assert record.started_at_utc <= record.completed_at_utc
    assert record.elapsed_seconds >= 0
    assert record.sample_status == sample_status_for(record.outcome)
    assert record.measurement["status"] in {"full", "partial", "unmeasured"}
    assert record.variant_id == manifest.episode_plan[0].variant_id
    assert record.trial_index == manifest.episode_plan[0].trial_index
    assert record.suite_content_digest == manifest.suite_content_digest
    assert record.scaffold_content_digest == manifest.scaffold_content_digest
    assert record.variant_content_digest == (
        _variants()[manifest.episode_plan[0].variant_id].content_digest
    )
    assert record.adapter == manifest.adapter.as_dict()
    assert record.contract_versions == dict(manifest.contract_versions)


@pytest.mark.parametrize(
    "usage",
    [
        {
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
            "latency_seconds": None,
        },
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": None,
            "latency_seconds": 0.0,
        },
        {
            "input_tokens": 1731,
            "output_tokens": 42,
            "cost_usd": None,
            "latency_seconds": 3.5,
        },
    ],
)
def test_usage_round_trips_exactly(tmp_path: Path, usage: dict[str, Any]) -> None:
    """Null is *not measured*; zero is a measurement of zero. Both are valid.

    A fake reports nothing and a real provider can genuinely bill nothing, so
    both survive the round trip unchanged — and a measurement that was taken
    comes back as the number that was taken, not as null and not as a string.

    ``cost_usd`` is null throughout because this manifest declares no cost cap
    and therefore pins no price. A cost is re-derived from the counts beside it
    at the run's own pricing policy, and a run that has no policy has no cost to
    record; a capped run's costs are round-tripped in
    :mod:`tests.test_runner_cost_control` and re-derived in
    :mod:`tests.test_cost_control_integrity`.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest, usage=usage))

    (record,) = read_ledger(path, manifest, _variants())

    assert dict(record.usage) == usage
    assert [type(value) for value in record.usage.values()] == [
        type(value) for value in usage.values()
    ]


def test_a_genuine_zero_turn_failure_is_a_valid_row(tmp_path: Path) -> None:
    """An episode can be over before it spends anything, and that is a record.

    A budget already exhausted when the loop starts ends the episode without a
    request, so ``turns_used`` and ``messages_used`` are genuinely zero. Reading
    zero as an impossible counter would refuse the one row that says the run
    attempted this episode and will not retry it.
    """
    manifest = _manifest()
    entry = manifest.episode_plan[0]
    variant = _variants()[entry.variant_id]
    times = iter([0.0])
    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=policy_following_script,
            identity=identity_for_test_double(),
            settings={},
        ),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
        clock=lambda: next(times, 100.0),
    )
    assert (result.outcome, result.turns_used, result.messages_used) == (
        OUTCOME_TIMEOUT,
        0,
        0,
    )
    path = tmp_path / "episodes.jsonl"
    append_episode(path, build_episode_record(manifest, entry, variant, result))

    (record,) = read_ledger(path, manifest, _variants())

    assert record.outcome == OUTCOME_TIMEOUT
    assert record.turns_used == 0
    assert record.messages_used == 0
    # Parsed evidence is recursively immutable, so a JSON array reads back as a
    # tuple; what matters is that the row records no steps at all.
    assert record.trajectory["steps"] == ()


def test_a_terminal_decision_with_secondary_reasons_round_trips(
    tmp_path: Path,
) -> None:
    """Secondary reason codes are part of the decision, not decoration.

    Every episode this build's fake produces closes with an empty list, which
    makes dropping the field on reconstruction invisible: an empty list is what
    the reconstructed decision defaults to. A decision that actually cites a
    second clause is the case that observes it.
    """
    manifest = _manifest()
    entry = manifest.episode_plan[0]
    variant = _variants()[entry.variant_id]
    codes = ["CORROBORATING_CLAUSE", "OPERATOR_NOTE"]

    def script(request: TurnRequest) -> AdapterCall:
        call = policy_following_script(request)
        if call.action != "complete_case":
            return call
        return AdapterCall(
            call.action, {**call.arguments, "secondary_reason_codes": list(codes)}
        )

    result = run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=script, identity=identity_for_test_double(), settings={}
        ),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )
    assert result.outcome == OUTCOME_SUCCESS
    path = tmp_path / "episodes.jsonl"
    append_episode(path, build_episode_record(manifest, entry, variant, result))

    (record,) = read_ledger(path, manifest, _variants())

    # Evidence is handed back recursively immutable, so the round-tripped array
    # is a tuple of the same codes in the same order.
    assert record.trajectory["terminal_decision"]["secondary_reason_codes"] == tuple(
        codes
    )
    assert record.trajectory["steps"][-1]["arguments"]["secondary_reason_codes"] == tuple(
        codes
    )


def test_a_row_whose_free_form_arguments_hold_infinity_is_refused(
    tmp_path: Path,
) -> None:
    """Step arguments are the one free-form payload a row holds.

    Every other field has a fixed shape, so this is the only place an arbitrary
    value can be smuggled in. ``json.loads`` accepts the ``Infinity`` literal
    and hands back a float canonical JSON cannot write, so a row carrying one
    could never be re-encoded to the bytes it claims to be — the shared
    JSON-safety walk refuses the row before any field is read, rather than each
    reader re-deriving the same rule.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    append_episode(path, _record(manifest))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["trajectory"]["steps"][0]["arguments"]["budget"] = float("inf")
    path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(LedgerCorruptionError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "is not a finite JSON value" in str(excinfo.value)


def test_an_out_of_order_row_at_the_end_of_the_plan_is_a_named_failure(
    tmp_path: Path,
) -> None:
    """Order is refused by name wherever it breaks, including the last position.

    The ledger must be an exact prefix of the plan, and the check reports which
    episode belonged in the position that disagreed. At the tail of the plan
    there is no *next* entry to report, so a diagnosis that looked one place too
    far would raise ``IndexError`` — an unhandled crash instead of the named
    ledger failure the CLI knows how to report.
    """
    manifest = _manifest()
    path = tmp_path / "episodes.jsonl"
    last = len(manifest.episode_plan) - 1
    # In plan order up to the tail, then the final two episodes swapped, so the
    # first disagreement lands on the last position the plan can name.
    for index in [*range(last - 1), last, last - 1]:
        append_episode(path, _record(manifest, index))

    with pytest.raises(LedgerError) as excinfo:
        read_ledger(path, manifest, _variants())

    assert "out of plan order" in str(excinfo.value)
