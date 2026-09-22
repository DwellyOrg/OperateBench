"""The states a run can actually have been in, and the ones it cannot.

A hash chain proves a row was not edited. It proves nothing about whether the
rows describe a run that could have happened: a journal whose only call faulted
and whose terminal row says ``scored``, a decision recorded against a body this
build refused to read, two attempts under a policy that allows one, an attempt
after the one that already answered, two rows claiming the same turn — all of
those chain perfectly and all of them are impossible.

So the state machine is checked as well as the chain, in both directions. The
writer refuses an impossible call *before* it is appended, because evidence that
had to be written and then disbelieved is evidence that was written. The reader
re-checks independently, because it reads rows long after — and without
importing — whatever wrote them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from operatebench.execution_ledger import (
    ABORTED_CODE,
    TERMINAL_ABORTED,
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    ExecutionLedgerError,
    ExecutionLedgerJournal,
    LedgerCall,
    read_execution_ledger,
)
from tests.execution_ledger_fixtures import (
    DIGEST_C,
    LATER,
    answered_attempt,
    call,
    call_rows,
    controls,
    decision,
    faulted_attempt,
    faulted_call,
    header,
    rechained,
    rows_of,
    terminal_row,
)


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


def _written(directory: Path, *calls: LedgerCall, kind: str, code: str | None) -> Path:
    """One journal written honestly, as the starting point for an attack."""
    path = directory / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        for entry in calls:
            journal.record_call(entry)
        journal.finalize(kind, ended_at=LATER, exclusion_code=code)
    return path


# -- a scored run is a run every call of which answered ------------------------


def test_a_scored_terminal_over_a_faulted_call_is_refused_by_the_writer(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(faulted_call(0))
        with pytest.raises(ExecutionLedgerError):
            journal.finalize(TERMINAL_SCORED, ended_at=LATER)


def test_a_scored_terminal_over_a_faulted_call_is_refused_by_the_reader(
    ledger_dir: Path,
) -> None:
    path = _written(
        ledger_dir, faulted_call(0), kind=TERMINAL_EXCLUDED, code="provider_transport"
    )
    rows = rows_of(path)
    terminal = terminal_row(rows)["terminal"]
    terminal["kind"] = TERMINAL_SCORED
    terminal["exclusion_code"] = None
    terminal["exclusion_detail"] = ""
    target = rechained(rows, path.with_name("scored-fault.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_scored_terminal_over_a_call_with_no_decision_is_refused(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0, decision=None))
        with pytest.raises(ExecutionLedgerError):
            journal.finalize(TERMINAL_SCORED, ended_at=LATER)


def test_an_excluded_run_states_a_cause_and_not_only_a_code(ledger_dir: Path) -> None:
    """An excluded ledger whose every call answered names nothing that stopped it."""
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        with pytest.raises(ExecutionLedgerError):
            journal.finalize(
                TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
            )


def test_an_excluded_run_stopped_on_its_call_budget_needs_no_fault(
    ledger_dir: Path,
) -> None:
    """The one exclusion whose cause is this build's own bound rather than a fault."""
    path = _written(ledger_dir, call(0), kind=TERMINAL_EXCLUDED, code="provider_budget")
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_EXCLUDED
    assert not audit.scored


def test_an_excluded_run_stopped_on_its_clock_needs_no_fault(
    ledger_dir: Path,
) -> None:
    """A deadline is this build's own bound, and it stops a turn before a request."""
    path = _written(ledger_dir, call(0), kind=TERMINAL_EXCLUDED, code="provider_deadline")
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_EXCLUDED
    assert not audit.scored


def test_an_excluded_run_that_made_no_call_at_all_is_written(ledger_dir: Path) -> None:
    """Refused before its first request: there is nothing to contradict the stop."""
    path = _written(ledger_dir, kind=TERMINAL_EXCLUDED, code="provider_transport")
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_EXCLUDED
    assert audit.totals.provider_calls == 0
    assert not audit.scored


def test_an_excluded_run_that_ran_out_of_clock_mid_turn_is_written(
    ledger_dir: Path,
) -> None:
    path = _written(
        ledger_dir,
        faulted_call(0, terminal_reason="deadline_exceeded"),
        kind=TERMINAL_EXCLUDED,
        code="provider_transport",
    )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.calls[0].terminal_reason == "deadline_exceeded"


def test_an_aborted_run_over_answered_calls_is_written_and_never_scored(
    ledger_dir: Path,
) -> None:
    path = _written(ledger_dir, call(0), kind=TERMINAL_ABORTED, code=ABORTED_CODE)
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_ABORTED
    assert not audit.scored


def test_a_faulted_run_writes_an_excluded_terminal_and_reads_back(
    ledger_dir: Path,
) -> None:
    path = _written(
        ledger_dir,
        call(0),
        faulted_call(1),
        kind=TERMINAL_EXCLUDED,
        code="provider_transport",
    )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_EXCLUDED
    assert audit.totals.fault_counts == {"provider_server_error": 1}


# -- a decision belongs to an answer this build read --------------------------


def test_a_decision_on_a_body_this_build_refused_to_read_is_refused() -> None:
    """``provider_response_invalid`` receives bytes and produces no answer."""
    with pytest.raises(ExecutionLedgerError):
        call(
            0,
            attempts=(
                faulted_attempt(
                    fault="provider_response_invalid",
                    http_status=None,
                    response_received=True,
                ),
            ),
            terminal_reason="fault_not_retryable",
            decision=decision(),
        )


def test_a_decision_on_a_transport_fault_is_refused() -> None:
    with pytest.raises(ExecutionLedgerError):
        call(0, attempts=(faulted_attempt(),), decision=decision())


def test_a_decision_needs_the_normalized_response_digest_that_binds_it() -> None:
    with pytest.raises(ExecutionLedgerError):
        call(
            0,
            attempts=(answered_attempt(response_normalized_digest_sha256=None),),
            decision=decision(),
        )


def test_an_answered_call_may_carry_no_decision(ledger_dir: Path) -> None:
    """A turn that raised after the answer arrived: the row survives without one."""
    path = _written(
        ledger_dir, call(0, decision=None), kind=TERMINAL_ABORTED, code=ABORTED_CODE
    )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.calls[0].decision is None


def test_a_rechained_decision_added_to_a_faulted_call_is_refused(
    ledger_dir: Path,
) -> None:
    path = _written(
        ledger_dir, faulted_call(0), kind=TERMINAL_EXCLUDED, code="provider_transport"
    )
    rows = rows_of(path)
    call_rows(rows)[0]["call"]["decision"] = {
        "kind": "COMPLETE",
        "decision_digest_sha256": DIGEST_C,
        "classification_code": None,
    }
    target = rechained(rows, path.with_name("decided-fault.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


# -- attempts: bounded, contiguous, and terminal at the answer ----------------


def test_two_attempts_under_a_one_attempt_policy_are_refused_by_the_writer(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with (
        ExecutionLedgerJournal.open(path, header()) as journal,
        pytest.raises(ExecutionLedgerError),
    ):
        journal.record_call(
            faulted_call(
                0,
                attempts=(
                    faulted_attempt(attempt_index=1),
                    faulted_attempt(attempt_index=2),
                ),
            )
        )


def test_two_attempts_under_a_one_attempt_policy_are_refused_by_the_reader(
    ledger_dir: Path,
) -> None:
    path = _written(
        ledger_dir, faulted_call(0), kind=TERMINAL_EXCLUDED, code="provider_transport"
    )
    rows = rows_of(path)
    attempts = call_rows(rows)[0]["call"]["attempts"]
    second: dict[str, Any] = json.loads(json.dumps(attempts[0]))
    second["attempt_index"] = 2
    attempts.append(second)
    terminal = terminal_row(rows)["terminal"]
    terminal["totals"]["attempts"] = 2
    terminal["totals"]["fault_counts"] = {"provider_server_error": 2}
    terminal["totals"]["forfeited_reservation_usd"] = "0.118685"
    terminal["totals"]["reserved_usd"] = "0.118685"
    target = rechained(rows, path.with_name("two-attempts.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_second_attempt_is_allowed_where_the_policy_allows_one(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    two = header(controls=controls(max_attempts_per_call=2))
    with ExecutionLedgerJournal.open(path, two) as journal:
        journal.record_call(
            faulted_call(
                0,
                attempts=(
                    faulted_attempt(attempt_index=1),
                    faulted_attempt(attempt_index=2),
                ),
            )
        )
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.totals.attempts == 2


def test_an_attempt_after_the_one_that_answered_is_refused() -> None:
    """A turn ends at its answer; there is nothing left for a retry to change."""
    provisional = call(
        0,
        attempts=(
            answered_attempt(attempt_index=1),
            faulted_attempt(attempt_index=2),
        ),
        decision=None,
    )
    with pytest.raises(ExecutionLedgerError):
        provisional.validate_for_ledger_version(3, "call 0")


def test_an_attempt_after_a_fault_that_repeating_cannot_change_is_refused() -> None:
    provisional = call(
        0,
        attempts=(
            faulted_attempt(
                attempt_index=1,
                fault="provider_authentication",
                http_status=401,
            ),
            faulted_attempt(attempt_index=2),
        ),
        decision=None,
        terminal_reason="retries_exhausted",
    )
    with pytest.raises(ExecutionLedgerError):
        provisional.validate_for_ledger_version(3, "call 0")


def test_attempt_indexes_are_contiguous_from_one() -> None:
    with pytest.raises(ExecutionLedgerError):
        call(0, attempts=(answered_attempt(attempt_index=2),))


# -- the turn's own exit reason has to match the attempts --------------------


def test_a_turn_that_faulted_cannot_state_it_ended_on_a_response() -> None:
    provisional = faulted_call(0, terminal_reason="response")
    with pytest.raises(ExecutionLedgerError):
        provisional.validate_for_ledger_version(3, "call 0")


def test_a_turn_that_answered_cannot_state_it_exhausted_its_retries() -> None:
    provisional = call(0, terminal_reason="retries_exhausted")
    with pytest.raises(ExecutionLedgerError):
        provisional.validate_for_ledger_version(3, "call 0")


def test_retries_exhausted_names_every_attempt_the_policy_allowed(
    ledger_dir: Path,
) -> None:
    """One attempt of a permitted two is a turn that stopped, not one that ran out."""
    path = ledger_dir / "run.ndjson"
    two = header(controls=controls(max_attempts_per_call=2))
    with (
        ExecutionLedgerJournal.open(path, two) as journal,
        pytest.raises(ExecutionLedgerError),
    ):
        journal.record_call(faulted_call(0, terminal_reason="retries_exhausted"))


def test_usage_reported_without_a_body_is_refused() -> None:
    with pytest.raises(ExecutionLedgerError):
        answered_attempt(
            outcome="fault",
            fault="provider_response_invalid",
            response_received=False,
            response_model=None,
            response_id_digest_sha256=None,
            response_stop_classification=None,
            response_normalized_digest_sha256=None,
        )


def test_response_evidence_on_an_attempt_that_produced_no_answer_is_refused() -> None:
    with pytest.raises(ExecutionLedgerError):
        faulted_attempt(
            fault="provider_response_invalid",
            http_status=None,
            response_received=True,
            response_model="gpt-5.6-luna",
        )


# -- one call per turn, and turns move forwards ------------------------------


def test_two_call_rows_for_one_turn_are_refused_by_the_writer(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0, invocation_index=1, turn_index=0))
        with pytest.raises(ExecutionLedgerError):
            journal.record_call(call(1, invocation_index=1, turn_index=0))


def test_two_call_rows_for_one_turn_are_refused_by_the_reader(
    ledger_dir: Path,
) -> None:
    path = _written(ledger_dir, call(0), call(1), kind=TERMINAL_SCORED, code=None)
    rows = rows_of(path)
    call_rows(rows)[1]["call"]["turn_index"] = call_rows(rows)[0]["call"]["turn_index"]
    target = rechained(rows, path.with_name("duplicate-turn.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_turn_that_goes_backwards_is_refused_by_the_writer(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0, invocation_index=1, turn_index=4))
        with pytest.raises(ExecutionLedgerError):
            journal.record_call(call(1, invocation_index=1, turn_index=3))


def test_a_turn_that_goes_backwards_is_refused_by_the_reader(
    ledger_dir: Path,
) -> None:
    path = _written(ledger_dir, call(0), call(1), kind=TERMINAL_SCORED, code=None)
    rows = rows_of(path)
    rows_by_call = call_rows(rows)
    rows_by_call[0]["call"]["turn_index"] = 4
    rows_by_call[1]["call"]["turn_index"] = 3
    target = rechained(rows, path.with_name("backwards-turn.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_new_invocation_restarts_the_turn_count(ledger_dir: Path) -> None:
    path = _written(
        ledger_dir,
        call(0, invocation_index=1, turn_index=7),
        call(1, invocation_index=2, turn_index=0),
        kind=TERMINAL_SCORED,
        code=None,
    )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.totals.provider_calls == 2


def test_an_invocation_that_goes_backwards_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0, invocation_index=2, turn_index=0))
        with pytest.raises(ExecutionLedgerError):
            journal.record_call(call(1, invocation_index=1, turn_index=9))


def test_a_call_past_the_run_s_own_bound_is_refused_by_the_writer(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    small = header(controls=controls(max_provider_calls=2, expected_provider_calls=1))
    with ExecutionLedgerJournal.open(path, small) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1))
        with pytest.raises(ExecutionLedgerError):
            journal.record_call(call(2))
