"""Money in a ledger is arithmetic, not a field.

The chain proves a row was not edited after it was written. It cannot prove an
amount is the amount that run was authorised for, because an attacker who
re-chains the whole journal can write any number they like into a row and any
matching number into the totals. A reader that adds those numbers up and reports
the sum is reporting the attacker's arithmetic.

So every amount here is *derived* rather than read: the reservation from the
call's own input bound, the run's output ceiling and the run's pinned rates; the
measured cost from the counts the provider reported at those same rates. What is
stored is checked against what follows, and a stored amount that does not follow
is refused. The three totals stay apart — a forfeited reservation is exposure
and never spend — and each is rebuilt from the derived values rather than from
the recorded ones.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from operatebench.execution_ledger import (
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    ExecutionLedgerError,
    ExecutionLedgerJournal,
    derive_measured_usd,
    derive_reservation_usd,
    read_execution_ledger,
)
from tests.execution_ledger_fixtures import (
    INPUT_TOKEN_UPPER_BOUND,
    LATER,
    MEASURED_USD,
    REPORTED_INPUT_TOKENS,
    REPORTED_OUTPUT_TOKENS,
    RESERVATION_USD,
    answered_attempt,
    call,
    call_rows,
    controls,
    faulted_attempt,
    faulted_call,
    header,
    pricing,
    rechained,
    rows_of,
    terminal_row,
)


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


def _scored(directory: Path) -> Path:
    path = directory / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    return path


def _excluded(directory: Path) -> Path:
    path = directory / "faulted.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(faulted_call(1))
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    return path


# -- the formulae themselves --------------------------------------------------


def test_a_reservation_is_the_run_s_bounds_at_the_run_s_rates() -> None:
    assert (
        derive_reservation_usd(
            input_token_upper_bound=INPUT_TOKEN_UPPER_BOUND, controls=controls()
        )
        == RESERVATION_USD
    )


def test_a_measured_cost_is_the_reported_counts_at_the_run_s_rates() -> None:
    assert (
        derive_measured_usd(
            input_tokens=REPORTED_INPUT_TOKENS,
            output_tokens=REPORTED_OUTPUT_TOKENS,
            controls=controls(),
        )
        == MEASURED_USD
    )


def test_the_derivation_is_exact_decimal_arithmetic_and_not_a_float() -> None:
    exact = (
        Decimal(REPORTED_INPUT_TOKENS) * Decimal("1.25")
        + Decimal(REPORTED_OUTPUT_TOKENS) * Decimal("10")
    ) / Decimal(1_000_000)
    derived = Decimal(
        derive_measured_usd(
            input_tokens=REPORTED_INPUT_TOKENS,
            output_tokens=REPORTED_OUTPUT_TOKENS,
            controls=controls(),
        )
    )
    assert derived == exact


def test_a_ledger_whose_amounts_follow_from_its_own_numbers_reads_clean(
    ledger_dir: Path,
) -> None:
    audit = read_execution_ledger(_scored(ledger_dir), require_complete=True)
    assert audit.status == TERMINAL_SCORED
    assert audit.totals.measured_cost_usd == "0.0128875"
    assert audit.totals.reserved_usd == "0.118685"
    assert audit.totals.forfeited_reservation_usd == "0"


# -- re-chained edits to the amounts themselves -------------------------------


def test_a_rechained_measured_cost_edit_is_refused(ledger_dir: Path) -> None:
    """The reviewer's attack: 0.00644375 becomes 0.0001, totals kept in step."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    for row in call_rows(rows):
        row["call"]["attempts"][0]["cost_measured_usd"] = "0.0001"
    terminal_row(rows)["terminal"]["totals"]["measured_cost_usd"] = "0.0002"
    target = rechained(rows, path.with_name("cheap.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_rechained_reservation_edit_is_refused(ledger_dir: Path) -> None:
    """0.0593425 becomes 0.01, totals kept in step, tokens and rates untouched."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    for row in call_rows(rows):
        row["call"]["attempts"][0]["cost_reservation_usd"] = "0.01"
    terminal_row(rows)["terminal"]["totals"]["reserved_usd"] = "0.02"
    target = rechained(rows, path.with_name("small-reservation.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_rechained_token_count_edit_is_refused(ledger_dir: Path) -> None:
    """Counts and cost are two readings of one measurement; moving one is a lie."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    for row in call_rows(rows):
        row["call"]["attempts"][0]["input_tokens"] = 41
    terminal = terminal_row(rows)["terminal"]
    terminal["totals"]["input_tokens"] = 82
    target = rechained(rows, path.with_name("fewer-tokens.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_rechained_input_bound_edit_is_refused(ledger_dir: Path) -> None:
    """The bound is what authorised the reservation, so it cannot move alone."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    for row in call_rows(rows):
        row["call"]["input_token_upper_bound"] = 100
    target = rechained(rows, path.with_name("small-bound.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def test_a_rechained_rate_edit_with_a_matching_pricing_digest_is_refused(
    ledger_dir: Path,
) -> None:
    """The digest covers the table, so an attacker can move both. The amounts stay."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    table = pricing(input_usd_per_mtok="0.01", output_usd_per_mtok="0.02")
    rows[0]["header"]["controls"]["pricing"] = table
    rows[0]["header"]["controls"]["pricing_digest_sha256"] = _digest_of(table)
    target = rechained(rows, path.with_name("cheap-rates.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


def _digest_of(table: dict[str, object]) -> str:
    from operatebench.execution_ledger import ledger_digest

    return ledger_digest(table)


def test_a_rechained_output_ceiling_edit_is_refused(ledger_dir: Path) -> None:
    """The ceiling is half of every reservation this run ever authorised."""
    path = _scored(ledger_dir)
    rows = rows_of(path)
    rows[0]["header"]["controls"]["max_output_tokens"] = 16
    target = rechained(rows, path.with_name("small-ceiling.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


# -- the writer will not put an underivable amount on disk --------------------


def test_the_writer_refuses_a_reservation_it_cannot_derive(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with (
        ExecutionLedgerJournal.open(path, header()) as journal,
        pytest.raises(ExecutionLedgerError),
    ):
        journal.record_call(
            call(0, attempts=(answered_attempt(cost_reservation_usd="0.01"),))
        )


def test_the_writer_refuses_a_measured_cost_it_cannot_derive(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with (
        ExecutionLedgerJournal.open(path, header()) as journal,
        pytest.raises(ExecutionLedgerError),
    ):
        journal.record_call(
            call(0, attempts=(answered_attempt(cost_measured_usd="0.0001"),))
        )


# -- what each settlement is allowed to say -----------------------------------


def test_a_measured_settlement_without_reported_usage_is_refused() -> None:
    with pytest.raises(ExecutionLedgerError):
        faulted_attempt(
            outcome="response",
            fault=None,
            http_status=None,
            response_received=True,
            usage_reported=False,
            cost_settlement="measured",
            cost_measured_usd=MEASURED_USD,
            cost_forfeited_usd=None,
            response_model="gpt-5.6-luna",
            response_stop_classification="completed",
        )


def test_a_forfeited_attempt_forfeits_the_whole_reservation_and_no_other_amount(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with (
        ExecutionLedgerJournal.open(path, header()) as journal,
        pytest.raises(ExecutionLedgerError),
    ):
        journal.record_call(
            faulted_call(0, attempts=(faulted_attempt(cost_forfeited_usd="0.01"),))
        )


def test_a_forfeited_reservation_is_never_reported_as_spend(ledger_dir: Path) -> None:
    audit = read_execution_ledger(_excluded(ledger_dir), require_complete=True)
    assert audit.totals.forfeited_reservation_usd == RESERVATION_USD
    assert audit.totals.measured_cost_usd == MEASURED_USD
    assert audit.totals.measured_cost_usd != audit.totals.reserved_usd


def test_a_rechained_forfeited_reservation_reported_as_measured_spend_is_refused(
    ledger_dir: Path,
) -> None:
    """The one edit that turns exposure into a claim about money that was spent."""
    path = _excluded(ledger_dir)
    rows = rows_of(path)
    faulted = call_rows(rows)[1]["call"]["attempts"][0]
    faulted["cost_settlement"] = "measured"
    faulted["cost_forfeited_usd"] = None
    faulted["cost_measured_usd"] = RESERVATION_USD
    terminal = terminal_row(rows)["terminal"]
    terminal["totals"]["forfeited_reservation_usd"] = "0"
    terminal["totals"]["measured_cost_usd"] = "0.0658"
    target = rechained(rows, path.with_name("forfeit-as-spend.ndjson"))
    with pytest.raises(ExecutionLedgerError):
        read_execution_ledger(target)


# -- the totals are rebuilt from the derived values ---------------------------


def test_the_totals_are_rebuilt_from_derived_amounts_not_recorded_ones(
    ledger_dir: Path,
) -> None:
    audit = read_execution_ledger(_excluded(ledger_dir), require_complete=True)
    derived_reservations = [
        derive_reservation_usd(
            input_token_upper_bound=entry.input_token_upper_bound, controls=controls()
        )
        for entry in audit.calls
    ]
    assert audit.totals.reserved_usd == "0.118685"
    assert set(derived_reservations) == {RESERVATION_USD}
    assert audit.totals.measured_cost_usd == derive_measured_usd(
        input_tokens=REPORTED_INPUT_TOKENS,
        output_tokens=REPORTED_OUTPUT_TOKENS,
        controls=controls(),
    )
