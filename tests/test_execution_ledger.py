"""The crash-safe execution ledger: one journal, one chain, one refusal each.

The ledger is a hash-chained NDJSON journal rather than one end-written
document, and every property that matters here follows from that: a row is
appended, flushed and fsynced before the next call is made, so a run that dies
mid-flight leaves a verifiable *prefix* rather than nothing, and that prefix can
never be read as a scored run.

Nothing in this module opens a socket, reads a credential or names a provider
endpoint it did not compute from this build's own settings.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import operatebench.execution_ledger as execution_ledger
import operatebench.providers.faults as live_faults
from operatebench.core.clock import format_timestamp
from operatebench.execution_ledger import (
    ABORTED_CODE,
    EXECUTION_LEDGER_VERSION,
    GENESIS_DIGEST,
    INTENDED_ARTIFACT_VERSION,
    ROW_CALL,
    ROW_HEADER,
    ROW_TERMINAL,
    TERMINAL_ABORTED,
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    ExecutionLedgerJournal,
    IncompleteExecutionLedgerError,
    LedgerAttempt,
    LedgerCall,
    LedgerChainError,
    LedgerControls,
    LedgerDecision,
    LedgerHeader,
    LedgerPathError,
    LedgerProviderIdentity,
    LedgerSchemaError,
    LedgerSecretError,
    LedgerStateError,
    LedgerTotals,
    _is_recoverable_json_object_prefix,
    assert_no_secret_material,
    new_execution_run_id,
    read_execution_ledger,
    recompute_totals,
    row_digest,
)
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.faults import TURN_END_PRE_DISPATCH_REFUSED

# -- fixtures ----------------------------------------------------------------

INSTANT = format_timestamp(1_900_000_000)
LATER = format_timestamp(1_900_000_060)
REPO_ROOT = Path(__file__).resolve().parents[1]

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def settings() -> dict[str, Any]:
    return {
        "api": "responses",
        "base_url": "https://api.openai.com/v1",
        "request_mapping": "lifecycle_openai_responses_model_request_v5",
        "sdk": "openai",
        "tool_choice": "required",
    }


def provider_identity() -> LedgerProviderIdentity:
    return LedgerProviderIdentity(
        provider="openai",
        api="responses",
        model="gpt-5.6-luna",
        base_url="https://api.openai.com/v1",
        implementation="OpenAIResponsesTransport",
        request_mapping="lifecycle_openai_responses_model_request_v5",
        protocol_version="operatebench.model.v3",
        sdk="openai",
        sdk_version="2.24.0",
        settings=settings(),
    )


def pricing() -> dict[str, Any]:
    return {
        "policy_id": "operator_pinned_v1",
        "input_usd_per_mtok": "1.25",
        "output_usd_per_mtok": "10.00",
        "rate_source": "operator_supplied_pinned_rates",
    }


def controls() -> LedgerControls:
    return LedgerControls(
        max_provider_calls=60,
        max_attempts_per_call=1,
        expected_provider_calls=46,
        token_hard_cap=1_650_300,
        max_output_tokens=4096,
        cost_cap_usd="1.00",
        cost_cap_policy="refuse_before_dispatch",
        turn_deadline_seconds=30.0,
        wall_clock_deadline_seconds=1800.0,
        pricing=pricing(),
    )


def header(**overrides: Any) -> LedgerHeader:
    fields: dict[str, Any] = {
        "execution_run_id": "exec_" + "0" * 32,
        "operation_instance_id": "opinst_" + "1" * 32,
        "operation_id": "lettings.maintenance.v0_1",
        "spec_digest_sha256": DIGEST_A,
        "scenario_id": "V1",
        "agent_id": "openai-gpt-5-6-luna",
        "agent_kind": "model",
        "intended_artifact_version": INTENDED_ARTIFACT_VERSION,
        "started_at": INSTANT,
        "provider": provider_identity(),
        "controls": controls(),
    }
    fields.update(overrides)
    return LedgerHeader(**fields)


def attempt(**overrides: Any) -> LedgerAttempt:
    fields: dict[str, Any] = {
        "attempt_index": 1,
        "outcome": "response",
        "fault": None,
        "http_status": None,
        "latency_seconds": 0.5,
        "response_received": True,
        "usage_reported": True,
        "cost_reservation_usd": "0.0593425",
        "cost_settlement": "measured",
        "cost_measured_usd": "0.00644375",
        "cost_forfeited_usd": None,
        "input_tokens": 4211,
        "output_tokens": 118,
        "response_model": "gpt-5.6-luna",
        "response_id_digest_sha256": DIGEST_B,
        "response_stop_classification": "completed",
        "response_normalized_digest_sha256": DIGEST_C,
    }
    fields.update(overrides)
    return LedgerAttempt(**fields)


def call(index: int = 0, **overrides: Any) -> LedgerCall:
    fields: dict[str, Any] = {
        "call_index": index,
        "invocation_index": 1,
        "turn_index": index,
        "request_digest_sha256": DIGEST_A,
        "prompt_digest_sha256": DIGEST_B,
        "observation_digest_sha256": DIGEST_C,
        "payload_digest_sha256": DIGEST_D,
        "wire_request_digest_sha256": DIGEST_A,
        "wire_request_bytes": 13682,
        "wire_request_method": "POST",
        "wire_request_url": "https://api.openai.com/v1/responses",
        "input_token_upper_bound": 14706,
        "started_at": INSTANT,
        "ended_at": LATER,
        "turn_latency_seconds": 0.75,
        "terminal_reason": "response",
        "attempts": (attempt(),),
        "decision": LedgerDecision(
            kind="RETRIEVE", decision_digest_sha256=DIGEST_D, classification_code=None
        ),
    }
    fields.update(overrides)
    return LedgerCall(**fields)


def faulted_call(index: int = 0, *, status: int = 500) -> LedgerCall:
    return call(
        index,
        attempts=(
            attempt(
                outcome="fault",
                fault="provider_server_error",
                http_status=status,
                response_received=True,
                usage_reported=False,
                cost_settlement="forfeited",
                cost_measured_usd=None,
                cost_forfeited_usd="0.0593425",
                input_tokens=None,
                output_tokens=None,
                response_model=None,
                response_id_digest_sha256=None,
                response_stop_classification=None,
                response_normalized_digest_sha256=None,
            ),
        ),
        terminal_reason="retries_exhausted",
        decision=None,
    )


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rewrite_ledger_version(path: Path, version: int) -> Path:
    """Construct another internally consistent version identity for reader tests."""
    rows = rows_of(path)
    previous = GENESIS_DIGEST
    for row in rows:
        row["ledger_version"] = version
        row["previous_row_digest_sha256"] = previous
        row["row_digest_sha256"] = row_digest(row)
        previous = row["row_digest_sha256"]
    target = path.with_name(f"ledger-v{version}.ndjson")
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return target


def _redigest(row: dict[str, Any]) -> dict[str, Any]:
    row["row_digest_sha256"] = row_digest(row)
    return row


def _complete_invalid_final_row(row: dict[str, Any], defect: str) -> Any:
    if defect == "unknown_field":
        row["unexpected"] = None
        return _redigest(row)
    if defect == "missing_field":
        del row[ROW_TERMINAL]
        return _redigest(row)
    if defect == "non_object_row":
        return []
    if defect == "invalid_row_kind":
        row["note"] = row.pop(ROW_TERMINAL)
        row["row_kind"] = "note"
        return _redigest(row)
    if defect == "previous_digest_mismatch":
        row["previous_row_digest_sha256"] = "f" * 64
        return _redigest(row)
    if defect == "invalid_row_hash":
        row["row_digest_sha256"] = "f" * 64
        return row
    if defect == "terminal_totals_mismatch":
        row[ROW_TERMINAL]["totals"]["provider_calls"] += 1
        return _redigest(row)
    if defect == "version_mismatch":
        row["ledger_version"] = 2
        return row
    raise AssertionError(f"unknown test defect: {defect}")


# -- the journal shape -------------------------------------------------------


def test_the_journal_writes_one_header_row_before_any_call_is_made(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    try:
        rows = rows_of(path)
        assert len(rows) == 1
        assert rows[0]["row_kind"] == ROW_HEADER
        assert rows[0]["row_index"] == 0
        assert rows[0]["ledger_version"] == EXECUTION_LEDGER_VERSION
        assert rows[0]["previous_row_digest_sha256"] == GENESIS_DIGEST
    finally:
        journal.close()


def test_current_writer_emits_execution_ledger_v3(ledger_dir: Path) -> None:
    path = ledger_dir / "current.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call())
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)

    assert EXECUTION_LEDGER_VERSION == 3
    assert {row["ledger_version"] for row in rows_of(path)} == {3}


def test_historical_v2_identity_is_an_independent_literal() -> None:
    source = (REPO_ROOT / "src/operatebench/execution_ledger.py").read_text()
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert assignments["EXECUTION_LEDGER_VERSION_V2"] == 2
    assert assignments["EXECUTION_LEDGER_VERSION"] == 3


def test_current_v3_writer_refuses_http_status_without_response_arrival(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "contradictory-http-arrival.ndjson"
    faulted = faulted_call()
    contradictory = replace(
        faulted,
        attempts=(
            replace(
                faulted.attempts[0],
                fault="provider_request_rejected",
                http_status=400,
                response_received=False,
            ),
        ),
    )

    with ExecutionLedgerJournal.open(path, header()) as journal:
        with pytest.raises(LedgerSchemaError, match=r"HTTP status.*response_received"):
            journal.record_call(contradictory)
        assert len(rows_of(path)) == 1


def test_current_v3_reader_refuses_http_status_without_response_arrival() -> None:
    payload = faulted_call().attempts[0].as_dict()
    payload.update(
        fault="provider_request_rejected", http_status=400, response_received=False
    )

    with pytest.raises(LedgerSchemaError, match=r"HTTP status.*response_received"):
        LedgerAttempt.from_row(payload, "row 1.call.attempts[0]", ledger_version=3)


def test_historical_v2_preserves_http_status_without_response_arrival() -> None:
    payload = faulted_call().attempts[0].as_dict()
    payload.update(
        fault="provider_request_rejected", http_status=400, response_received=False
    )

    restored = LedgerAttempt.from_row(payload, "row 1.call.attempts[0]", ledger_version=2)
    assert restored.http_status == 400
    assert restored.response_received is False


def test_each_ledger_version_owns_literal_fault_and_retry_tables() -> None:
    source = (REPO_ROOT / "src/operatebench/execution_ledger.py").read_text()
    tree = ast.parse(source)
    expected = {
        "ATTEMPT_OUTCOMES_V2",
        "ATTEMPT_OUTCOMES_V3",
        "ATTEMPT_SETTLEMENTS_V2",
        "ATTEMPT_SETTLEMENTS_V3",
        "PROVIDER_FAULTS_V2",
        "RETRYABLE_PROVIDER_FAULTS_V2",
        "PROVIDER_FAULTS_V3",
        "RETRYABLE_PROVIDER_FAULTS_V3",
        "TURN_TERMINAL_REASONS_V2",
        "TURN_TERMINAL_REASONS_V3",
    }
    literal_names = {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in expected
        and isinstance(node.value, ast.Tuple)
        and all(
            isinstance(item, ast.Constant)
            or (
                isinstance(item, ast.Starred)
                and isinstance(item.value, ast.Tuple)
                and not item.value.elts
            )
            for item in node.value.elts
        )
    }
    assert literal_names == expected
    assert execution_ledger.PROVIDER_FAULTS_V2 is not execution_ledger.PROVIDER_FAULTS_V3
    assert (
        execution_ledger.RETRYABLE_PROVIDER_FAULTS_V2
        is not execution_ledger.RETRYABLE_PROVIDER_FAULTS_V3
    )
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "operatebench.providers.faults"
        for alias in node.names
    }
    assert "PROVIDER_FAULTS" not in imported
    assert "RETRYABLE_PROVIDER_FAULTS" not in imported
    assert "ATTEMPT_OUTCOMES" not in imported
    assert "ATTEMPT_SETTLEMENTS" not in imported
    assert "TURN_TERMINAL_REASONS" not in imported


def test_durable_readers_ignore_live_provider_taxonomy_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_faults, "PROVIDER_FAULTS", ("provider_future_fault",))
    monkeypatch.setattr(
        live_faults, "RETRYABLE_PROVIDER_FAULTS", ("provider_future_fault",)
    )
    monkeypatch.setattr(live_faults, "ATTEMPT_OUTCOMES", ("mutant",))
    monkeypatch.setattr(live_faults, "ATTEMPT_SETTLEMENTS", ("mutant",))
    monkeypatch.setattr(live_faults, "TURN_TERMINAL_REASONS", ("mutant",))
    historical = faulted_call().attempts[0].as_dict()
    current = dict(historical)
    current["fault"] = "provider_spend_limit"

    assert LedgerAttempt.from_row(historical, "v2", ledger_version=2).fault == (
        "provider_server_error"
    )
    assert LedgerAttempt.from_row(current, "v3", ledger_version=3).fault == (
        "provider_spend_limit"
    )
    with pytest.raises(LedgerSchemaError, match="version 2"):
        LedgerAttempt.from_row(current, "v2", ledger_version=2)


@pytest.mark.parametrize("ledger_version", [2, 3])
def test_call_terminal_reason_is_validated_by_its_ledger_version(
    ledger_version: int,
) -> None:
    provisional = replace(faulted_call(), terminal_reason="future_terminal_reason")

    with pytest.raises(
        LedgerSchemaError, match=rf"terminal_reason.*version {ledger_version}"
    ):
        provisional.validate_for_ledger_version(ledger_version, "call 0")


def test_current_provider_taxonomy_has_an_explicit_fail_closed_v3_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert execution_ledger.PROVIDER_FAULTS_V3 == live_faults.PROVIDER_FAULTS
    assert execution_ledger.PROVIDER_FAULTS_V3 is not live_faults.PROVIDER_FAULTS
    assert (
        execution_ledger.RETRYABLE_PROVIDER_FAULTS_V3
        == live_faults.RETRYABLE_PROVIDER_FAULTS
    )
    assert (
        execution_ledger.RETRYABLE_PROVIDER_FAULTS_V3
        is not live_faults.RETRYABLE_PROVIDER_FAULTS
    )
    execution_ledger.assert_current_provider_ledger_compatibility()
    for fault in live_faults.PROVIDER_FAULTS:
        payload = faulted_call().attempts[0].as_dict()
        payload.update(fault=fault, http_status=None, response_received=False)
        assert LedgerAttempt.from_row(payload, "current", ledger_version=3).fault == fault

    monkeypatch.setattr(
        live_faults,
        "PROVIDER_FAULTS",
        (*live_faults.PROVIDER_FAULTS, "provider_future_fault"),
    )
    with pytest.raises(LedgerSchemaError, match=r"taxonomy.*ledger version 3"):
        execution_ledger.assert_current_provider_ledger_compatibility()


def test_every_row_is_flushed_before_the_next_one_is_recorded(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        assert len(rows_of(path)) == 2
        journal.record_call(call(1, turn_index=1))
        assert len(rows_of(path)) == 3
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
        assert len(rows_of(path)) == 4
    kinds = [row["row_kind"] for row in rows_of(path)]
    assert kinds == [ROW_HEADER, ROW_CALL, ROW_CALL, ROW_TERMINAL]


def test_pre_dispatch_refusal_preserves_prior_attempts_and_reads_back(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "pre-dispatch-refusal.ndjson"
    retry_controls = replace(controls(), max_attempts_per_call=2)
    retry_header = header(controls=retry_controls)
    prior_attempt = attempt(
        outcome="fault",
        fault="provider_server_error",
        http_status=500,
        response_received=True,
        usage_reported=False,
        cost_settlement="forfeited",
        cost_measured_usd=None,
        cost_forfeited_usd="0.0593425",
        input_tokens=None,
        output_tokens=None,
        response_model=None,
        response_id_digest_sha256=None,
        response_stop_classification=None,
        response_normalized_digest_sha256=None,
    )
    refused = call(
        attempts=(prior_attempt,),
        terminal_reason=TURN_END_PRE_DISPATCH_REFUSED,
        decision=None,
    )

    with ExecutionLedgerJournal.open(path, retry_header) as journal:
        journal.record_call(refused)
        journal.finalize(
            TERMINAL_ABORTED,
            ended_at=LATER,
            exclusion_code=ABORTED_CODE,
        )

    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_ABORTED
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == ABORTED_CODE
    assert audit.calls[0].attempts == (prior_attempt,)
    assert audit.calls[0].terminal_reason == TURN_END_PRE_DISPATCH_REFUSED


def test_each_row_carries_the_previous_rows_digest_and_its_own(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    rows = rows_of(path)
    for index, row in enumerate(rows):
        assert row["row_index"] == index
        body = {k: v for k, v in row.items() if k != "row_digest_sha256"}
        assert row["row_digest_sha256"] == row_digest(body)
    assert rows[1]["previous_row_digest_sha256"] == rows[0]["row_digest_sha256"]
    assert rows[2]["previous_row_digest_sha256"] == rows[1]["row_digest_sha256"]


def test_the_row_digest_is_over_the_canonical_row_without_its_own_digest() -> None:
    body = {"row_index": 0, "row_kind": ROW_HEADER}
    import hashlib

    expected = hashlib.sha256(
        canonical_json_text(body, "execution ledger row").encode("utf-8")
    ).hexdigest()
    assert row_digest(body) == expected
    assert row_digest({**body, "row_digest_sha256": "x" * 64}) == expected


def test_an_execution_run_id_is_opaque_and_labelled() -> None:
    minted = {new_execution_run_id() for _ in range(8)}
    assert len(minted) == 8
    for value in minted:
        assert value.startswith("exec_")
        assert len(value) == len("exec_") + 32
        assert set(value[5:]) <= set("0123456789abcdef")


# -- totals ------------------------------------------------------------------


def test_the_terminal_row_states_totals_derived_from_the_call_rows(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1, turn_index=1))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    totals = rows_of(path)[-1]["terminal"]["totals"]
    assert totals["provider_calls"] == 2
    assert totals["attempts"] == 2
    assert totals["input_tokens"] == 8422
    assert totals["output_tokens"] == 236
    assert totals["measured_cost_usd"] == "0.0128875"
    assert totals["forfeited_reservation_usd"] == "0"
    assert totals["reserved_usd"] == "0.118685"
    assert totals["fault_counts"] == {}


def test_a_forfeited_reservation_is_not_summed_as_measured_spend() -> None:
    totals = recompute_totals((call(0), faulted_call(1)))
    assert totals.measured_cost_usd == "0.00644375"
    assert totals.forfeited_reservation_usd == "0.0593425"
    assert totals.reserved_usd == "0.118685"
    assert totals.fault_counts == {"provider_server_error": 1}
    assert totals.as_dict()["measured_cost_usd"] != totals.as_dict()["reserved_usd"]


@pytest.mark.parametrize(
    "fault",
    [
        "provider_bad_request_unclassified",
        "provider_invalid_request_or_spend_limit",
        "provider_spend_limit",
    ],
)
def test_terminal_totals_validate_fault_vocabulary_at_their_ledger_version(
    fault: str,
) -> None:
    payload = recompute_totals(()).as_dict()
    payload["fault_counts"] = {fault: 1}

    with pytest.raises(LedgerSchemaError, match=rf"version 2.*{fault}"):
        LedgerTotals.from_row(payload, "terminal.totals", ledger_version=2)

    assert LedgerTotals.from_row(
        payload, "terminal.totals", ledger_version=3
    ).fault_counts == {fault: 1}


def test_a_terminal_row_whose_totals_do_not_follow_from_its_calls_is_refused(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    tampered = _rewrite(
        path, 2, lambda row: _set(row, ["terminal", "totals", "input_tokens"], 1)
    )
    with pytest.raises(LedgerChainError):
        read_execution_ledger(tampered)


# -- reading -----------------------------------------------------------------


def test_a_complete_scored_ledger_reads_back_as_scored(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        for index in range(3):
            journal.record_call(call(index, turn_index=index))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    audit = read_execution_ledger(path)
    assert audit.status == TERMINAL_SCORED
    assert audit.scored is True
    assert audit.complete is True
    assert audit.truncated_tail is False
    assert len(audit.calls) == 3
    assert audit.totals.provider_calls == 3
    assert audit.header.provider.model == "gpt-5.6-luna"


def test_a_valid_historical_v2_ledger_with_an_old_fault_still_reads(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "historical-old-fault.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(faulted_call())
        journal.finalize(
            TERMINAL_EXCLUDED,
            ended_at=LATER,
            exclusion_code="provider_transport",
        )

    audit = read_execution_ledger(rewrite_ledger_version(path, 2), require_complete=True)
    assert audit.calls[0].attempts[0].fault == "provider_server_error"


@pytest.mark.parametrize(
    "fault",
    [
        "provider_bad_request_unclassified",
        "provider_invalid_request_or_spend_limit",
        "provider_spend_limit",
    ],
)
def test_historical_v2_refuses_each_v3_only_fault(ledger_dir: Path, fault: str) -> None:
    path = ledger_dir / f"current-{fault}.ndjson"
    faulted = faulted_call()
    current_call = replace(
        faulted,
        attempts=(replace(faulted.attempts[0], fault=fault),),
    )
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(current_call)
        journal.finalize(
            TERMINAL_EXCLUDED,
            ended_at=LATER,
            exclusion_code="provider_transport",
        )

    with pytest.raises(LedgerSchemaError, match="version 2"):
        read_execution_ledger(rewrite_ledger_version(path, 2), require_complete=True)


@pytest.mark.parametrize(
    "fault",
    [
        "provider_bad_request_unclassified",
        "provider_invalid_request_or_spend_limit",
        "provider_spend_limit",
    ],
)
def test_current_v3_reads_each_new_fault(ledger_dir: Path, fault: str) -> None:
    path = ledger_dir / f"v3-{fault}.ndjson"
    faulted = faulted_call()
    current_call = replace(
        faulted,
        attempts=(replace(faulted.attempts[0], fault=fault),),
    )
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(current_call)
        journal.finalize(
            TERMINAL_EXCLUDED,
            ended_at=LATER,
            exclusion_code="provider_transport",
        )

    audit = read_execution_ledger(path, require_complete=True)
    assert audit.calls[0].attempts[0].fault == fault


def test_a_complete_excluded_ledger_reads_back_as_excluded_and_never_scored(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(faulted_call(1))
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    audit = read_execution_ledger(path)
    assert audit.status == TERMINAL_EXCLUDED
    assert audit.scored is False
    assert audit.complete is True
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == "provider_transport"
    assert audit.totals.forfeited_reservation_usd == "0.0593425"
    assert audit.totals.measured_cost_usd == "0.00644375"


def test_an_excluded_ledger_carries_no_business_outcome_by_schema(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(faulted_call(0))
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    text = path.read_text()
    for forbidden in (
        "final_state",
        "terminal_outcome",
        "evaluation",
        "reliable",
        "failed_dimensions",
        "trajectory",
    ):
        assert forbidden not in text, forbidden


# -- crash safety ------------------------------------------------------------


CHILD = """
import os, sys
sys.path.insert(0, {root!r})
from operatebench.execution_ledger import ExecutionLedgerJournal
from tests.test_execution_ledger import call, header
journal = ExecutionLedgerJournal.open({path!r}, header())
journal.record_call(call(0))
journal.record_call(call(1, turn_index=1))
os._exit(9)
"""


def test_a_process_killed_mid_run_leaves_a_verifiable_prefix(ledger_dir: Path) -> None:
    """The real property, proven the only way it can be: kill the writer.

    Every row is flushed and fsynced before the call that follows it, so a
    process that dies outright — no finaliser, no interpreter shutdown, no
    buffers drained — still leaves the rows it had already written. What it must
    never leave is a journal that reads as a scored run.
    """
    path = ledger_dir / "killed.ndjson"
    finished = subprocess.run(
        [sys.executable, "-c", CHILD.format(root=str(REPO_ROOT), path=str(path))],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert finished.returncode == 9, finished.stderr.decode()
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert audit.scored is False
    assert audit.complete is False
    assert len(audit.calls) == 2
    assert audit.rows_verified == 3
    with pytest.raises(IncompleteExecutionLedgerError):
        read_execution_ledger(path, require_complete=True)


def test_a_journal_with_no_terminal_row_reads_as_incomplete_never_scored(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    journal.record_call(call(1, turn_index=1))
    # No finalize, no close: the process died here.
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert audit.scored is False
    assert audit.complete is False
    assert len(audit.calls) == 2
    assert audit.totals.provider_calls == 2


def test_a_truncated_final_line_keeps_the_verified_prefix(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1, turn_index=1))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    whole = path.read_text()
    path.write_text(whole[: len(whole) - 40])
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert audit.scored is False
    assert audit.truncated_tail is True
    assert len(audit.calls) == 2
    assert audit.totals.provider_calls == 2


def test_an_incomplete_utf8_final_fragment_keeps_the_verified_prefix(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    journal.close()
    path.write_bytes(path.read_bytes() + b'{"terminal":"\xe2')

    audit = read_execution_ledger(path)

    assert audit.status == "incomplete"
    assert audit.truncated_tail is True
    assert len(audit.calls) == 1
    with pytest.raises(IncompleteExecutionLedgerError) as caught:
        read_execution_ledger(path, require_complete=True)
    assert caught.value.audit.truncated_tail is True
    assert len(caught.value.audit.calls) == 1


def test_every_proper_byte_prefix_of_representative_v2_and_v3_rows_is_recoverable(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "representative.ndjson"
    unicode_header = header(
        provider=replace(
            provider_identity(),
            settings={**settings(), "unicode_label": "模型-é"},
        )
    )
    with ExecutionLedgerJournal.open(path, unicode_header) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)

    checked = 0
    for version in (2, 3):
        version_path = rewrite_ledger_version(path, version)
        rows = version_path.read_bytes().splitlines()
        assert len(rows) == 3
        for row in rows:
            for cutoff in range(1, len(row)):
                assert _is_recoverable_json_object_prefix(row[:cutoff]), (
                    version,
                    cutoff,
                    row[:cutoff],
                )
                checked += 1
    assert checked > 0


def test_every_proper_prefix_across_canonical_json_lexical_states_is_recoverable() -> (
    None
):
    row = (
        b'{"array": [true, false, null, -12, 0, 3.50e-2, '
        b'{"escaped": "quote\\" slash\\\\ newline\\n unicode\\u20ac"}]}'
    )

    for cutoff in range(1, len(row)):
        assert _is_recoverable_json_object_prefix(row[:cutoff]), (cutoff, row[:cutoff])


@pytest.mark.parametrize("version", [2, 3])
def test_every_proper_call_and_terminal_row_cutoff_recovers_only_before_terminal(
    ledger_dir: Path,
    version: int,
) -> None:
    path = ledger_dir / "cutoff-source.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    rows = rewrite_ledger_version(path, version).read_bytes().splitlines()
    candidate = ledger_dir / f"cutoff-v{version}.ndjson"

    checked = 0
    for row_index in (1, 2):
        verified_prefix = b"\n".join(rows[:row_index]) + b"\n"
        for cutoff in range(1, len(rows[row_index])):
            candidate.write_bytes(verified_prefix + rows[row_index][:cutoff])
            audit = read_execution_ledger(candidate)
            assert audit.complete is False
            assert audit.truncated_tail is True
            with pytest.raises(IncompleteExecutionLedgerError):
                read_execution_ledger(candidate, require_complete=True)
            checked += 1
    assert checked > 0


@pytest.mark.parametrize(
    "fragment",
    [
        b"}{",
        b"{}garbage",
        b"\xff",
        b"\x80",
        b"\xc0",
        b"\xe2(",
        b"\xed\xa0\x80",
        b"\xf0\x80\x80\x80",
        b'{"x":"\\q',
        b'{"x":"\\u12x',
        b'{"x":"\x01',
        b'{"x":"line\nmore',
        b'{"x":]',
        b'{"x" 1',
        b'{"x"::',
        b'{"x":1 "y"',
        b'{"x":1,,',
        b'{"x":01',
        b'{"x":1.e',
        b'{"x":truX',
        b'{"x":True',
        b"{}{}",
        b'{"x":1}}',
        b'{"x":1,}',
        b"   ",
        b'"scalar',
        b"[",
    ],
)
@pytest.mark.parametrize("require_complete", [False, True])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_irrecoverable_final_fragment_before_terminal_is_refused(
    ledger_dir: Path,
    fragment: bytes,
    require_complete: bool,
    trailing_newline: bool,
) -> None:
    path = ledger_dir / "malformed-tail.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    journal.close()
    path.write_bytes(path.read_bytes() + fragment + (b"\n" if trailing_newline else b""))

    with pytest.raises(LedgerSchemaError, match=r"not one (UTF-8 )?JSON object"):
        read_execution_ledger(path, require_complete=require_complete)


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        (
            b'{"private":"credential-secret-fragment-91c2\xff"}',
            "not one UTF-8 JSON object",
        ),
        (b'{"private":"credential-secret-fragment-91c2",}', "not one JSON object"),
    ],
)
def test_decode_refusals_do_not_copy_secret_shaped_parser_input_to_public_errors(
    ledger_dir: Path,
    capsys: pytest.CaptureFixture[str],
    fragment: bytes,
    message: str,
) -> None:
    path = ledger_dir / "private-malformed-tail.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    journal.close()
    path.write_bytes(path.read_bytes() + fragment + b"\n")

    with pytest.raises(LedgerSchemaError, match=message) as caught:
        read_execution_ledger(path)

    streams = capsys.readouterr()
    public = f"{caught.value}\n{streams.out}\n{streams.err}"
    assert str(caught.value) == f"row 2: {message}"
    assert caught.value.__cause__ is None
    assert "credential-secret-fragment-91c2" not in public


@pytest.mark.parametrize(
    "raw",
    [
        b'{"credential-secret-fragment-91c2":1,"credential-secret-fragment-91c2":2}',
        b'{"nested":{"credential-secret-fragment-91c2":1,'
        b'"credential-secret-fragment-91c2":2}}',
    ],
)
def test_duplicate_key_refusal_has_a_fixed_input_free_error_graph(raw: bytes) -> None:
    secret = "credential-secret-fragment-91c2"

    with pytest.raises(LedgerSchemaError) as caught:
        execution_ledger._decode(raw, "row 4")

    error = caught.value
    assert str(error) == "row 4: duplicate object key"
    assert repr(error.args) == "('row 4: duplicate object key',)"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    traceback = error.__traceback__
    while traceback is not None:
        if Path(traceback.tb_frame.f_code.co_filename).name == "execution_ledger.py":
            assert secret not in repr(traceback.tb_frame.f_locals)
            assert all(value is not raw for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_decode_keeps_unique_keys_unchanged() -> None:
    assert execution_ledger._decode(b'{"outer":{"left":1,"right":2}}', "row 4") == {
        "outer": {"left": 1, "right": 2}
    }


def test_a_terminal_row_missing_its_newline_is_not_accepted_as_terminal(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:-3])
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert audit.truncated_tail is True


@pytest.mark.parametrize(("first_version", "final_version"), [(2, 3), (3, 2)])
def test_complete_final_row_without_newline_cannot_change_ledger_version(
    ledger_dir: Path,
    first_version: int,
    final_version: int,
) -> None:
    path = ledger_dir / "mixed-version.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    homogeneous = rewrite_ledger_version(path, first_version)
    rows = rows_of(homogeneous)
    rows[-1]["ledger_version"] = final_version
    homogeneous.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows))

    with pytest.raises(
        LedgerSchemaError,
        match=(
            rf"states execution ledger version {final_version}, but the journal began "
            rf"as version {first_version}"
        ),
    ):
        read_execution_ledger(homogeneous)


@pytest.mark.parametrize(
    ("defect", "error", "message"),
    [
        ("unknown_field", LedgerSchemaError, "field\\(s\\).*not part"),
        ("missing_field", LedgerSchemaError, "missing required field\\(s\\)"),
        ("non_object_row", LedgerSchemaError, "ledger row is a JSON object"),
        ("invalid_row_kind", LedgerSchemaError, "row_kind:.*not a value"),
        ("previous_digest_mismatch", LedgerChainError, "does not back-link"),
        ("invalid_row_hash", LedgerChainError, "does not hash"),
        ("terminal_totals_mismatch", LedgerChainError, "totals do not follow"),
        ("version_mismatch", LedgerSchemaError, "cannot change schema identity"),
    ],
)
def test_complete_invalid_final_row_without_newline_is_refused(
    ledger_dir: Path,
    defect: str,
    error: type[Exception],
    message: str,
) -> None:
    path = ledger_dir / f"complete-invalid-{defect}.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    rows: list[Any] = rows_of(path)
    rows[-1] = _complete_invalid_final_row(rows[-1], defect)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows))

    with pytest.raises(error, match=message):
        read_execution_ledger(path)


def test_complete_duplicate_key_final_row_without_newline_is_refused(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "complete-duplicate-key.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    lines = path.read_text().splitlines()
    lines[-1] = lines[-1][:-1] + ', "row_index": 2}'
    path.write_text("\n".join(lines))

    with pytest.raises(LedgerSchemaError, match="duplicate object key"):
        read_execution_ledger(path)


def test_valid_complete_final_row_without_newline_is_processed(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "complete-without-newline.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))

    audit = read_execution_ledger(path, require_complete=True)

    assert audit.status == TERMINAL_SCORED
    assert audit.complete is True
    assert audit.truncated_tail is False
    assert audit.rows_verified == 3


@pytest.mark.parametrize("require_complete", [False, True])
@pytest.mark.parametrize("corruption", [b"}{", b"\xff", b'{"row_kind":', b"{}"])
@pytest.mark.parametrize("reader_api", ["path", "descriptor"])
def test_nonempty_bytes_after_terminal_are_refused(
    ledger_dir: Path,
    require_complete: bool,
    corruption: bytes,
    reader_api: str,
) -> None:
    path = ledger_dir / "closed-ledger.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    path.write_bytes(path.read_bytes() + corruption)

    if reader_api == "path":
        with pytest.raises(LedgerChainError, match="follows the terminal row"):
            read_execution_ledger(path, require_complete=require_complete)
        return

    directory_fd = os.open(ledger_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(LedgerChainError, match="follows the terminal row"):
            read_execution_ledger(
                path.name,
                require_complete=require_complete,
                dir_fd=directory_fd,
            )
    finally:
        os.close(directory_fd)


@pytest.mark.parametrize("require_complete", [False, True])
@pytest.mark.parametrize("corruption", [b"}{", b"\xff", b'{"row_kind":', b"{}"])
def test_nonempty_bytes_glued_to_a_terminal_without_newline_are_refused(
    ledger_dir: Path,
    require_complete: bool,
    corruption: bytes,
) -> None:
    path = ledger_dir / "closed-ledger-without-newline.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    path.write_bytes(path.read_bytes().removesuffix(b"\n") + corruption)

    with pytest.raises(LedgerSchemaError, match=r"not one (UTF-8 )?JSON object"):
        read_execution_ledger(path, require_complete=require_complete)


def test_reading_an_incomplete_journal_strictly_raises_and_carries_the_prefix(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    with pytest.raises(IncompleteExecutionLedgerError) as caught:
        read_execution_ledger(path, require_complete=True)
    assert caught.value.audit.status == "incomplete"
    assert len(caught.value.audit.calls) == 1


def test_an_aborted_terminal_is_written_and_read_as_aborted(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_ABORTED, ended_at=LATER, exclusion_code=ABORTED_CODE)
    audit = read_execution_ledger(path)
    assert audit.status == TERMINAL_ABORTED
    assert audit.scored is False
    assert audit.complete is True


# -- refusals ----------------------------------------------------------------


def _set(row: dict[str, Any], path: list[str], value: Any) -> dict[str, Any]:
    node: Any = row
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return row


def _rewrite(path: Path, index: int, edit: Any) -> Path:
    rows = rows_of(path)
    rows[index] = edit(rows[index])
    target = path.with_name("tampered.ndjson")
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return target


def test_an_edited_row_breaks_its_own_digest(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    tampered = _rewrite(path, 1, lambda row: _set(row, ["call", "wire_request_bytes"], 1))
    with pytest.raises(LedgerChainError):
        read_execution_ledger(tampered)


def test_a_row_re_digested_after_editing_still_breaks_the_chain(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1, turn_index=1))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)

    def edit(row: dict[str, Any]) -> dict[str, Any]:
        _set(row, ["call", "wire_request_bytes"], 1)
        body = {k: v for k, v in row.items() if k != "row_digest_sha256"}
        row["row_digest_sha256"] = row_digest(body)
        return row

    tampered = _rewrite(path, 1, edit)
    with pytest.raises(LedgerChainError):
        read_execution_ledger(tampered)


def test_an_out_of_order_row_index_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.record_call(call(1, turn_index=1))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    rows = rows_of(path)
    rows[1], rows[2] = rows[2], rows[1]
    target = path.with_name("swapped.ndjson")
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with pytest.raises(LedgerChainError):
        read_execution_ledger(target)


def test_a_duplicated_row_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    lines = path.read_text().splitlines(keepends=True)
    target = path.with_name("duplicated.ndjson")
    target.write_text("".join([lines[0], lines[1], lines[1], lines[2]]))
    with pytest.raises(LedgerChainError):
        read_execution_ledger(target)


def test_a_row_after_the_terminal_row_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    lines = path.read_text().splitlines(keepends=True)
    target = path.with_name("extra.ndjson")
    target.write_text("".join([*lines, lines[1]]))
    with pytest.raises(LedgerChainError):
        read_execution_ledger(target)


def test_an_unknown_row_field_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)

    def edit(row: dict[str, Any]) -> dict[str, Any]:
        row["call"]["provider_note"] = "the model said it was confident"
        body = {k: v for k, v in row.items() if k != "row_digest_sha256"}
        row["row_digest_sha256"] = row_digest(body)
        row["previous_row_digest_sha256"] = row["previous_row_digest_sha256"]
        return row

    tampered = _rewrite(path, 1, edit)
    with pytest.raises((LedgerSchemaError, LedgerChainError)):
        read_execution_ledger(tampered)


def test_a_duplicate_json_key_in_a_row_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    lines = path.read_text().splitlines()
    lines[1] = lines[1][:-1] + ', "row_index": 9}'
    target = path.with_name("duplicate-key.ndjson")
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises((LedgerSchemaError, LedgerChainError)):
        read_execution_ledger(target)


def test_an_unknown_row_kind_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    tampered = _rewrite(path, 1, lambda row: _set(row, ["row_kind"], "note"))
    with pytest.raises((LedgerSchemaError, LedgerChainError)):
        read_execution_ledger(tampered)


def test_a_call_index_that_is_not_contiguous_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        with pytest.raises(LedgerStateError):
            journal.record_call(call(7))
        journal.finalize(TERMINAL_ABORTED, ended_at=LATER, exclusion_code=ABORTED_CODE)


def test_a_ledger_version_this_build_does_not_write_is_refused(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    tampered = _rewrite(path, 0, lambda row: _set(row, ["ledger_version"], 4))
    with pytest.raises((LedgerSchemaError, LedgerChainError)):
        read_execution_ledger(tampered)


# -- schema-level refusals on the dataclasses --------------------------------


def test_a_decision_without_a_response_bearing_attempt_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        call(
            0,
            attempts=(
                attempt(
                    outcome="fault",
                    fault="provider_server_error",
                    http_status=500,
                    response_received=True,
                    usage_reported=False,
                    cost_settlement="forfeited",
                    cost_measured_usd=None,
                    cost_forfeited_usd="0.05",
                    input_tokens=None,
                    output_tokens=None,
                    response_model=None,
                    response_id_digest_sha256=None,
                    response_stop_classification=None,
                    response_normalized_digest_sha256=None,
                ),
            ),
        )


def test_an_attempt_that_reports_usage_without_counts_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        attempt(usage_reported=True, input_tokens=None, output_tokens=None)


def test_an_attempt_settling_as_measured_without_a_measured_cost_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        attempt(cost_settlement="measured", cost_measured_usd=None)


def test_an_attempt_cannot_be_both_measured_and_forfeited() -> None:
    with pytest.raises(LedgerSchemaError):
        attempt(cost_measured_usd="0.01", cost_forfeited_usd="0.02")


def test_a_fault_outside_the_ledger_version_vocabulary_is_refused() -> None:
    provisional = replace(
        faulted_call().attempts[0], fault="provider_transport", http_status=None
    )
    with pytest.raises(LedgerSchemaError):
        provisional.validate_for_ledger_version(3, "attempt")


def test_a_stop_classification_this_build_does_not_write_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        attempt(response_stop_classification="the model finished politely")


def test_an_attempt_with_no_response_may_not_carry_response_evidence() -> None:
    with pytest.raises(LedgerSchemaError):
        attempt(
            outcome="fault",
            fault="provider_timeout",
            response_received=False,
            usage_reported=False,
            cost_settlement="forfeited",
            cost_measured_usd=None,
            cost_forfeited_usd="0.05",
            input_tokens=None,
            output_tokens=None,
        )


def test_a_scored_terminal_may_not_carry_an_exclusion_code() -> None:
    from operatebench.execution_ledger import LedgerTerminal

    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_SCORED,
            ended_at=LATER,
            exclusion_code="provider_transport",
            totals=recompute_totals((call(0),)),
        )


def test_an_excluded_terminal_must_carry_an_exclusion_code() -> None:
    from operatebench.execution_ledger import LedgerTerminal

    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_EXCLUDED,
            ended_at=LATER,
            exclusion_code=None,
            totals=recompute_totals((call(0),)),
        )


def test_an_intended_artifact_version_other_than_seven_is_refused() -> None:
    with pytest.raises(LedgerSchemaError):
        header(intended_artifact_version=6)


def test_a_header_states_the_digest_of_the_exact_settings_mapping() -> None:
    built = header()
    from operatebench.agents.transport import content_digest

    assert built.provider.settings_digest_sha256 == content_digest(settings())
    other = provider_identity()
    changed = {**settings(), "tool_choice": "auto"}
    assert (
        LedgerProviderIdentity(
            provider=other.provider,
            api=other.api,
            model=other.model,
            base_url=other.base_url,
            implementation=other.implementation,
            request_mapping=other.request_mapping,
            protocol_version=other.protocol_version,
            sdk=other.sdk,
            sdk_version=other.sdk_version,
            settings=changed,
        ).settings_digest_sha256
        != built.provider.settings_digest_sha256
    )


# -- secrets -----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": {"Authorization": "Bearer token-value"}},
        {"authorization": "anything"},
        {"api_key": "value"},
        {"note": "OPENAI_API_KEY=abc"},
        {"nested": [{"x-api-key": "value"}]},
        {"credential": "sk-abcdefghijklmnop"},
    ],
)
def test_the_recursive_scanner_refuses_credential_material(payload: Any) -> None:
    with pytest.raises(LedgerSecretError):
        assert_no_secret_material(payload, "test payload")


def test_the_scanner_refuses_unbounded_text_that_could_carry_prose() -> None:
    with pytest.raises(LedgerSecretError):
        assert_no_secret_material({"detail": "x" * 4096}, "test payload")


def test_the_scanner_accepts_this_builds_own_ledger_rows(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    for row in rows_of(path):
        assert_no_secret_material(row, "ledger row")


def test_a_row_carrying_provider_prose_is_refused_at_write_time(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        with pytest.raises((LedgerSchemaError, LedgerSecretError, TypeError)):
            journal.record_call(
                call(0, terminal_reason="the provider says it is busy right now")
            )
        journal.finalize(TERMINAL_ABORTED, ended_at=LATER, exclusion_code=ABORTED_CODE)


# -- write-once, root-only path semantics ------------------------------------


def test_the_journal_file_is_created_private_and_the_directory_must_be_private(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.finalize(TERMINAL_ABORTED, ended_at=LATER, exclusion_code=ABORTED_CODE)
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_world_readable_directory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "open"
    directory.mkdir(mode=0o755)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(directory / "run.ndjson", header())


def test_an_existing_file_is_never_overwritten(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    path.write_text("")
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(path, header())


def test_a_symlinked_journal_path_is_refused(ledger_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.ndjson"
    link = ledger_dir / "run.ndjson"
    link.symlink_to(target)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(link, header())


def test_a_symlinked_ancestor_directory_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(link / "run.ndjson", header())


def test_recording_after_finalising_is_refused(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    with pytest.raises(LedgerStateError):
        journal.record_call(call(0))
    with pytest.raises(LedgerStateError):
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    journal.close()


def test_the_journal_is_fsynced_row_by_row(ledger_dir: Path, monkeypatch: Any) -> None:
    synced: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    assert len(synced) >= 3


# -- reserved amounts render exactly -----------------------------------------


def test_amounts_are_exact_decimal_strings_never_binary_floats(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    totals = rows_of(path)[-1]["terminal"]["totals"]
    for key in ("measured_cost_usd", "forfeited_reservation_usd", "reserved_usd"):
        assert isinstance(totals[key], str)
        Decimal(totals[key])
    attempt_row = rows_of(path)[1]["call"]["attempts"][0]
    assert isinstance(attempt_row["cost_reservation_usd"], str)
    assert isinstance(attempt_row["cost_measured_usd"], str)
