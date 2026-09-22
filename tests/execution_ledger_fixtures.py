"""Shared builders for the adversarial execution-ledger suites.

The rows these build are the ones a real Lifecycle run writes, with real
arithmetic in them: the reservation is the run's pinned rates applied to the
call's own input bound and the output ceiling, and the measured cost is those
rates applied to the counts the provider reported. That matters here more than
it usually would — the accounting suite exists to prove a reader *derives* those
amounts rather than reading them, and a fixture carrying invented numbers could
not tell the difference.

``rechained`` is the attacker's tool. Editing one row and re-digesting it is not
an attack this build has ever been vulnerable to; editing a row, re-digesting it
and re-linking every row after it is, and it is what the accounting and state
suites use, because it is the only edit that leaves the chain itself intact.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from operatebench.core.clock import format_timestamp
from operatebench.execution_ledger import (
    GENESIS_DIGEST,
    INTENDED_ARTIFACT_VERSION,
    LedgerAttempt,
    LedgerCall,
    LedgerControls,
    LedgerDecision,
    LedgerHeader,
    LedgerProviderIdentity,
    row_digest,
)

INSTANT = format_timestamp(1_900_000_000)
LATER = format_timestamp(1_900_000_060)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64

#: The run's pinned rates, the output ceiling and the input bound every call in
#: these fixtures was authorised against. Every amount below is these numbers.
INPUT_RATE = Decimal("1.25")
OUTPUT_RATE = Decimal("10.00")
MAX_OUTPUT_TOKENS = 4096
INPUT_TOKEN_UPPER_BOUND = 14_706
REPORTED_INPUT_TOKENS = 4211
REPORTED_OUTPUT_TOKENS = 118

#: (14706 * 1.25 + 4096 * 10) / 1e6
RESERVATION_USD = "0.0593425"
#: (4211 * 1.25 + 118 * 10) / 1e6
MEASURED_USD = "0.00644375"


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


def pricing(**overrides: Any) -> dict[str, Any]:
    table: dict[str, Any] = {
        "policy_id": "operator_pinned_v1",
        "input_usd_per_mtok": "1.25",
        "output_usd_per_mtok": "10",
        "rate_source": "operator_supplied_pinned_rates",
    }
    table.update(overrides)
    return table


def controls(**overrides: Any) -> LedgerControls:
    fields: dict[str, Any] = {
        "max_provider_calls": 60,
        "max_attempts_per_call": 1,
        "expected_provider_calls": 46,
        "token_hard_cap": 1_650_300,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "cost_cap_usd": "1.00",
        "cost_cap_policy": "refuse_before_dispatch",
        "turn_deadline_seconds": 30.0,
        "wall_clock_deadline_seconds": 1800.0,
        "pricing": pricing(),
    }
    fields.update(overrides)
    return LedgerControls(**fields)


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


def answered_attempt(**overrides: Any) -> LedgerAttempt:
    """One attempt that received an answer this build could read and price."""
    fields: dict[str, Any] = {
        "attempt_index": 1,
        "outcome": "response",
        "fault": None,
        "http_status": None,
        "latency_seconds": 0.5,
        "response_received": True,
        "usage_reported": True,
        "cost_reservation_usd": RESERVATION_USD,
        "cost_settlement": "measured",
        "cost_measured_usd": MEASURED_USD,
        "cost_forfeited_usd": None,
        "input_tokens": REPORTED_INPUT_TOKENS,
        "output_tokens": REPORTED_OUTPUT_TOKENS,
        "response_model": "gpt-5.6-luna",
        "response_id_digest_sha256": DIGEST_B,
        "response_stop_classification": "completed",
        "response_normalized_digest_sha256": DIGEST_C,
    }
    fields.update(overrides)
    return LedgerAttempt(**fields)


def faulted_attempt(**overrides: Any) -> LedgerAttempt:
    """One attempt that met a fault, forfeiting the whole reservation."""
    fields: dict[str, Any] = {
        "attempt_index": 1,
        "outcome": "fault",
        "fault": "provider_server_error",
        "http_status": 500,
        "latency_seconds": 0.5,
        "response_received": True,
        "usage_reported": False,
        "cost_reservation_usd": RESERVATION_USD,
        "cost_settlement": "forfeited",
        "cost_measured_usd": None,
        "cost_forfeited_usd": RESERVATION_USD,
        "input_tokens": None,
        "output_tokens": None,
        "response_model": None,
        "response_id_digest_sha256": None,
        "response_stop_classification": None,
        "response_normalized_digest_sha256": None,
    }
    fields.update(overrides)
    return LedgerAttempt(**fields)


def decision(**overrides: Any) -> LedgerDecision:
    fields: dict[str, Any] = {
        "kind": "RETRIEVE",
        "decision_digest_sha256": DIGEST_D,
        "classification_code": None,
    }
    fields.update(overrides)
    return LedgerDecision(**fields)


def call(index: int = 0, **overrides: Any) -> LedgerCall:
    """One answered, decision-bearing call: what a scored run is made of."""
    fields: dict[str, Any] = {
        "call_index": index,
        "invocation_index": 1,
        "turn_index": index,
        "request_digest_sha256": DIGEST_A,
        "prompt_digest_sha256": DIGEST_B,
        "observation_digest_sha256": DIGEST_C,
        "payload_digest_sha256": DIGEST_D,
        "wire_request_digest_sha256": DIGEST_A,
        "wire_request_bytes": 13_682,
        "wire_request_method": "POST",
        "wire_request_url": "https://api.openai.com/v1/responses",
        "input_token_upper_bound": INPUT_TOKEN_UPPER_BOUND,
        "started_at": INSTANT,
        "ended_at": LATER,
        "turn_latency_seconds": 0.75,
        "terminal_reason": "response",
        "attempts": (answered_attempt(),),
        "decision": decision(),
    }
    fields.update(overrides)
    return LedgerCall(**fields)


def faulted_call(index: int = 0, **overrides: Any) -> LedgerCall:
    """One call that faulted: no decision, and the reason the turn stopped."""
    fields: dict[str, Any] = {
        "attempts": (faulted_attempt(),),
        "decision": None,
        "terminal_reason": "retries_exhausted",
    }
    fields.update(overrides)
    return call(index, **fields)


# -- reading and re-chaining a written journal --------------------------------


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rechained(rows: list[dict[str, Any]], target: Path) -> Path:
    """Write these rows as a journal whose chain verifies end to end.

    Every row is re-digested and re-linked, so the result is exactly what an
    attacker who understands the format produces: an edit no digest, back-link
    or index check can see. What is left to catch it is whether the numbers and
    the states in the rows still follow from each other.
    """
    previous = GENESIS_DIGEST
    written: list[str] = []
    for index, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != "row_digest_sha256"}
        body["row_index"] = index
        body["previous_row_digest_sha256"] = previous
        previous = row_digest(body)
        written.append(
            json.dumps({**body, "row_digest_sha256": previous}, sort_keys=True) + "\n"
        )
    target.write_text("".join(written))
    return target


def rechain_ledger_file(path: Path, mutate: Any) -> Path:
    """Read a written journal, apply one edit to its rows, and re-chain it.

    The convenience form of :func:`rechained` for suites that run a real
    episode first: the journal on disk is the one this build wrote, so the edit
    is applied to real rows rather than to a fixture that resembles them.
    """
    rows = rows_of(path)
    mutate(rows)
    return rechained(rows, path)


def call_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["row_kind"] == "call"]


def terminal_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["row_kind"] == "terminal")
