"""BoundaryBench schema-7 owns its historical provider semantics."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import boundarybench.adapter as adapter
from boundarybench import ledger
from boundarybench.ledger import LedgerError

HISTORICAL_SCHEMA_7_FAULTS = (
    "provider_authentication",
    "provider_configuration",
    "provider_network_error",
    "provider_rate_limited",
    "provider_request_rejected",
    "provider_response_invalid",
    "provider_server_error",
    "provider_timeout",
)
LIFECYCLE_V3_ONLY_FAULTS = (
    "provider_bad_request_unclassified",
    "provider_invalid_request_or_spend_limit",
    "provider_spend_limit",
)
HISTORICAL_SCHEMA_7_RETRYABLE_FAULTS = (
    "provider_network_error",
    "provider_rate_limited",
    "provider_server_error",
    "provider_timeout",
)
HISTORICAL_SCHEMA_7_PROVIDER_OUTCOMES = {
    "provider_authentication": "provider_auth_failure",
    "provider_configuration": "provider_configuration_failure",
    "provider_request_rejected": "provider_request_failure",
    "provider_rate_limited": "provider_rate_limited",
    "provider_server_error": "provider_failure",
    "provider_network_error": "provider_failure",
    "provider_timeout": "provider_failure",
    "provider_response_invalid": "provider_failure",
}
HISTORICAL_SCHEMA_7_ATTEMPT_OUTCOMES = (
    "fault",
    "response",
    "unclassified_error",
)
HISTORICAL_SCHEMA_7_SETTLEMENTS = ("forfeited", "measured")
HISTORICAL_SCHEMA_7_TERMINAL_REASONS = (
    "backoff_unaffordable",
    "cost_cap_exhausted",
    "deadline_exceeded",
    "fault_not_retryable",
    "pre_dispatch_refused",
    "response",
    "retries_exhausted",
    "unclassified_error",
)


def _attempt(fault: str) -> dict[str, object]:
    return {
        "index": 1,
        "outcome": "fault",
        "fault": fault,
        "http_status": 400,
        "latency_seconds": 0.5,
        "response_received": True,
        "usage_reported": False,
        "cost_reservation_usd": None,
        "cost_settlement": None,
    }


def test_schema_7_fault_vocabulary_is_exactly_the_historical_parent_set() -> None:
    assert ledger.LEDGER_SCHEMA_VERSION == 7
    assert ledger.PROVIDER_FAULTS_V7 == HISTORICAL_SCHEMA_7_FAULTS
    assert ledger.RETRYABLE_PROVIDER_FAULTS_V7 == HISTORICAL_SCHEMA_7_RETRYABLE_FAULTS
    assert ledger.ATTEMPT_OUTCOMES_V7 == HISTORICAL_SCHEMA_7_ATTEMPT_OUTCOMES
    assert ledger.ATTEMPT_SETTLEMENTS_V7 == HISTORICAL_SCHEMA_7_SETTLEMENTS
    assert ledger.TURN_TERMINAL_REASONS_V7 == HISTORICAL_SCHEMA_7_TERMINAL_REASONS
    assert ledger.PROVIDER_OUTCOMES_V7 == HISTORICAL_SCHEMA_7_PROVIDER_OUTCOMES


def test_schema_7_vocabularies_are_independent_literals_not_live_aliases() -> None:
    path = Path("src/boundarybench/ledger.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    expected = {
        "PROVIDER_FAULTS_V7": HISTORICAL_SCHEMA_7_FAULTS,
        "RETRYABLE_PROVIDER_FAULTS_V7": HISTORICAL_SCHEMA_7_RETRYABLE_FAULTS,
        "ATTEMPT_OUTCOMES_V7": HISTORICAL_SCHEMA_7_ATTEMPT_OUTCOMES,
        "ATTEMPT_SETTLEMENTS_V7": HISTORICAL_SCHEMA_7_SETTLEMENTS,
        "TURN_TERMINAL_REASONS_V7": HISTORICAL_SCHEMA_7_TERMINAL_REASONS,
    }
    literals = {}
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in expected
        ):
            assert isinstance(node.value, ast.Tuple)
            literals[node.target.id] = tuple(
                ast.literal_eval(item) for item in node.value.elts
            )
    assert literals == expected
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "boundarybench.adapter"
        for alias in node.names
    }
    durable_names = {
        "ATTEMPT_OUTCOME_FAULT",
        "ATTEMPT_OUTCOME_RESPONSE",
        "ATTEMPT_OUTCOME_UNCLASSIFIED",
        "ATTEMPT_OUTCOMES",
        "ATTEMPT_SETTLEMENT_FORFEITED",
        "ATTEMPT_SETTLEMENT_MEASURED",
        "ATTEMPT_SETTLEMENTS",
        "PROVIDER_FAULT_AUTHENTICATION",
        "PROVIDER_FAULT_CONFIGURATION",
        "PROVIDER_FAULT_NETWORK_ERROR",
        "PROVIDER_FAULT_RATE_LIMITED",
        "PROVIDER_FAULT_REQUEST_REJECTED",
        "PROVIDER_FAULT_RESPONSE_INVALID",
        "PROVIDER_FAULT_SERVER_ERROR",
        "PROVIDER_FAULT_TIMEOUT",
        "PROVIDER_FAULTS",
        "RETRYABLE_PROVIDER_FAULTS",
        "TURN_END_COST_CAP_EXHAUSTED",
        "TURN_END_DEADLINE_EXCEEDED",
        "TURN_END_EARLY_REASONS",
        "TURN_END_FAULT_NOT_RETRYABLE",
        "TURN_END_PRE_DISPATCH_REFUSED",
        "TURN_END_RESPONSE",
        "TURN_END_RETRIES_EXHAUSTED",
        "TURN_END_UNCLASSIFIED",
        "TURN_TERMINAL_REASONS",
    }
    assert not durable_names & imported


def test_schema_7_provider_failure_kinds_and_outcomes_are_historical_literals() -> None:
    assert {
        ledger.KIND_PROVIDER_AUTHENTICATION,
        ledger.KIND_PROVIDER_CONFIGURATION,
        ledger.KIND_PROVIDER_NETWORK_ERROR,
        ledger.KIND_PROVIDER_RATE_LIMITED,
        ledger.KIND_PROVIDER_REQUEST_REJECTED,
        ledger.KIND_PROVIDER_RESPONSE_INVALID,
        ledger.KIND_PROVIDER_SERVER_ERROR,
        ledger.KIND_PROVIDER_TIMEOUT,
    } == set(HISTORICAL_SCHEMA_7_FAULTS)
    for kind, outcome in HISTORICAL_SCHEMA_7_PROVIDER_OUTCOMES.items():
        assert ledger.classify_failure(phase="adapter_call", kind=kind)[0] == outcome


def test_schema_7_reader_semantics_ignore_adapter_vocabulary_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ATTEMPT_OUTCOMES",
        "ATTEMPT_SETTLEMENTS",
        "PROVIDER_FAULTS",
        "RETRYABLE_PROVIDER_FAULTS",
        "TURN_TERMINAL_REASONS",
        "TURN_END_EARLY_REASONS",
    ):
        monkeypatch.setattr(adapter, name, ("mutant",))
    monkeypatch.setattr(adapter, "PROVIDER_FAULT_SERVER_ERROR", "mutant")

    assert (
        ledger._provider_attempt(_attempt("provider_server_error"), 0, "row 0", None)[
            "fault"
        ]
        == "provider_server_error"
    )
    with pytest.raises(LedgerError, match="unknown provider fault"):
        ledger._provider_attempt(_attempt("mutant"), 0, "row 0", None)
    assert (
        ledger._permitted_terminal_reasons(
            [{"outcome": "fault", "fault": "provider_server_error"}], 2
        )
        == ledger.TURN_END_EARLY_REASONS_V7
    )


@pytest.mark.parametrize("fault", LIFECYCLE_V3_ONLY_FAULTS)
def test_schema_7_rejects_every_lifecycle_v3_only_fault(fault: str) -> None:
    with pytest.raises(LedgerError, match="unknown provider fault") as caught:
        ledger._provider_attempt(_attempt(fault), 0, "row 0", None)
    assert fault in str(caught.value)
    assert str(list(HISTORICAL_SCHEMA_7_FAULTS)) in str(caught.value)


@pytest.mark.parametrize("fault", HISTORICAL_SCHEMA_7_FAULTS)
def test_schema_7_still_accepts_every_historical_fault(fault: str) -> None:
    assert ledger._provider_attempt(_attempt(fault), 0, "row 0", None)["fault"] == fault
