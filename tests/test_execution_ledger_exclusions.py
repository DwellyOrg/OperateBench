"""Canonical provider failures remain complete exclusions, never free-form codes."""

import pytest

from operatebench.execution_ledger import (
    ABORTED_CODE,
    EXCLUSION_CODES,
    TERMINAL_EXCLUDED,
    LedgerSchemaError,
    LedgerTerminal,
    recompute_totals,
)
from operatebench.providers.faults import PROVIDER_FAULTS
from tests.test_execution_ledger import INSTANT


@pytest.mark.parametrize("code", PROVIDER_FAULTS)
def test_canonical_provider_exclusion_round_trip(code):
    terminal = LedgerTerminal(
        kind=TERMINAL_EXCLUDED,
        ended_at=INSTANT,
        exclusion_code=code,
        exclusion_detail="",
        totals=recompute_totals(()),
    )
    assert LedgerTerminal.from_row(terminal.as_dict(), "terminal") == terminal


def test_exclusion_vocabulary_is_exact_and_unknown_codes_fail_closed():
    assert set(EXCLUSION_CODES) == set(PROVIDER_FAULTS) | {
        ABORTED_CODE,
        "provider_budget",
        "provider_deadline",
        "provider_protocol",
        "provider_transport",
    }
    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_EXCLUDED,
            ended_at=INSTANT,
            exclusion_code="provider_future_fault",
            exclusion_detail="",
            totals=recompute_totals(()),
        )
