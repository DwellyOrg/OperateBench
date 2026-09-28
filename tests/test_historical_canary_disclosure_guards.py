# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Frozen read disclosures must never hide failure or successor recovery."""

from copy import deepcopy

import pytest

from operatebench.domains.lettings.maintenance import operation
from tests.historical_canary_controls import historical_canary_gate_build  # noqa: F401
from tools import run_lifecycle_v1_canary as canary


@pytest.mark.usefixtures("historical_canary_gate_build")
def test_historical_projection_leaves_delivered_evidence_unchanged():
    assert canary.OPERATEBENCH_VERSION == "0.12.0"
    record = {"communications": [{"dispatch_status": "DELIVERED"}]}
    before = deepcopy(record)
    assert operation.serve_maintenance_retrieval(record, [], "2026-01-01T00:00:00Z") == ()
    assert record == before


@pytest.mark.usefixtures("historical_canary_gate_build")
@pytest.mark.parametrize(
    "message",
    [
        {"dispatch_status": "FAILED"},
        {"dispatch_status": "DELIVERED", "recovery_of": 0},
    ],
)
def test_historical_projection_refuses_failure_or_recovery_without_mutation(message):
    record = {"communications": [message]}
    before = deepcopy(record)
    with pytest.raises(AssertionError):
        operation.serve_maintenance_retrieval(record, [], "2026-01-01T00:00:00Z")
    assert record == before


@pytest.mark.usefixtures("historical_canary_gate_build")
def test_historical_schema_refuses_successor_recovery():
    with pytest.raises(AssertionError):
        operation.action_schema_view(recovery_enabled=True)
    legacy = operation.action_schema_view(recovery_enabled=False)
    assert "recovery_of" not in legacy["send_message"]["optional"]
