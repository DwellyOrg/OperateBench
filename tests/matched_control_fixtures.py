"""Test-only current matched-control grammar for validator/decision-point tests.

This module never writes a manifest and does not define a benchmark control. Its
payload exists only in memory to exercise current validators and decision-point
machinery after the historical shipped draft became superseded.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from operatebench.decision_points import action_read_requirement_surface
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
)


def test_only_current_validator_payload(
    historical_payload: Mapping[str, Any], spec: Any
) -> dict[str, Any]:
    """Derive neutral current grammar solely for in-memory mutation tests.

    The result is not a matched benchmark control, is not evidence, is not
    shipped under ``examples/`` or the package, and must never be counted as a
    matched control.
    """
    current = copy.deepcopy(dict(historical_payload))
    current["control_id"] = "test_only_current_validator_grammar"
    current["control_version"] = "test-only"
    current["operation_id"] = spec.operation_id
    current["operation_type"] = spec.operation_type
    current["authored_against_operation_version"] = spec.operation_version
    current["semantic_scenario_id"] = spec.semantic_scenario_id
    current["matched_surface"]["action_read_requirements"] = list(
        action_read_requirement_surface()
    )
    current["matched_surface"]["action_schemas"] = {
        action_type: {
            "required": dict(required),
            "optional": dict(optional),
        }
        for action_type, (
            required,
            optional,
        ) in MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.items()
    }
    exception_outcome = next(
        outcome
        for point in current["decision_points"]
        for outcome in point["admissible_outcomes"]
        if outcome["action_type"] == "request_exception_resolution"
    )
    exception_outcome["outcome_constraints"]["approval_checkpoint_id"] = {
        "kind": "bound_field",
        "binding": "lapsed_approval",
        "field": "checkpoint_id",
    }
    return current
