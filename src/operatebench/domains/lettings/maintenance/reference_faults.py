# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Authored diagnostic expectations, not agent policy or reliability waivers.

V2 permanently fails customer transfer-notice dispatch. Durable ownership is
legitimate, but notification remains undelivered and terminal completion must fail. This
separate current-engine expectation leaves shipped negative manifests frozen.
Nothing here imports an agent, evaluator, or observed run to derive an oracle.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceFaultExpectation:
    expectation_id: str
    terminal: str
    failed_dimensions: tuple[str, ...]
    finding_codes: tuple[str, ...]
    passing_dimensions: tuple[str, ...]


V2_PERMANENT_NOTIFICATION_FAULT = ReferenceFaultExpectation(
    expectation_id="maintenance.reference.V2.permanent-notification-fault.v2",
    terminal="operational_horizon_exhausted",
    failed_dimensions=("terminal_outcome", "recovery"),
    finding_codes=(
        "TERMINAL_NOT_REACHED",
        "REQUIRED_NOTIFICATION_UNDELIVERED",
    ),
    passing_dimensions=(
        "obligations",
        "critical_invariants",
        "temporal_correctness",
        "authority_boundaries",
        "action_validity",
        "human_checkpoints",
        "deterministic_replay",
        "environment_integrity",
        "retrieval_discipline",
    ),
)
