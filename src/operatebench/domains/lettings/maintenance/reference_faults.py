# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Authored diagnostic expectations, not agent policy or reliability waivers.

V2 permanently fails customer transfer-notice dispatch. Durable ownership is
legitimate, but recovery and the notification obligation must fail. This
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
    expectation_id="maintenance.reference.V2.permanent-notification-fault.v1",
    terminal="transferred_to_human_ownership",
    failed_dimensions=("recovery", "obligations"),
    finding_codes=(
        "OBLIGATION_MESSAGE_NOT_ESTABLISHED",
        "REQUIRED_NOTIFICATION_UNDELIVERED",
    ),
    passing_dimensions=(
        "terminal_outcome",
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
