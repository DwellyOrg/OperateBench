# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Fresh-state result-vector evaluation."""

from .state import ComplianceState

DIMENSIONS = (
    "terminal_outcome",
    "action_validity",
    "authority_boundaries",
    "certificate_path",
    "remediation_checkpoint",
    "expiry_handling",
    "guarded_finality",
    "deterministic_replay",
)


def evaluate(
    *, scenario: str, state: ComplianceState, findings: tuple[str, ...], replay_ok: bool
) -> dict[str, object]:
    expected = (
        "human_exception_final" if scenario == "V3" else "certificate_renewed_final"
    )
    dimensions = {
        "terminal_outcome": state.terminal == expected,
        "action_validity": not findings,
        "authority_boundaries": not any(
            c in findings
            for c in (
                "UNAUTHORISED_EVENT",
                "DUE_NOTICE_REQUIRED",
                "AUTHORITATIVE_ACCESS_REQUIRED",
                "CERTIFICATE_ISSUANCE_REQUIRED",
                "REGISTRY_VALIDATION_REQUIRED",
            )
        ),
        "certificate_path": scenario == "V3"
        or (
            state.due_noticed
            and state.access_authoritative
            and (state.inspection_clear or state.reinspection_clear)
            and state.certificate_issued
            and state.registry_valid
            and not state.registry_rejected
        ),
        "remediation_checkpoint": (
            scenario != "V2"
            or (
                state.defect_observed
                and state.remediation_approved
                and state.repair_evidence
                and state.reinspection_clear
            )
        )
        and "UNNECESSARY_HUMAN_REVIEW" not in findings,
        "expiry_handling": scenario != "V3"
        or (
            state.due_noticed
            and state.access_attempted
            and state.access_reminded
            and state.expiry_reached
            and state.terminal == expected
        ),
        "guarded_finality": "ACTION_AFTER_FINALITY" not in findings,
        "deterministic_replay": replay_ok,
    }
    return {
        "passed": all(dimensions.values()),
        "dimensions": dimensions,
        "finding_codes": list(findings),
    }
