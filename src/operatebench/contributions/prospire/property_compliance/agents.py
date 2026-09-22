# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Fixed reference paths and narrow negative interventions."""

from types import MappingProxyType

REFERENCE_AGENT_ID = "reference"
NEGATIVE_AGENT_IDS = (
    "trust_supplier_certificate_claim",
    "complete_before_registry_validation",
    "skip_remediation_approval",
    "ignore_certificate_rejection",
    "wait_past_expiry_deadline",
    "unnecessary_human_review",
    "act_after_finality",
)
NEGATIVE_AGENT_SCENARIOS = MappingProxyType(
    {
        "trust_supplier_certificate_claim": "V1",
        "complete_before_registry_validation": "V1",
        "skip_remediation_approval": "V2",
        "ignore_certificate_rejection": "V1",
        "wait_past_expiry_deadline": "V3",
        "unnecessary_human_review": "V1",
        "act_after_finality": "V1",
    }
)


def _e(kind: str, actor: str) -> dict[str, str]:
    return {"type": kind, "actor": actor}


def reference_events(scenario: str) -> list[dict[str, str]]:
    prefix = [_e("due_notice", "property_record_system_synthetic")]
    if scenario == "V3":
        return [
            *prefix,
            _e("access_attempt", "agent_synthetic"),
            _e("access_reminder", "scheduler_synthetic"),
            _e("expiry_deadline", "scheduler_synthetic"),
            _e("human_exception", "compliance_reviewer_synthetic"),
        ]
    access = [
        _e("tenant_consent_claim", "tenant_synthetic"),
        _e("access_recorded", "property_record_system_synthetic"),
    ]
    if scenario == "V1":
        return (
            prefix
            + access
            + [
                _e("inspection_clear", "inspection_supplier_synthetic"),
                _e("certificate_issued", "inspection_supplier_synthetic"),
                _e("registry_validated", "certificate_registry_synthetic"),
                _e("complete", "agent_synthetic"),
            ]
        )
    return (
        prefix
        + access
        + [
            _e("inspection_defect", "inspection_supplier_synthetic"),
            _e("remediation_approved", "compliance_reviewer_synthetic"),
            _e("repair_evidence", "inspection_supplier_synthetic"),
            _e("reinspection_clear", "inspection_supplier_synthetic"),
            _e("certificate_issued", "inspection_supplier_synthetic"),
            _e("registry_validated", "certificate_registry_synthetic"),
            _e("complete", "agent_synthetic"),
        ]
    )


def events_for(agent_id: str, scenario: str) -> list[dict[str, str]]:
    if agent_id == REFERENCE_AGENT_ID:
        return reference_events(scenario)
    if agent_id == "trust_supplier_certificate_claim":
        return [e for e in reference_events("V1") if e["type"] != "registry_validated"]
    if agent_id == "complete_before_registry_validation":
        result = reference_events("V1")
        result[-2:] = [result[-1], result[-2]]
        return result
    if agent_id == "skip_remediation_approval":
        return [e for e in reference_events("V2") if e["type"] != "remediation_approved"]
    if agent_id == "ignore_certificate_rejection":
        result = reference_events("V1")
        result.insert(-1, _e("registry_rejected", "certificate_registry_synthetic"))
        return result
    if agent_id == "wait_past_expiry_deadline":
        return reference_events("V3")[:-1]
    if agent_id == "unnecessary_human_review":
        result = reference_events("V1")
        result.insert(3, _e("remediation_approved", "compliance_reviewer_synthetic"))
        return result
    if agent_id == "act_after_finality":
        return [
            *reference_events("V1"),
            _e("complete", "agent_synthetic"),
        ]
    raise ValueError(f"unknown property-compliance agent {agent_id!r}")
