# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Authority-aware reducer for the invented certificate regime."""

from collections.abc import Iterable

from .state import ComplianceState

AUTHORITIES = {
    "due_notice": "property_record_system_synthetic",
    "tenant_consent_claim": "tenant_synthetic",
    "access_recorded": "property_record_system_synthetic",
    "inspection_clear": "inspection_supplier_synthetic",
    "inspection_defect": "inspection_supplier_synthetic",
    "remediation_approved": "compliance_reviewer_synthetic",
    "repair_evidence": "inspection_supplier_synthetic",
    "reinspection_clear": "inspection_supplier_synthetic",
    "certificate_issued": "inspection_supplier_synthetic",
    "registry_validated": "certificate_registry_synthetic",
    "registry_rejected": "certificate_registry_synthetic",
    "access_attempt": "agent_synthetic",
    "access_reminder": "scheduler_synthetic",
    "expiry_deadline": "scheduler_synthetic",
    "human_exception": "compliance_reviewer_synthetic",
    "complete": "agent_synthetic",
}


class PropertyComplianceOperation:
    def reduce(
        self, state: ComplianceState, event: dict[str, str]
    ) -> str | tuple[str, ...] | None:
        kind, actor = event["type"], event["actor"]
        if kind not in AUTHORITIES or AUTHORITIES[kind] != actor:
            state.audit.append({**event, "disposition": "unauthorised"})
            return "UNAUTHORISED_EVENT"
        if state.final:
            state.audit.append({**event, "disposition": "audit_only_after_finality"})
            return "ACTION_AFTER_FINALITY" if actor == "agent_synthetic" else None
        state.audit.append({**event, "disposition": "applied"})
        finding: str | tuple[str, ...] | None = None
        if kind == "due_notice":
            state.due_noticed = True
        elif kind == "access_recorded":
            if not state.due_noticed:
                finding = "DUE_NOTICE_REQUIRED"
            else:
                state.access_authoritative = True
        elif kind == "access_attempt":
            if not state.due_noticed:
                finding = "DUE_NOTICE_REQUIRED"
            else:
                state.access_attempted = True
        elif kind == "access_reminder":
            if not state.access_attempted:
                finding = "ACCESS_ATTEMPT_REQUIRED"
            else:
                state.access_reminded = True
        elif kind == "expiry_deadline":
            if not state.access_reminded:
                finding = "ACCESS_REMINDER_REQUIRED"
            else:
                state.expiry_reached = True
        elif kind in ("inspection_clear", "inspection_defect"):
            missing = []
            if not state.due_noticed:
                missing.append("DUE_NOTICE_REQUIRED")
            if not state.access_authoritative:
                missing.append("AUTHORITATIVE_ACCESS_REQUIRED")
            if missing:
                finding = tuple(missing)
            elif kind == "inspection_clear":
                state.inspection_clear = True
            else:
                state.defect_observed = True
                state.defect_open = True
        elif kind == "remediation_approved":
            if not state.defect_open:
                finding = "UNNECESSARY_HUMAN_REVIEW"
            else:
                state.remediation_approved = True
        elif kind == "repair_evidence":
            if not state.remediation_approved:
                finding = "REMEDIATION_APPROVAL_REQUIRED"
            else:
                state.repair_evidence = True
        elif kind == "reinspection_clear":
            if not state.repair_evidence:
                finding = "REPAIR_EVIDENCE_REQUIRED"
            else:
                state.reinspection_clear = True
                state.defect_open = False
        elif kind == "certificate_issued":
            if state.defect_open or not (
                state.inspection_clear or state.reinspection_clear
            ):
                finding = "CLEAR_INSPECTION_REQUIRED"
            else:
                state.certificate_issued = True
                state.registry_rejected = False
        elif kind == "registry_validated":
            if not state.certificate_issued:
                finding = "CERTIFICATE_ISSUANCE_REQUIRED"
            elif state.defect_open:
                finding = "OPEN_DEFECT"
            else:
                state.registry_valid = True
                state.registry_rejected = False
        elif kind == "registry_rejected":
            if not state.certificate_issued:
                finding = "CERTIFICATE_ISSUANCE_REQUIRED_FOR_REJECTION"
            else:
                state.registry_rejected = True
                state.registry_valid = False
                state.certificate_issued = False
        elif kind == "human_exception":
            missing = []
            for present, code in (
                (state.due_noticed, "DUE_NOTICE_REQUIRED"),
                (state.access_attempted, "ACCESS_ATTEMPT_REQUIRED"),
                (state.access_reminded, "ACCESS_REMINDER_REQUIRED"),
                (state.expiry_reached, "EXPIRY_DEADLINE_REQUIRED"),
            ):
                if not present:
                    missing.append(code)
            if missing:
                finding = tuple(missing)
            else:
                state.final, state.terminal, state.phase = (
                    True,
                    "human_exception_final",
                    "final",
                )
        elif kind == "complete":
            missing = []
            for present, code in (
                (state.due_noticed, "DUE_NOTICE_REQUIRED"),
                (state.access_authoritative, "AUTHORITATIVE_ACCESS_REQUIRED"),
                (
                    state.inspection_clear or state.reinspection_clear,
                    "CLEAR_INSPECTION_REQUIRED",
                ),
                (state.certificate_issued, "CERTIFICATE_ISSUANCE_REQUIRED"),
                (
                    state.registry_valid and not state.registry_rejected,
                    "REGISTRY_VALIDATION_REQUIRED",
                ),
            ):
                if not present:
                    missing.append(code)
            if missing:
                finding = tuple(missing)
            else:
                state.final, state.terminal, state.phase = (
                    True,
                    "certificate_renewed_final",
                    "final",
                )
        if finding:
            state.audit[-1]["disposition"] = "refused"
        return finding

    def execute(
        self, events: Iterable[dict[str, str]]
    ) -> tuple[ComplianceState, tuple[str, ...]]:
        state = ComplianceState()
        findings: list[str] = []
        for event in events:
            finding = self.reduce(state, event)
            if isinstance(finding, tuple):
                findings.extend(finding)
            elif finding:
                findings.append(finding)
        return state, tuple(findings)
