# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Pack-owned state for the invented certificate renewal."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComplianceState:
    phase: str = "due"
    due_noticed: bool = False
    access_authoritative: bool = False
    inspection_clear: bool = False
    defect_observed: bool = False
    defect_open: bool = False
    remediation_approved: bool = False
    repair_evidence: bool = False
    reinspection_clear: bool = False
    certificate_issued: bool = False
    registry_valid: bool = False
    registry_rejected: bool = False
    access_attempted: bool = False
    access_reminded: bool = False
    expiry_reached: bool = False
    final: bool = False
    terminal: str | None = None
    audit: list[dict[str, Any]] = field(default_factory=list)

    def canonical(self) -> dict[str, Any]:
        return {
            name: ([dict(row) for row in value] if name == "audit" else value)
            for name, value in self.__dict__.items()
        }

    def operational_canonical(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if name != "audit"}
