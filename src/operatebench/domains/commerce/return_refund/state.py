"""Mutable record for the synthetic commerce return/refund operation.

The record deliberately uses only domain-neutral Python containers.  Claims,
authoritative records, human checkpoints, money movement and obligations live
in separate registries so a retrieval result cannot silently upgrade one kind
of evidence into another.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

PHASE_NEW = "NEW"
PHASE_RETURN_REQUESTED = "RETURN_REQUESTED"
PHASE_AWAITING_HANDOVER = "AWAITING_HANDOVER"
PHASE_AWAITING_INSPECTION = "AWAITING_INSPECTION"
PHASE_AWAITING_APPROVAL = "AWAITING_APPROVAL"
PHASE_READY_TO_REFUND = "READY_TO_REFUND"
PHASE_REFUND_PENDING = "REFUND_PENDING"
PHASE_REFUND_SETTLED = "REFUND_SETTLED"
PHASE_PROVISIONAL = "PROVISIONAL"
PHASE_REFUND_REOPENED = "REFUND_REOPENED"
PHASE_REFUND_DENIED = "REFUND_DENIED"
PHASE_RETURN_EXPIRED = "RETURN_EXPIRED"
PHASE_FINAL_SUCCESS = "FINAL_SUCCESS"
PHASE_FINAL_REFUND_DENIED = "FINAL_REFUND_DENIED"
PHASE_FINAL_RETURN_EXPIRED = "FINAL_RETURN_EXPIRED"

AUTHORISATION_ISSUED = "ISSUED"
AUTHORISATION_VERIFIED = "VERIFIED"
AUTHORISATION_EXPIRED = "EXPIRED"

CHECKPOINT_OPEN = "OPEN"
CHECKPOINT_APPROVED = "APPROVED"
CHECKPOINT_REJECTED = "REJECTED"
CHECKPOINT_EXPIRED = "EXPIRED"

REFUND_REQUESTED = "REQUESTED"
REFUND_SETTLED = "SETTLED"
REFUND_REVERSED = "REVERSED"

OBLIGATION_OPEN = "OPEN"
OBLIGATION_DISCHARGED = "DISCHARGED"
OBLIGATION_CANCELLED = "CANCELLED"
OBLIGATION_BREACHED = "BREACHED"

TERMINAL_COMPLETED_REFUND_SETTLED = "completed_refund_settled"
TERMINAL_CLOSED_REFUND_DENIED = "closed_refund_denied"
TERMINAL_CLOSED_RETURN_EXPIRED = "closed_return_expired"
PRODUCIBLE_TERMINALS: tuple[str, ...] = (
    TERMINAL_COMPLETED_REFUND_SETTLED,
    TERMINAL_CLOSED_REFUND_DENIED,
    TERMINAL_CLOSED_RETURN_EXPIRED,
)

OBSERVABLE_ROOTS: tuple[str, ...] = (
    "case",
    "authorisations",
    "customer_claims",
    "carrier_records",
    "inspections",
    "checkpoints",
    "refunds",
    "communications",
    "obligations",
)


@dataclass
class ReturnRefundState:
    """One operation instance's mutable system-of-record state."""

    phase: str = PHASE_NEW
    return_id: str | None = None
    order_id: str | None = None
    amount_minor: int | None = None
    currency: str | None = None
    current_cycle_id: str | None = None
    terminal: dict[str, Any] | None = None
    replay_final: bool = False
    provisional: dict[str, Any] | None = None
    authorisations: dict[str, dict[str, Any]] = field(default_factory=dict)
    customer_claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    carrier_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    inspections: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    communications: dict[str, dict[str, Any]] = field(default_factory=dict)
    obligations: dict[str, dict[str, Any]] = field(default_factory=dict)
    hidden: dict[str, Any] = field(default_factory=dict)

    def case_record(self) -> dict[str, Any]:
        return {
            "return_id": self.return_id,
            "order_id": self.order_id,
            "amount_minor": self.amount_minor,
            "currency": self.currency,
            "current_cycle_id": self.current_cycle_id,
            "phase": self.phase,
            "provisional": deepcopy(self.provisional),
            "terminal": deepcopy(self.terminal),
            "replay_final": self.replay_final,
        }

    def _roots(self) -> dict[str, Any]:
        return {
            "case": self.case_record(),
            "authorisations": deepcopy(self.authorisations),
            "customer_claims": deepcopy(self.customer_claims),
            "carrier_records": deepcopy(self.carrier_records),
            "inspections": deepcopy(self.inspections),
            "checkpoints": deepcopy(self.checkpoints),
            "refunds": deepcopy(self.refunds),
            "communications": deepcopy(self.communications),
            "obligations": deepcopy(self.obligations),
        }

    def project(self, roots: list[str] | tuple[str, ...]) -> dict[str, Any]:
        available = self._roots()
        unknown = sorted(set(roots) - set(available))
        if unknown:
            raise KeyError(f"unknown observable return/refund root(s): {unknown}")
        return {root: available[root] for root in roots}

    def observable(self) -> dict[str, Any]:
        return self._roots()

    def canonical(self) -> dict[str, Any]:
        body = self._roots()
        body["hidden"] = deepcopy(self.hidden)
        return body

    def open_checkpoints(self) -> list[dict[str, Any]]:
        return [
            checkpoint
            for checkpoint in self.checkpoints.values()
            if checkpoint.get("status") == CHECKPOINT_OPEN
        ]

    def open_obligations(self) -> list[dict[str, Any]]:
        return [
            obligation
            for obligation in self.obligations.values()
            if obligation.get("status") == OBLIGATION_OPEN
        ]

    def current_refund(self) -> dict[str, Any] | None:
        candidates = [
            refund
            for refund in self.refunds.values()
            if refund.get("cycle_id") == self.current_cycle_id
        ]
        return candidates[-1] if candidates else None


__all__ = [
    "AUTHORISATION_EXPIRED",
    "AUTHORISATION_ISSUED",
    "AUTHORISATION_VERIFIED",
    "CHECKPOINT_APPROVED",
    "CHECKPOINT_EXPIRED",
    "CHECKPOINT_OPEN",
    "CHECKPOINT_REJECTED",
    "OBLIGATION_BREACHED",
    "OBLIGATION_CANCELLED",
    "OBLIGATION_DISCHARGED",
    "OBLIGATION_OPEN",
    "OBSERVABLE_ROOTS",
    "PHASE_AWAITING_APPROVAL",
    "PHASE_AWAITING_HANDOVER",
    "PHASE_AWAITING_INSPECTION",
    "PHASE_FINAL_REFUND_DENIED",
    "PHASE_FINAL_RETURN_EXPIRED",
    "PHASE_FINAL_SUCCESS",
    "PHASE_NEW",
    "PHASE_PROVISIONAL",
    "PHASE_READY_TO_REFUND",
    "PHASE_REFUND_DENIED",
    "PHASE_REFUND_PENDING",
    "PHASE_REFUND_REOPENED",
    "PHASE_REFUND_SETTLED",
    "PHASE_RETURN_EXPIRED",
    "PHASE_RETURN_REQUESTED",
    "PRODUCIBLE_TERMINALS",
    "REFUND_REQUESTED",
    "REFUND_REVERSED",
    "REFUND_SETTLED",
    "TERMINAL_CLOSED_REFUND_DENIED",
    "TERMINAL_CLOSED_RETURN_EXPIRED",
    "TERMINAL_COMPLETED_REFUND_SETTLED",
    "ReturnRefundState",
]
