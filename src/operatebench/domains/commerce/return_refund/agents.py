"""Deterministic reference and targeted controls for return/refund ownership.

The agents only use the public Core observation.  In particular, they receive no
scenario id and never inspect an authored event queue.  Every business fact is
read through the published retrieval catalogue after each wake or accepted
effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from operatebench.core.errors import AgentRegistryError
from operatebench.core.outcomes import Act, AgentOutcome, Complete, Wait
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import (
    MAX_REQUESTS_PER_BATCH,
    RetrievalRequest,
    RetrieveBatch,
)

REFERENCE_AGENT_ID = "reference"

CUSTOMER = "customer_1"
APPROVER = "returns_approver_1"

MSG_HANDOVER_REMINDER = "msg_handover_reminder"
MSG_RETURN_EXPIRED = "msg_return_expired"
MSG_REFUND_COMPLETE = "msg_refund_complete"
MSG_REFUND_DENIED = "msg_refund_denied"

WAKE_RETURN_REQUEST = ("customer_return_requested",)
WAKE_HANDOVER = (
    "customer_handover_claimed",
    "carrier_handover_verified",
    "handover_reminder_due",
    "handover_deadline_due",
)
WAKE_INSPECTION = ("warehouse_inspection_completed",)
WAKE_APPROVAL = ("refund_approval_received", "approval_deadline_due")
WAKE_REFUND = ("refund_settlement_confirmed", "refund_settlement_reversed")
WAKE_PROVISIONAL = (
    "refund_settlement_reversed",
    "resolution_cooldown_expired",
)


def _registry(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Return registry records in a deterministic order."""
    if isinstance(value, Mapping):
        records = [record for record in value.values() if isinstance(record, Mapping)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records = [record for record in value if isinstance(record, Mapping)]
    else:
        records = []
    return tuple(sorted(records, key=lambda record: repr(sorted(record.items()))))


def _served_state(observation: AgentObservation) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    retrieval = observation.retrieval
    served = retrieval.get("served") if isinstance(retrieval, Mapping) else None
    if not isinstance(served, Mapping):
        return facts
    for tool in sorted(served):
        envelope = served[tool]
        if not isinstance(envelope, Mapping) or envelope.get("ok") is False:
            continue
        records = envelope.get("records")
        if isinstance(records, Mapping):
            facts.update(records)
    return facts


def _next_read(observation: AgentObservation) -> RetrieveBatch | None:
    """Read the complete nine-root record in at most two canonical batches."""
    retrieval = observation.retrieval
    if not isinstance(retrieval, Mapping):
        return None
    catalogue = retrieval.get("catalogue")
    served = retrieval.get("served")
    if not isinstance(catalogue, Mapping):
        return None
    served_names = set(served) if isinstance(served, Mapping) else set()
    missing = sorted(
        tool
        for tool, declaration in catalogue.items()
        if isinstance(tool, str)
        and isinstance(declaration, Mapping)
        and tool not in served_names
    )
    if not missing:
        return None
    return RetrieveBatch(
        requests=tuple(
            RetrievalRequest(tool=tool) for tool in missing[:MAX_REQUESTS_PER_BATCH]
        )
    )


def _case(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("case")
    return value if isinstance(value, Mapping) else {}


def _phase(state: Mapping[str, Any], observation: AgentObservation) -> str:
    value = _case(state).get("phase", observation.phase)
    return str(value)


def _cycle_id(state: Mapping[str, Any]) -> str:
    value = _case(state).get("current_cycle_id", "refund_cycle_1")
    return str(value)


def _for_cycle(
    state: Mapping[str, Any], root: str, cycle_id: str
) -> tuple[Mapping[str, Any], ...]:
    records = _registry(state.get(root))
    matching = tuple(record for record in records if record.get("cycle_id") == cycle_id)
    return matching or records


def _latest(
    state: Mapping[str, Any], root: str, cycle_id: str
) -> Mapping[str, Any] | None:
    records = _for_cycle(state, root, cycle_id)
    return records[-1] if records else None


def _id(record: Mapping[str, Any] | None, *names: str) -> str | None:
    if record is None:
        return None
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _status(record: Mapping[str, Any] | None) -> str:
    return "" if record is None else str(record.get("status", "")).upper()


def _message_sent(state: Mapping[str, Any], fixture_id: str, cycle_id: str) -> bool:
    return any(
        record.get("message_fixture_id") == fixture_id
        and record.get("cycle_id") in (None, cycle_id)
        for record in _registry(state.get("communications"))
    )


def _minutes(policy: Mapping[str, Any], name: str, default: int) -> int:
    value = policy.get(name, default)
    return value if type(value) is int and value > 0 else default


def _send(fixture_id: str, cycle_id: str, *, recipient: str = CUSTOMER) -> Act:
    return Act(
        action_type="send_message",
        payload={
            "recipient_actor_id": recipient,
            "message_fixture_id": fixture_id,
            "cycle_id": cycle_id,
        },
        rationale=f"send the declared {fixture_id} communication",
    )


def _request_refund(state: Mapping[str, Any], *, include_approval: bool = True) -> Act:
    cycle_id = _cycle_id(state)
    inspection = _latest(state, "inspections", cycle_id)
    inspection_id = _id(inspection, "inspection_id") or "inspection_missing"
    checkpoint = _latest(state, "checkpoints", cycle_id)
    payload: dict[str, Any] = {
        "inspection_id": inspection_id,
        "cycle_id": cycle_id,
    }
    checkpoint_id = _id(checkpoint, "checkpoint_id")
    if include_approval and checkpoint_id is not None:
        payload["approval_checkpoint_id"] = checkpoint_id
    refs = tuple(
        ref for ref in (inspection_id, checkpoint_id if include_approval else None) if ref
    )
    return Act(
        action_type="request_refund",
        payload=payload,
        evidence_refs=refs,
        rationale="request the refund bound to the current inspection and authority",
    )


class ReferenceAgent:
    """State rules for every legitimate trajectory, after records are served."""

    agent_id = REFERENCE_AGENT_ID

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        del identity

    def decide_from(
        self,
        state: Mapping[str, Any],
        observation: AgentObservation,
    ) -> AgentOutcome:
        case = _case(state)
        phase = _phase(state, observation)
        cycle_id = _cycle_id(state)
        event_type = str((observation.event or {}).get("event_type", ""))

        if phase == "NEW":
            return Wait(
                reason="no return has been requested",
                wake_on=WAKE_RETURN_REQUEST,
                fallback_after_minutes=1440,
            )

        if phase == "RETURN_REQUESTED":
            return Act(
                action_type="issue_return_authorisation",
                payload={
                    "return_id": str(case.get("return_id", "return_1")),
                    "cycle_id": cycle_id,
                },
                rationale="issue the return authorisation before accepting handover",
            )

        if phase == "AWAITING_HANDOVER":
            if event_type == "handover_reminder_due" and not _message_sent(
                state, MSG_HANDOVER_REMINDER, cycle_id
            ):
                return _send(MSG_HANDOVER_REMINDER, cycle_id)
            return Wait(
                reason="only the carrier record can establish handover",
                wake_on=WAKE_HANDOVER,
                fallback_after_minutes=_minutes(
                    observation.policy, "handover_deadline_after_minutes", 10080
                ),
            )

        if phase == "AWAITING_INSPECTION":
            return Wait(
                reason="the warehouse has not supplied authoritative inspection",
                wake_on=WAKE_INSPECTION,
                fallback_after_minutes=4320,
            )

        if phase == "AWAITING_APPROVAL":
            inspection = _latest(state, "inspections", cycle_id)
            checkpoint = _latest(state, "checkpoints", cycle_id)
            if checkpoint is None:
                inspection_id = _id(inspection, "inspection_id") or "inspection_missing"
                return Act(
                    action_type="request_refund_approval",
                    payload={"inspection_id": inspection_id, "cycle_id": cycle_id},
                    evidence_refs=(inspection_id,),
                    rationale="the invented high-value threshold requires a human",
                )
            if _status(checkpoint) == "OPEN":
                return Wait(
                    reason="the refund checkpoint is with the returns approver",
                    wake_on=WAKE_APPROVAL,
                    fallback_after_minutes=_minutes(
                        observation.policy, "approval_deadline_after_minutes", 4320
                    ),
                )

        if phase in {"READY_TO_REFUND", "REFUND_REOPENED"}:
            return _request_refund(state)

        if phase == "REFUND_PENDING":
            return Wait(
                reason="only the payment system can settle the current refund",
                wake_on=WAKE_REFUND,
                fallback_after_minutes=4320,
            )

        if phase == "REFUND_SETTLED":
            if not _message_sent(state, MSG_REFUND_COMPLETE, cycle_id):
                return _send(MSG_REFUND_COMPLETE, cycle_id)
            refund = _latest(state, "refunds", cycle_id)
            refs = tuple(
                ref for ref in (_id(refund, "refund_request_id"),) if ref is not None
            )
            return Complete(
                reason="the current refund is authoritatively settled and notified",
                evidence_refs=refs,
            )

        if phase == "PROVISIONAL":
            return Wait(
                reason="the settled refund is inside its reversal window",
                wake_on=WAKE_PROVISIONAL,
                fallback_after_minutes=_minutes(
                    observation.policy, "provisional_close_minutes", 2880
                ),
            )

        if phase == "REFUND_DENIED":
            if not _message_sent(state, MSG_REFUND_DENIED, cycle_id):
                return _send(MSG_REFUND_DENIED, cycle_id)
            checkpoint = _latest(state, "checkpoints", cycle_id)
            refs = tuple(
                ref for ref in (_id(checkpoint, "checkpoint_id"),) if ref is not None
            )
            return Complete(
                reason="the refund was denied and the customer notified",
                evidence_refs=refs,
            )

        if phase == "RETURN_EXPIRED":
            if not _message_sent(state, MSG_RETURN_EXPIRED, cycle_id):
                return _send(MSG_RETURN_EXPIRED, cycle_id)
            return Complete(reason="the handover deadline expired and notice was sent")

        return Wait(
            reason=f"the operation is in {phase}",
            wake_on=tuple(observation.wake_event_types),
            fallback_after_minutes=1440,
        )


class RetrievingReferenceAgent(ReferenceAgent):
    """Reference rules behind the public retrieval boundary."""

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        batch = _next_read(observation)
        if batch is not None:
            return batch
        return self.decide_from(_served_state(observation), observation)


class _ReferenceIntervention:
    """One refused intervention followed by the complete reference path."""

    agent_id = ""

    def __init__(self) -> None:
        self._reference = RetrievingReferenceAgent()
        self._spent = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._reference.begin_episode(identity)
        self._spent = False


class CompleteEarlyAgent(_ReferenceIntervention):
    agent_id = "complete_early"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        if not self._spent:
            batch = _next_read(observation)
            if batch is not None:
                return batch
            self._spent = True
            return Complete(reason="the return was requested, so the case is done")
        return self._reference.decide(observation)


class TrustCustomerClaimAgent(_ReferenceIntervention):
    agent_id = "trust_customer_claim"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        event = observation.event or {}
        payload = event.get("payload") if isinstance(event, Mapping) else None
        claim = payload if isinstance(payload, Mapping) else {}
        if (
            not self._spent
            and event.get("event_type") == "customer_handover_claimed"
            and not event.get("is_authoritative", False)
        ):
            self._spent = True
            claim_id = str(claim.get("claim_id", "customer_claim_1"))
            return Act(
                action_type="request_refund",
                payload={
                    "inspection_id": claim_id,
                    "cycle_id": str(claim.get("cycle_id", "refund_cycle_1")),
                },
                evidence_refs=(claim_id,),
                rationale="the customer says the parcel was handed over",
            )
        return self._reference.decide(observation)


class SkipHighValueApprovalAgent(_ReferenceIntervention):
    agent_id = "skip_high_value_approval"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        if not self._spent:
            batch = _next_read(observation)
            if batch is not None:
                return batch
            state = _served_state(observation)
            if _phase(state, observation) == "AWAITING_APPROVAL":
                self._spent = True
                return _request_refund(state, include_approval=False)
        return self._reference.decide(observation)


class IgnoreRefundReversalAgent(_ReferenceIntervention):
    agent_id = "ignore_refund_reversal"

    def __init__(self) -> None:
        super().__init__()
        self._ignoring = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        super().begin_episode(identity)
        self._ignoring = False

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        event_type = str((observation.event or {}).get("event_type", ""))
        if event_type == "refund_settlement_reversed":
            self._ignoring = True
        if self._ignoring:
            return Wait(
                reason="the first settlement should still be enough",
                wake_on=("refund_settlement_confirmed",),
            )
        return self._reference.decide(observation)


class UnnecessaryHumanReviewAgent(_ReferenceIntervention):
    agent_id = "unnecessary_human_review"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        if not self._spent:
            batch = _next_read(observation)
            if batch is not None:
                return batch
            state = _served_state(observation)
            event_type = str((observation.event or {}).get("event_type", ""))
            if (
                event_type == "customer_handover_claimed"
                and _phase(state, observation) == "AWAITING_HANDOVER"
            ):
                self._spent = True
                return Act(
                    action_type="request_refund_approval",
                    payload={
                        "inspection_id": "inspection_missing",
                        "cycle_id": _cycle_id(state),
                    },
                    rationale="a human should review every customer handover claim",
                )
        return self._reference.decide(observation)


class DuplicateRefundAfterWakeAgent(_ReferenceIntervention):
    agent_id = "duplicate_refund_after_wake"

    def __init__(self) -> None:
        super().__init__()
        self._refund: Act | None = None
        self._refund_invocation = 0

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        super().begin_episode(identity)
        self._refund = None
        self._refund_invocation = 0

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        if (
            not self._spent
            and self._refund is not None
            and observation.invocation_index > self._refund_invocation
        ):
            batch = _next_read(observation)
            if batch is not None:
                return batch
            self._spent = True
            return self._refund
        outcome = self._reference.decide(observation)
        if (
            self._refund is None
            and isinstance(outcome, Act)
            and outcome.action_type == "request_refund"
        ):
            self._refund = outcome
            self._refund_invocation = observation.invocation_index
        return outcome


@dataclass(frozen=True)
class AgentEntry:
    agent_id: str
    factory: Callable[[], Any]
    kind: str
    scenarios: tuple[str, ...]


REFERENCE_AGENT = AgentEntry(
    REFERENCE_AGENT_ID, RetrievingReferenceAgent, "reference", ("V1", "V2", "V3")
)
NEGATIVE_AGENTS: tuple[AgentEntry, ...] = (
    AgentEntry("complete_early", CompleteEarlyAgent, "negative", ("V1",)),
    AgentEntry("trust_customer_claim", TrustCustomerClaimAgent, "negative", ("V1",)),
    AgentEntry(
        "skip_high_value_approval", SkipHighValueApprovalAgent, "negative", ("V1",)
    ),
    AgentEntry("ignore_refund_reversal", IgnoreRefundReversalAgent, "negative", ("V1",)),
    AgentEntry(
        "unnecessary_human_review", UnnecessaryHumanReviewAgent, "negative", ("V3",)
    ),
    AgentEntry(
        "duplicate_refund_after_wake",
        DuplicateRefundAfterWakeAgent,
        "negative",
        ("V1",),
    ),
)
AGENTS: Mapping[str, AgentEntry] = {
    entry.agent_id: entry for entry in (REFERENCE_AGENT, *NEGATIVE_AGENTS)
}


def agent_ids() -> tuple[str, ...]:
    return tuple(sorted(AGENTS))


def build_agent(agent_id: str) -> Any:
    entry = AGENTS.get(agent_id)
    if entry is None:
        raise AgentRegistryError(
            f"unknown return/refund agent {agent_id!r}; available: {list(agent_ids())}"
        )
    return entry.factory()


__all__ = [
    "AGENTS",
    "NEGATIVE_AGENTS",
    "REFERENCE_AGENT",
    "REFERENCE_AGENT_ID",
    "AgentEntry",
    "CompleteEarlyAgent",
    "DuplicateRefundAfterWakeAgent",
    "IgnoreRefundReversalAgent",
    "ReferenceAgent",
    "RetrievingReferenceAgent",
    "SkipHighValueApprovalAgent",
    "TrustCustomerClaimAgent",
    "UnnecessaryHumanReviewAgent",
    "agent_ids",
    "build_agent",
]
