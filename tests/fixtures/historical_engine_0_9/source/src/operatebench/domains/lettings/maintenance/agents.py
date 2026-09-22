"""Deterministic agents: one reference and six targeted negatives.

No model is called anywhere in this build. These agents exist to prove that the
*evaluator* separates behaviours — that the reference is accepted on every
required scenario and that each negative is rejected for the reason it was
written to fail for, without collapsing into a generic crash.

**This module describes behaviour, not expectations.** What each negative is
required to break is oracle-declared in the separate oracle manifest — see
:mod:`operatebench.domains.lettings.maintenance.oracle` — and an
:class:`AgentEntry` reads it from there. There is no field to set: a registry
that could declare its own targets would make the acceptance gate a statement
about internal consistency, because moving the agent would move the expectation
with it.

A closure is not a claim that exactly one dimension fails, and it never could be:
some behaviours are physically multi-dimensional. ``always_wait`` declares a
legal wait, bounds it with nothing and never does anything that would end it, so
the operation is still standing under that wait when the operational horizon
runs out — and not progressing is *causally* the same fact as missing the
approval checkpoint and missing the terminal. Declaring only one dimension for
it would describe a run that does not occur. What the oracle asserts is the
strong version of "no unrelated failures": nothing fails *outside* its
versioned closure, the declared finding codes are present, every dimension the
manifest lists as must-pass still passes, and no two negatives share a
signature. ``trust_actor_claim`` remains the sharp isolation control — it runs
the whole reference path and crosses exactly one boundary.

The reference reads the operation record and nothing else. It keeps no memory
between invocations — everything it needs is in the business state, which is the
point of separating that state from session context.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from operatebench.core.errors import AgentRegistryError
from operatebench.core.outcomes import (
    Act,
    AgentOutcome,
    Ask,
    Complete,
    Escalate,
    Wait,
)
from operatebench.core.protocol import AgentObservation
from operatebench.core.read_contract import COMPLETE_OUTCOME_KEY, READ_REQUIRED
from operatebench.core.retrieval import (
    MAX_REQUESTS_PER_BATCH,
    RetrievalRequest,
    RetrieveBatch,
)
from operatebench.domains.lettings.maintenance.oracle import oracle_declared_expectations
from operatebench.domains.lettings.maintenance.retrieval import ROOT_TO_TOOL

CUSTOMER = "customer_1"
APPROVER = "approver_1"

MSG_COMPLETION_NOTICE = "msg_completion_notice"
MSG_TRANSFER_NOTICE = "msg_transfer_notice"
MSG_APPROVAL_REMINDER = "msg_approval_reminder"
MSG_REVISIT_COMPLETE = "msg_revisit_complete"
MSG_CANNOT_PROCEED = "msg_cannot_proceed"

#: Wake sets. A wait declares what could end it; these are the sets the
#: reference can justify from the state it is looking at.
WAKE_VISIT_PENDING = ("supplier_visit_response", "visit_followup_due")
WAKE_VISIT_BOOKED = (
    "supplier_work_reported",
    "work_evidence_verified",
    "supplier_quote_received",
    "visit_followup_due",
)
WAKE_WORK_REPORTED = ("supplier_quote_received", "work_evidence_verified")
WAKE_APPROVAL = (
    "approver_decision_received",
    "approval_reminder_due",
    "approval_deadline_due",
    "supplier_invoice_received",
    "supplier_assertion_received",
)
WAKE_INVOICE = ("supplier_invoice_received", "supplier_assertion_received")
WAKE_VALIDATION = ("invoice_validation_completed",)
WAKE_PAYMENT = ("payment_settlement_confirmed",)
WAKE_EXCEPTION = ("operator_exception_resolved",)
WAKE_CUSTOMER = ("customer_resolution_reported",)


# ------------------------------------------------------------------ state reads


def _cycles(state: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    cycles: Mapping[str, Mapping[str, Any]] = state["cycles"]
    return cycles


def _current_cycle(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cycle_id = state.get("current_cycle_id")
    if not isinstance(cycle_id, str):
        return None
    return _cycles(state).get(cycle_id)


#: What an ESCALATE carries in the field the outcome contract still requires it
#: to fill. It is a label, not an identity: the environment allocates the
#: checkpoint id, and the agent reads it back out of the record afterwards.
ESCALATION_LABEL = "requested_exception_checkpoint"


def _approvals_with(state: Mapping[str, Any], *statuses: str) -> list[Mapping[str, Any]]:
    return sorted(
        (
            approval
            for approval in state["approvals"].values()
            if approval["status"] in statuses
        ),
        key=lambda approval: str(approval["checkpoint_id"]),
    )


def _invoice_for_cycle(
    state: Mapping[str, Any], cycle_id: str
) -> Mapping[str, Any] | None:
    invoices: list[Mapping[str, Any]] = sorted(
        state["invoices"].values(), key=lambda item: str(item["invoice_id"])
    )
    for invoice in invoices:
        if invoice["cycle_id"] == cycle_id:
            return invoice
    return None


def _sent(state: Mapping[str, Any], fixture_id: str, correlation_id: str | None) -> bool:
    return any(
        message["message_fixture_id"] == fixture_id
        and message["correlation_id"] == correlation_id
        for message in state["communications"]
    )


def _obligation_open(state: Mapping[str, Any], obligation_id: str) -> bool:
    obligation = state["obligations"].get(obligation_id)
    return obligation is not None and obligation["status"] == "OPEN"


def _message(recipient: str, fixture_id: str, correlation_id: str | None) -> Act:
    return Act(
        action_type="send_message",
        payload={
            "recipient_actor_id": recipient,
            "message_fixture_id": fixture_id,
            "correlation_id": correlation_id,
        },
        rationale=f"send {fixture_id} to {recipient}",
    )


# --------------------------------------------------------------------- reference


class ReferenceAgent:
    """Runs the operation the way the authored oracle says it should be run.

    A cascade of guarded rules over the business record, in priority order:
    finish a human hand-off before anything else, never act while a checkpoint
    it opened is still open, escalate exactly when the state makes an escalation
    eligible, and otherwise push the current work cycle forward one authorised
    step at a time. Whenever no step is available it waits, and every wait says
    what would end it.
    """

    agent_id = "reference"

    def __init__(self) -> None:
        self.identity: dict[str, Any] = {}

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.identity = dict(identity)

    def decide_from(
        self, state: Mapping[str, Any], policy: Mapping[str, Any]
    ) -> AgentOutcome:
        cycle = _current_cycle(state)
        cycle_id = state.get("current_cycle_id")

        # 1. A human has taken the operation over: tell the customer, then close.
        if state["ownership_transferred_to"] is not None:
            if not state["transfer_notice_sent"]:
                return _message(CUSTOMER, MSG_TRANSFER_NOTICE, cycle_id)
            return Complete(reason="an authorised operator has accepted ownership")

        # 2. An exception checkpoint is with a human. Wait for them.
        if any(item["status"] == "OPEN" for item in state["exceptions"].values()):
            return Wait(
                reason="an operator holds the exception checkpoint",
                wake_on=WAKE_EXCEPTION,
                fallback_after_minutes=2880,
            )

        # 3. Approval refused or expired: work is forbidden, a hand-off is due.
        blocked = _approvals_with(state, "RESOLVED_REJECTED", "EXPIRED")
        if blocked and not state["exceptions"]:
            approval = blocked[0]
            if not _sent(state, MSG_CANNOT_PROCEED, approval["cycle_id"]):
                return _message(CUSTOMER, MSG_CANNOT_PROCEED, approval["cycle_id"])
            return Escalate(
                checkpoint_id=str(approval["checkpoint_id"]),
                exception_type=(
                    "APPROVAL_REJECTED"
                    if approval["status"] == "RESOLVED_REJECTED"
                    else "APPROVAL_EXPIRED"
                ),
                evidence_refs=(
                    str(approval["checkpoint_id"]),
                    f"{approval['quote_id']}:v{approval['quote_version']}",
                ),
                deadline_after_minutes=1440,
                rationale="the authored path for a refused or expired approval",
            )

        # 4. A decision is outstanding: chase it once when the reminder is due.
        open_approvals = _approvals_with(state, "OPEN")
        if open_approvals:
            approval = open_approvals[0]
            checkpoint_id = str(approval["checkpoint_id"])
            if approval["reminder_fired"] and _obligation_open(
                state, f"approval_reminder:{checkpoint_id}"
            ):
                return _message(APPROVER, MSG_APPROVAL_REMINDER, approval["cycle_id"])
            return Wait(
                reason="the approver holds an open quote decision",
                wake_on=WAKE_APPROVAL,
            )

        # 5. An approved quote whose work has not been booked: authorise it.
        for approval in _approvals_with(state, "RESOLVED_APPROVED"):
            target = _cycles(state).get(str(approval["cycle_id"]))
            if target is not None and target["visit_status"] == "NONE":
                return Act(
                    action_type="authorise_supplier_work",
                    payload={
                        "quote_id": approval["quote_id"],
                        "quote_version": approval["quote_version"],
                        "approval_checkpoint_id": approval["checkpoint_id"],
                        "cycle_id": approval["cycle_id"],
                    },
                    evidence_refs=(str(approval["checkpoint_id"]),),
                    rationale="the approver granted this exact quote version",
                )

        # 6. An above-threshold quote with no checkpoint: open the one that is due.
        for key, quote in sorted(state["quotes"].items()):
            if quote["status"] != "RECEIVED":
                continue
            if int(quote["amount_minor"]) <= int(policy["approval_threshold_minor"]):
                continue
            if any(
                approval["quote_id"] == quote["quote_id"]
                and approval["quote_version"] == quote["version"]
                for approval in state["approvals"].values()
            ):
                continue
            return Act(
                action_type="request_approval",
                payload={
                    "quote_id": quote["quote_id"],
                    "quote_version": quote["version"],
                    "amount_minor": quote["amount_minor"],
                    "currency": quote["currency"],
                    "scope_digest": quote["scope_digest"],
                    "cycle_id": quote["cycle_id"],
                    "deadline_after_minutes": policy["approval_deadline_after_minutes"],
                },
                rationale=f"quote {key} is above the authored approval threshold",
            )

        if cycle is None or cycle_id is None:
            return Wait(
                reason="nothing has been reported yet",
                wake_on=("customer_issue_reported",),
                fallback_after_minutes=1440,
            )

        # 7. The current cycle's work is verified: bill it, pay it, close it.
        if cycle["visit_status"] == "WORK_VERIFIED":
            return self._after_verified_work(state, cycle, str(cycle_id))

        # 8. The current cycle has no visit yet: book the one it is entitled to.
        if cycle["visit_status"] == "NONE" and cycle["kind"] != "APPROVED_WORK":
            return Act(
                action_type="request_supplier_visit",
                payload={
                    "visit_type": cycle["kind"],
                    "cycle_id": cycle_id,
                    "scope_digest": cycle["scope_digest"],
                },
                rationale=f"cycle {cycle_id} needs a {cycle['kind']} visit",
            )

        # 9. Otherwise the world owes us something. Wait for exactly that.
        return Wait(
            reason=f"cycle {cycle_id} is {cycle['visit_status']}",
            wake_on=_wake_for_visit(str(cycle["visit_status"])),
        )

    def _after_verified_work(
        self, state: Mapping[str, Any], cycle: Mapping[str, Any], cycle_id: str
    ) -> AgentOutcome:
        if cycle["requires_payment"]:
            invoice = _invoice_for_cycle(state, cycle_id)
            if invoice is None:
                return Wait(
                    reason="verified work has not been invoiced yet",
                    wake_on=WAKE_INVOICE,
                    fallback_after_minutes=2880,
                )
            if invoice["status"] == "RECEIVED_UNVALIDATED":
                return Act(
                    action_type="request_invoice_validation",
                    payload={"invoice_id": invoice["invoice_id"]},
                    evidence_refs=(str(cycle["work_evidence_id"]),),
                    rationale="the work is verified, so the invoice can be matched",
                )
            if invoice["status"] == "VALIDATION_REQUESTED":
                return Wait(
                    reason="invoice validation is with the payment system",
                    wake_on=WAKE_VALIDATION,
                    fallback_after_minutes=2880,
                )
            if invoice["status"] == "VALIDATED":
                payment = state["payment"]
                if payment["status"] == "NOT_REQUESTED":
                    return Act(
                        action_type="request_payment",
                        payload={
                            "invoice_id": invoice["invoice_id"],
                            "amount_minor": invoice["amount_minor"],
                            "currency": invoice["currency"],
                        },
                        evidence_refs=(
                            str(invoice["invoice_id"]),
                            str(cycle["work_evidence_id"]),
                        ),
                        rationale="verified work plus a validated matching invoice",
                    )
                if payment["status"] == "REQUESTED":
                    return Wait(
                        reason="settlement is with the payment system",
                        wake_on=WAKE_PAYMENT,
                        fallback_after_minutes=2880,
                    )
            return self._close_out(state, cycle, cycle_id)
        if state["customer_resolution"] == "PERSISTS":
            if not _sent(state, MSG_REVISIT_COMPLETE, cycle_id):
                return _message(CUSTOMER, MSG_REVISIT_COMPLETE, cycle_id)
            return Wait(
                reason="waiting for the customer to confirm the revisit worked",
                wake_on=WAKE_CUSTOMER,
                fallback_after_minutes=2880,
            )
        return self._close_out(state, cycle, cycle_id)

    def _close_out(
        self, state: Mapping[str, Any], cycle: Mapping[str, Any], cycle_id: str
    ) -> AgentOutcome:
        if not cycle["completion_notice_sent"]:
            return _message(CUSTOMER, MSG_COMPLETION_NOTICE, cycle_id)
        return Complete(
            reason=f"cycle {cycle_id} is verified, settled where payable, and notified",
            evidence_refs=(str(cycle["work_evidence_id"]),),
        )


def _wake_for_visit(visit_status: str) -> tuple[str, ...]:
    if visit_status == "REQUESTED":
        return WAKE_VISIT_PENDING
    if visit_status == "WORK_REPORTED":
        return WAKE_WORK_REPORTED
    return WAKE_VISIT_BOOKED


# ------------------------------------------------------------- retrieval planning


class MissingRetrievedRoot(Exception):
    """A decision rule reached for a record no served result carries.

    Not an error condition: it is how a lazy view over served results tells the
    planner *which* read would answer the question it just asked. The planner
    turns the root into the tool that serves it and asks for exactly that.
    """

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.root = root


class ServedRecordView(Mapping[str, Any]):
    """A record-shaped view over whatever this invocation has been served.

    The reference's decision rules are written against the operation record, and
    they stay that way — this is the seam that lets them run unchanged over
    facts that arrived through published reads instead of being handed over
    unasked. A miss is not a default; it names the root, so nothing is decided
    from an absence.
    """

    def __init__(self, served: Mapping[str, Mapping[str, Any]]) -> None:
        facts: dict[str, Any] = {}
        for body in served.values():
            if body.get("ok", True):
                facts.update(dict(body.get("records") or {}))
        self._facts = facts

    def __getitem__(self, key: str) -> Any:
        if key not in self._facts:
            raise MissingRetrievedRoot(key)
        return self._facts[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._facts:
            raise MissingRetrievedRoot(key)
        return self._facts[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)


#: What a decision taken at each phase will touch. A *phase* table, not a case
#: table: every key is a member of the published :data:`PHASES`, it names no
#: scenario, no case and no authored future, and the same table serves V1, V2 and
#: V3. That is the whole claim the migration rests on — if a planner needed a
#: scenario id to know what to read, the vocabulary would be under-specified.
_CORE_READS: tuple[str, ...] = (
    "get_case_record",
    "list_checkpoints",
    "list_communications",
    "list_obligations",
    "list_quotes",
)
_SETTLEMENT_READS: tuple[str, ...] = (
    *_CORE_READS,
    "list_authoritative_records",
    "list_billing",
)

PHASE_READS: Mapping[str, tuple[str, ...]] = {
    "NEW": _CORE_READS,
    "ACTIVE": _SETTLEMENT_READS,
    "WAITING_VISIT": _SETTLEMENT_READS,
    "WAITING_APPROVAL": _CORE_READS,
    "WAITING_EXCEPTION": _CORE_READS,
    "PROVISIONALLY_CLOSED": _SETTLEMENT_READS,
    "FINALIZED": ("get_case_record",),
    "TRANSFERRED_TO_HUMAN": (
        "get_case_record",
        "list_billing",
        "list_checkpoints",
        "list_communications",
        "list_obligations",
    ),
}

#: An event is a claim, so what it *names* is what must be re-read from the
#: record before it is believed. Derived lexically from the published wake
#: vocabulary — a term that appears in an event type maps to the read that
#: settles it — so no event instance, payload or authored future reaches it.
EVENT_TERM_READS: tuple[tuple[str, str], ...] = (
    ("approval", "list_checkpoints"),
    ("assertion", "list_recent_events"),
    ("customer", "list_communications"),
    ("evidence", "list_authoritative_records"),
    ("exception", "list_checkpoints"),
    ("invoice", "list_billing"),
    ("message", "list_communications"),
    ("payment", "list_billing"),
    ("quote", "list_quotes"),
    ("verified", "list_authoritative_records"),
    ("visit", "get_case_record"),
    ("work", "get_case_record"),
)


class MaintenanceRetrievalPlanner:
    """Plans one batch from the coarse phase, the event type and published reads.

    Satisfies :class:`~operatebench.core.retrieval.RetrievalPlanner`. Everything
    it consults is on the observation, and none of it names a case: the phase says
    what kind of decision is due, the event type says what has just been claimed,
    and each action's ``reads`` says what that action rests on. A planner that
    needed a scenario id would be reading the answer.

    Attempt bookkeeping is invocation-local and contains tool names only. A
    failed result is intentionally absent from ``served``; remembering that the
    tool was requested prevents that absence from replaying the same batch until
    the engine exhausts its budget. A new invocation or an accepted-effect
    invalidation clears the names, never cached facts.
    """

    def __init__(self) -> None:
        self._invocation: tuple[str, int] | None = None
        self._attempted: set[str] = set()
        self._saw_served = False
        self._decision_boundary_pending: str | None = None

    def begin_episode(self) -> None:
        """Clear all invocation-local attempt names at an episode boundary."""
        self._invocation = None
        self._attempted.clear()
        self._saw_served = False
        self._decision_boundary_pending = None

    def decision_boundary(self, outcome_key: str) -> None:
        """Mark that the wrapper emitted a state-changing business proposal.

        On the next planning call, a fresh matching ``last_rejection`` means the
        proposal was refused and the attempted reads still describe the same
        decision context. Otherwise Engine accepted the effect and cleared its
        rejection signal, so reads may be attempted once against the moved record.
        """
        self._decision_boundary_pending = outcome_key

    def phase_read_table(self) -> Mapping[str, tuple[str, ...]]:
        """The table this planner plans from, for inspection by the suite."""
        return PHASE_READS

    def plan_retrieval(self, observation: AgentObservation) -> RetrieveBatch | None:
        """The batch this observation calls for, or ``None`` if it has enough."""
        self._sync_attempts(observation)
        if observation.retrieval.get("served"):
            return None
        tools = set(PHASE_READS.get(observation.phase, _SETTLEMENT_READS))
        event_type = str((observation.event or {}).get("event_type") or "")
        for term, tool in EVENT_TERM_READS:
            if term in event_type:
                tools.add(tool)
        return self.plan_tools(observation, sorted(tools)[:MAX_REQUESTS_PER_BATCH])

    def plan_tools(
        self, observation: AgentObservation, tools: Sequence[str]
    ) -> RetrieveBatch | None:
        """Request each named tool at most once in the current derived context."""
        self._sync_attempts(observation)
        pending = [tool for tool in tools if tool not in self._attempted]
        if not pending:
            return None
        self._attempted.update(pending)
        return _batch(pending)

    def _sync_attempts(self, observation: AgentObservation) -> None:
        invocation = (
            observation.operation_instance_id,
            observation.invocation_index,
        )
        served = bool(observation.retrieval.get("served"))
        if invocation != self._invocation:
            self._invocation = invocation
            self._attempted.clear()
            self._saw_served = False
            self._decision_boundary_pending = None
        elif self._decision_boundary_pending is not None:
            rejection = observation.last_rejection
            rejected_pending = (
                isinstance(rejection, Mapping)
                and rejection.get("action_type") == self._decision_boundary_pending
            )
            if not rejected_pending:
                self._attempted.clear()
            self._decision_boundary_pending = None
        elif self._saw_served and not served:
            # Within one invocation the engine empties a non-empty served view
            # only after an accepted own effect. The record moved, so reads may
            # be attempted once again against the new derived context.
            self._attempted.clear()
        self._saw_served = served


def _batch(tools: Sequence[str]) -> RetrieveBatch:
    return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in tools))


def _outcome_read_key(outcome: AgentOutcome) -> str | None:
    """Which published schema an outcome's read requirements come from.

    Every business outcome the operation executes, not only ACT. An ASK sends a
    message, an ESCALATE opens a checkpoint and a COMPLETE ends the operation —
    all three are refused by handlers that read the record, so all three declare
    what they rest on. A WAIT changes nothing and commits nothing, so it rests on
    nothing and returns ``None``.
    """
    if isinstance(outcome, Act):
        return str(outcome.action_type)
    if isinstance(outcome, Ask):
        return "send_message"
    if isinstance(outcome, Escalate):
        return "request_exception_resolution"
    if isinstance(outcome, Complete):
        return COMPLETE_OUTCOME_KEY
    return None


def _published_required_reads(
    observation: AgentObservation, outcome_key: str | None
) -> list[str]:
    if outcome_key is None:
        return []
    schema = observation.action_schemas.get(outcome_key) or {}
    reads = schema.get("reads") or {}
    return sorted(tool for tool, strength in reads.items() if strength == READ_REQUIRED)


def _outstanding_reads(
    observation: AgentObservation, outcome: AgentOutcome
) -> tuple[str | None, list[str]]:
    """The outcome's key and required reads this invocation has not served."""
    key = _outcome_read_key(outcome)
    served = observation.retrieval.get("served") or {}
    return key, [
        tool for tool in _published_required_reads(observation, key) if tool not in served
    ]


def _served_view(observation: AgentObservation) -> ServedRecordView:
    return ServedRecordView(observation.retrieval.get("served") or {})


class RetrievingReferenceAgent:
    """The reference, running the same rules over records it asked for.

    The decision cascade is :class:`ReferenceAgent`'s, unchanged and delegated to
    — this is a planner in front of it, not a rewrite, which is the only way
    "the terminals are preserved" means anything. What changes is where the facts
    come from: a batch is proposed whenever this invocation has been served
    nothing, and after every accepted own effect the agent asks again rather than
    acting on a view of a document it has just moved.

    It keeps no memory between invocations and no cache within one. There is
    nowhere for a stale read to live.
    """

    agent_id = "reference"

    def __init__(self) -> None:
        self._inner = ReferenceAgent()
        self._planner = MaintenanceRetrievalPlanner()
        #: Batches the planner could not derive from the phase table alone.
        #: Counted rather than hidden: a top-up is the measurement that says the
        #: table is under-specified, and a planner that quietly repaired itself
        #: would report a clean run over a vocabulary that does not work.
        self.topup_batches = 0
        self.topup_causes: list[str] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)
        self._planner.begin_episode()
        self.topup_batches = 0
        self.topup_causes = []

    def phase_read_table(self) -> Mapping[str, tuple[str, ...]]:
        return self._planner.phase_read_table()

    def plan_retrieval(self, observation: AgentObservation) -> RetrieveBatch | None:
        return self._planner.plan_retrieval(observation)

    def decision_boundary(self, outcome_key: str) -> None:
        """Forward a business-outcome boundary to the retrieval planner."""
        self._planner.decision_boundary(outcome_key)

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        planned = self.plan_retrieval(observation)
        if planned is not None:
            return planned
        served = observation.retrieval["served"]
        try:
            outcome = self._inner.decide_from(
                ServedRecordView(served), observation.policy
            )
        except MissingRetrievedRoot as missing:
            tool = ROOT_TO_TOOL.get(missing.root)
            if tool is None or tool in served:
                # Nothing left to read. Wait rather than decide from an absence.
                return Wait(
                    reason="the record does not carry what this decision needs",
                    fallback_after_minutes=1440,
                )
            topup = self._planner.plan_tools(observation, (tool,))
            if topup is None:
                return Wait(
                    reason="a required record read failed in this invocation",
                    fallback_after_minutes=1440,
                )
            self.topup_batches += 1
            self.topup_causes.append(f"{observation.phase}:missing_root:{missing.root}")
            return topup
        # The action contract publishes what each action rests on. Satisfy it
        # from the contract, never by being refused: an agent that discovers its
        # read set by refusal is being measured on guessing the interface.
        key, outstanding = _outstanding_reads(observation, outcome)
        if outstanding:
            topup = self._planner.plan_tools(observation, outstanding)
            if topup is None:
                return Wait(
                    reason="a required action read failed in this invocation",
                    fallback_after_minutes=1440,
                )
            self.topup_batches += 1
            self.topup_causes.append(f"{observation.phase}:{key}:{outstanding}")
            return topup
        if key is not None:
            self.decision_boundary(key)
        return outcome


# --------------------------------------------------------------------- negatives


class _RetrievingNegative:
    """Shared plumbing for a negative that reads before it acts.

    Every deterministic agent in this build reaches the record the way a model
    does: one published catalogue, one batch object, one budget. A negative that
    kept a direct handle on the operation record would be a privileged path, and
    a benchmark with a privileged path is measuring two protocols rather than one.

    So a negative's *intervention* is what it decides, never how it reads. The
    ones below plan an ordinary batch and top up from the published reads exactly
    as the reference does; the ones whose intervention **is** a read failure —
    trusting a claim, holding a fact across a wake, repeating an accepted effect —
    deviate at that one point and nowhere else.
    """

    def __init__(self) -> None:
        self._planner = MaintenanceRetrievalPlanner()

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._planner.begin_episode()

    def _planned(self, observation: AgentObservation) -> RetrieveBatch | None:
        return self._planner.plan_retrieval(observation)

    def _gate(
        self, observation: AgentObservation, outcome: AgentOutcome
    ) -> AgentOutcome | RetrieveBatch:
        """Satisfy the outcome's published reads from the contract, not by refusal."""
        key, outstanding = _outstanding_reads(observation, outcome)
        if outstanding:
            topup = self._planner.plan_tools(observation, outstanding)
            if topup is not None:
                return topup
            return Wait(
                reason="a required action read failed in this invocation",
                fallback_after_minutes=1440,
            )
        if key is not None:
            self._planner.decision_boundary(key)
        return outcome


class CompleteEarlyAgent(_RetrievingNegative):
    """Declares victory instead of running the operation.

    Intended failure: a terminal proposal that the guard refuses because no work
    has authoritative verification — premature completion, over and over. It reads
    what the terminal publishes first, so what is refused is the *claim that the
    operation is finished* and not a failure to look.
    """

    agent_id = "complete_early"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        planned = self._planned(observation)
        if planned is not None:
            return planned
        return self._gate(observation, Complete(reason="the case looks finished to me"))


class AlwaysActAgent(_RetrievingNegative):
    """Acts on every turn and never waits for anything.

    Intended failure: it books a visit, then keeps booking the same visit,
    because it has no notion of waiting for the world to answer. It reads the
    record before each attempt, so the record plainly shows the visit it is about
    to request again — which is why the repetition is also a retrieval-discipline
    failure and not only a duplicate.
    """

    agent_id = "always_act"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        planned = self._planned(observation)
        if planned is not None:
            return planned
        state = _served_view(observation)
        try:
            cycle_id = state.get("current_cycle_id") or "work_cycle_1"
            cycle = _cycles(state).get(str(cycle_id), {"kind": "DIAGNOSTIC"})
        except MissingRetrievedRoot:
            cycle_id, cycle = "work_cycle_1", {"kind": "DIAGNOSTIC"}
        return self._gate(
            observation,
            Act(
                action_type="request_supplier_visit",
                payload={
                    "visit_type": cycle.get("kind", "DIAGNOSTIC"),
                    "cycle_id": cycle_id,
                    "scope_digest": cycle.get("scope_digest"),
                },
                rationale="doing something is better than waiting",
            ),
        )


class AlwaysWaitAgent:
    """Waits for an operator resolution it never causes, with no fallback.

    Intended failure: temporal non-completion. The wait itself is legal — it
    names a wake type this operation publishes, and the engine accepts it on the
    public contract, as it must: whether such an event is coming is authored
    future the agent cannot see. What fails is what the agent does next, which is
    nothing: it opens no checkpoint that could produce a resolution and declares
    no fallback deadline that would bound the wait, so the operation is still
    standing under it when the operational horizon runs out.

    It retrieves nothing, and that is correct rather than an omission: a WAIT
    changes nothing and commits nothing, so the read contract requires nothing of
    it. This control therefore leaves ``retrieval_discipline`` standing, which is
    what separates "never progressed" from "acted without looking".
    """

    agent_id = "always_wait"

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: AgentObservation) -> AgentOutcome:
        return Wait(
            reason="somebody will sort this out eventually",
            wake_on=WAKE_EXCEPTION,
        )


class AlwaysEscalateAgent(_RetrievingNegative):
    """Hands every invocation to a human, eligible or not.

    Intended failure: unnecessary human load. The state justifies no exception
    at all, so the checkpoint is refused as ineligible.
    """

    agent_id = "always_escalate"

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        planned = self._planned(observation)
        if planned is not None:
            return planned
        return self._gate(
            observation,
            Escalate(
                checkpoint_id=ESCALATION_LABEL,
                exception_type="APPROVAL_REJECTED",
                evidence_refs=(),
                deadline_after_minutes=1440,
                rationale="a human should look at this",
            ),
        )


class IgnoreNewEventsAgent(_RetrievingNegative):
    """Stops reading the record once a decision is outstanding.

    Intended failure: it keeps answering from the records it was served when it
    opened the approval checkpoint. After the decision arrives it never asks for
    anything again, declares the same legal wait from the frozen snapshot and
    remains non-terminal until the operational horizon expires.

    Its frozen snapshot only ever yields a WAIT, which rests on no read, so this
    control does *not* fail retrieval discipline — the distinction from
    ``stale_after_wake``, which proposes a business outcome on a fact it should
    have re-read, is exactly that.
    """

    agent_id = "ignore_new_events"

    def __init__(self) -> None:
        super().__init__()
        self._reference = ReferenceAgent()
        self._frozen: Mapping[str, Any] | None = None

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        super().begin_episode(identity)
        self._reference.begin_episode(identity)
        self._frozen = None

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        if self._frozen is not None:
            return self._gate(
                observation,
                self._reference.decide_from(
                    ServedRecordView(self._frozen), observation.policy
                ),
            )
        planned = self._planned(observation)
        if planned is not None:
            return planned
        served = observation.retrieval["served"]
        view = ServedRecordView(served)
        try:
            approvals = view["approvals"]
        except MissingRetrievedRoot:
            approvals = {}
        if approvals:
            self._frozen = dict(served)
        try:
            outcome = self._reference.decide_from(view, observation.policy)
        except MissingRetrievedRoot as missing:
            tool = ROOT_TO_TOOL.get(missing.root)
            if tool is None or tool in served:
                return Wait(
                    reason="the record does not carry what this decision needs",
                    fallback_after_minutes=1440,
                )
            topup = self._planner.plan_tools(observation, (tool,))
            if topup is not None:
                return topup
            return Wait(
                reason="a required record read failed in this invocation",
                fallback_after_minutes=1440,
            )
        return self._gate(observation, outcome)


class TrustActorClaimAgent:
    """Believes the supplier's word once, and asks to pay on the strength of it.

    Intended failure: it cites what the *event envelope* asserted as the evidence
    for a payment, without ever asking the verification service whether the
    record holds it. That is the whole distinction the retrieval protocol exists
    to draw — an event says someone asserted this, and only a retrieved
    authoritative result says the record holds it — and this control crosses it
    once and then runs the reference path to the authored terminal.
    """

    agent_id = "trust_actor_claim"

    def __init__(self) -> None:
        self._inner = RetrievingReferenceAgent()
        self._trusted = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)
        self._trusted = False

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        event = observation.event or {}
        payload: Mapping[str, Any] = event.get("payload") or {}
        if (
            not self._trusted
            and not event.get("is_authoritative", False)
            and str(payload.get("assertion")) == "WORK_COMPLETE"
        ):
            self._trusted = True
            self._inner.decision_boundary("request_payment")
            return Act(
                action_type="request_payment",
                payload={
                    "invoice_id": str(payload.get("invoice_id") or "invoice_1"),
                    "amount_minor": 0,
                    "currency": "GBP",
                },
                evidence_refs=(
                    str(payload.get("assertion_id") or event.get("event_id") or ""),
                ),
                rationale="the supplier says the work is complete",
            )
        return self._inner.decide(observation)


class StaleAfterWakeAgent:
    """Keeps one served fact across a wake and acts on it without re-reading.

    Intended failure: not an absence — it *did* read, and the record it read has
    since moved underneath it. Exactly once, on the first turn after a wake, it
    answers from the batch it was served in the previous invocation instead of
    asking again; the guard names that ``STALE_RETRIEVED_RECORD`` rather than
    ``ACTION_WITHOUT_RETRIEVAL``, and the evaluator reaches the same conclusion
    from the record on its own.

    It deliberately declines to make its one stale proposal an authoritative one.
    A proposal that needs the verification service *and* cites what an event
    asserted is the ``trust_actor_claim`` intervention wearing a different name;
    holding this control off that case is what keeps the two separable.
    """

    agent_id = "stale_after_wake"

    def __init__(self) -> None:
        self._inner = RetrievingReferenceAgent()
        self._rules = ReferenceAgent()
        self._cached: dict[str, Any] | None = None
        self._cached_invocation = 0
        self._spent = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)
        self._rules.begin_episode(identity)
        self._cached = None
        self._cached_invocation = 0
        self._spent = False

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        served = observation.retrieval.get("served") or {}
        if served:
            self._cached = dict(served)
            self._cached_invocation = observation.invocation_index
        stale = self._stale_outcome(observation, served)
        if stale is not None:
            self._spent = True
            key = _outcome_read_key(stale)
            if key is not None:
                self._inner.decision_boundary(key)
            return stale
        return self._inner.decide(observation)

    def _stale_outcome(
        self, observation: AgentObservation, served: Mapping[str, Any]
    ) -> AgentOutcome | None:
        if self._spent or served or self._cached is None:
            return None
        if observation.turn_index != 0:
            return None
        if observation.invocation_index <= self._cached_invocation:
            return None
        try:
            outcome = self._rules.decide_from(
                ServedRecordView(self._cached), observation.policy
            )
        except MissingRetrievedRoot:
            return None
        key = _outcome_read_key(outcome)
        if key is None:
            return None
        reads = _published_required_reads(observation, key)
        if not reads or "list_authoritative_records" in reads:
            return None
        return outcome


class DuplicateAfterWakeAgent:
    """Re-proposes its own last accepted action on the first turn after a wake.

    Intended failure: a causal duplicate. It reads freshly first — so this is not
    an absent-read or a stale-read failure — and then asks the operation to do
    again something the record it just read says is already done. The domain
    refuses it as a duplicate; independently of that refusal, the evaluator binds
    the repeated proposal to the earlier accepted effect and names the pattern.
    """

    agent_id = "duplicate_after_wake"

    def __init__(self) -> None:
        self._inner = RetrievingReferenceAgent()
        self._first_act: Act | None = None
        self._first_act_invocation = 0
        self._spent = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)
        self._first_act = None
        self._first_act_invocation = 0
        self._spent = False

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        served = observation.retrieval.get("served") or {}
        if (
            not self._spent
            and served
            and self._first_act is not None
            and observation.invocation_index > self._first_act_invocation
        ):
            self._spent = True
            self._inner.decision_boundary(self._first_act.action_type)
            return self._first_act
        outcome = self._inner.decide(observation)
        if (
            self._first_act is None
            and isinstance(outcome, Act)
            and outcome.action_type == "request_supplier_visit"
        ):
            self._first_act = outcome
            self._first_act_invocation = observation.invocation_index
        return outcome


@dataclass(frozen=True)
class AgentEntry:
    """One registered agent: what it is, how to build it, and where it runs.

    What it is *expected to break* is deliberately not here. ``targets`` and
    ``expected_findings`` are read-only views onto the oracle-declared control in
    the separate oracle manifest, so this registry can describe a behaviour
    but cannot author, widen or narrow the expectation the gate holds it to. An
    agent the oracle does not declare has no expectations at all — which is
    a failure of the gate, not a licence.
    """

    agent_id: str
    factory: Callable[[], Any]
    kind: str
    description: str
    scenarios: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expected_failure_closure(self) -> tuple[str, ...]:
        """The oracle's causal failure closure for this agent, or ``()``.

        Every dimension this behaviour is oracle-declared to take down, including
        the ones it takes down as a consequence of the one it was written for.
        The gate compares it as an exact set, so a failure outside the closure
        breaks it — that, and not "exactly one dimension failed", is what "for
        the intended reason without unrelated failures" means here.
        """
        control = oracle_declared_expectations(self.agent_id)
        return () if control is None else control.expected_failed_dimensions

    @property
    def targets(self) -> tuple[str, ...]:
        """:attr:`expected_failure_closure` under its shorter name."""
        return self.expected_failure_closure

    @property
    def expected_findings(self) -> tuple[str, ...]:
        """The oracle's required finding codes: what makes the failure intended."""
        control = oracle_declared_expectations(self.agent_id)
        return () if control is None else control.required_finding_codes

    @property
    def must_pass_dimensions(self) -> tuple[str, ...]:
        """The dimensions the oracle requires this behaviour to leave standing."""
        control = oracle_declared_expectations(self.agent_id)
        return () if control is None else control.must_pass_dimensions


REFERENCE_AGENT = AgentEntry(
    agent_id="reference",
    factory=RetrievingReferenceAgent,
    kind="reference",
    description="runs the authored oracle path on every required scenario",
    scenarios=("V1", "V2", "V3"),
)

#: The negatives this build ships. Each entry says what the agent *does*; what
#: it is required to break is oracle-declared in the oracle manifest, keyed by the
#: same ``agent_id``, and the suite asserts the two sets agree.
NEGATIVE_AGENTS: tuple[AgentEntry, ...] = (
    AgentEntry(
        agent_id="complete_early",
        factory=CompleteEarlyAgent,
        kind="negative",
        description="proposes a terminal transition before any work is verified",
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="always_act",
        factory=AlwaysActAgent,
        kind="negative",
        description="never waits; repeats an action the state has already accepted",
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="always_wait",
        factory=AlwaysWaitAgent,
        kind="negative",
        description=(
            "waits for an operator resolution it never causes, with no fallback "
            "deadline, until the operational horizon runs out"
        ),
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="always_escalate",
        factory=AlwaysEscalateAgent,
        kind="negative",
        description="asks for a human on every invocation, eligible or not",
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="ignore_new_events",
        factory=IgnoreNewEventsAgent,
        kind="negative",
        description="answers from stale state once a decision is outstanding",
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="trust_actor_claim",
        factory=TrustActorClaimAgent,
        kind="negative",
        description=(
            "cites what the event envelope asserted as the evidence for a payment, "
            "without ever reading the authoritative record"
        ),
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="stale_after_wake",
        factory=StaleAfterWakeAgent,
        kind="negative",
        description=(
            "holds one served fact across a wake and proposes a business outcome "
            "from it without reading again"
        ),
        scenarios=("V1",),
    ),
    AgentEntry(
        agent_id="duplicate_after_wake",
        factory=DuplicateAfterWakeAgent,
        kind="negative",
        description=(
            "reads freshly after a wake and then re-proposes the action its own "
            "last accepted effect already performed"
        ),
        scenarios=("V1",),
    ),
)

AGENTS: Mapping[str, AgentEntry] = {
    entry.agent_id: entry for entry in (REFERENCE_AGENT, *NEGATIVE_AGENTS)
}


def build_agent(agent_id: str) -> Any:
    """Construct a fresh agent by name. Episodes never share agent instances."""
    entry = AGENTS.get(agent_id)
    if entry is None:
        raise AgentRegistryError(
            f"no agent {agent_id!r} in this build; available agents are {sorted(AGENTS)}"
        )
    return entry.factory()


def agent_entry(agent_id: str) -> AgentEntry:
    entry = AGENTS.get(agent_id)
    if entry is None:
        raise AgentRegistryError(
            f"no agent {agent_id!r} in this build; available agents are {sorted(AGENTS)}"
        )
    return entry


def agent_ids() -> Sequence[str]:
    return sorted(AGENTS)


__all__ = [
    "AGENTS",
    "EVENT_TERM_READS",
    "NEGATIVE_AGENTS",
    "PHASE_READS",
    "REFERENCE_AGENT",
    "AgentEntry",
    "AlwaysActAgent",
    "AlwaysEscalateAgent",
    "AlwaysWaitAgent",
    "CompleteEarlyAgent",
    "DuplicateAfterWakeAgent",
    "IgnoreNewEventsAgent",
    "MaintenanceRetrievalPlanner",
    "MissingRetrievedRoot",
    "ReferenceAgent",
    "RetrievingReferenceAgent",
    "ServedRecordView",
    "StaleAfterWakeAgent",
    "TrustActorClaimAgent",
    "agent_entry",
    "agent_ids",
    "build_agent",
]
