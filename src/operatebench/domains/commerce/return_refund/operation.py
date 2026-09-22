"""Executable synthetic commerce return/refund operation.

Customer messages are claims.  Carrier handover, warehouse inspection and
payment settlement are distinct authoritative records.  A high-value refund
also requires one exact human checkpoint, and a settlement is provisional
until its cooldown expires without a later reversal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from operatebench.core.clock import parse_timestamp, shift_minutes
from operatebench.core.errors import SpecSchemaError
from operatebench.core.events import Event
from operatebench.core.protocol import (
    EnvironmentContext,
    EpisodePlan,
    PlannedEvent,
    PlannedTrigger,
    Verdict,
)
from operatebench.core.read_contract import (
    COMPLETE_OUTCOME_KEY,
    PayloadRegistryKey,
    ReadRequirementContract,
    resolve_required_evidence,
)
from operatebench.core.retrieval import (
    AUTHORITY_ACTOR_CLAIM,
    RetrievalRequest,
    ToolResult,
)
from operatebench.domains.commerce.return_refund import spec as return_refund_spec
from operatebench.domains.commerce.return_refund.retrieval import (
    RETURN_REFUND_RETRIEVAL_TOOLS,
    return_refund_retrieval_catalogue,
    return_refund_retrieval_record_contract,
    serve_return_refund_retrieval,
)
from operatebench.domains.commerce.return_refund.spec import (
    RETURN_REFUND_EVENT_TYPES,
    RETURN_REFUND_ROLE_AUTHORITY,
    RETURN_REFUND_WAKE_EVENT_TYPES,
    ReturnRefundSpec,
    ScenarioSpec,
    action_schema_view,
    validate_action_payload,
    validate_event_payload,
)
from operatebench.domains.commerce.return_refund.state import (
    AUTHORISATION_EXPIRED,
    AUTHORISATION_ISSUED,
    AUTHORISATION_VERIFIED,
    CHECKPOINT_APPROVED,
    CHECKPOINT_EXPIRED,
    CHECKPOINT_OPEN,
    CHECKPOINT_REJECTED,
    OBLIGATION_BREACHED,
    OBLIGATION_CANCELLED,
    OBLIGATION_DISCHARGED,
    OBLIGATION_OPEN,
    PHASE_AWAITING_APPROVAL,
    PHASE_AWAITING_HANDOVER,
    PHASE_AWAITING_INSPECTION,
    PHASE_FINAL_REFUND_DENIED,
    PHASE_FINAL_RETURN_EXPIRED,
    PHASE_FINAL_SUCCESS,
    PHASE_NEW,
    PHASE_PROVISIONAL,
    PHASE_READY_TO_REFUND,
    PHASE_REFUND_DENIED,
    PHASE_REFUND_PENDING,
    PHASE_REFUND_REOPENED,
    PHASE_REFUND_SETTLED,
    PHASE_RETURN_EXPIRED,
    PHASE_RETURN_REQUESTED,
    REFUND_REQUESTED,
    REFUND_REVERSED,
    REFUND_SETTLED,
    TERMINAL_CLOSED_REFUND_DENIED,
    TERMINAL_CLOSED_RETURN_EXPIRED,
    TERMINAL_COMPLETED_REFUND_SETTLED,
    ReturnRefundState,
)

ACTION_ISSUE_RETURN_AUTHORISATION = "issue_return_authorisation"
ACTION_REQUEST_REFUND_APPROVAL = "request_refund_approval"
ACTION_REQUEST_REFUND = "request_refund"
ACTION_SEND_MESSAGE = "send_message"
ACTION_COMPLETE = COMPLETE_OUTCOME_KEY
ACTION_TYPES: tuple[str, ...] = (
    ACTION_ISSUE_RETURN_AUTHORISATION,
    ACTION_REQUEST_REFUND_APPROVAL,
    ACTION_REQUEST_REFUND,
    ACTION_SEND_MESSAGE,
    ACTION_COMPLETE,
)

READ_GET_RETURN_CASE = "get_return_case"
READ_LIST_AUTHORISATIONS = "list_authorisations"
READ_LIST_CUSTOMER_CLAIMS = "list_customer_claims"
READ_LIST_CARRIER_RECORDS = "list_carrier_records"
READ_LIST_INSPECTIONS = "list_inspections"
READ_LIST_CHECKPOINTS = "list_checkpoints"
READ_LIST_REFUNDS = "list_refunds"
READ_LIST_COMMUNICATIONS = "list_communications"
READ_LIST_OBLIGATIONS = "list_obligations"
READ_TOOLS: tuple[str, ...] = tuple(sorted(RETURN_REFUND_RETRIEVAL_TOOLS))

EVENT_CUSTOMER_RETURN_REQUESTED = "customer_return_requested"
EVENT_CUSTOMER_HANDOVER_CLAIMED = "customer_handover_claimed"
EVENT_CARRIER_HANDOVER_VERIFIED = "carrier_handover_verified"
EVENT_WAREHOUSE_INSPECTION_COMPLETED = "warehouse_inspection_completed"
EVENT_REFUND_APPROVAL_RECEIVED = "refund_approval_received"
EVENT_REFUND_SETTLEMENT_CONFIRMED = "refund_settlement_confirmed"
EVENT_REFUND_SETTLEMENT_REVERSED = "refund_settlement_reversed"
EVENT_HANDOVER_REMINDER_DUE = "handover_reminder_due"
EVENT_HANDOVER_DEADLINE_DUE = "handover_deadline_due"
EVENT_APPROVAL_DEADLINE_DUE = "approval_deadline_due"
EVENT_RESOLUTION_COOLDOWN_EXPIRED = "resolution_cooldown_expired"
EVENT_TYPES: tuple[str, ...] = tuple(sorted(RETURN_REFUND_EVENT_TYPES))

MSG_HANDOVER_REMINDER = "msg_handover_reminder"
MSG_RETURN_EXPIRED = "msg_return_expired"
MSG_REFUND_COMPLETE = "msg_refund_complete"
MSG_REFUND_DENIED = "msg_refund_denied"

CHECKPOINT_HIGH_VALUE_REFUND = "high_value_refund_approval"
CUSTOMER_ACTOR_ID = "customer_1"
FIRST_REFUND_CYCLE = "refund_cycle_1"
RECOVERY_REFUND_CYCLE = "refund_cycle_2"
ELIGIBLE_DISPOSITION = "ELIGIBLE_FOR_REFUND"


class ReturnRefundOperation:
    """One scenario of the fully synthetic return/refund operation."""

    def __init__(self, spec: ReturnRefundSpec, scenario_id: str) -> None:
        self.spec = spec
        self.scenario: ScenarioSpec = spec.scenario(scenario_id)
        self.policy = spec.policy

    # ------------------------------------------------------------ plan/state

    def build_plan(self) -> EpisodePlan:
        events = tuple(
            PlannedEvent(
                event_id=authored.event_id,
                event_type=authored.event_type,
                actor_id=authored.actor_id,
                sequence=authored.sequence,
                payload=authored.payload_snapshot(),
                triggers_agent=authored.triggers_agent,
                at=authored.at,
                trigger=(
                    PlannedTrigger(
                        authored.trigger.kind,
                        authored.trigger.type_name,
                        authored.trigger.cycle_id,
                    )
                    if authored.trigger is not None
                    else None
                ),
                delay_minutes=authored.delay_minutes,
            )
            for authored in self.scenario.events
        )
        return EpisodePlan(
            starts_at=self.scenario.starts_at,
            events=events,
            horizon_minutes=self.policy.horizon_minutes,
            max_turns_per_invocation=self.policy.max_turns_per_invocation,
            max_retrieval_batches_per_invocation=(
                self.policy.max_retrieval_batches_per_invocation
            ),
        )

    def initial_state(self) -> ReturnRefundState:
        state = ReturnRefundState()
        state.hidden = self.spec.hidden_state_snapshot()
        return state

    def canonical_state(self, state: ReturnRefundState) -> Mapping[str, Any]:
        return state.canonical()

    def coarse_phase(self, state: ReturnRefundState) -> str:
        return state.phase

    # ------------------------------------------------------------- retrieval

    def event_authority(self, actor_id: str, event_type: str) -> str:
        actor = self.spec.actors.get(actor_id)
        role = actor.role if actor is not None else ""
        return RETURN_REFUND_ROLE_AUTHORITY.get(role, AUTHORITY_ACTOR_CLAIM)

    def retrieval_catalogue(self) -> Mapping[str, Mapping[str, Any]]:
        return return_refund_retrieval_catalogue()

    def retrieval_record_contract(self) -> list[str | int]:
        return return_refund_retrieval_record_contract()

    def read_requirements(self) -> ReadRequirementContract:
        return return_refund_spec.return_refund_action_evidence_contract()

    def serve_retrieval(
        self,
        state: ReturnRefundState,
        requests: Sequence[RetrievalRequest],
        as_of: str,
    ) -> Sequence[ToolResult]:
        roots = sorted(
            {
                root
                for request in requests
                for root in RETURN_REFUND_RETRIEVAL_TOOLS[request.tool].roots
            }
        )
        return serve_return_refund_retrieval(state.project(roots), requests, as_of)

    def policy_view(self) -> Mapping[str, Any]:
        return {
            "currency": self.policy.currency,
            "refund_amount_minor": self.policy.refund_amount_minor,
            "approval_threshold_minor": self.policy.approval_threshold_minor,
            "handover_reminder_after_minutes": (
                self.policy.handover_reminder_after_minutes
            ),
            "handover_deadline_after_minutes": (
                self.policy.handover_deadline_after_minutes
            ),
            "approval_deadline_after_minutes": (
                self.policy.approval_deadline_after_minutes
            ),
            "provisional_close_minutes": self.policy.provisional_close_minutes,
            "completion_notice_within_minutes": (
                self.policy.completion_notice_within_minutes
            ),
        }

    def actor_roles(self) -> Mapping[str, str]:
        return {actor_id: actor.role for actor_id, actor in self.spec.actors.items()}

    def message_fixture_ids(self) -> Sequence[str]:
        return tuple(sorted(self.spec.message_fixtures))

    def action_schemas(self) -> Mapping[str, Mapping[str, Any]]:
        agent = self.spec.actors["agent"]
        published = action_schema_view()
        offered = {
            action_type: schema
            for action_type, schema in published.items()
            if action_type == COMPLETE_OUTCOME_KEY or agent.holds(action_type)
        }
        return offered

    def wake_event_vocabulary(self) -> Sequence[str]:
        return RETURN_REFUND_WAKE_EVENT_TYPES

    def is_replay_final(self, state: ReturnRefundState) -> bool:
        return state.replay_final

    def terminal_outcome(self, state: ReturnRefundState) -> str | None:
        if state.terminal is None:
            return None
        outcome = state.terminal.get("outcome")
        return str(outcome) if outcome is not None else None

    def current_cycle_id(self, state: ReturnRefundState) -> str | None:
        return state.current_cycle_id

    # ---------------------------------------------------------------- events

    def reduce_event(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        actor = self.spec.actors.get(event.actor_id)
        if actor is None:
            return Verdict.refused("UNKNOWN_ACTOR", "event actor is not registered")
        required_authority = RETURN_REFUND_EVENT_TYPES.get(event.event_type)
        reducer = _EVENT_REDUCERS.get(event.event_type)
        if required_authority is None or reducer is None:
            return Verdict.refused(
                "UNKNOWN_EVENT_TYPE", f"no reducer for {event.event_type!r}"
            )
        if not actor.holds(required_authority):
            return Verdict.refused(
                "EVENT_ACTOR_LACKS_AUTHORITY",
                f"actor {event.actor_id!r} lacks {required_authority!r}",
            )
        try:
            validate_event_payload(
                event.event_type, event.payload, f"event {event.event_id}"
            )
        except SpecSchemaError as exc:
            return Verdict.refused("MALFORMED_EVENT_PAYLOAD", str(exc))
        return reducer(self, state, event, context)

    def _reduce_customer_return_requested(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        if state.phase != PHASE_NEW:
            return Verdict.refused("DUPLICATE_RETURN_REQUEST", "return already exists")
        payload = event.payload
        if (
            payload["amount_minor"] != self.policy.refund_amount_minor
            or payload["currency"] != self.policy.currency
            or payload["cycle_id"] != FIRST_REFUND_CYCLE
        ):
            return Verdict.refused(
                "RETURN_POLICY_MISMATCH",
                "the return must match the authored amount, currency and first cycle",
            )
        state.return_id = str(payload["return_id"])
        state.order_id = str(payload["order_id"])
        state.amount_minor = int(payload["amount_minor"])
        state.currency = str(payload["currency"])
        state.current_cycle_id = str(payload["cycle_id"])
        state.phase = PHASE_RETURN_REQUESTED
        context.record(
            "return_requested",
            {"return_id": state.return_id, "cycle_id": state.current_cycle_id},
        )
        return Verdict.ok(cycle_id=state.current_cycle_id)

    def _reduce_customer_handover_claimed(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        if (
            payload["return_id"] != state.return_id
            or payload["cycle_id"] != state.current_cycle_id
            or payload["authorisation_id"] not in state.authorisations
        ):
            return Verdict.refused(
                "CUSTOMER_CLAIM_BINDING_MISMATCH",
                "the claim does not name the current return authorisation",
            )
        claim_id = str(payload["claim_id"])
        if claim_id in state.customer_claims:
            return Verdict.refused("DUPLICATE_CUSTOMER_CLAIM", "claim id already exists")
        state.customer_claims[claim_id] = {
            "claim_id": claim_id,
            "return_id": state.return_id,
            "authorisation_id": str(payload["authorisation_id"]),
            "cycle_id": state.current_cycle_id,
            "claimed_at": context.now,
            "authority": "actor_claim",
        }
        context.record(
            "actor_claim_recorded",
            {"claim_id": claim_id, "authoritative": False},
        )
        return Verdict(
            accepted=True,
            code="CUSTOMER_CLAIM_RECORDED",
            cycle_id=state.current_cycle_id,
            observational=True,
        )

    def _reduce_carrier_handover_verified(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        authorisation_id = str(payload["authorisation_id"])
        authorisation = state.authorisations.get(authorisation_id)
        if authorisation is None:
            return Verdict.refused(
                "UNKNOWN_RETURN_AUTHORISATION", "no such authorisation"
            )
        if (
            payload["return_id"] != state.return_id
            or payload["cycle_id"] != authorisation["cycle_id"]
            or payload["handed_over"] is not True
        ):
            return Verdict.refused(
                "CARRIER_BINDING_MISMATCH",
                "carrier verification must affirm the current authorised return",
            )
        if authorisation["status"] != AUTHORISATION_ISSUED:
            return Verdict.refused(
                "RETURN_AUTHORISATION_NOT_OPEN",
                f"authorisation is {authorisation['status']}",
            )
        record_id = str(payload["carrier_record_id"])
        if record_id in state.carrier_records:
            return Verdict.refused("DUPLICATE_CARRIER_RECORD", "record id already exists")
        state.carrier_records[record_id] = {
            "carrier_record_id": record_id,
            "return_id": state.return_id,
            "authorisation_id": authorisation_id,
            "cycle_id": str(payload["cycle_id"]),
            "handed_over": True,
            "verified_at": context.now,
            "authority": "authoritative_verification",
        }
        authorisation["status"] = AUTHORISATION_VERIFIED
        authorisation["verified_at"] = context.now
        context.cancel_timer(_handover_reminder_timer(authorisation_id))
        context.cancel_timer(_handover_deadline_timer(authorisation_id))
        _close_obligation(
            state,
            context,
            _handover_obligation(authorisation_id),
            OBLIGATION_DISCHARGED,
        )
        _close_obligation(
            state,
            context,
            _handover_reminder_obligation(authorisation_id),
            OBLIGATION_CANCELLED,
        )
        state.phase = PHASE_AWAITING_INSPECTION
        return Verdict.ok(
            cycle_id=state.current_cycle_id,
            bindings={"authorisation_id": authorisation_id},
        )

    def _reduce_warehouse_inspection_completed(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        if state.phase != PHASE_AWAITING_INSPECTION:
            return Verdict.refused(
                "INSPECTION_NOT_EXPECTED",
                f"warehouse inspection cannot be accepted from {state.phase}",
            )
        if not _has_verified_handover(state):
            return Verdict.refused(
                "HANDOVER_NOT_VERIFIED",
                "warehouse inspection cannot precede authoritative carrier handover",
            )
        cycle_id = state.current_cycle_id
        if cycle_id is None:
            return Verdict.refused(
                "NO_ACTIVE_REFUND_CYCLE",
                "warehouse inspection arrived without an active return cycle",
            )
        if (
            payload["return_id"] != state.return_id
            or payload["cycle_id"] != cycle_id
            or payload["amount_minor"] != state.amount_minor
            or payload["currency"] != state.currency
        ):
            return Verdict.refused(
                "INSPECTION_BINDING_MISMATCH",
                "inspection does not match the current return, amount and currency",
            )
        inspection_id = str(payload["inspection_id"])
        if inspection_id in state.inspections:
            return Verdict.refused("DUPLICATE_INSPECTION", "inspection already exists")
        disposition = str(payload["disposition"])
        state.inspections[inspection_id] = {
            "inspection_id": inspection_id,
            "return_id": state.return_id,
            "cycle_id": cycle_id,
            "disposition": disposition,
            "amount_minor": state.amount_minor,
            "currency": state.currency,
            "inspected_at": context.now,
            "authority": "authoritative_verification",
        }
        if disposition != ELIGIBLE_DISPOSITION:
            state.phase = PHASE_REFUND_DENIED
            _open_obligation(
                state,
                context,
                _denial_notice_obligation(cycle_id),
                "refund_denial_notice",
                cycle_id,
                self.policy.completion_notice_within_minutes,
            )
        elif int(state.amount_minor or 0) > self.policy.approval_threshold_minor:
            state.phase = PHASE_AWAITING_APPROVAL
        else:
            state.phase = PHASE_READY_TO_REFUND
        return Verdict.ok(cycle_id=cycle_id)

    def _reduce_refund_approval_received(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        checkpoint_id = str(payload["checkpoint_id"])
        checkpoint = state.checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return Verdict.refused("UNKNOWN_CHECKPOINT", "no such refund checkpoint")
        if checkpoint["status"] != CHECKPOINT_OPEN:
            return Verdict.refused("CHECKPOINT_NOT_OPEN", "checkpoint is not open")
        expected = {
            "inspection_id": checkpoint["inspection_id"],
            "cycle_id": checkpoint["cycle_id"],
            "amount_minor": checkpoint["amount_minor"],
            "currency": checkpoint["currency"],
        }
        if any(payload[name] != value for name, value in expected.items()):
            return Verdict.refused(
                "APPROVAL_BINDING_MISMATCH",
                "decision does not match the checkpoint's inspection and amount",
            )
        decision = str(payload["decision"])
        if decision not in {"APPROVED", "REJECTED"}:
            return Verdict.refused("UNKNOWN_APPROVAL_DECISION", "unknown decision")
        checkpoint["decision"] = decision
        checkpoint["status"] = (
            CHECKPOINT_APPROVED if decision == "APPROVED" else CHECKPOINT_REJECTED
        )
        checkpoint["resolved_at"] = context.now
        context.cancel_timer(_approval_deadline_timer(checkpoint_id))
        _close_obligation(
            state,
            context,
            _approval_obligation(checkpoint_id),
            OBLIGATION_DISCHARGED,
        )
        if decision == "APPROVED":
            state.phase = PHASE_READY_TO_REFUND
            code = "REFUND_APPROVED"
        else:
            state.phase = PHASE_REFUND_DENIED
            _open_obligation(
                state,
                context,
                _denial_notice_obligation(str(checkpoint["cycle_id"])),
                "refund_denial_notice",
                str(checkpoint["cycle_id"]),
                self.policy.completion_notice_within_minutes,
            )
            code = "REFUND_DENIED"
        return Verdict.ok(code=code, cycle_id=str(checkpoint["cycle_id"]))

    def _reduce_refund_settlement_confirmed(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        request_id = str(payload["refund_request_id"])
        refund = state.refunds.get(request_id)
        if refund is None:
            return Verdict.refused("UNKNOWN_REFUND_REQUEST", "no such refund request")
        if refund["status"] != REFUND_REQUESTED:
            return Verdict.refused(
                "REFUND_REQUEST_NOT_OPEN", f"refund is {refund['status']}"
            )
        expected = {
            "refund_request_id": request_id,
            "inspection_id": refund["inspection_id"],
            "cycle_id": refund["cycle_id"],
            "amount_minor": refund["amount_minor"],
            "currency": refund["currency"],
        }
        if any(payload[name] != value for name, value in expected.items()):
            return Verdict.refused(
                "REFUND_SETTLEMENT_BINDING_MISMATCH",
                "settlement does not match the exact refund request",
            )
        refund["status"] = REFUND_SETTLED
        refund["settlement_id"] = str(payload["settlement_id"])
        refund["settled_at"] = context.now
        state.phase = PHASE_REFUND_SETTLED
        _close_obligation(
            state,
            context,
            _settlement_obligation(request_id),
            OBLIGATION_DISCHARGED,
        )
        _open_obligation(
            state,
            context,
            _completion_notice_obligation(str(refund["cycle_id"])),
            "refund_completion_notice",
            str(refund["cycle_id"]),
            self.policy.completion_notice_within_minutes,
        )
        return Verdict.ok(cycle_id=str(refund["cycle_id"]))

    def _reduce_refund_settlement_reversed(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        request_id = str(payload["refund_request_id"])
        refund = state.refunds.get(request_id)
        if refund is None:
            return Verdict.refused("UNKNOWN_REFUND_REQUEST", "no such refund request")
        if (
            refund["status"] != REFUND_SETTLED
            or payload["cycle_id"] != refund["cycle_id"]
            or state.provisional is None
            or state.provisional.get("refund_request_id") != request_id
        ):
            return Verdict.refused(
                "REVERSAL_BINDING_MISMATCH",
                "reversal must name the provisionally closed settled refund",
            )
        refund["status"] = REFUND_REVERSED
        refund["reversal_id"] = str(payload["reversal_id"])
        refund["reversal_reason_code"] = str(payload["reason_code"])
        refund["reversed_at"] = context.now
        context.cancel_timer(str(state.provisional["timer_event_id"]))
        state.provisional = None
        state.current_cycle_id = RECOVERY_REFUND_CYCLE
        state.phase = PHASE_REFUND_REOPENED
        _open_obligation(
            state,
            context,
            _recovery_obligation(RECOVERY_REFUND_CYCLE),
            "refund_recovery",
            RECOVERY_REFUND_CYCLE,
            self.policy.horizon_minutes,
        )
        context.record(
            "refund_reopened",
            {
                "reversed_refund_request_id": request_id,
                "new_cycle_id": RECOVERY_REFUND_CYCLE,
            },
        )
        return Verdict.ok(code="REFUND_REOPENED", cycle_id=RECOVERY_REFUND_CYCLE)

    def _reduce_handover_reminder_due(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        authorisation_id = str(event.payload["authorisation_id"])
        timer_problem = _timer_event_verdict(
            event, _handover_reminder_timer(authorisation_id)
        )
        if timer_problem is not None:
            return timer_problem
        authorisation = state.authorisations.get(authorisation_id)
        if authorisation is None or authorisation["status"] != AUTHORISATION_ISSUED:
            return Verdict.refused(
                "RETURN_AUTHORISATION_NOT_OPEN", "reminder has no open authorisation"
            )
        if event.payload["cycle_id"] != authorisation["cycle_id"]:
            return Verdict.refused("TIMER_BINDING_MISMATCH", "wrong reminder cycle")
        authorisation["reminder_due"] = True
        _open_obligation(
            state,
            context,
            _handover_reminder_obligation(authorisation_id),
            "handover_reminder",
            str(authorisation["cycle_id"]),
            self.policy.handover_deadline_after_minutes
            - self.policy.handover_reminder_after_minutes,
        )
        return Verdict.ok(cycle_id=str(authorisation["cycle_id"]))

    def _reduce_handover_deadline_due(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        authorisation_id = str(event.payload["authorisation_id"])
        timer_problem = _timer_event_verdict(
            event, _handover_deadline_timer(authorisation_id)
        )
        if timer_problem is not None:
            return timer_problem
        authorisation = state.authorisations.get(authorisation_id)
        if authorisation is None or authorisation["status"] != AUTHORISATION_ISSUED:
            return Verdict.refused(
                "RETURN_AUTHORISATION_NOT_OPEN", "deadline has no open authorisation"
            )
        if event.payload["cycle_id"] != authorisation["cycle_id"]:
            return Verdict.refused("TIMER_BINDING_MISMATCH", "wrong deadline cycle")
        authorisation["status"] = AUTHORISATION_EXPIRED
        authorisation["expired_at"] = context.now
        state.phase = PHASE_RETURN_EXPIRED
        _close_obligation(
            state,
            context,
            _handover_obligation(authorisation_id),
            OBLIGATION_BREACHED,
        )
        _close_obligation(
            state,
            context,
            _handover_reminder_obligation(authorisation_id),
            OBLIGATION_BREACHED,
        )
        _open_obligation(
            state,
            context,
            _expiry_notice_obligation(str(authorisation["cycle_id"])),
            "return_expiry_notice",
            str(authorisation["cycle_id"]),
            self.policy.completion_notice_within_minutes,
        )
        return Verdict.ok(code="RETURN_EXPIRED", cycle_id=str(authorisation["cycle_id"]))

    def _reduce_approval_deadline_due(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        checkpoint_id = str(event.payload["checkpoint_id"])
        timer_problem = _timer_event_verdict(
            event, _approval_deadline_timer(checkpoint_id)
        )
        if timer_problem is not None:
            return timer_problem
        checkpoint = state.checkpoints.get(checkpoint_id)
        if checkpoint is None or checkpoint["status"] != CHECKPOINT_OPEN:
            return Verdict.refused("CHECKPOINT_NOT_OPEN", "approval deadline is stale")
        if event.payload["cycle_id"] != checkpoint["cycle_id"]:
            return Verdict.refused("TIMER_BINDING_MISMATCH", "wrong checkpoint cycle")
        checkpoint["status"] = CHECKPOINT_EXPIRED
        checkpoint["resolved_at"] = context.now
        state.phase = PHASE_REFUND_DENIED
        _close_obligation(
            state,
            context,
            _approval_obligation(checkpoint_id),
            OBLIGATION_BREACHED,
        )
        _open_obligation(
            state,
            context,
            _denial_notice_obligation(str(checkpoint["cycle_id"])),
            "refund_denial_notice",
            str(checkpoint["cycle_id"]),
            self.policy.completion_notice_within_minutes,
        )
        return Verdict.ok(
            code="REFUND_APPROVAL_EXPIRED", cycle_id=str(checkpoint["cycle_id"])
        )

    def _reduce_resolution_cooldown_expired(
        self, state: ReturnRefundState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        if state.phase != PHASE_PROVISIONAL or state.provisional is None:
            return Verdict.refused(
                "NO_PROVISIONAL_REFUND", "there is no provisional refund to finalize"
            )
        request_id = str(event.payload["refund_request_id"])
        cycle_id = str(event.payload["cycle_id"])
        timer_problem = _timer_event_verdict(
            event,
            str(state.provisional["timer_event_id"]),
            not_before=str(state.provisional["expires_at"]),
        )
        if timer_problem is not None:
            return timer_problem
        if (
            state.provisional.get("refund_request_id") != request_id
            or state.provisional.get("cycle_id") != cycle_id
        ):
            return Verdict.refused(
                "COOLDOWN_BINDING_MISMATCH", "cooldown names a different refund"
            )
        refund = state.refunds.get(request_id)
        if refund is None or refund["status"] != REFUND_SETTLED:
            return Verdict.refused(
                "REFUND_NOT_SETTLED", "a reversed or missing refund cannot finalize"
            )
        if state.open_checkpoints() or state.open_obligations():
            return Verdict.refused(
                "OPEN_WORK_AT_FINALIZATION", "checkpoints or obligations remain open"
            )
        state.provisional = None
        state.phase = PHASE_FINAL_SUCCESS
        state.replay_final = True
        state.terminal = {
            "outcome": TERMINAL_COMPLETED_REFUND_SETTLED,
            "at": context.now,
            "cycle_id": cycle_id,
            "refund_request_id": request_id,
        }
        context.record(
            "terminal_accepted",
            {
                "outcome": TERMINAL_COMPLETED_REFUND_SETTLED,
                "replay_final": True,
                "cycle_id": cycle_id,
            },
        )
        return Verdict.ok(
            code="REFUND_FINALIZED",
            cycle_id=cycle_id,
            bindings={"refund_request_id": request_id},
        )

    # --------------------------------------------------------------- actions

    def apply_action(
        self,
        state: ReturnRefundState,
        action_type: str,
        payload: Mapping[str, Any],
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict:
        if state.replay_final:
            return Verdict.refused(
                "OPERATION_IS_REPLAY_FINAL", "operation accepts no more actions"
            )
        if not self.spec.actors["agent"].holds(action_type):
            return Verdict.refused(
                "ACTION_OUTSIDE_AGENT_AUTHORITY", f"agent lacks {action_type!r}"
            )
        handler = _ACTION_HANDLERS.get(action_type)
        if handler is None:
            return Verdict.refused("UNKNOWN_ACTION_TYPE", "unknown action")
        try:
            validate_action_payload(action_type, payload, f"action {action_type}")
        except SpecSchemaError as exc:
            return Verdict.refused("MALFORMED_ACTION_PAYLOAD", str(exc))
        unresolved = _unresolved_evidence(state, evidence_refs)
        if unresolved is not None:
            return unresolved
        return handler(self, state, payload, tuple(evidence_refs), context)

    def _act_issue_return_authorisation(
        self,
        state: ReturnRefundState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        if state.phase != PHASE_RETURN_REQUESTED:
            return Verdict.refused(
                "DUPLICATE_RETURN_AUTHORISATION",
                "a return authorisation may be issued exactly once",
            )
        if (
            payload["return_id"] != state.return_id
            or payload["cycle_id"] != state.current_cycle_id
        ):
            return Verdict.refused(
                "RETURN_AUTHORISATION_BINDING_MISMATCH",
                "authorisation must name the current return and cycle",
            )
        authorisation_id = _allocate("return_authorisation", set(state.authorisations))
        state.authorisations[authorisation_id] = {
            "authorisation_id": authorisation_id,
            "return_id": state.return_id,
            "cycle_id": state.current_cycle_id,
            "status": AUTHORISATION_ISSUED,
            "issued_at": context.now,
            "verified_at": None,
            "expired_at": None,
            "reminder_due": False,
        }
        state.phase = PHASE_AWAITING_HANDOVER
        context.schedule_timer(
            _handover_reminder_timer(authorisation_id),
            EVENT_HANDOVER_REMINDER_DUE,
            "scheduler",
            self.policy.handover_reminder_after_minutes,
            {"authorisation_id": authorisation_id, "cycle_id": state.current_cycle_id},
        )
        context.schedule_timer(
            _handover_deadline_timer(authorisation_id),
            EVENT_HANDOVER_DEADLINE_DUE,
            "scheduler",
            self.policy.handover_deadline_after_minutes,
            {"authorisation_id": authorisation_id, "cycle_id": state.current_cycle_id},
        )
        _open_obligation(
            state,
            context,
            _handover_obligation(authorisation_id),
            "authoritative_handover",
            state.current_cycle_id,
            self.policy.handover_deadline_after_minutes,
        )
        return Verdict.ok(
            cycle_id=state.current_cycle_id,
            bindings={"authorisation_id": authorisation_id},
        )

    def _act_request_refund_approval(
        self,
        state: ReturnRefundState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        inspection = state.inspections.get(str(payload["inspection_id"]))
        if (
            inspection is None
            or inspection.get("disposition") != ELIGIBLE_DISPOSITION
            or int(inspection.get("amount_minor", 0))
            <= self.policy.approval_threshold_minor
            or state.phase != PHASE_AWAITING_APPROVAL
        ):
            return Verdict.refused(
                "UNNECESSARY_CHECKPOINT_ATTEMPT",
                "no eligible above-threshold authoritative inspection needs review",
            )
        if payload["cycle_id"] != state.current_cycle_id:
            return Verdict.refused("STALE_REFUND_CYCLE", "approval names a stale cycle")
        if any(
            checkpoint.get("inspection_id") == inspection["inspection_id"]
            for checkpoint in state.checkpoints.values()
        ):
            return Verdict.refused(
                "DUPLICATE_CHECKPOINT", "inspection already has review"
            )
        citation = _required_evidence_verdict(
            state,
            ACTION_REQUEST_REFUND_APPROVAL,
            payload,
            evidence_refs,
            "MISSING_INSPECTION_EVIDENCE",
        )
        if citation is not None:
            return citation
        checkpoint_id = _allocate("refund_checkpoint", set(state.checkpoints))
        deadline_at = shift_minutes(
            context.now,
            self.policy.approval_deadline_after_minutes,
            context="refund approval deadline",
        )
        state.checkpoints[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": CHECKPOINT_HIGH_VALUE_REFUND,
            "inspection_id": inspection["inspection_id"],
            "cycle_id": state.current_cycle_id,
            "amount_minor": state.amount_minor,
            "currency": state.currency,
            "status": CHECKPOINT_OPEN,
            "decision": None,
            "opened_at": context.now,
            "deadline_at": deadline_at,
            "resolved_at": None,
        }
        context.schedule_timer(
            _approval_deadline_timer(checkpoint_id),
            EVENT_APPROVAL_DEADLINE_DUE,
            "scheduler",
            self.policy.approval_deadline_after_minutes,
            {"checkpoint_id": checkpoint_id, "cycle_id": state.current_cycle_id},
        )
        _open_obligation(
            state,
            context,
            _approval_obligation(checkpoint_id),
            "refund_approval_decision",
            state.current_cycle_id,
            self.policy.approval_deadline_after_minutes,
        )
        context.record(
            "checkpoint_opened",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_HIGH_VALUE_REFUND,
                "required": True,
            },
        )
        return Verdict.ok(
            cycle_id=state.current_cycle_id,
            bindings={"checkpoint_id": checkpoint_id},
        )

    def _act_request_refund(
        self,
        state: ReturnRefundState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        cycle_id = str(payload["cycle_id"])
        if cycle_id != state.current_cycle_id:
            return Verdict.refused("STALE_REFUND_CYCLE", "refund names a stale cycle")
        if any(refund.get("cycle_id") == cycle_id for refund in state.refunds.values()):
            return Verdict.refused(
                "DUPLICATE_REFUND_REQUEST", "this refund cycle already has a request"
            )
        inspection = state.inspections.get(str(payload["inspection_id"]))
        if inspection is None or inspection.get("disposition") != ELIGIBLE_DISPOSITION:
            return Verdict.refused(
                "INSPECTION_NOT_ELIGIBLE", "refund needs an eligible inspection"
            )
        if not _has_verified_handover(state):
            return Verdict.refused(
                "HANDOVER_NOT_VERIFIED", "customer claim cannot replace carrier record"
            )
        if int(state.amount_minor or 0) > self.policy.approval_threshold_minor:
            checkpoint_id = payload.get("approval_checkpoint_id")
            checkpoint = (
                state.checkpoints.get(str(checkpoint_id))
                if isinstance(checkpoint_id, str)
                else None
            )
            if checkpoint is None:
                return Verdict.refused(
                    "REFUND_APPROVAL_BYPASSED",
                    "above-threshold refund has no matching approved checkpoint",
                )
            if (
                checkpoint.get("status") != CHECKPOINT_APPROVED
                or checkpoint.get("inspection_id") != inspection["inspection_id"]
                or checkpoint.get("amount_minor") != state.amount_minor
                or checkpoint.get("currency") != state.currency
            ):
                return Verdict.refused(
                    "REFUND_APPROVAL_BYPASSED",
                    "checkpoint does not approve this exact inspection and amount",
                )
        if state.phase not in {PHASE_READY_TO_REFUND, PHASE_REFUND_REOPENED}:
            return Verdict.refused(
                "REFUND_NOT_READY", f"refund cannot be requested from {state.phase}"
            )
        citation = _required_evidence_verdict(
            state,
            ACTION_REQUEST_REFUND,
            payload,
            evidence_refs,
            "MISSING_INSPECTION_EVIDENCE",
        )
        if citation is not None:
            return citation
        request_id = _allocate("refund_request", set(state.refunds))
        state.refunds[request_id] = {
            "refund_request_id": request_id,
            "inspection_id": inspection["inspection_id"],
            "approval_checkpoint_id": payload.get("approval_checkpoint_id"),
            "cycle_id": cycle_id,
            "amount_minor": state.amount_minor,
            "currency": state.currency,
            "status": REFUND_REQUESTED,
            "requested_at": context.now,
            "settlement_id": None,
            "settled_at": None,
            "reversal_id": None,
            "reversal_reason_code": None,
            "reversed_at": None,
        }
        state.phase = PHASE_REFUND_PENDING
        _close_obligation(
            state,
            context,
            _recovery_obligation(cycle_id),
            OBLIGATION_DISCHARGED,
        )
        _open_obligation(
            state,
            context,
            _settlement_obligation(request_id),
            "authoritative_refund_settlement",
            cycle_id,
            self.policy.horizon_minutes,
        )
        return Verdict.ok(cycle_id=cycle_id, bindings={"refund_request_id": request_id})

    def _act_send_message(
        self,
        state: ReturnRefundState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        recipient = str(payload["recipient_actor_id"])
        fixture_id = str(payload["message_fixture_id"])
        cycle_id = str(payload.get("cycle_id") or state.current_cycle_id or "")
        if recipient not in self.spec.actors:
            return Verdict.refused("UNKNOWN_RECIPIENT", "recipient is not registered")
        if recipient != CUSTOMER_ACTOR_ID:
            return Verdict.refused(
                "MESSAGE_RECIPIENT_MISMATCH",
                "return/refund operational notices must be sent to the customer",
            )
        if fixture_id not in self.spec.message_fixtures:
            return Verdict.refused("UNKNOWN_MESSAGE_FIXTURE", "message is not declared")
        if any(
            item.get("message_fixture_id") == fixture_id
            and item.get("cycle_id") == cycle_id
            for item in state.communications.values()
        ):
            return Verdict.refused("DUPLICATE_MESSAGE", "message already committed")
        obligation_id: str
        if fixture_id == MSG_HANDOVER_REMINDER:
            authorisation = _current_authorisation(state)
            if authorisation is None or authorisation.get("reminder_due") is not True:
                return Verdict.refused("REMINDER_NOT_DUE", "handover reminder is not due")
            obligation_id = _handover_reminder_obligation(
                str(authorisation["authorisation_id"])
            )
        elif fixture_id == MSG_RETURN_EXPIRED:
            if state.phase != PHASE_RETURN_EXPIRED:
                return Verdict.refused("RETURN_NOT_EXPIRED", "no expiry to announce")
            obligation_id = _expiry_notice_obligation(cycle_id)
        elif fixture_id == MSG_REFUND_COMPLETE:
            if state.phase != PHASE_REFUND_SETTLED:
                return Verdict.refused("REFUND_NOT_SETTLED", "no settlement to announce")
            obligation_id = _completion_notice_obligation(cycle_id)
        elif fixture_id == MSG_REFUND_DENIED:
            if state.phase != PHASE_REFUND_DENIED:
                return Verdict.refused("REFUND_NOT_DENIED", "no denial to announce")
            obligation_id = _denial_notice_obligation(cycle_id)
        else:
            return Verdict.refused(
                "MESSAGE_HAS_NO_OPERATIONAL_PURPOSE",
                "fixture is diagnostic and has no valid transition",
            )
        obligation = state.obligations.get(obligation_id)
        if obligation is None or obligation.get("status") != OBLIGATION_OPEN:
            return Verdict.refused("MESSAGE_OBLIGATION_NOT_OPEN", "message is not owed")
        communication_id = _allocate("communication", set(state.communications))
        dispatched = not context.dispatch_fails(fixture_id)
        state.communications[communication_id] = {
            "communication_id": communication_id,
            "recipient_actor_id": recipient,
            "message_fixture_id": fixture_id,
            "cycle_id": cycle_id,
            "committed_at": context.now,
            "dispatched": dispatched,
        }
        _close_obligation(state, context, obligation_id, OBLIGATION_DISCHARGED)
        context.record(
            "side_effect" if dispatched else "side_effect_failed",
            {
                "channel": "message_dispatch",
                "message_fixture_id": fixture_id,
                "recipient_actor_id": recipient,
                "committed_decision_preserved": True,
                "status": "DELIVERED" if dispatched else "FAILED",
            },
        )
        return Verdict.ok(
            cycle_id=cycle_id,
            bindings={"communication_id": communication_id},
        )

    # -------------------------------------------------------------- terminal

    def apply_terminal(
        self,
        state: ReturnRefundState,
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict:
        if state.replay_final:
            return Verdict.refused("OPERATION_IS_REPLAY_FINAL", "operation has ended")
        unresolved = _unresolved_evidence(state, evidence_refs)
        if unresolved is not None:
            return unresolved
        if state.phase == PHASE_PROVISIONAL:
            return Verdict.refused(
                "ALREADY_PROVISIONAL", "only the cooldown can finalize this refund"
            )
        open_checkpoints = sorted(
            str(item["checkpoint_id"]) for item in state.open_checkpoints()
        )
        if open_checkpoints:
            return Verdict.refused(
                "OPEN_CHECKPOINT", f"open checkpoint(s): {open_checkpoints}"
            )
        open_obligations = sorted(
            str(item["obligation_id"]) for item in state.open_obligations()
        )
        if open_obligations:
            return Verdict.refused(
                "OPEN_OBLIGATION", f"open obligation(s): {open_obligations}"
            )
        if state.phase == PHASE_REFUND_DENIED:
            return self._finalize_immediately(
                state,
                context,
                PHASE_FINAL_REFUND_DENIED,
                TERMINAL_CLOSED_REFUND_DENIED,
                "REFUND_DENIAL_CLOSED",
            )
        if state.phase == PHASE_RETURN_EXPIRED:
            return self._finalize_immediately(
                state,
                context,
                PHASE_FINAL_RETURN_EXPIRED,
                TERMINAL_CLOSED_RETURN_EXPIRED,
                "RETURN_EXPIRY_CLOSED",
            )
        if state.phase != PHASE_REFUND_SETTLED:
            return Verdict.refused(
                "PREMATURE_COMPLETION", f"cannot complete from {state.phase}"
            )
        refund = state.current_refund()
        if refund is None or refund.get("status") != REFUND_SETTLED:
            return Verdict.refused(
                "REFUND_NOT_SETTLED", "current cycle has no settled refund"
            )
        if not _has_verified_handover(state):
            return Verdict.refused("HANDOVER_NOT_VERIFIED", "carrier record is absent")
        inspection = state.inspections.get(str(refund["inspection_id"]))
        if inspection is None or inspection.get("disposition") != ELIGIBLE_DISPOSITION:
            return Verdict.refused(
                "INSPECTION_NOT_ELIGIBLE", "inspection is not eligible"
            )
        if int(refund["amount_minor"]) > self.policy.approval_threshold_minor:
            checkpoint = state.checkpoints.get(str(refund.get("approval_checkpoint_id")))
            if checkpoint is None or checkpoint.get("status") != CHECKPOINT_APPROVED:
                return Verdict.refused(
                    "REFUND_APPROVAL_BYPASSED", "settled refund lacks approval"
                )
        if not _message_committed(state, MSG_REFUND_COMPLETE, str(refund["cycle_id"])):
            return Verdict.refused(
                "REFUND_NOTICE_NOT_SENT", "customer has not been told of settlement"
            )
        timer_id = _cooldown_timer(str(refund["refund_request_id"]))
        expires_at = shift_minutes(
            context.now,
            self.policy.provisional_close_minutes,
            context="refund resolution cooldown",
        )
        state.phase = PHASE_PROVISIONAL
        state.provisional = {
            "refund_request_id": refund["refund_request_id"],
            "cycle_id": refund["cycle_id"],
            "accepted_at": context.now,
            "expires_at": expires_at,
            "timer_event_id": timer_id,
        }
        context.schedule_timer(
            timer_id,
            EVENT_RESOLUTION_COOLDOWN_EXPIRED,
            "scheduler",
            self.policy.provisional_close_minutes,
            {
                "refund_request_id": refund["refund_request_id"],
                "cycle_id": refund["cycle_id"],
            },
        )
        context.record(
            "provisional_close_entered",
            {
                "refund_request_id": refund["refund_request_id"],
                "cycle_id": refund["cycle_id"],
                "expires_at": expires_at,
                "replay_final": False,
            },
        )
        return Verdict.ok(
            code="PROVISIONAL_REFUND_CLOSE",
            cycle_id=str(refund["cycle_id"]),
            bindings={"refund_request_id": str(refund["refund_request_id"])},
        )

    def _finalize_immediately(
        self,
        state: ReturnRefundState,
        context: EnvironmentContext,
        phase: str,
        outcome: str,
        code: str,
    ) -> Verdict:
        state.phase = phase
        state.replay_final = True
        state.terminal = {
            "outcome": outcome,
            "at": context.now,
            "cycle_id": state.current_cycle_id,
        }
        context.record(
            "terminal_accepted",
            {
                "outcome": outcome,
                "replay_final": True,
                "cycle_id": state.current_cycle_id,
            },
        )
        return Verdict.ok(code=code, cycle_id=state.current_cycle_id)


# ----------------------------------------------------------------- helpers


def _allocate(prefix: str, used: set[str]) -> str:
    index = len(used) + 1
    while f"{prefix}_{index}" in used:
        index += 1
    return f"{prefix}_{index}"


def _current_authorisation(state: ReturnRefundState) -> dict[str, Any] | None:
    candidates = [
        item
        for item in state.authorisations.values()
        if item.get("cycle_id") == FIRST_REFUND_CYCLE
    ]
    return candidates[-1] if candidates else None


def _has_verified_handover(state: ReturnRefundState) -> bool:
    return any(
        item.get("handed_over") is True
        and item.get("authority") == "authoritative_verification"
        for item in state.carrier_records.values()
    )


def _message_committed(state: ReturnRefundState, fixture_id: str, cycle_id: str) -> bool:
    return any(
        item.get("message_fixture_id") == fixture_id
        and item.get("cycle_id") == cycle_id
        and item.get("recipient_actor_id") == CUSTOMER_ACTOR_ID
        for item in state.communications.values()
    )


def _timer_event_verdict(
    event: Event,
    expected_event_id: str,
    *,
    not_before: str | None = None,
) -> Verdict | None:
    if (
        event.caused_by != "timer"
        or event.actor_id != "scheduler"
        or event.event_id != expected_event_id
    ):
        return Verdict.refused(
            "TIMER_PROVENANCE_MISMATCH",
            "timer transition requires the exact runtime-scheduled event identity",
        )
    if not_before is not None and parse_timestamp(
        event.at, f"timer event {event.event_id}"
    ) < parse_timestamp(not_before, f"timer event {event.event_id} lower bound"):
        return Verdict.refused(
            "TIMER_FIRED_EARLY",
            f"timer transition cannot occur before {not_before}",
        )
    return None


def _unresolved_evidence(
    state: ReturnRefundState, evidence_refs: Sequence[str]
) -> Verdict | None:
    claim_ids = set(state.customer_claims)
    claimed = sorted(set(evidence_refs) & claim_ids)
    if claimed:
        return Verdict.refused(
            "CLAIM_TREATED_AS_AUTHORITATIVE",
            "a customer's handover claim is not a carrier verification record",
        )
    resolvable = (
        set(state.authorisations)
        | set(state.carrier_records)
        | set(state.inspections)
        | set(state.checkpoints)
        | set(state.refunds)
        | set(state.communications)
        | set(state.obligations)
    )
    unknown = sorted(set(evidence_refs) - resolvable)
    if unknown:
        return Verdict.refused(
            "UNRESOLVED_EVIDENCE", f"evidence reference(s) do not resolve: {unknown}"
        )
    return None


def _required_evidence_verdict(
    state: ReturnRefundState,
    action_type: str,
    payload: Mapping[str, Any],
    evidence_refs: Sequence[str],
    code: str,
) -> Verdict | None:
    contract = return_refund_spec.return_refund_action_evidence_contract()
    roots = sorted(
        {
            selector.root
            for selector in contract.evidence_for(action_type)
            if isinstance(selector, PayloadRegistryKey)
        }
    )
    resolution = resolve_required_evidence(
        contract, action_type, payload, state.project(roots)
    )
    missing = sorted(set(resolution.required) - set(evidence_refs))
    if resolution.problem is None and not missing:
        return None
    return Verdict.refused(
        code,
        "required evidence did not resolve completely: "
        + (resolution.problem or f"missing {missing}"),
    )


def _open_obligation(
    state: ReturnRefundState,
    context: EnvironmentContext,
    obligation_id: str,
    kind: str,
    cycle_id: str | None,
    due_after_minutes: int,
) -> None:
    if obligation_id in state.obligations:
        return
    due_at = shift_minutes(
        context.now, due_after_minutes, context=f"obligation {obligation_id}"
    )
    state.obligations[obligation_id] = {
        "obligation_id": obligation_id,
        "kind": kind,
        "cycle_id": cycle_id,
        "status": OBLIGATION_OPEN,
        "opened_at": context.now,
        "due_at": due_at,
        "closed_at": None,
    }
    context.record(
        "obligation_created",
        {"obligation_id": obligation_id, "kind": kind, "due_at": due_at},
    )


def _close_obligation(
    state: ReturnRefundState,
    context: EnvironmentContext,
    obligation_id: str,
    status: str,
) -> None:
    obligation = state.obligations.get(obligation_id)
    if obligation is None or obligation.get("status") != OBLIGATION_OPEN:
        return
    if status == OBLIGATION_DISCHARGED and parse_timestamp(
        context.now, f"obligation {obligation_id} close"
    ) > parse_timestamp(str(obligation["due_at"]), f"obligation {obligation_id} due"):
        status = OBLIGATION_BREACHED
    obligation["status"] = status
    obligation["closed_at"] = context.now
    context.record(
        {
            OBLIGATION_DISCHARGED: "obligation_discharged",
            OBLIGATION_CANCELLED: "obligation_cancelled",
            OBLIGATION_BREACHED: "obligation_breached",
        }[status],
        {"obligation_id": obligation_id, "kind": obligation["kind"]},
    )


def _handover_reminder_timer(authorisation_id: str) -> str:
    return f"timer_handover_reminder:{authorisation_id}"


def _handover_deadline_timer(authorisation_id: str) -> str:
    return f"timer_handover_deadline:{authorisation_id}"


def _approval_deadline_timer(checkpoint_id: str) -> str:
    return f"timer_approval_deadline:{checkpoint_id}"


def _cooldown_timer(refund_request_id: str) -> str:
    return f"timer_resolution_cooldown:{refund_request_id}"


def _handover_obligation(authorisation_id: str) -> str:
    return f"handover_verification:{authorisation_id}"


def _handover_reminder_obligation(authorisation_id: str) -> str:
    return f"handover_reminder:{authorisation_id}"


def _approval_obligation(checkpoint_id: str) -> str:
    return f"refund_approval:{checkpoint_id}"


def _settlement_obligation(refund_request_id: str) -> str:
    return f"refund_settlement:{refund_request_id}"


def _completion_notice_obligation(cycle_id: str) -> str:
    return f"refund_completion_notice:{cycle_id}"


def _denial_notice_obligation(cycle_id: str) -> str:
    return f"refund_denial_notice:{cycle_id}"


def _expiry_notice_obligation(cycle_id: str) -> str:
    return f"return_expiry_notice:{cycle_id}"


def _recovery_obligation(cycle_id: str) -> str:
    return f"refund_recovery:{cycle_id}"


_EventReducer = Callable[
    [ReturnRefundOperation, ReturnRefundState, Event, EnvironmentContext], Verdict
]
_ActionHandler = Callable[
    [
        ReturnRefundOperation,
        ReturnRefundState,
        Mapping[str, Any],
        tuple[str, ...],
        EnvironmentContext,
    ],
    Verdict,
]

_EVENT_REDUCERS: Mapping[str, _EventReducer] = {
    EVENT_CUSTOMER_RETURN_REQUESTED: (
        ReturnRefundOperation._reduce_customer_return_requested
    ),
    EVENT_CUSTOMER_HANDOVER_CLAIMED: (
        ReturnRefundOperation._reduce_customer_handover_claimed
    ),
    EVENT_CARRIER_HANDOVER_VERIFIED: (
        ReturnRefundOperation._reduce_carrier_handover_verified
    ),
    EVENT_WAREHOUSE_INSPECTION_COMPLETED: (
        ReturnRefundOperation._reduce_warehouse_inspection_completed
    ),
    EVENT_REFUND_APPROVAL_RECEIVED: (
        ReturnRefundOperation._reduce_refund_approval_received
    ),
    EVENT_REFUND_SETTLEMENT_CONFIRMED: (
        ReturnRefundOperation._reduce_refund_settlement_confirmed
    ),
    EVENT_REFUND_SETTLEMENT_REVERSED: (
        ReturnRefundOperation._reduce_refund_settlement_reversed
    ),
    EVENT_HANDOVER_REMINDER_DUE: ReturnRefundOperation._reduce_handover_reminder_due,
    EVENT_HANDOVER_DEADLINE_DUE: ReturnRefundOperation._reduce_handover_deadline_due,
    EVENT_APPROVAL_DEADLINE_DUE: ReturnRefundOperation._reduce_approval_deadline_due,
    EVENT_RESOLUTION_COOLDOWN_EXPIRED: (
        ReturnRefundOperation._reduce_resolution_cooldown_expired
    ),
}

_ACTION_HANDLERS: Mapping[str, _ActionHandler] = {
    ACTION_ISSUE_RETURN_AUTHORISATION: (
        ReturnRefundOperation._act_issue_return_authorisation
    ),
    ACTION_REQUEST_REFUND_APPROVAL: ReturnRefundOperation._act_request_refund_approval,
    ACTION_REQUEST_REFUND: ReturnRefundOperation._act_request_refund,
    ACTION_SEND_MESSAGE: ReturnRefundOperation._act_send_message,
}


__all__ = [
    "ACTION_COMPLETE",
    "ACTION_ISSUE_RETURN_AUTHORISATION",
    "ACTION_REQUEST_REFUND",
    "ACTION_REQUEST_REFUND_APPROVAL",
    "ACTION_SEND_MESSAGE",
    "ACTION_TYPES",
    "CHECKPOINT_HIGH_VALUE_REFUND",
    "CUSTOMER_ACTOR_ID",
    "ELIGIBLE_DISPOSITION",
    "EVENT_APPROVAL_DEADLINE_DUE",
    "EVENT_CARRIER_HANDOVER_VERIFIED",
    "EVENT_CUSTOMER_HANDOVER_CLAIMED",
    "EVENT_CUSTOMER_RETURN_REQUESTED",
    "EVENT_HANDOVER_DEADLINE_DUE",
    "EVENT_HANDOVER_REMINDER_DUE",
    "EVENT_REFUND_APPROVAL_RECEIVED",
    "EVENT_REFUND_SETTLEMENT_CONFIRMED",
    "EVENT_REFUND_SETTLEMENT_REVERSED",
    "EVENT_RESOLUTION_COOLDOWN_EXPIRED",
    "EVENT_TYPES",
    "EVENT_WAREHOUSE_INSPECTION_COMPLETED",
    "FIRST_REFUND_CYCLE",
    "MSG_HANDOVER_REMINDER",
    "MSG_REFUND_COMPLETE",
    "MSG_REFUND_DENIED",
    "MSG_RETURN_EXPIRED",
    "READ_GET_RETURN_CASE",
    "READ_LIST_AUTHORISATIONS",
    "READ_LIST_CARRIER_RECORDS",
    "READ_LIST_CHECKPOINTS",
    "READ_LIST_COMMUNICATIONS",
    "READ_LIST_CUSTOMER_CLAIMS",
    "READ_LIST_INSPECTIONS",
    "READ_LIST_OBLIGATIONS",
    "READ_LIST_REFUNDS",
    "READ_TOOLS",
    "RECOVERY_REFUND_CYCLE",
    "ReturnRefundOperation",
]
