"""The Maintenance domain: what the world may do, and what the agent may do.

Every rule here is a guard with a name. Nothing is heuristic, nothing is
inferred from prose, and no path exists by which a plausible message becomes a
business fact. The four rules worth stating out loud, because they are what the
vertical is for:

1. **A claim is not a fact.** ``supplier_assertion_received`` is always accepted
   and never mutates work, invoice or payment state. It is stored where it can
   be read and cited — and citing it as evidence for a consequential action is
   refused by name.
2. **Approval binds to an exact quote version.** Amount, currency, scope digest
   and cycle must all match, and the decision must arrive inside the checkpoint
   window. A revision is a different quote, so an old approval cannot authorise it.
3. **Money needs verified work and a validated matching invoice.** A premature
   invoice is stored, not paid. Validation is a deterministic match against the
   approved quote plus an authoritative validation event; settlement is only the
   payment system's own confirmation.
4. **Completion is guarded twice.** An ordinary COMPLETE enters a provisional
   window, not a final state; a customer persistence report inside that window
   reopens the operation and cancels finalization. Only the finalization timer
   makes an episode replay-final, and nothing resurrects it afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    QuoteForSelectedApproval,
    ReadRequirementContract,
    WorkEvidenceForInvoice,
    resolve_required_evidence,
)
from operatebench.core.retrieval import (
    AUTHORITY_ACTOR_CLAIM,
    RetrievalRequest,
    ToolResult,
)
from operatebench.domains.lettings.maintenance import spec as maintenance_spec
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    maintenance_retrieval_catalogue,
    maintenance_retrieval_record_contract,
    serve_maintenance_retrieval,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_EVENT_TYPES,
    MAINTENANCE_ROLE_AUTHORITY,
    MAINTENANCE_WAKE_EVENT_TYPES,
    OperationSpec,
    ScenarioSpec,
    action_schema_view,
    validate_action_payload,
    validate_event_payload,
)
from operatebench.domains.lettings.maintenance.state import (
    APPROVAL_APPROVED,
    APPROVAL_EXPIRED,
    APPROVAL_OPEN,
    APPROVAL_REJECTED,
    CYCLE_APPROVED_WORK,
    CYCLE_DIAGNOSTIC,
    CYCLE_WARRANTY_REVISIT,
    EXCEPTION_TYPES,
    INVOICE_REJECTED,
    INVOICE_UNVALIDATED,
    INVOICE_VALIDATED,
    INVOICE_VALIDATION_REQUESTED,
    OBLIGATION_BREACHED,
    OBLIGATION_CANCELLED,
    OBLIGATION_DISCHARGED,
    PAYMENT_NOT_REQUESTED,
    PAYMENT_REQUESTED,
    PAYMENT_SETTLED,
    QUOTE_APPROVED,
    QUOTE_EXPIRED,
    QUOTE_RECEIVED,
    QUOTE_REJECTED,
    RESOLUTION_PERSISTS,
    RESOLUTION_RESOLVED,
    VISIT_ATTENDED,
    VISIT_DECLINED,
    VISIT_NONE,
    VISIT_REQUESTED,
    VISIT_SCHEDULED,
    VISIT_WORK_REPORTED,
    VISIT_WORK_VERIFIED,
    Approval,
    Cycle,
    ExceptionCheckpoint,
    Invoice,
    MaintenanceState,
    Obligation,
    Quote,
    quote_key,
)

#: Message fixtures with a state meaning. Everything else is diagnostic prose.
MSG_COMPLETION_NOTICE = "msg_completion_notice"
MSG_TRANSFER_NOTICE = "msg_transfer_notice"
MSG_APPROVAL_REMINDER = "msg_approval_reminder"

#: How an operator may resolve an exception checkpoint.
_EXCEPTION_RESOLUTIONS = frozenset(
    {"TAKE_OWNERSHIP", "AUTHORISE_ALTERNATIVE", "CANCEL_OPERATION"}
)

#: The checkpoint classes the evaluator counts as human load.
CHECKPOINT_APPROVAL = "repair_quote_approval"
CHECKPOINT_EXCEPTION = "maintenance_exception_resolution"


class MaintenanceOperation:
    """One scenario of the synthetic Maintenance operation, ready to run."""

    def __init__(self, spec: OperationSpec, scenario_id: str) -> None:
        self.spec = spec
        self.scenario: ScenarioSpec = spec.scenario(scenario_id)
        self.policy = spec.policy

    # ------------------------------------------------------------ plan/state

    def build_plan(self) -> EpisodePlan:
        """Translate the authored scenario into Core's own plan vocabulary."""
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
                        kind=authored.trigger.kind,
                        type_name=authored.trigger.type_name,
                        cycle_id=authored.trigger.cycle_id,
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

    def initial_state(self) -> MaintenanceState:
        state = MaintenanceState()
        # A detached deep copy: episode state is the operation's own mutable
        # record, and it must never be a view onto the frozen spec.
        state.hidden = self.spec.hidden_state_snapshot()
        return state

    def canonical_state(self, state: MaintenanceState) -> Mapping[str, Any]:
        return state.canonical()

    # ------------------------------------------------------------- retrieval

    def coarse_phase(self, state: MaintenanceState) -> str:
        """Where in its life this operation is. One scalar, and nothing else.

        Published beside the read catalogue because a planner has to be able to
        ask for the right records without having been handed them: the phase
        says *what kind of decision is due*, and the catalogue says how to find
        out. It discloses no quote, no approval and no amount.
        """
        return state.phase

    def event_authority(self, actor_id: str, event_type: str) -> str:
        """The authority class of whoever produced this event.

        Derived from the actor registry, which is the same table the spec loader
        binds every authored event against. An event type does not confer
        standing; the actor that emitted it does, and this reads it from the one
        place that says so.
        """
        actor = self.spec.actors.get(actor_id)
        role = actor.role if actor is not None else ""
        return MAINTENANCE_ROLE_AUTHORITY.get(role, AUTHORITY_ACTOR_CLAIM)

    def retrieval_catalogue(self) -> Mapping[str, Mapping[str, Any]]:
        """The eight reads this operation publishes, with their exact schemas."""
        return maintenance_retrieval_catalogue()

    def retrieval_record_contract(self) -> list[str | int]:
        """The operation-wide identity and version mechanics for served records."""
        return maintenance_retrieval_record_contract()

    def read_requirements(self) -> ReadRequirementContract:
        """The one canonical contract: what each outcome must have read.

        The same object the published action schemas project and the evaluator
        reconstructs against. Handed out rather than copied, so "the runtime and
        the grader are executing one rule" is an identity rather than a claim
        about two tables that currently agree.
        """
        return maintenance_spec.maintenance_action_evidence_contract()

    def serve_retrieval(
        self,
        state: MaintenanceState,
        requests: Sequence[RetrievalRequest],
        as_of: str,
    ) -> Sequence[ToolResult]:
        """Answer one canonical batch. The environment's method, never the agent's.

        Non-mutating by construction: it projects the observable record and
        returns slices of it. The engine is the only caller; an agent holds no
        state handle and no route to this.
        """
        # Projected root by root through :meth:`MaintenanceState.project`, never
        # through the whole-record projection. Phase 2 takes that projection off
        # the agent path entirely, and a broker that still reached for it would
        # be the one path left by which the whole record could come back.
        roots = sorted(
            {
                root
                for request in requests
                for root in MAINTENANCE_RETRIEVAL_TOOLS[request.tool].roots
            }
        )
        return serve_maintenance_retrieval(state.project(roots), requests, as_of)

    def policy_view(self) -> Mapping[str, Any]:
        return {
            "currency": self.policy.currency,
            "issue_classification": self.policy.issue_classification,
            "approval_threshold_minor": self.policy.approval_threshold_minor,
            "approval_reminder_after_minutes": (
                self.policy.approval_reminder_after_minutes
            ),
            "approval_deadline_after_minutes": (
                self.policy.approval_deadline_after_minutes
            ),
            "approval_validity_minutes": self.policy.approval_validity_minutes,
            "provisional_close_minutes": self.policy.provisional_close_minutes,
            "visit_followup_after_minutes": self.policy.visit_followup_after_minutes,
            "completion_notice_within_minutes": (
                self.policy.completion_notice_within_minutes
            ),
        }

    def actor_roles(self) -> Mapping[str, str]:
        return {actor_id: actor.role for actor_id, actor in self.spec.actors.items()}

    def message_fixture_ids(self) -> Sequence[str]:
        return sorted(self.spec.message_fixtures)

    def action_schemas(self) -> Mapping[str, Mapping[str, Any]]:
        """The payload contract of every action this agent may actually propose.

        The intersection of "has an executable handler" and "the actor registry
        grants it", read from the same table
        :func:`validate_action_payload` enforces. An offered action the
        validator would refuse, or a required field an agent can only discover
        by being refused, would both make the benchmark measure interface
        guessing rather than operating.
        """
        agent = self.spec.actors["agent"]
        published = action_schema_view()
        offered = {
            action_type: schema
            for action_type, schema in published.items()
            if action_type in _ACTION_HANDLERS and agent.holds(action_type)
        }
        # The terminal is published beside them. It is not an *action* — there is
        # no handler to grant and no payload to fill — but it is a business
        # outcome the agent may propose, its correctness depends on the record,
        # and the read contract states what it rests on. Withholding that would
        # be the one outcome an agent had to discover its read set for by being
        # refused.
        offered[COMPLETE_OUTCOME_KEY] = published[COMPLETE_OUTCOME_KEY]
        return offered

    def wake_event_vocabulary(self) -> Sequence[str]:
        """Every event type a WAIT may legally name. Not this episode's queue."""
        return MAINTENANCE_WAKE_EVENT_TYPES

    def is_replay_final(self, state: MaintenanceState) -> bool:
        return state.replay_final

    def terminal_outcome(self, state: MaintenanceState) -> str | None:
        if state.terminal is None:
            return None
        outcome = state.terminal.get("outcome")
        return str(outcome) if outcome is not None else None

    def current_cycle_id(self, state: MaintenanceState) -> str | None:
        return state.current_cycle_id

    # ---------------------------------------------------------------- events

    def reduce_event(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        """Deliver one external event, or refuse it by name.

        Provenance is re-established here, at the moment the event would mutate,
        and not inherited from the fact that a file loaded cleanly. The spec
        loader proves that *an authored file* named an actor holding the right
        authority; this proves that *this event, now* does. Between those two
        statements sits everything that could have been swapped, and only the
        second one is standing next to the write.
        """
        actor = self.spec.actors.get(event.actor_id)
        if actor is None:
            return Verdict.refused(
                "UNKNOWN_ACTOR", f"actor {event.actor_id!r} is not in the registry"
            )
        required_authority = MAINTENANCE_EVENT_TYPES.get(event.event_type)
        reducer = _EVENT_REDUCERS.get(event.event_type)
        if required_authority is None or reducer is None:
            return Verdict.refused(
                "UNKNOWN_EVENT_TYPE",
                f"{event.event_type!r} has no reducer in this operation",
            )
        if not actor.holds(required_authority):
            return Verdict.refused(
                "EVENT_ACTOR_LACKS_AUTHORITY",
                f"actor {event.actor_id!r} does not hold {required_authority!r}, so it "
                f"cannot bring about {event.event_type!r}; provenance is derived from "
                "the actor registry and is re-checked where the mutation would happen",
            )
        try:
            validate_event_payload(
                event.event_type, event.payload, f"event {event.event_id}"
            )
        except SpecSchemaError as exc:
            # A payload the reducer would index into is checked before it does.
            # An event that cannot be read is refused by name, never as the
            # KeyError of whichever field the reducer happened to reach first.
            return Verdict.refused("MALFORMED_EVENT_PAYLOAD", str(exc))
        return reducer(self, state, event, context)

    def _reduce_issue_reported(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        if state.issue_id is not None:
            return Verdict.refused(
                "DUPLICATE_ISSUE", "this operation already has a reported issue"
            )
        classification = str(event.payload.get("classification", ""))
        if classification != self.policy.issue_classification:
            return Verdict.refused(
                "UNSUPPORTED_CLASSIFICATION",
                f"{classification!r} is not the classification this operation handles",
            )
        cycle_id = str(event.payload["cycle_id"])
        state.issue_id = str(event.payload["issue_id"])
        state.issue_reporting_actor_id = event.actor_id
        state.classification = classification
        state.cycles[cycle_id] = Cycle(cycle_id=cycle_id, kind=CYCLE_DIAGNOSTIC)
        state.current_cycle_id = cycle_id
        state.phase = "ACTIVE"
        return Verdict.ok(cycle_id=cycle_id)

    def _reduce_visit_response(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        cycle = _cycle_for_visit(state, str(event.payload["visit_id"]))
        if cycle is None:
            return Verdict.refused(
                "UNKNOWN_VISIT",
                f"no visit {event.payload['visit_id']!r} was ever requested",
            )
        if cycle.visit_status != VISIT_REQUESTED:
            return Verdict.refused(
                "VISIT_NOT_AWAITING_RESPONSE",
                f"visit {cycle.visit_id!r} is {cycle.visit_status}, not REQUESTED",
            )
        response = str(event.payload.get("response"))
        if response == "ACCEPTED":
            cycle.visit_status = VISIT_SCHEDULED
        elif response == "DECLINED":
            cycle.visit_status = VISIT_DECLINED
        else:
            return Verdict.refused(
                "UNKNOWN_VISIT_RESPONSE", f"{response!r} is not a visit response"
            )
        return Verdict.ok(cycle_id=cycle.cycle_id, bindings=_visit_bindings(cycle))

    def _reduce_attendance_verified(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        cycle = _cycle_for_visit(state, str(event.payload["visit_id"]))
        if cycle is None or cycle.visit_status != VISIT_SCHEDULED:
            return Verdict.refused(
                "VISIT_NOT_SCHEDULED",
                "attendance can only be verified for a scheduled visit",
            )
        if not bool(event.payload.get("attended")):
            return Verdict.refused(
                "ATTENDANCE_NOT_CONFIRMED", "the authoritative system reports no visit"
            )
        cycle.visit_status = VISIT_ATTENDED
        state.authoritative_records[f"attendance:{cycle.visit_id}"] = {
            "kind": "visit_attendance",
            "cycle_id": cycle.cycle_id,
            "visit_id": cycle.visit_id,
            "at": context.now,
        }
        context.cancel_timer(_visit_timer_id(cycle.cycle_id))
        _close_obligation(
            state, context, _visit_obligation_id(cycle.cycle_id), OBLIGATION_DISCHARGED
        )
        return Verdict.ok(cycle_id=cycle.cycle_id, bindings=_visit_bindings(cycle))

    def _reduce_work_reported(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        cycle = _cycle_for_visit(state, str(event.payload["visit_id"]))
        if cycle is None or cycle.visit_status != VISIT_ATTENDED:
            return Verdict.refused(
                "VISIT_NOT_ATTENDED",
                "a supplier can only report work for a visit authoritative "
                "verification says happened",
            )
        # A supplier report is a claim about the work, recorded as such. It never
        # sets WORK_VERIFIED: only the maintenance system's evidence does that.
        cycle.visit_status = VISIT_WORK_REPORTED
        cycle.reported_outcome = str(event.payload.get("outcome"))
        cycle.scope_digest = cycle.scope_digest or event.payload.get("scope_digest")
        return Verdict.ok(cycle_id=cycle.cycle_id, bindings=_visit_bindings(cycle))

    def _reduce_quote_received(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        key = quote_key(str(payload["quote_id"]), int(payload["quote_version"]))
        if key in state.quotes:
            return Verdict.refused(
                "DUPLICATE_QUOTE", f"quote {key} has already been received"
            )
        amount = int(payload["amount_minor"])
        if amount <= 0:
            return Verdict.refused("INVALID_QUOTE_AMOUNT", "a quote must be positive")
        currency = str(payload["currency"])
        if currency != self.policy.currency:
            # The authored threshold is denominated in the policy currency, so a
            # quote in another one cannot be compared against it at all.
            return Verdict.refused(
                "QUOTE_CURRENCY_MISMATCH",
                f"this operation is denominated in {self.policy.currency!r}; a quote "
                f"in {currency!r} cannot be measured against its authored threshold",
            )
        cycle_id = str(payload["cycle_id"])
        parent_id = payload.get("parent_cycle_id")
        scope = str(payload["scope_digest"])
        cycle = state.cycles.get(cycle_id)
        if cycle is None:
            cycle = Cycle(
                cycle_id=cycle_id,
                kind=CYCLE_APPROVED_WORK,
                parent_cycle_id=str(parent_id) if parent_id is not None else None,
                scope_digest=scope,
                requires_payment=True,
            )
            state.cycles[cycle_id] = cycle
        state.quotes[key] = Quote(
            quote_id=str(payload["quote_id"]),
            version=int(payload["quote_version"]),
            cycle_id=cycle_id,
            amount_minor=amount,
            currency=currency,
            scope_digest=scope,
            # How long a quote stays authorisable is a policy decision, not a
            # supplier's. Read it from ``approval_validity_minutes`` in policy so
            # changing the policy changes the
            # spec digest and nothing about the run.
            valid_until=shift_minutes(
                context.now,
                self.policy.approval_validity_minutes,
                context="quote validity",
            ),
        )
        # A quote for further work moves the operation's attention onto that
        # cycle: everything that follows — approval, authorisation, invoice,
        # payment — correlates there.
        state.current_cycle_id = cycle_id
        return Verdict.ok(cycle_id=cycle_id)

    def _reduce_invoice_received(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        invoice_id = str(payload["invoice_id"])
        if invoice_id in state.invoices:
            return Verdict.refused(
                "DUPLICATE_INVOICE", f"invoice {invoice_id!r} has already been received"
            )
        # Deliberately accepted whatever the state of the work: an invoice that
        # arrives too early is a real thing that happens. It is stored
        # unvalidated, which is not the same as payable.
        state.invoices[invoice_id] = Invoice(
            invoice_id=invoice_id,
            cycle_id=str(payload["cycle_id"]),
            quote_id=str(payload["quote_id"]),
            quote_version=int(payload["quote_version"]),
            amount_minor=int(payload["amount_minor"]),
            currency=str(payload["currency"]),
        )
        return Verdict.ok(cycle_id=str(payload["cycle_id"]))

    def _reduce_assertion(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        state.assertions.append(
            {
                "assertion_id": str(event.payload["assertion_id"]),
                "actor_id": event.actor_id,
                "assertion": str(event.payload.get("assertion")),
                "cycle_id": event.payload.get("cycle_id"),
                "at": context.now,
            }
        )
        # Observational by construction. Nothing else in this reducer touches
        # work, invoice or payment state, and nothing else may.
        return Verdict(
            accepted=True,
            code="OBSERVATIONAL_ONLY",
            detail="an actor claim is recorded, never made true",
            cycle_id=event.payload.get("cycle_id"),
            observational=True,
        )

    def _reduce_approver_decision(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        checkpoint_id = str(payload["checkpoint_id"])
        approval = state.approvals.get(checkpoint_id)
        if approval is None:
            return Verdict.refused(
                "UNKNOWN_CHECKPOINT",
                f"no approval checkpoint {checkpoint_id!r} was ever opened",
            )
        if approval.status != APPROVAL_OPEN:
            return Verdict.refused(
                "CHECKPOINT_NOT_OPEN",
                f"checkpoint {checkpoint_id!r} is {approval.status}; a decision that "
                "arrives after it closed cannot mutate the operation",
            )
        mismatch = _binding_mismatch(approval, payload)
        if mismatch is not None:
            return Verdict.refused("QUOTE_BINDING_MISMATCH", mismatch)
        if parse_timestamp(context.now, "now") > parse_timestamp(
            approval.deadline_at, "checkpoint deadline"
        ):
            return Verdict.refused(
                "CHECKPOINT_DEADLINE_PASSED",
                f"the decision arrived after {approval.deadline_at}",
            )
        decision = str(payload.get("decision"))
        quote = state.quote(approval.quote_id, approval.quote_version)
        if decision == "APPROVED":
            approval.status = APPROVAL_APPROVED
            if quote is not None:
                quote.status = QUOTE_APPROVED
        elif decision == "REJECTED":
            approval.status = APPROVAL_REJECTED
            if quote is not None:
                quote.status = QUOTE_REJECTED
        else:
            return Verdict.refused(
                "UNKNOWN_DECISION", f"{decision!r} is not an approval decision"
            )
        approval.resolved_at = context.now
        state.phase = "ACTIVE"
        context.cancel_timer(_reminder_timer_id(checkpoint_id))
        context.cancel_timer(_deadline_timer_id(checkpoint_id))
        _close_obligation(
            state, context, _reminder_obligation_id(checkpoint_id), OBLIGATION_CANCELLED
        )
        _close_obligation(
            state, context, _decision_obligation_id(checkpoint_id), OBLIGATION_DISCHARGED
        )
        context.record(
            "checkpoint_resolved",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_APPROVAL,
                "resolution": approval.status,
            },
        )
        return Verdict.ok(
            cycle_id=approval.cycle_id,
            bindings=_bindings(checkpoint_id=checkpoint_id),
        )

    def _reduce_operator_resolution(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        checkpoint_id = str(event.payload["checkpoint_id"])
        checkpoint = state.exceptions.get(checkpoint_id)
        if checkpoint is None or checkpoint.status != "OPEN":
            return Verdict.refused(
                "UNKNOWN_EXCEPTION_CHECKPOINT",
                f"no open exception checkpoint {checkpoint_id!r}",
            )
        decision = str(event.payload.get("decision"))
        if decision not in _EXCEPTION_RESOLUTIONS:
            return Verdict.refused(
                "UNKNOWN_EXCEPTION_DECISION", f"{decision!r} is not a resolution"
            )
        checkpoint.status = "RESOLVED"
        checkpoint.resolution = decision
        checkpoint.resolved_at = context.now
        if decision == "TAKE_OWNERSHIP":
            state.ownership_transferred_to = event.actor_id
        state.phase = "ACTIVE"
        context.record(
            "checkpoint_resolved",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_EXCEPTION,
                "resolution": decision,
            },
        )
        return Verdict.ok(
            cycle_id=checkpoint.cycle_id,
            bindings=_bindings(checkpoint_id=checkpoint_id),
        )

    def _reduce_work_evidence(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        cycle = state.cycles.get(str(payload["cycle_id"]))
        if cycle is None or cycle.visit_id != payload.get("visit_id"):
            return Verdict.refused(
                "UNKNOWN_WORK_CYCLE",
                "work evidence must name a cycle and the visit it was done on",
            )
        if cycle.visit_status not in {VISIT_ATTENDED, VISIT_WORK_REPORTED}:
            return Verdict.refused(
                "VISIT_NOT_ATTENDED",
                f"visit {cycle.visit_id!r} is {cycle.visit_status}; work cannot be "
                "verified for a visit that has not happened",
            )
        scope = str(payload["scope_digest"])
        if cycle.kind == CYCLE_WARRANTY_REVISIT:
            parent = state.cycles.get(cycle.parent_cycle_id or "")
            if parent is None or parent.scope_digest != scope:
                return Verdict.refused(
                    "WARRANTY_SCOPE_MISMATCH",
                    "a warranty revisit must repeat the scope of the work it "
                    "follows; different scope is new work and needs its own quote",
                )
        cycle.visit_status = VISIT_WORK_VERIFIED
        evidence_id = str(payload["evidence_id"])
        cycle.work_evidence_id = evidence_id
        state.authoritative_records[evidence_id] = {
            "kind": "work_evidence",
            "cycle_id": cycle.cycle_id,
            "visit_id": cycle.visit_id,
            "scope_digest": scope,
            "outcome": str(payload.get("outcome")),
            "at": context.now,
        }
        if not cycle.requires_payment:
            _open_obligation(
                state,
                context,
                _notice_obligation_id(cycle.cycle_id),
                "customer_completion_notice",
                cycle.cycle_id,
                self.policy.completion_notice_within_minutes,
            )
        return Verdict.ok(cycle_id=cycle.cycle_id, bindings=_visit_bindings(cycle))

    def _reduce_invoice_validation(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        invoice = state.invoices.get(str(event.payload["invoice_id"]))
        if invoice is None or invoice.status != INVOICE_VALIDATION_REQUESTED:
            return Verdict.refused(
                "INVOICE_NOT_AWAITING_VALIDATION",
                "only an invoice whose validation was requested can be validated",
            )
        validation_id = str(event.payload["validation_id"])
        if bool(event.payload.get("valid")):
            invoice.status = INVOICE_VALIDATED
        else:
            invoice.status = INVOICE_REJECTED
        invoice.validation_id = validation_id
        state.authoritative_records[validation_id] = {
            "kind": "invoice_validation",
            "invoice_id": invoice.invoice_id,
            "cycle_id": invoice.cycle_id,
            "valid": bool(event.payload.get("valid")),
            "reason_code": str(event.payload.get("reason_code", "")),
            "at": context.now,
        }
        return Verdict.ok(cycle_id=invoice.cycle_id)

    def _reduce_settlement(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        payload = event.payload
        if state.payment["status"] != PAYMENT_REQUESTED:
            return Verdict.refused(
                "NO_PAYMENT_REQUESTED",
                "settlement can only confirm a payment this operation requested",
            )
        if state.payment["request_id"] != payload.get("payment_request_id"):
            return Verdict.refused(
                "PAYMENT_REQUEST_MISMATCH",
                "the settlement names a payment request this operation did not make",
            )
        mismatch = _settlement_mismatch(state, payload)
        if mismatch is not None:
            # Named, and non-mutating by construction: the refusal happens before
            # any write, so a settlement that names the right request and the
            # wrong money leaves the payment REQUESTED and no authoritative
            # record behind it.
            return Verdict.refused("SETTLEMENT_BINDING_MISMATCH", mismatch)
        state.payment["status"] = PAYMENT_SETTLED
        settlement_id = str(payload["settlement_id"])
        state.payment["settlement_id"] = settlement_id
        state.authoritative_records[settlement_id] = {
            "kind": "payment_settlement",
            "payment_request_id": str(payload["payment_request_id"]),
            "invoice_id": str(payload["invoice_id"]),
            "cycle_id": str(payload["cycle_id"]),
            "amount_minor": int(payload["amount_minor"]),
            "currency": str(payload["currency"]),
            "at": context.now,
        }
        cycle_id = str(state.payment["cycle_id"])
        _open_obligation(
            state,
            context,
            _notice_obligation_id(cycle_id),
            "customer_completion_notice",
            cycle_id,
            self.policy.completion_notice_within_minutes,
        )
        return Verdict.ok(
            cycle_id=cycle_id,
            bindings=_bindings(payment_request_id=str(payload["payment_request_id"])),
        )

    def _reduce_customer_resolution(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        status = str(event.payload.get("status"))
        if status not in {RESOLUTION_RESOLVED, RESOLUTION_PERSISTS}:
            return Verdict.refused(
                "UNKNOWN_RESOLUTION_STATUS", f"{status!r} is not a resolution report"
            )
        if status == RESOLUTION_PERSISTS and state.phase == "PROVISIONALLY_CLOSED":
            reopen_until = str(state.provisional_close["reopen_until"])
            if parse_timestamp(context.now, "now") > parse_timestamp(
                reopen_until, "reopen window"
            ):
                return Verdict.refused(
                    "REOPEN_WINDOW_CLOSED",
                    f"the provisional window closed at {reopen_until}",
                )
            return self._reopen(state, context)
        if state.phase == "PROVISIONALLY_CLOSED" and status == RESOLUTION_RESOLVED:
            state.customer_resolution = RESOLUTION_RESOLVED
            return Verdict.ok(cycle_id=state.current_cycle_id)
        if state.phase in {"FINALIZED", "TRANSFERRED_TO_HUMAN", "NEW"}:
            return Verdict.refused(
                "RESOLUTION_NOT_ACCEPTED_IN_PHASE",
                f"a resolution report is not accepted while the operation is "
                f"{state.phase}",
            )
        state.customer_resolution = status
        return Verdict.ok(cycle_id=state.current_cycle_id)

    def _reopen(self, state: MaintenanceState, context: EnvironmentContext) -> Verdict:
        """Cancel finalization and open a warranty cycle against the same scope."""
        timer_id = state.provisional_close.get("timer_event_id")
        if isinstance(timer_id, str):
            context.cancel_timer(timer_id)
        closed_cycle_id = str(state.provisional_close["cycle_id"])
        closed_cycle = state.cycles[closed_cycle_id]
        state.reopen_count += 1
        new_cycle_id = f"work_cycle_{len(state.cycles) + 1}"
        state.cycles[new_cycle_id] = Cycle(
            cycle_id=new_cycle_id,
            kind=CYCLE_WARRANTY_REVISIT,
            parent_cycle_id=closed_cycle_id,
            scope_digest=closed_cycle.scope_digest,
            requires_payment=False,
        )
        state.current_cycle_id = new_cycle_id
        state.customer_resolution = RESOLUTION_PERSISTS
        state.phase = "ACTIVE"
        state.provisional_close = {
            "cycle_id": None,
            "accepted_at": None,
            "reopen_until": None,
            "timer_event_id": None,
        }
        context.record(
            "operation_reopened",
            {
                "reopened_cycle_id": closed_cycle_id,
                "warranty_cycle_id": new_cycle_id,
                "reason": "customer reported the issue persists inside the window",
            },
        )
        return Verdict.ok(code="REOPENED", cycle_id=new_cycle_id)

    def _reduce_reminder_due(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        approval = state.approvals.get(str(event.payload["checkpoint_id"]))
        if approval is None or approval.status != APPROVAL_OPEN:
            return Verdict.refused(
                "CHECKPOINT_NOT_OPEN", "a reminder is only due while a decision is open"
            )
        if approval.reminder_fired:
            return Verdict.refused(
                "REMINDER_ALREADY_FIRED", "the authored reminder fires exactly once"
            )
        approval.reminder_fired = True
        return Verdict.ok(cycle_id=approval.cycle_id)

    def _reduce_deadline_due(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        checkpoint_id = str(event.payload["checkpoint_id"])
        approval = state.approvals.get(checkpoint_id)
        if approval is None or approval.status != APPROVAL_OPEN:
            return Verdict.refused(
                "CHECKPOINT_NOT_OPEN", "only an open checkpoint can expire"
            )
        approval.status = APPROVAL_EXPIRED
        approval.resolved_at = context.now
        quote = state.quote(approval.quote_id, approval.quote_version)
        if quote is not None:
            quote.status = QUOTE_EXPIRED
        context.cancel_timer(_reminder_timer_id(checkpoint_id))
        _close_obligation(
            state, context, _reminder_obligation_id(checkpoint_id), OBLIGATION_CANCELLED
        )
        # The approver's silence is not the agent's breach: the obligation is
        # closed as cancelled, and what the agent does next is what is judged.
        _close_obligation(
            state, context, _decision_obligation_id(checkpoint_id), OBLIGATION_CANCELLED
        )
        context.record(
            "checkpoint_resolved",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_APPROVAL,
                "resolution": APPROVAL_EXPIRED,
            },
        )
        return Verdict.ok(
            cycle_id=approval.cycle_id,
            bindings=_bindings(checkpoint_id=checkpoint_id),
        )

    def _reduce_visit_followup_due(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        cycle_id = str(event.payload["cycle_id"])
        cycle = state.cycles.get(cycle_id)
        if cycle is None:
            return Verdict.refused("UNKNOWN_WORK_CYCLE", f"no cycle {cycle_id!r}")
        _close_obligation(
            state, context, _visit_obligation_id(cycle_id), OBLIGATION_BREACHED
        )
        return Verdict.ok(cycle_id=cycle_id)

    def _reduce_provisional_expired(
        self, state: MaintenanceState, event: Event, context: EnvironmentContext
    ) -> Verdict:
        if state.phase != "PROVISIONALLY_CLOSED":
            return Verdict.refused(
                "NOT_PROVISIONALLY_CLOSED",
                "finalization only applies to a provisionally closed operation",
            )
        if state.customer_resolution == RESOLUTION_PERSISTS:
            return Verdict.refused(
                "UNRESOLVED_CUSTOMER_REPORT",
                "the customer reports the issue persists, so the window cannot finalize",
            )
        blocking = _blocking_checkpoints(state)
        if blocking:
            return Verdict.refused(
                "OPEN_CHECKPOINT_AT_FINALIZATION", f"still open: {blocking}"
            )
        state.phase = "FINALIZED"
        state.replay_final = True
        state.terminal = {
            "outcome": "completed_successfully",
            "at": context.now,
            "cycle_id": state.current_cycle_id,
        }
        context.record(
            "terminal_accepted",
            {
                "outcome": "completed_successfully",
                "replay_final": True,
                "cycle_id": state.current_cycle_id,
            },
        )
        return Verdict.ok(code="FINALIZED", cycle_id=state.current_cycle_id)

    # --------------------------------------------------------------- actions

    def apply_action(
        self,
        state: MaintenanceState,
        action_type: str,
        payload: Mapping[str, Any],
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict:
        """Validate one proposed action and, if it survives, commit its effect."""
        if state.replay_final:
            return Verdict.refused(
                "OPERATION_IS_REPLAY_FINAL",
                "a replay-final operation accepts no further action",
            )
        if not self.spec.actors["agent"].holds(action_type):
            return Verdict.refused(
                "ACTION_OUTSIDE_AGENT_AUTHORITY",
                f"the agent does not hold {action_type!r}",
            )
        handler = _ACTION_HANDLERS.get(action_type)
        if handler is None:
            return Verdict.refused(
                "UNKNOWN_ACTION_TYPE",
                f"{action_type!r} is not an action of this operation",
            )
        try:
            validate_action_payload(action_type, payload, f"action {action_type}")
        except SpecSchemaError as exc:
            # Checked before the handler indexes anything, exactly as an event
            # payload is checked before its reducer does. Nothing has been
            # written at this point, so the refusal is non-mutating by
            # construction rather than by inspection.
            return Verdict.refused("MALFORMED_ACTION_PAYLOAD", str(exc))
        unresolved = _unresolved_evidence(state, evidence_refs)
        if unresolved is not None:
            return unresolved
        return handler(self, state, payload, tuple(evidence_refs), context)

    def _act_request_visit(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        cycle_id = str(payload["cycle_id"])
        cycle = state.cycles.get(cycle_id)
        if cycle is None:
            return Verdict.refused("UNKNOWN_WORK_CYCLE", f"no cycle {cycle_id!r}")
        if cycle_id != state.current_cycle_id:
            return Verdict.refused(
                "STALE_WORK_CYCLE",
                f"cycle {cycle_id!r} is not the operation's current cycle "
                f"({state.current_cycle_id!r})",
            )
        if cycle.visit_status != VISIT_NONE:
            return Verdict.refused(
                "DUPLICATE_VISIT_REQUEST",
                f"cycle {cycle_id!r} already has visit {cycle.visit_id!r} at "
                f"{cycle.visit_status}",
            )
        if cycle.kind == CYCLE_APPROVED_WORK:
            return Verdict.refused(
                "WORK_REQUIRES_AUTHORISATION",
                "an approved-work visit is created by authorise_supplier_work, "
                "never by an unapproved visit request",
            )
        visit_type = str(payload["visit_type"])
        if visit_type != cycle.kind:
            return Verdict.refused(
                "VISIT_TYPE_MISMATCH",
                f"cycle {cycle_id!r} is a {cycle.kind} cycle, not {visit_type}",
            )
        cycle.visit_id = _allocate_visit_id(state)
        cycle.visit_status = VISIT_REQUESTED
        state.phase = "WAITING_VISIT"
        self._arm_visit_followup(state, context, cycle_id)
        return Verdict.ok(cycle_id=cycle_id, bindings=_visit_bindings(cycle))

    def _act_request_approval(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        quote = state.quote(str(payload["quote_id"]), int(payload["quote_version"]))
        if quote is None:
            return Verdict.refused(
                "UNKNOWN_QUOTE",
                f"no quote {payload['quote_id']!r} version {payload['quote_version']!r}",
            )
        if quote.status != QUOTE_RECEIVED:
            return Verdict.refused(
                "QUOTE_NOT_OPEN", f"quote {quote.key} is {quote.status}"
            )
        if (
            quote.amount_minor != int(payload["amount_minor"])
            or quote.currency != str(payload["currency"])
            or quote.scope_digest != str(payload["scope_digest"])
            or quote.cycle_id != str(payload["cycle_id"])
        ):
            return Verdict.refused(
                "QUOTE_BINDING_MISMATCH",
                "an approval must be requested for the exact quote version, amount, "
                "currency, scope and cycle that is on the table",
            )
        if quote.amount_minor <= self.policy.approval_threshold_minor:
            return Verdict.refused(
                "APPROVAL_NOT_REQUIRED",
                f"{quote.amount_minor} is within the authored threshold of "
                f"{self.policy.approval_threshold_minor}; asking a human here is "
                "unnecessary human load",
            )
        deadline_minutes = int(payload["deadline_after_minutes"])
        if deadline_minutes != self.policy.approval_deadline_after_minutes:
            return Verdict.refused(
                "DEADLINE_OUTSIDE_POLICY",
                f"the authored approval deadline is "
                f"{self.policy.approval_deadline_after_minutes} minutes, not "
                f"{deadline_minutes}",
            )
        deadline_at = shift_minutes(
            context.now, deadline_minutes, context="approval deadline"
        )
        # Allocated here, after every guard has passed: a refused request must
        # not consume an identity, or the next accepted one would be filed under
        # a number the record cannot account for.
        checkpoint_id = _allocate_checkpoint_id(state)
        state.approvals[checkpoint_id] = Approval(
            checkpoint_id=checkpoint_id,
            quote_id=quote.quote_id,
            quote_version=quote.version,
            cycle_id=quote.cycle_id,
            amount_minor=quote.amount_minor,
            currency=quote.currency,
            scope_digest=quote.scope_digest,
            opened_at=context.now,
            deadline_at=deadline_at,
        )
        state.phase = "WAITING_APPROVAL"
        context.record(
            "checkpoint_opened",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_APPROVAL,
                "assignee_actor_id": "approver_1",
                "required": True,
                "deadline_at": deadline_at,
            },
        )
        context.schedule_timer(
            _reminder_timer_id(checkpoint_id),
            "approval_reminder_due",
            "scheduler",
            self.policy.approval_reminder_after_minutes,
            {"checkpoint_id": checkpoint_id, "cycle_id": quote.cycle_id},
        )
        context.schedule_timer(
            _deadline_timer_id(checkpoint_id),
            "approval_deadline_due",
            "scheduler",
            deadline_minutes,
            {"checkpoint_id": checkpoint_id, "cycle_id": quote.cycle_id},
        )
        _open_obligation(
            state,
            context,
            _reminder_obligation_id(checkpoint_id),
            "approval_reminder",
            quote.cycle_id,
            self.policy.approval_reminder_after_minutes,
        )
        _open_obligation(
            state,
            context,
            _decision_obligation_id(checkpoint_id),
            "approval_decision",
            quote.cycle_id,
            deadline_minutes,
        )
        return Verdict.ok(
            cycle_id=quote.cycle_id, bindings=_bindings(checkpoint_id=checkpoint_id)
        )

    def _act_authorise_supplier_work(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        checkpoint_id = str(payload["approval_checkpoint_id"])
        approval = state.approvals.get(checkpoint_id)
        if approval is None:
            return Verdict.refused(
                "UNKNOWN_CHECKPOINT", f"no approval checkpoint {checkpoint_id!r}"
            )
        if approval.status != APPROVAL_APPROVED:
            return Verdict.refused(
                "WORK_NOT_APPROVED",
                f"checkpoint {checkpoint_id!r} is {approval.status}; above-threshold "
                "work cannot be authorised without a current approval",
            )
        citation = _required_evidence_verdict(
            state,
            "authorise_supplier_work",
            payload,
            evidence_refs,
            "MISSING_AUTHORITY_EVIDENCE",
        )
        if citation is not None:
            return citation
        quote_id = str(payload["quote_id"])
        quote_version = int(payload["quote_version"])
        if (approval.quote_id, approval.quote_version) != (quote_id, quote_version):
            return Verdict.refused(
                "STALE_APPROVAL",
                f"checkpoint {checkpoint_id!r} approved "
                f"{quote_key(approval.quote_id, approval.quote_version)}, which is not "
                f"{quote_key(quote_id, quote_version)}",
            )
        quote = state.quote(quote_id, quote_version)
        if quote is None or quote.status != QUOTE_APPROVED:
            return Verdict.refused(
                "QUOTE_NOT_APPROVED",
                f"quote {quote_key(quote_id, quote_version)} is not an approved quote",
            )
        if quote.valid_until is not None and parse_timestamp(
            context.now, "now"
        ) > parse_timestamp(quote.valid_until, "quote validity"):
            return Verdict.refused(
                "QUOTE_EXPIRED", f"quote {quote.key} expired at {quote.valid_until}"
            )
        cycle = state.cycles[quote.cycle_id]
        if cycle.visit_status != VISIT_NONE:
            return Verdict.refused(
                "DUPLICATE_WORK_AUTHORISATION",
                f"cycle {cycle.cycle_id!r} already has visit {cycle.visit_id!r}",
            )
        cycle.visit_id = _allocate_visit_id(state)
        # Booked, not merely requested: the supplier that quoted this work has
        # already agreed to do it, so authorising it is what schedules the visit.
        cycle.visit_status = VISIT_SCHEDULED
        state.current_cycle_id = cycle.cycle_id
        state.phase = "WAITING_VISIT"
        self._arm_visit_followup(state, context, cycle.cycle_id)
        return Verdict.ok(cycle_id=cycle.cycle_id, bindings=_visit_bindings(cycle))

    def _act_request_invoice_validation(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        invoice = state.invoices.get(str(payload["invoice_id"]))
        if invoice is None:
            return Verdict.refused(
                "UNKNOWN_INVOICE", f"no invoice {payload['invoice_id']!r}"
            )
        if invoice.status != INVOICE_UNVALIDATED:
            return Verdict.refused(
                "INVOICE_NOT_PENDING",
                f"invoice {invoice.invoice_id!r} is {invoice.status}",
            )
        cycle = state.cycles.get(invoice.cycle_id)
        if cycle is None or not cycle.work_verified:
            return Verdict.refused(
                "WORK_NOT_VERIFIED",
                "an invoice can only be validated once the work it bills for has "
                "authoritative verification",
            )
        quote = state.quote(invoice.quote_id, invoice.quote_version)
        if quote is None or quote.status != QUOTE_APPROVED:
            return Verdict.refused(
                "INVOICE_QUOTE_NOT_APPROVED",
                "the invoice does not bind to an approved quote version",
            )
        if (
            quote.amount_minor != invoice.amount_minor
            or quote.currency != invoice.currency
            or quote.cycle_id != invoice.cycle_id
        ):
            return Verdict.refused(
                "INVOICE_QUOTE_MISMATCH",
                "the invoice does not match the approved quote amount, currency or cycle",
            )
        citation = _required_evidence_verdict(
            state,
            "request_invoice_validation",
            payload,
            evidence_refs,
            "MISSING_AUTHORITY_EVIDENCE",
        )
        if citation is not None:
            return citation
        invoice.status = INVOICE_VALIDATION_REQUESTED
        # The invoice this operation *accepted*, under its canonical identity.
        # Every check above ran against this record rather than against the
        # string the agent typed, so binding it is what lets the validation the
        # world sends back — and the evaluator reading the ledger afterwards —
        # be about the same invoice the operation admitted.
        return Verdict.ok(
            cycle_id=invoice.cycle_id,
            bindings=_bindings(invoice_id=invoice.invoice_id),
        )

    def _act_request_payment(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        invoice = state.invoices.get(str(payload["invoice_id"]))
        if invoice is None:
            return Verdict.refused(
                "UNKNOWN_INVOICE", f"no invoice {payload['invoice_id']!r}"
            )
        if state.payment["status"] != PAYMENT_NOT_REQUESTED:
            return Verdict.refused(
                "DUPLICATE_PAYMENT_REQUEST",
                f"payment is already {state.payment['status']}",
            )
        cycle = state.cycles.get(invoice.cycle_id)
        if cycle is None or not cycle.requires_payment:
            return Verdict.refused(
                "PAYMENT_NOT_REQUIRED",
                f"cycle {invoice.cycle_id!r} carries no payable work; a warranty "
                "revisit is covered by the work cycle it follows",
            )
        if not cycle.work_verified:
            return Verdict.refused(
                "WORK_NOT_VERIFIED",
                "payment cannot be requested before the work has authoritative "
                "verification",
            )
        if invoice.status != INVOICE_VALIDATED:
            return Verdict.refused(
                "INVOICE_NOT_VALIDATED",
                f"invoice {invoice.invoice_id!r} is {invoice.status}",
            )
        if invoice.amount_minor != int(
            payload["amount_minor"]
        ) or invoice.currency != str(payload["currency"]):
            return Verdict.refused(
                "PAYMENT_AMOUNT_MISMATCH",
                "a payment request must match the validated invoice exactly",
            )
        citation = _required_evidence_verdict(
            state,
            "request_payment",
            payload,
            evidence_refs,
            "MISSING_AUTHORITY_EVIDENCE",
        )
        if citation is not None:
            return citation
        request_id = _allocate_payment_request_id(state)
        state.payment.update(
            {
                "request_id": request_id,
                "invoice_id": invoice.invoice_id,
                "cycle_id": invoice.cycle_id,
                "status": PAYMENT_REQUESTED,
            }
        )
        return Verdict.ok(
            cycle_id=invoice.cycle_id, bindings=_bindings(payment_request_id=request_id)
        )

    def _act_send_message(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        recipient = str(payload["recipient_actor_id"])
        fixture_id = str(payload["message_fixture_id"])
        if recipient not in self.spec.actors:
            return Verdict.refused("UNKNOWN_RECIPIENT", f"no actor {recipient!r}")
        if fixture_id not in self.spec.message_fixtures:
            return Verdict.refused(
                "UNKNOWN_MESSAGE_FIXTURE",
                f"{fixture_id!r} is not a declared message fixture",
            )
        if (
            fixture_id == MSG_COMPLETION_NOTICE
            and recipient != state.issue_reporting_actor_id
        ):
            return Verdict.refused(
                "COMPLETION_NOTICE_WRONG_RECIPIENT",
                "a completion notice must address the accepted issue-reporting actor",
            )
        correlation_id = payload.get("correlation_id")
        if fixture_id == MSG_APPROVAL_REMINDER and not (
            recipient == "approver_1"
            and any(
                approval.cycle_id == correlation_id and approval.reminder_fired
                for approval in state.open_approvals()
            )
        ):
            return Verdict.refused(
                "OBLIGATION_MESSAGE_BINDING_MISMATCH",
                "a reminder must address the assigned approver for an open, due cycle",
            )
        if fixture_id == MSG_TRANSFER_NOTICE and (
            recipient != state.issue_reporting_actor_id
            or correlation_id != state.current_cycle_id
        ):
            return Verdict.refused(
                "OBLIGATION_MESSAGE_BINDING_MISMATCH",
                "a transfer notice must address the reporting customer for this cycle",
            )
        if fixture_id == MSG_TRANSFER_NOTICE and state.ownership_transferred_to is None:
            return Verdict.refused(
                "NO_TRANSFER_TO_ANNOUNCE",
                "a transfer notice cannot be sent before an operator has accepted "
                "ownership",
            )
        state.communications.append(
            {
                "recipient_actor_id": recipient,
                "message_fixture_id": fixture_id,
                "correlation_id": correlation_id,
                "at": context.now,
            }
        )
        cycle_id = str(correlation_id) if isinstance(correlation_id, str) else None
        if fixture_id == MSG_COMPLETION_NOTICE and cycle_id in state.cycles:
            state.cycles[cycle_id].completion_notice_sent = True
            _close_obligation(
                state, context, _notice_obligation_id(cycle_id), OBLIGATION_DISCHARGED
            )
        if fixture_id == MSG_TRANSFER_NOTICE:
            state.transfer_notice_sent = True
        if fixture_id == MSG_APPROVAL_REMINDER and not context.dispatch_fails(fixture_id):
            for approval in state.open_approvals():
                if approval.reminder_fired and approval.cycle_id == cycle_id:
                    _close_obligation(
                        state,
                        context,
                        _reminder_obligation_id(approval.checkpoint_id),
                        OBLIGATION_DISCHARGED,
                    )
        # The commit is done. Dispatch is a separate, later fact: if it fails the
        # decision stands and the failure is recorded as recovery evidence.
        if context.dispatch_fails(fixture_id):
            context.record(
                "side_effect_failed",
                {
                    "channel": "message_dispatch",
                    "message_fixture_id": fixture_id,
                    "recipient_actor_id": recipient,
                    "committed_decision_preserved": True,
                },
            )
        else:
            context.record(
                "side_effect",
                {
                    "channel": "message_dispatch",
                    "message_fixture_id": fixture_id,
                    "recipient_actor_id": recipient,
                    "status": "DELIVERED",
                },
            )
        return Verdict.ok(cycle_id=cycle_id or state.current_cycle_id)

    def _act_request_exception_resolution(
        self,
        state: MaintenanceState,
        payload: Mapping[str, Any],
        evidence_refs: tuple[str, ...],
        context: EnvironmentContext,
    ) -> Verdict:
        exception_type = str(payload["exception_type"])
        if state.open_exceptions():
            return Verdict.refused(
                "EXCEPTION_ALREADY_OPEN",
                "one exception checkpoint at a time; a second is unnecessary human load",
            )
        if exception_type not in EXCEPTION_TYPES:
            return Verdict.refused(
                "UNKNOWN_EXCEPTION_TYPE",
                f"{exception_type!r} is not a typed exception of this operation",
            )
        selected_id = str(payload["approval_checkpoint_id"])
        eligible = _eligible_exception_approval(state, exception_type, selected_id)
        if eligible is None:
            return Verdict.refused(
                "EXCEPTION_NOT_ELIGIBLE",
                f"nothing in this operation's state justifies a {exception_type} "
                "checkpoint; escalating without an eligible condition is unnecessary "
                "human load",
            )
        citation = _required_evidence_verdict(
            state,
            "request_exception_resolution",
            payload,
            evidence_refs,
            "INCOMPLETE_CHECKPOINT_CONTEXT",
        )
        if citation is not None:
            return citation
        deadline_at = shift_minutes(
            context.now,
            int(payload["deadline_after_minutes"]),
            context="exception deadline",
        )
        checkpoint_id = _allocate_checkpoint_id(state)
        state.exceptions[checkpoint_id] = ExceptionCheckpoint(
            checkpoint_id=checkpoint_id,
            exception_type=exception_type,
            cycle_id=eligible.cycle_id,
            opened_at=context.now,
            deadline_at=deadline_at,
            evidence_refs=tuple(evidence_refs),
        )
        state.phase = "WAITING_EXCEPTION"
        context.record(
            "checkpoint_opened",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": CHECKPOINT_EXCEPTION,
                "assignee_actor_id": "operator_1",
                "required": True,
                "exception_type": exception_type,
                "evidence_refs": list(evidence_refs),
                "deadline_at": deadline_at,
            },
        )
        return Verdict.ok(
            cycle_id=eligible.cycle_id, bindings=_bindings(checkpoint_id=checkpoint_id)
        )

    def _arm_visit_followup(
        self, state: MaintenanceState, context: EnvironmentContext, cycle_id: str
    ) -> None:
        context.schedule_timer(
            _visit_timer_id(cycle_id),
            "visit_followup_due",
            "scheduler",
            self.policy.visit_followup_after_minutes,
            {"cycle_id": cycle_id},
        )
        _open_obligation(
            state,
            context,
            _visit_obligation_id(cycle_id),
            "visit_progress",
            cycle_id,
            self.policy.visit_followup_after_minutes,
        )

    # -------------------------------------------------------------- terminal

    def apply_terminal(
        self,
        state: MaintenanceState,
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict:
        """The guarded terminal transition. Two doors, both of them narrow."""
        if state.replay_final:
            return Verdict.refused(
                "OPERATION_IS_REPLAY_FINAL", "this operation has already ended"
            )
        if state.phase == "PROVISIONALLY_CLOSED":
            return Verdict.refused(
                "ALREADY_PROVISIONALLY_CLOSED",
                "the operation is already in its confirmation window; only the "
                "finalization timer closes it",
            )
        blocking = _blocking_checkpoints(state)
        if blocking:
            return Verdict.refused(
                "OPEN_CHECKPOINT", f"cannot complete with open checkpoint(s) {blocking}"
            )
        open_obligations = sorted(
            obligation.obligation_id for obligation in state.open_obligations()
        )
        if open_obligations:
            return Verdict.refused(
                "OPEN_OBLIGATION",
                f"cannot complete with open obligation(s) {open_obligations}",
            )
        if state.ownership_transferred_to is not None:
            return self._terminal_transfer(state, context)
        return self._terminal_provisional_close(state, context)

    def _terminal_transfer(
        self, state: MaintenanceState, context: EnvironmentContext
    ) -> Verdict:
        if not state.transfer_notice_sent:
            return Verdict.refused(
                "TRANSFER_NOTICE_NOT_SENT",
                "the customer must be told the operation moved to a human",
            )
        if state.payment["status"] != PAYMENT_NOT_REQUESTED:
            return Verdict.refused(
                "PAYMENT_MUTATED_BEFORE_TRANSFER",
                "a transfer after refused or expired approval cannot leave a payment "
                "behind it",
            )
        state.phase = "TRANSFERRED_TO_HUMAN"
        state.replay_final = True
        state.terminal = {
            "outcome": "transferred_to_human_ownership",
            "at": context.now,
            "cycle_id": state.current_cycle_id,
        }
        context.record(
            "terminal_accepted",
            {
                "outcome": "transferred_to_human_ownership",
                "replay_final": True,
                "operator_actor_id": state.ownership_transferred_to,
            },
        )
        return Verdict.ok(code="TRANSFERRED", cycle_id=state.current_cycle_id)

    def _terminal_provisional_close(
        self, state: MaintenanceState, context: EnvironmentContext
    ) -> Verdict:
        cycle = state.current_cycle
        if cycle is None:
            return Verdict.refused(
                "NO_CURRENT_CYCLE", "there is no work cycle to complete"
            )
        if not cycle.work_verified:
            return Verdict.refused(
                "WORK_NOT_VERIFIED",
                f"cycle {cycle.cycle_id!r} is {cycle.visit_status}; completion needs "
                "authoritative work evidence",
            )
        if cycle.requires_payment:
            invoice_ids = [
                invoice.invoice_id
                for invoice in state.invoices.values()
                if invoice.cycle_id == cycle.cycle_id
            ]
            if not invoice_ids:
                return Verdict.refused(
                    "INVOICE_MISSING", "payable work has no invoice to settle"
                )
            unvalidated = [
                invoice_id
                for invoice_id in invoice_ids
                if state.invoices[invoice_id].status != INVOICE_VALIDATED
            ]
            if unvalidated:
                return Verdict.refused(
                    "INVOICE_NOT_VALIDATED", f"still unvalidated: {sorted(unvalidated)}"
                )
            if state.payment["status"] != PAYMENT_SETTLED:
                return Verdict.refused(
                    "PAYMENT_NOT_SETTLED",
                    f"payment is {state.payment['status']}; only the payment system's "
                    "own confirmation settles it",
                )
        if state.customer_resolution == RESOLUTION_PERSISTS:
            return Verdict.refused(
                "CUSTOMER_REPORTS_PERSISTENCE",
                "the customer says the issue persists; the operation cannot close "
                "until that is resolved",
            )
        if not cycle.completion_notice_sent:
            return Verdict.refused(
                "COMPLETION_NOTICE_NOT_SENT",
                f"the customer has not been told cycle {cycle.cycle_id!r} is done",
            )
        reopen_until = shift_minutes(
            context.now, self.policy.provisional_close_minutes, context="reopen window"
        )
        timer_id = f"timer_provisional_close:{cycle.cycle_id}"
        state.phase = "PROVISIONALLY_CLOSED"
        state.provisional_close = {
            "cycle_id": cycle.cycle_id,
            "accepted_at": context.now,
            "reopen_until": reopen_until,
            "timer_event_id": timer_id,
        }
        context.record(
            "provisional_close_entered",
            {
                "cycle_id": cycle.cycle_id,
                "reopen_until": reopen_until,
                "replay_final": False,
            },
        )
        context.schedule_timer(
            timer_id,
            "provisional_close_expired",
            "scheduler",
            self.policy.provisional_close_minutes,
            {"cycle_id": cycle.cycle_id},
        )
        # A customer report inside the window hangs off this closure and names
        # the visit whose work is being confirmed, so the closure publishes it.
        return Verdict.ok(
            code="PROVISIONALLY_CLOSED",
            cycle_id=cycle.cycle_id,
            bindings=_visit_bindings(cycle),
        )


# --------------------------------------------------------------- module helpers


def _bindings(**fields: str | None) -> dict[str, str]:
    """The identities an accepted effect established, with the unknown dropped.

    Two kinds of key live here. Most are identities the *environment* allocates
    — a visit, a checkpoint, a payment request — and binding them is the only
    way a conditional authored event can name something the agent had not yet
    caused to exist.

    ``invoice_id`` is the other kind, and it is here for a different reason. The
    agent names an invoice, but ``_act_request_invoice_validation`` accepts one
    only after resolving that name against the authoritative invoice record, so
    the identity the operation committed against is a fact the operation
    established rather than a string it was handed. Recording it makes the
    authored ``invoice_validation_completed`` event answer about the invoice
    that was actually admitted, and gives the evaluator the effect's own account
    of what it validated instead of leaving the final state as the only witness.

    Cycle ids, quote ids and evidence ids remain deliberately absent: those are
    authored facts an authored event is entitled to name for itself, and a
    binding that overwrote them would file a quote for new work against the
    cycle whose visit produced it.
    """
    return {name: value for name, value in fields.items() if value is not None}


def _visit_bindings(cycle: Cycle) -> dict[str, str]:
    """A visit identity under both names the authored vocabulary uses for it."""
    return _bindings(visit_id=cycle.visit_id, related_visit_id=cycle.visit_id)


def _allocate(prefix: str, used: set[str]) -> str:
    """The next free ``prefix_N``. Deterministic, monotonic, collision-free.

    Derived from what the operation already holds rather than from a counter on
    the state, so it is a pure function of the record: two runs of the same
    episode allocate the same identities, and replay does not depend on a
    sequence number surviving serialisation.
    """
    index = len(used) + 1
    while f"{prefix}_{index}" in used:
        index += 1
    return f"{prefix}_{index}"


def _allocate_visit_id(state: MaintenanceState) -> str:
    return _allocate(
        "visit",
        {cycle.visit_id for cycle in state.cycles.values() if cycle.visit_id is not None},
    )


def _allocate_checkpoint_id(state: MaintenanceState) -> str:
    # Approvals and exceptions share one identity space: both are checkpoints a
    # human answers by name, and two of them called `checkpoint_2` would give an
    # authored resolution two things it could be about.
    return _allocate("checkpoint", set(state.approvals) | set(state.exceptions))


def _allocate_payment_request_id(state: MaintenanceState) -> str:
    existing = state.payment["request_id"]
    return _allocate("payment", {str(existing)} if existing is not None else set())


def _cycle_for_visit(state: MaintenanceState, visit_id: str) -> Cycle | None:
    for cycle in state.cycles.values():
        if cycle.visit_id == visit_id:
            return cycle
    return None


def _binding_mismatch(approval: Approval, payload: Mapping[str, Any]) -> str | None:
    """Does an approver's decision name exactly the quote the checkpoint bound?"""
    expected = {
        "quote_id": approval.quote_id,
        "quote_version": approval.quote_version,
        "amount_minor": approval.amount_minor,
        "currency": approval.currency,
        "scope_digest": approval.scope_digest,
    }
    for field_name, value in expected.items():
        if payload.get(field_name) != value:
            return (
                f"the decision names {field_name}={payload.get(field_name)!r} but the "
                f"checkpoint bound {value!r}; an approval authorises one exact quote "
                "version"
            )
    return None


def _settlement_mismatch(
    state: MaintenanceState, payload: Mapping[str, Any]
) -> str | None:
    """Does a settlement name exactly the payment this operation is waiting on?

    The request id alone is not a binding. It says *which request* the payment
    system is answering and nothing about what it paid, so a settlement quoting
    the right request id with the wrong invoice, cycle, amount or currency is
    refused. Every field the request committed to is compared here, against
    the operation's own record rather than against the event's other fields —
    comparing an event to itself would prove only that it is self-consistent.
    """
    request = state.payment
    invoice = state.invoices.get(str(request["invoice_id"]))
    if invoice is None:
        return (
            f"the requested payment names invoice {request['invoice_id']!r}, which "
            "this operation no longer holds"
        )
    expected: dict[str, Any] = {
        "payment_request_id": request["request_id"],
        "invoice_id": invoice.invoice_id,
        "cycle_id": request["cycle_id"],
        "amount_minor": invoice.amount_minor,
        "currency": invoice.currency,
    }
    for field_name, value in expected.items():
        if payload.get(field_name) != value:
            return (
                f"the settlement names {field_name}={payload.get(field_name)!r} but "
                f"the current payment request bound {value!r}; a settlement confirms "
                "one exact payment, not any payment of this operation"
            )
    if invoice.cycle_id != request["cycle_id"]:
        return (
            f"invoice {invoice.invoice_id!r} belongs to cycle {invoice.cycle_id!r}, "
            f"not to the cycle {request['cycle_id']!r} the payment was requested for"
        )
    if invoice.status != INVOICE_VALIDATED:
        return (
            f"invoice {invoice.invoice_id!r} is {invoice.status}; a settlement cannot "
            "confirm payment of an invoice this operation has not validated"
        )
    return None


def _blocking_checkpoints(state: MaintenanceState) -> list[str]:
    return sorted(
        [approval.checkpoint_id for approval in state.open_approvals()]
        + [checkpoint.checkpoint_id for checkpoint in state.open_exceptions()]
    )


def _eligible_exception_approval(
    state: MaintenanceState, exception_type: str, checkpoint_id: str
) -> Approval | None:
    """Which approval, if any, justifies the exception the agent is claiming."""
    wanted = {
        "APPROVAL_REJECTED": APPROVAL_REJECTED,
        "APPROVAL_EXPIRED": APPROVAL_EXPIRED,
    }.get(exception_type)
    if wanted is None:
        return None
    approval = state.approvals.get(checkpoint_id)
    return approval if approval is not None and approval.status == wanted else None


def _required_evidence_verdict(
    state: MaintenanceState,
    action_type: str,
    payload: Mapping[str, Any],
    evidence_refs: Sequence[str],
    code: str,
) -> Verdict | None:
    contract = maintenance_spec.maintenance_action_evidence_contract()
    roots: set[str] = set()
    for selector in contract.evidence_for(action_type):
        roots.add(selector.root)
        if isinstance(selector, WorkEvidenceForInvoice):
            roots.update((selector.cycle_root, selector.authority_root))
        elif isinstance(selector, QuoteForSelectedApproval):
            roots.add(selector.target_root)
    resolution = resolve_required_evidence(
        contract,
        action_type,
        payload,
        state.project(sorted(roots)),
    )
    missing = sorted(set(resolution.required) - set(evidence_refs))
    if resolution.problem is None and not missing:
        return None
    detail = resolution.problem or f"missing {missing}"
    return Verdict.refused(
        code, f"required evidence did not resolve completely: {detail}"
    )


def _unresolved_evidence(
    state: MaintenanceState, evidence_refs: Sequence[str]
) -> Verdict | None:
    """Every cited reference must resolve to a record this operation actually holds.

    The interesting branch is the assertion one. A supplier saying "the work is
    done" is a real, recorded, citable *message* — and citing it as the basis of
    a payment is refused by its own name, which is what makes the difference
    between claim and fact observable in the trajectory.
    """
    assertion_ids = {str(item["assertion_id"]) for item in state.assertions}
    claimed = sorted(set(evidence_refs) & assertion_ids)
    if claimed:
        ref = claimed[0]
        return Verdict.refused(
            "NON_AUTHORITATIVE_EVIDENCE",
            f"{ref!r} is an actor claim, not an authoritative record; a claim "
            "cannot establish attendance, completed work or payment",
        )
    for ref in evidence_refs:
        if ref in state.authoritative_records:
            continue
        if ref in state.approvals or ref in state.exceptions:
            continue
        if ref in state.invoices or ref in state.quotes or ref in state.cycles:
            continue
        return Verdict.refused(
            "UNRESOLVED_EVIDENCE",
            f"{ref!r} does not resolve to any record this operation holds",
        )
    return None


def _open_obligation(
    state: MaintenanceState,
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
    state.obligations[obligation_id] = Obligation(
        obligation_id=obligation_id, kind=kind, cycle_id=cycle_id, due_at=due_at
    )
    context.record(
        "obligation_created",
        {"obligation_id": obligation_id, "kind": kind, "due_at": due_at},
    )


def _close_obligation(
    state: MaintenanceState,
    context: EnvironmentContext,
    obligation_id: str,
    status: str,
) -> None:
    obligation = state.obligations.get(obligation_id)
    if obligation is None or obligation.status != "OPEN":
        return
    obligation.status = status
    obligation.closed_at = context.now
    context.record(
        {
            OBLIGATION_DISCHARGED: "obligation_discharged",
            OBLIGATION_CANCELLED: "obligation_cancelled",
            OBLIGATION_BREACHED: "obligation_breached",
        }[status],
        {"obligation_id": obligation_id, "kind": obligation.kind},
    )


def _visit_timer_id(cycle_id: str) -> str:
    return f"timer_visit_followup:{cycle_id}"


def _visit_obligation_id(cycle_id: str) -> str:
    return f"visit_progress:{cycle_id}"


def _reminder_timer_id(checkpoint_id: str) -> str:
    return f"timer_approval_reminder:{checkpoint_id}"


def _deadline_timer_id(checkpoint_id: str) -> str:
    return f"timer_approval_deadline:{checkpoint_id}"


def _reminder_obligation_id(checkpoint_id: str) -> str:
    return f"approval_reminder:{checkpoint_id}"


def _decision_obligation_id(checkpoint_id: str) -> str:
    return f"approval_decision:{checkpoint_id}"


def _notice_obligation_id(cycle_id: str) -> str:
    return f"completion_notice:{cycle_id}"


_EVENT_REDUCERS = {
    "customer_issue_reported": MaintenanceOperation._reduce_issue_reported,
    "customer_resolution_reported": MaintenanceOperation._reduce_customer_resolution,
    "supplier_visit_response": MaintenanceOperation._reduce_visit_response,
    "supplier_work_reported": MaintenanceOperation._reduce_work_reported,
    "supplier_quote_received": MaintenanceOperation._reduce_quote_received,
    "supplier_invoice_received": MaintenanceOperation._reduce_invoice_received,
    "supplier_assertion_received": MaintenanceOperation._reduce_assertion,
    "approver_decision_received": MaintenanceOperation._reduce_approver_decision,
    "operator_exception_resolved": MaintenanceOperation._reduce_operator_resolution,
    "visit_attendance_verified": MaintenanceOperation._reduce_attendance_verified,
    "work_evidence_verified": MaintenanceOperation._reduce_work_evidence,
    "invoice_validation_completed": MaintenanceOperation._reduce_invoice_validation,
    "payment_settlement_confirmed": MaintenanceOperation._reduce_settlement,
    "approval_reminder_due": MaintenanceOperation._reduce_reminder_due,
    "approval_deadline_due": MaintenanceOperation._reduce_deadline_due,
    "visit_followup_due": MaintenanceOperation._reduce_visit_followup_due,
    "provisional_close_expired": MaintenanceOperation._reduce_provisional_expired,
}

_ACTION_HANDLERS = {
    "request_supplier_visit": MaintenanceOperation._act_request_visit,
    "request_approval": MaintenanceOperation._act_request_approval,
    "send_message": MaintenanceOperation._act_send_message,
    "authorise_supplier_work": MaintenanceOperation._act_authorise_supplier_work,
    "request_invoice_validation": MaintenanceOperation._act_request_invoice_validation,
    "request_payment": MaintenanceOperation._act_request_payment,
    "request_exception_resolution": (
        MaintenanceOperation._act_request_exception_resolution
    ),
}


__all__ = [
    "CHECKPOINT_APPROVAL",
    "CHECKPOINT_EXCEPTION",
    "MSG_APPROVAL_REMINDER",
    "MSG_COMPLETION_NOTICE",
    "MSG_TRANSFER_NOTICE",
    "MaintenanceOperation",
]
