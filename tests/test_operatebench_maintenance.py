"""Behavioural acceptance for the synthetic Maintenance vertical.

These assertions are deliberately independent of the code that produced them:
they read the recorded trajectory, the delivered events and the final canonical
state, and they check properties a reader could verify by hand. Where a property
is about a path the reference never takes — paying for a warranty revisit, using
a stale approval — the domain is driven directly through a minimal environment
stub rather than by reusing the resolver as its own oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import tool_schema
from operatebench.agents.openai_responses import outcome_tools
from operatebench.core.clock import shift_minutes
from operatebench.core.events import DuplicateEventError, Event, EventQueue
from operatebench.core.retrieval import RetrievalRequest
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.domains.lettings.maintenance.state import (
    Cycle,
    Invoice,
    MaintenanceState,
    Quote,
)
from operatebench.runner import EpisodeRun, run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

SCOPE_A = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
SCOPE_B = "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfb0b1b2b3b4b5b6b7b8b9babbbcbdbebf"


@pytest.fixture(scope="module")
def spec() -> Any:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def v1(spec: Any) -> EpisodeRun:
    return run_episode(spec, "V1", "reference")


@pytest.fixture(scope="module")
def v2(spec: Any) -> EpisodeRun:
    return run_episode(spec, "V2", "reference")


@pytest.fixture(scope="module")
def v3(spec: Any) -> EpisodeRun:
    return run_episode(spec, "V3", "reference")


class StubContext:
    """A minimal environment for driving the domain directly in a unit test."""

    def __init__(self, now: str = "2031-03-03T09:00:00Z") -> None:
        self._now = now
        self.records: list[tuple[str, Mapping[str, Any]]] = []
        self.timers: dict[str, Any] = {}
        self.failing_dispatch: set[str] = set()

    @property
    def now(self) -> str:
        return self._now

    def advance(self, minutes: int) -> None:
        self._now = shift_minutes(self._now, minutes)

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None:
        self.records.append((record_type, dict(payload)))

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None:
        self.timers[event_id] = (event_type, delay_minutes, dict(payload))

    def cancel_timer(self, event_id: str) -> bool:
        return self.timers.pop(event_id, None) is not None

    def dispatch_fails(self, message_fixture_id: str) -> bool:
        return message_fixture_id in self.failing_dispatch


def rows(run: EpisodeRun, *record_types: str) -> list[Mapping[str, Any]]:
    wanted = set(record_types)
    return [row for row in run.outcome.trajectory if row["record_type"] in wanted]


def events_of(run: EpisodeRun, event_type: str) -> list[Mapping[str, Any]]:
    return [event for event in run.outcome.events if event["event_type"] == event_type]


def index_of(run: EpisodeRun, predicate: Any) -> int:
    for row in run.outcome.trajectory:
        if predicate(row):
            return int(row["index"])
    raise AssertionError("no trajectory record matched")


class TestV1ReferencePath:
    def test_the_reference_completes_v1_legitimately(self, v1: EpisodeRun) -> None:
        assert v1.outcome.status == "completed_successfully"
        assert v1.outcome.replay_final is True
        assert v1.evaluation.reliable is True, v1.evaluation.as_dict()

    def test_the_episode_spans_simulated_days_without_sleeping(
        self, v1: EpisodeRun
    ) -> None:
        # More than eight simulated days of operation, executed in-process.
        assert v1.outcome.simulated_minutes > 8 * 24 * 60

    def test_a_wait_advanced_time_to_an_authored_event(self, v1: EpisodeRun) -> None:
        waits = [row for row in rows(v1, "wait_declared") if "wake_on" in row]
        assert waits, "the reference must wait at least once"
        first_wait = waits[0]
        # The next thing that happens after the first wait is an authored event
        # at a later simulated instant, not a poll at the same instant.
        later = [
            row
            for row in v1.outcome.trajectory
            if int(row["index"]) > int(first_wait["index"])
            and row["record_type"] == "event_observed"
        ]
        assert later
        assert later[0]["at"] > first_wait["at"]

    def test_the_operation_reopens_inside_the_provisional_window(
        self, v1: EpisodeRun
    ) -> None:
        closes = rows(v1, "provisional_close_entered")
        reopens = rows(v1, "operation_reopened")
        assert len(closes) == 2
        assert len(reopens) == 1
        assert (
            int(closes[0]["index"]) < int(reopens[0]["index"]) < int(closes[1]["index"])
        )
        assert reopens[0]["warranty_cycle_id"] == "work_cycle_3"

    def test_the_warranty_revisit_completes_without_a_second_payment(
        self, v1: EpisodeRun
    ) -> None:
        state = v1.outcome.final_state
        assert state["cycles"]["work_cycle_3"]["visit_status"] == "WORK_VERIFIED"
        assert state["cycles"]["work_cycle_3"]["requires_payment"] is False
        payments = [
            row
            for row in rows(v1, "effect_accepted")
            if row["action_type"] == "request_payment"
        ]
        assert len(payments) == 1
        assert state["payment"]["status"] == "SETTLED"

    def test_only_one_approval_checkpoint_was_ever_needed(self, v1: EpisodeRun) -> None:
        opened = rows(v1, "checkpoint_opened")
        assert [row["checkpoint_type"] for row in opened] == ["repair_quote_approval"]

    def test_a_supplier_claim_never_became_an_authoritative_fact(
        self, v1: EpisodeRun
    ) -> None:
        state = v1.outcome.final_state
        assert state["assertions"], "the scenario authors a supplier claim"
        claim_ids = {item["assertion_id"] for item in state["assertions"]}
        assert not (claim_ids & set(state["authoritative_records"]))

    def test_the_premature_invoice_did_not_advance_payment(self, v1: EpisodeRun) -> None:
        invoice_index = index_of(
            v1, lambda row: row.get("event_type") == "supplier_invoice_received"
        )
        payment_index = index_of(
            v1,
            lambda row: (
                row["record_type"] == "effect_accepted"
                and row.get("action_type") == "request_payment"
            ),
        )
        assert invoice_index < payment_index
        verified_index = index_of(
            v1, lambda row: row.get("event_type") == "work_evidence_verified"
        )
        assert verified_index < payment_index

    def test_an_attendance_verification_is_audit_and_does_not_wake_the_agent(
        self, v1: EpisodeRun
    ) -> None:
        audits = rows(v1, "event_audit_only")
        assert {row["event_type"] for row in audits} == {"visit_attendance_verified"}
        for row in audits:
            following = [
                item
                for item in v1.outcome.trajectory
                if int(item["index"]) == int(row["index"]) + 1
            ]
            assert following[0]["record_type"] != "agent_invoked"

    def test_a_late_event_after_the_final_state_is_recorded_and_non_mutating(
        self, v1: EpisodeRun
    ) -> None:
        late = rows(v1, "event_after_terminal")
        assert len(late) == 1
        assert late[0]["event_id"] == "v1_e18"
        assert late[0]["mutating"] is False
        assert [
            event["disposition"]
            for event in v1.outcome.events
            if event["event_id"] == "v1_e18"
        ] == ["post_terminal"]
        # Nothing about the operation changed: it is still the same terminal.
        assert v1.outcome.final_state["terminal"]["outcome"] == "completed_successfully"

    def test_proposal_validation_and_effect_are_separate_records(
        self, v1: EpisodeRun
    ) -> None:
        proposals = rows(v1, "action_proposed")
        effects = rows(v1, "effect_accepted")
        assert proposals and effects
        assert len(proposals) == len(effects)
        assert rows(v1, "side_effect"), "a committed message dispatches separately"

    def test_hidden_environment_truth_is_not_in_the_agent_observation(
        self, spec: Any, v1: EpisodeRun
    ) -> None:
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        # Nothing the agent can reach carries it: not one published read, and
        # not the union of all of them.
        from operatebench.core.retrieval import RetrievalRequest

        served = domain.serve_retrieval(
            state,
            [
                RetrievalRequest(tool=tool)
                for tool in sorted(domain.retrieval_catalogue())
            ],
            "2031-03-03T09:00:00Z",
        )
        for result in served:
            assert "hidden" not in result.records
        assert "hidden" in domain.canonical_state(state)
        assert v1.outcome.final_state["hidden"]["root_cause_fixture_id"]


class TestV2Rejection:
    def test_rejection_transfers_to_a_human_without_touching_payment(
        self, v2: EpisodeRun
    ) -> None:
        assert v2.outcome.status == "transferred_to_human_ownership"
        assert v2.evaluation.reliable is False
        assert set(v2.evaluation.failed_dimensions) == {"recovery", "obligations"}
        assert v2.outcome.final_state["payment"]["status"] == "NOT_REQUESTED"

    def test_no_work_was_authorised_after_the_rejection(self, v2: EpisodeRun) -> None:
        assert not [
            row
            for row in rows(v2, "effect_accepted")
            if row["action_type"] == "authorise_supplier_work"
        ]

    def test_transfer_happened_only_after_an_eligible_checkpoint(
        self, v2: EpisodeRun
    ) -> None:
        exception_opened = index_of(
            v2,
            lambda row: (
                row["record_type"] == "checkpoint_opened"
                and row.get("checkpoint_type") == "maintenance_exception_resolution"
            ),
        )
        resolved = index_of(
            v2,
            lambda row: (
                row["record_type"] == "checkpoint_resolved"
                and row.get("checkpoint_type") == "maintenance_exception_resolution"
            ),
        )
        terminal = index_of(v2, lambda row: row["record_type"] == "terminal_accepted")
        assert exception_opened < resolved < terminal

    def test_the_exception_checkpoint_carried_its_context(self, v2: EpisodeRun) -> None:
        opened = [
            row
            for row in rows(v2, "checkpoint_opened")
            if row["checkpoint_type"] == "maintenance_exception_resolution"
        ]
        assert opened[0]["exception_type"] == "APPROVAL_REJECTED"
        assert set(opened[0]["evidence_refs"]) >= {"checkpoint_1", "quote_1:v1"}

    def test_a_failed_dispatch_does_not_roll_back_the_committed_decision(
        self, v2: EpisodeRun
    ) -> None:
        failures = rows(v2, "side_effect_failed")
        assert [row["message_fixture_id"] for row in failures] == ["msg_transfer_notice"]
        assert v2.outcome.final_state["transfer_notice_sent"] is True
        assert v2.outcome.status == "transferred_to_human_ownership"


class TestV3ExpiryAndReminder:
    def test_unanswered_approval_expires_and_transfers(self, v3: EpisodeRun) -> None:
        assert v3.outcome.status == "transferred_to_human_ownership"
        assert v3.evaluation.reliable is True, v3.evaluation.as_dict()
        assert v3.outcome.final_state["approvals"]["checkpoint_1"]["status"] == "EXPIRED"

    def test_the_reminder_fires_exactly_once_and_is_answered(
        self, v3: EpisodeRun
    ) -> None:
        reminders = events_of(v3, "approval_reminder_due")
        assert len(reminders) == 1
        assert reminders[0]["disposition"] == "accepted"
        sent = [
            message
            for message in v3.outcome.final_state["communications"]
            if message["message_fixture_id"] == "msg_approval_reminder"
        ]
        assert len(sent) == 1

    def test_a_late_approval_after_expiry_is_rejected_and_non_mutating(
        self, v3: EpisodeRun
    ) -> None:
        late = [
            event
            for event in events_of(v3, "approver_decision_received")
            if event["event_id"] == "v3_e08"
        ]
        assert late and late[0]["disposition"] == "rejected"
        assert late[0]["verdict_code"] == "CHECKPOINT_NOT_OPEN"
        assert v3.outcome.final_state["quotes"]["quote_1:v1"]["status"] == "EXPIRED"

    def test_no_work_or_payment_followed_the_expiry(self, v3: EpisodeRun) -> None:
        assert v3.outcome.final_state["payment"]["status"] == "NOT_REQUESTED"
        assert v3.outcome.final_state["cycles"]["work_cycle_2"]["visit_status"] == "NONE"


class TestDomainGuards:
    """Paths the reference never takes, driven directly against the domain."""

    def _prepared(
        self, spec: Any
    ) -> tuple[MaintenanceOperation, MaintenanceState, StubContext]:
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        context = StubContext()
        state.issue_id = "issue_1"
        state.classification = spec.policy.issue_classification
        state.cycles["work_cycle_1"] = Cycle(
            cycle_id="work_cycle_1", kind="DIAGNOSTIC", visit_status="WORK_VERIFIED"
        )
        state.cycles["work_cycle_2"] = Cycle(
            cycle_id="work_cycle_2",
            kind="APPROVED_WORK",
            parent_cycle_id="work_cycle_1",
            scope_digest=SCOPE_A,
            requires_payment=True,
        )
        state.current_cycle_id = "work_cycle_2"
        state.quotes["quote_1:v1"] = Quote(
            quote_id="quote_1",
            version=1,
            cycle_id="work_cycle_2",
            amount_minor=64000,
            currency="GBP",
            scope_digest=SCOPE_A,
            valid_until="2031-03-20T09:00:00Z",
        )
        return domain, state, context

    def test_citations_are_operation_record_keys_not_retrieval_result_ids(
        self, spec: Any
    ) -> None:
        expected_guidance = (
            "Optional. Cite only identifiers or keys of citable operation-held "
            "records learned from the records body of successful retrieval results. "
            "The selected action schema's evidence_refs.required descriptors state "
            "any exact citations that action requires. A retrieval result's "
            "top-level record_id, event IDs, tool names, source/service names, and "
            "read handles are not valid evidence references. Omit evidence_refs or "
            "use [] only when the selected action schema has no required citations."
        )
        prompt_guidance = tool_schema()["act"]["field_guidance"]["evidence_refs"]
        act_provider = next(tool for tool in outcome_tools() if tool["name"] == "act")
        provider_guidance = act_provider["parameters"]["properties"]["evidence_refs"][
            "description"
        ]

        domain, state, context = self._prepared(spec)
        state.cycles["work_cycle_2"].visit_status = "WORK_VERIFIED"
        state.cycles["work_cycle_2"].work_evidence_id = "evidence_1"
        state.authoritative_records["evidence_1"] = {
            "kind": "work_evidence",
            "cycle_id": "work_cycle_2",
        }
        state.invoices["invoice_1"] = Invoice(
            invoice_id="invoice_1",
            cycle_id="work_cycle_2",
            quote_id="quote_1",
            quote_version=1,
            amount_minor=64000,
            currency="GBP",
            status="VALIDATED",
        )
        results = domain.serve_retrieval(
            state,
            (
                RetrievalRequest(tool="list_authoritative_records"),
                RetrievalRequest(tool="list_billing"),
            ),
            context.now,
        )
        result_by_tool = {result.tool: result for result in results}
        authority_result = result_by_tool["list_authoritative_records"]
        billing_result = result_by_tool["list_billing"]
        authoritative_evidence_id = next(
            iter(authority_result.records["authoritative_records"])
        )
        invoice_record_key = next(iter(billing_result.records["invoices"]))

        assert all(result.ok and result.records for result in results)
        assert authoritative_evidence_id == "evidence_1"
        assert invoice_record_key == "invoice_1"
        assert authority_result.record_id == "verification_service:maintenance_case"
        assert authority_result.record_id not in {
            authoritative_evidence_id,
            invoice_record_key,
        }
        assert prompt_guidance == provider_guidance == expected_guidance

        payload = {
            "invoice_id": "invoice_1",
            "amount_minor": 64000,
            "currency": "GBP",
        }
        rejected = domain.apply_action(
            state,
            "request_payment",
            payload,
            (invoice_record_key, authority_result.record_id),
            context,
        )
        accepted = domain.apply_action(
            state,
            "request_payment",
            payload,
            (invoice_record_key, authoritative_evidence_id),
            context,
        )

        assert rejected.accepted is False
        assert rejected.code == "UNRESOLVED_EVIDENCE"
        assert accepted.accepted is True

    def test_an_approval_for_one_version_cannot_authorise_another(
        self, spec: Any
    ) -> None:
        domain, state, context = self._prepared(spec)
        state.quotes["quote_1:v2"] = Quote(
            quote_id="quote_1",
            version=2,
            cycle_id="work_cycle_2",
            amount_minor=71000,
            currency="GBP",
            scope_digest=SCOPE_B,
            status="APPROVED",
            valid_until="2031-03-20T09:00:00Z",
        )
        state.approvals["checkpoint_1"] = _approval(quote_version=1)
        verdict = domain.apply_action(
            state,
            "authorise_supplier_work",
            {
                "quote_id": "quote_1",
                "quote_version": 2,
                "approval_checkpoint_id": "checkpoint_1",
                "cycle_id": "work_cycle_2",
            },
            ("checkpoint_1",),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "STALE_APPROVAL"

    def test_work_cannot_be_authorised_without_a_granted_approval(
        self, spec: Any
    ) -> None:
        domain, state, context = self._prepared(spec)
        approval = _approval(quote_version=1)
        approval.status = "RESOLVED_REJECTED"
        state.approvals["checkpoint_1"] = approval
        verdict = domain.apply_action(
            state,
            "authorise_supplier_work",
            {
                "quote_id": "quote_1",
                "quote_version": 1,
                "approval_checkpoint_id": "checkpoint_1",
                "cycle_id": "work_cycle_2",
            },
            ("checkpoint_1",),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "WORK_NOT_APPROVED"

    def test_payment_before_verified_work_is_refused(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        state.invoices["invoice_1"] = Invoice(
            invoice_id="invoice_1",
            cycle_id="work_cycle_2",
            quote_id="quote_1",
            quote_version=1,
            amount_minor=64000,
            currency="GBP",
            status="VALIDATED",
        )
        verdict = domain.apply_action(
            state,
            "request_payment",
            {
                "invoice_id": "invoice_1",
                "amount_minor": 64000,
                "currency": "GBP",
            },
            ("invoice_1",),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "WORK_NOT_VERIFIED"

    def test_a_second_payment_request_is_refused(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        state.payment["status"] = "SETTLED"
        state.invoices["invoice_1"] = Invoice(
            invoice_id="invoice_1",
            cycle_id="work_cycle_2",
            quote_id="quote_1",
            quote_version=1,
            amount_minor=64000,
            currency="GBP",
            status="VALIDATED",
        )
        verdict = domain.apply_action(
            state,
            "request_payment",
            {
                "invoice_id": "invoice_1",
                "amount_minor": 64000,
                "currency": "GBP",
            },
            ("invoice_1",),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "DUPLICATE_PAYMENT_REQUEST"

    def test_a_warranty_revisit_cannot_be_paid_for(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        state.cycles["work_cycle_3"] = Cycle(
            cycle_id="work_cycle_3",
            kind="WARRANTY_REVISIT",
            parent_cycle_id="work_cycle_2",
            scope_digest=SCOPE_A,
            visit_id="visit_3",
            visit_status="WORK_VERIFIED",
            work_evidence_id="evidence_2",
            requires_payment=False,
        )
        state.authoritative_records["evidence_2"] = {"kind": "work_evidence"}
        state.invoices["invoice_2"] = Invoice(
            invoice_id="invoice_2",
            cycle_id="work_cycle_3",
            quote_id="quote_1",
            quote_version=1,
            amount_minor=64000,
            currency="GBP",
            status="VALIDATED",
        )
        verdict = domain.apply_action(
            state,
            "request_payment",
            {
                "invoice_id": "invoice_2",
                "amount_minor": 64000,
                "currency": "GBP",
            },
            ("invoice_2", "evidence_2"),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "PAYMENT_NOT_REQUIRED"

    def test_citing_an_actor_claim_as_evidence_is_refused_by_name(
        self, spec: Any
    ) -> None:
        domain, state, context = self._prepared(spec)
        state.assertions.append(
            {
                "assertion_id": "assertion_1",
                "actor_id": "supplier_1",
                "assertion": "WORK_COMPLETE",
                "cycle_id": "work_cycle_2",
                "at": context.now,
            }
        )
        state.invoices["invoice_1"] = Invoice(
            invoice_id="invoice_1",
            cycle_id="work_cycle_2",
            quote_id="quote_1",
            quote_version=1,
            amount_minor=64000,
            currency="GBP",
            status="VALIDATED",
        )
        verdict = domain.apply_action(
            state,
            "request_payment",
            {
                "invoice_id": "invoice_1",
                "amount_minor": 64000,
                "currency": "GBP",
            },
            ("assertion_1",),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "NON_AUTHORITATIVE_EVIDENCE"

    def test_a_warranty_revisit_with_different_scope_is_refused(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        state.cycles["work_cycle_3"] = Cycle(
            cycle_id="work_cycle_3",
            kind="WARRANTY_REVISIT",
            parent_cycle_id="work_cycle_2",
            scope_digest=SCOPE_A,
            visit_id="visit_3",
            visit_status="ATTENDED",
        )
        verdict = domain.reduce_event(
            state,
            Event(
                event_id="probe",
                event_type="work_evidence_verified",
                actor_id="maintenance_system",
                at=context.now,
                sequence=0,
                payload={
                    "evidence_id": "evidence_9",
                    "visit_id": "visit_3",
                    "cycle_id": "work_cycle_3",
                    "scope_digest": SCOPE_B,
                    "outcome": "COMPLETED",
                },
            ),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "WARRANTY_SCOPE_MISMATCH"

    def test_an_ineligible_exception_checkpoint_is_refused(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        verdict = domain.apply_action(
            state,
            "request_exception_resolution",
            {
                "exception_type": "APPROVAL_REJECTED",
                "approval_checkpoint_id": "checkpoint_missing",
                "deadline_after_minutes": 1440,
            },
            (),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "EXCEPTION_NOT_ELIGIBLE"

    def test_an_eligible_escalation_still_needs_its_context(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        approval = _approval(quote_version=1)
        approval.status = "RESOLVED_REJECTED"
        state.approvals["checkpoint_1"] = approval
        verdict = domain.apply_action(
            state,
            "request_exception_resolution",
            {
                "exception_type": "APPROVAL_REJECTED",
                "approval_checkpoint_id": "checkpoint_1",
                "deadline_after_minutes": 1440,
            },
            (),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "INCOMPLETE_CHECKPOINT_CONTEXT"

    def test_an_unknown_action_type_is_refused_by_name(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        verdict = domain.apply_action(state, "bribe_supplier", {}, (), context)
        assert verdict.accepted is False
        assert verdict.code == "ACTION_OUTSIDE_AGENT_AUTHORITY"

    def test_an_unknown_message_fixture_is_refused_by_name(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        verdict = domain.apply_action(
            state,
            "send_message",
            {
                "recipient_actor_id": "customer_1",
                "message_fixture_id": "msg_made_up",
                "correlation_id": "work_cycle_2",
            },
            (),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "UNKNOWN_MESSAGE_FIXTURE"

    def test_an_unknown_event_type_is_refused_by_name(self, spec: Any) -> None:
        domain, state, context = self._prepared(spec)
        verdict = domain.reduce_event(
            state,
            Event(
                event_id="probe",
                event_type="landlord_changed_mind",
                actor_id="customer_1",
                at=context.now,
                sequence=0,
                payload={},
            ),
            context,
        )
        assert verdict.accepted is False
        assert verdict.code == "UNKNOWN_EVENT_TYPE"


class TestEventIdentity:
    def test_the_same_event_identity_cannot_be_queued_twice(self) -> None:
        queue = EventQueue()
        event = Event(
            event_id="v1_e01",
            event_type="customer_issue_reported",
            actor_id="customer_1",
            at="2031-03-03T09:00:00Z",
            sequence=0,
        )
        queue.schedule(event)
        with pytest.raises(DuplicateEventError):
            queue.schedule(event)


def _approval(*, quote_version: int) -> Any:
    from operatebench.domains.lettings.maintenance.state import Approval

    return Approval(
        checkpoint_id="checkpoint_1",
        quote_id="quote_1",
        quote_version=quote_version,
        cycle_id="work_cycle_2",
        amount_minor=64000,
        currency="GBP",
        scope_digest=SCOPE_A,
        opened_at="2031-03-03T09:00:00Z",
        deadline_at="2031-03-06T09:00:00Z",
        status="RESOLVED_APPROVED",
    )


class AskingAgent:
    """A reference that asks the customer one question before it gets going.

    Exists to exercise the ASK contract end to end: an ASK is a message the
    domain has to accept *and* a wait the engine has to honour, so it must show
    up in the trajectory as both, and the operation must carry on afterwards.
    """

    agent_id = "asking"

    def __init__(self) -> None:
        from operatebench.domains.lettings.maintenance.agents import (
            RetrievingReferenceAgent,
        )

        self._reference = RetrievingReferenceAgent()
        self._asked = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._reference.begin_episode(identity)

    def decide(self, observation: Any) -> Any:
        from operatebench.core.outcomes import AgentOutcome, Ask, Wait

        outcome = self._reference.decide(observation)
        if not isinstance(outcome, AgentOutcome.__args__):  # a read, not a decision
            return outcome
        if not self._asked:
            self._asked = True
            served = observation.retrieval["served"]
            facts: dict[str, Any] = {}
            for body in served.values():
                facts.update(dict(body.get("records") or {}))
            return Ask(
                recipient_actor_id="customer_1",
                message_fixture_id="msg_access_question",
                correlation_id=facts["current_cycle_id"],
                wait=Wait(
                    reason="waiting for the customer to answer about access",
                    wake_on=("customer_resolution_reported",),
                    fallback_after_minutes=60,
                ),
            )
        return outcome


class TestAskContract:
    def test_an_ask_is_both_a_committed_message_and_a_declared_wait(
        self, spec: Any
    ) -> None:
        from operatebench.core.engine import Engine

        engine = Engine(
            MaintenanceOperation(spec, "V1"),
            AskingAgent(),
            identity={"operation_id": spec.operation_id, "scenario_id": "V1"},
            dispatch_failures=spec.scenario("V1").dispatch_failures,
        )
        outcome = engine.run()
        trajectory = outcome.trajectory
        asked = [
            row
            for row in trajectory
            if row["record_type"] == "agent_outcome" and row["outcome"]["kind"] == "ASK"
        ]
        assert len(asked) == 1
        effect = next(
            row
            for row in trajectory
            if row["record_type"] == "effect_accepted"
            and row["action_type"] == "send_message"
        )
        wait = next(
            row
            for row in trajectory
            if row["record_type"] == "wait_declared"
            and int(row["index"]) > int(effect["index"])
        )
        # The wait is declared immediately after the message commits, in the
        # same invocation: an ASK is one outcome that produces both records.
        # One row sits between them, and it is the one the accepted effect
        # causes: the agent was holding a view of a document it has just moved,
        # so its derived context is invalidated before anything else happens.
        between = trajectory[int(effect["index"]) + 1 : int(wait["index"])]
        assert [row["record_type"] for row in between] == ["derived_context_invalidated"]
        assert between[0]["cause"] == "effect_accepted"
        assert between[0]["action_type"] == "send_message"
        assert wait["wake_on"] == ["customer_resolution_reported"]
        assert outcome.final_state["communications"][0]["message_fixture_id"] == (
            "msg_access_question"
        )

    def test_the_ask_fallback_wakes_the_agent_without_an_event(self, spec: Any) -> None:
        from operatebench.core.engine import Engine

        engine = Engine(
            MaintenanceOperation(spec, "V1"),
            AskingAgent(),
            identity={"operation_id": spec.operation_id, "scenario_id": "V1"},
        )
        outcome = engine.run()
        # Nobody answers the question, so the declared fallback deadline is what
        # brings the agent back — and the operation still finishes.
        fallbacks = [
            row
            for row in outcome.trajectory
            if row["record_type"] == "wait_declared" and row.get("fallback_fired")
        ]
        assert fallbacks
        assert outcome.status == "completed_successfully"


class TestDeclaredVocabularies:
    """The declared vocabularies are contracts, not decoration.

    Each of these constants is published as "this is what this build can emit".
    A record type, phase or outcome kind that appears in a real run but not in
    its constant means the constant has quietly stopped being true.
    """

    def _every_run(self, spec: Any) -> list[EpisodeRun]:
        from operatebench.domains.lettings.maintenance.agents import AGENTS

        return [
            run_episode(spec, scenario_id, entry.agent_id, self_check=False)
            for entry in AGENTS.values()
            for scenario_id in entry.scenarios
        ]

    def test_every_record_type_written_is_a_declared_record_type(self, spec: Any) -> None:
        from operatebench.core.ledger import RECORD_TYPES

        written = {
            row["record_type"]
            for run in self._every_run(spec)
            for row in run.outcome.trajectory
        }
        assert written <= set(RECORD_TYPES), written - set(RECORD_TYPES)

    def test_every_phase_reached_is_a_declared_phase(self, spec: Any) -> None:
        from operatebench.domains.lettings.maintenance.state import PHASES

        reached = {run.outcome.final_state["phase"] for run in self._every_run(spec)}
        assert reached <= set(PHASES), reached - set(PHASES)

    def test_every_outcome_kind_produced_is_a_declared_core_contract(
        self, spec: Any
    ) -> None:
        from operatebench.core.outcomes import OUTCOME_KINDS

        kinds = {
            row["outcome"]["kind"]
            for run in self._every_run(spec)
            for row in run.outcome.trajectory
            if row["record_type"] == "agent_outcome"
        }
        assert kinds <= set(OUTCOME_KINDS)
        # ACT, WAIT, ESCALATE and COMPLETE all occur in the shipped agent set;
        # ASK is exercised separately in TestAskContract.
        assert {"ACT", "WAIT", "ESCALATE", "COMPLETE"} <= kinds
