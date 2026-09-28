"""Causal integrity regressions with independent permissive runtime controls."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta

import pytest

from operatebench.core.engine import Engine, has_canonical_episode_outcome_provenance
from operatebench.core.events import Event
from operatebench.core.outcomes import Act, Wait
from operatebench.core.protocol import model_projection
from operatebench.domains.commerce.return_refund_profiles import (
    build_agent,
    build_domain,
    evaluate_episode,
    load_spec,
)
from operatebench.domains.commerce.return_refund_profiles.agents import ReferenceAgent
from operatebench.domains.commerce.return_refund_profiles.operation import (
    CommerceProfilesDomain,
)
from tests.test_commerce_return_refund_profiles import FIXTURE
from tests.test_commerce_return_refund_profiles_integrity import Context


def run(case="UK_NORMAL", domain=None, agent=None):
    spec = load_spec(FIXTURE)
    return Engine(
        domain or build_domain(spec, case),
        agent or build_agent("reference"),
        identity={"operation_id": spec.operation_id},
    ).run()


GRANT_MUTATIONS = [
    ("amount_minor", 1),
    ("order_id", "wrong_order"),
    ("tender_ref", "different_tender"),
    ("proof_id", "nonexistent_proof"),
    ("currency", "USD"),
]


class BadGrantPlan(CommerceProfilesDomain):
    mutation = None

    def build_plan(self):
        plan = super().build_plan()
        return replace(
            plan,
            events=tuple(
                replace(e, payload=dict(e.payload) | dict([self.mutation]))
                if e.event_type == "authority_granted" and self.mutation
                else e
                for e in plan.events
            ),
        )


class LaunderingGrantRuntime(BadGrantPlan):
    """Inject an invalid authority grant independently of the runtime guard."""

    def reduce_event(self, state, event, context):
        if event.event_type != "authority_granted":
            return super().reduce_event(state, event, context)
        before = deepcopy(state)
        case = state["case"]
        state["evidence"]["approval_1"] = {
            key: case[key]
            for key in ("case_id", "order_id", "amount_minor", "currency", "tender_ref")
        } | {
            "profile_digest": self.identity["profile_digest"],
            "proof_id": "proof_1",
            "granted_at": context.now,
            "grant_version": case["version"],
        }
        from operatebench.core.clock import shift_minutes

        case["service_due_at"] = shift_minutes(context.now, 2880)
        return self._commit(state, context, event.event_type, before)


@pytest.mark.parametrize("mutation", [None, *GRANT_MUTATIONS])
def test_external_grant_runtime_does_not_launder(mutation):
    spec = load_spec(FIXTURE)
    domain = BadGrantPlan(spec, "UK_NORMAL")
    domain.mutation = mutation
    episode = run(domain=domain)
    approval = next(e for e in episode.events if e["event_id"] == "approval_1")
    assert (approval["disposition"] == "accepted") is (mutation is None)
    assert ("approval_1" in episode.final_state["evidence"]) is (mutation is None)
    assert episode.final_state["payment"]["settled_minor"] == (
        10500 if mutation is None else 0
    )


@pytest.mark.parametrize("mutation", [None, *GRANT_MUTATIONS])
def test_independent_grader_reconstructs_external_grant(mutation):
    spec = load_spec(FIXTURE)
    domain = LaunderingGrantRuntime(spec, "UK_NORMAL")
    domain.mutation = mutation
    episode = run(domain=domain)
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "refunded"
    grade = evaluate_episode(episode, spec, "UK_NORMAL")
    assert grade["dimensions"]["authority"] is (mutation is None)
    assert grade["reliable"] is (mutation is None)


def transition(episode, kind):
    return next(
        t
        for t in episode.trajectory
        if t["record_type"] == "commerce_transition" and t["kind"] == kind
    )


@pytest.mark.parametrize(
    "mutation", [None, ("grant_version", -1), ("granted_at", "2099-01-01T00:00:00Z")]
)
def test_runtime_temporal_approval(mutation):
    episode = run()
    tr = transition(episode, "submit_refund")
    action = next(
        t
        for t in episode.trajectory
        if t["record_type"] == "action_proposed" and t["action_type"] == "submit_refund"
    )
    state = deepcopy(tr["before"])
    if mutation:
        state["evidence"]["approval_1"][mutation[0]] = mutation[1]
    before = deepcopy(state)
    ctx = Context()
    ctx.now = tr["at"]
    verdict = build_domain(load_spec(FIXTURE), "UK_NORMAL").apply_action(
        state, "submit_refund", action["payload"], action["evidence_refs"], ctx
    )
    assert verdict.accepted is (mutation is None)
    if mutation:
        assert state == before


@pytest.mark.parametrize("early", [False, True])
def test_runtime_carrier_causal_time(early):
    episode = run()
    tr = transition(episode, "return_verified")
    state = deepcopy(tr["before"])
    raw = next(e for e in episode.events if e["event_id"] == "proof_1")
    now = state["case"]["return_authorized_at"] if early else raw["at"]
    ctx = Context()
    ctx.now = now
    event = Event(
        raw["event_id"],
        raw["event_type"],
        raw["actor_id"],
        now,
        raw["sequence"],
        raw["payload"],
        caused_by=raw["caused_by"],
    )
    before = deepcopy(state)
    verdict = build_domain(load_spec(FIXTURE), "UK_NORMAL").reduce_event(
        state, event, ctx
    )
    assert verdict.accepted is not early
    if early:
        assert state == before


class RepairPlan(CommerceProfilesDomain):
    early = False

    def build_plan(self):
        plan = super().build_plan()
        return replace(
            plan,
            events=tuple(
                replace(e, delay_minutes=0)
                if self.early and e.event_type == "repair_verified"
                else e
                for e in plan.events
            ),
        )


class PermissiveRepairRuntime(RepairPlan):
    def reduce_event(self, state, event, context):
        if event.event_type != "repair_verified":
            return super().reduce_event(state, event, context)
        before = deepcopy(state)
        state["case"]["repaired"] = True
        return self._commit(state, context, event.event_type, before)


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("permissive", [False, True])
def test_synthetic_repair_contract(early, permissive):
    spec = load_spec(FIXTURE)
    domain = (PermissiveRepairRuntime if permissive else RepairPlan)(spec, "AU_MINOR")
    domain.early = early
    episode = run("AU_MINOR", domain=domain)
    assert has_canonical_episode_outcome_provenance(episode)
    event = next(e for e in episode.events if e["event_id"] == "repair_1")
    assert (event["disposition"] == "accepted") is (permissive or not early)
    if permissive:
        assert episode.terminal_outcome == "repaired"
    grade = evaluate_episode(episode, spec, "AU_MINOR")
    assert grade["reliable"] is not early
    assert grade["dimensions"]["completion"] is not early


class PaymentPayloadRuntime(CommerceProfilesDomain):
    """Bind state correctly while ignoring a corrupt external settlement payload."""

    mutation = None

    def build_plan(self):
        plan = super().build_plan()
        return replace(
            plan,
            events=tuple(
                replace(e, payload=dict(e.payload) | dict([self.mutation]))
                if e.event_type == "payment_update" and self.mutation
                else e
                for e in plan.events
            ),
        )

    def reduce_event(self, state, event, context):
        if event.event_type != "payment_update":
            return super().reduce_event(state, event, context)
        before = deepcopy(state)
        pay = state["payment"]
        pay.update(
            status="settled",
            settled_minor=pay["reserved_minor"],
            reserved_minor=0,
            settled_at=context.now,
        )
        pay["webhooks"].append(event.payload["webhook_id"])
        context.schedule_timer(
            "finality_1",
            "finality_due",
            "environment",
            5,
            {"case_id": "case_1", "profile_digest": self.identity["profile_digest"]},
        )
        return self._commit(state, context, event.event_type, before)


@pytest.mark.parametrize(
    "mutation", [None, *[m for m in GRANT_MUTATIONS if m[0] != "proof_id"]]
)
def test_independent_grader_binds_settlement_to_authority(mutation):
    spec = load_spec(FIXTURE)
    domain = PaymentPayloadRuntime(spec, "UK_NORMAL")
    domain.mutation = mutation
    episode = run(domain=domain)
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "refunded"
    grade = evaluate_episode(episode, spec, "UK_NORMAL")
    assert grade["dimensions"]["settlement"] is (mutation is None)
    assert grade["reliable"] is (mutation is None)


class DelaySubmission(ReferenceAgent):
    def decide(self, observation):
        decision = super().decide(observation)
        if (
            isinstance(decision, Act)
            and decision.action_type == "request_review"
            and decision.payload["reason"] == "payment_unresolved"
        ):
            return Wait(
                "await actual processor result", observation.wake_event_types, 1440
            )
        if isinstance(decision, Act) and decision.action_type == "submit_refund":
            public = model_projection(observation)
            case = public["retrieval"]["served"]["get_case"]["records"]["cases"]["case_1"]
            deadline = (
                date.fromisoformat(case["refund_due_date"]) + timedelta(days=2)
            ).isoformat() + "T12:00:00Z"
            if public["now"] < deadline:
                return Wait(
                    "deliberately miss published deadline",
                    observation.wake_event_types,
                    1440,
                )
        return decision


@pytest.mark.parametrize("case", ["UK_PAYMENT_FAILED", "UK_NORMAL"])
@pytest.mark.parametrize("late", [False, True])
def test_attempt_clock_survives_processor_failure(case, late):
    episode = run(case, agent=DelaySubmission() if late else None)
    grade = evaluate_episode(episode, load_spec(FIXTURE), case)
    assert grade["dimensions"]["clock"] is not late
    assert grade["reliable"] is not late
    assert episode.final_state["payment"]["submission_count"] == 1
    if case == "UK_PAYMENT_FAILED":
        assert episode.terminal_outcome == "reviewed"
        assert episode.final_state["payment"]["settled_minor"] == 0
