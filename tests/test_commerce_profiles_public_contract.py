"""Public commerce preconditions and refusal causes, not relaxed grading."""

from copy import deepcopy

import pytest

from operatebench.core.outcomes import Act
from operatebench.core.protocol import AgentObservation
from operatebench.domains.commerce.return_refund_profiles import (
    build_domain,
    evaluate_episode,
    load_spec,
)
from operatebench.domains.commerce.return_refund_profiles.agents import ReferenceAgent
from tests import test_property_compliance_public_contract as compliance_contract
from tests.test_commerce_return_refund_profiles import FIXTURE
from tests.test_commerce_return_refund_profiles_integrity import Context
from tests.test_commerce_return_refund_profiles_remediation import run, transition


def test_workflow_preconditions_reach_actual_provider_payload():
    seen = []

    class Agent(ReferenceAgent):
        def decide(self, observation):
            public = compliance_contract.provider_public(observation)
            seen.append(public)
            return super().decide(AgentObservation(**public))

    episode = run(agent=Agent())
    assert evaluate_episode(episode, load_spec(FIXTURE), "UK_NORMAL")["reliable"]
    for public in seen:
        policy = public["policy"]
        assert policy["public_contract_version"] == 2
        workflow = policy["workflow"]
        for rule in (
            "decision=review before request_review",
            "status=unknown",
            "status=pending",
            "WAIT",
            "finality_ready=true before notify_customer",
            "ACT and RETRIEVE do not advance time",
            "COMPLETE also cites case_1",
        ):
            assert rule in workflow
        assert "proof_1" in workflow and "approval_1" in workflow


@pytest.mark.parametrize(
    "mode,code",
    [
        ("duplicate", "DUPLICATE_ENTITLEMENT"),
        ("mismatch", "ENTITLEMENT_MISMATCH"),
        ("premature", "NOTICE_NOT_READY"),
        ("finality", "FINALITY_NOT_READY"),
        ("notified", "ALREADY_NOTIFIED"),
        ("wrong", "FALSE_NOTICE"),
        ("ready", "ACCEPTED"),
    ],
)
def test_refusal_causes_do_not_claim_applied_harm(mode, code):
    domain = build_domain(load_spec(FIXTURE), "UK_NORMAL")
    ctx = Context()
    if mode in ("duplicate", "mismatch"):
        state = domain.initial_state()
        if mode == "duplicate":
            state["case"]["decision"] = "refund"
        action = "decide_entitlement"
        fields = {"decision": "refund" if mode == "duplicate" else "review"}
    else:
        tr = transition(run(), "notify_customer")
        state = deepcopy(tr["before"])
        ctx.now = tr["at"]
        action = "notify_customer"
        fields = {"message": "reviewed" if mode == "wrong" else "refunded"}
        if mode == "premature":
            state["payment"]["status"] = "pending"
            state["case"]["finality_ready"] = False
        elif mode == "finality":
            state["case"]["finality_ready"] = False
        elif mode == "notified":
            state["case"]["notified"] = "refunded"
    payload = {
        "case_id": "case_1",
        "version": state["case"]["version"],
        "profile_digest": domain.identity["profile_digest"],
        **fields,
    }
    before = deepcopy(state)
    verdict = domain.apply_action(state, action, payload, ("case_1",), ctx)
    assert verdict.code == code
    assert verdict.accepted is (mode == "ready")
    if not verdict.accepted:
        assert state == before
    if mode == "finality":
        assert "WAIT" in verdict.detail


@pytest.mark.parametrize(
    "decision,requested,code",
    [
        (None, False, "REVIEW_DECISION_REQUIRED"),
        ("review", True, "REVIEW_ALREADY_REQUESTED"),
        ("review", False, "ACCEPTED"),
        ("refund", False, "UNJUSTIFIED_REVIEW"),
    ],
)
def test_review_order_and_repeat_diagnostics(decision, requested, code):
    domain = build_domain(load_spec(FIXTURE), "UK_MISSING_DISCLOSURE")
    state = domain.initial_state()
    state["case"].update(decision=decision, review_requested=requested)
    payload = {
        "case_id": "case_1",
        "profile_digest": domain.identity["profile_digest"],
        "version": state["case"]["version"],
        "reason": "scope_or_rights",
    }
    before = deepcopy(state)
    verdict = domain.apply_action(
        state, "request_review", payload, ("case_1",), Context()
    )
    assert verdict.code == code
    if not verdict.accepted:
        assert state == before
    if decision is None:
        assert "decide_entitlement" in verdict.detail and "review" in verdict.detail


def test_duplicate_refund_feedback_does_not_instruct_pending_query():
    domain = build_domain(load_spec(FIXTURE), "UK_NORMAL")
    tr = transition(run(), "submit_refund")
    state = deepcopy(tr["after"])
    payload = {
        "case_id": "case_1",
        "profile_digest": domain.identity["profile_digest"],
        "version": state["case"]["version"],
        "intent_id": state["payment"]["intent_id"],
        **{key: state["case"][key] for key in ("amount_minor", "currency", "tender_ref")},
    }
    before = deepcopy(state)
    verdict = domain.apply_action(state, "submit_refund", payload, ("case_1",), Context())
    assert verdict.code == "DUPLICATE_REFUND"
    assert "status=unknown" in verdict.detail
    assert "pending" in verdict.detail and "WAIT" in verdict.detail
    assert state == before


def test_duplicate_entitlement_then_recovery_has_same_effects_but_is_unsafe():
    class Repeat(ReferenceAgent):
        attempted = False

        def decide(self, observation):
            public = compliance_contract.provider_public(observation)
            result = super().decide(AgentObservation(**public))
            if (
                isinstance(result, Act)
                and result.action_type == "authorize_return"
                and not self.attempted
            ):
                self.attempted = True
                return Act(
                    "decide_entitlement",
                    dict(result.payload) | {"decision": "refund"},
                    evidence_refs=result.evidence_refs,
                    rationale="deliberate duplicate then recovery",
                )
            return result

    reference = run()
    recovered = run(agent=Repeat())
    grade = evaluate_episode(recovered, load_spec(FIXTURE), "UK_NORMAL")
    assert not grade["reliable"] and not grade["dimensions"]["safe_actions"]
    assert recovered.final_state == reference.final_state
    assert recovered.terminal_outcome == "refunded"
    assert "DUPLICATE_ENTITLEMENT" in {f["code"] for f in grade["findings"]}

    def effects(ep):
        return [t for t in ep.trajectory if t["record_type"] == "commerce_transition"]

    assert [t["kind"] for t in effects(recovered)] == [
        t["kind"] for t in effects(reference)
    ]
