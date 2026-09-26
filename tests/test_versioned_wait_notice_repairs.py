# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Independent causal controls for engine 0.13, including adversarial ledgers."""

from copy import deepcopy

import pytest

from operatebench.domains.lettings.maintenance.evaluator import (
    _record_integrity,
    _retrieval_discipline,
    _temporal,
)

REPEATED = "REPEATED_ACTION_AFTER_WAKE"


def repeats(rows, events=()):
    return [
        f
        for f in _retrieval_discipline(rows, events=events).findings
        if f.code == REPEATED
    ]


@pytest.fixture
def notice_trace():
    rows = []
    events = []

    def row(record_type, **kw):
        rows.append(
            dict(
                index=len(rows), at="2031-01-01T00:00:00Z", record_type=record_type, **kw
            )
        )

    def event(name, kind, actor, payload):
        row(
            "event_observed",
            event_id=name,
            event_type=kind,
            actor_id=actor,
            code="ACCEPTED",
        )
        events.append(
            {
                "event_id": name,
                "event_type": kind,
                "actor_id": actor,
                "at": rows[-1]["at"],
                "payload": payload,
                "disposition": "accepted",
                "verdict_code": "ACCEPTED",
            }
        )

    payload = {
        "recipient_actor_id": "reporter",
        "message_fixture_id": "msg_completion_notice",
        "correlation_id": "repair",
    }
    event("issue", "customer_issue_reported", "reporter", {})
    event(
        "verification",
        "work_evidence_verified",
        "verifier",
        {"cycle_id": "repair", "visit_id": "visit"},
    )
    row("agent_invoked", invocation_index=1)
    row(
        "action_proposed",
        proposal_id="early",
        action_type="send_message",
        payload=payload.copy(),
    )
    row(
        "side_effect",
        channel="message_dispatch",
        status="DELIVERED",
        **{k: v for k, v in payload.items() if k != "correlation_id"},
    )
    row(
        "effect_accepted",
        proposal_id="early",
        action_type="send_message",
        code="ACCEPTED",
        cycle_id="repair",
    )
    row(
        "action_proposed",
        proposal_id="pay",
        action_type="request_payment",
        payload={"invoice_id": "invoice", "amount_minor": 42, "currency": "GBP"},
    )
    row(
        "effect_accepted",
        proposal_id="pay",
        action_type="request_payment",
        code="ACCEPTED",
        cycle_id="repair",
        bindings={"payment_request_id": "payment"},
    )
    row(
        "obligation_created",
        obligation_id="completion_notice:repair",
        kind="customer_completion_notice",
    )
    event(
        "settlement",
        "payment_settlement_confirmed",
        "bank",
        {
            "cycle_id": "repair",
            "payment_request_id": "payment",
            "invoice_id": "invoice",
            "amount_minor": 42,
            "currency": "GBP",
        },
    )
    row("agent_invoked", invocation_index=2)
    row(
        "action_proposed",
        proposal_id="required",
        action_type="send_message",
        payload=payload.copy(),
    )
    row(
        "obligation_discharged",
        obligation_id="completion_notice:repair",
        kind="customer_completion_notice",
    )
    row(
        "side_effect",
        channel="message_dispatch",
        status="DELIVERED",
        **{k: v for k, v in payload.items() if k != "correlation_id"},
    )
    row(
        "effect_accepted",
        proposal_id="required",
        action_type="send_message",
        code="ACCEPTED",
        cycle_id="repair",
    )
    return rows, events


def test_new_completion_duty_is_not_repetition(notice_trace):
    assert not repeats(*notice_trace)


@pytest.mark.parametrize("number", [3, 4])
def test_later_wasted_notices_stay_repetitions(notice_trace, number):
    rows, events = notice_trace
    for n in range(3, number + 1):
        rows.append({"record_type": "agent_invoked", "invocation_index": n})
        p = deepcopy(rows[11])
        p["proposal_id"] = f"waste{n}"
        rows.append(p)
        e = deepcopy(rows[14])
        e["proposal_id"] = p["proposal_id"]
        rows.append(e)
    assert len(repeats(rows, events)) == number - 2


@pytest.mark.parametrize(
    "mutation",
    [
        "supplier",
        "second_customer",
        "wrong_cycle",
        "refs_only",
        "no_creation",
        "forged_kind",
        "forged_id",
        "creation_before_early",
        "already_closed",
        "no_event",
        "future_event",
        "wrong_event_cycle",
        "wrong_money",
        "no_payment",
        "no_verification",
        "future_verification",
        "wrong_acceptance",
        "wrong_time",
        "no_discharge",
        "wrong_discharge",
        "no_delivery",
        "future_acceptance",
        "future_discharge",
        "old_wrong_acceptance",
        "old_wrong_time",
        "duplicate_creation",
    ],
)
def test_causal_tampering_cannot_excuse_a_repeat(notice_trace, mutation):
    rows, events = notice_trace
    if mutation in {"supplier", "second_customer"}:
        for r in rows:
            if r.get("action_type") == "send_message" and "payload" in r:
                r["payload"]["recipient_actor_id"] = mutation
            if r.get("record_type") == "side_effect":
                r["recipient_actor_id"] = mutation
    elif mutation == "wrong_cycle":
        for r in (rows[3], rows[11]):
            r["payload"]["correlation_id"] = "other"
        for r in (rows[5], rows[14]):
            r["cycle_id"] = "other"
    elif mutation == "refs_only":
        rows[11]["evidence_refs"] = ["fresh-document"]
        rows.pop(8)
    elif mutation == "no_creation":
        rows.pop(8)
    elif mutation == "forged_kind":
        rows[8]["kind"] = "arbitrary_duty"
    elif mutation == "forged_id":
        rows[8]["obligation_id"] = "arbitrary:repair"
    elif mutation == "creation_before_early":
        rows.insert(2, rows.pop(8))
    elif mutation == "already_closed":
        rows.insert(10, deepcopy(rows[12]))
    elif mutation == "no_event":
        events.pop()
    elif mutation == "future_event":
        rows.append(rows.pop(9))
    elif mutation == "wrong_event_cycle":
        events[-1]["payload"]["cycle_id"] = "other"
    elif mutation == "wrong_money":
        events[-1]["payload"]["amount_minor"] = 43
    elif mutation == "no_payment":
        rows.pop(7)
    elif mutation == "no_verification":
        events.pop(1)
    elif mutation == "future_verification":
        rows.append(rows.pop(1))
    elif mutation == "wrong_acceptance":
        rows[-1]["proposal_id"] = "unrelated"
    elif mutation == "wrong_time":
        rows[-1]["at"] = "2031-01-02T00:00:00Z"
    elif mutation == "no_discharge":
        rows.pop(12)
    elif mutation == "wrong_discharge":
        rows[12]["obligation_id"] = "other"
    elif mutation == "no_delivery":
        rows.pop(13)
    elif mutation == "future_acceptance":
        rows.insert(14, {"record_type": "agent_invoked", "invocation_index": 3})
    elif mutation == "future_discharge":
        rows.append(rows.pop(12))
    elif mutation == "old_wrong_acceptance":
        rows[5]["cycle_id"] = "other"
    elif mutation == "old_wrong_time":
        rows[5]["at"] = "2030-01-01T00:00:00Z"
    elif mutation == "duplicate_creation":
        rows.insert(8, deepcopy(rows[8]))
    assert len(repeats(rows, events)) == 1


@pytest.mark.parametrize(
    "action", ["request_payment", "request_supplier_visit", "authorise_supplier_work"]
)
def test_other_effects_do_not_gain_a_discharge_exception(notice_trace, action):
    rows, events = notice_trace
    for r in rows:
        if r.get("action_type") == "send_message":
            r["action_type"] = action
    assert len(repeats(rows, events)) == 1


@pytest.mark.parametrize("code", ["UNKNOWN_WAKE_EVENT_TYPE", "WAIT_AFTER_REPLAY_FINAL"])
def test_actual_invalid_waits_stay_failures(code):
    rows = [
        {
            "index": 0,
            "at": "2031-01-01T00:00:00Z",
            "record_type": "wait_declared",
            "wake_on": ["known"],
        },
        {
            "index": 1,
            "at": "2031-01-01T00:00:00Z",
            "record_type": "wait_rejected",
            "code": code,
            "detail": "invalid",
        },
    ]
    assert code in {
        f.code for f in _temporal(rows, [], _record_integrity(rows, [])).findings
    }


def test_unsolicited_interrupt_is_informational():
    rows = [
        {
            "index": 0,
            "at": "2031-01-01T00:00:00Z",
            "record_type": "wait_declared",
            "wake_on": ["known"],
        },
        {
            "index": 1,
            "at": "2031-01-01T00:00:00Z",
            "record_type": "wait_rejected",
            "code": "UNDECLARED_WAKE_EVENT",
            "detail": "interrupt",
        },
    ]
    before = deepcopy(rows)
    result = _temporal(rows, [], _record_integrity(rows, []))
    assert "UNDECLARED_WAKE_EVENT" not in {f.code for f in result.findings}
    assert rows == before


@pytest.mark.parametrize("payable", [False, True])
def test_verification_creation_requires_a_nonpayable_visit(notice_trace, payable):
    rows, events = notice_trace
    # A permitted early notice can precede verification on a nonpayable visit.
    # A payable cycle cannot forge a duty at verification instead of settlement.
    rows[6]["action_type"] = "request_supplier_visit"
    rows[6]["payload"] = {
        "cycle_id": "repair",
        "visit_type": "APPROVED_WORK" if payable else "DIAGNOSTIC",
    }
    rows[7]["action_type"] = "request_supplier_visit"
    rows[7]["bindings"] = {"visit_id": "visit"}
    rows[9].update(
        event_type="work_evidence_verified", event_id="verification", actor_id="verifier"
    )
    events.pop()
    rows.pop(1)
    assert len(repeats(rows, events)) == int(payable)


def test_audit_delivery_can_create_a_duty_before_a_later_wake(notice_trace):
    rows, events = notice_trace
    rows[9]["record_type"] = "event_audit_only"
    events[-1]["disposition"] = "audit"
    assert not repeats(rows, events)


def test_reused_proposal_identity_cannot_establish_new_effect(notice_trace):
    rows, events = notice_trace
    rows[11]["proposal_id"] = rows[14]["proposal_id"] = "early"
    assert len(repeats(rows, events)) == 1


def test_malformed_identical_payload_is_not_an_exception(notice_trace):
    rows, events = notice_trace
    rows[3]["payload"] = rows[11]["payload"] = ["not-a-message"]
    assert len(repeats(rows, events)) == 1


@pytest.mark.parametrize("kind", ["unrelated", "verification_lookalike"])
def test_shuffled_same_time_event_cannot_supply_the_cause(notice_trace, kind):
    """Same-instant reordering must not let a different event become the cause."""
    rows, events = notice_trace
    at = rows[8]["at"]
    eid = "shuffled"
    etype = "landlord_note_added" if kind == "unrelated" else "work_evidence_verified"
    rows.insert(
        9,
        {
            "index": 9,
            "at": at,
            "record_type": "event_observed",
            "event_id": eid,
            "event_type": etype,
            "actor_id": "verifier",
            "code": "ACCEPTED",
        },
    )
    events.append(
        {
            "event_id": eid,
            "event_type": etype,
            "actor_id": "verifier",
            "at": at,
            "payload": {"cycle_id": "repair", "visit_id": "visit"},
            "disposition": "accepted",
            "verdict_code": "ACCEPTED",
        }
    )
    assert len(repeats(rows, events)) == 1


def test_duplicate_event_id_in_the_tape_cannot_bind_the_cause(notice_trace):
    """Two tape entries sharing the causal event_id must be ambiguous, not accepted."""
    rows, events = notice_trace
    events.append(deepcopy(events[-1]))
    assert len(repeats(rows, events)) == 1


def test_two_reporting_actors_cannot_authorise_the_recipient(notice_trace):
    """An ambiguous reporting actor must not authorise the notice recipient."""
    rows, events = notice_trace
    at = rows[0]["at"]
    rows.insert(
        1,
        {
            "index": 1,
            "at": at,
            "record_type": "event_observed",
            "event_id": "issue2",
            "event_type": "customer_issue_reported",
            "actor_id": "reporter2",
            "code": "ACCEPTED",
        },
    )
    events.append(
        {
            "event_id": "issue2",
            "event_type": "customer_issue_reported",
            "actor_id": "reporter2",
            "at": at,
            "payload": {},
            "disposition": "accepted",
            "verdict_code": "ACCEPTED",
        }
    )
    assert len(repeats(rows, events)) == 1


def test_selected_wait_interrupt_keeps_business_effects(monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    from operatebench.core.outcomes import Wait
    from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode

    spec = load_spec(Path("examples/operatebench/maintenance_v0_1.yaml"))
    baseline = run_episode(spec, "V1", "reference")
    original = RetrievingReferenceAgent.decide

    def selected(self, observation):
        outcome = original(self, observation)
        if isinstance(outcome, Wait):
            return replace(outcome, wake_on=("customer_issue_reported",))
        return outcome

    monkeypatch.setattr(RetrievingReferenceAgent, "decide", selected)
    interrupted = run_episode(spec, "V1", "reference")
    assert interrupted.reliable
    assert interrupted.outcome.final_state == baseline.outcome.final_state
    assert any(
        row.get("code") == "UNDECLARED_WAKE_EVENT"
        for row in interrupted.outcome.trajectory
    )


def test_payment_settlement_requires_a_second_completion_notice(monkeypatch):
    from pathlib import Path

    from operatebench.core.outcomes import Act
    from operatebench.domains.lettings.maintenance.agents import (
        CUSTOMER,
        MSG_COMPLETION_NOTICE,
        ReferenceAgent,
        _message,
    )
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode

    original_decide = ReferenceAgent.decide_from
    original_close = ReferenceAgent._close_out

    def decide(self, state, policy):
        outcome = original_decide(self, state, policy)
        if (
            isinstance(outcome, Act)
            and outcome.action_type == "request_payment"
            and not getattr(self, "_early_notice_sent", False)
        ):
            # Instance-local state gives the deterministic rerun its own early notice.
            self._early_notice_sent = True
            return _message(CUSTOMER, MSG_COMPLETION_NOTICE, state["current_cycle_id"])
        return outcome

    def close_out(self, state, cycle, cycle_id):
        obligation = state["obligations"].get(f"completion_notice:{cycle_id}")
        if obligation and obligation["status"] == "OPEN":
            return _message(CUSTOMER, MSG_COMPLETION_NOTICE, cycle_id)
        return original_close(self, state, cycle, cycle_id)

    monkeypatch.setattr(ReferenceAgent, "decide_from", decide)
    monkeypatch.setattr(ReferenceAgent, "_close_out", close_out)
    spec = load_spec(
        Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
    )
    run = run_episode(spec, "V1", "reference")
    assert run.reliable
    assert next(
        d for d in run.evaluation.dimensions if d.name == "deterministic_replay"
    ).ok
    rows, events = run.outcome.trajectory, run.outcome.events
    notices = [
        row
        for row in rows
        if row["record_type"] == "action_proposed"
        and row.get("payload", {}).get("message_fixture_id") == MSG_COMPLETION_NOTICE
    ]
    early = notices[0]
    early, required = [
        row
        for row in notices
        if row["payload"]["correlation_id"] == early["payload"]["correlation_id"]
    ]
    assert early["payload"] == required["payload"]
    obligation_id = f"completion_notice:{required['payload']['correlation_id']}"
    created, discharged = [
        row for row in rows if row.get("obligation_id") == obligation_id
    ]
    assert created["record_type"] == "obligation_created"
    assert created["kind"] == "customer_completion_notice"
    assert discharged["record_type"] == "obligation_discharged"
    early_accepted, required_accepted = [
        row
        for row in rows
        if row["record_type"] == "effect_accepted"
        and row.get("proposal_id") in {early["proposal_id"], required["proposal_id"]}
    ]
    assert rows.index(early_accepted) < rows.index(created) < rows.index(required)
    assert rows.index(required) < rows.index(discharged) < rows.index(required_accepted)
    cause = rows[rows.index(created) + 1]
    assert cause["event_type"] == "payment_settlement_confirmed"
    assert not repeats(rows, events)
    # The real event tape must establish the cause, not just the obligation rows.
    unmatched = repeats(
        rows, [event for event in events if event["event_id"] != cause["event_id"]]
    )
    assert len(unmatched) == 1
    assert unmatched[0].at == required["at"]
