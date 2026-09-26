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


def test_unsolicited_interrupt_without_provenance_fails():
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
    assert "WAIT_INTERRUPT_PROVENANCE_INVALID" in {f.code for f in result.findings}
    assert result.counts["unsolicited_interrupts"] == 0
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


@pytest.fixture(scope="module")
def runtime_wait_interrupt():
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

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RetrievingReferenceAgent, "decide", selected)
        interrupted = run_episode(spec, "V1", "reference")
    return baseline, interrupted


def test_selected_wait_interrupt_keeps_business_effects(runtime_wait_interrupt):
    baseline, interrupted = runtime_wait_interrupt
    from operatebench.artifact import build_artifact, validate_artifact

    assert interrupted.reliable
    validate_artifact(build_artifact(interrupted))
    rows, events = interrupted.outcome.trajectory, interrupted.outcome.events
    result = _temporal(rows, events, _record_integrity(rows, events))
    assert result.ok
    assert result.counts["unsolicited_interrupts"] == sum(
        row.get("code") == "UNDECLARED_WAKE_EVENT" for row in rows
    )
    assert result.counts["waits_declared"] == sum(
        row["record_type"] == "wait_declared" and "wake_on" in row for row in rows
    )
    assert result.counts["reminders_fired"] == sum(
        event["event_type"] == "approval_reminder_due"
        and event["disposition"] == "accepted"
        for event in events
    )
    assert interrupted.outcome.final_state == baseline.outcome.final_state
    assert any(
        row.get("code") == "UNDECLARED_WAKE_EVENT"
        for row in interrupted.outcome.trajectory
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_id",
        "unknown_id",
        "malformed_id",
        "unrelated_id",
        "no_delivery",
        "duplicate_delivery",
        "duplicate_record",
        "no_observed",
        "audit_record",
        "rejected",
        "audit",
        "post_terminal",
        "no_trigger",
        "malformed_trigger",
        "wrong_actor",
        "wrong_type",
        "wrong_time",
        "wrong_verdict",
        "diagnostic_time",
        "cancelled",
        "no_wait",
        "in_wake_on",
        "empty_wake",
        "malformed_wake",
        "fallback_fired",
        "expired_fallback",
        "malformed_fallback",
        "old_wait",
        "intervening_delivery",
        "duplicate_diagnostic",
        "missing_trigger",
        "missing_type",
        "missing_actor",
        "missing_time",
        "malformed_delivery",
        "missing_verdict",
        "replacement_wait",
    ],
)
def test_runtime_interrupt_requires_current_delivery_provenance(
    runtime_wait_interrupt, mutation
):
    _, run = runtime_wait_interrupt
    rows = deepcopy(list(run.outcome.trajectory))
    events = deepcopy(list(run.outcome.events))
    position = next(
        i for i, row in enumerate(rows) if row.get("code") == "UNDECLARED_WAKE_EVENT"
    )
    # Keep the real prefix through the first interrupt, isolating its count.
    rows = rows[: position + 1]
    recorded_ids = {
        row.get("event_id") for row in rows if row["record_type"].startswith("event_")
    }
    events = [event for event in events if event["event_id"] in recorded_ids]
    diagnostic, observed = rows[-1], rows[-2]
    event = next(event for event in events if event["event_id"] == diagnostic["event_id"])
    wait = next(
        row
        for row in reversed(rows[:-2])
        if row["record_type"] == "wait_declared" and "wake_on" in row
    )
    valid = _temporal(rows, events, _record_integrity(rows, events))
    assert valid.ok
    assert valid.counts["unsolicited_interrupts"] == 1
    if mutation in {
        "missing_trigger",
        "missing_type",
        "missing_actor",
        "missing_time",
        "missing_verdict",
    }:
        key = {
            "missing_trigger": "triggers_agent",
            "missing_type": "event_type",
            "missing_actor": "actor_id",
            "missing_time": "at",
            "missing_verdict": "verdict_code",
        }[mutation]
        event.pop(key)
    elif mutation == "malformed_delivery":
        events[events.index(event)] = None
    elif mutation == "replacement_wait":
        replacement = deepcopy(wait)
        replacement["wake_on"] = [event["event_type"]]
        replacement["at"] = observed["at"]
        rows.insert(-2, replacement)
    elif mutation == "missing_id":
        diagnostic.pop("event_id")
    elif mutation == "unknown_id":
        diagnostic["event_id"] = "absent"
    elif mutation == "malformed_id":
        diagnostic["event_id"] = []
    elif mutation == "unrelated_id":
        diagnostic["event_id"] = events[0]["event_id"]
    elif mutation == "no_delivery":
        events.remove(event)
    elif mutation == "duplicate_delivery":
        events.append(deepcopy(event))
    elif mutation == "duplicate_record":
        rows.insert(-1, deepcopy(observed))
    elif mutation == "no_observed":
        rows.remove(observed)
    elif mutation == "audit_record":
        observed["record_type"] = "event_audit_only"
    elif mutation in {"rejected", "audit", "post_terminal"}:
        event["disposition"] = mutation
    elif mutation in {"no_trigger", "malformed_trigger"}:
        event["triggers_agent"] = False if mutation == "no_trigger" else 1
    elif mutation.startswith("wrong_"):
        key = {
            "wrong_actor": "actor_id",
            "wrong_type": "event_type",
            "wrong_time": "at",
            "wrong_verdict": "verdict_code",
        }[mutation]
        event[key] = "2030-01-01T00:00:00Z" if key == "at" else "other"
    elif mutation == "diagnostic_time":
        diagnostic["at"] = "2030-01-01T00:00:00Z"
    elif mutation == "cancelled":
        rows.insert(
            -2,
            {
                "record_type": "timer_cancelled",
                "at": observed["at"],
                "event_id": event["event_id"],
            },
        )
    elif mutation == "no_wait":
        rows.remove(wait)
    elif mutation in {"in_wake_on", "empty_wake", "malformed_wake"}:
        wait["wake_on"] = {
            "in_wake_on": [event["event_type"]],
            "empty_wake": [],
            "malformed_wake": 7,
        }[mutation]
    elif mutation == "fallback_fired":
        rows.insert(
            -2,
            {
                "record_type": "wait_declared",
                "at": observed["at"],
                "fallback_fired": True,
            },
        )
    elif mutation in {"expired_fallback", "malformed_fallback"}:
        wait["fallback_at"] = (
            "2030-01-01T00:00:00Z" if mutation == "expired_fallback" else []
        )
    elif mutation == "old_wait":
        rows.insert(-2, {"record_type": "agent_invoked", "at": observed["at"]})
    elif mutation == "intervening_delivery":
        rows.insert(-2, deepcopy(observed))
        rows[-3]["event_id"] = "another_delivery"
    elif mutation == "duplicate_diagnostic":
        rows.append(deepcopy(diagnostic))
    else:
        raise AssertionError(mutation)
    for index, row in enumerate(rows):
        row["index"] = index
    before = deepcopy((rows, events))
    result = _temporal(rows, events, _record_integrity(rows, events))
    assert "WAIT_INTERRUPT_PROVENANCE_INVALID" in {f.code for f in result.findings}
    assert not result.ok
    assert result.counts["unsolicited_interrupts"] == (
        1 if mutation == "duplicate_diagnostic" else 0
    )
    assert (rows, events) == before


@pytest.mark.parametrize("intervening", ["event_audit_only", "event_rejected"])
def test_nontrigger_delivery_preserves_standing_wait(runtime_wait_interrupt, intervening):
    _, run = runtime_wait_interrupt
    rows = deepcopy(list(run.outcome.trajectory))
    events = deepcopy(list(run.outcome.events))
    position = next(
        i for i, row in enumerate(rows) if row.get("code") == "UNDECLARED_WAKE_EVENT"
    )
    observed = deepcopy(rows[position - 1])
    event = deepcopy(
        next(event for event in events if event["event_id"] == observed["event_id"])
    )
    observed.update(record_type=intervening, event_id="nontrigger")
    event.update(
        event_id="nontrigger",
        triggers_agent=False,
        disposition="audit" if intervening == "event_audit_only" else "rejected",
    )
    rows.insert(position - 1, observed)
    events.append(event)
    # Equal-instant event delivery wins over a fallback (the engine uses <).
    wait = next(
        row
        for row in reversed(rows[: position - 1])
        if row["record_type"] == "wait_declared" and "wake_on" in row
    )
    wait["fallback_at"] = observed["at"]
    for index, row in enumerate(rows):
        row["index"] = index
    result = _temporal(rows, events, _record_integrity(rows, events))
    assert result.ok
    assert result.counts["unsolicited_interrupts"] == sum(
        row.get("code") == "UNDECLARED_WAKE_EVENT" for row in rows
    )


def test_interrupt_provenance_does_not_swallow_internal_bugs(
    runtime_wait_interrupt, monkeypatch
):
    from operatebench.domains.lettings.maintenance import evaluator

    _, run = runtime_wait_interrupt
    rows, events = run.outcome.trajectory, run.outcome.events
    integrity = _record_integrity(rows, events)

    def broken_clock(*args):
        raise RuntimeError("internal bug")

    monkeypatch.setattr(evaluator, "parse_timestamp", broken_clock)
    with pytest.raises(RuntimeError, match="internal bug"):
        _temporal(rows, events, integrity)


def test_actual_interrupt_at_invocation_limit_is_still_informational(monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    from operatebench.core.outcomes import Wait
    from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
    from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode

    original_plan = MaintenanceOperation.build_plan
    original_decide = RetrievingReferenceAgent.decide

    def limited_plan(self):
        return replace(original_plan(self), max_invocations=1)

    def selected(self, observation):
        outcome = original_decide(self, observation)
        return (
            replace(outcome, wake_on=("customer_issue_reported",))
            if isinstance(outcome, Wait)
            else outcome
        )

    monkeypatch.setattr(MaintenanceOperation, "build_plan", limited_plan)
    monkeypatch.setattr(RetrievingReferenceAgent, "decide", selected)
    run = run_episode(
        load_spec(Path("examples/operatebench/maintenance_v0_1.yaml")),
        "V1",
        "reference",
        self_check=False,
    )
    rows, events = run.outcome.trajectory, run.outcome.events
    result = _temporal(rows, events, _record_integrity(rows, events))
    assert result.counts["unsolicited_interrupts"] == 1
    assert "WAIT_INTERRUPT_PROVENANCE_INVALID" not in {f.code for f in result.findings}
    assert "NEVER_YIELDED_CONTROL" in {f.code for f in result.findings}
    assert sum(row["record_type"] == "agent_invoked" for row in rows) == 1


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
