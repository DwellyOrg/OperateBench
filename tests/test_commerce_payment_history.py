"""Independent payment history checks with successful permissive producers."""

from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.core.engine import has_canonical_episode_outcome_provenance
from operatebench.domains.commerce.return_refund_profiles.operation import (
    CommerceProfilesDomain,
)
from tests.test_commerce_return_refund_profiles_remediation import (
    FIXTURE,
    evaluate_episode,
    load_spec,
    run,
)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        ("status", "failed"),
        ("amount_minor", 1),
        ("intent_id", "wrong"),
        ("currency", "USD"),
        ("tender_ref", "wrong"),
        ("webhook_id", []),
        ("webhook_id", {}),
        ("webhook_id", None),
        ("delay_minutes", 11),
    ],
)
def test_query_history_not_permissive_settlement(mutation):
    class Domain(CommerceProfilesDomain):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan,
                events=tuple(
                    (
                        replace(e, delay_minutes=mutation[1])
                        if mutation[0] == "delay_minutes"
                        else replace(e, payload=dict(e.payload) | dict([mutation]))
                    )
                    if mutation and e.event_id == "query_result_1"
                    else e
                    for e in plan.events
                ),
            )

        def reduce_event(self, state, event, context):
            if event.event_id != "query_result_1":
                return super().reduce_event(state, event, context)
            before = deepcopy(state)
            pay = state["payment"]
            pay.update(
                status="settled",
                settled_minor=pay["reserved_minor"],
                reserved_minor=0,
                settled_at=context.now,
            )
            pay["webhooks"].append("webhook_2")
            context.schedule_timer(
                "finality_1",
                "finality_due",
                "environment",
                5,
                {"case_id": "case_1", "profile_digest": self.identity["profile_digest"]},
            )
            return self._commit(state, context, event.event_type, before)

    spec = load_spec(FIXTURE)
    episode = run("UK_LABEL_UNKNOWN", domain=Domain(spec, "UK_LABEL_UNKNOWN"))
    grade = evaluate_episode(episode, spec, "UK_LABEL_UNKNOWN")
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "refunded"
    assert all(
        e["disposition"] == "accepted"
        for e in episode.events
        if e["event_type"] == "payment_update"
    )
    assert grade["reliable"] is (mutation is None)
    failed = {key for key, ok in grade["dimensions"].items() if not ok}
    expected = {"settlement", "deduplication", "clock"} if mutation else set()
    if mutation and mutation[0] == "delay_minutes":
        expected = {"deduplication"}
    assert failed == expected
    if "clock" in expected:
        assert {f["code"] for f in grade["findings"] if f["dimension"] == "clock"} == {
            "REFUND_CLOCK_INSUFFICIENT_DATA"
        }


@pytest.mark.parametrize(
    "mode", ["valid", "no_marker", "marker_only", "conflict", "reset", "early"]
)
def test_duplicate_evidence_and_finality_come_from_core(mode):
    from operatebench.core.clock import shift_minutes
    from operatebench.core.protocol import Verdict

    class Domain(CommerceProfilesDomain):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan,
                events=tuple(
                    replace(e, payload=dict(e.payload) | {"status": "failed"})
                    if mode == "conflict" and e.event_id == "duplicate_result_1"
                    else e
                    for e in plan.events
                ),
            )

        def reduce_event(self, state, event, context):
            if event.event_type == "finality_due":
                before = deepcopy(state)
                state["case"]["finality_ready"] = True
                return self._commit(state, context, event.event_type, before)
            if event.event_id == "duplicate_result_1":
                if mode in ("no_marker", "conflict"):
                    return Verdict.ok(code="DUPLICATE_IGNORED", cycle_id="return_cycle_1")
                if mode == "marker_only":
                    context.record(
                        "commerce_duplicate_ignored",
                        {"webhook_id": "webhook_2", "intent_id": "refund_1"},
                    )
                    return Verdict.refused("TEST_REFUSAL", "marker is not acceptance")
                if mode == "reset":
                    before = deepcopy(state)
                    state["payment"]["settled_at"] = context.now
                    return self._commit(state, context, event.event_type, before)

            class Proxy:
                def __getattr__(self, name):
                    return getattr(context, name)

                def schedule_timer(
                    self, timer_id, event_type, actor_id, delay_minutes, payload
                ):
                    if mode == "early" and event_type == "finality_due":
                        delay_minutes = 3
                    return context.schedule_timer(
                        timer_id, event_type, actor_id, delay_minutes, payload
                    )

            return super().reduce_event(state, event, Proxy())

    spec = load_spec(FIXTURE)
    episode = run("UK_LABEL_UNKNOWN", domain=Domain(spec, "UK_LABEL_UNKNOWN"))
    grade = evaluate_episode(episode, spec, "UK_LABEL_UNKNOWN")
    assert has_canonical_episode_outcome_provenance(episode)
    assert episode.terminal_outcome == "refunded"
    failed = {key for key, ok in grade["dimensions"].items() if not ok}
    assert (
        failed
        == {
            "valid": set(),
            "no_marker": set(),
            "marker_only": {"deduplication"},
            "conflict": {"settlement", "deduplication", "clock"},
            "reset": {"settlement", "completion", "clock"},
            "early": {"completion"},
        }[mode]
    )
    if mode in ("conflict", "reset"):
        assert {f["code"] for f in grade["findings"] if f["dimension"] == "clock"} == {
            "REFUND_CLOCK_INSUFFICIENT_DATA"
        }
    if mode in ("valid", "no_marker"):
        query = next(e for e in episode.events if e["event_id"] == "query_result_1")
        assert episode.final_state["payment"]["settled_at"] == query["at"]
        assert shift_minutes(query["at"], 5) == next(
            e["at"] for e in episode.events if e["event_type"] == "finality_due"
        )


@pytest.mark.parametrize(
    "case",
    [
        "UK_NORMAL",
        "UK_LABEL_UNKNOWN",
        "UK_PAYMENT_FAILED",
        "DE_NORMAL",
        "CA_POLICY",
        "AU_MAJOR",
    ],
)
def test_empty_processor_history_matches_initial_payment(case):
    from tests.test_commerce_refund_clock import StopBefore

    spec = load_spec(FIXTURE)
    episode = run(case, agent=StopBefore("submit_refund"))
    assert has_canonical_episode_outcome_provenance(episode)
    assert not any(
        e["event_type"] == "payment_update" and e["disposition"] == "accepted"
        for e in episode.events
    )
    payment = episode.final_state["payment"]
    assert (payment["status"], payment["settled_minor"], payment.get("settled_at")) == (
        "none",
        0,
        None,
    )
    grade = evaluate_episode(episode, spec, case)
    codes = {f["code"] for f in grade["findings"]}
    assert {"REFUND_NOT_SUBMITTED", "NO_CAUSAL_PROCESSOR_RESULT"} <= codes
    assert not grade["reliable"]
    assert not grade["dimensions"]["settlement"]
    assert "PROCESSOR_HISTORY_STATE_MISMATCH" not in codes


@pytest.mark.parametrize("accepted_history", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [("status", "failed"), ("settled_minor", 1), ("settled_at", "2026-01-13T12:00:00Z")],
)
def test_payment_state_mismatch_still_fails(accepted_history, mutation):
    from tests.test_commerce_refund_clock import StopBefore

    class Domain(CommerceProfilesDomain):
        def _commit(self, state, context, kind, before):
            if kind == ("payment_update" if accepted_history else "authority_granted"):
                state["payment"][mutation[0]] = mutation[1]
            return super()._commit(state, context, kind, before)

    spec = load_spec(FIXTURE)
    episode = run(
        domain=Domain(spec, "UK_NORMAL"),
        agent=None if accepted_history else StopBefore("submit_refund"),
    )
    assert has_canonical_episode_outcome_provenance(episode)
    assert (
        any(
            e["event_type"] == "payment_update" and e["disposition"] == "accepted"
            for e in episode.events
        )
        is accepted_history
    )
    grade = evaluate_episode(episode, spec, "UK_NORMAL")
    assert "PROCESSOR_HISTORY_STATE_MISMATCH" in {f["code"] for f in grade["findings"]}
    assert not grade["dimensions"]["settlement"]
    assert not grade["reliable"]


def test_query_cannot_be_first_accepted_processor_result():
    from operatebench.core.protocol import Verdict

    class Domain(CommerceProfilesDomain):
        def reduce_event(self, state, event, context):
            if event.event_id == "processor_1":
                # Permit the reference to query while refusing the initial delivery.
                state["payment"]["status"] = "unknown"
                return Verdict.refused("TEST_REFUSAL", "initial delivery refused")
            return super().reduce_event(state, event, context)

    spec = load_spec(FIXTURE)
    episode = run("UK_LABEL_UNKNOWN", domain=Domain(spec, "UK_LABEL_UNKNOWN"))
    accepted = [
        e["event_id"]
        for e in episode.events
        if e["event_type"] == "payment_update" and e["disposition"] == "accepted"
    ]
    assert accepted[0] == "query_result_1"
    grade = evaluate_episode(episode, spec, "UK_LABEL_UNKNOWN")
    assert "INVALID_PROCESSOR_STATUS_HISTORY" in {f["code"] for f in grade["findings"]}
    assert not grade["reliable"]


@pytest.mark.parametrize(
    "case",
    [
        "UK_MISSING_DISCLOSURE",
        "DE_FAULT_REVIEW",
        "CA_NO_POLICY",
        "AU_MINOR",
        "AU_CHANGE_OF_MIND",
    ],
)
def test_non_refund_without_submission_retains_reliable_grade(case):
    episode = run(case)
    assert episode.final_state["payment"]["submission_count"] == 0
    grade = evaluate_episode(episode, load_spec(FIXTURE), case)
    assert grade["reliable"]
    assert grade["dimensions"]["settlement"]
    assert not grade["findings"]
