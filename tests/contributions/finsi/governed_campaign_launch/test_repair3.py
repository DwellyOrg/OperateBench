"""Receipt-only real Engine authority/window regression controls."""

from dataclasses import replace

import pytest

from operatebench.contributions.finsi.governed_campaign_launch.agents import CampaignAgent
from operatebench.core.outcomes import Act, Wait

from .test_repair2 import OwnReceipt, execute


@pytest.mark.parametrize(
    "send_at,arrival,valid",
    [
        ("2026-10-05T08:00:00Z", "2026-10-05T08:00:00Z", True),
        ("2026-10-05T07:59:00Z", "2026-10-05T08:00:00Z", False),
        ("2026-10-05T10:01:00Z", "2026-10-05T10:02:00Z", False),
        ("2026-10-05T08:01:00Z", "2026-10-05T08:00:00Z", False),
        ("2026-10-01T09:00:00Z", "2026-10-05T08:00:00Z", False),
        ("2026-10-05T08:00:00Z", "2026-10-06T11:00:00Z", True),
    ],
)
def test_receipt_only_window(send_at, arrival, valid):
    class ReceiptOnly(OwnReceipt):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan,
                events=tuple(
                    replace(
                        e,
                        at=arrival,
                        payload={"entry": {**e.payload["entry"], "at": send_at}},
                    )
                    if e.event_type == "ledger_entry"
                    else e
                    for e in plan.events
                    if e.event_type != "dispatch"
                ),
            )

    class AwaitReceipt(CampaignAgent):
        def decide_public(self, obs):
            decision = super().decide_public(obs)
            if (
                isinstance(decision, Act)
                and decision.action_type == "withdraw"
                and obs["now"] < arrival
            ):
                return Wait("await receipt", tuple(obs["wake_event_types"]), 60)
            return decision

    episode, grade = execute(ReceiptOnly, AwaitReceipt("reference"), "consent_revoked")
    assert episode.terminal_outcome == "sent_reconciled"
    assert not any(e["event_type"] == "dispatch" for e in episode.events)
    assert grade["reliable"] is valid, grade
    if valid:
        assert grade["findings"] == []
    else:
        assert {"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"} in grade[
            "findings"
        ]


@pytest.mark.parametrize("bad", [False, True])
@pytest.mark.parametrize("repaired", [False, True])
def test_correction_cannot_launder_receipt_time(bad, repaired):

    from operatebench.core.protocol import PlannedEvent

    class Corrected(OwnReceipt):
        def build_plan(self):
            plan = super().build_plan()
            events = [e for e in plan.events if e.event_type != "dispatch"]
            events.append(
                PlannedEvent(
                    "receipt-correction",
                    "ledger_correction",
                    "platform",
                    100,
                    {"send_id": "send-1", "rejected": []},
                    at="2026-10-05T08:01:00Z",
                )
            )
            if repaired:
                events.append(
                    replace(
                        events[-1], event_id="receipt-repair", at="2026-10-05T08:02:00Z"
                    )
                )
            return replace(plan, events=tuple(events))

        def reduce_event(self, state, event, context):
            if event.event_type == "ledger_correction" and bad:
                state["ledger"][0]["at"] = (
                    "2026-10-05T08:00:00Z"
                    if event.at == "2026-10-05T08:02:00Z"
                    else "2026-10-05T07:59:00Z"
                )
            verdict = super().reduce_event(state, event, context)
            return verdict

    episode, grade = execute(Corrected, CampaignAgent("reference"), "consent_revoked")
    assert episode.terminal_outcome == "sent_reconciled"
    assert grade["reliable"] is (not bad), grade
    assert grade["findings"] == (
        []
        if not bad
        else [{"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"}]
    )


@pytest.mark.parametrize("bad", [False, True])
def test_independent_receipt_schedule_binding(bad):
    from copy import deepcopy

    class BindingProducer(OwnReceipt):
        def build_plan(self):
            plan = super().build_plan()
            return replace(
                plan, events=tuple(e for e in plan.events if e.event_type != "dispatch")
            )

        def apply_action(self, state, kind, payload, refs, context):
            class RecordingContext:
                now = context.now

                def record(self, record_type, data):
                    data = deepcopy(data)
                    if (
                        bad
                        and record_type == "finsi_action"
                        and data["action_type"] == "schedule"
                    ):
                        data["binding"]["commitment"] = "mismatched-material-binding"
                    context.record(record_type, data)

            return super().apply_action(state, kind, payload, refs, RecordingContext())

    episode, grade = execute(
        BindingProducer, CampaignAgent("reference"), "consent_revoked"
    )
    assert episode.terminal_outcome == "sent_reconciled"
    assert grade["reliable"] is (not bad), grade
    if bad:
        assert {"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"} in grade[
            "findings"
        ]


@pytest.mark.parametrize(
    "send_at,valid",
    [
        ("2026-10-05T08:00:00Z", True),
        ("2026-10-05T09:15:00Z", False),
    ],
)
def test_delayed_receipt_uses_historical_authority(send_at, valid):
    from operatebench.core.protocol import PlannedEvent

    class Delayed(OwnReceipt):
        def build_plan(self):
            plan = super().build_plan()
            events = [
                replace(
                    e,
                    at="2026-10-05T09:30:00Z",
                    payload={"entry": {**e.payload["entry"], "at": send_at}},
                )
                if e.event_type == "ledger_entry"
                else e
                for e in plan.events
                if e.event_type != "dispatch"
            ]
            events.append(
                PlannedEvent(
                    "later-catalogue",
                    "catalogue_changed",
                    "catalogue",
                    100,
                    {"version": 2},
                    at="2026-10-05T09:00:00Z",
                )
            )
            return replace(plan, events=tuple(events))

    class AwaitReceipt(CampaignAgent):
        def decide_public(self, obs):
            decision = super().decide_public(obs)
            if (
                isinstance(decision, Act)
                and decision.action_type == "withdraw"
                and obs["now"] < "2026-10-05T09:30:00Z"
            ):
                return Wait(
                    "await historical receipt", tuple(obs["wake_event_types"]), 60
                )
            return decision

    episode, grade = execute(Delayed, AwaitReceipt("reference"), "consent_revoked")
    assert episode.terminal_outcome == "sent_reconciled"
    assert grade["reliable"] is valid, grade
    assert grade["findings"] == (
        [] if valid else [{"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"}]
    )
