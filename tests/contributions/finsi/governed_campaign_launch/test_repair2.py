"""Real Core controls for causal receipts, scheduling, approvals and proof selection."""

from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.contributions.finsi.governed_campaign_launch.agents import CampaignAgent
from operatebench.contributions.finsi.governed_campaign_launch.evaluator import (
    evaluate_episode,
)
from operatebench.contributions.finsi.governed_campaign_launch.operation import (
    WINDOWS,
    CampaignDomain,
    document,
    schedule_key,
)
from operatebench.contributions.finsi.governed_campaign_launch.spec import (
    digest,
    load_spec,
)
from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Wait

from .test_pack import FIXTURE, event, prepared


def execute(domain, agent, case="baseline"):
    spec = load_spec(FIXTURE)
    episode = Engine(
        domain(spec, case),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "repair2",
            "agent_id": "reference",
        },
    ).run()
    return episode, evaluate_episode(episode, spec, case)


class OwnReceipt(CampaignDomain):
    bad = False

    def build_plan(self):
        plan = super().build_plan()
        accepted = (
            ["member-0000"] if self.bad else [f"member-{i:04d}" for i in range(1, 2340)]
        )
        entry = {
            "key": digest(
                [
                    "campaign-synthetic",
                    digest(document(1)),
                    "approval-1",
                    list(WINDOWS[0]),
                ]
            ),
            "send_id": "send-1",
            "accepted": accepted,
            "accepted_commitment": None if self.bad else digest(accepted),
            "registry_version": None if self.bad else "2",
            "filtered": [] if self.bad else ["member-0000"],
            "at": WINDOWS[0][0],
        }
        replaced = False
        events = []
        for e in plan.events:
            if e.event_type == "dispatch" and e.at == WINDOWS[0][0] and not replaced:
                e = replace(e, event_type="ledger_entry", payload={"entry": entry})
                replaced = True
            events.append(e)
        return replace(plan, events=tuple(events))


@pytest.mark.parametrize("bad", [False, True])
def test_own_ledger_receipt_cannot_launder_harm(bad):
    class Receipt(OwnReceipt):
        pass

    Receipt.bad = bad
    e, g = execute(Receipt, CampaignAgent("reference"), "consent_revoked")
    assert e.terminal_outcome == ("TRANSFERRED_UNRESOLVED" if bad else "sent_reconciled")
    assert g["reliable"] is (not bad), g
    assert g["findings"] == (
        [{"dimension": "consent", "code": "CONSENT_UNVERIFIED_AT_SEND"}] if bad else []
    )


class Midwindow(CampaignAgent):
    target = "2026-10-05T08:30:00Z"

    def decide_public(self, obs):
        decision = super().decide_public(obs)
        if (
            isinstance(decision, Act)
            and decision.action_type == "schedule"
            and obs["now"] < self.target
        ):
            from datetime import datetime

            minutes = int(
                (
                    datetime.fromisoformat(self.target.replace("Z", "+00:00"))
                    - datetime.fromisoformat(obs["now"].replace("Z", "+00:00"))
                ).total_seconds()
                / 60
            )
            return Wait(
                "schedule within approved window", tuple(obs["wake_event_types"]), minutes
            )
        return decision


@pytest.mark.parametrize(
    "target", ["2026-10-05T08:00:00Z", "2026-10-05T08:30:00Z", "2026-10-05T09:59:00Z"]
)
def test_accepted_within_window_schedule_really_dispatches(target):
    agent = Midwindow("reference")
    agent.target = target
    e, g = execute(CampaignDomain, agent)
    assert e.terminal_outcome == "sent_reconciled" and g["reliable"], g
    assert len(e.final_state["ledger"]) == 1
    assert e.final_state["ledger"][0]["at"] == target


class Reapprove(CampaignAgent):
    def decide_public(self, obs):
        served = obs["retrieval"].get("served", {})
        if "get_campaign" in served:
            s = served["get_campaign"]["records"]["record"]
            if (
                s["withdrawn"]
                and not s["status_owed"]
                and s["window"] == list(WINDOWS[0])
            ):
                return Act("draft", {})
        return super().decide_public(obs)


def test_same_content_new_window_has_new_causal_human_approval():
    e, g = execute(CampaignDomain, Reapprove("reference"), "late_settlement")
    assert g["reliable"] and e.terminal_outcome == "sent_reconciled", g
    assert e.final_state["version"] == 1
    human_responses = [
        e
        for e in e.events
        if e["event_type"] == "approval_record"
        and e["disposition"] in ("accepted", "audit")
    ]
    assert len(human_responses) == 2, "Each window needs its own causal human response"
    requests = list(e.final_state["requests"].values())
    assert len(requests) == 2 and requests[0]["request_id"] != requests[1]["request_id"]
    assert e.final_state["approval"]["window"] == list(WINDOWS[1])
    assert e.final_state["approval"]["id"] == "approval-1-2"


def test_old_response_is_audit_only_cannot_rebind_new_request():
    d, s, c = prepared()
    old = deepcopy(s["request"])
    c.now = WINDOWS[0][1]
    assert d.apply_action(s, "withdraw", {}, (), c).accepted
    assert d.apply_action(s, "status", {}, (), c).accepted
    assert d.apply_action(s, "draft", {}, (), c).accepted
    assert d.apply_action(s, "push", {}, (), c).accepted
    assert event(
        d, s, c, "document_ready", {"document": document(1), "attempt": 2}
    ).accepted
    assert d.apply_action(s, "verify", {}, (), c).accepted
    assert d.apply_action(
        s, "request_approval", {"role": "send_approver"}, (), c
    ).accepted
    before = deepcopy(s)
    payload = {
        "version": 1,
        "role": "send_approver",
        "request_id": old["request_id"],
        "reviewed_binding": digest(old),
        "approval_id": old["approval_id"],
    }
    assert event(d, s, c, "approval_record", payload).code == "AUDIT_ONLY"
    assert s == before
    assert (
        d.apply_action(s, "request_approval", {"role": "send_approver"}, (), c).code
        == "DUPLICATE_REQUEST"
    )


def test_compact_view_retains_canonical_sets_and_requires_bound_proof():
    d, s, c = prepared()
    assert d.apply_action(
        s,
        "schedule",
        {"key": schedule_key(s)},
        (),
        c,
    ).accepted
    c.now = WINDOWS[0][0]
    assert event(d, s, c, "dispatch", {"window_index": 0}).accepted
    from types import SimpleNamespace

    view = d.serve_retrieval(s, [SimpleNamespace(tool="get_campaign")], c.now)[0].records[
        "record"
    ]
    assert "members" not in view and "members" not in view["snapshot"]
    assert view["snapshot"]["member_count"] == len(s["snapshot"]["members"]) == 2340
    assert "accepted" not in view["ledger"][0]
    assert (
        d.apply_action(s, "reconcile", {"proof": "count:2340"}, (), c).code
        == "CONSENT_UNVERIFIED_AT_SEND"
    )
    assert d.apply_action(
        s, "reconcile", {"proof": view["identity_proof"]["handle"]}, (), c
    ).accepted
    assert event(
        d, s, c, "ledger_correction", {"send_id": "send-1", "rejected": ["member-0001"]}
    ).accepted
    assert (
        d.apply_action(
            s, "reconcile", {"proof": view["identity_proof"]["handle"]}, (), c
        ).code
        == "CONSENT_UNVERIFIED_AT_SEND"
    )
    assert d.apply_action(
        s, "reconcile", {"proof": d.identity_proof(s)["handle"]}, (), c
    ).accepted


@pytest.mark.parametrize("future_version", [False, True])
def test_receipt_uses_send_time_consent_not_delivery_or_future_registry(future_version):
    from operatebench.core.protocol import PlannedEvent

    class DelayedReceipt(CampaignDomain):
        def build_plan(self):
            plan = super().build_plan()
            accepted = [f"member-{i:04d}" for i in range(2340)]
            if future_version:
                accepted = accepted[1:]
            entry = {
                "key": digest(
                    [
                        "campaign-synthetic",
                        digest(document(1)),
                        "approval-1",
                        list(WINDOWS[0]),
                    ]
                ),
                "send_id": "send-1",
                "accepted": accepted,
                "accepted_commitment": digest(accepted),
                "registry_version": "2" if future_version else "1",
                "filtered": ["member-0000"] if future_version else [],
                "at": WINDOWS[0][0],
            }
            events = [
                replace(
                    e,
                    event_type="ledger_entry",
                    payload={"entry": entry},
                    at="2026-10-05T09:30:00Z",
                )
                if e.event_type == "dispatch" and e.at == WINDOWS[0][0]
                else e
                for e in plan.events
            ]
            events.append(
                PlannedEvent(
                    "later-revocation",
                    "consent_revoked",
                    "registry",
                    100,
                    {"member": "member-0000"},
                    at="2026-10-05T09:00:00Z",
                )
            )
            return replace(plan, events=tuple(events))

    e, g = execute(DelayedReceipt, CampaignAgent("reference"))
    assert e.terminal_outcome == "sent_reconciled"
    assert g["reliable"] is (not future_version), g
    if future_version:
        assert g["findings"] == [
            {"dimension": "consent", "code": "CONSENT_UNVERIFIED_AT_SEND"}
        ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", []),
        ("reviewed_binding", "forged"),
        ("approval_id", "approval-other"),
    ],
)
def test_malformed_or_mismatched_human_binding_is_atomic(field, value):
    d, s, c = prepared()
    req = s["request"]
    payload = {
        "version": 1,
        "role": "send_approver",
        "request_id": req["request_id"],
        "reviewed_binding": digest(req),
        "approval_id": req["approval_id"],
    }
    payload[field] = value
    before, audit = deepcopy(s), deepcopy(c.records)
    assert not event(d, s, c, "approval_record", payload).accepted
    assert s == before and c.records == audit
    payload[field] = {
        "request_id": req["request_id"],
        "reviewed_binding": digest(req),
        "approval_id": req["approval_id"],
    }[field]
    assert event(d, s, c, "approval_record", payload).accepted


def test_corrupt_producer_cannot_launder_wrong_set_with_pleasant_proof():
    from operatebench.contributions.finsi.governed_campaign_launch.operation import plus
    from operatebench.core.protocol import Verdict

    class Corrupt(CampaignDomain):
        def identity_proof(self, s):
            report = super().identity_proof(s)
            if report:
                report.pop("handle")
                report["unauthorized"] = []
                report["handle"] = digest(report)
            return report

        def reduce_event(self, s, e, c):
            verdict = super().reduce_event(s, e, c)
            if e.event_type == "dispatch" and s["ledger"]:
                row = s["ledger"][0]
                row["accepted"][0] = "outsider"
                row["accepted_commitment"] = digest(row["accepted"])
                c.record(
                    "finsi_event",
                    {
                        "event_type": "dispatch",
                        "actor_id": "platform",
                        "payload": dict(e.payload),
                        "ledger": deepcopy(s["ledger"]),
                    },
                )
            return verdict

        def apply_action(self, s, k, p, refs, c):
            if k == "reconcile":
                assert p["proof"] == self.identity_proof(s)["handle"]
                s["reconciled"] = True
                s["cooldown_until"] = plus(c.now, 1440)
                c.record(
                    "finsi_action",
                    {
                        "action_type": k,
                        "payload": dict(p),
                        "version": s["version"],
                        "stored": None,
                        "binding": None,
                    },
                )
                return Verdict.ok()
            return super().apply_action(s, k, p, refs, c)

    e, g = execute(Corrupt, CampaignAgent("reference"), "consent_revoked")
    assert e.terminal_outcome == "sent_reconciled"
    assert g["findings"] == [
        {"dimension": "consent", "code": "CONSENT_UNVERIFIED_AT_SEND"}
    ], g
