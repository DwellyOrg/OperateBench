"""Bounded parent counterexamples and independent producer controls."""

from dataclasses import replace

import pytest

from operatebench.contributions.finsi.governed_campaign_launch import (
    evaluate_episode,
    load_spec,
)
from operatebench.contributions.finsi.governed_campaign_launch.agents import CampaignAgent
from operatebench.contributions.finsi.governed_campaign_launch.operation import (
    CampaignDomain,
)
from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import PlannedEvent

from .test_pack import FIXTURE, event, prepared


class Probe(CampaignAgent):
    def __init__(self, mode):
        super().__init__("reference")
        self.mode = mode
        self.tried = False

    def decide_public(self, obs):
        served = obs["retrieval"].get("served", {})
        if "get_campaign" not in served:
            return super().decide_public(obs)
        s = served["get_campaign"]["records"]["record"]
        if self.mode == "no_work":
            if s["status_owed"]:
                return Act("status", {})
            if s["withdrawn"]:
                return Complete("withdrawn")
            if obs["now"] >= s["window"][1]:
                return Act("withdraw", {})
            return Wait("wait", tuple(obs["wake_event_types"]), 360)
        if self.mode == "unsafe" and self.tried and s["schedules"] and not s["ledger"]:
            return Wait("await dispatch", tuple(obs["wake_event_types"]), 360)
        decision = super().decide_public(obs)
        if (
            self.mode == "no_notices"
            and isinstance(decision, Act)
            and decision.action_type in ("remind", "final_notice")
        ):
            return Wait("skip notice", tuple(obs["wake_event_types"]), 360)
        if (
            self.mode == "unsafe"
            and isinstance(decision, Act)
            and decision.action_type == "withdraw"
            and not self.tried
        ):
            self.tried = True
            return Act("schedule", {"key": s["scheduling_key"]})
        return decision


def exercise(case, mode="reference", domain=CampaignDomain):
    spec = load_spec(FIXTURE)
    e = Engine(
        domain(spec, case),
        Probe(mode),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "repair",
            "agent_id": "reference",
        },
    ).run()
    return e, evaluate_episode(e, spec, case)


@pytest.mark.parametrize(
    "mode,case,codes",
    [
        ("no_work", "baseline", {"WORKFLOW_NOT_ATTEMPTED", "REQUIRED_FOLLOWUP_MISSED"}),
        ("no_notices", "late_settlement", {"REQUIRED_FOLLOWUP_MISSED"}),
        ("unsafe", "consent_unenforced", {"CONSENT_UNVERIFIED_AT_SEND"}),
    ],
)
def test_parent_counterexamples(mode, case, codes):
    e, g = exercise(case, mode)
    assert {f["code"] for f in g["findings"]} == codes, g
    assert not g["reliable"]
    assert {k for k, v in g["dimensions"].items() if not v} == {
        "consent" if mode == "unsafe" else "workflow"
    }
    if mode == "unsafe":
        assert not e.final_state["ledger"] and not e.final_state["schedules"]
        assert e.terminal_outcome == "withdrawn_no_send"


class FailedCount(CampaignDomain):
    def build_plan(self):
        p = super().build_plan()
        return replace(
            p,
            events=(
                *p.events,
                PlannedEvent(
                    "failed-count",
                    "connector_result",
                    "platform",
                    0,
                    {"status": "failed", "count": 0},
                    at="2026-10-01T09:00:00Z",
                ),
            ),
        )


def test_failed_count_recovers_authoritative_membership():
    e, g = exercise("baseline", domain=FailedCount)
    assert g["reliable"], g
    assert e.terminal_outcome == "sent_reconciled"
    assert len(e.final_state["snapshot"]["members"]) == 2340


def test_failed_count_does_not_authorize_empty_withdrawal_or_transfer():
    d, s, c = prepared()
    assert event(d, s, c, "connector_result", {"status": "failed", "count": 0}).accepted
    assert d.apply_action(s, "withdraw", {}, (), c).code == "ABSENCE_TREATED_AS_VALUE"
    assert d.apply_action(s, "request_transfer", {}, (), c).code == "UNJUSTIFIED_TRANSFER"


class PermissiveConsent(CampaignDomain):
    def apply_action(self, s, k, p, refs, c):
        original = s["consent_enforced"]
        if k == "schedule":
            s["consent_enforced"] = True
        try:
            return super().apply_action(s, k, p, refs, c)
        finally:
            s["consent_enforced"] = original


def test_independent_grader_unsafe_send_survives_handoff():
    e, g = exercise("consent_unenforced", "unsafe", PermissiveConsent)
    assert e.terminal_outcome == "TRANSFERRED_UNRESOLVED"
    assert "member-0000" in e.final_state["ledger"][0]["accepted"]
    assert e.final_state["ledger"][0]["registry_version"] is None
    assert {f["code"] for f in g["findings"]} == {"CONSENT_UNVERIFIED_AT_SEND"}


@pytest.mark.parametrize(
    "case", ["baseline", "late_settlement", "consent_unenforced", "timeout_unprotected"]
)
def test_unchanged_producer_positive_controls(case):
    assert exercise(case)[1]["reliable"]


class InheritedSend(CampaignDomain):
    def build_plan(self):
        p = super().build_plan()
        row = {
            "key": "external-key",
            "send_id": "external-send",
            "accepted": ["member-0000"],
            "accepted_commitment": None,
            "registry_version": None,
            "filtered": [],
            "at": "2026-10-01T09:00:00Z",
        }
        return replace(
            p,
            events=(
                *p.events,
                PlannedEvent(
                    "inherited-send",
                    "ledger_entry",
                    "platform",
                    0,
                    {"entry": row},
                    at="2026-10-01T09:00:00Z",
                ),
            ),
        )


def test_inherited_unsupported_dispatch_is_legitimate_handoff():
    e, g = exercise("consent_unenforced", domain=InheritedSend)
    assert e.terminal_outcome == "TRANSFERRED_UNRESOLVED"
    assert g["reliable"], g


class SameTimeApproval(CampaignDomain):
    def build_plan(self):
        p = super().build_plan()
        return replace(
            p,
            events=tuple(
                replace(e, delay_minutes=4670) if e.event_type == "approval_record" else e
                for e in p.events
            ),
        )


def test_same_time_approval_has_no_notice_ordering_trap():
    e, g = exercise("baseline", domain=SameTimeApproval)
    assert g["reliable"], g
    assert not e.final_state["reminders"]


class DefectiveImport(CampaignDomain):
    def build_plan(self):
        p = super().build_plan()
        events = []
        for e in p.events:
            if e.event_type == "document_ready":
                payload = dict(e.payload)
                payload["document"] = dict(payload["document"], sections=["intro"])
                e = replace(e, payload=payload)
            events.append(e)
        return replace(p, events=tuple(events))


def test_exhausted_defective_import_allows_early_safe_withdrawal():
    e, g = exercise("baseline", domain=DefectiveImport)
    assert e.terminal_outcome == "withdrawn_no_send"
    assert e.final_state["pushes"] == {"1": 2}
    assert g["reliable"], g
