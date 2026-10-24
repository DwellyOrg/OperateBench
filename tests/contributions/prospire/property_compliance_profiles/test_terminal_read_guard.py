"""Read-guard attempts remain faults after real Core recovery."""

import pytest

from operatebench.contributions.prospire.property_compliance_profiles.agents import (
    ComplianceAgent,
)
from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    ComplianceDomain,
)
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import Verdict, model_projection

from .test_pack import FIXTURE, load_spec, run_case


class Recover(ComplianceAgent):
    def __init__(self, stale, proposals, independent=False):
        super().__init__("wrong_recipient" if independent else "reference")
        self.stale = stale
        self.proposals = list(proposals)
        self.read = False

    def decide(self, observation):
        public = model_projection(observation)
        served = "get_compliance_record" in public["retrieval"]["served"]
        self.read |= served
        if not served and (self.read if self.stale else not self.read) and self.proposals:
            kind = self.proposals.pop(0)
            return (
                Complete("Attempt before fresh read")
                if kind == "terminal"
                else Act("request_inspection")
            )
        return super().decide(observation)


@pytest.mark.parametrize("stale", [False, True], ids=["absent", "stale"])
@pytest.mark.parametrize(
    "proposals", [("terminal",), ("action",), ("action", "terminal", "terminal")]
)
@pytest.mark.parametrize("independent", [False, True])
def test_read_guard_recovery(stale, proposals, independent):
    clean, control = run_case(
        "EN_normal", "wrong_recipient" if independent else "reference"
    )
    episode, grade = run_case("EN_normal", Recover(stale, proposals, independent))
    code = "STALE_RETRIEVED_RECORD" if stale else "ACTION_WITHOUT_RETRIEVAL"
    rejections = [
        (i, row)
        for i, row in enumerate(episode.trajectory)
        if row["record_type"] in {"terminal_rejected", "action_rejected"}
        and row["code"] == code
    ]
    assert len(rejections) == len(proposals)
    assert [r["record_type"] for _, r in rejections] == [
        "terminal_rejected" if p == "terminal" else "action_rejected" for p in proposals
    ]
    # Runtime prevention and recovery are independent of grading.
    assert episode.replay_final and episode.final_state == clean.final_state
    assert episode.terminal_outcome == clean.terminal_outcome == "VERIFIED_CURRENT"
    tail = episode.trajectory[rejections[-1][0] + 1 :]
    assert any(r["record_type"] == "retrieval_served" for r in tail)
    assert any(r["record_type"] == "compliance_complete" for r in tail)
    assert not grade["reliable"]
    assert grade["dimensions"] == control["dimensions"] | {"retrieval": False}
    finding = {"dimension": "retrieval", "code": code}
    assert grade["findings"].count(finding) == 1
    assert [f for f in grade["findings"] if f != finding] == control["findings"]


def test_valid_control():
    episode, grade = run_case("EN_normal")
    assert episode.replay_final and grade["reliable"]
    assert grade["findings"] == []


@pytest.mark.parametrize(
    "code",
    ["FINALITY", "REPORTS_MISSING", "ACTED_ON_CLAIM_WITHOUT_RECORD", "WAIT_POLICY"],
)
def test_unrelated_terminal_codes_are_not_new_faults(code):
    class RefuseOnce(ComplianceDomain):
        refused = False

        def apply_terminal(self, state, evidence_refs, context):
            if not self.refused:
                self.refused = True
                return Verdict.refused(code, "Synthetic domain refusal boundary")
            return super().apply_terminal(state, evidence_refs, context)

    episode, grade = run_case(
        "EN_normal", domain=RefuseOnce(load_spec(FIXTURE), "EN_normal")
    )
    assert any(
        r["record_type"] == "terminal_rejected" and r["code"] == code
        for r in episode.trajectory
    )
    assert episode.replay_final and grade["reliable"]


def test_wait_policy_remains_unchanged():
    class InvalidWait(ComplianceAgent):
        attempted = False

        def decide(self, observation):
            if not self.attempted:
                self.attempted = True
                return Wait("Unknown wake control", ("not_a_compliance_event",), 120)
            return super().decide(observation)

    clean, control = run_case("EN_normal")
    episode, grade = run_case("EN_normal", InvalidWait("reference"))
    assert any(r["record_type"] == "wait_rejected" for r in episode.trajectory)
    assert episode.final_state == clean.final_state
    assert grade == control
