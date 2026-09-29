"""Canonical refused attempts are audited without becoming accepted evidence."""

from dataclasses import replace

import pytest

from operatebench.contributions.prospire.property_compliance_profiles import (
    REFERENCE_SCENARIOS,
    evaluate_episode,
)
from operatebench.contributions.prospire.property_compliance_profiles.evaluator import (
    DIMENSIONS,
)
from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    ComplianceDomain,
)
from operatebench.contributions.prospire.property_compliance_profiles.spec import plain
from operatebench.core.engine import has_canonical_episode_outcome_provenance
from operatebench.core.protocol import PlannedEvent, PlannedTrigger, Verdict

from .test_pack import FIXTURE, load_spec, run_case


def codes(grade):
    assert set(grade) == {"reliable", "dimensions", "findings", "terminal_outcome"}
    assert set(grade["dimensions"]) == set(DIMENSIONS)
    assert all(type(value) is bool for value in grade["dimensions"].values())
    assert all(set(f) == {"dimension", "code"} for f in grade["findings"])
    return {f["code"] for f in grade["findings"]}


def test_binding_does_not_hide_late_remedy():
    class Binding(ComplianceDomain):
        def initial_state(self):
            state = super().initial_state()
            state["binding"] = dict(state["binding"], spec_digest="wrong")
            return state

    episode, grade = run_case(
        "EN_remediation", "delay_remedy", Binding(load_spec(FIXTURE), "EN_remediation")
    )
    assert has_canonical_episode_outcome_provenance(episode)
    assert codes(grade) == {"CASE_PROFILE_BINDING", "LATE_REMEDIATION"}
    assert not grade["dimensions"]["temporal"]


class Rejected(ComplianceDomain):
    mode = "verification"
    recovery = True
    wrong_code = False

    def build_plan(self):
        plan = super().build_plan()
        if self.mode in {"version", "leak", "prefix", "valid_prefix"}:
            report = plain(self.scenario["replacement"])
            report["supersedes"] = "nonexistent"
            reports = [report]
            if self.mode in {"leak", "prefix", "valid_prefix"}:
                first = plain(self.scenario["replacement"])
                report = dict(first, document_id="installation-v3")
                report["supersedes"] = (
                    first["supersedes"] if self.mode == "leak" else first["document_id"]
                )
                reports = [first, report]
                if self.mode == "prefix":
                    reports.append(
                        dict(report, document_id="installation-v4", supersedes="bad")
                    )
            event = PlannedEvent(
                "bad-revision",
                "report_replaced",
                "inspector",
                99,
                payload={"reports": reports},
                trigger=PlannedTrigger("action", "provisionally_close"),
                delay_minutes=1,
            )
        else:
            report = self.scenario["reports"][0]
            payload = {
                "document_id": report["document_id"],
                "verification_id": "confirmation-" + report["document_id"],
                "property_id": report["property_id"],
                "defects_resolved": [
                    f["id"] for f in report["findings"] if f["code"] != "C3"
                ],
                "qualified": False,
                "standards_met": True,
                "next_due_unchanged": report["next_due"],
            }
            if self.mode == "unknown":
                payload.update(document_id="unknown", qualified=True)
            elif self.mode == "id":
                payload.update(verification_id="wrong", qualified=True)
            elif self.mode == "clock":
                payload.update(next_due_unchanged="2099-01-01", qualified=True)
            event = PlannedEvent(
                "bad-verification",
                "verification_received",
                "verifier",
                99,
                payload=payload,
                trigger=PlannedTrigger("action", "request_remediation"),
                delay_minutes=11,
            )
        events = plan.events
        if not self.recovery or self.mode in {"leak", "prefix", "valid_prefix"}:
            events = tuple(e for e in events if e.event_type != "report_replaced")
        return replace(plan, events=(*events, event))

    def reduce_event(self, state, event, context):
        if self.mode == "valid_prefix" and event.event_id == "bad-revision":
            context.record(
                "compliance_event",
                {
                    "event_type": event.event_type,
                    "actor_id": event.actor_id,
                    "payload": plain(event.payload),
                },
            )
            return Verdict.refused("VERSION", "valid sequence refused")
        if (
            not self.recovery
            and event.event_type == "verification_received"
            and event.event_id != "bad-verification"
        ):
            # Remove legitimate recovery before recording; preserve prerecord rules.
            return Verdict.refused("AUTHORITY", "recovery unavailable")
        verdict = super().reduce_event(state, event, context)
        if self.wrong_code and event.event_id.startswith("bad-"):
            return Verdict.refused("VERSION", "wrong refusal code")
        return verdict


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("verification", {"BROKEN_REPAIR_CHAIN", "VERIFICATION"}),
        ("version", {"BROKEN_SUPERSESSION", "VERSION"}),
        ("unknown", {"BROKEN_REPAIR_CHAIN", "VERIFICATION"}),
        ("id", {"BROKEN_REPAIR_CHAIN", "VERIFICATION"}),
        ("clock", {"REPAIR_RESET_PERIODIC_CLOCK", "VERIFICATION"}),
    ],
)
def test_canonical_rejection_then_recovery(mode, expected):
    scenario = "EN_reopen" if mode == "version" else "EN_remediation"
    domain = Rejected(load_spec(FIXTURE), scenario)
    domain.mode = mode
    episode, grade = run_case(scenario, domain=domain)
    assert has_canonical_episode_outcome_provenance(episode)
    assert any(e["disposition"] == "rejected" for e in episode.events)
    assert episode.terminal_outcome == "VERIFIED_CURRENT"
    assert codes(grade) == expected
    assert grade["dimensions"]["provenance"]


@pytest.mark.parametrize("mode", ["verification", "version"])
def test_no_recovery_does_not_apply_rejected_evidence(mode):
    scenario = "EN_reopen" if mode == "version" else "EN_remediation"
    domain = Rejected(load_spec(FIXTURE), scenario)
    domain.mode, domain.recovery = mode, False
    episode, grade = run_case(scenario, domain=domain)
    found = codes(grade)
    assert "EVENT_AUTHORITY_JOIN" not in found
    assert "CAUSAL_STATE_MISMATCH" not in found
    if mode == "verification":
        assert "UNRESOLVED_WORKFLOW" in found
        assert not episode.final_state["verifications"]
    else:
        assert episode.terminal_outcome == "VERIFIED_CURRENT"
        assert found == {"BROKEN_SUPERSESSION", "VERSION"}
        assert episode.final_state["latest"]["installation"] == "installation-v1"


@pytest.mark.parametrize("mode", ["leak", "prefix"])
def test_partial_multireport_mutation_is_not_accepted(mode):
    domain = Rejected(load_spec(FIXTURE), "EN_reopen")
    domain.mode = mode
    episode, grade = run_case("EN_reopen", domain=domain)
    assert has_canonical_episode_outcome_provenance(episode)
    found = codes(grade)
    assert {"BROKEN_SUPERSESSION", "VERSION", "CAUSAL_STATE_MISMATCH"} <= found
    assert (
        not {
            "EVENT_AUTHORITY_JOIN",
            "UNJUSTIFIED_EVENT_REJECTION",
            "AUTHORED_REPORT_MISMATCH",
        }
        & found
    )
    assert episode.final_state["latest"]["installation"] == (
        "installation-v2" if mode == "leak" else "installation-v3"
    )


def test_wrong_refusal_code_is_unjustified_but_audit_continues():
    domain = Rejected(load_spec(FIXTURE), "EN_remediation")
    domain.wrong_code = True
    _, grade = run_case("EN_remediation", domain=domain)
    assert codes(grade) == {
        "BROKEN_REPAIR_CHAIN",
        "VERSION",
        "UNJUSTIFIED_EVENT_REJECTION",
    }


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
def test_cross_scenario_report_shape(scenario):
    episode, _ = run_case("EN_normal")
    grade = evaluate_episode(episode, load_spec(FIXTURE), scenario)
    found = codes(grade)
    if scenario != "EN_normal":
        assert "CASE_PROFILE_BINDING" in found


def test_unauthenticated_dimensions_are_explicitly_skipped():
    grade = evaluate_episode({}, load_spec(FIXTURE), "EN_normal")
    assert codes(grade) == {
        "UNAUTHENTICATED_EPISODE",
        "NOT_EVALUATED_UNAUTHENTICATED_EPISODE",
    }
    assert not any(grade["dimensions"].values())
    assert len(grade["findings"]) == 11
    assert grade["terminal_outcome"] == "UNTRUSTED"


def test_valid_multireport_prefix_does_not_prove_a_version_defect():
    domain = Rejected(load_spec(FIXTURE), "EN_reopen")
    domain.mode = "valid_prefix"
    episode, grade = run_case("EN_reopen", domain=domain)
    assert codes(grade) == {"VERSION", "UNJUSTIFIED_EVENT_REJECTION"}
    assert episode.terminal_outcome == "VERIFIED_CURRENT"
