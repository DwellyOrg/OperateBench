"""Executable four-profile contribution contract tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from operatebench.contributions.prospire.property_compliance_profiles import (
    NEGATIVE_CONTROLS,
    REFERENCE_SCENARIOS,
    build_agent,
    build_domain,
    evaluate_episode,
    load_spec,
)
from operatebench.core.engine import Engine
from operatebench.core.protocol import model_projection

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = (
    ROOT / "examples/contributions/prospire/property_compliance_profiles/operation.yaml"
)
MODULE = "operatebench.contributions.prospire.property_compliance_profiles"


def test_reference_real_core_normal():
    spec = load_spec(FIXTURE)
    outcome = Engine(
        build_domain(spec, "EN_normal"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": "reference",
        },
    ).run()
    assert outcome.terminal_outcome == "VERIFIED_CURRENT"
    assert evaluate_episode(outcome, spec, "EN_normal")["reliable"]


def test_strict_immutable_profile_spec():
    spec = load_spec(FIXTURE)
    assert len(spec.scenarios) >= 12
    assert len(spec.content_digest) == 64
    assert set(REFERENCE_SCENARIOS) == set(spec.scenarios)
    import pytest

    with pytest.raises(TypeError):
        spec.scenarios["invented"] = {}


def run_case(scenario, agent="reference", domain=None):
    spec = load_spec(FIXTURE)
    episode = Engine(
        domain or build_domain(spec, scenario),
        build_agent(agent) if isinstance(agent, str) else agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": agent if isinstance(agent, str) else agent.agent_id,
        },
    ).run()
    return episode, evaluate_episode(episode, spec, scenario)


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
def test_all_reference_cases(scenario):
    episode, grade = run_case(scenario)
    assert grade["reliable"], (scenario, grade, episode.status)
    expected = (
        "ACCESS_BLOCKED_REFERRED"
        if scenario.endswith("_access")
        else "EVIDENCE_REFERRED"
        if scenario.endswith("_qualification")
        or scenario
        in ("AU_gas_expired", "NZ_assessor_only", "EN_wrong_property", "EN_no_response")
        else "VERIFIED_CURRENT"
    )
    assert episode.terminal_outcome == expected
    if scenario.startswith("NZ_"):
        assert all(r["next_due"] is None for r in episode.final_state["reports"].values())
    if scenario.endswith("_reopen"):
        assert episode.final_state["generation"] == 2
        assert len(episode.final_state["reports"]) > len(episode.final_state["latest"])
        assert episode.final_state["verifications"]


@pytest.mark.parametrize(
    "control", NEGATIVE_CONTROLS, ids=lambda c: c["agent_id"] + "-" + c["scenario_id"]
)
def test_precise_negative_controls(control):
    episode, grade = run_case(control["scenario_id"], control["agent_id"])
    assert not grade["reliable"]
    assert {k for k, v in grade["dimensions"].items() if not v} == set(
        control["failed_dimensions"]
    ), grade
    assert {f["code"] for f in grade["findings"]} == set(control["finding_codes"]), grade
    assert all(grade["dimensions"][d] for d in control["must_pass"])
    assert episode.replay_final, (
        "negative must recover without unrelated terminal failure"
    )


def test_alternate_council_first_is_legitimate():
    episode, grade = run_case("EN_remediation", "alternate_delivery")
    assert grade["reliable"], grade
    sends = [
        r["payload"]["recipient"]
        for r in episode.trajectory
        if r["record_type"] == "compliance_action" and r["action_type"] == "send_bundle"
    ]
    assert sends == ["council", "tenant"]


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
def test_json_public_observation_only(scenario):
    class Projected:
        agent_id = "reference"

        def begin_episode(self, identity):
            assert set(identity) <= {"operation_id", "operation_instance_id", "agent_id"}
            self.delegate = build_agent("reference")
            self.delegate.begin_episode(identity)

        def decide(self, observation):
            public = json.loads(json.dumps(model_projection(observation)))
            forbidden = {
                "scenario_id",
                "expected_terminal",
                "pending_events",
                "scenario_digest",
                "replacement",
                "response_delay_minutes",
            }

            def check(x):
                if isinstance(x, dict):
                    assert not forbidden.intersection(x)
                    for v in x.values():
                        check(v)
                elif isinstance(x, list):
                    for v in x:
                        check(v)

            check(public)
            return self.delegate.decide_public(public)

    _, grade = run_case(scenario, Projected())
    assert grade["reliable"], grade


def test_forged_and_wrong_case_fail_closed():
    episode, _ = run_case("EN_normal")
    spec = load_spec(FIXTURE)
    forged = replace(episode)
    assert not evaluate_episode(forged, spec, "EN_normal")["reliable"]
    assert not evaluate_episode(episode, spec, "SC_normal")["dimensions"]["provenance"]
    assert not evaluate_episode({}, spec, "EN_normal")["reliable"]


@pytest.mark.parametrize("mutation", ["unknown", "date", "profile", "duplicate"])
def test_strict_spec_refuses_unreviewed_rules(tmp_path, mutation):
    body = json.loads(FIXTURE.read_text())
    if mutation == "unknown":
        body["unknown"] = True
    elif mutation == "date":
        body["scenarios"]["EN_normal"]["reports"][0]["next_due"] = "never"
    elif mutation == "profile":
        body["scenarios"]["EN_normal"]["profile_id"] = "global"
    raw = json.dumps(body)
    if mutation == "duplicate":
        raw = raw.replace('"version": "0.2.0"', '"version": "0.2.0", "version": "0.2.0"')
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_spec(path)


def test_independent_grader_rejects_corrupt_reducer_acceptance():
    from operatebench.contributions.prospire.property_compliance_profiles import operation
    from operatebench.core.protocol import Verdict

    class Corrupt(operation.ComplianceDomain):
        def apply_action(self, state, action_type, payload, evidence_refs, context):
            if action_type == "accept_report":
                state["accepted"].append(payload["document_id"])
                context.record(
                    "compliance_action",
                    {
                        "action_type": action_type,
                        "payload": dict(payload),
                        "evidence_refs": list(evidence_refs),
                    },
                )
                return Verdict.ok()
            return super().apply_action(
                state, action_type, payload, evidence_refs, context
            )

    spec = load_spec(FIXTURE)
    _, grade = run_case(
        "AU_qualification", "trust_bad_report", Corrupt(spec, "AU_qualification")
    )
    assert not grade["dimensions"]["qualification"]
    assert any(f["code"] == "ACCEPTED_UNQUALIFIED" for f in grade["findings"])


def test_independent_grader_rejects_unjustified_evidence_referral():
    from operatebench.contributions.prospire.property_compliance_profiles import operation
    from operatebench.core.outcomes import Act

    class Corrupt(operation.ComplianceDomain):
        def _review_reason(self, state, now):
            return "QUALIFICATION"

    class Refers:
        agent_id = "reference"

        def begin_episode(self, identity):
            self.delegate = build_agent("reference")
            self.sent = False

        def decide(self, observation):
            public = model_projection(observation)
            if (
                "get_compliance_record" in public["retrieval"].get("served", {})
                and not self.sent
            ):
                self.sent = True
                return Act("request_review", {"reason": "QUALIFICATION"})
            return self.delegate.decide_public(public)

    spec = load_spec(FIXTURE)
    _, grade = run_case("EN_normal", Refers(), Corrupt(spec, "EN_normal"))
    assert not grade["reliable"], (
        "review acknowledgement is not evidence that referral was justified"
    )
    assert not grade["dimensions"]["decision_validity"]


def test_calendar_and_nz_standards_boundaries():
    from operatebench.contributions.prospire.property_compliance_profiles import operation

    report_problem = operation.report_problem
    from operatebench.contributions.prospire.property_compliance_profiles.spec import (
        plain,
    )

    spec = load_spec(FIXTURE)
    s = spec.scenarios["EN_normal"]
    r = plain(s["reports"][0])
    r["next_due"] = "2026-09-26"
    policy = plain(spec.profiles[s["profile_id"]])
    assert report_problem(r, s["initial"], policy, "2026-09-25T23:59:00Z") is None
    assert report_problem(r, s["initial"], policy, "2026-09-26T23:59:00Z") is None
    assert report_problem(r, s["initial"], policy, "2026-09-27T00:00:00Z") == "EXPIRY"
    nz = spec.scenarios["NZ_normal"]
    r = plain(nz["reports"][0])
    del r["standards"]["ventilation"]
    assert (
        report_problem(
            r,
            nz["initial"],
            plain(spec.profiles[nz["profile_id"]]),
            "2026-09-26T09:00:00Z",
        )
        == "STANDARDS"
    )


def test_victoria_independent_clocks_and_nz_no_expiry():
    episode, grade = run_case("AU_normal")
    assert grade["reliable"]
    assert (
        episode.final_state["reports"]["electrical-v1"]["next_due"]
        != episode.final_state["reports"]["gas-v1"]["next_due"]
    )
    episode, grade = run_case("NZ_reopen")
    assert grade["reliable"]
    assert episode.final_state["reports"]["healthy_homes-v2"]["next_due"] is None
    assert (
        episode.final_state["verifications"]["healthy_homes-v2"]["next_due_unchanged"]
        is None
    )


def test_nz_adverse_assessment_records_failed_standard_before_repair():
    spec = load_spec(FIXTURE)
    report = spec.scenarios["NZ_remediation"]["reports"][0]
    assert report["standards"]["ventilation"]["met"] is False
    assert report["findings"][0]["standard"] == "ventilation"
    episode, grade = run_case("NZ_remediation")
    assert grade["reliable"], grade
    assert episode.final_state["verifications"]["healthy_homes-v1"]["standards_met"]


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
def test_persisted_core_decision_tape_replays_exactly(scenario, tmp_path):
    from operatebench.agents.playback import (
        RecordedOutcomeAgent,
        RecordingAgent,
        tape_from_records,
    )

    recorder = RecordingAgent(build_agent("reference"))
    original, grade = run_case(scenario, recorder)
    assert grade["reliable"]
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(recorder.tape().as_list()))
    playback = RecordedOutcomeAgent(
        tape_from_records(json.loads(path.read_text())), agent_id="reference"
    )
    replay, replay_grade = run_case(scenario, playback)
    playback.check_exhausted()
    assert original.as_dict() == replay.as_dict()
    assert grade == replay_grade


def test_tape_rejects_wrong_profile_before_accepting_decisions():
    from operatebench.agents.playback import RecordedOutcomeAgent, RecordingAgent

    recorder = RecordingAgent(build_agent("reference"))
    run_case("EN_normal", recorder)
    playback = RecordedOutcomeAgent(recorder.tape(), agent_id="reference")
    with pytest.raises(Exception, match="observation"):
        run_case("SC_normal", playback)


def test_canonical_state_must_match_causal_report_history():
    from operatebench.contributions.prospire.property_compliance_profiles import operation

    class Corrupt(operation.ComplianceDomain):
        def canonical_state(self, state):
            snapshot = super().canonical_state(state)
            if snapshot["final"]:
                snapshot["reports"]["installation-v1"]["next_due"] = "2099-01-01"
            return snapshot

    spec = load_spec(FIXTURE)
    _, grade = run_case("EN_normal", domain=Corrupt(spec, "EN_normal"))
    assert not grade["dimensions"]["provenance"]


def test_late_remedy_preserves_breach_after_present_safety_recovers():
    from operatebench.contributions.prospire.property_compliance_profiles.agents import (
        AGENT_IDS,
    )

    assert "delay_remedy" in AGENT_IDS
    episode, grade = run_case("EN_remediation", "delay_remedy")
    assert episode.terminal_outcome == "CURRENT_WITH_HISTORICAL_BREACH"
    assert episode.final_state["historical_breach"]
    assert {k for k, v in grade["dimensions"].items() if not v} == {"temporal"}, grade
    assert {x["code"] for x in grade["findings"]} == {"LATE_REMEDIATION"}
