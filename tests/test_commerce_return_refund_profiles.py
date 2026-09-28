"""Successor Commerce tests; no SDK registration or provider required."""

import json
from pathlib import Path

import pytest

from operatebench.domains.commerce.return_refund_profiles import (
    NEGATIVE_CONTROLS,
    REFERENCE_SCENARIOS,
    build_agent,
    build_domain,
    evaluate_episode,
    load_spec,
    profile_identity,
)

EXPECTED = {
    "UK_NORMAL": ("refunded", 10500, "2026-01-27"),
    "UK_LABEL_UNKNOWN": ("refunded", 10500, "2026-01-27"),
    "UK_MISSING_DISCLOSURE": ("reviewed", 0, None),
    "UK_PAYMENT_FAILED": ("reviewed", 0, "2026-01-27"),
    "DE_NORMAL": ("refunded", 10500, "2026-01-26"),
    "DE_WITHHOLDING": ("refunded", 10500, "2026-01-26"),
    "DE_MISSING_DISCLOSURE": ("reviewed", 0, None),
    "DE_FAULT_REVIEW": ("reviewed", 0, None),
    "CA_POLICY": ("refunded", 10000, None),
    "CA_NO_POLICY": ("denied_voluntary", 0, None),
    "CA_UNKNOWN_DISPLAY": ("reviewed", 0, None),
    "CA_FAULT_REVIEW": ("reviewed", 0, None),
    "AU_MAJOR": ("refunded", 10000, None),
    "AU_MINOR": ("repaired", 0, None),
    "AU_CHANGE_OF_MIND": ("denied_voluntary", 0, None),
    "AU_UNCERTAIN_FAULT": ("reviewed", 0, None),
}

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/operatebench/commerce_return_refund_profiles/operation.yaml"


def test_uk_reference_is_a_real_core_settlement():
    assert FIXTURE.exists(), "new strict country fixture is not implemented"
    from operatebench.core.engine import Engine
    from operatebench.domains.commerce.return_refund_profiles import (
        build_agent,
        build_domain,
        load_spec,
    )

    spec = load_spec(FIXTURE)
    episode = Engine(
        build_domain(spec, "UK_NORMAL"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "commerce-test-instance",
            "scenario_id": "UK_NORMAL",
            "agent_id": "reference",
            "spec_digest_sha256": spec.content_digest,
        },
    ).run()
    assert episode.terminal_outcome == "refunded"
    assert episode.final_state["payment"]["settled_minor"] == 10500
    assert episode.final_state["case"]["refund_due_date"] == "2026-01-27"


def run(scenario, agent="reference", spec=None):
    from operatebench.core.engine import Engine
    from operatebench.domains.commerce.return_refund_profiles import (
        build_agent,
        build_domain,
        load_spec,
    )

    spec = spec or load_spec(FIXTURE)
    return Engine(
        build_domain(spec, scenario),
        build_agent(agent),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "commerce-test-instance",
            "scenario_id": scenario,
            "agent_id": agent,
            "spec_digest_sha256": spec.content_digest,
        },
    ).run()


def test_independent_grader_checks_real_causal_completion_and_rejects_forgery():
    from dataclasses import replace

    import operatebench.domains.commerce.return_refund_profiles as api

    assert hasattr(api, "evaluate_episode"), "independent grading API missing"
    spec = api.load_spec(FIXTURE)
    episode = run("UK_NORMAL")
    grade = api.evaluate_episode(episode, spec, "UK_NORMAL")
    assert grade["reliable"] is True, grade
    assert all(grade["dimensions"].values())
    assert api.evaluate_episode(replace(episode), spec, "UK_NORMAL")["reliable"] is False
    assert api.evaluate_episode(episode, spec, "DE_NORMAL")["reliable"] is False
    early = api.evaluate_episode(run("UK_NORMAL", "complete_early"), spec, "UK_NORMAL")
    assert early["dimensions"]["completion"] is True
    assert early["dimensions"]["safe_actions"] is False


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
@pytest.mark.parametrize("agent", ["reference", "reference_alternative"])
def test_all_reference_branches_against_literal_expectations(scenario, agent):
    spec = load_spec(FIXTURE)
    e = run(scenario, agent, spec)
    terminal, amount, due = EXPECTED[scenario]
    assert (
        e.terminal_outcome,
        e.final_state["payment"]["settled_minor"],
        e.final_state["case"]["refund_due_date"],
    ) == (terminal, amount, due)
    grade = evaluate_episode(e, spec, scenario)
    assert grade["reliable"], grade
    assert all(event["disposition"] == "accepted" for event in e.events)
    assert e.final_state["identity"] == profile_identity(spec, scenario)


@pytest.mark.parametrize("control", NEGATIVE_CONTROLS, ids=lambda c: c["agent_id"])
def test_separate_negative_expectations(control):
    spec = load_spec(FIXTURE)
    g = evaluate_episode(
        run(control["scenario_id"], control["agent_id"], spec),
        spec,
        control["scenario_id"],
    )
    assert not g["reliable"], g
    assert sorted(k for k, v in g["dimensions"].items() if not v) == sorted(
        control["failed_dimensions"]
    ), g
    assert {f["code"] for f in g["findings"]} == set(control["finding_codes"]), g
    assert all(g["dimensions"][k] for k in control["must_pass"]), g


@pytest.mark.parametrize("scenario", REFERENCE_SCENARIOS)
def test_real_model_parser_and_zero_provider_replay(scenario):
    from operatebench.agents.model import ModelAgent
    from operatebench.agents.playback import RecordedModelAgent, RecordingAgent
    from operatebench.core.engine import Engine
    from tests.model_transport import TEST_MODEL, observation_from_dict, response_for

    class OfflineTransport:
        def __init__(self):
            self.reference = build_agent("reference")
            self.reference.begin_episode({})
            self.digests = []

        def send(self, request):
            public = json.loads(json.dumps(request.prompt))["observation"]
            assert not {
                "scenario_id",
                "expected_terminal",
                "hidden_state",
                "state",
                "scenario_digest",
            } & set(public)
            serialized = json.dumps(public)
            assert "proof_delay_minutes" not in serialized
            assert "payment_mode" not in serialized
            assert "expected_terminal" not in serialized
            assert public["policy"]["supported_outcomes"] == ["ACT", "WAIT", "COMPLETE"]
            self.digests.append(request.request_digest_sha256)
            return response_for(self.reference.decide(observation_from_dict(public)))

    spec = load_spec(FIXTURE)
    transport = OfflineTransport()
    model = ModelAgent(
        transport, model=TEST_MODEL, agent_id="offline-mock", max_transport_calls=240
    )
    recorder = RecordingAgent(model)
    identity = {
        "operation_id": spec.operation_id,
        "operation_instance_id": "commerce-model-test",
        "scenario_id": scenario,
        "agent_id": model.agent_id,
        "spec_digest_sha256": spec.content_digest,
    }

    def execute(agent):
        return Engine(build_domain(spec, scenario), agent, identity=identity).run()

    original = execute(recorder)
    grade = evaluate_episode(original, spec, scenario)
    assert grade["reliable"], grade
    playback = RecordedModelAgent(
        recorder.tape(),
        agent_id=model.agent_id,
        model=TEST_MODEL,
        max_output_tokens=model.max_output_tokens,
        request_digests=transport.digests,
    )
    replay = execute(playback)
    playback.check_exhausted()
    assert playback.provider_calls == 0
    assert original.as_dict() == replay.as_dict()
    assert evaluate_episode(replay, spec, scenario) == grade
