"""Offline regressions for the bounded benchmark validity correction."""

import importlib
import json

import pytest

from operatebench.agents.model import ModelAgent
from operatebench.agents.outcome_contract import AGENT_TOOL_NAMES
from tests.test_lifecycle_anthropic_messages import observation


@pytest.mark.parametrize(
    "module, model",
    [
        ("openai_responses", "gpt-5.6-luna"),
        ("anthropic_messages", "claude-haiku-4-5-20251001"),
        ("anthropic_messages", "claude-sonnet-5"),
        ("mistral_chat", "mistral-small-2603"),
        ("xai_chat_completions", "grok-4.5"),
        ("xai_responses", "grok-4.5"),
        ("xai_responses", "grok-4.6"),
    ],
)
def test_wire_instructions_declare_all_six_decisions(module, model):
    adapter = importlib.import_module(f"operatebench.agents.{module}")
    request = ModelAgent(
        None, model=model, max_output_tokens=8192 if model == "grok-4.6" else 4096
    ).build_request(observation())
    payload = adapter.build_model_payload(request, model=model)
    instructions = payload.get("instructions", payload.get("system"))
    if instructions is None:
        instructions = payload["messages"][0]["content"]
    assert "five" not in instructions
    assert (
        "six" not in instructions
    )  # count and names derive from the canonical authority
    assert f"{len(AGENT_TOOL_NAMES)} decision tools" in instructions
    for name in AGENT_TOOL_NAMES:
        assert name in instructions
    assert len(payload["tools"]) == len(AGENT_TOOL_NAMES) == 6
    json.dumps(payload)


@pytest.mark.parametrize(
    "call, classification",
    [
        ("unknown", "ModelNamedAnUnknownTool"),
        ("missing", "ModelToolArgumentsMalformed"),
        ("arguments", "ModelToolArgumentsMalformed"),
    ],
)
def test_engine_malformed_feedback_is_safe_complete_and_repairable(call, classification):
    from pathlib import Path

    from operatebench.agents.model import parse_tool_call
    from operatebench.agents.playback import decision_record, outcome_from_record
    from operatebench.agents.transport import ToolCall
    from operatebench.core.engine import Engine
    from operatebench.core.outcomes import outcome_contract_problem
    from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
    from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
    from operatebench.domains.lettings.maintenance.spec import load_spec

    secret = "synthetic-private-provider-output"
    calls = {
        "unknown": ToolCall(secret, {}),
        "missing": ToolCall("retrieve", {}),
        "arguments": ToolCall("retrieve", secret),
    }
    malformed = parse_tool_call(calls[call])
    # The transient detail is untrusted, even if it happens to look like parser prose.
    malformed.detail = secret
    recovered = []

    class Stop(Exception):
        pass

    class Agent(RetrievingReferenceAgent):
        def decide(self, obs):
            if obs.turn_index == 0:
                return malformed
            if obs.turn_index == 1:
                detail = obs.last_rejection["detail"]
                assert secret not in detail
                assert classification in detail
                for name in AGENT_TOOL_NAMES:
                    assert name in detail
                if call != "unknown":
                    assert "required" in detail and "requests" in detail
                assert detail == outcome_contract_problem(
                    outcome_from_record(decision_record(malformed))
                )
                recovered.append(detail)
                return parse_tool_call(
                    ToolCall(
                        "retrieve",
                        {"requests": [{"tool": "get_case_record", "arguments": {}}]},
                    )
                )
            assert obs.retrieval["served"]
            raise Stop

    spec = load_spec(
        Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
    )
    with pytest.raises(Stop):
        Engine(
            MaintenanceOperation(spec, "V1"),
            Agent(),
            identity={"operation_id": spec.operation_id},
        ).run()
    assert recovered


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_reference_visible_wire_solves(scenario, monkeypatch):
    run = _reference_run(scenario, monkeypatch)
    assert run.reliable is (scenario != "V2")
    assert set(run.evaluation.failed_dimensions) == (
        {"recovery", "obligations"} if scenario == "V2" else set()
    )


def _reference_run(scenario, monkeypatch, mode="baseline", two_customers=False):
    from pathlib import Path

    from operatebench.core.events import EventQueue
    from operatebench.core.outcomes import Act
    from operatebench.core.protocol import model_projection
    from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode

    original = RetrievingReferenceAgent.decide

    def decide(self, obs):
        wire = json.loads(json.dumps(model_projection(obs)))

        class Visible:
            def __getattr__(self, key):
                assert key in wire, f"non-model-visible access: {key}"
                return wire[key]

        if two_customers and wire.get("retrieval", {}).get("served", {}).get(
            "get_case_record"
        ):
            record = wire["retrieval"]["served"]["get_case_record"]["records"]
            if record["issue_id"] is not None:
                assert record["issue_reporting_actor_id"] == "customer_1"
        outcome = original(self, Visible())
        if (
            isinstance(outcome, Act)
            and outcome.action_type == "send_message"
            and outcome.payload["message_fixture_id"] == "msg_completion_notice"
        ):
            payload = dict(outcome.payload)
            if mode == "wrong_recipient":
                payload["recipient_actor_id"] = "supplier_1"
            elif mode == "other_customer":
                payload["recipient_actor_id"] = "customer_2"
            elif mode == "supplier_ordinary_once" and not getattr(
                self, "_ordinary", False
            ):
                self._ordinary = True
                payload["recipient_actor_id"] = "supplier_1"
                payload["message_fixture_id"] = "msg_access_question"
                payload.pop("correlation_id", None)
            elif mode == "omit_once" and not getattr(self, "_omitted", False):
                self._omitted = True
                payload.pop("correlation_id", None)
            outcome = Act(
                outcome.action_type, payload, outcome.evidence_refs, outcome.rationale
            )
        return outcome

    monkeypatch.setattr(RetrievingReferenceAgent, "decide", decide)

    def forbidden(*args):
        raise AssertionError("future queue access")

    monkeypatch.setattr(EventQueue, "pending_event_types", forbidden)
    spec = load_spec(
        Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
    )
    if two_customers:
        from dataclasses import replace

        actors = dict(spec.actors)
        customer = next(actor for actor in actors.values() if actor.role == "customer")
        actors["customer_2"] = replace(customer, actor_id="customer_2")
        spec = replace(spec, actors=actors)
        spec = replace(spec, spec_digest_sha256=spec.compute_digest())
    return run_episode(spec, scenario, "reference")


def test_real_engine_other_customer_is_refused(monkeypatch):
    run = _reference_run("V1", monkeypatch, "other_customer", two_customers=True)
    assert any(
        row.get("code") == "COMPLETION_NOTICE_WRONG_RECIPIENT"
        for row in run.outcome.trajectory
    )
    assert not any(
        c["completion_notice_sent"] for c in run.outcome.final_state["cycles"].values()
    )


def test_real_engine_reporting_customer_succeeds_with_two_customers(monkeypatch):
    run = _reference_run("V1", monkeypatch, two_customers=True)
    assert run.reliable


def test_real_engine_supplier_notice_cannot_complete_customer_obligation(monkeypatch):
    run = _reference_run("V1", monkeypatch, "wrong_recipient")
    assert not run.reliable
    assert run.outcome.status != "completed_successfully"
    assert not any(
        c["completion_notice_sent"] for c in run.outcome.final_state["cycles"].values()
    )


def test_omitted_correlation_notice_can_recover_without_global_requirement(monkeypatch):
    run = _reference_run("V1", monkeypatch, "omit_once")
    assert run.reliable
    notices = [
        m
        for m in run.outcome.final_state["communications"]
        if m["message_fixture_id"] == "msg_completion_notice"
    ]
    assert any(m["correlation_id"] is None for m in notices)
    assert any(m["correlation_id"] is not None for m in notices)
    rows = list(run.outcome.trajectory)
    proposed = next(
        i
        for i, row in enumerate(rows)
        if row.get("record_type") == "action_proposed"
        and row.get("payload", {}).get("message_fixture_id") == "msg_completion_notice"
        and row["payload"].get("correlation_id") is None
    )
    accepted = next(
        i
        for i, row in enumerate(rows)
        if row.get("record_type") == "effect_accepted"
        and row.get("proposal_id") == rows[proposed]["proposal_id"]
    )
    assert not any(
        row.get("record_type") == "obligation_discharged"
        for row in rows[proposed:accepted]
    )


def test_independent_evaluator_rejects_wrong_recipient_despite_discharged_flags(
    monkeypatch,
):
    from copy import deepcopy

    from operatebench.domains.lettings.maintenance.evaluator import _obligations

    run = _reference_run("V1", monkeypatch)
    trajectory = deepcopy(list(run.outcome.trajectory))
    final_state = deepcopy(dict(run.outcome.final_state))
    # Preserve every accepted-effect and discharged-obligation claim. Change the
    # proposal/durable communication recipient only: runtime rejection is not the oracle.
    for row in trajectory:
        if (
            row.get("record_type") == "action_proposed"
            and row.get("payload", {}).get("message_fixture_id")
            == "msg_completion_notice"
        ):
            row["payload"]["recipient_actor_id"] = "supplier_1"
    for message in final_state["communications"]:
        if message["message_fixture_id"] == "msg_completion_notice":
            message["recipient_actor_id"] = "supplier_1"
    result = _obligations(trajectory, final_state, run.outcome.status)
    assert not result.ok
    assert "CUSTOMER_NOTICE_NOT_ESTABLISHED" in {f.code for f in result.findings}


def test_completion_correlation_is_conditionally_disclosed_on_model_wire():
    from pathlib import Path

    from operatebench.core.engine import Engine
    from operatebench.core.protocol import model_projection
    from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
    from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
    from operatebench.domains.lettings.maintenance.spec import load_spec

    class Stop(Exception):
        pass

    class Agent(RetrievingReferenceAgent):
        def decide(self, obs):
            if obs.turn_index == 0:
                from operatebench.agents.model import parse_tool_call
                from operatebench.agents.transport import ToolCall

                return parse_tool_call(
                    ToolCall(
                        "retrieve",
                        {
                            "requests": [
                                {"tool": tool, "arguments": {}}
                                for tool in obs.retrieval["catalogue"]
                            ]
                        },
                    )
                )
            schema = json.loads(json.dumps(model_projection(obs)))["action_schemas"][
                "send_message"
            ]
            assert "correlation_id" not in schema["required"]
            guidance = schema["field_guidance"]["correlation_id"]
            assert "completion: match-only discharge" in guidance
            assert "cycle_id" in guidance
            assert "discharge" in guidance
            assert "else optional" in guidance
            assert "match-only discharge; mismatch accepted" in guidance
            assert "approval reminder: open/due approval" in guidance
            assert "transfer: current/transferred" in guidance
            recipients = schema["field_guidance"]["recipient_actor_id"]
            assert "get_case_record.issue_reporting_actor_id" in recipients
            assert "completion/transfer notices:" in recipients
            assert "approval reminder: approver_1" in recipients
            assert "else any" in recipients
            raise Stop

    spec = load_spec(
        Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
    )
    with pytest.raises(Stop):
        Engine(
            MaintenanceOperation(spec, "V1"),
            Agent(),
            identity={"operation_id": spec.operation_id},
        ).run()


def test_ordinary_supplier_communication_is_accepted(monkeypatch):
    run = _reference_run("V1", monkeypatch, "supplier_ordinary_once")
    assert run.reliable
    assert any(
        m["recipient_actor_id"] == "supplier_1"
        and m["message_fixture_id"] == "msg_access_question"
        and m["correlation_id"] is None
        for m in run.outcome.final_state["communications"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "other_customer",
        "accepted_cycle",
        "accepted_time",
        "before_obligation",
        "no_acceptance",
    ],
)
def test_independent_notice_causality_rejects_invalid_records(monkeypatch, mutation):
    from copy import deepcopy

    from operatebench.domains.lettings.maintenance.evaluator import _obligations

    run = _reference_run("V1", monkeypatch, two_customers=True)
    trajectory = deepcopy(list(run.outcome.trajectory))
    state = deepcopy(dict(run.outcome.final_state))
    proposal = next(
        row
        for row in trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("payload", {}).get("message_fixture_id") == "msg_completion_notice"
    )
    accepted = next(
        row
        for row in trajectory
        if row.get("record_type") == "effect_accepted"
        and row.get("proposal_id") == proposal["proposal_id"]
    )
    if mutation == "other_customer":
        proposal["payload"]["recipient_actor_id"] = "customer_2"
        # Forge the durable state as well; the oracle must derive the actor from events.
        state["issue_reporting_actor_id"] = "customer_2"
        for message in state["communications"]:
            if message["at"] == proposal["at"]:
                message["recipient_actor_id"] = "customer_2"
    elif mutation == "accepted_cycle":
        accepted["cycle_id"] = "different_cycle"
    elif mutation == "accepted_time":
        accepted["at"] = "2031-03-01T00:00:00Z"
    elif mutation == "no_acceptance":
        trajectory.remove(accepted)
    else:
        created = next(
            row
            for row in trajectory
            if row.get("record_type") == "obligation_created"
            and row.get("obligation_id")
            == "completion_notice:" + proposal["payload"]["correlation_id"]
        )
        trajectory.remove(proposal)
        trajectory.insert(trajectory.index(created), proposal)
    result = _obligations(trajectory, state, run.outcome.status)
    assert not result.ok
    assert "CUSTOMER_NOTICE_NOT_ESTABLISHED" in {f.code for f in result.findings}
