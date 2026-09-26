# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Bounded public budget and parser-to-engine repair regressions; no live I/O."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from operatebench.agents.model import ModelAgent, parse_tool_call
from operatebench.agents.transport import ToolCall
from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, outcome_contract_problem
from operatebench.core.protocol import model_projection
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.evaluator import _human
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec

FIXTURE = Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"


class StopProbe(Exception):
    pass


@pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
@pytest.mark.parametrize("budget", [None, 0, 2])
def test_actual_projection_budget_matches_enforcement_and_http(scenario_id, budget):
    from tests.test_lifecycle_openai_bridge import (
        MODEL,
        ReferenceDrivenOpenAI,
        transport_for,
        user_text,
    )

    spec = load_spec(FIXTURE)
    if budget is not None:
        scenarios = dict(spec.scenarios)
        scenarios[scenario_id] = replace(
            scenarios[scenario_id], human_checkpoint_budget=budget
        )
        spec = replace(spec, scenarios=scenarios)
    scenario = spec.scenario(scenario_id)
    seen = []

    class Probe(RetrievingReferenceAgent):
        def decide(self, observation):
            seen.append(observation)
            raise StopProbe

    with pytest.raises(StopProbe):
        Engine(
            MaintenanceOperation(spec, scenario_id),
            Probe(),
            identity={"operation_id": spec.operation_id},
        ).run()
    projection = model_projection(seen[0])
    policy = projection["policy"]
    assert policy["human_checkpoint_budget"] == scenario.human_checkpoint_budget
    assert "maximum total" in policy["human_checkpoint_guidance"]
    assert "Exceeding it fails" in policy["human_checkpoint_guidance"]
    rows = [
        {
            "record_type": "checkpoint_opened",
            "checkpoint_type": scenario.required_checkpoint_types[0],
        }
        for _ in range(scenario.human_checkpoint_budget + 1)
    ]
    evaluated = _human(rows, scenario)
    assert evaluated.counts["budget"] == policy["human_checkpoint_budget"]
    assert not evaluated.ok
    assert "HUMAN_BUDGET_EXCEEDED" in {f.code for f in evaluated.findings}

    # Real SDK serialization into MockTransport: no sockets or credentials.
    wire = ReferenceDrivenOpenAI()
    with wire.client() as client:
        transport = transport_for(client)
        request = ModelAgent(None, model=MODEL).build_request(seen[0])
        transport.send(request)
    body = wire.bodies[0]
    assert json.loads(user_text(body))["observation"]["policy"] == policy
    assert "current observation" in body["instructions"]


@pytest.mark.parametrize(
    "wrong_refs", [{"private-marker": "private-value"}, "private-marker", [3]]
)
def test_wrong_act_refs_repair_on_next_real_engine_invocation(wrong_refs):
    spec = load_spec(FIXTURE)
    repaired = []

    class Agent(RetrievingReferenceAgent):
        pending = None
        checked = False

        def decide(self, observation):
            wire = model_projection(observation)
            if self.pending is not None:
                detail = wire["last_rejection"]["detail"]
                assert (
                    "For act, evidence_refs must be an array of non-empty strings"
                    in detail
                )
                assert "For retrieve, requests must be an array of objects" in detail
                assert "private-marker" not in json.dumps(wire)
                assert "private-value" not in json.dumps(wire)
                corrected = parse_tool_call(ToolCall("act", self.pending))
                assert isinstance(corrected, Act)
                assert outcome_contract_problem(corrected) is None
                self.pending = None
                self.checked = True
                return corrected
            if self.checked:
                assert wire["last_rejection"] is None
                repaired.append(True)
                raise StopProbe
            outcome = super().decide(observation)
            if isinstance(outcome, Act):
                self.pending = {
                    "action_type": outcome.action_type,
                    "payload": dict(outcome.payload),
                    "evidence_refs": list(outcome.evidence_refs),
                    "rationale": outcome.rationale,
                }
                malformed = parse_tool_call(
                    ToolCall("act", {**self.pending, "evidence_refs": wrong_refs})
                )
                assert type(malformed).__name__ == "ModelToolArgumentsMalformed"
                return malformed
            return outcome

    with pytest.raises(StopProbe):
        Engine(
            MaintenanceOperation(spec, "V1"),
            Agent(),
            identity={"operation_id": spec.operation_id},
        ).run()
    assert repaired == [True]


def test_real_redundant_intervention_still_exceeds_published_budget():
    spec = load_spec(FIXTURE)

    class Agent(RetrievingReferenceAgent):
        approval = None
        stage = 0

        def decide(self, observation):
            if self.approval is not None:
                assert observation.last_rejection is None
                self.stage += 1
                if self.stage == 1:
                    return RetrieveBatch(
                        tuple(
                            RetrievalRequest(tool, {})
                            for tool in (
                                "get_case_record",
                                "list_quotes",
                                "list_checkpoints",
                            )
                        )
                    )
                if self.stage == 2:
                    return self.approval
                raise StopProbe
            outcome = super().decide(observation)
            if isinstance(outcome, Act) and outcome.action_type == "request_approval":
                self.approval = outcome
            return outcome

    operation = MaintenanceOperation(spec, "V1")
    engine = Engine(operation, Agent(), identity={"operation_id": spec.operation_id})
    with pytest.raises(StopProbe):
        engine.run()
    evaluated = _human(engine.ledger.as_list(), spec.scenario("V1"))
    assert evaluated.counts["human_touches"] == 2
    assert (
        evaluated.counts["budget"] == operation.policy_view()["human_checkpoint_budget"]
    )
    assert [finding.code for finding in evaluated.findings] == ["HUMAN_BUDGET_EXCEEDED"]
