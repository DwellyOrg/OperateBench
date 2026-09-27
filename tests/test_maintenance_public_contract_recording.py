# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Public exception vocabulary and single-observer integration regressions."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from operatebench.agents.model import ModelAgent
from operatebench.agents.playback import RecordingAgent
from operatebench.core.engine import Engine
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.domains.lettings.maintenance.state import EXCEPTION_TYPES
from operatebench.partial_evidence import PartialExecutionEvidence
from operatebench.runner import _execute

FIXTURE = Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"


def test_actual_projected_model_prompt_publishes_exception_names_and_eligibility():
    prompts = []

    class Capture(RetrievingReferenceAgent):
        def decide(self, observation):
            request = ModelAgent(None, model="offline").build_request(observation)
            prompts.append(json.loads(json.dumps(request.prompt)))
            return super().decide(observation)

    spec = load_spec(FIXTURE)
    Engine(MaintenanceOperation(spec, "V3"), Capture(), identity={}).run()
    schemas = [
        p["observation"]["action_schemas"]["request_exception_resolution"]
        for p in prompts
    ]
    ready = [s for s in schemas if "required" in s]
    assert ready
    for schema in ready:
        guidance = schema.get("field_guidance", {})
        assert set(guidance) >= set(EXCEPTION_TYPES)
        assert "RESOLVED_REJECTED" in guidance["APPROVAL_REJECTED"]
        assert "EXPIRED" in guidance["APPROVAL_EXPIRED"]
        assert "not eligible" in guidance["EVIDENCE_CONFLICT"]
        assert "not eligible" in guidance["REPEATED_VISIT_FAILURE"]


def test_one_partial_decision_per_sealed_decision_and_inner_call():
    partial = PartialExecutionEvidence({})

    class Counted(RetrievingReferenceAgent):
        calls = 0

        def decide(self, observation):
            self.calls += 1
            return super().decide(observation)

    inner = Counted()
    recorder = RecordingAgent(inner, observer=partial.decision)
    result = _execute(
        load_spec(FIXTURE),
        "V3",
        "reference",
        lambda: recorder,
        f"opinst_{uuid4().hex}",
        partial,
    )
    decisions = [r["decision"] for r in partial.rows if r["kind"] == "decision"]
    sealed = [d.as_dict() for d in result.tape.decisions]
    assert decisions == sealed
    assert result.tape == recorder.tape()
    assert len(decisions) == inner.calls


@pytest.mark.parametrize("scenario", ["V2", "V3"])
def test_sdk_mock_has_one_call_partial_decision_and_canonical_tape(scenario, monkeypatch):
    import socket

    from tests.test_lifecycle_openai_bridge import (
        MODEL,
        ReferenceDrivenOpenAI,
        transport_for,
        user_text,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("offline SDK control attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    wire = ReferenceDrivenOpenAI()
    partial = PartialExecutionEvidence({})
    with wire.client() as client:
        model = ModelAgent(transport_for(client), model=MODEL)
        recorder = RecordingAgent(model, observer=partial.decision)
        result = _execute(
            load_spec(FIXTURE),
            scenario,
            "offline",
            lambda: recorder,
            f"opinst_{uuid4().hex}",
            partial,
        )
    decisions = [r["decision"] for r in partial.rows if r["kind"] == "decision"]
    assert decisions == [d.as_dict() for d in result.tape.decisions]
    assert result.tape == recorder.tape()
    assert len(decisions) == wire.calls == len(model.execution_record().attempts)
    prompts = [json.loads(user_text(b)) for b in wire.bodies]
    ready = [
        p["observation"]["action_schemas"]["request_exception_resolution"]
        for p in prompts
        if "required"
        in p["observation"]["action_schemas"]["request_exception_resolution"]
    ]
    assert ready and all(set(s["field_guidance"]) >= set(EXCEPTION_TYPES) for s in ready)
    if scenario == "V2":
        assert result.outcome.status == "operational_horizon_exhausted"
        assert not result.outcome.final_state["transfer_notice_sent"]
        seen_failed = [
            p
            for p in prompts
            if any(
                row.get("dispatch_status") == "FAILED"
                for result in p["observation"]["retrieval"]["served"].values()
                for row in result.get("records", {}).get("communications", [])
            )
        ]
        assert seen_failed
        assert all(p["observation"]["last_rejection"] is None for p in seen_failed)
    else:
        assert result.outcome.status == "transferred_to_human_ownership"


@pytest.mark.parametrize("exception_type", EXCEPTION_TYPES)
@pytest.mark.parametrize(
    "status", ["OPEN", "RESOLVED_APPROVED", "RESOLVED_REJECTED", "EXPIRED"]
)
def test_published_exception_eligibility_matches_executable_guard(exception_type, status):
    from operatebench.domains.lettings.maintenance.operation import (
        _eligible_exception_approval,
    )
    from operatebench.domains.lettings.maintenance.spec import action_schema_view
    from operatebench.domains.lettings.maintenance.state import Approval, MaintenanceState

    state = MaintenanceState()
    approval = Approval(
        "checkpoint",
        "quote",
        1,
        "cycle",
        100,
        "GBP",
        "scope",
        "2031-03-03T09:00:00Z",
        "2031-03-04T09:00:00Z",
        status=status,
    )
    state.approvals["checkpoint"] = approval
    eligible = _eligible_exception_approval(state, exception_type, "checkpoint")
    guidance = action_schema_view()["request_exception_resolution"]["field_guidance"]
    wanted = {"APPROVAL_REJECTED": "RESOLVED_REJECTED", "APPROVAL_EXPIRED": "EXPIRED"}
    assert (eligible is approval) == (status == wanted.get(exception_type))
    assert _eligible_exception_approval(state, exception_type, "other") is None
    if exception_type in wanted:
        assert wanted[exception_type] in guidance[exception_type]
    else:
        assert "not eligible" in guidance[exception_type]
