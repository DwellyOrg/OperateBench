# SPDX-License-Identifier: Apache-2.0
"""Synthetic ordinary retrievals must survive tape serialization exactly."""

import json

import httpx
import openai
import pytest

from operatebench.agents.evidence import decision_projection
from operatebench.agents.model import ModelAgent
from operatebench.agents.playback import (
    RecordedModelAgent,
    RecordedOutcomeAgent,
    RecordingAgent,
    decision_record,
    outcome_from_record,
    tape_from_records,
)
from operatebench.agents.transport import content_digest
from operatebench.agents.xai_responses import XAIResponsesTransport
from operatebench.core.outcomes import Complete
from operatebench.domains.lettings.maintenance.spec import load_spec
from tests.test_operatebench_retrieval_engine import ScriptedAgent, batch, rows, run


@pytest.mark.parametrize(
    "tools",
    [
        ("list_quotes", "get_case_record", "list_quotes"),
        ("unknown_zeta", "unknown_alpha", "unknown_zeta"),
        ("list_quotes", "unknown_alpha", "list_quotes", "unknown_alpha"),
        ("list_quotes", "get_case_record"),
    ],
)
def test_serialized_tape_preserves_requests_and_exact_engine_replay(tools):
    spec = load_spec("examples/operatebench/maintenance_v0_1.yaml")
    requested = batch(*tools)
    recorder = RecordingAgent(
        ScriptedAgent([requested], tail=Complete(reason="synthetic control"))
    )
    original, _ = run(spec, recorder)
    serialized = json.loads(json.dumps(recorder.tape().as_list()))
    tape = tape_from_records(serialized)
    playback = RecordedOutcomeAgent(tape, agent_id=recorder.agent_id)
    replayed, _ = run(spec, playback)
    playback.check_exhausted()

    # Check the actual defect first: refused batches retain the original count.
    assert replayed.trajectory == original.trajectory
    assert replayed.final_state == original.final_state
    assert replayed.events == original.events
    assert replayed.status == original.status
    assert tape.decisions[0].outcome["requests"] == [
        request.as_dict() for request in requested.requests
    ]
    assert outcome_from_record(tape.decisions[0].outcome) == requested
    assert decision_record(requested) == tape.decisions[0].outcome
    assert decision_projection(requested).decision_digest_sha256 == content_digest(
        tape.decisions[0].outcome
    )
    refused = rows(original, "retrieval_refused")
    if any(tool.startswith("unknown_") for tool in tools):
        assert len(refused) == 1
        assert refused[0]["code"] == "UNKNOWN_RETRIEVAL_TOOL"
        assert refused[0]["request_count"] == len(tools)
    else:
        assert refused == []
        assert len(rows(original, "retrieval_served")) == len(set(tools))


def test_sdk_full_retrieval_json_keeps_full_names_through_parser_and_tape():
    """An independent synthetic wire control, not attribution of a past response."""
    spec = load_spec("examples/operatebench/maintenance_v0_1.yaml")
    probe = ScriptedAgent([], tail=Complete(reason="synthetic observation"))
    run(spec, probe)
    observation = probe.observations[0]
    requested = batch("list_quotes", "get_case_record", "list_quotes")
    arguments = {"requests": [request.as_dict() for request in requested.requests]}
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_synthetic_retrieval",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "grok-4.6",
                "output": [
                    {
                        "id": "fc_synthetic_retrieval",
                        "type": "function_call",
                        "call_id": "call_synthetic_retrieval",
                        "name": "retrieve",
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 10,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 30,
                },
            },
        )

    with openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        lane = XAIResponsesTransport(model="grok-4.6", client=client, deadline_seconds=30)
        recorder = RecordingAgent(
            ModelAgent(lane, model="grok-4.6", max_output_tokens=8192)
        )
        recorder.begin_episode({})
        parsed = recorder.decide(observation)
    assert len(calls) == 1
    # This assertion isolates the SDK/adapter/model parser from tape projection.
    assert parsed == requested
    tape = tape_from_records(json.loads(json.dumps(recorder.tape().as_list())))
    assert tape.decisions[0].outcome == {"kind": "RETRIEVE", **arguments}
    playback = RecordedModelAgent(
        tape,
        agent_id=recorder.agent_id,
        model="grok-4.6",
        max_output_tokens=8192,
        request_digests=[recorder.inner.build_request(observation).request_digest_sha256],
    )
    playback.begin_episode({})
    assert playback.decide(observation) == requested
    playback.check_exhausted()
    assert playback.provider_calls == 0
