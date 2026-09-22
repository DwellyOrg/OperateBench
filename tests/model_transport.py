"""In-process doubles for the OperateBench model boundary.

Nothing here opens a socket, imports a provider SDK or names a vendor. The
boundary under test is :mod:`operatebench.agents.transport`, whose whole surface
is one method, so a double is a class with that method — which is the point of
keeping the boundary that small.

:class:`ReferenceDrivenTransport` is the one that makes an end-to-end model
episode possible. Scripting a model that drives an eleven-day maintenance
operation to a legitimate terminal by hand would be scripting the reference
agent badly; instead this answers each request by handing the *public
observation carried in that request* to the shipped reference agent and encoding
what it decides as a structured tool call. The decisions are the reference's,
the path they travel is the model boundary's, and the episode reaches a real
terminal — which is what a test of replay needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from operatebench.core.outcomes import Act, AgentOutcome, Ask, Escalate, Wait
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import build_agent

#: The model a test run records as its identity. Not a real model id, and
#: deliberately not shaped like one: nothing in this suite may be mistaken for
#: evidence about a released model.
TEST_MODEL = "test-model-not-a-provider"

#: Prose a fake response carries beside its structured call, so a test can assert
#: that no durable record ever quotes it.
PROVIDER_PROSE = "PROVIDER-PROSE-THAT-MUST-NOT-BE-DURABLE"


def observation_from_dict(payload: Mapping[str, Any]) -> AgentObservation:
    """Rebuild the observation a request carries, from the request alone.

    The inverse of
    :func:`~operatebench.core.protocol.model_projection`, and it reads every
    field of it — including the action schemas and the wake vocabulary, which a
    real model needs in order to answer at all. Nothing here reconstructs a field
    the projection does not carry: the double sees exactly what a provider would
    see, which is the only way an end-to-end test of this boundary means
    anything.
    """
    return AgentObservation(
        now=payload["now"],
        operation_id=payload["operation_id"],
        operation_instance_id=payload["operation_instance_id"],
        invocation_index=payload["invocation_index"],
        turn_index=payload["turn_index"],
        policy=payload["policy"],
        actors=payload["actors"],
        message_fixture_ids=payload["message_fixture_ids"],
        action_schemas=payload["action_schemas"],
        wake_event_types=payload["wake_event_types"],
        last_rejection=payload["last_rejection"],
        phase=payload["phase"],
        event=payload["event"],
        retrieval=payload["retrieval"],
    )


def tool_call_for(outcome: AgentOutcome | RetrieveBatch) -> ToolCall:
    """Encode one Core object as the structured call a model would emit."""
    if isinstance(outcome, RetrieveBatch):
        return ToolCall(
            "retrieve",
            {"requests": [request.as_dict() for request in outcome.requests]},
        )
    if isinstance(outcome, Act):
        return ToolCall(
            "act",
            {
                "action_type": outcome.action_type,
                "payload": dict(outcome.payload),
                "evidence_refs": list(outcome.evidence_refs),
                "rationale": outcome.rationale,
            },
        )
    if isinstance(outcome, Wait):
        return ToolCall("wait", _wait_arguments(outcome))
    if isinstance(outcome, Ask):
        return ToolCall(
            "ask",
            {
                "recipient_actor_id": outcome.recipient_actor_id,
                "message_fixture_id": outcome.message_fixture_id,
                "correlation_id": outcome.correlation_id,
                "wait": _wait_arguments(outcome.wait),
            },
        )
    if isinstance(outcome, Escalate):
        return ToolCall(
            "escalate",
            {
                "checkpoint_id": outcome.checkpoint_id,
                "exception_type": outcome.exception_type,
                "evidence_refs": list(outcome.evidence_refs),
                "deadline_after_minutes": outcome.deadline_after_minutes,
                "rationale": outcome.rationale,
            },
        )
    return ToolCall(
        "complete",
        {"reason": outcome.reason, "evidence_refs": list(outcome.evidence_refs)},
    )


def _wait_arguments(wait: Wait) -> dict[str, Any]:
    return {
        "reason": wait.reason,
        "wake_on": list(wait.wake_on),
        "fallback_after_minutes": wait.fallback_after_minutes,
    }


def response_for(
    outcome: AgentOutcome | RetrieveBatch, *, model: str = TEST_MODEL
) -> ModelResponse:
    """One well-formed provider response carrying one decision, plus prose."""
    return ModelResponse(
        model=model,
        stop_reason="tool_use",
        tool_calls=(tool_call_for(outcome),),
        text=PROVIDER_PROSE,
        output_tokens=29,
    )


class ReferenceDrivenTransport:
    """Answers every request with what the reference agent would have decided.

    ``interpose`` lets a test replace the answer at a chosen call index with
    something else — a malformed response, a fault — without hand-scripting the
    rest of an eleven-day episode around it.
    """

    def __init__(
        self,
        *,
        agent_id: str = "reference",
        model: str = TEST_MODEL,
        interpose: Mapping[int, ModelResponse | Exception] | None = None,
        max_calls: int | None = None,
    ) -> None:
        self._agent = build_agent(agent_id)
        self._model = model
        self._interpose = dict(interpose or {})
        self._max_calls = max_calls
        self.requests: list[ModelRequest] = []
        self.calls = 0

    def begin(self, identity: Mapping[str, Any]) -> None:
        self._agent.begin_episode(identity)

    def send(self, request: ModelRequest) -> ModelResponse:
        index = self.calls
        self.calls += 1
        self.requests.append(request)
        if self._max_calls is not None and self.calls > self._max_calls:
            raise AssertionError(
                f"the transport was called {self.calls} times against a cap of "
                f"{self._max_calls}; a path that must not reach the provider did"
            )
        if index in self._interpose:
            answer = self._interpose[index]
            if isinstance(answer, Exception):
                raise answer
            return answer
        observation = observation_from_dict(request.prompt["observation"])
        return response_for(self._agent.decide(observation), model=self._model)


class CountingTransport:
    """A transport that answers nothing and only counts. For refusal paths."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def send(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        raise self._error
