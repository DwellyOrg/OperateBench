"""The provider surface, stated in full: a request, a response, and one method.

This module is deliberately the smallest thing that can be called a provider
boundary. It names no vendor, imports no SDK and opens no socket. A concrete
provider is something a caller injects, which is what makes "this episode made
zero provider calls" a property a test can *prove* rather than a claim a comment
makes: hand the agent :class:`ForbiddenTransport` and any call at all is a named
failure with a count attached.

Two separations earn their place here.

**A provider failure is not a business outcome.** A socket that closed, a
gateway that returned 502 and a response that does not satisfy the model
protocol are all statements about the *execution*, not about the operation. They
raise :class:`ProviderFailure`, which carries an :class:`ExecutionRecord` — the
attempts actually made and the fault that stopped them — so the run is excluded
with its reason recorded rather than completed with an outcome nobody produced.

**Retries are a pinned setting, not a default.** :data:`RETRY_COUNT` is zero for
this build. A retried call is a second provider run under one recorded identity,
and until the artefact can express that, the honest number is none. A bounded
stop — deadline or budget — does not discard what came before it: the execution
record keeps every attempt already made.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol

from boundarybench.jsonsafe import canonical_json_text
from operatebench.core.errors import OperateBenchError
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.partial_evidence import PartialExecutionEvidence

#: How many times a failed provider call is retried. Zero, and pinned: a retry
#: is a second provider run, and an artefact that recorded one attempt while the
#: run made three would be describing a request that did not happen.
RETRY_COUNT = 0

#: Attempts one decision is allowed to make. ``RETRY_COUNT`` retries plus the
#: attempt itself, stated as one number so the two cannot drift apart.
MAX_ATTEMPTS = RETRY_COUNT + 1

#: Why an execution stopped, when it did not stop because the operation ended.
#: Stable, machine-readable, and disjoint from every business terminal.
FAULT_TRANSPORT = "provider_transport"
FAULT_PROTOCOL = "provider_protocol"
FAULT_DEADLINE = "provider_deadline"
FAULT_BUDGET = "provider_budget"

#: Every fault this build classifies. A fault outside the set is refused at
#: construction rather than recorded as an unknown string that later reads as a
#: category nobody defined.
FAULTS: tuple[str, ...] = (
    FAULT_TRANSPORT,
    FAULT_PROTOCOL,
    FAULT_DEADLINE,
    FAULT_BUDGET,
)

#: What produced the decisions in a run. ``deterministic`` is an in-process
#: agent, ``model`` is a transport-backed one, ``recorded`` is playback.
SOURCE_DETERMINISTIC = "deterministic"
SOURCE_MODEL = "model"
SOURCE_RECORDED = "recorded"

OUTCOME_SOURCES: tuple[str, ...] = (
    SOURCE_DETERMINISTIC,
    SOURCE_MODEL,
    SOURCE_RECORDED,
)


class TransportError(OperateBenchError):
    """The provider boundary could not deliver a usable response."""


class ProviderFailure(TransportError):
    """Execution stopped at the provider, and the run is excluded rather than scored.

    Carries the :class:`ExecutionRecord` as it stood when the failure was
    raised, so a caller that excludes the run can say how many attempts it made
    and which fault ended it without reconstructing either.
    """

    def __init__(self, record: ExecutionRecord, detail: str) -> None:
        super().__init__(detail)
        self.record = record
        self.detail = detail
        self.partial_evidence: PartialExecutionEvidence | None = None

    @property
    def fault(self) -> str:
        return self.record.exclusion_code or FAULT_TRANSPORT


class ProviderCallForbiddenError(TransportError):
    """Something called the provider on a path that is required not to.

    Raised by :class:`ForbiddenTransport` on any call at all, and by
    :class:`RecordingTransport` on a call past the end of its script — which is
    the same statement in a test that scripts exactly the calls a run is allowed
    to make. Either way the failure lands at the moment of the call, naming the
    invocation and turn, rather than as a digest that quietly differs later.
    """


class SingleUseAgentError(TransportError):
    """A single-use agent was asked to execute an episode a second time.

    The determinism self-check historically re-executed the episode. For an
    agent whose decisions come from a provider that is a second provider run,
    and a stochastic one need not agree with the first — so the check would
    either cost twice and prove nothing, or be skipped and leave the run
    unreliable. Refused by name, with the alternative in the message.
    """


@dataclass(frozen=True)
class ToolCall:
    """One structured outcome a model asked for, by name and arguments.

    The *only* channel a decision travels on. Prose the model also produced is
    not read here and never reaches durable evidence — see
    :attr:`ModelResponse.text`.

    Both fields are typed ``Any`` because both are *provider output*, and this
    boundary exists to be the place that refuses provider output rather than the
    place that assumes it. A model can name a tool that is not a string and send
    arguments that are a list, and
    :func:`~operatebench.agents.model.parse_tool_call` classifies each of those
    by name — so a narrower annotation here would describe what a well-behaved
    provider sends rather than what this object has to be able to hold.
    """

    name: Any
    arguments: Any = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    """Exactly what identifies one provider request, and nothing that does not.

    The observation digest is the load-bearing field. It binds this request to
    the *exact public observation* that provoked it, so a recorded decision
    cannot be replayed against a different state that happens to arrive at the
    same index — see :mod:`operatebench.agents.playback`.

    ``prompt`` is the material a concrete transport renders and sends. It is
    carried so this boundary is usable by a real provider, and it is deliberately
    *outside* :meth:`identity`: what a request has to pin is that the same input
    produced the same call, which its digest states without putting the rendered
    prompt into a record that is compared field by field across builds.
    """

    model: str
    protocol_version: str
    max_output_tokens: int
    tool_names: tuple[str, ...]
    observation_digest_sha256: str
    prompt_digest_sha256: str
    invocation_index: int
    turn_index: int
    prompt: Mapping[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """What this build pins about a request, in one mapping."""
        return {
            "model": self.model,
            "protocol_version": self.protocol_version,
            "max_output_tokens": self.max_output_tokens,
            "tool_names": list(self.tool_names),
            "observation_digest_sha256": self.observation_digest_sha256,
            "prompt_digest_sha256": self.prompt_digest_sha256,
            "invocation_index": self.invocation_index,
            "turn_index": self.turn_index,
        }

    @property
    def request_digest_sha256(self) -> str:
        return content_digest(self.identity())


@dataclass(frozen=True)
class ModelResponse:
    """What a transport hands back. One field of it is not durable.

    ``text`` is whatever prose accompanied the structured call. It exists so a
    transport can be honest about what it received; nothing in this package puts
    it into a trajectory, an artefact or a classification detail. A canonical
    record that quoted provider prose would make two runs that decided the same
    thing compare as different runs, and would carry unreviewed model output
    into a file the benchmark publishes.
    """

    model: str
    stop_reason: str
    tool_calls: tuple[ToolCall, ...] = ()
    text: str = ""
    output_tokens: int = 0


class ModelTransport(Protocol):
    """One method. Everything a provider is, as far as OperateBench is concerned."""

    def send(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class ProviderTurnEvidence:
    """Everything one turn measured, offered to an optional recorder.

    A value object rather than a widening of :class:`ModelResponse`: what a
    recorder needs is the kernel's own telemetry and usage *beside* the request
    identity and the body that was built from it, and none of that belongs on
    the object the parser reads a decision out of.

    ``fault`` is the *kernel's* fault name when the turn ended in one —
    ``provider_rate_limited``, ``provider_server_error`` — and not the single
    Lifecycle value those collapse into on the way to an execution record.
    Recording the collapsed value would lose exactly the distinction an operator
    reading an outage needs.
    """

    request: ModelRequest
    payload: Mapping[str, Any]
    settings: Mapping[str, Any]
    telemetry: Any
    usage: Any
    response: ModelResponse | None = None
    response_id_digest_sha256: str | None = None
    fault: str | None = None


class ProviderTurnObserver(Protocol):
    """What a transport hands its per-turn evidence to, if it has one.

    Structural and one method wide. A transport that has no observer does
    exactly what it did before observers existed, which is the property that
    makes evidence capture unable to change a byte of a request.
    """

    def observe_provider_turn(self, evidence: ProviderTurnEvidence) -> None: ...


@dataclass(frozen=True)
class ProviderAttempt:
    """One call that was made, and how it ended."""

    invocation_index: int
    turn_index: int
    request_digest_sha256: str
    outcome: str
    fault: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "invocation_index": self.invocation_index,
            "turn_index": self.turn_index,
            "request_digest_sha256": self.request_digest_sha256,
            "outcome": self.outcome,
            "fault": self.fault,
        }


@dataclass(frozen=True)
class ExecutionRecord:
    """How the decisions in a run were produced — and whether it counts.

    Separate from the operation's own result on purpose. ``excluded`` says the
    run cannot be scored, and ``exclusion_code`` says which fault ended it; a
    scored run has ``excluded`` false and no code. There is no arrangement of
    these fields that reports a business terminal for an execution that never
    reached one.

    ``model``, ``protocol_version`` and ``max_output_tokens`` are the three
    settings that decide what :meth:`ModelRequest.identity` hashes to, so they
    are exactly the settings a replay needs in order to rebuild the requests a
    recorded run made and check them against the attempts recorded here. They
    travel together for that reason: recording the model without the ceiling it
    was asked under would leave the provider identity as a label beside the
    decisions rather than a part of what produced them.
    """

    outcome_source: str
    model: str | None
    protocol_version: str | None
    max_output_tokens: int | None
    transport_calls: int
    retry_count: int
    attempts: tuple[ProviderAttempt, ...]
    excluded: bool
    exclusion_code: str | None
    exclusion_detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome_source": self.outcome_source,
            "model": self.model,
            "protocol_version": self.protocol_version,
            "max_output_tokens": self.max_output_tokens,
            "transport_calls": self.transport_calls,
            "retry_count": self.retry_count,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "excluded": self.excluded,
            "exclusion_code": self.exclusion_code,
            "exclusion_detail": self.exclusion_detail,
        }


def execution_from_record(payload: Mapping[str, Any]) -> ExecutionRecord:
    """Rebuild an execution record from an artefact's ``agent_execution``.

    The structural read that follows validation. It reconstructs rather than
    trusting the mapping wholesale, so a field the artefact carries and this
    build does not know is dropped here rather than travelling on inside a
    record that claims to be one of ours.
    """
    return ExecutionRecord(
        outcome_source=str(payload["outcome_source"]),
        model=payload["model"],
        protocol_version=payload["protocol_version"],
        max_output_tokens=payload["max_output_tokens"],
        transport_calls=int(payload["transport_calls"]),
        retry_count=int(payload["retry_count"]),
        attempts=tuple(
            ProviderAttempt(
                invocation_index=int(attempt["invocation_index"]),
                turn_index=int(attempt["turn_index"]),
                request_digest_sha256=str(attempt["request_digest_sha256"]),
                outcome=str(attempt["outcome"]),
                fault=attempt["fault"],
            )
            for attempt in payload["attempts"]
        ),
        excluded=bool(payload["excluded"]),
        exclusion_code=payload["exclusion_code"],
        exclusion_detail=str(payload["exclusion_detail"]),
    )


def in_process_execution(source: str = SOURCE_DETERMINISTIC) -> ExecutionRecord:
    """The execution record of a run that never had a provider to call."""
    if source not in OUTCOME_SOURCES:
        raise TransportError(
            f"{source!r} is not an outcome source this build writes; known sources "
            f"are {list(OUTCOME_SOURCES)}"
        )
    return ExecutionRecord(
        outcome_source=source,
        model=None,
        protocol_version=None,
        max_output_tokens=None,
        transport_calls=0,
        retry_count=RETRY_COUNT,
        attempts=(),
        excluded=False,
        exclusion_code=None,
        exclusion_detail="",
    )


class ForbiddenTransport:
    """A transport that refuses to be one, and counts every attempt to use it.

    The proof object for "this path made zero provider calls". Playback and the
    determinism self-check install it; a test asserts :attr:`calls` is zero
    afterwards, which is a stronger statement than "no exception was raised"
    because it also fails a path that swallowed the refusal.
    """

    def __init__(self) -> None:
        self.calls = 0

    def send(self, request: ModelRequest) -> NoReturn:
        self.calls += 1
        raise ProviderCallForbiddenError(
            f"a provider call was made on a path that must not make one "
            f"(invocation {request.invocation_index}, turn {request.turn_index}); "
            "recorded outcomes are replayed, never re-requested"
        )


class RecordingTransport:
    """A scripted in-process transport that keeps every request it was given.

    Not a provider and not pretending to be one: it is the injectable double the
    model boundary is tested through. A scripted entry that is an exception is
    raised instead of returned, which is how transport and protocol faults get
    exercised without a network that can fail on cue.

    A call past the end of the script raises
    :class:`ProviderCallForbiddenError` rather than an "out of data" error. A
    test that scripts exactly the calls a run is allowed to make is stating a
    budget, and the honest name for exceeding it is the one playback uses.
    """

    def __init__(self, responses: Sequence[ModelResponse | Exception]) -> None:
        self._responses = list(responses)
        self.scripted = len(self._responses)
        self.requests: list[ModelRequest] = []
        self.calls = 0

    def send(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.requests.append(request)
        if not self._responses:
            raise ProviderCallForbiddenError(
                f"a provider call was made past the end of the scripted transport "
                f"(invocation {request.invocation_index}, turn "
                f"{request.turn_index}); this run is allowed {self.scripted} call(s) "
                f"and has now made {self.calls}"
            )
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def exhausted(self) -> bool:
        return not self._responses


def content_digest(payload: Any) -> str:
    """SHA-256 over canonical UTF-8 JSON. The one digest function this seam uses."""
    return hashlib.sha256(
        canonical_json_text(payload, "operatebench model boundary").encode("utf-8")
    ).hexdigest()


def observation_digest(observation: AgentObservation) -> str:
    """The digest of the exact public observation an agent was handed.

    Over :func:`~operatebench.core.protocol.model_projection` — the allowlisted
    model-visible view and nothing else, because binding a decision to anything
    the agent could not have seen binds it to something it did not answer. It is
    the same projection :meth:`ModelAgent.build_request` sends, which is what
    makes "this decision was produced from this observation" and "this request
    carried this observation" the same statement. This lives here, at the bottom
    of the import graph, so both the model agent and playback compute it the
    same way rather than from two definitions that can drift.
    """
    return content_digest(model_projection(observation))


__all__ = [
    "FAULTS",
    "FAULT_BUDGET",
    "FAULT_DEADLINE",
    "FAULT_PROTOCOL",
    "FAULT_TRANSPORT",
    "MAX_ATTEMPTS",
    "OUTCOME_SOURCES",
    "RETRY_COUNT",
    "SOURCE_DETERMINISTIC",
    "SOURCE_MODEL",
    "SOURCE_RECORDED",
    "ExecutionRecord",
    "ForbiddenTransport",
    "ModelRequest",
    "ModelResponse",
    "ModelTransport",
    "ProviderAttempt",
    "ProviderCallForbiddenError",
    "ProviderFailure",
    "ProviderTurnEvidence",
    "ProviderTurnObserver",
    "RecordingTransport",
    "SingleUseAgentError",
    "ToolCall",
    "TransportError",
    "content_digest",
    "execution_from_record",
    "in_process_execution",
    "observation_digest",
]
