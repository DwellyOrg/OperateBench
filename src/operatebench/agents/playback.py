"""Recorded outcomes, and re-execution from them without a provider.

The problem this solves is narrow and specific. An episode's determinism check
executes the episode twice; for a deterministic in-process agent that is free,
and for a model agent it is a second provider run whose answers need not match
the first. So a model episode could either pay twice for an answer that proves
nothing, or skip the check and be graded ``REPLAY_NOT_CHECKED`` — which is to
say, never be reliable.

What is recorded instead is the *decision tape*: for every call the engine made
to the agent, the structured outcome that came back and the SHA-256 digest of
the exact public observation that produced it. Re-execution then runs the real
environment, the real domain semantics and the real evaluator, and takes the
agent's decisions off the tape.

The observation digest is what makes that a check rather than a recital. A tape
replayed by index alone would agree with any run that made the same number of
calls; binding each decision to the observation it answered means the recorded
decision must be offered *the state it was actually produced from*. Reorder two
entries, retarget the tape at another scenario, edit the state the run reached,
drop an entry or add one, and playback refuses by name instead of producing a
plausible different run.

A model run's playback binds one thing more. :class:`RecordedModelAgent`
rebuilds the request each decision answered — from the model, protocol version
and output ceiling the record names, against the observation the episode is
offering — and requires it to hash to the request digest recorded beside that
decision. Without it the provider a run is attributed to is a string nothing
reads, and editing it produces a record that replays clean under another
model's name.

Playback holds a :class:`~operatebench.agents.transport.ForbiddenTransport`
where a provider would be, so "this made zero provider calls" is enforced at the
point of the call and counted, not asserted in a docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from operatebench.agents.model import (
    MalformedModelOutcome,
    ModelAgent,
    classification,
)
from operatebench.agents.transport import (
    SOURCE_RECORDED,
    ForbiddenTransport,
    content_digest,
    observation_digest,
)
from operatebench.core.errors import ArtifactError
from operatebench.core.outcomes import (
    Act,
    AgentOutcome,
    Ask,
    Complete,
    Escalate,
    Wait,
    outcome_as_dict,
)
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import (
    RETRIEVE_KIND,
    RetrievalRequest,
    RetrieveBatch,
)

#: The kind a malformed decision is recorded under. Not one of the five: the
#: tape has to be able to say "the agent produced something that was not a
#: decision", or a run whose model misbehaved could not be replayed at all.
MALFORMED_KIND = "MALFORMED"

#: Exactly the fields one recorded decision carries.
DECISION_FIELDS: tuple[str, ...] = (
    "invocation_index",
    "turn_index",
    "observation_digest_sha256",
    "outcome",
)


class PlaybackError(ArtifactError):
    """A recorded tape does not describe the run it is being replayed against."""


class TapeExhaustedError(PlaybackError):
    """The episode asked for a decision the tape does not have."""


class TapeUnconsumedError(PlaybackError):
    """The episode ended with recorded decisions left over."""


class ObservationMismatchError(PlaybackError):
    """A recorded decision was offered an observation it was not produced from."""


class DecisionOrderError(PlaybackError):
    """A recorded decision does not sit where the episode asked for it."""


class RequestIdentityMismatchError(PlaybackError):
    """A recorded attempt is not the request this run's provider identity produces."""


@dataclass(frozen=True)
class RecordedDecision:
    """One call to the agent: what it was shown, and what it decided.

    ``observation_digest_sha256`` is over the *public* observation only — the
    projection the agent was actually handed. Binding to hidden environment
    truth would bind a decision to something its author could not have seen.
    """

    invocation_index: int
    turn_index: int
    observation_digest_sha256: str
    outcome: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "invocation_index": self.invocation_index,
            "turn_index": self.turn_index,
            "observation_digest_sha256": self.observation_digest_sha256,
            "outcome": dict(self.outcome),
        }


@dataclass(frozen=True)
class DecisionTape:
    """Every decision one episode's agent made, in the order it made them."""

    decisions: tuple[RecordedDecision, ...] = ()

    def __len__(self) -> int:
        return len(self.decisions)

    def as_list(self) -> list[dict[str, Any]]:
        return [decision.as_dict() for decision in self.decisions]

    @property
    def digest(self) -> str:
        return content_digest(self.as_list())


def tape_from_records(records: Sequence[Mapping[str, Any]]) -> DecisionTape:
    """Rebuild a tape from an artefact's ``decisions`` array.

    The array has already been held to its shape by the artefact validator; this
    is the structural read that follows it, and it re-states the two conditions
    the validator cannot check in isolation.
    """
    decisions: list[RecordedDecision] = []
    for position, record in enumerate(records):
        missing = sorted(set(DECISION_FIELDS) - set(record))
        if missing:
            raise PlaybackError(
                f"recorded decision {position} is missing required field(s) {missing}"
            )
        decisions.append(
            RecordedDecision(
                invocation_index=int(record["invocation_index"]),
                turn_index=int(record["turn_index"]),
                observation_digest_sha256=str(record["observation_digest_sha256"]),
                outcome=dict(record["outcome"]),
            )
        )
    return DecisionTape(tuple(decisions))


# ------------------------------------------------------------------- recording


class RecordingAgent:
    """A transparent wrapper that keeps the tape as the episode runs.

    Transparent is the requirement: it forwards the observation unchanged and
    returns the inner agent's answer unchanged, so an episode recorded through
    it is byte-for-byte the episode that would have run without it. A wrapper
    that normalised anything would make the tape a record of the wrapper.
    """

    def __init__(
        self,
        inner: Any,
        *,
        observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._observer = observer
        self._inner = inner
        self.agent_id = str(getattr(inner, "agent_id", "agent"))
        self._decisions: list[RecordedDecision] = []

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def single_use(self) -> bool:
        return bool(getattr(self._inner, "single_use", False))

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._decisions = []
        self._inner.begin_episode(identity)

    def decide(self, observation: AgentObservation) -> Any:
        digest = observation_digest(observation)
        outcome = self._inner.decide(observation)
        self._decisions.append(
            RecordedDecision(
                invocation_index=observation.invocation_index,
                turn_index=observation.turn_index,
                observation_digest_sha256=digest,
                outcome=decision_record(outcome),
            )
        )
        if self._observer is not None:
            self._observer(self._decisions[-1].as_dict())
        return outcome

    def tape(self) -> DecisionTape:
        return DecisionTape(tuple(self._decisions))


def decision_record(outcome: Any) -> dict[str, Any]:
    """The durable projection of whatever the agent returned.

    A well-formed decision is projected by Core's own
    :func:`~operatebench.core.outcomes.outcome_as_dict`, so the tape and the
    trajectory say the same thing in the same words. Anything else is recorded
    as its stable classification code and nothing more — no provider prose, no
    unknown tool name, no argument the model invented.
    """
    if isinstance(outcome, MalformedModelOutcome):
        return {"kind": MALFORMED_KIND, "code": outcome.code}
    if isinstance(outcome, RetrieveBatch):
        return {"kind": RETRIEVE_KIND, "requests": _tape_requests(outcome)}
    if isinstance(outcome, (Act, Wait, Ask, Escalate, Complete)):
        return outcome_as_dict(outcome)
    # An object that is neither. Recorded as the classification the engine will
    # refuse it under, so the replay reproduces the same non-mutating refusal.
    return {"kind": MALFORMED_KIND, "code": "MODEL_OUTPUT_UNCLASSIFIED"}


def _tape_requests(batch: RetrieveBatch) -> list[dict[str, Any]]:
    """The durable projection of a batch's requests.

    Preserve the exact request order and multiplicity, including unknown tools.
    Canonicalization belongs to the environment when it serves a valid batch,
    not to the tape: a refused batch records its original request_count, which
    replay must reproduce. Sorting or deduplicating here loses what was asked.

    Whether a tool exists, whether the batch is within its bounds and whether the
    budget is spent are the environment's questions against a live catalogue.
    The tape does not answer them; it records what was asked.
    """
    return [request.as_dict() for request in batch.requests]


def outcome_from_record(
    record: Mapping[str, Any],
) -> AgentOutcome | RetrieveBatch | MalformedModelOutcome:
    """Rebuild one decision from its recorded projection."""
    kind = record.get("kind")
    if kind == RETRIEVE_KIND:
        try:
            return RetrieveBatch(
                tuple(
                    RetrievalRequest(
                        tool=entry["tool"], arguments=dict(entry.get("arguments") or {})
                    )
                    for entry in record["requests"]
                )
            )
        except (KeyError, TypeError) as exc:
            raise PlaybackError(
                "a recorded RETRIEVE decision is not shaped like one this build "
                f"writes ({type(exc).__name__})"
            ) from exc
    if kind == MALFORMED_KIND:
        code = record.get("code")
        if code == "MODEL_OUTPUT_UNCLASSIFIED":
            return MalformedModelOutcome()
        return classification(str(code))
    try:
        if kind == "ACT":
            return Act(
                action_type=record["action_type"],
                payload=record["payload"],
                evidence_refs=tuple(record["evidence_refs"]),
                rationale=record["rationale"],
            )
        if kind == "WAIT":
            return _wait(record)
        if kind == "ASK":
            return Ask(
                recipient_actor_id=record["recipient_actor_id"],
                message_fixture_id=record["message_fixture_id"],
                wait=_wait(record["wait"]),
                correlation_id=record["correlation_id"],
            )
        if kind == "ESCALATE":
            return Escalate(
                checkpoint_id=record["checkpoint_id"],
                exception_type=record["exception_type"],
                evidence_refs=tuple(record["evidence_refs"]),
                deadline_after_minutes=record["deadline_after_minutes"],
                rationale=record["rationale"],
            )
        if kind == "COMPLETE":
            return Complete(
                reason=record["reason"],
                evidence_refs=tuple(record["evidence_refs"]),
            )
    except (KeyError, TypeError) as exc:
        raise PlaybackError(
            f"a recorded {kind!r} decision is not shaped like one this build writes "
            f"({type(exc).__name__})"
        ) from exc
    raise PlaybackError(
        f"{kind!r} is not a decision kind this build records; a tape carrying one "
        "describes a run this build cannot reproduce"
    )


def _wait(record: Mapping[str, Any]) -> Wait:
    return Wait(
        reason=record["reason"],
        wake_on=tuple(record["wake_on"]),
        fallback_after_minutes=record["fallback_after_minutes"],
    )


# -------------------------------------------------------------------- playback


class RecordedOutcomeAgent:
    """Replays a tape against a live episode, refusing anything that does not fit.

    Not a stub that returns answers in order. Every decision is checked against
    the observation the episode is *currently* offering: its invocation and turn
    must be the ones the decision was made on, and the digest of the observation
    must be the digest recorded beside it. That is what stops a tape from
    "reproducing" a run it never described.

    The cursor is reset in :meth:`begin_episode`, so the same agent object
    replays identically however many times it is run — which is the property the
    determinism check needs, and which a stub holding a consumed iterator would
    quietly fail.
    """

    #: Playback never touches a provider, so it can always run again.
    single_use = False

    #: What produced its decisions, for the run's execution record.
    outcome_source = SOURCE_RECORDED

    def __init__(self, tape: DecisionTape, *, agent_id: str = "recorded") -> None:
        self.agent_id = agent_id
        self._tape = tape
        self._cursor = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def tape(self) -> DecisionTape:
        return self._tape

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._cursor = 0

    def decide(self, observation: AgentObservation) -> Any:
        if self._cursor >= len(self._tape):
            raise TapeExhaustedError(
                f"the episode asked for decision {self._cursor} and the recorded tape "
                f"holds {len(self._tape)}; a missing outcome is refused rather than "
                "improvised, because an improvised decision would be the benchmark's "
                "and not the agent's"
            )
        decision = self._tape.decisions[self._cursor]
        if (decision.invocation_index, decision.turn_index) != (
            observation.invocation_index,
            observation.turn_index,
        ):
            raise DecisionOrderError(
                f"recorded decision {self._cursor} was made on invocation "
                f"{decision.invocation_index} turn {decision.turn_index} and is being "
                f"replayed at invocation {observation.invocation_index} turn "
                f"{observation.turn_index}; a tape whose order moved describes a "
                "different run"
            )
        offered = observation_digest(observation)
        if offered != decision.observation_digest_sha256:
            raise ObservationMismatchError(
                f"recorded decision {self._cursor} was made from an observation with "
                f"digest {decision.observation_digest_sha256}, and the episode is "
                f"offering {offered}; a decision is bound to what produced it, so an "
                "index-only replay is not a replay"
            )
        self._cursor += 1
        return outcome_from_record(decision.outcome)

    def check_exhausted(self) -> None:
        """Refuse a run that ended with recorded decisions left unused."""
        if self._cursor != len(self._tape):
            raise TapeUnconsumedError(
                f"the episode ended after {self._cursor} decision(s) and the recorded "
                f"tape holds {len(self._tape)}; an extra outcome means the tape and "
                "the run it claims to describe are not the same run"
            )


class RecordedModelAgent(RecordedOutcomeAgent):
    """Playback that also rebuilds the provider request each decision answered.

    Everything :class:`RecordedOutcomeAgent` does, plus the check that makes a
    recorded model's identity load-bearing rather than descriptive. For each
    decision it rebuilds the :class:`~operatebench.agents.transport.ModelRequest`
    that this run's recorded provider identity — model, protocol version and
    output ceiling — would have produced *from the observation the episode is
    now offering*, and requires it to hash to the request digest recorded beside
    the decision.

    That closes the last way a model artefact can lie cheaply. The decisions,
    the states they answered and the trajectory they produced are all bound to
    each other by the tape; without this, the *model* those decisions are
    attributed to is a string nothing reads, and editing it produces a record
    that replays clean under another model's name. Editing it now moves every
    request digest in the run.

    The request is built and hashed; it is never sent. The transport handed to
    the rebuilt agent is a
    :class:`~operatebench.agents.transport.ForbiddenTransport`, so "this check
    reached no provider" is enforced at the point of a call rather than
    asserted, and :attr:`provider_calls` is the count a caller can read back.
    """

    def __init__(
        self,
        tape: DecisionTape,
        *,
        agent_id: str = "recorded",
        model: str,
        max_output_tokens: int,
        request_digests: Sequence[str],
    ) -> None:
        super().__init__(tape, agent_id=agent_id)
        self._transport = ForbiddenTransport()
        self._rebuilder = ModelAgent(
            self._transport,
            model=model,
            agent_id=agent_id,
            max_output_tokens=max_output_tokens,
        )
        self._request_digests = tuple(request_digests)

    @property
    def provider_calls(self) -> int:
        """How many provider calls this playback made. Always zero."""
        return self._transport.calls

    def decide(self, observation: AgentObservation) -> Any:
        index = self.cursor
        outcome = super().decide(observation)
        if index >= len(self._request_digests):
            raise RequestIdentityMismatchError(
                f"decision {index} has no recorded provider attempt beside it; every "
                "decision in a model run came back from exactly one call, so a tape "
                "longer than the attempts describes a different run"
            )
        rebuilt = self._rebuilder.build_request(observation).request_digest_sha256
        recorded = self._request_digests[index]
        if rebuilt != recorded:
            raise RequestIdentityMismatchError(
                f"recorded attempt {index} carries request digest {recorded}, and the "
                f"model identity this artefact names produces {rebuilt} for the "
                "observation that decision answered; the model, protocol version and "
                "output ceiling a run records are part of what produced its "
                "decisions, not a label beside them"
            )
        return outcome


__all__ = [
    "DECISION_FIELDS",
    "MALFORMED_KIND",
    "RETRIEVE_KIND",
    "DecisionOrderError",
    "DecisionTape",
    "ObservationMismatchError",
    "PlaybackError",
    "RecordedDecision",
    "RecordedModelAgent",
    "RecordedOutcomeAgent",
    "RecordingAgent",
    "RequestIdentityMismatchError",
    "TapeExhaustedError",
    "TapeUnconsumedError",
    "decision_record",
    "outcome_from_record",
    "tape_from_records",
]
