"""The five things an agent may return, and what each one must declare.

``ACT``, ``WAIT``, ``ASK``, ``ESCALATE``, ``COMPLETE``. The contract that earns
its place here is WAIT's: a wait must say *what would wake it*, either as event
types it expects or as a fallback deadline it will not sleep past. An agent that
returns "I am waiting" with neither has not made a decision the environment can
honour, and the benchmark has to be able to tell that apart from a correct wait
for an event that is genuinely coming. It is refused at construction, so a
wait-forever agent fails on its own contract rather than by timing out.

None of these are state mutations. An ``ACT`` is a *proposal*; whether it
commits is decided by the domain's validator and, where the operation says so,
by a human. The ledger keeps proposal, validation, accepted effect and
post-commit side effect apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from operatebench.core.errors import OutcomeError, WaitContractError

#: The Core outcome vocabulary, in the order the contract states it.
OUTCOME_KINDS: tuple[str, ...] = ("ACT", "WAIT", "ASK", "ESCALATE", "COMPLETE")


@dataclass(frozen=True)
class Act:
    """Perform an authorised business action. A proposal, not a commit."""

    action_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: Sequence[str] = ()
    rationale: str = ""

    kind: str = field(default="ACT", init=False)

    def __post_init__(self) -> None:
        if not self.action_type:
            raise OutcomeError("ACT must name an action type")


@dataclass(frozen=True)
class Wait:
    """Wait for declared events or a declared deadline. Never for wall-clock time."""

    reason: str
    wake_on: Sequence[str] = ()
    fallback_after_minutes: int | None = None

    kind: str = field(default="WAIT", init=False)

    def __post_init__(self) -> None:
        if not self.wake_on and self.fallback_after_minutes is None:
            raise WaitContractError(
                "WAIT must declare a wake condition (event types) or a fallback "
                "deadline in minutes; a wait with neither can never end and is "
                "refused rather than executed"
            )
        if self.fallback_after_minutes is not None and self.fallback_after_minutes <= 0:
            raise WaitContractError(
                "WAIT fallback deadline must be a positive number of simulated "
                f"minutes, got {self.fallback_after_minutes}"
            )


@dataclass(frozen=True)
class Ask:
    """Request missing information from a named actor, and wait for the answer.

    Carries its own wait: asking without declaring what the answer looks like is
    the same unbounded wait WAIT refuses, wearing a different hat.
    """

    recipient_actor_id: str
    message_fixture_id: str
    wait: Wait
    correlation_id: str | None = None

    kind: str = field(default="ASK", init=False)

    def __post_init__(self) -> None:
        if not self.recipient_actor_id:
            raise OutcomeError("ASK must name the actor being asked")


@dataclass(frozen=True)
class Escalate:
    """Create a justified human or exception checkpoint, and wait for its resolution."""

    checkpoint_id: str
    exception_type: str
    evidence_refs: Sequence[str] = ()
    deadline_after_minutes: int = 1440
    rationale: str = ""

    kind: str = field(default="ESCALATE", init=False)

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.exception_type:
            raise OutcomeError(
                "ESCALATE must name both the checkpoint it opens and the typed "
                "exception it claims; an untyped escalation cannot be judged "
                "necessary or unnecessary"
            )


@dataclass(frozen=True)
class Complete:
    """Propose a guarded terminal transition. The guard, not the agent, decides."""

    reason: str = ""
    evidence_refs: Sequence[str] = ()

    kind: str = field(default="COMPLETE", init=False)


#: Every outcome the engine accepts. A union rather than a base class so a
#: domain cannot widen the contract by subclassing.
AgentOutcome = Act | Wait | Ask | Escalate | Complete


#: The stable code the engine records when an agent's outcome cannot be read as
#: one of the five. Named rather than typed: the point of the boundary is that a
#: malformed outcome becomes a recorded refusal the episode survives, not an
#: exception that ends a run halfway through with a traceback.
MALFORMED_OUTCOME_CODE = "MALFORMED_AGENT_OUTCOME"

#: Sequences that are not sequences of references. A bare string satisfies
#: ``Sequence[str]`` and iterates into characters, which is exactly the shape an
#: agent produces by writing ``evidence_refs="evidence_1"``.
_NOT_A_SEQUENCE = (str, bytes, bytearray, Mapping)


def _refs_problem(value: Any, what: str) -> str | None:
    if isinstance(value, _NOT_A_SEQUENCE) or not isinstance(value, Sequence):
        return (
            f"{what} must be a sequence of reference strings, got {type(value).__name__}"
        )
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, str) or not item:
            return (
                f"{what}[{position}] must be a non-empty reference string, got {item!r}"
            )
    return None


def _text_problem(value: Any, what: str) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        return f"{what} must be a non-empty string, got {type(value).__name__}"
    return None


def _wait_problem(wait: Any) -> str | None:
    if not isinstance(wait, Wait):
        return f"the wait it carries is a {type(wait).__name__}, not a WAIT"
    if not isinstance(wait.reason, str):
        return "WAIT reason must be a string"
    problem = _refs_problem(wait.wake_on, "WAIT wake_on")
    if problem is not None:
        return problem
    deadline = wait.fallback_after_minutes
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, int)
    ):
        return "WAIT fallback_after_minutes must be a whole number of minutes or None"
    return None


def outcome_contract_problem(outcome: Any) -> str | None:
    """Why this object cannot be read as an agent outcome, or ``None`` if it can.

    Called at the public engine boundary, before anything indexes, serialises or
    dispatches on what an agent returned. The dataclasses refuse the obvious
    emptiness at construction, but they are not typed at runtime: an ``ACT``
    whose payload is a list, whose evidence refs are a bare string, or whose
    ``quote_version`` holds a container is a well-constructed object and a
    malformed decision; this check returns a structured refusal before a domain
    can surface a ``TypeError``.
    """
    if not isinstance(outcome, (Act, Wait, Ask, Escalate, Complete)):
        # Never forward transient parser/provider detail. Classification-only
        # tapes reproduce this same guidance without retaining raw output.
        name = type(outcome).__name__
        guidance = {
            "ModelNamedAnUnknownTool": (
                "Use a declared tool name, not an inner retrieval name."
            ),
            "ModelToolArgumentsMalformed": (
                "Check the declared argument types and required fields; retrieve "
                "requires requests, an array of tool/arguments objects."
            ),
        }.get(name, "Return one structured decision with all required fields.")
        tools = [kind.lower() for kind in (*OUTCOME_KINDS, "RETRIEVE")]
        return (
            f"an agent must return exactly one decision using {tools}, got "
            f"{name}. {guidance}"
        )
    if isinstance(outcome, Act):
        problem = _text_problem(outcome.action_type, "ACT action_type")
        if problem is not None:
            return problem
        if not isinstance(outcome.payload, Mapping):
            return f"ACT payload must be a mapping, got {type(outcome.payload).__name__}"
        for key in outcome.payload:
            if not isinstance(key, str) or not key:
                return f"ACT payload field name {key!r} must be a non-empty string"
        problem = _refs_problem(outcome.evidence_refs, "ACT evidence_refs")
        if problem is not None:
            return problem
        if not isinstance(outcome.rationale, str):
            return "ACT rationale must be a string"
        return None
    if isinstance(outcome, Wait):
        return _wait_problem(outcome)
    if isinstance(outcome, Ask):
        for value, what in (
            (outcome.recipient_actor_id, "ASK recipient_actor_id"),
            (outcome.message_fixture_id, "ASK message_fixture_id"),
        ):
            problem = _text_problem(value, what)
            if problem is not None:
                return problem
        correlation = outcome.correlation_id
        if correlation is not None and _text_problem(correlation, "ASK correlation_id"):
            return "ASK correlation_id must be a non-empty string or None"
        return _wait_problem(outcome.wait)
    if isinstance(outcome, Escalate):
        for value, what in (
            (outcome.checkpoint_id, "ESCALATE checkpoint_id"),
            (outcome.exception_type, "ESCALATE exception_type"),
        ):
            problem = _text_problem(value, what)
            if problem is not None:
                return problem
        problem = _refs_problem(outcome.evidence_refs, "ESCALATE evidence_refs")
        if problem is not None:
            return problem
        deadline = outcome.deadline_after_minutes
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline <= 0:
            return (
                "ESCALATE deadline_after_minutes must be a positive whole number of "
                f"minutes, got {deadline!r}"
            )
        if not isinstance(outcome.rationale, str):
            return "ESCALATE rationale must be a string"
        return None
    if not isinstance(outcome.reason, str):
        return "COMPLETE reason must be a string"
    return _refs_problem(outcome.evidence_refs, "COMPLETE evidence_refs")


def outcome_as_dict(outcome: AgentOutcome) -> dict[str, Any]:
    """A detached JSON projection of an outcome, for the trajectory ledger."""
    if isinstance(outcome, Act):
        return {
            "kind": "ACT",
            "action_type": outcome.action_type,
            "payload": dict(outcome.payload),
            "evidence_refs": list(outcome.evidence_refs),
            "rationale": outcome.rationale,
        }
    if isinstance(outcome, Wait):
        return {
            "kind": "WAIT",
            "reason": outcome.reason,
            "wake_on": list(outcome.wake_on),
            "fallback_after_minutes": outcome.fallback_after_minutes,
        }
    if isinstance(outcome, Ask):
        return {
            "kind": "ASK",
            "recipient_actor_id": outcome.recipient_actor_id,
            "message_fixture_id": outcome.message_fixture_id,
            "correlation_id": outcome.correlation_id,
            "wait": outcome_as_dict(outcome.wait),
        }
    if isinstance(outcome, Escalate):
        return {
            "kind": "ESCALATE",
            "checkpoint_id": outcome.checkpoint_id,
            "exception_type": outcome.exception_type,
            "evidence_refs": list(outcome.evidence_refs),
            "deadline_after_minutes": outcome.deadline_after_minutes,
            "rationale": outcome.rationale,
        }
    return {
        "kind": "COMPLETE",
        "reason": outcome.reason,
        "evidence_refs": list(outcome.evidence_refs),
    }


__all__ = [
    "MALFORMED_OUTCOME_CODE",
    "OUTCOME_KINDS",
    "Act",
    "AgentOutcome",
    "Ask",
    "Complete",
    "Escalate",
    "OutcomeError",
    "Wait",
    "WaitContractError",
    "outcome_as_dict",
    "outcome_contract_problem",
]
