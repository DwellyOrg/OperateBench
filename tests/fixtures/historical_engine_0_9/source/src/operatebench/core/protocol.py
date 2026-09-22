"""The seam between Core and a domain pack, in one file.

Core runs operations it does not understand. That only works if the things it
*does* understand are stated here and nowhere else: an episode plan (what the
world will do, and what it is waiting for), a verdict (was this accepted, and
under what name if not), an observation (what the agent is allowed to see) and
two protocols — one a domain implements, one an agent implements.

The dependency runs one way. This module imports the rest of Core and nothing
else; a domain pack imports it and hands Core plain Core-owned objects. That is
why :class:`EpisodePlan` is defined here rather than reusing the domain's own
scenario type: the domain translates its spec into this, so Core never learns
what a quote is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from operatebench.core.clock import parse_timestamp
from operatebench.core.errors import MalformedTimestampError
from operatebench.core.outcomes import AgentOutcome
from operatebench.core.read_contract import READ_OPTIONAL, ReadRequirementContract
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch, ToolResult
from operatebench.core.retrieval_evidence import (
    PUBLIC_RECORD_VERSION_ALGORITHM,
    public_record_version,
)

#: What a conditional authored event is waiting for.
TRIGGER_ON_ACTION = "action"
TRIGGER_ON_EVENT = "event"


@dataclass(frozen=True)
class PlannedTrigger:
    """The cause a conditional event hangs off, plus an optional correlation."""

    kind: str
    type_name: str
    cycle_id: str | None = None

    def matches(self, kind: str, type_name: str, cycle_id: str | None) -> bool:
        if self.kind != kind or self.type_name != type_name:
            return False
        return self.cycle_id is None or self.cycle_id == cycle_id


@dataclass(frozen=True)
class PlannedEvent:
    """One authored external event, in Core's own vocabulary.

    Either ``at`` is set (it happens at an instant) or ``trigger`` is set (it
    happens ``delay_minutes`` after its cause). Never both: the domain's spec
    loader has already refused that.
    """

    event_id: str
    event_type: str
    actor_id: str
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    triggers_agent: bool = True
    at: str | None = None
    trigger: PlannedTrigger | None = None
    delay_minutes: int = 0


@dataclass(frozen=True)
class EpisodePlan:
    """Everything Core needs to run one episode's world."""

    starts_at: str
    events: tuple[PlannedEvent, ...]
    horizon_minutes: int
    max_turns_per_invocation: int
    max_invocations: int = 200
    #: How many retrieval batches one invocation may have served. A *separate*
    #: budget from ``max_turns_per_invocation``, which bounds business outcomes:
    #: collapsing the two would make reading compete with deciding, and an agent
    #: that read carefully would run out of turns for doing the work.
    max_retrieval_batches_per_invocation: int = 6


@dataclass(frozen=True)
class Verdict:
    """The answer to "was this allowed?", with a name for "no".

    ``code`` is stable and machine-readable — it is what the negative-agent
    contract asserts on, so "this agent failed for the intended reason" is
    checkable rather than inferred from prose.

    ``bindings`` is what the accepted effect *established*: the canonical
    identities the operation now holds because of it. It is how a conditional
    authored event learns which visit, checkpoint or payment request it is
    about. Without it the only way to correlate a future event with the cause it
    hangs off is an authored literal, which is a guess about what the agent will
    call something — and a guess the agent can invalidate by choosing a
    different, equally legal string. Empty on a refusal: nothing was
    established, so there is nothing to bind to.
    """

    accepted: bool
    code: str
    detail: str = ""
    cycle_id: str | None = None
    observational: bool = False
    bindings: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        code: str = "ACCEPTED",
        *,
        cycle_id: str | None = None,
        detail: str = "",
        bindings: Mapping[str, str] | None = None,
    ) -> Verdict:
        return cls(
            accepted=True,
            code=code,
            detail=detail,
            cycle_id=cycle_id,
            bindings=dict(bindings or {}),
        )

    @classmethod
    def refused(cls, code: str, detail: str) -> Verdict:
        return cls(accepted=False, code=code, detail=detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "code": self.code,
            "detail": self.detail,
            "cycle_id": self.cycle_id,
            "observational": self.observational,
            "bindings": dict(self.bindings),
        }


@dataclass(frozen=True)
class AgentObservation:
    """What the agent sees when it is woken.

    Two halves, and the line between them is the whole design.

    The agent is told *the rules*: which actions exist, the exact payload
    contract of each one, and the vocabulary of events that can wake it. An
    agent that has to discover a required field by being refused is being
    measured on guessing the interface rather than on running the operation,
    and a wait declared against an event type this operation has never heard of
    is a mistake nobody needed to make.

    The agent is not told *the answers*: hidden environment truth, the pending
    event queue and its order, the scenario's authored expectation, the oracle
    and anything else it has not been told through the record. The disclosure
    below is derived from the operation's static vocabulary alone, so it is the
    same in every scenario — which is what makes "it discloses no answer"
    checkable rather than asserted. Session memory an agent keeps between
    invocations is its own business and is never operation state.

    That is why this record carries ``operation_instance_id`` and not
    ``scenario_id``. The scenario identity *is* the answer's name — it selects
    the authored expectation, the expected terminal and the oracle control in
    one lookup — so an agent that reads it can be tuned per case without running
    the operation, and every measurement taken afterwards is a measurement of
    recognition. The instance identity says which run this is and nothing about
    which case it is: opaque, random, minted once per run (see
    :mod:`operatebench.core.instance`). The scenario identity does not disappear;
    it moves to the run metadata, the artefact and the evaluator, which is where
    a grader reads it and an agent does not.

    ``operation_id`` stays. It is *operation* identity — which operation this
    is, the same string in every scenario of it — and the agent is already told
    the operation's whole static vocabulary: its actions, their payload
    contracts and its wake event types. Removing it would hide nothing that the
    disclosure does not already state, and it is what a record of a decision
    names when it says which operation the decision was about.
    """

    now: str
    operation_id: str
    operation_instance_id: str
    invocation_index: int
    turn_index: int
    policy: Mapping[str, Any]
    actors: Mapping[str, str]
    message_fixture_ids: Sequence[str]
    #: Every action this agent may propose, mapped to ``{"required": ...,
    #: "optional": ...}`` field-name-to-kind tables. Exactly what the domain's
    #: validator enforces, read from the same table it enforces it from.
    action_schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: The event types that can wake this agent, sorted. The operation's
    #: vocabulary, not this episode's queue: it says what a wait may name, never
    #: what is coming or when.
    wake_event_types: Sequence[str] = ()
    last_rejection: Mapping[str, Any] | None = None
    #: The operation's coarse current phase. One scalar, not a projection of the
    #: record: it says *where in its life* the operation is, which is what a
    #: planner needs in order to ask for the right records, and it says nothing
    #: about what any of them hold.
    phase: str = ""
    #: The event that woke this invocation, as an envelope rather than a fact.
    #: It replaces the bare ``trigger`` this record carried before Phase 2, and
    #: the replacement is the point rather than a rename: a trigger presented on
    #: its own reads as a statement about the world, and this says who produced
    #: it and whether their word settles anything.
    #: It carries the authority class of whoever produced it and whether that
    #: authority is authoritative, because an event says *someone asserted this*
    #: and only a retrieved record says *the record holds it*.
    event: Mapping[str, Any] | None = None
    #: The read affordances: the published catalogue, the results served **in
    #: this invocation only**, how many batches are left and how large one may
    #: be. Empty at the first turn of every invocation, by construction.
    retrieval: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "now": self.now,
            "operation_id": self.operation_id,
            "operation_instance_id": self.operation_instance_id,
            "invocation_index": self.invocation_index,
            "turn_index": self.turn_index,
            "policy": dict(self.policy),
            "actors": dict(self.actors),
            "message_fixture_ids": list(self.message_fixture_ids),
            "action_schemas": {
                action_type: deepcopy(dict(schema))
                for action_type, schema in sorted(self.action_schemas.items())
            },
            "wake_event_types": list(self.wake_event_types),
            "last_rejection": (
                dict(self.last_rejection) if self.last_rejection is not None else None
            ),
            "phase": self.phase,
            "event": dict(self.event) if self.event is not None else None,
            "retrieval": dict(self.retrieval),
        }


#: Exactly what a model may be shown, named once. The allowlist is the contract:
#: a field added to :class:`AgentObservation` and not to this tuple does not
#: reach a provider, which is the direction the failure has to point. The
#: alternative — projecting whatever the observation happens to serialise —
#: makes every future internal field a disclosure by default, and the field that
#: leaks that way is the one nobody thought about.
MODEL_VISIBLE_FIELDS: tuple[str, ...] = (
    "operation_instance_id",
    "now",
    "invocation_index",
    "turn_index",
    "policy",
    "actors",
    "message_fixture_ids",
    "action_schemas",
    "wake_event_types",
    "last_rejection",
    "operation_id",
    # Phase 2 of the tool-mediated migration. ``state`` is gone: not renamed,
    # not emptied, not summarised. The coarse ``phase`` says *where in its life*
    # the operation is, the ``event`` envelope says what someone has asserted,
    # and every authoritative fact reaches an agent only as the answer to a read
    # it asked for. That is what closes the construct claim Phase 1A left open.
    "phase",
    "event",
    "retrieval",
)


def model_projection(observation: AgentObservation) -> dict[str, Any]:
    """The model-visible view of one observation, built field by field.

    Deliberately not ``observation.as_dict()`` filtered by the allowlist, and
    deliberately not ``as_dict()`` at all: the values are read off the named
    attributes, so an object whose serialisation says something else cannot put
    it in front of a provider. Detached — dictionaries and sequences are copied
    — because what a request is hashed under must not be a live view of state
    the engine is still mutating.
    """
    return {
        "operation_instance_id": observation.operation_instance_id,
        "operation_id": observation.operation_id,
        "now": observation.now,
        "invocation_index": observation.invocation_index,
        "turn_index": observation.turn_index,
        "policy": dict(observation.policy),
        "actors": dict(observation.actors),
        "message_fixture_ids": list(observation.message_fixture_ids),
        "action_schemas": _action_schema_projection(observation),
        "wake_event_types": list(observation.wake_event_types),
        "last_rejection": (
            dict(observation.last_rejection)
            if observation.last_rejection is not None
            else None
        ),
        "phase": observation.phase,
        "event": dict(observation.event) if observation.event is not None else None,
        "retrieval": _retrieval_projection(observation.retrieval),
    }


def _retrieval_projection(retrieval: Mapping[str, Any]) -> dict[str, Any]:
    """Detach the read affordance and keep its compact contract JSON-native."""
    projected = deepcopy(dict(retrieval))
    record_contract = projected.get("record_contract")
    if isinstance(record_contract, (list, tuple)):
        projected["record_contract"] = list(record_contract)
    return projected


def _action_schema_projection(
    observation: AgentObservation,
) -> dict[str, dict[str, Any]]:
    """Show the static contract part needed at this read/action boundary.

    Required reads guard each business action until they have succeeded. Optional
    reads remain visible as metadata but never hide the payload and citation half
    or become a runtime prerequisite. Splitting each immutable schema at that
    already-visible boundary avoids dumping both halves without selecting an
    action or disclosing anything about the records themselves.
    """
    served = (
        observation.retrieval.get("served")
        if isinstance(observation.retrieval, Mapping)
        else None
    )
    catalogue = (
        observation.retrieval.get("catalogue")
        if isinstance(observation.retrieval, Mapping)
        else None
    )
    record_contract = (
        observation.retrieval.get("record_contract")
        if isinstance(observation.retrieval, Mapping)
        else None
    )
    successful = _successful_served_tools(served, catalogue, record_contract)
    projected: dict[str, dict[str, Any]] = {}
    for action_type, schema in sorted(observation.action_schemas.items()):
        reads = schema.get("reads")
        declared_reads = reads if isinstance(reads, Mapping) else {}
        optional_reads = {
            tool: deepcopy(strength)
            for tool, strength in sorted(declared_reads.items())
            if strength == READ_OPTIONAL
        }
        unmet_required = {
            tool: deepcopy(strength)
            for tool, strength in sorted(declared_reads.items())
            if strength != READ_OPTIONAL and tool not in successful
        }
        if unmet_required:
            projected[action_type] = {
                "reads": {**unmet_required, **optional_reads},
            }
        else:
            projected[action_type] = {
                name: deepcopy(schema[name])
                for name in ("required", "optional", "evidence_refs")
                if name in schema
            }
            if optional_reads:
                projected[action_type]["reads"] = optional_reads
    return projected


def _successful_served_tools(
    served: Any, catalogue: Any, record_contract: Any
) -> frozenset[str]:
    """Return successful catalogue-bound tools; malformed input fails closed."""
    catalogue_fields = {"source", "authority", "schema_id", "arguments"}
    if (
        not isinstance(served, Mapping)
        or not isinstance(catalogue, Mapping)
        or not isinstance(record_contract, (list, tuple))
        or len(record_contract) != 4
        or not all(type(value) is str and bool(value) for value in record_contract[:3])
        or record_contract[2] != PUBLIC_RECORD_VERSION_ALGORITHM
        or type(record_contract[3]) is not int
        or not 1 <= record_contract[3] <= 64
    ):
        return frozenset()
    identity_suffix, version_context, _algorithm, version_length = record_contract
    try:
        for tool, entry in catalogue.items():
            if (
                type(tool) is not str
                or not tool
                or not isinstance(entry, Mapping)
                or set(entry) != catalogue_fields
                or not all(
                    type(entry.get(name)) is str and bool(entry.get(name))
                    for name in ("source", "authority", "schema_id")
                )
                or not isinstance(entry.get("arguments"), Mapping)
                or not all(
                    type(name) is str and type(shape) is str
                    for name, shape in entry["arguments"].items()
                )
            ):
                return frozenset()
    except (TypeError, ValueError, RecursionError):
        return frozenset()
    envelope_fields = {
        "tool",
        "source",
        "authority",
        "record_id",
        "record_version",
        "as_of",
        "schema_id",
        "records",
        "ok",
        "error_code",
    }
    try:
        for tool, envelope in served.items():
            if (
                not isinstance(tool, str)
                or tool not in catalogue
                or not isinstance(envelope, Mapping)
                or set(envelope) != envelope_fields
                or envelope.get("tool") != tool
                or any(
                    envelope.get(name) != catalogue[tool].get(name)
                    for name in ("source", "authority", "schema_id")
                )
                or not all(
                    type(envelope.get(name)) is str and bool(envelope.get(name))
                    for name in (
                        "tool",
                        "source",
                        "authority",
                        "record_id",
                        "record_version",
                        "as_of",
                        "schema_id",
                    )
                )
                or not isinstance(envelope.get("records"), Mapping)
                or type(envelope.get("ok")) is not bool
                or not (
                    envelope.get("error_code") is None
                    or (
                        type(envelope.get("error_code")) is str
                        and bool(envelope.get("error_code"))
                    )
                )
            ):
                return frozenset()
            parse_timestamp(envelope["as_of"], f"served.{tool}.as_of")
            entry = catalogue[tool]
            if envelope["record_id"] != f"{entry['source']}:{identity_suffix}":
                return frozenset()
            if envelope["record_version"] != public_record_version(
                envelope["records"],
                context=version_context,
                length=version_length,
            ):
                return frozenset()
    except (MalformedTimestampError, TypeError, ValueError, RecursionError):
        return frozenset()
    return frozenset(
        tool
        for tool, envelope in served.items()
        if envelope.get("ok") is True and envelope.get("error_code") is None
    )


#: What an agent is told about the episode it is starting. The same separation
#: one level up: a run's internal identity names the scenario, the spec digest
#: and the authored expectation, and none of that is the agent's to know.
AGENT_VISIBLE_IDENTITY_FIELDS: tuple[str, ...] = (
    "operation_id",
    "operation_instance_id",
    "agent_id",
)


def agent_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The episode identity an agent is handed, from the run's internal one."""
    return {name: identity.get(name, "") for name in AGENT_VISIBLE_IDENTITY_FIELDS}


class EnvironmentContext(Protocol):
    """What a domain may ask of the environment while it reduces something.

    Deliberately small. A domain may look at the clock, write a trajectory
    record, arm or disarm a timer and ask whether a dispatch is authored to
    fail. It cannot deliver events, invoke the agent or move time.
    """

    @property
    def now(self) -> str: ...

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None: ...

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None: ...

    def cancel_timer(self, event_id: str) -> bool: ...

    def dispatch_fails(self, message_fixture_id: str) -> bool: ...


class OperationDomain(Protocol):
    """What a domain pack must implement for Core to run its operation."""

    def build_plan(self) -> EpisodePlan: ...

    def initial_state(self) -> Any: ...

    def canonical_state(self, state: Any) -> Mapping[str, Any]: ...

    def coarse_phase(self, state: Any) -> str: ...

    def event_authority(self, actor_id: str, event_type: str) -> str: ...

    def retrieval_catalogue(self) -> Mapping[str, Mapping[str, Any]]: ...

    def retrieval_record_contract(self) -> Sequence[str | int]: ...

    def read_requirements(self) -> ReadRequirementContract: ...

    def serve_retrieval(
        self,
        state: Any,
        requests: Sequence[RetrievalRequest],
        as_of: str,
    ) -> Sequence[ToolResult]: ...

    def policy_view(self) -> Mapping[str, Any]: ...

    def actor_roles(self) -> Mapping[str, str]: ...

    def message_fixture_ids(self) -> Sequence[str]: ...

    def action_schemas(self) -> Mapping[str, Mapping[str, Any]]: ...

    def wake_event_vocabulary(self) -> Sequence[str]: ...

    def reduce_event(
        self, state: Any, event: Any, context: EnvironmentContext
    ) -> Verdict: ...

    def apply_action(
        self,
        state: Any,
        action_type: str,
        payload: Mapping[str, Any],
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict: ...

    def apply_terminal(
        self,
        state: Any,
        evidence_refs: Sequence[str],
        context: EnvironmentContext,
    ) -> Verdict: ...

    def is_replay_final(self, state: Any) -> bool: ...

    def terminal_outcome(self, state: Any) -> str | None: ...

    def current_cycle_id(self, state: Any) -> str | None: ...


class OperationAgent(Protocol):
    """What an evaluated agent must implement. Deterministic in this build.

    ``decide`` returns one of *six* objects, not five. A
    :class:`~operatebench.core.retrieval.RetrieveBatch` is not an
    :data:`~operatebench.core.outcomes.AgentOutcome` — the engine branches on it
    by type, one frame before the outcome contract is consulted, so widening the
    contract five outcomes are held to was never the price of adding a read. But
    it *is* something an agent may return, and a signature that said otherwise
    would be a signature every retrieving agent in this build violates.
    """

    agent_id: str

    def begin_episode(self, identity: Mapping[str, Any]) -> None: ...

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch: ...


__all__ = [
    "AGENT_VISIBLE_IDENTITY_FIELDS",
    "MODEL_VISIBLE_FIELDS",
    "TRIGGER_ON_ACTION",
    "TRIGGER_ON_EVENT",
    "AgentObservation",
    "EnvironmentContext",
    "EpisodePlan",
    "OperationAgent",
    "OperationDomain",
    "PlannedEvent",
    "PlannedTrigger",
    "Verdict",
    "agent_identity",
    "model_projection",
]
