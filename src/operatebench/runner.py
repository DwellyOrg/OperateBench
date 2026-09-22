"""Executing episodes, and the causal acceptance contract over them.

:func:`run_episode` is the only place an episode is produced. It reproduces the
episode a second time and compares canonical digests, which is what fills the
``deterministic_replay`` dimension: a run that cannot reproduce itself has no
business claiming replayability of its artefact.

*How* it reproduces it is the part that had to change. Executing the agent again
is free for a deterministic in-process agent and is a second provider run for a
model one — and a stochastic agent need not agree with itself, so that check
would cost twice and prove nothing. Turning it off was worse: the evaluator then
records ``REPLAY_NOT_CHECKED`` and a model episode can never be reliable.

So there are two self-check modes. :data:`SELF_CHECK_RERUN` executes the agent
again, and is what a deterministic agent still gets. :data:`SELF_CHECK_PLAYBACK`
re-executes the *environment* — real clock, real events, real domain semantics,
real evaluator — while taking the agent's decisions off the recorded tape, each
one bound to the digest of the observation that produced it. It reaches no
provider, and it is the stricter check of the two: a rerun asks whether the same
code produced the same digests, while playback asks whether the recorded
decisions, offered the states they were actually made from, rebuild the run.

An agent that declares itself ``single_use`` is refused in rerun mode by name
rather than being executed, and paid for, twice.

The identity a run carries is minted inside :func:`run_episode` and cannot be
supplied to it. The public run surface is provider-capable, and an identity a
caller chooses is one two runs can share — which is what the opaque per-run
identity exists to prevent, and which no amount of *validating* a supplied value
would prevent, because a scenario-derived string can be perfectly well-formed.
Reproducing a persisted record is the one case that legitimately needs an
existing identity, and it has its own surfaces: the private
:func:`_reproduce_episode`, which the artefact reader uses, and
:func:`playback_episode`, which can hold nothing but a recorded tape. Both
convert what they are given at their boundary, and so does :func:`_execute`,
before any agent begins.

:func:`check_maintenance` is the CI gate. It is stricter than "the reference
passes and the negatives fail", because that is satisfiable by an evaluator that
rejects everything except one hard-coded trajectory. For every negative it reads
the expectation out of the **separate oracle manifest** — never out of the
agent registry — and requires that the set of failed dimensions is *exactly* the
oracle-declared causal failure closure, that the required finding codes are
present, and that every dimension the manifest lists as must-pass actually
passed. An agent that fails the right dimension for the wrong reason, takes down
a dimension nobody registered, or is not oracle-declared at all, breaks the gate.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Protocol

from operatebench.agents.playback import (
    DecisionTape,
    RecordedOutcomeAgent,
    RecordingAgent,
)
from operatebench.agents.transport import (
    SOURCE_DETERMINISTIC,
    SOURCE_RECORDED,
    ExecutionRecord,
    ProviderFailure,
    SingleUseAgentError,
    in_process_execution,
)
from operatebench.core.engine import (
    Engine,
    EpisodeOutcome,
    has_canonical_episode_outcome_provenance,
)
from operatebench.core.errors import OperateBenchError
from operatebench.core.evaluation import OperationEvaluation
from operatebench.core.instance import (
    new_operation_instance_id,
    require_operation_instance_id,
)
from operatebench.domains.lettings.maintenance.agents import (
    NEGATIVE_AGENTS,
    REFERENCE_AGENT,
    AgentEntry,
    agent_entry,
    build_agent,
)
from operatebench.domains.lettings.maintenance.evaluator import _evaluate_fresh
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.oracle import (
    NegativeControlOracle,
    negative_control_oracle,
)
from operatebench.domains.lettings.maintenance.reference_faults import (
    V2_PERMANENT_NOTIFICATION_FAULT,
)
from operatebench.domains.lettings.maintenance.spec import (
    AuthoredEvent,
    ConditionalTrigger,
    ExpectedEventRejection,
    OperationSpec,
    ScenarioSpec,
)
from operatebench.partial_evidence import PartialExecutionEvidence
from operatebench.version import OPERATEBENCH_VERSION


class EvidenceRecorder(Protocol):
    """What :func:`run_episode` needs of a provider evidence recorder.

    Structural, and four methods wide, so this module composes one without
    importing it: a recorder holds an SDK client's wire capture and a pricing
    policy, and a runner that imported those would drag a provider dependency
    into every deterministic run in this build.

    The implementation is
    :class:`~operatebench.agents.evidence.ProviderEvidenceRecorder`.
    """

    def begin(self, *, operation_instance_id: str) -> Any: ...

    def finalize_scored(self) -> None: ...

    def finalize_excluded(self, exclusion_code: str) -> None: ...

    def finalize_aborted(self) -> None: ...

    def provider_execution_binding(self) -> Mapping[str, Any] | None: ...


#: Execute the agent a second time and compare digests. What a deterministic
#: in-process agent gets, and what a single-use one is refused.
SELF_CHECK_RERUN = "rerun"

#: Re-execute the environment from the recorded decision tape. No agent is run
#: and no provider is reached; see :mod:`operatebench.agents.playback`.
SELF_CHECK_PLAYBACK = "playback"

SELF_CHECK_MODES: tuple[str, ...] = (SELF_CHECK_RERUN, SELF_CHECK_PLAYBACK)

#: A callable that builds the agent for one execution.
#:
#: Typed at ``Any`` rather than at :class:`OperationAgent` deliberately. A model
#: agent may return a *classification* instead of one of the five outcomes, which
#: is wider than the protocol — and correctly so: the engine's public boundary is
#: the thing that refuses it, by name and without mutating anything. Narrowing
#: the type here would mean the only way to hand the engine a malformed decision
#: is to lie to the type checker, and then the refusal path would never be
#: exercised by anything that resembles a real agent.
AgentFactory = Callable[[], Any]


class ReproductionAgentError(OperateBenchError):
    """A reproduction was offered an agent that is not a replay of the record.

    Narrow on purpose. It is not a claim that a caller reaching a private
    function is untrusted — Python has no such boundary and pretending otherwise
    would be theatre. It is the structural statement that :func:`_reproduce_episode`
    only has two shapes that mean anything: rerun the registry agent the record
    names, or replay the record's own decisions. An agent that would obtain new
    decisions produces a *different* run under the recorded run's identity, and
    the resulting comparison would report agreement or divergence about two
    things that were never the same run.
    """


def _recorded_agent(
    agent_factory: AgentFactory, operation_instance_id: str
) -> AgentFactory:
    """One validated recorded agent, returned by a one-shot factory.

    The caller has already validated ``operation_instance_id``. It is named in
    the refusal only as an operation instance, never quoted: a persisted value is
    provenance, not diagnostic text supplied by a provider.
    """
    agent = agent_factory()
    if not isinstance(agent, RecordedOutcomeAgent):
        raise ReproductionAgentError(
            "episode reproduction under a persisted operation instance accepts "
            "only RecordedOutcomeAgent or RecordedModelAgent; an agent that can "
            "obtain new decisions would create a different run under the recorded "
            "run's identity"
        )
    return lambda: agent


@dataclass(frozen=True)
class EpisodeRun:
    """One executed episode, its evaluation, and the identity of both.

    ``tape`` and ``execution`` are what make the episode reproducible without
    re-executing the agent: the decisions it made, each bound to the observation
    that produced it, and the record of how those decisions were produced.
    """

    spec: OperationSpec
    scenario_id: str
    agent_id: str
    outcome: EpisodeOutcome
    evaluation: OperationEvaluation
    #: This run, named without naming the case it ran. Minted once per run and
    #: reused by the self-check, because every recorded decision is bound to the
    #: digest of an observation that carries it.
    operation_instance_id: str = ""
    tape: DecisionTape = field(default_factory=DecisionTape)
    execution: ExecutionRecord = field(default_factory=in_process_execution)
    agent_kind: str | None = None
    #: The binding to the execution ledger that witnessed this run's provider
    #: session, or ``None`` for a run that reached no provider.
    #:
    #: It is attached by :func:`run_episode` *after* the recorder has finalised
    #: a scored journal and never before, and there is deliberately no path that
    #: builds one from the run alone. A binding minted from what a run believes
    #: about itself would be a claim about a provider with nothing durable
    #: behind it, which is precisely what contract 7 exists to refuse.
    provider_execution: Mapping[str, Any] | None = None

    @property
    def reliable(self) -> bool:
        return self.evaluation.reliable

    def identity(self) -> dict[str, Any]:
        scenario = self.spec.scenario(self.scenario_id)
        return {
            "engine_version": OPERATEBENCH_VERSION,
            "operation": self.spec.identity_payload(),
            "operation_instance_id": self.operation_instance_id,
            "scenario_id": self.scenario_id,
            "semantic_scenario_id": scenario.semantic_scenario_id,
            "expected_terminal": scenario.expected_terminal,
            "agent_id": self.agent_id,
            "agent_kind": self.resolved_agent_kind(),
        }

    def resolved_agent_kind(self) -> str:
        """The agent's kind, from the registry unless the run states its own.

        An injected agent is not in the shipped registry and must not be looked
        up there: an unregistered id would raise, and borrowing a registered
        one's kind would attribute a model run to a deterministic control.
        """
        if self.agent_kind is not None:
            return self.agent_kind
        return agent_entry(self.agent_id).kind

    def summary(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "status": self.outcome.status,
            "reliable": self.reliable,
            "failed_dimensions": list(self.evaluation.failed_dimensions),
            "finding_codes": sorted(set(self.evaluation.finding_codes)),
            "simulated_minutes": self.outcome.simulated_minutes,
            "agent_invocations": self.outcome.invocations,
            "final_state_digest_sha256": self.outcome.final_state_digest_sha256,
            "trajectory_digest_sha256": self.outcome.trajectory_digest_sha256,
        }


@dataclass(frozen=True)
class ExecutedEpisode:
    """One execution: what happened, what was decided, and how it was decided."""

    outcome: EpisodeOutcome
    tape: DecisionTape
    execution: ExecutionRecord
    single_use: bool


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _FreshEvaluationInput:
    """Private bridge from one fresh execution to maintenance grading."""

    outcome: EpisodeOutcome
    replay_ok: bool | None
    scenario: ScenarioSpec
    _executed: ExecutedEpisode = field(repr=False, compare=False)
    _snapshot: _FreshOutcomeSnapshot = field(repr=False, compare=False)

    def __copy__(self) -> Any:
        raise TypeError("fresh evaluation authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        raise TypeError("fresh evaluation authority cannot be deep-copied")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("fresh evaluation authority cannot be pickled")


@dataclass(frozen=True, slots=True)
class _FreshOutcomeSnapshot:
    """Detached evaluator-consumed projection of an authenticated outcome."""

    status: str
    replay_final: bool
    simulated_minutes: int
    invocations: int
    final_state: Mapping[str, Any]
    trajectory: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]


class _FreshFingerprintRefusal(ValueError):
    """A mint value has no bounded private fingerprint representation."""


class _FreshFingerprintState:
    """Bound work before traversing values held across the grading boundary."""

    __slots__ = ("active", "remaining_bytes", "remaining_nodes")

    def __init__(self) -> None:
        self.active: set[int] = set()
        self.remaining_bytes = 8 * 1024 * 1024
        self.remaining_nodes = 100_000

    def take_node(self) -> None:
        self.remaining_nodes -= 1
        if self.remaining_nodes < 0:
            raise _FreshFingerprintRefusal("fresh evaluation fingerprint is oversized")

    def take_bytes(self, size: int) -> None:
        self.remaining_bytes -= size
        if self.remaining_bytes < 0:
            raise _FreshFingerprintRefusal("fresh evaluation fingerprint is oversized")


def _fingerprint_frame(tag: bytes, parts: Sequence[bytes]) -> bytes:
    return tag + b"".join(len(part).to_bytes(8, "big") + part for part in parts)


def _fresh_value_fingerprint(value: Any, state: _FreshFingerprintState) -> bytes:
    state.take_node()
    value_type = type(value)
    if value is None:
        state.take_bytes(1)
        return b"n"
    if value_type is bool:
        state.take_bytes(2)
        return b"b1" if value else b"b0"
    if value_type is int:
        assert isinstance(value, int)
        if value.bit_length() > 16_384:
            raise _FreshFingerprintRefusal("fresh evaluation fingerprint is oversized")
        encoded = str(value).encode("ascii")
        state.take_bytes(len(encoded) + 1)
        return b"i" + encoded
    if value_type is float:
        assert isinstance(value, float)
        encoded = value.hex().encode("ascii")
        state.take_bytes(len(encoded) + 1)
        return b"f" + encoded
    if value_type is str:
        assert isinstance(value, str)
        encoded = value.encode()
        state.take_bytes(len(encoded) + 1)
        return b"s" + encoded
    if value_type is bytes:
        assert isinstance(value, bytes)
        state.take_bytes(len(value) + 1)
        return b"y" + value

    allowed_dataclasses = (
        _FreshOutcomeSnapshot,
        ScenarioSpec,
        AuthoredEvent,
        ConditionalTrigger,
        ExpectedEventRejection,
    )
    is_supported_dataclass = value_type in allowed_dataclasses
    is_supported_container = value_type in (
        dict,
        MappingProxyType,
        list,
        tuple,
        set,
        frozenset,
    )
    if not is_supported_dataclass and not is_supported_container:
        raise _FreshFingerprintRefusal("unsupported fresh evaluation fingerprint value")

    identity = id(value)
    if identity in state.active:
        raise _FreshFingerprintRefusal("cyclic fresh evaluation fingerprint value")
    if len(state.active) >= 64:
        raise _FreshFingerprintRefusal("fresh evaluation fingerprint is too deep")
    state.active.add(identity)
    try:
        if is_supported_dataclass:
            class_name = f"{value_type.__module__}.{value_type.__qualname__}".encode()
            dataclass_fields = fields(value)
            field_literals = tuple(
                (item.name.encode("utf-8"), str(item.type).encode("utf-8"))
                for item in dataclass_fields
            )
            state.take_bytes(
                1
                + 8 * (1 + 3 * len(dataclass_fields))
                + len(class_name)
                + sum(
                    len(field_name) + len(field_type)
                    for field_name, field_type in field_literals
                )
            )
            parts = [class_name]
            for item, (field_name, field_type) in zip(
                dataclass_fields, field_literals, strict=True
            ):
                parts.extend(
                    (
                        field_name,
                        field_type,
                        _fresh_value_fingerprint(getattr(value, item.name), state),
                    )
                )
            encoded = _fingerprint_frame(b"d", parts)
        elif value_type in (dict, MappingProxyType):
            state.take_bytes(1 + 25 * len(value))
            pairs = sorted(
                _fingerprint_frame(
                    b"p",
                    (
                        _fresh_value_fingerprint(key, state),
                        _fresh_value_fingerprint(item, state),
                    ),
                )
                for key, item in value.items()
            )
            encoded = _fingerprint_frame(b"m" if value_type is dict else b"r", pairs)
        elif value_type in (list, tuple):
            state.take_bytes(1 + 8 * len(value))
            encoded = _fingerprint_frame(
                b"l" if value_type is list else b"t",
                tuple(_fresh_value_fingerprint(item, state) for item in value),
            )
        else:
            state.take_bytes(1 + 8 * len(value))
            encoded = _fingerprint_frame(
                b"e" if value_type is set else b"z",
                sorted(_fresh_value_fingerprint(item, state) for item in value),
            )
    finally:
        state.active.remove(identity)
    return encoded


def _fresh_fingerprint(value: Any) -> bytes:
    return _fresh_value_fingerprint(value, _FreshFingerprintState())


@dataclass(frozen=True, slots=True)
class _FreshMintRecord:
    wrapper_ref: weakref.ReferenceType[_FreshEvaluationInput]
    creator_pid: int
    outcome_id: int
    executed_id: int
    snapshot_id: int
    scenario_id: int
    replay_ok: bool | None
    snapshot_fingerprint: bytes
    scenario_fingerprint: bytes


_FRESH_MINT_LOCK = threading.Lock()
_FRESH_MINTS: dict[int, _FreshMintRecord] = {}


def _retire_fresh_mint(
    wrapper_id: int, wrapper_ref: weakref.ReferenceType[_FreshEvaluationInput]
) -> None:
    with _FRESH_MINT_LOCK:
        record = _FRESH_MINTS.get(wrapper_id)
        if record is not None and record.wrapper_ref is wrapper_ref:
            del _FRESH_MINTS[wrapper_id]


def _register_fresh_evaluation(fresh: _FreshEvaluationInput) -> None:
    wrapper_id = id(fresh)
    wrapper_ref = weakref.ref(
        fresh, lambda reference: _retire_fresh_mint(wrapper_id, reference)
    )
    record = _FreshMintRecord(
        wrapper_ref=wrapper_ref,
        creator_pid=os.getpid(),
        outcome_id=id(fresh.outcome),
        executed_id=id(fresh._executed),
        snapshot_id=id(fresh._snapshot),
        scenario_id=id(fresh.scenario),
        replay_ok=fresh.replay_ok,
        snapshot_fingerprint=_fresh_fingerprint(fresh._snapshot),
        scenario_fingerprint=_fresh_fingerprint(fresh.scenario),
    )
    with _FRESH_MINT_LOCK:
        _FRESH_MINTS[wrapper_id] = record


def _fresh_record_matches(fresh: _FreshEvaluationInput, record: _FreshMintRecord) -> bool:
    if (
        record.wrapper_ref() is not fresh
        or record.creator_pid != os.getpid()
        or id(fresh.outcome) != record.outcome_id
        or id(fresh._executed) != record.executed_id
        or id(fresh._snapshot) != record.snapshot_id
        or id(fresh.scenario) != record.scenario_id
        or fresh.replay_ok is not record.replay_ok
    ):
        return False
    try:
        return (
            _fresh_fingerprint(fresh._snapshot) == record.snapshot_fingerprint
            and _fresh_fingerprint(fresh.scenario) == record.scenario_fingerprint
        )
    except _FreshFingerprintRefusal:
        return False


def _freeze_evaluation_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _freeze_evaluation_value(key): _freeze_evaluation_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_evaluation_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_evaluation_value(item) for item in value)
    return value


def _freeze_evaluation_scenario(scenario: ScenarioSpec) -> ScenarioSpec:
    events = tuple(
        replace(
            event,
            payload=_freeze_evaluation_value(event.payload),
            trigger=None if event.trigger is None else replace(event.trigger),
        )
        for event in scenario.events
    )
    return replace(
        scenario,
        required_checkpoint_types=tuple(scenario.required_checkpoint_types),
        dispatch_failures=frozenset(scenario.dispatch_failures),
        expected_event_rejections=tuple(
            replace(expected) for expected in scenario.expected_event_rejections
        ),
        events=events,
    )


def _admit_fresh_evaluation(
    fresh: object,
) -> tuple[_FreshOutcomeSnapshot, ScenarioSpec, bool | None]:
    if (
        not isinstance(fresh, _FreshEvaluationInput)
        or type(fresh) is not _FreshEvaluationInput
    ):
        raise ValueError("fresh evaluation authority refused unminted or changed input")
    with _FRESH_MINT_LOCK:
        record = _FRESH_MINTS.get(id(fresh))
        if (
            record is None
            or record.wrapper_ref() is not fresh
            or record.creator_pid != os.getpid()
        ):
            raise ValueError(
                "fresh evaluation authority refused unminted or changed input"
            )
    admitted = _fresh_record_matches(
        fresh, record
    ) and has_canonical_episode_outcome_provenance(fresh.outcome)
    with _FRESH_MINT_LOCK:
        admitted = (
            admitted
            and _FRESH_MINTS.get(id(fresh)) is record
            and record.wrapper_ref() is fresh
            and record.creator_pid == os.getpid()
        )
    if not admitted:
        raise ValueError("fresh evaluation authority refused unminted or changed input")
    return fresh._snapshot, fresh.scenario, fresh.replay_ok


def _execute(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    agent_factory: AgentFactory | None,
    operation_instance_id: str,
    partial_evidence: PartialExecutionEvidence | None = None,
) -> ExecutedEpisode:
    """Run one episode once, recording every decision as it is made.

    The first thing that happens is not execution: the spec re-derives its own
    identity and refuses if what it now holds no longer hashes to the digest it
    claims. A stale digest would otherwise bind changed semantics — the run
    artefact records ``spec_digest_sha256`` as the provenance of everything in
    it, so an episode that runs under one must be running the operation that
    digest describes.

    The agent is wrapped in a :class:`~operatebench.agents.playback.RecordingAgent`
    unconditionally. The wrapper forwards observation and outcome unchanged, so
    a recorded episode is byte-for-byte the episode that would have run without
    it, and every run — deterministic or not — comes out with a tape.

    ``operation_instance_id`` is converted here, before anything is built and
    long before ``begin_episode``, and this is the engine boundary rather than a
    convenience for public callers: every episode this build executes passes
    through it, so an identity that is not one this build writes is a named
    refusal instead of a value an observation carries to an agent.
    """
    instance_id = require_operation_instance_id(
        operation_instance_id, "operation_instance_id at the episode engine boundary"
    )
    spec.verify_identity()
    scenario = spec.scenario(scenario_id)
    domain = MaintenanceOperation(spec, scenario_id)
    agent = build_agent(agent_id) if agent_factory is None else agent_factory()
    recorder = RecordingAgent(
        agent, observer=None if partial_evidence is None else partial_evidence.decision
    )
    engine = Engine(
        domain,
        recorder,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": instance_id,
            "scenario_id": scenario_id,
            "agent_id": agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
        partial_evidence=partial_evidence,
    )
    try:
        outcome = engine.run()
    except ProviderFailure:
        if partial_evidence is not None:
            partial_evidence.state(
                domain.canonical_state(engine.state), at=engine.clock.now
            )
        raise
    if isinstance(agent, RecordedOutcomeAgent):
        # An episode that ended with recorded decisions left over is not the
        # episode the tape describes, however well the digests happen to agree.
        agent.check_exhausted()
    record = getattr(agent, "execution_record", None)
    source = str(getattr(agent, "outcome_source", SOURCE_DETERMINISTIC))
    execution = record() if callable(record) else in_process_execution(source)
    return ExecutedEpisode(
        outcome=outcome,
        tape=recorder.tape(),
        execution=execution,
        single_use=recorder.single_use,
    )


def execute_episode(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    *,
    agent_factory: AgentFactory | None = None,
) -> EpisodeOutcome:
    """Run one episode once. Deterministic in, deterministic out.

    Provider-capable — ``agent_factory`` may build a model agent — so the
    instance identity is minted here and there is no parameter that would let a
    caller name one. See :func:`run_episode` for why that is a refusal by
    construction rather than a validated argument.
    """
    return _execute(
        spec,
        scenario_id,
        agent_id,
        agent_factory,
        new_operation_instance_id(),
    ).outcome


def playback_episode(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    tape: DecisionTape,
    *,
    operation_instance_id: str,
) -> ExecutedEpisode:
    """Re-execute one episode from recorded decisions, reaching no provider.

    Everything except the agent is real: the clock, the event queue, the domain
    validators, the ledger. Only the decisions are recorded, and each is refused
    unless the observation the episode offers is the one it was made from.

    ``operation_instance_id`` is required rather than minted here, and that is
    the whole reason it is a parameter. Every recorded decision is bound to the
    digest of the observation that produced it, and that observation carries the
    instance identity; replaying under a fresh one would offer each decision an
    observation it was not produced from, and playback would refuse the run it is
    supposed to be reproducing.

    That it may take a persisted identity at all rests on there being no agent
    parameter: this function builds a
    :class:`~operatebench.agents.playback.RecordedOutcomeAgent` over the tape it
    was handed and nothing else, so no caller can point it at a provider. The
    identity is converted at this boundary, before the engine is built and
    before any agent begins.
    """
    return _execute(
        spec,
        scenario_id,
        agent_id,
        lambda: RecordedOutcomeAgent(tape, agent_id=agent_id),
        require_operation_instance_id(
            operation_instance_id, "operation_instance_id offered to playback_episode"
        ),
    )


def _reproduces(first: EpisodeOutcome, second: EpisodeOutcome) -> bool:
    return (
        second.final_state_digest_sha256 == first.final_state_digest_sha256
        and second.trajectory_digest_sha256 == first.trajectory_digest_sha256
    )


def _reproduce_episode(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    *,
    operation_instance_id: str,
    self_check: bool = True,
    agent_factory: AgentFactory | None = None,
    self_check_mode: str | None = None,
    agent_kind: str | None = None,
) -> EpisodeRun:
    """Run and evaluate one episode under an identity that already exists.

    Private, and the only surface in this module that accepts one. It exists for
    *reproduction*: :mod:`operatebench.artifact` rerunning or replaying a
    persisted record has to execute under the identity that record carries,
    because every recorded decision is bound to the digest of an observation
    that names it. Nothing else may reach it — a caller who could choose the
    identity of a fresh run could give two runs, on two scenarios, the same one,
    and hand the result to a provider-backed agent.

    That is why this surface takes an identity and :func:`run_episode` does not,
    and it is also why the two are separate functions rather than one with a
    default: both evaluate an episode through :func:`_run_and_evaluate`, and the
    only thing this one adds is what a reproduction is allowed to be.

    Two shapes, and no third. ``agent_factory=None`` is a *deterministic rerun*:
    the agent comes from the shipped registry, by the name the record carries,
    which is free to execute and reaches nothing. A supplied ``agent_factory``
    is a *replay*, and must build a
    :class:`~operatebench.agents.playback.RecordedOutcomeAgent` — which includes
    :class:`~operatebench.agents.playback.RecordedModelAgent`, whose extra work
    is rebuilding each request digest against a
    :class:`~operatebench.agents.transport.ForbiddenTransport`. Anything else is
    refused by name here, before the agent is begun and before anything it holds
    could be spent, because reproducing a record means re-executing *its*
    decisions: an agent that would ask a provider for new ones is producing a
    different run under the identity of the recorded one.

    The factory is called once, to see what it builds, and the instance it built
    is what runs — the check would otherwise be about an agent the episode never
    saw. The identity is converted before that call and again at
    :func:`_execute`'s boundary, so a malformed one is a named refusal before
    anything is built at all.
    """
    instance_id = require_operation_instance_id(
        operation_instance_id, "operation_instance_id offered for episode reproduction"
    )
    reproduction_factory = (
        None if agent_factory is None else _recorded_agent(agent_factory, instance_id)
    )
    return _run_and_evaluate(
        spec,
        scenario_id,
        agent_id,
        operation_instance_id=instance_id,
        self_check=self_check,
        agent_factory=reproduction_factory,
        self_check_mode=self_check_mode,
        agent_kind=agent_kind,
    )


def _run_and_evaluate(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    *,
    operation_instance_id: str,
    self_check: bool = True,
    agent_factory: AgentFactory | None = None,
    self_check_mode: str | None = None,
    agent_kind: str | None = None,
    partial_evidence: PartialExecutionEvidence | None = None,
) -> EpisodeRun:
    """Execute one episode under a given identity, self-check it, and grade it.

    The trusted primitive both surfaces above are built on, and it holds no
    policy about where the identity came from or what the agent may be: those
    are exactly what distinguishes :func:`run_episode` (mints its own identity,
    provider-capable) from :func:`_reproduce_episode` (takes a persisted one,
    recorded agents only). Keeping the shared execution here means neither
    surface can acquire the other's permissions by delegating to it.

    The identity is converted here as well as at :func:`_execute`'s boundary, so
    it is refused by name before anything is executed rather than at whichever
    execution happens to run first.
    """
    instance_id = require_operation_instance_id(
        operation_instance_id, "operation_instance_id offered for episode execution"
    )
    fresh = _execute_and_self_check(
        spec,
        scenario_id,
        agent_id,
        operation_instance_id=instance_id,
        self_check=self_check,
        agent_factory=agent_factory,
        self_check_mode=self_check_mode,
        partial_evidence=partial_evidence,
    )
    executed = fresh._executed
    outcome = fresh.outcome
    evaluation = _evaluate_fresh(fresh)
    return EpisodeRun(
        spec=spec,
        scenario_id=scenario_id,
        agent_id=agent_id,
        outcome=outcome,
        evaluation=evaluation,
        operation_instance_id=instance_id,
        tape=executed.tape,
        execution=executed.execution,
        agent_kind=agent_kind,
    )


def _execute_and_self_check(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    *,
    operation_instance_id: str,
    self_check: bool = True,
    agent_factory: AgentFactory | None = None,
    self_check_mode: str | None = None,
    partial_evidence: PartialExecutionEvidence | None = None,
) -> _FreshEvaluationInput:
    """Execute and self-check without importing grading into the runtime step."""
    instance_id = require_operation_instance_id(
        operation_instance_id, "operation_instance_id offered for episode execution"
    )
    executed = _execute(
        spec, scenario_id, agent_id, agent_factory, instance_id, partial_evidence
    )
    outcome = executed.outcome
    mode = self_check_mode
    if mode is None:
        mode = SELF_CHECK_PLAYBACK if executed.single_use else SELF_CHECK_RERUN
    if mode not in SELF_CHECK_MODES:
        raise SingleUseAgentError(
            f"{mode!r} is not a determinism self-check this build performs; the "
            f"modes are {list(SELF_CHECK_MODES)}"
        )

    replay_ok: bool | None = None
    if self_check:
        if mode == SELF_CHECK_RERUN:
            if executed.single_use:
                raise SingleUseAgentError(
                    f"agent {agent_id!r} is single-use, so re-executing it to check "
                    "determinism would be a second provider run whose answers need "
                    f"not match the first; use self_check_mode="
                    f"{SELF_CHECK_PLAYBACK!r}, which reproduces the episode from the "
                    "recorded decisions and reaches no provider"
                )
            repeat = _execute(
                spec, scenario_id, agent_id, agent_factory, instance_id
            ).outcome
        else:
            repeat = playback_episode(
                spec,
                scenario_id,
                agent_id,
                executed.tape,
                operation_instance_id=instance_id,
            ).outcome
        replay_ok = _reproduces(outcome, repeat)

    if not has_canonical_episode_outcome_provenance(outcome):
        raise ValueError("fresh evaluation authority requires an Engine-minted outcome")
    snapshot = _FreshOutcomeSnapshot(
        status=outcome.status,
        replay_final=outcome.replay_final,
        simulated_minutes=outcome.simulated_minutes,
        invocations=outcome.invocations,
        final_state=_freeze_evaluation_value(outcome.final_state),
        trajectory=_freeze_evaluation_value(outcome.trajectory),
        events=_freeze_evaluation_value(outcome.events),
    )
    fresh = _FreshEvaluationInput(
        outcome,
        replay_ok,
        _freeze_evaluation_scenario(spec.scenario(scenario_id)),
        executed,
        snapshot,
    )
    _register_fresh_evaluation(fresh)
    return fresh


def run_episode(
    spec: OperationSpec,
    scenario_id: str,
    agent_id: str,
    *,
    self_check: bool = True,
    agent_factory: AgentFactory | None = None,
    self_check_mode: str | None = None,
    agent_kind: str | None = None,
    evidence_recorder: EvidenceRecorder | None = None,
) -> EpisodeRun:
    """Execute and evaluate one episode, including the determinism self-check.

    ``self_check_mode`` defaults to the mode the agent can actually honour:
    playback for a single-use agent, rerun for anything else. Passing it
    explicitly forces the choice, and forcing a rerun on a single-use agent is
    refused rather than silently downgraded — a caller who asked for a second
    provider run should be told that is what they asked for.

    The operation instance identity is minted here, once, and used by both the
    execution and the self-check. There is deliberately no parameter for it.
    This function is provider-capable — ``agent_factory`` may build a
    :class:`~operatebench.agents.model.ModelAgent`, and that agent is handed the
    identity inside every observation — so an identity a caller could choose is
    an identity two runs could share, and "this run, and nothing about which run
    it is" would become a label a caller controls. Validating a supplied one
    would not close that: a scenario-derived string can be perfectly well-formed
    and still be the same value on a V1 run and a V2 one. What closes it is that
    the value is freshly random, minted inside this call, and impossible to name
    from outside. Reproducing a *persisted* run is a different operation with a
    different surface: :func:`_reproduce_episode` for a record with its own tape
    and identity, :func:`playback_episode` for a tape alone.

    ``evidence_recorder`` is optional, and when it is absent this function is
    exactly what it was before the parameter existed: a deterministic in-process
    run records no provider evidence because it made no provider call.

    When one is supplied, the order here is the contract. The instance identity
    is minted *first* and the recorder is begun with it, so the journal's header
    names the run before its first request leaves. The recorder is then
    finalised on every exit: ``scored`` for a run that reached an evaluated
    episode, ``excluded`` for a
    :class:`~operatebench.agents.transport.ProviderFailure` — which still writes
    no episode artefact, exactly as before — and ``aborted`` for anything else,
    while the process is still alive to write it. A process that dies outright
    writes nothing further, and the journal's verified prefix is what survives.
    """
    instance_id = new_operation_instance_id()
    partial = PartialExecutionEvidence(
        {
            "operation_instance_id": instance_id,
            "operation_id": spec.operation_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
            "scenario_id": scenario_id,
            "agent_id": agent_id,
        }
    )
    if evidence_recorder is not None:
        evidence_recorder.begin(operation_instance_id=instance_id)
        durable_partial = getattr(evidence_recorder, "partial_evidence", None)
        if isinstance(durable_partial, PartialExecutionEvidence):
            partial = durable_partial
    try:
        run = _run_and_evaluate(
            spec,
            scenario_id,
            agent_id,
            operation_instance_id=instance_id,
            self_check=self_check,
            agent_factory=agent_factory,
            self_check_mode=self_check_mode,
            agent_kind=agent_kind,
            partial_evidence=partial,
        )
    except ProviderFailure as failure:
        failure.partial_evidence = partial
        if evidence_recorder is not None:
            evidence_recorder.finalize_excluded(failure.fault)
        partial.finish("excluded", failure.fault)
        raise
    except BaseException:
        if evidence_recorder is not None:
            evidence_recorder.finalize_aborted()
        partial.finish("aborted")
        raise
    if evidence_recorder is None:
        partial.finish("scored")
        return run
    evidence_recorder.finalize_scored()
    partial.finish("scored")
    # Only now. The binding states the digest of a *complete* journal, so it
    # cannot exist until the terminal row is durable — and asking for it before
    # this line would either produce a digest over a prefix or force the
    # recorder to predict the row it has not written yet. A recorder that
    # finalised something other than a scored run returns ``None`` and the run
    # carries no binding, which is what makes "a model artefact without a scored
    # ledger is refused" a fact about the runner rather than about the writer.
    return replace(run, provider_execution=evidence_recorder.provider_execution_binding())


@dataclass(frozen=True)
class AgentCheck:
    """What one (agent, scenario) pair was required to do, and whether it did.

    For a negative, ``expected_targets``, ``expected_findings`` and
    ``required_passing`` are copied out of the oracle manifest that graded the
    run, so a report says which expectation it was held to rather than only
    whether it met one.
    """

    agent_id: str
    scenario_id: str
    kind: str
    reliable: bool
    failed_dimensions: tuple[str, ...]
    passed_dimensions: tuple[str, ...]
    finding_codes: tuple[str, ...]
    expected_targets: tuple[str, ...]
    expected_findings: tuple[str, ...]
    required_passing: tuple[str, ...]
    problems: tuple[str, ...]
    reference_expectation_id: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "ok": self.ok,
            "reliable": self.reliable,
            "failed_dimensions": list(self.failed_dimensions),
            "passed_dimensions": list(self.passed_dimensions),
            "finding_codes": list(self.finding_codes),
            "expected_targets": list(self.expected_targets),
            "expected_findings": list(self.expected_findings),
            "required_passing": list(self.required_passing),
            "problems": list(self.problems),
            "reference_expectation_id": self.reference_expectation_id,
        }


@dataclass(frozen=True)
class CheckReport:
    """The causal acceptance contract over the whole agent set.

    The oracle's identity travels with the verdict. A report that said "OK"
    without naming the manifest it was graded against would be unfalsifiable
    the moment the expectations changed.
    """

    operation_id: str
    spec_digest_sha256: str
    oracle_id: str
    oracle_version: str
    oracle_digest_sha256: str
    checks: tuple[AgentCheck, ...]

    @property
    def ok(self) -> bool:
        # Non-empty first, because ``all()`` over nothing is True. A report that
        # ran no reference and no negative control is not an acceptance, it is the
        # absence of one, and anything reading this field would have no way to
        # tell a 0/0 report apart from a gate that actually passed.
        return bool(self.checks) and all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "spec_digest_sha256": self.spec_digest_sha256,
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "oracle_digest_sha256": self.oracle_digest_sha256,
            "ok": self.ok,
            "checks": [check.as_dict() for check in self.checks],
        }


def check_agent(
    spec: OperationSpec,
    entry: AgentEntry,
    scenario_id: str,
    *,
    oracle: NegativeControlOracle | None = None,
) -> AgentCheck:
    """Run one agent on one scenario and hold it to the oracle-declared contract.

    V1/V3 references must be reliable. V2 separately checks the authored permanent
    notification fault, including an exact terminal, failures and survivors;
    passing this diagnostic does not make the V2 episode reliable.

    ``oracle`` defaults to the shipped manifest. It is a parameter so the suite
    can hold the *gate* to account — point a deliberately wrong expectation at a
    real agent and the gate has to notice — not so a caller can soften one.
    """
    manifest = negative_control_oracle() if oracle is None else oracle
    run = run_episode(spec, scenario_id, entry.agent_id)
    evaluation = run.evaluation
    failed = evaluation.failed_dimensions
    passed = evaluation.passed_dimensions
    codes = tuple(sorted(set(evaluation.finding_codes)))
    problems: list[str] = []
    expected_targets: tuple[str, ...] = ()
    expected_findings: tuple[str, ...] = ()
    required_passing: tuple[str, ...] = ()

    reference_expectation_id = None
    if entry.kind == "reference":
        if (
            entry.agent_id == "reference"
            and spec.operation_id == "lettings_maintenance_synthetic_v1"
            and scenario_id == "V2"
            and spec.scenario(scenario_id).dispatch_failures
            == frozenset({"msg_transfer_notice"})
        ):
            expectation = V2_PERMANENT_NOTIFICATION_FAULT
            reference_expectation_id = expectation.expectation_id
            expected_targets = expectation.failed_dimensions
            expected_findings = expectation.finding_codes
            required_passing = expectation.passing_dimensions
            if (
                evaluation.reliable
                or not evaluation.legitimate_completion
                or run.outcome.status != expectation.terminal
                or set(failed) != set(expected_targets)
                or set(codes) != set(expected_findings)
                or set(passed) != set(required_passing)
            ):
                problems.append(
                    "reference fault-control outcome differs from authored expectation "
                    f"{reference_expectation_id}: terminal={run.outcome.status!r}, "
                    f"failed={list(failed)}, findings={list(codes)}, "
                    f"passed={list(passed)}"
                )
        elif not evaluation.reliable:
            problems.append(
                f"the reference agent must be reliable on {scenario_id}; it failed "
                f"{list(failed)} with findings {list(codes)} and ended "
                f"{run.outcome.status!r}"
            )
    elif not manifest.has_control(entry.agent_id):
        # Not "unconstrained": ungraded. A negative with nothing oracle-declared
        # cannot be evidence that the evaluator is causal, so it fails here
        # rather than passing by default.
        problems.append(
            f"negative agent {entry.agent_id!r} has no oracle-declared control in "
            f"oracle {manifest.oracle_id!r}; a negative with no declared "
            "expectation is ungraded, not unconstrained"
        )
    else:
        control = manifest.control(entry.agent_id)
        expected_targets = control.expected_failed_dimensions
        expected_findings = control.required_finding_codes
        required_passing = control.must_pass_dimensions

        if control.scenario_id != scenario_id:
            problems.append(
                f"negative agent {entry.agent_id!r} is oracle-declared against "
                f"scenario {control.scenario_id!r} and was run on {scenario_id!r}; "
                "an expectation authored for one scenario says nothing about another"
            )
        if evaluation.reliable:
            problems.append(
                f"negative agent {entry.agent_id!r} was accepted as reliable on "
                f"{scenario_id}"
            )
        if set(failed) != set(expected_targets):
            # Both directions matter. A dimension outside the closure is an
            # unrelated failure nobody oracle-declared; a declared one that
            # survived means the closure describes a run that did not happen.
            outside = sorted(set(failed) - set(expected_targets))
            unreached = sorted(set(expected_targets) - set(failed))
            problems.append(
                f"failed dimensions {sorted(failed)} are not the oracle-declared "
                f"causal failure closure {sorted(expected_targets)}: outside the "
                f"oracle-declared closure {outside}, oracle-declared but did not fail "
                f"{unreached}"
            )
        missing = sorted(set(expected_findings) - set(codes))
        if missing:
            problems.append(
                f"the intended reason is missing: expected finding code(s) {missing}, "
                f"got {list(codes)}"
            )
        lost = sorted(set(required_passing) - set(passed))
        if lost:
            problems.append(
                f"dimensions {lost} must remain passing for this control and did "
                "not; a negative that takes unrelated dimensions with it is a "
                "generic crash rather than a targeted failure"
            )

    return AgentCheck(
        agent_id=entry.agent_id,
        scenario_id=scenario_id,
        kind=entry.kind,
        reliable=evaluation.reliable,
        failed_dimensions=failed,
        passed_dimensions=passed,
        finding_codes=codes,
        expected_targets=expected_targets,
        expected_findings=expected_findings,
        required_passing=required_passing,
        problems=tuple(problems),
        reference_expectation_id=reference_expectation_id,
    )


def _scenarios_for(
    spec: OperationSpec, entry: AgentEntry, manifest: NegativeControlOracle
) -> tuple[str, ...]:
    """Which scenarios one entry is checked on.

    A negative is checked on the scenario its oracle control names, so the
    registry cannot move a control off the scenario it was authored against. An
    agent the oracle does not know still runs — on whatever the registry says —
    because the resulting check has to *fail*, and a check that never ran cannot.
    """
    if entry.kind != "reference" and manifest.has_control(entry.agent_id):
        return (manifest.control(entry.agent_id).scenario_id,)
    return tuple(entry.scenarios or spec.scenario_ids)


def check_maintenance(
    spec: OperationSpec,
    *,
    agent_ids: Sequence[str] | None = None,
    oracle: NegativeControlOracle | None = None,
) -> CheckReport:
    """Run the reference across every required scenario and every targeted negative.

    An empty ``agent_ids`` selects the whole agent set, exactly as ``None`` does,
    which is the reading the other operation packs give a falsy selection. The
    alternative is worse than inconsistent: a selection that named nothing would
    run no reference and no negative control, and the 0/0 report it produced
    would carry a verdict a caller could publish as causal acceptance.
    """
    manifest = negative_control_oracle() if oracle is None else oracle
    entries: Sequence[AgentEntry]
    if not agent_ids:
        entries = (REFERENCE_AGENT, *NEGATIVE_AGENTS)
    else:
        entries = [agent_entry(agent_id) for agent_id in agent_ids]
    checks: list[AgentCheck] = []
    for entry in entries:
        for scenario_id in _scenarios_for(spec, entry, manifest):
            checks.append(check_agent(spec, entry, scenario_id, oracle=manifest))
    return CheckReport(
        operation_id=spec.operation_id,
        spec_digest_sha256=spec.spec_digest_sha256,
        oracle_id=manifest.oracle_id,
        oracle_version=manifest.oracle_version,
        oracle_digest_sha256=manifest.oracle_digest_sha256,
        checks=tuple(checks),
    )


__all__ = [
    "SELF_CHECK_MODES",
    "SELF_CHECK_PLAYBACK",
    "SELF_CHECK_RERUN",
    "SOURCE_RECORDED",
    "AgentCheck",
    "AgentFactory",
    "CheckReport",
    "EpisodeRun",
    "EvidenceRecorder",
    "ExecutedEpisode",
    "check_agent",
    "check_maintenance",
    "execute_episode",
    "playback_episode",
    "run_episode",
]
