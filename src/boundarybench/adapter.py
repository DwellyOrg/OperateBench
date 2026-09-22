"""The provider-neutral adapter boundary.

Everything a model implementation may see about an episode passes through
:class:`TurnRequest`, and everything it may say back is one :class:`AdapterCall`.
Keeping both as explicit value objects is what makes the boundary checkable: the
compiled :class:`~boundarybench.compiler.Variant` — which carries the expected
disposition, the expected reason, the required evidence set and the variant's
content digest — never reaches adapter code at all, so an adapter cannot cheat
even by accident, and a provider integration cannot quietly start depending on
grading state.

No adapter here calls a network. This module defines the contract and ships one
deterministic fake for tests and CI; network providers live in separate modules.

Part of that contract is now *shared* rather than Boundary's. The fault, attempt,
settlement and turn-terminal vocabularies, the exception base and the
provider-side error, the single-flight guard, and the four value objects a turn
measures itself with are defined in :mod:`operatebench.providers` — the provider
kernel the Lifecycle Track composes too — and imported here. They are the same
objects under the same names, so ``isinstance``, this module's ``__all__`` and
every historical import are unchanged; what has moved is only where each of them
is written down, and the two section banners below say which.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from boundarybench.environment import Environment, query_affordances
from boundarybench.freezing import to_json
from boundarybench.scaffold import (
    FACT_PARAMETER,
    ActionSchema,
    Scaffold,
    project_action_surface,
)
from operatebench.providers.faults import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
    ATTEMPT_OUTCOMES,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    ATTEMPT_SETTLEMENTS,
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_CONFIGURATION,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_REQUEST_REJECTED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_TIMEOUT,
    PROVIDER_FAULTS,
    RETRYABLE_PROVIDER_FAULTS,
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_EARLY_REASONS,
    TURN_END_FAULT_NOT_RETRYABLE,
    TURN_END_PRE_DISPATCH_REFUSED,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
    TURN_END_UNCLASSIFIED,
    TURN_TERMINAL_REASONS,
    AdapterBusyError,
    AdapterError,
    AdapterProviderError,
    SingleFlight,
)
from operatebench.providers.telemetry import (
    AdapterUsage,
    ProviderAttempt,
    ProviderTelemetry,
    TurnDeadline,
)

#: The adapter contract version. Recorded in the run manifest, so a change to
#: what an adapter is shown or may answer is visible in run identity.
#:
#: ``0.1.0-dev.2`` moved what an implementation must *measure*. ``next_call``
#: has to clear its measurement before it does anything else, and ``last_usage``
#: has to report what the current call consumed even when that call ends in a
#: failure — see :class:`ModelAdapter`. A run under each records different
#: usage for the same provider traffic, so their identities have to say so.
#:
#: ``0.2.0`` added a second thing an implementation must report: structured
#: provider-attempt evidence (:meth:`ModelAdapter.last_telemetry`), under the
#: same two rules. Each turn records the requests it actually made and the
#: outcome of every attempt.
#:
#: ``0.3.0`` widened the closed fault set: an implementation may now report
#: :data:`PROVIDER_FAULT_CONFIGURATION`, and a refusal that names the run's
#: pinned model or resource must be reported as that rather than as a request
#: rejection. The two mean different things to a run — one stops it, the other
#: fails one episode — and the durable fault class preserves that distinction.
#:
#: ``0.4.0`` widened what a measurement may carry: an implementation may now
#: report a measured ``cost_usd`` on :class:`AdapterUsage`, computed from the
#: provider-reported token counts at the run's pinned pricing policy, and may
#: refuse to dispatch a request its run's remaining authorised budget cannot
#: cover. A row written under ``0.3.0`` reports a null cost for traffic a
#: ``0.4.0`` row prices, and a ``0.3.0`` run sends requests a ``0.4.0`` run
#: refuses, so the two are not the same experiment.
#:
#: ``0.5.0`` made every attempt state its own money. A
#: :class:`ProviderAttempt` now records the reservation that authorised it and
#: how that reservation was resolved, so a run's exposure can be re-derived from
#: the attempts that produced it rather than read back from a total the same
#: file states; an attempt this integration could not classify is recorded as
#: :data:`ATTEMPT_OUTCOME_UNCLASSIFIED` instead of vanishing from the evidence;
#: and an implementation must refuse a response whose measured cost exceeds the
#: reservation that authorised it rather than settling it. A ``0.4.0`` row
#: cannot answer what its attempts reserved, and a ``0.4.0`` run commits a
#: settlement a ``0.5.0`` run stops on.
#:
#: ``0.6.0`` made every turn state *why its retry loop stopped*. A
#: :class:`ProviderTelemetry` now carries a :data:`TURN_TERMINAL_REASONS` value,
#: and this contract names which faults may be followed by another attempt
#: (:data:`RETRYABLE_PROVIDER_FAULTS`). Together they turn a turn's attempt list
#: from a bag the reader could only count into a sequence it can check: a
#: response ends the turn, a fault that repeating cannot change ends the turn,
#: and a turn that stopped with retries still permitted has to name the durable
#: reason it stopped. A ``0.5.0`` row states no such reason, so what its
#: attempts were allowed to be cannot be decided from it.
#:
#: ``0.7.0`` moved what an adapter is *shown about its own arguments*. A
#: fact-acquisition action in :attr:`TurnRequest.actions` now carries the exact,
#: deterministic set of fact keys it can retrieve, on the argument that names
#: them (:data:`~boundarybench.scaffold.FACT_PARAMETER`), derived from the Cube
#: that is running. Up to ``0.6.0`` that argument was an unconstrained string
#: while the environment scored it against the Cube's own retrievable set, so a
#: model could name a key the case never had and be recorded as having chosen
#: wrongly rather than as having been asked an unanswerable question. An
#: implementation under this contract can
#: state what may be asked for; one under ``0.6.0`` could not, and its rows do
#: not mean the same thing.
#:
#: ``0.8.0`` moved what an acquisition action *means*. The argument it enumerates
#: is now a **query key** resolved by the Cube's authored query-resolution
#: registry (:mod:`boundarybench.queries`), not a fact key that always yields a
#: fact: a key the case offers may resolve to ``reveal``, ``unknown``,
#: ``not_recorded`` or ``out_of_scope``, and an adapter is shown the resolved
#: outcome on the observation it gets back
#: (:attr:`ObservedFact.outcome`). Up to ``0.7.0`` every offered key returned a
#: value by construction and no observation could say otherwise, so a run under
#: that contract could not have asked, or recorded, any of the other three
#: answers. The enum, the tool schemas and the environment's answer are all
#: projections of the one registry, so a run under this contract offers a
#: strictly different interface and its rows do not mean the same thing.
ADAPTER_CONTRACT_VERSION = "0.8.0"


# -- the exception taxonomy --------------------------------------------------
#
# :class:`AdapterError`, :class:`AdapterProviderError`, :class:`AdapterBusyError`
# and :class:`SingleFlight` are imported above from
# :mod:`operatebench.providers.faults`, together with the fault, attempt,
# settlement and turn-terminal vocabularies, and re-exported here under the names
# they have always had. They moved because they describe what a *provider call*
# can do rather than what the Boundary Track asks of one, and the Lifecycle Track
# composes the same kernel; they are the same objects, so ``isinstance``, this
# hierarchy and every existing ``except`` clause are unchanged.
#
# The two classes below stay here, because both are statements about this
# track's own protocol: a malformed call against a Boundary scaffold, and an
# answer cut off by the ceiling a Boundary run pinned.


class AdapterProtocolError(AdapterError):
    """The adapter answered with something that is not one allowed call.

    Distinct from an adapter *crashing*: this is a well-behaved implementation
    producing a malformed or unknown call, which is a model/scaffold protocol
    failure rather than an infrastructure one.
    """


class AdapterOutputLimitError(AdapterError):
    """The answer was cut off by the run's own output-token ceiling.

    Deliberately **not** an :class:`AdapterProtocolError`, and not a subclass of
    one. A truncated answer is not a model that cannot follow a schema: the
    model was still speaking when a limit *this run pinned* stopped it, and the
    fragment that survived is not the action it was making. Filing it under the
    model blamed the model for the run's configuration, and it also buried the
    one signal that would tell an operator what to change — the ceiling.

    Its message is fixed detail plus this build's own pinned ceiling. The
    provider's stated stop reason is not quoted, for the same reason no other
    provider text is: it is a durable row's content and it arrived from outside.
    """


# -- the unclassified case ---------------------------------------------------
#
# The classes above and :class:`AdapterProviderError` are what a *cooperating*
# adapter raises, and all of them carry detail this contract has already
# constrained. Everything else an adapter can
# raise is unclassified: an SDK error the integration declined to map, a bug in
# a third-party adapter, a library exception from four frames down. Its type is
# useful and its message is not — an SDK routinely builds one out of a response
# body, and a third-party adapter may build one out of anything at all — so the
# rule lives here, at the boundary that defines what an adapter is, rather than
# at each call site the runner happens to have.

#: The two calls the runner makes on an adapter. A closed set, because the name
#: is written into a durable failure detail and so may not be free text.
ADAPTER_BOUNDARY_NEXT_CALL = "next_call"
ADAPTER_BOUNDARY_LAST_USAGE = "last_usage"
ADAPTER_BOUNDARIES: tuple[str, ...] = (
    ADAPTER_BOUNDARY_LAST_USAGE,
    ADAPTER_BOUNDARY_NEXT_CALL,
)


def redacted_adapter_detail(exception: BaseException, *, boundary: str) -> str:
    """The durable sentence an unclassified adapter exception is recorded as.

    Fixed text, the exception's class and the boundary call it came out of —
    and nothing else. ``str(exception)`` is deliberately dropped: it is the one
    part of an exception that is unbounded, externally authored and, on the far
    side of a provider integration, frequently a verbatim copy of a response
    body or a request. A ledger row is durable evidence that a report renders
    and an audit reads, and text that arrived from outside does not belong in
    one however convenient it is while debugging.

    What survives is enough to act on. The class is what the row is *filed*
    under — :data:`~boundarybench.ledger.KIND_ADAPTER_EXCEPTION` takes its
    ``error_class`` from it — so a run whose episodes all failed with the same
    exception type still says so in a report's error-class counts, and an
    operator still knows which of the two boundary calls broke.
    """
    if boundary not in ADAPTER_BOUNDARIES:
        raise ValueError(
            f"{boundary!r} is not an adapter boundary this contract defines; "
            f"allowed: {list(ADAPTER_BOUNDARIES)}"
        )
    return (
        f"the adapter raised {type(exception).__name__} from {boundary}(); this "
        "contract does not classify it, so the episode is recorded as an adapter "
        "failure by that class. The exception's own message is not recorded: it is "
        "text from outside this build, of unbounded shape and provider-influenced "
        "content, and a durable failure row carries only fixed detail and "
        "closed-set values"
    )


# -- what the adapter is shown ----------------------------------------------


@dataclass(frozen=True)
class ObservedFact:
    """One observation handle the environment has revealed to the agent.

    Deliberately not :meth:`Observation.as_dict`: that projection carries
    ``evidence_relevant``, which states whether citing the handle would score.

    ``outcome`` is how a query the agent asked resolved, from
    :data:`~boundarybench.queries.QUERY_OUTCOMES`, and ``None`` on everything the
    agent did not ask for. It is shown because it *is* the answer: an agent that
    asked a question and got a handle carrying no value has to be able to read
    whether the user could not determine it, the record does not hold it, or the
    question was not relevant. It says nothing about the case's decision — the
    registry is Cube-level and invariant across arms — and it appears only on a
    handle the agent has already obtained.
    """

    observation_id: str
    source: str
    key: str
    fact: Mapping[str, Any]
    outcome: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source": self.source,
            "key": self.key,
            "fact": to_json(self.fact),
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class TranscriptEntry:
    """One action already taken and what the environment returned for it."""

    index: int
    action: str
    arguments: Mapping[str, Any]
    revealed_observation_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "arguments": to_json(self.arguments),
            "revealed_observation_ids": list(self.revealed_observation_ids),
        }


@dataclass(frozen=True)
class TurnRequest:
    """Everything an adapter may see when choosing its next action.

    The field list is the boundary. Adding a field here is a deliberate decision
    about what an agent is allowed to know, so ``variant``, ``expected_*``,
    ``required_evidence_ids`` and any evaluator verdict are absent by
    construction rather than by convention.
    """

    system_prompt: str
    scaffold_id: str
    scaffold_version: str
    actions: tuple[ActionSchema, ...]
    available_actions: tuple[str, ...]
    observations: tuple[ObservedFact, ...]
    transcript: tuple[TranscriptEntry, ...]
    turns_remaining: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "scaffold_id": self.scaffold_id,
            "scaffold_version": self.scaffold_version,
            "actions": [action.as_dict() for action in self.actions],
            "available_actions": list(self.available_actions),
            "observations": [entry.as_dict() for entry in self.observations],
            "transcript": [entry.as_dict() for entry in self.transcript],
            "turns_remaining": self.turns_remaining,
        }


def build_turn_request(
    *, scaffold: Scaffold, environment: Environment, turns_remaining: int
) -> TurnRequest:
    """Project the live episode down to what the agent has actually observed.

    The environment is the only source: an adapter sees the handles it was
    given and the steps it took, never the variant those came from.

    ``actions`` and ``available_actions`` are one list, projected once from the
    Cube's ``allowed_actions`` by :func:`~boundarybench.scaffold.
    project_action_surface`. The schemas and names are the same card-derived
    surface, so a case that
    excluded ``call_tool`` still declared a ``call_tool`` tool and then refused
    any use of it from metadata the agent had never been shown. An action
    surface that is not the case's own surface measures whether a model guesses
    the difference, which is not what this benchmark claims to measure.

    The same argument applies one level down, to an action's *arguments*, which
    is why the fact keys each acquisition channel can retrieve are projected
    from the same variant by :func:`~boundarybench.environment.
    query_affordances` and become part of the action schemas here. An
    acquisition argument the case cannot enumerate is one the environment grades
    against a hidden namespace.
    """
    variant = environment.variant
    actions = project_action_surface(
        scaffold,
        variant.allowed_actions,
        fact_affordances=query_affordances(variant),
    )
    return TurnRequest(
        system_prompt=scaffold.system_prompt,
        scaffold_id=scaffold.scaffold_id,
        scaffold_version=scaffold.scaffold_version,
        actions=actions,
        available_actions=tuple(action.name for action in actions),
        observations=tuple(
            ObservedFact(
                observation_id=observation.observation_id,
                source=observation.source,
                key=observation.key,
                fact=observation.fact,
                outcome=observation.outcome,
            )
            for observation in environment.observed()
        ),
        transcript=tuple(
            TranscriptEntry(
                index=step.index,
                action=step.action,
                arguments=step.arguments,
                revealed_observation_ids=step.revealed_observation_ids,
            )
            for step in environment.steps
        ),
        turns_remaining=turns_remaining,
    )


def offered_queries(request: TurnRequest) -> dict[str, list[str]]:
    """Which query keys each action of this turn accepts, for the model.

    Derived from :attr:`TurnRequest.actions` and from nothing else, so a
    provider adapter that renders this alongside its tool schemas is rendering
    the same enum the schemas carry rather than a second opinion about it. An
    action with no enumerable acquisition argument is absent, which is what a
    case offering no acquisition channel at all should say — an empty mapping,
    not a set of empty lists.

    Provider-neutral on purpose. A model-visible representation of what may be
    asked for is part of the interface under measurement, so it may not be
    something each integration invents; this is the one derivation, and any
    integration that shows the affordance in prose or JSON shows this.

    What it does *not* say is what any of those keys will return. The offered set
    is the Cube's query registry, which resolves some keys to a fact and others
    to ``unknown``, ``not_recorded`` or ``out_of_scope``; publishing the outcome
    beside the key would answer every question before it was asked.
    """
    return {
        action.name: list(parameter.enum)
        for action in request.actions
        for parameter in action.parameters
        if parameter.name == FACT_PARAMETER and parameter.enum is not None
    }


#: The name this projection had while every offered key returned a fact. Kept so
#: existing callers keep working; the offered set is now the query registry, and
#: some of its keys resolve to no value at all.
fact_affordances = offered_queries


# -- what the adapter answers ------------------------------------------------
#
# :class:`TurnDeadline`, :class:`AdapterUsage`, :class:`ProviderAttempt` and
# :class:`ProviderTelemetry` are imported above from
# :mod:`operatebench.providers.telemetry` and re-exported here. A wall-clock
# budget, two token counts and a price, and what one request met are the same
# facts whichever track projected the body, so they are defined once for both and
# these are the same classes — a dataclass compares, hashes and serialises exactly
# as it did.
#
# :class:`AdapterCall` stays: it is one action against a Boundary scaffold, which
# is this track's own answer shape and nobody else's.


@dataclass(frozen=True)
class AdapterCall:
    """Exactly one action and its arguments."""

    action: str
    arguments: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "arguments": to_json(self.arguments)}


@dataclass(frozen=True)
class AdapterIdentity:
    """Who answered, and with which implementation.

    ``provider``/``model`` name the service under test; ``implementation``/
    ``version`` name the code that talked to it. Both halves change results, so
    both are hashed into run identity.
    """

    provider: str
    model: str
    implementation: str
    version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "implementation": self.implementation,
            "version": self.version,
        }


class ModelAdapter(Protocol):
    """The whole provider surface: identity, settings and one next action.

    ``last_usage`` reports what the *current* call cost. The runner sums it
    across the episode, so an implementation returns one turn's measurement and
    never a running total.

    Two rules make that measurement trustworthy on the paths where it matters
    most, and both are part of :data:`ADAPTER_CONTRACT_VERSION` rather than
    conventions an integration may interpret:

    * **Cleared first.** ``next_call`` clears the measurement before it does
      anything else — before it serialises a request, before it opens a socket.
      A call that fails before any response arrives therefore reports nothing,
      rather than leaving the *previous* call's numbers readable as if they
      were this one's.
    * **Reported even on failure.** A response that arrived was paid for. Its
      tokens and the turn's latency are recorded as soon as it is measurable,
      so ``last_usage`` still answers for the consumption after ``next_call``
      has raised a protocol failure over the same response. What was never
      received is never invented: a transport failure with no response measures
      nothing, which is not the same claim as measuring zero.

    ``last_telemetry`` reports what the current call *did*, on the same two
    rules: cleared first, reported even on failure. It answers ``None`` when the
    implementation makes no provider request at all — an in-process fake has no
    attempts to report, and reporting zero attempts would claim it tried and
    failed. An implementation that does call a provider answers with a
    :class:`ProviderTelemetry` on every path, including the ones that raise.

    Both rules are statements about *one* call, and that is what makes the third
    rule necessary: **an adapter instance executes one turn at a time.** Its
    measurement is per-instance state that the turn in flight owns, so a second
    overlapping ``next_call`` on the same instance is refused immediately with
    :class:`AdapterBusyError` — it dispatches nothing, resets nothing and leaves
    the turn in flight untouched. Sequential reuse of one adapter, which is what
    an episode does, is unaffected; a caller that wants two turns at once gives
    each one its own adapter. Every provider adapter in this build enforces it
    through :class:`SingleFlight`.
    """

    @property
    def identity(self) -> AdapterIdentity: ...

    @property
    def settings(self) -> Mapping[str, Any]: ...

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall: ...

    def last_usage(self) -> AdapterUsage: ...

    def last_telemetry(self) -> ProviderTelemetry | None: ...


# -- the released fake, and the test double ----------------------------------
#
# There are two of these and the difference is the whole point of the section.
#
# :class:`PolicyFollowingFakeAdapter` is *released*. Its identity and its
# behaviour are both fixed by this file: nothing can be injected into it, so
# "the fake run" names exactly one behaviour, and two runs that record the fake
# identity really did do the same thing. It is what ``run-suite --adapter
# fake-scripted`` runs, and it is the only adapter the CLI can construct.
#
# :class:`ScriptedTestAdapter` is a *test double*. It takes an arbitrary script
# because exercising a timeout, a malformed call or a provider that raises needs
# one. That flexibility is exactly why it may not claim to be anything else: its
# provider is fixed to :data:`TEST_DOUBLE_PROVIDER`, so it can neither name a
# real service nor borrow the released fake's identity, and a run it executes
# hashes to a different ``run_id`` from a released fake run by construction.
#
# The rule both obey: a fake produces no model measurement, and nothing either
# of them outputs may be reported as a benchmark result.

#: A script: a pure function of the request. The signature is the point — a
#: script cannot consult anything the agent was not shown.
FakeScript = Callable[[TurnRequest], AdapterCall]

#: The released fake's identity. Fixed here, not supplied by a caller.
FAKE_PROVIDER = "fake"
FAKE_MODEL = "policy-following-deterministic"
FAKE_IMPLEMENTATION = "policy_following_fake"

#: The test double's provider. Reserved: no real service is ever called
#: ``test-double``, and the released fake is not, either.
TEST_DOUBLE_PROVIDER = "test-double"
TEST_DOUBLE_MODEL = "scripted-test-double"
TEST_DOUBLE_IMPLEMENTATION = "scripted_test_double"

#: The providers that are this process rather than a service. Both are reserved
#: names owned by this module — nothing outside it can construct an adapter
#: claiming either — which is what makes "is this an external provider run?" a
#: decidable question rather than a guess about a name.
#:
#: Stated here, at the boundary that defines what an adapter is, because the run
#: manifest derives a run's *track* from it: a run against one of these executes
#: no model and must say so, and a run against anything else is an execution
#: against an external provider and must say that instead.
IN_PROCESS_PROVIDERS: frozenset[str] = frozenset({FAKE_PROVIDER, TEST_DOUBLE_PROVIDER})


class FakeAdapterProvenanceError(AdapterError):
    """A test double tried to claim an identity that is not its own.

    Refused rather than recorded. A script-injectable adapter that could state
    ``openai/gpt-4o`` would produce a ledger whose rows assert a provider that
    was never contacted, and one that could state the released fake's identity
    would make two different behaviours share a single ``run_id``.
    """


def fake_identity() -> AdapterIdentity:
    """The one identity a released fake run records. Takes no parameters."""
    return AdapterIdentity(
        provider=FAKE_PROVIDER,
        model=FAKE_MODEL,
        implementation=FAKE_IMPLEMENTATION,
        version=ADAPTER_CONTRACT_VERSION,
    )


def identity_for_test_double(
    *,
    model: str = TEST_DOUBLE_MODEL,
    implementation: str = TEST_DOUBLE_IMPLEMENTATION,
    version: str = ADAPTER_CONTRACT_VERSION,
) -> AdapterIdentity:
    """An identity for a test double. The provider is not a parameter."""
    return AdapterIdentity(
        provider=TEST_DOUBLE_PROVIDER,
        model=model,
        implementation=implementation,
        version=version,
    )


@dataclass(frozen=True)
class PolicyFollowingFakeAdapter:
    """The released deterministic fake. **Not a model.**

    It exists so the runner, the ledger and the resume path can be exercised end
    to end without a network, and so CI has a reference execution whose result
    is fixed by released code rather than by whatever a caller passed in. There
    is no ``script`` field: the behaviour is :func:`policy_following_script`,
    which is part of this build's contract and moves only when
    :data:`ADAPTER_CONTRACT_VERSION` does.

    Settings remain a parameter because they are recorded in run identity: two
    runs with different settings are two different runs, by their ids.
    """

    settings: Mapping[str, Any]

    @property
    def identity(self) -> AdapterIdentity:
        return fake_identity()

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        # The deadline is accepted and ignored: a fake does no I/O, so it has
        # nothing to cancel. Accepting it keeps the fake on the same contract a
        # provider adapter will implement.
        return policy_following_script(request)

    def last_usage(self) -> AdapterUsage:
        # A fake has no tokens, no cost and no meaningful latency. Reporting
        # zeros would be a fabricated measurement, so every field stays null.
        return AdapterUsage()

    def last_telemetry(self) -> ProviderTelemetry | None:
        # No provider was asked, so there is nothing to report. ``None``, not an
        # empty telemetry: zero attempts would say this adapter tried to reach a
        # service and got nowhere, which is a different and false claim.
        return None


@dataclass(frozen=True)
class ScriptedTestAdapter:
    """A script-injectable adapter **for tests only**, and never a fake run.

    The script is a pure function of :class:`TurnRequest`, which keeps even a
    test fixture on the right side of the boundary: it can force any behaviour
    it likes, but only from what the agent can see. The identity is constrained
    rather than free — see :class:`FakeAdapterProvenanceError`.
    """

    script: FakeScript
    identity: AdapterIdentity
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.identity.provider != TEST_DOUBLE_PROVIDER:
            raise FakeAdapterProvenanceError(
                f"a scripted test adapter may only run as provider "
                f"{TEST_DOUBLE_PROVIDER!r}, not {self.identity.provider!r}. Its "
                "behaviour is supplied by the caller, so it may claim neither a "
                "real provider nor the released fake identity; a run it executes "
                "is a test double's run and its run_id says so."
            )

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        return self.script(request)

    def last_usage(self) -> AdapterUsage:
        return AdapterUsage()

    def last_telemetry(self) -> ProviderTelemetry | None:
        return None


def _observation_by_source(request: TurnRequest, source: str) -> ObservedFact | None:
    for observation in request.observations:
        if observation.source == source:
            return observation
    return None


def _taken(request: TurnRequest, action: str) -> bool:
    return any(entry.action == action for entry in request.transcript)


def _requested_facts(request: TurnRequest) -> set[str]:
    facts: set[str] = set()
    for entry in request.transcript:
        value = entry.arguments.get("fact")
        if isinstance(value, str):
            facts.add(value)
    return facts


def _matches(condition: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    key = condition["fact"]
    if key not in facts:
        return False
    actual, expected = facts[key], condition["value"]
    op = condition["op"]
    if op == "eq":
        return type(actual) is type(expected) and bool(actual == expected)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if op == "gt":
        return bool(actual > expected)
    if op == "lt":
        return bool(actual < expected)
    if op == "gte":
        return bool(actual >= expected)
    return bool(actual <= expected)


def policy_following_script(request: TurnRequest) -> AdapterCall:
    """The fake's default behaviour: apply the policy the case presented.

    It reads the active policy out of the observations, obtains whichever facts
    the rules reference, then answers. That is deliberately the same information
    a real agent gets — the script has no access to the expected disposition, so
    when it happens to be right, it is right for the observable reason.
    """
    if not _taken(request, "read_records"):
        return AdapterCall("read_records", {})

    policy = _observation_by_source(request, "active_policy")
    if policy is None:  # pragma: no cover - read_records always reveals it
        return AdapterCall("complete_case", _terminal_arguments("STOP", "UNKNOWN", ()))

    known: dict[str, Any] = {}
    for observation in request.observations:
        known.update(observation.fact)

    rules = policy.fact["rules"]
    cited = [policy.observation_id]
    for rule in rules:
        needed = [condition["fact"] for condition in rule["when"]]
        for key in needed:
            if key in known:
                continue
            if key in _requested_facts(request):
                continue
            channel = _elicitation_channel(request, key)
            if channel is None:  # pragma: no cover - defensive
                break
            return AdapterCall(channel, {"fact": key})
        if all(_matches(condition, known) for condition in rule["when"]):
            for observation in request.observations:
                if observation.key in needed and observation.observation_id not in cited:
                    cited.append(observation.observation_id)
            return _decide(request, rule, tuple(cited))
        for observation in request.observations:
            if observation.key in needed and observation.observation_id not in cited:
                cited.append(observation.observation_id)
    # No presented rule covers the observed facts: say so rather than guess.
    return AdapterCall(
        "complete_case", _terminal_arguments("STOP", "NO_APPLICABLE_RULE", ())
    )


def _elicitation_channel(request: TurnRequest, key: str) -> str | None:
    """Which offered action this turn says can retrieve ``key``.

    The affordance is consulted first, because it is the answer: a case may offer
    both channels and carry different keys on each, and picking the first one
    that exists would then ask the wrong one and be refused for it. The fallback
    is the first channel offered, which is what a case that enumerates nothing
    for its channels would need — and no such case can be projected, so it stands
    only for a request built some other way.
    """
    affordances = offered_queries(request)
    for candidate in ("ask_user", "call_tool"):
        if key in affordances.get(candidate, ()):
            return candidate
    for candidate in ("ask_user", "call_tool"):
        if candidate in request.available_actions:
            return candidate
    return None


def _decide(
    request: TurnRequest, rule: Mapping[str, Any], cited: tuple[str, ...]
) -> AdapterCall:
    """Act first when the rule says act, then close the case."""
    if rule["decision"] == "ACT":
        pending = [
            action
            for action in request.available_actions
            if action not in _PROTOCOL_ACTIONS and not _taken(request, action)
        ]
        if pending:
            # Called by its own name: a workflow action is offered as its own
            # tool, so naming it is the whole call and there is no argument to
            # get wrong.
            return AdapterCall(pending[0], {})
    return AdapterCall(
        "complete_case",
        _terminal_arguments(str(rule["decision"]), str(rule["primary_reason"]), cited),
    )


_PROTOCOL_ACTIONS = frozenset({"read_records", "ask_user", "call_tool", "complete_case"})


def _terminal_arguments(
    disposition: str, reason: str, cited: Sequence[str]
) -> dict[str, Any]:
    return {
        "disposition": disposition,
        "primary_reason_code": reason,
        "secondary_reason_codes": [],
        "evidence_refs": list(cited),
    }


__all__ = [
    "ADAPTER_BOUNDARIES",
    "ADAPTER_BOUNDARY_LAST_USAGE",
    "ADAPTER_BOUNDARY_NEXT_CALL",
    "ADAPTER_CONTRACT_VERSION",
    "ATTEMPT_OUTCOMES",
    "ATTEMPT_OUTCOME_FAULT",
    "ATTEMPT_OUTCOME_RESPONSE",
    "ATTEMPT_OUTCOME_UNCLASSIFIED",
    "ATTEMPT_SETTLEMENTS",
    "ATTEMPT_SETTLEMENT_FORFEITED",
    "ATTEMPT_SETTLEMENT_MEASURED",
    "FAKE_IMPLEMENTATION",
    "FAKE_MODEL",
    "FAKE_PROVIDER",
    "IN_PROCESS_PROVIDERS",
    "PROVIDER_FAULTS",
    "PROVIDER_FAULT_AUTHENTICATION",
    "PROVIDER_FAULT_CONFIGURATION",
    "PROVIDER_FAULT_NETWORK_ERROR",
    "PROVIDER_FAULT_RATE_LIMITED",
    "PROVIDER_FAULT_REQUEST_REJECTED",
    "PROVIDER_FAULT_RESPONSE_INVALID",
    "PROVIDER_FAULT_SERVER_ERROR",
    "PROVIDER_FAULT_TIMEOUT",
    "RETRYABLE_PROVIDER_FAULTS",
    "TEST_DOUBLE_IMPLEMENTATION",
    "TEST_DOUBLE_MODEL",
    "TEST_DOUBLE_PROVIDER",
    "TURN_END_BACKOFF_UNAFFORDABLE",
    "TURN_END_COST_CAP_EXHAUSTED",
    "TURN_END_DEADLINE_EXCEEDED",
    "TURN_END_EARLY_REASONS",
    "TURN_END_FAULT_NOT_RETRYABLE",
    "TURN_END_PRE_DISPATCH_REFUSED",
    "TURN_END_RESPONSE",
    "TURN_END_RETRIES_EXHAUSTED",
    "TURN_END_UNCLASSIFIED",
    "TURN_TERMINAL_REASONS",
    "AdapterBusyError",
    "AdapterCall",
    "AdapterError",
    "AdapterIdentity",
    "AdapterOutputLimitError",
    "AdapterProtocolError",
    "AdapterProviderError",
    "AdapterUsage",
    "FakeAdapterProvenanceError",
    "FakeScript",
    "ModelAdapter",
    "ObservedFact",
    "PolicyFollowingFakeAdapter",
    "ProviderAttempt",
    "ProviderTelemetry",
    "ScriptedTestAdapter",
    "SingleFlight",
    "TranscriptEntry",
    "TurnDeadline",
    "TurnRequest",
    "build_turn_request",
    "fact_affordances",
    "fake_identity",
    "identity_for_test_double",
    "offered_queries",
    "policy_following_script",
    "redacted_adapter_detail",
]
