"""Deterministic environment for a single variant.

The environment is the only thing that mints observation handles and the only
thing that records mutations. A solver can therefore never cite information it
was not given, and the evaluator can always tell what was known at which step.

There is no LLM anywhere in this loop: the user simulator is a lookup over the
variant's precompiled observations.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from boundarybench.compiler import Variant
from boundarybench.queries import SOURCE_BY_ACTION
from boundarybench.trajectory import (
    TERMINAL_ACTION,
    Observation,
    Step,
    TerminalDecision,
    Trajectory,
)

READ_ACTION = "read_records"

#: Which action delivers an answer from each elicitable source. Public, because
#: a caller has to be able to ask which channel a query arrives on *before*
#: acting: recording the step first and objecting afterwards would file an
#: action the agent never took.
#:
#: Inverted from the resolution contract's own table rather than restated, so
#: the channel a registry entry names and the channel a step is filed under are
#: one mapping read in two directions.
ACTION_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {source: action for action, source in SOURCE_BY_ACTION.items()}
)


class EnvironmentError(RuntimeError):
    """Raised when a solver attempts an action the environment forbids."""


def channel_for(variant: Variant, query_key: str) -> str | None:
    """Which acquisition action this variant accepts ``query_key`` on, if any.

    Public so both the live dispatch loop and trajectory replay ask the exact
    same question about a query: which single channel the Cube's registry
    assigns it. ``None`` when the Cube offers no query under that name — a key
    that is not in the registry at all, or a fact that is already in the initial
    record and so is read rather than asked for. Both of those are left to
    :meth:`Environment.obtain`, which refuses them with their own messages.
    """
    return variant.query_registry.channel_for(query_key)


def query_affordances(variant: Variant) -> Mapping[str, tuple[str, ...]]:
    """Which query keys each acquisition channel this case offers accepts.

    The affordance the model is shown, and the same answer
    :func:`channel_for` and :meth:`Environment.obtain` give — because all three
    read the one compiled registry rather than each deriving an opinion. Two
    tables of this fact are exactly the defect this contract closes: the tool
    schema said ``fact`` was any string, and the environment scored it against
    the Cube's declared elicitation channels. A model could then ask for a key
    that was plausible under the policy text and be refused using a namespace it
    had never received. Under the resolution
    contract such a key is either in the registry — with an authored outcome,
    which may well be ``unknown`` — or it is not offered at all.

    One entry per elicitation action the case *allows*, always — including an
    action no query is offered on, which gets an empty tuple rather than being
    silently dropped. That distinction is the whole point: an offered channel
    with nothing to ask it is a case that cannot be executed as authored, and it
    has to be visible to the projection that would otherwise put an
    unconstrained argument in front of a model
    (:func:`~boundarybench.scaffold.project_action_surface`).

    Keys are sorted by the registry itself, so the enum a Cube offers is fixed
    by the Cube and not by the order its card happened to declare things in. The
    result is immutable at both levels — a read-only mapping of tuples — because
    it is handed to the projection that builds a model-facing interface out of
    it, and a caller that could edit it could edit what a run offered after the
    fact.
    """
    offered = set(variant.allowed_actions) & set(ACTION_BY_SOURCE.values())
    registered = variant.query_registry.affordances()
    return MappingProxyType(
        {action: tuple(registered.get(action, ())) for action in sorted(offered)}
    )


#: The name this projection had before queries could resolve to anything but a
#: fact. Kept as an alias because a Cube's reveal queries *are* its retrievable
#: fact keys, and every caller of the old name wanted the offered enum; the new
#: name is used everywhere in this build, because the enum now also carries keys
#: whose authored answer is that there is no value to retrieve.
retrievable_fact_keys = query_affordances


class Environment:
    """Mediates every interaction between a solver and one variant."""

    def __init__(self, variant: Variant) -> None:
        self.variant = variant
        self._steps: list[Step] = []
        self._observed: list[str] = []
        self._mutations: list[tuple[int, str]] = []
        self._terminal: TerminalDecision | None = None

    # -- introspection -----------------------------------------------------

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(self._steps)

    @property
    def observed_ids(self) -> tuple[str, ...]:
        return tuple(self._observed)

    def observed(self) -> tuple[Observation, ...]:
        return tuple(
            obs
            for obs in self.variant.observations
            if obs.observation_id in self._observed
        )

    def known_facts(self) -> dict[str, Any]:
        """Every fact the solver has legitimately seen so far."""
        facts: dict[str, Any] = {}
        for obs in self.observed():
            facts.update(obs.fact)
        return facts

    # -- actions -----------------------------------------------------------

    def read_records(self) -> tuple[Observation, ...]:
        """Reveal everything available at the start of the case."""
        revealed = self.variant.initial_observations()
        self._record(READ_ACTION, {}, revealed)
        return revealed

    def obtain(self, query_key: str) -> Observation:
        """Resolve one query through the channel the Cube's registry assigns it.

        The keys this accepts are the keys :func:`query_affordances` offers,
        because both read the one compiled registry. A key outside that set is
        refused here as a last resort only: the model is shown the set as an
        enum, so a live turn asking for anything else is refused as an argument
        failure before this is reached — and is never mapped, by anything here,
        onto one of the four authored outcomes.

        What comes back is the authored resolution and nothing else. A ``reveal``
        returns the fact; every other outcome returns a handle carrying the query
        key and the outcome. Either way the environment's state and the case's
        policy are untouched: a question moves information, never state.

        Asking twice reveals once and is recorded twice. The handle is the same
        object and no observation is revealed again — there is no further
        information to move — but the action was dispatched, so it is filed like
        any other, with nothing revealed and nothing mutated. Recording it is
        what keeps the episode's own counters honest: a dispatched action costs
        a turn, and a turn with no step is a model action that left no audit
        trail. That gap is what made a run of twelve completed provider turns
        unreadable against a trajectory of nine steps.
        """
        entry = self.variant.query_registry.entry(query_key)
        if entry is not None:
            target = self._resolution_observation(query_key)
            if target.observation_id in self._observed:
                self._record(entry.action, {"fact": query_key}, ())
                return target
            self._record(entry.action, {"fact": query_key}, (target,))
            return target

        # Not a query at all. A fact already in the initial record is read, not
        # asked for, and its own message says so; anything else this Cube simply
        # does not have.
        initial_target = next(
            (
                obs
                for obs in self.variant.observations
                if obs.evidence_relevant and query_key in obs.fact
            ),
            None,
        )
        if initial_target is None:
            raise EnvironmentError(
                f"variant {self.variant.variant_id} has no observation for query "
                f"{query_key!r}"
            )
        if initial_target.observation_id in self._observed:
            return initial_target
        raise EnvironmentError(
            f"fact {query_key!r} is available from the start but records have "
            "not been read yet"
        )

    def _resolution_observation(self, query_key: str) -> Observation:
        """The one on-request observation this Cube compiled for ``query_key``.

        The registry and the observation set are built from each other by the
        compiler, so a registry entry with no observation is a broken build
        rather than a case a run can meet; it is still named rather than
        allowed to surface as a ``StopIteration``.
        """
        for observation in self.variant.observations:
            if observation.availability == "on_request" and observation.key == query_key:
                return observation
        raise EnvironmentError(  # pragma: no cover - defensive
            f"variant {self.variant.variant_id} resolves query {query_key!r} but "
            "compiled no observation for it"
        )

    def act(self, action: str) -> None:
        """Perform a state-mutating, irreversible action."""
        if action not in self.variant.irreversible_actions:
            raise EnvironmentError(
                f"action {action!r} is not allowed for variant {self.variant.variant_id}"
            )
        index = self._record(action, {}, (), mutations=(action,))
        self._mutations.append((index, action))

    def complete_case(
        self,
        *,
        disposition: str,
        primary_reason_code: str,
        secondary_reason_codes: tuple[str, ...] = (),
        evidence_refs: tuple[str, ...] = (),
    ) -> None:
        """Issue the structured terminal decision (spec §13).

        Recording happens first: if the case has already terminated the step is
        rejected and the stored decision is left untouched, so a refused second
        call cannot overwrite the first one.
        """
        decision = TerminalDecision(
            disposition=disposition,
            primary_reason_code=primary_reason_code,
            secondary_reason_codes=tuple(secondary_reason_codes),
            evidence_refs=tuple(evidence_refs),
        )
        self._record(
            TERMINAL_ACTION,
            {
                "disposition": disposition,
                "primary_reason_code": primary_reason_code,
                "secondary_reason_codes": list(secondary_reason_codes),
                "evidence_refs": list(evidence_refs),
            },
            (),
        )
        self._terminal = decision

    # -- result ------------------------------------------------------------

    def trajectory(self) -> Trajectory:
        return Trajectory(
            variant_id=self.variant.variant_id,
            steps=tuple(self._steps),
            terminal_decision=self._terminal,
            observed_ids=tuple(self._observed),
            mutation_history=tuple(self._mutations),
        )

    # -- internals ---------------------------------------------------------

    def _record(
        self,
        action: str,
        arguments: Mapping[str, Any],
        revealed: tuple[Observation, ...],
        mutations: tuple[str, ...] = (),
    ) -> int:
        if action not in self.variant.allowed_actions:
            raise EnvironmentError(
                f"action {action!r} is not allowed for variant "
                f"{self.variant.variant_id}; the card permits "
                f"{list(self.variant.allowed_actions)}"
            )
        if self._steps and self._steps[-1].action == TERMINAL_ACTION:
            raise EnvironmentError(
                f"cannot take action {action!r}: the case reached its terminal "
                "decision already"
            )
        index = len(self._steps) + 1
        for obs in revealed:
            if obs.observation_id not in self._observed:
                self._observed.append(obs.observation_id)
        self._steps.append(
            Step(
                index=index,
                action=action,
                arguments=dict(arguments),
                revealed_observation_ids=tuple(o.observation_id for o in revealed),
                mutations=mutations,
            )
        )
        return index


# -- replay --------------------------------------------------------------


_ELICITATION_ACTIONS: frozenset[str] = frozenset(ACTION_BY_SOURCE.values())


def replay_trajectory(variant: Variant, trajectory: Trajectory) -> Trajectory:
    """Regenerate the trajectory a stored one claims by replaying it for real.

    Rebuilding the typed object and re-encoding it only proves a row is
    internally consistent; it says nothing about whether the environment would
    ever have produced these steps in this order. This drives a fresh
    :class:`Environment` through the recorded steps, in the recorded order,
    using the exact dispatch a live episode uses — same channel, same
    irreversible-action and terminal semantics — and hands back whatever the
    environment actually produces. The caller compares that against the stored
    trajectory; anything the environment would not have produced this way
    disagrees with it.
    """
    environment = Environment(variant)
    for step in trajectory.steps:
        _replay_step(environment, step)
    return environment.trajectory()


def replay_turn_states(variant: Variant, trajectory: Trajectory) -> Iterator[Environment]:
    """The environment as each turn of the episode found it, in order.

    A turn's request is a projection of the environment the turn started from,
    so anything that has to rederive what a turn asked for — its size, and
    therefore what it cost to authorise — has to stand where that turn stood.
    A completed turn adds exactly one recorded step and a turn that died before
    the environment saw a validated action adds none, so the state turn ``t``
    started from is the state after ``t - 1`` replayed steps. That correspondence
    is not assumed here: the ledger proves it per row from the recorded fault
    before it asks (see ``ledger._verify_counters``).

    One environment is advanced and yielded repeatedly rather than copied, so a
    caller must use each state before asking for the next one. Yielding
    snapshots would mean a second definition of what an environment's state is,
    and the whole point of replaying is that there is only the real one.
    """
    environment = Environment(variant)
    yield environment
    for step in trajectory.steps:
        _replay_step(environment, step)
        yield environment


def _replay_step(environment: Environment, step: Step) -> None:
    action = step.action
    arguments = step.arguments
    try:
        if action == READ_ACTION:
            environment.read_records()
            return
        if action in _ELICITATION_ACTIONS:
            fact = arguments["fact"]
            if not isinstance(fact, str):
                raise EnvironmentError(
                    f"step {step.index} calls {action!r} without a string 'fact' argument"
                )
            expected = channel_for(environment.variant, fact)
            if expected is not None and expected != action:
                raise EnvironmentError(
                    f"step {step.index}: fact {fact!r} is not available through "
                    f"{action!r}; this case delivers it through {expected!r}"
                )
            environment.obtain(fact)
            return
        if action == TERMINAL_ACTION:
            environment.complete_case(
                disposition=str(arguments["disposition"]),
                primary_reason_code=str(arguments["primary_reason_code"]),
                secondary_reason_codes=tuple(
                    str(item) for item in arguments["secondary_reason_codes"]
                ),
                evidence_refs=tuple(str(item) for item in arguments["evidence_refs"]),
            )
            return
        environment.act(action)
    except EnvironmentError:
        raise
    except (KeyError, TypeError) as exc:
        raise EnvironmentError(
            f"step {step.index}: {action!r} has malformed arguments: {exc}"
        ) from exc
