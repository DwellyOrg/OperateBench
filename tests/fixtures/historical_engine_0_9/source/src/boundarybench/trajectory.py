"""Deterministic trajectory contract.

An episode is a sequence of steps produced by a solver against an environment.
The environment owns every identifier: solvers never mint observation handles,
which is what lets the evaluator distinguish grounded evidence from fabricated
references (spec §15).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from boundarybench.freezing import deep_freeze, to_json
from boundarybench.queries import OUTCOME_REVEAL, QUERY_OUTCOMES, REVEALING_OUTCOMES

#: Observation sources. Everything the agent can ever cite is one of these.
OBSERVATION_SOURCES: frozenset[str] = frozenset(
    {
        "active_policy",
        "initial_record",
        "user_answer",
        "tool_output",
        "user_statement",
    }
)

TERMINAL_ACTION = "complete_case"


@dataclass(frozen=True)
class Observation:
    """An immutable, system-generated handle over one piece of information.

    ``outcome`` is what a *query* resolved to, from
    :data:`~boundarybench.queries.QUERY_OUTCOMES`, and is ``None`` on everything
    that is not a query resolution: the active policy, the initial record, a
    pressure statement, a fact that was in context from the start. It is the
    field that makes a non-reveal answer sayable at all, and the constraints
    below are what keep it from being sayable dishonestly — a non-reveal handle
    carries no value and cannot present itself as fact evidence, so nothing
    downstream can read "the user could not tell me" as "the user told me".
    """

    observation_id: str
    source: str
    key: str
    fact: Mapping[str, Any]
    availability: str
    evidence_relevant: bool
    outcome: str | None = None

    def __post_init__(self) -> None:
        if self.source not in OBSERVATION_SOURCES:
            raise ValueError(f"unknown observation source {self.source!r}")
        if self.outcome is not None and self.outcome not in QUERY_OUTCOMES:
            raise ValueError(
                f"observation {self.observation_id!r} declares query outcome "
                f"{self.outcome!r}; this build resolves exactly "
                f"{list(QUERY_OUTCOMES)}"
            )
        object.__setattr__(self, "fact", deep_freeze(self.fact))
        if self.outcome is None:
            return
        if self.outcome in REVEALING_OUTCOMES:
            if not self.fact:
                raise ValueError(
                    f"observation {self.observation_id!r} resolves to "
                    f"{OUTCOME_REVEAL!r} and carries no fact; a reveal is the "
                    "authored value or it is not a reveal"
                )
            return
        if self.fact:
            raise ValueError(
                f"observation {self.observation_id!r} resolves to {self.outcome!r} "
                f"and carries the value(s) {sorted(self.fact)}. Only "
                f"{OUTCOME_REVEAL!r} carries a value: a non-reveal answer states the "
                "query key and the outcome, so it can never be read back as "
                "evidence of something nobody revealed"
            )
        if self.evidence_relevant:
            raise ValueError(
                f"observation {self.observation_id!r} resolves to {self.outcome!r} "
                "and claims to be evidence. A non-reveal outcome cannot satisfy a "
                "predicate that requires a revealed fact value; a construct that "
                "makes one load-bearing has to declare that explicitly"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "source": self.source,
            "key": self.key,
            "fact": to_json(self.fact),
            "availability": self.availability,
            "evidence_relevant": self.evidence_relevant,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class TerminalDecision:
    """The structured terminal call, per spec §13."""

    disposition: str
    primary_reason_code: str
    secondary_reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "primary_reason_code": self.primary_reason_code,
            "secondary_reason_codes": list(self.secondary_reason_codes),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class Step:
    """One agent action and everything the environment returned for it."""

    index: int
    action: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    revealed_observation_ids: tuple[str, ...] = ()
    mutations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", deep_freeze(self.arguments))

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "arguments": to_json(self.arguments),
            "revealed_observation_ids": list(self.revealed_observation_ids),
            "mutations": list(self.mutations),
        }


@dataclass(frozen=True)
class Trajectory:
    """A complete, replayable episode over a single variant."""

    variant_id: str
    steps: tuple[Step, ...]
    terminal_decision: TerminalDecision | None
    observed_ids: tuple[str, ...]
    mutation_history: tuple[tuple[int, str], ...]

    @property
    def terminal_indices(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.steps if s.action == TERMINAL_ACTION)

    def observed_before(self, step_index: int) -> frozenset[str]:
        """Observation ids revealed strictly before ``step_index``."""
        revealed: set[str] = set()
        for step in self.steps:
            if step.index >= step_index:
                break
            revealed.update(step.revealed_observation_ids)
        return frozenset(revealed)

    def first_index_of(self, action: str) -> int | None:
        for step in self.steps:
            if step.action == action:
                return step.index
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "steps": [step.as_dict() for step in self.steps],
            "terminal_decision": (
                self.terminal_decision.as_dict() if self.terminal_decision else None
            ),
            "observed_ids": list(self.observed_ids),
            "mutation_history": [list(item) for item in self.mutation_history],
        }
