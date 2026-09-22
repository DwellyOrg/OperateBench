"""The deterministic question-resolution contract.

Every request a model makes for information resolves here, through an authored,
immutable, Cube-level registry — never through a runtime judgement about what a
question "probably means". The registry answers exactly three questions about a
query key, and it answers them the same way for every cell and every pressure
probe of the Cube:

* which single acquisition channel the Cube offers it on;
* which of four closed outcomes it returns;
* and, for ``reveal`` and only for ``reveal``, which authored fact it carries.

The four outcomes are the whole vocabulary. ``reveal`` returns one existing
authored fact through its assigned channel; ``unknown`` says the user or tool
cannot determine the information; ``not_recorded`` says the relevant system does
not hold it; ``out_of_scope`` says the request is not relevant to this workflow
decision. A non-reveal answer carries the query key and the outcome, and nothing
else: no value, no prose, no state change, and nothing a predicate that requires
a revealed fact value can be satisfied by.

This module is the one source. The compiled registry drives the neutral action
argument enums, the provider tool schemas, the model-visible affordances, the
environment's dispatch and outcome generation, the observations and their
handles, evaluator replay, and content/suite/run identity. It deliberately
imports nothing from the rest of the package, so nothing can layer a second
opinion about resolution on top of it.

What it is *not*: there is no fuzzy matching, no natural-language classifier and
no inference. A key outside the offered enum is not mapped to one of these four
outcomes by anything here — it is a malformed argument against the interface the
model was shown, and stays one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

#: How this build resolves a model's request for information, recorded in
#: adapter settings and in compiled content so it is inside every identity a run
#: has. A Cube can answer with a fact or another declared resolution; the
#: contract makes every outcome explicit. A run
#: recorded under this value resolved every accepted query through an authored
#: registry with four closed outcomes.
QUERY_RESOLUTION_CONTRACT = "cube_query_resolution_registry_v1"

#: Returns one authored existing fact through its assigned channel.
OUTCOME_REVEAL = "reveal"
#: The user or the tool cannot determine the requested information.
OUTCOME_UNKNOWN = "unknown"
#: The information is absent from the relevant system or record.
OUTCOME_NOT_RECORDED = "not_recorded"
#: The request is not relevant to this workflow decision.
OUTCOME_OUT_OF_SCOPE = "out_of_scope"

#: The closed set, sorted. Closed on purpose: an outcome is authored evidence,
#: so a Cube cannot introduce a fifth meaning by naming one.
QUERY_OUTCOMES: tuple[str, ...] = (
    OUTCOME_NOT_RECORDED,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_REVEAL,
    OUTCOME_UNKNOWN,
)

#: The outcomes that carry an authored fact value. Exactly one, and stated as a
#: set rather than a comparison so the rule has one home.
REVEALING_OUTCOMES: frozenset[str] = frozenset({OUTCOME_REVEAL})

#: The acquisition channels a Cube may offer a query on. The protocol's two
#: elicitation actions and nothing else — reading records is not a question, and
#: a workflow action moves state rather than information.
QUERY_ACTIONS: tuple[str, ...] = ("ask_user", "call_tool")

#: Which observation source each channel delivers through. A non-reveal answer
#: still arrives *from the user* or *from the tool*: it is that party saying they
#: cannot answer, which is an observation about the channel and not a new kind of
#: source.
SOURCE_BY_ACTION: Mapping[str, str] = MappingProxyType(
    {"ask_user": "user_answer", "call_tool": "tool_output"}
)

#: The entry was projected from a version-1 card's
#: ``facts.<key>.elicitation_action``. Recorded rather than erased, because the
#: projection is a claim this build makes about a card that never stated it, and
#: a reader has to be able to tell it from something an author wrote.
ORIGIN_PROJECTED_FACT = "projected_fact"
#: The entry was written by a card author. Every non-reveal outcome is one of
#: these, always: nothing infers a non-reveal outcome from text.
ORIGIN_AUTHORED = "authored"
QUERY_ORIGINS: tuple[str, ...] = (ORIGIN_AUTHORED, ORIGIN_PROJECTED_FACT)


class QueryResolutionError(ValueError):
    """The registry, or one of its entries, is not a contract this build can run."""


@dataclass(frozen=True)
class QueryEntry:
    """One authored resolution: a query key, its channel, and its outcome.

    Every field is a string from a closed set or a key name, so the whole entry
    is immutable by construction — there is no container here for a value, a
    disposition, a reason code or a policy label to be smuggled in through, which
    is what keeps a non-reveal answer from carrying evidence it has not got.
    """

    query_key: str
    action: str
    outcome: str
    fact_key: str | None
    origin: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_key, str) or not self.query_key.strip():
            raise QueryResolutionError(
                f"a query key must be a non-empty string, got {self.query_key!r}"
            )
        if self.action not in QUERY_ACTIONS:
            raise QueryResolutionError(
                f"query {self.query_key!r} names acquisition channel "
                f"{self.action!r}, which is not one this protocol offers; allowed: "
                f"{list(QUERY_ACTIONS)}"
            )
        if self.outcome not in QUERY_OUTCOMES:
            raise QueryResolutionError(
                f"query {self.query_key!r} declares outcome {self.outcome!r}; this "
                f"contract resolves exactly {list(QUERY_OUTCOMES)}"
            )
        if self.origin not in QUERY_ORIGINS:
            raise QueryResolutionError(
                f"query {self.query_key!r} declares origin {self.origin!r}; allowed: "
                f"{list(QUERY_ORIGINS)}"
            )
        if self.reveals:
            if not isinstance(self.fact_key, str) or not self.fact_key.strip():
                raise QueryResolutionError(
                    f"query {self.query_key!r} resolves to {OUTCOME_REVEAL!r} and "
                    "names no fact to reveal; a reveal is exactly one existing "
                    "authored fact and nothing else"
                )
        elif self.fact_key is not None:
            raise QueryResolutionError(
                f"query {self.query_key!r} resolves to {self.outcome!r} and names "
                f"fact {self.fact_key!r}. Only {OUTCOME_REVEAL!r} carries a fact: a "
                "non-reveal answer states the query key and the outcome, so that it "
                "can never be read back as evidence of a value nobody revealed"
            )

    @property
    def reveals(self) -> bool:
        return self.outcome in REVEALING_OUTCOMES

    @property
    def source(self) -> str:
        """The observation source this entry's answer arrives from."""
        return SOURCE_BY_ACTION[self.action]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_key": self.query_key,
            "action": self.action,
            "outcome": self.outcome,
            "fact_key": self.fact_key,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class QueryRegistry:
    """Every query one Cube offers, in one immutable, canonically ordered object.

    Cube-level and invariant: the same registry is attached to all four cells and
    both pressure probes, so what may be asked, and what asking returns, cannot
    be read as a signal about which arm a variant is running.

    Entries are sorted by query key at construction, so the offered enum is fixed
    by the Cube rather than by the order its card happened to declare things in,
    and two cards with the same content compile to the same bytes.
    """

    entries: tuple[QueryEntry, ...] = ()
    by_key: Mapping[str, QueryEntry] = field(
        default_factory=lambda: MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        for entry in entries:
            if not isinstance(entry, QueryEntry):
                raise QueryResolutionError(
                    f"a query registry holds QueryEntry values, got "
                    f"{type(entry).__name__}"
                )
        ordered = tuple(sorted(entries, key=lambda item: item.query_key))
        seen: dict[str, QueryEntry] = {}
        for entry in ordered:
            if entry.query_key in seen:
                raise QueryResolutionError(
                    f"duplicate query key {entry.query_key!r}: a Cube offers each "
                    "query on exactly one channel with exactly one outcome, so two "
                    "entries for one key describe no resolution this build can make"
                )
            seen[entry.query_key] = entry
        object.__setattr__(self, "entries", ordered)
        object.__setattr__(self, "by_key", MappingProxyType(seen))

    def entry(self, query_key: str) -> QueryEntry | None:
        """The resolution for this key, or ``None`` when the Cube offers none."""
        return self.by_key.get(query_key)

    def channel_for(self, query_key: str) -> str | None:
        """Which channel this Cube accepts ``query_key`` on, if it accepts it."""
        entry = self.by_key.get(query_key)
        return None if entry is None else entry.action

    def affordances(self) -> Mapping[str, tuple[str, ...]]:
        """The offered query keys per channel: the enum, and the one derivation.

        Only channels that actually carry a query appear. A channel a Cube allows
        and offers nothing on is a case that cannot be executed as authored, and
        it is refused where the action surface is fixed rather than papered over
        with an empty enum here.

        Immutable at both levels, because this is handed to the projection that
        builds a model-facing interface out of it.
        """
        keys: dict[str, list[str]] = {}
        for entry in self.entries:
            keys.setdefault(entry.action, []).append(entry.query_key)
        return MappingProxyType(
            {action: tuple(found) for action, found in sorted(keys.items())}
        )

    def revealed_fact_keys(self) -> tuple[str, ...]:
        """Every fact key some query reveals, sorted and deduplicated."""
        return tuple(sorted({entry.fact_key for entry in self.entries if entry.fact_key}))

    def as_dict(self) -> dict[str, Any]:
        """The rendering that enters compiled content and therefore identity.

        The contract name travels with the entries deliberately: a build that
        resolved the same entries under different rules is not the same
        experiment, and a digest that covered only the entries could not say so.
        """
        return {
            "contract": QUERY_RESOLUTION_CONTRACT,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def registry_from_entries(entries: Iterable[QueryEntry]) -> QueryRegistry:
    """Build a registry from any iterable of entries. Ordering is not the caller's."""
    return QueryRegistry(entries=tuple(entries))


__all__ = [
    "ORIGIN_AUTHORED",
    "ORIGIN_PROJECTED_FACT",
    "OUTCOME_NOT_RECORDED",
    "OUTCOME_OUT_OF_SCOPE",
    "OUTCOME_REVEAL",
    "OUTCOME_UNKNOWN",
    "QUERY_ACTIONS",
    "QUERY_ORIGINS",
    "QUERY_OUTCOMES",
    "QUERY_RESOLUTION_CONTRACT",
    "REVEALING_OUTCOMES",
    "SOURCE_BY_ACTION",
    "QueryEntry",
    "QueryRegistry",
    "QueryResolutionError",
    "registry_from_entries",
]
