"""Machine-readable ConstructCard schema for a binary Sparse Autonomy Cube.

The card is the human-authored ground truth. Everything downstream (compiler,
environment, evaluator, solvers) is derived from it, so the schema layer is
deliberately strict: unknown keys, unknown enum members and dangling references
are errors rather than warnings.

This slice supports only the binary ACT/STOP ontology on a 2x2 state x policy
table. Richer operational dispositions (ASK, WAIT, REQUEST_APPROVAL, ESCALATE,
REFUSE) are explicitly out of scope here.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from boundarybench import jsonsafe
from boundarybench.freezing import deep_freeze
from boundarybench.queries import (
    ORIGIN_AUTHORED,
    ORIGIN_PROJECTED_FACT,
    OUTCOME_REVEAL,
    QUERY_ACTIONS,
    QUERY_OUTCOMES,
    QueryEntry,
    QueryRegistry,
    QueryResolutionError,
)

#: The card format this build authors. ``2`` added the authored
#: ``query_resolution`` registry: a Cube may now state, for a query key it
#: offers, that the answer is ``unknown``, ``not_recorded`` or ``out_of_scope``
#: rather than a fact. Version 1 cards remain executable and mean exactly what
#: they meant — every ``facts.<key>.elicitation_action`` projects to an implicit
#: ``reveal`` entry — but they may not author the section, because a build that
#: read a registry out of a card claiming version 1 would be reading a contract
#: that version does not have.
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
#: The first version whose cards may author a query-resolution registry.
QUERY_RESOLUTION_SCHEMA_VERSION = 2

STATE_KEYS: tuple[str, ...] = ("S0", "S1")
POLICY_KEYS: tuple[str, ...] = ("P0", "P1")
DISPOSITIONS: frozenset[str] = frozenset({"ACT", "STOP"})
#: How many of the four ``state x policy`` cells each cube type must call ACT.
#: The type is the whole balance contract: a `1-ACT` Cube is mostly boundary and
#: a `3-ACT` Cube is mostly permission, and a card cannot be authored with the
#: other shape's table. Richer tables (2-ACT diagonals) are admissible under
#: neither, which is what keeps a Cube's answer pattern from being guessable.
EXPECTED_ACT_CELLS: Mapping[str, int] = MappingProxyType({"1-ACT": 1, "3-ACT": 3})
#: Derived, so the vocabulary and the balance rule can never drift apart.
CUBE_TYPES: frozenset[str] = frozenset(EXPECTED_ACT_CELLS)
#: Likewise, only the two fact-location mechanics the environment can execute.
#: 'tool_mediated' and 'runtime_emergent' need environment work that does not
#: exist yet.
FACT_LOCATIONS: frozenset[str] = frozenset({"initial_state", "elicited"})
#: Only synthetic material is admissible in this slice.
PRIVACY_STATUSES: frozenset[str] = frozenset({"SYNTHETIC_ONLY", "SYNTHETIC_POLICY"})
AVAILABILITIES: frozenset[str] = frozenset({"initial", "on_request"})
#: Derived from the resolution contract rather than restated, so the channels a
#: fact may arrive on and the channels a query may be asked on are one set.
ELICITATION_ACTIONS: frozenset[str] = frozenset(QUERY_ACTIONS)
CONDITION_OPS: frozenset[str] = frozenset({"eq", "ne", "lte", "lt", "gte", "gt"})

#: A cube id becomes the namespace of every variant id, so it must stay free of
#: the '::' separator, whitespace and case ambiguity.
CUBE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: The action every card needs in order to see its own records.
READ_ACTION = "read_records"
#: The terminal call.
TERMINAL_ACTION = "complete_case"

#: Actions the protocol itself owns. They move information, never state, so
#: none of them may double as an irreversible action.
PROTOCOL_ACTIONS: frozenset[str] = frozenset(
    {READ_ACTION, TERMINAL_ACTION} | ELICITATION_ACTIONS
)

#: Invariant ids the evaluator knows how to check. The compiler refuses cards
#: that name an invariant with no executable checker behind it.
KNOWN_INVARIANTS: tuple[str, ...] = (
    "NO_IRREVERSIBLE_BEFORE_REQUIRED_EVIDENCE",
    "NO_IRREVERSIBLE_ACTION_WHEN_STOP",
    "NO_FABRICATED_EVIDENCE_REFERENCE",
)

_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "cube_id",
    "domain",
    "construct_family",
    "workflow",
    "cube_type",
    "fact_location",
    "privacy_status",
    "disclaimer",
    "why_it_matters",
    "reason_code_precedence",
    "state_axis",
    "policy_axis",
    "disposition_table",
    "facts",
    "pressure_probes",
    "allowed_actions",
    "irreversible_actions",
    "critical_invariants",
)

_OPTIONAL_KEYS: tuple[str, ...] = ("distractors", "query_resolution")

#: The fields one authored query-resolution entry may state. ``fact`` is present
#: only on a ``reveal``; there is deliberately no field for a value, a label, a
#: disposition or a reason, so a non-reveal entry has nowhere to put one.
_QUERY_ENTRY_KEYS: tuple[str, ...] = ("query_key", "action", "outcome", "fact")


class SchemaError(ValueError):
    """Raised when a construct card is structurally invalid."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SchemaError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _require_text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _require(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context}: field {key!r} must be a non-empty string")
    return value


def _require_mapping(
    mapping: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = _require(mapping, key, context)
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context}: field {key!r} must be a mapping")
    return value


def _require_sequence(
    mapping: Mapping[str, Any], key: str, context: str
) -> Sequence[Any]:
    value = _require(mapping, key, context)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{context}: field {key!r} must be a list")
    return value


def _render_keys(keys: Iterable[Any]) -> list[str]:
    """Render a key collection that may mix types, without sorting them."""
    return sorted(repr(key) for key in keys)


def _require_string_keys(raw: Mapping[Any, Any], context: str) -> None:
    """Authored mappings are keyed by strings; anything else is a mistake."""
    offenders = [key for key in raw if not isinstance(key, str)]
    if offenders:
        raise SchemaError(
            f"{context}: mapping keys must be strings, got {_render_keys(offenders)}"
        )


def _reject_unknown_keys(
    raw: Mapping[str, Any], allowed: Iterable[str], context: str
) -> None:
    _require_string_keys(raw, context)
    unknown = set(raw) - set(allowed)
    if unknown:
        raise SchemaError(
            f"{context}: unknown keys {_render_keys(unknown)}; allowed: {sorted(allowed)}"
        )


def _require_string_list(
    mapping: Mapping[str, Any], key: str, context: str, *, unique: bool = True
) -> tuple[str, ...]:
    """A non-empty list of non-empty strings, optionally without duplicates."""
    values = _require_sequence(mapping, key, context)
    if not values:
        raise SchemaError(f"{context}: {key!r} must not be empty")
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise SchemaError(
                f"{context}: {key}[{index}] must be a non-empty string, got {item!r}"
            )
    if unique and len(set(values)) != len(values):
        duplicates = sorted({v for v in values if list(values).count(v) > 1})
        raise SchemaError(f"{context}: {key!r} contains duplicates {duplicates}")
    return tuple(values)


def ensure_json_safe(value: Any, context: str) -> Any:
    """Reject authored values JSON cannot represent exactly.

    YAML happily produces dates, timestamps, non-finite floats, self-referential
    containers through anchors, and strings holding a lone UTF-16 surrogate.
    None of those survive a canonical digest or ``compile --json`` intact, so
    they are refused at the door rather than coerced later.

    The walk itself lives in :mod:`boundarybench.jsonsafe`, because a card is not
    the only artefact that has to answer this question and two implementations of
    "JSON-safe" would eventually disagree about one. This wrapper only restates
    the answer as a :class:`SchemaError`, so a card author still gets a card
    error.
    """
    try:
        return jsonsafe.ensure_json_safe(value, context)
    except jsonsafe.JsonSafetyError as exc:
        raise SchemaError(str(exc)) from exc


def _cell_key(state: str, policy: str) -> str:
    return f"{state}_{policy}"


@dataclass(frozen=True)
class Condition:
    """A single atomic test against an operational fact."""

    fact: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        # Frozen here rather than only in from_dict, so a directly constructed
        # condition carries the same immutable semantics as an authored one.
        object.__setattr__(self, "value", deep_freeze(self.value))

    def holds(self, facts: Mapping[str, Any]) -> bool:
        if self.fact not in facts:
            return False
        observed = facts[self.fact]
        if self.op == "eq":
            return bool(observed == self.value)
        if self.op == "ne":
            return bool(observed != self.value)
        try:
            if self.op == "lt":
                return bool(observed < self.value)
            if self.op == "lte":
                return bool(observed <= self.value)
            if self.op == "gt":
                return bool(observed > self.value)
            if self.op == "gte":
                return bool(observed >= self.value)
        except TypeError as exc:
            raise SchemaError(
                f"condition {self.fact!r} {self.op} {self.value!r} compares "
                f"incompatible types: fact value is "
                f"{type(observed).__name__}, declared value is "
                f"{type(self.value).__name__}"
            ) from exc
        raise SchemaError(f"unknown condition operator {self.op!r}")

    @classmethod
    def from_dict(cls, raw: Any, context: str) -> Condition:
        if not isinstance(raw, Mapping):
            raise SchemaError(f"{context}: condition must be a mapping")
        fact = _require_text(raw, "fact", context)
        op = _require_text(raw, "op", context)
        if op not in CONDITION_OPS:
            raise SchemaError(
                f"{context}: unknown condition operator {op!r}; "
                f"allowed: {sorted(CONDITION_OPS)}"
            )
        if "value" not in raw:
            raise SchemaError(f"{context}: condition must declare 'value'")
        unknown = set(raw) - {"fact", "op", "value"}
        if unknown:
            raise SchemaError(
                f"{context}: unknown condition keys {_render_keys(unknown)}"
            )
        ensure_json_safe(raw["value"], f"{context}.value")
        return cls(fact=fact, op=op, value=raw["value"])


@dataclass(frozen=True)
class Rule:
    """One clause of a versioned operational policy. First match wins."""

    id: str
    conditions: tuple[Condition, ...]
    decision: str
    primary_reason: str

    def matches(self, facts: Mapping[str, Any]) -> bool:
        return all(condition.holds(facts) for condition in self.conditions)

    @property
    def referenced_facts(self) -> tuple[str, ...]:
        return tuple(condition.fact for condition in self.conditions)

    @classmethod
    def from_dict(cls, raw: Any, context: str) -> Rule:
        if not isinstance(raw, Mapping):
            raise SchemaError(f"{context}: rule must be a mapping")
        unknown = set(raw) - {"id", "when", "decision", "primary_reason"}
        if unknown:
            raise SchemaError(f"{context}: unknown rule keys {_render_keys(unknown)}")
        rule_id = _require_text(raw, "id", context)
        decision = _require_text(raw, "decision", context)
        if decision not in DISPOSITIONS:
            raise SchemaError(
                f"{context}: rule {rule_id!r} declares disposition {decision!r}; "
                f"this slice supports only {sorted(DISPOSITIONS)}"
            )
        conditions = tuple(
            Condition.from_dict(item, f"{context}.{rule_id}.when[{index}]")
            for index, item in enumerate(_require_sequence(raw, "when", context))
        )
        return cls(
            id=rule_id,
            conditions=conditions,
            decision=decision,
            primary_reason=_require_text(raw, "primary_reason", context),
        )


@dataclass(frozen=True)
class StateArm:
    key: str
    label: str
    description: str
    facts: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyArm:
    key: str
    label: str
    description: str
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class FactSpec:
    """Where a deciding fact lives and how the agent may obtain it."""

    key: str
    availability: str
    elicitation_action: str | None
    question: str | None


@dataclass(frozen=True)
class PressureProbeSpec:
    direction: str
    base_cell: str
    cues: tuple[str, ...]


@dataclass(frozen=True)
class ConstructCard:
    """Human-authored ground truth for a single Sparse Autonomy Cube."""

    schema_version: int
    cube_id: str
    domain: str
    construct_family: str
    workflow: str
    cube_type: str
    fact_location: str
    privacy_status: str
    disclaimer: str
    why_it_matters: str
    reason_code_precedence: tuple[str, ...]
    state_axis: Mapping[str, StateArm]
    policy_axis: Mapping[str, PolicyArm]
    disposition_table: Mapping[str, str]
    facts: Mapping[str, FactSpec]
    distractors: tuple[Mapping[str, Any], ...]
    pressure_probes: Mapping[str, PressureProbeSpec]
    #: Every query this Cube resolves, and how. Cube-level and invariant across
    #: cells and probes; see :mod:`boundarybench.queries`.
    query_registry: QueryRegistry
    allowed_actions: tuple[str, ...]
    irreversible_actions: tuple[str, ...]
    critical_invariants: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    # -- accessors ---------------------------------------------------------

    def disposition(self, state: str, policy: str) -> str:
        return self.disposition_table[_cell_key(state, policy)]

    def facts_for(self, state: str) -> Mapping[str, Any]:
        return self.state_axis[state].facts

    def reason_rank(self, reason_code: str) -> int:
        return self.reason_code_precedence.index(reason_code)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConstructCard:
        if not isinstance(raw, Mapping):
            raise SchemaError("construct card must be a mapping")
        raw = copy.deepcopy(dict(raw))
        context = "construct_card"

        _require_string_keys(raw, context)
        unknown = set(raw) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)
        if unknown:
            raise SchemaError(
                f"{context}: unknown top-level keys {_render_keys(unknown)}"
            )

        version = _require(raw, "schema_version", context)
        # bool is an int subclass, so an exact type check is what keeps
        # `schema_version: true` from being read as version 1.
        if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaError(
                f"{context}: unsupported schema_version {version!r}; "
                f"this build supports {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        cube_id = _require_text(raw, "cube_id", context)
        if not CUBE_ID_PATTERN.fullmatch(cube_id):
            raise SchemaError(
                f"{context}: cube_id {cube_id!r} must start with a lowercase "
                "letter or digit and use only lowercase letters, digits, "
                "underscores and hyphens"
            )
        domain = _require_text(raw, "domain", context)
        construct_family = _require_text(raw, "construct_family", context)
        workflow = _require_text(raw, "workflow", context)

        cube_type = _require_text(raw, "cube_type", context)
        if cube_type not in CUBE_TYPES:
            raise SchemaError(
                f"{context}: unknown cube_type {cube_type!r}; "
                f"allowed: {sorted(CUBE_TYPES)}"
            )

        fact_location = _require_text(raw, "fact_location", context)
        if fact_location not in FACT_LOCATIONS:
            raise SchemaError(
                f"{context}: unknown fact_location {fact_location!r}; "
                f"allowed: {sorted(FACT_LOCATIONS)}"
            )

        privacy_status = _require_text(raw, "privacy_status", context)
        if privacy_status not in PRIVACY_STATUSES:
            raise SchemaError(
                f"{context}: privacy_status {privacy_status!r} is not admissible; "
                f"this slice accepts only {sorted(PRIVACY_STATUSES)}"
            )

        disclaimer = _require_text(raw, "disclaimer", context)
        why_it_matters = _require_text(raw, "why_it_matters", context)

        precedence = _require_string_list(raw, "reason_code_precedence", context)

        state_axis = cls._parse_state_axis(raw, context)
        facts = cls._parse_facts(raw, state_axis, context)
        policy_axis = cls._parse_policy_axis(raw, precedence, facts, context)
        disposition_table = cls._parse_disposition_table(raw, context)
        pressure_probes = cls._parse_pressure_probes(raw, disposition_table, context)

        allowed_actions = _require_string_list(raw, "allowed_actions", context)
        irreversible_actions = _require_string_list(raw, "irreversible_actions", context)
        conflicting = sorted(set(irreversible_actions) & PROTOCOL_ACTIONS)
        if conflicting:
            raise SchemaError(
                f"{context}: irreversible_actions {conflicting} name protocol "
                "actions; reading records, eliciting a fact and closing the case "
                "move information, not state, so they can never be irreversible"
            )
        dangling = [a for a in irreversible_actions if a not in allowed_actions]
        if dangling:
            raise SchemaError(
                f"{context}: irreversible_actions {dangling} are not in allowed_actions"
            )
        if TERMINAL_ACTION not in allowed_actions:
            raise SchemaError(
                f"{context}: allowed_actions must include {TERMINAL_ACTION!r}"
            )
        if READ_ACTION not in allowed_actions:
            raise SchemaError(
                f"{context}: allowed_actions must include {READ_ACTION!r}; the "
                "active policy and the initial record can only be reached that way"
            )
        missing_channels = sorted(
            {
                spec.elicitation_action
                for spec in facts.values()
                if spec.elicitation_action is not None
                and spec.elicitation_action not in allowed_actions
            }
        )
        if missing_channels:
            raise SchemaError(
                f"{context}: facts are declared to arrive through "
                f"{missing_channels}, but those actions are not in allowed_actions; "
                "the card could not be executed"
            )

        invariants = _require_string_list(raw, "critical_invariants", context)
        unknown_invariants = [i for i in invariants if i not in KNOWN_INVARIANTS]
        if unknown_invariants:
            raise SchemaError(
                f"{context}: no executable checker for critical_invariants "
                f"{unknown_invariants}; known: {list(KNOWN_INVARIANTS)}"
            )

        distractors = cls._parse_distractors(raw, facts, context)
        query_registry = cls._parse_query_resolution(
            raw,
            facts=facts,
            distractors=distractors,
            allowed_actions=allowed_actions,
            version=version,
            context=context,
        )

        return cls(
            schema_version=version,
            cube_id=cube_id,
            domain=domain,
            construct_family=construct_family,
            workflow=workflow,
            cube_type=cube_type,
            fact_location=fact_location,
            privacy_status=privacy_status,
            disclaimer=disclaimer,
            why_it_matters=why_it_matters,
            reason_code_precedence=precedence,
            state_axis=MappingProxyType(state_axis),
            policy_axis=MappingProxyType(policy_axis),
            disposition_table=MappingProxyType(disposition_table),
            facts=MappingProxyType(facts),
            distractors=distractors,
            pressure_probes=MappingProxyType(pressure_probes),
            query_registry=query_registry,
            allowed_actions=allowed_actions,
            irreversible_actions=irreversible_actions,
            critical_invariants=invariants,
            raw=deep_freeze(raw),
        )

    # -- section parsers ---------------------------------------------------

    @staticmethod
    def _parse_state_axis(raw: Mapping[str, Any], context: str) -> dict[str, StateArm]:
        section = _require_mapping(raw, "state_axis", context)
        _require_string_keys(section, f"{context}.state_axis")
        if tuple(sorted(section)) != STATE_KEYS:
            raise SchemaError(
                f"{context}: state_axis must declare exactly {list(STATE_KEYS)}, "
                f"got {sorted(section)}"
            )
        arms: dict[str, StateArm] = {}
        for key in STATE_KEYS:
            arm_context = f"{context}.state_axis.{key}"
            arm = section[key]
            if not isinstance(arm, Mapping):
                raise SchemaError(f"{arm_context}: must be a mapping")
            _reject_unknown_keys(arm, ("label", "description", "facts"), arm_context)
            facts = _require_mapping(arm, "facts", arm_context)
            if not facts:
                raise SchemaError(f"{arm_context}: facts must not be empty")
            for fact_key, fact_value in facts.items():
                if not isinstance(fact_key, str) or not fact_key.strip():
                    raise SchemaError(
                        f"{arm_context}: fact keys must be non-empty strings, "
                        f"got {fact_key!r}"
                    )
                ensure_json_safe(fact_value, f"{arm_context}.facts.{fact_key}")
            arms[key] = StateArm(
                key=key,
                label=_require_text(arm, "label", arm_context),
                description=_require_text(arm, "description", arm_context),
                facts=deep_freeze(facts),
            )
        if set(arms["S0"].facts) != set(arms["S1"].facts):
            raise SchemaError(
                f"{context}: state_axis arms must declare the same fact keys "
                "so that variants stay symmetric before the causal reveal"
            )
        if dict(arms["S0"].facts) == dict(arms["S1"].facts):
            raise SchemaError(f"{context}: state_axis arms S0 and S1 are identical")
        return arms

    @staticmethod
    def _parse_facts(
        raw: Mapping[str, Any], state_axis: Mapping[str, StateArm], context: str
    ) -> dict[str, FactSpec]:
        section = _require_mapping(raw, "facts", context)
        _require_string_keys(section, f"{context}.facts")
        declared = set(state_axis["S0"].facts)
        if set(section) != declared:
            raise SchemaError(
                f"{context}: facts must describe exactly the state_axis fact keys "
                f"{sorted(declared)}, got {sorted(section)}"
            )
        specs: dict[str, FactSpec] = {}
        for key, spec in section.items():
            spec_context = f"{context}.facts.{key}"
            if not isinstance(spec, Mapping):
                raise SchemaError(f"{spec_context}: must be a mapping")
            _reject_unknown_keys(
                spec, ("availability", "elicitation_action", "question"), spec_context
            )
            availability = _require_text(spec, "availability", spec_context)
            if availability not in AVAILABILITIES:
                raise SchemaError(
                    f"{spec_context}: unknown availability {availability!r}; "
                    f"allowed: {sorted(AVAILABILITIES)}"
                )
            elicitation_action = spec.get("elicitation_action")
            if availability == "on_request":
                if elicitation_action not in ELICITATION_ACTIONS:
                    raise SchemaError(
                        f"{spec_context}: on_request facts need an elicitation_action "
                        f"in {sorted(ELICITATION_ACTIONS)}"
                    )
            elif elicitation_action is not None:
                raise SchemaError(
                    f"{spec_context}: initial facts must not declare an "
                    "elicitation_action"
                )
            question = spec.get("question")
            if elicitation_action == "ask_user":
                # A user-elicited fact is worthless without the question to ask.
                question = _require_text(spec, "question", spec_context)
            elif question is not None and (
                not isinstance(question, str) or not question.strip()
            ):
                raise SchemaError(
                    f"{spec_context}: 'question' must be a non-empty string when present"
                )
            specs[key] = FactSpec(
                key=key,
                availability=availability,
                elicitation_action=elicitation_action,
                question=question,
            )
        return specs

    @staticmethod
    def _parse_policy_axis(
        raw: Mapping[str, Any],
        precedence: Sequence[str],
        facts: Mapping[str, FactSpec],
        context: str,
    ) -> dict[str, PolicyArm]:
        section = _require_mapping(raw, "policy_axis", context)
        _require_string_keys(section, f"{context}.policy_axis")
        if tuple(sorted(section)) != POLICY_KEYS:
            raise SchemaError(
                f"{context}: policy_axis must declare exactly {list(POLICY_KEYS)}, "
                f"got {sorted(section)}"
            )
        seen_rule_ids: set[str] = set()
        arms: dict[str, PolicyArm] = {}
        for key in POLICY_KEYS:
            arm_context = f"{context}.policy_axis.{key}"
            arm = section[key]
            if not isinstance(arm, Mapping):
                raise SchemaError(f"{arm_context}: must be a mapping")
            _reject_unknown_keys(arm, ("label", "description", "rules"), arm_context)
            rules = tuple(
                Rule.from_dict(item, arm_context)
                for item in _require_sequence(arm, "rules", arm_context)
            )
            if not rules:
                raise SchemaError(f"{arm_context}: rules must not be empty")
            for rule in rules:
                if rule.id in seen_rule_ids:
                    raise SchemaError(f"{arm_context}: duplicate rule id {rule.id!r}")
                seen_rule_ids.add(rule.id)
                if rule.primary_reason not in precedence:
                    raise SchemaError(
                        f"{arm_context}: rule {rule.id!r} uses reason code "
                        f"{rule.primary_reason!r} which is absent from "
                        "reason_code_precedence"
                    )
                for fact_key in rule.referenced_facts:
                    if fact_key not in facts:
                        raise SchemaError(
                            f"{arm_context}: rule {rule.id!r} references unknown fact "
                            f"{fact_key!r}"
                        )
            arms[key] = PolicyArm(
                key=key,
                label=_require_text(arm, "label", arm_context),
                description=_require_text(arm, "description", arm_context),
                rules=rules,
            )
        return arms

    @staticmethod
    def _parse_distractors(
        raw: Mapping[str, Any], facts: Mapping[str, FactSpec], context: str
    ) -> tuple[Mapping[str, Any], ...]:
        section = raw.get("distractors")
        if section is None:
            return ()
        if isinstance(section, (str, bytes)) or not isinstance(section, Sequence):
            raise SchemaError(f"{context}: distractors must be a list")

        parsed: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(section):
            item_context = f"{context}.distractors[{index}]"
            if not isinstance(item, Mapping):
                raise SchemaError(
                    f"{item_context}: each distractor must be a mapping with a "
                    "'key' and a 'value'"
                )
            unknown = set(item) - {"key", "value"}
            if unknown:
                raise SchemaError(
                    f"{item_context}: unknown distractor fields {_render_keys(unknown)}; "
                    "only 'key' and 'value' are allowed"
                )
            key = _require_text(item, "key", item_context)
            if "value" not in item:
                raise SchemaError(f"{item_context}: missing required field 'value'")
            ensure_json_safe(item["value"], f"{item_context}.value")
            if item["value"] is None:
                raise SchemaError(f"{item_context}: field 'value' must not be null")
            if key in seen:
                raise SchemaError(f"{item_context}: duplicate distractor key {key!r}")
            if key in facts:
                raise SchemaError(
                    f"{item_context}: distractor key {key!r} collides with a "
                    "declared deciding fact; a distractor must never carry a fact "
                    "the policy can rest on"
                )
            seen.add(key)
            parsed.append(
                MappingProxyType({"key": key, "value": deep_freeze(item["value"])})
            )
        return tuple(parsed)

    @staticmethod
    def _parse_query_resolution(
        raw: Mapping[str, Any],
        *,
        facts: Mapping[str, FactSpec],
        distractors: Sequence[Mapping[str, Any]],
        allowed_actions: Sequence[str],
        version: int,
        context: str,
    ) -> QueryRegistry:
        """The Cube's whole query-resolution contract: projection plus authoring.

        Two things go in and one comes out. Every ``on_request`` fact projects to
        an implicit ``reveal`` entry whose query key is the fact key — that is
        what a version-1 card always meant, stated explicitly for the first time
        — and every authored entry is added beside it, unchanged and unguessed.

        Nothing infers a non-reveal outcome. A card that wants a query to answer
        ``unknown`` says so; there is no text classifier, no fallback and no
        heuristic that can produce one, which is the whole reason the registry
        exists rather than a simulator.
        """
        entries: list[QueryEntry] = [
            QueryEntry(
                query_key=key,
                action=str(spec.elicitation_action),
                outcome=OUTCOME_REVEAL,
                fact_key=key,
                origin=ORIGIN_PROJECTED_FACT,
            )
            for key, spec in sorted(facts.items())
            if spec.elicitation_action is not None
        ]
        projected = {entry.query_key for entry in entries}

        section = raw.get("query_resolution")
        if section is not None:
            if version < QUERY_RESOLUTION_SCHEMA_VERSION:
                raise SchemaError(
                    f"{context}: 'query_resolution' is a schema_version "
                    f"{QUERY_RESOLUTION_SCHEMA_VERSION} section, and this card "
                    f"declares schema_version {version}. A version-{version} card's "
                    "queries are exactly its elicitable facts; authoring other "
                    "outcomes is a different contract and says so in its version"
                )
            if isinstance(section, (str, bytes)) or not isinstance(section, Sequence):
                raise SchemaError(f"{context}: query_resolution must be a list")
            reserved = {str(item["key"]) for item in distractors}
            for index, item in enumerate(section):
                entry_context = f"{context}.query_resolution[{index}]"
                if not isinstance(item, Mapping):
                    raise SchemaError(f"{entry_context}: must be a mapping")
                _reject_unknown_keys(item, _QUERY_ENTRY_KEYS, entry_context)
                query_key = _require_text(item, "query_key", entry_context)
                action = _require_text(item, "action", entry_context)
                outcome = _require_text(item, "outcome", entry_context)
                if outcome not in QUERY_OUTCOMES:
                    raise SchemaError(
                        f"{entry_context}: unknown outcome {outcome!r}; this contract "
                        f"resolves exactly {list(QUERY_OUTCOMES)}"
                    )
                if action not in allowed_actions:
                    raise SchemaError(
                        f"{entry_context}: query {query_key!r} is offered on "
                        f"{action!r}, which is not in allowed_actions "
                        f"{list(allowed_actions)}; a query the case cannot be asked "
                        "is one no run could ever resolve"
                    )
                if query_key in projected:
                    raise SchemaError(
                        f"{entry_context}: duplicate query key {query_key!r}; it is "
                        "already the implicit reveal of the fact of the same name, "
                        "and a Cube resolves each query exactly one way"
                    )
                if query_key in facts or query_key in reserved:
                    raise SchemaError(
                        f"{entry_context}: query key {query_key!r} names a fact or "
                        "distractor this card already declares; a query key is a "
                        "distinct name in the same namespace the model is shown"
                    )
                fact_key = item.get("fact")
                if outcome == OUTCOME_REVEAL:
                    if not isinstance(fact_key, str) or not fact_key.strip():
                        raise SchemaError(
                            f"{entry_context}: a {OUTCOME_REVEAL!r} query must name "
                            "the existing fact it reveals in its 'fact' field"
                        )
                    spec = facts.get(fact_key)
                    if spec is None:
                        raise SchemaError(
                            f"{entry_context}: query {query_key!r} reveals fact "
                            f"{fact_key!r}, which this card does not declare"
                        )
                    if spec.elicitation_action != action:
                        raise SchemaError(
                            f"{entry_context}: query {query_key!r} reveals fact "
                            f"{fact_key!r} through channel {action!r}, but the fact "
                            f"is assigned to {spec.elicitation_action!r}; a fact "
                            "arrives on exactly one channel"
                        )
                elif fact_key is not None:
                    raise SchemaError(
                        f"{entry_context}: outcome {outcome!r} names fact "
                        f"{fact_key!r}. Only {OUTCOME_REVEAL!r} carries a fact value; "
                        "every other outcome answers with the query key and the "
                        "outcome alone"
                    )
                entries.append(
                    QueryEntry(
                        query_key=query_key,
                        action=action,
                        outcome=outcome,
                        fact_key=fact_key if outcome == OUTCOME_REVEAL else None,
                        origin=ORIGIN_AUTHORED,
                    )
                )

        try:
            return QueryRegistry(entries=tuple(entries))
        except QueryResolutionError as exc:
            raise SchemaError(f"{context}.query_resolution: {exc}") from exc

    @staticmethod
    def _parse_disposition_table(raw: Mapping[str, Any], context: str) -> dict[str, str]:
        section = _require_mapping(raw, "disposition_table", context)
        _require_string_keys(section, f"{context}.disposition_table")
        expected = {_cell_key(s, p) for s in STATE_KEYS for p in POLICY_KEYS}
        if set(section) != expected:
            raise SchemaError(
                f"{context}: disposition_table must cover exactly {sorted(expected)}, "
                f"got {sorted(section)}"
            )
        table: dict[str, str] = {}
        for cell, value in section.items():
            if not isinstance(value, str):
                raise SchemaError(
                    f"{context}: disposition_table[{cell}] must be a string, got "
                    f"{type(value).__name__}"
                )
            if value not in DISPOSITIONS:
                raise SchemaError(
                    f"{context}: disposition_table[{cell}] is {value!r}; this slice "
                    f"supports only the binary ontology {sorted(DISPOSITIONS)}"
                )
            table[cell] = value
        return table

    @staticmethod
    def _parse_pressure_probes(
        raw: Mapping[str, Any], table: Mapping[str, str], context: str
    ) -> dict[str, PressureProbeSpec]:
        section = _require_mapping(raw, "pressure_probes", context)
        _require_string_keys(section, f"{context}.pressure_probes")
        if set(section) != {"alarmist", "reassurance"}:
            raise SchemaError(
                f"{context}: pressure_probes must declare exactly "
                f"['alarmist', 'reassurance'], got {sorted(section)}"
            )
        required_disposition = {"alarmist": "ACT", "reassurance": "STOP"}
        probes: dict[str, PressureProbeSpec] = {}
        for direction, spec in section.items():
            probe_context = f"{context}.pressure_probes.{direction}"
            if not isinstance(spec, Mapping):
                raise SchemaError(f"{probe_context}: must be a mapping")
            _reject_unknown_keys(spec, ("base_cell", "cues"), probe_context)
            base_cell = _require_text(spec, "base_cell", probe_context)
            if base_cell not in table:
                raise SchemaError(
                    f"{probe_context}: base_cell {base_cell!r} is not a table cell"
                )
            expected = required_disposition[direction]
            if table[base_cell] != expected:
                raise SchemaError(
                    f"{probe_context}: {direction} probe must be built on a correct "
                    f"{expected} cell, but {base_cell} is {table[base_cell]}"
                )
            cues = _require_string_list(spec, "cues", probe_context)
            probes[direction] = PressureProbeSpec(
                direction=direction, base_cell=base_cell, cues=cues
            )
        return probes
