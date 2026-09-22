"""Cube compiler.

Takes a validated :class:`~boundarybench.schema.ConstructCard` and emits exactly
six task variants: four ``state x policy`` cells plus one alarmist and one
reassurance pressure probe (spec §6).

Balance is a compile-time constraint, not a post-hoc report (spec §7): a card
whose declared 2x2 table contradicts its machine-readable policy, or which lacks
a live state edge, a live policy edge or the corresponding preserve edges, does
not compile.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from boundarybench.freezing import deep_freeze, to_json
from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from boundarybench.queries import QueryRegistry
from boundarybench.schema import (
    EXPECTED_ACT_CELLS,
    POLICY_KEYS,
    STATE_KEYS,
    ConstructCard,
    PolicyArm,
    Rule,
)
from boundarybench.trajectory import Observation

CELL_KIND = "cell"
PROBE_KIND = "probe"


class CompileError(ValueError):
    """Raised when a construct card is internally inconsistent or unbalanced."""


def _cell(state: str, policy: str) -> str:
    return f"{state}_{policy}"


def _canonical_digest(payload: Any) -> str:
    """Hash a payload strictly: no coercion, no non-finite floats, no surrogates.

    ``default=str`` would quietly stringify anything unexpected and hand back a
    digest for something the payload does not actually contain, so the schema
    guarantees JSON-safety up front and this stays strict. The proof is taken
    again here rather than assumed: this is the last point before bytes exist,
    and a payload the encoder cannot carry must fail as a compile error, not as
    a ``UnicodeEncodeError`` from inside a hash.
    """
    try:
        return hashlib.sha256(canonical_json_bytes(payload, "variant")).hexdigest()
    except JsonSafetyError as exc:
        raise CompileError(f"variant content cannot be digested: {exc}") from exc


@dataclass(frozen=True)
class Variant:
    """One executable task variant.

    ``variant_id`` is a stable semantic id derived from the cube and the cell or
    probe it names; it does not change when content changes. ``content_digest``
    is the content-addressed part: it covers the task surface and the grading
    contract, so any change to either changes the digest.
    """

    variant_id: str
    cube_id: str
    kind: str
    state: str
    policy: str
    base_cell: str
    pressure_direction: str | None
    pressure_cues: tuple[str, ...]
    expected_disposition: str
    expected_primary_reason: str
    matched_rule_id: str
    #: The Cube's query-resolution contract. The *same* object on every variant
    #: of a Cube — cells and probes alike — because what may be asked and what
    #: asking returns is a property of the Cube and not of the arm under
    #: measurement. It is carried on the variant rather than looked up from the
    #: card because the variant is what the environment, the adapter boundary
    #: and replay are given.
    query_registry: QueryRegistry
    observations: tuple[Observation, ...]
    required_evidence_ids: tuple[str, ...]
    expected_mutations: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    irreversible_actions: tuple[str, ...]
    critical_invariants: tuple[str, ...]
    content_digest: str

    def observation(self, observation_id: str) -> Observation | None:
        for obs in self.observations:
            if obs.observation_id == observation_id:
                return obs
        return None

    def initial_observations(self) -> tuple[Observation, ...]:
        return tuple(o for o in self.observations if o.availability == "initial")

    def on_request_observations(self) -> tuple[Observation, ...]:
        return tuple(o for o in self.observations if o.availability == "on_request")

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "cube_id": self.cube_id,
            "kind": self.kind,
            "state": self.state,
            "policy": self.policy,
            "base_cell": self.base_cell,
            "pressure_direction": self.pressure_direction,
            "pressure_cues": to_json(self.pressure_cues),
            "expected_disposition": self.expected_disposition,
            "expected_primary_reason": self.expected_primary_reason,
            "matched_rule_id": self.matched_rule_id,
            "query_registry": self.query_registry.as_dict(),
            "observations": [o.as_dict() for o in self.observations],
            "required_evidence_ids": list(self.required_evidence_ids),
            "expected_mutations": list(self.expected_mutations),
            "allowed_actions": list(self.allowed_actions),
            "irreversible_actions": list(self.irreversible_actions),
            "critical_invariants": list(self.critical_invariants),
            "content_digest": self.content_digest,
        }


def variant_content_digest(variant: Variant) -> str:
    """What a compiled variant hashes to *now*, from the fields it carries now.

    Public, and used by the compiler itself to mint the digest in the first
    place, so the two can never drift apart. It exists because ``content_digest``
    travels inside the object it describes: ``dataclasses.replace`` rebuilds a
    frozen :class:`Variant` with a different ``expected_disposition`` and the
    original pin, and a run that only compares stored fields will happily grade
    against a contract the suite never contained.

    The payload is the task surface *and* the grading contract — everything that
    decides what the agent sees, what it may do and what counts as correct.
    ``cube_id``, ``kind`` and ``base_cell`` are absent because they are naming,
    not content, and the variant id they compose is in here.
    """
    return _canonical_digest(
        {
            "variant_id": variant.variant_id,
            "state": variant.state,
            "policy": variant.policy,
            "pressure_direction": variant.pressure_direction,
            "pressure_cues": list(variant.pressure_cues),
            "expected_disposition": variant.expected_disposition,
            "expected_primary_reason": variant.expected_primary_reason,
            "matched_rule_id": variant.matched_rule_id,
            "query_registry": variant.query_registry.as_dict(),
            "observations": [o.as_dict() for o in variant.observations],
            "required_evidence_ids": list(variant.required_evidence_ids),
            "expected_mutations": list(variant.expected_mutations),
            "allowed_actions": list(variant.allowed_actions),
            "irreversible_actions": list(variant.irreversible_actions),
            "critical_invariants": list(variant.critical_invariants),
        }
    )


@dataclass(frozen=True)
class Cube:
    """A compiled Sparse Autonomy Cube."""

    cube_id: str
    card: ConstructCard
    variants: tuple[Variant, ...]
    #: The Cube-level query-resolution contract every variant carries.
    query_registry: QueryRegistry
    live_state_edges: tuple[tuple[str, str, str], ...]
    preserve_state_edges: tuple[tuple[str, str, str], ...]
    live_policy_edges: tuple[tuple[str, str, str], ...]
    preserve_policy_edges: tuple[tuple[str, str, str], ...]
    constant_strategy_scores: Mapping[str, int]

    def variant(self, variant_id: str) -> Variant:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise KeyError(variant_id)

    def cell_variant(self, cell: str) -> Variant:
        return self.variant(f"{self.cube_id}::{CELL_KIND}::{cell}")

    def probe_variant(self, direction: str) -> Variant:
        return self.variant(f"{self.cube_id}::{PROBE_KIND}::{direction}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "cube_id": self.cube_id,
            "cube_type": self.card.cube_type,
            "fact_location": self.card.fact_location,
            "privacy_status": self.card.privacy_status,
            "disclaimer": self.card.disclaimer,
            "live_state_edges": [list(e) for e in self.live_state_edges],
            "preserve_state_edges": [list(e) for e in self.preserve_state_edges],
            "live_policy_edges": [list(e) for e in self.live_policy_edges],
            "preserve_policy_edges": [list(e) for e in self.preserve_policy_edges],
            "constant_strategy_scores": to_json(self.constant_strategy_scores),
            "query_registry": self.query_registry.as_dict(),
            "variants": [v.as_dict() for v in self.variants],
        }


# -- resolution -------------------------------------------------------------


def attempted_rules(arm: PolicyArm, facts: Mapping[str, Any]) -> tuple[Rule, ...]:
    """The rules first-match evaluation walks, up to and including the winner.

    Selecting a later rule is only justified because the earlier ones were
    checked and missed, so the facts those earlier rules were evaluated against
    are part of what grounds the decision.
    """
    walked: list[Rule] = []
    for rule in arm.rules:
        walked.append(rule)
        if rule.matches(facts):
            return tuple(walked)
    return tuple(walked)


def _resolve_cell(card: ConstructCard, state: str, policy: str) -> Rule:
    """Return the rule the active policy applies to this cell, precedence-checked."""
    cell = _cell(state, policy)
    arm: PolicyArm = card.policy_axis[policy]
    facts = card.facts_for(state)

    matching = [rule for rule in arm.rules if rule.matches(facts)]
    if not matching:
        raise CompileError(
            f"cell {cell}: no rule in policy {policy} covers this state; "
            "every cell must have a determinate ground truth"
        )
    matched = matching[0]

    best = min(matching, key=lambda rule: card.reason_rank(rule.primary_reason))
    if best.primary_reason != matched.primary_reason:
        raise CompileError(
            f"cell {cell}: first matching rule {matched.id!r} reports "
            f"{matched.primary_reason!r}, but rule {best.id!r} also applies and "
            f"outranks it in precedence; reorder the rules so first-match agrees "
            "with the declared reason precedence"
        )

    declared = card.disposition(state, policy)
    if matched.decision != declared:
        raise CompileError(
            f"cell {cell}: disposition_table declares {declared!r} but rule "
            f"{matched.id!r} of policy {policy} yields {matched.decision!r}; "
            "the human table and the machine-readable policy must agree"
        )
    return matched


def _validate_edges(
    card: ConstructCard,
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    live_state, preserve_state = [], []
    for policy in POLICY_KEYS:
        a, b = _cell("S0", policy), _cell("S1", policy)
        edge = (policy, a, b)
        if card.disposition_table[a] != card.disposition_table[b]:
            live_state.append(edge)
        else:
            preserve_state.append(edge)

    live_policy, preserve_policy = [], []
    for state in STATE_KEYS:
        a, b = _cell(state, "P0"), _cell(state, "P1")
        edge = (state, a, b)
        if card.disposition_table[a] != card.disposition_table[b]:
            live_policy.append(edge)
        else:
            preserve_policy.append(edge)

    if not live_state:
        raise CompileError(
            "cube has no live state edge: the state axis never moves the decision"
        )
    if not live_policy:
        raise CompileError(
            "cube has no live policy edge: the policy axis never moves the decision"
        )
    if not preserve_state:
        raise CompileError(
            "cube has no preserve state edge: every state change crosses the boundary"
        )
    if not preserve_policy:  # pragma: no cover
        # Unreachable for a binary 2x2 table: if both rows cross the boundary
        # then the columns are either both constant (caught by the live state
        # edge check) or both live (caught by the preserve state edge check).
        # Kept for symmetry so a wider table would still be validated.
        raise CompileError(
            "cube has no preserve policy edge: every policy change crosses the boundary"
        )
    return (
        tuple(live_state),
        tuple(preserve_state),
        tuple(live_policy),
        tuple(preserve_policy),
    )


def _validate_balance(card: ConstructCard) -> None:
    act_cells = sum(1 for value in card.disposition_table.values() if value == "ACT")
    # Derived from the declared type, never hard-coded: the schema guarantees
    # the type is one the table shape is defined for.
    expected = EXPECTED_ACT_CELLS[card.cube_type]
    if act_cells != expected:
        raise CompileError(
            f"cube_type {card.cube_type!r} requires exactly {expected} ACT cell(s), "
            f"but the disposition table has {act_cells}"
        )


def _validate_fact_location(card: ConstructCard) -> None:
    availabilities = {spec.availability for spec in card.facts.values()}
    if card.fact_location == "elicited" and "on_request" not in availabilities:
        raise CompileError(
            "fact_location 'elicited' requires at least one deciding fact with "
            "availability 'on_request'; every fact here is already in context"
        )
    if card.fact_location == "initial_state" and availabilities != {"initial"}:
        raise CompileError(
            "fact_location 'initial_state' requires every deciding fact to be "
            "available from the start"
        )


# -- variant construction ---------------------------------------------------


def _build_observations(
    card: ConstructCard, state: str, policy: str, cues: tuple[str, ...]
) -> tuple[Observation, ...]:
    """Build the variant's observation set in canonical, deterministic order.

    Two sources, in one fixed order. Everything already in the record comes
    first — the active policy, the distractors, then the facts this case starts
    with — and after them, one observation per entry of the Cube's
    query-resolution registry, in the registry's own canonical order.

    That second loop is the whole of question resolution. A ``reveal`` entry
    produces the authored value of the fact it names, exactly as this compiler
    always produced it for an elicitable fact; every other outcome produces an
    observation carrying the query key and the outcome and *no* value, marked
    not evidence-relevant, so it can neither ground a decision nor be mistaken
    for one that could.
    """
    drafts: list[tuple[str, str, dict[str, Any], str, bool, str | None]] = []

    arm = card.policy_axis[policy]
    # The handle is deliberately arm-independent: the policy intervention must
    # not be readable from the observation id (spec §18.1, pre-reveal symmetry).
    drafts.append(
        (
            "active_policy",
            "applicable_policy",
            {
                "active_policy": policy,
                "policy_label": arm.label,
                "policy_description": arm.description,
                "rules": [
                    {
                        "id": rule.id,
                        "when": [
                            {"fact": c.fact, "op": c.op, "value": c.value}
                            for c in rule.conditions
                        ],
                        "decision": rule.decision,
                        "primary_reason": rule.primary_reason,
                    }
                    for rule in arm.rules
                ],
            },
            "initial",
            True,
            None,
        )
    )

    for distractor in card.distractors:
        drafts.append(
            (
                "initial_record",
                str(distractor["key"]),
                {str(distractor["key"]): distractor["value"]},
                "initial",
                False,
                None,
            )
        )

    for key in sorted(card.facts):
        spec = card.facts[key]
        if spec.availability != "initial":
            # Reached through a query, so it is the registry's to emit below.
            continue
        drafts.append(
            (
                "initial_record",
                key,
                {key: card.facts_for(state)[key]},
                "initial",
                True,
                None,
            )
        )

    for entry in card.query_registry.entries:
        if entry.reveals:
            value = card.facts_for(state)[str(entry.fact_key)]
            drafts.append(
                (
                    entry.source,
                    entry.query_key,
                    {str(entry.fact_key): value},
                    "on_request",
                    True,
                    entry.outcome,
                )
            )
            continue
        drafts.append(
            (entry.source, entry.query_key, {}, "on_request", False, entry.outcome)
        )

    if cues:
        drafts.append(
            (
                "user_statement",
                "irrelevant_pressure",
                {"statements": list(cues)},
                "initial",
                False,
                None,
            )
        )

    return tuple(
        Observation(
            observation_id=f"obs:{source}:{key}:{index:02d}",
            source=source,
            key=key,
            fact=fact,
            availability=availability,
            evidence_relevant=relevant,
            outcome=outcome,
        )
        for index, (source, key, fact, availability, relevant, outcome) in enumerate(
            drafts, 1
        )
    )


def _required_evidence_ids(
    observations: tuple[Observation, ...], attempted: tuple[Rule, ...]
) -> tuple[str, ...]:
    """Active policy plus every fact first-match evaluation had to consult.

    That is the whole chain, not just the winning rule: a later rule is only
    the right answer because the earlier ones were checked against real facts
    and missed, so those facts are part of the justification.
    """
    needed_keys = {key for rule in attempted for key in rule.referenced_facts}
    required = [o.observation_id for o in observations if o.source == "active_policy"]
    required.extend(
        o.observation_id
        for o in observations
        if o.evidence_relevant and o.key in needed_keys
    )
    return tuple(required)


def _build_variant(
    card: ConstructCard,
    *,
    kind: str,
    state: str,
    policy: str,
    matched: Rule,
    attempted: tuple[Rule, ...],
    pressure_direction: str | None,
    cues: tuple[str, ...],
    suffix: str,
) -> Variant:
    observations = _build_observations(card, state, policy, cues)
    required = _required_evidence_ids(observations, attempted)
    disposition = matched.decision
    mutations = card.irreversible_actions if disposition == "ACT" else ()

    # Built once with an empty pin, then pinned by hashing the built object. The
    # digest is therefore *by construction* what recomputing it from the
    # variant's own fields gives, so a checker and the compiler cannot disagree
    # about what a variant hashes to.
    variant = Variant(
        variant_id=f"{card.cube_id}::{kind}::{suffix}",
        cube_id=card.cube_id,
        kind=kind,
        state=state,
        policy=policy,
        base_cell=_cell(state, policy),
        pressure_direction=pressure_direction,
        pressure_cues=cues,
        expected_disposition=disposition,
        expected_primary_reason=matched.primary_reason,
        matched_rule_id=matched.id,
        query_registry=card.query_registry,
        observations=observations,
        required_evidence_ids=required,
        expected_mutations=mutations,
        allowed_actions=card.allowed_actions,
        irreversible_actions=card.irreversible_actions,
        critical_invariants=card.critical_invariants,
        content_digest="",
    )
    return replace(variant, content_digest=variant_content_digest(variant))


def compile_cube(card: ConstructCard) -> Cube:
    """Compile a construct card into six executable variants."""
    matched_rules: dict[str, Rule] = {}
    walked_rules: dict[str, tuple[Rule, ...]] = {}
    for state in STATE_KEYS:
        for policy in POLICY_KEYS:
            cell = _cell(state, policy)
            matched_rules[cell] = _resolve_cell(card, state, policy)
            walked_rules[cell] = attempted_rules(
                card.policy_axis[policy], card.facts_for(state)
            )

    live_state, preserve_state, live_policy, preserve_policy = _validate_edges(card)
    _validate_balance(card)
    _validate_fact_location(card)

    variants: list[Variant] = []
    for state in STATE_KEYS:
        for policy in POLICY_KEYS:
            cell = _cell(state, policy)
            variants.append(
                _build_variant(
                    card,
                    kind=CELL_KIND,
                    state=state,
                    policy=policy,
                    matched=matched_rules[cell],
                    attempted=walked_rules[cell],
                    pressure_direction=None,
                    cues=(),
                    suffix=cell,
                )
            )

    for direction in ("alarmist", "reassurance"):
        probe = card.pressure_probes[direction]
        state, policy = probe.base_cell.split("_")
        variants.append(
            _build_variant(
                card,
                kind=PROBE_KIND,
                state=state,
                policy=policy,
                matched=matched_rules[probe.base_cell],
                attempted=walked_rules[probe.base_cell],
                pressure_direction=direction,
                cues=probe.cues,
                suffix=direction,
            )
        )

    scores = {
        "always_act": sum(1 for v in variants if v.expected_disposition == "ACT"),
        "always_stop": sum(1 for v in variants if v.expected_disposition == "STOP"),
    }

    return Cube(
        cube_id=card.cube_id,
        card=card,
        variants=tuple(variants),
        query_registry=card.query_registry,
        live_state_edges=live_state,
        preserve_state_edges=preserve_state,
        live_policy_edges=live_policy,
        preserve_policy_edges=preserve_policy,
        constant_strategy_scores=deep_freeze(scores),
    )
