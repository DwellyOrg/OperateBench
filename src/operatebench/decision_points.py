"""Decision points: one state, one observation, one outcome, one admissibility call.

A Lifecycle episode grades a whole trajectory. That is the right unit for the
Lifecycle claim and the wrong unit for a *matched control*, because two arms with
different temporal and recovery dimensions cannot have their episode-level
``reliable`` compared and mean anything. So this module works one level down. It
names a **decision point** — a moment in the operation identified by a
preregistered *semantic state predicate*, never by a line number in a trajectory
and never by "whatever the reference agent did third" — drives the real engine
and the real domain until that predicate first holds, freezes the public
:class:`~operatebench.core.protocol.AgentObservation` the engine would have
handed an agent at that instant, and grades exactly one returned outcome against
a closed admissible set.

The shared estimand is therefore :data:`SHARED_ESTIMAND` —
*decision-point admissibility* — and nothing here computes, imports or exposes an
episode verdict. That is deliberate: the whole point of the construct is that the
control and the Lifecycle arm are compared on a quantity they both actually have.

Three properties are load-bearing and each one is asserted by the offline suite
rather than assumed here:

* **The observation is the public projection, unchanged.** The engine builds it;
  this module stores a detached JSON snapshot of it and rebuilds a fresh
  observation on every access, so a grader cannot mutate what the next grader
  sees, and nothing that was not in the projection can be added to it.
* **Nothing about the future is in the encounter.** The frozen observation is
  state as of that instant. The pending event queue, the authored expectation,
  the evaluator's findings and the environment's hidden truth are all absent from
  it because the engine never put them in it.
* **The driver is not the answer.** The episode is moved along the reference path
  because a state has to be *reached* somehow, but the driver's own decision at
  the decision point is discarded rather than recorded, and the admissible set is
  read from a manifest that this module cannot write.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from typing import Any

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.core.engine import Engine
from operatebench.core.errors import OperateBenchError
from operatebench.core.outcomes import (
    OUTCOME_KINDS,
    Act,
    AgentOutcome,
    Ask,
    Complete,
    Escalate,
    Wait,
    outcome_contract_problem,
)
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    maintenance_retrieval_catalogue,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_READ_CONTRACT,
    OperationSpec,
)
from operatebench.domains.lettings.maintenance.state import (
    Approval,
    Cycle,
    ExceptionCheckpoint,
    Invoice,
    MaintenanceState,
    Obligation,
    Quote,
)

#: The one quantity a matched decision-point control and its Lifecycle arm share.
#: Named here so a report can state it rather than leave a reader to infer that
#: two numbers printed side by side are estimates of the same thing.
SHARED_ESTIMAND = "decision_point_admissibility"

#: The agent that moves the world while the harness waits for a predicate to
#: hold. It is a *driver*, not an oracle: its decision at the decision point is
#: never stored on the encounter and never consulted when grading.
DRIVER_AGENT_ID = "reference"

#: The unit this construct grades, and how many outcomes it takes at that unit.
#: Stated as data rather than as prose in a docstring, because a report that
#: prints an admissibility rate has to be able to say what its numerator counted:
#: one outcome — the first the agent returns at the matched state — per decision
#: point, and no part of the trajectory around it.
GRADED_UNIT = "decision_point"
GRADED_OUTCOMES_PER_DECISION_POINT = 1


class DecisionPointError(OperateBenchError):
    """A decision point cannot be evaluated, reached or graded as stated."""


class DecisionPointUnreachedError(DecisionPointError):
    """A preregistered state predicate never held in the episode it names."""


# --------------------------------------------------------------- the projection


def _record_fields(record_type: type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(record_type))


#: The live record type behind each registry the public observation carries, so
#: the fields a control may name and the kind of value each field holds are both
#: read off the operation rather than restated beside it.
_RECORD_TYPES: Mapping[str, type] = {
    "cycles": Cycle,
    "quotes": Quote,
    "approvals": Approval,
    "exceptions": ExceptionCheckpoint,
    "invoices": Invoice,
    "obligations": Obligation,
}

#: The record registries the public observation carries, and the fields each of
#: their records actually has. Read off the live record types rather than
#: restated, so renaming a field in the operation breaks every control that named
#: it instead of silently making the control describe a record that is gone.
RECORD_COLLECTIONS: Mapping[str, tuple[str, ...]] = {
    name: _record_fields(record_type) for name, record_type in _RECORD_TYPES.items()
}

#: Where an evidence reference can resolve, and what class of thing it finds
#: there. ``assertions`` is in the table precisely so a control can declare it and
#: be held to declaring it as a *claim*: relabelling an actor's assertion as an
#: authoritative record is the single most useful sham to be able to refuse.
EVIDENCE_COLLECTIONS: Mapping[str, str] = {
    "authoritative_records": "authoritative_record",
    "cycles": "operation_record",
    "quotes": "operation_record",
    "approvals": "operation_record",
    "exceptions": "operation_record",
    "invoices": "operation_record",
    "assertions": "actor_assertion",
}

#: Whether a reference into each class may be cited as evidence for a
#: consequential action. A claim cannot: that is the operation's rule, and a
#: control that says otherwise is not matched to it.
EVIDENCE_CITABILITY: Mapping[str, bool] = {
    "authoritative_record": True,
    "operation_record": True,
    "actor_assertion": False,
}

_CITABLE_COLLECTIONS: tuple[str, ...] = tuple(
    sorted(
        collection
        for collection, klass in EVIDENCE_COLLECTIONS.items()
        if EVIDENCE_CITABILITY[klass]
    )
)


def observation_projection_fields() -> tuple[str, ...]:
    """The public observation's own field names, in the order Core declares them."""
    return tuple(field.name for field in fields(AgentObservation))


def retrieved_record_roots() -> tuple[str, ...]:
    """Every root a *published read* projects, canonically ordered.

    Derived from the read catalogue, not from the operation's whole-record
    projection. Phase 2 takes that projection off the agent path, so "the roots a
    control may name" stopped being "everything the record holds" and became
    "everything a published read returns" — and those are the same roots only for
    as long as the catalogue covers the record, which the catalogue's own
    coverage check is what asserts.
    """
    return tuple(
        sorted(
            {root for tool in MAINTENANCE_RETRIEVAL_TOOLS.values() for root in tool.roots}
        )
    )


def retrieval_catalogue_surface() -> tuple[str, ...]:
    """The published read vocabulary, as ``tool:source:authority`` triples.

    A tool name alone would let a control declare the right vocabulary against
    the wrong provenance — the same eight reads answered by different services,
    or an actor claim relabelled as a system of record. The triple is what an arm
    is actually offered.
    """
    catalogue = maintenance_retrieval_catalogue()
    return tuple(
        f"{tool}:{entry['source']}:{entry['authority']}"
        for tool, entry in sorted(catalogue.items())
    )


def action_read_requirement_surface() -> tuple[str, ...]:
    """Each outcome's strength-sensitive reads, as ``outcome:requirements``.

    Read off the one canonical contract the runtime and the evaluator execute, so
    a control that declares a different read requirement is declaring a different
    operation rather than a differently-worded one. Outcome keys and tool names
    are RFC 3986 percent-encoded components (only unreserved characters remain
    literal), then required reads remain bare while optional reads append the
    literal ``=optional`` marker. Percent is itself encoded, so component text
    cannot impersonate a marker or delimiter. Today's all-required Maintenance
    surface contains only unreserved names and therefore remains byte-identical.
    """
    return tuple(
        f"{_read_requirement_component(key)}:{','.join(_read_requirement_codec(key))}"
        for key in MAINTENANCE_READ_CONTRACT.outcome_keys()
    )


def _read_requirement_codec(outcome_key: str) -> tuple[str, ...]:
    """Encode the contract's two accepted strengths without delimiter collisions."""
    reads = MAINTENANCE_READ_CONTRACT.schema_part(outcome_key)
    encoded: list[str] = []
    for tool, strength in sorted(reads.items()):
        component = _read_requirement_component(tool)
        if strength == "required":
            encoded.append(component)
        elif strength == "optional":
            encoded.append(f"{component}=optional")
        else:  # The canonical contract constructor rejects every other strength.
            raise ValueError(f"unsupported read requirement strength {strength!r}")
    return tuple(encoded)


def _read_requirement_component(value: str) -> str:
    """Canonical RFC 3986 component: ASCII unreserved characters only."""
    unreserved = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    return "".join(
        chr(byte) if byte in unreserved else f"%{byte:02X}"
        for byte in value.encode("utf-8")
    )


def _mapping_roots() -> Mapping[str, tuple[str, ...]]:
    """Roots that are a fixed record rather than a registry or a scalar."""
    blank = MaintenanceState().project(("payment", "provisional_close"))
    return {
        name: tuple(sorted(blank[name]))
        for name in ("payment", "provisional_close")
        if isinstance(blank[name], Mapping)
    }


#: What a handle used as a *comparand* may resolve to. ``unknown`` is a scalar
#: whose type this build cannot derive from the live projection — the leaves of a
#: fixed record, which the operation declares as ``Any`` — and it is deliberately
#: not interchangeable with a typed kind: a comparison that needs a number will
#: not accept one.
SCALAR_KIND_STRING = "string"
SCALAR_KIND_INTEGER = "integer"
SCALAR_KIND_BOOLEAN = "boolean"
SCALAR_KIND_UNTYPED = "unknown"
SCALAR_KINDS: tuple[str, ...] = (
    SCALAR_KIND_STRING,
    SCALAR_KIND_INTEGER,
    SCALAR_KIND_BOOLEAN,
    SCALAR_KIND_UNTYPED,
)

#: Why a handle cannot stand on one side of a comparison. Named, because "the
#: bound did not apply" with no name is exactly the silence this closes: a
#: comparand that cannot resolve to one value makes a maximum, a threshold or an
#: equality vanish at grading time while the manifest still reads as if it holds.
COMPARAND_UNKNOWN_PATH = "UNKNOWN_OBSERVATION_PATH"
COMPARAND_NOT_SCALAR = "NON_SCALAR_COMPARAND"
COMPARAND_TYPE_MISMATCH = "COMPARAND_TYPE_MISMATCH"
COMPARAND_NOT_A_SET = "MEMBERSHIP_COMPARAND_NOT_A_SET"
COMPARAND_EMPTY_SET = "MEMBERSHIP_COMPARAND_EMPTY_SET"
COMPARAND_PROBLEM_CODES: tuple[str, ...] = (
    COMPARAND_UNKNOWN_PATH,
    COMPARAND_NOT_SCALAR,
    COMPARAND_TYPE_MISMATCH,
    COMPARAND_NOT_A_SET,
    COMPARAND_EMPTY_SET,
)

_SCALAR_KIND_BY_TOKEN: Mapping[str, str] = {
    "str": SCALAR_KIND_STRING,
    "int": SCALAR_KIND_INTEGER,
    "bool": SCALAR_KIND_BOOLEAN,
}


#: What a field holds when the operation declares it as a collection of values
#: rather than as one. Deliberately *not* a scalar kind — it never appears in
#: :data:`SCALAR_KINDS` and no comparand ever resolves to it. It exists so that
#: "this field is a tuple" and "this field is an ``Any`` nobody typed" stop being
#: the same answer: the first is a comparison the runtime settles without reading
#: any record, and the second is one a control may legitimately make.
FIELD_KIND_CONTAINER = "container"

#: The type names that declare a collection. ``str`` is a sequence in Python and
#: is not one of these: a name is one value, and comparing against it is exactly
#: what a control is for.
_CONTAINER_TOKENS: frozenset[str] = frozenset(
    {
        "tuple",
        "list",
        "set",
        "frozenset",
        "dict",
        "Sequence",
        "MutableSequence",
        "Mapping",
        "MutableMapping",
        "AbstractSet",
        "Collection",
        "Iterable",
    }
)


def _annotation_tokens(annotation: Any) -> list[str]:
    """The alternatives a live field annotation names, ``None`` dropped."""
    text = (
        annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    )
    return [token.strip() for token in str(text).split("|") if token.strip() != "None"]


def _annotation_scalar_kind(annotation: Any) -> str | None:
    """The scalar kind a live field annotation declares, or ``None`` if it is not one."""
    named = _annotation_tokens(annotation)
    if len(named) != 1:
        return None
    return _SCALAR_KIND_BY_TOKEN.get(named[0])


def _annotation_is_container(annotation: Any) -> bool:
    """Whether a live field annotation declares a collection rather than one value."""
    return any(
        token.split("[")[0].rsplit(".", 1)[-1] in _CONTAINER_TOKENS
        for token in _annotation_tokens(annotation)
    )


def _annotation_field_kind(annotation: Any) -> str | None:
    """What a record field holds, as far as its live type declares it.

    Three answers, and the distance between the last two is the point: a scalar
    kind, :data:`FIELD_KIND_CONTAINER` for a field declared as a collection, and
    ``None`` for a leaf whose type this build cannot derive. A collection has no
    ordering against a number, no equality with a name and no membership of a
    list of them — a control that states one has written a comparison whose
    answer is the same in every state — while an untyped leaf is a value like any
    other, and refusing it too would take comparisons a control may fairly make.
    """
    if _annotation_is_container(annotation):
        return FIELD_KIND_CONTAINER
    return _annotation_scalar_kind(annotation)


#: What kind of thing each record field holds, where its record type declares it.
#: A comparison has two sides: a field that holds a status word can no more be
#: ordered against a threshold than a currency code can, and a field that holds a
#: tuple of evidence handles can be neither ordered, equalled nor looked for in a
#: list — so the left of a record condition is read here for exactly the reason
#: the right is.
RECORD_FIELD_KINDS: Mapping[str, Mapping[str, str]] = {
    name: {
        entry.name: kind
        for entry in fields(record_type)
        if (kind := _annotation_field_kind(entry.type)) is not None
    }
    for name, record_type in _RECORD_TYPES.items()
}


def value_scalar_kind(value: Any) -> str | None:
    """The scalar kind a stated literal is, or ``None`` if it is not one value."""
    if isinstance(value, bool):
        return SCALAR_KIND_BOOLEAN
    if isinstance(value, int):
        return SCALAR_KIND_INTEGER
    if isinstance(value, str):
        return SCALAR_KIND_STRING
    return None


def literal_is_one_value(value: Any) -> bool:
    """Whether a stated literal is one value a comparison could ever read.

    A scalar primitive is one. ``None`` is one too, and deliberately so: the
    operation declares fields such as ``exceptions.resolution`` and
    ``cycles.parent_cycle_id`` as ``str | None``, so null is a value those
    records genuinely hold and a clause that names it states a rule some state
    decides. Everything else — a list, a tuple, a set, a mapping, or any other
    object no field of the operation could hold — is not one value, and
    :func:`_compare` answers a comparison against it the same way in every state.

    Stated as one positive test rather than as a list of refused shapes, because
    a list of refused shapes is a list somebody has to remember to add to: the
    next literal shape a control states is refused here by default, and a shape
    the projection starts carrying is admitted by :func:`value_scalar_kind`,
    which reads the live types. ``str`` is one value and not a sequence of them,
    for the same reason it is not a container in :data:`_CONTAINER_TOKENS`.
    """
    return value is None or value_scalar_kind(value) is not None


def _scalar_kinds(policy: Mapping[str, Any]) -> Mapping[str, str]:
    """Every handle that resolves to one value, and what kind of value that is.

    Read off the live state type and the live policy view rather than restated, so
    a field that becomes a registry, or a count that becomes a flag, stops being
    an admissible comparand instead of quietly resolving to something else.
    """
    kinds: dict[str, str] = {}
    observable = MaintenanceState().observable()
    for entry in fields(MaintenanceState):
        if entry.name not in observable:
            continue
        kind = _annotation_scalar_kind(entry.type)
        if kind is not None:
            kinds[f"state.{entry.name}"] = kind
    for root, keys in _mapping_roots().items():
        for key in keys:
            # A fixed record's leaves are declared ``Any`` by the operation, so
            # they are scalars of no derivable kind rather than of every kind.
            kinds[f"state.{root}.{key}"] = SCALAR_KIND_UNTYPED
    for name, value in policy.items():
        kind = value_scalar_kind(value)
        if kind is not None:
            kinds[f"policy.{name}"] = kind
    return kinds


@dataclass(frozen=True)
class ProjectionSchema:
    """Every path a control is allowed to name, derived from the live projection.

    Built from the operation rather than authored beside it. ``state.hidden`` is
    absent for the reason the whole construct exists: the agent's projection does
    not carry environment truth, so a control that names it is describing a
    different observation than the one the engine hands out.
    """

    state_roots: tuple[str, ...]
    record_collections: Mapping[str, tuple[str, ...]]
    mapping_roots: Mapping[str, tuple[str, ...]]
    policy_keys: tuple[str, ...]
    scalar_kinds: Mapping[str, str]
    record_field_kinds: Mapping[str, Mapping[str, str]]

    def path_problem(self, path: str) -> str | None:
        """Why this dotted handle is not a path into the projection, or ``None``."""
        if not isinstance(path, str) or not path:
            return "a fact handle must be a non-empty dotted path"
        parts = path.split(".")
        root = parts[0]
        if root == "policy":
            if len(parts) != 2 or parts[1] not in self.policy_keys:
                return (
                    f"{path!r} is not a policy handle; the operation's policy view "
                    f"carries {list(self.policy_keys)}"
                )
            return None
        if root != "state":
            return (
                f"{path!r} must start with 'state.' or 'policy.'; the public "
                "observation has no other fact surface a control may name"
            )
        if len(parts) < 2 or parts[1] not in self.state_roots:
            offered = parts[1] if len(parts) > 1 else ""
            return (
                f"{path!r} names {offered!r}, which the observable state does not "
                f"carry; it carries {list(self.state_roots)}"
            )
        name = parts[1]
        if name in self.record_collections:
            if len(parts) == 2:
                return None
            if len(parts) != 4 or parts[2] != "*":
                return (
                    f"{path!r} must address a registry as 'state.{name}' or "
                    f"'state.{name}.*.<field>'"
                )
            if parts[3] not in self.record_collections[name]:
                return (
                    f"{path!r} names field {parts[3]!r}, which a {name} record does "
                    f"not carry; it carries {list(self.record_collections[name])}"
                )
            return None
        if name in self.mapping_roots:
            if len(parts) == 2:
                return None
            if len(parts) != 3 or parts[2] not in self.mapping_roots[name]:
                return (
                    f"{path!r} names a field {name!r} does not carry; it carries "
                    f"{list(self.mapping_roots[name])}"
                )
            return None
        if len(parts) != 2:
            return f"{path!r} addresses into {name!r}, which is not a record"
        return None

    def scalar_kind(self, path: str) -> str | None:
        """The kind of value this handle resolves to, or ``None`` if it is not one."""
        return self.scalar_kinds.get(path)

    def comparand_problem(
        self, path: str, *, required_kind: str | None = None
    ) -> tuple[str, str] | None:
        """Why this handle cannot be compared against, as ``(code, detail)``.

        A comparand is not a fact handle. ``state.quotes`` is a perfectly good
        thing for a control to declare it may look at, and a nonsense thing to
        bound a deadline by: the registry is not a number, so the bound would
        resolve to nothing and an apparently enforced maximum would silently stop
        applying. Every such handle is refused here, before anything runs.
        """
        unresolved = self.path_problem(path)
        if unresolved is not None:
            return (COMPARAND_UNKNOWN_PATH, unresolved)
        kind = self.scalar_kind(path)
        if kind is None:
            return (
                COMPARAND_NOT_SCALAR,
                f"{path!r} addresses a registry, a record set or a whole record, not "
                "one value; a comparison against it can never resolve, so the "
                "constraint that states it would not be enforced at all",
            )
        if required_kind is not None and kind != required_kind:
            return (
                COMPARAND_TYPE_MISMATCH,
                f"{path!r} resolves to a value of kind {kind!r} and this comparison "
                f"requires {required_kind!r}",
            )
        return None

    def record_field_kind(self, collection: str, field_name: str) -> str | None:
        """What a record field holds, or ``None`` where the type declares nothing.

        Either a scalar kind or :data:`FIELD_KIND_CONTAINER`. ``None`` means the
        operation types this leaf as ``Any``, which is not the same answer and is
        not treated as one.
        """
        return self.record_field_kinds.get(collection, {}).get(field_name)

    def record_field_problem(self, collection: str, field_name: str) -> str | None:
        known = self.record_collections.get(collection)
        if known is None:
            return (
                f"{collection!r} is not one of the observation's record registries "
                f"{sorted(self.record_collections)}"
            )
        if field_name not in known:
            return (
                f"a {collection} record does not carry {field_name!r}; it carries "
                f"{list(known)}"
            )
        return None


def projection_schema(spec: OperationSpec, scenario_id: str) -> ProjectionSchema:
    """The projection a control must be matched to, for one scenario of one spec."""
    policy = MaintenanceOperation(spec, scenario_id).policy_view()
    return ProjectionSchema(
        state_roots=retrieved_record_roots(),
        record_collections=RECORD_COLLECTIONS,
        mapping_roots=_mapping_roots(),
        policy_keys=tuple(sorted(policy)),
        scalar_kinds=_scalar_kinds(policy),
        record_field_kinds=RECORD_FIELD_KINDS,
    )


def resolve_path(path: str, observation: Mapping[str, Any]) -> Any:
    """Read one scalar handle out of a frozen observation payload."""
    parts = path.split(".")
    if parts[0] == "policy":
        return observation["policy"].get(parts[1])
    cursor: Any = observation_facts(observation)
    for part in parts[1:]:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(part)
    return cursor


# ------------------------------------------------------------------ predicates

#: The comparisons a preregistered predicate may make. Deliberately few, and all
#: total: a predicate that can only compare is a predicate a reader can check by
#: hand against a state dump.
CONDITION_OPS: tuple[str, ...] = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in")
SCALAR_OPS: tuple[str, ...] = ("eq", "ne", "in", "not_in")
QUANTIFIERS: tuple[str, ...] = ("exists", "none", "exactly_one")

#: What each comparison needs on its right, stated where ``_compare`` implements
#: it so the two cannot drift apart. Ordering answers ``False`` for anything that
#: is not a whole number on either side, and membership answers ``False`` for any
#: right-hand side that is not a list — so a clause that states the wrong shape
#: is not a strict rule, it is a comparison no state could ever satisfy.
ORDERING_OPS: tuple[str, ...] = ("gt", "gte", "lt", "lte")
MEMBERSHIP_OPS: tuple[str, ...] = ("in", "not_in")

#: The quantifiers that select *one* record and may therefore declare a binding.
#: ``none`` selects nothing, and ``exists`` holds for one match or for nine — a
#: binding taken from it would be "whichever record sorted first", which is a
#: position and not a semantic selection. Everything a control reads through a
#: binding has to come from a record the predicate actually singled out.
SINGULAR_QUANTIFIERS: tuple[str, ...] = ("exactly_one",)


def same_value(left: Any, right: Any) -> bool:
    """Equality that does not confuse a yes/no answer with a quantity.

    ``True == 1`` in Python, so a predicate or an admissible set that compared by
    value alone would accept a flag where it asked for a version, an amount or a
    count — and would do it silently, which is the whole problem.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return bool(left == right)


def _compare(op: str, left: Any, right: Any) -> bool:
    if op == "eq":
        return same_value(left, right)
    if op == "ne":
        return not same_value(left, right)
    if op == "in":
        return isinstance(right, (list, tuple)) and any(
            same_value(left, item) for item in right
        )
    if op == "not_in":
        return isinstance(right, (list, tuple)) and not any(
            same_value(left, item) for item in right
        )
    if not isinstance(left, int) or isinstance(left, bool):
        return False
    if not isinstance(right, int) or isinstance(right, bool):
        return False
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    return left <= right if op == "lte" else False


@dataclass(frozen=True)
class RecordCondition:
    """One comparison against one field of one record in a registry."""

    field: str
    op: str
    value: Any = None
    value_ref: str | None = None

    def holds(self, record: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
        right = (
            resolve_path(self.value_ref, observation)
            if self.value_ref is not None
            else self.value
        )
        return _compare(self.op, record.get(self.field), right)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "op": self.op,
            "value": self.value,
            "value_ref": self.value_ref,
        }


@dataclass(frozen=True)
class QuantifiedClause:
    """ "Exactly one / at least one / no record in this registry looks like this."

    ``bind`` is what makes an admissible outcome expressible without naming a
    fixture identifier: the clause finds *the quote on the table*, and the
    admissible ACT is required to carry that quote's own amount and scope rather
    than a literal copied out of the scenario file.
    """

    collection: str
    quantifier: str
    where: tuple[RecordCondition, ...]
    bind: str | None = None

    kind: str = "quantified"

    def matches(
        self, observation: Mapping[str, Any]
    ) -> tuple[bool, Mapping[str, Any] | None, str]:
        registry = observation_facts(observation).get(self.collection)
        if not isinstance(registry, Mapping):
            return False, None, f"the observation carries no {self.collection!r} registry"
        hits = [
            record
            for _, record in sorted(registry.items())
            if isinstance(record, Mapping)
            and all(condition.holds(record, observation) for condition in self.where)
        ]
        if self.quantifier == "none":
            return (
                not hits,
                None,
                f"{len(hits)} {self.collection} record(s) match a clause requiring none",
            )
        if self.quantifier == "exists":
            return (
                bool(hits),
                (hits[0] if hits else None),
                (f"no {self.collection} record matches"),
            )
        matched = len(hits) == 1
        return (
            matched,
            (hits[0] if matched else None),
            f"{len(hits)} {self.collection} record(s) match a clause requiring one",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "collection": self.collection,
            "quantifier": self.quantifier,
            "bind": self.bind,
            "where": [condition.as_dict() for condition in self.where],
        }


@dataclass(frozen=True)
class ScalarClause:
    """One comparison against one scalar handle, possibly against a bound record."""

    path: str
    op: str
    value: Any = None
    value_ref: str | None = None
    binding: str | None = None
    binding_field: str | None = None

    kind: str = "scalar"

    def matches(
        self, observation: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]]
    ) -> tuple[bool, str]:
        left = resolve_path(self.path, observation)
        if self.binding is not None:
            record = bindings.get(self.binding)
            if record is None:
                return False, f"no clause bound {self.binding!r} before this one"
            right: Any = record.get(str(self.binding_field))
        elif self.value_ref is not None:
            right = resolve_path(self.value_ref, observation)
        else:
            right = self.value
        return (
            _compare(self.op, left, right),
            f"{self.path} is {left!r}, which does not satisfy {self.op} {right!r}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "op": self.op,
            "value": self.value,
            "value_ref": self.value_ref,
            "binding": self.binding,
            "binding_field": self.binding_field,
        }


Clause = QuantifiedClause | ScalarClause


@dataclass(frozen=True)
class PredicateMatch:
    """Whether a state predicate held, what it bound, and why it did not."""

    matched: bool
    bindings: Mapping[str, Mapping[str, Any]]
    reason: str


@dataclass(frozen=True)
class StatePredicate:
    """A conjunction of clauses over the public observation. Nothing else."""

    clauses: tuple[Clause, ...]

    @property
    def bindings(self) -> tuple[str, ...]:
        return tuple(
            clause.bind
            for clause in self.clauses
            if isinstance(clause, QuantifiedClause) and clause.bind is not None
        )

    def evaluate(self, observation: Mapping[str, Any]) -> PredicateMatch:
        bound: dict[str, Mapping[str, Any]] = {}
        for position, clause in enumerate(self.clauses):
            if isinstance(clause, QuantifiedClause):
                matched, record, reason = clause.matches(observation)
                if not matched:
                    return PredicateMatch(False, {}, f"clause[{position}]: {reason}")
                if clause.bind is not None and record is not None:
                    bound[clause.bind] = dict(record)
                continue
            matched, reason = clause.matches(observation, bound)
            if not matched:
                return PredicateMatch(False, {}, f"clause[{position}]: {reason}")
        return PredicateMatch(True, bound, "every clause holds")

    def as_dict(self) -> dict[str, Any]:
        return {"clauses": [clause.as_dict() for clause in self.clauses]}


# ----------------------------------------------------------------- constraints

#: How an admissible outcome may pin one field. ``bound_field`` is the important
#: one: it says "the value the state already carries", which is what stops an
#: admissible set from being a transcription of one fixture's identifiers.
CONSTRAINT_KINDS: tuple[str, ...] = (
    "bound_field",
    "free_identifier",
    "one_of",
    "exact",
    "integer_range",
    "optional_integer_range",
    "string_set",
)


@dataclass(frozen=True)
class Constraint:
    """One admissibility constraint on one field of one outcome."""

    kind: str
    binding: str | None = None
    field: str | None = None
    values: tuple[Any, ...] = ()
    value: Any = None
    minimum: int | None = None
    maximum: int | None = None
    maximum_ref: str | None = None
    must_include: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()

    def problem(
        self,
        value: Any,
        bindings: Mapping[str, Mapping[str, Any]],
        observation: Mapping[str, Any],
        where: str,
    ) -> str | None:
        if self.kind == "bound_field":
            record = bindings.get(str(self.binding))
            if record is None:
                return f"{where}: the predicate bound no {self.binding!r}"
            expected = record.get(str(self.field))
            if value != expected or type(value) is not type(expected):
                return (
                    f"{where}: must carry the state's own {self.binding}.{self.field} "
                    f"({expected!r}), got {value!r}"
                )
            return None
        if self.kind == "free_identifier":
            if isinstance(value, bool) or not isinstance(value, str) or not value:
                return f"{where}: must be a non-empty identifier, got {value!r}"
            return None
        if self.kind == "one_of":
            if not any(same_value(value, member) for member in self.values):
                return f"{where}: {value!r} is not one of {list(self.values)}"
            return None
        if self.kind == "exact":
            if same_value(value, self.value):
                return None
            return f"{where}: must be {self.value!r}, got {value!r}"
        if self.kind == "string_set":
            return self._set_problem(value, where)
        return self._range_problem(value, observation, where)

    def _set_problem(self, value: Any, where: str) -> str | None:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return f"{where}: must be a sequence of names, got {type(value).__name__}"
        names = list(value)
        if any(not isinstance(name, str) or not name for name in names):
            return f"{where}: every entry must be a non-empty name, got {names!r}"
        missing = sorted(set(self.must_include) - set(names))
        if missing:
            return f"{where}: must include {missing}, got {names}"
        outside = sorted(set(names) - set(self.allowed))
        if outside:
            return (
                f"{where}: {outside} are outside the admissible set {list(self.allowed)}"
            )
        return None

    def _range_problem(
        self, value: Any, observation: Mapping[str, Any], where: str
    ) -> str | None:
        if value is None:
            if self.kind == "optional_integer_range":
                return None
            return f"{where}: must be a whole number of minutes, got None"
        if isinstance(value, bool) or not isinstance(value, int):
            # A yes/no answer is not a quantity, and ``True`` is an ``int``.
            return f"{where}: must be an integer, got {type(value).__name__}"
        if self.minimum is not None and value < self.minimum:
            return f"{where}: must not be below {self.minimum}, got {value}"
        ceiling = self.maximum
        if self.maximum_ref is not None:
            resolved = resolve_path(self.maximum_ref, observation)
            if isinstance(resolved, bool) or not isinstance(resolved, int):
                # A bound that cannot resolve is not an absent bound. The match
                # validator refuses such a handle before anything runs; if one
                # reaches here anyway, the outcome is refused with the handle
                # named rather than graded against a maximum that has vanished.
                return (
                    f"{where}: the admissible ceiling {self.maximum_ref!r} does not "
                    f"resolve to a whole number in this state (it resolves to "
                    f"{resolved!r}), so this constraint cannot be applied"
                )
            ceiling = resolved
        if ceiling is not None and value > ceiling:
            return f"{where}: must not exceed {ceiling}, got {value}"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "binding": self.binding,
            "field": self.field,
            "values": list(self.values),
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "maximum_ref": self.maximum_ref,
            "must_include": list(self.must_include),
            "allowed": list(self.allowed),
        }


#: What an admissible outcome may say about the references an outcome cites.
EVIDENCE_POLICIES: tuple[str, ...] = ("empty_only", "citable_only", "any")

#: The fields of each outcome kind a control may constrain, read off Core's own
#: outcome types. ``ACT`` is absent because its constrainable surface is the
#: *action payload*, which the operation declares per action type.
OUTCOME_CONSTRAINABLE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "WAIT": tuple(name for name in _record_fields(Wait) if name != "kind"),
    "ASK": tuple(name for name in _record_fields(Ask) if name != "kind"),
    "ESCALATE": tuple(name for name in _record_fields(Escalate) if name != "kind"),
    "COMPLETE": tuple(name for name in _record_fields(Complete) if name != "kind"),
}


def _citable_refs(state: Mapping[str, Any]) -> frozenset[str]:
    refs: set[str] = set()
    for collection in _CITABLE_COLLECTIONS:
        registry = state.get(collection)
        if isinstance(registry, Mapping):
            refs.update(str(key) for key in registry)
    return frozenset(refs)


@dataclass(frozen=True)
class AdmissibleOutcome:
    """One member of a decision point's closed admissible set."""

    outcome_id: str
    outcome_kind: str
    rationale: str
    action_type: str | None = None
    constraints: Mapping[str, Constraint] = dataclass_field(default_factory=dict)
    evidence_refs_policy: str = "any"

    def problem(
        self,
        outcome: AgentOutcome,
        bindings: Mapping[str, Mapping[str, Any]],
        observation: Mapping[str, Any],
    ) -> str | None:
        """Why this outcome is not this admissible member, or ``None`` if it is."""
        kind = getattr(outcome, "kind", "")
        if kind != self.outcome_kind:
            return f"outcome kind {kind!r} is not {self.outcome_kind!r}"
        if isinstance(outcome, Act):
            if outcome.action_type != self.action_type:
                return f"action {outcome.action_type!r} is not {self.action_type!r}"
            body: Mapping[str, Any] = outcome.payload
            where_prefix = "payload"
        else:
            body = {
                name: getattr(outcome, name)
                for name in OUTCOME_CONSTRAINABLE_FIELDS[self.outcome_kind]
            }
            where_prefix = self.outcome_kind
        for name, constraint in self.constraints.items():
            problem = constraint.problem(
                body.get(name), bindings, observation, f"{where_prefix}.{name}"
            )
            if problem is not None:
                return problem
        if isinstance(outcome, Act):
            unknown = sorted(set(outcome.payload) - set(self.constraints))
            if unknown:
                return f"payload carries unconstrained field(s) {unknown}"
        return self._evidence_problem(outcome, observation)

    def _evidence_problem(
        self, outcome: AgentOutcome, observation: Mapping[str, Any]
    ) -> str | None:
        refs = tuple(getattr(outcome, "evidence_refs", ()))
        if self.evidence_refs_policy == "any":
            return None
        if self.evidence_refs_policy == "empty_only":
            return (
                None
                if not refs
                else f"cites evidence {list(refs)} where none is admissible"
            )
        citable = _citable_refs(observation_facts(observation))
        outside = [ref for ref in refs if ref not in citable]
        if outside:
            return (
                f"cites {outside}, which do not resolve to a citable record in this "
                "state; a claim is not a fact and an unresolved reference is neither"
            )
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "outcome_kind": self.outcome_kind,
            "action_type": self.action_type,
            "rationale": self.rationale,
            "evidence_refs_policy": self.evidence_refs_policy,
            "outcome_constraints": {
                name: constraint.as_dict()
                for name, constraint in sorted(self.constraints.items())
            },
        }


# -------------------------------------------------------------- decision points


@dataclass(frozen=True)
class ObservableFact:
    """A fact handle the decision is allowed to rest on, and what class it is."""

    fact_id: str
    path: str
    evidence_class: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "path": self.path,
            "evidence_class": self.evidence_class,
            "note": self.note,
        }


@dataclass(frozen=True)
class EvidenceProvenance:
    """Where a citable reference at this decision point comes from."""

    collection: str
    evidence_class: str
    citable_as_evidence: bool
    produced_by_actor: str
    produced_by_authority: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "evidence_class": self.evidence_class,
            "citable_as_evidence": self.citable_as_evidence,
            "produced_by_actor": self.produced_by_actor,
            "produced_by_authority": self.produced_by_authority,
            "note": self.note,
        }


@dataclass(frozen=True)
class DecisionPoint:
    """One preregistered decision point of one scenario."""

    decision_point_id: str
    scenario_id: str
    label: str
    state_predicate: StatePredicate
    observable_facts: tuple[ObservableFact, ...]
    evidence_provenance: tuple[EvidenceProvenance, ...]
    admissible_outcomes: tuple[AdmissibleOutcome, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_point_id": self.decision_point_id,
            "scenario_id": self.scenario_id,
            "label": self.label,
            "state_predicate": self.state_predicate.as_dict(),
            "observable_facts": [fact.as_dict() for fact in self.observable_facts],
            "evidence_provenance": [
                entry.as_dict() for entry in self.evidence_provenance
            ],
            "admissible_outcomes": [
                outcome.as_dict() for outcome in self.admissible_outcomes
            ],
        }


@dataclass(frozen=True)
class DecisionPointEncounter:
    """The frozen, detached observation one decision point was reached at.

    ``_observation_json`` rather than a live mapping, and a fresh object on every
    :meth:`observation` call, because the same encounter is handed to several
    graders in a row: an agent that edited the state it was shown would otherwise
    change what the next one sees, and "the observation is immutable" would be a
    comment rather than a property.
    """

    decision_point_id: str
    scenario_id: str
    now: str
    invocation_index: int
    turn_index: int
    bindings: Mapping[str, Mapping[str, Any]]
    resolved_evidence: Mapping[str, tuple[str, ...]]
    observation_digest_sha256: str
    _observation_json: str

    def observation(self) -> AgentObservation:
        """A fresh, detached copy of exactly what the engine handed the driver."""
        return AgentObservation(**json.loads(self._observation_json))

    def observation_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(self._observation_json)
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_point_id": self.decision_point_id,
            "scenario_id": self.scenario_id,
            "now": self.now,
            "invocation_index": self.invocation_index,
            "turn_index": self.turn_index,
            "bindings": {name: dict(record) for name, record in self.bindings.items()},
            "resolved_evidence": {
                collection: list(refs)
                for collection, refs in sorted(self.resolved_evidence.items())
            },
            "observation_digest_sha256": self.observation_digest_sha256,
            "observation": self.observation_payload(),
        }


class _PredicateProbe:
    """A driver wrapper that watches for a predicate and freezes what it sees.

    Wrapping rather than reaching into the engine is the whole design: the engine
    hands an observation to whatever satisfies its agent protocol, so a probe that
    satisfies that protocol sees exactly the projection an evaluated agent would,
    with no engine change and no privileged access. What the wrapped driver
    *returns* is passed straight back and never stored.
    """

    def __init__(self, inner: Any, predicate: StatePredicate) -> None:
        self._inner = inner
        self._predicate = predicate
        self.agent_id = str(getattr(inner, "agent_id", DRIVER_AGENT_ID))
        self.captured: tuple[AgentObservation, PredicateMatch] | None = None

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)

    @staticmethod
    def _catalogue_batch(observation: AgentObservation) -> RetrieveBatch | None:
        """One batch for every published read this invocation has not served."""
        catalogue = (observation.retrieval or {}).get("catalogue") or {}
        served = (observation.retrieval or {}).get("served") or {}
        outstanding = sorted(set(catalogue) - set(served))
        budget = int((observation.retrieval or {}).get("batch_budget_remaining") or 0)
        if not outstanding or budget <= 0:
            return None
        return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in outstanding))

    def decide(self, observation: AgentObservation) -> AgentOutcome | RetrieveBatch:
        match = (
            self._predicate.evaluate(observation.as_dict())
            if self.captured is None
            else None
        )
        if match is not None and match.matched:
            # Every arm is offered the *same* surface, and since Phase 2 that
            # surface is what has been read. So the driver establishes the whole
            # published catalogue at the moment it is about to freeze, once. A
            # frozen encounter carrying only the reads this particular driver
            # happened to need would hand each arm a different fact surface —
            # an arm difference nobody chose, at the one moment the control
            # exists to hold constant.
            full = self._catalogue_batch(observation)
            if full is not None:
                return full
        outcome: AgentOutcome = self._inner.decide(observation)
        # A decision point is a moment a *decision* was taken, so it is captured
        # at a turn that produced one. Since reading became an explicit act, the
        # first turn at which a predicate holds is usually the turn the driver
        # spent asking for records — and freezing that one would offer every
        # graded agent an observation at which nothing had been read yet, and
        # then grade the read it correctly asked for as though it were the
        # decision. The turn that follows is the same instant of the same
        # invocation, carries what the driver read, and is where the outcome is.
        if match is not None and match.matched and not isinstance(outcome, RetrieveBatch):
            self.captured = (observation, match)
        return outcome


def reach_decision_point(
    spec: OperationSpec, point: DecisionPoint
) -> DecisionPointEncounter:
    """Run the real engine over the real domain until the predicate first holds.

    No provider is contacted and no network is touched: the driver is one of the
    shipped deterministic agents, and everything the harness reports comes out of
    the engine's own observation projection.
    """
    scenario = spec.scenario(point.scenario_id)
    domain = MaintenanceOperation(spec, point.scenario_id)
    probe = _PredicateProbe(build_agent(DRIVER_AGENT_ID), point.state_predicate)
    engine = Engine(
        domain,
        probe,
        identity={
            "operation_id": spec.operation_id,
            "scenario_id": point.scenario_id,
            "agent_id": f"{DRIVER_AGENT_ID}:decision_point_driver",
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    )
    engine.run()
    if probe.captured is None:
        raise DecisionPointUnreachedError(
            f"decision point {point.decision_point_id!r} declares a state predicate "
            f"that never held in scenario {point.scenario_id!r}; a decision point "
            "nothing reaches grades nothing"
        )
    observation, match = probe.captured
    payload = observation.as_dict()
    return DecisionPointEncounter(
        decision_point_id=point.decision_point_id,
        scenario_id=point.scenario_id,
        now=observation.now,
        invocation_index=observation.invocation_index,
        turn_index=observation.turn_index,
        bindings={name: dict(record) for name, record in match.bindings.items()},
        resolved_evidence=_resolved_evidence(point, payload),
        observation_digest_sha256=hashlib.sha256(
            canonical_json_bytes(payload, "decision point observation")
        ).hexdigest(),
        _observation_json=json.dumps(payload, sort_keys=True),
    )


def observation_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The fact surface a control predicate reads, out of a frozen observation.

    Phase 2 took the normalized projection off the agent path, so this is no
    longer "the record": it is *what has been established by reading*, assembled
    from the results this invocation has been served. A control whose predicate
    names a root nothing was read for resolves to nothing rather than to the
    record's hidden answer, which is the honest reading of the moment — at this
    decision point, what was known is what was asked for.

    Deliberately not a field on the frozen payload. That payload is the exact
    model-visible observation and has to rebuild as one; a synthesised ``state``
    beside it would be a fact surface no agent was ever handed.
    """
    facts: dict[str, Any] = {}
    served = (payload.get("retrieval") or {}).get("served") or {}
    for body in served.values():
        if isinstance(body, Mapping) and body.get("ok", True):
            facts.update(dict(body.get("records") or {}))
    return facts


def _resolved_evidence(
    point: DecisionPoint, payload: Mapping[str, Any]
) -> Mapping[str, tuple[str, ...]]:
    """Which declared evidence handles actually resolve in this state."""
    state = observation_facts(payload)
    resolved: dict[str, tuple[str, ...]] = {}
    for entry in point.evidence_provenance:
        registry = state.get(entry.collection)
        if isinstance(registry, Mapping):
            resolved[entry.collection] = tuple(sorted(str(key) for key in registry))
        elif isinstance(registry, list):
            resolved[entry.collection] = tuple(
                sorted(
                    str(item.get("assertion_id", ""))
                    for item in registry
                    if isinstance(item, Mapping)
                )
            )
        else:
            resolved[entry.collection] = ()
    return resolved


# --------------------------------------------------------------------- grading

#: Why one outcome was not admissible. A closed vocabulary, because "not
#: admissible" without a name is not a result anybody can act on.
NOT_AN_OUTCOME = "NOT_AN_OUTCOME"
MALFORMED_OUTCOME = "MALFORMED_OUTCOME"
OUTCOME_KIND_NOT_ADMISSIBLE = "OUTCOME_KIND_NOT_ADMISSIBLE"
ACTION_TYPE_NOT_ADMISSIBLE = "ACTION_TYPE_NOT_ADMISSIBLE"
OUTCOME_FIELDS_NOT_ADMISSIBLE = "OUTCOME_FIELDS_NOT_ADMISSIBLE"
ADMISSIBLE = "ADMISSIBLE"


@dataclass(frozen=True)
class AdmissibilityVerdict:
    """Admissible or not, which member it matched, and the exact reason if not."""

    decision_point_id: str
    admissible: bool
    reason_code: str
    detail: str
    matched_outcome_id: str | None
    candidate_reasons: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_point_id": self.decision_point_id,
            "estimand": SHARED_ESTIMAND,
            "graded_unit": GRADED_UNIT,
            "graded_outcomes": GRADED_OUTCOMES_PER_DECISION_POINT,
            "admissible": self.admissible,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "matched_outcome_id": self.matched_outcome_id,
            "candidate_reasons": dict(sorted(self.candidate_reasons.items())),
        }


def grade_outcome(
    point: DecisionPoint, encounter: DecisionPointEncounter, outcome: Any
) -> AdmissibilityVerdict:
    """Grade exactly one returned outcome against the closed admissible set."""
    if encounter.decision_point_id != point.decision_point_id:
        raise DecisionPointError(
            f"encounter {encounter.decision_point_id!r} does not belong to decision "
            f"point {point.decision_point_id!r}"
        )
    problem = outcome_contract_problem(outcome)
    if problem is not None:
        code = (
            NOT_AN_OUTCOME
            if not isinstance(outcome, (Act, Wait, Ask, Escalate, Complete))
            else MALFORMED_OUTCOME
        )
        return AdmissibilityVerdict(
            decision_point_id=point.decision_point_id,
            admissible=False,
            reason_code=code,
            detail=problem,
            matched_outcome_id=None,
            candidate_reasons={},
        )
    payload = encounter.observation_payload()
    reasons: dict[str, str] = {}
    for candidate in point.admissible_outcomes:
        why = candidate.problem(outcome, encounter.bindings, payload)
        if why is None:
            return AdmissibilityVerdict(
                decision_point_id=point.decision_point_id,
                admissible=True,
                reason_code=ADMISSIBLE,
                detail=f"matches admissible outcome {candidate.outcome_id!r}",
                matched_outcome_id=candidate.outcome_id,
                candidate_reasons={},
            )
        reasons[candidate.outcome_id] = why
    kind = getattr(outcome, "kind", "")
    kinds = {candidate.outcome_kind for candidate in point.admissible_outcomes}
    if kind not in kinds:
        code = OUTCOME_KIND_NOT_ADMISSIBLE
    elif isinstance(outcome, Act) and outcome.action_type not in {
        candidate.action_type
        for candidate in point.admissible_outcomes
        if candidate.outcome_kind == "ACT"
    }:
        code = ACTION_TYPE_NOT_ADMISSIBLE
    else:
        code = OUTCOME_FIELDS_NOT_ADMISSIBLE
    return AdmissibilityVerdict(
        decision_point_id=point.decision_point_id,
        admissible=False,
        reason_code=code,
        detail="; ".join(
            f"{outcome_id}: {reason}" for outcome_id, reason in sorted(reasons.items())
        ),
        matched_outcome_id=None,
        candidate_reasons=reasons,
    )


def decide_at(encounter: DecisionPointEncounter, agent: Any) -> Any:
    """Offer one agent the frozen observation and take exactly one outcome back.

    The agent sees the public projection and nothing else — no trajectory, no
    pending queue, no expectation — which is the same deal the Lifecycle arm
    offers it at that instant.
    """
    agent.begin_episode(
        {
            "operation_id": encounter.observation_payload()["operation_id"],
            "scenario_id": encounter.scenario_id,
            "agent_id": getattr(agent, "agent_id", "unnamed"),
        }
    )
    return agent.decide(encounter.observation())


__all__ = [
    "ACTION_TYPE_NOT_ADMISSIBLE",
    "ADMISSIBLE",
    "COMPARAND_EMPTY_SET",
    "COMPARAND_NOT_A_SET",
    "COMPARAND_NOT_SCALAR",
    "COMPARAND_PROBLEM_CODES",
    "COMPARAND_TYPE_MISMATCH",
    "COMPARAND_UNKNOWN_PATH",
    "CONDITION_OPS",
    "CONSTRAINT_KINDS",
    "DRIVER_AGENT_ID",
    "EVIDENCE_CITABILITY",
    "EVIDENCE_COLLECTIONS",
    "EVIDENCE_POLICIES",
    "GRADED_OUTCOMES_PER_DECISION_POINT",
    "GRADED_UNIT",
    "MALFORMED_OUTCOME",
    "MEMBERSHIP_OPS",
    "NOT_AN_OUTCOME",
    "ORDERING_OPS",
    "OUTCOME_CONSTRAINABLE_FIELDS",
    "OUTCOME_FIELDS_NOT_ADMISSIBLE",
    "OUTCOME_KINDS",
    "OUTCOME_KIND_NOT_ADMISSIBLE",
    "QUANTIFIERS",
    "RECORD_COLLECTIONS",
    "RECORD_FIELD_KINDS",
    "SCALAR_KINDS",
    "SCALAR_KIND_BOOLEAN",
    "SCALAR_KIND_INTEGER",
    "SCALAR_KIND_STRING",
    "SCALAR_KIND_UNTYPED",
    "SCALAR_OPS",
    "SHARED_ESTIMAND",
    "SINGULAR_QUANTIFIERS",
    "AdmissibilityVerdict",
    "AdmissibleOutcome",
    "Clause",
    "Constraint",
    "DecisionPoint",
    "DecisionPointEncounter",
    "DecisionPointError",
    "DecisionPointUnreachedError",
    "EvidenceProvenance",
    "ObservableFact",
    "PredicateMatch",
    "ProjectionSchema",
    "QuantifiedClause",
    "RecordCondition",
    "ScalarClause",
    "StatePredicate",
    "action_read_requirement_surface",
    "decide_at",
    "grade_outcome",
    "literal_is_one_value",
    "observation_projection_fields",
    "projection_schema",
    "reach_decision_point",
    "resolve_path",
    "retrieval_catalogue_surface",
    "retrieved_record_roots",
    "same_value",
    "value_scalar_kind",
]
