"""The matched decision-point control, and the contract that says it is matched.

A control arm is only a control if it is *the same thing* as the arm it is
compared with, minus the one difference under study. That claim is easy to make
in prose and easy to lose in code, so this module makes it a document with a
validator: a control manifest declares, in machine-readable form, the state
predicate that selects each decision point, the fact handles the decision may
rest on, where the citable evidence at that point comes from, the action and
outcome schemas on offer, and the closed set of admissible outcomes — and
:func:`validate_match` holds every one of those against the *live* operation.

The validator's job is to be able to say **no** before anything executes. Move a
field, rename an action payload key, credit a verification to an actor that
cannot produce it, or widen the admissible set to an action the control never
declared, and the match fails with a named code. That is what stops this from
being a sham control that agrees with whatever the code currently does.

Three things this module deliberately does not do:

* It does not compute or import an episode verdict. The shared estimand is
  :data:`~operatebench.decision_points.SHARED_ESTIMAND`; comparing episode-level
  reliability across arms with different temporal and recovery dimensions is not
  a comparison, and there is no code path here that would let a report imply one.
* It does not author admissible outcomes. It reads them, and it refuses to
  present a project-authored draft as an independent review it has not had.
* It has no call that completes the independent-review slot. That gate is human,
  and a gate with an API is not a gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from operatebench.core.errors import OperateBenchError
from operatebench.core.outcomes import OUTCOME_KINDS
from operatebench.core.protocol import EpisodePlan
from operatebench.core.retrieval import MAX_REQUESTS_PER_BATCH
from operatebench.decision_points import (
    COMPARAND_EMPTY_SET,
    COMPARAND_NOT_A_SET,
    COMPARAND_NOT_SCALAR,
    COMPARAND_TYPE_MISMATCH,
    CONDITION_OPS,
    CONSTRAINT_KINDS,
    EVIDENCE_CITABILITY,
    EVIDENCE_COLLECTIONS,
    EVIDENCE_POLICIES,
    FIELD_KIND_CONTAINER,
    GRADED_OUTCOMES_PER_DECISION_POINT,
    GRADED_UNIT,
    MEMBERSHIP_OPS,
    ORDERING_OPS,
    OUTCOME_CONSTRAINABLE_FIELDS,
    QUANTIFIERS,
    SCALAR_KIND_BOOLEAN,
    SCALAR_KIND_INTEGER,
    SCALAR_KIND_STRING,
    SCALAR_OPS,
    SHARED_ESTIMAND,
    SINGULAR_QUANTIFIERS,
    AdmissibleOutcome,
    Clause,
    Constraint,
    DecisionPoint,
    EvidenceProvenance,
    ObservableFact,
    QuantifiedClause,
    RecordCondition,
    ScalarClause,
    StatePredicate,
    action_read_requirement_surface,
    literal_is_one_value,
    observation_projection_fields,
    projection_schema,
    retrieval_catalogue_surface,
    value_scalar_kind,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    OperationSpec,
)

#: Where the shipped draft manifest lives. A repository example rather than a
#: packaged resource: it is a template awaiting an independent author, not
#: something a released wheel should be able to present as a validated control.
CONTROL_EXAMPLE_DIR = ("examples", "operatebench", "controls")
CONTROL_MANIFEST_NAME = "matched_decision_points_draft_v0_3.yaml"

#: The Lifecycle arms Stage 1 is matched against. V2 is deliberately absent: this
#: control was authored against V1 and V3, and a manifest that quietly widened to
#: a third arm would be claiming coverage nobody wrote.
MATCHED_ARMS: tuple[str, ...] = ("V1", "V3")

#: The two things a manifest may say about who wrote its admissible sets.
INDEPENDENCE_DRAFT = "PROJECT_AUTHORED_DRAFT_NOT_INDEPENDENT"
INDEPENDENCE_INDEPENDENT = "INDEPENDENT_HUMAN_AUTHORED"
INDEPENDENCE_VALUES: tuple[str, ...] = (INDEPENDENCE_DRAFT, INDEPENDENCE_INDEPENDENT)

DRAFT_REVIEW_STATUS = "AWAITING_INDEPENDENT_REVIEW"
DRAFT_EVIDENCE_CLASS = "DRAFT_TEMPLATE_NOT_VALIDATION"

SUPPORTED_SCHEMA_VERSION = 1

#: The top-level blocks the *review target* digest deliberately leaves out. The
#: provenance block says who authored the control and carries the review itself,
#: so a digest that covered it could not be the thing a review names: adding the
#: attestation would change the value the attestation has to state. What is left
#: is the control's semantic content — the predicates, the fact and evidence
#: handles, the surface and the admissible sets — which is what a reviewer reads.
REVIEW_TARGET_EXCLUDED_BLOCKS: tuple[str, ...] = ("provenance",)

#: Domain separation, so a review target digest can never be mistaken for, or
#: substituted with, the document digest of some other manifest.
REVIEW_TARGET_DOMAIN = b"operatebench.matched_control.review_target.v1\n"


class MatchManifestError(OperateBenchError):
    """A control manifest cannot be read, or is not matched to the operation."""


class _DuplicateKeyError(yaml.constructor.ConstructorError):
    """A mapping in the document states the same key twice."""


class _StrictLoader(yaml.SafeLoader):
    """``yaml.safe_load`` with last-wins removed.

    A manifest that states a key twice is ambiguous, and the ambiguity is exactly
    the kind that matters here: two ``maximum`` lines, or two ``admissible_outcomes``
    blocks, would load as whichever came last while reading as though both were in
    force. A human reviewer signing this document cannot see which one the code
    took, so the loader refuses instead of choosing.
    """


def _construct_strict_mapping(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:
            # An unhashable key is not a duplicate question; PyYAML's own
            # constructor refuses it a few lines below, by name.
            continue
        if duplicate:
            raise _DuplicateKeyError(
                "while reading a control manifest",
                node.start_mark,
                f"key {key!r} is stated twice in the same mapping; a manifest that "
                "says two things is not a manifest, and this build will not silently "
                "keep the last one",
                key_node.start_mark,
            )
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_strict_mapping
)


# ------------------------------------------------------------------- utilities


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchManifestError(
            f"{where}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _fields(payload: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise MatchManifestError(
            f"{where}: unknown field(s) {unknown}; this build knows {sorted(allowed)}"
        )
    missing = sorted(set(allowed) - set(payload))
    if missing:
        raise MatchManifestError(f"{where}: missing required field(s) {missing}")


def _text(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise MatchManifestError(
            f"{where}: {key} must be a non-empty string, got {type(value).__name__}"
        )
    return value.strip()


def _flag(payload: Mapping[str, Any], key: str, where: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise MatchManifestError(f"{where}: {key} must be true or false")
    return value


def _count(payload: Mapping[str, Any], key: str, where: str) -> int:
    value = payload[key]
    # ``bool`` is an ``int`` in Python, and a yes/no answer is not a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchManifestError(
            f"{where}: {key} must be an integer, got {type(value).__name__}"
        )
    return value


def _optional_int(payload: Mapping[str, Any], key: str, where: str) -> int | None:
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchManifestError(
            f"{where}: {key} must be an integer or null, got {type(value).__name__}"
        )
    return value


def _optional_text(payload: Mapping[str, Any], key: str, where: str) -> str | None:
    return None if payload[key] is None else _text(payload, key, where)


def _names(
    payload: Mapping[str, Any], key: str, where: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or (not value and not allow_empty):
        raise MatchManifestError(
            f"{where}: {key} must be a non-empty list of names, got "
            f"{type(value).__name__}"
        )
    names: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, str) or not item.strip():
            raise MatchManifestError(f"{where}: {key} carries a non-name entry {item!r}")
        if item in names:
            raise MatchManifestError(f"{where}: {key} names {item!r} twice")
        names.append(item)
    return tuple(names)


def _one_of(
    payload: Mapping[str, Any], key: str, allowed: Sequence[str], where: str
) -> str:
    value = _text(payload, key, where)
    if value not in allowed:
        raise MatchManifestError(
            f"{where}: {key} is {value!r}, which is not one of {list(allowed)}"
        )
    return value


def _list(value: Any, where: str, *, allow_empty: bool = False) -> Sequence[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise MatchManifestError(
            f"{where}: expected a non-empty list, got {type(value).__name__}"
        )
    return value


# -------------------------------------------------------------------- parsing

_CONDITION_FIELDS = ("field", "op", "value", "value_ref")
_QUANTIFIED_FIELDS = ("kind", "collection", "quantifier", "bind", "where")
_SCALAR_FIELDS = ("kind", "path", "op", "value", "value_ref", "binding", "binding_field")

#: Which extra keys each constraint kind carries. Stated once so a constraint
#: that names a field its kind has no use for is refused rather than ignored.
_CONSTRAINT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "bound_field": ("binding", "field"),
    "free_identifier": (),
    "one_of": ("values",),
    "exact": ("value",),
    "integer_range": ("minimum", "maximum", "maximum_ref"),
    "optional_integer_range": ("minimum", "maximum", "maximum_ref"),
    "string_set": ("must_include", "allowed"),
}


def _parse_condition(raw: Any, where: str) -> RecordCondition:
    payload = _mapping(raw, where)
    _fields(payload, _CONDITION_FIELDS, where)
    op = _one_of(payload, "op", CONDITION_OPS, where)
    if (payload["value"] is None) == (payload["value_ref"] is None):
        raise MatchManifestError(
            f"{where}: a condition states exactly one comparand — a literal 'value' "
            "or a 'value_ref' handle — and this one states both or neither"
        )
    return RecordCondition(
        field=_text(payload, "field", where),
        op=op,
        value=payload["value"],
        value_ref=_optional_text(payload, "value_ref", where),
    )


def _parse_clause(raw: Any, where: str) -> Clause:
    payload = _mapping(raw, where)
    kind = payload.get("kind")
    if kind == "quantified":
        _fields(payload, _QUANTIFIED_FIELDS, where)
        return QuantifiedClause(
            collection=_text(payload, "collection", where),
            quantifier=_one_of(payload, "quantifier", QUANTIFIERS, where),
            bind=_optional_text(payload, "bind", where),
            where=tuple(
                _parse_condition(item, f"{where}: where[{index}]")
                for index, item in enumerate(_list(payload["where"], f"{where}: where"))
            ),
        )
    if kind == "scalar":
        _fields(payload, _SCALAR_FIELDS, where)
        stated = [
            name
            for name in ("value", "value_ref", "binding")
            if payload[name] is not None
        ]
        if len(stated) != 1:
            raise MatchManifestError(
                f"{where}: a scalar clause states exactly one comparand — 'value', "
                f"'value_ref' or 'binding' — and this one states {stated}"
            )
        binding = _optional_text(payload, "binding", where)
        binding_field = _optional_text(payload, "binding_field", where)
        if (binding is None) != (binding_field is None):
            raise MatchManifestError(
                f"{where}: 'binding' and 'binding_field' are one handle and are "
                "stated together or not at all"
            )
        return ScalarClause(
            path=_text(payload, "path", where),
            op=_one_of(payload, "op", SCALAR_OPS, where),
            value=payload["value"],
            value_ref=_optional_text(payload, "value_ref", where),
            binding=binding,
            binding_field=binding_field,
        )
    raise MatchManifestError(
        f"{where}: clause kind {kind!r} is not one of ['quantified', 'scalar']"
    )


def _parse_constraint(raw: Any, where: str) -> Constraint:
    payload = _mapping(raw, where)
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in CONSTRAINT_KINDS:
        raise MatchManifestError(
            f"{where}: constraint kind {kind!r} is not one of {list(CONSTRAINT_KINDS)}"
        )
    _fields(payload, ("kind", *_CONSTRAINT_FIELDS[kind]), where)
    if kind == "bound_field":
        return Constraint(
            kind=kind,
            binding=_text(payload, "binding", where),
            field=_text(payload, "field", where),
        )
    if kind == "free_identifier":
        return Constraint(kind=kind)
    if kind == "one_of":
        return Constraint(kind=kind, values=_names(payload, "values", where))
    if kind == "exact":
        return Constraint(kind=kind, value=payload["value"])
    if kind == "string_set":
        allowed = _names(payload, "allowed", where)
        must_include = _names(payload, "must_include", where, allow_empty=True)
        outside = sorted(set(must_include) - set(allowed))
        if outside:
            raise MatchManifestError(
                f"{where}: must_include names {outside}, which the allowed set does "
                "not carry; a constraint cannot require what it also forbids"
            )
        return Constraint(kind=kind, must_include=must_include, allowed=allowed)
    minimum = _optional_int(payload, "minimum", where)
    maximum = _optional_int(payload, "maximum", where)
    maximum_ref = _optional_text(payload, "maximum_ref", where)
    if maximum is not None and maximum_ref is not None:
        raise MatchManifestError(
            f"{where}: a range states a literal 'maximum' or a 'maximum_ref' handle, "
            "never both"
        )
    return Constraint(
        kind=kind, minimum=minimum, maximum=maximum, maximum_ref=maximum_ref
    )


_ADMISSIBLE_FIELDS = (
    "outcome_id",
    "outcome_kind",
    "action_type",
    "rationale",
    "evidence_refs_policy",
    "outcome_constraints",
)


def _parse_admissible(raw: Any, where: str) -> AdmissibleOutcome:
    payload = _mapping(raw, where)
    _fields(payload, _ADMISSIBLE_FIELDS, where)
    outcome_kind = _one_of(payload, "outcome_kind", OUTCOME_KINDS, where)
    action_type = _optional_text(payload, "action_type", where)
    if (outcome_kind == "ACT") != (action_type is not None):
        raise MatchManifestError(
            f"{where}: an ACT names the action it proposes and nothing else does; "
            f"this one is {outcome_kind} with action_type {action_type!r}"
        )
    constraints_raw = _mapping(payload["outcome_constraints"], f"{where}: constraints")
    constraints = {
        name: _parse_constraint(body, f"{where}: outcome_constraints[{name!r}]")
        for name, body in constraints_raw.items()
    }
    return AdmissibleOutcome(
        outcome_id=_text(payload, "outcome_id", where),
        outcome_kind=outcome_kind,
        action_type=action_type,
        rationale=_text(payload, "rationale", where),
        evidence_refs_policy=_one_of(
            payload, "evidence_refs_policy", EVIDENCE_POLICIES, where
        ),
        constraints=constraints,
    )


_FACT_FIELDS = ("fact_id", "path", "evidence_class", "note")
_PROVENANCE_ENTRY_FIELDS = (
    "collection",
    "evidence_class",
    "citable_as_evidence",
    "produced_by_actor",
    "produced_by_authority",
    "note",
)
_DECISION_POINT_FIELDS = (
    "decision_point_id",
    "scenario_id",
    "label",
    "state_predicate",
    "observable_facts",
    "evidence_provenance",
    "admissible_outcomes",
)


def _parse_decision_point(raw: Any, where: str) -> DecisionPoint:
    payload = _mapping(raw, where)
    _fields(payload, _DECISION_POINT_FIELDS, where)
    decision_point_id = _text(payload, "decision_point_id", where)
    named = f"{where} ({decision_point_id})"

    predicate_raw = _mapping(payload["state_predicate"], f"{named}: state_predicate")
    _fields(predicate_raw, ("clauses",), f"{named}: state_predicate")
    clauses = tuple(
        _parse_clause(item, f"{named}: clauses[{index}]")
        for index, item in enumerate(_list(predicate_raw["clauses"], f"{named}: clauses"))
    )

    facts: list[ObservableFact] = []
    for index, item in enumerate(
        _list(payload["observable_facts"], f"{named}: observable_facts")
    ):
        place = f"{named}: observable_facts[{index}]"
        body = _mapping(item, place)
        _fields(body, _FACT_FIELDS, place)
        facts.append(
            ObservableFact(
                fact_id=_text(body, "fact_id", place),
                path=_text(body, "path", place),
                evidence_class=_text(body, "evidence_class", place),
                note=_text(body, "note", place),
            )
        )

    provenance: list[EvidenceProvenance] = []
    for index, item in enumerate(
        _list(payload["evidence_provenance"], f"{named}: evidence_provenance")
    ):
        place = f"{named}: evidence_provenance[{index}]"
        body = _mapping(item, place)
        _fields(body, _PROVENANCE_ENTRY_FIELDS, place)
        provenance.append(
            EvidenceProvenance(
                collection=_text(body, "collection", place),
                evidence_class=_text(body, "evidence_class", place),
                citable_as_evidence=_flag(body, "citable_as_evidence", place),
                produced_by_actor=_text(body, "produced_by_actor", place),
                produced_by_authority=_text(body, "produced_by_authority", place),
                note=_text(body, "note", place),
            )
        )

    admissible: list[AdmissibleOutcome] = []
    seen: set[str] = set()
    for index, item in enumerate(
        _list(payload["admissible_outcomes"], f"{named}: admissible_outcomes")
    ):
        outcome = _parse_admissible(item, f"{named}: admissible_outcomes[{index}]")
        if outcome.outcome_id in seen:
            raise MatchManifestError(
                f"{named}: admissible outcome {outcome.outcome_id!r} is declared "
                "twice; a closed set with a duplicate member is not a closed set"
            )
        seen.add(outcome.outcome_id)
        admissible.append(outcome)

    return DecisionPoint(
        decision_point_id=decision_point_id,
        scenario_id=_text(payload, "scenario_id", named),
        label=_text(payload, "label", named),
        state_predicate=StatePredicate(clauses=clauses),
        observable_facts=tuple(facts),
        evidence_provenance=tuple(provenance),
        admissible_outcomes=tuple(admissible),
    )


# ------------------------------------------------------------------- the shape


@dataclass(frozen=True)
class ActionSchema:
    """One action's payload contract, as the control declares it."""

    required: Mapping[str, str]
    optional: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"required": dict(self.required), "optional": dict(self.optional)}


@dataclass(frozen=True)
class MatchedSurface:
    """What the control claims the agent is offered, in the agent's own terms.

    Read-requirement identity uses a backwards-compatible injective codec.
    Outcome keys and tool names are RFC 3986 percent-encoded components with only
    unreserved characters left literal; required tools are then bare components
    and optional tools append the literal ``=optional`` marker.
    """

    observation_projection_fields: tuple[str, ...]
    #: The published read vocabulary, as ``tool:source:authority``. This replaces
    #: ``observation_state_roots`` at draft 0.3, and the replacement is the seam:
    #: Phase 2 took the normalized projection off the agent path, so what an arm
    #: is offered is no longer "these roots of the record" but "these reads, from
    #: these services, under these authorities".
    retrieval_catalogue: tuple[str, ...]
    #: How many batches one invocation may have served, and how large one may be.
    #: An arm that read as often as it liked would not be the same surface as one
    #: that could not, whatever else matched.
    retrieval_batch_budget: tuple[int, int]
    #: Each outcome's exact strength-sensitive reads. Encoded required components
    #: remain bare; encoded optional components append ``=optional``.
    action_read_requirements: tuple[str, ...]
    outcome_kinds: tuple[str, ...]
    action_schemas: Mapping[str, ActionSchema]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_projection_fields": list(self.observation_projection_fields),
            "retrieval_catalogue": list(self.retrieval_catalogue),
            "retrieval_batch_budget": list(self.retrieval_batch_budget),
            "action_read_requirements": list(self.action_read_requirements),
            "outcome_kinds": list(self.outcome_kinds),
            "action_schemas": {
                name: schema.as_dict()
                for name, schema in sorted(self.action_schemas.items())
            },
        }


@dataclass(frozen=True)
class IndependentReview:
    """A completed independent review of the admissible sets. Never self-issued."""

    reviewer_id: str
    reviewer_affiliation: str
    reviewed_at: str
    attestation_digest_sha256: str
    reviewed_manifest_digest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewer_affiliation": self.reviewer_affiliation,
            "reviewed_at": self.reviewed_at,
            "attestation_digest_sha256": self.attestation_digest_sha256,
            "reviewed_manifest_digest_sha256": self.reviewed_manifest_digest_sha256,
        }


@dataclass(frozen=True)
class Provenance:
    """Who wrote the admissible sets, and what that makes them evidence of."""

    authored_by: str
    independence: str
    evidence_class: str
    review_status: str
    independent_review: IndependentReview | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "authored_by": self.authored_by,
            "independence": self.independence,
            "evidence_class": self.evidence_class,
            "review_status": self.review_status,
            "independent_review": (
                None
                if self.independent_review is None
                else self.independent_review.as_dict()
            ),
        }


@dataclass(frozen=True)
class Denominator:
    """What the control's true denominator is, stated rather than implied.

    Several decision points inside one Semantic Scenario are several *readings of
    the same case*, not several cases. A manifest that claimed otherwise would be
    inflating its n by construction, so the loader refuses it.
    """

    semantic_scenarios: int
    decision_points_are_independent_semantic_samples: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_scenarios": self.semantic_scenarios,
            "decision_points_are_independent_semantic_samples": (
                self.decision_points_are_independent_semantic_samples
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class MatchManifest:
    """The whole control document, with the identity of what it was read from."""

    schema_version: int
    control_id: str
    control_version: str
    operation_id: str
    operation_type: str
    authored_against_operation_version: str
    semantic_scenario_id: str
    shared_estimand: str
    denominator: Denominator
    provenance: Provenance
    matched_surface: MatchedSurface
    decision_points: tuple[DecisionPoint, ...]
    manifest_digest_sha256: str
    review_target_digest_sha256: str
    source: str

    @property
    def review_is_of_this_control(self) -> bool:
        """Whether the review on record names this control's semantic content.

        A review is a review *of something*. The reviewer computes
        :func:`review_target_digest` over the control they read and names it in
        their attestation; anything else — a placeholder, a digest of an earlier
        draft, a digest of a document nobody has seen — is not a review of this.
        """
        review = self.provenance.independent_review
        return (
            review is not None
            and review.reviewed_manifest_digest_sha256 == self.review_target_digest_sha256
        )

    @property
    def is_independent(self) -> bool:
        """Whether these admissible sets are independent of the project. Draft: no."""
        return (
            self.provenance.independence == INDEPENDENCE_INDEPENDENT
            and self.provenance.independent_review is not None
            and self.review_is_of_this_control
        )

    def decision_point(self, decision_point_id: str) -> DecisionPoint:
        for point in self.decision_points:
            if point.decision_point_id == decision_point_id:
                return point
        raise MatchManifestError(
            f"{self.source}: no decision point {decision_point_id!r}; this control "
            f"declares {[point.decision_point_id for point in self.decision_points]}"
        )

    def points_for(self, scenario_id: str) -> tuple[DecisionPoint, ...]:
        return tuple(
            point for point in self.decision_points if point.scenario_id == scenario_id
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "control_id": self.control_id,
            "control_version": self.control_version,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "authored_against_operation_version": (
                self.authored_against_operation_version
            ),
            "semantic_scenario_id": self.semantic_scenario_id,
            "shared_estimand": self.shared_estimand,
            "denominator": self.denominator.as_dict(),
            "provenance": self.provenance.as_dict(),
            "matched_surface": self.matched_surface.as_dict(),
            "decision_points": [point.as_dict() for point in self.decision_points],
        }

    def identity(self) -> dict[str, Any]:
        """What a report records so a reader knows which control it is reading.

        The graded unit and the denominator are in here rather than in a docstring
        because a reader who is handed a rate has to be able to see, from the same
        record, what one observation of it was: one outcome at one decision point,
        over one Semantic Scenario, from a set nobody independent has reviewed yet.
        """
        return {
            "control_id": self.control_id,
            "control_version": self.control_version,
            "manifest_digest_sha256": self.manifest_digest_sha256,
            "review_target_digest_sha256": self.review_target_digest_sha256,
            "shared_estimand": self.shared_estimand,
            "graded_unit": GRADED_UNIT,
            "graded_outcomes_per_decision_point": GRADED_OUTCOMES_PER_DECISION_POINT,
            "semantic_scenarios": self.denominator.semantic_scenarios,
            "decision_points_are_independent_semantic_samples": (
                self.denominator.decision_points_are_independent_semantic_samples
            ),
            "independence": self.provenance.independence,
            "is_independent": self.is_independent,
        }


# --------------------------------------------------------------------- loading

_TOP_LEVEL_FIELDS = (
    "schema_version",
    "control_id",
    "control_version",
    "operation_id",
    "operation_type",
    "authored_against_operation_version",
    "semantic_scenario_id",
    "shared_estimand",
    "denominator",
    "provenance",
    "matched_surface",
    "decision_points",
)
_DENOMINATOR_FIELDS = (
    "semantic_scenarios",
    "decision_points_are_independent_semantic_samples",
    "note",
)
_PROVENANCE_FIELDS = (
    "authored_by",
    "independence",
    "evidence_class",
    "review_status",
    "independent_review",
)
_REVIEW_FIELDS = (
    "reviewer_id",
    "reviewer_affiliation",
    "reviewed_at",
    "attestation_digest_sha256",
    "reviewed_manifest_digest_sha256",
)
_SURFACE_FIELDS = (
    "observation_projection_fields",
    "retrieval_catalogue",
    "retrieval_batch_budget",
    "action_read_requirements",
    "outcome_kinds",
    "action_schemas",
)
_SCHEMA_FIELDS = ("required", "optional")


def _parse_denominator(raw: Any, where: str) -> Denominator:
    payload = _mapping(raw, where)
    _fields(payload, _DENOMINATOR_FIELDS, where)
    independent = _flag(
        payload, "decision_points_are_independent_semantic_samples", where
    )
    if independent:
        raise MatchManifestError(
            f"{where}: decision points inside one Semantic Scenario are readings of "
            "the same case and may not be declared independent semantic samples; a "
            "control that says otherwise inflates its own n"
        )
    scenarios = _count(payload, "semantic_scenarios", where)
    if scenarios != 1:
        raise MatchManifestError(
            f"{where}: semantic_scenarios must be 1 — this control is matched to one "
            f"Semantic Scenario and the denominator is that scenario, got {scenarios}"
        )
    return Denominator(
        semantic_scenarios=scenarios,
        decision_points_are_independent_semantic_samples=independent,
        note=_text(payload, "note", where),
    )


def _parse_provenance(raw: Any, where: str) -> Provenance:
    payload = _mapping(raw, where)
    _fields(payload, _PROVENANCE_FIELDS, where)
    authored_by = _text(payload, "authored_by", where)
    independence = _one_of(payload, "independence", INDEPENDENCE_VALUES, where)
    review_status = _text(payload, "review_status", where)
    raw_review = payload["independent_review"]

    if independence == INDEPENDENCE_DRAFT:
        if raw_review is not None:
            raise MatchManifestError(
                f"{where}: a {INDEPENDENCE_DRAFT} manifest is a draft awaiting review "
                "and cannot also carry a completed independent_review"
            )
        if review_status != DRAFT_REVIEW_STATUS:
            raise MatchManifestError(
                f"{where}: a draft's review_status is {DRAFT_REVIEW_STATUS!r}, got "
                f"{review_status!r}"
            )
        return Provenance(
            authored_by=authored_by,
            independence=independence,
            evidence_class=_text(payload, "evidence_class", where),
            review_status=review_status,
            independent_review=None,
        )

    if raw_review is None:
        raise MatchManifestError(
            f"{where}: {INDEPENDENCE_INDEPENDENT} requires a completed "
            "independent_review block naming the reviewer and their attestation; "
            "independence is not a field a manifest can simply assert"
        )
    review = _mapping(raw_review, f"{where}: independent_review")
    _fields(review, _REVIEW_FIELDS, f"{where}: independent_review")
    reviewer_id = _text(review, "reviewer_id", f"{where}: independent_review")
    affiliation = _text(review, "reviewer_affiliation", f"{where}: independent_review")
    if authored_by in {reviewer_id, affiliation}:
        raise MatchManifestError(
            f"{where}: independent_review is a self-review — the authoring party "
            f"{authored_by!r} cannot be the reviewer or the reviewer's affiliation"
        )
    return Provenance(
        authored_by=authored_by,
        independence=independence,
        evidence_class=_text(payload, "evidence_class", where),
        review_status=review_status,
        independent_review=IndependentReview(
            reviewer_id=reviewer_id,
            reviewer_affiliation=affiliation,
            reviewed_at=_text(review, "reviewed_at", f"{where}: independent_review"),
            attestation_digest_sha256=_digest_text(
                review, "attestation_digest_sha256", f"{where}: independent_review"
            ),
            reviewed_manifest_digest_sha256=_digest_text(
                review, "reviewed_manifest_digest_sha256", f"{where}: independent_review"
            ),
        ),
    )


def _digest_text(payload: Mapping[str, Any], key: str, where: str) -> str:
    value = _text(payload, key, where)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise MatchManifestError(
            f"{where}: {key} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _parse_surface(raw: Any, where: str) -> MatchedSurface:
    payload = _mapping(raw, where)
    _fields(payload, _SURFACE_FIELDS, where)
    schemas_raw = _mapping(payload["action_schemas"], f"{where}: action_schemas")
    schemas: dict[str, ActionSchema] = {}
    for name, body in schemas_raw.items():
        place = f"{where}: action_schemas[{name!r}]"
        entry = _mapping(body, place)
        _fields(entry, _SCHEMA_FIELDS, place)
        schemas[str(name)] = ActionSchema(
            required=_kinds(entry["required"], f"{place}.required"),
            optional=_kinds(entry["optional"], f"{place}.optional"),
        )
    if not schemas:
        raise MatchManifestError(
            f"{where}: a control that offers no action offers no decision"
        )
    return MatchedSurface(
        observation_projection_fields=_names(
            payload, "observation_projection_fields", where
        ),
        retrieval_catalogue=_names(payload, "retrieval_catalogue", where),
        retrieval_batch_budget=_budget(payload, where),
        action_read_requirements=_names(payload, "action_read_requirements", where),
        outcome_kinds=_names(payload, "outcome_kinds", where),
        action_schemas=schemas,
    )


def _budget(payload: Mapping[str, Any], where: str) -> tuple[int, int]:
    """The declared read budget: batches per invocation, requests per batch."""
    raw = payload.get("retrieval_batch_budget")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in raw
        )
    ):
        raise MatchManifestError(
            f"{where}: retrieval_batch_budget states two positive integers — batches "
            "per invocation and requests per batch — and this states "
            f"{raw!r}"
        )
    return (int(raw[0]), int(raw[1]))


def _kinds(raw: Any, where: str) -> Mapping[str, str]:
    payload = _mapping(raw, where)
    kinds: dict[str, str] = {}
    for name, kind in payload.items():
        if not isinstance(name, str) or not name:
            raise MatchManifestError(f"{where}: field name {name!r} must be a string")
        if isinstance(kind, bool) or not isinstance(kind, str) or not kind:
            raise MatchManifestError(
                f"{where}: field {name!r} must declare a primitive kind as a string"
            )
        kinds[name] = kind
    return kinds


def _digest(payload: Mapping[str, Any], source: str) -> str:
    """SHA-256 over the parsed semantic content, not the file's bytes.

    A reflow or a comment does not move it; an edit to a predicate, an evidence
    handle or an admissible outcome does.
    """
    try:
        return hashlib.sha256(
            canonical_json_bytes(dict(payload), f"{source}: control manifest")
        ).hexdigest()
    except JsonSafetyError as exc:
        raise MatchManifestError(
            f"{source}: this control cannot be given a content identity: {exc}"
        ) from exc


def review_target_digest(payload: Mapping[str, Any], source: str) -> str:
    """The digest an independent reviewer signs: this control's semantic content.

    Not the document digest. :func:`_digest` covers the whole manifest including
    the provenance block, and the review lives in the provenance block — so a
    reviewer asked to name the document digest would have to name a value that
    their own signature changes. This digest covers everything else: the state
    predicates, the fact handles, the evidence provenance, the matched surface,
    the denominator and the admissible sets. It is computable before any review
    exists, it does not move when the reviewer's name or timestamp is edited, and
    it does move when any of the semantics a reviewer actually read are changed.
    """
    body = {
        name: value
        for name, value in _mapping(payload, source).items()
        if name not in REVIEW_TARGET_EXCLUDED_BLOCKS
    }
    try:
        content = canonical_json_bytes(body, f"{source}: control review target")
    except JsonSafetyError as exc:
        raise MatchManifestError(
            f"{source}: this control cannot be given a review target: {exc}"
        ) from exc
    return hashlib.sha256(REVIEW_TARGET_DOMAIN + content).hexdigest()


def build_match_manifest(payload: Mapping[str, Any], source: str) -> MatchManifest:
    """Read one control manifest out of an already-parsed mapping."""
    body = _mapping(payload, source)
    _fields(body, _TOP_LEVEL_FIELDS, source)
    version = _count(body, "schema_version", source)
    if version != SUPPORTED_SCHEMA_VERSION:
        raise MatchManifestError(
            f"{source}: schema_version {version} is not the {SUPPORTED_SCHEMA_VERSION} "
            "this build reads"
        )
    estimand = _text(body, "shared_estimand", source)
    if estimand != SHARED_ESTIMAND:
        raise MatchManifestError(
            f"{source}: shared_estimand must be {SHARED_ESTIMAND!r} — the quantity "
            "this control and its Lifecycle arm both have — got "
            f"{estimand!r}"
        )
    points: list[DecisionPoint] = []
    seen: set[str] = set()
    for index, item in enumerate(
        _list(body["decision_points"], f"{source}: decision_points")
    ):
        point = _parse_decision_point(item, f"{source}: decision_points[{index}]")
        if point.decision_point_id in seen:
            raise MatchManifestError(
                f"{source}: decision point {point.decision_point_id!r} is declared "
                "twice; one predicate per decision point, or a report has two answers"
            )
        seen.add(point.decision_point_id)
        points.append(point)

    manifest_payload = dict(body)
    target = review_target_digest(body, source)
    provenance = _parse_provenance(body["provenance"], f"{source}: provenance")
    review = provenance.independent_review
    if review is not None and review.reviewed_manifest_digest_sha256 != target:
        raise MatchManifestError(
            f"{source}: the independent_review names reviewed_manifest_digest_sha256 "
            f"{review.reviewed_manifest_digest_sha256!r}, and this control's review "
            f"target digest is {target!r}; a review of a different document is not a "
            "review of this one, and independence is not a field a manifest may assert"
        )
    return MatchManifest(
        schema_version=version,
        control_id=_text(body, "control_id", source),
        control_version=_text(body, "control_version", source),
        operation_id=_text(body, "operation_id", source),
        operation_type=_text(body, "operation_type", source),
        authored_against_operation_version=_text(
            body, "authored_against_operation_version", source
        ),
        semantic_scenario_id=_text(body, "semantic_scenario_id", source),
        shared_estimand=estimand,
        denominator=_parse_denominator(body["denominator"], f"{source}: denominator"),
        provenance=provenance,
        matched_surface=_parse_surface(
            body["matched_surface"], f"{source}: matched_surface"
        ),
        decision_points=tuple(points),
        manifest_digest_sha256=_digest(manifest_payload, source),
        review_target_digest_sha256=target,
        source=source,
    )


def match_manifest_path() -> Path:
    """Where the shipped draft control manifest lives in this repository."""
    return (
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath(*CONTROL_EXAMPLE_DIR, CONTROL_MANIFEST_NAME)
    )


def load_match_manifest(path: str | Path) -> MatchManifest:
    """Read and validate a control manifest from disk."""
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MatchManifestError(f"cannot read control manifest {source}: {exc}") from exc
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except _DuplicateKeyError as exc:
        raise MatchManifestError(f"{source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MatchManifestError(f"{source}: not readable as YAML: {exc}") from exc
    return build_match_manifest(_mapping(raw, source), source)


# ------------------------------------------------------------ the match itself


@dataclass(frozen=True)
class MatchProblem:
    """One named way this control is not matched to the operation it names."""

    code: str
    where: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "where": self.where, "detail": self.detail}


@dataclass(frozen=True)
class MatchReport:
    """Whether the control is matched, and every way in which it is not."""

    control_id: str
    manifest_digest_sha256: str
    review_target_digest_sha256: str
    operation_id: str
    spec_digest_sha256: str
    shared_estimand: str
    is_independent: bool
    problems: tuple[MatchProblem, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "manifest_digest_sha256": self.manifest_digest_sha256,
            "review_target_digest_sha256": self.review_target_digest_sha256,
            "operation_id": self.operation_id,
            "spec_digest_sha256": self.spec_digest_sha256,
            "shared_estimand": self.shared_estimand,
            "graded_unit": GRADED_UNIT,
            "graded_outcomes_per_decision_point": GRADED_OUTCOMES_PER_DECISION_POINT,
            "is_independent": self.is_independent,
            "ok": self.ok,
            "problems": [problem.as_dict() for problem in self.problems],
        }


def _surface_problems(manifest: MatchManifest, spec: OperationSpec) -> list[MatchProblem]:
    problems: list[MatchProblem] = []
    surface = manifest.matched_surface
    where = "matched_surface"
    if surface.observation_projection_fields != observation_projection_fields():
        problems.append(
            MatchProblem(
                "OBSERVATION_PROJECTION_MISMATCH",
                where,
                "the control declares an observation projection that is not the one "
                "Core hands agents: declared "
                f"{list(surface.observation_projection_fields)}, live "
                f"{list(observation_projection_fields())}",
            )
        )
    if surface.retrieval_catalogue != retrieval_catalogue_surface():
        problems.append(
            MatchProblem(
                "RETRIEVAL_CATALOGUE_MISMATCH",
                where,
                "the control declares a read vocabulary that is not the one this "
                f"operation publishes: declared {list(surface.retrieval_catalogue)}, "
                f"live {list(retrieval_catalogue_surface())}",
            )
        )
    live_budget = (
        EpisodePlan.__dataclass_fields__["max_retrieval_batches_per_invocation"].default,
        MAX_REQUESTS_PER_BATCH,
    )
    if surface.retrieval_batch_budget != live_budget:
        problems.append(
            MatchProblem(
                "RETRIEVAL_BUDGET_MISMATCH",
                where,
                "the control declares a read budget that is not the one an arm is "
                f"offered: declared {list(surface.retrieval_batch_budget)}, live "
                f"{list(live_budget)}",
            )
        )
    if surface.action_read_requirements != action_read_requirement_surface():
        problems.append(
            MatchProblem(
                "ACTION_READ_REQUIREMENT_MISMATCH",
                where,
                "the control declares read requirements that are not the canonical "
                f"contract: declared {list(surface.action_read_requirements)}, live "
                f"{list(action_read_requirement_surface())}",
            )
        )
    if surface.outcome_kinds != OUTCOME_KINDS:
        problems.append(
            MatchProblem(
                "OUTCOME_VOCABULARY_MISMATCH",
                where,
                f"the control offers {list(surface.outcome_kinds)}; Core's outcome "
                f"vocabulary is {list(OUTCOME_KINDS)}",
            )
        )
    if set(surface.action_schemas) != set(MAINTENANCE_ACTION_PAYLOAD_SCHEMAS):
        # A matched control offers the operation's whole action affordance, not a
        # convenient subset: an agent that is only shown the actions somebody
        # already decided were relevant has been handed the answer.
        problems.append(
            MatchProblem(
                "ACTION_VOCABULARY_MISMATCH",
                f"{where}.action_schemas",
                f"the control offers {sorted(surface.action_schemas)}; the operation "
                f"offers {sorted(MAINTENANCE_ACTION_PAYLOAD_SCHEMAS)}",
            )
        )
    for action_type, schema in sorted(surface.action_schemas.items()):
        live = MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.get(action_type)
        if live is None:
            problems.append(
                MatchProblem(
                    "UNKNOWN_ACTION_TYPE",
                    f"{where}.action_schemas[{action_type!r}]",
                    f"{action_type!r} is not an action of this operation; it offers "
                    f"{sorted(MAINTENANCE_ACTION_PAYLOAD_SCHEMAS)}",
                )
            )
            continue
        required, optional = live
        if dict(schema.required) != dict(required) or dict(schema.optional) != dict(
            optional
        ):
            problems.append(
                MatchProblem(
                    "ACTION_SCHEMA_MISMATCH",
                    f"{where}.action_schemas[{action_type!r}]",
                    "the control offers a different payload contract than the "
                    f"operation accepts: declared {schema.as_dict()}, live "
                    f"{ {'required': dict(required), 'optional': dict(optional)} }",
                )
            )
    if spec.operation_id != manifest.operation_id:
        problems.append(
            MatchProblem(
                "OPERATION_IDENTITY_MISMATCH",
                "operation_id",
                f"the control names operation {manifest.operation_id!r} and was "
                f"validated against {spec.operation_id!r}",
            )
        )
    if spec.operation_type != manifest.operation_type:
        problems.append(
            MatchProblem(
                "OPERATION_IDENTITY_MISMATCH",
                "operation_type",
                f"the control names operation type {manifest.operation_type!r} and "
                f"was validated against {spec.operation_type!r}",
            )
        )
    if spec.semantic_scenario_id != manifest.semantic_scenario_id:
        problems.append(
            MatchProblem(
                "SEMANTIC_SCENARIO_MISMATCH",
                "semantic_scenario_id",
                f"the control is authored against {manifest.semantic_scenario_id!r} "
                f"and the operation runs {spec.semantic_scenario_id!r}",
            )
        )
    return problems


def _comparand_problem(
    schema: Any, path: str, place: str, *, required_kind: str | None = None
) -> MatchProblem | None:
    """One handle held to being a single value of the kind the comparison needs."""
    problem = schema.comparand_problem(path, required_kind=required_kind)
    if problem is None:
        return None
    code, detail = problem
    return MatchProblem(code, place, detail)


def _kinds_cannot_meet(left: str | None, right: str | None) -> bool:
    """Whether two scalar kinds settle a comparison without consulting any state.

    Only kinds the projection actually declares count. ``unknown`` is a leaf the
    operation types as ``Any``, and a kind nothing derived is not a kind that
    disagrees — refusing on either would narrow what a control may say rather
    than refuse what it cannot mean.
    """
    known = (SCALAR_KIND_STRING, SCALAR_KIND_INTEGER, SCALAR_KIND_BOOLEAN)
    return left in known and right in known and left != right


#: The operators whose answer against a comparand nothing could ever equal is
#: *true* rather than false. They are the negations, and the distinction is not
#: cosmetic: it decides which half of a vacuity an author is looking at.
_NEGATED_OPS: frozenset[str] = frozenset({"ne", "not_in"})


def _fixed_answer(op: str) -> str:
    """How a comparison no state can decide reads, for the operator that states it.

    ``eq`` against something no field could be is false wherever it is evaluated;
    ``ne`` against the same thing is true wherever it is evaluated, and ``in`` and
    ``not_in`` divide the same way. All of them are the one defect — a conjunct
    that is a constant — and saying "could never hold" of a negation would send an
    author looking for the wrong mistake.
    """
    if op in _NEGATED_OPS:
        return "the comparison holds in every state rather than only where it says"
    return "the comparison could never hold"


def _members_that_are_not_values(value: Sequence[Any]) -> list[Any]:
    """The stated members that are not one value a field could ever hold.

    A collection is the usual one; anything else the projection cannot carry —
    a fractional number, an object — is the same defect in a different shape.
    ``None`` is not one of them: a field the operation declares as ``str | None``
    is null in the states where nothing filled it in, so a list that names null
    tests something a record can be.
    """
    return [item for item in value if not literal_is_one_value(item)]


def _non_scalar_literal_problem(op: str, value: Any, place: str) -> MatchProblem | None:
    """Why a stated literal is not the single value a scalar comparison reads.

    ``approvals.status eq ['OPEN']`` reads as the guard it sits in and is not one:
    ``same_value`` never reads a status word as equal to a list, so the clause is
    false for every approval there will ever be — and under the ``none``
    quantifier, true in every state the operation could reach. The same literal
    under ``ne`` is the constant read from the other end, and under an ordering
    operator it is false again, because ``_compare`` orders nothing that is not a
    whole number. So the shape of the right-hand side is checked once, for every
    operator that takes one value, rather than one operator at a time.
    """
    if literal_is_one_value(value):
        return None
    return MatchProblem(
        COMPARAND_NOT_SCALAR,
        place,
        f"{op!r} compares one value with one value and this clause states "
        f"{value!r}, which is a {type(value).__name__} and not one value; nothing "
        f"a comparison reads can ever be it, so {_fixed_answer(op)}",
    )


def _container_field_problem(
    schema: Any, collection: str, field_name: str, op: str, place: str
) -> MatchProblem | None:
    """Why a field the operation declares as a collection stands on no side of one.

    ``exceptions.evidence_refs`` is a tuple of handles. Ordering it against a
    count answers ``False`` for every exception, equality with a name answers
    ``False`` for every exception, and membership asks whether a whole list is one
    of a list of values. None of the three reads the record, so a clause that
    states one is a fixed answer wearing the shape of a rule.
    """
    if schema.record_field_kind(collection, field_name) != FIELD_KIND_CONTAINER:
        return None
    return MatchProblem(
        COMPARAND_NOT_SCALAR,
        place,
        f"a {collection} record declares {field_name!r} as a collection of values "
        f"rather than one value, and {op!r} compares single values; no state "
        "decides this comparison, its answer is the same in every one of them",
    )


def _comparison_problems(
    schema: Any,
    op: str,
    value: Any,
    value_ref: str | None,
    place: str,
    *,
    left_kind: str | None = None,
) -> list[MatchProblem]:
    """Every way this comparison is already answered before any state reaches it.

    A predicate is a rule about states. A comparison whose operator and whose
    right-hand side cannot meet — an ordering against a currency code, a
    membership test against a single handle — is not a strict rule but a fixed
    answer: false for every record under ``exists`` or ``exactly_one``, and
    therefore *true in every state* under ``none``. That is a guard which reads
    as a refusal and enforces nothing, so it is refused here by name.
    """
    if op in MEMBERSHIP_OPS:
        if value_ref is not None:
            return [
                MatchProblem(
                    COMPARAND_NOT_A_SET,
                    place,
                    f"{op!r} asks whether a value is one of a list, and {value_ref!r} "
                    "is a handle to a single value — every comparand handle is, by "
                    "rule; the comparison could never hold",
                )
            ]
        if not isinstance(value, (list, tuple)):
            return [
                MatchProblem(
                    COMPARAND_NOT_A_SET,
                    place,
                    f"{op!r} asks whether a value is one of a list and this clause "
                    f"states {value!r}; the comparison could never hold",
                )
            ]
        if not value:
            return [
                MatchProblem(
                    COMPARAND_EMPTY_SET,
                    place,
                    f"{op!r} against a list of nothing has the same answer for every "
                    "value there could ever be; a membership test with no members "
                    "states a rule it does not have",
                )
            ]
        unreadable = _members_that_are_not_values(value)
        if unreadable:
            return [
                MatchProblem(
                    COMPARAND_NOT_SCALAR,
                    place,
                    f"{op!r} asks whether one value is one of a list of values, and "
                    f"this clause lists {unreadable!r}, which are not single values "
                    f"a field could hold; nothing a comparison reads is one of them, "
                    f"so {_fixed_answer(op)}",
                )
            ]
        if all(_kinds_cannot_meet(left_kind, value_scalar_kind(item)) for item in value):
            return [
                MatchProblem(
                    COMPARAND_TYPE_MISMATCH,
                    place,
                    f"this clause tests a value of kind {left_kind!r} for membership "
                    f"of {list(value)!r}, and no member of that list is one; "
                    f"{_fixed_answer(op)}",
                )
            ]
        return []
    if value_ref is not None:
        problem = _comparand_problem(
            schema,
            value_ref,
            place,
            required_kind=SCALAR_KIND_INTEGER if op in ORDERING_OPS else None,
        )
        if problem is not None:
            return [problem]
    else:
        # Shape before kind, and in that order for a reason: a list is not a
        # number *and* is not one value, and "not one value" is the defect the
        # author has to fix. Naming it as a type mismatch would describe a
        # comparison between two kinds where there is only one value to have one.
        literal = _non_scalar_literal_problem(op, value, place)
        if literal is not None:
            return [literal]
        if op in ORDERING_OPS and (isinstance(value, bool) or not isinstance(value, int)):
            return [
                MatchProblem(
                    COMPARAND_TYPE_MISMATCH,
                    place,
                    f"{op!r} orders two numbers and this clause states {value!r}; the "
                    "comparison could never hold",
                )
            ]
    right_kind = (
        schema.scalar_kind(value_ref)
        if value_ref is not None
        else value_scalar_kind(value)
    )
    if _kinds_cannot_meet(left_kind, right_kind):
        return [
            MatchProblem(
                COMPARAND_TYPE_MISMATCH,
                place,
                f"this clause compares a value of kind {left_kind!r} with one of kind "
                f"{right_kind!r}; no state decides that comparison, its answer is "
                "the same in every one of them",
            )
        ]
    return []


def _condition_problems(
    schema: Any, clause: QuantifiedClause, place: str
) -> list[MatchProblem]:
    problems: list[MatchProblem] = []
    for position, condition in enumerate(clause.where):
        at = f"{place}.where[{position}]"
        problem = schema.record_field_problem(clause.collection, condition.field)
        if problem is not None:
            # A field the record does not carry has no kind, so every comparand
            # check below would be answered about a field that is not there. One
            # name for one defect: the author has a field to fix, not two.
            problems.append(MatchProblem("UNKNOWN_RECORD_FIELD", at, problem))
            continue
        container = _container_field_problem(
            schema, clause.collection, condition.field, condition.op, at
        )
        if container is not None:
            problems.append(container)
            continue
        problems.extend(
            _comparison_problems(
                schema,
                condition.op,
                condition.value,
                condition.value_ref,
                at,
                left_kind=schema.record_field_kind(clause.collection, condition.field),
            )
        )
    return problems


def _predicate_problems(
    point: DecisionPoint, schema: Any, where: str
) -> list[MatchProblem]:
    problems: list[MatchProblem] = []
    for index, clause in enumerate(point.state_predicate.clauses):
        place = f"{where}.state_predicate.clauses[{index}]"
        if isinstance(clause, QuantifiedClause):
            if clause.collection not in schema.record_collections:
                problems.append(
                    MatchProblem(
                        "UNKNOWN_OBSERVATION_COLLECTION",
                        place,
                        f"{clause.collection!r} is not one of the observation's record "
                        f"registries {sorted(schema.record_collections)}",
                    )
                )
                continue
            problems.extend(_condition_problems(schema, clause, place))
            continue
        # Both sides of a scalar clause are single values: a comparison against a
        # registry, a record set or a whole record silently never holds.
        comparand = _comparand_problem(schema, clause.path, place)
        if comparand is not None:
            problems.append(comparand)
        # A clause states exactly one comparand, so one that binds has no literal
        # and no handle to check here. It is held to its binding instead, by the
        # layer that knows which record was bound: that record's field types are
        # the operation's, not the projection's.
        if clause.binding is None:
            problems.extend(
                _comparison_problems(
                    schema,
                    clause.op,
                    clause.value,
                    clause.value_ref,
                    place,
                    left_kind=schema.scalar_kind(clause.path),
                )
            )
    return problems


def _binders(
    point: DecisionPoint, where: str
) -> tuple[dict[str, tuple[int, str]], list[MatchProblem]]:
    """The bindings this predicate declares, and every way a declaration is not one."""
    binders: dict[str, tuple[int, str]] = {}
    problems: list[MatchProblem] = []
    for index, clause in enumerate(point.state_predicate.clauses):
        if not isinstance(clause, QuantifiedClause) or clause.bind is None:
            continue
        place = f"{where}.state_predicate.clauses[{index}]"
        if clause.quantifier not in SINGULAR_QUANTIFIERS:
            problems.append(
                MatchProblem(
                    "BINDING_QUANTIFIER_NOT_SINGULAR",
                    place,
                    f"a {clause.quantifier!r} clause binds {clause.bind!r}; only "
                    f"{list(SINGULAR_QUANTIFIERS)} selects exactly one record, and a "
                    "binding taken from anything else is either absent or a position "
                    "in a sort order rather than a record the predicate identified",
                )
            )
            continue
        if clause.bind in binders:
            first, _ = binders[clause.bind]
            problems.append(
                MatchProblem(
                    "DUPLICATE_BINDING",
                    place,
                    f"{clause.bind!r} is already bound by clauses[{first}]; two records "
                    "under one name means a reader cannot tell which one an admissible "
                    "outcome is held against",
                )
            )
            continue
        binders[clause.bind] = (index, clause.collection)
    return binders, problems


def _bound_comparison_problems(
    schema: Any, clause: ScalarClause, collection: str, field_name: str, place: str
) -> list[MatchProblem]:
    """Every way a comparison against a bound record field is already answered.

    The binding resolves to one field of one record, so the same shams are open
    here as on any other comparison: a field that is a collection and not a value,
    a membership test whose right-hand side is a single value, and two kinds that
    never meet. Each makes the conjunct a constant, and a predicate that never
    holds reaches no decision point.
    """
    container = _container_field_problem(schema, collection, field_name, clause.op, place)
    if container is not None:
        return [container]
    if clause.op in MEMBERSHIP_OPS:
        return [
            MatchProblem(
                COMPARAND_NOT_A_SET,
                place,
                f"{clause.op!r} asks whether a value is one of a list, and "
                f"{clause.binding!r}.{field_name} is one field of one record; the "
                "comparison could never hold",
            )
        ]
    left_kind = schema.scalar_kind(clause.path)
    right_kind = schema.record_field_kind(collection, field_name)
    if _kinds_cannot_meet(left_kind, right_kind):
        return [
            MatchProblem(
                COMPARAND_TYPE_MISMATCH,
                place,
                f"this clause compares {clause.path}, of kind {left_kind!r}, with "
                f"{field_name} of a bound {collection} record, of kind "
                f"{right_kind!r}; no state decides that comparison, its answer is "
                "the same in every one of them",
            )
        ]
    return []


def _binding_problems(
    point: DecisionPoint, schema: Any, where: str
) -> list[MatchProblem]:
    """Every binding an admissible outcome or a scalar clause reads must be bound.

    Bound, and bound *first*: a clause that reads a binding declared later in the
    same predicate evaluates against nothing, so the predicate quietly never holds
    and the decision point is never reached — a control that grades no agent while
    reading as though it grades one.
    """
    binders, problems = _binders(point, where)
    consumers: list[tuple[str, int | None, str, str | None, ScalarClause | None]] = [
        (
            f"{where}.state_predicate.clauses[{index}]",
            index,
            clause.binding,
            clause.binding_field,
            clause,
        )
        for index, clause in enumerate(point.state_predicate.clauses)
        if isinstance(clause, ScalarClause) and clause.binding is not None
    ]
    for outcome in point.admissible_outcomes:
        for name, constraint in sorted(outcome.constraints.items()):
            if constraint.kind == "bound_field":
                consumers.append(
                    (
                        f"{where}.admissible_outcomes[{outcome.outcome_id!r}].{name}",
                        None,
                        str(constraint.binding),
                        constraint.field,
                        None,
                    )
                )
    for place, position, binding, binding_field, clause in consumers:
        bound = binders.get(binding)
        if bound is None:
            problems.append(
                MatchProblem(
                    "UNKNOWN_BINDING",
                    place,
                    f"nothing in this predicate binds {binding!r} in a way it may be "
                    "read; a control cannot read a record its own predicate never "
                    "singled out",
                )
            )
            continue
        declared_at, collection = bound
        if position is not None and declared_at >= position:
            problems.append(
                MatchProblem(
                    "BINDING_USED_BEFORE_BOUND",
                    place,
                    f"{binding!r} is read here and bound by clauses[{declared_at}]; a "
                    "predicate is evaluated in the order it is written, so this clause "
                    "would compare against nothing and never hold",
                )
            )
            continue
        if binding_field is None:
            continue
        problem = schema.record_field_problem(collection, binding_field)
        if problem is not None:
            problems.append(MatchProblem("UNKNOWN_RECORD_FIELD", place, problem))
            continue
        if clause is not None:
            problems.extend(
                _bound_comparison_problems(
                    schema, clause, collection, binding_field, place
                )
            )
    return problems


def _evidence_problems(
    point: DecisionPoint, spec: OperationSpec, where: str
) -> list[MatchProblem]:
    problems: list[MatchProblem] = []
    for index, entry in enumerate(point.evidence_provenance):
        place = f"{where}.evidence_provenance[{index}]"
        live_class = EVIDENCE_COLLECTIONS.get(entry.collection)
        if live_class is None:
            problems.append(
                MatchProblem(
                    "UNKNOWN_EVIDENCE_COLLECTION",
                    place,
                    f"{entry.collection!r} holds nothing an evidence reference can "
                    f"resolve to; the operation resolves references in "
                    f"{sorted(EVIDENCE_COLLECTIONS)}",
                )
            )
            continue
        if entry.evidence_class != live_class:
            problems.append(
                MatchProblem(
                    "EVIDENCE_CLASS_MISMATCH",
                    place,
                    f"{entry.collection!r} holds {live_class!r} entries and the "
                    f"control calls them {entry.evidence_class!r}",
                )
            )
        if entry.citable_as_evidence != EVIDENCE_CITABILITY[live_class]:
            problems.append(
                MatchProblem(
                    "EVIDENCE_CITABILITY_MISMATCH",
                    place,
                    f"a {live_class!r} is "
                    f"{'citable' if EVIDENCE_CITABILITY[live_class] else 'not citable'} "
                    f"in this operation and the control says otherwise",
                )
            )
        actor = spec.actors.get(entry.produced_by_actor)
        if actor is None:
            problems.append(
                MatchProblem(
                    "EVIDENCE_ACTOR_NOT_IN_REGISTRY",
                    place,
                    f"{entry.produced_by_actor!r} is not an actor of this operation",
                )
            )
            continue
        if not actor.holds(entry.produced_by_authority):
            problems.append(
                MatchProblem(
                    "EVIDENCE_ACTOR_LACKS_AUTHORITY",
                    place,
                    f"{entry.produced_by_actor!r} does not hold "
                    f"{entry.produced_by_authority!r}, so it cannot be the provenance "
                    "of this evidence",
                )
            )
    return problems


def _admissible_problems(
    point: DecisionPoint, manifest: MatchManifest, schema: Any, where: str
) -> list[MatchProblem]:
    problems: list[MatchProblem] = []
    for outcome in point.admissible_outcomes:
        place = f"{where}.admissible_outcomes[{outcome.outcome_id!r}]"
        if outcome.outcome_kind == "ACT":
            declared = manifest.matched_surface.action_schemas.get(
                str(outcome.action_type)
            )
            if declared is None:
                problems.append(
                    MatchProblem(
                        "ADMISSIBLE_OUTCOME_UNDECLARED_ACTION",
                        place,
                        f"the admissible set offers {outcome.action_type!r}, which "
                        "this control's matched surface never declared",
                    )
                )
                continue
            allowed = set(declared.required) | set(declared.optional)
            unknown = sorted(set(outcome.constraints) - allowed)
            if unknown:
                problems.append(
                    MatchProblem(
                        "ADMISSIBLE_OUTCOME_UNKNOWN_FIELD",
                        place,
                        f"constrains {unknown}, which the {outcome.action_type!r} "
                        f"payload does not carry; it carries {sorted(allowed)}",
                    )
                )
            missing = sorted(set(declared.required) - set(outcome.constraints))
            if missing:
                problems.append(
                    MatchProblem(
                        "ADMISSIBLE_OUTCOME_UNCONSTRAINED_FIELD",
                        place,
                        f"leaves required payload field(s) {missing} unconstrained; an "
                        "admissible set with a free required field is not closed",
                    )
                )
        else:
            allowed = set(OUTCOME_CONSTRAINABLE_FIELDS[outcome.outcome_kind])
            unknown = sorted(set(outcome.constraints) - allowed)
            if unknown:
                problems.append(
                    MatchProblem(
                        "ADMISSIBLE_OUTCOME_UNKNOWN_FIELD",
                        place,
                        f"constrains {unknown}, which a {outcome.outcome_kind} does "
                        f"not carry; it carries {sorted(allowed)}",
                    )
                )
        for name, constraint in sorted(outcome.constraints.items()):
            if constraint.maximum_ref is None:
                continue
            # A ceiling is a number. A handle that resolves to a registry, a
            # record, a flag or a name resolves to no ceiling at all, and the
            # maximum this admissible set appears to enforce would silently not be
            # enforced against anything.
            comparand = _comparand_problem(
                schema,
                constraint.maximum_ref,
                f"{place}.{name}",
                required_kind=SCALAR_KIND_INTEGER,
            )
            if comparand is not None:
                problems.append(comparand)
    return problems


def validate_match(manifest: MatchManifest, spec: OperationSpec) -> MatchReport:
    """Hold every claim the control makes against the live operation.

    Returns a report rather than raising, because "here is every way this control
    is not matched" is more useful than the first one. :func:`require_match` is
    the gate that refuses.
    """
    problems: list[MatchProblem] = list(_surface_problems(manifest, spec))
    if manifest.authored_against_operation_version != spec.operation_version:
        problems.append(
            MatchProblem(
                "OPERATION_IDENTITY_MISMATCH",
                "authored_against_operation_version",
                f"the control was authored against operation version "
                f"{manifest.authored_against_operation_version!r} and the spec is "
                f"{spec.operation_version!r}",
            )
        )
    for point in manifest.decision_points:
        where = f"decision_points[{point.decision_point_id!r}]"
        if point.scenario_id not in spec.scenario_ids:
            problems.append(
                MatchProblem(
                    "UNKNOWN_SCENARIO",
                    where,
                    f"the operation declares scenarios {list(spec.scenario_ids)} and "
                    f"this decision point names {point.scenario_id!r}",
                )
            )
            continue
        if point.scenario_id not in MATCHED_ARMS:
            problems.append(
                MatchProblem(
                    "SCENARIO_OUTSIDE_MATCHED_ARMS",
                    where,
                    f"{point.scenario_id!r} is outside the arms this control is "
                    f"matched against {list(MATCHED_ARMS)}",
                )
            )
            continue
        schema = projection_schema(spec, point.scenario_id)
        problems.extend(_predicate_problems(point, schema, where))
        problems.extend(_binding_problems(point, schema, where))
        problems.extend(_evidence_problems(point, spec, where))
        problems.extend(_admissible_problems(point, manifest, schema, where))
        for index, fact in enumerate(point.observable_facts):
            problem = schema.path_problem(fact.path)
            if problem is not None:
                problems.append(
                    MatchProblem(
                        "UNKNOWN_OBSERVATION_PATH",
                        f"{where}.observable_facts[{index}]",
                        problem,
                    )
                )
    return MatchReport(
        control_id=manifest.control_id,
        manifest_digest_sha256=manifest.manifest_digest_sha256,
        review_target_digest_sha256=manifest.review_target_digest_sha256,
        operation_id=spec.operation_id,
        spec_digest_sha256=spec.spec_digest_sha256,
        shared_estimand=manifest.shared_estimand,
        is_independent=manifest.is_independent,
        problems=tuple(problems),
    )


def require_match(manifest: MatchManifest, spec: OperationSpec) -> MatchReport:
    """The gate. A control that is not matched does not get to run."""
    report = validate_match(manifest, spec)
    if not report.ok:
        codes = sorted({problem.code for problem in report.problems})
        detail = "; ".join(
            f"{problem.code} at {problem.where}: {problem.detail}"
            for problem in report.problems
        )
        raise MatchManifestError(
            f"{manifest.source}: this control is not matched to operation "
            f"{spec.operation_id!r} — {codes} — {detail}"
        )
    return report


__all__ = [
    "CONTROL_MANIFEST_NAME",
    "DRAFT_EVIDENCE_CLASS",
    "DRAFT_REVIEW_STATUS",
    "INDEPENDENCE_DRAFT",
    "INDEPENDENCE_INDEPENDENT",
    "INDEPENDENCE_VALUES",
    "MATCHED_ARMS",
    "REVIEW_TARGET_DOMAIN",
    "REVIEW_TARGET_EXCLUDED_BLOCKS",
    "SUPPORTED_SCHEMA_VERSION",
    "ActionSchema",
    "Denominator",
    "IndependentReview",
    "MatchManifest",
    "MatchManifestError",
    "MatchProblem",
    "MatchReport",
    "MatchedSurface",
    "Provenance",
    "build_match_manifest",
    "load_match_manifest",
    "match_manifest_path",
    "require_match",
    "review_target_digest",
    "validate_match",
]
