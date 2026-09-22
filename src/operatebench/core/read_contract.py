"""Canonical per-outcome read and evidence obligations.

One immutable contract is projected to agents and consumed independently by the
runtime and evaluator.  Selectors are a deliberately small algebra: payload
registry identities, fields reached through a linked record, and composite keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

READ_REQUIRED = "required"
READ_OPTIONAL = "optional"
READ_REQUIREMENT_STRENGTHS: tuple[str, ...] = (READ_REQUIRED, READ_OPTIONAL)
COMPLETE_OUTCOME_KEY = "complete"

ACTION_WITHOUT_RETRIEVAL = "ACTION_WITHOUT_RETRIEVAL"
STALE_RETRIEVED_RECORD = "STALE_RETRIEVED_RECORD"
ACTED_ON_CLAIM_WITHOUT_RECORD = "ACTED_ON_CLAIM_WITHOUT_RECORD"
READ_GUARD_CODES: tuple[str, ...] = tuple(
    sorted(
        {
            ACTED_ON_CLAIM_WITHOUT_RECORD,
            ACTION_WITHOUT_RETRIEVAL,
            STALE_RETRIEVED_RECORD,
        }
    )
)

IDENTITY_CODEC = "identity"
QUOTE_KEY_CODEC = "quote_key"
EVIDENCE_CODECS = (IDENTITY_CODEC, QUOTE_KEY_CODEC)
REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED = "REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED"


def _normalise_selector_strings(selector: Any) -> None:
    """Detach selector scalars without calling string-like external objects."""
    for name in selector.__dataclass_fields__:
        value = getattr(selector, name)
        object.__setattr__(selector, name, value if type(value) is str else "")


@dataclass(frozen=True)
class PayloadRegistryKey:
    """Cite the payload identity after resolving it through ``tool``/``root``."""

    tool: str
    root: str
    payload_field: str

    def __post_init__(self) -> None:
        _normalise_selector_strings(self)

    def projection(self) -> tuple[str, dict[str, str]]:
        return (
            self.payload_field,
            {
                "from": f"{self.tool}.{self.root}",
                "select": f"key == payload.{self.payload_field}",
            },
        )


@dataclass(frozen=True)
class WorkEvidenceForInvoice:
    """Cite authoritative work evidence for the payload-selected invoice."""

    tool: str
    root: str
    payload_field: str
    cycle_tool: str
    cycle_root: str
    authority_tool: str
    authority_root: str

    def __post_init__(self) -> None:
        _normalise_selector_strings(self)

    def projection(self) -> tuple[str, dict[str, str]]:
        return (
            "work_evidence_id",
            {
                "from": f"{self.authority_tool}.{self.authority_root}",
                "select": (
                    "key == cycle.work_evidence_id; kind == work_evidence; "
                    "cycle_id == cycle.cycle_id where "
                    f"invoice={self.tool}.{self.root}[payload.{self.payload_field}], "
                    f"cycle={self.cycle_tool}.{self.cycle_root}[invoice.cycle_id]"
                ),
            },
        )


@dataclass(frozen=True)
class QuoteForSelectedApproval:
    """Cite the fixed-codec quote bound by the payload-selected approval."""

    tool: str
    root: str
    payload_field: str
    target_tool: str
    target_root: str

    def __post_init__(self) -> None:
        _normalise_selector_strings(self)

    def projection(self) -> tuple[str, dict[str, str]]:
        return (
            "quote_record_key",
            {
                "from": f"{self.target_tool}.{self.target_root}",
                "select": (
                    "key == approval.quote_id + ':v' + approval.quote_version where "
                    f"approval={self.tool}.{self.root}[payload.{self.payload_field}]"
                ),
            },
        )


EvidenceSelector = PayloadRegistryKey | WorkEvidenceForInvoice | QuoteForSelectedApproval
CLOSED_SELECTOR_TYPES = frozenset(
    {PayloadRegistryKey, WorkEvidenceForInvoice, QuoteForSelectedApproval}
)


def _detached_immutable(value: Any, active: set[int] | None = None) -> Any:
    """Snapshot malformed external declarations without trusting their shape."""
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise RecursionError("recursive action evidence declaration")
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            pairs = [
                (_detached_immutable(key, active), _detached_immutable(item, active))
                for key, item in value.items()
            ]
        finally:
            active.remove(identity)
        return MappingProxyType(dict(pairs))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        active.add(identity)
        try:
            return tuple(_detached_immutable(item, active) for item in value)
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        active.add(identity)
        try:
            return frozenset(_detached_immutable(item, active) for item in value)
        finally:
            active.remove(identity)
    return value


@dataclass(frozen=True)
class EvidenceResolution:
    required: tuple[str, ...]
    problem: str | None = None


@dataclass(frozen=True)
class OutcomeEvidenceObligation:
    reads: Any = field(default_factory=dict)
    evidence_refs: Any = ()
    declaration_problem: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        problem = None
        reads: Mapping[str, str] = MappingProxyType({})
        refs: tuple[Any, ...] = ()
        try:
            if isinstance(self.reads, Mapping):
                detached_reads: dict[str, str] = {}
                for key, strength in self.reads.items():
                    if (
                        type(key) is not str
                        or not key
                        or type(strength) is not str
                        or strength not in READ_REQUIREMENT_STRENGTHS
                    ):
                        problem = (
                            "reads must use exact non-empty string keys and strengths"
                        )
                        continue
                    detached_reads[key] = strength
                reads = MappingProxyType(detached_reads)
            else:
                problem = "reads is not a mapping"
            if isinstance(self.evidence_refs, Sequence) and not isinstance(
                self.evidence_refs, (str, bytes)
            ):
                refs = tuple(_detached_immutable(item) for item in self.evidence_refs)
            else:
                problem = problem or "evidence_refs is not a sequence"
        except (TypeError, ValueError, RecursionError):
            problem = "action evidence contract construction failed"
            reads = MappingProxyType({})
            refs = ()
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "declaration_problem", problem)


@dataclass(frozen=True)
class ActionEvidenceContract:
    """Outcome-keyed read and citation requirements, deeply immutable."""

    outcomes: Mapping[str, OutcomeEvidenceObligation | Mapping[str, Any]] = field(
        default_factory=dict
    )
    declaration_problem: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        frozen: dict[str, Any] = {}
        problem: str | None = None
        if not isinstance(self.outcomes, Mapping):
            object.__setattr__(self, "outcomes", MappingProxyType(frozen))
            object.__setattr__(
                self, "declaration_problem", "contract outcomes is not a mapping"
            )
            return
        try:
            for key, raw in self.outcomes.items():
                if type(key) is not str or not key:
                    problem = "contract outcome keys must be exact non-empty strings"
                    continue
                if isinstance(raw, OutcomeEvidenceObligation):
                    obligation = raw
                elif isinstance(raw, Mapping):
                    entries = list(raw.items())
                    if not all(type(name) is str for name, _value in entries):
                        problem = problem or (
                            f"the {key!r} outcome obligation has malformed fields"
                        )
                        entries = []
                    copied = dict(entries)
                    structured = {"reads", "evidence_refs"}
                    unknown = set(copied) - structured
                    legacy = bool(copied) and all(
                        type(value) is str for value in copied.values()
                    )
                    if unknown and not legacy and problem is None:
                        problem = (
                            f"the {key!r} outcome obligation has unknown field(s) "
                            f"{sorted(unknown)}"
                        )
                    reads = copied if legacy else copied.get("reads", {})
                    obligation = OutcomeEvidenceObligation(
                        reads=reads,
                        evidence_refs=copied.get("evidence_refs", ()),
                    )
                else:
                    obligation = raw
                if (
                    isinstance(obligation, OutcomeEvidenceObligation)
                    and obligation.declaration_problem is not None
                    and problem is None
                ):
                    problem = (
                        obligation.declaration_problem
                        if obligation.declaration_problem
                        == "action evidence contract construction failed"
                        else f"the {key!r} outcome {obligation.declaration_problem}"
                    )
                frozen[key] = obligation
        except (TypeError, ValueError, RecursionError):
            frozen = {}
            problem = "action evidence contract construction failed"
        object.__setattr__(self, "outcomes", MappingProxyType(frozen))
        object.__setattr__(self, "declaration_problem", problem)

    @property
    def requirements(self) -> Mapping[str, Mapping[str, str]]:
        """Compatibility view for the original read-only consumers."""
        return MappingProxyType(
            {
                key: value.reads
                for key, value in self.outcomes.items()
                if isinstance(value, OutcomeEvidenceObligation) and value.reads
            }
        )

    def outcome_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.outcomes))

    def required_tools_for(self, outcome_key: str) -> tuple[str, ...]:
        """Return only reads whose declared strength gates the outcome."""
        obligation = self.outcomes.get(outcome_key)
        if not isinstance(obligation, OutcomeEvidenceObligation):
            return ()
        return tuple(
            sorted(
                tool
                for tool, strength in obligation.reads.items()
                if strength == READ_REQUIRED
            )
        )

    def tools_for(self, outcome_key: str) -> tuple[str, ...]:
        """Return every declared tool name for metadata and surface identity.

        This accessor describes the published read affordance.  Runtime and
        evaluator gating must use :meth:`required_tools_for` instead.
        """
        obligation = self.outcomes.get(outcome_key)
        if not isinstance(obligation, OutcomeEvidenceObligation):
            return ()
        return tuple(sorted(obligation.reads))

    def evidence_for(self, outcome_key: str) -> tuple[EvidenceSelector, ...]:
        obligation = self.outcomes.get(outcome_key)
        return (
            obligation.evidence_refs
            if isinstance(obligation, OutcomeEvidenceObligation)
            else ()
        )

    def schema_part(self, outcome_key: str) -> dict[str, str]:
        obligation = self.outcomes.get(outcome_key)
        if not isinstance(obligation, OutcomeEvidenceObligation):
            return {}
        return {tool: str(value) for tool, value in sorted(obligation.reads.items())}

    def evidence_schema_part(
        self, outcome_key: str
    ) -> dict[str, dict[str, dict[str, str]]]:
        projected: dict[str, dict[str, str]] = {}
        for selector in self.evidence_for(outcome_key):
            if type(selector) not in CLOSED_SELECTOR_TYPES:
                raise ValueError("unknown evidence selector")
            if type(selector) is PayloadRegistryKey:
                assert isinstance(selector, PayloadRegistryKey)
                key, value = PayloadRegistryKey.projection(selector)
            elif type(selector) is WorkEvidenceForInvoice:
                assert isinstance(selector, WorkEvidenceForInvoice)
                key, value = WorkEvidenceForInvoice.projection(selector)
            else:
                assert isinstance(selector, QuoteForSelectedApproval)
                key, value = QuoteForSelectedApproval.projection(selector)
            projected[key] = value
        return {"required": projected}

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "reads": self.schema_part(key),
                "evidence_refs": self.evidence_schema_part(key),
            }
            for key in self.outcome_keys()
        }


# Backward-compatible import name.  This is an alias, not a sibling policy type:
# every caller now receives the one outcome/evidence contract implementation.
ReadRequirementContract = ActionEvidenceContract


def action_evidence_contract_problem(
    contract: ActionEvidenceContract,
    *,
    catalogue: Mapping[str, Mapping[str, Any]],
    outcome_keys: Sequence[str],
    payload_schemas: Mapping[str, tuple[Mapping[str, str], Mapping[str, str]]],
    retrieval_tools: Mapping[str, Any],
) -> str | None:
    """Return why a declaration is not executable, failing closed."""
    if contract.declaration_problem is not None:
        return contract.declaration_problem
    known = set(outcome_keys)

    def ownership_problem(tool: Any, root: Any, relationship: str) -> str | None:
        if (
            not isinstance(tool, str)
            or tool not in catalogue
            or tool not in retrieval_tools
        ):
            return f"the {relationship} names unknown retrieval tool {tool!r}"
        if not isinstance(root, str):
            return f"the {relationship} root must be a string"
        owners = [
            name
            for name, spec in retrieval_tools.items()
            if root in getattr(spec, "roots", ())
        ]
        if owners != [tool]:
            return (
                f"the {relationship} has invalid tool-root ownership: {tool!r}/"
                f"{root!r} is owned by {sorted(owners)}"
            )
        return None

    for key in contract.outcome_keys():
        if key not in known:
            return f"the action evidence contract names unknown outcome {key!r}"
        obligation = contract.outcomes[key]
        if not isinstance(obligation, OutcomeEvidenceObligation):
            return f"the {key!r} outcome obligation is not a mapping"
        for tool, strength in obligation.reads.items():
            if tool not in catalogue:
                return f"the {key!r} requirement names unknown read {tool!r}"
            if strength not in READ_REQUIREMENT_STRENGTHS:
                return f"the {key!r} read {tool!r} has unknown strength {strength!r}"
        payload_fields = set().union(*(payload_schemas.get(key) or ({}, {})))
        seen: set[EvidenceSelector] = set()
        projected_keys: set[str] = set()
        for selector in obligation.evidence_refs:
            if type(selector) not in CLOSED_SELECTOR_TYPES:
                return f"the {key!r} outcome has an unknown evidence selector"
            scalar_values = (selector.tool, selector.root, selector.payload_field)
            if not all(isinstance(value, str) and value for value in scalar_values):
                return f"the {key!r} selector has an incomplete scalar binding"
            extra_values: tuple[str, ...]
            if isinstance(selector, WorkEvidenceForInvoice):
                extra_values = (
                    selector.cycle_tool,
                    selector.cycle_root,
                    selector.authority_tool,
                    selector.authority_root,
                )
            elif isinstance(selector, QuoteForSelectedApproval):
                extra_values = (selector.target_tool, selector.target_root)
            else:
                extra_values = ()
            if not all(isinstance(value, str) and value for value in extra_values):
                return f"the {key!r} selector has an incomplete tool-root binding"
            if selector in seen:
                return f"the {key!r} outcome has a duplicate evidence selector"
            seen.add(selector)
            problem = ownership_problem(selector.tool, selector.root, f"{key!r} selector")
            if problem is not None:
                return problem
            if selector.payload_field not in payload_fields:
                return (
                    f"the {key!r} selector names unknown payload field "
                    f"{selector.payload_field!r}"
                )
            if isinstance(selector, WorkEvidenceForInvoice):
                for tool, root, relation in (
                    (selector.cycle_tool, selector.cycle_root, "cycle"),
                    (selector.authority_tool, selector.authority_root, "authority"),
                ):
                    problem = ownership_problem(
                        tool, root, f"{key!r} work-evidence selector {relation}"
                    )
                    if problem is not None:
                        return problem
            if isinstance(selector, QuoteForSelectedApproval):
                problem = ownership_problem(
                    selector.target_tool,
                    selector.target_root,
                    f"{key!r} quote selector target",
                )
                if problem is not None:
                    return problem
            try:
                if type(selector) is PayloadRegistryKey:
                    assert isinstance(selector, PayloadRegistryKey)
                    projection = PayloadRegistryKey.projection(selector)
                elif type(selector) is WorkEvidenceForInvoice:
                    assert isinstance(selector, WorkEvidenceForInvoice)
                    projection = WorkEvidenceForInvoice.projection(selector)
                else:
                    assert isinstance(selector, QuoteForSelectedApproval)
                    projection = QuoteForSelectedApproval.projection(selector)
                if (
                    not isinstance(projection, tuple)
                    or len(projection) != 2
                    or not isinstance(projection[0], str)
                    or not projection[0]
                    or not isinstance(projection[1], Mapping)
                ):
                    return f"the {key!r} selector has a malformed projection"
                projected_key = projection[0]
            except Exception:
                return f"the {key!r} selector projection cannot be constructed"
            if projected_key in projected_keys:
                return (
                    f"the {key!r} outcome has duplicate evidence projection key "
                    f"{projected_key!r}"
                )
            projected_keys.add(projected_key)
    return None


def read_contract_problem(
    contract: ActionEvidenceContract,
    *,
    catalogue: Mapping[str, Mapping[str, Any]],
    outcome_keys: Sequence[str],
) -> str | None:
    """Compatibility validator for read-only declarations."""
    if contract.declaration_problem is not None:
        return contract.declaration_problem
    known = set(outcome_keys)
    for key in contract.outcome_keys():
        if key not in known:
            return (
                f"the read contract states requirements for {key!r}, which is not an "
                f"outcome this operation produces; it produces {sorted(known)}"
            )
        obligation = contract.outcomes[key]
        if not isinstance(obligation, OutcomeEvidenceObligation):
            return f"the {key!r} requirement must be a mapping"
        for tool, strength in obligation.reads.items():
            if tool not in catalogue:
                return (
                    f"the {key!r} requirement names read {tool!r}, which the "
                    "published retrieval catalogue does not offer; it publishes "
                    f"{sorted(catalogue)}"
                )
            if strength not in READ_REQUIREMENT_STRENGTHS:
                return (
                    f"the {key!r} requirement gives {tool!r} strength {strength!r}; "
                    f"this build has {list(READ_REQUIREMENT_STRENGTHS)}"
                )
    return None


def resolve_required_evidence(
    contract: ActionEvidenceContract,
    outcome_key: str,
    payload: Mapping[str, Any],
    records: Mapping[str, Any],
) -> EvidenceResolution:
    """Resolve required IDs without selecting or adding citations."""
    if not isinstance(payload, Mapping) or not isinstance(records, Mapping):
        return EvidenceResolution((), "the evidence inputs are not mappings")
    resolved: list[str] = []
    for selector in contract.evidence_for(outcome_key):
        if type(selector) not in CLOSED_SELECTOR_TYPES:
            return EvidenceResolution((), "the evidence selector declaration is unknown")
        registry = records.get(selector.root)
        selected_id = payload.get(selector.payload_field)
        if not isinstance(registry, Mapping) or not isinstance(selected_id, str):
            return EvidenceResolution((), "the payload selector cannot be resolved")
        selected = registry.get(selected_id)
        if not isinstance(selected, Mapping):
            return EvidenceResolution((), f"no {selector.root} record {selected_id!r}")
        if isinstance(selector, PayloadRegistryKey):
            resolved.append(selected_id)
            continue
        if isinstance(selector, WorkEvidenceForInvoice):
            cycle_id = selected.get("cycle_id")
            cycles = records.get(selector.cycle_root)
            cycle = cycles.get(cycle_id) if isinstance(cycles, Mapping) else None
            if not isinstance(cycle_id, str) or not isinstance(cycle, Mapping):
                return EvidenceResolution((), "the invoice cycle is missing")
            evidence_id = cycle.get("work_evidence_id")
            authorities = records.get(selector.authority_root)
            authority = (
                authorities.get(evidence_id) if isinstance(authorities, Mapping) else None
            )
            if not isinstance(evidence_id, str) or not isinstance(authority, Mapping):
                return EvidenceResolution(
                    (), "the authoritative work evidence is missing"
                )
            if authority.get("kind") != "work_evidence":
                return EvidenceResolution(
                    (), "the authoritative evidence kind does not match"
                )
            if authority.get("cycle_id") != cycle_id:
                return EvidenceResolution(
                    (), "the authoritative evidence cycle does not match"
                )
            resolved.append(evidence_id)
            continue
        quote_id = selected.get("quote_id")
        quote_version = selected.get("quote_version")
        if not (
            isinstance(quote_id, str)
            and isinstance(quote_version, int)
            and not isinstance(quote_version, bool)
        ):
            return EvidenceResolution((), "the quote evidence key cannot be encoded")
        value = f"{quote_id}:v{quote_version}"
        targets = records.get(selector.target_root)
        if not isinstance(targets, Mapping) or value not in targets:
            return EvidenceResolution((), "the selected approval quote is missing")
        resolved.append(value)
    return EvidenceResolution(tuple(resolved))


__all__ = [
    "ACTED_ON_CLAIM_WITHOUT_RECORD",
    "ACTION_WITHOUT_RETRIEVAL",
    "COMPLETE_OUTCOME_KEY",
    "EVIDENCE_CODECS",
    "READ_GUARD_CODES",
    "READ_REQUIRED",
    "READ_REQUIREMENT_STRENGTHS",
    "REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED",
    "STALE_RETRIEVED_RECORD",
    "ActionEvidenceContract",
    "EvidenceResolution",
    "OutcomeEvidenceObligation",
    "PayloadRegistryKey",
    "QuoteForSelectedApproval",
    "ReadRequirementContract",
    "WorkEvidenceForInvoice",
    "action_evidence_contract_problem",
    "read_contract_problem",
    "resolve_required_evidence",
]
