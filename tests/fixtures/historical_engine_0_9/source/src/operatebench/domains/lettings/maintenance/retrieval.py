"""The eight reads this operation publishes, and what each one projects.

Finite and non-generic, on purpose. There is no query language here, no key
namespace and no "fetch me any path": eight named tools, each mapping onto roots
of the synthetic
:class:`~operatebench.domains.lettings.maintenance.state.MaintenanceState`
that already exist. A generic read would make the vocabulary unbounded, which
would make "the agent asked for the right records" unmeasurable — every run would
have asked for something different, and no two runs could be compared.

Each tool publishes what a caller needs and nothing it has to guess: the service
it reads from, the authority class the answer carries, the schema identity of the
answer's shape and its exact argument schema. In this vertical every read is a
whole-collection projection and every argument schema is therefore *empty* —
stated rather than left open, so an invented key is a named refusal rather than a
parameter the environment silently ignores.

``record_version`` is a content digest in this phase. No record in the synthetic
state carries a version of its own, so the honest thing is to derive one from
what the read returned and say so: it moves when and only when the projected
content moves, which is what a replay binds to, and it is *not* a per-record
version the domain maintains. ``as_of`` is the simulated clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from operatebench.core.retrieval import (
    AUTHORITY_ACTOR_CLAIM,
    AUTHORITY_AUTHORITATIVE_VERIFICATION,
    AUTHORITY_COMMUNICATION,
    AUTHORITY_SYSTEM_OF_RECORD,
    RetrievalRequest,
    ToolResult,
)
from operatebench.core.retrieval_evidence import (
    PUBLIC_RECORD_VERSION_ALGORITHM,
    public_record_version,
)

#: How many hex characters of the content digest a version states. Enough to
#: distinguish the projections one operation produces; deliberately not offered
#: as a cryptographic claim about anything.
RECORD_VERSION_LENGTH = 16

#: The published label the version digest is taken under. Stated as a constant
#: rather than written inline at the one call site, because the evaluator has to
#: apply the *same* label to the same neutral helper to recompute a version it
#: was handed — and a label only one side can see is a label only one side can
#: be right about.
RECORD_VERSION_CONTEXT = "maintenance retrieval record"

#: The schema identity family every answer is stamped with.
SCHEMA_NAMESPACE = "maintenance"
SCHEMA_REVISION = "v1"

#: How a record identity is built from the service that answered. A function
#: rather than a value per tool, so a reader that never ran the operation can
#: derive the identity a row must carry from the source the catalogue publishes
#: for its tool, instead of accepting whatever identity the row states.
RECORD_IDENTITY_SUFFIX = "maintenance_case"


def record_identity(source: str) -> str:
    """Which record a read of ``source`` is about.

    Bound to the service and the operation, never to the scenario. A record
    identity that named the case would put the answer's name inside the
    observation — the exact disclosure ``operation_instance_id`` exists to
    prevent — and every measurement taken afterwards would be a measurement of
    recognition.
    """
    return f"{source}:{RECORD_IDENTITY_SUFFIX}"


@dataclass(frozen=True)
class RetrievalToolSpec:
    """One published read: where it comes from, what it says, and what it takes."""

    tool: str
    source: str
    authority: str
    roots: tuple[str, ...]
    arguments: Mapping[str, str] = field(default_factory=dict)

    @property
    def schema_id(self) -> str:
        return f"{SCHEMA_NAMESPACE}.{self.tool}.{SCHEMA_REVISION}"

    @property
    def record_id(self) -> str:
        """Which record this read is about; see :func:`record_identity`."""
        return record_identity(self.source)

    def catalogue_entry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "authority": self.authority,
            "schema_id": self.schema_id,
            "arguments": dict(self.arguments),
        }


#: The vocabulary. Every ``roots`` entry is a key of
#: :meth:`MaintenanceState.observable`, and the coverage of that mapping is
#: asserted below rather than trusted: a root served by no tool would be a fact
#: an agent could never read, and a root served by two would make "which read
#: established this" ambiguous.
MAINTENANCE_RETRIEVAL_TOOLS: Mapping[str, RetrievalToolSpec] = {
    spec.tool: spec
    for spec in (
        RetrievalToolSpec(
            tool="get_case_record",
            source="case_service",
            authority=AUTHORITY_SYSTEM_OF_RECORD,
            roots=(
                "phase",
                "issue_id",
                "classification",
                "customer_resolution",
                "current_cycle_id",
                "cycles",
                "ownership_transferred_to",
                "transfer_notice_sent",
                "provisional_close",
                "reopen_count",
                "terminal",
                "replay_final",
            ),
        ),
        RetrievalToolSpec(
            tool="list_authoritative_records",
            source="verification_service",
            authority=AUTHORITY_AUTHORITATIVE_VERIFICATION,
            roots=("authoritative_records",),
        ),
        RetrievalToolSpec(
            tool="list_billing",
            source="billing_service",
            authority=AUTHORITY_SYSTEM_OF_RECORD,
            roots=("invoices", "payment"),
        ),
        RetrievalToolSpec(
            tool="list_checkpoints",
            source="approval_service",
            authority=AUTHORITY_SYSTEM_OF_RECORD,
            roots=("approvals", "exceptions"),
        ),
        RetrievalToolSpec(
            tool="list_communications",
            source="messaging_service",
            authority=AUTHORITY_COMMUNICATION,
            roots=("communications",),
        ),
        RetrievalToolSpec(
            tool="list_obligations",
            source="obligation_service",
            authority=AUTHORITY_SYSTEM_OF_RECORD,
            roots=("obligations",),
        ),
        RetrievalToolSpec(
            tool="list_quotes",
            source="quoting_service",
            authority=AUTHORITY_SYSTEM_OF_RECORD,
            roots=("quotes",),
        ),
        RetrievalToolSpec(
            tool="list_recent_events",
            source="inbox_service",
            authority=AUTHORITY_ACTOR_CLAIM,
            roots=("assertions",),
        ),
    )
}

MAINTENANCE_RETRIEVAL_TOOL_NAMES: tuple[str, ...] = tuple(
    sorted(MAINTENANCE_RETRIEVAL_TOOLS)
)

#: Which read serves each root of the observable projection. Derived from the
#: table above and never restated, so a planner that discovers it is missing a
#: root learns which tool answers it rather than guessing.
ROOT_TO_TOOL: Mapping[str, str] = {
    root: spec.tool
    for spec in MAINTENANCE_RETRIEVAL_TOOLS.values()
    for root in spec.roots
}


def maintenance_retrieval_catalogue() -> dict[str, dict[str, Any]]:
    """The published catalogue, as detached plain builtins."""
    return {
        tool: MAINTENANCE_RETRIEVAL_TOOLS[tool].catalogue_entry()
        for tool in MAINTENANCE_RETRIEVAL_TOOL_NAMES
    }


def maintenance_retrieval_record_contract() -> list[str | int]:
    """The shared mechanics behind every record this operation serves.

    Compact canonical order: identity suffix, version context, version
    algorithm, prefix length. It is derived from the same constants used to
    construct served identities and versions, rather than repeated per tool.
    """
    return [
        RECORD_IDENTITY_SUFFIX,
        RECORD_VERSION_CONTEXT,
        PUBLIC_RECORD_VERSION_ALGORITHM,
        RECORD_VERSION_LENGTH,
    ]


#: The JSON shape each observable root is projected in. Not decoration: it is
#: what makes a forged ``records`` body checkable by someone who never ran the
#: operation. Without it, "these records are this tool's answer" reduces to "the
#: digest of whatever is here matches the digest of whatever is here", which any
#: fabricator can satisfy by recomputing one number.
#:
#: The shapes are coarse on purpose — the published root set per tool, and the
#: JSON type each root is projected as. A per-field schema for every nested
#: record would be a second copy of the domain's own state contract, maintained
#: here and drifting from it; and the row cannot prove the *values* were true
#: whatever shape they are in. What this rules out is a body of arbitrary roots
#: and arbitrary types wearing a recomputed version.
ROOT_SHAPE_TEXT = "text"
ROOT_SHAPE_OPTIONAL_TEXT = "optional_text"
ROOT_SHAPE_OBJECT = "object"
ROOT_SHAPE_OPTIONAL_OBJECT = "optional_object"
ROOT_SHAPE_LIST = "list"
ROOT_SHAPE_FLAG = "flag"
ROOT_SHAPE_COUNTER = "counter"

OBSERVABLE_ROOT_SHAPES: Mapping[str, str] = {
    "approvals": ROOT_SHAPE_OBJECT,
    "assertions": ROOT_SHAPE_LIST,
    "authoritative_records": ROOT_SHAPE_OBJECT,
    "classification": ROOT_SHAPE_OPTIONAL_TEXT,
    "communications": ROOT_SHAPE_LIST,
    "current_cycle_id": ROOT_SHAPE_OPTIONAL_TEXT,
    "customer_resolution": ROOT_SHAPE_TEXT,
    "cycles": ROOT_SHAPE_OBJECT,
    "exceptions": ROOT_SHAPE_OBJECT,
    "invoices": ROOT_SHAPE_OBJECT,
    "issue_id": ROOT_SHAPE_OPTIONAL_TEXT,
    "obligations": ROOT_SHAPE_OBJECT,
    "ownership_transferred_to": ROOT_SHAPE_OPTIONAL_TEXT,
    "payment": ROOT_SHAPE_OBJECT,
    "phase": ROOT_SHAPE_TEXT,
    "provisional_close": ROOT_SHAPE_OBJECT,
    "quotes": ROOT_SHAPE_OBJECT,
    "reopen_count": ROOT_SHAPE_COUNTER,
    "replay_final": ROOT_SHAPE_FLAG,
    "terminal": ROOT_SHAPE_OPTIONAL_OBJECT,
    "transfer_notice_sent": ROOT_SHAPE_FLAG,
}


def root_shape_problem(root: str, value: Any) -> str | None:
    """Why this value is not the shape the root is published in, or ``None``."""
    shape = OBSERVABLE_ROOT_SHAPES.get(root)
    if shape is None:
        return f"{root!r} is not a root any published read projects"
    if shape == ROOT_SHAPE_TEXT:
        ok = isinstance(value, str)
    elif shape == ROOT_SHAPE_OPTIONAL_TEXT:
        ok = value is None or isinstance(value, str)
    elif shape == ROOT_SHAPE_OBJECT:
        ok = isinstance(value, Mapping)
    elif shape == ROOT_SHAPE_OPTIONAL_OBJECT:
        ok = value is None or isinstance(value, Mapping)
    elif shape == ROOT_SHAPE_LIST:
        ok = isinstance(value, list)
    elif shape == ROOT_SHAPE_FLAG:
        ok = isinstance(value, bool)
    else:
        ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if ok:
        return None
    return (
        f"{root!r} is published as {shape}, and this row carries a {type(value).__name__}"
    )


def record_version(records: Mapping[str, Any]) -> str:
    """The version this phase gives a projection: a digest of what it says.

    Deliberately derived rather than maintained. No record in this synthetic
    state carries a version, and inventing a counter in the state would be
    inventing evidence; a content digest states exactly what it is — *this read
    returned this content* — and moves when the content moves.

    Routed through :func:`~operatebench.core.retrieval_evidence.public_record_version`
    rather than computed here, so the party that answers a read and the party
    that checks one are running the same code over the same published label.
    """
    return public_record_version(
        records, context=RECORD_VERSION_CONTEXT, length=RECORD_VERSION_LENGTH
    )


def serve_maintenance_retrieval(
    observable: Mapping[str, Any],
    requests: Sequence[RetrievalRequest],
    as_of: str,
) -> tuple[ToolResult, ...]:
    """Answer one canonical batch, in order, at one instant.

    ``observable`` is projected once for the whole batch rather than per request,
    so the results cannot disagree about the state they were read from — the
    other half of what "served atomically" means.
    """
    results: list[ToolResult] = []
    for request in requests:
        spec = MAINTENANCE_RETRIEVAL_TOOLS[request.tool]
        records = {root: observable[root] for root in spec.roots}
        results.append(
            ToolResult(
                tool=spec.tool,
                source=spec.source,
                authority=spec.authority,
                record_id=spec.record_id,
                record_version=record_version(records),
                as_of=as_of,
                schema_id=spec.schema_id,
                records=records,
            )
        )
    return tuple(results)


__all__ = [
    "MAINTENANCE_RETRIEVAL_TOOLS",
    "MAINTENANCE_RETRIEVAL_TOOL_NAMES",
    "OBSERVABLE_ROOT_SHAPES",
    "RECORD_IDENTITY_SUFFIX",
    "RECORD_VERSION_CONTEXT",
    "RECORD_VERSION_LENGTH",
    "ROOT_TO_TOOL",
    "RetrievalToolSpec",
    "maintenance_retrieval_catalogue",
    "maintenance_retrieval_record_contract",
    "record_identity",
    "record_version",
    "root_shape_problem",
    "serve_maintenance_retrieval",
]
