"""Finite retrieval vocabulary for the synthetic return/refund operation."""

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

RECORD_VERSION_LENGTH = 16
RECORD_VERSION_CONTEXT = "commerce return/refund retrieval record"
RECORD_IDENTITY_SUFFIX = "return_refund_case"
SCHEMA_NAMESPACE = "commerce.return_refund"
SCHEMA_REVISION = "v1"


@dataclass(frozen=True)
class RetrievalToolSpec:
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
        return f"{self.source}:{RECORD_IDENTITY_SUFFIX}"

    def catalogue_entry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "authority": self.authority,
            "schema_id": self.schema_id,
            "arguments": dict(self.arguments),
        }


RETURN_REFUND_RETRIEVAL_TOOLS: Mapping[str, RetrievalToolSpec] = {
    item.tool: item
    for item in (
        RetrievalToolSpec(
            "get_return_case", "returns_service", AUTHORITY_SYSTEM_OF_RECORD, ("case",)
        ),
        RetrievalToolSpec(
            "list_authorisations",
            "returns_service",
            AUTHORITY_SYSTEM_OF_RECORD,
            ("authorisations",),
        ),
        RetrievalToolSpec(
            "list_customer_claims",
            "returns_inbox",
            AUTHORITY_ACTOR_CLAIM,
            ("customer_claims",),
        ),
        RetrievalToolSpec(
            "list_carrier_records",
            "carrier_service",
            AUTHORITY_AUTHORITATIVE_VERIFICATION,
            ("carrier_records",),
        ),
        RetrievalToolSpec(
            "list_inspections",
            "warehouse_service",
            AUTHORITY_AUTHORITATIVE_VERIFICATION,
            ("inspections",),
        ),
        RetrievalToolSpec(
            "list_checkpoints",
            "approval_service",
            AUTHORITY_SYSTEM_OF_RECORD,
            ("checkpoints",),
        ),
        RetrievalToolSpec(
            "list_refunds", "payment_service", AUTHORITY_SYSTEM_OF_RECORD, ("refunds",)
        ),
        RetrievalToolSpec(
            "list_communications",
            "messaging_service",
            AUTHORITY_COMMUNICATION,
            ("communications",),
        ),
        RetrievalToolSpec(
            "list_obligations",
            "obligation_service",
            AUTHORITY_SYSTEM_OF_RECORD,
            ("obligations",),
        ),
    )
}

RETURN_REFUND_READ_TOOLS: tuple[str, ...] = tuple(sorted(RETURN_REFUND_RETRIEVAL_TOOLS))
ROOT_TO_TOOL: Mapping[str, str] = {
    root: item.tool
    for item in RETURN_REFUND_RETRIEVAL_TOOLS.values()
    for root in item.roots
}
OBSERVABLE_ROOT_SHAPES: Mapping[str, str] = dict.fromkeys(ROOT_TO_TOOL, "object")


def return_refund_retrieval_catalogue() -> dict[str, dict[str, Any]]:
    return {
        tool: RETURN_REFUND_RETRIEVAL_TOOLS[tool].catalogue_entry()
        for tool in RETURN_REFUND_READ_TOOLS
    }


def return_refund_retrieval_record_contract() -> list[str | int]:
    return [
        RECORD_IDENTITY_SUFFIX,
        RECORD_VERSION_CONTEXT,
        PUBLIC_RECORD_VERSION_ALGORITHM,
        RECORD_VERSION_LENGTH,
    ]


def record_version(records: Mapping[str, Any]) -> str:
    return public_record_version(
        records, context=RECORD_VERSION_CONTEXT, length=RECORD_VERSION_LENGTH
    )


def root_shape_problem(root: str, value: Any) -> str | None:
    if root not in OBSERVABLE_ROOT_SHAPES:
        return f"{root!r} is not a published return/refund root"
    if isinstance(value, Mapping):
        return None
    return f"{root!r} is published as object, got {type(value).__name__}"


def serve_return_refund_retrieval(
    observable: Mapping[str, Any],
    requests: Sequence[RetrievalRequest],
    as_of: str,
) -> tuple[ToolResult, ...]:
    results: list[ToolResult] = []
    for request in requests:
        item = RETURN_REFUND_RETRIEVAL_TOOLS[request.tool]
        records = {root: observable[root] for root in item.roots}
        results.append(
            ToolResult(
                tool=item.tool,
                source=item.source,
                authority=item.authority,
                record_id=item.record_id,
                record_version=record_version(records),
                as_of=as_of,
                schema_id=item.schema_id,
                records=records,
            )
        )
    return tuple(results)


__all__ = [
    "OBSERVABLE_ROOT_SHAPES",
    "RECORD_IDENTITY_SUFFIX",
    "RECORD_VERSION_CONTEXT",
    "RECORD_VERSION_LENGTH",
    "RETURN_REFUND_READ_TOOLS",
    "RETURN_REFUND_RETRIEVAL_TOOLS",
    "ROOT_TO_TOOL",
    "RetrievalToolSpec",
    "record_version",
    "return_refund_retrieval_catalogue",
    "return_refund_retrieval_record_contract",
    "root_shape_problem",
    "serve_return_refund_retrieval",
]
