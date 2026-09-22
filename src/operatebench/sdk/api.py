"""The small, trusted interface between the CLI and a shipped operation pack.

This is deliberately a command-dispatch contract, not a dynamic plugin system.
Every implementation is reviewed source imported by the static built-in registry.
A spec cannot name Python to import, and a scaffold is never imported or registered
by the command that creates it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

from boundarybench.freezing import to_json
from operatebench.jsonsafe import check_text, ensure_json_safe

PACK_API_VERSION = 2
PACK_CAPABILITIES: tuple[str, ...] = ("validate", "run", "replay", "check")
PackStatus = Literal["development", "incubator"]
ContributionKind = Literal["maintainer", "partner"]


@dataclass(frozen=True, slots=True)
class ValidateRequest:
    """Static validation of one explicitly selected pack's fixture."""

    spec_path: Path


@dataclass(frozen=True, slots=True)
class RunRequest:
    """One offline development episode and its pack-owned artefact."""

    spec_path: Path
    scenario_id: str
    agent_id: str
    output_path: Path
    self_check: bool = True


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Reproduce a pack-owned run record against its bound fixture."""

    spec_path: Path
    run_path: Path


@dataclass(frozen=True, slots=True)
class CheckRequest:
    """Run a pack's reference and separately authored negative-control gate."""

    spec_path: Path
    agent_ids: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """A domain-neutral CLI result.

    ``contract_passed`` is the only source of exit code 0 versus 2. Bad pack
    input is not encoded here: pack implementations raise
    :class:`OperateBenchError`, and the CLI maps that named refusal to exit code
    1. A malformed command line is a separate usage error.

    Payload values use the JSON-like semantics accepted by
    :func:`operatebench.jsonsafe.ensure_json_safe`: string-keyed mappings,
    sequences, and JSON scalar values. They are recursively copied into detached
    plain JSON at construction and projected into fresh plain ``dict``/``list``
    values on every :attr:`payload` access. No arbitrary object copying hooks are
    invoked.
    """

    contract_passed: bool
    payload: Mapping[str, Any] = field(default_factory=dict)
    text_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.contract_passed) is not bool:
            raise TypeError(
                "operation-pack contract_passed must be bool, got "
                f"{type(self.contract_passed).__name__}"
            )
        payload = dict(object.__getattribute__(self, "payload"))
        ensure_json_safe(payload, "operation-pack command result")
        detached = cast(dict[str, Any], to_json(payload))
        ensure_json_safe(detached, "operation-pack command result")
        lines = tuple(self.text_lines)
        for position, line in enumerate(lines):
            if type(line) is not str:
                raise TypeError(
                    "operation-pack text line "
                    f"{position} has type {type(line).__name__}, expected str"
                )
            check_text(line, f"operation-pack text line {position}")
        object.__setattr__(self, "payload", detached)
        object.__setattr__(self, "text_lines", lines)

    def __getattribute__(self, name: str) -> Any:
        """Project the public payload field without exposing stored aliases."""

        value = object.__getattribute__(self, name)
        if name == "payload":
            return cast(dict[str, Any], to_json(value))
        return value


@dataclass(frozen=True, slots=True)
class OperationPackMetadata:
    """Stable registry metadata; not a run or Artifact-8 identity."""

    pack_id: str
    operation_type: str
    operation_id: str
    pack_version: str
    display_name: str
    owner_id: str
    owner_display_name: str
    contribution_kind: ContributionKind
    status: PackStatus
    privacy_status: Literal["SYNTHETIC_ONLY"]
    default_spec: str | None
    agent_ids: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = PACK_CAPABILITIES
    sdk_api_version: int = PACK_API_VERSION
    evidence_eligible: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "operation_type": self.operation_type,
            "operation_id": self.operation_id,
            "pack_version": self.pack_version,
            "display_name": self.display_name,
            "owner_id": self.owner_id,
            "owner_display_name": self.owner_display_name,
            "contribution_kind": self.contribution_kind,
            "status": self.status,
            "privacy_status": self.privacy_status,
            "default_spec": self.default_spec,
            "agent_ids": list(self.agent_ids),
            "aliases": list(self.aliases),
            "capabilities": list(self.capabilities),
            "sdk_api_version": self.sdk_api_version,
            "evidence_eligible": self.evidence_eligible,
        }


@runtime_checkable
class OperationPack(Protocol):
    """Complete offline commands owned by one statically shipped domain pack."""

    metadata: OperationPackMetadata

    def validate(self, request: ValidateRequest) -> CommandResult: ...

    def run(self, request: RunRequest) -> CommandResult: ...

    def replay(self, request: ReplayRequest) -> CommandResult: ...

    def check(self, request: CheckRequest) -> CommandResult: ...


__all__ = [
    "PACK_API_VERSION",
    "PACK_CAPABILITIES",
    "CheckRequest",
    "CommandResult",
    "ContributionKind",
    "OperationPack",
    "OperationPackMetadata",
    "PackStatus",
    "ReplayRequest",
    "RunRequest",
    "ValidateRequest",
]
