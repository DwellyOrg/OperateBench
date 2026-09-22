"""Public authoring surface for statically reviewed operation packs."""

from __future__ import annotations

from operatebench.sdk.api import (
    PACK_API_VERSION,
    PACK_CAPABILITIES,
    CheckRequest,
    CommandResult,
    ContributionKind,
    OperationPack,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.company_scaffold import (
    CompanyScaffoldResult,
    scaffold_company_operation,
)
from operatebench.sdk.errors import (
    OperationPackError,
    OperationPackMismatchError,
    OperationPackRegistryError,
    OperationScaffoldError,
    UnknownOperationPackError,
)
from operatebench.sdk.registry import OperationPackRegistry
from operatebench.sdk.scaffold import scaffold_operation

__all__ = [
    "PACK_API_VERSION",
    "PACK_CAPABILITIES",
    "CheckRequest",
    "CommandResult",
    "CompanyScaffoldResult",
    "ContributionKind",
    "OperationPack",
    "OperationPackError",
    "OperationPackMetadata",
    "OperationPackMismatchError",
    "OperationPackRegistry",
    "OperationPackRegistryError",
    "OperationScaffoldError",
    "ReplayRequest",
    "RunRequest",
    "UnknownOperationPackError",
    "ValidateRequest",
    "scaffold_company_operation",
    "scaffold_operation",
]
