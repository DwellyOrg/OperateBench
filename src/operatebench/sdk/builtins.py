"""The only operation packs this build imports and dispatches."""

from __future__ import annotations

from operatebench.contributions.prospire.property_compliance.pack import (
    PACK as PROPERTY_COMPLIANCE_PACK,
)
from operatebench.domains.commerce.return_refund.pack import RETURN_REFUND_PACK
from operatebench.domains.lettings.maintenance.pack import MAINTENANCE_PACK
from operatebench.sdk.registry import OperationPackRegistry

BUILTIN_PACKS = OperationPackRegistry(
    (MAINTENANCE_PACK, RETURN_REFUND_PACK, PROPERTY_COMPLIANCE_PACK)
)

__all__ = ["BUILTIN_PACKS"]
