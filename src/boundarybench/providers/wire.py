"""The exact JSON a provider sent, under its historical name.

The implementation now lives in :mod:`operatebench.providers.wire`, unchanged.
Reading the bytes before an SDK's typed reading of them is not a Boundary rule —
an SDK coerces, drops and resolves leniently whichever track built the request —
so it belongs in the shared provider kernel that both tracks compose.

This module re-exports **the same objects**, not equivalents: ``WireResponse``
here *is* ``WireResponse`` there, so ``isinstance`` and every existing ``except``
clause behave exactly as they did, and there is no second implementation to drift
from the first.

Historical imports — ``from boundarybench.providers.wire import WireResponse``
and the rest — keep working and keep meaning the same thing.
"""

from __future__ import annotations

from operatebench.providers.usage import MAX_EXACT_TOKEN_COUNT
from operatebench.providers.wire import (
    MAX_PROVIDER_RESPONSE_ID_CHARACTERS,
    MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES,
    MAX_WIRE_DEPTH,
    WireResponse,
    WireShape,
    checked_wire_list,
    checked_wire_mapping,
    checked_wire_object,
    checked_wire_string,
    declared_wire_fields,
    is_wire_token_count,
    wire_invalid,
    wire_json_object,
    wire_token_usage,
)

__all__ = [
    # Re-exported beside this module's own names because the token-domain bound
    # is stated once, in :mod:`operatebench.providers.usage`, and both readers
    # that apply it have always been able to name it from here.
    "MAX_EXACT_TOKEN_COUNT",
    "MAX_PROVIDER_RESPONSE_ID_CHARACTERS",
    "MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES",
    "MAX_WIRE_DEPTH",
    "WireResponse",
    "WireShape",
    "checked_wire_list",
    "checked_wire_mapping",
    "checked_wire_object",
    "checked_wire_string",
    "declared_wire_fields",
    "is_wire_token_count",
    "wire_invalid",
    "wire_json_object",
    "wire_token_usage",
]
