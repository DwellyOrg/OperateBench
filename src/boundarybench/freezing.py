"""Deep freezing and detached JSON projection.

Compiled artefacts carry authored data that must not change after its content
digest was taken, so every nested structure is frozen on the way in and copied
back out as plain built-ins on the way to JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from operatebench.jsonsafe import (
    to_json,
    to_json_preserving_tuples,
)


def deep_freeze(value: Any) -> Any:
    """Recursively convert a parsed value into an immutable structure.

    Mappings become read-only proxies over already-frozen children, and every
    sequence becomes a tuple. Scalars pass through unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    return value


__all__ = ["deep_freeze", "to_json", "to_json_preserving_tuples"]
