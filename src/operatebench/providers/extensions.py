"""Exact mapping-only extension primitives, not a general deep-freeze policy.

Sequence values deliberately pass through unchanged. APIs whose contracts freeze
lists and tuples (xAI Responses) retain their separate conversion policy.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def frozen_contract(node: Any) -> Any:
    """Copy mapping nodes into read-only mappings, preserving iteration order."""
    if isinstance(node, Mapping):
        return MappingProxyType(
            {key: frozen_contract(value) for key, value in node.items()}
        )
    return node


def plain_contract(node: Any) -> Any:
    """Copy mapping nodes into JSON-encodable dictionaries; leave sequences alone."""
    if isinstance(node, Mapping):
        return {key: plain_contract(value) for key, value in node.items()}
    return node


def typed_extras(value: Any) -> Mapping[str, Any]:
    """One parsed object's undeclared fields, as this SDK parked them."""
    extra = getattr(value, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}
