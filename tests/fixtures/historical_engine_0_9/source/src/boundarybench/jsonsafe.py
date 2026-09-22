"""The one recursive JSON and text safety primitive, under its historical name.

The implementation now lives in :mod:`operatebench.jsonsafe`, unchanged. It moved
because the provider kernel under :mod:`operatebench.providers` has to be able to
canonicalise a request body without importing the Boundary Track — a kernel both
tracks compose cannot depend on one of them — and canonical UTF-8 JSON was never
a Boundary construct in the first place: it is the encoding every artefact in
this distribution is hashed as, and :mod:`operatebench.core` was already reaching
across for it.

This module re-exports **the same objects**, not equivalents. Every name below is
bound to the object :mod:`operatebench.jsonsafe` defines, so ``isinstance``, the
:class:`~operatebench.jsonsafe.JsonSafetyError` hierarchy and every existing
``except`` clause behave exactly as they did. Nothing here re-implements
anything, and there is deliberately no second definition to drift from the first.

Historical imports — ``from boundarybench.jsonsafe import canonical_json_text``
and the rest — keep working and keep meaning the same thing.
"""

from __future__ import annotations

from operatebench.jsonsafe import (
    MAX_JSON_DEPTH,
    MAX_JSON_INTEGER_DIGITS,
    REPLACEMENT_CHARACTER,
    CyclicStructureError,
    JsonSafetyError,
    NestingDepthError,
    NonJsonValueError,
    TextEncodingError,
    UnrenderableValueError,
    canonical_json_bytes,
    canonical_json_text,
    check_text,
    ensure_json_safe,
    ensure_raw_json_depth,
    sanitized_text,
    surrogate_index,
    to_json,
    to_json_preserving_tuples,
)

__all__ = [
    "MAX_JSON_DEPTH",
    "MAX_JSON_INTEGER_DIGITS",
    "REPLACEMENT_CHARACTER",
    "CyclicStructureError",
    "JsonSafetyError",
    "NestingDepthError",
    "NonJsonValueError",
    "TextEncodingError",
    "UnrenderableValueError",
    "canonical_json_bytes",
    "canonical_json_text",
    "check_text",
    "ensure_json_safe",
    "ensure_raw_json_depth",
    "sanitized_text",
    "surrogate_index",
    "to_json",
    "to_json_preserving_tuples",
]
