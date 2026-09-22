"""OperateBench — a benchmark for agents that own longitudinal operations.

This package ships Core — the domain-neutral operation, time, event, outcome,
ledger and evaluator protocols — together with one synthetic Lifecycle vertical,
:mod:`operatebench.domains.lettings.maintenance`.

Distribution ``operatebench`` installs two importable packages from one wheel:
this one, and ``boundarybench``, the historical Boundary Track package kept for
compatibility under its own version line and its own manifest semantics (see
``docs/VERSIONING.md``). Neither package imports the other's domain semantics.
Two shared primitives are the exceptions, and both are infrastructure rather
than construct. :mod:`operatebench.jsonsafe` is canonical UTF-8 JSON and its
safety checks, which is an encoding utility rather than a Boundary construct;
:mod:`boundarybench.jsonsafe` re-exports it under the name it has always had.
:mod:`operatebench.providers` is the provider kernel — the endpoint, retry, cost
and response-contract machinery of one provider call — which the Boundary Track
composes and the Lifecycle Track is meant to; it imports neither track, and no
track type appears in it.

The public export here is deliberately small. Import from the submodules
directly.
"""

from __future__ import annotations

from operatebench.cards import (
    CardValidationError,
    load_card_json,
    load_field_census,
    recompute_card_digest,
    validate_card,
    validate_field_census,
)
from operatebench.version import CARD_SCHEMA_VERSION, OPERATEBENCH_VERSION

__all__ = [
    "CARD_SCHEMA_VERSION",
    "OPERATEBENCH_VERSION",
    "CardValidationError",
    "load_card_json",
    "load_field_census",
    "recompute_card_digest",
    "validate_card",
    "validate_field_census",
]
