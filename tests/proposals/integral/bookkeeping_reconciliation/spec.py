"""Strict invented single-month input. JSON is the fixture's YAML subset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from operatebench.core.errors import SpecSchemaError

PACK_ID = "accounting.bookkeeping_reconciliation.synthetic.v1"
OPERATION_ID = "bookkeeping_reconciliation_synthetic_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Spec:
    data: dict[str, Any]
    content_digest: str


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecSchemaError("duplicate fixture key")
        result[key] = value
    return result


def load_spec(path: str | Path) -> Spec:
    body = Path(path).read_bytes()
    if len(body) > 16384:
        raise SpecSchemaError("fixture exceeds 16 KiB")
    try:
        data = json.loads(body, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SpecSchemaError("invalid fixture JSON/YAML subset") from exc
    fixed = {
        "operation_type": PACK_ID,
        "operation_id": OPERATION_ID,
        "version": "0.1.0",
        "privacy_status": "SYNTHETIC_ONLY",
        "company": "Example Lantern Workshop",
        "month": "2026-08",
        "transaction_id": "bank-line-1",
        "document_id": "receipt-1",
        "amount_minor": 12000,
        "currency": "SYN",
    }
    if not isinstance(data, dict) or set(data) != {*fixed, "scenarios"}:
        raise SpecSchemaError("unexpected fixture fields")
    if any(type(data[k]) is not type(v) or data[k] != v for k, v in fixed.items()):
        raise SpecSchemaError("wrong operation or invented case identity")
    scenarios = data["scenarios"]
    if not isinstance(scenarios, dict) or set(scenarios) != {
        "match",
        "conflict",
        "missing",
        "failed_zero",
        "reopen",
    }:
        raise SpecSchemaError("unexpected scenarios")
    for row in scenarios.values():
        if not isinstance(row, dict) or set(row) != {
            "receipt",
            "connector",
            "correction",
        }:
            raise SpecSchemaError("unexpected scenario fields")
        if (
            row["receipt"] not in ("match", "conflict", "missing")
            or row["connector"] not in ("ok", "failed")
            or type(row["correction"]) is not bool
        ):
            raise SpecSchemaError("invalid synthetic scenario")
    return Spec(data, digest(data))
