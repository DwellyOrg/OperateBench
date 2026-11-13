"""Finite immutable synthetic specification, pinned independently of file location."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .input import InputError, read_json
from .input import unique as unique

PACK_ID = "partner.finsi.governed_campaign_launch.synthetic"
OPERATION_TYPE = "finsi.governed_campaign_launch"
REFERENCE_SCENARIOS = (
    "baseline",
    "consent_revoked",
    "consent_unenforced",
    "partial_import",
    "chat_claim",
    "supersession",
    "timeout_idempotent",
    "timeout_unprotected",
    "preview",
    "late_settlement",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Spec:
    operation_id: str
    content_digest: str
    scenarios: Mapping[str, Any]


def load_spec(path: str | Path) -> Spec:
    body = read_json(path, 100000)
    if digest(body) != FIXTURE_DIGEST:
        raise InputError("unreviewed finite fixture")
    return Spec(
        body["operation_id"],
        digest(body),
        MappingProxyType({k: MappingProxyType(v) for k, v in body["scenarios"].items()}),
    )


FIXTURE_DIGEST = "f27b692c5168acd23e0ed06ecefd0371521afb1064d0ee102045ab67c5b7e67e"
