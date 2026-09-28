"""Strict, finite authored fixture: JSON is a YAML subset; no YAML tags or code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

PACK_ID = "lettings.property_compliance.profiles.v1"
OPERATION_TYPE = "lettings.property_compliance.profiles"
VERSION = "0.1.0"
FIXTURE_DIGEST = "0cb47835c4e95fb8d2c53ea24e86ea44a63e0f831feebdb2f0ac5b85ee06c309"
REFERENCE_SCENARIOS = (
    "EN_normal",
    "EN_remediation",
    "EN_access",
    "EN_reopen",
    "EN_qualification",
    "SC_normal",
    "SC_remediation",
    "SC_access",
    "SC_reopen",
    "SC_qualification",
    "AU_normal",
    "AU_remediation",
    "AU_access",
    "AU_reopen",
    "AU_qualification",
    "NZ_normal",
    "NZ_remediation",
    "NZ_access",
    "NZ_reopen",
    "NZ_qualification",
    "SC_checklist",
    "AU_gas_expired",
    "NZ_assessor_only",
    "EN_wrong_property",
    "EN_no_response",
)


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def unique(pairs: Any) -> Any:
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate fixture key")
        result[k] = v
    return result


@dataclass(frozen=True)
class Spec:
    operation_id: str
    content_digest: str
    profiles: Mapping[str, Any]
    scenarios: Mapping[str, Any]


def load_spec(path: str | Path) -> Spec:
    raw = Path(path).read_bytes()
    if len(raw) > 1000000:
        raise ValueError("fixture exceeds one MiB")
    body = json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )
    if digest(body) != FIXTURE_DIGEST:
        raise ValueError(
            "not the reviewed finite 0.1.0 fixture; unknown fields, "
            "rules, dates and cases are refused"
        )
    return Spec(
        body["operation_id"],
        digest(body),
        freeze(body["profiles"]),
        freeze(body["scenarios"]),
    )


def profile_identity(spec: Spec, scenario_id: str) -> dict[str, str]:
    p = spec.profiles[spec.scenarios[scenario_id]["profile_id"]]
    return {
        "profile_id": p["profile_id"],
        "profile_version": p["profile_version"],
        "profile_digest": digest(p),
        "jurisdiction": p["jurisdiction"],
    }
