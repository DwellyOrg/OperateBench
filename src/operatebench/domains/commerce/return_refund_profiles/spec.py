"""Strict, recursively immutable, snapshot-bound development fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from operatebench.core.errors import SpecSchemaError

PACK_ID = "commerce.return_refund.profiles.v1"
OPERATION_TYPE = "commerce.return_refund.profiles"
VERSION = "0.2.0"


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
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
            thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


PROFILES = freeze(
    json.loads((Path(__file__).parent / "resources/profiles.json").read_text())
)


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader: Any, node: Any) -> Any:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if type(key) is not str or key in result:
            raise SpecSchemaError("keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)

BOOL_FIELDS = (
    "ordinary_goods",
    "jurisdiction_stipulated",
    "disclosed",
    "policy_display_proven",
    "unused",
    "rejection_timely_stipulated",
    "original_packaging",
    "claim_label_only",
)
DATE_FIELDS = (
    "purchase_date",
    "delivery_date",
    "notice_sent_date",
    "notice_received_date",
    "dispatch_date",
)
INT_FIELDS = (
    "price_minor",
    "standard_delivery_minor",
    "premium_delivery_minor",
    "proof_delay_minutes",
)
ENUM_FIELDS = {
    "profile_id": tuple(PROFILES),
    "channel": ("distance", "in_store"),
    "reason": ("change_of_mind", "fault"),
    "merchant_policy": ("M0", "M30"),
    "fault": ("none", "minor", "major", "unknown"),
    "consumer_choice": ("refund", "repair", "replacement"),
    "transaction_proof": ("order_reference", "linked_serial", "unknown"),
    "payment_mode": ("normal", "unknown", "failed"),
}
SCENARIO_FIELDS = set(BOOL_FIELDS + DATE_FIELDS + INT_FIELDS + ("starts_at",)) | set(
    ENUM_FIELDS
)


@dataclass(frozen=True)
class Spec:
    operation_id: str
    content_digest: str
    scenarios: Mapping[str, Any]
    profiles: Mapping[str, Any]

    @property
    def spec_digest_sha256(self) -> str:
        return self.content_digest


def load_spec(path: str | Path) -> Spec:
    raw = Path(path).read_bytes()
    if not 0 < len(raw) <= 262144:
        raise SpecSchemaError("fixture size outside 1..262144 bytes")
    try:
        # No aliases: avoids recursive graphs, shared mutable facts and expansion.
        if any(
            isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for t in yaml.scan(raw)
        ):
            raise SpecSchemaError("YAML aliases/anchors are unsupported")
        data = yaml.load(raw, Loader=StrictLoader)
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise SpecSchemaError("malformed fixture") from exc
    expected = {
        "pack_id",
        "operation_type",
        "version",
        "operation_id",
        "profiles",
        "scenarios",
    }
    if type(data) is not dict or set(data) != expected:
        raise SpecSchemaError("wrong top-level fields")
    for k, v in [
        ("pack_id", PACK_ID),
        ("operation_type", OPERATION_TYPE),
        ("version", VERSION),
        ("operation_id", "commerce_return_refund_profiles"),
    ]:
        if data[k] != v:
            raise SpecSchemaError(f"unsupported {k}")
    if data["profiles"] != {k: digest(v) for k, v in PROFILES.items()}:
        raise SpecSchemaError("profile identity/digest mismatch")
    cases = data["scenarios"]
    if type(cases) is not dict or not 1 <= len(cases) <= 64:
        raise SpecSchemaError("scenarios must be a bounded mapping")
    for name, s in cases.items():
        if (
            not name
            or len(name) > 64
            or not name.replace("_", "").isalnum()
            or type(s) is not dict
            or set(s) != SCENARIO_FIELDS
        ):
            raise SpecSchemaError("invalid scenario fields or identity")
        for field in BOOL_FIELDS:
            if type(s[field]) is not bool:
                raise SpecSchemaError(f"{field} must be boolean")
        for field in INT_FIELDS:
            if type(s[field]) is not int or not 0 <= s[field] <= 1000000:
                raise SpecSchemaError(f"{field} must be bounded integer")
        if (
            not 1 <= s["price_minor"] <= 100000
            or not 1 <= s["proof_delay_minutes"] <= 28800
        ):
            raise SpecSchemaError("price/delay outside supported range")
        for field, values in ENUM_FIELDS.items():
            if type(s[field]) is not str or s[field] not in values:
                raise SpecSchemaError(f"unsupported {field}")
        for field in DATE_FIELDS:
            try:
                if (
                    type(s[field]) is not str
                    or date.fromisoformat(s[field]).isoformat() != s[field]
                ):
                    raise ValueError()
            except ValueError as exc:
                raise SpecSchemaError(f"invalid civil date {field}") from exc
        try:
            start = datetime.strptime(s["starts_at"], "%Y-%m-%dT%H:%M:%SZ")
            if start.strftime("%Y-%m-%dT%H:%M:%SZ") != s["starts_at"]:
                raise ValueError("noncanonical UTC timestamp")
        except (ValueError, TypeError) as exc:
            raise SpecSchemaError("starts_at requires canonical UTC timestamp") from exc
        if not (
            s["purchase_date"]
            <= s["delivery_date"]
            <= s["notice_sent_date"]
            <= s["notice_received_date"]
            <= s["dispatch_date"]
        ):
            raise SpecSchemaError("inconsistent chronology")
        if (
            start.date().isoformat() != s["notice_received_date"]
            or start.year != 2026
            or start.month != 1
        ):
            raise SpecSchemaError("only pinned January 2026 synthetic calendar supported")
        proof_date = (
            (start.replace(tzinfo=UTC) + timedelta(minutes=s["proof_delay_minutes"]))
            .astimezone(ZoneInfo(PROFILES[s["profile_id"]]["timezone"]))
            .date()
        )
        if date.fromisoformat(s["dispatch_date"]) > proof_date:
            raise SpecSchemaError("dispatch cannot occur after supplied carrier proof")
        if (
            s["profile_id"] in ("UK", "DE")
            and s["reason"] == "change_of_mind"
            and s["disclosed"]
            and date.fromisoformat(s["dispatch_date"])
            > date.fromisoformat(s["notice_sent_date"]) + timedelta(days=14)
        ):
            raise SpecSchemaError(
                "late dispatch is outside this automatic return profile"
            )
        if s["reason"] == "change_of_mind" and s["fault"] != "none":
            raise SpecSchemaError("inconsistent reason/fault")
    return Spec(
        data["operation_id"], hashlib.sha256(raw).hexdigest(), freeze(cases), PROFILES
    )


def profile_identity(spec: Spec, scenario_id: str) -> dict[str, str]:
    p = spec.profiles[spec.scenarios[scenario_id]["profile_id"]]
    return {k: p[k] for k in ("profile_id", "profile_version", "jurisdiction")} | {
        "profile_digest": digest(p)
    }
