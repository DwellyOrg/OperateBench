# SPDX-License-Identifier: Apache-2.0
"""Independent executable contract for audit-integrity successor resources."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "schemas"
PACKAGED = ROOT / "src" / "operatebench" / "resources" / "identity"
FILES = (
    "episode-outcome-v3.schema.json",
    "execution-record-v2.schema.json",
    "executed-runtime-evidence-v2.schema.json",
    "audit-integrity-successor-registry-v1.schema.json",
    "audit-integrity-successor-registry-v1.json",
    "audit-integrity-successor-census-v1.schema.json",
    "audit-integrity-successor-census-v1.json",
    "audit-integrity-successor-v1.golden.json",
)
REFS = (
    "episode-outcome-v3.schema.json",
    "execution-record-v2.schema.json",
    "executed-runtime-evidence-v2.schema.json",
    "decision-tape-v1.schema.json",
    "provider-execution-binding-v1.schema.json",
    "domain-generated-audit-budget-evidence-v1.schema.json",
)
CODES = (
    "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    "DOMAIN_AUDIT_COUNTER_INTEGRITY",
)
DOMAIN = "operatebench.digest.executed_runtime_evidence.v2"
SCHEMA = "operatebench.executed_runtime_evidence.v2"
AUDIT_DOMAIN = "operatebench.digest.domain-generated-audit-budget-evidence.v1"
AUDIT_CONTRACT = "operatebench.domain-generated-audit-budget.v1"
EXPECTED_BOUNDS: Mapping[str, int] = MappingProxyType(
    {
        "max_depth": 64,
        "max_non_audit_nodes": 100_000,
        "max_non_audit_payload_utf8_bytes": 8_388_608,
        "max_total_nodes": 2_100_000,
        "max_total_payload_utf8_bytes": 268_435_456,
        "max_container_members": 4096,
        "audit_records_max_items": 100_000,
        "max_string_utf8_bytes": 1_048_576,
        "safe_integer_min": -9_007_199_254_740_991,
        "safe_integer_max": 9_007_199_254_740_991,
        "audit_declared_max_integer": 9_223_372_036_854_775_807,
    }
)
# Compatibility name for pre-existing arithmetic and golden assertions. The oracle's
# authority is the independently authored immutable literal above.
BOUNDS = EXPECTED_BOUNDS
EXPECTED_NAMES = (
    "zero_deterministic_success",
    "model_started_success",
    "assigned_preexecution_model_not_started_error",
    "midexecution_model_started_error",
    "post_finality_error",
)
EXPECTED_CASES = {
    "zero_deterministic_success": (
        "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
        "opinst_00000000000000000000000000000000",
        '{"kind":"domain_terminal"}',
        "completed_successfully",
        True,
        0,
        "deterministic",
        None,
        None,
        None,
        "not_applicable",
        0,
        0,
        0,
        None,
        None,
        AUDIT_CONTRACT,
        "1" * 64,
        0,
        ("terminal_summary",),
        None,
        None,
        0,
        "01f7581992512e419cddb1f8655e2068c92821e65593f88cd84d24af1bcdc92f",
        2126,
        "89a9853979987cd58d2a637fb7662ab1a17dab836dfb6375a19f2128e7c0afb9",
    ),
    "model_started_success": (
        "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
        "opinst_99999999999999999999999999999999",
        '{"kind":"domain_terminal"}',
        "completed_successfully",
        True,
        1,
        "model",
        "synthetic-model",
        "operatebench.model.v4",
        4096,
        "started",
        1,
        1,
        1,
        ("openai", "responses", "synthetic-model"),
        "b07408ab6156ddec3227fb2d6770b0e5c77bf1c3ba80673919d84c1e1675a46d",
        AUDIT_CONTRACT,
        "2" * 64,
        3,
        ("committed_intent", "committed_intent", "committed_intent", "terminal_summary"),
        None,
        None,
        3,
        "f5b3e0bf2a1b3c805164e787bf70a201b19029cea5dd4bf7f26033b6a2980494",
        4358,
        "fa12efe2e4238c172aaacc497abaff0626214e9ec0bfd45fffa087ad7cc203e0",
    ),
    "assigned_preexecution_model_not_started_error": (
        "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
        "opinst_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        '{"code":"DOMAIN_AUDIT_CONTRACT_VIOLATION","contract_id":"operatebench.domain-generated-audit-budget.v1","kind":"integrity_error"}',
        None,
        False,
        0,
        "model",
        "synthetic-model",
        "operatebench.model.v4",
        4096,
        "not_started",
        0,
        0,
        0,
        None,
        None,
        AUDIT_CONTRACT,
        "6" * 64,
        99999,
        ("integrity_finding", "terminal_summary"),
        "v1_capacity_exceeded",
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        0,
        "5f123a364b2ba3dc32b468ed5fb58f3ea8e2886636918f4a496ba304a3864d21",
        2626,
        "d5af801a828f20f0e2093f0e43872a904e59d6642392165b0adccc349bd3ca85",
    ),
    "midexecution_model_started_error": (
        "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
        "opinst_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        '{"code":"DOMAIN_AUDIT_BUDGET_EXCEEDED","contract_id":"operatebench.domain-generated-audit-budget.v1","kind":"integrity_error"}',
        None,
        False,
        1,
        "model",
        "synthetic-model",
        "operatebench.model.v4",
        4096,
        "started",
        1,
        1,
        1,
        ("openai", "responses", "synthetic-model"),
        "b07408ab6156ddec3227fb2d6770b0e5c77bf1c3ba80673919d84c1e1675a46d",
        AUDIT_CONTRACT,
        "2" * 64,
        3,
        (
            "committed_intent",
            "committed_intent",
            "committed_intent",
            "integrity_finding",
            "terminal_summary",
        ),
        "valid_reservation_above_declared_aggregate",
        "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        3,
        "c4f474c56892931eb8e9fc287591dc2da81f877ed10cc2ee27152803e64f8905",
        4867,
        "9dcceda655e7bb83418ce4c056bf574afb138b37b971a7daaec8dbfbe8002ae6",
    ),
    "post_finality_error": (
        "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc",
        "opinst_cccccccccccccccccccccccccccccccc",
        '{"code":"DOMAIN_AUDIT_CONTRACT_VIOLATION","contract_id":"operatebench.domain-generated-audit-budget.v1","kind":"integrity_error"}',
        "completed_successfully",
        True,
        0,
        "deterministic",
        None,
        None,
        None,
        "not_applicable",
        0,
        0,
        0,
        None,
        None,
        AUDIT_CONTRACT,
        "6" * 64,
        3,
        ("integrity_finding", "terminal_summary"),
        "post_finality_attempt",
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        0,
        "76f005324ee39197a922c7d808f6f13d3fd3adf48797932f6e56347ec4c0cc8e",
        2613,
        "34f55c26c3ec5f4b5db56816cdf6051a91e2f7c4caf157d297e27b97ba58bc7b",
    ),
}
EXPECTED_REFUSALS = {
    (2, 1): "accept_historical_no_audit_claim",
    (3, 1): "unsupported_outcome_schema_version",
    (2, 2): "outcome_evidence_generation_mismatch",
    (3, 2): "shape_only_no_authority",
}
REFUSAL_FIELDS = (
    "integrity_code",
    "stage",
    "source_id_rule",
    "requested_count_rule",
    "transition_id_rule",
    "invocation_index_rule",
    "pre_operation_count_rule",
    "pre_source_count_rule",
    "declared_max_source",
    "finding_required",
    "finding_integrity_code",
)
EXPECTED_REFUSAL_ROWS = {
    "unsupported_or_wrong_contract": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "non_exact_type_or_boolean": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "negative_value": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "duplicate_or_unauthorized_order": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "source_or_aggregate_mismatch": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "v1_capacity_exceeded": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "pre_identity",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "no_evidence",
        "eligibility_candidate_only_no_evidence",
        False,
        None,
    ),
    "unknown_source_or_kind": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "assigned_transition",
        "exact_attempted_source",
        "positive_exact_attempted_batch_size",
        "exact_attempted_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before_or_null_if_unknown",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    ),
    "invalid_batch_or_identity": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "assigned_transition",
        "exact_attempted_source",
        "zero",
        "exact_attempted_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    ),
    "reservation_lifecycle_violation": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "assigned_transition",
        "exact_reservation_source",
        "positive_exact_reservation_size",
        "exact_reservation_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    ),
    "post_finality_attempt": (
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "assigned_preexecution",
        "null",
        "zero",
        "null",
        "null",
        "exact_authoritative_committed_before",
        "null",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    ),
    "checked_signed_int64_arithmetic_overflow": (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "assigned_transition",
        "exact_attempted_source",
        "positive_exact_attempted_batch_size",
        "exact_attempted_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
    ),
    "malformed_runtime_counter": (
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "assigned_preexecution",
        "null",
        "zero",
        "null",
        "null",
        "zero_untrusted_counter",
        "null",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_COUNTER_INTEGRITY",
    ),
    "valid_reservation_above_declared_aggregate": (
        "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "assigned_transition",
        "exact_attempted_source",
        "positive_exact_attempted_batch_size",
        "exact_attempted_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    ),
    "valid_reservation_above_declared_source": (
        "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "assigned_transition",
        "exact_attempted_source",
        "positive_exact_attempted_batch_size",
        "exact_attempted_transition",
        "exact_current_or_null_when_not_invocation_scoped",
        "exact_authoritative_committed_before",
        "exact_authoritative_source_committed_before",
        "accepted_card_value",
        True,
        "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    ),
}
EXPECTED_CENSUS_ROWS = (
    (
        "/episode_outcome/status",
        "Core runtime outcome contract",
        "future Core terminal disposition",
        "future successor evidence composer and replay verifier",
        "exactly_one_closed_tag",
        "nonnull",
    ),
    (
        "/execution_record/provider_dispatch_state",
        "execution recorder contract",
        "future provider dispatch boundary",
        "future successor evidence verifier",
        "exactly_one_enum",
        "nonnull",
    ),
    (
        "/executed_runtime_evidence/domain_generated_audit_budget_evidence",
        "Core runtime audit evidence",
        "future authoritative Core reservation accounting",
        "future successor evidence composer and verifier",
        "exactly_one_complete_v1_object",
        "nonnull",
    ),
    (
        "/executed_runtime_evidence/domain_generated_audit_budget_evidence_digest_sha256",
        "executed evidence composer contract",
        "frozen audit evidence wrapped projection",
        "future successor evidence verifier",
        "exactly_one_sha256",
        "nonnull",
    ),
)
EXPECTED_DEPENDENCIES = (
    "operatebench.domain-generated-audit-budget.v1",
    "operation_core_content_digest_sha256",
    "future authorized Card-v2 aggregate and source declaration",
)
EXPECTED_FROZEN = (
    "episode-outcome-v2.schema.json",
    "execution-record-v1.schema.json",
    "executed-runtime-evidence-v1.schema.json",
    "executed-runtime-evidence-v1.golden.json",
    "decision-tape-v1.schema.json",
    "provider-execution-binding-v1.schema.json",
    "domain-generated-audit-budget-v1.schema.json",
    "domain-generated-audit-budget-evidence-v1.schema.json",
    "domain-generated-audit-budget-registry-v1.schema.json",
    "domain-generated-audit-budget-registry-v1.json",
    "domain-generated-audit-budget-census-v1.schema.json",
    "domain-generated-audit-budget-census-v1.json",
    "domain-generated-audit-budget-v1.golden.json",
    "operation-card-v1.schema.json",
    "operation-card-v2.schema.json",
    "runtime-identity-adapter-registry-v1.json",
    "runtime-identity-adapter-census-v1.json",
)


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


_AUDIT_MEMBER = "domain_generated_audit_budget_evidence"
_AUDIT_RECORDS_PATH = (_AUDIT_MEMBER, "records")
_AUDIT_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "contract_id",
        "operation_core_content_digest_sha256",
        "operation_instance_id",
        "declared_max_events",
        "records",
    }
)


def _utf8_size(value: str, *, label: str) -> int:
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} contains a lone surrogate") from exc


def _canonical_bounded(value: Any, *, label: str, maximum: int) -> bytes:
    encoded = canonical(value)
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds {maximum} bytes")
    return encoded


def _reject_raw_depth(text: str, *, maximum: int) -> None:
    """Reject excessive container nesting before the recursive stdlib parser runs."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                raise ValueError("container depth exceeds bounded profile before parse")
        elif character in "]}" and depth:
            depth -= 1


def validate_bounded_profile(
    value: Any, *, bounds: Mapping[str, int] = EXPECTED_BOUNDS
) -> None:
    """Execute the frozen fail-closed profile without recursive Python traversal.

    Containers alone contribute to depth: a container at the document root is depth
    one, nested containers add one, and scalar leaves add none. Structural nodes are
    every JSON value/container, every object key, and every array-element slot. The
    non-audit projection omits exactly the root audit member (key plus value subtree),
    and only after its envelope has the protocol's exact key set.
    """
    audit = value.get(_AUDIT_MEMBER) if isinstance(value, dict) else None
    if (
        isinstance(value, dict)
        and _AUDIT_MEMBER in value
        and (not isinstance(audit, dict) or set(audit) != _AUDIT_ENVELOPE_KEYS)
    ):
        raise ValueError("exact audit envelope is required for bounded exception")

    total_nodes = 0
    non_audit_nodes = 0
    root_depth = 1 if isinstance(value, (dict, list)) else 0
    stack: list[tuple[Any, tuple[str | int, ...], int, bool]] = [
        (value, (), root_depth, True)
    ]
    while stack:
        item, path, depth, is_non_audit = stack.pop()
        total_nodes += 1
        non_audit_nodes += int(is_non_audit)
        if total_nodes > bounds["max_total_nodes"]:
            raise ValueError("total nodes exceed bounded profile")
        if non_audit_nodes > bounds["max_non_audit_nodes"]:
            raise ValueError("non-audit nodes exceed bounded profile")

        if isinstance(item, str):
            if _utf8_size(item, label="JSON string") > bounds["max_string_utf8_bytes"]:
                raise ValueError("string UTF-8 bytes exceed bounded profile")
            continue
        if item is None or isinstance(item, (bool, int, float)):
            continue
        if not isinstance(item, (dict, list)):
            raise ValueError("value is not a JSON type")
        if depth > bounds["max_depth"]:
            raise ValueError("container depth exceeds bounded profile")

        member_limit = (
            bounds["audit_records_max_items"]
            if path == _AUDIT_RECORDS_PATH and isinstance(item, list)
            else bounds["max_container_members"]
        )
        if len(item) > member_limit:
            raise ValueError("container members exceed bounded profile")

        child_depth = depth + 1
        if isinstance(item, list):
            for index in range(len(item) - 1, -1, -1):
                # An array position is a structural node independent of its value.
                total_nodes += 1
                non_audit_nodes += int(is_non_audit)
                if total_nodes > bounds["max_total_nodes"]:
                    raise ValueError("total nodes exceed bounded profile")
                if non_audit_nodes > bounds["max_non_audit_nodes"]:
                    raise ValueError("non-audit nodes exceed bounded profile")
                child = item[index]
                stack.append(
                    (
                        child,
                        (*path, index),
                        child_depth if isinstance(child, (dict, list)) else depth,
                        is_non_audit,
                    )
                )
        else:
            for key, child in reversed(tuple(item.items())):
                if not isinstance(key, str):
                    raise ValueError("JSON object key is not a string")
                if (
                    _utf8_size(key, label="JSON object key")
                    > bounds["max_string_utf8_bytes"]
                ):
                    raise ValueError("string UTF-8 bytes exceed bounded profile")
                child_is_non_audit = is_non_audit and not (
                    path == () and key == _AUDIT_MEMBER
                )
                # Object keys are structural nodes; the exact audit key is omitted
                # together with its value from the non-audit projection.
                total_nodes += 1
                non_audit_nodes += int(child_is_non_audit)
                if total_nodes > bounds["max_total_nodes"]:
                    raise ValueError("total nodes exceed bounded profile")
                if non_audit_nodes > bounds["max_non_audit_nodes"]:
                    raise ValueError("non-audit nodes exceed bounded profile")
                stack.append(
                    (
                        child,
                        (*path, key),
                        child_depth if isinstance(child, (dict, list)) else depth,
                        child_is_non_audit,
                    )
                )

    _canonical_bounded(
        value,
        label="canonical JSON UTF-8 bytes",
        maximum=bounds["max_total_payload_utf8_bytes"],
    )
    projection = (
        {key: item for key, item in value.items() if key != _AUDIT_MEMBER}
        if isinstance(value, dict) and _AUDIT_MEMBER in value
        else value
    )
    _canonical_bounded(
        projection,
        label="non-audit canonical JSON UTF-8 bytes",
        maximum=bounds["max_non_audit_payload_utf8_bytes"],
    )


def strict_json_loads(
    raw: bytes | str, *, bounds: Mapping[str, int] = EXPECTED_BOUNDS
) -> Any:
    """Bound raw UTF-8, then reject ambiguity and execute the structural profile."""
    if isinstance(raw, bytes):
        raw_size = len(raw)
        if raw_size > bounds["max_total_payload_utf8_bytes"]:
            raise ValueError("raw JSON UTF-8 bytes exceed bounded profile")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("raw JSON is not strict UTF-8") from exc
    elif isinstance(raw, str):
        raw_size = _utf8_size(raw, label="raw JSON")
        text = raw
    else:
        raise TypeError("raw JSON must be bytes or str")
    if raw_size > bounds["max_total_payload_utf8_bytes"]:
        raise ValueError("raw JSON UTF-8 bytes exceed bounded profile")
    _reject_raw_depth(text, maximum=bounds["max_depth"])
    value = json.loads(text, object_pairs_hook=_closed_object)
    validate_bounded_profile(value, bounds=bounds)
    return value


def load(name: str) -> Any:
    return strict_json_loads((DOCS / name).read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def audit_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": AUDIT_DOMAIN,
        "content": {
            k: value[k]
            for k in (
                "contract_id",
                "operation_core_content_digest_sha256",
                "operation_instance_id",
                "declared_max_events",
                "records",
            )
        },
    }


def outer_projection(value: dict[str, Any]) -> dict[str, Any]:
    content = copy.deepcopy(value)
    del content["runtime_evidence_digest_sha256"]
    return {"domain": DOMAIN, "schema": SCHEMA, "content": content}


def registry() -> Registry:
    result = Registry()
    for name in REFS:
        document = load(name)
        result = result.with_resource(document["$id"], Resource.from_contents(document))
    return result


def _instant(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"calendar instant: {label}") from exc
    return parsed.replace(tzinfo=UTC)


def _validate_outcome_content(outcome: dict[str, Any]) -> None:
    started = _instant(outcome["started_at"], label="started_at")
    ended = _instant(outcome["ended_at"], label="ended_at")
    if ended < started:
        raise ValueError("instant ordering")
    elapsed = int((ended - started).total_seconds())
    if elapsed % 60 or elapsed // 60 != outcome["simulated_minutes"]:
        raise ValueError("elapsed minutes")

    event_keys: list[tuple[datetime, int, str]] = []
    for index, event in enumerate(outcome["events"]):
        at = _instant(event["at"], label=f"events/{index}/at")
        if not started <= at <= ended:
            raise ValueError("event instant range")
        event_keys.append((at, event["sequence"], event["event_id"]))
    if any(later <= earlier for earlier, later in pairwise(event_keys)):
        raise ValueError("event ordering")
    if len({event["event_id"] for event in outcome["events"]}) != len(outcome["events"]):
        raise ValueError("event identity")

    if digest(outcome["final_state"]) != outcome["final_state_digest_sha256"]:
        raise ValueError("final state digest")
    if digest(outcome["trajectory"]) != outcome["trajectory_digest_sha256"]:
        raise ValueError("trajectory digest")


def validate_semantics(evidence: dict[str, Any]) -> None:
    validate_bounded_profile(evidence)
    Draft202012Validator(
        load("executed-runtime-evidence-v2.schema.json"), registry=registry()
    ).validate(evidence)
    outcome = evidence["outcome"]
    execution = evidence["execution_record"]
    audit = evidence["domain_generated_audit_budget_evidence"]
    records = audit["records"]
    summary = records[-1]
    findings = [row for row in records if row["record_type"] == "integrity_finding"]
    commits = [row for row in records if row["record_type"] == "committed_intent"]

    _validate_outcome_content(outcome)

    assert evidence["operation_instance_id"] == audit["operation_instance_id"]
    assert audit["contract_id"] == AUDIT_CONTRACT
    assert audit["operation_core_content_digest_sha256"]
    assert (
        digest(audit_projection(audit))
        == evidence["domain_generated_audit_budget_evidence_digest_sha256"]
    )
    assert (
        digest(outer_projection(evidence)) == evidence["runtime_evidence_digest_sha256"]
    )

    # Frozen audit V1 ordering, joins, counters, and exact assigned refusal rows.
    phase = "commits"
    batches: dict[tuple[str, str], list[int]] = {}
    for row in records:
        assert row["contract_id"] == AUDIT_CONTRACT
        kind = row["record_type"]
        if kind == "committed_intent":
            assert phase == "commits"
            batches.setdefault((row["transition_id"], row["source_id"]), []).append(
                row["batch_ordinal"]
            )
        elif kind == "integrity_finding":
            assert phase == "commits"
            phase = "finding"
        else:
            assert kind == "terminal_summary" and phase in {"commits", "finding"}
            phase = "summary"
    assert phase == "summary"
    assert sum(row["record_type"] == "terminal_summary" for row in records) == 1
    assert len(findings) <= 1
    assert [row["operation_committed_count"] for row in commits] == list(
        range(1, len(commits) + 1)
    )
    assert len({row["intent_id"] for row in commits}) == len(commits)
    assert len({row["intent_digest_sha256"] for row in commits}) == len(commits)
    assert all(ordinals == list(range(len(ordinals))) for ordinals in batches.values())
    assert summary["declared_max_events"] == audit["declared_max_events"]
    assert summary["committed_count"] == len(commits)
    for finding in findings:
        assert finding["declared_max_events"] == audit["declared_max_events"]
        refusal = finding["refusal_class"]
        if refusal == "valid_reservation_above_declared_source":
            raise AssertionError("source overrun requires authenticated declaration")
        expected = EXPECTED_REFUSAL_ROWS[refusal]
        rule = dict(zip(REFUSAL_FIELDS, expected, strict=True))
        exceptional_preexecution = refusal in {
            "source_or_aggregate_mismatch",
            "v1_capacity_exceeded",
        }
        assert rule["finding_integrity_code"] == finding["integrity_code"] or (
            exceptional_preexecution
            and finding["integrity_code"] == "DOMAIN_AUDIT_CONTRACT_VIOLATION"
        )
        if rule["stage"] == "pre_identity":
            assert exceptional_preexecution
            assert finding["source_id"] is None
            assert finding["requested_count"] == 0
            assert finding["transition_id"] is None
            assert finding["invocation_index"] is None
            assert finding["operation_committed_before"] == 0
            assert finding["source_committed_before"] is None
        elif rule["stage"] == "assigned_preexecution":
            assert finding["source_id"] is None
            assert finding["requested_count"] == 0
            assert finding["transition_id"] is None
            assert finding["invocation_index"] is None
            assert finding["source_committed_before"] is None
        else:
            assert isinstance(finding["source_id"], str) and finding["source_id"]
            assert isinstance(finding["transition_id"], str) and finding["transition_id"]
            if rule["requested_count_rule"] == "zero":
                assert finding["requested_count"] == 0
            else:
                assert finding["requested_count"] > 0
            assert finding["operation_committed_before"] == len(commits)
            assert finding["source_committed_before"] == sum(
                row["source_id"] == finding["source_id"] for row in commits
            )
        if finding["invocation_index"] is not None:
            assert finding["invocation_index"] <= outcome["invocations"]
        if refusal == "valid_reservation_above_declared_aggregate":
            assert finding["integrity_code"] == "DOMAIN_AUDIT_BUDGET_EXCEEDED"
            assert (
                len(commits) + finding["requested_count"] > audit["declared_max_events"]
            )

    kind = outcome["status"]["kind"]
    if kind == "domain_terminal":
        assert outcome["terminal_outcome"] is not None
        assert outcome["replay_final"] is True
    if kind == "integrity_error":
        assert summary["result"] == "ERROR" and len(findings) == 1
        assert records[-2] == findings[0]
        assert outcome["status"]["contract_id"] == audit["contract_id"]
        assert (
            findings[0]["integrity_code"]
            == summary["integrity_code"]
            == outcome["status"]["code"]
        )
    else:
        assert summary["result"] == "SUCCESS"
        assert summary["integrity_code"] is None
        assert not findings

    state = execution["provider_dispatch_state"]
    calls = execution["transport_calls"]
    attempts = execution["attempts"]
    decisions = evidence["decision_tape"]["decisions"]
    binding = evidence["provider_binding"]
    assert digest(decisions) == evidence["decision_tape"]["decisions_digest_sha256"]
    if execution["outcome_source"] != "model":
        assert decisions == []
        assert state == "not_applicable" and calls == 0 and attempts == []
        assert binding is None
        assert (
            execution["model"]
            is execution["protocol_version"]
            is execution["max_output_tokens"]
            is None
        )
    elif state == "not_started":
        assert calls == 0 and attempts == [] and decisions == [] and binding is None
        assert kind == "integrity_error" and findings[0]["invocation_index"] is None
        refusal = findings[0]["refusal_class"]
        assert refusal in {
            "post_finality_attempt",
            "malformed_runtime_counter",
            "source_or_aggregate_mismatch",
            "v1_capacity_exceeded",
        }
        assert execution["model"] and execution["protocol_version"]
        assert execution["max_output_tokens"] > 0
    else:
        assert state == "started"
        assert calls == len(attempts) == len(decisions) >= 1
        assert binding is not None
        assert execution["model"] and execution["protocol_version"]
        assert execution["max_output_tokens"] > 0
        assert binding["model"] == execution["model"]
        assert binding["provider_calls"] == binding["attempts"] == calls
        assert binding["decision_call_index"] == list(range(calls))
        pairs = [(row["invocation_index"], row["turn_index"]) for row in decisions]
        assert pairs == sorted(pairs) and len(pairs) == len(set(pairs))
        assert all(1 <= invocation <= outcome["invocations"] for invocation, _ in pairs)
        for attempt, decision in zip(attempts, decisions, strict=True):
            assert (attempt["invocation_index"], attempt["turn_index"]) == (
                decision["invocation_index"],
                decision["turn_index"],
            )
            assert attempt["fault"] is None
            expected_outcome = (
                "classified" if decision["outcome"]["kind"] == "MALFORMED" else "decided"
            )
            assert attempt["outcome"] == expected_outcome
    if findings:
        finding_invocation = findings[0]["invocation_index"]
        if finding_invocation is not None:
            assert all(
                row["invocation_index"] <= finding_invocation
                for row in [*attempts, *decisions]
            )
    # Frozen V1 canonical evidence refuses exclusions entirely.
    assert execution["excluded"] is False
    assert execution["exclusion_code"] is None
    assert execution["exclusion_detail"] == ""


def test_successor_resource_set_is_closed_meta_valid_and_packaged_byte_identical() -> (
    None
):
    assert set(FILES) <= {p.name for p in DOCS.iterdir()}
    for path in (*DOCS.glob("*.json"), *PACKAGED.glob("*.json")):
        strict_json_loads(path.read_bytes())
    for name in FILES:
        docs_raw = (DOCS / name).read_bytes()
        package_raw = (PACKAGED / name).read_bytes()
        assert docs_raw == package_raw
        assert docs_raw.endswith(b"\n")
        strict_json_loads(docs_raw)
    for name in (*REFS[:3], FILES[3], FILES[5]):
        Draft202012Validator.check_schema(load(name))
    assert registry()


def test_episode_outcome_v3_retains_v2_fields_and_freezes_tagged_status_only() -> None:
    old, new = (
        load("episode-outcome-v2.schema.json"),
        load("episode-outcome-v3.schema.json"),
    )
    assert new["properties"]["schema"] == {"const": "operatebench.episode_outcome.v3"}
    assert new["properties"]["schema_version"] == {"const": 3}
    assert new["required"] == old["required"]
    assert set(new["properties"]) == set(old["properties"])
    for field in set(old["properties"]) - {"schema", "schema_version", "status"}:
        assert new["properties"][field] == old["properties"][field]
    branches = new["properties"]["status"]["oneOf"]
    assert [b["properties"]["kind"]["const"] for b in branches] == [
        "domain_terminal",
        "engine_halt",
        "integrity_error",
    ]
    assert tuple(branches[1]["properties"]["code"]["enum"]) == (
        "operation_deadlock",
        "operational_horizon_exhausted",
    )
    assert branches[2]["properties"]["contract_id"]["const"] == AUDIT_CONTRACT
    assert tuple(branches[2]["properties"]["code"]["enum"]) == CODES
    assert "primary_status" not in new["properties"]
    assert not any(
        "audit" in key or ("digest" in key and key not in old["properties"])
        for key in new["properties"]
    )


def test_execution_record_v2_is_exact_v1_plus_dispatch_state() -> None:
    old, new = (
        load("execution-record-v1.schema.json"),
        load("execution-record-v2.schema.json"),
    )
    assert new["required"] == [*old["required"], "provider_dispatch_state"]
    assert set(new["properties"]) == set(old["properties"]) | {"provider_dispatch_state"}
    for field in old["properties"]:
        expected = old["properties"][field]
        if field == "schema":
            expected = {"const": "operatebench.execution_record.v2"}
        if field == "schema_version":
            expected = {"const": 2}
        assert new["properties"][field] == expected
    assert new["properties"]["provider_dispatch_state"]["enum"] == [
        "not_applicable",
        "not_started",
        "started",
    ]


def test_evidence_v2_exact_refs_required_audit_and_self_digest() -> None:
    old, new = (
        load("executed-runtime-evidence-v1.schema.json"),
        load("executed-runtime-evidence-v2.schema.json"),
    )
    assert new["required"] == [
        *old["required"][:-1],
        "domain_generated_audit_budget_evidence",
        "domain_generated_audit_budget_evidence_digest_sha256",
        old["required"][-1],
    ]
    assert new["properties"]["outcome"] == {"$ref": "episode-outcome-v3.schema.json"}
    assert new["properties"]["execution_record"] == {
        "$ref": "execution-record-v2.schema.json"
    }
    assert new["properties"]["domain_generated_audit_budget_evidence"] == {
        "$ref": "domain-generated-audit-budget-evidence-v1.schema.json"
    }
    assert set(new["properties"]) == set(old["properties"]) | {
        "domain_generated_audit_budget_evidence",
        "domain_generated_audit_budget_evidence_digest_sha256",
    }


def _validate_golden_vectors(golden: dict[str, Any]) -> None:
    assert golden["digest_domain"] == DOMAIN
    assert golden["bounds"] == BOUNDS
    assert tuple(v["name"] for v in golden["valid_vectors"]) == EXPECTED_NAMES
    for vector in golden["valid_vectors"]:
        name, evidence = vector["name"], vector["evidence"]
        validate_semantics(evidence)
        wrapped = canonical(outer_projection(evidence))
        assert wrapped.decode() == vector["canonical_wrapper_utf8"]
        assert wrapped.hex() == vector["canonical_wrapper_utf8_hex"]
        assert len(wrapped) == vector["canonical_wrapper_bytes"]
        assert hashlib.sha256(wrapped).hexdigest() == vector["expected_sha256"]
        audit = evidence["domain_generated_audit_budget_evidence"]
        execution = evidence["execution_record"]
        outcome = evidence["outcome"]
        records = audit["records"]
        finding = next(
            (r for r in records if r["record_type"] == "integrity_finding"), None
        )
        binding = evidence["provider_binding"]
        assert EXPECTED_CASES[name] == (
            evidence["identity_manifest_digest_sha256"],
            evidence["operation_instance_id"],
            canonical(outcome["status"]).decode(),
            outcome["terminal_outcome"],
            outcome["replay_final"],
            outcome["invocations"],
            execution["outcome_source"],
            execution["model"],
            execution["protocol_version"],
            execution["max_output_tokens"],
            execution["provider_dispatch_state"],
            execution["transport_calls"],
            len(execution["attempts"]),
            len(evidence["decision_tape"]["decisions"]),
            None
            if binding is None
            else (binding["provider"], binding["api"], binding["model"]),
            None if binding is None else digest(binding),
            audit["contract_id"],
            audit["operation_core_content_digest_sha256"],
            audit["declared_max_events"],
            tuple(r["record_type"] for r in records),
            None if finding is None else finding["refusal_class"],
            records[-1]["integrity_code"],
            records[-1]["committed_count"],
            evidence["domain_generated_audit_budget_evidence_digest_sha256"],
            vector["canonical_wrapper_bytes"],
            vector["expected_sha256"],
        )
        assert evidence["runtime_evidence_digest_sha256"] == vector["expected_sha256"]
        assert vector["authority_status"] == "wire_shape_only_no_runtime_authority"


def test_golden_vectors_validate_recompute_and_pin_semantics() -> None:
    _validate_golden_vectors(load("audit-integrity-successor-v1.golden.json"))


def test_coherent_manifest_and_terminal_repins_fail_independent_golden_literals() -> None:
    for field, value in (
        ("identity_manifest_digest_sha256", "f" * 64),
        ("terminal_outcome", "different_terminal"),
    ):
        golden = load("audit-integrity-successor-v1.golden.json")
        vector = golden["valid_vectors"][0]
        target = (
            vector["evidence"]
            if field.startswith("identity")
            else vector["evidence"]["outcome"]
        )
        target[field] = value
        _reseal(vector["evidence"])
        wrapped = canonical(outer_projection(vector["evidence"]))
        vector["canonical_wrapper_utf8"] = wrapped.decode()
        vector["canonical_wrapper_utf8_hex"] = wrapped.hex()
        vector["canonical_wrapper_bytes"] = len(wrapped)
        vector["expected_sha256"] = hashlib.sha256(wrapped).hexdigest()
        with pytest.raises(AssertionError):
            _validate_golden_vectors(golden)


def test_registry_census_compatibility_bounds_and_contract_only_scope() -> None:
    reg, census = load(FILES[4]), load(FILES[6])
    assert reg["implementation_status"] == "contract_only"
    assert tuple(reg["resources"]) == FILES
    assert reg["digest_domains"] == {
        "outer": DOMAIN,
        "audit": AUDIT_DOMAIN,
        "outer_wrapper": (
            "exact sorted canonical JSON {domain,schema,content}; omit only "
            "runtime_evidence_digest_sha256"
        ),
    }
    assert reg["refusal_codes"] == [
        "unsupported_outcome_schema_version",
        "outcome_evidence_generation_mismatch",
    ]
    assert reg["status_kinds"] == ["domain_terminal", "engine_halt", "integrity_error"]
    assert reg["provider_dispatch_states"] == ["not_applicable", "not_started", "started"]
    assert tuple(reg["dependencies"]) == EXPECTED_DEPENDENCIES
    assert tuple(reg["frozen_historical_resources"]) == EXPECTED_FROZEN
    assert reg["bounds"] == BOUNDS
    assert reg["compatibility_matrix"] == [
        {
            "outcome_version": 2,
            "evidence_version": 1,
            "result": "accept_historical_no_audit_claim",
        },
        {
            "outcome_version": 3,
            "evidence_version": 1,
            "result": "unsupported_outcome_schema_version",
        },
        {
            "outcome_version": 2,
            "evidence_version": 2,
            "result": "outcome_evidence_generation_mismatch",
        },
        {
            "outcome_version": 3,
            "evidence_version": 2,
            "result": "shape_only_no_authority",
        },
    ]
    audit_reg = load("domain-generated-audit-budget-registry-v1.json")
    assert len(audit_reg["refusal_mapping"]) == len(EXPECTED_REFUSAL_ROWS) == 14
    assert {
        row["refusal_class"]: tuple(row[field] for field in REFUSAL_FIELDS)
        for row in audit_reg["refusal_mapping"]
    } == EXPECTED_REFUSAL_ROWS
    assert reg["card_compatibility"] == {
        "card_v1": "no_audit_claim",
        "card_v2_missing_source_declaration": "preidentity_refusal",
        "artifact_8": "no_migration_no_rewrite",
    }
    assert reg["deferred"] == [
        "runtime",
        "adapters",
        "evaluator",
        "replay",
        "maintenance_source_declarations",
        "no_time",
        "artifact_9",
    ]
    assert census["scope"] == "new_successor_fields_only_not_artifact9"
    assert (
        census["entry_count"] == len(census["entries"]) == len(EXPECTED_CENSUS_ROWS) == 4
    )
    assert (
        tuple(
            (
                entry["field_path"],
                entry["owner"],
                entry["source"],
                entry["consumer"],
                entry["cardinality"],
                entry["nullability"],
            )
            for entry in census["entries"]
        )
        == EXPECTED_CENSUS_ROWS
    )
    changed = {p.as_posix() for p in ROOT.rglob("*") if p.is_file()}
    assert not any(
        "runtime-identity-adapter-registry-v2" in p or "artifact-9" in p for p in changed
    )


@pytest.mark.parametrize(
    "kind,terminal,replay,valid",
    [
        ({"kind": "domain_terminal"}, "done", True, True),
        ({"kind": "domain_terminal"}, None, True, False),
        ({"kind": "domain_terminal"}, "done", False, False),
        ({"kind": "engine_halt", "code": "operation_deadlock"}, None, False, True),
        ({"kind": "engine_halt", "code": "operation_deadlock"}, "done", True, True),
        (
            {"kind": "integrity_error", "contract_id": AUDIT_CONTRACT, "code": CODES[0]},
            None,
            False,
            True,
        ),
    ],
)
def test_status_schema_branches_are_closed(
    kind: dict[str, Any], terminal: str | None, replay: bool, valid: bool
) -> None:
    evidence = copy.deepcopy(load(FILES[7])["valid_vectors"][0]["evidence"])
    evidence["outcome"]["status"] = kind
    evidence["outcome"]["terminal_outcome"] = terminal
    evidence["outcome"]["replay_final"] = replay
    validator = Draft202012Validator(load("episode-outcome-v3.schema.json"))
    if valid:
        validator.validate(evidence["outcome"])
    else:
        with pytest.raises(ValidationError):
            validator.validate(evidence["outcome"])


def test_algorithmic_worst_case_proof_stays_below_owner_total_bound() -> None:
    # Per-record maxima are computed without materializing 100,000 rows. The longest
    # schema-valid committed row has 255-byte intent/source/transition IDs and fixed
    # maximum integers/digest. JSON punctuation/key costs are measured canonically.
    row = {
        "record_type": "committed_intent",
        "contract_id": AUDIT_CONTRACT,
        "intent_digest_sha256": "f" * 64,
        "intent_id": "a" * 255,
        "source_id": "a" * 255,
        "counted_kind": "scheduled_audit_event_intent",
        "transition_id": "a" * 255,
        "batch_ordinal": 99998,
        "operation_committed_count": 99998,
    }
    per_record = len(canonical(row))
    audit_envelope_overhead = len(
        canonical(
            {
                "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
                "schema_version": 1,
                "contract_id": AUDIT_CONTRACT,
                "operation_core_content_digest_sha256": "f" * 64,
                "operation_instance_id": "a" * 255,
                "declared_max_events": 2**63 - 1,
                "records": [],
            }
        )
    )
    schema_worst_case = (
        100_000 * per_record
        + 99_999
        + audit_envelope_overhead
        + BOUNDS["max_non_audit_payload_utf8_bytes"]
    )
    assert per_record == 1102
    assert audit_envelope_overhead == 591
    assert schema_worst_case == 118_689_198
    assert schema_worst_case < BOUNDS["max_total_payload_utf8_bytes"]
    audit_envelope_nodes = 16
    simultaneous_node_maxima = (
        100_000 * 20 + audit_envelope_nodes + BOUNDS["max_non_audit_nodes"]
    )
    assert simultaneous_node_maxima == 2_100_016
    assert simultaneous_node_maxima > BOUNDS["max_total_nodes"]
    assert simultaneous_node_maxima - BOUNDS["max_total_nodes"] == 16


def _reseal(evidence: dict[str, Any]) -> None:
    audit = evidence["domain_generated_audit_budget_evidence"]
    evidence["domain_generated_audit_budget_evidence_digest_sha256"] = digest(
        audit_projection(audit)
    )
    evidence["runtime_evidence_digest_sha256"] = digest(outer_projection(evidence))


def _golden_evidence(name: str) -> dict[str, Any]:
    vectors = load("audit-integrity-successor-v1.golden.json")["valid_vectors"]
    return copy.deepcopy(next(v["evidence"] for v in vectors if v["name"] == name))


def _reseal_content(evidence: dict[str, Any]) -> None:
    """Refresh every content-derived section digest and the outer content seal."""
    outcome = evidence["outcome"]
    outcome["final_state_digest_sha256"] = digest(outcome["final_state"])
    outcome["trajectory_digest_sha256"] = digest(outcome["trajectory"])
    tape = evidence["decision_tape"]
    tape["decisions_digest_sha256"] = digest(tape["decisions"])
    _reseal(evidence)


def _event(event_id: str, at: str, sequence: int) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": "probe",
        "actor_id": "actor.probe",
        "at": at,
        "sequence": sequence,
        "payload": {},
        "triggers_agent": False,
        "caused_by": None,
        "disposition": "accepted",
        "verdict_code": "OK",
    }


def _append_later_provider_activity(evidence: dict[str, Any]) -> None:
    attempt = copy.deepcopy(evidence["execution_record"]["attempts"][0])
    attempt.update(invocation_index=2, request_digest_sha256="8" * 64)
    evidence["execution_record"]["attempts"].append(attempt)
    evidence["execution_record"]["transport_calls"] = 2
    decision = copy.deepcopy(evidence["decision_tape"]["decisions"][0])
    decision.update(invocation_index=2, observation_digest_sha256="9" * 64)
    evidence["decision_tape"]["decisions"].append(decision)
    evidence["provider_binding"].update(
        provider_calls=2, attempts=2, decision_call_index=[0, 1]
    )
    evidence["outcome"]["invocations"] = 2


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "domain-terminal-null-and-not-final",
            lambda e: (
                e["outcome"].__setitem__("terminal_outcome", None),
                e["outcome"].__setitem__("replay_final", False),
            ),
        ),
        (
            "started-model-plan-null",
            lambda e: [
                e["execution_record"].__setitem__(key, None)
                for key in ("model", "protocol_version", "max_output_tokens")
            ],
        ),
        (
            "model-to-recorded-laundering",
            lambda e: (
                e["execution_record"].update(
                    outcome_source="recorded",
                    model=None,
                    protocol_version=None,
                    max_output_tokens=None,
                    provider_dispatch_state="not_applicable",
                    transport_calls=0,
                    attempts=[],
                ),
                e.__setitem__("provider_binding", None),
            ),
        ),
        (
            "provider-model-identity-drift",
            lambda e: e["provider_binding"].__setitem__("model", "other-model"),
        ),
        (
            "attempt-decision-index-drift",
            lambda e: e["execution_record"]["attempts"][0].__setitem__("turn_index", 9),
        ),
        (
            "failed-attempt-with-decision",
            lambda e: e["execution_record"]["attempts"][0].update(
                outcome="failed", fault="provider_transport"
            ),
        ),
        (
            "invocation-out-of-outcome-range",
            lambda e: e["outcome"].__setitem__("invocations", 0),
        ),
        (
            "excluded-without-code",
            lambda e: e["execution_record"].__setitem__("excluded", True),
        ),
    ],
)
def test_independent_oracle_rejects_coherently_resealed_v1_semantic_attacks(
    name: str, mutate: Any
) -> None:
    evidence = _golden_evidence("model_started_success")
    mutate(evidence)
    _reseal(evidence)
    with pytest.raises((AssertionError, ValidationError)):
        validate_semantics(evidence)


def _set_finding_field(field: str, value: Any) -> Any:
    def mutate(evidence: dict[str, Any]) -> None:
        records = evidence["domain_generated_audit_budget_evidence"]["records"]
        records[-2][field] = value

    return mutate


@pytest.mark.parametrize(
    "name,base,mutate,refresh_sections",
    [
        (
            "stale-final-state-digest",
            "zero_deterministic_success",
            lambda e: e["outcome"]["final_state"].update(changed=True),
            False,
        ),
        (
            "stale-trajectory-digest",
            "zero_deterministic_success",
            lambda e: e["outcome"]["trajectory"].append({"changed": True}),
            False,
        ),
        (
            "invalid-start-calendar-instant",
            "zero_deterministic_success",
            lambda e: e["outcome"].__setitem__("started_at", "2031-02-31T09:00:00Z"),
            False,
        ),
        (
            "ended-before-started",
            "zero_deterministic_success",
            lambda e: e["outcome"].update(
                started_at="2031-03-03T10:00:00Z", ended_at="2031-03-03T09:00:00Z"
            ),
            False,
        ),
        (
            "elapsed-minute-mismatch",
            "zero_deterministic_success",
            lambda e: e["outcome"].__setitem__("ended_at", "2031-03-03T09:01:00Z"),
            False,
        ),
        (
            "invalid-event-calendar-instant",
            "zero_deterministic_success",
            lambda e: e["outcome"]["events"].append(
                _event("event.bad", "2031-02-31T09:00:00Z", 0)
            ),
            False,
        ),
        (
            "event-before-outcome-range",
            "zero_deterministic_success",
            lambda e: e["outcome"]["events"].append(
                _event("event.early", "2031-03-03T08:59:00Z", 0)
            ),
            False,
        ),
        (
            "event-order-descends",
            "zero_deterministic_success",
            lambda e: e["outcome"].update(
                ended_at="2031-03-03T09:01:00Z",
                simulated_minutes=1,
                events=[
                    _event("event.later", "2031-03-03T09:01:00Z", 1),
                    _event("event.earlier", "2031-03-03T09:00:00Z", 0),
                ],
            ),
            False,
        ),
        (
            "source-committed-before-mismatch",
            "midexecution_model_started_error",
            _set_finding_field("source_committed_before", 0),
            False,
        ),
        (
            "finding-invocation-beyond-outcome",
            "midexecution_model_started_error",
            _set_finding_field("invocation_index", 99),
            False,
        ),
        (
            "attempt-and-decision-after-finding-invocation",
            "midexecution_model_started_error",
            _append_later_provider_activity,
            True,
        ),
    ],
)
def test_content_oracle_rejects_derivable_outcome_and_audit_coherence_attacks(
    name: str, base: str, mutate: Any, refresh_sections: bool
) -> None:
    evidence = _golden_evidence(base)
    mutate(evidence)
    (_reseal_content if refresh_sections else _reseal)(evidence)
    with pytest.raises((AssertionError, ValidationError, ValueError)):
        validate_semantics(evidence)


@pytest.mark.parametrize(
    "label,mutate",
    [
        (
            "terminal-identity",
            lambda e: e["outcome"].__setitem__("terminal_outcome", "different_terminal"),
        ),
        ("final-state", lambda e: e["outcome"]["final_state"].update(legal=True)),
        ("trajectory", lambda e: e["outcome"]["trajectory"].append({"legal": True})),
        (
            "provider-model-identity",
            lambda e: (
                e["execution_record"].__setitem__("model", "other-synthetic-model"),
                e["provider_binding"].__setitem__("model", "other-synthetic-model"),
            ),
        ),
        (
            "provider-sidecar-only-wire-candidate-fields",
            lambda e: e["provider_binding"].update(
                execution_run_id="exec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                execution_ledger_digest_sha256="a" * 64,
                settings_digest_sha256="b" * 64,
                pricing_digest_sha256="c" * 64,
            ),
        ),
    ],
)
def test_generic_content_accepts_legal_coherent_repins_but_exact_golden_rejects_them(
    label: str, mutate: Any
) -> None:
    golden = load("audit-integrity-successor-v1.golden.json")
    vector = next(
        v for v in golden["valid_vectors"] if v["name"] == "model_started_success"
    )
    mutate(vector["evidence"])
    _reseal_content(vector["evidence"])
    validate_semantics(vector["evidence"])
    wrapped = canonical(outer_projection(vector["evidence"]))
    vector.update(
        canonical_wrapper_utf8=wrapped.decode(),
        canonical_wrapper_utf8_hex=wrapped.hex(),
        canonical_wrapper_bytes=len(wrapped),
        expected_sha256=hashlib.sha256(wrapped).hexdigest(),
    )
    with pytest.raises(AssertionError):
        _validate_golden_vectors(golden)


@pytest.mark.parametrize(
    "name,refusal_class,code",
    [
        (
            "transition-class-under-not-started",
            "unknown_source_or_kind",
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        ),
        (
            "aggregate-overrun-code-laundering",
            "valid_reservation_above_declared_aggregate",
            "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        ),
    ],
)
def test_refusal_matrix_rejects_coherent_stage_and_code_laundering(
    name: str, refusal_class: str, code: str
) -> None:
    base = (
        "assigned_preexecution_model_not_started_error"
        if name.startswith("transition")
        else "midexecution_model_started_error"
    )
    evidence = _golden_evidence(base)
    finding = evidence["domain_generated_audit_budget_evidence"]["records"][-2]
    finding["refusal_class"] = refusal_class
    finding["integrity_code"] = code
    evidence["domain_generated_audit_budget_evidence"]["records"][-1][
        "integrity_code"
    ] = code
    evidence["outcome"]["status"]["code"] = code
    _reseal(evidence)
    with pytest.raises((AssertionError, ValidationError)):
        validate_semantics(evidence)


def test_generic_content_oracle_rejects_unauthenticated_source_overrun_label() -> None:
    evidence = _golden_evidence("midexecution_model_started_error")
    finding = evidence["domain_generated_audit_budget_evidence"]["records"][-2]
    finding["refusal_class"] = "valid_reservation_above_declared_source"
    _reseal(evidence)

    with pytest.raises(AssertionError):
        validate_semantics(evidence)


def test_raw_reference_parser_rejects_duplicate_keys_and_escaped_lone_surrogates() -> (
    None
):
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        strict_json_loads(b'{"schema_version":2,"schema_version":1}')
    with pytest.raises(ValueError, match="surrogate"):
        strict_json_loads(b'{"value":"\\ud800"}')


def _bounds(**overrides: int) -> Mapping[str, int]:
    return MappingProxyType({**EXPECTED_BOUNDS, **overrides})


def _nested_arrays(container_depth: int) -> list[Any]:
    root: list[Any] = []
    cursor = root
    for _ in range(container_depth - 1):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    return root


def _audit_envelope(records: list[Any]) -> dict[str, Any]:
    return {
        "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
        "schema_version": 1,
        "contract_id": AUDIT_CONTRACT,
        "operation_core_content_digest_sha256": "f" * 64,
        "operation_instance_id": "opinst_" + "a" * 32,
        "declared_max_events": 0,
        "records": records,
    }


def test_qodo_regression_bounded_profile_runs_before_jsonschema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _golden_evidence("zero_deterministic_success")
    evidence["decision_tape"]["decisions"] = _nested_arrays(65)

    def schema_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("jsonschema ran before bounded validation")

    monkeypatch.setattr(Draft202012Validator, "validate", schema_must_not_run)
    with pytest.raises(ValueError, match="depth"):
        validate_semantics(evidence)


def test_bounded_profile_depth_convention_accepts_root_at_64_and_rejects_65() -> None:
    validate_bounded_profile(_nested_arrays(64))
    with pytest.raises(ValueError, match="depth"):
        validate_bounded_profile(_nested_arrays(65))


@pytest.mark.parametrize(
    "container", [{str(index): None for index in range(4096)}, [None] * 4096]
)
def test_bounded_profile_accepts_ordinary_container_at_member_limit(
    container: Any,
) -> None:
    validate_bounded_profile(container)


@pytest.mark.parametrize(
    "container", [{str(index): None for index in range(4097)}, [None] * 4097]
)
def test_bounded_profile_rejects_ordinary_container_above_member_limit(
    container: Any,
) -> None:
    with pytest.raises(ValueError, match="container members"):
        validate_bounded_profile(container)


def test_only_exact_audit_records_path_receives_cardinality_exception() -> None:
    records = [None] * 4097
    validate_bounded_profile(
        {"domain_generated_audit_budget_evidence": _audit_envelope(records)}
    )
    with pytest.raises(ValueError, match="container members"):
        validate_bounded_profile({"records": records})
    with pytest.raises(ValueError, match="audit envelope"):
        validate_bounded_profile(
            {
                "domain_generated_audit_budget_evidence": {
                    "records": records,
                    "laundered": [],
                }
            }
        )


def test_audit_records_cardinality_boundary_and_plus_one() -> None:
    validate_bounded_profile(
        {"domain_generated_audit_budget_evidence": _audit_envelope([None] * 100_000)}
    )
    with pytest.raises(ValueError, match="container members"):
        validate_bounded_profile(
            {"domain_generated_audit_budget_evidence": _audit_envelope([None] * 100_001)}
        )


def test_string_utf8_boundary_and_plus_one() -> None:
    validate_bounded_profile("x" * 1_048_576)
    with pytest.raises(ValueError, match="string UTF-8 bytes"):
        validate_bounded_profile("x" * 1_048_577)


def test_total_node_boundary_counts_value_key_and_array_slot_nodes() -> None:
    # [0, 0] = array value + two array slots + two scalar values = five nodes.
    validate_bounded_profile([0, 0], bounds=_bounds(max_total_nodes=5))
    with pytest.raises(ValueError, match="total nodes"):
        validate_bounded_profile([0, 0], bounds=_bounds(max_total_nodes=4))


def test_non_audit_node_boundary_omits_only_the_exact_audit_member() -> None:
    value = {"x": [0], "domain_generated_audit_budget_evidence": _audit_envelope([])}
    # Projection {"x":[0]} = root + key + array + slot + scalar = five nodes.
    validate_bounded_profile(value, bounds=_bounds(max_non_audit_nodes=5))
    with pytest.raises(ValueError, match="non-audit nodes"):
        validate_bounded_profile(value, bounds=_bounds(max_non_audit_nodes=4))


def test_raw_and_canonical_total_byte_boundaries_do_not_require_huge_allocations() -> (
    None
):
    assert strict_json_loads(b"[]", bounds=_bounds(max_total_payload_utf8_bytes=2)) == []
    assert strict_json_loads("[]", bounds=_bounds(max_total_payload_utf8_bytes=2)) == []
    with pytest.raises(ValueError, match="raw JSON UTF-8 bytes"):
        strict_json_loads(b"[]", bounds=_bounds(max_total_payload_utf8_bytes=1))
    with pytest.raises(ValueError, match="raw JSON UTF-8 bytes"):
        strict_json_loads(b"\xff\xff", bounds=_bounds(max_total_payload_utf8_bytes=1))
    validate_bounded_profile("é", bounds=_bounds(max_total_payload_utf8_bytes=4))
    with pytest.raises(ValueError, match="canonical JSON UTF-8 bytes"):
        validate_bounded_profile("é", bounds=_bounds(max_total_payload_utf8_bytes=3))


def test_raw_parser_rejects_depth_before_stdlib_recursion() -> None:
    raw = b"[" * 65 + b"]" * 65
    with pytest.raises(ValueError, match=r"depth.*before parse"):
        strict_json_loads(raw)


def test_non_audit_canonical_byte_boundary_and_plus_one() -> None:
    value = {"x": "é", "domain_generated_audit_budget_evidence": _audit_envelope([])}
    projected_bytes = len(canonical({"x": "é"}))
    validate_bounded_profile(
        value, bounds=_bounds(max_non_audit_payload_utf8_bytes=projected_bytes)
    )
    with pytest.raises(ValueError, match="non-audit canonical JSON UTF-8 bytes"):
        validate_bounded_profile(
            value, bounds=_bounds(max_non_audit_payload_utf8_bytes=projected_bytes - 1)
        )


def test_aggregate_decision_evidence_above_8_mib_rejects_before_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _golden_evidence("zero_deterministic_success")
    evidence["decision_tape"]["decisions"] = ["x" * 1_048_000 for _ in range(9)]

    def schema_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("jsonschema ran before bounded validation")

    monkeypatch.setattr(Draft202012Validator, "validate", schema_must_not_run)
    with pytest.raises(ValueError, match="non-audit canonical JSON UTF-8 bytes"):
        validate_semantics(evidence)


def test_expected_bounds_are_independent_literals_matching_registry_and_golden() -> None:
    expected = {
        "max_depth": 64,
        "max_non_audit_nodes": 100_000,
        "max_non_audit_payload_utf8_bytes": 8_388_608,
        "max_total_nodes": 2_100_000,
        "max_total_payload_utf8_bytes": 268_435_456,
        "max_container_members": 4096,
        "audit_records_max_items": 100_000,
        "max_string_utf8_bytes": 1_048_576,
        "safe_integer_min": -9_007_199_254_740_991,
        "safe_integer_max": 9_007_199_254_740_991,
        "audit_declared_max_integer": 9_223_372_036_854_775_807,
    }
    assert dict(EXPECTED_BOUNDS) == expected
    assert load("audit-integrity-successor-registry-v1.json")["bounds"] == expected
    assert load("audit-integrity-successor-v1.golden.json")["bounds"] == expected


def test_safe_integer_schema_does_not_treat_boolean_as_integer() -> None:
    evidence = _golden_evidence("zero_deterministic_success")
    evidence["domain_generated_audit_budget_evidence"]["declared_max_events"] = True
    validate_bounded_profile(evidence)
    with pytest.raises(ValidationError):
        Draft202012Validator(
            load("executed-runtime-evidence-v2.schema.json"), registry=registry()
        ).validate(evidence)
