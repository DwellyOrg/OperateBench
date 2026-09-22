# SPDX-License-Identifier: Apache-2.0
"""Executable, implementation-independent B3.3 runtime-evidence contract checks.

Nothing here reads an expectation out of the artefact it is auditing. The bounds,
the two golden byte counts and hashes, the load-bearing semantic content of both
vectors, the adversarial census, and the clean-tracked-source-tree refusal
vocabulary are all stated as literals in this file; the resources must agree with
them rather than the other way round.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import cache, lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "schemas"
PACKAGED = ROOT / "src" / "operatebench" / "resources" / "identity"
SCHEMAS = (
    "episode-outcome-v2.schema.json",
    "decision-tape-v1.schema.json",
    "execution-record-v1.schema.json",
    "provider-execution-binding-v1.schema.json",
    "executed-runtime-evidence-v1.schema.json",
    "build-provenance-descriptor-v1.schema.json",
)
GOLDEN = "executed-runtime-evidence-v1.golden.json"
TREE_GOLDEN = "clean-tracked-source-tree-v1.golden.json"
TREE_SCHEMA = "clean-tracked-source-tree-v1.schema.json"

# RFC 4.1 bounds, authored here rather than read from the golden resource.
MAX_DEPTH = 64
MAX_NODES = 100_000
MAX_CONTAINER_MEMBERS = 4096
MAX_STRING_UTF8_BYTES = 1_048_576
MAX_PAYLOAD_UTF8_BYTES = 8_388_608
SAFE_INTEGER = 2**53 - 1
DIGEST_DOMAIN = "operatebench.digest.executed_runtime_evidence.v1"
EVIDENCE_SCHEMA = "operatebench.executed_runtime_evidence.v1"
SELF_DIGEST_FIELD = "runtime_evidence_digest_sha256"

# RFC 4.4 pins. A coherently repinned golden file must fail against these.
PROVIDER_NULL_BYTES = 1379
PROVIDER_NULL_SHA256 = "309b01dd7e08c61e1314b972467c922403f38ef097555ab7f762c125a06c2704"
PROVIDER_NON_NULL_BYTES = 2415
PROVIDER_NON_NULL_SHA256 = (
    "970ec85c12c5d9bbf562198a3ab016d66abcec2d4abba628e7a6e161419d393e"
)
# The semantic content each pin stands for, so that a rewritten vector cannot keep
# the name while changing what it demonstrates.
LOAD_BEARING = {
    "provider_null": {
        "operation_instance_id": "opinst_00000000000000000000000000000000",
        "identity_manifest_digest_sha256": (
            "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc"
        ),
        "outcome_status": "completed_successfully",
        "outcome_terminal": "completed_successfully",
        "replay_final": True,
        "started_at": "2031-03-03T09:00:00Z",
        "ended_at": "2031-03-03T09:00:00Z",
        "simulated_minutes": 0,
        "invocations": 0,
        "outcome_source": "deterministic",
        "transport_calls": 0,
        "decisions": 0,
        "attempts": 0,
        "provider_binding_is_null": True,
    },
    "provider_non_null": {
        "operation_instance_id": "opinst_99999999999999999999999999999999",
        "identity_manifest_digest_sha256": (
            "251e36a96f2c4cf474d06cef7d3ff3d39aab75211b4b9d191f1c4c4647b9a1bc"
        ),
        "outcome_status": "completed_successfully",
        "outcome_terminal": "completed_successfully",
        "replay_final": True,
        "started_at": "2031-03-03T09:00:00Z",
        "ended_at": "2031-03-03T09:00:00Z",
        "simulated_minutes": 0,
        "invocations": 1,
        "outcome_source": "model",
        "transport_calls": 1,
        "decisions": 1,
        "attempts": 1,
        "provider_binding_is_null": False,
    },
}

NULL_VECTOR_NAME = "provider_null"
MODEL_VECTOR_NAME = "provider_non_null"
ENGINE_HALTS = ("operation_deadlock", "operational_horizon_exhausted")
INSTANT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SCHEMA_REFUSAL = "<schema>"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _document(name: str) -> Any:
    return _load(DOCS / name)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


@lru_cache(maxsize=1)
def _registry() -> Registry:
    registry = Registry()
    for name in SCHEMAS:
        document = _document(name)
        registry = registry.with_resource(
            document["$id"], Resource.from_contents(document)
        )
    return registry


@cache
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(_document(name), registry=_registry())


def _validate_schema(name: str, value: Any) -> None:
    _validator(name).validate(value)


def _strict_json(value: Any, *, depth: int = 0, nodes: list[int] | None = None) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_NODES or depth > MAX_DEPTH:
        raise ValueError("bounded JSON profile")
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if not -SAFE_INTEGER <= value <= SAFE_INTEGER:
            raise ValueError("safe integer")
        return
    if type(value) is str:
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("surrogate code point")
        if len(value.encode("utf-8", "surrogatepass")) > MAX_STRING_UTF8_BYTES:
            raise ValueError("string utf8 bytes")
        return
    if type(value) is list:
        if len(value) > MAX_CONTAINER_MEMBERS:
            raise ValueError("container members")
        for item in value:
            _strict_json(item, depth=depth + 1, nodes=nodes)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("string-keyed object")
        if len(value) > MAX_CONTAINER_MEMBERS:
            raise ValueError("container members")
        for key, item in value.items():
            _strict_json(key, depth=depth + 1, nodes=nodes)
            _strict_json(item, depth=depth + 1, nodes=nodes)
        return
    raise ValueError("strict JSON type")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _projection(evidence: dict[str, Any]) -> dict[str, Any]:
    content = dict(evidence)
    content.pop(SELF_DIGEST_FIELD, None)
    return {"domain": DIGEST_DOMAIN, "schema": EVIDENCE_SCHEMA, "content": content}


def _refresh_runtime_digest(evidence: dict[str, Any]) -> dict[str, Any]:
    evidence[SELF_DIGEST_FIELD] = _digest(_projection(evidence))
    return evidence


def _reseal(evidence: dict[str, Any]) -> dict[str, Any]:
    """Coherently refresh every section digest and then the self digest."""
    outcome = evidence["outcome"]
    outcome["final_state_digest_sha256"] = _digest(outcome["final_state"])
    outcome["trajectory_digest_sha256"] = _digest(outcome["trajectory"])
    tape = evidence["decision_tape"]
    tape["decisions_digest_sha256"] = _digest(tape["decisions"])
    return _refresh_runtime_digest(evidence)


def _instant(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.strptime(value, INSTANT_FORMAT)
    except ValueError as error:
        raise ValueError(f"calendar instant: {label}") from error
    return parsed.replace(tzinfo=UTC)


def _validate_outcome(outcome: dict[str, Any]) -> None:
    terminal = outcome["terminal_outcome"]
    status = outcome["status"]
    if terminal is None:
        if status not in ENGINE_HALTS:
            raise ValueError("terminal status coherence")
    elif status not in ENGINE_HALTS:
        if status != terminal:
            raise ValueError("terminal identity")
        if outcome["replay_final"] is not True:
            raise ValueError("replay final")
    started = _instant(outcome["started_at"], label="started_at")
    ended = _instant(outcome["ended_at"], label="ended_at")
    for index, event in enumerate(outcome["events"]):
        _instant(event["at"], label=f"events/{index}/at")
    if ended < started:
        raise ValueError("instant ordering")
    elapsed = int((ended - started).total_seconds())
    if elapsed % 60 or elapsed // 60 != outcome["simulated_minutes"]:
        raise ValueError("elapsed minutes")
    if _digest(outcome["final_state"]) != outcome["final_state_digest_sha256"]:
        raise ValueError("final state digest")
    if _digest(outcome["trajectory"]) != outcome["trajectory_digest_sha256"]:
        raise ValueError("trajectory digest")


def _validate_tape(tape: dict[str, Any]) -> None:
    keys = [
        (decision["invocation_index"], decision["turn_index"])
        for decision in tape["decisions"]
    ]
    if any(later <= earlier for earlier, later in pairwise(keys)):
        raise ValueError("decision ordering")
    if _digest(tape["decisions"]) != tape["decisions_digest_sha256"]:
        raise ValueError("decision digest")


def _validate_execution(execution: dict[str, Any]) -> None:
    source = execution["outcome_source"]
    nullable = ("model", "protocol_version", "max_output_tokens")
    if source == "model":
        if any(execution[field] is None for field in nullable):
            raise ValueError("model nullability")
    else:
        if any(execution[field] is not None for field in nullable):
            raise ValueError("non-model nullability")
        if execution["transport_calls"] != 0:
            raise ValueError("non-model transport calls")
        if execution["attempts"]:
            raise ValueError("non-model attempts")
    if execution["excluded"]:
        raise ValueError("excluded execution")
    if execution["exclusion_code"] is not None or execution["exclusion_detail"] != "":
        raise ValueError("exclusion coherence")
    for attempt in execution["attempts"]:
        if attempt["outcome"] == "failed":
            raise ValueError("failed attempt")
        if attempt["fault"] is not None:
            raise ValueError("fault coherence")


def _validate_cross_record(evidence: dict[str, Any]) -> None:
    outcome = evidence["outcome"]
    decisions = evidence["decision_tape"]["decisions"]
    execution = evidence["execution_record"]
    attempts = execution["attempts"]
    binding = evidence["provider_binding"]
    recorded = [decision["invocation_index"] for decision in decisions]
    recorded += [attempt["invocation_index"] for attempt in attempts]
    if recorded and outcome["invocations"] < max(recorded):
        raise ValueError("invocation coverage")
    if execution["outcome_source"] != "model":
        if binding is not None:
            raise ValueError("non-model binding forbidden")
        return
    if binding is None:
        raise ValueError("model binding required")
    if binding["model"] != execution["model"]:
        raise ValueError("model mismatch")
    if execution["transport_calls"] != len(attempts) or len(attempts) != len(decisions):
        raise ValueError("model cardinality")
    if binding["attempts"] != len(attempts) or binding["provider_calls"] != len(attempts):
        raise ValueError("provider cardinality")
    if binding["decision_call_index"] != list(range(len(decisions))):
        raise ValueError("call index")
    for attempt, decision in zip(attempts, decisions, strict=True):
        if (attempt["invocation_index"], attempt["turn_index"]) != (
            decision["invocation_index"],
            decision["turn_index"],
        ):
            raise ValueError("attempt binding")
        malformed = decision["outcome"]["kind"] == "MALFORMED"
        expected = "classified" if malformed else "decided"
        if attempt["outcome"] != expected:
            raise ValueError("attempt kind pairing")


def _validate_semantics(evidence: dict[str, Any]) -> None:
    _strict_json(evidence)
    if len(_canonical(evidence)) > MAX_PAYLOAD_UTF8_BYTES:
        raise ValueError("payload bytes")
    _validate_schema("executed-runtime-evidence-v1.schema.json", evidence)
    _validate_outcome(evidence["outcome"])
    _validate_tape(evidence["decision_tape"])
    _validate_execution(evidence["execution_record"])
    _validate_cross_record(evidence)
    if _digest(_projection(evidence)) != evidence[SELF_DIGEST_FIELD]:
        raise ValueError("runtime evidence digest")


def _vectors() -> list[dict[str, Any]]:
    return _document(GOLDEN)["valid_vectors"]


def _vector(name: str) -> dict[str, Any]:
    return next(vector for vector in _vectors() if vector["name"] == name)


def _evidence(name: str) -> dict[str, Any]:
    return copy.deepcopy(_vector(name)["evidence"])


def _pointers(value: Any, prefix: str = "") -> Iterator[str]:
    if prefix:
        yield prefix
    if type(value) is dict:
        for key, item in value.items():
            yield from _pointers(item, f"{prefix}/{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _pointers(item, f"{prefix}/{index}")


def _resolve(value: Any, pointer: str) -> tuple[Any, str | int]:
    parts = pointer.split("/")[1:]
    parent: Any = value
    for part in parts[:-1]:
        parent = parent[int(part)] if type(parent) is list else parent[part]
    tail = parts[-1]
    return parent, int(tail) if type(parent) is list else tail


def _delete(value: Any, pointer: str) -> None:
    parent, key = _resolve(value, pointer)
    del parent[key]  # type: ignore[arg-type]


def _substitute(value: Any, pointer: str) -> None:
    parent, key = _resolve(value, pointer)
    current = parent[key]
    if current is None:
        replacement: Any = 0
    elif type(current) is bool:
        replacement = not current
    elif type(current) is int:
        replacement = current + 1
    elif type(current) is str:
        replacement = current + "z"
    elif type(current) is list:
        replacement = [] if current else [None]
    else:
        replacement = {} if current else {"unowned": 1}
    parent[key] = replacement


# ---------------------------------------------------------------------------
# Schema surface, packaged parity and lexical portability
# ---------------------------------------------------------------------------


def _patterns(value: Any) -> Iterator[str]:
    if type(value) is dict:
        for key, item in value.items():
            if key == "pattern" and type(item) is str:
                yield item
            yield from _patterns(item)
    elif type(value) is list:
        for item in value:
            yield from _patterns(item)


def test_schemas_meta_validate_offline_and_packaged_copies_match() -> None:
    assert _registry()
    for name in (*SCHEMAS, TREE_SCHEMA):
        Draft202012Validator.check_schema(_document(name))
    for name in (*SCHEMAS, TREE_SCHEMA, GOLDEN, TREE_GOLDEN):
        assert (DOCS / name).read_bytes() == (PACKAGED / name).read_bytes()
    Draft202012Validator(_document(TREE_SCHEMA)).validate(_document(TREE_GOLDEN))


@pytest.mark.parametrize(
    ("property_name", "expected"),
    (
        ("accepted_modes", ("100644", "100755", "120000")),
        (
            "refusal_codes",
            (
                "dirty_tracked_content",
                "duplicate_path",
                "frame_length_overflow",
                "mode_observation_unavailable",
                "non_canonical_path",
                "out_of_order",
                "portable_path_collision",
                "staged_index_drift",
                "submodule_entry",
                "unreadable_bytes",
                "unsafe_path",
                "unsupported_mode",
            ),
        ),
    ),
)
def test_clean_tree_fixed_vocabularies_require_exact_complete_arrays(
    property_name: str, expected: tuple[str, ...]
) -> None:
    validator = Draft202012Validator(_document(TREE_SCHEMA)["properties"][property_name])
    validator.validate(list(expected))
    for prefix_length in range(len(expected)):
        with pytest.raises(ValidationError):
            validator.validate(list(expected[:prefix_length]))
    with pytest.raises(ValidationError):
        validator.validate([*expected, "unexpected"])
    with pytest.raises(ValidationError):
        validator.validate([expected[1], expected[0], *expected[2:]])
    with pytest.raises(ValidationError):
        validator.validate(["unexpected", *expected[1:]])
    for non_array in (True, "unexpected", {}, None):
        with pytest.raises(ValidationError):
            validator.validate(non_array)


def test_every_pattern_uses_an_absolute_ending_rather_than_a_dollar_anchor() -> None:
    seen = 0
    for name in (*SCHEMAS, TREE_SCHEMA):
        for pattern in _patterns(_document(name)):
            seen += 1
            assert pattern.startswith("^"), (name, pattern)
            assert not pattern.endswith("$"), (name, pattern)
            assert pattern.endswith(r"(?![\s\S])"), (name, pattern)
    assert seen >= 25


@pytest.mark.parametrize(
    "suffix", ("\n", "\r", "\r\n", "\u2028", "\u2029", " ", "\t", "\x00")
)
@pytest.mark.parametrize(
    "pointer",
    (
        "/identity_manifest_digest_sha256",
        "/operation_instance_id",
        "/runtime_evidence_digest_sha256",
        "/outcome/final_state_digest_sha256",
        "/outcome/started_at",
        "/decision_tape/decisions_digest_sha256",
    ),
)
def test_trailing_line_terminators_are_outside_every_exact_lexical_language(
    pointer: str, suffix: str
) -> None:
    evidence = _evidence("provider_null")
    parent, key = _resolve(evidence, pointer)
    parent[key] = parent[key] + suffix
    if pointer != "/runtime_evidence_digest_sha256":
        _refresh_runtime_digest(evidence)
    with pytest.raises((ValueError, ValidationError)):
        _validate_semantics(evidence)


@pytest.mark.parametrize("suffix", ("\n", "\r", "\r\n", "\u2028", "\u2029"))
def test_trailing_line_terminators_are_outside_the_terminal_identifier_language(
    suffix: str,
) -> None:
    evidence = _evidence("provider_null")
    evidence["outcome"]["status"] += suffix
    evidence["outcome"]["terminal_outcome"] += suffix
    _refresh_runtime_digest(evidence)
    with pytest.raises((ValueError, ValidationError)):
        _validate_semantics(evidence)


@pytest.mark.parametrize("suffix", ("\n", "\r", "\u2028"))
def test_build_descriptor_lexical_fields_reject_line_terminator_suffixes(
    suffix: str,
) -> None:
    schema = _document("build-provenance-descriptor-v1.schema.json")
    validator = Draft202012Validator(schema)
    descriptor = {
        "schema": "operatebench.build_provenance_descriptor.v1",
        "schema_version": 1,
        "source_commit": "a" * 40,
        "source_tree_algorithm": "operatebench.clean_tracked_source_tree.v1",
        "source_tree_digest_sha256": "b" * 64,
        "tree_clean": True,
        "uv_lock_sha256": "c" * 64,
        "build_mode": "checkout",
        "distribution_artifacts": None,
        "toolchain": {
            "python_implementation": "cpython",
            "python_version": "3.11.15",
            "build_backend": "hatchling",
            "build_backend_version": "1.32.0",
        },
    }
    validator.validate(descriptor)
    for field in (
        "source_commit",
        "source_tree_digest_sha256",
        "uv_lock_sha256",
    ):
        suffixed = dict(descriptor)
        suffixed[field] = str(descriptor[field]) + suffix
        with pytest.raises(ValidationError):
            validator.validate(suffixed)
    suffixed = dict(descriptor)
    suffixed["toolchain"] = dict(
        descriptor["toolchain"], python_version="3.11.15" + suffix
    )
    with pytest.raises(ValidationError):
        validator.validate(suffixed)


# ---------------------------------------------------------------------------
# Golden vectors: independent pins, not resource-derived labels
# ---------------------------------------------------------------------------


def test_provider_binding_text_keeps_the_stricter_ledger_length_limit() -> None:
    schema = _document("provider-execution-binding-v1.schema.json")
    text_fields = {"model", "measured_cost_usd", "forfeited_reservation_usd"}
    fixed_length = {
        "execution_run_id": 37,
        "execution_ledger_digest_sha256": 64,
        "settings_digest_sha256": 64,
        "pricing_digest_sha256": 64,
    }
    for field in text_fields:
        assert schema["properties"][field]["maxLength"] == 256, field
    validator = Draft202012Validator(schema)
    for field, width in fixed_length.items():
        assert width <= 256
        binding = copy.deepcopy(
            _vector(MODEL_VECTOR_NAME)["evidence"]["provider_binding"]
        )
        binding[field] = binding[field] + "0"
        with pytest.raises(ValidationError):
            validator.validate(binding)
    for field in text_fields:
        binding = copy.deepcopy(
            _vector(MODEL_VECTOR_NAME)["evidence"]["provider_binding"]
        )
        binding[field] = "1" * 257
        with pytest.raises(ValidationError):
            validator.validate(binding)


def test_golden_vectors_match_independently_stated_byte_counts_and_hashes() -> None:
    assert [vector["name"] for vector in _vectors()] == [
        "provider_null",
        "provider_non_null",
    ]
    pins = {
        "provider_null": (PROVIDER_NULL_BYTES, PROVIDER_NULL_SHA256),
        "provider_non_null": (PROVIDER_NON_NULL_BYTES, PROVIDER_NON_NULL_SHA256),
    }
    for name, (expected_bytes, expected_sha256) in pins.items():
        vector = _vector(name)
        encoded = _canonical(_projection(vector["evidence"]))
        assert len(encoded) == expected_bytes
        assert hashlib.sha256(encoded).hexdigest() == expected_sha256
        assert vector["evidence"][SELF_DIGEST_FIELD] == expected_sha256
        # The resource's own stored representations must agree with the pins.
        assert vector["canonical_wrapper_bytes"] == expected_bytes
        assert vector["expected_sha256"] == expected_sha256
        assert encoded.decode() == vector["canonical_wrapper_utf8"]
        assert encoded.hex() == vector["canonical_wrapper_utf8_hex"]
        _validate_semantics(copy.deepcopy(vector["evidence"]))


def test_golden_vectors_carry_the_exact_load_bearing_semantics_the_pins_stand_for() -> (
    None
):
    for name, expected in LOAD_BEARING.items():
        evidence = _evidence(name)
        outcome = evidence["outcome"]
        execution = evidence["execution_record"]
        actual = {
            "operation_instance_id": evidence["operation_instance_id"],
            "identity_manifest_digest_sha256": evidence[
                "identity_manifest_digest_sha256"
            ],
            "outcome_status": outcome["status"],
            "outcome_terminal": outcome["terminal_outcome"],
            "replay_final": outcome["replay_final"],
            "started_at": outcome["started_at"],
            "ended_at": outcome["ended_at"],
            "simulated_minutes": outcome["simulated_minutes"],
            "invocations": outcome["invocations"],
            "outcome_source": execution["outcome_source"],
            "transport_calls": execution["transport_calls"],
            "decisions": len(evidence["decision_tape"]["decisions"]),
            "attempts": len(execution["attempts"]),
            "provider_binding_is_null": evidence["provider_binding"] is None,
        }
        assert actual == expected


def test_a_coherent_golden_repin_is_refused_by_the_independent_pins() -> None:
    """A rewritten vector whose stored representations all agree still fails."""
    vector = copy.deepcopy(_vector("provider_null"))
    vector["evidence"]["operation_instance_id"] = "opinst_" + "1" * 32
    _refresh_runtime_digest(vector["evidence"])
    encoded = _canonical(_projection(vector["evidence"]))
    vector["canonical_wrapper_utf8"] = encoded.decode()
    vector["canonical_wrapper_utf8_hex"] = encoded.hex()
    vector["canonical_wrapper_bytes"] = len(encoded)
    vector["expected_sha256"] = hashlib.sha256(encoded).hexdigest()
    # Internally coherent, and still refused by both independent pins.
    _validate_semantics(copy.deepcopy(vector["evidence"]))
    assert vector["expected_sha256"] != PROVIDER_NULL_SHA256
    assert (
        vector["evidence"]["operation_instance_id"]
        != LOAD_BEARING["provider_null"]["operation_instance_id"]
    )


def test_bounds_resource_matches_the_independently_authored_constants() -> None:
    assert _document(GOLDEN)["bounds"] == {
        "max_depth": MAX_DEPTH,
        "max_nodes": MAX_NODES,
        "max_container_members": MAX_CONTAINER_MEMBERS,
        "max_string_utf8_bytes": MAX_STRING_UTF8_BYTES,
        "max_payload_utf8_bytes": MAX_PAYLOAD_UTF8_BYTES,
        "safe_integer_max": SAFE_INTEGER,
        "safe_integer_min": -SAFE_INTEGER,
    }
    assert _document(GOLDEN)["digest_domain"] == DIGEST_DOMAIN
    assert _document(GOLDEN)["independent_pins"] == [
        {
            "name": "provider_null",
            "canonical_wrapper_bytes": PROVIDER_NULL_BYTES,
            "expected_sha256": PROVIDER_NULL_SHA256,
        },
        {
            "name": "provider_non_null",
            "canonical_wrapper_bytes": PROVIDER_NON_NULL_BYTES,
            "expected_sha256": PROVIDER_NON_NULL_SHA256,
        },
    ]


# ---------------------------------------------------------------------------
# The generated adversarial census
# ---------------------------------------------------------------------------


class _Forged(dict):
    """A mapping that is not an exact dict; the closed profile must refuse it."""


def _duplicate_key_probe() -> None:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    json.loads(
        '{"schema_version":1,"schema_version":1}', object_pairs_hook=reject_duplicates
    )


def _nested(depth: int) -> Any:
    node: Any = {}
    for _ in range(depth):
        node = {"n": node}
    return node


def _mutate(name: str, apply: Callable[[dict[str, Any]], None], *, reseal: bool = True):
    def build() -> dict[str, Any]:
        evidence = _evidence(name)
        apply(evidence)
        if reseal:
            _reseal(evidence)
        return evidence

    return build


def _profile(name: str, apply: Callable[[dict[str, Any]], None]):
    """A closed-profile mutation: never resealed, since it is refused before digests."""
    return _mutate(name, apply, reseal=False)


NULL = "provider_null"
MODEL = "provider_non_null"


def _set(pointer: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(evidence: dict[str, Any]) -> None:
        parent, key = _resolve(evidence, pointer)
        parent[key] = value

    return apply


def _descending_tape(evidence: dict[str, Any]) -> None:
    decisions = evidence["decision_tape"]["decisions"]
    attempts = evidence["execution_record"]["attempts"]
    second = copy.deepcopy(decisions[0])
    second.update(invocation_index=1, turn_index=0)
    decisions[0].update(invocation_index=1, turn_index=1)
    decisions.append(second)
    extra = copy.deepcopy(attempts[0])
    extra.update(invocation_index=1, turn_index=0)
    attempts[0].update(invocation_index=1, turn_index=1)
    attempts.append(extra)
    evidence["execution_record"]["transport_calls"] = 2
    evidence["provider_binding"].update(
        attempts=2, provider_calls=2, decision_call_index=[0, 1]
    )


CENSUS: dict[str, tuple[Callable[[], dict[str, Any]] | None, str]] = {
    # profile
    "duplicate_key": (None, "duplicate key"),
    "unknown_field": (_profile(NULL, _set("/unowned", True)), SCHEMA_REFUSAL),
    "missing_field": (
        _profile(NULL, lambda e: e.pop("operation_instance_id")),
        SCHEMA_REFUSAL,
    ),
    "bool_for_integer": (
        _profile(NULL, _set("/outcome/invocations", True)),
        SCHEMA_REFUSAL,
    ),
    "float": (_profile(NULL, _set("/outcome/invocations", 1.0)), "strict JSON type"),
    "lone_surrogate": (
        _profile(NULL, _set("/execution_record/exclusion_detail", "\ud800")),
        "surrogate code point",
    ),
    "non_string_key": (
        _profile(NULL, _set("/outcome/final_state", {1: "one"})),
        "string-keyed object",
    ),
    "tuple_container": (
        _profile(NULL, _set("/outcome/trajectory", ())),
        "strict JSON type",
    ),
    "mapping_subclass_forgery": (
        _profile(
            MODEL,
            lambda e: e.__setitem__("provider_binding", _Forged(e["provider_binding"])),
        ),
        "strict JSON type",
    ),
    "unsafe_integer": (
        _profile(NULL, _set("/outcome/invocations", 2**53)),
        "safe integer",
    ),
    "over_depth": (
        _profile(NULL, _set("/outcome/final_state", _nested(MAX_DEPTH + 2))),
        "bounded JSON profile",
    ),
    "over_nodes": (
        _profile(
            NULL,
            _set(
                "/outcome/trajectory",
                [[None] * 30 for _ in range(MAX_CONTAINER_MEMBERS)],
            ),
        ),
        "bounded JSON profile",
    ),
    "over_container_members": (
        _profile(NULL, _set("/outcome/trajectory", [None] * (MAX_CONTAINER_MEMBERS + 1))),
        "container members",
    ),
    "over_string_bytes": (
        _profile(
            NULL,
            _set("/execution_record/exclusion_detail", "a" * (MAX_STRING_UTF8_BYTES + 1)),
        ),
        "string utf8 bytes",
    ),
    "over_payload_bytes": (
        _profile(
            NULL,
            _set(
                "/outcome/trajectory",
                ["a" * 2100 for _ in range(MAX_CONTAINER_MEMBERS)],
            ),
        ),
        "payload bytes",
    ),
    # lexical
    "newline_suffixed_digest": (
        _mutate(NULL, _set("/identity_manifest_digest_sha256", "a" * 64 + "\n")),
        SCHEMA_REFUSAL,
    ),
    "carriage_return_suffixed_identifier": (
        _mutate(NULL, _set("/operation_instance_id", "opinst_" + "0" * 32 + "\r")),
        SCHEMA_REFUSAL,
    ),
    "line_separator_suffixed_terminal": (
        _mutate(
            NULL,
            lambda e: e["outcome"].update(
                status="completed_successfully\u2028",
                terminal_outcome="completed_successfully\u2028",
            ),
        ),
        SCHEMA_REFUSAL,
    ),
    # outcome
    "terminal_status_mismatch": (
        _mutate(NULL, _set("/outcome/terminal_outcome", "cancelled")),
        "terminal identity",
    ),
    "business_terminal_not_replay_final": (
        _mutate(NULL, _set("/outcome/replay_final", False)),
        "replay final",
    ),
    "impossible_calendar_date": (
        _mutate(NULL, _set("/outcome/started_at", "2031-02-31T09:00:00Z")),
        "calendar instant",
    ),
    "ended_before_started": (
        _mutate(NULL, _set("/outcome/started_at", "2031-03-03T10:00:00Z")),
        "instant ordering",
    ),
    "elapsed_minutes_mismatch": (
        _mutate(NULL, _set("/outcome/ended_at", "2031-03-03T10:00:00Z")),
        "elapsed minutes",
    ),
    "event_instant_not_calendar_valid": (
        _mutate(
            NULL,
            lambda e: e["outcome"]["events"].append(
                {
                    "event_id": "e1",
                    "event_type": "maintenance.reported",
                    "actor_id": "a1",
                    "at": "2031-04-31T09:00:00Z",
                    "sequence": 0,
                    "payload": {},
                    "triggers_agent": False,
                    "caused_by": None,
                    "disposition": "accepted",
                    "verdict_code": "ok",
                }
            ),
        ),
        "calendar instant",
    ),
    # tape
    "non_increasing_decision_order": (
        _mutate(MODEL, _descending_tape),
        "decision ordering",
    ),
    "stale_section_digest": (
        _mutate(
            NULL,
            _set("/outcome/trajectory_digest_sha256", "d" * 64),
            reseal=False,
        ),
        "trajectory digest",
    ),
    "stale_runtime_digest": (
        _mutate(NULL, _set("/outcome/invocations", 7), reseal=False),
        "runtime evidence digest",
    ),
    "self_digest_only_mutation": (
        _mutate(NULL, _set("/runtime_evidence_digest_sha256", "e" * 64), reseal=False),
        "runtime evidence digest",
    ),
    # execution
    "deterministic_source_model_forgery": (
        _mutate(
            NULL,
            lambda e: e["execution_record"].update(
                model="forged", protocol_version="forged.v1", max_output_tokens=8
            ),
        ),
        "non-model nullability",
    ),
    "deterministic_source_transport_calls": (
        _mutate(NULL, _set("/execution_record/transport_calls", 1)),
        "non-model transport calls",
    ),
    "deterministic_source_attempt_present": (
        _mutate(
            NULL,
            _set(
                "/execution_record/attempts",
                [
                    {
                        "invocation_index": 1,
                        "turn_index": 0,
                        "request_digest_sha256": "3" * 64,
                        "outcome": "decided",
                        "fault": None,
                    }
                ],
            ),
        ),
        "non-model attempts",
    ),
    "model_source_null_protocol": (
        _mutate(MODEL, _set("/execution_record/protocol_version", None)),
        "model nullability",
    ),
    "model_source_null_token_ceiling": (
        _mutate(MODEL, _set("/execution_record/max_output_tokens", None)),
        "model nullability",
    ),
    "excluded_execution_record": (
        _mutate(
            NULL,
            lambda e: e["execution_record"].update(
                excluded=True, exclusion_code="operator_withdrawn"
            ),
        ),
        "excluded execution",
    ),
    "exclusion_code_without_exclusion": (
        _mutate(NULL, _set("/execution_record/exclusion_code", "operator_withdrawn")),
        "exclusion coherence",
    ),
    "fault_outcome_mismatch": (
        _mutate(MODEL, _set("/execution_record/attempts/0/fault", "provider_transport")),
        "fault coherence",
    ),
    "attempt_outcome_failed": (
        _mutate(
            MODEL,
            lambda e: e["execution_record"]["attempts"][0].update(
                outcome="failed", fault="provider_deadline"
            ),
        ),
        "failed attempt",
    ),
    "decision_kind_attempt_outcome_mismatch": (
        _mutate(MODEL, _set("/execution_record/attempts/0/outcome", "classified")),
        "attempt kind pairing",
    ),
    # evidence
    "invocation_coverage_shortfall": (
        _mutate(
            MODEL,
            lambda e: e["outcome"].update(invocations=0, ended_at="2031-03-03T09:00:00Z"),
        ),
        "invocation coverage",
    ),
    "provider_binding_source_mismatch": (
        _mutate(
            NULL,
            lambda e: e.__setitem__(
                "provider_binding",
                copy.deepcopy(_vector(MODEL)["evidence"]["provider_binding"]),
            ),
        ),
        "non-model binding forbidden",
    ),
    "provider_binding_missing_for_model": (
        _mutate(MODEL, _set("/provider_binding", None)),
        "model binding required",
    ),
    "provider_binding_model_mismatch": (
        _mutate(MODEL, _set("/provider_binding/model", "other-model")),
        "model mismatch",
    ),
    "provider_cardinality_mismatch": (
        _mutate(MODEL, _set("/provider_binding/provider_calls", 2)),
        "provider cardinality",
    ),
    "decision_call_index_not_contiguous": (
        _mutate(MODEL, _set("/provider_binding/decision_call_index", [1])),
        "call index",
    ),
    "attempt_decision_index_mismatch": (
        _mutate(MODEL, _set("/execution_record/attempts/0/turn_index", 5)),
        "attempt binding",
    ),
    "swapped_ledger_identity": (
        _mutate(
            MODEL,
            _set("/provider_binding/execution_run_id", "exec_" + "8" * 32),
            reseal=False,
        ),
        "runtime evidence digest",
    ),
    "swapped_pricing_or_settings_digest": (
        _mutate(
            MODEL,
            lambda e: e["provider_binding"].update(
                settings_digest_sha256="a" * 64, pricing_digest_sha256="b" * 64
            ),
            reseal=False,
        ),
        "runtime evidence digest",
    ),
    "swapped_ledger_totals": (
        _mutate(
            MODEL,
            lambda e: e["provider_binding"].update(
                input_tokens=97, output_tokens=31, measured_cost_usd="12.5"
            ),
            reseal=False,
        ),
        "runtime evidence digest",
    ),
    "audit_companion_field_on_wire": (
        _profile(
            MODEL,
            _set("/provider_binding/execution_ledger_path", "/tmp/ledger.ndjson"),
        ),
        SCHEMA_REFUSAL,
    ),
}


def test_census_codes_and_classes_agree_with_the_frozen_resource() -> None:
    resource = _document(GOLDEN)["malformed_vectors"]
    assert [item["code"] for item in resource] == list(CENSUS)
    assert {item["expected"] for item in resource} == {"reject"}
    assert _document(GOLDEN)["malformed_vector_classes"] == sorted(
        {item["class"] for item in resource}
    )
    assert len(resource) == len(CENSUS) == 49


@pytest.mark.parametrize("code", list(CENSUS))
def test_every_census_code_has_an_executable_constructor_that_is_refused(
    code: str,
) -> None:
    build, marker = CENSUS[code]
    if build is None:
        with pytest.raises(ValueError, match=marker):
            _duplicate_key_probe()
        return
    evidence = build()
    if marker is SCHEMA_REFUSAL:
        with pytest.raises(ValidationError):
            _validate_semantics(evidence)
        return
    with pytest.raises(ValueError, match=marker):
        _validate_semantics(evidence)


# ---------------------------------------------------------------------------
# The generated per-field mutation matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("provider_null", "provider_non_null"))
def test_deleting_any_wire_field_is_refused(name: str) -> None:
    base = _evidence(name)
    pointers = list(_pointers(base))
    assert pointers
    for pointer in pointers:
        evidence = _evidence(name)
        _delete(evidence, pointer)
        with pytest.raises((ValueError, ValidationError)):
            _validate_semantics(evidence)


@pytest.mark.parametrize("name", ("provider_null", "provider_non_null"))
def test_substituting_any_wire_field_is_refused_and_moves_the_canonical_bytes(
    name: str,
) -> None:
    base = _evidence(name)
    pointers = list(_pointers(base))
    original = _canonical(base)
    covered: set[str] = set()
    projection_stable: set[str] = set()
    for pointer in pointers:
        evidence = _evidence(name)
        _substitute(evidence, pointer)
        assert _canonical(evidence) != original, pointer
        if _canonical(_projection(evidence)) == _canonical(_projection(base)):
            projection_stable.add(pointer)
        with pytest.raises((ValueError, ValidationError)):
            _validate_semantics(evidence)
        covered.add(pointer)
    assert covered == set(pointers)
    # Exactly one wire field is omitted from the digest projection: the self digest.
    assert projection_stable == {f"/{SELF_DIGEST_FIELD}"}


@pytest.mark.parametrize("name", ("provider_null", "provider_non_null"))
def test_mutation_matrix_covers_every_record_and_every_declared_field(
    name: str,
) -> None:
    pointers = set(_pointers(_evidence(name)))
    evidence = _evidence(name)
    for field in _document("executed-runtime-evidence-v1.schema.json")["required"]:
        assert f"/{field}" in pointers
    for record, schema in (
        ("outcome", "episode-outcome-v2.schema.json"),
        ("decision_tape", "decision-tape-v1.schema.json"),
        ("execution_record", "execution-record-v1.schema.json"),
    ):
        for field in _document(schema)["required"]:
            assert f"/{record}/{field}" in pointers
    if evidence["provider_binding"] is not None:
        required = _document("provider-execution-binding-v1.schema.json")["required"]
        assert len(required) == 17
        for field in required:
            assert f"/provider_binding/{field}" in pointers


def test_validation_never_mutates_the_caller_graph_and_aliases_are_not_observable() -> (
    None
):
    evidence = _evidence(MODEL)
    before = _canonical(evidence)
    _validate_semantics(evidence)
    assert _canonical(evidence) == before

    aliased = _evidence(MODEL)
    shared = {"shared": [1, 2, 3]}
    aliased["outcome"]["final_state"] = shared
    aliased["outcome"]["trajectory"] = [shared, shared]
    _reseal(aliased)
    detached = copy.deepcopy(aliased)
    detached["outcome"]["final_state"] = copy.deepcopy(shared)
    detached["outcome"]["trajectory"] = [copy.deepcopy(shared), copy.deepcopy(shared)]
    _reseal(detached)
    _validate_semantics(aliased)
    _validate_semantics(detached)
    assert _canonical(aliased) == _canonical(detached)
    assert aliased[SELF_DIGEST_FIELD] == detached[SELF_DIGEST_FIELD]
    # A fresh wire projection shares no container identity with its source.
    projection = json.loads(_canonical(aliased).decode())
    assert projection["outcome"]["trajectory"][0] is not shared


# ---------------------------------------------------------------------------
# Positive state-machine coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "terminal"),
    (
        ("operation_deadlock", None),
        ("operation_deadlock", "completed_successfully"),
        ("operational_horizon_exhausted", None),
        ("operational_horizon_exhausted", "completed_successfully"),
    ),
)
def test_both_engine_halts_preserve_null_or_retained_domain_terminal(
    status: str, terminal: str | None
) -> None:
    evidence = _evidence(NULL)
    evidence["outcome"].update(
        status=status, terminal_outcome=terminal, replay_final=False
    )
    _reseal(evidence)
    _validate_semantics(evidence)


def test_whole_elapsed_minutes_and_a_malformed_decision_pairing_are_accepted() -> None:
    evidence = _evidence(MODEL)
    evidence["outcome"].update(ended_at="2031-03-03T09:07:00Z", simulated_minutes=7)
    evidence["decision_tape"]["decisions"][0]["outcome"] = {
        "kind": "MALFORMED",
        "code": "MODEL_NO_TOOL_CALL",
    }
    evidence["execution_record"]["attempts"][0]["outcome"] = "classified"
    _reseal(evidence)
    _validate_semantics(evidence)


# ---------------------------------------------------------------------------
# operatebench.clean_tracked_source_tree.v1
# ---------------------------------------------------------------------------

TREE_ALGORITHM = "operatebench.clean_tracked_source_tree.v1"
TREE_PREFIX = TREE_ALGORITHM.encode("ascii")
TREE_MODES = ("100644", "100755", "120000")
TREE_MAX_PATH_BYTES = 4096
TREE_U64_MAX = 2**64 - 1
TREE_REFUSALS = (
    "dirty_tracked_content",
    "duplicate_path",
    "frame_length_overflow",
    "mode_observation_unavailable",
    "non_canonical_path",
    "out_of_order",
    "portable_path_collision",
    "staged_index_drift",
    "submodule_entry",
    "unreadable_bytes",
    "unsafe_path",
    "unsupported_mode",
)


class TreeRefusal(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _admit(mode: str, path: bytes, content: bytes | None) -> None:
    if mode == "160000":
        raise TreeRefusal("submodule_entry")
    if mode not in TREE_MODES:
        raise TreeRefusal("unsupported_mode")
    if not path or len(path) > TREE_MAX_PATH_BYTES:
        raise TreeRefusal("unsafe_path")
    try:
        text = path.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TreeRefusal("unsafe_path") from error
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise TreeRefusal("unsafe_path")
    if unicodedata.normalize("NFC", text) != text:
        raise TreeRefusal("non_canonical_path")
    if any(byte < 0x20 or byte == 0x7F or byte == 0x5C for byte in path):
        raise TreeRefusal("unsafe_path")
    if text.startswith("/"):
        raise TreeRefusal("unsafe_path")
    components = text.split("/")
    if any(part in ("", ".", "..") for part in components):
        raise TreeRefusal("unsafe_path")
    if any(part.lower() == ".git" for part in components):
        raise TreeRefusal("unsafe_path")
    if content is None:
        raise TreeRefusal("unreadable_bytes")


def _portable_path_key(path: bytes) -> bytes:
    ascii_lower = bytes.maketrans(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz"
    )
    return path.translate(ascii_lower)


def _admit_entries(entries: list[tuple[str, bytes, bytes]]) -> None:
    previous: bytes | None = None
    portable_paths: dict[bytes, bytes] = {}
    for mode, path, content in entries:
        _admit(mode, path, content)
        if previous is not None:
            if path == previous:
                raise TreeRefusal("duplicate_path")
            if path < previous:
                raise TreeRefusal("out_of_order")
        portable = _portable_path_key(path)
        prior = portable_paths.get(portable)
        if prior is not None and prior != path:
            raise TreeRefusal("portable_path_collision")
        portable_paths[portable] = path
        previous = path


def _tree_u64(length: int) -> bytes:
    if not 0 <= length <= TREE_U64_MAX:
        raise TreeRefusal("frame_length_overflow")
    return length.to_bytes(8, "big")


def _tree_frame(entries: list[tuple[str, bytes, bytes]]) -> bytes:
    """Reference framing: build the whole byte string, then hash it once."""
    _admit_entries(entries)
    out = bytearray(TREE_PREFIX)
    out.append(0)
    for mode, path, content in entries:
        mode_bytes = mode.encode("ascii")
        out += _tree_u64(len(mode_bytes))
        out += mode_bytes
        out += _tree_u64(len(path))
        out += path
        out += _tree_u64(len(content))
        out += content
    return bytes(out)


def _tree_digest_streaming(entries: list[tuple[str, bytes, bytes]]) -> str:
    """Independent recomputation: never materialise the framed byte string."""
    _admit_entries(entries)
    digest = hashlib.sha256()
    digest.update(TREE_PREFIX)
    digest.update(b"\x00")
    for mode, path, content in entries:
        for field in (
            _tree_u64(len(mode.encode("ascii"))),
            mode.encode("ascii"),
            _tree_u64(len(path)),
            path,
            _tree_u64(len(content)),
            content,
        ):
            digest.update(field)
    return digest.hexdigest()


def _tree_entries(vector: dict[str, Any]) -> list[tuple[str, bytes, bytes]]:
    return [
        (
            entry["mode"],
            bytes.fromhex(entry["path_hex"]),
            bytes.fromhex(entry["content_hex"]),
        )
        for entry in vector["entries"]
    ]


def _tree_golden() -> dict[str, Any]:
    return _document(TREE_GOLDEN)


def test_tree_vector_resource_states_the_independently_authored_contract() -> None:
    golden = _tree_golden()
    assert golden["algorithm"] == TREE_ALGORITHM
    assert golden["prefix_utf8"] == TREE_ALGORITHM
    assert golden["prefix_utf8_hex"] == TREE_PREFIX.hex()
    assert golden["separator_byte_hex"] == "00"
    assert golden["length_field_bytes"] == 8
    assert golden["enumeration_command"] == "git ls-tree -r -z --full-tree HEAD"
    assert tuple(golden["accepted_modes"]) == TREE_MODES
    assert golden["max_path_utf8_bytes"] == TREE_MAX_PATH_BYTES
    assert tuple(golden["refusal_codes"]) == TREE_REFUSALS
    assert [item["order"] for item in golden["precondition_vectors"]] == [1, 2, 3]
    assert [item["refusal_code"] for item in golden["precondition_vectors"]] == [
        "staged_index_drift",
        "dirty_tracked_content",
        "mode_observation_unavailable",
    ]
    assert {vector["name"] for vector in golden["accept_vectors"]} == {
        "empty_tree",
        "single_regular_file",
        "regular_executable_symlink",
        "utf8_path_and_empty_blob",
        "nested_paths_in_git_emission_order",
    }
    assert {vector["refusal_code"] for vector in golden["reject_vectors"]} <= set(
        TREE_REFUSALS
    )


def test_empty_tree_digest_is_the_independently_stated_prefix_hash() -> None:
    expected = "68f6186b2db6dc772dc66283dfa332d841dcdc93e80a200afe7e2ee725906f43"
    assert hashlib.sha256(TREE_PREFIX + b"\x00").hexdigest() == expected
    assert _tree_digest_streaming([]) == expected
    empty = next(
        vector
        for vector in _tree_golden()["accept_vectors"]
        if vector["name"] == "empty_tree"
    )
    assert empty["expected_sha256"] == expected
    assert empty["framed_byte_count"] == 42


@pytest.mark.parametrize(
    "name",
    (
        "empty_tree",
        "single_regular_file",
        "regular_executable_symlink",
        "utf8_path_and_empty_blob",
        "nested_paths_in_git_emission_order",
    ),
)
def test_accept_vectors_recompute_identically_through_two_implementations(
    name: str,
) -> None:
    vector = next(
        item for item in _tree_golden()["accept_vectors"] if item["name"] == name
    )
    entries = _tree_entries(vector)
    framed = _tree_frame(entries)
    assert framed.hex() == vector["framed_bytes_hex"]
    assert len(framed) == vector["framed_byte_count"]
    assert hashlib.sha256(framed).hexdigest() == vector["expected_sha256"]
    assert _tree_digest_streaming(entries) == vector["expected_sha256"]
    # Independently reconstruct the stored fixed-width framing field by field.
    rebuilt = bytearray(TREE_PREFIX)
    rebuilt.append(0)
    for mode, path, content in entries:
        mode_bytes = mode.encode("ascii")
        for field in (mode_bytes, path, content):
            rebuilt += len(field).to_bytes(8, "big")
            rebuilt += field
    assert bytes(rebuilt) == framed
    if len(entries) > 1:
        with pytest.raises(TreeRefusal) as excinfo:
            _tree_digest_streaming(list(reversed(entries)))
        assert excinfo.value.code == "out_of_order"


def test_every_reject_vector_raises_its_exact_refusal_code() -> None:
    vectors = _tree_golden()["reject_vectors"]
    assert len(vectors) >= 22
    seen: set[str] = set()
    for vector in vectors:
        if "entry" in vector:
            entry = vector["entry"]
            content = (
                None
                if entry["content_hex"] is None
                else bytes.fromhex(entry["content_hex"])
            )
            with pytest.raises(TreeRefusal) as excinfo:
                _admit(entry["mode"], bytes.fromhex(entry["path_hex"]), content)
            assert excinfo.value.code == vector["refusal_code"], vector["name"]
        elif "entries" in vector:
            with pytest.raises(TreeRefusal) as excinfo:
                _tree_frame(_tree_entries(vector))
            assert excinfo.value.code == vector["refusal_code"], vector["name"]
        elif vector["name"] == "fixed_frame_length_overflow":
            with pytest.raises(TreeRefusal) as excinfo:
                _tree_u64(2**64)
            assert excinfo.value.code == vector["refusal_code"]
        seen.add(vector["refusal_code"])
    assert seen == set(TREE_REFUSALS) - {
        "mode_observation_unavailable",
        "staged_index_drift",
    }


def test_non_nfc_path_refuses_before_framing() -> None:
    decomposed = "cafe\N{COMBINING ACUTE ACCENT}.txt".encode()
    assert unicodedata.normalize("NFC", decomposed.decode()).encode() != decomposed
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_frame([("100644", decomposed, b"x")])
    assert excinfo.value.code == "non_canonical_path"


@pytest.mark.parametrize(
    ("entries", "code"),
    (
        (
            [("100644", b"a", b"one"), ("100644", b"a", b"two")],
            "duplicate_path",
        ),
        (
            [("100644", b"A", b"one"), ("100644", b"a", b"two")],
            "portable_path_collision",
        ),
        (
            [("100644", b"b", b"one"), ("100644", b"a", b"two")],
            "out_of_order",
        ),
    ),
)
def test_entry_sequence_refuses_duplicates_collisions_and_out_of_order(
    entries: list[tuple[str, bytes, bytes]], code: str
) -> None:
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_frame(entries)
    assert excinfo.value.code == code


def test_fixed_width_u64_framing_rejects_overflow() -> None:
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_u64(2**64)
    assert excinfo.value.code == "frame_length_overflow"


def test_single_file_frame_uses_only_fixed_width_u64_lengths() -> None:
    expected = (
        TREE_PREFIX
        + b"\x00"
        + (6).to_bytes(8, "big")
        + b"100644"
        + (5).to_bytes(8, "big")
        + b"a.txt"
        + (6).to_bytes(8, "big")
        + b"hello\n"
    )
    assert _tree_frame([("100644", b"a.txt", b"hello\n")]) == expected


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment.update(
        GIT_CONFIG_GLOBAL=str(repo / "absent-global-config"),
        GIT_CONFIG_SYSTEM=str(repo / "absent-system-config"),
        GIT_AUTHOR_NAME="B33 Fixture",
        GIT_AUTHOR_EMAIL="b33@example.invalid",
        GIT_COMMITTER_NAME="B33 Fixture",
        GIT_COMMITTER_EMAIL="b33@example.invalid",
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        env=environment,
    )


def _tree_preconditions(repo: Path) -> None:
    if os.name != "posix":
        raise TreeRefusal("mode_observation_unavailable")
    if _git(repo, "diff-index", "--cached", "--quiet", "HEAD", "--").returncode != 0:
        raise TreeRefusal("staged_index_drift")
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status.stdout != b"":
        raise TreeRefusal("dirty_tracked_content")
    for setting in ("core.fileMode", "core.symlinks"):
        result = _git(repo, "config", "--type=bool", "--get", setting)
        if result.returncode == 0 and result.stdout.strip() != b"true":
            raise TreeRefusal("mode_observation_unavailable")
        if result.returncode not in (0, 1):
            raise TreeRefusal("mode_observation_unavailable")


def _tree_enumerate(repo: Path) -> list[tuple[str, bytes, bytes]]:
    raw = _git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD").stdout
    entries: list[tuple[str, bytes, bytes]] = []
    for record in raw.split(b"\x00"):
        if not record:
            continue
        header, _, path = record.partition(b"\t")
        mode = header.split(b" ")[0].decode("ascii")
        if mode == "160000":
            raise TreeRefusal("submodule_entry")
        if mode not in TREE_MODES:
            raise TreeRefusal("unsupported_mode")
        _admit(mode, path, b"")
        target = repo / os.fsdecode(path)
        try:
            content = (
                os.fsencode(os.readlink(target))
                if mode == "120000"
                else target.read_bytes()
            )
        except OSError as error:
            raise TreeRefusal("unreadable_bytes") from error
        entries.append((mode, path, content))
    return entries


def _tree_source_digest(repo: Path) -> str:
    _tree_preconditions(repo)
    return _tree_digest_streaming(_tree_enumerate(repo))


def _build_repo(root: Path) -> Path:
    repo = root / "fixture"
    repo.mkdir()
    assert _git(repo, "init", "-q", "-b", "main").returncode == 0
    _git(repo, "config", "user.name", "B33 Fixture")
    _git(repo, "config", "user.email", "b33@example.invalid")
    _git(repo, "config", "core.fileMode", "true")
    _git(repo, "config", "core.symlinks", "true")
    (repo / "README.md").write_bytes(b"# fixture\n")
    (repo / "tools").mkdir()
    script = repo / "tools" / "run.sh"
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    (repo / "link").symlink_to("README.md")
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "fixture").returncode == 0
    return repo


requires_posix_git = pytest.mark.skipif(
    os.name != "posix" or shutil.which("git") is None,
    reason="the clean tracked source tree algorithm is specified for POSIX Git",
)


@requires_posix_git
def test_a_real_git_checkout_reproduces_the_frozen_three_mode_vector(
    tmp_path: Path,
) -> None:
    repo = _build_repo(tmp_path)
    entries = _tree_enumerate(repo)
    assert [(mode, path.decode()) for mode, path, _ in entries] == [
        ("100644", "README.md"),
        ("120000", "link"),
        ("100755", "tools/run.sh"),
    ]
    vector = next(
        item
        for item in _tree_golden()["accept_vectors"]
        if item["name"] == "regular_executable_symlink"
    )
    assert entries == _tree_entries(vector)
    assert _tree_source_digest(repo) == vector["expected_sha256"]
    assert _tree_frame(entries).hex() == vector["framed_bytes_hex"]


@requires_posix_git
def test_dirty_and_staged_trees_refuse_with_their_exact_codes(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    assert _tree_source_digest(repo)
    (repo / "README.md").write_bytes(b"# edited\n")
    with pytest.raises(TreeRefusal) as dirty:
        _tree_source_digest(repo)
    assert dirty.value.code == "dirty_tracked_content"
    assert _git(repo, "add", "README.md").returncode == 0
    with pytest.raises(TreeRefusal) as staged:
        _tree_source_digest(repo)
    assert staged.value.code == "staged_index_drift"


@requires_posix_git
def test_untracked_nonignored_content_refuses_but_ignored_content_does_not(
    tmp_path: Path,
) -> None:
    repo = _build_repo(tmp_path)
    (repo / ".gitignore").write_text("ignored.txt\n")
    assert _git(repo, "add", ".gitignore").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "ignore rule").returncode == 0
    (repo / "ignored.txt").write_bytes(b"ignored")
    assert _tree_source_digest(repo)
    (repo / "untracked.txt").write_bytes(b"untracked")
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_source_digest(repo)
    assert excinfo.value.code == "dirty_tracked_content"


@requires_posix_git
def test_unobservable_modes_refuse_rather_than_guess(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    _git(repo, "config", "core.fileMode", "false")
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_source_digest(repo)
    assert excinfo.value.code == "mode_observation_unavailable"


@requires_posix_git
@pytest.mark.parametrize("setting", ("core.fileMode", "core.symlinks"))
def test_explicitly_false_mode_setting_refuses(tmp_path: Path, setting: str) -> None:
    repo = _build_repo(tmp_path)
    assert _git(repo, "config", setting, "false").returncode == 0
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_source_digest(repo)
    assert excinfo.value.code == "mode_observation_unavailable"


@requires_posix_git
@pytest.mark.parametrize("setting", ("core.fileMode", "core.symlinks"))
def test_explicitly_true_mode_setting_is_accepted(tmp_path: Path, setting: str) -> None:
    repo = _build_repo(tmp_path)
    assert _git(repo, "config", setting, "true").returncode == 0
    assert _tree_source_digest(repo)


@requires_posix_git
@pytest.mark.parametrize("setting", ("core.fileMode", "core.symlinks"))
def test_unset_mode_setting_uses_posix_git_enabled_default(
    tmp_path: Path, setting: str
) -> None:
    repo = _build_repo(tmp_path)
    assert _git(repo, "config", "--unset", setting).returncode == 0
    assert _tree_source_digest(repo)


@requires_posix_git
def test_a_committed_gitlink_refuses_as_a_submodule_entry(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert (
        _git(
            repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor/dep"
        ).returncode
        == 0
    )
    assert _git(repo, "commit", "-q", "-m", "gitlink").returncode == 0
    with pytest.raises(TreeRefusal) as excinfo:
        _tree_enumerate(repo)
    assert excinfo.value.code == "submodule_entry"


def test_build_descriptor_schema_does_not_pin_source_commit_eligibility() -> None:
    schema = _document("build-provenance-descriptor-v1.schema.json")
    source_commit = schema["properties"]["source_commit"]
    assert "const" not in source_commit
    assert source_commit["pattern"] == r"^[0-9a-f]{40}(?![\s\S])"
    assert schema["properties"]["source_tree_algorithm"]["const"] == TREE_ALGORITHM
    description = schema["description"].lower()
    for term in ("git metadata", "generated", "ignored"):
        assert term in description
