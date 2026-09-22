"""Pure-offline Identity Manifest v1 identities and trust-boundary validation.

This module provides construction, recomputation, projection-backed manifest
validation, and bounded component-requirement evaluation.  It does not import
replay, Artifact, evaluator, provider, result, or analysis logic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

from operatebench.jsonsafe import (
    JsonSafetyError,
    canonical_json_bytes,
    ensure_raw_json_depth,
)

IDENTITY_MANIFEST_SCHEMA: Final = "operatebench.identity_manifest.v1"
COMPONENT_IDENTITY_SCHEMA: Final = "operatebench.component_identity.v1"
COMPOSITE_IDENTITY_SCHEMA: Final = "operatebench.composite_identity.v1"
COMPATIBILITY_CONTRACT_SCHEMA: Final = "operatebench.compatibility_contract.v1"
IDENTITY_MANIFEST_PROJECTION_CENSUS_SCHEMA: Final = (
    "operatebench.identity_manifest_projection_census.v1"
)
MANIFEST_DIGEST_DOMAIN: Final = "operatebench.digest.identity_manifest.v1"
IDENTITY_SCHEMA_VERSION: Final = 1
OWNED_MUTATION_CONTRACT_VERSION: Final = 1

# Frozen Identity Manifest v1 semantic resource envelope.
MAX_OWNED_PROJECTION_FIELDS: Final = 4_096
MAX_STRICT_JSON_NODES: Final = 65_536
MAX_CANONICAL_UTF8_PAYLOAD_BYTES: Final = 4 * 1024 * 1024
MAX_IDENTITY_STRING_BYTES: Final = 1024 * 1024
MAX_STRICT_JSON_CONTAINER_ITEMS: Final = 4_096
MAX_INTEROPERABLE_INTEGER: Final = 2**53 - 1
MAX_COMPONENT_IDENTITY_BYTES: Final = 336 * 1024
MAX_COMPONENT_IDENTITY_NODES: Final = 5_600
MAX_IDENTITY_RESOURCE_BYTES: Final = 4 * 1024 * 1024
MAX_IDENTITY_RESOURCE_READ_CALLS: Final = 128
_RESOURCE_READ_CHUNK_BYTES: Final = 64 * 1024

_RESOURCE_PACKAGE = "operatebench.resources.identity"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_POINTER_RE = re.compile(r"^/(?:[^/~*]|~0|~1)+(?:/(?:[^/~*]|~0|~1)+)*$")
_COMPONENT_DAG: Final = {
    "card_schema_content": (),
    "build_provenance_content": (),
    "operation_core_content": ("card_schema_content",),
    "semantic_scenario_content": ("operation_core_content", "card_schema_content"),
    "variant_content": ("semantic_scenario_content", "card_schema_content"),
    "runtime_contract_content": ("operation_core_content", "card_schema_content"),
    "scaffold_content": ("runtime_contract_content",),
    "arm_protocol_content": (
        "runtime_contract_content",
        "semantic_scenario_content",
        "variant_content",
    ),
    "provider_run_plan_content": (
        "runtime_contract_content",
        "scaffold_content",
        "build_provenance_content",
    ),
    "evaluator_bundle_content": (
        "operation_core_content",
        "card_schema_content",
        "runtime_contract_content",
    ),
    "analysis_contract_content": ("evaluator_bundle_content",),
}
_MAX_COMPONENT_NAME_CHARS: Final = max(map(len, _COMPONENT_DAG))
_COMPOSITE_MEMBERS: Final = {
    "runtime_bundle": (
        "operation_core_content",
        "semantic_scenario_content",
        "variant_content",
        "runtime_contract_content",
        "scaffold_content",
        "arm_protocol_content",
        "provider_run_plan_content",
        "build_provenance_content",
    ),
    "experiment_bundle": (
        "runtime_bundle",
        "evaluator_bundle_content",
        "analysis_contract_content",
    ),
}
_EXPERIMENT_LEAF_CLOSURE: Final = (
    *_COMPOSITE_MEMBERS["runtime_bundle"],
    "evaluator_bundle_content",
    "analysis_contract_content",
)

_FrozenJSONObject: TypeAlias = dict[str, Any] | MappingProxyType[str, Any]


class IdentityValidationError(ValueError):
    """An identity value fails the closed Identity Manifest v1 contract."""


@dataclass(frozen=True, slots=True)
class RequirementV1:
    """One normalized direct component compatibility requirement."""

    component: str
    compatible: _FrozenJSONObject


@dataclass(frozen=True, slots=True)
class RequirementEvaluationV1:
    """Detached result of the v1 component-requirement conjunction."""

    requirement_contract_version: int
    satisfied: bool
    reasons: tuple[str, ...]


REQUIREMENT_EVALUATION_REASON_ORDER: Final = (
    "unknown_named_requirement_predicate",
    "missing_component",
    "content_digest_mismatch",
    "schema_version_out_of_range",
)


@dataclass(frozen=True, slots=True)
class ComponentIdentityV1:
    """Detached immutable identity of one registry component projection."""

    name: str
    domain: str
    schema: str
    schema_version: int
    content_digest_sha256: str
    owned_fields: tuple[str, ...]
    owned_projection: _FrozenJSONObject
    excludes: tuple[str, ...]
    requires: tuple[RequirementV1, ...]


@dataclass(frozen=True, slots=True)
class CompositeIdentityV1:
    """Detached immutable identity of one ordered registry composite."""

    name: str
    domain: str
    schema: str
    schema_version: int
    members: tuple[str, ...]
    bundle_digest_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedIdentityManifest:
    """Freshly recomputed, detached Identity Manifest v1 trust-boundary result."""

    schema: str
    schema_version: int
    components: MappingProxyType[str, ComponentIdentityV1]
    composites: MappingProxyType[str, CompositeIdentityV1]
    compatibility_contract: MappingProxyType[str, Any]
    manifest_digest_sha256: str


@dataclass(frozen=True, slots=True)
class OwnedMutation:
    """One detached component-owned Identity Manifest v1 mutation."""

    mutation_contract_version: int
    owner: str
    changed_field_paths: tuple[str, ...]
    changed_requirement_components: tuple[str, ...]
    old_content_digest_sha256: str
    new_content_digest_sha256: str
    changed_composites: tuple[str, ...]


_IdentityRecord: TypeAlias = ComponentIdentityV1 | CompositeIdentityV1
_ResolvedMemberIdentities: TypeAlias = (
    dict[str, ComponentIdentityV1]
    | dict[str, CompositeIdentityV1]
    | dict[str, _IdentityRecord]
    | MappingProxyType[str, ComponentIdentityV1]
    | MappingProxyType[str, CompositeIdentityV1]
    | MappingProxyType[str, _IdentityRecord]
)


def _fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise IdentityValidationError(message)
    raise IdentityValidationError(message) from cause


def _strip_exception_tracebacks(exc: BaseException) -> None:
    """Detach traceback frames without changing exception identity or taxonomy."""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


# No typing import at runtime merely for the helper's bottom return type.
from typing import NoReturn  # noqa: E402


def _normalize_strict_json(
    value: Any,
    context: str,
    *,
    allow_frozen: bool = False,
    max_container_items: int | None = None,
    max_nodes: int | None = None,
    max_payload_bytes: int | None = None,
) -> Any:
    """Return trusted JSON after bounded validation and integer normalization."""
    nodes = 0
    payload_bytes = 0
    container_item_limit = (
        MAX_STRICT_JSON_CONTAINER_ITEMS
        if max_container_items is None
        else max_container_items
    )
    node_limit = MAX_STRICT_JSON_NODES if max_nodes is None else max_nodes
    payload_limit = (
        MAX_CANONICAL_UTF8_PAYLOAD_BYTES
        if max_payload_bytes is None
        else max_payload_bytes
    )

    def add_bytes(amount: int) -> None:
        nonlocal payload_bytes
        payload_bytes += amount
        if payload_bytes > payload_limit:
            raise OverflowError

    def quoted_size(text: str) -> int:
        if len(text) > MAX_IDENTITY_STRING_BYTES:
            raise OverflowError
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_IDENTITY_STRING_BYTES:
            raise OverflowError
        return len(json.dumps(text, ensure_ascii=False).encode("utf-8"))

    def walk(item: Any, depth: int = 0) -> Any:
        nonlocal nodes
        pairs: Any = None
        pair: Any = None
        key: Any = None
        child: Any = None
        try:
            nodes += 1
            if nodes > node_limit:
                raise OverflowError
            if item is None:
                add_bytes(4)
                return None
            if type(item) is bool:
                add_bytes(4 if item else 5)
                return item
            if type(item) is int:
                if not -MAX_INTEROPERABLE_INTEGER <= item <= MAX_INTEROPERABLE_INTEGER:
                    raise ValueError
                add_bytes(len(str(item)))
                return item
            if type(item) is float:
                if (
                    not math.isfinite(item)
                    or not item.is_integer()
                    or not -MAX_INTEROPERABLE_INTEGER <= item <= MAX_INTEROPERABLE_INTEGER
                ):
                    raise ValueError
                normalized = int(item)
                add_bytes(len(str(normalized)))
                return normalized
            if type(item) is str:
                add_bytes(quoted_size(item))
                return item
            if type(item) is dict or (allow_frozen and type(item) is MappingProxyType):
                if depth >= 64:
                    raise OverflowError
                expected_count: int | None = None
                if type(item) is dict:
                    expected_count = len(item)
                    if expected_count > container_item_limit:
                        raise OverflowError
                try:
                    pairs = iter(item.items())
                except BaseExceptionGroup:
                    raise
                except Exception as exc:
                    _fail(f"{context} is not canonical Identity v1 JSON", exc)
                add_bytes(2)
                normalized_object: dict[str, Any] = {}
                item_count = 0
                while True:
                    try:
                        pair = next(pairs)
                    except StopIteration:
                        break
                    except BaseExceptionGroup:
                        raise
                    except Exception as exc:
                        _fail(f"{context} is not canonical Identity v1 JSON", exc)
                    item_count += 1
                    if item_count > container_item_limit:
                        raise OverflowError
                    if type(pair) is not tuple or len(pair) != 2:
                        raise TypeError
                    key, child = pair
                    if type(key) is not str:
                        raise TypeError
                    if key in normalized_object:
                        raise TypeError
                    if item_count > 1:
                        add_bytes(1)
                    add_bytes(quoted_size(key) + 1)
                    normalized_object[key] = walk(child, depth + 1)
                    pair = None
                    key = None
                    child = None
                if expected_count is not None and item_count != expected_count:
                    raise TypeError
                return normalized_object
            if type(item) is list or (allow_frozen and type(item) is tuple):
                if depth >= 64 or len(item) > container_item_limit:
                    raise OverflowError
                add_bytes(2 + max(0, len(item) - 1))
                normalized_array: list[Any] = []
                for index in range(len(item)):
                    child = item[index]
                    normalized_array.append(walk(child, depth + 1))
                    child = None
                return normalized_array
            raise TypeError
        finally:
            # Tracebacks retain frame locals. Clear every alias into the caller's
            # tree before a normalization failure can leave this frame suspended.
            item = None
            pairs = None
            pair = None
            key = None
            child = None

    try:
        return walk(value)
    except IdentityValidationError:
        raise
    except OverflowError as exc:
        _fail(f"{context} exceeds Identity v1 resource limits", exc)
    except (JsonSafetyError, UnicodeError, ValueError, TypeError) as exc:
        _fail(f"{context} is not canonical Identity v1 JSON", exc)
    finally:
        # The public validator passes one raw caller root at a time. Do not let
        # this outer normalization frame keep that root alive after rejection.
        value = None


def _strict_json(value: Any, context: str, *, allow_frozen: bool = False) -> None:
    """Validate the bounded Identity v1 JSON profile."""
    _normalize_strict_json(value, context, allow_frozen=allow_frozen)


def _freeze(value: Any) -> Any:
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if type(value) in (dict, MappingProxyType):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    _fail("unsupported identity projection object")


def _duplicates_refused(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, context: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        ensure_raw_json_depth(text, context)
        value = json.loads(
            text,
            object_pairs_hook=_duplicates_refused,
            parse_float=lambda token: (_ for _ in ()).throw(
                ValueError(f"forbidden float {token}")
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"forbidden constant {token}")
            ),
        )
        if type(value) is not dict:
            raise TypeError("resource root is not an object")
        _strict_json(value, context)
        return cast(dict[str, Any], value)
    except IdentityValidationError:
        raise
    except (
        UnicodeError,
        JsonSafetyError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        _fail(f"invalid packaged identity resource {context!r}", exc)


def _read_resource_bytes(name: str) -> bytes:
    resource_package: Any = None
    resource: Any = None
    stream: Any = None
    open_failure: BaseException | None = None
    try:
        resource_package = files(_RESOURCE_PACKAGE)
    except BaseException as exc:
        open_failure = exc
    if open_failure is not None:
        resource_package = None
        _strip_exception_tracebacks(open_failure)
        if isinstance(open_failure, ImportError):
            _fail("cannot resolve packaged identity resource package", open_failure)
        raise open_failure
    try:
        resource = resource_package.joinpath(name)
        stream = resource.open("rb")
    except BaseException as exc:
        open_failure = exc
    if open_failure is not None:
        resource = None
        resource_package = None
        _strip_exception_tracebacks(open_failure)
        if isinstance(open_failure, OSError):
            _fail(f"cannot read packaged identity resource {name!r}", open_failure)
        raise open_failure
    chunks: list[bytes] = []
    total = 0
    read_calls = 0
    chunk: object = None
    primary_failure: BaseException | None = None
    close_failure: BaseException | None = None
    result: bytes | None = None
    try:
        while True:
            if read_calls >= MAX_IDENTITY_RESOURCE_READ_CALLS:
                _fail(f"packaged identity resource {name!r} exceeds resource limits")
            remaining = MAX_IDENTITY_RESOURCE_BYTES + 1 - total
            requested = min(_RESOURCE_READ_CHUNK_BYTES, remaining)
            read_calls += 1
            chunk = stream.read(requested)
            if type(chunk) is not bytes:
                chunk = None
                _fail(
                    "packaged identity resource stream returned non-bytes",
                    TypeError("packaged resource read must return bytes"),
                )
            if chunk == b"":
                break
            if len(chunk) > requested:
                chunk = None
                _fail(
                    "packaged identity resource stream exceeded requested read size",
                    ValueError("packaged resource read exceeded requested size"),
                )
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_IDENTITY_RESOURCE_BYTES:
                _fail(f"packaged identity resource {name!r} exceeds resource limits")
    except BaseException as exc:
        primary_failure = exc
    try:
        stream.close()
    except BaseException as exc:
        close_failure = exc
    if primary_failure is None and close_failure is None:
        try:
            result = b"".join(chunks)
        except BaseException as exc:
            primary_failure = exc

    chunks.clear()
    chunk = None
    stream = None
    resource = None
    resource_package = None

    if primary_failure is not None:
        _strip_exception_tracebacks(primary_failure)
        if close_failure is not None:
            _strip_exception_tracebacks(close_failure)
            close_failure = None
        if isinstance(primary_failure, OSError):
            _fail(f"cannot read packaged identity resource {name!r}", primary_failure)
        raise primary_failure
    if close_failure is not None:
        _strip_exception_tracebacks(close_failure)
        if isinstance(close_failure, BaseExceptionGroup):
            raise close_failure
        if isinstance(close_failure, Exception):
            _fail(f"cannot read packaged identity resource {name!r}", close_failure)
        raise close_failure
    assert result is not None
    return result


_RESOURCE_SHA256: Final = {
    "identity-manifest-v1.schema.json": (
        "a09a81be34c172df781af0b8ec98f6e0de3a67540cc042412b5560998ab450d5"
    ),
    "identity-component-registry-v1.json": (
        "852690707e5864ba3d15591094a9c1872dbce2ec4458483c2523ea554553db5f"
    ),
    "identity-compatibility-v1.registry.json": (
        "bcb84fdac5a545e34f002113eed1ac7231d7387c0cd61a6043acc6895282ba44"
    ),
    "identity-manifest-digest-v1.golden.json": (
        "01bceeb07cf195415d519f570e13c697b80f0c4ab4ae7945763fcc1a83225948"
    ),
    "identity-manifest-projection-census-v1.schema.json": (
        "52e9f87f423f86f7ef3ffd0a8a7fe17fdff3a4cd0d4fe07afcb7281b1521ad12"
    ),
}


@cache
def _cached_resource(name: str) -> MappingProxyType[str, Any]:
    raw = _read_resource_bytes(name)
    if type(raw) is not bytes:
        _fail("packaged identity resource reader returned malformed bytes")
    expected = _RESOURCE_SHA256.get(name)
    if expected is None or hashlib.sha256(raw).hexdigest() != expected:
        _fail(
            "packaged identity resource bytes do not match the frozen contract",
            ValueError("packaged identity resource digest mismatch"),
        )
    value = _load_json_bytes(raw, name)
    if name == "identity-component-registry-v1.json":
        _validate_component_registry(value)
    elif name == "identity-compatibility-v1.registry.json":
        _validate_compatibility_registry(value)
    return cast(MappingProxyType[str, Any], _freeze(value))


@cache
def _named_requirement_predicate_allowlist() -> frozenset[tuple[str, int]]:
    registry = _cached_resource("identity-compatibility-v1.registry.json")
    contract = cast(MappingProxyType[str, Any], registry["contract"])
    rows = cast(
        tuple[MappingProxyType[str, Any], ...],
        contract["named_requirement_predicates"],
    )
    return frozenset(
        (cast(str, row["predicate"]), cast(int, row["predicate_version"])) for row in rows
    )


def _clear_resource_caches() -> None:
    """Clear private resource caches for isolated trust-boundary tests."""
    _named_requirement_predicate_allowlist.cache_clear()
    _cached_resource.cache_clear()


def _resource(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], _plain(_cached_resource(name)))


def load_identity_schema() -> dict[str, Any]:
    """Return a fresh copy of the strict Draft 2020-12 manifest schema."""
    return _resource("identity-manifest-v1.schema.json")


def load_component_registry() -> dict[str, Any]:
    """Return a fresh copy of the frozen component/composite DAG registry."""
    return _resource("identity-component-registry-v1.json")


def load_compatibility_registry() -> dict[str, Any]:
    """Return a fresh copy of the closed compatibility vocabulary registry."""
    return _resource("identity-compatibility-v1.registry.json")


def _evaluate_component_requirements(
    requirements: list[dict[str, Any]],
    resolved_components: dict[str, ComponentIdentityV1],
) -> RequirementEvaluationV1:
    normalized_requirements: list[dict[str, Any]] = []
    validated_components: dict[str, ComponentIdentityV1] = {}
    name: str | None = None
    candidate: ComponentIdentityV1 | None = None
    raw: dict[str, Any] | None = None
    component: str | None = None
    compatible: dict[str, Any] | None = None
    compatible_value: Any = None
    kind: str | None = None
    target: ComponentIdentityV1 | None = None
    expected: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    predicate: str | None = None
    predicate_version: int | None = None
    allowlist: frozenset[tuple[str, int]] = frozenset()
    reasons: set[str] = set()
    try:
        normalized_requirements = cast(
            list[dict[str, Any]],
            _normalize_strict_json(
                requirements,
                "component requirements",
                max_container_items=MAX_STRICT_JSON_CONTAINER_ITEMS,
                max_nodes=MAX_STRICT_JSON_NODES,
                max_payload_bytes=MAX_CANONICAL_UTF8_PAYLOAD_BYTES,
            ),
        )
        for name in sorted(resolved_components):
            candidate = _validate_self_contained_component(resolved_components[name])
            if candidate.name != name:
                _fail(f"resolved component {name!r} has the wrong identity name")
            validated_components[name] = candidate

        allowlist = _named_requirement_predicate_allowlist()
        for index, raw in enumerate(normalized_requirements):
            _exact_keys(
                raw, {"component", "compatible"}, f"component requirement {index}"
            )
            component = _namespaced(
                raw["component"], f"component requirement {index} component"
            )
            if component not in _COMPONENT_DAG:
                _fail("unknown component requirement")
            compatible_value = raw["compatible"]
            if type(compatible_value) is not dict:
                _fail(
                    f"component requirement {index} compatible must be an ordinary object"
                )
            compatible = compatible_value
            kind = compatible.get("kind")
            if type(kind) is not str:
                _fail(
                    f"component requirement {index} compatible must declare a string kind"
                )
            target = validated_components.get(component)
            if target is None:
                reasons.add("missing_component")

            if kind == "exact_digest":
                _exact_keys(
                    compatible,
                    {"kind", "content_digest_sha256"},
                    "exact-digest component requirement",
                )
                expected = _digest(
                    compatible["content_digest_sha256"], "required component digest"
                )
                if target is not None and target.content_digest_sha256 != expected:
                    reasons.add("content_digest_mismatch")
            elif kind == "closed_schema_version_range":
                _exact_keys(
                    compatible,
                    {"kind", "minimum_schema_version", "maximum_schema_version"},
                    "closed schema-version component requirement",
                )
                minimum = _positive_int(
                    compatible["minimum_schema_version"], "minimum_schema_version"
                )
                maximum = _positive_int(
                    compatible["maximum_schema_version"], "maximum_schema_version"
                )
                if minimum > maximum:
                    _fail("closed schema version range minimum exceeds maximum")
                if target is not None and not minimum <= target.schema_version <= maximum:
                    reasons.add("schema_version_out_of_range")
            elif kind == "named_predicate":
                _exact_keys(
                    compatible,
                    {"kind", "predicate", "predicate_version"},
                    "named-predicate component requirement",
                )
                predicate = _namespaced(
                    compatible["predicate"], "component requirement predicate"
                )
                if "." not in predicate:
                    _fail("component requirement predicate must use a dotted namespace")
                predicate_version = _positive_int(
                    compatible["predicate_version"], "predicate_version"
                )
                if (predicate, predicate_version) not in allowlist:
                    reasons.add("unknown_named_requirement_predicate")
            else:
                _fail("unknown component requirement kind")

            raw = None
            component = None
            compatible = None
            compatible_value = None
            kind = None
            target = None
            expected = None
            minimum = None
            maximum = None
            predicate = None
            predicate_version = None

        ordered = tuple(
            reason for reason in REQUIREMENT_EVALUATION_REASON_ORDER if reason in reasons
        )
        return RequirementEvaluationV1(1, not ordered, ordered)
    finally:
        requirements = []
        resolved_components = {}
        normalized_requirements.clear()
        validated_components.clear()
        name = None
        candidate = None
        raw = None
        component = None
        compatible = None
        compatible_value = None
        kind = None
        target = None
        expected = None
        minimum = None
        maximum = None
        predicate = None
        predicate_version = None
        allowlist = frozenset()
        reasons.clear()


def evaluate_component_requirements(
    requirements: list[dict[str, Any]],
    resolved_components: dict[str, ComponentIdentityV1],
) -> RequirementEvaluationV1:
    """Evaluate only the bounded conjunction of explicit component requirements.

    The empty list is the true empty conjunction.  Satisfaction is not permission
    to replay, regrade, recompute analysis, compare, or pool results.
    """

    failure_message: str | None = None
    name: object | None = None
    try:
        if type(requirements) is not list:
            _fail("compatibility requirements must be an ordinary array")
        if type(resolved_components) is not dict:
            _fail("resolved components must be an ordinary object")
        if len(resolved_components) > len(_COMPONENT_DAG):
            _fail("resolved components exceed the closed Identity v1 component set")
        for name in resolved_components:
            if type(name) is not str:
                _fail("resolved component names must be exact strings")
            if len(name) > _MAX_COMPONENT_NAME_CHARS or name not in _COMPONENT_DAG:
                _fail(
                    "resolved component name is outside the closed Identity v1 "
                    "component set"
                )
            name = None
        if len(requirements) > MAX_STRICT_JSON_CONTAINER_ITEMS:
            _fail("compatibility requirements exceed Identity v1 resource limits")
        return _evaluate_component_requirements(requirements, resolved_components)
    except IdentityValidationError as exc:
        if (
            type(exc) is IdentityValidationError
            and len(exc.args) == 1
            and type(exc.args[0]) is str
        ):
            failure_message = exc.args[0]
        else:
            failure_message = "component requirement evaluation rejected malformed input"
        _strip_exception_tracebacks(exc)
    finally:
        requirements = []
        resolved_components = {}
        name = None
    assert failure_message is not None
    raise IdentityValidationError(failure_message) from None


def load_identity_golden_vectors() -> dict[str, Any]:
    """Return detached canonical-byte and digest conformance vectors."""
    return _resource("identity-manifest-digest-v1.golden.json")


def load_identity_manifest_projection_census_schema() -> dict[str, Any]:
    """Return a fresh copy of the strict manifest-projection census schema."""

    return _resource("identity-manifest-projection-census-v1.schema.json")


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if type(value) is not dict:
        _fail(f"{context} must be an ordinary object")
    if any(type(key) is not str for key in value):
        _fail(f"{context} keys must be exact strings")
    if set(value) != expected:
        _fail(f"{context} fields must be exactly {sorted(expected)!r}")


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare normalized strict-JSON trees without Python's bool/int aliasing."""

    value_type = type(left)
    if value_type is not type(right):
        return False
    if left is None:
        return True
    if value_type in (bool, int, str):
        return bool(left == right)
    if value_type is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if value_type is dict:
        if len(left) != len(right):
            return False
        return all(
            type(key) is str
            and key in right
            and _exact_json_equal(left_value, right[key])
            for key, left_value in left.items()
        )
    return False


def _positive_int(value: Any, context: str) -> int:
    if type(value) is int:
        normalized = value
    elif type(value) is float and math.isfinite(value) and value.is_integer():
        normalized = int(value)
    else:
        _fail(f"{context} must be a positive interoperable integer")
    if not 1 <= normalized <= MAX_INTEROPERABLE_INTEGER:
        _fail(f"{context} must be a positive interoperable integer")
    return normalized


def _namespaced(value: Any, context: str) -> str:
    if type(value) is not str:
        _fail(f"{context} must be a lowercase namespaced string")
    if len(value) > MAX_IDENTITY_STRING_BYTES:
        _fail(f"{context} exceeds Identity v1 resource limits")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        _fail(f"{context} must be a lowercase namespaced string", exc)
    if len(encoded) > MAX_IDENTITY_STRING_BYTES:
        _fail(f"{context} exceeds Identity v1 resource limits")
    if _NAME_RE.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase namespaced string")
    return value


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        _fail(f"{context} must be 64 lowercase hexadecimal characters")
    if _DIGEST_RE.fullmatch(value) is None:
        _fail(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _validate_component_registry(registry: dict[str, Any]) -> None:
    _exact_keys(
        registry,
        {"schema", "schema_version", "components", "composites"},
        "component registry",
    )
    if registry["schema"] != "operatebench.identity_component_registry.v1":
        _fail("unsupported component registry schema")
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        _fail("unsupported component registry schema_version")
    components = registry["components"]
    composites = registry["composites"]
    if type(components) is not list or type(composites) is not list:
        _fail("component registry groups must be arrays")
    component_names: list[str] = []
    domains: list[str] = [MANIFEST_DIGEST_DOMAIN]
    edges: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(components):
        if type(row) is not dict:
            _fail(f"component registry row {index} must be an object")
        _exact_keys(
            row,
            {"name", "domain", "schema", "schema_version", "requires"},
            f"component row {index}",
        )
        name = _namespaced(row["name"], f"component row {index} name")
        domain = _namespaced(row["domain"], f"component row {index} domain")
        if row["schema"] != COMPONENT_IDENTITY_SCHEMA:
            _fail(f"component {name} declares the wrong identity schema")
        if type(row["schema_version"]) is not int or row["schema_version"] != 1:
            _fail(f"component {name} declares the wrong identity schema_version")
        requires = row["requires"]
        if type(requires) is not list or any(type(item) is not str for item in requires):
            _fail(f"component {name} requires must be a string array")
        if len(requires) != len(set(requires)):
            _fail(f"component {name} has duplicate requirements")

        component_names.append(name)
        domains.append(domain)
        edges[name] = tuple(requires)
    if len(component_names) != len(set(component_names)):
        _fail("component registry has duplicate names")
    if tuple(component_names) != tuple(_COMPONENT_DAG) or edges != _COMPONENT_DAG:
        _fail("component registry is not the frozen Identity v1 DAG")
    seen: set[str] = set()
    for name in component_names:
        for dependency in edges[name]:
            if dependency not in seen:
                _fail(
                    f"component {name} has missing, forward, or cyclic dependency "
                    f"{dependency!r}"
                )
        seen.add(name)
    composite_names: list[str] = []
    available = set(component_names)
    for index, row in enumerate(composites):
        if type(row) is not dict:
            _fail(f"composite registry row {index} must be an object")
        _exact_keys(
            row,
            {"name", "domain", "schema", "schema_version", "members"},
            f"composite row {index}",
        )
        name = _namespaced(row["name"], f"composite row {index} name")
        domains.append(_namespaced(row["domain"], f"composite row {index} domain"))
        if row["schema"] != COMPOSITE_IDENTITY_SCHEMA:
            _fail(f"composite {name} declares the wrong identity schema")
        if type(row["schema_version"]) is not int or row["schema_version"] != 1:
            _fail(f"composite {name} declares the wrong identity schema_version")
        members = row["members"]
        if type(members) is not list or any(type(item) is not str for item in members):
            _fail(f"composite {name} members must be a string array")
        if not members or len(members) != len(set(members)):
            _fail(f"composite {name} members must be nonempty and unique")
        if any(member not in available for member in members):
            _fail(f"composite {name} has missing, forward, or cyclic member")
        composite_names.append(name)
        available.add(name)
    if len(composite_names) != len(set(composite_names)) or set(component_names) & set(
        composite_names
    ):
        _fail("registry names collide")
    registered_composites = {row["name"]: tuple(row["members"]) for row in composites}
    if (
        tuple(composite_names) != tuple(_COMPOSITE_MEMBERS)
        or registered_composites != _COMPOSITE_MEMBERS
    ):
        _fail("component registry is not the frozen Identity v1 composite graph")
    if len(domains) != len(set(domains)):
        _fail("identity digest domains must be globally unique")


def _validate_compatibility_registry(registry: dict[str, Any]) -> None:
    _exact_keys(
        registry,
        {"schema", "schema_version", "contract", "reason_codes"},
        "compatibility registry",
    )
    if registry["schema"] != "operatebench.identity_compatibility_registry.v1":
        _fail("unsupported compatibility registry schema")
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        _fail("unsupported compatibility registry schema_version")
    contract = registry["contract"]
    if type(contract) is not dict:
        _fail("compatibility contract must be an object")
    _exact_keys(
        contract,
        {
            "schema",
            "schema_version",
            "requirement_kinds",
            "decision_predicates",
            "named_requirement_predicates",
        },
        "compatibility contract",
    )
    if (
        contract["schema"] != COMPATIBILITY_CONTRACT_SCHEMA
        or type(contract["schema_version"]) is not int
        or contract["schema_version"] != 1
    ):
        _fail("unsupported compatibility contract")
    expected_kinds = {
        "exact_digest": 1,
        "closed_schema_version_range": 1,
        "named_predicate": 1,
    }
    expected_decisions = dict.fromkeys(
        (
            "can_runtime_replay",
            "can_evaluator_regrade",
            "can_analysis_recompute",
            "can_compare",
            "can_pool",
        ),
        1,
    )
    if (
        contract["requirement_kinds"] != expected_kinds
        or contract["decision_predicates"] != expected_decisions
    ):
        _fail("compatibility registry vocabulary is not the closed v1 vocabulary")
    if registry["reason_codes"] != []:
        _fail("compatibility registry reason codes are not the closed v1 vocabulary")
    if contract["named_requirement_predicates"] != []:
        _fail("Identity v1 does not register named requirement predicates")


def _component_rows() -> dict[str, dict[str, Any]]:
    registry = load_component_registry()
    return {row["name"]: row for row in registry["components"]}


def _composite_rows() -> dict[str, dict[str, Any]]:
    registry = load_component_registry()
    return {row["name"]: row for row in registry["composites"]}


def _normalize_compatible(value: Any) -> MappingProxyType[str, Any]:
    if type(value) is not dict:
        _fail("requirement compatible value must be an object")
    kind = value.get("kind")
    if type(kind) is not str:
        _fail("compatibility requirement kind must be an exact string")
    normalized: dict[str, Any]
    if kind == "exact_digest":
        _exact_keys(value, {"kind", "content_digest_sha256"}, "exact_digest requirement")
        normalized = {
            "kind": kind,
            "content_digest_sha256": _digest(
                value["content_digest_sha256"], "required digest"
            ),
        }
    elif kind == "closed_schema_version_range":
        _exact_keys(
            value,
            {"kind", "minimum_schema_version", "maximum_schema_version"},
            "version-range requirement",
        )
        minimum = _positive_int(value["minimum_schema_version"], "minimum_schema_version")
        maximum = _positive_int(value["maximum_schema_version"], "maximum_schema_version")
        if minimum > maximum:
            _fail("closed schema version range minimum exceeds maximum")
        normalized = {
            "kind": kind,
            "minimum_schema_version": minimum,
            "maximum_schema_version": maximum,
        }
    elif kind == "named_predicate":
        _exact_keys(
            value,
            {"kind", "predicate", "predicate_version"},
            "named-predicate requirement",
        )
        predicate = _namespaced(value["predicate"], "requirement predicate")
        if "." not in predicate:
            _fail("requirement predicate must use a dotted namespace")
        version = _positive_int(value["predicate_version"], "predicate_version")
        normalized = {"kind": kind, "predicate": predicate, "predicate_version": version}
    else:
        _fail("unsupported compatibility requirement kind")
    return cast(MappingProxyType[str, Any], _freeze(normalized))


def _normalize_requirements(
    value: Any, expected: tuple[str, ...]
) -> tuple[RequirementV1, ...]:
    if type(value) is not list:
        _fail("component requirements must be an array")
    if len(value) != len(expected):
        _fail("component requirements do not match registry cardinality")
    result: list[RequirementV1] = []
    for index, row in enumerate(value):
        if type(row) is not dict:
            _fail(f"requirement {index} must be an object")
        _exact_keys(row, {"component", "compatible"}, f"requirement {index}")
        if type(row["component"]) is not str:
            _fail(f"requirement {index} component must be an exact string")
        result.append(
            RequirementV1(row["component"], _normalize_compatible(row["compatible"]))
        )
    if tuple(item.component for item in result) != tuple(expected):
        _fail(
            "requirements must name direct dependencies exactly in registry order "
            f"{tuple(expected)!r}"
        )
    return tuple(result)


def _validate_pointer(path: Any) -> str:
    if (
        type(path) is not str
        or len(path) > 1024
        or _POINTER_RE.fullmatch(path) is None
        or "*" in path
    ):
        _fail("owned field is not a concrete non-root non-empty-token JSON pointer")
    return path


def _component_wire_projection(identity: ComponentIdentityV1) -> dict[str, Any]:
    """Build the plain wire projection without invoking public validation recursively."""
    return {
        "domain": identity.domain,
        "schema": identity.schema,
        "schema_version": identity.schema_version,
        "content_digest_sha256": identity.content_digest_sha256,
        "owned_fields": list(identity.owned_fields),
        "owned_projection": _plain(identity.owned_projection),
        "excludes": list(identity.excludes),
        "requires": [
            {"component": item.component, "compatible": _plain(item.compatible)}
            for item in identity.requires
        ],
    }


def recompute_component(
    name: str, owned_projection: dict[str, Any], requires: list[dict[str, Any]]
) -> ComponentIdentityV1:
    """Recompute one component over exactly its detached owned projection.

    Caller JSON objects and arrays must be ordinary built-in ``dict`` and ``list``
    instances. Requirements are normalized metadata and intentionally excluded
    from the component content digest; composite and manifest identities include them.
    """
    name = _namespaced(name, "component name")
    try:
        row = _component_rows()[name]
    except KeyError as exc:
        _fail(f"unknown identity component {name!r}", exc)
    normalized_requires = _normalize_requirements(requires, _COMPONENT_DAG[name])
    if type(owned_projection) is not dict:
        _fail("owned_projection must be an ordinary JSON object")
    if len(owned_projection) > MAX_OWNED_PROJECTION_FIELDS:
        _fail("owned projection exceeds Identity v1 resource limits")
    if any(type(path) is not str for path in owned_projection):
        _fail("owned projection keys must be exact strings")
    normalized_projection = _normalize_strict_json(
        owned_projection, f"owned projection for {name}"
    )
    paths = tuple(sorted(_validate_pointer(path) for path in owned_projection))
    detached = {path: normalized_projection[path] for path in paths}
    wrapper = {
        "domain": row["domain"],
        "schema": COMPONENT_IDENTITY_SCHEMA,
        "content": detached,
    }
    _strict_json(wrapper, f"component {name} digest wrapper")
    digest = hashlib.sha256(
        canonical_json_bytes(wrapper, f"component {name} digest wrapper")
    ).hexdigest()
    # V1 owns exact positive projections; there are no registry-level exclusions.
    excludes: tuple[str, ...] = ()
    result = ComponentIdentityV1(
        name,
        row["domain"],
        COMPONENT_IDENTITY_SCHEMA,
        row["schema_version"],
        digest,
        paths,
        cast(MappingProxyType[str, Any], _freeze(detached)),
        excludes,
        normalized_requires,
    )
    context = f"component {name} identity"
    complete = _normalize_strict_json(
        _component_wire_projection(result),
        context,
        max_nodes=MAX_COMPONENT_IDENTITY_NODES,
        max_payload_bytes=MAX_COMPONENT_IDENTITY_BYTES,
    )
    canonical = canonical_json_bytes(complete, context)
    if len(canonical) > MAX_COMPONENT_IDENTITY_BYTES:
        _fail(f"{context} exceeds Identity v1 resource limits")
    return result


def _validate_component_identity(
    identity: Any, member: str, row: dict[str, Any], composite: str
) -> ComponentIdentityV1:
    if type(identity) is not ComponentIdentityV1:
        _fail(f"composite {composite} member {member!r} is not a component identity")
    candidate = identity
    if (
        type(candidate.name) is not str
        or type(candidate.domain) is not str
        or type(candidate.schema) is not str
        or type(candidate.schema_version) is not int
        or type(candidate.content_digest_sha256) is not str
        or type(candidate.owned_fields) is not tuple
        or type(candidate.owned_projection) not in (dict, MappingProxyType)
        or type(candidate.excludes) is not tuple
        or type(candidate.requires) is not tuple
    ):
        _fail(f"composite {composite} member {member!r} has malformed identity fields")
    _namespaced(candidate.name, "component identity name")
    if len(candidate.owned_fields) > MAX_OWNED_PROJECTION_FIELDS:
        _fail("component owned fields exceed Identity v1 resource limits")
    if len(candidate.excludes) != 0:
        _fail("Identity v1 component exclusions must be empty")
    if len(candidate.requires) != len(_COMPONENT_DAG[member]):
        _fail("component requirements do not match registry cardinality")
    if (
        candidate.name != member
        or candidate.domain != row["domain"]
        or candidate.schema != COMPONENT_IDENTITY_SCHEMA
        or candidate.schema_version != 1
    ):
        _fail(f"composite {composite} member {member!r} has the wrong component identity")
    requirements: list[dict[str, Any]] = []
    for requirement in candidate.requires:
        if (
            type(requirement) is not RequirementV1
            or type(requirement.component) is not str
            or type(requirement.compatible) not in (dict, MappingProxyType)
        ):
            _fail(f"composite {composite} member {member!r} has malformed requirements")
        normalized_compatible = _normalize_strict_json(
            requirement.compatible,
            "component identity requirement",
            allow_frozen=True,
            max_container_items=3,
        )
        requirements.append(
            {
                "component": requirement.component,
                "compatible": normalized_compatible,
            }
        )
    projection = _normalize_strict_json(
        candidate.owned_projection,
        "component identity owned projection",
        allow_frozen=True,
    )
    if type(projection) is not dict:
        _fail(f"composite {composite} member {member!r} has malformed projection")
    recomputed = recompute_component(member, projection, requirements)
    if len(candidate.owned_fields) != len(recomputed.owned_fields):
        _fail(f"composite {composite} member {member!r} has malformed owned fields")
    if any(type(path) is not str for path in candidate.owned_fields):
        _fail(f"composite {composite} member {member!r} has malformed owned fields")
    recomputed_requirements = [
        {"component": item.component, "compatible": _plain(item.compatible)}
        for item in recomputed.requires
    ]
    if (
        candidate.name != recomputed.name
        or candidate.domain != recomputed.domain
        or candidate.schema != recomputed.schema
        or candidate.schema_version != recomputed.schema_version
        or candidate.content_digest_sha256 != recomputed.content_digest_sha256
        or candidate.owned_fields != recomputed.owned_fields
        or projection != _plain(recomputed.owned_projection)
        or candidate.excludes != recomputed.excludes
        or requirements != recomputed_requirements
    ):
        _fail(f"composite {composite} member {member!r} is not a complete valid identity")
    return recomputed


def _validate_self_contained_component(identity: Any) -> ComponentIdentityV1:
    if type(identity) is not ComponentIdentityV1:
        _fail("identity projection requires a complete component identity")
    name = _namespaced(identity.name, "component identity name")
    try:
        row = _component_rows()[name]
    except KeyError as exc:
        _fail("identity projection names an unknown component", exc)
    return _validate_component_identity(identity, name, row, "identity projection")


def _validate_projectable_composite(
    identity: Any,
    resolved_member_identities: _ResolvedMemberIdentities | None,
) -> CompositeIdentityV1:
    if type(identity) is not CompositeIdentityV1:
        _fail("identity projection requires a composite identity")
    candidate = identity
    if (
        type(candidate.name) is not str
        or type(candidate.domain) is not str
        or type(candidate.schema) is not str
        or type(candidate.schema_version) is not int
        or type(candidate.members) is not tuple
        or type(candidate.bundle_digest_sha256) is not str
    ):
        _fail("identity projection requires a complete composite identity")
    name = _namespaced(candidate.name, "composite identity name")
    try:
        row = _composite_rows()[name]
    except KeyError as exc:
        _fail("identity projection names an unknown composite", exc)
    expected_members = tuple(row["members"])
    if len(candidate.members) != len(expected_members):
        _fail("identity projection requires a complete composite identity")
    if any(type(member) is not str for member in candidate.members):
        _fail("identity projection requires a complete composite identity")
    if (
        candidate.domain != row["domain"]
        or candidate.schema != COMPOSITE_IDENTITY_SCHEMA
        or candidate.schema_version != 1
        or candidate.members != expected_members
    ):
        _fail("identity projection requires a complete composite identity")
    _digest(candidate.bundle_digest_sha256, "composite bundle digest")
    if resolved_member_identities is None:
        _fail("composite projection requires its exact resolved leaves")
    recomputed = recompute_bundle(candidate.name, resolved_member_identities)
    if (
        candidate.name != recomputed.name
        or candidate.domain != recomputed.domain
        or candidate.schema != recomputed.schema
        or candidate.schema_version != recomputed.schema_version
        or candidate.members != recomputed.members
        or candidate.bundle_digest_sha256 != recomputed.bundle_digest_sha256
    ):
        _fail("identity projection requires a complete composite identity")
    return recomputed


def identity_to_json(
    value: _IdentityRecord,
    *,
    resolved_member_identities: _ResolvedMemberIdentities | None = None,
) -> dict[str, Any]:
    """Project an identity only after digest recomputation.

    Component records are self-contained and are fully recomputed here. Composite
    records omit their leaf closure on the wire, so their exact eight- or ten-leaf
    resolver is mandatory. That resolver must be an ordinary ``dict`` or an exact
    ``MappingProxyType``. Serialized records are never digest authority.
    """
    if type(value) is ComponentIdentityV1:
        component_candidate = _validate_self_contained_component(value)
        return _component_wire_projection(component_candidate)
    if type(value) is CompositeIdentityV1:
        composite_candidate = _validate_projectable_composite(
            value, resolved_member_identities
        )
        return {
            "domain": composite_candidate.domain,
            "schema": composite_candidate.schema,
            "schema_version": composite_candidate.schema_version,
            "members": list(composite_candidate.members),
            "bundle_digest_sha256": composite_candidate.bundle_digest_sha256,
        }
    _fail("unsupported identity object")


def _compute_bundle(
    name: str,
    identities: dict[str, _IdentityRecord],
) -> CompositeIdentityV1:
    row = _composite_rows()[name]
    component_rows = _component_rows()
    content: list[dict[str, Any]] = []
    for member in row["members"]:
        member_identity = identities[member]
        if member in component_rows:
            component = _validate_component_identity(
                member_identity, member, component_rows[member], name
            )
            projected = {
                "domain": component.domain,
                "schema": component.schema,
                "schema_version": component.schema_version,
                "content_digest_sha256": component.content_digest_sha256,
                "owned_fields": list(component.owned_fields),
                "owned_projection": _plain(component.owned_projection),
                "excludes": list(component.excludes),
                "requires": [
                    {
                        "component": item.component,
                        "compatible": _plain(item.compatible),
                    }
                    for item in component.requires
                ],
            }
        else:
            _fail(f"composite {name} cannot accept caller-supplied composite members")
        content.append({"name": member, "identity": projected})
    wrapper = {
        "domain": row["domain"],
        "schema": COMPOSITE_IDENTITY_SCHEMA,
        "content": {"members": content},
    }
    _strict_json(wrapper, f"composite {name} digest wrapper")
    digest = hashlib.sha256(
        canonical_json_bytes(wrapper, f"composite {name} digest wrapper")
    ).hexdigest()
    return CompositeIdentityV1(
        name,
        row["domain"],
        COMPOSITE_IDENTITY_SCHEMA,
        row["schema_version"],
        tuple(row["members"]),
        digest,
    )


def _compute_experiment_bundle(
    resolved_member_identities: dict[str, _IdentityRecord],
) -> CompositeIdentityV1:
    runtime_members = {
        member: resolved_member_identities[member]
        for member in _COMPOSITE_MEMBERS["runtime_bundle"]
    }
    runtime = _compute_bundle("runtime_bundle", runtime_members)
    component_rows = _component_rows()
    evaluator = _validate_component_identity(
        resolved_member_identities["evaluator_bundle_content"],
        "evaluator_bundle_content",
        component_rows["evaluator_bundle_content"],
        "experiment_bundle",
    )
    analysis = _validate_component_identity(
        resolved_member_identities["analysis_contract_content"],
        "analysis_contract_content",
        component_rows["analysis_contract_content"],
        "experiment_bundle",
    )
    runtime_projection = {
        "domain": runtime.domain,
        "schema": runtime.schema,
        "schema_version": runtime.schema_version,
        "members": list(runtime.members),
        "bundle_digest_sha256": runtime.bundle_digest_sha256,
    }
    content = [
        {"name": "runtime_bundle", "identity": runtime_projection},
        {
            "name": "evaluator_bundle_content",
            "identity": {
                "domain": evaluator.domain,
                "schema": evaluator.schema,
                "schema_version": evaluator.schema_version,
                "content_digest_sha256": evaluator.content_digest_sha256,
                "owned_fields": list(evaluator.owned_fields),
                "owned_projection": _plain(evaluator.owned_projection),
                "excludes": list(evaluator.excludes),
                "requires": [
                    {
                        "component": item.component,
                        "compatible": _plain(item.compatible),
                    }
                    for item in evaluator.requires
                ],
            },
        },
        {
            "name": "analysis_contract_content",
            "identity": {
                "domain": analysis.domain,
                "schema": analysis.schema,
                "schema_version": analysis.schema_version,
                "content_digest_sha256": analysis.content_digest_sha256,
                "owned_fields": list(analysis.owned_fields),
                "owned_projection": _plain(analysis.owned_projection),
                "excludes": list(analysis.excludes),
                "requires": [
                    {
                        "component": item.component,
                        "compatible": _plain(item.compatible),
                    }
                    for item in analysis.requires
                ],
            },
        },
    ]
    row = _composite_rows()["experiment_bundle"]
    wrapper = {
        "domain": row["domain"],
        "schema": COMPOSITE_IDENTITY_SCHEMA,
        "content": {"members": content},
    }
    _strict_json(wrapper, "composite experiment_bundle digest wrapper")
    digest = hashlib.sha256(
        canonical_json_bytes(wrapper, "composite experiment_bundle digest wrapper")
    ).hexdigest()
    return CompositeIdentityV1(
        "experiment_bundle",
        row["domain"],
        COMPOSITE_IDENTITY_SCHEMA,
        row["schema_version"],
        tuple(row["members"]),
        digest,
    )


def _snapshot_resolved_member_identities(
    value: Any, expected_count: int
) -> dict[str, ComponentIdentityV1 | CompositeIdentityV1]:
    """Detach a fixed-width resolver before invoking any identity value hooks."""
    if type(value) is dict:
        if len(value) != expected_count:
            _fail("resolved leaves must match the fixed leaf closure")
        for member in value:
            if type(member) is not str:
                _fail("resolved member identity names must be exact strings")
        return cast(dict[str, ComponentIdentityV1 | CompositeIdentityV1], dict(value))
    if type(value) is not MappingProxyType:
        _fail("resolved member identities must be an ordinary or frozen mapping")

    try:
        pairs = iter(value.items())
    except BaseExceptionGroup:
        raise
    except Exception as exc:
        _fail("resolved member identities could not be snapshotted", exc)
    snapshot: dict[str, ComponentIdentityV1 | CompositeIdentityV1] = {}
    while True:
        try:
            pair = next(pairs)
        except StopIteration:
            break
        except BaseExceptionGroup:
            raise
        except Exception as exc:
            _fail("resolved member identities could not be snapshotted", exc)
        if len(snapshot) == expected_count:
            _fail("resolved leaves must match the fixed leaf closure")
        if type(pair) is not tuple or len(pair) != 2:
            _fail("resolved member identity entries must be exact pairs")
        member, identity = pair
        if type(member) is not str:
            _fail("resolved member identity names must be exact strings")
        if member in snapshot:
            _fail("resolved member identity names must be unique")
        snapshot[member] = identity
    if len(snapshot) != expected_count:
        _fail("resolved leaves must match the fixed leaf closure")
    return snapshot


def recompute_bundle(
    name: str,
    resolved_member_identities: _ResolvedMemberIdentities,
) -> CompositeIdentityV1:
    """Recompute from exact leaves in an ordinary ``dict`` or ``MappingProxyType``."""
    name = _namespaced(name, "bundle name")
    if name not in _COMPOSITE_MEMBERS:
        _fail("unknown identity composite")
    expected = (
        _COMPOSITE_MEMBERS["runtime_bundle"]
        if name == "runtime_bundle"
        else _EXPERIMENT_LEAF_CLOSURE
    )
    snapshot = _snapshot_resolved_member_identities(
        resolved_member_identities, len(expected)
    )
    if any(member not in snapshot for member in expected):
        _fail(f"composite {name} resolved leaves must match the fixed leaf closure")
    if name == "runtime_bundle":
        return _compute_bundle(name, snapshot)
    return _compute_experiment_bundle(snapshot)


def recompute_manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash an ordinary-dict/list JSON tree, excluding its self-digest.

    This computes a digest; it is not full manifest validation.
    """
    if type(manifest) is not dict:
        _fail("identity manifest must be an ordinary object")
    normalized_manifest = _normalize_strict_json(manifest, "identity manifest")
    if type(normalized_manifest) is not dict:
        _fail("identity manifest must be an ordinary object")
    expected = {
        "schema",
        "schema_version",
        "components",
        "composites",
        "compatibility_contract",
        "manifest_digest_sha256",
    }
    _exact_keys(normalized_manifest, expected, "identity manifest")
    if (
        type(normalized_manifest["schema"]) is not str
        or type(normalized_manifest["schema_version"]) is not int
    ):
        _fail("unsupported identity manifest schema or version")
    if (
        normalized_manifest["schema"] != IDENTITY_MANIFEST_SCHEMA
        or normalized_manifest["schema_version"] != 1
    ):
        _fail("unsupported identity manifest schema or version")
    content = {
        key: value
        for key, value in normalized_manifest.items()
        if key != "manifest_digest_sha256"
    }
    wrapper = {
        "domain": MANIFEST_DIGEST_DOMAIN,
        "schema": IDENTITY_MANIFEST_SCHEMA,
        "content": content,
    }
    _strict_json(wrapper, "identity manifest digest wrapper")
    return hashlib.sha256(
        canonical_json_bytes(wrapper, "identity manifest digest wrapper")
    ).hexdigest()


def _precheck_normalized_manifest_validation_inputs(
    manifest: dict[str, Any],
    field_census: dict[str, Any],
    owned_projections: dict[str, Any],
) -> None:
    components = manifest.get("components")
    composites = manifest.get("composites")
    if type(components) is dict and len(components) != len(_COMPONENT_DAG):
        _fail("identity manifest components must match registry cardinality")
    if type(composites) is dict and len(composites) != len(_COMPOSITE_MEMBERS):
        _fail("identity manifest composites must match registry cardinality")
    entries = field_census.get("entries")
    if type(entries) is list and len(entries) > MAX_OWNED_PROJECTION_FIELDS:
        _fail("manifest projection census entries exceed Identity v1 resource limits")
    owners = field_census.get("owners")
    if type(owners) is list and len(owners) != len(_COMPONENT_DAG):
        _fail("manifest projection census owners must match registry cardinality")
    for name in _COMPONENT_DAG:
        projection = owned_projections.get(name)
        if type(projection) is not dict:
            _fail(f"owned projection for {name} must be an ordinary flat object")
        if len(projection) > MAX_OWNED_PROJECTION_FIELDS:
            _fail(f"owned projection for {name} exceeds Identity v1 resource limits")


def _validate_projection_census(census: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    if (
        census.get("schema") == "operatebench.identity_field_census.v1"
        or census.get("artifact_schema") == "operatebench.b1_card_contract.v1"
    ):
        _fail("B1 identity field census has insufficient scope for manifest validation")
    _exact_keys(
        census,
        {"schema", "schema_version", "manifest_schema", "owners", "entries"},
        "manifest projection census",
    )
    if (
        census["schema"] != IDENTITY_MANIFEST_PROJECTION_CENSUS_SCHEMA
        or type(census["schema_version"]) is not int
        or census["schema_version"] != 1
        or census["manifest_schema"] != IDENTITY_MANIFEST_SCHEMA
    ):
        _fail("unsupported manifest projection census schema or version")
    owners = census["owners"]
    if type(owners) is not list or tuple(owners) != tuple(_COMPONENT_DAG):
        _fail("manifest projection census owners must match registry order")
    entries = census["entries"]
    if type(entries) is not list:
        _fail("manifest projection census entries must be an array")
    if len(entries) > MAX_OWNED_PROJECTION_FIELDS:
        _fail("manifest projection census entries exceed Identity v1 resource limits")
    owner_index = {name: index for index, name in enumerate(_COMPONENT_DAG)}
    assigned: dict[str, list[str]] = {name: [] for name in _COMPONENT_DAG}
    seen: set[str] = set()
    ordering: list[tuple[int, str]] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            _fail(f"manifest projection census entry {index} must be an object")
        _exact_keys(entry, {"field_path", "owner"}, f"census entry {index}")
        owner = entry["owner"]
        if type(owner) is not str or owner not in owner_index:
            _fail("manifest projection census entry has unknown owner")
        path = _validate_pointer(entry["field_path"])
        if path in seen:
            _fail("manifest projection census field paths must be globally unique")
        seen.add(path)
        assigned[owner].append(path)
        ordering.append((owner_index[owner], path))
    if ordering != sorted(ordering):
        _fail("manifest projection census entries must be sorted by owner and field path")
    return {name: tuple(paths) for name, paths in assigned.items()}


def _plain_requirement_rows(value: Any, component: str) -> list[dict[str, Any]]:
    if type(value) is not list:
        _fail(f"component {component} is not a complete valid identity")
    return cast(list[dict[str, Any]], value)


def _requirement_is_satisfied(
    requirement: RequirementV1, target: ComponentIdentityV1
) -> None:
    compatible = requirement.compatible
    kind = compatible["kind"]
    if kind == "exact_digest":
        if compatible["content_digest_sha256"] != target.content_digest_sha256:
            _fail("exact requirement digest does not match recomputed target")
    elif kind == "closed_schema_version_range":
        if not (
            compatible["minimum_schema_version"]
            <= target.schema_version
            <= compatible["maximum_schema_version"]
        ):
            _fail("closed schema version range does not contain recomputed target")
    elif kind == "named_predicate":
        _fail("named predicate requirement is not registered in Identity v1")
    else:  # pragma: no cover - recompute_component closes the kind vocabulary
        _fail("unsupported compatibility requirement kind")


def _validate_normalized_identity_manifest(
    manifest: dict[str, Any],
    field_census: dict[str, Any],
    owned_projections: dict[str, Any],
) -> ValidatedIdentityManifest:
    """Validate only bounded, detached, exact-built-in manifest roots."""

    normalized_manifest = manifest
    normalized_census = field_census
    normalized_projections = owned_projections
    _precheck_normalized_manifest_validation_inputs(
        normalized_manifest, normalized_census, normalized_projections
    )

    # The explicit B1 refusal applies only after safe detachment. Hostile lookalikes
    # are rejected by strict-JSON normalization without invoking caller protocols.
    if normalized_census.get("schema") == "operatebench.identity_field_census.v1" or (
        normalized_census.get("artifact_schema") == "operatebench.b1_card_contract.v1"
    ):
        _fail("B1 identity field census has insufficient scope for manifest validation")
    census_paths = _validate_projection_census(normalized_census)
    _exact_keys(
        normalized_manifest,
        {
            "schema",
            "schema_version",
            "components",
            "composites",
            "compatibility_contract",
            "manifest_digest_sha256",
        },
        "identity manifest",
    )
    if (
        normalized_manifest["schema"] != IDENTITY_MANIFEST_SCHEMA
        or type(normalized_manifest["schema_version"]) is not int
        or normalized_manifest["schema_version"] != 1
    ):
        _fail("unsupported identity manifest schema or version")
    components_wire = normalized_manifest["components"]
    composites_wire = normalized_manifest["composites"]
    if type(components_wire) is not dict:
        _fail("identity manifest components must be an ordinary object")
    if type(composites_wire) is not dict:
        _fail("identity manifest composites must be an ordinary object")
    _exact_keys(components_wire, set(_COMPONENT_DAG), "identity manifest components")
    _exact_keys(composites_wire, set(_COMPOSITE_MEMBERS), "identity manifest composites")
    expected_contract = load_compatibility_registry()["contract"]
    if not _exact_json_equal(
        normalized_manifest["compatibility_contract"], expected_contract
    ):
        _fail("identity manifest compatibility contract is not the frozen v1 contract")

    _exact_keys(normalized_projections, set(_COMPONENT_DAG), "owned projections")
    projections: dict[str, dict[str, Any]] = {}
    for name in _COMPONENT_DAG:
        projection = normalized_projections[name]
        if type(projection) is not dict:
            _fail(f"owned projection for {name} must be an ordinary flat object")
        for path in projection:
            _validate_pointer(path)
        if tuple(sorted(projection)) != census_paths[name]:
            _fail(f"projection ownership for {name} does not match census paths")
        serialized = components_wire[name]
        if type(serialized) is not dict:
            _fail(f"component {name} is not a complete valid identity")
        serialized_projection = serialized.get("owned_projection")
        serialized_fields = serialized.get("owned_fields")
        if (
            type(serialized_projection) is not dict
            or type(serialized_fields) is not list
            or not _exact_json_equal(serialized_projection, projection)
            or tuple(serialized_fields) != census_paths[name]
        ):
            _fail(f"component {name} projection ownership is not complete and exact")
        projections[name] = projection

    recomputed_components: dict[str, ComponentIdentityV1] = {}
    component_wire_keys = {
        "domain",
        "schema",
        "schema_version",
        "content_digest_sha256",
        "owned_fields",
        "owned_projection",
        "excludes",
        "requires",
    }
    for name in _COMPONENT_DAG:
        serialized = cast(dict[str, Any], components_wire[name])
        _exact_keys(serialized, component_wire_keys, f"component {name}")
        recomputed = recompute_component(
            name, projections[name], _plain_requirement_rows(serialized["requires"], name)
        )
        if not _exact_json_equal(serialized, _component_wire_projection(recomputed)):
            _fail(f"component {name} is not a complete valid identity")
        recomputed_components[name] = recomputed

    for name in _COMPONENT_DAG:
        for requirement in recomputed_components[name].requires:
            _requirement_is_satisfied(
                requirement, recomputed_components[requirement.component]
            )

    runtime_leaves = {
        name: recomputed_components[name] for name in _COMPOSITE_MEMBERS["runtime_bundle"]
    }
    experiment_leaves = {
        name: recomputed_components[name] for name in _EXPERIMENT_LEAF_CLOSURE
    }
    runtime = recompute_bundle("runtime_bundle", runtime_leaves)
    experiment = recompute_bundle("experiment_bundle", experiment_leaves)
    recomputed_composites = {
        "runtime_bundle": runtime,
        "experiment_bundle": experiment,
    }
    for name, leaves in (
        ("runtime_bundle", runtime_leaves),
        ("experiment_bundle", experiment_leaves),
    ):
        serialized = composites_wire[name]
        if type(serialized) is not dict:
            _fail(f"composite {name} is not a complete valid identity")
        expected_wire = identity_to_json(
            recomputed_composites[name], resolved_member_identities=leaves
        )
        if not _exact_json_equal(serialized, expected_wire):
            _fail(f"composite {name} is not a complete valid identity")

    complete_wire = {
        "schema": IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: _component_wire_projection(recomputed_components[name])
            for name in _COMPONENT_DAG
        },
        "composites": {
            "runtime_bundle": identity_to_json(
                runtime, resolved_member_identities=runtime_leaves
            ),
            "experiment_bundle": identity_to_json(
                experiment, resolved_member_identities=experiment_leaves
            ),
        },
        "compatibility_contract": expected_contract,
        "manifest_digest_sha256": "0" * 64,
    }
    digest = recompute_manifest_digest(complete_wire)
    if normalized_manifest["manifest_digest_sha256"] != digest:
        _fail("identity manifest digest does not match fresh recomputation")
    return ValidatedIdentityManifest(
        IDENTITY_MANIFEST_SCHEMA,
        1,
        MappingProxyType(dict(recomputed_components)),
        MappingProxyType(dict(recomputed_composites)),
        cast(MappingProxyType[str, Any], _freeze(expected_contract)),
        digest,
    )


def validate_identity_manifest(
    manifest: dict[str, Any],
    field_census: dict[str, Any],
    owned_projections: dict[str, dict[str, Any]],
) -> ValidatedIdentityManifest:
    """Validate detached roots without retaining raw caller trees on failure."""

    normalized_manifest: dict[str, Any] | None = None
    normalized_census: dict[str, Any] | None = None
    normalized_projections: dict[str, Any] | None = None
    try:
        # Keep exact-root checks inline: a helper accepting all three raw
        # arguments would preserve every unvisited root in its traceback frame.
        if type(manifest) is not dict:
            _fail("identity manifest must be an ordinary object")
        if type(field_census) is not dict:
            _fail("manifest projection census must be an ordinary object")
        if type(owned_projections) is not dict:
            _fail("owned projections must be an ordinary object")
        if len(manifest) != 6:
            _fail("identity manifest fields do not match the closed v1 envelope")
        if len(field_census) not in (4, 5):
            _fail("manifest projection census fields do not match a supported envelope")
        if len(owned_projections) != len(_COMPONENT_DAG):
            _fail("owned projections must contain exactly the registered components")

        normalized_manifest = cast(
            dict[str, Any], _normalize_strict_json(manifest, "identity manifest")
        )
        manifest = {}
        normalized_census = cast(
            dict[str, Any],
            _normalize_strict_json(field_census, "manifest projection census"),
        )
        field_census = {}
        normalized_projections = cast(
            dict[str, Any],
            _normalize_strict_json(owned_projections, "owned projections"),
        )
        owned_projections = {}
        return _validate_normalized_identity_manifest(
            normalized_manifest, normalized_census, normalized_projections
        )
    finally:
        # Function parameters are frame locals. Rebinding every raw alias here is
        # what makes the public exception graph safe; ``from None`` would not.
        manifest = {}
        field_census = {}
        owned_projections = {}


def _snapshot_diff_mapping(
    value: Any, expected: tuple[str, ...], context: str
) -> dict[str, Any]:
    """Snapshot an exact frozen fixed-width mapping with bounded hostile traversal."""

    if type(value) is not MappingProxyType:
        _fail(f"{context} must be an exact frozen mapping")
    pairs: Any = None
    pair: Any = None
    key: Any = None
    item: Any = None
    snapshot: dict[str, Any] = {}
    try:
        try:
            pairs = iter(value.items())
        except BaseExceptionGroup:
            raise
        except Exception as exc:
            _fail(f"{context} could not be snapshotted", exc)
        while True:
            try:
                pair = next(pairs)
            except StopIteration:
                break
            except BaseExceptionGroup:
                raise
            except Exception as exc:
                _fail(f"{context} could not be snapshotted", exc)
            if len(snapshot) == len(expected):
                _fail(f"{context} must match frozen registry cardinality")
            if type(pair) is not tuple or len(pair) != 2:
                _fail(f"{context} entries must be exact pairs")
            key, item = pair
            if type(key) is not str:
                _fail(f"{context} names must be exact strings")
            if key in snapshot:
                _fail(f"{context} names must be unique")
            snapshot[key] = item
            pair = None
            key = None
            item = None
        if tuple(snapshot) != expected:
            _fail(f"{context} must match frozen registry order")
        return snapshot
    finally:
        value = None
        pairs = None
        pair = None
        key = None
        item = None


def _revalidate_diff_manifest(
    value: Any,
) -> tuple[
    dict[str, ComponentIdentityV1],
    dict[str, CompositeIdentityV1],
    dict[str, dict[str, Any]],
    dict[str, tuple[tuple[str, dict[str, Any]], ...]],
]:
    """Rebuild a forgeable validated-result record without a retained census."""

    if type(value) is not ValidatedIdentityManifest:
        _fail("identity diff inputs must be exact validated manifests")
    if (
        type(value.schema) is not str
        or value.schema != IDENTITY_MANIFEST_SCHEMA
        or type(value.schema_version) is not int
        or value.schema_version != 1
        or type(value.manifest_digest_sha256) is not str
    ):
        _fail("identity diff manifest envelope is malformed")
    raw_components = _snapshot_diff_mapping(
        value.components, tuple(_COMPONENT_DAG), "identity diff components"
    )
    raw_composites = _snapshot_diff_mapping(
        value.composites, tuple(_COMPOSITE_MEMBERS), "identity diff composites"
    )
    compatibility = cast(
        dict[str, Any],
        _normalize_strict_json(
            value.compatibility_contract,
            "identity diff compatibility contract",
            allow_frozen=True,
        ),
    )
    expected_contract = load_compatibility_registry()["contract"]
    if not _exact_json_equal(compatibility, expected_contract):
        _fail("identity diff compatibility contract is not the frozen v1 contract")

    rows = _component_rows()
    components: dict[str, ComponentIdentityV1] = {}
    projections: dict[str, dict[str, Any]] = {}
    requirements: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {}
    owners: dict[str, str] = {}
    for name in _COMPONENT_DAG:
        recomputed = _validate_component_identity(
            raw_components[name], name, rows[name], "identity diff"
        )
        projection = cast(
            dict[str, Any],
            _normalize_strict_json(
                recomputed.owned_projection,
                "identity diff owned projection",
                allow_frozen=True,
            ),
        )
        for path in recomputed.owned_fields:
            if path in owners:
                _fail("identity diff projection paths must have unique ownership")
            owners[path] = name
        requirement_rows = tuple(
            (
                requirement.component,
                cast(
                    dict[str, Any],
                    _normalize_strict_json(
                        requirement.compatible,
                        "identity diff component requirement",
                        allow_frozen=True,
                        max_container_items=3,
                    ),
                ),
            )
            for requirement in recomputed.requires
        )
        components[name] = recomputed
        projections[name] = projection
        requirements[name] = requirement_rows
    for name in _COMPONENT_DAG:
        for requirement in components[name].requires:
            _requirement_is_satisfied(requirement, components[requirement.component])

    runtime_leaves = {
        name: components[name] for name in _COMPOSITE_MEMBERS["runtime_bundle"]
    }
    experiment_leaves = {name: components[name] for name in _EXPERIMENT_LEAF_CLOSURE}
    recomputed_composites = {
        "runtime_bundle": recompute_bundle("runtime_bundle", runtime_leaves),
        "experiment_bundle": recompute_bundle("experiment_bundle", experiment_leaves),
    }
    composite_rows = _composite_rows()
    for name in _COMPOSITE_MEMBERS:
        candidate = raw_composites[name]
        if type(candidate) is not CompositeIdentityV1:
            _fail("identity diff composite record is malformed")
        expected = recomputed_composites[name]
        row = composite_rows[name]
        if (
            type(candidate.name) is not str
            or type(candidate.domain) is not str
            or type(candidate.schema) is not str
            or type(candidate.schema_version) is not int
            or type(candidate.members) is not tuple
            or len(candidate.members) != len(row["members"])
            or any(type(member) is not str for member in candidate.members)
            or type(candidate.bundle_digest_sha256) is not str
            or candidate.name != name
            or candidate.domain != row["domain"]
            or candidate.schema != COMPOSITE_IDENTITY_SCHEMA
            or candidate.schema_version != 1
            or candidate.members != tuple(row["members"])
            or candidate.bundle_digest_sha256 != expected.bundle_digest_sha256
        ):
            _fail("identity diff composite record is not freshly recomputed")

    complete_wire = {
        "schema": IDENTITY_MANIFEST_SCHEMA,
        "schema_version": 1,
        "components": {
            name: _component_wire_projection(components[name]) for name in _COMPONENT_DAG
        },
        "composites": {
            "runtime_bundle": identity_to_json(
                recomputed_composites["runtime_bundle"],
                resolved_member_identities=runtime_leaves,
            ),
            "experiment_bundle": identity_to_json(
                recomputed_composites["experiment_bundle"],
                resolved_member_identities=experiment_leaves,
            ),
        },
        "compatibility_contract": expected_contract,
        "manifest_digest_sha256": "0" * 64,
    }
    fresh_digest = recompute_manifest_digest(complete_wire)
    if value.manifest_digest_sha256 != fresh_digest:
        _fail("identity diff manifest digest is not freshly recomputed")
    return components, recomputed_composites, projections, requirements


def identity_diff(
    old: ValidatedIdentityManifest, new: ValidatedIdentityManifest
) -> tuple[OwnedMutation, ...]:
    """Return deterministic component-owned changes between exact valid manifests."""

    old_state: Any = None
    new_state: Any = None
    failure_message: str | None = None
    try:
        old_state = _revalidate_diff_manifest(old)
        old = cast(ValidatedIdentityManifest, None)
        new_state = _revalidate_diff_manifest(new)
        new = cast(ValidatedIdentityManifest, None)
        old_components, old_composites, old_projections, old_requirements = old_state
        new_components, new_composites, new_projections, new_requirements = new_state
        old_owners = {
            path: owner for owner in _COMPONENT_DAG for path in old_projections[owner]
        }
        new_owners = {
            path: owner for owner in _COMPONENT_DAG for path in new_projections[owner]
        }
        if any(
            path in new_owners and new_owners[path] != owner
            for path, owner in old_owners.items()
        ):
            _fail("identity diff refuses projection path owner reassignment")
        composite_changes = {
            name
            for name in _COMPOSITE_MEMBERS
            if old_composites[name].bundle_digest_sha256
            != new_composites[name].bundle_digest_sha256
        }
        result: list[OwnedMutation] = []
        for owner in _COMPONENT_DAG:
            old_projection = old_projections[owner]
            new_projection = new_projections[owner]
            changed_paths = tuple(
                path
                for path in sorted(set(old_projection) | set(new_projection))
                if path not in old_projection
                or path not in new_projection
                or not _exact_json_equal(old_projection[path], new_projection[path])
            )
            old_requirement_rows = old_requirements[owner]
            new_requirement_rows = new_requirements[owner]
            changed_requirements = tuple(
                old_name
                for (old_name, old_compatible), (new_name, new_compatible) in zip(
                    old_requirement_rows, new_requirement_rows, strict=True
                )
                if old_name != new_name
                or not _exact_json_equal(old_compatible, new_compatible)
            )
            if not changed_paths and not changed_requirements:
                continue
            if owner == "card_schema_content":
                closure: tuple[str, ...] = ()
            elif owner in ("evaluator_bundle_content", "analysis_contract_content"):
                closure = ("experiment_bundle",)
            else:
                closure = ("runtime_bundle", "experiment_bundle")
            result.append(
                OwnedMutation(
                    OWNED_MUTATION_CONTRACT_VERSION,
                    owner,
                    changed_paths,
                    changed_requirements,
                    old_components[owner].content_digest_sha256,
                    new_components[owner].content_digest_sha256,
                    tuple(name for name in closure if name in composite_changes),
                )
            )
        return tuple(result)
    except IdentityValidationError as exc:
        failure_message = (
            exc.args[0]
            if type(exc) is IdentityValidationError
            and len(exc.args) == 1
            and type(exc.args[0]) is str
            else "identity diff rejected malformed input"
        )
        _strip_exception_tracebacks(exc)
    except BaseException as exc:
        _strip_exception_tracebacks(exc)
        raise
    finally:
        old = cast(ValidatedIdentityManifest, None)
        new = cast(ValidatedIdentityManifest, None)
        old_state = None
        new_state = None
    assert failure_message is not None
    raise IdentityValidationError(failure_message) from None


__all__ = [
    "COMPATIBILITY_CONTRACT_SCHEMA",
    "COMPONENT_IDENTITY_SCHEMA",
    "COMPOSITE_IDENTITY_SCHEMA",
    "IDENTITY_MANIFEST_PROJECTION_CENSUS_SCHEMA",
    "IDENTITY_MANIFEST_SCHEMA",
    "IDENTITY_SCHEMA_VERSION",
    "MANIFEST_DIGEST_DOMAIN",
    "OWNED_MUTATION_CONTRACT_VERSION",
    "ComponentIdentityV1",
    "CompositeIdentityV1",
    "IdentityValidationError",
    "OwnedMutation",
    "RequirementEvaluationV1",
    "RequirementV1",
    "ValidatedIdentityManifest",
    "evaluate_component_requirements",
    "identity_diff",
    "identity_to_json",
    "load_compatibility_registry",
    "load_component_registry",
    "load_identity_golden_vectors",
    "load_identity_manifest_projection_census_schema",
    "load_identity_schema",
    "recompute_bundle",
    "recompute_component",
    "recompute_manifest_digest",
    "validate_identity_manifest",
]
