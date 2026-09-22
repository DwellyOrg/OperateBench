"""Offline Card/Census v1 and v2 loading, validation, and identity APIs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Never, cast

from operatebench.jsonsafe import (
    MAX_JSON_DEPTH,
    JsonSafetyError,
    canonical_json_bytes,
    ensure_json_safe,
    ensure_raw_json_depth,
)
from operatebench.version import (
    CARD_SCHEMA_VERSION,
    OPERATION_CARD_V1_SCHEMA_VERSION,
    OPERATION_CARD_V2_SCHEMA_VERSION,
    SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION,
    SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION,
    VARIANT_CARD_V1_SCHEMA_VERSION,
    VARIANT_CARD_V2_SCHEMA_VERSION,
)

OPERATION_CARD_SCHEMA: Final = "operatebench.operation_card.v1"
SEMANTIC_SCENARIO_CARD_SCHEMA: Final = "operatebench.semantic_scenario_card.v1"
VARIANT_CARD_SCHEMA: Final = "operatebench.variant_card.v1"
CARD_SCHEMAS: Final = (
    OPERATION_CARD_SCHEMA,
    SEMANTIC_SCENARIO_CARD_SCHEMA,
    VARIANT_CARD_SCHEMA,
)
OPERATION_CARD_V2_SCHEMA: Final = "operatebench.operation_card.v2"
SEMANTIC_SCENARIO_CARD_V2_SCHEMA: Final = "operatebench.semantic_scenario_card.v2"
VARIANT_CARD_V2_SCHEMA: Final = "operatebench.variant_card.v2"
SUPPORTED_CARD_SCHEMA_VERSIONS: Final = MappingProxyType(
    {
        OPERATION_CARD_SCHEMA: OPERATION_CARD_V1_SCHEMA_VERSION,
        SEMANTIC_SCENARIO_CARD_SCHEMA: SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION,
        VARIANT_CARD_SCHEMA: VARIANT_CARD_V1_SCHEMA_VERSION,
        OPERATION_CARD_V2_SCHEMA: OPERATION_CARD_V2_SCHEMA_VERSION,
        SEMANTIC_SCENARIO_CARD_V2_SCHEMA: SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION,
        VARIANT_CARD_V2_SCHEMA: VARIANT_CARD_V2_SCHEMA_VERSION,
    }
)
_RESOURCE_PACKAGE = "operatebench.resources.cards"
_CENSUS_SCHEMA_FILE = "identity-field-census-v1.schema.json"
_SCHEMA_FILES = {
    OPERATION_CARD_SCHEMA: "operation-card-v1.schema.json",
    SEMANTIC_SCENARIO_CARD_SCHEMA: "semantic-scenario-card-v1.schema.json",
    VARIANT_CARD_SCHEMA: "variant-card-v1.schema.json",
    OPERATION_CARD_V2_SCHEMA: "operation-card-v2.schema.json",
    SEMANTIC_SCENARIO_CARD_V2_SCHEMA: "semantic-scenario-card-v2.schema.json",
    VARIANT_CARD_V2_SCHEMA: "variant-card-v2.schema.json",
}
_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*(?:@[0-9]+\.[0-9]+\.[0-9]+)?$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_CENSUS_OWNERS = (
    "card_schema_content",
    "operation_core_content",
    "semantic_scenario_content",
    "variant_content",
)
_CENSUS_ATOMIC_SCHEMA_DEFS = frozenset(
    {"fieldShapeMap", "jsonSchemaSubset", "predicateAst"}
)
_CENSUS_CARD_ROOTS = {
    OPERATION_CARD_SCHEMA: ("operation_card", "operation_core_content"),
    SEMANTIC_SCENARIO_CARD_SCHEMA: (
        "semantic_scenario_card",
        "semantic_scenario_content",
    ),
    VARIANT_CARD_SCHEMA: ("variant_card", "variant_content"),
}
_CENSUS_OWNED_OBJECT_PATHS = frozenset({"/content/policy_contract"})
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SAFE_PATTERN_MAX_CHARS = 256
_SAFE_PATTERN_MAX_BYTES = 256
_SAFE_PATTERN_INPUT_MAX_CHARS = 4096
_SAFE_PATTERN_MAX_REPEAT = 4096
_CONCRETE_PATH_TYPE = type(Path())
_MAX_PROJECTED_NODES = 100_000
_MAX_CONTAINER_MEMBERS = 4_096
_MAX_STRING_UTF8_BYTES = 1_048_576
_MAX_AGGREGATE_STRING_UTF8_BYTES = 8_388_608
_MAX_CANONICAL_CARD_BYTES = 8_388_608
_DECLARED_UNTRUSTED_PROTOCOL_ERRORS = (
    AssertionError,
    OSError,
    RecursionError,
    TypeError,
    UnicodeError,
    ValueError,
)


class CardValidationError(ValueError):
    """A Card/Census v1 value fails its closed offline contract."""


class _UnsafeProjectionError(CardValidationError):
    """A projection error whose hidden graph may still contain caller objects."""


class _DetachedProjectionError(CardValidationError):
    """A bounded projection error whose graph is caller-object-free."""


def _detach_projection_error(error: CardValidationError) -> tuple[str, str | None]:
    """Copy bounded diagnostics without retaining the unsafe exception graph."""
    message = (
        error.args[0]
        if len(error.args) == 1 and type(error.args[0]) is str
        else "malformed in-memory Card/Census value"
    )
    if len(message) > _MAX_STRING_UTF8_BYTES:
        message = "malformed in-memory Card/Census value"
    cause = error.__cause__
    safe_detail = (
        cause.args[0]
        if type(cause) is JsonSafetyError
        and cause.__traceback__ is None
        and len(cause.args) == 1
        and type(cause.args[0]) is str
        else None
    )
    return message, safe_detail


def _raise_detached_projection_error(message: str, detail: str | None) -> Never:
    if detail is None:
        raise _DetachedProjectionError(message)
    raise _DetachedProjectionError(message) from JsonSafetyError(detail)


def _detach_domain_error(error: CardValidationError, fallback: str) -> str:
    message = (
        error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else fallback
    )
    if len(message) > _MAX_STRING_UTF8_BYTES:
        message = fallback
    return message


def _raise_domain_error(message: str, cause: BaseException | None) -> Never:
    if cause is None:
        raise CardValidationError(message)
    raise CardValidationError(message) from cause


def _duplicates_refused(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CardValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CardValidationError(f"{context} is not UTF-8") from exc
    try:
        ensure_raw_json_depth(text, context)
        return json.loads(
            text,
            object_pairs_hook=_duplicates_refused,
            parse_float=lambda value: (_ for _ in ()).throw(
                CardValidationError(f"{context} contains forbidden float {value}")
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                CardValidationError(f"{context} contains forbidden constant {value}")
            ),
        )
    except CardValidationError:
        raise
    except (JsonSafetyError, json.JSONDecodeError) as exc:
        raise CardValidationError(f"invalid {context}") from exc


def _read_resource_bytes(name: str) -> bytes:
    try:
        return files(_RESOURCE_PACKAGE).joinpath(name).read_bytes()
    except OSError as exc:
        raise CardValidationError(f"cannot read packaged resource {name!r}") from exc


@lru_cache(maxsize=None)  # noqa: UP033 - explicit bounded API supports cache_clear
def _cached_resource_json(name: str) -> Mapping[str, Any]:
    value = _load_json_bytes(_read_resource_bytes(name), name)
    if not isinstance(value, dict):
        raise CardValidationError(f"packaged resource {name} is not an object")
    return cast(Mapping[str, Any], _freeze(value))


def _clear_resource_caches() -> None:
    """Clear private package-resource caches (primarily for isolated tests)."""
    _cached_resource_json.cache_clear()


def _resource_json(name: str) -> dict[str, Any]:
    return cast(dict[str, Any], card_to_json(_cached_resource_json(name)))


def load_card_registry() -> dict[str, Any]:
    """Return a detached plain-JSON projection of the packaged Card v1 registry."""
    return _resource_json("card-registry-v1.json")


def load_card_registry_v2() -> dict[str, Any]:
    """Return a detached plain-JSON projection of the additive Card v2 registry."""
    return _resource_json("card-registry-v2.json")


def load_schema(schema: str) -> dict[str, Any]:
    """Load one exact Card v1 schema without permitting network resolution."""
    if type(schema) is not str:
        error = TypeError("schema name must be an exact str")
        raise CardValidationError("malformed card schema name") from error
    try:
        name = _SCHEMA_FILES[schema]
    except KeyError as exc:
        raise CardValidationError(f"unsupported card schema {schema!r}") from exc
    return _resource_json(name)


def load_field_census() -> dict[str, Any]:
    """Load the complete B1-owned surface census (not an Artifact 9 census)."""
    return _resource_json("identity-field-census-v1.registry.json")


def load_field_census_v2() -> dict[str, Any]:
    """Load the independently frozen additive Card v2 field census."""
    return _resource_json("identity-field-census-v2.registry.json")


def _strict_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CardValidationError(f"{path} contains a lone surrogate") from exc
        return
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise CardValidationError(f"{path} integer is outside signed 64-bit range")
        return
    if isinstance(value, float):
        raise CardValidationError(f"{path} contains a forbidden float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CardValidationError(f"{path} has a non-string object key")
            _strict_json(item, f"{path}/{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, f"{path}/{index}")
        return
    raise CardValidationError(f"{path} has non-JSON value {type(value).__name__}")


def _resolve_ref(
    ref: str, *, defs_file: str = "card-defs-v1.schema.json"
) -> Mapping[str, Any]:
    external_prefixes = (
        "card-defs-v1.schema.json#/$defs/",
        "card-defs-v2.schema.json#/$defs/",
    )
    local_prefix = "#/$defs/"
    matched_file = next(
        (candidate for candidate in external_prefixes if ref.startswith(candidate)), None
    )
    if matched_file is not None:
        name = ref[len(matched_file) :]
        resolved_defs_file = matched_file.split("#", 1)[0]
    elif ref.startswith(local_prefix):
        name = ref[len(local_prefix) :]
        resolved_defs_file = defs_file
    else:
        raise CardValidationError(f"non-offline or unsupported $ref {ref!r}")
    if "/" in name or resolved_defs_file not in {
        "card-defs-v1.schema.json",
        "card-defs-v2.schema.json",
    }:
        raise CardValidationError(f"non-offline or unsupported $ref {ref!r}")
    defs = _cached_resource_json(resolved_defs_file).get("$defs")
    if not isinstance(defs, Mapping):
        raise CardValidationError(
            f"packaged {resolved_defs_file} is missing a valid $defs object"
        )
    target = defs.get(name)
    if not isinstance(target, Mapping):
        raise CardValidationError(f"unresolved offline $ref {ref!r}")
    return target


def _equal_json(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _safe_pattern_error(path: str, detail: str) -> CardValidationError:
    return CardValidationError(f"{path} is not a safe regex: {detail}")


def _validate_safe_pattern(pattern: str, path: str) -> None:
    """Validate the Card v1 linear-time regex subset without executing it.

    Card-authored patterns use printable ASCII and are capped concatenations of
    literal atoms, approved escaped literals or ``dDsSwW`` categories, character
    classes/ranges, and boundary anchors. An atom may carry one simple or bounded
    quantifier. All
    grouping, alternation, wildcards, advanced escapes, and ambiguous repetition
    constructs fail closed before matching; allowing at most one variable-width
    quantifier prevents cross-product backtracking between repeated atoms.
    """
    if any(not " " <= character <= "~" for character in pattern):
        raise _safe_pattern_error(
            path, "pattern contains a character outside printable ASCII U+0020..U+007E"
        )
    if len(pattern) > _SAFE_PATTERN_MAX_CHARS:
        raise _safe_pattern_error(path, "pattern exceeds 256 characters")
    try:
        encoded = pattern.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _safe_pattern_error(path, "pattern is not UTF-8") from exc
    if len(encoded) > _SAFE_PATTERN_MAX_BYTES:
        raise _safe_pattern_error(path, "pattern exceeds 256 UTF-8 bytes")

    def consume_escape(index: int, *, in_class: bool) -> int:
        if index + 1 >= len(pattern):
            raise _safe_pattern_error(path, "trailing escape")
        escaped = pattern[index + 1]
        if escaped.isdigit():
            raise _safe_pattern_error(path, "backreferences are forbidden")
        if escaped.isalnum() and escaped not in "dDsSwW":
            detail = "advanced class escape" if in_class else "advanced escape"
            raise _safe_pattern_error(path, f"{detail} is forbidden")
        return index + 2

    def consume_class(index: int) -> int:
        index += 1
        if index < len(pattern) and pattern[index] == "^":
            index += 1
        class_has_atom = False
        range_pending = False
        while index < len(pattern):
            character = pattern[index]
            if character == "]":
                if not class_has_atom or range_pending:
                    raise _safe_pattern_error(path, "empty or malformed character class")
                return index + 1
            if character == "[":
                raise _safe_pattern_error(path, "nested character class is forbidden")
            if character == "\\":
                index = consume_escape(index, in_class=True)
                class_has_atom = True
                range_pending = False
                continue
            if character == "-":
                if range_pending:
                    raise _safe_pattern_error(path, "malformed character-class range")
                if (
                    class_has_atom
                    and index + 1 < len(pattern)
                    and pattern[index + 1] != "]"
                ):
                    range_pending = True
                    index += 1
                    continue
            elif range_pending:
                range_pending = False
            class_has_atom = True
            index += 1
        raise _safe_pattern_error(path, "unterminated character class")

    def consume_quantifier(index: int) -> tuple[int, bool]:
        if pattern[index] in "*+?":
            return index + 1, True
        end = pattern.find("}", index + 1)
        if end < 0:
            raise _safe_pattern_error(path, "unterminated bounded quantifier")
        parts = pattern[index + 1 : end].split(",")
        if (
            len(parts) > 2
            or not parts[0].isdigit()
            or (len(parts) == 2 and parts[1] and not parts[1].isdigit())
        ):
            raise _safe_pattern_error(path, "invalid bounded quantifier")
        minimum = int(parts[0])
        maximum = (
            minimum
            if len(parts) == 1
            else int(parts[1])
            if parts[1]
            else _SAFE_PATTERN_MAX_REPEAT
        )
        if maximum < minimum or maximum > _SAFE_PATTERN_MAX_REPEAT:
            raise _safe_pattern_error(path, "bounded quantifier exceeds safe limit")
        return end + 1, minimum != maximum

    index = 0
    previous_atom_quantified = False
    has_variable_quantifier = False
    while index < len(pattern):
        character = pattern[index]
        if character == "^":
            if index != 0:
                raise _safe_pattern_error(path, "start anchor is only allowed first")
            index += 1
            continue
        if character == "$":
            if index != len(pattern) - 1:
                raise _safe_pattern_error(path, "end anchor is only allowed last")
            index += 1
            continue
        if character == "[":
            index = consume_class(index)
        elif character == "\\":
            index = consume_escape(index, in_class=False)
        elif character in "()|.*+?{}]":
            raise _safe_pattern_error(path, f"unsupported metacharacter {character!r}")
        else:
            index += 1

        quantified = index < len(pattern) and pattern[index] in "*+?{"
        if quantified:
            if previous_atom_quantified:
                raise _safe_pattern_error(path, "adjacent quantified atoms are forbidden")
            index, variable_width = consume_quantifier(index)
            if variable_width and has_variable_quantifier:
                raise _safe_pattern_error(
                    path, "multiple variable-width quantifiers are forbidden"
                )
            has_variable_quantifier = has_variable_quantifier or variable_width
            if index < len(pattern) and pattern[index] in "*+?{":
                raise _safe_pattern_error(path, "repeated quantifier is forbidden")
        previous_atom_quantified = quantified

    try:
        re.compile(pattern)
    except re.error as exc:
        raise _safe_pattern_error(path, "pattern does not compile") from exc


def _pattern_fullmatch(
    pattern: str, value: str, path: str, *, trusted_pattern: bool
) -> bool:
    """Use the sole regex match path; card-authored expressions always pass the gate."""
    if not trusted_pattern:
        _validate_safe_pattern(pattern, path)
        if len(value) > _SAFE_PATTERN_INPUT_MAX_CHARS:
            raise CardValidationError(f"{path} regex input is too long")
    try:
        return re.fullmatch(pattern, value) is not None
    except re.error as exc:
        raise CardValidationError(f"{path} regex does not compile") from exc


def _validate_schema(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    *,
    trusted_patterns: bool = True,
    enforce_canonical_sets: bool = True,
    defs_file: str = "card-defs-v1.schema.json",
) -> None:
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str):
            raise CardValidationError(f"malformed $ref at {path}")
        next_defs_file = ref.split("#", 1)[0] if not ref.startswith("#") else defs_file
        _validate_schema(
            value,
            _resolve_ref(ref, defs_file=defs_file),
            path,
            trusted_patterns=trusted_patterns,
            enforce_canonical_sets=enforce_canonical_sets,
            defs_file=next_defs_file,
        )
    for branch in schema.get("allOf", ()):
        _validate_schema(
            value,
            branch,
            path,
            trusted_patterns=trusted_patterns,
            enforce_canonical_sets=enforce_canonical_sets,
            defs_file=defs_file,
        )
    if "oneOf" in schema:
        matches = 0
        for branch in schema["oneOf"]:
            try:
                _validate_schema(
                    value,
                    branch,
                    path,
                    trusted_patterns=trusted_patterns,
                    enforce_canonical_sets=enforce_canonical_sets,
                    defs_file=defs_file,
                )
            except CardValidationError:
                continue
            matches += 1
        if matches != 1:
            raise CardValidationError(f"{path} must match exactly one allowed shape")
    if "const" in schema and not _equal_json(value, schema["const"]):
        raise CardValidationError(f"{path} must equal {schema['const']!r}")
    enum_values = schema.get("enum")
    if (
        isinstance(enum_values, Sequence)
        and not isinstance(enum_values, (str, bytes, bytearray))
        and not any(_equal_json(value, item) for item in enum_values)
    ):
        raise CardValidationError(f"{path} is not in the closed enum")
    expected = schema.get("type")
    type_checks = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    type_ok = type_checks.get(expected, True) if isinstance(expected, str) else True
    if not type_ok:
        raise CardValidationError(f"{path} must be {expected}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise CardValidationError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise CardValidationError(f"{path} is too long")
        if "pattern" in schema and not _pattern_fullmatch(
            schema["pattern"],
            value,
            path,
            trusted_pattern=trusted_patterns,
        ):
            raise CardValidationError(f"{path} has invalid lexical form")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise CardValidationError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise CardValidationError(f"{path} is above its maximum")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise CardValidationError(f"{path} is missing required keys {missing}")
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        property_names = schema.get("propertyNames")
        for key, item in value.items():
            if property_names is not None:
                _validate_schema(
                    key,
                    property_names,
                    f"{path}/<key>",
                    trusted_patterns=trusted_patterns,
                    enforce_canonical_sets=enforce_canonical_sets,
                    defs_file=defs_file,
                )
            if key in properties:
                _validate_schema(
                    item,
                    properties[key],
                    f"{path}/{key}",
                    trusted_patterns=trusted_patterns,
                    enforce_canonical_sets=enforce_canonical_sets,
                    defs_file=defs_file,
                )
                continue
            matching = [
                branch
                for pattern, branch in patterns.items()
                if _pattern_fullmatch(
                    pattern,
                    key,
                    f"{path}/<key>",
                    trusted_pattern=trusted_patterns,
                )
            ]
            if matching:
                for branch in matching:
                    _validate_schema(
                        item,
                        branch,
                        f"{path}/{key}",
                        trusted_patterns=trusted_patterns,
                        enforce_canonical_sets=enforce_canonical_sets,
                        defs_file=defs_file,
                    )
                continue
            if schema.get("additionalProperties") is False:
                raise CardValidationError(f"{path} contains unknown key {key!r}")
            additional = schema.get("additionalProperties")
            if isinstance(additional, Mapping):
                _validate_schema(
                    item,
                    additional,
                    f"{path}/{key}",
                    trusted_patterns=trusted_patterns,
                    enforce_canonical_sets=enforce_canonical_sets,
                    defs_file=defs_file,
                )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise CardValidationError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise CardValidationError(f"{path} has too many items")
        if schema.get("uniqueItems"):
            encoded = [canonical_json_bytes(item, path) for item in value]
            if len(set(encoded)) != len(encoded):
                raise CardValidationError(f"{path} contains duplicate set items")
            if enforce_canonical_sets and encoded != sorted(encoded):
                raise CardValidationError(f"{path} set items are not canonically sorted")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema(
                    item,
                    item_schema,
                    f"{path}/{index}",
                    trusted_patterns=trusted_patterns,
                    enforce_canonical_sets=enforce_canonical_sets,
                    defs_file=defs_file,
                )


def recompute_card_digest(card: Mapping[str, Any]) -> str:
    """Recompute the exact Card v1 self-excluding canonical digest."""
    try:
        plain = card_to_json(card)
    finally:
        card = cast(Mapping[str, Any], None)
    if not isinstance(plain, dict):
        raise CardValidationError("card digest input must be an object")
    schema_name = plain.get("schema")
    schema_version = plain.get("schema_version")
    if (
        type(schema_name) is not str
        or type(schema_version) is not int
        or SUPPORTED_CARD_SCHEMA_VERSIONS.get(schema_name) != schema_version
    ):
        raise CardValidationError("unsupported card schema/version pair")
    try:
        wrapper = {
            "domain": plain["schema"],
            "schema_version": plain["schema_version"],
            "id": plain["id"],
            "content": plain["content"],
        }
    except KeyError as exc:
        raise CardValidationError(
            f"card is missing digest input {exc.args[0]!r}"
        ) from exc
    _strict_json(wrapper)
    try:
        encoded = canonical_json_bytes(wrapper, "card digest wrapper")
    except CardValidationError:
        raise
    except (JsonSafetyError, RecursionError, UnicodeError) as exc:
        raise CardValidationError("malformed in-memory card digest wrapper") from exc
    return hashlib.sha256(encoded).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _untrusted_mapping_items(
    value: Mapping[Any, Any], context: str
) -> list[tuple[Any, Any]]:
    """Read at most one-over the mapping width cap without trusting ``len``."""
    try:
        iterator = iter(value.items())
        items: list[tuple[Any, Any]] = []
        for index in range(_MAX_CONTAINER_MEMBERS + 1):
            try:
                pair = next(iterator)
            except StopIteration:
                return items
            if index == _MAX_CONTAINER_MEMBERS:
                raise _UnsafeProjectionError(
                    f"{context} exceeds {_MAX_CONTAINER_MEMBERS} members"
                )
            key, child = pair
            items.append((key, child))
        return items  # pragma: no cover - the bounded loop always returns or raises
    except _UnsafeProjectionError:
        raise
    except _DECLARED_UNTRUSTED_PROTOCOL_ERRORS as exc:
        raise _UnsafeProjectionError(f"malformed {context}") from exc
    except Exception:
        raise _UnsafeProjectionError(f"malformed {context}") from None


def _untrusted_sequence_items(value: Sequence[Any], context: str) -> list[Any]:
    """Read at most one-over the sequence width cap without trusting ``len``."""
    try:
        iterator = iter(value)
        items: list[Any] = []
        for index in range(_MAX_CONTAINER_MEMBERS + 1):
            try:
                item = next(iterator)
            except StopIteration:
                return items
            if index == _MAX_CONTAINER_MEMBERS:
                raise _UnsafeProjectionError(
                    f"{context} exceeds {_MAX_CONTAINER_MEMBERS} members"
                )
            items.append(item)
        return items  # pragma: no cover - the bounded loop always returns or raises
    except _UnsafeProjectionError:
        raise
    except _DECLARED_UNTRUSTED_PROTOCOL_ERRORS as exc:
        raise _UnsafeProjectionError(f"malformed {context}") from exc
    except Exception:
        raise _UnsafeProjectionError(f"malformed {context}") from None


def _projection_refusal(detail: str) -> CardValidationError:
    cause = JsonSafetyError(detail)
    error = _UnsafeProjectionError(f"malformed in-memory Card/Census value: {detail}")
    error.__cause__ = cause
    return error


class _ProjectionBudget:
    """One exact aggregate admission budget shared by a Card reference graph."""

    def __init__(self) -> None:
        self.nodes = 0
        self.string_bytes = 0

    def node(self) -> None:
        self.nodes += 1
        if self.nodes > _MAX_PROJECTED_NODES:
            raise _projection_refusal(
                f"value exceeds {_MAX_PROJECTED_NODES} traversed nodes"
            )

    def text(self, value: str) -> None:
        try:
            size = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            raise _projection_refusal("text contains a lone surrogate") from None
        if size > _MAX_STRING_UTF8_BYTES:
            raise _projection_refusal(
                f"string/key exceeds {_MAX_STRING_UTF8_BYTES} UTF-8 bytes"
            )
        self.string_bytes += size
        if self.string_bytes > _MAX_AGGREGATE_STRING_UTF8_BYTES:
            raise _projection_refusal(
                "value exceeds "
                f"{_MAX_AGGREGATE_STRING_UTF8_BYTES} aggregate UTF-8 string/key bytes"
            )


def _project_untrusted_json(
    value: Any,
    *,
    ancestors: set[int],
    depth: int,
    budget: _ProjectionBudget,
) -> Any:
    budget.node()
    if value is None or type(value) in {bool, int, float}:
        return value
    if type(value) is str:
        budget.text(value)
        return value
    if isinstance(value, (str, bool, int, float)):
        raise _projection_refusal("JSON primitive subclasses are not JSON-safe")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise _projection_refusal("binary values are not JSON-safe")
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in ancestors:
            raise _projection_refusal("cyclic reference detected in mapping")
        if depth >= MAX_JSON_DEPTH:
            raise _projection_refusal(
                f"value is nested deeper than {MAX_JSON_DEPTH} JSON levels"
            )
        items = _untrusted_mapping_items(value, "in-memory Card/Census mapping")
        ancestors.add(marker)
        try:
            projected: dict[str, Any] = {}
            for key, child in items:
                if type(key) is not str:
                    raise _projection_refusal("mapping keys must be exact strings")
                budget.node()
                budget.text(key)
                if key in projected:
                    raise _projection_refusal(f"duplicate mapping key {key!r}")
                projected[key] = _project_untrusted_json(
                    child,
                    ancestors=ancestors,
                    depth=depth + 1,
                    budget=budget,
                )
            return projected
        finally:
            ancestors.discard(marker)
    if isinstance(value, Sequence):
        marker = id(value)
        if marker in ancestors:
            raise _projection_refusal("cyclic reference detected in sequence")
        if depth >= MAX_JSON_DEPTH:
            raise _projection_refusal(
                f"value is nested deeper than {MAX_JSON_DEPTH} JSON levels"
            )
        items = _untrusted_sequence_items(value, "in-memory Card/Census sequence")
        ancestors.add(marker)
        try:
            return [
                _project_untrusted_json(
                    child,
                    ancestors=ancestors,
                    depth=depth + 1,
                    budget=budget,
                )
                for child in items
            ]
        finally:
            ancestors.discard(marker)
    raise _projection_refusal("unsupported value is not JSON-safe")


def card_to_json(value: Any) -> Any:
    """Return a recursively detached mutable plain-JSON projection."""
    failure: tuple[str, str | None] | None = None
    plain: Any = None
    try:
        plain = _project_untrusted_json(
            value, ancestors=set(), depth=0, budget=_ProjectionBudget()
        )
    except _UnsafeProjectionError as caught:
        failure = _detach_projection_error(caught)
    finally:
        value = None
    if failure is not None:
        message, detail = failure
        failure = None
        _raise_detached_projection_error(message, detail)
    try:
        ensure_json_safe(plain, "Card/Census value")
    except JsonSafetyError as exc:
        raise CardValidationError("malformed in-memory Card/Census value") from exc
    return plain


def _check_canonical_card_size(value: Any) -> None:
    try:
        size = len(canonical_json_bytes(value, "projected Card"))
    except CardValidationError:
        raise
    except Exception:
        raise CardValidationError("malformed canonicalized Card") from None
    if size > _MAX_CANONICAL_CARD_BYTES:
        raise CardValidationError(
            f"canonicalized Card exceeds {_MAX_CANONICAL_CARD_BYTES} bytes"
        )


def _validate_timestamp(value: str, path: str) -> None:
    if _TIMESTAMP_RE.fullmatch(value) is None:
        raise CardValidationError(f"{path} is not an exact six-fraction UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise CardValidationError(
            f"{path} is not a Gregorian calendar timestamp"
        ) from exc


def _validate_field_shape(value: Any, shape: Mapping[str, Any], path: str) -> None:
    kind = shape["kind"]
    if kind == "NULLABLE_STRING" and value is None:
        return
    if kind in ("string", "NULLABLE_STRING"):
        if not isinstance(value, str):
            raise CardValidationError(f"{path} payload value must be string")
        if not shape["min_length"] <= len(value) <= shape["max_length"]:
            raise CardValidationError(f"{path} payload string length is invalid")
        if shape["pattern"] is not None and not _pattern_fullmatch(
            shape["pattern"], value, path, trusted_pattern=False
        ):
            raise CardValidationError(f"{path} payload string pattern is invalid")
        if shape["enum"] is not None and value not in shape["enum"]:
            raise CardValidationError(f"{path} payload string is outside its enum")
    elif kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CardValidationError(f"{path} payload value must be integer")
        if not shape["minimum"] <= value <= shape["maximum"]:
            raise CardValidationError(f"{path} payload integer is outside its bounds")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise CardValidationError(f"{path} payload value must be boolean")
    elif kind == "null":
        if value is not None:
            raise CardValidationError(f"{path} payload value must be null")
    elif kind == "array":
        if not isinstance(value, list):
            raise CardValidationError(f"{path} payload value must be array")
        if not shape["min_items"] <= len(value) <= shape["max_items"]:
            raise CardValidationError(f"{path} payload array length is invalid")
        if shape["unique_items"]:
            encoded = [canonical_json_bytes(item, path) for item in value]
            if len(encoded) != len(set(encoded)):
                raise CardValidationError(f"{path} payload array items must be unique")
        for index, item in enumerate(value):
            _validate_field_shape(item, shape["items"], f"{path}/{index}")
    elif kind == "object":
        if not isinstance(value, Mapping):
            raise CardValidationError(f"{path} payload value must be object")
        fields = shape["fields"]
        required = {name for name, entry in fields.items() if entry["required"]}
        allowed = set(fields)
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise CardValidationError(f"{path} payload object shape mismatch")
        for name, item in value.items():
            _validate_field_shape(item, fields[name]["shape"], f"{path}/{name}")


def _reference_index(
    references: Iterable[Mapping[str, Any]], *, budget: _ProjectionBudget
) -> dict[tuple[str, str], dict[str, Any]]:
    """Project and validate one complete, bounded reference graph."""
    try:
        iterator = iter(references)
        materialized: list[Mapping[str, Any]] = []
        for position in range(_MAX_CONTAINER_MEMBERS + 1):
            try:
                candidate = next(iterator)
            except StopIteration:
                break
            if position == _MAX_CONTAINER_MEMBERS:
                raise _UnsafeProjectionError(
                    f"card reference collection exceeds {_MAX_CONTAINER_MEMBERS} members"
                )
            materialized.append(candidate)
    except _UnsafeProjectionError:
        raise
    except _DECLARED_UNTRUSTED_PROTOCOL_ERRORS as exc:
        raise _UnsafeProjectionError("malformed card reference collection") from exc
    except Exception:
        raise _UnsafeProjectionError("malformed card reference collection") from None
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in materialized:
        target = _project_untrusted_json(
            candidate, ancestors=set(), depth=0, budget=budget
        )
        if not isinstance(target, dict):
            raise CardValidationError("card reference target must be an object")
        _check_canonical_card_size(target)
        schema = target.get("schema")
        identifier = target.get("id")
        if type(schema) is not str or type(identifier) is not str:
            raise CardValidationError("card reference target has malformed schema/id")
        key = (schema, identifier)
        if key in index:
            raise CardValidationError(f"ambiguous card reference target {key}")
        index[key] = target
    validated: set[tuple[str, str]] = set()
    active: set[tuple[str, str]] = set()

    def validate_target(key: tuple[str, str]) -> None:
        if key in validated:
            return
        if key in active:
            raise CardValidationError("card reference graph contains a cycle")
        target = index[key]
        active.add(key)
        try:
            _validate_card_shape_and_digest(target)
            schema = target["schema"]
            if schema in (OPERATION_CARD_SCHEMA, OPERATION_CARD_V2_SCHEMA):
                _validate_operation(target["content"])
            elif schema in (
                SEMANTIC_SCENARIO_CARD_SCHEMA,
                SEMANTIC_SCENARIO_CARD_V2_SCHEMA,
            ):
                operation = _resolve_exact_target(
                    target["content"]["operation_core"], index
                )
                operation_key = (operation["schema"], operation["id"])
                validate_target(operation_key)
                expected = (
                    OPERATION_CARD_V2_SCHEMA
                    if schema == SEMANTIC_SCENARIO_CARD_V2_SCHEMA
                    else OPERATION_CARD_SCHEMA
                )
                if operation["schema"] != expected:
                    raise CardValidationError(
                        "scenario target does not resolve to a same-version operation"
                    )
                _validate_scenario(target["content"], operation)
            elif schema in (VARIANT_CARD_SCHEMA, VARIANT_CARD_V2_SCHEMA):
                scenario = _resolve_exact_target(
                    target["content"]["semantic_scenario"], index
                )
                scenario_key = (scenario["schema"], scenario["id"])
                validate_target(scenario_key)
                expected_scenario = (
                    SEMANTIC_SCENARIO_CARD_V2_SCHEMA
                    if schema == VARIANT_CARD_V2_SCHEMA
                    else SEMANTIC_SCENARIO_CARD_SCHEMA
                )
                if scenario["schema"] != expected_scenario:
                    raise CardValidationError(
                        "variant target does not resolve to a same-version scenario"
                    )
                operation = _resolve_exact_target(
                    scenario["content"]["operation_core"], index
                )
                _validate_variant(target["content"], operation)
            else:
                raise CardValidationError(f"unsupported card reference target {key}")
            validated.add(key)
        except CardValidationError:
            raise CardValidationError("invalid card reference target") from None
        finally:
            active.discard(key)

    for key in index:
        validate_target(key)
    return index


def _reachable_reference_keys(
    card: Mapping[str, Any], index: Mapping[tuple[str, str], dict[str, Any]]
) -> set[tuple[str, str]]:
    """Return the exact transitive references authoritative for ``card``."""
    reachable: set[tuple[str, str]] = set()
    pending: list[Mapping[str, Any]] = []
    schema = card["schema"]
    if schema in (SEMANTIC_SCENARIO_CARD_SCHEMA, SEMANTIC_SCENARIO_CARD_V2_SCHEMA):
        pending.append(card["content"]["operation_core"])
    elif schema in (VARIANT_CARD_SCHEMA, VARIANT_CARD_V2_SCHEMA):
        pending.append(card["content"]["semantic_scenario"])
    while pending:
        ref = pending.pop()
        target = _resolve_exact_target(ref, index)
        key = (target["schema"], target["id"])
        if key in reachable:
            continue
        reachable.add(key)
        target_schema = target["schema"]
        if target_schema in (
            SEMANTIC_SCENARIO_CARD_SCHEMA,
            SEMANTIC_SCENARIO_CARD_V2_SCHEMA,
        ):
            pending.append(target["content"]["operation_core"])
        elif target_schema in (VARIANT_CARD_SCHEMA, VARIANT_CARD_V2_SCHEMA):
            pending.append(target["content"]["semantic_scenario"])
    return reachable


def _resolve_exact_target(
    ref: Mapping[str, Any], index: Mapping[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    target = index.get((ref["schema"], ref["id"]))
    if (
        target is None
        or target.get("content_digest_sha256") != ref["content_digest_sha256"]
    ):
        raise CardValidationError(
            "card reference does not resolve to exact schema/id/digest"
        )
    return target


def _validate_card_shape_and_digest(plain: dict[str, Any]) -> None:
    schema_name = plain.get("schema")
    schema_version = plain.get("schema_version")
    if (
        type(schema_name) is not str
        or type(schema_version) is not int
        or SUPPORTED_CARD_SCHEMA_VERSIONS.get(schema_name) != schema_version
    ):
        raise CardValidationError(
            f"unsupported card schema/version pair {schema_name!r}/{schema_version!r}"
        )
    _strict_json(plain)
    defs_file = (
        "card-defs-v2.schema.json" if schema_version == 2 else "card-defs-v1.schema.json"
    )
    _validate_schema(plain, load_schema(schema_name), "$", defs_file=defs_file)
    if recompute_card_digest(plain) != plain["content_digest_sha256"]:
        raise CardValidationError("card reference target digest is stale or invalid")


def _validate_field_shape_patterns(shape: Mapping[str, Any], path: str) -> None:
    kind = shape["kind"]
    if kind in ("string", "NULLABLE_STRING") and shape["pattern"] is not None:
        _validate_safe_pattern(shape["pattern"], f"{path}/pattern")
    elif kind == "array":
        _validate_field_shape_patterns(shape["items"], f"{path}/items")
    elif kind == "object":
        for name, entry in shape["fields"].items():
            _validate_field_shape_patterns(entry["shape"], f"{path}/fields/{name}/shape")


def _validate_field_shape_map_patterns(shapes: Mapping[str, Any], path: str) -> None:
    for name, entry in shapes.items():
        _validate_field_shape_patterns(entry["shape"], f"{path}/{name}/shape")


def _validate_schema_patterns(schema: Mapping[str, Any], path: str) -> None:
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        _validate_safe_pattern(pattern, f"{path}/pattern")
    for keyword in ("properties", "patternProperties"):
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            for name, child in children.items():
                if keyword == "patternProperties":
                    _validate_safe_pattern(name, f"{path}/{keyword}/<pattern>")
                if isinstance(child, Mapping):
                    _validate_schema_patterns(child, f"{path}/{keyword}/{name}")
    for keyword in ("items", "propertyNames", "additionalProperties"):
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            _validate_schema_patterns(child, f"{path}/{keyword}")


def _validate_operation_patterns(content: Mapping[str, Any]) -> None:
    _validate_schema_patterns(
        content["state_contract"]["initial_state_schema"],
        "$/content/state_contract/initial_state_schema",
    )
    for contract_kind in ("event_contracts", "action_contracts"):
        for index, contract in enumerate(content[contract_kind]):
            for field_kind in ("required", "optional"):
                _validate_field_shape_map_patterns(
                    contract[field_kind],
                    f"$/content/{contract_kind}/{index}/{field_kind}",
                )
    for index, tool in enumerate(content["retrieval_contract"]["catalogue"]):
        _validate_field_shape_map_patterns(
            tool["arguments"],
            f"$/content/retrieval_contract/catalogue/{index}/arguments",
        )


def _validate_operation(content: Mapping[str, Any]) -> None:
    # Gate every card-authored regex before any later validation can match one.
    _validate_operation_patterns(content)
    authority = content["authority_contract"]
    roles = authority["roles"]
    role_ids = [item["role_id"] for item in roles]
    if len(set(role_ids)) != len(role_ids):
        raise CardValidationError("operation role IDs collide")
    event_types = [item["event_type"] for item in content["event_contracts"]]
    action_types = [item["action_type"] for item in content["action_contracts"]]
    if len(set(event_types)) != len(event_types) or len(set(action_types)) != len(
        action_types
    ):
        raise CardValidationError("operation contract definitions collide")
    registered_events = [item["event_type"] for item in authority["event_authority"]]
    registered_actions = [item["action_type"] for item in authority["action_authority"]]
    if sorted(registered_events) != sorted(event_types) or sorted(
        registered_actions
    ) != sorted(action_types):
        raise CardValidationError("operation authority and contract definitions mismatch")
    known_authorities = {item for role in roles for item in role["authorities"]}
    declared = [item["required_authority"] for item in authority["event_authority"]]
    declared += [item["required_authority"] for item in authority["action_authority"]]
    if not set(declared).issubset(known_authorities):
        raise CardValidationError("operation contract requires unknown authority")
    tools = [item["tool_id"] for item in content["retrieval_contract"]["catalogue"]]
    selectors = [
        item["selector_id"]
        for item in content["retrieval_contract"]["evidence_requirements"]
    ]
    if len(tools) != len(set(tools)):
        raise CardValidationError("operation retrieval tool IDs collide")
    if len(selectors) != len(set(selectors)):
        raise CardValidationError("operation evidence selector IDs collide")
    if any(
        tool not in set(tools)
        for requirement in content["retrieval_contract"]["evidence_requirements"]
        for tool in requirement["required_tools"]
    ):
        raise CardValidationError("operation evidence requirement has unknown tool")


_MISSING = object()


def _pointer_value(root: Any, pointer: str) -> Any:
    if pointer == "":
        return root
    current = root
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            return _MISSING
        current = current[token]
    return current


def _evaluate_predicate_v2(
    predicate: Mapping[str, Any],
    state: Mapping[str, Any],
    bindings: Mapping[str, Any] | None = None,
) -> tuple[bool, Mapping[str, Any] | None]:
    """Evaluate the closed Card v2 predicate AST with exact binding semantics."""
    active = dict(bindings or {})

    def expression_value(expression: Mapping[str, Any], scope: Mapping[str, Any]) -> Any:
        kind = expression["kind"]
        if kind == "literal":
            return expression["value"]
        if kind == "pointer":
            return _pointer_value(state, expression["pointer"])
        if kind == "binding":
            bound = scope.get(expression["binding"], _MISSING)
            return _pointer_value(bound, expression["pointer"])
        raise CardValidationError("unknown Card v2 value-expression kind")

    def walk(
        clause: Mapping[str, Any], scope: dict[str, Any]
    ) -> tuple[bool, dict[str, Any] | None]:
        kind = clause["kind"]
        if kind == "all":
            accumulated = dict(scope)
            for operand in clause["operands"]:
                truth, result = walk(operand, accumulated)
                if not truth or result is None:
                    return False, None
                accumulated = result
            return True, accumulated
        if kind == "any":
            for operand in clause["operands"]:
                truth, result = walk(operand, dict(scope))
                if truth:
                    return True, result
            return False, None
        if kind == "not":
            truth, _ = walk(clause["operand"], dict(scope))
            return (not truth, dict(scope) if not truth else None)
        if kind == "scalar":
            left = expression_value(clause["left"], scope)
            right = expression_value(clause["right"], scope)
            if left is _MISSING or right is _MISSING:
                return False, None
            operator = clause["operator"]
            equal = type(left) is type(right) and left == right
            try:
                if operator == "EQ":
                    truth = equal
                elif operator == "NE":
                    truth = not equal
                elif operator in {"LT", "LTE", "GT", "GTE"}:
                    if type(left) is not type(right) or type(left) not in {int, str}:
                        truth = False
                    else:
                        truth = {
                            "LT": left < right,
                            "LTE": left <= right,
                            "GT": left > right,
                            "GTE": left >= right,
                        }[operator]
                elif operator in {"IN", "NOT_IN"}:
                    if type(right) is not list:
                        truth = False
                    else:
                        contained = any(
                            type(left) is type(item) and left == item for item in right
                        )
                        truth = contained if operator == "IN" else not contained
                elif operator == "MATCHES":
                    truth = (
                        type(left) is str
                        and type(right) is str
                        and bool(
                            _pattern_fullmatch(
                                right, left, "$predicate", trusted_pattern=False
                            )
                        )
                    )
                else:
                    raise CardValidationError("unknown Card v2 scalar operator")
            except TypeError:
                truth = False
            return (truth, dict(scope) if truth else None)
        if kind == "quantified":
            collection = _pointer_value(state, clause["collection"])
            if not isinstance(collection, Mapping):
                return False, None
            if any(type(key) is not str for key in collection):
                return False, None
            matches: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
            for key in sorted(collection):
                element = collection[key]
                if not isinstance(element, Mapping):
                    continue
                nested_scope = {**scope, clause["binding"]: element}
                truth, result = walk(clause["predicate"], nested_scope)
                if truth and result is not None:
                    matches.append((element, result))
            quantifier = clause["quantifier"]
            if quantifier == "EXACTLY_ONE":
                if len(matches) != 1:
                    return False, None
                element, result = matches[0]
                return True, {**result, clause["binding"]: element}
            if quantifier == "ANY":
                if not matches:
                    return False, None
                element, result = matches[0]
                return True, {**result, clause["binding"]: element}
            if quantifier == "NONE":
                return (False, None) if matches else (True, dict(scope))
            if quantifier == "ALL":
                object_count = sum(
                    isinstance(item, Mapping) for item in collection.values()
                )
                return (
                    (
                        True,
                        {
                            **matches[-1][1],
                            clause["binding"]: matches[-1][0],
                        }
                        if matches
                        else dict(scope),
                    )
                    if len(matches) == object_count
                    else (False, None)
                )
            raise CardValidationError("unknown Card v2 quantifier")
        raise CardValidationError("unknown Card v2 predicate kind")

    truth, result = walk(predicate, active)
    return truth, result if truth else None


def _validate_value_expression(
    expression: Mapping[str, Any], allowed_pointers: set[str], bindings: set[str]
) -> None:
    if expression["kind"] == "pointer" and expression["pointer"] not in allowed_pointers:
        pointer = expression["pointer"]
        raise CardValidationError(
            f"scenario predicate pointer {pointer!r} is not operation-declared"
        )
    if expression["kind"] == "binding" and expression["binding"] not in bindings:
        raise CardValidationError(
            f"scenario predicate binding {expression['binding']!r} is unresolved"
        )


def _validate_predicate(
    predicate: Mapping[str, Any], allowed_pointers: set[str], bindings: set[str]
) -> None:
    kind = predicate["kind"]
    if kind in ("all", "any"):
        for operand in predicate["operands"]:
            _validate_predicate(operand, allowed_pointers, bindings)
    elif kind == "not":
        _validate_predicate(predicate["operand"], allowed_pointers, bindings)
    elif kind == "scalar":
        _validate_value_expression(predicate["left"], allowed_pointers, bindings)
        _validate_value_expression(predicate["right"], allowed_pointers, bindings)
    elif kind == "quantified":
        if predicate["collection"] not in allowed_pointers:
            raise CardValidationError(
                f"scenario predicate pointer {predicate['collection']!r} is not "
                "operation-declared"
            )
        binding = predicate["binding"]
        if binding in bindings:
            raise CardValidationError(
                f"scenario predicate binding {binding!r} would redeclare "
                "an active binding"
            )
        nested_bindings = bindings | {binding}
        _validate_predicate(predicate["predicate"], allowed_pointers, nested_bindings)


def _validate_scenario(content: Mapping[str, Any], operation: Mapping[str, Any]) -> None:
    operation_content = operation["content"]
    state_contract = operation_content["state_contract"]
    allowed_pointers = set(state_contract["canonical_state_projection"])
    allowed_pointers.update(
        operation_content["transition_contract"]["common_predicate_inputs"]
    )
    points = content["decision_points"]
    point_ids = [point["point_id"] for point in points]
    if len(set(point_ids)) != len(point_ids) or len(
        {point["order"] for point in points}
    ) != len(points):
        raise CardValidationError(
            "decision point identities and order values must be unique"
        )
    obligations = [
        item["obligation_id"] for item in content["common_outcome_obligations"]
    ]
    hazards = [item["hazard_id"] for item in content["authored_hazards"]]
    if set(point_ids) & set(hazards):
        raise CardValidationError("scenario hazard and decision point IDs collide")
    dimensions = [item["dimension_id"] for item in content["allowed_variant_dimensions"]]
    for label, identifiers in (
        ("obligation", obligations),
        ("hazard", hazards),
        ("variant dimension", dimensions),
    ):
        if len(set(identifiers)) != len(identifiers):
            raise CardValidationError(f"scenario {label} identities collide")
    if any(point["common_obligation_id"] not in obligations for point in points):
        raise CardValidationError("scenario decision has unresolved obligation")
    seen: set[str] = set()
    for point in points:
        _validate_predicate(point["reach_predicate"], allowed_pointers, set())
        if any(prior not in seen for prior in point["required_prior_point_ids"]):
            raise CardValidationError(
                "decision point prerequisite is unresolved or not prior"
            )
        seen.add(point["point_id"])
    for hazard in content["authored_hazards"]:
        _validate_predicate(hazard["predicate"], allowed_pointers, set())
    node_ids = [node["node_id"] for node in content["causal_graph"]["nodes"]]
    if len(set(node_ids)) != len(node_ids):
        raise CardValidationError("scenario causal node identities collide")
    nodes = set(node_ids)
    if nodes != set(point_ids) | set(hazards):
        raise CardValidationError("scenario causal nodes do not match points and hazards")
    graph: dict[str, list[str]] = {node: [] for node in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    for edge in content["causal_graph"]["edges"]:
        if edge["cause_node_id"] not in nodes or edge["effect_node_id"] not in nodes:
            raise CardValidationError("causal edge is unresolved")
        graph[edge["cause_node_id"]].append(edge["effect_node_id"])
        indegree[edge["effect_node_id"]] += 1
    ready = deque(node for node in node_ids if indegree[node] == 0)
    visited_count = 0
    while ready:
        node = ready.popleft()
        visited_count += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited_count != len(node_ids):
        raise CardValidationError("causal graph contains a cycle")


def _validate_variant(content: Mapping[str, Any], operation: Mapping[str, Any]) -> None:
    _validate_timestamp(content["starts_at"], "$/content/starts_at")
    fixture_id_list = [fixture["fixture_id"] for fixture in content["message_fixtures"]]
    if len(fixture_id_list) != len(set(fixture_id_list)):
        raise CardValidationError("variant fixture id is duplicate")
    fixture_ids = set(fixture_id_list)
    if not set(content["dispatch_failures"]).issubset(fixture_ids):
        raise CardValidationError("dispatch failure has unresolved fixture binding")
    actor_id_list = [actor["actor_id"] for actor in content["actors"]]
    if len(actor_id_list) != len(set(actor_id_list)):
        raise CardValidationError("variant actor id is duplicate")
    actor_ids = set(actor_id_list)
    event_ids: set[str] = set()
    sequences: set[int] = set()
    last_absolute_key: tuple[str, int, str] | None = None
    for event_index, event in enumerate(content["events"]):
        event_path = f"$/content/events/{event_index}"
        if event["event_id"] in event_ids or event["authored_sequence"] in sequences:
            raise CardValidationError(
                f"{event_path} variant event identity/sequence is duplicate"
            )
        event_ids.add(event["event_id"])
        sequences.add(event["authored_sequence"])
        if event["actor_id"] not in actor_ids:
            raise CardValidationError(f"{event_path}/actor_id is unresolved")
        if (event["at"] is None) == (event["trigger"] is None):
            raise CardValidationError(
                f"{event_path} requires exactly one of at and trigger"
            )
        if event["at"] is not None:
            _validate_timestamp(event["at"], f"{event_path}/at")
            absolute_key = (
                event["at"],
                event["authored_sequence"],
                event["event_id"],
            )
            if last_absolute_key is not None and absolute_key < last_absolute_key:
                raise CardValidationError(
                    f"{event_path} absolute variant event is not in canonical "
                    "authored order"
                )
            last_absolute_key = absolute_key
    events = content["events"]
    conditional = [
        (index, event)
        for index, event in enumerate(events)
        if event["trigger"] is not None
    ]
    last_conditional_key_by_owner: dict[bytes, tuple[int, str]] = {}
    for event_index, event in conditional:
        event_path = f"$/content/events/{event_index}"
        trigger = event["trigger"]
        owner = canonical_json_bytes(
            {
                "kind": trigger["kind"],
                "type_name": trigger["type_name"],
                "correlation": trigger["correlation"],
            },
            f"conditional event causal owner at {event_path}",
        )
        key = event["authored_sequence"], event["event_id"]
        if (
            owner in last_conditional_key_by_owner
            and key < last_conditional_key_by_owner[owner]
        ):
            raise CardValidationError(
                f"{event_path} conditional variant event is not in canonical "
                "authored order"
            )
        last_conditional_key_by_owner[owner] = key
    rejection_indices: dict[str, int] = {}
    for rejection_index, rejection in enumerate(content["expected_event_rejections"]):
        rejection_path = f"$/content/expected_event_rejections/{rejection_index}/event_id"
        event_id = rejection["event_id"]
        if event_id not in event_ids:
            raise CardValidationError(f"{rejection_path} {event_id!r} is unresolved")
        if event_id in rejection_indices:
            first_path = (
                "$/content/expected_event_rejections/"
                f"{rejection_indices[event_id]}/event_id"
            )
            raise CardValidationError(
                f"{rejection_path} duplicates event_id {event_id!r} first declared at "
                f"{first_path}"
            )
        rejection_indices[event_id] = rejection_index
    _validate_schema(
        content["hidden_initial_values"],
        operation["content"]["state_contract"]["initial_state_schema"],
        "$/content/hidden_initial_values",
        trusted_patterns=False,
    )
    if (
        content["expected_terminal"]
        not in operation["content"]["transition_contract"]["terminal_kinds"]
    ):
        raise CardValidationError("variant expected terminal is not operation-declared")
    roles = {
        role["role_id"]: set(role["authorities"])
        for role in operation["content"]["authority_contract"]["roles"]
    }
    actors = {actor["actor_id"]: actor["role_id"] for actor in content["actors"]}
    for actor_index, actor in enumerate(content["actors"]):
        if actor["role_id"] not in roles:
            raise CardValidationError(
                f"$/content/actors/{actor_index}/role_id {actor['role_id']!r} is not "
                "operation-declared"
            )
    authority = {
        item["event_type"]: item["required_authority"]
        for item in operation["content"]["authority_contract"]["event_authority"]
    }
    contracts = {
        item["event_type"]: item for item in operation["content"]["event_contracts"]
    }
    trigger_contracts = {
        "event": contracts,
        "action": {
            item["action_type"]: item for item in operation["content"]["action_contracts"]
        },
    }
    for event_index, event in enumerate(content["events"]):
        event_path = f"$/content/events/{event_index}"
        if event["event_type"] not in contracts:
            raise CardValidationError(
                f"{event_path}/event_type is not operation-declared"
            )
        if (
            actors[event["actor_id"]] not in roles
            or authority.get(event["event_type"]) not in roles[actors[event["actor_id"]]]
        ):
            raise CardValidationError(f"{event_path}/actor_id authority mismatch")
        contract = contracts[event["event_type"]]
        if set(event["payload"]) != set(contract["required"]) | (
            set(event["payload"]) & set(contract["optional"])
        ):
            raise CardValidationError(f"{event_path}/payload shape mismatch")
        field_contracts = {**contract["required"], **contract["optional"]}
        for field_name, field_value in event["payload"].items():
            _validate_field_shape(
                field_value,
                field_contracts[field_name]["shape"],
                f"{event_path}/payload/{field_name}",
            )
        trigger = event["trigger"]
        if trigger is not None:
            trigger_contract = trigger_contracts[trigger["kind"]].get(
                trigger["type_name"]
            )
            if trigger_contract is None:
                raise CardValidationError(
                    f"{event_path}/trigger/type_name is not operation-declared"
                )
            correlation_fields = {
                **trigger_contract["required"],
                **trigger_contract["optional"],
            }
            if not set(trigger["correlation"]).issubset(correlation_fields):
                raise CardValidationError(
                    f"{event_path}/trigger/correlation has undeclared fields"
                )
            for field_name, field_value in trigger["correlation"].items():
                try:
                    _validate_field_shape(
                        field_value,
                        correlation_fields[field_name]["shape"],
                        f"{event_path}/trigger/correlation/{field_name}",
                    )
                except CardValidationError as exc:
                    raise CardValidationError(
                        f"{event_path}/trigger/correlation/{field_name} is malformed"
                    ) from exc


def validate_card(
    card: Mapping[str, Any], *, references: Iterable[Mapping[str, Any]] = ()
) -> Mapping[str, Any]:
    """Validate a Card v1 and return a recursively immutable detached value."""
    budget = _ProjectionBudget()
    projection_failure: tuple[str, str | None] | None = None
    plain: Any = None
    try:
        plain = _project_untrusted_json(card, ancestors=set(), depth=0, budget=budget)
    except _UnsafeProjectionError as error:
        projection_failure = _detach_projection_error(error)
    finally:
        card = cast(Mapping[str, Any], None)
    if projection_failure is not None:
        references = ()
        message, detail = projection_failure
        projection_failure = None
        _raise_detached_projection_error(message, detail)
    _check_canonical_card_size(plain)
    if not isinstance(plain, dict):
        raise CardValidationError("Card v1 value must be an object")
    _validate_card_shape_and_digest(plain)
    schema_name = plain["schema"]
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if schema_name != OPERATION_CARD_SCHEMA:
        try:
            index = _reference_index(references, budget=budget)
        except _UnsafeProjectionError as error:
            projection_failure = _detach_projection_error(error)
        finally:
            references = ()
        if projection_failure is not None:
            message, detail = projection_failure
            projection_failure = None
            _raise_detached_projection_error(message, detail)
    references = ()
    if schema_name in (OPERATION_CARD_SCHEMA, OPERATION_CARD_V2_SCHEMA):
        _validate_operation(plain["content"])
    elif schema_name in (
        SEMANTIC_SCENARIO_CARD_SCHEMA,
        SEMANTIC_SCENARIO_CARD_V2_SCHEMA,
    ):
        operation = _resolve_exact_target(plain["content"]["operation_core"], index)
        expected_operation_schema = (
            OPERATION_CARD_V2_SCHEMA
            if schema_name == SEMANTIC_SCENARIO_CARD_V2_SCHEMA
            else OPERATION_CARD_SCHEMA
        )
        if operation["schema"] != expected_operation_schema:
            raise CardValidationError(
                "scenario reference is not a same-version operation"
            )
        _validate_scenario(plain["content"], operation)
    elif schema_name in (VARIANT_CARD_SCHEMA, VARIANT_CARD_V2_SCHEMA):
        scenario = _resolve_exact_target(plain["content"]["semantic_scenario"], index)
        expected_scenario_schema = (
            SEMANTIC_SCENARIO_CARD_V2_SCHEMA
            if schema_name == VARIANT_CARD_V2_SCHEMA
            else SEMANTIC_SCENARIO_CARD_SCHEMA
        )
        expected_operation_schema = (
            OPERATION_CARD_V2_SCHEMA
            if schema_name == VARIANT_CARD_V2_SCHEMA
            else OPERATION_CARD_SCHEMA
        )
        if scenario["schema"] != expected_scenario_schema:
            raise CardValidationError("variant reference is not a same-version scenario")
        operation = _resolve_exact_target(scenario["content"]["operation_core"], index)
        if operation["schema"] != expected_operation_schema:
            raise CardValidationError("variant scenario operation reference is invalid")
        _validate_variant(plain["content"], operation)
    reachable = _reachable_reference_keys(plain, index)
    if plain["schema_version"] == 2 and set(index) != reachable:
        raise CardValidationError("card reference collection contains extraneous targets")
    return cast(Mapping[str, Any], _freeze(plain))


def _load_card_json_worker(
    source: str | bytes | Path,
    *,
    generation: int,
    references: Iterable[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Parse, project, generation-check, validate, and freeze one card."""
    context = f"Card v{generation} JSON"
    if type(source) is _CONCRETE_PATH_TYPE:
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise CardValidationError(f"cannot read {context} path {source!s}") from exc
    elif type(source) is bytes:
        raw = source
    elif type(source) is str:
        try:
            raw = source.encode("utf-8")
        except UnicodeEncodeError:
            raise CardValidationError(f"{context} is not UTF-8") from None
    else:
        error = TypeError(f"{context} source must have an exact supported type")
        raise CardValidationError(f"malformed {context} source") from error
    if generation == 2 and len(raw) > _MAX_CANONICAL_CARD_BYTES:
        raise CardValidationError(
            f"{context} raw UTF-8 exceeds {_MAX_CANONICAL_CARD_BYTES} bytes"
        )
    value = _load_json_bytes(raw, context)
    if not isinstance(value, dict):
        raise CardValidationError(f"{context} must be an object")
    plain = card_to_json(value)
    expected = (
        {
            OPERATION_CARD_SCHEMA: OPERATION_CARD_V1_SCHEMA_VERSION,
            SEMANTIC_SCENARIO_CARD_SCHEMA: SEMANTIC_SCENARIO_CARD_V1_SCHEMA_VERSION,
            VARIANT_CARD_SCHEMA: VARIANT_CARD_V1_SCHEMA_VERSION,
        }
        if generation == 1
        else {
            OPERATION_CARD_V2_SCHEMA: OPERATION_CARD_V2_SCHEMA_VERSION,
            SEMANTIC_SCENARIO_CARD_V2_SCHEMA: SEMANTIC_SCENARIO_CARD_V2_SCHEMA_VERSION,
            VARIANT_CARD_V2_SCHEMA: VARIANT_CARD_V2_SCHEMA_VERSION,
        }
    )
    if (
        not isinstance(plain, dict)
        or type(plain.get("schema")) is not str
        or type(plain.get("schema_version")) is not int
        or expected.get(plain["schema"]) != plain["schema_version"]
    ):
        raise CardValidationError(f"unsupported Card v{generation} schema/version pair")
    return validate_card(plain, references=references)


def _load_card_json(
    source: str | bytes | Path,
    *,
    generation: int,
    references: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Close caller-facing errors for one exact versioned JSON loader."""
    context = f"Card v{generation} JSON"
    trusted_failure: tuple[str, BaseException | None] | None = None
    detached: tuple[str, str | None] | None = None
    detached_message: str | None = None
    try:
        return _load_card_json_worker(
            source, generation=generation, references=references
        )
    except _DetachedProjectionError as error:
        detached = _detach_projection_error(error)
    except CardValidationError as error:
        message = error.args[0] if error.args and type(error.args[0]) is str else ""
        if message.startswith(
            (f"cannot read {context} path", f"malformed {context} source")
        ):
            trusted_failure = (message, error.__cause__)
        else:
            detached_message = _detach_domain_error(error, f"malformed {context}")
    finally:
        source = cast(str | bytes | Path, None)
        references = ()
    if detached is not None:
        message, detail = detached
        detached = None
        _raise_detached_projection_error(message, detail)
    if detached_message is not None:
        raise CardValidationError(detached_message)
    if trusted_failure is None:  # pragma: no cover
        raise AssertionError("unreachable")
    message, cause = trusted_failure
    trusted_failure = None
    _raise_domain_error(message, cause)


def load_card_json(
    source: str | bytes | Path, *, references: Iterable[Mapping[str, Any]] = ()
) -> Mapping[str, Any]:
    """Load exactly one legacy Card v1, preserving whitespace-padded JSON input."""
    try:
        return _load_card_json(source, generation=1, references=references)
    finally:
        source = cast(str | bytes | Path, None)
        references = ()


def load_card_json_v2(
    source: str | bytes | Path, *, references: Iterable[Mapping[str, Any]] = ()
) -> Mapping[str, Any]:
    """Load exactly one Card v2 with an 8 MiB pre-parse raw UTF-8 bound."""
    try:
        return _load_card_json(source, generation=2, references=references)
    finally:
        source = cast(str | bytes | Path, None)
        references = ()


def _census_schema_leaves(
    schema: Mapping[str, Any],
    *,
    field_path: str,
    schema_path: str,
    expanded_arrays: frozenset[str],
) -> set[str]:
    """Derive census leaves from the independently packaged Card wire schema."""
    ref = schema.get("$ref")
    if isinstance(ref, str):
        ref_name = ref.rsplit("/", 1)[-1]
        if ref_name in _CENSUS_ATOMIC_SCHEMA_DEFS:
            return {field_path}
        return _census_schema_leaves(
            _resolve_ref(ref),
            field_path=field_path,
            schema_path=schema_path,
            expanded_arrays=expanded_arrays,
        )

    branches = schema.get("oneOf")
    if isinstance(branches, Sequence):
        non_null = [branch for branch in branches if branch.get("type") != "null"]
        if len(non_null) == 1:
            return _census_schema_leaves(
                non_null[0],
                field_path=field_path,
                schema_path=schema_path,
                expanded_arrays=expanded_arrays,
            )
        return {field_path}

    if schema_path in _CENSUS_OWNED_OBJECT_PATHS:
        return {field_path}

    properties = schema.get("properties")
    if schema.get("type") == "object" and isinstance(properties, Mapping):
        leaves: set[str] = set()
        for name, child in properties.items():
            escaped = name.replace("~", "~0").replace("/", "~1")
            leaves.update(
                _census_schema_leaves(
                    child,
                    field_path=f"{field_path}/{escaped}",
                    schema_path=f"{schema_path}/{escaped}",
                    expanded_arrays=expanded_arrays,
                )
            )
        return leaves

    if schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            item_ref = items.get("$ref")
            item_schema = _resolve_ref(item_ref) if isinstance(item_ref, str) else items
            if isinstance(item_schema.get("properties"), Mapping) and (
                not isinstance(item_ref, str)
                or item_ref.rsplit("/", 1)[-1] not in _CENSUS_ATOMIC_SCHEMA_DEFS
            ):
                if schema_path not in expanded_arrays:
                    raise CardValidationError(
                        f"Card registry does not classify record array {schema_path}"
                    )
                return _census_schema_leaves(
                    items,
                    field_path=field_path,
                    schema_path=schema_path,
                    expanded_arrays=expanded_arrays,
                )
        return {field_path}

    return {field_path}


def _expected_census_contract_leaves() -> dict[str, str]:
    """Build the canonical path/owner map without consulting the census entries."""
    registry = load_card_registry()
    set_arrays = registry.get("set_class_array_json_pointers")
    ordered_arrays = registry.get("ordered_array_json_pointers")
    documents = registry.get("schema_documents")
    if not isinstance(set_arrays, list):
        raise CardValidationError("packaged Card registry set arrays are malformed")
    if not isinstance(ordered_arrays, list):
        raise CardValidationError("packaged Card registry ordered arrays are malformed")
    if not isinstance(documents, list):
        raise CardValidationError("packaged Card registry documents are malformed")
    expanded_arrays = frozenset((*set_arrays, *ordered_arrays))

    expected: dict[str, str] = {}
    for schema_name, (root, owner) in _CENSUS_CARD_ROOTS.items():
        for path in _census_schema_leaves(
            load_schema(schema_name),
            field_path=f"/{root}",
            schema_path="",
            expanded_arrays=expanded_arrays,
        ):
            expected[path] = owner

    offline_paths: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping) or not isinstance(
            document.get("offline_path"), str
        ):
            raise CardValidationError("packaged Card registry document is malformed")
        offline_paths.add(document["offline_path"])
    offline_paths.update(
        {
            "card-registry-v1.json",
            "identity-field-census-v1.schema.json",
            "identity-field-census-v1.registry.json",
        }
    )
    expected.update(
        {f"/contract_resources/{name}": "card_schema_content" for name in offline_paths}
    )
    return expected


def _expected_census_consumer_tuples() -> dict[str, tuple[str, ...]]:
    """Load the compact normative owner/consumer oracle, independently of census."""
    configured = load_card_registry().get("census_consumer_tuples_by_owner")
    if not isinstance(configured, Mapping) or set(configured) != set(_CENSUS_OWNERS):
        raise CardValidationError("packaged Card registry consumers are malformed")
    expected: dict[str, tuple[str, ...]] = {}
    for owner in _CENSUS_OWNERS:
        consumers = configured[owner]
        if (
            not isinstance(consumers, list)
            or not consumers
            or any(
                not isinstance(consumer, str) or not consumer for consumer in consumers
            )
            or len(consumers) != len(set(consumers))
        ):
            raise CardValidationError("packaged Card registry consumers are malformed")
        expected[owner] = tuple(consumers)
    return expected


def _reject_non_list_census_arrays(
    value: Any,
    path: str = "$",
    *,
    ancestors: set[int] | None = None,
    depth: int = 0,
) -> None:
    """Reject Python sequence shapes that a JSON array cannot preserve exactly."""
    if ancestors is None:
        ancestors = set()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise _projection_refusal("binary values are not JSON-safe")
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in ancestors:
            raise _projection_refusal("cyclic reference detected in mapping")
        if depth >= MAX_JSON_DEPTH:
            raise _projection_refusal(
                f"value is nested deeper than {MAX_JSON_DEPTH} JSON levels"
            )
        items = _untrusted_mapping_items(value, "field census mapping")
        ancestors.add(marker)
        try:
            for key, item in items:
                if type(key) is not str:
                    raise _projection_refusal("mapping keys must be exact strings")
                _reject_non_list_census_arrays(
                    item,
                    f"{path}/{key}",
                    ancestors=ancestors,
                    depth=depth + 1,
                )
        finally:
            ancestors.discard(marker)
        return
    if isinstance(value, list):
        marker = id(value)
        if marker in ancestors:
            raise _projection_refusal("cyclic reference detected in sequence")
        if depth >= MAX_JSON_DEPTH:
            raise _projection_refusal(
                f"value is nested deeper than {MAX_JSON_DEPTH} JSON levels"
            )
        items = _untrusted_sequence_items(value, "field census sequence")
        ancestors.add(marker)
        try:
            for index, item in enumerate(items):
                _reject_non_list_census_arrays(
                    item,
                    f"{path}/{index}",
                    ancestors=ancestors,
                    depth=depth + 1,
                )
        finally:
            ancestors.discard(marker)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raise CardValidationError(f"{path} must be a JSON array represented as a list")


def _validate_field_census(
    census: Mapping[str, Any],
    *,
    artifact_schema: str | None = None,
    source_root: Path | None = None,
) -> Mapping[str, Any]:
    """Validate B1's card census, optionally checking sources under a trusted root."""
    if not isinstance(census, MappingProxyType):
        _reject_non_list_census_arrays(census)
    plain = card_to_json(census)
    _strict_json(plain)
    try:
        _validate_schema(
            plain,
            _cached_resource_json(_CENSUS_SCHEMA_FILE),
            "$",
            enforce_canonical_sets=False,
        )
    except CardValidationError as exc:
        raise CardValidationError(
            "invalid census source path, owner, consumer, duplicate, "
            "or unexpanded pattern"
        ) from exc
    required = {"schema", "schema_version", "artifact_schema", "entries"}
    if not isinstance(plain, dict) or set(plain) != required:
        raise CardValidationError("field census has unknown or missing keys")
    if (
        plain["schema"] != "operatebench.identity_field_census.v1"
        or type(plain["schema_version"]) is not int
        or plain["schema_version"] != CARD_SCHEMA_VERSION
    ):
        raise CardValidationError("unsupported field census schema/version")
    if artifact_schema == "operatebench.artifact.v9":
        raise CardValidationError(
            "Artifact 9 cannot depend on B1 census until its concrete leaf schema "
            "is supplied"
        )
    if plain["artifact_schema"] != "operatebench.b1_card_contract.v1":
        raise CardValidationError("field census is not the B1 card contract surface")
    seen: dict[str, str] = {}
    owners_seen: set[str] = set()
    expected_consumers = _expected_census_consumer_tuples()
    for entry in plain["entries"]:
        expected = {
            "field_path_pattern",
            "owner",
            "source_paths",
            "consumers",
            "nullability",
            "cardinality",
        }
        if not isinstance(entry, dict) or set(entry) != expected:
            raise CardValidationError("field census entry has unknown or missing keys")
        path = entry["field_path_pattern"]
        owner = entry["owner"]
        if owner not in _CENSUS_OWNERS:
            raise CardValidationError(f"field census path {path} has invalid owner")
        if path in seen:
            raise CardValidationError(f"duplicate/multiple owners for census path {path}")
        seen[path] = owner
        owners_seen.add(owner)
        if entry["cardinality"] != "exact" or "*" in path:
            raise CardValidationError(f"unexpanded census pattern {path}")
        if not entry["source_paths"] or not entry["consumers"]:
            raise CardValidationError(f"absent source or consumer for census path {path}")
        if (
            not isinstance(entry["consumers"], list)
            or tuple(entry["consumers"]) != expected_consumers[owner]
        ):
            raise CardValidationError(
                f"census path {path} does not have its canonical consumer tuple"
            )
        for source in entry["source_paths"]:
            if not isinstance(source, str) or not _safe_source_path(source):
                raise CardValidationError(f"unsafe census source path {source!r}")
            if source_root is not None:
                try:
                    trusted_root = source_root.resolve(strict=True)
                    candidate = (trusted_root / source).resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise CardValidationError(f"stale source path {source}") from exc
                if not candidate.is_relative_to(trusted_root):
                    raise CardValidationError(
                        f"census source path escapes root: {source}"
                    )
    missing_owners = sorted(set(_CENSUS_OWNERS) - owners_seen)
    if missing_owners:
        raise CardValidationError(
            f"field census has missing owner roots {missing_owners}"
        )
    expected_leaves = _expected_census_contract_leaves()
    if seen != expected_leaves:
        missing = sorted(set(expected_leaves) - set(seen))
        extra = sorted(set(seen) - set(expected_leaves))
        wrong_owner = sorted(
            path
            for path in set(seen) & set(expected_leaves)
            if seen[path] != expected_leaves[path]
        )
        raise CardValidationError(
            "field census does not match authoritative contract leaves: "
            f"missing={missing}, extra={extra}, wrong_owner={wrong_owner}"
        )
    return cast(Mapping[str, Any], _freeze(plain))


def validate_field_census(
    census: Mapping[str, Any],
    *,
    artifact_schema: str | None = None,
    source_root: Path | None = None,
) -> Mapping[str, Any]:
    """Validate B1's census with a closed domain-error boundary."""
    trusted_failure: tuple[str, BaseException | None] | None = None
    detached: tuple[str, str | None] | None = None
    try:
        if artifact_schema is not None and type(artifact_schema) is not str:
            error = TypeError("artifact_schema must be None or an exact str")
            raise CardValidationError("malformed census artifact schema") from error
        if source_root is not None and type(source_root) is not _CONCRETE_PATH_TYPE:
            error = TypeError("source_root must be None or an exact platform Path")
            raise CardValidationError("malformed census source root") from error
        return _validate_field_census(
            census, artifact_schema=artifact_schema, source_root=source_root
        )
    except _UnsafeProjectionError as error:
        detached = _detach_projection_error(error)
    except _DetachedProjectionError as error:
        detached = _detach_projection_error(error)
    except CardValidationError as error:
        message = error.args[0] if error.args and type(error.args[0]) is str else ""
        if message.startswith(
            (
                "stale source path",
                "malformed census artifact schema",
                "malformed census source root",
            )
        ):
            trusted_failure = (message, error.__cause__)
        else:
            detached = _detach_projection_error(error)
    finally:
        census = cast(Mapping[str, Any], None)
        artifact_schema = None
        source_root = None
    if detached is not None:
        message, detail = detached
        detached = None
        _raise_detached_projection_error(message, detail)
    if trusted_failure is None:  # pragma: no cover - return or caught is exhaustive
        raise AssertionError("unreachable")
    message, cause = trusted_failure
    trusted_failure = None
    _raise_domain_error(message, cause)


_V2_CENSUS_CANONICAL_SHA256: Final = (
    "9c2e60ba816dc3c10ef98ae32ceefc95553f88cd3342847406e9bc8def252d5f"
)
# Structurally independent semantic oracle: every unlisted exact owner/path row
# defaults to ``never``.  Card v2 intentionally has no explicitly nullable
# identity field; this literal cannot be coordinated with resource/hash repins.
_V2_EXPLICIT_NULL_ALLOWED: Final[frozenset[tuple[str, str]]] = frozenset()


def _validate_field_census_v2_worker(
    census: Mapping[str, Any], *, source_root: Path | None = None
) -> Mapping[str, Any]:
    """Validate the independently authored exact Card v2 leaf/owner census."""
    if source_root is not None and type(source_root) is not _CONCRETE_PATH_TYPE:
        error = TypeError("source_root must be None or an exact platform Path")
        raise CardValidationError("malformed census source root") from error
    if not isinstance(census, MappingProxyType):
        _reject_non_list_census_arrays(census)
    plain = card_to_json(census)
    if not isinstance(plain, dict):
        raise CardValidationError("Card v2 census must be an object")
    _strict_json(plain)
    _validate_schema(
        plain,
        _cached_resource_json("identity-field-census-v2.schema.json"),
        "$",
        enforce_canonical_sets=False,
    )
    for entry in plain["entries"]:
        row = (entry["owner"], entry["field_path_pattern"])
        expected_nullability = (
            "explicit_null_allowed" if row in _V2_EXPLICIT_NULL_ALLOWED else "never"
        )
        if entry["nullability"] != expected_nullability:
            raise CardValidationError(f"Card v2 census nullability is wrong for {row!r}")
    encoded = canonical_json_bytes(plain, "Card v2 field census")
    if hashlib.sha256(encoded).hexdigest() != _V2_CENSUS_CANONICAL_SHA256:
        raise CardValidationError(
            "Card v2 census differs from the independent exact leaf/owner oracle"
        )
    if source_root is not None:
        try:
            trusted_root = source_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CardValidationError("stale Card v2 census source root") from exc
        for entry in plain["entries"]:
            for source in entry["source_paths"]:
                if not _safe_source_path(source):
                    raise CardValidationError(f"unsafe census source path {source!r}")
                try:
                    candidate = (trusted_root / source).resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise CardValidationError(f"stale source path {source}") from exc
                if not candidate.is_relative_to(trusted_root):
                    raise CardValidationError(
                        f"census source path escapes root: {source}"
                    )
    return cast(Mapping[str, Any], _freeze(plain))


def validate_field_census_v2(
    census: Mapping[str, Any], *, source_root: Path | None = None
) -> Mapping[str, Any]:
    """Validate the independently authored exact Card v2 leaf/owner census."""
    trusted_failure: tuple[str, BaseException | None] | None = None
    detached: tuple[str, str | None] | None = None
    try:
        return _validate_field_census_v2_worker(census, source_root=source_root)
    except _UnsafeProjectionError as error:
        detached = _detach_projection_error(error)
    except _DetachedProjectionError as error:
        detached = _detach_projection_error(error)
    except CardValidationError as error:
        message = error.args[0] if error.args and type(error.args[0]) is str else ""
        if message.startswith(
            (
                "stale source path",
                "stale Card v2 census source root",
                "malformed census source root",
            )
        ):
            trusted_failure = (message, error.__cause__)
        else:
            detached = _detach_projection_error(error)
    finally:
        census = cast(Mapping[str, Any], None)
        source_root = None
    if detached is not None:
        message, detail = detached
        detached = None
        _raise_detached_projection_error(message, detail)
    if trusted_failure is None:  # pragma: no cover - return or caught is exhaustive
        raise AssertionError("unreachable")
    message, cause = trusted_failure
    trusted_failure = None
    _raise_domain_error(message, cause)


def _safe_source_path(source: str) -> bool:
    if (
        not source
        or "\\" in source
        or re.match(r"^[A-Za-z]:", source) is not None
        or any(ord(character) <= 0x1F or ord(character) == 0x7F for character in source)
    ):
        return False
    path = Path(source)
    if path.is_absolute() or path.as_posix() != source:
        return False
    parts = path.parts
    return bool(parts) and all(part not in ("", ".", "..") for part in parts)


__all__ = [
    "CARD_SCHEMAS",
    "CARD_SCHEMA_VERSION",
    "OPERATION_CARD_SCHEMA",
    "OPERATION_CARD_V2_SCHEMA",
    "SEMANTIC_SCENARIO_CARD_SCHEMA",
    "SEMANTIC_SCENARIO_CARD_V2_SCHEMA",
    "SUPPORTED_CARD_SCHEMA_VERSIONS",
    "VARIANT_CARD_SCHEMA",
    "VARIANT_CARD_V2_SCHEMA",
    "CardValidationError",
    "card_to_json",
    "load_card_json",
    "load_card_json_v2",
    "load_card_registry",
    "load_card_registry_v2",
    "load_field_census",
    "load_field_census_v2",
    "load_schema",
    "recompute_card_digest",
    "validate_card",
    "validate_field_census",
    "validate_field_census_v2",
]
