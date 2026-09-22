"""The one recursive JSON and text safety primitive.

Every artefact this build persists — construct cards, semantic manifests, suite
manifests, scaffolds, run manifests, adapter settings and ledger rows — is
canonical UTF-8 JSON, and every identity it claims is a SHA-256 over exactly
those bytes. So there is exactly one question to ask of any value before it is
hashed or written: *can canonical UTF-8 JSON carry this back unchanged?*

Three adversarial shapes answer "no" and receive named boundary errors:

* **Lone UTF-16 surrogates.** ``json.loads`` and PyYAML can both produce
  ``"\\ud800"`` as a Python string, while ``str.encode("utf-8")`` cannot encode
  it. A surrogate is not text this build can round-trip, so it is refused by
  name at the boundary rather than at the encoder.
* **Cycles.** A YAML anchor or a hand-built mapping can point at itself.
* **Depth.** A structure nested past :data:`MAX_JSON_DEPTH` is refused before
  the interpreter's own stack runs out, so the failure is a stated limit rather
  than a ``RecursionError`` from whichever frame happened to be deepest.
* **Integer width.** CPython refuses to convert an integer of more than
  :data:`MAX_JSON_INTEGER_DIGITS` decimal digits to text, so a wide enough
  ``int`` makes ``json.dumps`` raise a bare ``ValueError`` — at the point a
  durable row was being written, in a type no caller of this module catches.
  It is refused as a stated limit instead.

Callers translate :class:`JsonSafetyError` into their own domain error, so the
operator sees "this scaffold" or "this ledger row" rather than a generic
encoding complaint. What they do not do is wrap a hash call in ``try/except
UnicodeEncodeError``: the check happens once, here, before anything encodes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

#: Maximum number of open JSON arrays and objects at any point in a persisted
#: document. Raw JSON is checked against this public limit before parsing, and
#: constructed values are checked before serialisation. No artefact this build
#: writes comes close.
MAX_JSON_DEPTH = 64

#: The most decimal digits an integer in a persisted payload may have.
#:
#: CPython's own limit on integer-to-text conversion, restated here because this
#: module is where the consequence lands: past it, ``str(value)`` and therefore
#: ``json.dumps`` raise ``ValueError``, and a boundary whose job is to say
#: whether a payload can be carried would be answering by raising the
#: interpreter's exception instead of its own. Every number this build persists
#: is orders of magnitude inside it — the token domain stops sixteen digits in —
#: so nothing real is refused by it.
MAX_JSON_INTEGER_DIGITS = 4300

#: The same limit as a magnitude: an integer converts to text exactly when its
#: absolute value is strictly below this. Obtained by arithmetic once, at
#: import, so the bound itself is never rendered either.
#:
#: This is the rule, not an approximation of it. Digits and bits do not divide
#: the integers at the same places, so a bound stated in bits alone must answer
#: the same way for two integers that share a bit length and straddle the digit
#: limit — and ``10 ** MAX_JSON_INTEGER_DIGITS``, the first integer the
#: interpreter refuses, is one of exactly that pair.
_MAX_JSON_INTEGER_MAGNITUDE = 10**MAX_JSON_INTEGER_DIGITS

#: The bit length of that bound, kept only as a fast path. Bit length is O(1)
#: and settles every integer except those sharing this many bits with the bound
#: itself; those are settled by comparing against the bound.
_MAX_JSON_INTEGER_BITS = _MAX_JSON_INTEGER_MAGNITUDE.bit_length()

#: The replacement written into *diagnostic* text that carries a lone surrogate.
#: Diagnostics are prose about a failure, not evidence of one, so they are
#: repaired rather than refused; see :func:`sanitized_text`.
REPLACEMENT_CHARACTER = "�"

_SURROGATE_LOW = 0xD800
_SURROGATE_HIGH = 0xDFFF

# Captured from the defining built-in rather than resolved through a caller's
# metaclass, whose data descriptors take precedence even for type.__getattribute__.
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]


def _type_mro(cls: type[Any]) -> tuple[type[Any], ...]:
    """Return ``cls``'s actual MRO without invoking metaclass descriptors."""
    return cast(tuple[type[Any], ...], _TYPE_MRO_DESCRIPTOR.__get__(cls, type))


class JsonSafetyError(ValueError):
    """A value cannot be carried by canonical UTF-8 JSON."""


class TextEncodingError(JsonSafetyError):
    """A string carries a lone UTF-16 surrogate, which UTF-8 cannot encode."""


class CyclicStructureError(JsonSafetyError):
    """A structure contains itself, so it has no finite serialisation."""


class NestingDepthError(JsonSafetyError):
    """A structure nests deeper than this build will serialise."""


class NonJsonValueError(JsonSafetyError):
    """A value is of a type JSON has no representation for."""


class UnrenderableValueError(JsonSafetyError):
    """A value is of a JSON type but cannot be rendered into JSON text.

    One shape reaches this: an integer wider than
    :data:`MAX_JSON_INTEGER_DIGITS` decimal digits, which the interpreter will
    not convert to text at all. It is a JSON number by type and not one by
    reality, which is why it needs a name of its own rather than
    :class:`NonJsonValueError`.

    The value is never quoted in the message, and for the one shape that raises
    it that is not a policy choice: rendering it is the operation that failed.
    """


def _json_key(value: Any) -> Any:
    """Return an accepted string key as an exact built-in without user hooks."""
    value_type = type(value)
    if value_type is str:
        return value
    for base in _type_mro(value_type):
        if base is str:
            return str.__str__(value)
    return value


def to_json(value: Any) -> Any:
    """Project a frozen structure back into detached plain dict/list built-ins.

    The result shares no mutable state with the source, so a caller may edit the
    payload freely without reaching back into a compiled object. Accepted scalar
    subclasses are reduced through base descriptors, never dynamic conversion
    hooks; unsupported values remain unsupported for the JSON-safety boundary to
    reject. Cycles and structures deeper than the shared JSON limit are refused
    during this projection itself.
    """
    return _to_json(value, set(), 0, "$", preserve_tuples=False, root_key_admission=None)


def to_json_preserving_tuples(value: Any) -> Any:
    """Detach a JSON value while preserving exact built-in tuples.

    Exact ``dict``, ``list``, and ``tuple`` containers retain those recursive
    public types and are always freshly allocated. Other ``Mapping`` and
    ``Sequence`` implementations retain :func:`to_json`'s canonical projection
    to built-in ``dict`` and ``list``. Scalar normalization, accepted wire
    values, cycle detection, and depth limits are otherwise identical to
    :func:`to_json`; in particular, this function does not make an unsupported
    value JSON-safe.
    """
    return _to_json(value, set(), 0, "$", preserve_tuples=True, root_key_admission=None)


def _to_json_preserving_tuples_with_root_key_admission(
    value: Any, root_key_admission: Callable[[str], None]
) -> Any:
    """Project once, admitting each normalized root key before its child."""
    return _project_mapping(
        value,
        set(),
        0,
        "$",
        preserve_tuples=True,
        root_key_admission=root_key_admission,
    )


def _to_json(
    value: Any,
    ancestors: set[int],
    depth: int,
    path: str,
    *,
    preserve_tuples: bool,
    root_key_admission: Callable[[str], None] | None,
) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value if type(value) is str else str.__str__(value)
    if isinstance(value, int):
        return value if type(value) is int else int.__int__(value)
    if isinstance(value, float):
        return value if type(value) is float else float.__float__(value)
    if isinstance(value, Mapping):
        return _project_mapping(
            value,
            ancestors,
            depth,
            path,
            preserve_tuples=preserve_tuples,
            root_key_admission=root_key_admission,
        )
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        marker = _enter_container(value, ancestors, depth, path, "sequence")
        try:
            items = (
                _to_json(
                    item,
                    ancestors,
                    depth + 1,
                    f"{path}[{index}]",
                    preserve_tuples=preserve_tuples,
                    root_key_admission=None,
                )
                for index, item in enumerate(value)
            )
            if preserve_tuples and type(value) is tuple:
                return tuple(items)
            return list(items)
        finally:
            ancestors.discard(marker)
    return value


def _project_mapping(
    value: Any,
    ancestors: set[int],
    depth: int,
    path: str,
    *,
    preserve_tuples: bool,
    root_key_admission: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Project an already-admitted mapping without repeating its shape check."""
    marker = _enter_container(value, ancestors, depth, path, "mapping")
    try:
        projected: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _json_key(key)
            if type(normalized_key) is not str:
                raise NonJsonValueError(
                    f"JSON projection: mapping key at {path} must be an exact string"
                )
            if root_key_admission is not None:
                root_key_admission(normalized_key)
            if normalized_key in projected:
                raise NonJsonValueError(
                    f"JSON projection: duplicate mapping key {normalized_key!r} at {path}"
                )
            child_path = f"{path}.{normalized_key}"
            projected[normalized_key] = _to_json(
                item,
                ancestors,
                depth + 1,
                child_path,
                preserve_tuples=preserve_tuples,
                root_key_admission=None,
            )
        return projected
    finally:
        ancestors.discard(marker)


def _enter_container(
    value: Any,
    ancestors: set[int],
    depth: int,
    path: str,
    container_kind: str,
) -> int:
    marker = id(value)
    if marker in ancestors:
        raise CyclicStructureError(
            "JSON projection: cyclic reference detected at "
            f"{path} in {container_kind}; a structure that contains itself has no "
            "serialisation"
        )
    if depth + 1 > MAX_JSON_DEPTH:
        raise NestingDepthError(
            f"JSON projection: nested deeper than {MAX_JSON_DEPTH} levels at {path}; "
            "a structure this build cannot serialise is refused rather than projected"
        )
    ancestors.add(marker)
    return marker


def ensure_raw_json_depth(
    text: str, context: str, *, max_depth: int = MAX_JSON_DEPTH
) -> str:
    """Return raw JSON text unless its array/object nesting exceeds ``max_depth``.

    This is deliberately not a JSON validator. It tracks only structural
    delimiters outside strings, including escaped quotes and backslashes, and
    leaves malformed syntax to the configured JSON parser and its domain error.
    Running it before parsing makes the depth refusal independent of interpreter
    recursion behavior.
    """
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
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise NestingDepthError(
                    f"{context} is nested too deeply to read; maximum JSON nesting "
                    f"depth is {max_depth}"
                )
        elif character in "]}" and depth:
            depth -= 1
    return text


def surrogate_index(text: str) -> int | None:
    """The position of the first lone surrogate in ``text``, or ``None``.

    Python strings are sequences of code points, so a surrogate that appears in
    one is unpaired by construction: there is no byte sequence UTF-8 could
    encode it as. Scanning is therefore a simple range test.
    """
    for index, character in enumerate(text):
        if _SURROGATE_LOW <= ord(character) <= _SURROGATE_HIGH:
            return index
    return None


def check_text(value: str, context: str) -> str:
    """Return ``value`` when UTF-8 can encode it; raise by name when it cannot."""
    position = surrogate_index(value)
    if position is not None:
        raise TextEncodingError(
            f"{context}: text carries a lone UTF-16 surrogate "
            f"(U+{ord(value[position]):04X}) at position {position}; UTF-8 cannot "
            "encode it, so it is refused rather than hashed or persisted"
        )
    return value


def sanitized_text(value: str) -> str:
    """Replace every lone surrogate with U+FFFD. For diagnostics only.

    The explicit rule: text that *describes* a failure — an exception message
    from a provider integration, say — must never be able to stop the failure
    being recorded. Such text is not evidence of what an adapter said, so the
    unencodable code points are replaced and the record is written. Text that
    *is* evidence is refused instead; the runner classifies it as malformed
    protocol data rather than quietly rewriting what a model produced.
    """
    if surrogate_index(value) is None:
        return value
    return "".join(
        REPLACEMENT_CHARACTER
        if _SURROGATE_LOW <= ord(character) <= _SURROGATE_HIGH
        else character
        for character in value
    )


def _renders_as_decimal_text(value: int) -> bool:
    """Whether this interpreter will convert ``value`` to decimal text.

    Measured, never rendered: rendering it is exactly what the interpreter
    refuses to do past the limit, so a check that formatted the value to decide
    whether it could be formatted would raise the failure it exists to report.

    Magnitude rather than value, because the sign is not part of the size the
    interpreter counts; and magnitude rather than bits alone, because bits are
    the coarser measure and the boundary has to fall exactly where the
    interpreter's does.
    """
    magnitude = int.__abs__(value)
    if int.bit_length(magnitude) < _MAX_JSON_INTEGER_BITS:
        return True
    return bool(magnitude < _MAX_JSON_INTEGER_MAGNITUDE)


def ensure_json_safe(value: Any, context: str, *, max_depth: int = MAX_JSON_DEPTH) -> Any:
    """Prove a value survives canonical UTF-8 JSON, and hand it back unchanged.

    Returns the original object rather than a copy: callers that want a detached
    projection already have :func:`~boundarybench.freezing.to_json`, and a check
    that silently rebuilt its input would make it impossible to check something
    in place.
    """
    _walk(value, context, set(), 0, max_depth)
    return value


def _walk(
    value: Any, context: str, ancestors: set[int], depth: int, max_depth: int
) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        text = value if type(value) is str else str.__str__(value)
        check_text(text, context)
        return
    if isinstance(value, int):
        integer = value if type(value) is int else int.__int__(value)
        if not _renders_as_decimal_text(integer):
            raise UnrenderableValueError(
                f"{context}: an integer of more than {MAX_JSON_INTEGER_DIGITS} "
                "decimal digits cannot be converted to text by this interpreter, "
                "so it can be neither serialised nor hashed. The value is not "
                "quoted here, because quoting it is the operation that fails"
            )
        return
    if isinstance(value, float):
        number = value if type(value) is float else float.__float__(value)
        if not math.isfinite(number):
            raise NonJsonValueError(
                f"{context}: {number!r} is not a finite number; JSON cannot "
                "represent NaN or Infinity"
            )
        return
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in ancestors:
            raise CyclicStructureError(
                f"{context}: cyclic reference detected in mapping; a structure that "
                "contains itself has no serialisation"
            )
        depth += 1
        if depth > max_depth:
            raise NestingDepthError(
                f"{context}: nested deeper than {max_depth} levels; a structure this "
                "build cannot serialise is refused rather than parsed"
            )
        ancestors.add(marker)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise NonJsonValueError(
                        f"{context}: mappings must use string keys, got {key!r} of "
                        f"type {type(key).__name__}"
                    )
                normalized_key = key if type(key) is str else str.__str__(key)
                check_text(normalized_key, f"{context}: key {normalized_key!r}")
                _walk(
                    item,
                    f"{context}.{normalized_key}",
                    ancestors,
                    depth,
                    max_depth,
                )
        finally:
            ancestors.discard(marker)
        return
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        marker = id(value)
        if marker in ancestors:
            raise CyclicStructureError(
                f"{context}: cyclic reference detected in sequence; a structure that "
                "contains itself has no serialisation"
            )
        depth += 1
        if depth > max_depth:
            raise NestingDepthError(
                f"{context}: nested deeper than {max_depth} levels; a structure this "
                "build cannot serialise is refused rather than parsed"
            )
        ancestors.add(marker)
        try:
            for index, item in enumerate(value):
                _walk(item, f"{context}[{index}]", ancestors, depth, max_depth)
        finally:
            ancestors.discard(marker)
        return
    raise NonJsonValueError(
        f"{context}: value of type {type(value).__name__} is not JSON-safe; use "
        "null, bool, int, finite float, string, list or string-keyed mapping"
    )


#: Fragments of CPython's own wording for refusing an integer-to-text
#: conversion — currently "Exceeds the limit (N digits) for integer string
#: conversion; use sys.set_int_max_str_digits() to increase the limit", the same
#: sentence on every version this build supports.
#:
#: Two independent fragments, matched case-insensitively, because the recognition
#: has to survive a rewording: the interpreter names the operation in one and the
#: knob that controls it in the other, and either alone identifies the failure.
#: Nothing here is matched against a payload value, so no provider-supplied text
#: can make an unrelated failure look like this one.
_INTEGER_TEXT_LIMIT_MARKERS = ("integer string conversion", "int_max_str_digits")


def _is_integer_text_limit_error(error: ValueError) -> bool:
    """Whether ``error`` is the interpreter refusing to render a wide integer.

    Read from the message because CPython gives this failure no type of its own:
    it is a bare ``ValueError``, indistinguishable by class from the encoder's
    circular-reference refusal.
    """
    message = str(error).casefold()
    return any(marker in message for marker in _INTEGER_TEXT_LIMIT_MARKERS)


def canonical_json_text(payload: Any, context: str) -> str:
    """The one canonical encoding, taken only after the payload is proven safe.

    Sorted keys and compact separators make the encoding independent of
    authoring order; ``ensure_ascii`` stays off so the bytes are real UTF-8
    rather than escapes, which is exactly why the surrogate check has to happen
    first; ``allow_nan`` stays off so a value JSON cannot represent is never
    hashed as if it could.

    One ``ValueError`` out of the encoder is translated: the interpreter's
    refusal to convert a wide integer to text. Every caller of this function
    catches :class:`JsonSafetyError` and rewrites it into its own domain error —
    "this ledger row", "this scaffold" — and that bare ``ValueError`` walks past
    all of them to surface as a traceback from whichever writer happened to be
    running. :func:`ensure_json_safe` refuses such an integer first, so what is
    left here is the payload that was not the payload the walk proved: a mapping
    whose contents are computed rather than stored can differ between the two
    reads.

    Every other ``ValueError`` propagates as the object the encoder raised. A
    structure that became circular after the walk proved it acyclic raises one,
    and reporting that as an integer too wide to render would name a cause that
    is not there — sending an operator to look for a number in a row whose
    actual fault is a cycle. The translation names one failure, so it recognises
    that one failure.

    ``TypeError`` is deliberately *not* caught. A value of a type JSON cannot
    carry is already refused by name before the encoder runs, so what is left for
    the encoder to complain about is a mistake in this build — and dressing one
    up as a boundary refusal would report a payload as untrustworthy when the
    untrustworthy thing was the code.
    """
    ensure_json_safe(payload, context)
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        if not _is_integer_text_limit_error(exc):
            raise
        raise UnrenderableValueError(
            f"{context}: this build could not render a value it had already "
            "proven safe into canonical JSON text. The value is not quoted here: "
            "it is the value rendering failed on, and a diagnostic that tried to "
            "include it would fail the same way"
        ) from exc


def canonical_json_bytes(payload: Any, context: str) -> bytes:
    """The exact bytes every content digest in this build is taken over."""
    return canonical_json_text(payload, context).encode("utf-8")


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
