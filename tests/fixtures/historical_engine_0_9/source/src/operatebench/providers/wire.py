"""The exact JSON a provider sent, read before its own SDK's reading of it.

Every adapter in this build takes two things out of a provider response: one
action, and the two token counts a turn is priced and capped on. Until now both
were read off the SDK's *typed* model, and a typed model is not a transcript of
the wire. Three distinct things happen between the bytes and the object, and
each of them is invisible from the far side:

* **coercion.** ``"prompt_tokens": true`` reaches the model as ``1`` and
  ``"12"`` reaches it as ``12``. Both are then non-negative integers, both pass
  every check downstream, and neither is a measurement anybody took.
* **dropping.** One SDK in this build discards a field its own schema does not
  declare *before* the response object exists, so an undeclared field — a
  surface that moved under a pinned SDK, or content that reached the body from
  somewhere other than the model — cannot be refused by inspecting the object.
* **leniency.** A union resolves to whatever arm fits, or to nothing, so a
  ``function`` that is a list survives the parse and fails several frames later
  as an ``AttributeError`` carrying the interpreter's words, outside the fault
  taxonomy and after the attempt has already been published as a success.

So the wire is checked here first, on the bytes the provider actually sent, and
only then is the SDK's object trusted for anything. An SDK-coerced value is not
evidence of a wire type.

Everything this module raises is
:data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`, and everything
it raises is a fixed sentence plus a count or a closed-set label this build
chose. Nothing that arrived in the body is quoted — not a field name, not a
value, not the text that failed to decode — because a durable failure row
carries only this build's own detail, and a response body is
provider-controlled text of unbounded shape.

Nothing here opens a socket or imports a provider SDK.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeGuard, TypeVar, cast

from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)
from operatebench.providers.response import MAX_RESPONSE_DEPTH
from operatebench.providers.usage import MAX_EXACT_TOKEN_COUNT, TokenUsage

#: How deep this module will walk a decoded body before refusing it as one it
#: cannot check. The same limit, and the same argument, as
#: :data:`~operatebench.providers.response.MAX_RESPONSE_DEPTH`: finite, so a body
#: built to recurse forever fails as a stated limit rather than as a
#: ``RecursionError`` from whichever frame happened to be deepest.
MAX_WIRE_DEPTH = MAX_RESPONSE_DEPTH

#: Provider response identifiers are correlation evidence, not model output. A
#: kilobyte is ample for documented identifiers and keeps both validation and
#: hashing bounded. The byte ceiling is stated separately so a maximum-length
#: Unicode string cannot multiply the hashing work without limit.
MAX_PROVIDER_RESPONSE_ID_CHARACTERS = 1024
MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES = 4096

#: One provider's parsed response object, whatever its SDK calls it.
Parsed = TypeVar("Parsed")


def wire_invalid(detail: str) -> AdapterProviderError:
    """The one refusal this module raises, under the one fault it raises it as.

    A response-invalid provider fault rather than a protocol failure, and that
    distinction is load-bearing everywhere it is used: nothing about the
    *model's* behaviour is wrong when a body does not state the contract this
    build asked for. So the attempt is recorded as one that received a response,
    its usage is left unmeasured rather than zero, and the reservation that
    authorised the request stays as exposure.
    """
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


class _DuplicateWireKeyError(ValueError):
    """A JSON object on the wire named the same key twice, so it has two readings.

    Private, and never raised past :func:`wire_json_object`: it exists to tell
    one decoding failure from another inside a single ``json.loads`` call.
    """


def _object_with_unique_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """One JSON object, refusing the ones that name a key twice.

    ``json.loads`` keeps the last value for a repeated key and says nothing.
    On a response body that is a real ambiguity: ``{"prompt_tokens": 1,
    "prompt_tokens": 900000}`` is one usage block that two readers price two
    ways, and whichever this decoder happened to keep would become the run's
    measurement.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateWireKeyError("a JSON object names the same key twice")
        seen[key] = value
    return seen


def _check_depth(value: Any, depth: int) -> None:
    """Refuse a decoded body this module would not finish walking."""
    if depth > MAX_WIRE_DEPTH:
        raise wire_invalid(
            "the response nests deeper than this build will read before deciding "
            "whether it is the response contract it asked for, so it is refused "
            f"rather than parsed; the limit is {MAX_WIRE_DEPTH} levels"
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


def wire_json_object(payload: str, *, kind: str) -> Mapping[str, Any]:
    """The body the provider sent, decoded strictly, or a refusal.

    Strict about three things a lenient decode would let past: text that is not
    decodable JSON at all, a top-level value that is not a JSON object, and an
    object anywhere in the body that names the same key twice.

    ``RecursionError`` is caught beside ``ValueError`` because the decoder
    reaches its own limit before this module's: either way the answer is that
    the body is not one this build will read, and it must be said as a named
    fault rather than as a stack overflow escaping the turn.
    """
    try:
        decoded: Any = json.loads(payload, object_pairs_hook=_object_with_unique_keys)
    except _DuplicateWireKeyError as exc:
        raise wire_invalid(
            f"the {kind} carries a JSON object that names the same key more than "
            "once, so it does not state one value for it and this build will not "
            "pick the winner. The keys and values are not recorded here: they are "
            "provider-controlled and a durable failure row records only this "
            "build's own fixed detail and closed-set values"
        ) from exc
    except (ValueError, RecursionError) as exc:
        raise wire_invalid(
            f"the {kind} is not decodable JSON, so what the provider sent cannot "
            "be determined at all. The text that failed to decode is not recorded "
            "here: it is provider-controlled and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise wire_invalid(
            f"the {kind} is not a JSON object, so it states none of the fields "
            "this API documents and there is nothing in it this build can read"
        )
    _check_depth(decoded, 0)
    return decoded


@dataclass(frozen=True)
class WireShape:
    """One documented JSON object: what it may state, and what it must.

    ``allowed`` is the closed set of fields this build claims the object can
    carry, and anything outside it fails closed. It is taken from the pinned
    SDK's own model for that object — see :func:`declared_wire_fields` — rather
    than hand-listed, because a hand-listed set drifts from the schema it claims
    to transcribe and the direction it drifts in is the silent one: a field the
    vendor documents and this build forgot would refuse every real response.

    ``required`` is the smaller set this build actually reads. A documented
    field that is absent is only a failure when something here was going to read
    it — which is exactly what makes a missing token count a refusal and a
    missing ``incomplete_details`` an ordinary absence.
    """

    kind: str
    allowed: frozenset[str]
    required: tuple[str, ...]


def declared_wire_fields(model: type) -> frozenset[str]:
    """Every field one pinned SDK response model declares, by its wire name.

    The alias where there is one, because the alias is the name the *wire* uses
    and the attribute name is the SDK's convenience. A model with no declared
    fields is refused rather than silently producing an empty allowlist, which
    would make every field on that object undeclared and every real response
    invalid.
    """
    fields = getattr(model, "model_fields", None)
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(
            f"{model!r} declares no response fields this build can read a wire "
            "contract from; the allowed field set of a documented object is taken "
            "from the pinned SDK's own model for it, never hand-listed"
        )
    return frozenset(
        str(getattr(info, "alias", None) or name) for name, info in fields.items()
    )


def checked_wire_mapping(value: Any, *, kind: str) -> Mapping[str, Any]:
    """Prove a node the API documents as an object is one, before it is read."""
    if not isinstance(value, Mapping):
        raise wire_invalid(
            f"the response does not carry a {kind} that is a JSON object, so there "
            "is nothing in it this build can read. What arrived in that field is "
            "not recorded here: it is provider-controlled and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    return value


def checked_wire_object(value: Any, shape: WireShape) -> Mapping[str, Any]:
    """Prove one documented object states exactly what it is allowed to state.

    Three refusals, and they are three different failures. The value is not an
    object at all; it carries a field neither the vendor's schema nor this
    integration can describe; or it omits one this build was about to read. Only
    the *count* of undeclared fields is recorded, never their names: they are
    provider-controlled text, and the count is this build's own arithmetic.
    """
    body = checked_wire_mapping(value, kind=shape.kind)
    undeclared = sum(1 for key in body if str(key) not in shape.allowed)
    if undeclared:
        raise wire_invalid(
            f"the {shape.kind} carries {undeclared} field(s) that the provider's "
            "own published response schema, as this pinned SDK version models it, "
            "does not declare. This build accepts one action and two token counts "
            "out of this body, so a part of it that neither the SDK nor this "
            "integration can describe is refused whole rather than ignored: the "
            "safe readings of an undeclared field are a surface that moved under a "
            "pinned SDK and content that did not come from the model. The field "
            "names and values are not recorded here: they are provider-controlled "
            "text, and a durable failure row records only this build's own fixed "
            "detail and closed-set values"
        )
    missing = sum(1 for field in shape.required if field not in body)
    if missing:
        raise wire_invalid(
            f"the {shape.kind} omits {missing} of the {len(shape.required)} field(s) "
            "this build reads out of it, so the body does not state the contract "
            "this run asked for and there is nothing here to read it as. Which "
            "fields were absent is this build's own closed-set detail and is "
            "recorded as a count; nothing the provider sent is recorded"
        )
    return body


def checked_wire_list(value: Any, *, kind: str) -> Sequence[Any]:
    """Prove the response's own collection of answers is a JSON array."""
    if not isinstance(value, list):
        raise wire_invalid(
            f"the response does not carry a list of {kind}, so it does not state "
            "the shape this API documents and there is nothing in it this build "
            "can read as the action the model took. What arrived in that field is "
            "not recorded here: it is provider-controlled and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    return value


def checked_wire_string(value: Any, *, kind: str) -> str:
    """Prove a field the API documents as text is text.

    A ``bool`` is refused explicitly even though it is not a ``str`` in Python,
    for the same reason it is refused as a token count: a lenient reader
    somewhere downstream will render it as one, and a tool name of ``true``
    names no action any scaffold declares.
    """
    if not isinstance(value, str):
        raise wire_invalid(
            f"the response states a {kind} that is not a JSON string, so it does "
            "not name what this API documents there. The value is not recorded "
            "here: it is provider-controlled and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        )
    return value


def is_wire_token_count(value: Any) -> TypeGuard[int]:
    """Whether a value on the wire is a token count this build can price on.

    Exactly a non-negative JSON integer. ``None`` is *not* a count here, which is
    the difference between this and
    :func:`~operatebench.providers.usage.is_token_count`: that one answers
    "did the SDK report it", and a null there is an honest absence. This one
    answers "did the provider state it on the wire", and a null there is a
    required field the body did not send.

    ``bool`` is refused although Python makes it an ``int``: ``true`` tokens is
    not a measurement, and a JSON boolean is not a JSON number.

    The upper bound is :data:`~operatebench.providers.usage.MAX_EXACT_TOKEN_COUNT`, the
    one this build's prices and thresholds already state, imported rather than
    restated so the two cannot drift. It is the largest integer a JSON number
    carries exactly, which is precisely the property a value read *off the wire*
    has to have: a count that does not survive being written into a ledger row
    and read back is a count no later reader can re-derive the turn's cost from.
    A body may state any integer it likes — including one Python will not
    convert to text — so the bound is applied here, at the read, rather than
    discovered by whichever encoder later tried to render it.
    """
    return type(value) is int and 0 <= value <= MAX_EXACT_TOKEN_COUNT


def wire_token_usage(usage: Mapping[str, Any], *, fields: Sequence[str]) -> TokenUsage:
    """The two counts a turn is priced on, exactly as the wire stated them.

    Both must be present and both must be exact non-negative integers inside the
    domain a JSON number carries unchanged. Every other shape — absent, null,
    boolean, string, float, negative, past the exact bound — is refused,
    and refused as the whole response rather than as an unmeasured turn: this
    build multiplies these counts by the run's pinned rates and commits the
    product against its cost cap, so a count it cannot verify is not a count it
    may bank at any value, including zero.

    ``fields`` is given by the caller because the surfaces disagree about what
    the same two numbers are called, and a reader that proved one pair present
    and then read the other would have proven nothing.
    """
    counts: list[int] = []
    for field in fields:
        value = usage.get(field)
        if is_wire_token_count(value):
            counts.append(value)
        else:
            raise wire_invalid(
                "the response's usage block states a token count that is not one "
                "on the wire: this build prices a turn by multiplying the "
                "provider's counts by the run's pinned rates and commits the "
                "product against the run's cost cap, so each count it reads must be "
                "present and must be an exact non-negative whole number no larger "
                "than the largest integer a JSON number carries exactly. A JSON "
                "boolean, string, fractional number or null is not one, whatever "
                "the SDK coerces it to. The value is not recorded here: it is "
                "provider-controlled, and a durable failure row records only this "
                "build's own fixed detail and closed-set values"
            )
    return TokenUsage(counts[0], counts[1])


class WireResponse(Generic[Parsed]):
    """One provider answer, held as the bytes that arrived beside the SDK's reading.

    Both halves are computed lazily and remembered, and the order they are
    consumed in is the point. The raw JSON is decoded and checked *first*, so a
    body that is not the contract this build asked for is refused before the
    SDK's object is trusted for a token count or an action. The typed reading is
    then produced from the same bytes, by the real SDK, because the action still
    has to be read out of a real response model rather than out of a mapping
    this build hand-walked.

    The text is held for the duration of one turn and is never written anywhere:
    it is a provider-controlled body, and nothing durable in this build carries
    one.
    """

    __slots__ = (
        "_kind",
        "_parse",
        "_parsed",
        "_parsed_ready",
        "_pre_json_check",
        "_pre_json_checked",
        "_response_id_admitted",
        "_response_id_digest",
        "_text",
        "_wire",
    )

    def __init__(
        self,
        text: str,
        parse: Callable[[], Parsed],
        *,
        kind: str,
        pre_json_check: Callable[[str], None] | None = None,
    ) -> None:
        self._text = text
        self._parse = parse
        self._kind = kind
        self._wire: Mapping[str, Any] | None = None
        self._parsed: Any = None
        self._parsed_ready = False
        self._pre_json_check = pre_json_check
        self._pre_json_checked = False
        self._response_id_admitted = False
        self._response_id_digest: str | None = None

    def __repr__(self) -> str:
        """Names the kind of body, never a byte of it."""
        return f"<WireResponse {self._kind}>"

    @property
    def wire(self) -> Mapping[str, Any]:
        """The exact JSON object the provider sent, decoded strictly."""
        if not self._pre_json_checked and self._pre_json_check is not None:
            self._pre_json_check(self._text)
            self._pre_json_checked = True
        if self._wire is None:
            self._wire = wire_json_object(self._text, kind=self._kind)
        return self._wire

    @property
    def parsed(self) -> Parsed:
        """The SDK's typed reading of the same bytes.

        A failure to produce one is a response-invalid fault and not an adapter
        crash: the body arrived, and what could not be done with it is this
        build's own parsing step. ``ValueError`` covers a pydantic validation
        error and a JSON decode error alike; ``TypeError`` covers a shape the
        model's constructor could not take at all.
        """
        if not self._parsed_ready:
            try:
                self._parsed = self._parse()
            except (ValueError, TypeError) as exc:
                raise wire_invalid(
                    f"the {self._kind} arrived but this run's pinned SDK could not "
                    "read it as the response model this API documents, so there is "
                    "no typed answer here to take an action out of. The parser's "
                    "own message is not recorded here: it quotes the body, which is "
                    "provider-controlled text"
                ) from exc
            self._parsed_ready = True
        return cast("Parsed", self._parsed)

    def admit_response_id(self, *, field: str = "id") -> None:
        """Validate and bind a provider identifier before usage settlement.

        Absence and the empty string intentionally carry no digest. Otherwise the
        raw JSON and SDK model must state the same built-in string, it must be
        Unicode-scalar-safe, and it must fit both finite limits above. Refusal
        details never quote the identifier, parser text, or encoding exception.
        """
        self._response_id_admitted = False
        self._response_id_digest = None
        raw_wire = self.wire
        raw = raw_wire.get(field)
        typed = getattr(self.parsed, field, None)
        if field not in raw_wire:
            # SDKs may synthesize defaults that never appeared on the wire. Only
            # the two deliberate no-evidence states may agree with raw absence;
            # a synthesized non-empty identifier is otherwise unbound evidence.
            if typed is not None and not (type(typed) is str and not typed):
                raise wire_invalid(
                    "the pinned SDK supplied a non-empty response identifier that "
                    "was absent from the JSON on the wire, so it is not accepted or "
                    "recorded as correlation evidence"
                )
            self._response_id_admitted = True
            return
        if type(raw) is not str:
            raise wire_invalid(
                "the response identifier is not a JSON string, so it cannot be "
                "used as bounded durable correlation evidence; its value is not "
                "recorded"
            )
        if len(raw) > MAX_PROVIDER_RESPONSE_ID_CHARACTERS:
            raise wire_invalid(
                "the response identifier exceeds this build's character limit, so "
                "it is refused rather than hashed without a bound; its value is not "
                "recorded"
            )
        try:
            encoded = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise wire_invalid(
                "the response identifier is not Unicode scalar text, so it cannot "
                "be encoded as durable correlation evidence; its value and the "
                "encoding exception are not recorded"
            ) from exc
        if len(encoded) > MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES:
            raise wire_invalid(
                "the response identifier exceeds this build's UTF-8 byte limit, so "
                "it is refused rather than hashed without a bound; its value is not "
                "recorded"
            )
        if type(typed) is not str or typed != raw:
            raise wire_invalid(
                "the pinned SDK's response identifier does not exactly match the "
                "JSON string on the wire, so neither reading is accepted or recorded"
            )
        self._response_id_digest = (
            None if not encoded else hashlib.sha256(encoded).hexdigest()
        )
        self._response_id_admitted = True

    @property
    def response_id_digest_sha256(self) -> str | None:
        """The digest admitted before settlement, never a late read of the body."""
        return self._response_id_digest if self._response_id_admitted else None


__all__ = [
    "MAX_PROVIDER_RESPONSE_ID_CHARACTERS",
    "MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES",
    "MAX_WIRE_DEPTH",
    "WireResponse",
    "WireShape",
    "checked_wire_list",
    "checked_wire_mapping",
    "checked_wire_object",
    "checked_wire_string",
    "declared_wire_fields",
    "is_wire_token_count",
    "wire_invalid",
    "wire_json_object",
    "wire_token_usage",
]
