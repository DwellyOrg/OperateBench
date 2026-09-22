"""Proving an SDK's typed reading of a body is one this build can describe.

This is the *typed* half of the response contract; the exact-JSON half is
:mod:`operatebench.providers.wire`, and it runs first. What is left for here is
what only the parsed object can answer: which fields the vendor's own schema
declares, whether an SDK resolved a union leniently into something an attribute
read would crash on, and whether the body carries a field neither the SDK nor
the integration can describe.

Everything raised is
:data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`, and
everything raised is a fixed sentence plus a count or a closed-set label this
build chose. Nothing that arrived in the body is quoted.

Nothing here opens a socket or imports a provider SDK.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)

#: How deep :func:`check_response_contract` will walk a parsed response before
#: refusing it as one this build cannot check. Far deeper than any response
#: shape these APIs document, and finite so a body built to recurse forever
#: fails as a stated limit rather than as a ``RecursionError`` from whichever
#: frame happened to be deepest. The same argument :mod:`operatebench.jsonsafe`
#: makes for its own depth limit.
MAX_RESPONSE_DEPTH = 32

#: The default nested-extension allowance: nothing, on nothing. Shared and
#: immutable, because a default argument that could be mutated by one caller
#: would widen what every other lane in this build accepts.
_NO_MODEL_EXTRAS: Mapping[type, frozenset[str]] = MappingProxyType({})


def _declared_fields(value: Any) -> Mapping[str, Any] | None:
    """One parsed response object's declared fields, or ``None`` if it is not one."""
    fields = getattr(type(value), "model_fields", None)
    if not isinstance(fields, Mapping):
        return None
    return {name: getattr(value, name, None) for name in fields}


def _unknown_field_count(
    value: Any,
    depth: int,
    allowed: frozenset[str] = frozenset(),
    by_model: Mapping[type, frozenset[str]] = _NO_MODEL_EXTRAS,
) -> int:
    """How many fields the provider added that its own SDK does not declare.

    Recursive on purpose, and over the *parsed* objects rather than over the raw
    body. These SDKs build response models that accept unknown fields and park
    them in ``model_extra``, so a field the vendor's own schema does not declare
    is preserved silently at every level: on the response, on an output item or
    a choice, on a tool call, and on the nested function object inside it. A
    check that looked only at the top level would pass a body whose action was
    carried by a tool call nobody can describe.

    ``allowed`` names the extras an integration has *already validated itself*
    against a named, versioned contract of its own, and it applies to this level
    only — the recursive calls below pass none. So an adapter that has proven a
    top-level block against a fixed shape can say so here without widening what
    is accepted anywhere else in the body, and an integration that names none
    (which is every other one in this build) counts every extra exactly as
    before.

    ``by_model`` is the same permission for an extension that does *not* live at
    the root. It is keyed by the pinned SDK's own response model for the object
    that may carry it, so what an integration widens is "this named field, on
    this documented object" rather than "this name, anywhere in the body" — the
    latter would let a nested object smuggle in a field under a name proven
    somewhere else entirely. It travels down the recursion because the objects
    it names are nested by construction, and it is empty by default, so an
    integration that names none counts every extra at every depth exactly as
    before.
    """
    if depth > MAX_RESPONSE_DEPTH:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the response nests deeper than this build will read before deciding "
            "whether it is the response contract it asked for, so it is refused "
            f"rather than parsed; the limit is {MAX_RESPONSE_DEPTH} levels",
        )
    if isinstance(value, (str, bytes, int, float)) or value is None:
        return 0
    count = 0
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, Mapping):
        permitted = allowed | by_model.get(type(value), frozenset())
        count += sum(1 for key in extra if str(key) not in permitted)
    declared = _declared_fields(value)
    if declared is not None:
        for item in declared.values():
            count += _unknown_field_count(item, depth + 1, by_model=by_model)
        return count
    if isinstance(value, Mapping):
        for item in value.values():
            count += _unknown_field_count(item, depth + 1, by_model=by_model)
        return count
    if isinstance(value, Sequence):
        for item in value:
            count += _unknown_field_count(item, depth + 1, by_model=by_model)
    return count


def check_response_contract(
    response: Any,
    *,
    allowed_root_extras: frozenset[str] = frozenset(),
    allowed_extras_by_model: Mapping[type, frozenset[str]] = _NO_MODEL_EXTRAS,
) -> None:
    """Refuse a response carrying anything the provider's own schema omits.

    Fail-closed, and deliberately stricter than "ignore what you do not
    understand". This build accepts an action from a response and prices a turn
    from its usage block; a field neither the SDK's models nor this integration
    knows about is, by construction, a part of the answer this build cannot say
    anything about — and the two shapes it would most plausibly take are a
    surface that changed under a pinned SDK version and content that reached the
    response from somewhere other than the model. Both are cases where the safe
    reading of the body is "this is not the contract I asked for", and both are
    invisible to every other check here.

    Raised as
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`
    rather than as a protocol failure, and therefore *before* the action or the
    token counts are accepted: nothing about the model's behaviour is wrong
    here, and the attempt is still recorded as one that received a response,
    with its reservation kept as exposure rather than settled at a cost nobody
    measured.

    ``allowed_root_extras`` is the one narrow way out, and it is narrow on
    purpose. An integration may name the top-level extras it has *itself*
    validated against a fixed, versioned contract it publishes and hashes into
    run identity — see the OpenAI lane's
    :data:`~operatebench.providers.openai_responses.SERVER_EXTENSION_SCHEMA`.
    It defaults to nothing, it is never read from the response, and it applies
    to the root object only, so every other integration in this build — and
    every nested object in every body — still accepts exactly zero undeclared
    fields.

    ``allowed_extras_by_model`` is the same way out for a contract whose fields
    do not arrive at the root — see the xAI lane's
    :data:`~boundarybench.providers.xai_openai_compat.RESPONSE_EXTENSION_SCHEMA`,
    where the compatibility layer states one field on the assistant message, two
    on the usage block and a pair inside a detail object the SDK does declare.
    Keying it by the SDK model of the object entitled to carry each name is what
    keeps it as narrow as the root allowance: a name proven for the usage block
    buys nothing for a message, and both default to nothing.

    The number of unknown fields is recorded; their names and values are not.
    They are provider-controlled text of unbounded shape, and a durable failure
    row carries only this build's own fixed detail.
    """
    count = _unknown_field_count(
        response, 0, allowed_root_extras, allowed_extras_by_model
    )
    if count:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            f"the response carries {count} field(s) that the provider's own "
            "published response schema, as this pinned SDK version models it, "
            "does not declare. This build accepts one action and two token counts "
            "out of this body, so a part of it that neither the SDK nor this "
            "integration can describe is refused whole rather than ignored: the "
            "safe readings of an undeclared field are a surface that moved under "
            "a pinned SDK and content that did not come from the model. The field "
            "names and values are not recorded here: they are provider-controlled "
            "text, and a durable failure row records only this build's own fixed "
            "detail and closed-set values",
        )


def check_response_collection(value: Any, *, kind: str) -> None:
    """Prove the response's own collection of answers is one.

    These SDKs parse a response leniently — a ``null`` where the schema declares
    a list becomes ``None`` on the model rather than a validation error — so the
    first thing that would notice is whichever line iterated it, four frames
    down, as a ``TypeError`` carrying the SDK's own words. That failure is
    indistinguishable from this integration crashing, it escapes the turn
    without the attempt evidence being published, and the text it carries is not
    something this build chose.

    So the shape is checked here instead, at the same boundary that checks the
    response's stated model, and refused by name.
    """
    if not isinstance(value, list):
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            f"the response does not carry a list of {kind}, so it does not state "
            "the shape this API documents and there is nothing in it this build "
            "can read as the action the model took. What arrived in that field is "
            "not recorded here: it is provider-controlled and a durable failure "
            "row records only this build's own fixed detail and closed-set values",
        )


def check_response_object(value: Any, *, fields: Sequence[str], kind: str) -> None:
    """Prove a sub-object the API documents is one, before it is read.

    The companion to :func:`check_response_collection`, and it exists for the
    same leniency: where these SDKs' own models declare an object, a ``[]``, a
    ``"x"`` or a ``7`` on the wire is carried through as that value rather than
    refused, and a body that omits the object's fields entirely produces a model
    that has none of them. Nothing notices until an attribute is read off it,
    and what surfaces then is an ``AttributeError`` carrying the interpreter's
    words — raised from outside this contract's fault set, indistinguishable
    from this integration crashing, and, when the container is the usage block,
    raised *before* the attempt that received the body was ever published.

    So the object is proven here, by the fields the reader is about to use,
    rather than by its Python class: an SDK resolves a union leniently, so an
    ``isinstance`` would answer "the SDK fell back to a model" rather than "the
    body states what this API documents".

    Refused as
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID` and
    not as a protocol failure. A container the transport could not have produced
    from a documented body says nothing about how the *model* behaved, so the
    attempt is recorded as one that received a response, its usage is left
    unmeasured rather than zero, and the reservation that authorised the request
    stays as exposure.

    Neither the value nor the field names that were missing are recorded: they
    are provider-controlled, and a durable failure row carries only this build's
    own fixed detail and closed-set values.
    """
    if not all(hasattr(value, field) for field in fields):
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            f"the response does not carry a {kind} that states the fields this "
            "API documents, so there is nothing in it this build can read. What "
            "arrived in that field is not recorded here: it is provider-controlled "
            "and a durable failure row records only this build's own fixed detail "
            "and closed-set values",
        )


__all__ = [
    "MAX_RESPONSE_DEPTH",
    "check_response_collection",
    "check_response_contract",
    "check_response_object",
]
