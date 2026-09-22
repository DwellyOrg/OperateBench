"""Reading the arguments of one tool call, strictly, for whichever track asked.

Every tool-calling API in this build delivers a call's arguments as *text the
model wrote*, and every track in this build turns that text into a durable
record — a Boundary ledger row, a Lifecycle decision. The rules for reading it
are therefore not a vendor's and not a track's: they are this benchmark's
statement about what counts as one unambiguous answer, and they are written once
here so the two tracks cannot drift into reading the same bytes two ways.

The rules themselves are in :func:`decode_tool_arguments`. What is worth naming
up here is the one that is easy to miss: a JSON object that names the same key
twice is *refused* rather than resolved. ``json.loads`` keeps the last value and
says nothing, which turns a real ambiguity into a silent choice made by a
decoder, and the durable row this build would write would record only the
winner.

The exception a refusal is raised as is the caller's, passed in as ``error``.
That is the same shape every other check in this kernel takes — see
:func:`~operatebench.providers.config.check_endpoint` — and for the same reason:
each track has its own exception taxonomy, and a kernel that raised into one of
them could not be composed by the other.

The refusal messages are the ones the Boundary Track shipped, word for word.
Two of them say "action" and "ledger row", which are that track's words rather
than this module's; they are kept because this lift is required to be
behaviour-identical for a track whose runs are already on disk, and a message an
operator has read before is not worth re-wording for tidiness.

Nothing here opens a socket, imports a provider SDK or reaches into a track.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from operatebench.jsonsafe import JsonSafetyError, ensure_json_safe


class _DuplicateJsonKeyError(ValueError):
    """A JSON object sent the same key twice, so it has two readings.

    Private, and never raised past :func:`decode_tool_arguments`: it exists to
    tell one decoding failure from another inside a single ``json.loads`` call,
    not to be part of any track's exception taxonomy.
    """


def _object_with_unique_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """One JSON object, refusing the ones that name a key twice.

    ``json.loads`` keeps the last value for a repeated key and says nothing.
    That is a real ambiguity rather than a pedantic one: ``{"fact":"a",
    "fact":"b"}`` is one call that two readers resolve two ways, and the durable
    record this build would write records only the winner — so the evidence
    would state an argument the model may not have meant and no reader could
    tell. Every nested object is checked, because this hook runs on each one.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateJsonKeyError("a JSON object names the same key twice")
        seen[key] = value
    return seen


def decode_tool_arguments(
    raw: Any, *, label: str, error: type[Exception]
) -> dict[str, Any]:
    """The arguments one tool call carries, decoded strictly or refused.

    Strictly, because every one of these APIs delivers the arguments as *text
    the model wrote* and this build turns them into a durable record. Four ways
    that text can fail to name one unambiguous answer are refused here, by name:

    * **it is not decodable JSON** — what the model asked for cannot be
      determined at all;
    * **an object names a key twice** — two readers resolve it two ways, and the
      record would state whichever one this decoder happened to keep;
    * **it is not a JSON object with string keys** — a call's arguments are a
      mapping, and a list or a bare string is not one;
    * **it carries a value canonical UTF-8 JSON cannot round-trip** — a lone
      surrogate, a NaN, a structure nested past this build's stated limit. Such
      a value cannot be written as evidence of what the model said, and a record
      that dropped or repaired it would claim the model said something else.

    What is *not* decided here is whether the arguments suit the thing they were
    sent to. Each track already checks a call against its own declared schema —
    required arguments, unknown arguments, declared types, closed-set values —
    and doing that twice would put one rule in two places that can drift. An
    argument this decoder accepts may still be refused there, and a call carrying
    an argument no schema declared is expected to travel this far and be refused
    as a recorded failure rather than be dropped silently by the transport.

    ``raw`` may be the JSON *string* these APIs normally send or an
    already-decoded mapping — one SDK in this build types that field as either.

    Every message raised is a fixed sentence. Nothing that arrived in the
    response is quoted: not a key, not a value, not the text that failed to
    decode.
    """
    if isinstance(raw, str):
        try:
            arguments: Any = json.loads(raw, object_pairs_hook=_object_with_unique_keys)
        except _DuplicateJsonKeyError as exc:
            raise error(
                "the call's arguments name the same key more than once, so they do "
                "not state one value for it and this build will not pick the "
                "winner. The keys and values are not quoted here: they are "
                "model-authored text and a durable failure row records only "
                "stable, closed-set detail"
            ) from exc
        except (ValueError, TypeError) as exc:
            raise error(
                "the call's arguments are not decodable JSON, so what the model "
                "asked for cannot be determined. The text it produced is not "
                "quoted here: it is model-authored and a durable failure row "
                "records only stable, closed-set detail"
            ) from exc
    else:
        arguments = raw
    if not isinstance(arguments, Mapping) or not all(
        isinstance(key, str) for key in arguments
    ):
        raise error(
            "the call did not carry a JSON object with string keys as its "
            "arguments, so it names no action this build can record"
        )
    decoded = dict(arguments)
    try:
        ensure_json_safe(decoded, label)
    except JsonSafetyError as exc:
        raise error(
            "the call's arguments carry a value this build cannot record in a "
            "ledger row, so the answer cannot be preserved as evidence of what "
            "the model said. The offending value is not quoted here"
        ) from exc
    return decoded


__all__ = ["decode_tool_arguments"]
