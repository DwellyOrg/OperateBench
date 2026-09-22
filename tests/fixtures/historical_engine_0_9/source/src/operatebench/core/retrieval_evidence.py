"""The neutral ground a retrieved record's evidence is computed on.

Two parties need the same two answers about one served read, and neither may be
the authority for the other.

The **broker** — the domain that answered the read — states a version and hands
back a body. The **evaluator** — which is handed a trajectory by someone it has
no reason to trust — has to decide whether that version is the version those
records have. If the evaluator asked the broker, it would be asking the party
whose claim is under examination; if it reimplemented the digest, the two would
drift and the check would quietly become "these two algorithms agree", which is
a different claim and a weaker one.

So the algorithm lives here, in Core, knowing nothing about any domain. A domain
supplies its own context label and version length and gets a version; the
evaluator supplies the same published constants and the row's own ``records``
and gets the version those records must carry. Neither computes anything, and
the row's stated version is an *output to compare against* rather than an input
to either side.

:func:`detached_records` is the other half of the same idea. A ledger row that
held a reference to the mapping the domain still owns would be a row whose
contents could move after it was written; a row whose body cannot be carried by
canonical UTF-8 JSON is a row no digest and no artefact can state. Both are
settled once, here, at the moment the row is built.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from boundarybench.jsonsafe import canonical_json_bytes, ensure_json_safe

#: Stable public name for the version mechanics implemented below. This names an
#: encoding contract; it is not a claim that a shortened content digest provides
#: authenticity or any other cryptographic guarantee.
PUBLIC_RECORD_VERSION_ALGORITHM = "canonical-json-sha256-prefix-v1"


def detached_records(records: Mapping[str, Any], context: str) -> dict[str, Any]:
    """A plain-builtin copy of one served projection, proven JSON-safe.

    Detached because a durable row must not share mutable state with the live
    operation: the record a decision rested on is what was true when it was
    served, and a row that could be edited afterwards by the state moving is not
    evidence of anything. Proven safe because the row is hashed into the
    trajectory digest and written into an artefact, and a value canonical JSON
    cannot carry has to be refused where it entered rather than at the encoder.
    """
    body = _plain(records)
    ensure_json_safe(body, context)
    assert isinstance(body, dict)
    return body


def _plain(value: Any) -> Any:
    """Rebuild a value out of plain ``dict``/``list``/scalars, recursively."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    ):
        return [_plain(item) for item in value]
    return value


def public_record_version(
    records: Mapping[str, Any], *, context: str, length: int
) -> str:
    """The version a projection carries: a digest of exactly what it says.

    ``context`` is the domain's published label for the encoding, and ``length``
    is how many hex characters of the digest that domain states. Both are the
    domain's to publish and this function's to *apply*, which is what keeps one
    algorithm between the party that answers a read and the party that checks it.
    """
    return hashlib.sha256(canonical_json_bytes(records, context)).hexdigest()[:length]


__all__ = [
    "PUBLIC_RECORD_VERSION_ALGORITHM",
    "detached_records",
    "public_record_version",
]
