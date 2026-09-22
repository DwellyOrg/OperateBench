"""The two counts a turn is priced on, and what one of them may be.

A provider reports usage and this build multiplies it by a pinned rate, commits
the product against a cap and writes it into durable evidence. So "is this a
token count" is a question with a fail-closed answer rather than a type hint, and
it is asked in exactly one place for every track and every provider.

Nothing here opens a socket or imports a provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)

#: The largest token count this build treats as exact.
#:
#: 2**53 - 1 is the largest integer that is exactly representable as an
#: IEEE-754 double, and therefore the largest one that survives a JSON number in
#: every reader this build's artefacts have to cross — which matters because
#: token counts and thresholds are written into durable rows and hashed policy
#: payloads that other languages read. It is also some nine orders of magnitude
#: above the largest published context window, so nothing real is refused by it.
MAX_EXACT_TOKEN_COUNT = 2**53 - 1


@dataclass(frozen=True)
class TokenUsage:
    """The two counts a turn is priced on, as this build read them.

    ``None`` on either is "the response did not report it", which is not the
    same claim as zero and is never rendered as one.
    """

    input_tokens: int | None
    output_tokens: int | None

    @property
    def measurable(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


def is_token_count(value: Any) -> bool:
    """Whether a provider-reported token count is one this build can price on.

    ``None`` is a count: it means "the response did not report it", which is not
    zero and is never rendered as one. Everything else must be a non-negative
    integer, and a ``bool`` is not one — Python makes ``True`` an ``int`` and
    ``True`` tokens is not a measurement anybody took.

    Bounded above as well as below, by :data:`MAX_EXACT_TOKEN_COUNT` — the same
    constant the price tables and the tier thresholds are bounded by, imported
    rather than restated. A count past it is not a measurement this build can
    carry: it does not survive a JSON number in the readers a durable row has to
    cross, so a cost derived from it could not be re-derived from the row that
    recorded it. Past that bound the failure also stops being arithmetic — a
    large enough integer cannot be converted to text at all, which turns the
    *serialiser* into the thing that fails, several layers below the reader that
    should have refused the value.
    """
    if value is None:
        return True
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_EXACT_TOKEN_COUNT
    )


def checked_token_usage(input_tokens: Any, output_tokens: Any) -> TokenUsage:
    """The two counts a turn is priced on, or a refusal of the whole response.

    The counts are the provider's own, they are multiplied by the run's pinned
    price, and the product is committed against the run's cost cap and written
    into durable evidence. A negative count would therefore *credit* the run
    against its own cap — the one direction a cost control must never move — and
    a ``bool`` would price a turn at one token. A count past the exactly
    representable domain is the third of the same kind: it prices a turn at a
    number the row recording it could not be read back as. None of them is a
    measurement, so none of them is allowed to become an
    :class:`~operatebench.providers.telemetry.AdapterUsage`, reach settlement or
    reach a durable row.

    Refused as
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`,
    which keeps the honest reading of what happened: a response *did* arrive, so
    the attempt records that it did, and the reservation that authorised the
    request stays as exposure rather than settling to a zero nobody measured.
    Invalid usage is not free usage.

    The offending value is not quoted. It is provider-controlled, and the
    classification — which of the two counts, and that it was unusable — is
    closed-set detail this build chose.
    """
    for label, value in (("input", input_tokens), ("output", output_tokens)):
        if not is_token_count(value):
            raise AdapterProviderError(
                PROVIDER_FAULT_RESPONSE_INVALID,
                f"the response reports a {label} token count that is not one: this "
                "build prices a turn by multiplying the provider's counts by the "
                "run's pinned rates and commits the product against the run's cost "
                "cap, so a count that is absent is null and a count that is present "
                "must be a non-negative whole number no larger than the largest "
                "integer a JSON number carries exactly. The value is not recorded "
                "here: it is provider-controlled text, and a durable failure row "
                "records only this build's own fixed detail and closed-set values",
            )
    return TokenUsage(input_tokens, output_tokens)


__all__ = [
    "MAX_EXACT_TOKEN_COUNT",
    "TokenUsage",
    "checked_token_usage",
    "is_token_count",
]
