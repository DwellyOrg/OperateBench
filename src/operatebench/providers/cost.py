"""What a provider request costs a run, before and after it is made.

The retry loop in :mod:`operatebench.providers.executor` reserves an amount
before every request, resolves it afterwards, and records both on the attempt.
The *policy* it reserves against — the price table, the cap, the arithmetic — is
a track's own concern and stays there. What has to be here is the small vocabulary
the loop itself speaks: the shape of a guard, the two refusals it can raise, the
one bound it reserves against and the one way a USD amount is rendered.

Those four are here rather than in a track for the same reason the retry loop is:
a second definition of "the conservative bound" or "how a reservation renders"
would drift, and a drifted bound makes every stored reservation of an otherwise
correct run unre-derivable — a ledger that was written honestly and can no longer
be read.

The guard itself is a :class:`~typing.Protocol`. The kernel never constructs one
and never decides what a token costs; it asks whatever the track supplied.

Nothing here opens a socket or imports a provider SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Protocol

from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.faults import AdapterError


class BudgetError(ValueError):
    """The run's limits, or an operation on them, are not usable."""


class CostCapExceededError(AdapterError):
    """The remaining authorised budget cannot cover this request.

    An :class:`~operatebench.providers.faults.AdapterError` because it is raised
    at the adapter boundary, in place of the request that was not made — the
    runner classifies it there, records it durably and stops the run. It carries
    no provider text: every number in its message is this build's own arithmetic
    over its own pinned settings.
    """


class CostReservationBreachedError(AdapterError):
    """A response cost more than the reservation that authorised the request.

    The one case a per-request bound cannot rule out: the bound is computed from
    the serialised body and this run's pinned output ceiling, and the counts come
    back from the provider. If they exceed it, the request was already made and
    the tokens were already consumed, so there is no version of this that ends
    with the run having spent only what it authorised.

    What this build refuses to do is *absorb* it. The measured cost is banked at
    its measured value — hiding real spend is the failure that matters — and this
    is raised in place of the answer, so the turn is classified as a reservation
    breach, recorded durably, and the run stops instead of continuing under a cap
    it has already passed. The amounts are carried on the exception because the
    row that records it has to state all three. An optional settled aggregate
    controller instead rejects out-of-bound token claims as unmeasurable,
    durably retains the entire wire bound as unknown, and supplies
    ``measured_usd=None``. The executor then records forfeiture, not a measured
    settlement. The guard must close that reservation before raising.

    An :class:`~operatebench.providers.faults.AdapterError` for the same reason
    :class:`CostCapExceededError` is: it surfaces at the adapter boundary, where
    the runner classifies it. It carries no provider text — every number in it is
    this build's own arithmetic over its own pinned settings and pricing policy.
    """

    def __init__(
        self,
        message: str,
        *,
        measured_usd: Decimal | None,
        reservation_usd: Decimal,
        cap_usd: Decimal,
    ) -> None:
        super().__init__(message)
        self.measured_usd = measured_usd
        self.reservation_usd = reservation_usd
        self.cap_usd = cap_usd


#: Tokens added to every request's input bound on top of the serialised body.
#:
#: The body is the only thing this build can measure, and it is not everything
#: the provider counts: tool declarations and system content are re-framed
#: server-side in a form no client sees. A fixed, stated allowance is the honest
#: way to cover that — it is generous relative to the framing of a request this
#: scaffold produces, and it is a constant rather than a guess per request.
REQUEST_OVERHEAD_TOKEN_ALLOWANCE = 1024


def conservative_input_token_bound(serialised_request_bytes: int) -> int:
    """An upper bound on the input tokens one request can be charged for.

    Every token a byte-pair encoder produces consumes at least one byte of the
    UTF-8 it was built from, so the encoded body's byte length bounds its token
    count from above. :data:`REQUEST_OVERHEAD_TOKEN_ALLOWANCE` is added for the
    framing the body does not show. The result is deliberately loose: a bound
    that is occasionally far too large costs a run some head-room, and a bound
    that is once too small costs it the guarantee.
    """
    if type(serialised_request_bytes) is not int or serialised_request_bytes < 0:
        raise BudgetError(
            "a request's serialised size must be a non-negative integer number of "
            f"bytes, got {serialised_request_bytes!r}"
        )
    return serialised_request_bytes + REQUEST_OVERHEAD_TOKEN_ALLOWANCE


def usd_text(amount: Decimal) -> str:
    """One rendering for every USD amount this build reports.

    Fixed-point rather than scientific, and without the trailing fractional
    zeros decimal addition accumulates: summing twelve amounts of scale six
    yields a scale-six zero, and ``"0.000000"`` and ``"0"`` are the same number
    reported two ways. One rendering means two reports of the same spend compare
    equal as strings, which is what a reader and a test both need.
    """
    text = format(amount, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def request_input_token_bound(payload: Mapping[str, Any], label: str) -> int:
    """An upper bound on the input tokens one exact request can be charged for.

    Derived from the body the SDK will send, canonically encoded, because that
    is the only thing about the request this build can measure. Every token a
    byte-pair encoder emits consumes at least one byte of its UTF-8 source, so
    the encoded length bounds the count from above; the allowance
    :func:`conservative_input_token_bound` adds on top covers the provider-side
    framing of tools and system content that the body does not show. The result
    is used only to *refuse* requests, so being loose costs head-room and being
    tight would cost the guarantee.

    ``label`` is the caller's own name for the request, so a failure while
    encoding names the request that failed.
    """
    return conservative_input_token_bound(
        len(canonical_json_text(payload, label).encode("utf-8"))
    )


class CostGuard(Protocol):
    """What the retry loop needs of a run's cost control, and nothing more.

    Structural rather than a base class. The kernel authorises a request before it
    is made, then either cancels that reservation if a final local admission
    refuses before dispatch, settles it at measured cost after a response, or
    forfeits it as exposure once the request may have crossed the wire. It never
    reads a cap, a price table or a running total, because deciding what a token
    costs is the track's business and not the transport's.

    A guard is expected to raise :class:`CostCapExceededError` from
    :meth:`authorize` when the run's remaining authorised budget cannot cover the
    request, and :class:`CostReservationBreachedError` from :meth:`settle` when a
    measured cost exceeds the reservation that authorised it. Those are the two
    refusals the loop knows how to record.
    """

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        """Permit one provider request, or refuse it. Returns the reservation."""
        ...

    def settle(
        self, reservation: Decimal, *, input_tokens: int | None, output_tokens: int | None
    ) -> Decimal | None:
        """Replace an outstanding reservation with what the response reported."""
        ...

    def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
        """Release an outstanding reservation for a request never dispatched."""
        ...

    def forfeit(self, reservation: Decimal) -> None:
        """Keep an outstanding reservation as exposure: nothing was measured."""
        ...


__all__ = [
    "REQUEST_OVERHEAD_TOKEN_ALLOWANCE",
    "BudgetError",
    "CostCapExceededError",
    "CostGuard",
    "CostReservationBreachedError",
    "conservative_input_token_bound",
    "request_input_token_bound",
    "usd_text",
]
