"""What one turn asked of a provider, what it cost, and what each attempt met.

``AdapterUsage`` says what a turn cost. It cannot say what the turn *did*: a turn
that was rate limited twice and then answered reports exactly the same usage as
one that answered first time, and a turn that never got a body reported nothing
at all — which made it indistinguishable from a turn that never happened. So an
integration also reports structured evidence of its attempts.

Everything in these value objects is either a closed-set value
:mod:`operatebench.providers.faults` defines, a small integer, an exact decimal
string, or a locally measured duration. No provider message, header, body,
request id, credential text or exception string can reach any of them by
construction, because there is no field one could be put in.

None of it names a track. A deadline is wall-clock, a usage is two counts and a
price, an attempt is what one request met — and each of those means the same
thing whichever projection built the body.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from operatebench.providers.faults import (
    ATTEMPT_OUTCOMES,
    ATTEMPT_SETTLEMENTS,
    PROVIDER_FAULTS,
    TURN_TERMINAL_REASONS,
)


@dataclass(frozen=True)
class TurnDeadline:
    """How much of the episode budget is left when a call is dispatched.

    Deliberately *not* part of a track's request object. The request is the
    agent's view of the case; the deadline is an infrastructure parameter for the
    integration, and mixing the two would put a wall-clock fact into the thing
    that is supposed to contain only what an agent observed.

    This is the seam a real provider adapter needs. In-process Python cannot
    forcibly kill an arbitrary blocking callable, so a runner cannot interrupt a
    hung adapter from outside; what it can do is (a) hand every call the budget
    it must not exceed, so a network integration can set its own socket or stream
    timeout and poll :meth:`cancelled` between chunks, and (b) check the clock
    the instant the call returns, so an overrun is recorded as a timeout rather
    than a success. The contract ships the seam and post-return enforcement;
    forcible cancellation of an uncooperative in-process callable is out of scope
    and is not claimed.

    ``started_at`` is the *absolute* instant, on the same monotonic clock the
    executor reads, from which ``remaining_seconds`` was counted. It exists
    because a caller usually does real work — checking a request, building a
    body, checking the body against the settings, bounding the tokens — between
    deciding the budget and reaching the loop that spends it, and a loop that
    started its own stopwatch on entry would hand that work out of the budget
    for free. Stating the origin makes the two halves measure one interval:
    every attempt's timeout is what is left *of the turn*, not what is left of
    the loop.

    ``None`` is the honest "not stated", and it means exactly what this contract
    always meant — the budget begins when the loop does. A caller that states
    only ``remaining_seconds`` gets bit-for-bit the behaviour it got before this
    field existed, which is why it is optional rather than defaulted to a
    sentinel instant.
    """

    remaining_seconds: float
    cancelled: Callable[[], bool]
    started_at: float | None = None


@dataclass(frozen=True)
class AdapterUsage:
    """What one turn cost. Every field is nullable: a fake reports none of it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class ProviderAttempt:
    """One request to the provider, and what came back — or did not."""

    index: int
    outcome: str
    #: The named fault, on a failed attempt. ``None`` on one that got a response.
    fault: str | None = None
    #: The HTTP status, where there was one. ``None`` is "no status", which is
    #: what a dropped connection or a client-side timeout has, and is never 0.
    http_status: int | None = None
    #: Locally measured wall-clock for this attempt alone. ``None`` means the
    #: implementation did not measure it, which is not the same as instant.
    latency_seconds: float | None = None
    #: Whether an HTTP response reached the SDK, including a non-success response.
    response_received: bool = False
    #: Whether that body carried usage this build could read.
    usage_reported: bool = False
    #: The conservative upper bound the run's cost guard authorised for this
    #: attempt before it was made, as an exact decimal string in USD. ``None``
    #: means the run has no cost cap and so authorised no amount — never that it
    #: authorised nothing. A string rather than a JSON number for the reason
    #: every amount here is: a binary float is not the amount that was reserved.
    cost_reservation_usd: str | None = None
    #: How that reservation was resolved, from
    #: :data:`~operatebench.providers.faults.ATTEMPT_SETTLEMENTS`, or ``None``
    #: when there was no reservation to resolve. Recorded per attempt because an
    #: episode's exposure is the sum of the forfeited ones, and a total that
    #: cannot be re-derived from its own parts is a claim.
    cost_settlement: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in ATTEMPT_OUTCOMES:
            raise ValueError(
                f"{self.outcome!r} is not a provider attempt outcome this contract "
                f"defines; allowed: {list(ATTEMPT_OUTCOMES)}"
            )
        if self.fault is not None and self.fault not in PROVIDER_FAULTS:
            raise ValueError(
                f"{self.fault!r} is not a provider fault this contract defines; "
                f"allowed: {list(PROVIDER_FAULTS)}"
            )
        if (self.cost_reservation_usd is None) != (self.cost_settlement is None):
            raise ValueError(
                "an attempt states a reservation and how it was resolved together "
                "or states neither; a reservation with no settlement is money this "
                "run cannot account for, and a settlement with no reservation "
                f"resolves nothing. Got {self.cost_reservation_usd!r} and "
                f"{self.cost_settlement!r}"
            )
        if (
            self.cost_settlement is not None
            and self.cost_settlement not in ATTEMPT_SETTLEMENTS
        ):
            raise ValueError(
                f"{self.cost_settlement!r} is not a settlement this contract "
                f"defines; allowed: {list(ATTEMPT_SETTLEMENTS)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "outcome": self.outcome,
            "fault": self.fault,
            "http_status": self.http_status,
            "latency_seconds": self.latency_seconds,
            "response_received": self.response_received,
            "usage_reported": self.usage_reported,
            "cost_reservation_usd": self.cost_reservation_usd,
            "cost_settlement": self.cost_settlement,
        }


@dataclass(frozen=True)
class ProviderTelemetry:
    """Every attempt one turn made, and what the turn as a whole measured.

    An empty ``attempts`` is a real observation and not a missing one: it is what
    a turn whose budget was spent before a request could be dispatched actually
    did. "No telemetry at all" is a different claim and is spelled ``None`` — an
    in-process fake has no attempts to report, and reporting zero attempts would
    claim it tried and failed.
    """

    attempts: tuple[ProviderAttempt, ...] = ()
    #: Locally measured wall-clock for the whole turn, retries and backoff waits
    #: included. ``None`` means unmeasured, never zero.
    turn_latency_seconds: float | None = None
    #: Why the turn's retry loop stopped, from
    #: :data:`~operatebench.providers.faults.TURN_TERMINAL_REASONS`. ``None`` is
    #: the cleared state — a turn whose request never reached the loop at all,
    #: and which therefore has no exit to report.
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.terminal_reason is not None and (
            self.terminal_reason not in TURN_TERMINAL_REASONS
        ):
            raise ValueError(
                f"{self.terminal_reason!r} is not a reason a turn may end for that "
                f"this contract defines; allowed: {list(TURN_TERMINAL_REASONS)}"
            )

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def response_received(self) -> bool:
        return any(attempt.response_received for attempt in self.attempts)

    @property
    def usage_reported(self) -> bool:
        return any(attempt.usage_reported for attempt in self.attempts)

    @property
    def fault_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            if attempt.fault is not None:
                counts[attempt.fault] = counts.get(attempt.fault, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        """The stored form, with the derived summary spelled out.

        The four derived fields are written rather than left to the reader
        because a durable row is read by things that are not this build. They are
        re-derived from ``attempts`` when a row is read back, so a row whose
        summary does not follow from its own attempts is refused.
        """
        return {
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "attempt_count": self.attempt_count,
            "fault_counts": self.fault_counts,
            "response_received": self.response_received,
            "usage_reported": self.usage_reported,
            "turn_latency_seconds": self.turn_latency_seconds,
            "terminal_reason": self.terminal_reason,
        }


__all__ = [
    "AdapterUsage",
    "ProviderAttempt",
    "ProviderTelemetry",
    "TurnDeadline",
]
