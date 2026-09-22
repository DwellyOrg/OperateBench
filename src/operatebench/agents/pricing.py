"""What a Lifecycle provider call is authorised to cost, and what it did cost.

The shared kernel reserves an amount before every request and resolves it
afterwards, and it deliberately does not know what a token costs: the price
table is the track's business (see :mod:`operatebench.providers.cost`). The
Lifecycle Track has had no such table, which is why a Lifecycle run's cost has
until now been ``null`` — the guard was a constructor parameter nothing
supplied.

This module is the Lifecycle Track's answer, and it states plainly what it is:

**The rates are the operator's, pinned, and digested.** :class:`LifecyclePricingPolicy`
carries the two rates a run was priced at, the identity of the policy and the
source of the rates as a closed-vocabulary value. ``rate_source`` is
:data:`RATE_SOURCE_OPERATOR` — an operator stated these numbers for this run —
and this build makes **no claim that they are any vendor's current published
prices**. The digest travels into the execution ledger beside every amount, so a
reader knows which table an amount was computed at and can refuse one computed
at another.

**The guard reserves before dispatch and never absorbs a breach.** It satisfies
:class:`~operatebench.providers.cost.CostGuard` structurally, so the real shared
retry loop authorises through it, records the reservation on the attempt, and
settles or forfeits it — no second reservation arithmetic exists here.

**A forfeited reservation is not spend.** :attr:`LifecycleCostGuard.measured_usd`
is what the provider's reported counts came to; :attr:`forfeited_usd` is budget
consumed for attempts whose actual cost is unknown. They are separate totals,
and nothing in this module or the ledger adds them together and calls the result
what a run cost.

Nothing here opens a socket, reads a credential or imports a provider SDK.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from operatebench.core.errors import OperateBenchError
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.cost import (
    CostCapExceededError,
    CostReservationBreachedError,
    usd_text,
)

#: Tokens a rate is quoted per. One million, stated once.
TOKENS_PER_RATE_UNIT = 1_000_000

#: The rates were supplied by the operator running this experiment and pinned
#: for it. Not a claim about any vendor's price list, current or historical.
RATE_SOURCE_OPERATOR = "operator_supplied_pinned_rates"

#: Every source a rate may be attributed to. One value, because this build has
#: exactly one honest thing to say about where its numbers came from.
RATE_SOURCES: tuple[str, ...] = (RATE_SOURCE_OPERATOR,)

#: The arithmetic precision every amount here is computed under. Well above what
#: a run's accumulated amounts can need: a total that rounded would be a cap
#: enforced against a number that is nobody's arithmetic.
PRICING_PRECISION = 60


class PricingPolicyError(OperateBenchError):
    """A pricing policy, or an amount offered to one, is not usable."""


def _rate(value: object, name: str) -> Decimal:
    if isinstance(value, float) or not isinstance(value, (Decimal, int, str)):
        raise PricingPolicyError(
            f"{name} must be an exact Decimal, integer or decimal string; a binary "
            f"float is not the rate anybody agreed to. Got {type(value).__name__}"
        )
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise PricingPolicyError(f"{name} is not a decimal amount") from exc
    if not rate.is_finite() or rate < 0:
        raise PricingPolicyError(f"{name} must be finite and non-negative, got {rate}")
    return rate


@dataclass(frozen=True)
class LifecyclePricingPolicy:
    """Two rates, an identity and where the rates came from.

    Frozen and digested. The digest is over the stored form — the rates as exact
    decimal *strings* — so a run's amounts and the table they were computed at
    are bound together by something a reader can recompute.
    """

    policy_id: str
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    rate_source: str = RATE_SOURCE_OPERATOR

    def __post_init__(self) -> None:
        if not self.policy_id or not self.policy_id.strip():
            raise PricingPolicyError(
                "a pricing policy is named, so an amount computed under it can be "
                "attributed to a table rather than to a number somebody typed"
            )
        if self.rate_source not in RATE_SOURCES:
            raise PricingPolicyError(
                f"{self.rate_source!r} is not a rate source this build states; "
                f"allowed: {list(RATE_SOURCES)}. This build does not claim any "
                "vendor's published prices"
            )
        object.__setattr__(
            self,
            "input_usd_per_mtok",
            _rate(self.input_usd_per_mtok, "input_usd_per_mtok"),
        )
        object.__setattr__(
            self,
            "output_usd_per_mtok",
            _rate(self.output_usd_per_mtok, "output_usd_per_mtok"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "input_usd_per_mtok": usd_text(self.input_usd_per_mtok),
            "output_usd_per_mtok": usd_text(self.output_usd_per_mtok),
            "rate_source": self.rate_source,
        }

    @property
    def digest_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_text(self.as_dict(), "lifecycle pricing policy").encode(
                "utf-8"
            )
        ).hexdigest()

    def price(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        """What one call's reported counts come to, at this table's rates."""
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            if type(value) is not int or value < 0:
                raise PricingPolicyError(
                    f"{name} must be a non-negative integer count, got {value!r}"
                )
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            return (
                Decimal(input_tokens) * self.input_usd_per_mtok
                + Decimal(output_tokens) * self.output_usd_per_mtok
            ) / Decimal(TOKENS_PER_RATE_UNIT)

    def worst_case_usd(
        self, *, calls: int, input_token_bound: int, max_output_tokens: int
    ) -> Decimal:
        """The most a run of this shape could cost, if every call were the largest.

        Pessimistic on purpose, and computed from the run's own bounds rather
        than from what a scenario is expected to need: a preflight that refused
        only runs whose *expected* cost exceeded the cap would authorise a run
        that could exceed it.
        """
        if type(calls) is not int or calls < 1:
            raise PricingPolicyError(f"calls must be a positive integer, got {calls!r}")
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            return Decimal(calls) * self.price(
                input_tokens=input_token_bound, output_tokens=max_output_tokens
            )


class TokenCapExceededError(CostCapExceededError):
    """The run's cumulative token reservations would pass its fixed ceiling.

    A :class:`~operatebench.providers.cost.CostCapExceededError` rather than a
    new kind of refusal, and deliberately so: the kernel's retry loop already
    knows that error means *this request was never dispatched*. It records the
    turn as having stopped on its budget, resolves nothing it did not reserve,
    and the run is excluded at the provider boundary. A separate exception type
    would need the same handling written a second time, and a second copy is
    how a bound stops being enforced on one of the paths.

    The ceiling itself is a fixed property of this build rather than an
    operator's number. It bounds worst-case exposure — every authorised call at
    the largest request this scaffold produces, plus the whole output ceiling —
    and moving it is a code change, not a command-line argument.
    """


@dataclass(frozen=True)
class GuardEvent:
    """How one reservation was resolved. Exactly one of the two amounts is set."""

    reservation_usd: Decimal
    measured_usd: Decimal | None = None
    forfeited_usd: Decimal | None = None


class LifecycleCostGuard:
    """One run's authorised budget, enforced before every provider request.

    Structurally a :class:`~operatebench.providers.cost.CostGuard`, so the shared
    retry loop drives it: authorise, dispatch, then settle at the measured cost
    or forfeit the reservation as exposure. Nothing here re-implements the
    reserve-before-dispatch discipline; it supplies the price table that
    discipline was always missing.
    """

    def __init__(
        self,
        *,
        policy: LifecyclePricingPolicy,
        cap_usd: Decimal,
        max_output_tokens: int,
        token_hard_cap: int | None = None,
    ) -> None:
        if not isinstance(policy, LifecyclePricingPolicy):
            raise PricingPolicyError(
                "a Lifecycle cost guard enforces a stated pricing policy, not "
                f"{type(policy).__name__}"
            )
        cap = _rate(cap_usd, "cap_usd")
        if cap <= 0:
            raise PricingPolicyError(
                "a cost cap is a positive amount; a zero cap authorises nothing and "
                "is refused rather than read as 'no limit'"
            )
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise PricingPolicyError("max_output_tokens must be a positive integer")
        if token_hard_cap is not None and (
            type(token_hard_cap) is not int or token_hard_cap < 1
        ):
            raise PricingPolicyError(
                "a token ceiling is a positive integer count of tokens, or absent"
            )
        self._policy = policy
        self._cap = cap
        self._max_output_tokens = max_output_tokens
        self._token_hard_cap = token_hard_cap
        self._reserved_tokens = 0
        self._measured = Decimal(0)
        self._forfeited = Decimal(0)
        self._reserved = Decimal(0)
        self._outstanding = Decimal(0)
        self._last: GuardEvent | None = None
        self._pending_events: list[GuardEvent] = []
        # Equal-priced requests remain distinct outstanding authorisations. The
        # retry loop is serial, so the only pre-dispatch cancellation is for the
        # request it just authorised; retaining exposures in LIFO order binds
        # that cancellation to the exact token bound even when an input rate of
        # zero makes differently sized requests price equally.
        self._pending: dict[Decimal, list[int]] = {}

    @property
    def policy(self) -> LifecyclePricingPolicy:
        return self._policy

    @property
    def cap_usd(self) -> Decimal:
        return self._cap

    @property
    def measured_usd(self) -> Decimal:
        """What the provider's own reported counts came to. Actual spend."""
        return self._measured

    @property
    def forfeited_usd(self) -> Decimal:
        """Budget consumed for attempts whose actual cost is unknown. Not spend."""
        return self._forfeited

    @property
    def reserved_usd(self) -> Decimal:
        """Everything ever authorised before dispatch, resolved or not."""
        return self._reserved

    @property
    def committed_usd(self) -> Decimal:
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            return self._measured + self._forfeited + self._outstanding

    def take_events(self) -> tuple[GuardEvent, ...]:
        """Every resolution since the last take, in the order they happened.

        A queue rather than a single value because a turn may resolve more than
        one reservation — one per attempt — and a recorder that read only the
        last one would write the same amount onto every attempt of that turn.
        This build's retry policy allows one attempt per call, so today the
        queue holds one entry; taking them in order is what keeps that from
        being an assumption the row silently depends on.
        """
        taken, self._pending_events = tuple(self._pending_events), []
        return taken

    def last_event(self) -> GuardEvent | None:
        """How the most recent reservation was resolved, or ``None``.

        Read by the evidence recorder so the amount that reaches a durable row
        is the exact ``Decimal`` this guard computed, rather than a float
        rounded on its way through a usage object.
        """
        return self._last

    @property
    def token_hard_cap(self) -> int | None:
        """The fixed ceiling on this run's cumulative token exposure, if any."""
        return self._token_hard_cap

    @property
    def reserved_tokens(self) -> int:
        """Every token this run has authorised: each call's bound plus the ceiling."""
        return self._reserved_tokens

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        exposure = input_tokens_upper_bound + self._max_output_tokens
        if self._token_hard_cap is not None:
            projected_tokens = self._reserved_tokens + exposure
            if projected_tokens > self._token_hard_cap:
                raise TokenCapExceededError(
                    "this run's token ceiling cannot cover the request. The ceiling is "
                    f"{self._token_hard_cap} token(s), {self._reserved_tokens} are "
                    f"already reserved and this request's worst case is {exposure} "
                    f"({input_tokens_upper_bound} in, {self._max_output_tokens} out), "
                    f"which would commit {projected_tokens}. It is refused before "
                    "dispatched rather than discovered afterwards"
                )
        reservation = self._policy.price(
            input_tokens=input_tokens_upper_bound,
            output_tokens=self._max_output_tokens,
        )
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            projected = self._measured + self._forfeited + self._outstanding + reservation
        if projected > self._cap:
            raise CostCapExceededError(
                "this run's remaining authorised budget cannot cover the request. "
                f"The cap is {usd_text(self._cap)} USD and authorising this request "
                f"would commit {usd_text(projected)} USD, so it is refused before "
                "anything is dispatched rather than discovered afterwards"
            )
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            self._outstanding += reservation
            self._reserved += reservation
        # Counted only once the request is authorised, so a refusal leaves the
        # run's recorded exposure exactly where it was.
        self._reserved_tokens += exposure
        self._pending.setdefault(reservation, []).append(exposure)
        return reservation

    def settle(
        self,
        reservation: Decimal,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> Decimal | None:
        if input_tokens is None or output_tokens is None:
            self.forfeit(reservation)
            return None
        measured = self._policy.price(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        self._release(reservation)
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            self._outstanding -= reservation
            self._measured += measured
        self._last = GuardEvent(reservation_usd=reservation, measured_usd=measured)
        self._pending_events.append(self._last)
        if measured > reservation:
            raise CostReservationBreachedError(
                "a response cost more than the reservation that authorised it. The "
                "measured amount is banked at its measured value — hiding real spend "
                "is the failure that matters — and the run stops rather than "
                "continuing under a cap it has already passed",
                measured_usd=measured,
                reservation_usd=reservation,
                cap_usd=self._cap,
            )
        return measured

    def forfeit(self, reservation: Decimal) -> None:
        self._release(reservation)
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            self._outstanding -= reservation
            self._forfeited += reservation
        self._last = GuardEvent(reservation_usd=reservation, forfeited_usd=reservation)
        self._pending_events.append(self._last)

    def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
        """Release authorisation for a request proven not to have dispatched."""
        # Pricing validates the supplied token bound and proves it reproduces the
        # reservation. Exposure is checked before the pending entry is removed.
        recomputed = self._policy.price(
            input_tokens=input_tokens_upper_bound,
            output_tokens=self._max_output_tokens,
        )
        exposure = input_tokens_upper_bound + self._max_output_tokens
        if recomputed != reservation:
            raise PricingPolicyError(
                "the Lifecycle cancellation token bound does not reproduce the "
                "reservation being released"
            )
        self._release(reservation, expected_exposure=exposure)
        with localcontext() as context:
            context.prec = PRICING_PRECISION
            self._outstanding -= reservation
        self._reserved_tokens -= exposure

    def _release(
        self, reservation: Decimal, *, expected_exposure: int | None = None
    ) -> int:
        try:
            exposures = self._pending[reservation]
        except KeyError as exc:
            raise PricingPolicyError(
                "the Lifecycle reservation being resolved is not outstanding; "
                "resolving it twice or resolving an unauthorised amount would make "
                "the guard's accounting negative or ambiguous"
            ) from exc
        exposure = exposures[-1]
        if expected_exposure is not None and exposure != expected_exposure:
            raise PricingPolicyError(
                "the Lifecycle cancellation token exposure is not the most recent "
                "equal-priced authorisation; no outstanding total was changed"
            )
        exposures.pop()
        if not exposures:
            del self._pending[reservation]
        return exposure

    def as_dict(self) -> Mapping[str, Any]:
        """The three totals, apart, as exact decimal strings."""
        return {
            "cap_usd": usd_text(self._cap),
            "measured_cost_usd": usd_text(self._measured),
            "forfeited_reservation_usd": usd_text(self._forfeited),
            "reserved_usd": usd_text(self._reserved),
            "pricing_digest_sha256": self._policy.digest_sha256,
        }


__all__ = [
    "PRICING_PRECISION",
    "RATE_SOURCES",
    "RATE_SOURCE_OPERATOR",
    "TOKENS_PER_RATE_UNIT",
    "GuardEvent",
    "LifecycleCostGuard",
    "LifecyclePricingPolicy",
    "PricingPolicyError",
    "TokenCapExceededError",
]
