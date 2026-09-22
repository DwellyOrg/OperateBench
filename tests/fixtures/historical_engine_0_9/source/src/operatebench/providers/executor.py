"""One turn's provider call: the retry loop, the cost guard and the evidence.

This is the part of a provider call that no track and no vendor gets an opinion
about. What a turn reserves before it dispatches, when it forfeits, which exit it
names, what it publishes before it raises — those are the properties a durable
row is read against, and an integration that got one of them subtly wrong would
produce evidence that looks exactly like evidence. So it is written once.

Everything provider-specific arrives as a callback and everything track-specific
never arrives at all: :meth:`TurnExecutor.run` takes a payload mapping, a
deadline, a token bound and four functions. It has never heard of a
``TurnRequest``, a ``ModelRequest``, a scaffold or an outcome.

Nothing in this module opens a socket, reads a credential or imports a provider
SDK.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Generic, TypeVar

from operatebench.providers.config import ProviderConfigurationError
from operatebench.providers.cost import (
    CostCapExceededError,
    CostGuard,
    CostReservationBreachedError,
    usd_text,
)
from operatebench.providers.faults import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_FAULT_NOT_RETRYABLE,
    TURN_END_PRE_DISPATCH_REFUSED,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
    TURN_END_UNCLASSIFIED,
    AdapterProviderError,
    Fault,
    fault_detail,
)
from operatebench.providers.telemetry import (
    AdapterUsage,
    ProviderAttempt,
    ProviderTelemetry,
    TurnDeadline,
)
from operatebench.providers.usage import TokenUsage

# -- the retry policy ---------------------------------------------------------


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """The bounded, deterministic retry policy an adapter owns.

    Owned by the adapter rather than delegated to the SDK. An SDK's own retry
    loop is disabled because a hidden retry spends the episode's wall-clock
    budget without the runner or the deadline knowing, and a benchmark that
    cannot say how many provider calls an episode made cannot report latency
    honestly.

    Not to be confused with a run manifest's ``retry_policy``, which is about
    *episodes*: a failed episode is still never re-run. This is one turn's
    transport-level policy, and it is inside adapter settings, so it is inside
    run identity all the same.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "backoff_multiplier": self.backoff_multiplier,
        }


#: The smallest multiplier a retry policy may state.
#:
#: One, and not zero: a multiplier below one *shrinks* the wait on every retry,
#: so a policy whose settings read like a backoff would in fact hammer the
#: provider faster the worse the outage got, and a multiplier of zero would
#: retry with no wait at all after the first pause. One is the exact lower bound
#: at which "back off" remains true — it means a constant wait — so it is the
#: bound rather than an arbitrary safety margin.
MINIMUM_BACKOFF_MULTIPLIER = 1.0


def check_retry_policy(
    policy: ProviderRetryPolicy,
    *,
    provider: str,
    error: type[ProviderConfigurationError],
) -> ProviderRetryPolicy:
    """Prove a retry policy is one, before a client exists or a request is sent.

    The policy is inside adapter settings, so it is inside ``configuration_id``
    and inside the run manifest: it is this run's durable claim about how many
    times a turn may ask the provider and how long it waits between attempts. A
    policy the loop cannot honour makes that claim false in the direction that
    costs money and wall-clock — ``max_attempts=0`` records a bounded retry
    policy and then dispatches a request the settings say it would not make,
    ``NaN`` compares false against every bound so a backoff derived from it is
    never "too long to afford", and ``inf`` sleeps out the episode.

    Every value is checked for *exact* type and finite range rather than for
    truthiness. ``bool`` is refused where an ``int`` is required even though
    Python makes it one: ``max_attempts=True`` is an attempt count nobody
    chose, and settings that recorded ``true`` where a count belongs would be
    hashed into run identity as a number they do not name.

    Called before the credential is read and before a client is constructed, and
    again in the adapter's own constructor, because those are the two ways a
    policy reaches the dispatch loop.
    """
    if not isinstance(policy, ProviderRetryPolicy):
        raise error(
            f"a {provider} run's retry policy must be a ProviderRetryPolicy; the "
            "policy is hashed into run identity as this run's statement of how "
            "many times a turn may ask the provider, and an object of another "
            f"type cannot be that statement. Got {type(policy).__name__}"
        )
    attempts = policy.max_attempts
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise error(
            "a retry policy's max_attempts must be an integer of at least 1, and "
            "not a bool. A turn makes one request before it can retry anything, so "
            "a policy of 0 or fewer describes a run that sends nothing while the "
            "loop sends one request anyway — and the settings hashed into run "
            f"identity would name a policy this build does not follow. Got {attempts!r}"
        )
    backoff = policy.initial_backoff_seconds
    if (
        isinstance(backoff, bool)
        or not isinstance(backoff, (int, float))
        or not math.isfinite(backoff)
        or backoff < 0
    ):
        raise error(
            "a retry policy's initial_backoff_seconds must be a finite number of "
            "zero or more seconds, and not a bool. A negative wait is not a wait, "
            "and NaN compares false against every bound this loop checks a backoff "
            "against — so a policy carrying one would pass the affordability check "
            "that exists to stop a retry outliving the turn's wall-clock budget. "
            f"Got {backoff!r}"
        )
    multiplier = policy.backoff_multiplier
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(multiplier)
        or multiplier < MINIMUM_BACKOFF_MULTIPLIER
    ):
        raise error(
            "a retry policy's backoff_multiplier must be a finite number of at "
            f"least {MINIMUM_BACKOFF_MULTIPLIER}, and not a bool. Below one the "
            "wait shrinks on every retry, so settings that read as a backoff would "
            "in fact ask the provider faster the worse the outage got; NaN and "
            "infinity are not schedules at all. A multiplier of exactly one is a "
            f"constant wait, which is why it is the bound. Got {multiplier!r}"
        )
    return policy


#: Zero. Every adapter owns its retries; see :class:`ProviderRetryPolicy`.
SDK_MAX_RETRIES = 0

# -- the turn ----------------------------------------------------------------

#: What an adapter waits on, and what it reads the clock with. Injected so the
#: retry and deadline paths are exercised deterministically and instantly.
Sleep = Callable[[float], None]
Clock = Callable[[], float]

#: One provider's parsed response object, whatever its SDK calls it.
Response = TypeVar("Response")


class TurnExecutor(Generic[Response]):
    """The retry loop, the cost guard and the attempt evidence, once.

    Everything provider-specific arrives as a callback, and there are exactly
    four of them because there are exactly four things the providers disagree
    about:

    ``dispatch``
        make one request with a per-attempt timeout and return the parsed
        response, or raise the SDK's own exception.
    ``classify``
        map one SDK exception onto this contract's fault set, or return ``None``
        to say "this is not the provider's doing" and let it out unchanged.
    ``check_identity``
        raise if the response does not state that the pinned model produced it.
    ``usage_of``
        read the two token counts off the response, or raise
        :class:`~operatebench.providers.faults.AdapterProviderError` to refuse a
        body it cannot read a count off of. Refusing here ends the turn with the
        reservation kept as exposure — see :meth:`run`.

    The loop itself — what it reserves, when it forfeits, which exit it names,
    what it publishes before raising — is identical for every provider and is
    written once, here, because those are the properties a durable row is read
    against and an integration that got one of them subtly wrong would produce
    evidence that looks exactly like evidence.
    """

    def __init__(
        self,
        *,
        retry: ProviderRetryPolicy,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
    ) -> None:
        self._retry = retry
        self._sleep = sleep
        self._clock = clock
        self._cost_guard = cost_guard
        self._usage = AdapterUsage()
        self._telemetry = ProviderTelemetry()

    @property
    def cost_guard(self) -> CostGuard | None:
        return self._cost_guard

    def last_usage(self) -> AdapterUsage:
        return self._usage

    def last_telemetry(self) -> ProviderTelemetry:
        """What this turn asked of the provider, and what each attempt met.

        Never ``None``: these adapters do call a provider, so every turn has an
        answer to give — including a turn whose budget was spent before a
        request could be dispatched, which honestly reports zero attempts.
        """
        return self._telemetry

    def clear(self) -> None:
        """Drop both measurements, before anything else a turn does.

        Called by ``next_call`` first — before the request is even serialised —
        because everything after it can fail, and a measurement that outlives
        the call it measured is stale evidence.

        Attempt evidence that survived into the next turn would be worse than
        stale usage: it would attribute one turn's provider faults to another
        turn, which is the exact question the evidence exists to answer.
        """
        self._usage = AdapterUsage()
        self._telemetry = ProviderTelemetry()

    def record_pre_dispatch_refusal(self, *, started_at: float) -> None:
        """Publish a local configuration refusal detected before :meth:`run`.

        The caller starts the local latency measurement immediately before its
        entry-time client-state check and calls this only when that check raises
        its own :class:`ProviderConfigurationError` subclass. No exception is
        accepted or inspected here. The refusal occurred before cost
        authorisation and before a provider attempt, so this method touches
        neither and publishes exactly zero attempts with empty usage.

        Supported adapter entry points clear at the start of every turn. This
        method resets both measurements as well: direct sequential reuse cannot
        retain a previous turn's usage or attempts if a caller reaches this
        publisher without clearing first.
        """
        self._usage = AdapterUsage()
        self._telemetry = ProviderTelemetry(
            attempts=(),
            turn_latency_seconds=self._clock() - started_at,
            terminal_reason=TURN_END_PRE_DISPATCH_REFUSED,
        )

    def run(
        self,
        *,
        payload: Mapping[str, Any],
        deadline: TurnDeadline,
        token_bound: int,
        dispatch: Callable[[float], Response],
        check_before_dispatch: Callable[[], None],
        classify: Callable[[Exception], Fault | None],
        check_identity: Callable[[Response], None],
        usage_of: Callable[[Response], TokenUsage],
    ) -> Response:
        """One turn's provider call, retried within the turn's own budget.

        The deadline is the ceiling on the whole loop, not on one request: the
        remaining budget is recomputed before every attempt and handed to
        ``dispatch`` as that attempt's timeout, so retrying cannot extend an
        episode past the wall-clock it was given. A wait that would not leave
        time for the request it precedes is not taken at all — sleeping through
        the rest of the budget and then failing wastes the turn to learn
        nothing.

        Only :class:`Exception` is caught, and only the subset ``classify``
        recognises. A ``KeyboardInterrupt`` is not a provider fault and is not
        retried, delayed or reclassified.

        **Two instants, deliberately kept apart.** ``origin`` is where the
        turn's *budget* is counted from and ``started`` is where this loop was
        entered. They are the same instant unless the caller stated a
        :attr:`~operatebench.providers.telemetry.TurnDeadline.started_at`, in
        which case the caller did work — request checks, body construction,
        payload validation, the token bound — inside the budget and before this
        method, and every attempt's timeout has to be what is left after it.
        Telemetry keeps measuring from ``started``: a turn's recorded latency is
        what *this loop* took, and moving it would silently re-define a number
        that is already in durable rows.
        """
        started = self._clock()
        origin = started if deadline.started_at is None else deadline.started_at
        wait = self._retry.initial_backoff_seconds
        attempts = 0
        record: list[ProviderAttempt] = []

        def left() -> float:
            return deadline.remaining_seconds - (self._clock() - origin)

        def commit(now: float, reason: str) -> None:
            """Publish what the turn has done so far, before anything raises.

            Called on every exit from this loop rather than only on the ones
            that succeed. The whole point of this evidence is that it survives a
            failure: a turn that made three requests and got nothing is exactly
            the turn whose attempts an operator needs to see, and it is also the
            turn from which nothing else can be learned.

            The instant is passed in rather than read here. Every exit already
            has one in hand, and taking a second reading would make the turn's
            recorded latency depend on how many times this loop happened to
            consult the clock rather than on how long the turn took.
            """
            self._telemetry = ProviderTelemetry(
                attempts=tuple(record),
                turn_latency_seconds=now - started,
                terminal_reason=reason,
            )

        while True:
            # One reading, used as both this attempt's start and the instant the
            # remaining budget is computed at. Two would drift apart under a
            # clock that moves between them. It is subtracted from ``origin``
            # once and only here, so a stated origin is never deducted twice.
            attempt_started = self._clock()
            budget = deadline.remaining_seconds - (attempt_started - origin)
            if budget <= 0.0 or deadline.cancelled():
                commit(attempt_started, TURN_END_DEADLINE_EXCEEDED)
                raise AdapterProviderError(
                    PROVIDER_FAULT_TIMEOUT,
                    fault_detail(
                        Fault(PROVIDER_FAULT_TIMEOUT, True, None),
                        attempts=attempts,
                        reason=(
                            "the turn's remaining wall-clock budget was spent or "
                            "cancelled before a request could be made"
                        ),
                    ),
                )
            # Before the attempt is counted, and before a socket is opened: a
            # request the run's remaining authorised budget cannot cover is not
            # made at all. The refusal raises out of this loop without touching
            # the attempt record, because an attempt that never happened is not
            # evidence of one that did. ``commit`` still publishes what the turn
            # did before it, so a turn refused on its third retry still reports
            # the two attempts it made.
            reservation: Decimal | None = None
            if self._cost_guard is not None:
                try:
                    reservation = self._cost_guard.authorize(
                        input_tokens_upper_bound=token_bound
                    )
                except CostCapExceededError:
                    commit(self._clock(), TURN_END_COST_CAP_EXHAUSTED)
                    raise
            # Rendered once, here, and carried onto every attempt record this
            # iteration can produce: the amount a row states as reserved is the
            # amount the guard actually authorised for that same request.
            reserved = None if reservation is None else usd_text(reservation)
            # The injected client is mutable. This final local admission runs
            # after authorisation so mutation from a trusted guard is visible,
            # but before the attempt count and socket boundary. A refusal here
            # cancels the reservation: no request may have crossed the wire.
            try:
                check_before_dispatch()
            except ProviderConfigurationError:
                # The refusal itself is already known before reservation cleanup.
                # Publish it first so a broken guard cannot leave stale telemetry;
                # then let any cancellation failure escape rather than hiding an
                # accounting invariant violation behind the configuration error.
                commit(self._clock(), TURN_END_PRE_DISPATCH_REFUSED)
                if self._cost_guard is not None and reservation is not None:
                    self._cost_guard.cancel(
                        reservation, input_tokens_upper_bound=token_bound
                    )
                raise
            attempts += 1
            try:
                response = dispatch(budget)
            except Exception as exception:
                failed_at = self._clock()
                # An attempt that returned nothing this build could measure keeps
                # its whole conservative bound as exposure. Whether the provider
                # billed for it is not something a client can determine, and the
                # only error that can breach a cap is under-stating it.
                self._forfeit(reservation)
                fault = classify(exception)
                if fault is None:
                    # Not the provider's doing. Left to the runner, which records
                    # it as this integration's own failure rather than an outage.
                    # The attempt is still recorded — the request was made, the
                    # time it took is the turn's whatever broke afterwards, and
                    # the reservation it forfeited is money the run has to
                    # account for.
                    record.append(
                        ProviderAttempt(
                            index=attempts,
                            outcome=ATTEMPT_OUTCOME_UNCLASSIFIED,
                            latency_seconds=failed_at - attempt_started,
                            cost_reservation_usd=reserved,
                            cost_settlement=(
                                None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                            ),
                        )
                    )
                    commit(failed_at, TURN_END_UNCLASSIFIED)
                    raise
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_FAULT,
                        fault=fault.fault,
                        http_status=fault.status,
                        latency_seconds=failed_at - attempt_started,
                        # Not always ``False``. A typed HTTP-status exception, or
                        # an SDK that read a whole 200 body and then refused to
                        # parse it, proves a response reached the SDK. Neither is
                        # an accepted model answer. See
                        # :attr:`Fault.response_received`.
                        response_received=fault.response_received,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                if not fault.retryable:
                    commit(failed_at, TURN_END_FAULT_NOT_RETRYABLE)
                    raise AdapterProviderError(
                        fault.fault,
                        fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "this fault is not retried, because repeating the "
                                "same request cannot change the answer"
                            ),
                        ),
                    ) from exception
                if attempts >= self._retry.max_attempts:
                    commit(failed_at, TURN_END_RETRIES_EXHAUSTED)
                    raise AdapterProviderError(
                        fault.fault,
                        fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "the adapter's bounded retry policy is exhausted "
                                "and no answer was received"
                            ),
                        ),
                    ) from exception
                pause, wait = wait, wait * self._retry.backoff_multiplier
                # Two different facts, kept apart: a cancelled episode ran out
                # of wall clock, and a backoff that will not fit is this
                # policy's own arithmetic over what is left of it.
                cancelled = deadline.cancelled()
                if cancelled or pause >= left():
                    commit(
                        failed_at,
                        TURN_END_DEADLINE_EXCEEDED
                        if cancelled
                        else TURN_END_BACKOFF_UNAFFORDABLE,
                    )
                    raise AdapterProviderError(
                        fault.fault,
                        fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "the turn's remaining wall-clock budget cannot pay "
                                "for the next backoff and the request after it, so "
                                "the retry was abandoned rather than started"
                            ),
                        ),
                    ) from exception
                self._sleep(pause)
                continue
            # Before the measurement, and so before anything is parsed out of
            # the body: a response whose stated model is not the pinned one is
            # refused whole. This is the one narrow exception to "a response
            # that arrived was paid for, so it is measured". Every usage figure
            # this build records is attributed to the adapter identity on the
            # row that carries it, so banking these counts would attribute spend
            # to a model that did not produce it — the same falsehood as
            # accepting the action would be. A protocol-invalid answer *from the
            # pinned model* is the opposite case and is measured, below.
            answered_at = self._clock()
            try:
                check_identity(response)
            except AdapterProviderError:
                # A body arrived, so the attempt is recorded as one that got a
                # response — and as one that reported no usage, because this
                # build refuses to attribute those counts to a model that did
                # not produce them. Recording it as a transport fault instead
                # would say no answer came back, which is not what happened.
                self._forfeit(reservation)
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_FAULT,
                        fault=PROVIDER_FAULT_RESPONSE_INVALID,
                        latency_seconds=answered_at - attempt_started,
                        response_received=True,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                commit(answered_at, TURN_END_FAULT_NOT_RETRYABLE)
                raise

            breach: CostReservationBreachedError | None = None
            # Reading the counts is the second place a usage block is proven,
            # and it is the one that runs with a reservation outstanding. Every
            # other exit from this loop resolves that reservation, publishes one
            # attempt and names a terminal reason before it raises; this one has
            # to do the same, because an exception carried straight out would
            # leave the guard holding a pending reservation no later episode can
            # begin past, and telemetry claiming the turn made no attempt at all
            # over a request that really was sent and really may be billed.
            try:
                usage = usage_of(response)
            except AdapterProviderError as exception:
                # The body arrived and this build cannot read a count off it.
                # Recorded exactly as the identity refusal above is: a response
                # *was* received, no usage was reported, and the reservation
                # stays whole as exposure rather than settling to a cost nobody
                # measured. The turn is over — repeating the request cannot make
                # the same body readable — and the exception goes out unchanged,
                # because it already carries this contract's fault name and this
                # build's own fixed detail.
                self._forfeit(reservation)
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_FAULT,
                        fault=exception.fault,
                        latency_seconds=answered_at - attempt_started,
                        response_received=True,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                commit(answered_at, TURN_END_FAULT_NOT_RETRYABLE)
                raise
            except Exception:
                # Not the provider's doing: a body came back and this
                # integration's own reading of it failed in a way it never
                # classified. Named as this build's own outcome rather than as a
                # fault, for the same reason the dispatch path does it — a fault
                # this build did not observe would be a fabricated diagnosis,
                # and a reader counting outages would count a bug in here as
                # one. The attempt still says a response was received, because
                # one was, and the reservation is still money this run has to
                # account for.
                self._forfeit(reservation)
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_UNCLASSIFIED,
                        latency_seconds=answered_at - attempt_started,
                        response_received=True,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                commit(answered_at, TURN_END_UNCLASSIFIED)
                raise
            try:
                settlement = self._record_usage(
                    usage, seconds=answered_at - started, reservation=reservation
                )
            except CostReservationBreachedError as exc:
                # The response arrived, was measured and was banked at its
                # measured value; what failed is the authorisation, not the
                # observation. So the attempt is recorded exactly as the
                # successful one it was — including the settlement, which really
                # did measure — and only then does the turn raise. Committing the
                # evidence first is the whole point: the row that stops the run
                # is the one that has to show what the run had spent when it did.
                breach, settlement = exc, ATTEMPT_SETTLEMENT_MEASURED
            record.append(
                ProviderAttempt(
                    index=attempts,
                    outcome=ATTEMPT_OUTCOME_RESPONSE,
                    latency_seconds=answered_at - attempt_started,
                    response_received=True,
                    usage_reported=usage.measurable,
                    cost_reservation_usd=reserved,
                    cost_settlement=settlement,
                )
            )
            commit(answered_at, TURN_END_RESPONSE)
            if breach is not None:
                raise breach
            return response

    def _forfeit(self, reservation: Decimal | None) -> None:
        """Resolve an outstanding reservation as exposure, if there is one."""
        if self._cost_guard is not None and reservation is not None:
            self._cost_guard.forfeit(reservation)

    def _record_usage(
        self,
        usage: TokenUsage,
        *,
        seconds: float,
        reservation: Decimal | None = None,
    ) -> str | None:
        """One turn's measurement, under one stated convention.

        * The token counts are the provider's own, on the response that
          *returned*, accepted or not. Recorded before the answer is parsed, so
          a turn that ends in a protocol failure still reports what the provider
          charged for producing it: those tokens were consumed, and a benchmark
          that omitted them would under-report real spend on exactly the turns
          where a model misbehaved. Attempts that failed in transport returned
          no body to measure, so they contribute nothing rather than zero.
        * A turn's measurement covers every response the turn received. At most
          one of them can be measurable in this build — a response either ends
          the turn or ends the run, so the retry loop never gets a second body —
          which is why this assigns rather than sums.
        * Cache-related counters are deliberately not folded in. This build
          enables no caching feature, so there is nothing cached to count, and
          summing fields whose meaning depends on a feature that is off would
          make the number mean different things in different runs.
        * ``latency_seconds`` is locally measured wall-clock for the whole turn:
          from immediately before the first provider request to immediately
          after the measured response returned, *including* retried attempts and
          the backoff waits between them. It is not the provider's server-side
          processing time, and nothing here can measure that.
        * ``cost_usd`` is computed **only** under a cost guard, at the run's
          pinned pricing policy, in decimal arithmetic. Without a guard this run
          has pinned no price, and a number derived from a table its manifest
          does not record would be a fabricated measurement dressed as a
          recorded one — so it stays ``None``. Under a guard, a response whose
          counts the price policy cannot use is settled as *unmeasured*: its
          reservation becomes exposure and the cost stays null, never zero.

        Returns how the reservation was resolved, for the attempt record that
        has to state it, or ``None`` when there was no reservation.
        """
        cost: Decimal | None = None
        settlement: str | None = None
        breach: CostReservationBreachedError | None = None
        if self._cost_guard is not None and reservation is not None:
            if usage.measurable:
                try:
                    cost = self._cost_guard.settle(
                        reservation,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                except CostReservationBreachedError as exc:
                    cost, breach = exc.measured_usd, exc
            else:
                # No counts this build could read, so nothing to settle against.
                # The reservation stays as exposure rather than resolving to a
                # zero nobody measured.
                self._cost_guard.forfeit(reservation)
            settlement = (
                ATTEMPT_SETTLEMENT_FORFEITED
                if cost is None
                else ATTEMPT_SETTLEMENT_MEASURED
            )
        self._usage = AdapterUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            # Converted once, here, at the boundary where a JSON row is written.
            # Every sum before this point is exact decimal arithmetic; see
            # ``boundarybench.budget``.
            cost_usd=None if cost is None else float(cost),
            latency_seconds=seconds,
        )
        if breach is not None:
            raise breach
        return settlement


__all__ = [
    "MINIMUM_BACKOFF_MULTIPLIER",
    "SDK_MAX_RETRIES",
    "Clock",
    "ProviderRetryPolicy",
    "Sleep",
    "TurnExecutor",
    "check_retry_policy",
]
