"""What can go wrong at a provider boundary, named once for every track.

A provider integration fails in ways an in-process fake cannot: the credential
is refused, the request is rejected, the account is rate limited, the service is
down, the network drops, the answer does not parse. Reporting all of those as
"the adapter raised" would put a wrong credential, an outage and a misconfigured
model id in one bucket, and an operator reading the evidence could not tell which
of them had happened.

So the vocabulary is named here, as closed sets, and it is here rather than in
either track because it is a statement about what a *provider call* may report.
A track's durable reader imports these names to build its taxonomy, so the two
can never drift.

Nothing in this module opens a socket, reads a credential or imports a provider
SDK.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# -- the exception taxonomy ---------------------------------------------------


class AdapterError(RuntimeError):
    """Base class for adapter-side failures."""


class AdapterBusyError(AdapterError):
    """A turn was asked of an adapter instance that is already executing one.

    A state error rather than a provider fault: nothing was asked of the
    provider and nothing about the run is wrong, so it is neither a recorded
    outage nor evidence about a model. It is the caller's own mistake, raised
    where the mistake is.
    """


class SingleFlight:
    """One turn at a time, per adapter instance.

    Every adapter in this build holds per-*instance* state that one turn owns
    while it runs: the last usage, the last attempt evidence, and — where the
    wire is read before an SDK parses it — the captured response the turn's
    checks are taken from. None of it is per-call, so two overlapping turns on
    one adapter interleave into one set of numbers: the second caller's reset
    erases the first's evidence, and a capture holding two bodies leaves both
    turns unable to say which one is theirs. Both failures are silent, and both
    produce durable rows that look exactly like correct ones.

    So an adapter instance is single-flight, and it is stated here rather than
    written four times: an integration that enforced it slightly differently
    would be the drift this contract exists to prevent.

    Acquisition is **non-blocking**, and that is the decision rather than an
    implementation detail. Waiting would make a benchmark's measured wall-clock
    depend on contention it does not record, and a turn that re-entered its own
    adapter — from a transport hook, a callback, a signal handler — would wait
    on a lock it is itself holding and never return. A refusal is immediate,
    happens before any state is reset and before any transport call is made, and
    leaves the turn in flight untouched. Released in a ``finally``, so a turn
    that raised does not wedge the adapter shut, and sequential reuse of one
    adapter — what every episode of every run actually does — is unaffected.
    """

    def __init__(self, *, adapter: str) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()

    def __enter__(self) -> SingleFlight:
        if not self._lock.acquire(blocking=False):
            raise AdapterBusyError(
                f"a {self._adapter} turn is already in flight on this adapter "
                "instance, and an adapter instance executes one turn at a time. "
                "Its usage, its attempt evidence and its response capture are "
                "per-instance state that the turn in flight owns, so a second "
                "overlapping call would reset the first turn's measurement and "
                "leave neither turn able to say which response was its own. This "
                "call is refused rather than queued: no request was made, nothing "
                "was reset, and the turn in flight is untouched. Give each "
                "concurrent caller its own adapter instance"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()


# -- provider faults ---------------------------------------------------------

#: The credential was refused, or it is not permitted to do this. Not retried.
PROVIDER_FAULT_AUTHENTICATION = "provider_authentication"
#: The provider says the *resource this run is pinned to* does not exist or is
#: not available to this account — the model id, most of all. Not retried, and
#: separate from :data:`PROVIDER_FAULT_REQUEST_REJECTED` because of what it
#: proves: the pinned model is part of configuration identity and is sent
#: identically on every turn of every episode, so a provider that cannot find it
#: has answered for the whole plan rather than for one payload. That is what
#: makes this one of exactly two faults a run may stop on.
PROVIDER_FAULT_CONFIGURATION = "provider_configuration"
#: The provider rejected *this request*: a malformed or oversized payload, a
#: field it will not accept. Retrying an invalid request only makes it invalid
#: again — but the next episode sends a different payload, so this says nothing
#: about the rest of the plan and never stops it.
PROVIDER_FAULT_REQUEST_REJECTED = "provider_request_rejected"
#: Anthropic documented that this request was refused because the owning
#: organisation or workspace reached its configured spend limit. Provider-owned
#: rather than a model outcome, and not retried.
PROVIDER_FAULT_SPEND_LIMIT = "provider_spend_limit"
#: Anthropic returned its invalid-request error type, but the privacy-safe
#: allowlist could not distinguish request validation from its documented 400
#: spend-limit case. Deliberately ambiguous and not retried.
PROVIDER_FAULT_INVALID_REQUEST_OR_SPEND_LIMIT = "provider_invalid_request_or_spend_limit"
#: Anthropic answered 400 without a recognised structured error shape. No raw
#: fallback is retained, and the response is not re-described as malformed input.
PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED = "provider_bad_request_unclassified"
#: Rate limited, and still rate limited after the adapter's bounded retries.
PROVIDER_FAULT_RATE_LIMITED = "provider_rate_limited"
#: The service failed on its side and was still failing after those retries.
PROVIDER_FAULT_SERVER_ERROR = "provider_server_error"
#: The request never completed: connection refused, reset, DNS, TLS.
PROVIDER_FAULT_NETWORK_ERROR = "provider_network_error"
#: The request was still outstanding when its per-attempt timeout expired.
PROVIDER_FAULT_TIMEOUT = "provider_timeout"
#: A response arrived and is not one this build can read as a Messages answer.
PROVIDER_FAULT_RESPONSE_INVALID = "provider_response_invalid"

#: Every fault an adapter may report. Closed on purpose: a durable row's
#: classification comes from this set, so an integration cannot invent a new
#: outcome by naming one.
PROVIDER_FAULTS: tuple[str, ...] = (
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED,
    PROVIDER_FAULT_CONFIGURATION,
    PROVIDER_FAULT_INVALID_REQUEST_OR_SPEND_LIMIT,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_REQUEST_REJECTED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_SPEND_LIMIT,
    PROVIDER_FAULT_TIMEOUT,
)

#: The faults a second attempt can answer differently, and therefore the only
#: ones another attempt may follow.
#:
#: Retryability is a property of the fault, not of the caller's patience: a
#: refused credential, a model this account cannot see, a payload the service
#: will not accept and a body this build cannot read all produce exactly the
#: same answer the second time. Stated here, in the contract, rather than only
#: inside the integration that classifies exceptions, because a durable reader
#: has to decide whether a recorded attempt was *allowed* to be followed by
#: another one and it reads rows long after — and without importing — whatever
#: wrote them.
RETRYABLE_PROVIDER_FAULTS: tuple[str, ...] = (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_TIMEOUT,
)


class AdapterProviderError(AdapterError):
    """A named, provider-side fault, carrying only redacted stable detail.

    The message becomes a durable ``failure_event.detail``, so what goes in it
    is a design decision rather than a convenience. A provider's own exception
    string, its response body, its headers and its request id are all excluded:
    they are attacker- and vendor-controlled text of unbounded shape, they can
    quote back part of a request, and a durable row is not the place to discover
    that. What an integration may put here is a fixed sentence plus values from
    a closed set — an HTTP status, an attempt count, the fault's own name.
    """

    def __init__(self, fault: str, detail: str) -> None:
        if fault not in PROVIDER_FAULTS:
            raise ValueError(
                f"{fault!r} is not a provider fault this contract defines; allowed: "
                f"{list(PROVIDER_FAULTS)}"
            )
        super().__init__(detail)
        self.fault = fault


# -- provider attempt evidence -----------------------------------------------

#: The attempt produced a response body from the provider.
ATTEMPT_OUTCOME_RESPONSE = "response"
#: The attempt ended in one of this contract's named provider faults.
ATTEMPT_OUTCOME_FAULT = "fault"
#: The request was made and what came back is not something this integration can
#: classify: not a response it could read, and not one of the named faults
#: either. Its own outcome rather than a fault, because naming a fault this
#: build did not observe would be a fabricated diagnosis — and rather than
#: nothing, because the request *was* sent, it may have been billed, and an
#: attempt missing from the evidence is exposure missing from the accounting.
ATTEMPT_OUTCOME_UNCLASSIFIED = "unclassified_error"
ATTEMPT_OUTCOMES: tuple[str, ...] = (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
)

#: The reservation was replaced by a cost measured from the provider's own
#: reported counts at the run's pinned pricing policy.
ATTEMPT_SETTLEMENT_MEASURED = "measured"
#: The reservation was kept in full as conservative exposure, because this
#: attempt returned no usage this build could read — or none it could attribute
#: to the model this run pinned.
ATTEMPT_SETTLEMENT_FORFEITED = "forfeited"
ATTEMPT_SETTLEMENTS: tuple[str, ...] = (
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
)

# -- why a turn's retry loop stopped ------------------------------------------
#
# A list of attempts says what happened; it does not say whether the list is
# *complete*. Most of the time it does not have to: a response ends the turn, a
# fault that repeating cannot change ends the turn, and a turn that made every
# attempt its run's policy permits could not have made another. The one state
# those rules cannot place is a turn that stopped early after a retryable
# fault — legitimately, because the wall clock ran out, the episode was
# cancelled, the next backoff would not fit, or the run's remaining authorised
# cost could not cover another request. That is indistinguishable, from the
# attempts alone, from a turn whose later attempts were simply deleted.
#
# So the reason is recorded at the exit it happened at, from this closed set,
# and is the minimum needed to tell those two apart.

#: The last attempt returned a response body, which ends the turn: the loop
#: returns and there is nothing left to retry.
TURN_END_RESPONSE = "response"
#: The last attempt met a fault that repeating cannot change.
TURN_END_FAULT_NOT_RETRYABLE = "fault_not_retryable"
#: Every attempt the run's pinned retry policy permits was made, and none of
#: them returned a response.
TURN_END_RETRIES_EXHAUSTED = "retries_exhausted"
#: The request was made and what came back could not be classified at all.
TURN_END_UNCLASSIFIED = "unclassified_error"
#: The turn's remaining wall-clock budget was spent, or the episode was
#: cancelled, before the next request could be made.
TURN_END_DEADLINE_EXCEEDED = "deadline_exceeded"
#: A retry the policy still permitted was abandoned rather than started: the
#: budget that was left could not pay for the backoff *and* the request after
#: it, and sleeping through the rest of it would have learned nothing.
TURN_END_BACKOFF_UNAFFORDABLE = "backoff_unaffordable"
#: The run's remaining authorised cost could not cover the next request, so it
#: was never dispatched.
TURN_END_COST_CAP_EXHAUSTED = "cost_cap_exhausted"
#: A final local configuration check refused the request after authorisation but
#: before the attempt was counted or any socket operation began.
TURN_END_PRE_DISPATCH_REFUSED = "pre_dispatch_refused"

TURN_TERMINAL_REASONS: tuple[str, ...] = (
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_FAULT_NOT_RETRYABLE,
    TURN_END_PRE_DISPATCH_REFUSED,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
    TURN_END_UNCLASSIFIED,
)

#: The reasons a turn may stop while its retry policy still permitted another
#: attempt. Each is a local pre-dispatch fact about this run's budget, clock or
#: admitted client configuration rather than a provider outcome, and none can be
#: derived from the attempts.
TURN_END_EARLY_REASONS: tuple[str, ...] = (
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_PRE_DISPATCH_REFUSED,
)


# -- one SDK exception, as this build's taxonomy reads it ---------------------


@dataclass(frozen=True)
class Fault:
    """What one SDK exception means, and whether repeating helps."""

    fault: str
    retryable: bool
    status: int | None
    #: Whether a provider response reached this process before the SDK raised.
    #: ``False`` for failures in transport. ``True`` for a status exception or an
    #: SDK that read a whole HTTP 200 and then refused to parse it: recording
    #: either as an attempt that received nothing would be a statement this build
    #: can see is false.
    response_received: bool = False

    def __post_init__(self) -> None:
        """Whether a fault is worth repeating is the contract's answer, not an
        integration's second opinion.

        A durable reader decides whether a recorded attempt was allowed to be
        followed by another one from :data:`RETRYABLE_PROVIDER_FAULTS`, and it
        never imports these modules. Two independent tables of the same fact
        would drift, and the direction that matters is the silent one: a fault
        classified as retryable here and terminal there produces real rows the
        reader refuses. So this is bound to that set rather than checked against
        it by hand.
        """
        if self.fault not in PROVIDER_FAULTS:
            raise AdapterError(
                f"{self.fault!r} is not a provider fault this contract defines; "
                f"allowed: {list(PROVIDER_FAULTS)}"
            )
        if self.retryable != (self.fault in RETRYABLE_PROVIDER_FAULTS):
            raise AdapterError(
                f"this integration classifies {self.fault!r} as "
                f"{'retryable' if self.retryable else 'terminal'}, and the adapter "
                "contract does not. Retryability is a property of the fault and is "
                f"named once, in RETRYABLE_PROVIDER_FAULTS: "
                f"{list(RETRYABLE_PROVIDER_FAULTS)}"
            )


def fault_detail(fault: Fault, *, attempts: int, reason: str) -> str:
    """The durable sentence one provider fault is recorded as.

    Assembled from a fixed template, the fault's own name, an HTTP status and a
    count — and nothing else. The provider's message, body, headers and request
    id are all in hand at the call site and all deliberately dropped: a durable
    row is evidence, and vendor- or attacker-controlled text of unbounded shape
    does not belong in one.
    """
    answered = (
        f"the provider answered HTTP {fault.status}"
        if fault.status is not None
        else "the request to the provider did not complete"
    )
    return (
        f"{answered} ({fault.fault}) after {attempts} attempt(s); {reason}. The "
        "provider's own message, response body, headers and request id are not "
        "recorded here by design"
    )


def classify_http_status(status: int) -> Fault:
    """One HTTP status, as this build's fault taxonomy reads it.

    Shared because it is not a provider's decision: these are the HTTP
    semantics, and three integrations reading them three ways would put the same
    outage in three buckets. What *is* per-provider is which SDK exception
    carries a status at all, and that stays in each integration.

    404 is the one 4xx that is about the *run* rather than the request: an API
    answers it when the model or resource named in the URL or body does not
    exist for this account. The model id is pinned in configuration identity and
    is sent identically on every turn of every episode, so this cannot come out
    differently later, which is what earns it the right to stop the whole plan.
    Every other 4xx is about the payload this turn happened to send and stays
    one episode's failure.

    Reaching this function means an SDK supplied an HTTP status, which in turn
    means an HTTP response reached it. The response is evidence of arrival, not
    an accepted model answer: callers retain no model output or usage from it.
    """
    if status in (401, 403):
        return Fault(PROVIDER_FAULT_AUTHENTICATION, False, status, response_received=True)
    if status == 404:
        return Fault(PROVIDER_FAULT_CONFIGURATION, False, status, response_received=True)
    if status == 408:
        return Fault(PROVIDER_FAULT_TIMEOUT, True, status, response_received=True)
    if status == 429:
        return Fault(PROVIDER_FAULT_RATE_LIMITED, True, status, response_received=True)
    if status >= 500:
        return Fault(PROVIDER_FAULT_SERVER_ERROR, True, status, response_received=True)
    return Fault(PROVIDER_FAULT_REQUEST_REJECTED, False, status, response_received=True)


__all__ = [
    "ATTEMPT_OUTCOMES",
    "ATTEMPT_OUTCOME_FAULT",
    "ATTEMPT_OUTCOME_RESPONSE",
    "ATTEMPT_OUTCOME_UNCLASSIFIED",
    "ATTEMPT_SETTLEMENTS",
    "ATTEMPT_SETTLEMENT_FORFEITED",
    "ATTEMPT_SETTLEMENT_MEASURED",
    "PROVIDER_FAULTS",
    "PROVIDER_FAULT_AUTHENTICATION",
    "PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED",
    "PROVIDER_FAULT_CONFIGURATION",
    "PROVIDER_FAULT_INVALID_REQUEST_OR_SPEND_LIMIT",
    "PROVIDER_FAULT_NETWORK_ERROR",
    "PROVIDER_FAULT_RATE_LIMITED",
    "PROVIDER_FAULT_REQUEST_REJECTED",
    "PROVIDER_FAULT_RESPONSE_INVALID",
    "PROVIDER_FAULT_SERVER_ERROR",
    "PROVIDER_FAULT_SPEND_LIMIT",
    "PROVIDER_FAULT_TIMEOUT",
    "RETRYABLE_PROVIDER_FAULTS",
    "TURN_END_BACKOFF_UNAFFORDABLE",
    "TURN_END_COST_CAP_EXHAUSTED",
    "TURN_END_DEADLINE_EXCEEDED",
    "TURN_END_EARLY_REASONS",
    "TURN_END_FAULT_NOT_RETRYABLE",
    "TURN_END_PRE_DISPATCH_REFUSED",
    "TURN_END_RESPONSE",
    "TURN_END_RETRIES_EXHAUSTED",
    "TURN_END_UNCLASSIFIED",
    "TURN_TERMINAL_REASONS",
    "AdapterBusyError",
    "AdapterError",
    "AdapterProviderError",
    "Fault",
    "SingleFlight",
    "classify_http_status",
    "fault_detail",
]
