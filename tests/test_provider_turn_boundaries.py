"""What the three new adapters do at the boundaries of one turn.

:mod:`tests.test_provider_adapter_hardening` covers what an adapter refuses from
a hostile client, body or policy. This module covers the paths *around* that
refusal — the ones a turn takes when the budget is gone before a request can be
made, when the provider answers with a shape the SDK parsed but this build
cannot read, when the answer measures more than the run authorised, and when
something inside the integration itself breaks. Each of them ends in durable
evidence, and each of them is a different sentence that evidence has to say.

The same rules apply as everywhere else in this suite: every test runs the real
installed SDK over an in-process ``httpx.MockTransport``, nothing opens a
socket, and no test asserts on the shape of an implementation it could not
observe from the outside.
"""

from __future__ import annotations

import socket
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from mistralai.client.errors import NoResponseError

from boundarybench.adapter import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_FAULT_NOT_RETRYABLE,
    TURN_END_PRE_DISPATCH_REFUSED,
    TURN_END_RESPONSE,
    TURN_END_UNCLASSIFIED,
    AdapterError,
    AdapterProtocolError,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import (
    CostCapExceededError,
    CostControls,
    CostReservationBreachedError,
    RunCostGuard,
)
from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.pricing import price_for
from boundarybench.providers import common, xai_openai_compat
from boundarybench.providers.common import (
    MAX_RESPONSE_DEPTH,
    Fault,
    ProviderRetryPolicy,
    TokenUsage,
    TurnExecutor,
)
from boundarybench.providers.mistral_chat import (
    MISTRAL_BASE_URL,
    MistralChatAdapter,
    MistralConfigurationError,
    build_mistral_adapter,
    check_client_endpoint,
    mistral_identity,
)
from boundarybench.providers.mistral_chat import (
    RequestProfile as MistralRequestProfile,
)
from boundarybench.providers.mistral_chat import (
    classify_exception as mistral_classify,
)
from boundarybench.providers.mistral_chat import (
    request_profile_for as mistral_profile_for,
)
from boundarybench.providers.openai_responses import (
    OPENAI_BASE_URL,
    OpenAIConfigurationError,
    OpenAIResponsesAdapter,
    build_openai_adapter,
    openai_identity,
)
from boundarybench.providers.openai_responses import (
    RequestProfile as OpenAIRequestProfile,
)
from boundarybench.providers.openai_responses import (
    build_client as build_openai_client,
)
from boundarybench.providers.openai_responses import (
    classify_exception as openai_classify,
)
from boundarybench.providers.openai_responses import (
    request_profile_for as openai_profile_for,
)
from boundarybench.providers.xai_openai_compat import (
    XAI_BASE_URL,
    XAIConfigurationError,
    XAIOpenAICompatAdapter,
    build_xai_adapter,
    xai_identity,
)
from boundarybench.providers.xai_openai_compat import (
    RequestProfile as XAIRequestProfile,
)
from boundarybench.providers.xai_openai_compat import (
    build_client as build_xai_client,
)
from boundarybench.providers.xai_openai_compat import (
    check_response_model as xai_check_response_model,
)
from boundarybench.providers.xai_openai_compat import (
    classify_exception as xai_classify,
)
from boundarybench.providers.xai_openai_compat import (
    request_profile_for as xai_profile_for,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from operatebench.providers.wire import WireResponse
from tests import mistral_transport, openai_transport
from tests.conftest import minimal_card

OPENAI_MODEL = "gpt-5.6-luna"
XAI_MODEL = "grok-4.5"
MISTRAL_MODEL = "mistral-small-2603"

#: Not a credential, and never sent anywhere: every test that hands this to a
#: factory asserts on the object the factory returned and dispatches nothing.
FAKE_KEY = "test-key-not-a-credential"

#: A configuration identity of the right shape. Its digest names nothing: the
#: point of the tests that use it is that an identity is checked for shape and
#: for membership before a credential is read, not that this one is real.
AUDITED_ID = "b" * 64


def _request() -> TurnRequest:
    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(cube.variants[0]),
        turns_remaining=12,
    )


def _deadline(seconds: float = 30.0, *, cancelled: bool = False) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: cancelled)


def _guard(
    provider: str,
    model: str,
    *,
    cap: Decimal = Decimal("100"),
    max_output_tokens: int = 1024,
) -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=cap,
            max_episodes=1,
            price=price_for(provider=provider, model=model),
        ),
        max_output_tokens=max_output_tokens,
        max_attempts_per_turn=3,
    )


def _openai_body(**kwargs: Any) -> dict[str, Any]:
    return openai_transport.responses_body(
        [openai_transport.function_call_item("read_records", "{}")],
        model=OPENAI_MODEL,
        **kwargs,
    )


def _xai_body(**kwargs: Any) -> dict[str, Any]:
    return openai_transport.chat_body(
        [openai_transport.tool_call("read_records", "{}")], model=XAI_MODEL, **kwargs
    )


def _mistral_body(**kwargs: Any) -> dict[str, Any]:
    return mistral_transport.chat_body(
        [mistral_transport.tool_call("read_records", "{}")],
        model=MISTRAL_MODEL,
        **kwargs,
    )


def _openai_turn(body: Any, *, guard: RunCostGuard | None = None, steps: int = 1):
    transport, client = openai_transport.scripted_client(*([body] * steps))
    return transport, OpenAIResponsesAdapter(
        model=OPENAI_MODEL, client=client, cost_guard=guard, sleep=lambda _s: None
    )


def _xai_turn(body: Any, *, guard: RunCostGuard | None = None, steps: int = 1):
    transport, client = openai_transport.xai_scripted_client(*([body] * steps))
    return transport, XAIOpenAICompatAdapter(
        model=XAI_MODEL, client=client, cost_guard=guard, sleep=lambda _s: None
    )


def _mistral_turn(body: Any, *, guard: RunCostGuard | None = None, steps: int = 1):
    transport, client = mistral_transport.scripted_client(*([body] * steps))
    return transport, MistralChatAdapter(
        model=MISTRAL_MODEL, client=client, cost_guard=guard, sleep=lambda _s: None
    )


# -- a turn that never reaches the provider -----------------------------------


@pytest.mark.parametrize(
    "deadline",
    [_deadline(0.0), _deadline(30.0, cancelled=True)],
    ids=["budget spent", "cancelled"],
)
def test_a_turn_with_no_wall_clock_left_makes_no_request(deadline: TurnDeadline) -> None:
    """The deadline is checked before the attempt, not after it.

    A spent or cancelled episode must not be able to open one more socket: the
    request would be made on a budget the run no longer has, and the turn that
    reported it would be the turn that had already ended. The evidence still
    exists and honestly reports zero attempts — a turn that made no request is a
    different thing from a turn whose request failed.
    """
    transport, adapter = _openai_turn(_openai_body())
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), deadline)
    assert raised.value.fault == PROVIDER_FAULT_TIMEOUT
    assert transport.calls == 0

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 0
    assert telemetry["terminal_reason"] == TURN_END_DEADLINE_EXCEEDED
    assert telemetry["response_received"] is False


def test_a_turn_the_cost_cap_cannot_cover_is_refused_before_the_socket() -> None:
    """The guard is asked before the attempt is counted, so nothing is spent.

    The refusal comes out as the cost-cap error the runner already knows, not as
    a provider fault: nothing about the provider went wrong, and dressing an
    exhausted budget as an outage would put a run's own spending decision in the
    bucket that means "the service failed".
    """
    guard = _guard("openai", OPENAI_MODEL, cap=Decimal("0.000001"))
    transport, adapter = _openai_turn(_openai_body(), guard=guard)
    with pytest.raises(CostCapExceededError):
        adapter.next_call(_request(), _deadline())
    assert transport.calls == 0

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 0
    assert telemetry["terminal_reason"] == TURN_END_COST_CAP_EXHAUSTED
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd == Decimal(0)


def test_a_failure_that_is_not_the_providers_is_recorded_as_this_builds_own() -> None:
    """An exception the classifier does not recognise is not made into an outage.

    ``classify`` returning ``None`` means "this is not the provider's doing".
    The exception is then let out exactly as it arrived, for the runner to record
    as this integration's failure — but the attempt is still published, because
    the request really was made, it really took time, and the reservation it
    forfeited is money the run has to account for. Silently reclassifying it
    would produce a ledger row blaming a provider for a bug in this build.
    """
    guard = _guard("openai", OPENAI_MODEL)
    executor: TurnExecutor[object] = TurnExecutor(
        retry=ProviderRetryPolicy(), sleep=lambda _s: None, cost_guard=guard
    )

    def dispatch(_timeout: float) -> object:
        raise RuntimeError("something inside this integration broke")

    with pytest.raises(RuntimeError):
        executor.run(
            payload={"model": OPENAI_MODEL},
            deadline=_deadline(),
            token_bound=64,
            dispatch=dispatch,
            check_before_dispatch=lambda: None,
            classify=lambda _exception: None,
            check_identity=lambda _response: None,
            usage_of=lambda _response: TokenUsage(1, 1),
        )

    telemetry = executor.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["terminal_reason"] == TURN_END_UNCLASSIFIED
    assert telemetry["attempts"][0]["outcome"] == ATTEMPT_OUTCOME_UNCLASSIFIED
    assert telemetry["attempts"][0]["fault"] is None
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


def test_a_later_pre_dispatch_refusal_preserves_the_attempt_already_made() -> None:
    guard = _guard("xai", XAI_MODEL)
    executor: TurnExecutor[object] = TurnExecutor(
        retry=ProviderRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0.0,
            backoff_multiplier=1.0,
        ),
        sleep=lambda _seconds: None,
        cost_guard=guard,
    )
    checks = 0

    def check_before_dispatch() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise XAIConfigurationError("the final local client check refused")

    def dispatch(_timeout: float) -> object:
        raise OSError("offline")

    with pytest.raises(XAIConfigurationError):
        executor.run(
            payload={"model": XAI_MODEL},
            deadline=_deadline(),
            token_bound=64,
            dispatch=dispatch,
            check_before_dispatch=check_before_dispatch,
            classify=lambda _exception: Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None),
            check_identity=lambda _response: None,
            usage_of=lambda _response: TokenUsage(1, 1),
        )

    telemetry = executor.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.attempts[0].fault == PROVIDER_FAULT_NETWORK_ERROR
    assert telemetry.attempts[0].cost_settlement == ATTEMPT_SETTLEMENT_FORFEITED
    assert telemetry.terminal_reason == TURN_END_PRE_DISPATCH_REFUSED
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


def test_a_pre_dispatch_cancellation_failure_is_visible_with_fresh_telemetry() -> None:
    class BrokenCancellationGuard:
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            return Decimal("0.25")

        def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
            assert input_tokens_upper_bound == 64
            raise RuntimeError("cancellation invariant failed")

        def settle(
            self,
            reservation: Decimal,
            *,
            input_tokens: int | None,
            output_tokens: int | None,
        ) -> Decimal | None:
            raise AssertionError("a refused dispatch must not settle")

        def forfeit(self, reservation: Decimal) -> None:
            raise AssertionError("a refused dispatch must not forfeit")

    def refuse_configuration() -> None:
        raise XAIConfigurationError("configuration refused")

    executor: TurnExecutor[object] = TurnExecutor(
        retry=ProviderRetryPolicy(), cost_guard=BrokenCancellationGuard()
    )

    with pytest.raises(RuntimeError, match="cancellation invariant failed"):
        executor.run(
            payload={"model": XAI_MODEL},
            deadline=_deadline(),
            token_bound=64,
            dispatch=lambda _timeout: pytest.fail("dispatch must not run"),
            check_before_dispatch=refuse_configuration,
            classify=lambda _exception: None,
            check_identity=lambda _response: None,
            usage_of=lambda _response: TokenUsage(1, 1),
        )

    telemetry = executor.last_telemetry()
    assert telemetry.attempt_count == 0
    assert telemetry.terminal_reason == TURN_END_PRE_DISPATCH_REFUSED


def test_an_entry_refusal_publisher_replaces_prior_turn_measurements_only() -> None:
    """A local refusal before ``run`` is fresh evidence, not the prior turn's."""
    guard = _guard("xai", XAI_MODEL)
    readings = iter((1.0, 1.0, 2.0, 5.0))
    executor: TurnExecutor[object] = TurnExecutor(
        retry=ProviderRetryPolicy(),
        clock=lambda: next(readings),
        cost_guard=guard,
    )
    executor.run(
        payload={"model": XAI_MODEL},
        deadline=_deadline(),
        token_bound=64,
        dispatch=lambda _timeout: object(),
        check_before_dispatch=lambda: None,
        classify=lambda _exception: None,
        check_identity=lambda _response: None,
        usage_of=lambda _response: TokenUsage(11, 7),
    )
    assert executor.last_telemetry().attempt_count == 1
    assert executor.last_usage().input_tokens == 11
    before = (guard.committed_usd, guard.measured_usd, guard.exposure_usd)

    executor.record_pre_dispatch_refusal(started_at=3.0)

    assert executor.last_usage().as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }
    telemetry = executor.last_telemetry()
    assert telemetry.attempts == ()
    assert telemetry.turn_latency_seconds == 2.0
    assert telemetry.terminal_reason == TURN_END_PRE_DISPATCH_REFUSED
    assert (guard.committed_usd, guard.measured_usd, guard.exposure_usd) == before


def test_raw_response_id_absence_refuses_a_typed_synthesized_identifier() -> None:
    response: WireResponse[SimpleNamespace] = WireResponse(
        "{}",
        lambda: SimpleNamespace(id="provider-synthesized-id"),
        kind="test response",
    )

    with pytest.raises(AdapterProviderError) as raised:
        response.admit_response_id()

    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert response.response_id_digest_sha256 is None


# -- what a turn costs when the answer arrives --------------------------------


def test_a_response_measuring_more_than_it_reserved_stops_the_run() -> None:
    """The spend is banked at what it measured, and the run stops on the breach.

    A provider that reports far more input tokens than the request could have
    carried commits spend the cap never authorised. Under-stating it would be
    the worse error, so the measurement is recorded at its measured value — and
    the attempt evidence is published *before* the breach is raised, because the
    row that stops a run is exactly the row that has to show what the run had
    spent when it did.
    """
    guard = _guard("openai", OPENAI_MODEL, max_output_tokens=8)
    body = _openai_body()
    body["usage"]["input_tokens"] = 5_000_000
    body["usage"]["output_tokens"] = 8
    body["usage"]["total_tokens"] = 5_000_008
    _transport, adapter = _openai_turn(body, guard=guard)

    with pytest.raises(CostReservationBreachedError):
        adapter.next_call(_request(), _deadline())

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["response_received"] is True
    assert telemetry["usage_reported"] is True
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_MEASURED
    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] == 5_000_000
    assert usage["cost_usd"] is not None and usage["cost_usd"] > 0
    assert guard.reservation_breached is True
    assert guard.measured_usd > 0


@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_an_answer_reporting_no_usage_is_unmeasured_and_not_free(provider: str) -> None:
    """A missing usage block is refused, and is never rendered as zero.

    A turn this build cannot price is a turn it must not settle. The counts are
    what a run's cost cap is enforced against, so a body that states none is not
    a cheap answer or an unmeasured one — it is a body that does not carry the
    contract this run asked for, and it is refused as one.

    The economics are the part that does not change: the reservation that
    authorised the request stays on the guard as exposure rather than settling
    to a cost nobody measured, so a run whose provider stopped reporting counts
    still cannot read as a run that became free.
    """
    if provider == "openai":
        guard = _guard("openai", OPENAI_MODEL)
        body = _openai_body()
        body["usage"] = None
        _transport, adapter = _openai_turn(body, guard=guard)
    else:
        guard = _guard("xai", XAI_MODEL)
        body = _xai_body()
        body["usage"] = None
        _transport, adapter = _xai_turn(body, guard=guard)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["cost_usd"] is None
    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["response_received"] is True
    assert telemetry["usage_reported"] is False
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


def test_one_reported_count_and_one_absent_count_is_still_unmeasured() -> None:
    """Half a measurement is not a measurement, and is not half a price.

    ``null`` on one of the two counts is a usage block that does not state what
    this API documents there. The turn cannot be priced from it, and pricing it
    as though the missing half were zero would under-report real spend — so the
    body is refused and the whole reservation is kept as exposure.
    """
    guard = _guard("openai", OPENAI_MODEL)
    body = _openai_body()
    body["usage"]["input_tokens"] = None
    _transport, adapter = _openai_turn(body, guard=guard)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] is None
    # Neither half is banked. The count that *was* reported is not a measurement
    # on its own, and recording it beside a null would put half a turn's spend
    # into a row that reads as a whole one.
    assert usage["output_tokens"] is None
    assert usage["cost_usd"] is None
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


# -- the count read that fails after the answer already arrived ---------------
#
# Every surface validates its usage block twice: once while the response is
# being admitted, and once in the function whose return value becomes a settled
# cost. The second one is the one that matters here. It runs *after* the turn
# has an outstanding reservation and after a body has arrived, so whatever it
# raises has to leave the same three things behind as every other exit from the
# loop: a resolved reservation, exactly one attempt that says what happened, and
# a named terminal reason. A turn that raised out of it with the reservation
# still pending would leave the guard unable to start the next episode at all.


def _refused_usage_turn(
    usage_of: Any, *, guard: RunCostGuard
) -> tuple[TurnExecutor[object], list[float], Exception]:
    """One turn whose response arrives intact and whose count read then fails.

    Returns the executor, the timeout each dispatch was given, and the exception
    the turn raised — which every caller asserts on, because *which* one came
    out is half of what these tests are about.
    """
    executor: TurnExecutor[object] = TurnExecutor(
        retry=ProviderRetryPolicy(), sleep=lambda _s: None, cost_guard=guard
    )
    dispatched: list[float] = []

    def dispatch(timeout: float) -> object:
        dispatched.append(timeout)
        return object()

    executor.clear()
    try:
        executor.run(
            payload={"model": OPENAI_MODEL},
            deadline=_deadline(),
            token_bound=64,
            dispatch=dispatch,
            check_before_dispatch=lambda: None,
            classify=lambda _exception: None,
            check_identity=lambda _response: None,
            usage_of=usage_of,
        )
    except Exception as exception:
        return executor, dispatched, exception
    raise AssertionError("a turn whose count read failed must not return a response")


def test_a_refused_count_read_forfeits_the_reservation_and_names_the_fault() -> None:
    """A response this build cannot read a count off of ends the turn, once.

    The count read is the second place a usage block is proven, and it runs with
    a reservation outstanding. When it refuses the response the turn is over:
    repeating the request cannot make the same body readable, so nothing is
    retried, the reservation stays whole as exposure rather than settling to a
    cost nobody measured, and the single attempt records what actually happened
    — a response *did* come back, and it reported no usage this build could use.

    The exception is re-raised exactly as it arrived. It already carries this
    contract's fault name and this build's own fixed detail, and rebuilding it
    here would either lose that detail or invent one.
    """
    guard = _guard("openai", OPENAI_MODEL)
    refusal = AdapterProviderError(
        PROVIDER_FAULT_RESPONSE_INVALID, "a count this build cannot price a turn on"
    )

    def usage_of(_response: object) -> TokenUsage:
        raise refusal

    executor, dispatched, raised = _refused_usage_turn(usage_of, guard=guard)
    assert raised is refusal

    telemetry = executor.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["terminal_reason"] == TURN_END_FAULT_NOT_RETRYABLE
    attempt = telemetry["attempts"][0]
    assert attempt["outcome"] == ATTEMPT_OUTCOME_FAULT
    assert attempt["fault"] == PROVIDER_FAULT_RESPONSE_INVALID
    assert attempt["response_received"] is True
    assert attempt["usage_reported"] is False
    assert attempt["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED
    assert telemetry["response_received"] is True
    assert telemetry["usage_reported"] is False
    # One request, and no second one: repeating it cannot make the same body
    # readable, so this fault is not retried.
    assert len(dispatched) == 1

    usage = executor.last_usage().as_dict()
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["cost_usd"] is None


def test_a_count_read_that_breaks_inside_this_build_is_not_a_provider_outage() -> None:
    """An unexpected failure in the count read is named as this build's own.

    Nothing about the provider went wrong: a body arrived and this integration's
    own reading of it raised something it never classified. Recording that as a
    provider fault would put a bug in this build into the bucket a reader counts
    outages from, so the attempt takes the outcome this contract already has for
    "the request was made and what came back is not something this integration
    can classify" — and names no fault at all.

    Everything the accounting needs is unchanged. The response really did
    arrive, so the attempt says so; the reservation is money the run has to
    account for either way, so it is forfeited; and the exception is let out
    exactly as it arrived, for the runner to record as this build's failure.
    """
    guard = _guard("openai", OPENAI_MODEL)
    broken = RuntimeError("the count read in this integration broke")

    def usage_of(_response: object) -> TokenUsage:
        raise broken

    executor, dispatched, raised = _refused_usage_turn(usage_of, guard=guard)
    assert raised is broken

    telemetry = executor.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["terminal_reason"] == TURN_END_UNCLASSIFIED
    attempt = telemetry["attempts"][0]
    assert attempt["outcome"] == ATTEMPT_OUTCOME_UNCLASSIFIED
    assert attempt["fault"] is None
    assert attempt["response_received"] is True
    assert attempt["usage_reported"] is False
    assert attempt["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED
    assert telemetry["fault_counts"] == {}
    assert len(dispatched) == 1

    usage = executor.last_usage().as_dict()
    assert usage["input_tokens"] is None
    assert usage["cost_usd"] is None


@pytest.mark.parametrize(
    "failure",
    [
        AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, "an unreadable count"),
        RuntimeError("the count read in this integration broke"),
    ],
    ids=["provider fault", "this build's own"],
)
def test_a_turn_whose_count_read_failed_leaves_the_guard_able_to_go_on(
    failure: Exception,
) -> None:
    """The reservation is resolved, so the next episode can begin at once.

    This is the property the whole path exists for. The guard permits exactly
    one outstanding request, so a reservation left pending by a turn that raised
    is not a stale number in a report — it is a run that cannot start another
    episode, reporting a budget error about a request nobody can now account
    for. Resolved as exposure and not as a settlement: no count was read, so
    nothing was measured, and the conservative bound is what stands.
    """
    guard = _guard("openai", OPENAI_MODEL)

    def usage_of(_response: object) -> TokenUsage:
        raise failure

    _executor, _dispatched, raised = _refused_usage_turn(usage_of, guard=guard)
    assert raised is failure

    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)
    assert guard.committed_usd == guard.exposure_usd
    guard.begin_episode()


@pytest.mark.parametrize(
    "failure",
    [
        AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, "an unreadable count"),
        RuntimeError("the count read in this integration broke"),
    ],
    ids=["provider fault", "this build's own"],
)
def test_a_turn_after_a_failed_count_read_reports_only_its_own_answer(
    failure: Exception,
) -> None:
    """The next turn's evidence is the next turn's, with nothing carried over.

    A failed count read publishes an attempt and leaves the usage empty. Both
    are cleared before the following turn does anything, so a run whose second
    turn succeeded reports one attempt that got a response and the counts that
    response actually stated — never the fault from the turn before it, and
    never a measurement from a turn that took none.
    """
    guard = _guard("openai", OPENAI_MODEL)

    def refusing(_response: object) -> TokenUsage:
        raise failure

    executor, _dispatched, raised = _refused_usage_turn(refusing, guard=guard)
    assert raised is failure

    executor.clear()
    assert executor.last_telemetry().as_dict()["terminal_reason"] is None
    executor.run(
        payload={"model": OPENAI_MODEL},
        deadline=_deadline(),
        token_bound=64,
        dispatch=lambda _timeout: object(),
        check_before_dispatch=lambda: None,
        classify=lambda _exception: None,
        check_identity=lambda _response: None,
        usage_of=lambda _response: TokenUsage(11, 7),
    )

    telemetry = executor.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["terminal_reason"] == TURN_END_RESPONSE
    assert telemetry["fault_counts"] == {}
    assert telemetry["attempts"][0]["outcome"] == ATTEMPT_OUTCOME_RESPONSE
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_MEASURED
    usage = executor.last_usage().as_dict()
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert guard.measured_usd > 0


def test_a_real_body_whose_counts_only_the_wire_refuses_ends_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same exit, reached through a real adapter over a real SDK.

    A surface admits a body before it prices it, and then proves the usage block
    a second time in the function whose return value becomes a settled cost. The
    second proof is not redundant: it is the one guarding the number that is
    multiplied by a rate and committed against the cap, and it is the only one
    left if an admissibility check is ever narrowed — as it is here, to the
    identity question this contract minimally requires.

    So the body is a real one, the client is the shipped one over an in-process
    transport, and the count that cannot be read is a JSON ``true`` — which this
    SDK hands to its typed model as one token, and which the wire reading
    refuses. What comes out is the shared loop's exit and not a crash four
    frames down: one attempt, a response received, no usage reported, the
    reservation kept whole as exposure, and a guard ready for the next episode.
    """

    def identity_only(response: Any, *, model: str) -> None:
        xai_check_response_model(response.parsed, model=model)

    monkeypatch.setattr(xai_openai_compat, "check_response_admissible", identity_only)
    guard = _guard("xai", XAI_MODEL)
    body = _xai_body()
    body["usage"]["prompt_tokens"] = True
    transport, adapter = _xai_turn(body, guard=guard)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert transport.calls == 1

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["terminal_reason"] == TURN_END_FAULT_NOT_RETRYABLE
    attempt = telemetry["attempts"][0]
    assert attempt["outcome"] == ATTEMPT_OUTCOME_FAULT
    assert attempt["fault"] == PROVIDER_FAULT_RESPONSE_INVALID
    assert attempt["response_received"] is True
    assert attempt["usage_reported"] is False
    assert attempt["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED

    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] is None
    assert usage["cost_usd"] is None
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)
    guard.begin_episode()


# -- a body the SDK parsed and this build still cannot read -------------------


def test_a_response_nested_past_this_builds_limit_is_refused_by_name() -> None:
    """A body built to recurse forever fails as a stated limit, not as a crash.

    The contract check walks the parsed response looking for fields the
    provider's own schema does not declare, and a body can nest arbitrarily
    deep. Without a bound, the failure would be a ``RecursionError`` from
    whichever frame happened to be deepest — text this build did not choose,
    raised from a place it cannot describe.
    """
    body = _openai_body()
    nest: dict[str, Any] = {}
    cursor = nest
    for _ in range(MAX_RESPONSE_DEPTH + 8):
        cursor["provider_authored_key"] = {}
        cursor = cursor["provider_authored_key"]
    body["metadata"] = nest
    _transport, adapter = _openai_turn(body)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert str(MAX_RESPONSE_DEPTH) in str(raised.value)
    assert "provider_authored_key" not in str(raised.value)


def test_an_openai_answer_that_did_not_complete_is_a_protocol_failure() -> None:
    """A status other than completed carries no action, and does not name itself.

    Truncation at this run's own ceiling is a separate, classified outcome. Every
    other unfinished status is the model's answer failing the protocol, and the
    status string the provider stated is not repeated into a durable row.
    """
    _transport, adapter = _openai_turn(_openai_body(status="failed"))
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "did not complete" in str(raised.value)
    assert "failed" not in str(raised.value)


def test_an_xai_answer_with_no_choices_states_no_action() -> None:
    body = openai_transport.chat_body([], model=XAI_MODEL)
    body["choices"] = []
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "0 choice(s)" in str(raised.value)


def test_xai_tool_calls_that_are_not_a_list_are_refused_by_name() -> None:
    """This SDK parses a response leniently, so the shape has to be checked.

    An object where the API documents a list is not a count of actions. Left
    alone it would reach ``len()`` as a ``TypeError`` carrying the interpreter's
    words, after the turn had already been measured and settled.

    Refused as a *response* fault rather than a protocol failure, because that
    is what it is: a body that does not state the shape this API documents says
    nothing about how the model behaved, and filing it under the model would put
    a malformed body in the bucket that means "the model answered badly".
    """
    body = openai_transport.chat_body([], model=XAI_MODEL)
    body["choices"][0]["message"]["tool_calls"] = {"not": "a list"}
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "does not carry a list of tool calls" in str(raised.value)
    assert "TypeError" not in str(raised.value)


def test_an_xai_tool_call_that_is_not_a_function_call_is_refused() -> None:
    """This SDK's tool-call union has a second arm, and it is not an action.

    A "custom" tool call carries free-form text where this scaffold requires a
    named action with structured arguments, so reading one would mean guessing
    what the model asked for.
    """
    body = openai_transport.chat_body(
        [{"id": "call_test", "type": "custom", "custom": {"name": "x", "input": "y"}}],
        model=XAI_MODEL,
    )
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "not a function call" in str(raised.value)


def test_an_xai_function_call_with_no_function_object_is_refused() -> None:
    """It declares itself an action and names none, which is not an adapter bug.

    The SDK's own model for a function tool call requires the function object,
    so a body that omits it is not the documented shape and is refused on the
    wire — before the attribute nobody can read is reached, and before the turn
    is measured.
    """
    body = openai_transport.chat_body(
        [{"id": "call_test", "type": "function"}], model=XAI_MODEL
    )
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "AttributeError" not in str(raised.value)


@pytest.mark.parametrize("count", [0, 2], ids=["none", "two"])
def test_a_mistral_answer_that_is_not_exactly_one_choice_is_refused(count: int) -> None:
    """One completion per turn: neither none nor two names what the agent did."""
    body = _mistral_body()
    choice = body["choices"][0]
    body["choices"] = [dict(choice, index=index) for index in range(count)]
    _transport, adapter = _mistral_turn(body)
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert f"{count} choice(s)" in str(raised.value)


def test_a_mistral_choice_with_no_message_is_a_protocol_failure() -> None:
    body = mistral_transport.chat_body([], model=MISTRAL_MODEL)
    body["choices"][0]["message"] = None
    _transport, adapter = _mistral_turn(body)
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "no message" in str(raised.value)


# -- one SDK exception, as this build's fault taxonomy reads it ----------------


@pytest.mark.parametrize("provider", ["openai", "xai"])
def test_a_read_timeout_is_a_retryable_timeout_fault(provider: str) -> None:
    """A timeout is the one transport failure worth repeating, and is retried.

    Raised from the transport itself rather than simulated, so the SDK maps it to
    its own timeout class exactly as it would in a live run, and the adapter's
    bounded retry policy is what decides how many times it is tried.
    """
    timeouts = [httpx.ReadTimeout("timed out")] * 3
    if provider == "openai":
        transport = openai_transport.RecordingTransport(timeouts)
        adapter: Any = OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=transport.client(), sleep=lambda _s: None
        )
    else:
        transport = openai_transport.RecordingTransport(timeouts)
        adapter = XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=transport.xai_client(), sleep=lambda _s: None
        )
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_TIMEOUT
    assert transport.calls == 3
    assert adapter.last_telemetry().as_dict()["attempt_count"] == 3


def test_an_unreadable_body_is_terminal_and_a_missing_one_is_not() -> None:
    """Two failures the SDKs report as one family, told apart on purpose.

    A response the SDK could not validate arrived — repeating the request cannot
    change what the service sends — so it is terminal. A response that never
    arrived at all is a transport failure and is worth one more attempt. Both are
    the provider's doing, and the contract, not this integration, decides which
    of them may be retried.
    """
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    unreadable = openai.APIResponseValidationError(
        response=httpx.Response(200, json={}, request=request), body=None
    )
    for classify in (openai_classify, xai_classify):
        fault = classify(unreadable)
        assert fault is not None
        assert fault.fault == PROVIDER_FAULT_RESPONSE_INVALID
        assert fault.retryable is False

    absent = mistral_classify(NoResponseError("no response received"))
    assert absent is not None
    assert absent.fault == PROVIDER_FAULT_NETWORK_ERROR
    assert absent.retryable is True


@pytest.mark.parametrize(
    "classify",
    [openai_classify, xai_classify, mistral_classify],
    ids=lambda f: f.__module__,
)
def test_an_exception_no_sdk_raised_is_not_classified_as_an_outage(classify: Any) -> None:
    """``None`` is the honest answer: this build broke, and should say so.

    Mapping an arbitrary exception onto a provider fault would let a bug in this
    integration be recorded as a service failure — and, if it were mapped to a
    retryable one, be repeated against a real endpoint at real cost.
    """
    assert classify(RuntimeError("not an SDK error")) is None


def test_a_fault_this_contract_does_not_define_cannot_be_constructed() -> None:
    """Retryability is named once, in the contract, and bound to it here.

    The ledger decides whether a recorded attempt was allowed to be followed by
    another one, and it never imports these modules. A fault classified as
    retryable in an integration and terminal in the contract would produce real
    rows the reader refuses, so the disagreement is refused at construction
    instead.
    """
    with pytest.raises(AdapterError):
        Fault("provider_ran_out_of_ideas", False, None)
    with pytest.raises(AdapterError) as raised:
        Fault(PROVIDER_FAULT_TIMEOUT, False, None)
    assert "Retryability is a property of the fault" in str(raised.value)


# -- what a client and a factory are trusted for ------------------------------


@pytest.mark.parametrize(
    "build,expected",
    [(build_openai_client, OPENAI_BASE_URL), (build_xai_client, XAI_BASE_URL)],
    ids=["openai", "xai"],
)
def test_the_shipped_client_factory_pins_the_endpoint_and_disables_sdk_retries(
    build: Any, expected: str
) -> None:
    """The one client this build constructs itself, and what it constructs it as.

    The endpoint is where the credential is sent, and the SDK's own retry loop
    would spend an episode's wall-clock on calls the runner never saw. Both are
    settled here rather than left to the SDK's defaults — which, for the xAI
    lane, are another vendor's endpoint entirely.
    """
    client = build(api_key=FAKE_KEY)
    assert common.normalized_endpoint(str(client.base_url)) == expected
    assert client.max_retries == 0


def _boundary_xai_client(
    transport: openai_transport.RecordingTransport,
    *,
    sdk_options: dict[str, Any] | None = None,
    http_options: dict[str, Any] | None = None,
) -> openai.OpenAI:
    options = dict(sdk_options or {})
    max_retries = options.pop("max_retries", 0)
    http_client = httpx.Client(
        transport=httpx.MockTransport(transport.handler),
        **(http_options or {}),
    )
    return openai.OpenAI(
        api_key=FAKE_KEY,
        base_url=XAI_BASE_URL,
        max_retries=max_retries,
        http_client=http_client,
        **options,
    )


def _mutate_boundary_xai_client(client: openai.OpenAI, mutation: str) -> None:
    secret = "unrecorded-client-state-secret"
    if mutation == "base_url":
        client.base_url = "https://redirect.invalid/v1"
    elif mutation == "organization":
        client.organization = secret
    elif mutation == "project":
        client.project = secret
    elif mutation == "sdk_headers":
        client._custom_headers = {"x-account-scope": secret}
    elif mutation == "sdk_query":
        client._custom_query = {"tenant": secret}
    elif mutation == "http_headers":
        client._client.headers["x-account-scope"] = secret
    elif mutation == "http_params":
        client._client.params = {"tenant": secret}
    elif mutation == "http_auth":
        client._client.auth = (secret, "not-a-credential")
    elif mutation == "http_cookies":
        client._client.cookies.set("session", secret)
    elif mutation == "request_hook":

        def redirect(request: httpx.Request) -> None:
            request.url = httpx.URL("https://redirect.invalid/v1/chat/completions")
            request.headers["x-hook-state"] = secret

        client._client.event_hooks["request"].append(redirect)
    elif mutation == "response_hook":

        def rewrite(response: httpx.Response) -> None:
            response._content = b"{}"

        client._client.event_hooks["response"].append(rewrite)
    elif mutation == "retries_false":
        client.max_retries = False
    elif mutation == "retries_float":
        client.max_retries = 0.0  # type: ignore[assignment]
    else:
        client.max_retries = 3


@pytest.mark.parametrize(
    ("sdk_options", "http_options"),
    [
        ({"default_headers": {"x-account-scope": "client-state-secret"}}, None),
        ({"default_query": {"tenant": "client-state-secret"}}, None),
        (None, {"headers": {"x-account-scope": "client-state-secret"}}),
        (None, {"params": {"tenant": "client-state-secret"}}),
        (None, {"auth": ("client-state-secret", "not-a-credential")}),
        (None, {"cookies": {"session": "client-state-secret"}}),
        (None, {"event_hooks": {"request": [lambda _request: None]}}),
        (None, {"event_hooks": {"response": [lambda _response: None]}}),
        ({"max_retries": False}, None),
        ({"max_retries": 0.0}, None),
        ({"max_retries": 3}, None),
    ],
)
def test_boundary_xai_refuses_unrecorded_client_state_at_construction(
    sdk_options: dict[str, Any] | None,
    http_options: dict[str, Any] | None,
) -> None:
    transport = openai_transport.RecordingTransport([_xai_body()])
    client = _boundary_xai_client(
        transport,
        sdk_options=sdk_options,
        http_options=http_options,
    )

    with pytest.raises(XAIConfigurationError) as raised:
        XAIOpenAICompatAdapter(model=XAI_MODEL, client=client)

    assert "client-state-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert transport.calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "base_url",
        "organization",
        "project",
        "sdk_headers",
        "sdk_query",
        "http_headers",
        "http_params",
        "http_auth",
        "http_cookies",
        "request_hook",
        "response_hook",
        "retries_false",
        "retries_float",
        "retries_nonzero",
    ],
)
def test_boundary_xai_refuses_post_construction_client_state_before_dispatch(
    mutation: str,
) -> None:
    transport = openai_transport.RecordingTransport([_xai_body()])
    client = _boundary_xai_client(transport)
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=client)
    _mutate_boundary_xai_client(client, mutation)

    with pytest.raises(XAIConfigurationError) as raised:
        adapter.next_call(_request(), _deadline())

    assert "unrecorded-client-state-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert transport.calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "base_url",
        "organization",
        "project",
        "sdk_headers",
        "sdk_query",
        "http_headers",
        "http_params",
        "http_auth",
        "http_cookies",
        "request_hook",
        "response_hook",
        "retries_false",
        "retries_float",
        "retries_nonzero",
    ],
)
def test_boundary_xai_final_client_state_refusal_cancels_without_dispatch(
    mutation: str,
) -> None:
    transport = openai_transport.RecordingTransport([_xai_body()])
    client = _boundary_xai_client(transport)
    inner = _guard("xai", XAI_MODEL)

    class MutatingGuard:
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            reservation = inner.authorize(
                input_tokens_upper_bound=input_tokens_upper_bound
            )
            _mutate_boundary_xai_client(client, mutation)
            return reservation

        def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
            inner.cancel(
                reservation,
                input_tokens_upper_bound=input_tokens_upper_bound,
            )

        def settle(
            self,
            reservation: Decimal,
            *,
            input_tokens: int | None,
            output_tokens: int | None,
        ) -> Decimal | None:
            return inner.settle(
                reservation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        def forfeit(self, reservation: Decimal) -> None:
            inner.forfeit(reservation)

    adapter = XAIOpenAICompatAdapter(
        model=XAI_MODEL,
        client=client,
        cost_guard=MutatingGuard(),  # type: ignore[arg-type]
    )

    with pytest.raises(XAIConfigurationError) as raised:
        adapter.next_call(_request(), _deadline())

    assert "unrecorded-client-state-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert transport.calls == 0
    assert adapter.last_telemetry().attempt_count == 0
    assert adapter.last_telemetry().terminal_reason == TURN_END_PRE_DISPATCH_REFUSED
    assert inner.exposure_usd == Decimal(0)
    assert inner.committed_usd == Decimal(0)
    inner.begin_episode()


@pytest.mark.parametrize("field", ["organization", "project"])
def test_an_xai_client_carrying_an_account_scope_is_refused(field: str) -> None:
    """The xAI lane has its own constructor, and its own copy of this refusal.

    An account scope does not move the request; it moves which account is billed
    for it and which policy applies to it. Adapter settings — hashed into run
    identity — record neither, so a run under one could not say who paid for it.
    """
    transport = openai_transport.RecordingTransport([_xai_body()])
    with pytest.raises(XAIConfigurationError) as raised:
        XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=transport.xai_client(**{field: "org_attacker"})
        )
    assert field in str(raised.value)
    assert "org_attacker" not in str(raised.value)
    assert transport.calls == 0


def test_a_mistral_client_this_build_cannot_read_a_server_url_from_is_refused() -> None:
    """No readable configuration means no proof of where the credential goes.

    The check refuses rather than assumes: an object that cannot state its server
    URL is not a client whose endpoint this run's settings can honestly record.
    """

    class NotAClient:
        sdk_configuration = None

    with pytest.raises(MistralConfigurationError) as raised:
        check_client_endpoint(NotAClient())  # type: ignore[arg-type]
    assert MISTRAL_BASE_URL in str(raised.value)


@pytest.mark.parametrize(
    "factory,model,variable,base_url,identity",
    [
        (
            build_openai_adapter,
            OPENAI_MODEL,
            "OPENAI_API_KEY",
            OPENAI_BASE_URL,
            openai_identity,
        ),
        (build_xai_adapter, XAI_MODEL, "XAI_API_KEY", XAI_BASE_URL, xai_identity),
        (
            build_mistral_adapter,
            MISTRAL_MODEL,
            "MISTRAL_API_KEY",
            MISTRAL_BASE_URL,
            mistral_identity,
        ),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_an_authorised_factory_builds_an_adapter_that_records_no_credential(
    factory: Any, model: str, variable: str, base_url: str, identity: Any
) -> None:
    """The whole configuration path, end to end, and nothing sent.

    The credential is read from the one variable it may come from and handed to
    the client; what the adapter *publishes* is the endpoint, the model and the
    request shape. Settings are hashed into run identity and written into a
    manifest, so a credential reaching them would be published — this asserts
    the value is nowhere in them.

    No request is made here. Building an adapter is not running one, which is the
    property that lets an operator preflight a configuration offline.
    """
    adapter = factory(
        model=model,
        configuration_id=AUDITED_ID,
        audited={AUDITED_ID},
        environ={variable: FAKE_KEY},
    )
    assert adapter.identity == identity(model)
    assert adapter.settings["base_url"] == base_url
    assert FAKE_KEY not in repr(adapter)
    assert FAKE_KEY not in str(dict(adapter.settings))
    assert adapter.last_telemetry().as_dict()["attempt_count"] == 0


@pytest.mark.parametrize(
    "factory,model,error",
    [
        (build_openai_adapter, OPENAI_MODEL, OpenAIConfigurationError),
        (build_xai_adapter, XAI_MODEL, XAIConfigurationError),
        (build_mistral_adapter, MISTRAL_MODEL, MistralConfigurationError),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_a_factory_refuses_an_unnamed_or_misshapen_configuration_identity(
    factory: Any, model: str, error: type[Exception]
) -> None:
    """Two different operator mistakes, refused as two different sentences.

    An invocation that named no configuration has nothing to check against the
    audited set. One that named something of another shape made a typo, and a
    typo must not be answered by sending a request — so it is refused as a typo
    rather than as an unauthorised run. Both happen before the credential is
    read: the environment here carries none.
    """
    with pytest.raises(error) as unnamed:
        factory(model=model, audited={AUDITED_ID}, environ={})
    assert "identity" in str(unnamed.value)

    with pytest.raises(error) as misshapen:
        factory(
            model=model,
            configuration_id="not-a-digest",
            audited={AUDITED_ID},
            environ={},
        )
    assert "hex digest" in str(misshapen.value)


@pytest.mark.parametrize(
    "factory,error",
    [
        (build_openai_adapter, OpenAIConfigurationError),
        (build_xai_adapter, XAIConfigurationError),
        (build_mistral_adapter, MistralConfigurationError),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_a_factory_refuses_a_model_identifier_that_is_not_one(
    factory: Any, error: type[Exception]
) -> None:
    """Refused first of all, because the model is what run identity names."""
    with pytest.raises(error) as raised:
        factory(model="not a model identifier", environ={})
    assert "model identifier" in str(raised.value)


@pytest.mark.parametrize(
    "adapter_type,model,client_of,error",
    [
        (
            OpenAIResponsesAdapter,
            OPENAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).client(),
            OpenAIConfigurationError,
        ),
        (
            XAIOpenAICompatAdapter,
            XAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).xai_client(),
            XAIConfigurationError,
        ),
        (
            MistralChatAdapter,
            MISTRAL_MODEL,
            lambda: mistral_transport.RecordingTransport([]).client(),
            MistralConfigurationError,
        ),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_a_retry_policy_of_another_type_is_not_a_retry_policy(
    adapter_type: Any, model: str, client_of: Any, error: type[Exception]
) -> None:
    """Settings hash a policy into run identity, so it has to be one.

    An object of another type cannot state how many times a turn may ask the
    provider, and a settings mapping built from one would record something that
    is not an attempt count as though it were.
    """
    with pytest.raises(error) as raised:
        adapter_type(model=model, client=client_of(), retry=object())
    assert "ProviderRetryPolicy" in str(raised.value)


@pytest.mark.parametrize(
    "adapter_type,model,client_of,profile,error",
    [
        (
            OpenAIResponsesAdapter,
            OPENAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).client(),
            OpenAIRequestProfile(
                profile_id="another_models_profile_v1", temperature=1.0, reasoning=None
            ),
            OpenAIConfigurationError,
        ),
        (
            XAIOpenAICompatAdapter,
            XAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).xai_client(),
            XAIRequestProfile(
                profile_id="another_models_profile_v1",
                temperature=1.0,
                reasoning_effort=None,
            ),
            XAIConfigurationError,
        ),
        (
            MistralChatAdapter,
            MISTRAL_MODEL,
            lambda: mistral_transport.RecordingTransport([]).client(),
            MistralRequestProfile(
                profile_id="another_models_profile_v1", temperature=1.0
            ),
            MistralConfigurationError,
        ),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_a_request_profile_from_another_model_cannot_shape_this_request(
    adapter_type: Any, model: str, client_of: Any, profile: Any, error: type[Exception]
) -> None:
    """A request shape is frozen per model, so it cannot be carried across models.

    One model refuses a sampling parameter another requires, and the profile is
    inside adapter settings and so inside run identity. A profile that belonged
    to a different model would send a body this run's own settings do not
    describe.
    """
    with pytest.raises(error) as raised:
        adapter_type(model=model, client=client_of(), profile=profile)
    assert "another_models_profile_v1" in str(raised.value)


@pytest.mark.parametrize(
    "adapter_type,model,client_of,profile_for,identity",
    [
        (
            OpenAIResponsesAdapter,
            OPENAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).client(),
            openai_profile_for,
            openai_identity,
        ),
        (
            XAIOpenAICompatAdapter,
            XAI_MODEL,
            lambda: openai_transport.RecordingTransport([]).xai_client(),
            xai_profile_for,
            xai_identity,
        ),
        (
            MistralChatAdapter,
            MISTRAL_MODEL,
            lambda: mistral_transport.RecordingTransport([]).client(),
            mistral_profile_for,
            mistral_identity,
        ),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_an_adapter_publishes_the_identity_profile_and_guard_it_was_built_with(
    adapter_type: Any, model: str, client_of: Any, profile_for: Any, identity: Any
) -> None:
    """What a row is attributed to, asserted against what was handed in.

    Every usage figure and every fault this build records is attributed to the
    adapter identity on the row that carries it, and the cost guard is the object
    a run's spend is enforced against. An adapter that reported either of them as
    something other than what it was constructed with would make those rows
    describe a different run.
    """
    guard = _guard("openai", OPENAI_MODEL)
    adapter = adapter_type(model=model, client=client_of(), cost_guard=guard)
    assert adapter.identity == identity(model)
    assert adapter.request_profile == profile_for(model)
    assert adapter.cost_guard is guard


# -- no network ----------------------------------------------------------------


def test_no_test_in_this_module_can_reach_a_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee the rest of this module rests on, made to fail loudly.

    Every client above is bound to an in-process transport, so nothing here
    should be able to open a socket even if it tried. This proves the attempt
    would be visible rather than quietly successful.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a test in this module tried to open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)

    transport, adapter = _openai_turn(_openai_body())
    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    assert transport.calls == 1
