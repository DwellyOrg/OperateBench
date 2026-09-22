"""The Anthropic adapter under a cost guard.

Every test here drives the real ``anthropic`` SDK over the in-process
``MockTransport`` from :mod:`tests.anthropic_transport`, so the request is
serialised by the SDK, the response is parsed into a real ``Message`` and the
usage the adapter measures is the usage the SDK read off the wire. No socket is
opened and no provider is contacted.

The tests bind three facts into one invariant: provider-reported usage, cost under
the run's reviewed price, and authorisation before dispatch.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from boundarybench.adapter import (
    AdapterProtocolError,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import (
    CostCapExceededError,
    CostControls,
    RunCostGuard,
    conservative_input_token_bound,
)
from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.pricing import price_for
from boundarybench.providers.anthropic_messages import (
    MAX_OUTPUT_TOKENS,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
    build_messages_request,
    request_input_token_bound,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from tests.anthropic_transport import (
    RecordingTransport,
    error_body,
    message_body,
    scripted_client,
    text_block,
    tool_use_block,
)
from tests.conftest import minimal_card

SONNET = "claude-sonnet-5"


def _request() -> TurnRequest:
    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(cube.variants[0]),
        turns_remaining=12,
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _guard(cap: str = "100") -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal(cap),
            max_episodes=None,
            price=price_for(provider="anthropic", model=SONNET),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )


def _adapter(
    client: object, guard: RunCostGuard | None, **kwargs: object
) -> AnthropicMessagesAdapter:
    return AnthropicMessagesAdapter(
        model=SONNET,
        client=client,  # type: ignore[arg-type]
        cost_guard=guard,
        sleep=lambda seconds: None,
        **kwargs,  # type: ignore[arg-type]
    )


STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


# -- the request-size bound --------------------------------------------------


def test_the_input_bound_is_derived_from_the_body_the_sdk_will_send() -> None:
    """Bounded by the body's own bytes, plus the stated framing allowance."""
    from boundarybench.jsonsafe import canonical_json_text

    payload = build_messages_request(_request(), model=SONNET)
    encoded = canonical_json_text(payload, "request").encode("utf-8")

    assert request_input_token_bound(payload) == conservative_input_token_bound(
        len(encoded)
    )
    # A token encodes at least one byte, so the bound is never below the body.
    assert request_input_token_bound(payload) > len(encoded)


# -- measured cost -----------------------------------------------------------


def test_a_measured_turn_prices_the_usage_the_provider_reported() -> None:
    transport, client = scripted_client(
        message_body([STOP_CALL], model=SONNET, input_tokens=137, output_tokens=29)
    )
    guard = _guard()
    adapter = _adapter(client, guard)

    adapter.next_call(_request(), _deadline())

    usage = adapter.last_usage()
    assert usage.input_tokens == 137
    assert usage.output_tokens == 29
    # 137 * 2/1M + 29 * 10/1M
    assert usage.cost_usd == pytest.approx(0.000564)
    assert guard.measured_usd == Decimal("0.000564")
    assert guard.exposure_usd == Decimal(0)
    assert transport.calls == 1


def test_without_a_guard_the_cost_stays_null_rather_than_being_invented() -> None:
    """No pinned price, no cost. A number here would be a fabricated measurement."""
    _transport, client = scripted_client(
        message_body([STOP_CALL], model=SONNET, input_tokens=137, output_tokens=29)
    )
    adapter = _adapter(client, None)

    adapter.next_call(_request(), _deadline())

    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_usage().cost_usd is None


def test_a_protocol_invalid_answer_that_reported_usage_still_records_its_cost() -> None:
    """A response that arrived was paid for even when this build cannot read it."""
    _transport, client = scripted_client(
        message_body(
            [text_block("no tool call")], model=SONNET, input_tokens=91, output_tokens=92
        )
    )
    guard = _guard()
    adapter = _adapter(client, guard)

    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())

    usage = adapter.last_usage()
    assert usage.input_tokens == 91
    assert usage.output_tokens == 92
    assert usage.cost_usd == pytest.approx(float(Decimal("0.001102")))
    assert guard.measured_usd == Decimal("0.001102")
    assert guard.exposure_usd == Decimal(0)


def test_an_answer_from_the_wrong_model_is_exposure_and_never_measured_cost() -> None:
    """Its tokens are not this model's spend, but the request was still made."""
    _transport, client = scripted_client(
        message_body(
            [STOP_CALL], model="claude-other-5", input_tokens=137, output_tokens=29
        )
    )
    guard = _guard()
    adapter = _adapter(client, guard)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_request(), _deadline())

    assert adapter.last_usage().cost_usd is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > 0


# -- exposure ----------------------------------------------------------------


def test_a_transport_fault_keeps_its_conservative_bound_as_exposure() -> None:
    _transport, client = scripted_client((401, error_body()))
    guard = _guard()
    adapter = _adapter(client, guard)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_request(), _deadline())

    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > 0
    assert guard.remaining_usd < Decimal("100")


def test_every_retried_attempt_is_separately_authorised_and_separately_exposed() -> None:
    """Retries are not free, and the guard sees each one before it is made."""
    transport, client = scripted_client(
        (500, error_body()), (500, error_body()), (500, error_body())
    )
    guard = _guard()
    adapter = _adapter(client, guard)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_request(), _deadline())

    assert transport.calls == 3
    single = guard.reservation_for(
        input_tokens_upper_bound=request_input_token_bound(
            build_messages_request(_request(), model=SONNET)
        )
    )
    assert guard.exposure_usd == single * 3


# -- fail closed -------------------------------------------------------------


def test_an_unaffordable_request_is_never_dispatched() -> None:
    """The proof is the transport: a refused request is a request not made."""
    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    guard = _guard("0.000001")
    adapter = _adapter(transport.client(), guard)

    with pytest.raises(CostCapExceededError):
        adapter.next_call(_request(), _deadline())

    assert transport.calls == 0


def test_the_refusal_names_no_provider_text_and_no_credential() -> None:
    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    guard = _guard("0.000001")
    adapter = _adapter(transport.client(), guard)

    with pytest.raises(CostCapExceededError) as refused:
        adapter.next_call(_request(), _deadline())

    message = str(refused.value)
    assert "USD" in message
    assert "test-key-not-a-credential" not in message
    assert "provider.invalid" not in message


def test_a_run_that_exhausts_its_budget_mid_turn_stops_before_the_next_retry() -> None:
    """The second attempt is refused because the first one's exposure ate the cap.

    The turn therefore ends on the budget refusal rather than on the provider
    fault it would otherwise have retried past: a run that has spent its
    authorisation stops, and it stops before the request rather than after it.
    """
    transport, client = scripted_client((500, error_body()), (500, error_body()))
    guard = _guard()
    adapter = _adapter(client, guard)
    single_bound = guard.reservation_for(
        input_tokens_upper_bound=request_input_token_bound(
            build_messages_request(_request(), model=SONNET)
        )
    )
    # A cap that pays for exactly one attempt.
    guard.seed(measured_usd=Decimal("100") - single_bound, exposure_usd=Decimal(0))

    with pytest.raises(CostCapExceededError):
        adapter.next_call(_request(), _deadline())

    assert transport.calls == 1
    assert guard.remaining_usd == Decimal(0)
    # The attempt that *was* made is still recorded, so the turn's evidence
    # survives the refusal that ended it.
    assert adapter.last_telemetry().attempt_count == 1


def test_the_guard_is_cleared_between_turns_so_no_reservation_leaks() -> None:
    transport = RecordingTransport(
        [],
        default=message_body(
            [STOP_CALL], model=SONNET, input_tokens=10, output_tokens=10
        ),
    )
    guard = _guard()
    adapter = _adapter(transport.client(), guard)

    adapter.next_call(_request(), _deadline())
    adapter.next_call(_request(), _deadline())

    assert transport.calls == 2
    assert guard.measured_usd == price_for(provider="anthropic", model=SONNET).cost(
        input_tokens=20, output_tokens=20
    )
