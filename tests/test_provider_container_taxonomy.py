"""Malformed sub-objects of a provider response, under the real SDKs.

These SDKs parse a body leniently: where their own models declare an object,
a ``[]``, a ``"x"`` or a ``7`` on the wire is carried through as that value
rather than refused. Nothing downstream of the parse notices until an attribute
is read off it, and by then the failure is an ``AttributeError`` carrying the
interpreter's words — indistinguishable from this integration crashing, raised
from outside the fault taxonomy, and in the usage case raised before the attempt
that received the body was ever published.

So the property under test here is one property in three places: a container the
API documents but the body did not send is refused *by name*, before anything is
read out of it, with the attempt evidence intact and the money it authorised
still held as exposure. Every test runs the genuine SDK over an in-process
``httpx.MockTransport``; none of them contacts a provider.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from boundarybench.adapter import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.compiler import Variant
from boundarybench.pricing import price_for
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.providers.openai_responses import (
    MAX_OUTPUT_TOKENS,
    OpenAIResponsesAdapter,
)
from boundarybench.providers.xai_openai_compat import XAIOpenAICompatAdapter
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.conftest import minimal_card
from tests.openai_transport import (
    RecordingTransport,
    chat_body,
    function_call_item,
    responses_body,
    tool_call,
)

OPENAI_MODEL = "gpt-test-20990101"
XAI_MODEL = "grok-test-20990101"

#: The three shapes a lenient parse lets through where an object is documented:
#: an empty list, a bare string and a bare number. Each one reaches a different
#: attribute error, and none of them is a container this build can read.
MALFORMED_CONTAINERS: tuple[Any, ...] = ([], "x", 7)


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    from boundarybench.compiler import compile_cube
    from boundarybench.schema import ConstructCard

    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return cube.variants[0]


def _turn_request() -> TurnRequest:
    from boundarybench.environment import Environment

    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(_variant()), turns_remaining=12
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _guard() -> RunCostGuard:
    """A real cost guard, so a refusal's effect on the money can be asserted.

    Priced under a model this build holds a reviewed price for. The price policy
    is what makes a reservation an amount; which model the transport claims is
    irrelevant to the guard and is checked by the adapter instead.
    """
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("10.00"),
            max_episodes=36,
            price=price_for(provider="openai", model="gpt-5.6-luna"),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=ProviderRetryPolicy().max_attempts,
    )


def _openai_body(usage: Any = None) -> dict[str, Any]:
    body = responses_body([function_call_item("read_records", "{}")], model=OPENAI_MODEL)
    if usage is not None:
        body["usage"] = usage
    return body


def _xai_body(usage: Any = None, message: Any = None) -> dict[str, Any]:
    body = chat_body([tool_call("read_records", "{}")], model=XAI_MODEL)
    if usage is not None:
        body["usage"] = usage
    if message is not None:
        body["choices"][0]["message"] = message
    return body


def _assert_refused_and_unmeasured(adapter: Any, guard: RunCostGuard) -> None:
    """The one shape every refusal here has to have.

    Four claims, and they are separate on purpose:

    * the turn failed *inside* the taxonomy, as a response-invalid provider
      fault, rather than as a raw ``AttributeError`` from the SDK or the
      interpreter;
    * the attempt that received the body is still published — a response did
      arrive, and a turn that hid it would report a transport failure that did
      not happen;
    * the usage is *unmeasured*, never zero: a container this build could not
      read is not a report of no tokens;
    * the reservation stays as exposure at its conservative upper bound rather
      than settling at a cost nobody measured.
    """
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.response_received is True
    assert telemetry.usage_reported is False
    attempt = telemetry.attempts[0]
    assert attempt.cost_settlement == "forfeited"
    assert adapter.last_usage().input_tokens is None
    assert adapter.last_usage().output_tokens is None
    assert adapter.last_usage().cost_usd is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd == Decimal(attempt.cost_reservation_usd)
    assert guard.exposure_usd > Decimal(0)


# -- OpenAI: the usage block --------------------------------------------------


@pytest.mark.parametrize("usage", MALFORMED_CONTAINERS)
def test_openai_usage_that_is_not_a_usage_object_is_a_named_response_fault(
    usage: Any,
) -> None:
    """``usage: []`` is not a token report, and reading it is not a crash."""
    transport = RecordingTransport([_openai_body(usage=usage)])
    guard = _guard()
    adapter = OpenAIResponsesAdapter(
        model=OPENAI_MODEL, client=transport.client(), cost_guard=guard
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_refused_and_unmeasured(adapter, guard)


def test_openai_malformed_usage_never_reaches_the_operator_as_an_attribute_error() -> (
    None
):
    """No exception from outside this contract escapes the turn."""
    transport = RecordingTransport([_openai_body(usage="x")])
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert not isinstance(caught.value, AttributeError)
    assert "input_tokens" not in str(caught.value)


def test_openai_a_later_turn_reports_its_own_evidence_and_not_the_refused_one() -> None:
    """Attempt evidence that outlived its turn would misattribute a fault."""
    transport = RecordingTransport([_openai_body(usage=[]), _openai_body()])
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_telemetry().usage_reported is False

    adapter.next_call(_turn_request(), _deadline())

    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.usage_reported is True
    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_usage().output_tokens == 29


# -- xAI: the usage block -----------------------------------------------------


@pytest.mark.parametrize("usage", MALFORMED_CONTAINERS)
def test_xai_usage_that_is_not_a_usage_object_is_a_named_response_fault(
    usage: Any,
) -> None:
    transport = RecordingTransport([_xai_body(usage=usage)])
    guard = _guard()
    adapter = XAIOpenAICompatAdapter(
        model=XAI_MODEL, client=transport.xai_client(), cost_guard=guard
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_refused_and_unmeasured(adapter, guard)


def test_xai_malformed_usage_never_reaches_the_operator_as_an_attribute_error() -> None:
    transport = RecordingTransport([_xai_body(usage=7)])
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert not isinstance(caught.value, AttributeError)
    assert "prompt_tokens" not in str(caught.value)


# -- xAI: the choice's message ------------------------------------------------


@pytest.mark.parametrize("message", MALFORMED_CONTAINERS)
def test_xai_a_choice_whose_message_is_not_a_message_is_a_named_response_fault(
    message: Any,
) -> None:
    """A scalar where the API documents a message states no action at all.

    Refused as a response fault rather than as a protocol failure, and refused
    before the body is measured: nothing about the *model's* behaviour is
    established by a container the transport could not have produced from a
    documented body, so filing it under the model would put a body-shape fault
    in the bucket that means "the model answered badly".
    """
    transport = RecordingTransport([_xai_body(message=message)])
    guard = _guard()
    adapter = XAIOpenAICompatAdapter(
        model=XAI_MODEL, client=transport.xai_client(), cost_guard=guard
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_refused_and_unmeasured(adapter, guard)


def test_xai_a_malformed_message_never_quotes_what_arrived_in_that_field() -> None:
    """A durable failure detail carries this build's own words and nothing else."""
    transport = RecordingTransport([_xai_body(message="secret-provider-text")])
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    detail = str(caught.value)
    assert "secret-provider-text" not in detail
    assert "tool_calls" not in detail


def test_xai_a_choice_that_is_not_a_choice_is_refused_before_it_is_read() -> None:
    """The list arrived; what is in it still has to be what this API documents."""
    body = _xai_body()
    body["choices"][0] = 7
    transport = RecordingTransport([body])
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert adapter.last_telemetry().response_received is True


def test_xai_a_tool_call_entry_that_is_not_a_tool_call_is_refused() -> None:
    """A scalar in ``tool_calls`` is a malformed body, not an unreadable union arm.

    Refused on the wire, and so *before* the turn's measurement rather than
    after it. That is the same rule every other refusal in this file follows: a
    body that does not state the contract this build asked for is not a body
    whose token counts this build banks. The reservation stays as exposure
    instead — which over-states what the run may have spent, in the one
    direction a cost cap can survive.
    """
    transport = RecordingTransport([chat_body([7], model=XAI_MODEL)])
    guard = _guard()
    adapter = XAIOpenAICompatAdapter(
        model=XAI_MODEL, client=transport.xai_client(), cost_guard=guard
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert adapter.last_telemetry().response_received is True
    assert adapter.last_usage().input_tokens is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > Decimal(0)


def test_xai_a_later_turn_reports_its_own_evidence_and_not_the_refused_one() -> None:
    transport = RecordingTransport([_xai_body(message=[]), _xai_body()])
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens is None

    adapter.next_call(_turn_request(), _deadline())

    assert adapter.last_telemetry().attempt_count == 1
    assert adapter.last_usage().input_tokens == 137
