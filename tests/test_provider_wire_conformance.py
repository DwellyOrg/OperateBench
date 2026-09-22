"""What arrived on the wire, checked before a typed SDK model is believed.

Every adapter in this build reads two things out of a provider response: one
action, and two token counts. Both are read off *typed* SDK models, and a typed
model is not a transcript of the wire. These SDKs coerce (``true`` and ``"12"``
become ``1`` and ``12``), they drop (a field the vendor's schema does not
declare disappears before anything here can see it), and they resolve unions
leniently (a ``function`` that is a list survives the parse and fails four
frames later as an ``AttributeError`` the taxonomy has no name for).

So the property under test here is one property in three integrations: the
**exact JSON the provider sent** is validated — field presence, field types,
nested shape and undeclared extras — before the typed model is trusted for
anything, and every way it can fail is a named fault with the attempt evidence
intact and the money it authorised still held as exposure.

Every test runs the genuine SDK over an in-process ``httpx.MockTransport``.
None of them contacts a provider.
"""

from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import httpx
import pytest
from mistralai.client import Mistral

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
from boundarybench.providers.mistral_chat import (
    MISTRAL_ADAPTER_VERSION,
    MISTRAL_RESPONSE_CAPTURE,
    CapturedHttpClient,
    MistralChatAdapter,
    ensure_capturing_client,
    mistral_settings,
)
from boundarybench.providers.openai_responses import (
    MAX_OUTPUT_TOKENS,
    OPENAI_ADAPTER_VERSION,
    OPENAI_RESPONSE_CAPTURE,
    OpenAIResponsesAdapter,
    openai_settings,
)
from boundarybench.providers.wire import MAX_WIRE_DEPTH
from boundarybench.providers.xai_openai_compat import (
    XAI_ADAPTER_VERSION,
    XAI_RESPONSE_CAPTURE,
    XAIOpenAICompatAdapter,
    xai_settings,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.conftest import minimal_card
from tests.mistral_transport import RecordingTransport as MistralTransport
from tests.mistral_transport import chat_body as mistral_chat_body
from tests.mistral_transport import tool_call as mistral_tool_call
from tests.openai_transport import (
    RecordingTransport,
    chat_body,
    function_call_item,
    responses_body,
    tool_call,
)

OPENAI_MODEL = "gpt-test-20990101"
XAI_MODEL = "grok-test-20990101"
MISTRAL_MODEL = "mistral-test-20990101"

#: The four values a token count may never be, however the SDK renders them.
#: ``True`` is an ``int`` to Python and ``1`` to a lenient parser; ``"12"`` is
#: coerced to twelve; a float and a null are not counts at all. None of them is
#: a measurement anybody took, and all four reach a cost cap if believed.
UNCOUNTABLE_TOKEN_VALUES: tuple[Any, ...] = (True, False, "12", 12.0, None)


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
    """A real cost guard, so a refusal's effect on the money can be asserted."""
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("10.00"),
            max_episodes=36,
            price=price_for(provider="openai", model="gpt-5.6-luna"),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=ProviderRetryPolicy().max_attempts,
    )


# -- bodies -------------------------------------------------------------------


def _openai_body(**overrides: Any) -> dict[str, Any]:
    body = responses_body([function_call_item("read_records", "{}")], model=OPENAI_MODEL)
    body.update(copy.deepcopy(overrides))
    return body


def _xai_body(**overrides: Any) -> dict[str, Any]:
    body = chat_body([tool_call("read_records", "{}")], model=XAI_MODEL)
    body.update(copy.deepcopy(overrides))
    return body


def _mistral_body(**overrides: Any) -> dict[str, Any]:
    body = mistral_chat_body(
        [mistral_tool_call("read_records", "{}")], model=MISTRAL_MODEL
    )
    body.update(copy.deepcopy(overrides))
    return body


# -- the three adapters, over their real SDKs ---------------------------------


def _openai(*steps: Any) -> tuple[OpenAIResponsesAdapter, RunCostGuard]:
    transport = RecordingTransport(list(steps))
    guard = _guard()
    return (
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=transport.client(), cost_guard=guard
        ),
        guard,
    )


def _xai(*steps: Any) -> tuple[XAIOpenAICompatAdapter, RunCostGuard]:
    transport = RecordingTransport(list(steps))
    guard = _guard()
    return (
        XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=transport.xai_client(), cost_guard=guard
        ),
        guard,
    )


def _mistral(*steps: Any) -> tuple[MistralChatAdapter, RunCostGuard]:
    transport = MistralTransport(list(steps))
    guard = _guard()
    return (
        MistralChatAdapter(
            model=MISTRAL_MODEL, client=transport.client(), cost_guard=guard
        ),
        guard,
    )


def _refused(adapter: Any, guard: RunCostGuard) -> AdapterProviderError:
    """Run one turn, demand a named response-invalid refusal, and check the money.

    Five claims, separate on purpose:

    * the turn failed *inside* the taxonomy rather than as an ``AttributeError``
      or a ``TypeError`` carrying the interpreter's words;
    * the attempt is published, and it says a response really did arrive;
    * no usage was measured — an unreadable count is not a report of no tokens;
    * the reservation stayed as exposure at its conservative upper bound;
    * nothing the provider wrote is in the durable detail.
    """
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    error = caught.value
    assert error.fault == PROVIDER_FAULT_RESPONSE_INVALID
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.response_received is True
    assert telemetry.usage_reported is False
    assert telemetry.attempts[0].cost_settlement == "forfeited"
    assert adapter.last_usage().input_tokens is None
    assert adapter.last_usage().output_tokens is None
    assert adapter.last_usage().cost_usd is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > Decimal(0)
    return error


# -- 1. the raw-response mode is part of what a run recorded -------------------


def test_every_adapter_records_how_it_captured_the_wire_response() -> None:
    """A run that validated the wire says so in the settings it hashes.

    The check is not free: the OpenAI SDK's raw-response surface adds a header
    to the request, and the Mistral client is wrapped so the ``httpx.Response``
    can be read before the SDK parses it. Both are result-affecting, so both are
    named in adapter settings — which are hashed into ``configuration_id`` — and
    the adapter versions move with them.
    """
    assert openai_settings(model="gpt-5.6-luna")["response_capture"] == (
        OPENAI_RESPONSE_CAPTURE
    )
    assert xai_settings(model="grok-4.5")["response_capture"] == XAI_RESPONSE_CAPTURE
    assert mistral_settings(model="mistral-small-2603")["response_capture"] == (
        MISTRAL_RESPONSE_CAPTURE
    )
    # Two SDKs, two mechanisms. The OpenAI and xAI lanes share a mode because
    # they share a client; the Mistral one cannot, and a reader must not have to
    # infer which a run used.
    modes = {OPENAI_RESPONSE_CAPTURE, XAI_RESPONSE_CAPTURE, MISTRAL_RESPONSE_CAPTURE}
    assert len(modes) == 2
    # ``0.2.0`` is the version each of the three reached by validating the wire.
    # All three have since moved past it for unrelated reasons — the OpenAI one
    # three times, as what it *sends* to its pinned model changed, then what it
    # *accepts* did, then that reading was narrowed to the exact observed shape;
    # the xAI one once, when it began accepting its compatibility endpoint's own
    # response extensions; the Mistral one once, when it began accepting this
    # service's own ``prompt_tokens_details`` object — so the assertion is per
    # lane rather than one number for all three. Reading the capture mode off a
    # shared version would stop being a check the moment any one lane moved.
    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert MISTRAL_ADAPTER_VERSION == "0.3.0"


def test_the_happy_path_still_parses_an_action_and_measures_the_wire_counts() -> None:
    """Validating the wire does not stop a well-formed answer being read."""
    for adapter in (
        _openai(_openai_body())[0],
        _xai(_xai_body())[0],
        _mistral(_mistral_body())[0],
    ):
        call = adapter.next_call(_turn_request(), _deadline())
        assert call.action == "read_records"
        usage = adapter.last_usage()
        assert usage.input_tokens == 137
        assert usage.output_tokens == 29
        assert adapter.last_telemetry().usage_reported is True


def test_the_request_body_is_unchanged_by_capturing_the_raw_response() -> None:
    """The wire *request* is the one the settings describe, header aside.

    Capturing the response must not change what is asked of the provider: a
    body that moved would make every recorded run incomparable with the ones
    before it, and the settings that describe it false.
    """
    from boundarybench.providers.openai_responses import build_responses_request
    from boundarybench.providers.xai_openai_compat import build_chat_request

    transport = RecordingTransport([_openai_body()])
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())
    adapter.next_call(_turn_request(), _deadline())
    assert transport.bodies[0] == build_responses_request(
        _turn_request(), model=OPENAI_MODEL
    )

    transport = RecordingTransport([_xai_body()])
    xai = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())
    xai.next_call(_turn_request(), _deadline())
    assert transport.bodies[0] == build_chat_request(_turn_request(), model=XAI_MODEL)


# -- 2. usage must be on the wire, and must be exact integers -----------------


@pytest.mark.parametrize("value", UNCOUNTABLE_TOKEN_VALUES)
def test_openai_refuses_a_coerced_or_absent_input_count(value: Any) -> None:
    """``true``/``"12"``/``12.0``/``null`` are not counts, whatever the SDK makes
    of them."""
    body = _openai_body()
    body["usage"] = {**body["usage"], "input_tokens": value}
    adapter, guard = _openai(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("value", UNCOUNTABLE_TOKEN_VALUES)
def test_xai_refuses_a_coerced_or_absent_prompt_count(value: Any) -> None:
    body = _xai_body()
    body["usage"] = {**body["usage"], "prompt_tokens": value}
    adapter, guard = _xai(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("value", UNCOUNTABLE_TOKEN_VALUES)
def test_mistral_refuses_a_coerced_or_absent_prompt_count(value: Any) -> None:
    body = _mistral_body()
    body["usage"] = {**body["usage"], "prompt_tokens": value}
    adapter, guard = _mistral(body)
    _refused(adapter, guard)


def test_a_boolean_count_is_never_priced_as_one_token() -> None:
    """The specific coercion this build must not inherit.

    ``prompt_tokens: true`` reaches the SDK's model as ``1``. A build that
    believed it would price a turn at one token and settle a reservation
    against a measurement nobody took.
    """
    body = _xai_body()
    body["usage"] = {"prompt_tokens": True, "completion_tokens": 29, "total_tokens": 30}
    adapter, guard = _xai(body)
    _refused(adapter, guard)
    assert adapter.last_usage().input_tokens is None


@pytest.mark.parametrize("usage", [{}, None])
def test_an_empty_or_absent_usage_block_is_refused_by_every_adapter(
    usage: Any,
) -> None:
    """A usage object with no counts is not zero tokens, and is not measured.

    ``{}`` is the shape that matters most: on the Responses surface it makes a
    turn *unmeasured*, and on Mistral's it parses into a usage model whose
    counts default to zero — a measured cost of nothing for a request that
    really consumed tokens. Both readings are wrong, and the wire states neither.
    """
    for factory, body in (
        (_openai, _openai_body()),
        (_xai, _xai_body()),
        (_mistral, _mistral_body()),
    ):
        if usage is None:
            body.pop("usage")
        else:
            body["usage"] = usage
        adapter, guard = factory(body)
        _refused(adapter, guard)


def test_a_negative_wire_count_is_refused_rather_than_crediting_the_run() -> None:
    """A negative count would *credit* a run against its own cap."""
    body = _xai_body()
    body["usage"] = {"prompt_tokens": -1, "completion_tokens": 29, "total_tokens": 28}
    adapter, guard = _xai(body)
    _refused(adapter, guard)


# -- 3. undeclared fields on the wire fail closed -----------------------------


def test_mistral_refuses_an_undeclared_field_its_sdk_silently_drops() -> None:
    """The failure a typed model cannot see at all.

    This SDK discards a field its own schema does not declare *before* anything
    in this build can inspect the response, so the extras check that works on
    the other two SDKs sees nothing. Reading the wire is the only way this can
    be refused, and refused it must be: an undeclared field is either a surface
    that moved under a pinned SDK or content that did not come from the model.
    """
    adapter, guard = _mistral(_mistral_body(surprise="whatever this is"))
    _refused(adapter, guard)


@pytest.mark.parametrize(
    "path",
    [
        ("choices", 0),
        ("choices", 0, "message"),
        ("choices", 0, "message", "tool_calls", 0),
        ("choices", 0, "message", "tool_calls", 0, "function"),
        ("usage",),
    ],
)
def test_mistral_refuses_an_undeclared_field_at_every_nested_level(
    path: tuple[Any, ...],
) -> None:
    """Top level, choice, message, tool call, function object and usage.

    A check that looked only at the top level would pass a body whose *action*
    was carried by a tool call nobody can describe.
    """
    body = _mistral_body()
    node: Any = body
    for step in path:
        node = node[step]
    node["undeclared_field"] = 1
    adapter, guard = _mistral(body)
    _refused(adapter, guard)


def test_an_undeclared_field_inside_the_token_details_is_refused() -> None:
    """The detail objects are part of the block this turn is priced on.

    This build reads no cached- or reasoning-token counter — it enables no
    caching feature — but a field nobody can describe *inside* the usage object
    is still a usage object that is not the one this API documents.
    """
    body = _openai_body()
    body["usage"] = {**body["usage"], "input_tokens_details": {"surprise": 1}}
    adapter, guard = _openai(body)
    _refused(adapter, guard)

    body = _xai_body()
    body["usage"] = {**body["usage"], "completion_tokens_details": {"surprise": 1}}
    adapter, guard = _xai(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("arguments", [7, [], None, True])
def test_mistral_refuses_tool_call_arguments_of_an_undocumented_shape(
    arguments: Any,
) -> None:
    """This SDK documents a JSON string *or* an object there, and nothing else."""
    body = _mistral_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    adapter, guard = _mistral(body)
    _refused(adapter, guard)


def test_mistral_accepts_the_decoded_object_form_of_tool_call_arguments() -> None:
    """The other documented shape still works: this SDK types the field as either."""
    body = _mistral_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {}
    adapter, _guard = _mistral(body)
    assert adapter.next_call(_turn_request(), _deadline()).action == "read_records"


@pytest.mark.parametrize(
    "path",
    [
        ("choices", 0),
        ("choices", 0, "message"),
        ("choices", 0, "message", "tool_calls", 0),
        ("choices", 0, "message", "tool_calls", 0, "function"),
        ("usage",),
    ],
)
def test_xai_refuses_an_undeclared_field_at_every_nested_level(
    path: tuple[Any, ...],
) -> None:
    body = _xai_body()
    node: Any = body
    for step in path:
        node = node[step]
    node["undeclared_field"] = 1
    adapter, guard = _xai(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("path", [("output", 0), ("usage",)])
def test_openai_refuses_an_undeclared_field_at_every_nested_level(
    path: tuple[Any, ...],
) -> None:
    body = _openai_body()
    node: Any = body
    for step in path:
        node = node[step]
    node["undeclared_field"] = 1
    adapter, guard = _openai(body)
    _refused(adapter, guard)


def test_an_undeclared_top_level_field_is_refused_by_every_adapter() -> None:
    for factory, body in (
        (_openai, _openai_body(undeclared_field=1)),
        (_xai, _xai_body(undeclared_field=1)),
        (_mistral, _mistral_body(undeclared_field=1)),
    ):
        adapter, guard = factory(body)
        _refused(adapter, guard)


# -- 4. a malformed tool call is a named fault, never an AttributeError --------


#: The four values that reach ``.name`` and raise a raw ``AttributeError`` under
#: a lenient parse. A list, a string, an integer and a boolean each survive the
#: SDK's union resolution and none of them names an action.
MALFORMED_FUNCTIONS: tuple[Any, ...] = ([], "read_records", 7, True)


@pytest.mark.parametrize("function", MALFORMED_FUNCTIONS)
def test_xai_refuses_a_malformed_function_object_by_name(function: Any) -> None:
    """The blocker: ``function`` as a list reaches ``.name`` and crashes.

    Worse than the crash is what the evidence said: the attempt was published as
    a *successful response with usage*, because the identity check passed and
    the counts were banked before the parse reached the malformed field. The
    turn then died outside the taxonomy.
    """
    body = _xai_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"] = function
    adapter, guard = _xai(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("function", MALFORMED_FUNCTIONS)
def test_mistral_refuses_a_malformed_function_object_by_name(function: Any) -> None:
    body = _mistral_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"] = function
    adapter, guard = _mistral(body)
    _refused(adapter, guard)


@pytest.mark.parametrize("name", [7, None, [], {}])
def test_a_tool_call_name_that_is_not_a_string_is_refused(name: Any) -> None:
    """A name is text. Anything else names no action the scaffold declares."""
    body = _xai_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = name
    adapter, guard = _xai(body)
    _refused(adapter, guard)


# -- 5. a body the SDK cannot parse still arrived -----------------------------


def test_mistral_records_that_a_malformed_http_200_body_arrived() -> None:
    """``ResponseValidationError`` carries a response. The telemetry must say so.

    The SDK raises before returning, so the executor saw an exception rather
    than a body — and recorded ``response_received=False``, which is a claim
    that nothing came back. Something did: HTTP 200, with a body this build
    could not parse. The exception carries it as ``raw_response``, so the honest
    record is an attempt that received a response, measured no usage, and kept
    its reservation as exposure.
    """
    adapter, guard = _mistral(
        {
            "id": "cmpl_test",
            "object": "chat.completion",
            "model": MISTRAL_MODEL,
            "choices": "not-a-list",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.attempts[0].response_received is True
    assert telemetry.usage_reported is False
    assert adapter.last_usage().input_tokens is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > Decimal(0)


def test_openai_records_that_an_unparseable_body_arrived() -> None:
    """The same claim on the other SDK: a 200 that will not parse still arrived."""
    adapter, guard = _openai(b'{"id": "resp_test", "object": ')
    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_telemetry().attempts[0].response_received is True
    assert guard.exposure_usd > Decimal(0)


# -- 6. the state a bad response leaves behind --------------------------------


def test_a_later_valid_call_clears_what_a_refused_one_left() -> None:
    """A refusal is this turn's evidence, not the next turn's."""
    for factory, bad, good in (
        (_openai, _openai_body(usage={}), _openai_body()),
        (_xai, _xai_body(usage={}), _xai_body()),
        (_mistral, _mistral_body(usage={}), _mistral_body()),
    ):
        adapter, guard = factory(bad, good)
        with pytest.raises(AdapterProviderError):
            adapter.next_call(_turn_request(), _deadline())
        held = guard.exposure_usd
        call = adapter.next_call(_turn_request(), _deadline())
        assert call.action == "read_records"
        assert adapter.last_usage().input_tokens == 137
        assert adapter.last_telemetry().attempt_count == 1
        assert adapter.last_telemetry().usage_reported is True
        # The forfeited exposure is a fact about the run and does not vanish;
        # what clears is the turn's own measurement and attempt record.
        assert guard.exposure_usd == held
        assert guard.measured_usd > Decimal(0)


def test_no_provider_text_reaches_a_refusals_durable_detail() -> None:
    """A refusal quotes this build's own words and closed-set values only."""
    secret = "PROVIDER-CONTROLLED-MARKER"
    body = _xai_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"] = secret
    body[secret] = secret
    adapter, guard = _xai(body)
    error = _refused(adapter, guard)
    assert secret not in str(error)


# -- 7. the Mistral capture wrapper -------------------------------------------


def test_an_injected_mistral_client_is_wrapped_once_and_only_once() -> None:
    """Wrapping is idempotent: two adapters on one client is one wrapper.

    A wrapper around a wrapper would hold two copies of a provider body and make
    "the response for this call" ambiguous — which is the one thing the capture
    exists to answer.
    """
    transport = MistralTransport([_mistral_body(), _mistral_body()])
    client: Mistral = transport.client()
    first = ensure_capturing_client(client)
    second = ensure_capturing_client(client)
    assert first is second
    assert isinstance(client.sdk_configuration.client, CapturedHttpClient)
    MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    assert client.sdk_configuration.client is first


def test_the_capture_is_cleared_before_each_call_and_holds_one_response() -> None:
    """Only the current call's response is consumed, and it is not kept after.

    A capture that survived its turn would let one turn's body be read as the
    next turn's — the same staleness the executor clears its measurement to
    avoid, one layer lower.
    """
    transport = MistralTransport([_mistral_body(), _mistral_body()])
    client = transport.client()
    adapter = MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    capture = client.sdk_configuration.client
    assert isinstance(capture, CapturedHttpClient)
    adapter.next_call(_turn_request(), _deadline())
    adapter.next_call(_turn_request(), _deadline())
    assert transport.calls == 2
    # Nothing is held between turns: the capture is cleared before the request
    # and again as soon as its one response has been read.
    assert capture.captured_count == 0


def test_a_capture_wrapper_delegates_build_request_and_close() -> None:
    """The wrapper is an ``HttpClient``, not a partial one.

    The SDK builds every request through this object and closes it on teardown.
    A wrapper that implemented only ``send`` would work until the first time the
    SDK did either.
    """
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    inner = httpx.Client(transport=transport)
    wrapper = CapturedHttpClient(inner)
    request = wrapper.build_request("POST", "https://api.mistral.ai/v1/chat/completions")
    assert isinstance(request, httpx.Request)
    wrapper.close()
    assert inner.is_closed


def test_a_response_that_names_a_key_twice_states_two_costs_and_is_refused() -> None:
    """``json.loads`` keeps the last value and says nothing. This build will not.

    A usage block that names ``prompt_tokens`` twice is one body that two
    readers price two ways, and whichever the decoder happened to keep would
    become the run's measurement. Sent as raw bytes because no Python value can
    be encoded into a duplicate key.
    """
    adapter, guard = _xai(
        b'{"id":"c","object":"chat.completion","created":1,"model":"'
        + XAI_MODEL.encode()
        + b'","choices":[],"usage":{"prompt_tokens":1,"prompt_tokens":900000,'
        b'"completion_tokens":2,"total_tokens":3}}'
    )
    _refused(adapter, guard)


@pytest.mark.parametrize("body", [b"[]", b'"a string"', b"7", b"null"])
def test_a_body_that_is_not_a_json_object_is_refused(body: bytes) -> None:
    """The response contract is an object. A list or a scalar states none of it."""
    adapter, guard = _xai(body)
    _refused(adapter, guard)


def test_a_body_nested_past_the_stated_limit_fails_as_a_limit_not_a_crash() -> None:
    """A body built to recurse forever is refused by a number this build chose."""
    depth = MAX_WIRE_DEPTH + 5
    nested = b"[" * depth + b"1" + b"]" * depth
    adapter, guard = _xai(
        b'{"id":"c","object":"chat.completion","created":1,"model":"'
        + XAI_MODEL.encode()
        + b'","choices":'
        + nested
        + b',"usage":{"prompt_tokens":1,'
        b'"completion_tokens":2,"total_tokens":3}}'
    )
    error = _refused(adapter, guard)
    assert "RecursionError" not in str(error)


def test_a_typed_reading_the_sdk_cannot_produce_is_a_named_fault() -> None:
    """A parse that fails is a response fault, never an adapter crash.

    The bytes arrived; what could not be done with them is this build's own
    parsing step, and it must not escape the turn as an untyped exception with
    the attempt evidence unpublished.
    """
    from boundarybench.providers.wire import WireResponse

    def explode() -> object:
        raise ValueError("the SDK's own words, which quote the body")

    response: WireResponse[object] = WireResponse("{}", explode, kind="test response")
    with pytest.raises(AdapterProviderError) as caught:
        assert response.parsed
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "quote the body" not in str(caught.value)
    assert repr(response) == "<WireResponse test response>"


def test_an_allowed_field_set_is_never_silently_empty() -> None:
    """A model with no declared fields would make every real response invalid.

    The allowed sets are taken from the pinned SDK's own models rather than
    hand-listed, which is only safe if "this is not one of those models" is a
    refusal rather than an empty allowlist.
    """
    from boundarybench.providers.wire import declared_wire_fields

    for candidate in (object, dict):
        with pytest.raises(ValueError, match="declares no response fields"):
            declared_wire_fields(candidate)


def test_the_mistral_capture_refuses_a_call_it_cannot_tie_to_one_response() -> None:
    """None captured, or more than one, has no answer — so it is not guessed at."""
    from boundarybench.providers.wire import wire_invalid

    assert isinstance(wire_invalid("x"), AdapterProviderError)
    inner = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    )
    wrapper = CapturedHttpClient(inner)
    with pytest.raises(AdapterProviderError) as caught:
        wrapper.take()
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    request = wrapper.build_request("POST", "https://api.mistral.ai/v1/chat/completions")
    wrapper.send(request)
    wrapper.send(request)
    assert wrapper.captured_count == 2
    # Names what it is and how much it is holding, never a byte of a body.
    assert repr(wrapper) == "<CapturedHttpClient captured=2>"
    with pytest.raises(AdapterProviderError):
        wrapper.take()
    wrapper.clear()
    wrapper.close()


def test_a_mistral_client_with_no_http_client_is_refused() -> None:
    """No client to wrap means no way to read the wire, so no request is made."""
    from boundarybench.providers.mistral_chat import MistralConfigurationError

    transport = MistralTransport([_mistral_body()])
    client = transport.client()
    client.sdk_configuration.client = None
    with pytest.raises(MistralConfigurationError):
        MistralChatAdapter(model=MISTRAL_MODEL, client=client)


def test_a_body_the_sdk_refused_to_parse_is_not_left_in_the_capture() -> None:
    """A turn that raised holds no body afterwards, and no body to be reused.

    The malformed HTTP 200 is the case where the capture is filled and the SDK
    then raises out of the same call, so the line that consumes the response is
    never reached. What is left behind is a whole provider body held on the
    adapter's client past the end of the turn — and, worse, the response the
    *next* turn's ``take`` would find waiting for it if the clear before that
    request ever failed to run. The turn's telemetry is unaffected: the
    exception carries its own ``raw_response``, which is what says a response
    arrived, so the capture can be emptied without touching that claim.
    """
    transport = MistralTransport(
        [
            {
                "id": "cmpl_test",
                "object": "chat.completion",
                "model": MISTRAL_MODEL,
                "choices": "not-a-list",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        ]
    )
    client = transport.client()
    guard = _guard()
    adapter = MistralChatAdapter(model=MISTRAL_MODEL, client=client, cost_guard=guard)
    capture = client.sdk_configuration.client
    assert isinstance(capture, CapturedHttpClient)
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    # The response telemetry is exactly what it was: a body did arrive.
    assert adapter.last_telemetry().attempts[0].response_received is True
    assert guard.exposure_usd > Decimal(0)
    assert capture.captured_count == 0


def test_a_second_adapter_over_one_captured_client_is_refused() -> None:
    """A capture belongs to one adapter instance, and the second is turned away.

    Two adapters over one client share one wrapper, and each holds its own
    single-flight lock — so neither lock sees the other's turn and both turns
    clear and fill the same capture. Locking across instances is not the fix:
    it would make one adapter's turn wait on another's, which a benchmark's
    measured wall-clock does not record. The claim is refused at construction
    instead, before any transport, and the adapter that already owns the
    capture is left exactly as it was.
    """
    from boundarybench.providers.mistral_chat import MistralConfigurationError

    transport = MistralTransport([_mistral_body(), _mistral_body()])
    client = transport.client()
    first = MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    capture = client.sdk_configuration.client
    assert isinstance(capture, CapturedHttpClient)
    with pytest.raises(MistralConfigurationError) as caught:
        MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    # The guidance names the remedy, and nothing else: no provider text, no body.
    assert "separate SDK client" in str(caught.value)
    assert transport.calls == 0
    # The refusal changed nothing the first adapter depends on: same wrapper,
    # same owner, and two more turns on it still run.
    assert client.sdk_configuration.client is capture
    assert first.next_call(_turn_request(), _deadline()).action == "read_records"
    assert first.next_call(_turn_request(), _deadline()).action == "read_records"
    assert transport.calls == 2
    assert capture.captured_count == 0


def test_separate_clients_give_each_adapter_a_capture_of_its_own() -> None:
    """The stated remedy works: one SDK client per adapter, and both run."""
    first_transport = MistralTransport([_mistral_body()])
    second_transport = MistralTransport([_mistral_body()])
    first_client = first_transport.client()
    second_client = second_transport.client()
    first = MistralChatAdapter(model=MISTRAL_MODEL, client=first_client)
    second = MistralChatAdapter(model=MISTRAL_MODEL, client=second_client)
    assert (
        first_client.sdk_configuration.client
        is not second_client.sdk_configuration.client
    )
    assert first.next_call(_turn_request(), _deadline()).action == "read_records"
    assert second.next_call(_turn_request(), _deadline()).action == "read_records"
    assert first_transport.calls == 1
    assert second_transport.calls == 1
