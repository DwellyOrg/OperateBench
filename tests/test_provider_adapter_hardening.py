"""What the three new adapters refuse from a client, a body and a policy.

Every test here is adversarial: it hands an adapter something a cooperating
provider and a careful operator would never produce — a client pointed at
another host, a response carrying a field the vendor's own schema does not
declare, a token count below zero, a retry policy that cannot be followed,
arguments that decode two ways — and asserts the adapter refuses it *by name*,
before the thing it protects has happened.

Every one of them runs the real installed SDK over an in-process
``httpx.MockTransport``. Nothing here opens a socket; one test proves that by
making the attempt fail loudly.

None of this is evidence about a model. It is evidence about the integration.
"""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from typing import Any

import httpx
import openai
import pytest
from mistralai.client import Mistral
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from boundarybench.adapter import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProtocolError,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.pricing import price_for
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.providers.mistral_chat import (
    MISTRAL_BASE_URL,
    MistralChatAdapter,
    MistralConfigurationError,
    build_mistral_adapter,
)
from boundarybench.providers.openai_responses import (
    OPENAI_BASE_URL,
    OpenAIConfigurationError,
    OpenAIResponsesAdapter,
    build_openai_adapter,
)
from boundarybench.providers.xai_openai_compat import (
    XAI_BASE_URL,
    XAIConfigurationError,
    XAIOpenAICompatAdapter,
    build_xai_adapter,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from tests import mistral_transport, openai_transport
from tests.conftest import minimal_card

OPENAI_MODEL = "gpt-5.6-luna"
XAI_MODEL = "grok-4.5"
MISTRAL_MODEL = "mistral-small-2603"

#: Somewhere that is nobody's endpoint. Nothing answers on it, and nothing here
#: would reach it if something did.
ATTACKER_ENDPOINT = "https://attacker.invalid/custom"


def _request() -> TurnRequest:
    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(cube.variants[0]),
        turns_remaining=12,
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _guard(provider: str, model: str) -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("100"),
            max_episodes=1,
            price=price_for(provider=provider, model=model),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )


# -- the endpoint a client will actually send to -------------------------------


def _openai_body(arguments: str = "{}") -> dict[str, Any]:
    return openai_transport.responses_body(
        [openai_transport.function_call_item("read_records", arguments)],
        model=OPENAI_MODEL,
    )


def _xai_body(arguments: str = "{}") -> dict[str, Any]:
    return openai_transport.chat_body(
        [openai_transport.tool_call("read_records", arguments)], model=XAI_MODEL
    )


def _mistral_body(arguments: str = "{}") -> dict[str, Any]:
    return mistral_transport.chat_body(
        [mistral_transport.tool_call("read_records", arguments)], model=MISTRAL_MODEL
    )


def _mistral_client(transport: mistral_transport.RecordingTransport, **kwargs: Any):
    """A real Mistral client on this transport, with a caller-chosen server URL.

    Built here rather than through ``build_client`` because ``build_client``
    pins the server URL and is not a way to reach another one — which is exactly
    the property under test.
    """
    kwargs.setdefault("server_url", MISTRAL_BASE_URL)
    return Mistral(
        api_key=mistral_transport.FAKE_API_KEY,
        client=httpx.Client(transport=httpx.MockTransport(transport.handler)),
        retry_config=RetryConfig("none", BackoffStrategy(1, 1, 1, 1), False),
        **kwargs,
    )


def test_the_openai_constructor_refuses_a_client_pointed_at_another_host() -> None:
    """A client that would send this credential elsewhere is refused, not used.

    The settings this adapter publishes — and the run manifest hashes — state
    ``https://api.openai.com/v1``. A client injected at another host makes that
    statement false while every artefact still reads as though it were true, and
    it sends the credential to whoever holds the other host.
    """
    transport = openai_transport.RecordingTransport([_openai_body()])
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=transport.client(base_url=ATTACKER_ENDPOINT)
        )
    assert OPENAI_BASE_URL in str(raised.value)
    assert transport.calls == 0


def test_the_xai_constructor_refuses_a_client_pointed_at_another_host() -> None:
    transport = openai_transport.RecordingTransport([_xai_body()])
    with pytest.raises(XAIConfigurationError) as raised:
        XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=transport.client(base_url=ATTACKER_ENDPOINT)
        )
    assert XAI_BASE_URL in str(raised.value)
    assert transport.calls == 0


def test_the_xai_constructor_refuses_a_client_left_on_the_sdks_own_default() -> None:
    """The default is OpenAI's endpoint, which is the failure mode that is silent.

    A forgotten ``base_url`` does not raise anywhere in the SDK: it sends an xAI
    credential to OpenAI and records ``https://api.x.ai/v1`` for having done so.
    """
    transport = openai_transport.RecordingTransport([_xai_body()])
    with pytest.raises(XAIConfigurationError):
        XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=transport.client(base_url=OPENAI_BASE_URL)
        )
    assert transport.calls == 0


def test_the_mistral_constructor_refuses_a_client_pointed_at_another_host() -> None:
    transport = mistral_transport.RecordingTransport([_mistral_body()])
    client = _mistral_client(transport, server_url="https://attacker.invalid")
    with pytest.raises(MistralConfigurationError) as raised:
        MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    assert MISTRAL_BASE_URL in str(raised.value)
    assert transport.calls == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1/custom",
        "https://api.openai.com/v2",
        "https://api.openai.com",
        "http://api.openai.com/v1",
        "https://api.openai.com:8443/v1",
        "https://api.openai.com/v1?tenant=attacker",
        "https://api.openai.com/v1#attacker",
        "https://api.openai.com/v1//",
    ],
)
def test_only_the_exact_pinned_endpoint_is_accepted(endpoint: str) -> None:
    """A path, a scheme, a port, a query or a fragment is a different endpoint.

    Each of these is *nearly* the official base URL, and each would send the
    request somewhere the manifest does not describe or in a form this build did
    not pin. Only a single trailing slash is normalised away, because that is
    the one difference the SDK itself introduces.
    """
    transport = openai_transport.RecordingTransport([_openai_body()])
    with pytest.raises(OpenAIConfigurationError):
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=transport.client(base_url=endpoint)
        )
    assert transport.calls == 0


@pytest.mark.parametrize(
    "endpoint",
    [OPENAI_BASE_URL, OPENAI_BASE_URL + "/", "https://api.openai.com:443/v1"],
)
def test_the_official_endpoint_is_accepted_in_both_trailing_slash_forms(
    endpoint: str,
) -> None:
    """The one normalisation this build performs, and proof it is not a hole.

    The SDK stores a trailing slash whichever form it was handed, so refusing
    one of the two would refuse the officially-built client. That is the only
    difference this build normalises away.

    The explicit ``:443`` is here to record where the boundary actually falls
    rather than where this build draws it: ``httpx`` removes a scheme's default
    port while parsing, so what reaches this check is already the official URL
    and there is nothing left to accept or refuse. A *non*-default port is a
    different endpoint and is refused above.
    """
    transport = openai_transport.RecordingTransport([_openai_body()])
    adapter = OpenAIResponsesAdapter(
        model=OPENAI_MODEL, client=transport.client(base_url=endpoint)
    )
    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    assert str(transport.requests[0].url).startswith(OPENAI_BASE_URL)


def test_an_endpoint_url_object_is_read_the_same_way_a_string_is() -> None:
    """An ``httpx.URL`` is what the SDK actually stores, so it is what is checked.

    Both directions are asserted from the object form: the official URL passed
    as a parsed object is accepted, and one carrying userinfo — a URL's own
    place to hide a credential — is not.
    """
    transport = openai_transport.RecordingTransport([_openai_body(), _openai_body()])
    OpenAIResponsesAdapter(
        model=OPENAI_MODEL, client=transport.client(base_url=httpx.URL(OPENAI_BASE_URL))
    )
    with pytest.raises(OpenAIConfigurationError):
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL,
            client=transport.client(
                base_url=httpx.URL("https://operator:hunter2@api.openai.com/v1")
            ),
        )


def test_the_refusal_never_quotes_the_endpoint_it_refused() -> None:
    """A base URL can carry a credential in its userinfo, so it is not echoed."""
    transport = openai_transport.RecordingTransport([_openai_body()])
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL,
            client=transport.client(base_url="https://operator:hunter2@evil.invalid/v1"),
        )
    message = str(raised.value)
    assert "hunter2" not in message
    assert "evil.invalid" not in message


@pytest.mark.parametrize("field", ["organization", "project"])
def test_a_client_carrying_an_account_scope_is_refused(field: str) -> None:
    """These do not move the request; they move who is billed for it.

    This build already refuses the environment variables that set them, and an
    injected client is the same hole through a different door.
    """
    transport = openai_transport.RecordingTransport([_openai_body()])
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=transport.client(**{field: "org_attacker"})
        )
    assert field in str(raised.value)
    assert transport.calls == 0


def test_an_endpoint_swapped_after_construction_is_caught_before_dispatch() -> None:
    """A client is mutable, so the check that matters is the last one before sending.

    The adapter is built against the pinned endpoint and passes; the client is
    then pointed elsewhere, as a compromised or careless caller could do between
    two turns. The turn is refused and no request is made.
    """
    transport = openai_transport.RecordingTransport([_openai_body()])
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())
    adapter._client.base_url = httpx.URL(ATTACKER_ENDPOINT)
    with pytest.raises(OpenAIConfigurationError):
        adapter.next_call(_request(), _deadline())
    assert transport.calls == 0


def test_an_xai_endpoint_swapped_after_construction_is_caught_before_dispatch() -> None:
    transport = openai_transport.RecordingTransport([_xai_body()])
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=transport.xai_client())
    adapter._client.base_url = httpx.URL(ATTACKER_ENDPOINT)
    with pytest.raises(XAIConfigurationError):
        adapter.next_call(_request(), _deadline())
    assert transport.calls == 0


def test_a_mistral_endpoint_swapped_after_construction_is_caught_before_dispatch() -> (
    None
):
    transport = mistral_transport.RecordingTransport([_mistral_body()])
    adapter = MistralChatAdapter(model=MISTRAL_MODEL, client=_mistral_client(transport))
    adapter._client.sdk_configuration.server_url = "https://attacker.invalid"
    with pytest.raises(MistralConfigurationError):
        adapter.next_call(_request(), _deadline())
    assert transport.calls == 0


# -- the response contract -----------------------------------------------------


def _openai_turn(body: Any, *, guard: RunCostGuard | None = None):
    transport, client = openai_transport.scripted_client(body)
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=client, cost_guard=guard)
    return transport, adapter


def _xai_turn(body: Any, *, guard: RunCostGuard | None = None):
    transport, client = openai_transport.xai_scripted_client(body)
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=client, cost_guard=guard)
    return transport, adapter


def _mistral_turn(body: Any, *, guard: RunCostGuard | None = None):
    transport, client = mistral_transport.scripted_client(body)
    adapter = MistralChatAdapter(model=MISTRAL_MODEL, client=client, cost_guard=guard)
    return transport, adapter


def test_an_unknown_top_level_response_field_is_refused_by_openai() -> None:
    """A field the vendor's own schema does not declare is not ignored.

    This build reads one action and two token counts out of a body it did not
    write. A part of that body neither the SDK nor this integration can describe
    is either a surface that moved under a pinned SDK version or content that
    did not come from the model, and both are reasons to refuse the whole thing
    rather than to read the parts that look familiar.
    """
    body = _openai_body()
    body["attacker_unknown_response_field"] = "CANARY"
    _transport, adapter = _openai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "CANARY" not in str(raised.value)


def test_an_unknown_field_on_an_openai_output_item_is_refused() -> None:
    """The recursion is the point: the action itself travels inside an item."""
    item = openai_transport.function_call_item("read_records", "{}")
    item["attacker_unknown_tool_call_field"] = "CANARY"
    _transport, adapter = _openai_turn(
        openai_transport.responses_body([item], model=OPENAI_MODEL)
    )
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_an_unknown_field_inside_the_openai_usage_block_is_refused() -> None:
    """Usage is priced and committed against a cap, so it is checked as hard."""
    body = _openai_body()
    body["usage"]["attacker_unknown_usage_field"] = 7
    _transport, adapter = _openai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_an_unknown_top_level_response_field_is_refused_by_xai() -> None:
    body = _xai_body()
    body["attacker_unknown_response_field"] = "CANARY"
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_an_unknown_field_on_an_xai_tool_call_is_refused() -> None:
    call = openai_transport.tool_call("read_records", "{}")
    call["attacker_unknown_tool_call_field"] = "CANARY"
    _transport, adapter = _xai_turn(openai_transport.chat_body([call], model=XAI_MODEL))
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_an_unknown_field_on_the_nested_xai_function_object_is_refused() -> None:
    """The deepest one, and the one a top-level check would never see.

    The function object is where the action's name and arguments live, so an
    undeclared field beside them is the least visible place a body can differ
    from the contract this build asked for.
    """
    call = openai_transport.tool_call("read_records", "{}")
    call["function"]["attacker_unknown_function_field"] = "CANARY"
    _transport, adapter = _xai_turn(openai_transport.chat_body([call], model=XAI_MODEL))
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_the_mistral_sdk_drops_unknown_fields_and_the_wire_check_still_sees_them() -> (
    None
):
    """The SDK still discards them. The adapter no longer inherits that blindness.

    Both halves are pinned, because they are two different facts.
    ``mistralai`` 2.9.2 models ignore undeclared fields at parse time rather
    than preserving them, so the recursive contract check that runs on the
    *parsed* object finds nothing to refuse — the extra field is gone before any
    object this build can inspect exists. That was, until this adapter read the
    wire, a real gap: the field was accepted in full and silently, and this
    build could not report what it never saw.

    So the body is now checked as the exact JSON the provider sent, before the
    SDK's reading of it is trusted, and the undeclared field is refused by name.
    The first assertion fails on the day the SDK starts preserving extras — which
    is when the first paragraph stops being true — and the second fails on the
    day this adapter stops reading the wire.
    """
    from mistralai.client.models import ChatCompletionResponse

    body = _mistral_body()
    body["attacker_unknown_response_field"] = "CANARY"
    parsed = ChatCompletionResponse.model_validate(body)
    assert parsed.model_extra in (None, {})

    _transport, adapter = _mistral_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "CANARY" not in str(raised.value)


def test_an_invalid_response_keeps_its_reservation_as_exposure_not_as_zero() -> None:
    """A body that arrived is not free just because it could not be read.

    The attempt records that a response *was* received, reports no usage, and
    the reservation that authorised the request stays on the guard as exposure.
    Settling it at zero would let a run whose every answer was unreadable report
    that it spent nothing.
    """
    guard = _guard("openai", OPENAI_MODEL)
    body = _openai_body()
    body["attacker_unknown_response_field"] = "CANARY"
    _transport, adapter = _openai_turn(body, guard=guard)
    with pytest.raises(AdapterProviderError):
        adapter.next_call(_request(), _deadline())

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["response_received"] is True
    assert telemetry["usage_reported"] is False
    assert telemetry["attempts"][0]["fault"] == PROVIDER_FAULT_RESPONSE_INVALID
    assert telemetry["attempts"][0]["cost_settlement"] == "forfeited"
    assert adapter.last_usage().as_dict()["cost_usd"] is None
    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


# -- malformed collections -----------------------------------------------------


def test_a_null_openai_output_becomes_a_named_refusal_not_a_type_error() -> None:
    """``output: null`` is the provider breaking its own schema, and is named as such.

    Before this, the first thing to notice was the loop that iterated it, four
    frames down, as a ``TypeError`` carrying the interpreter's words — raised
    *after* the turn had been measured and settled, so the failure escaped the
    boundary that publishes attempt evidence and carried text this build did not
    choose.
    """
    guard = _guard("openai", OPENAI_MODEL)
    malformed = openai_transport.responses_body([], model=OPENAI_MODEL)
    malformed["output"] = None
    transport, client = openai_transport.scripted_client(malformed, _openai_body())
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=client, cost_guard=guard)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "NoneType" not in str(raised.value)

    first = adapter.last_telemetry().as_dict()
    assert first["attempt_count"] == 1
    assert first["response_received"] is True
    assert first["usage_reported"] is False
    assert adapter.last_usage().as_dict()["input_tokens"] is None

    # The turn after it is a clean turn: the measurement is cleared before
    # anything else happens, so no part of the refused turn is attributed to it.
    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    second = adapter.last_telemetry().as_dict()
    assert second["attempt_count"] == 1
    assert second["terminal_reason"] == "response"
    assert second["usage_reported"] is True
    assert adapter.last_usage().as_dict()["input_tokens"] == 137
    assert transport.calls == 2


def test_a_null_xai_choices_becomes_a_named_refusal_not_a_type_error() -> None:
    guard = _guard("xai", XAI_MODEL)
    malformed = openai_transport.chat_body([], model=XAI_MODEL)
    malformed["choices"] = None
    transport, client = openai_transport.xai_scripted_client(malformed, _xai_body())
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=client, cost_guard=guard)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "NoneType" not in str(raised.value)
    assert adapter.last_telemetry().as_dict()["response_received"] is True

    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    assert adapter.last_telemetry().as_dict()["attempt_count"] == 1
    assert adapter.last_usage().as_dict()["input_tokens"] == 137
    assert transport.calls == 2


def test_an_xai_choice_with_no_message_is_a_protocol_failure() -> None:
    """A choice that states no message states no action, and says so."""
    body = openai_transport.chat_body([], model=XAI_MODEL)
    body["choices"][0]["message"] = None
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_a_null_mistral_choices_never_reaches_the_adapter_untyped() -> None:
    """This SDK validates strictly, and the refusal is still a named one.

    ``mistralai`` refuses the null while parsing rather than handing the adapter
    a ``None`` where a list belongs, and this integration already maps that
    SDK's validation error onto
    :data:`~boundarybench.adapter.PROVIDER_FAULT_RESPONSE_INVALID`. So the same
    body that escaped the other two integrations as a raw ``TypeError`` lands
    here on the same boundary they now land on — recorded, classified and
    carrying none of the SDK's own words.
    """
    body = _mistral_body()
    body["choices"] = None
    _transport, adapter = _mistral_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert "NoneType" not in str(raised.value)


# -- the retry policy ----------------------------------------------------------

_BAD_POLICIES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("zero attempts", {"max_attempts": 0}),
    ("negative attempts", {"max_attempts": -1}),
    ("boolean attempts", {"max_attempts": True}),
    ("float attempts", {"max_attempts": 3.0}),
    ("negative backoff", {"initial_backoff_seconds": -1.0}),
    ("nan backoff", {"initial_backoff_seconds": float("nan")}),
    ("infinite backoff", {"initial_backoff_seconds": float("inf")}),
    ("boolean backoff", {"initial_backoff_seconds": True}),
    ("shrinking multiplier", {"backoff_multiplier": 0.5}),
    ("zero multiplier", {"backoff_multiplier": 0.0}),
    ("nan multiplier", {"backoff_multiplier": float("nan")}),
    ("infinite multiplier", {"backoff_multiplier": float("inf")}),
    ("boolean multiplier", {"backoff_multiplier": True}),
)


@pytest.mark.parametrize(
    "label,overrides", _BAD_POLICIES, ids=[n for n, _ in _BAD_POLICIES]
)
def test_every_adapter_refuses_a_retry_policy_it_could_not_follow(
    label: str, overrides: dict[str, Any]
) -> None:
    """The policy is inside run identity, so a policy the loop ignores is a lie.

    ``max_attempts=0`` records a bounded policy and then sends a request anyway;
    ``NaN`` compares false against every bound the loop checks a backoff
    against, so it slips past the affordability check that stops a retry
    outliving the turn; a multiplier below one asks the provider *faster* the
    worse the outage gets. None of them is a schedule, and none of them is
    refused by anything else in the stack.
    """
    policy = ProviderRetryPolicy(**overrides)

    openai_steps = openai_transport.RecordingTransport([_openai_body()])
    with pytest.raises(OpenAIConfigurationError):
        OpenAIResponsesAdapter(
            model=OPENAI_MODEL, client=openai_steps.client(), retry=policy
        )
    assert openai_steps.calls == 0

    xai_steps = openai_transport.RecordingTransport([_xai_body()])
    with pytest.raises(XAIConfigurationError):
        XAIOpenAICompatAdapter(
            model=XAI_MODEL, client=xai_steps.xai_client(), retry=policy
        )
    assert xai_steps.calls == 0

    mistral_steps = mistral_transport.RecordingTransport([_mistral_body()])
    with pytest.raises(MistralConfigurationError):
        MistralChatAdapter(
            model=MISTRAL_MODEL, client=_mistral_client(mistral_steps), retry=policy
        )
    assert mistral_steps.calls == 0


@pytest.mark.parametrize(
    "factory,error",
    [
        (build_openai_adapter, OpenAIConfigurationError),
        (build_xai_adapter, XAIConfigurationError),
        (build_mistral_adapter, MistralConfigurationError),
    ],
)
def test_a_factory_refuses_a_bad_retry_policy_before_it_reads_a_credential(
    factory: Any, error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused pre-client, so an unusable policy never handles a key or opens a socket.

    Asserted through the message: with no credential in the environment and no
    audited configuration, several refusals are available, and the one that must
    win is the one that costs nothing.
    """
    for variable in ("OPENAI_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    model = {
        build_openai_adapter: OPENAI_MODEL,
        build_xai_adapter: XAI_MODEL,
        build_mistral_adapter: MISTRAL_MODEL,
    }[factory]
    with pytest.raises(error) as raised:
        factory(model=model, retry=ProviderRetryPolicy(max_attempts=0), environ={})
    assert "max_attempts" in str(raised.value)


def test_the_exact_lower_bounds_of_a_retry_policy_are_accepted() -> None:
    """One attempt, no first wait and a constant multiplier is a policy.

    The bounds are exact rather than defensive margins: a multiplier of one is a
    constant wait, which is still a backoff, and a zero first wait retries
    immediately once — both are schedules an operator can state and this loop
    can follow.
    """
    policy = ProviderRetryPolicy(
        max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
    )
    transport, client = openai_transport.scripted_client(_openai_body())
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=client, retry=policy)
    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    assert adapter.settings["retry"] == policy.as_dict()
    assert transport.calls == 1


# -- token counts --------------------------------------------------------------


@pytest.mark.parametrize("value", [-7, True, False, 1.5, "12"])
def test_openai_refuses_a_usage_count_that_is_not_a_count(value: Any) -> None:
    """A count is priced and committed against a cap, so it must be a count.

    A negative one *credits* the run against its own cap, which is the single
    direction a cost control must never move. A ``bool`` prices a turn at one
    token because Python makes ``True`` an ``int``. Neither is a measurement.
    """
    guard = _guard("openai", OPENAI_MODEL)
    body = _openai_body()
    body["usage"]["input_tokens"] = value
    _transport, adapter = _openai_turn(body, guard=guard)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert str(value) not in str(raised.value)
    assert adapter.last_usage().as_dict()["input_tokens"] is None
    assert adapter.last_usage().as_dict()["cost_usd"] is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > 0


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
def test_xai_refuses_a_negative_usage_count(field: str) -> None:
    body = _xai_body()
    body["usage"][field] = -7
    _transport, adapter = _xai_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
def test_mistral_refuses_a_negative_usage_count(field: str) -> None:
    body = _mistral_body()
    body["usage"][field] = -7
    _transport, adapter = _mistral_turn(body)
    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_a_valid_usage_block_is_still_measured_and_settled() -> None:
    """The refusals above must not have made every count unreadable."""
    guard = _guard("openai", OPENAI_MODEL)
    _transport, adapter = _openai_turn(_openai_body(), guard=guard)
    assert adapter.next_call(_request(), _deadline()).action == "read_records"
    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] == 137
    assert usage["output_tokens"] == 29
    assert usage["cost_usd"] is not None
    assert guard.measured_usd > 0


# -- tool call arguments -------------------------------------------------------


def test_openai_refuses_arguments_that_name_a_key_twice() -> None:
    """``{"a":1,"a":2}`` is one call two readers resolve two ways.

    ``json.loads`` keeps the last value and says nothing, so the ledger row
    would record an argument the model may not have meant, with no way for a
    reader to tell it had a choice.
    """
    _transport, adapter = _openai_turn(_openai_body('{"fact":"a","fact":"b"}'))
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "same key" in str(raised.value)
    assert "fact" not in str(raised.value)


def test_xai_refuses_arguments_that_name_a_key_twice() -> None:
    _transport, adapter = _xai_turn(_xai_body('{"fact":"a","fact":"b"}'))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_mistral_refuses_arguments_that_name_a_key_twice() -> None:
    _transport, adapter = _mistral_turn(_mistral_body('{"fact":"a","fact":"b"}'))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_a_nested_object_that_names_a_key_twice_is_refused_too() -> None:
    """The ambiguity is not only a top-level one."""
    _transport, adapter = _openai_turn(_openai_body('{"outer":{"fact":"a","fact":"b"}}'))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_arguments_carrying_a_lone_surrogate_are_refused() -> None:
    """Text UTF-8 cannot encode cannot be evidence of what the model said.

    Sent as raw bytes, because no Python value survives being encoded from one:
    the escape has to reach the decoder as the provider would send it.
    """
    body = _openai_body('{"fact": "\ud800"}')
    raw = json.dumps(body, ensure_ascii=True).encode("ascii")
    transport = openai_transport.RecordingTransport([raw])
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())
    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(_request(), _deadline())
    assert "cannot record" in str(raised.value)


@pytest.mark.parametrize("arguments", ['["fact"]', '"fact"', "12", "null", "true"])
def test_arguments_that_are_not_a_json_object_are_refused(arguments: str) -> None:
    """An action's arguments are a mapping. A list or a bare scalar is not one."""
    _transport, adapter = _openai_turn(_openai_body(arguments))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_arguments_carrying_a_non_json_number_are_refused() -> None:
    """``NaN`` decodes in Python and has no JSON representation to write back."""
    _transport, adapter = _openai_turn(_openai_body('{"fact": NaN}'))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


def test_arguments_nested_past_this_builds_limit_are_refused() -> None:
    """A stated limit, rather than whichever frame the interpreter gave up in."""
    from boundarybench.jsonsafe import MAX_JSON_DEPTH

    deep = '{"a":' * (MAX_JSON_DEPTH + 2) + "1" + "}" * (MAX_JSON_DEPTH + 2)
    _transport, adapter = _openai_turn(_openai_body(deep))
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_request(), _deadline())


@pytest.mark.parametrize(
    "turn,body",
    [
        (_openai_turn, _openai_body('{"attacker_unknown_argument":"CANARY"}')),
        (_xai_turn, _xai_body('{"attacker_unknown_argument":"CANARY"}')),
        (_mistral_turn, _mistral_body('{"attacker_unknown_argument":"CANARY"}')),
    ],
    ids=["openai", "xai", "mistral"],
)
def test_an_undeclared_action_argument_travels_as_an_adapter_call(
    turn: Any, body: Any
) -> None:
    """Deliberately *not* refused here, and this pins that decision.

    Whether an argument suits an action is the scaffold's question, and the
    runner already answers it against the ``TurnRequest`` it built. Answering it
    here as well would put one rule in two places that can drift — and the
    failure mode of the adapter winning is worse than the duplication: a call
    the model really made would be dropped at the transport instead of recorded
    as the protocol failure it is. The end-to-end test below is the other half
    of this claim.
    """
    _transport, adapter = turn(body)
    call = adapter.next_call(_request(), _deadline())
    assert call.action == "read_records"
    assert call.arguments == {"attacker_unknown_argument": "CANARY"}


def test_the_runner_refuses_the_undeclared_argument_the_adapter_passed_on() -> None:
    """The other half: the handoff, end to end, through the real runner.

    A provider answers with an argument the scaffold never declared; the adapter
    decodes it and hands it on; the runner refuses it against the schema the
    model was actually shown, and the episode ends as a recorded protocol
    failure rather than as a silently-dropped turn.
    """
    from boundarybench.ledger import KIND_MALFORMED_ARGUMENTS, PHASE_RESPONSE_VALIDATION
    from boundarybench.runner import RunLimits, run_episode

    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    body = _openai_body('{"attacker_unknown_argument":"CANARY"}')
    transport = openai_transport.RecordingTransport([], default=body)
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=transport.client())

    result = run_episode(
        variant=cube.variants[0],
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=adapter,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )

    assert not result.succeeded
    assert result.failure_event is not None
    assert result.failure_event["phase"] == PHASE_RESPONSE_VALIDATION
    assert result.failure_event["kind"] == KIND_MALFORMED_ARGUMENTS
    assert "unknown argument" in str(result.error_detail)
    assert transport.calls == 1


# -- no network ----------------------------------------------------------------


def test_no_test_in_this_module_can_reach_a_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proven by making it impossible rather than by asserting it in prose.

    Every way a socket can be *connected* is replaced with something that
    raises, and a full turn is then run against each of the three adapters.
    ``MockTransport`` answers every request inside the process, so all three
    complete — a transport that had quietly fallen back to a real connection
    would fail here instead.

    Connecting is what is blocked rather than socket construction: the
    interpreter builds socket pairs of its own for unrelated reasons, and a test
    that failed on those would be asserting something other than "this build
    did not call a provider".
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    for turn, body in (
        (_openai_turn, _openai_body()),
        (_xai_turn, _xai_body()),
        (_mistral_turn, _mistral_body()),
    ):
        transport, adapter = turn(body)
        assert adapter.next_call(_request(), _deadline()).action == "read_records"
        assert isinstance(transport.requests[0].url.host, str) and transport.calls == 1


def test_the_openai_sdk_under_test_is_the_pinned_one() -> None:
    """The SDK decides serialisation, parsing and error mapping, so it is pinned."""
    from importlib.metadata import version

    assert openai.__version__ == "2.53.0"
    assert version("mistralai") == "2.9.2"
