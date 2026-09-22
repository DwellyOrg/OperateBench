"""The xAI adapter, over xAI's OpenAI-compatible Chat Completions surface.

Every test here runs the real OpenAI SDK over an in-process transport (see
``tests.openai_transport``). No test contacts a provider, and none of them is
evidence about a model: they are evidence about the integration.

The first two tests are the ones that matter most about this integration in
particular. It does **not** use ``xai-sdk``, and everything it records about
itself has to say so — the implementation name, the SDK named in settings and
the documentation all have to agree that this is a compatibility surface rather
than the vendor's own client.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from boundarybench.adapter import (
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_CONFIGURATION,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant
from boundarybench.providers.xai_openai_compat import (
    GROK_4_5_MODEL,
    MAX_OUTPUT_TOKENS,
    XAI_BASE_URL,
    XAI_IMPLEMENTATION,
    XAIConfigurationError,
    XAIOpenAICompatAdapter,
    build_chat_request,
    build_xai_adapter,
    request_profile_for,
    xai_identity,
    xai_settings,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.conftest import minimal_card
from tests.openai_transport import (
    FAKE_API_KEY,
    RecordingTransport,
    chat_body,
    error_body,
    tool_call,
    xai_scripted_client,
)

PINNED_MODEL = "grok-test-20990101"


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


def _adapter(
    client: openai.OpenAI, *, model: str = PINNED_MODEL
) -> XAIOpenAICompatAdapter:
    return XAIOpenAICompatAdapter(model=model, client=client)


def _one_call(action: str = "read_records", arguments: str = "{}", **kwargs: object):
    return xai_scripted_client(
        chat_body([tool_call(action, arguments)], model=PINNED_MODEL, **kwargs)
    )


# -- the honesty of the name --------------------------------------------------


def test_the_implementation_name_says_it_is_a_compatibility_surface() -> None:
    """This build does not use ``xai-sdk``, so nothing here may imply that it does.

    The implementation string goes into adapter identity and is hashed into run
    identity, so it is the durable claim about *what code talked to the
    service*. A run recorded as having used the vendor's own client when it used
    an OpenAI-compatible endpoint would be false in the one place an audit
    checks.
    """
    assert XAI_IMPLEMENTATION == "xai_openai_compat_chat_completions"
    assert "xai_sdk" not in XAI_IMPLEMENTATION
    identity = xai_identity(GROK_4_5_MODEL)
    assert identity.provider == "xai"
    assert identity.implementation == XAI_IMPLEMENTATION


def test_the_settings_name_the_sdk_that_was_actually_used() -> None:
    """``sdk`` is ``openai``, because the OpenAI SDK is what serialises this."""
    settings = xai_settings(model=GROK_4_5_MODEL)
    assert settings["sdk"] == "openai"
    assert settings["sdk_version"] == openai.__version__
    assert settings["api"] == "chat_completions"
    assert settings["api_compatibility"] == "openai_compatible"
    assert settings["base_url"] == XAI_BASE_URL


def test_the_endpoint_is_pinned_to_the_documented_xai_base_url() -> None:
    assert XAI_BASE_URL == "https://api.x.ai/v1"


def test_the_request_goes_to_the_xai_chat_completions_path() -> None:
    """The compatibility surface is an endpoint, so prove the URL is that one."""
    transport = RecordingTransport(
        [chat_body([tool_call("read_records", "{}")], model=PINNED_MODEL)]
    )
    client = transport.xai_client(base_url=XAI_BASE_URL)
    _adapter(client).next_call(_turn_request(), _deadline())
    assert str(transport.requests[0].url) == "https://api.x.ai/v1/chat/completions"


# -- what goes on the wire ----------------------------------------------------


def test_the_installed_sdk_serializes_exactly_the_body_the_adapter_built() -> None:
    """The real SDK's wire body *is* the mapping this build produced."""
    transport, client = _one_call()
    request = _turn_request()
    _adapter(client).next_call(request, _deadline())
    assert transport.calls == 1
    assert transport.bodies[0] == build_chat_request(request, model=PINNED_MODEL)


def test_the_body_carries_the_pinned_settings_and_the_required_omissions() -> None:
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    body = transport.bodies[0]
    assert body["model"] == PINNED_MODEL
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    profile = request_profile_for(PINNED_MODEL)
    assert tuple(sorted(body)) == profile.sent_fields()
    for field in profile.omitted_fields():
        assert field not in body


def test_the_tool_schema_is_the_nested_chat_completions_shape() -> None:
    """Chat Completions nests the function under a ``function`` key.

    The opposite of the Responses API's flat shape, and not interchangeable with
    it, which is why each integration states which one it sends.
    """
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    for tool in transport.bodies[0]["tools"]:
        assert set(tool) == {"type", "function"}
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_the_system_prompt_travels_as_a_system_message() -> None:
    """This API has no ``instructions`` parameter, so the prompt is a message."""
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    messages = transport.bodies[0]["messages"]
    assert [entry["role"] for entry in messages] == ["system", "user"]


def test_the_request_shows_nothing_the_turn_request_does_not_carry() -> None:
    variant = _variant()
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    wire = json.dumps(transport.bodies[0], sort_keys=True)
    assert variant.variant_id not in wire
    assert variant.content_digest not in wire
    for forbidden in (
        "expected_disposition",
        "expected_primary_reason",
        "required_evidence",
        "expected_mutations",
        "matched_rule",
        "evidence_relevant",
        "answer_key",
    ):
        assert forbidden not in wire


# -- the answer ---------------------------------------------------------------


def test_one_tool_call_becomes_one_adapter_call() -> None:
    _transport, client = _one_call("ask_user", '{"fact": "boiler_age_years"}')
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.action == "ask_user"
    assert call.arguments == {"fact": "boiler_age_years"}


def test_prose_beside_the_call_is_ignored() -> None:
    _transport, client = xai_scripted_client(
        chat_body(
            [tool_call("read_records", "{}")],
            content="here is my reasoning",
            model=PINNED_MODEL,
        )
    )
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.action == "read_records"


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param(None, id="no-tool-calls-field"),
        pytest.param([], id="empty-list"),
        pytest.param(
            [
                tool_call("read_records", "{}", call_id="a"),
                tool_call("ask_user", '{"fact": "x"}', call_id="b"),
            ],
            id="two-calls",
        ),
    ],
)
def test_anything_other_than_exactly_one_call_is_a_protocol_failure(
    calls: list | None,
) -> None:
    _transport, client = xai_scripted_client(
        chat_body(
            calls, finish_reason="stop" if not calls else "tool_calls", model=PINNED_MODEL
        )
    )
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_a_tool_the_scaffold_does_not_declare_is_refused() -> None:
    _transport, client = _one_call("drop_database", "{}")
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param("[1, 2, 3]", id="not-an-object"),
    ],
)
def test_arguments_that_are_not_a_json_object_are_refused(arguments: str) -> None:
    _transport, client = _one_call("read_records", arguments)
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_a_lone_surrogate_in_the_arguments_is_refused() -> None:
    _transport, client = _one_call("ask_user", '{"fact": "\\ud800"}')
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_termination_at_the_output_ceiling_is_its_own_failure() -> None:
    """``finish_reason: length`` is the run's own ceiling, not a bad answer."""
    _transport, client = xai_scripted_client(
        chat_body(
            [tool_call("read_records", "{}")], finish_reason="length", model=PINNED_MODEL
        )
    )
    with pytest.raises(AdapterOutputLimitError) as caught:
        _adapter(client).next_call(_turn_request(), _deadline())
    assert str(MAX_OUTPUT_TOKENS) in str(caught.value)


def test_a_finish_reason_this_scaffold_does_not_model_is_a_protocol_failure() -> None:
    _transport, client = xai_scripted_client(
        chat_body(
            [tool_call("read_records", "{}")],
            finish_reason="content_filter",
            model=PINNED_MODEL,
        )
    )
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


# -- provenance and measurement -----------------------------------------------


def test_a_response_from_another_model_is_refused_before_the_action_or_usage() -> None:
    _transport, client = xai_scripted_client(
        chat_body([tool_call("read_records", "{}")], model="grok-something-else")
    )
    adapter = _adapter(client)
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert adapter.last_usage().input_tokens is None


def test_usage_is_read_from_the_chat_completions_field_names() -> None:
    """This surface reports ``prompt_tokens``/``completion_tokens``.

    Different names from the Responses API's, for the same two quantities. A
    build that read the wrong pair would price every turn at null and silently
    forfeit every reservation.
    """
    _transport, client = _one_call()
    adapter = _adapter(client)
    adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_usage().output_tokens == 29


def test_usage_is_reported_even_when_the_answer_fails_the_protocol() -> None:
    _transport, client = _one_call("drop_database", "{}")
    adapter = _adapter(client)
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137


# -- the fault taxonomy -------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "fault"),
    [
        (401, PROVIDER_FAULT_AUTHENTICATION),
        (404, PROVIDER_FAULT_CONFIGURATION),
    ],
)
def test_terminal_statuses_are_not_retried(status: int, fault: str) -> None:
    transport = RecordingTransport([(status, error_body())])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.xai_client()).next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 1


@pytest.mark.parametrize(
    ("status", "fault"),
    [(429, PROVIDER_FAULT_RATE_LIMITED), (503, PROVIDER_FAULT_SERVER_ERROR)],
)
def test_retryable_statuses_are_retried_to_the_policy_bound(
    status: int, fault: str
) -> None:
    transport = RecordingTransport([(status, error_body())] * 3)
    adapter = XAIOpenAICompatAdapter(
        model=PINNED_MODEL, client=transport.xai_client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 3


def test_a_dropped_connection_is_a_network_fault() -> None:
    transport = RecordingTransport([httpx.ConnectError("refused")] * 3)
    adapter = XAIOpenAICompatAdapter(
        model=PINNED_MODEL, client=transport.xai_client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_NETWORK_ERROR


def test_no_provider_text_reaches_a_durable_failure_detail() -> None:
    secret = "PROVIDER-PROSE-THAT-MUST-NOT-BE-DURABLE"
    transport = RecordingTransport([(401, error_body(message=secret))])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.xai_client()).next_call(_turn_request(), _deadline())
    assert secret not in str(caught.value)


def test_the_credential_never_appears_in_the_repr_or_the_settings() -> None:
    _transport, client = _one_call()
    adapter = _adapter(client)
    assert FAKE_API_KEY not in repr(adapter)
    assert FAKE_API_KEY not in json.dumps(dict(adapter.settings))


# -- configuration refusals ---------------------------------------------------


def test_a_missing_credential_refuses_before_a_client_exists() -> None:
    with pytest.raises(XAIConfigurationError) as caught:
        build_xai_adapter(
            model=GROK_4_5_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={},
        )
    assert "XAI_API_KEY" in str(caught.value)


def test_the_openai_credential_is_not_accepted_for_an_xai_run() -> None:
    """The compatibility surface uses OpenAI's *SDK*, not OpenAI's *account*.

    This is the specific hazard of reusing a client library: the SDK will read
    ``OPENAI_API_KEY`` from the environment if nothing is passed. A run that
    silently authenticated to xAI with a key meant for OpenAI — or worse, sent
    an OpenAI key to xAI's endpoint — is exactly what pinning one variable
    prevents.
    """
    with pytest.raises(XAIConfigurationError) as caught:
        build_xai_adapter(
            model=GROK_4_5_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={"OPENAI_API_KEY": "not-the-xai-key"},
        )
    assert "XAI_API_KEY" in str(caught.value)


@pytest.mark.parametrize("variable", ["XAI_BASE_URL", "OPENAI_BASE_URL"])
def test_an_environment_that_would_redirect_the_request_is_refused(variable: str) -> None:
    """``OPENAI_BASE_URL`` is refused too, and that is the point.

    It is the OpenAI SDK doing the talking, so *its* redirect variable would
    move an xAI run somewhere else just as effectively as an xAI-named one.
    """
    with pytest.raises(XAIConfigurationError) as caught:
        build_xai_adapter(
            model=GROK_4_5_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={variable: "https://elsewhere.invalid", "XAI_API_KEY": "k"},
        )
    assert variable in str(caught.value)


def test_an_unauthorised_configuration_is_refused_before_the_credential_is_read() -> None:
    with pytest.raises(XAIConfigurationError):
        build_xai_adapter(
            model=GROK_4_5_MODEL,
            configuration_id="b" * 64,
            audited={"a" * 64},
            environ={"XAI_API_KEY": "k"},
        )


def test_the_build_pre_authorises_nothing() -> None:
    with pytest.raises(XAIConfigurationError):
        build_xai_adapter(
            model=GROK_4_5_MODEL,
            configuration_id="a" * 64,
            environ={"XAI_API_KEY": "k"},
        )


def test_a_client_with_its_own_retry_loop_is_refused() -> None:
    transport = RecordingTransport([])
    with pytest.raises(XAIConfigurationError):
        XAIOpenAICompatAdapter(
            model=PINNED_MODEL, client=transport.xai_client(max_retries=2)
        )


@pytest.mark.parametrize("mutation", ["headers", "query", "auth", "cookies", "retry"])
def test_entry_client_mutation_publishes_fresh_refusal_telemetry(mutation: str) -> None:
    transport, client = _one_call()
    adapter = _adapter(client)
    adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_telemetry().attempt_count == 1
    http_client = client._client
    if mutation == "headers":
        http_client.headers["x-smuggled-route"] = "entry-refusal-secret"
    elif mutation == "query":
        http_client.params = {"tenant": "entry-refusal-secret"}
    elif mutation == "auth":
        http_client.auth = ("entry-refusal-secret", "not-a-credential")
    elif mutation == "cookies":
        http_client.cookies.set("session", "entry-refusal-secret")
    else:
        client.max_retries = 1

    with pytest.raises(XAIConfigurationError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert "entry-refusal-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert transport.calls == 1
    assert adapter.last_usage().as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }
    telemetry = adapter.last_telemetry()
    assert telemetry.attempts == ()
    assert telemetry.terminal_reason == "pre_dispatch_refused"
