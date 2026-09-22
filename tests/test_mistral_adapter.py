"""The Mistral chat-completions adapter.

Every test here runs the real SDK over an in-process transport (see
``tests.mistral_transport``). No test contacts a provider, and none of them is
evidence about a model: they are evidence about the integration.
"""

from __future__ import annotations

import json

import httpx
import pytest
from mistralai.client import Mistral

from boundarybench.adapter import (
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_CONFIGURATION,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_REQUEST_REJECTED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_TIMEOUT,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant
from boundarybench.providers.mistral_chat import (
    MAX_OUTPUT_TOKENS,
    MISTRAL_BASE_URL,
    MISTRAL_SMALL_MODEL,
    SDK_INJECTED_FIELDS,
    MistralChatAdapter,
    MistralConfigurationError,
    build_mistral_adapter,
    build_mistral_request,
    mistral_identity,
    mistral_settings,
    request_profile_for,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.conftest import minimal_card
from tests.mistral_transport import (
    FAKE_API_KEY,
    RecordingTransport,
    chat_body,
    error_body,
    scripted_client,
    tool_call,
)

PINNED_MODEL = "mistral-test-20990101"


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


def _adapter(client: Mistral, *, model: str = PINNED_MODEL) -> MistralChatAdapter:
    return MistralChatAdapter(model=model, client=client)


def _one_call(action: str = "read_records", arguments: str = "{}", **kwargs: object):
    return scripted_client(
        chat_body([tool_call(action, arguments)], model=PINNED_MODEL, **kwargs)
    )


# -- what goes on the wire ----------------------------------------------------


def test_the_wire_body_is_the_payload_plus_exactly_the_sdk_injected_fields() -> None:
    """The SDK adds ``stream: false``, and the settings say so rather than hide it.

    This is the one place in the build where what leaves the process is not
    exactly what the adapter built. Asserting the wire body against
    ``{**payload, **injected}`` rather than against the payload alone states the
    difference precisely: the adapter is answerable for its own fields, the SDK
    is answerable for the ones it adds, and a *new* field appearing from either
    side still fails this test.
    """
    transport, client = _one_call()
    request = _turn_request()
    _adapter(client).next_call(request, _deadline())
    assert transport.calls == 1
    payload = build_mistral_request(request, model=PINNED_MODEL)
    assert transport.bodies[0] == {**payload, **SDK_INJECTED_FIELDS}
    # The injected field is genuinely absent from what this build authored, so
    # the two halves of the claim cannot quietly become one.
    assert "stream" not in payload
    assert SDK_INJECTED_FIELDS == {"stream": False}


def test_the_settings_record_the_sdk_injected_fields() -> None:
    """Settings are the run's claim about what the provider received.

    A settings mapping that described only the adapter's own fields would be an
    incomplete description of the request that was actually sent.
    """
    settings = mistral_settings(model=MISTRAL_SMALL_MODEL)
    assert settings["sdk_injected_fields"] == {"stream": False}


def test_the_body_carries_the_pinned_settings_and_the_required_omissions() -> None:
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    body = transport.bodies[0]
    assert body["model"] == PINNED_MODEL
    assert body["max_tokens"] == MAX_OUTPUT_TOKENS
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    profile = request_profile_for(PINNED_MODEL)
    assert tuple(sorted(body)) == tuple(
        sorted({*profile.sent_fields(), *SDK_INJECTED_FIELDS})
    )
    for field in profile.omitted_fields():
        assert field not in body


def test_the_tool_schema_is_the_nested_shape_with_the_query_enum() -> None:
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    tools = transport.bodies[0]["tools"]
    assert tools
    for tool in tools:
        assert set(tool) == {"type", "function"}
        assert set(tool["function"]) == {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["additionalProperties"] is False


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


def test_the_endpoint_is_the_documented_one() -> None:
    assert MISTRAL_BASE_URL == "https://api.mistral.ai"


# -- the answer ---------------------------------------------------------------


def test_one_tool_call_becomes_one_adapter_call() -> None:
    _transport, client = _one_call("ask_user", '{"fact": "boiler_age_years"}')
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.action == "ask_user"
    assert call.arguments == {"fact": "boiler_age_years"}


def test_arguments_that_arrive_already_decoded_are_accepted() -> None:
    """This SDK types ``arguments`` as a string *or* an object, so both work.

    A build that assumed the string form would raise a protocol failure against
    a perfectly well-formed answer, and record the model as having misbehaved
    when it had not.
    """
    _transport, client = scripted_client(
        chat_body(
            [tool_call("ask_user", {"fact": "boiler_age_years"})], model=PINNED_MODEL
        )
    )
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.arguments == {"fact": "boiler_age_years"}


def test_prose_beside_the_call_is_ignored() -> None:
    _transport, client = scripted_client(
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
        pytest.param(None, id="no-tool-calls"),
        pytest.param(
            [
                tool_call("read_records", "{}", call_id="aaaaaaaaa"),
                tool_call("ask_user", '{"fact": "x"}', call_id="bbbbbbbbb"),
            ],
            id="two-calls",
        ),
    ],
)
def test_anything_other_than_exactly_one_call_is_a_protocol_failure(
    calls: list | None,
) -> None:
    _transport, client = scripted_client(
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


def test_a_tool_call_that_does_not_declare_itself_a_function_call_is_refused() -> None:
    """A call whose declared type this build does not model is not read as one.

    This SDK types a tool call's ``type`` as ``"function"`` *or* an unrecognised
    string, so a surface that grew a second kind of call would parse cleanly and
    arrive here carrying a ``function`` field the adapter would happily read. The
    name and arguments of a call this build cannot interpret are not evidence of
    the action the model took, and reading them as one would record an action
    nobody can show was requested. The xAI adapter refuses the same shape on the
    same grounds; a scaffold that accepted it from one vendor and refused it from
    another would be two different protocols.
    """
    call = tool_call("read_records", "{}")
    call["type"] = "custom"
    _transport, client = scripted_client(chat_body([call], model=PINNED_MODEL))
    with pytest.raises(AdapterProtocolError) as caught:
        _adapter(client).next_call(_turn_request(), _deadline())
    # Provider-controlled text stays out of the durable detail.
    assert "custom" not in str(caught.value)


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
    _transport, client = scripted_client(
        chat_body(
            [tool_call("read_records", "{}")], finish_reason="length", model=PINNED_MODEL
        )
    )
    with pytest.raises(AdapterOutputLimitError) as caught:
        _adapter(client).next_call(_turn_request(), _deadline())
    assert str(MAX_OUTPUT_TOKENS) in str(caught.value)


def test_a_finish_reason_this_scaffold_does_not_model_is_a_protocol_failure() -> None:
    _transport, client = scripted_client(
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
    _transport, client = scripted_client(
        chat_body([tool_call("read_records", "{}")], model="mistral-something-else")
    )
    adapter = _adapter(client)
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    assert adapter.last_usage().input_tokens is None


def test_usage_is_reported_even_when_the_answer_fails_the_protocol() -> None:
    _transport, client = _one_call("drop_database", "{}")
    adapter = _adapter(client)
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_usage().output_tokens == 29


# -- the fault taxonomy -------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "fault"),
    [
        (401, PROVIDER_FAULT_AUTHENTICATION),
        (403, PROVIDER_FAULT_AUTHENTICATION),
        (404, PROVIDER_FAULT_CONFIGURATION),
        (400, PROVIDER_FAULT_REQUEST_REJECTED),
        (422, PROVIDER_FAULT_REQUEST_REJECTED),
    ],
)
def test_terminal_statuses_map_to_their_faults_and_are_not_retried(
    status: int, fault: str
) -> None:
    transport = RecordingTransport([(status, error_body())])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.client()).next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 1


@pytest.mark.parametrize(
    ("status", "fault"),
    [
        (429, PROVIDER_FAULT_RATE_LIMITED),
        (500, PROVIDER_FAULT_SERVER_ERROR),
        (503, PROVIDER_FAULT_SERVER_ERROR),
        (408, PROVIDER_FAULT_TIMEOUT),
    ],
)
def test_retryable_statuses_are_retried_to_the_policy_bound(
    status: int, fault: str
) -> None:
    transport = RecordingTransport([(status, error_body())] * 3)
    adapter = MistralChatAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 3, "the adapter owns retries; the SDK's are disabled"


def test_a_dropped_connection_is_a_network_fault() -> None:
    transport = RecordingTransport([httpx.ConnectError("refused")] * 3)
    adapter = MistralChatAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_NETWORK_ERROR


def test_a_read_timeout_is_a_timeout_fault() -> None:
    transport = RecordingTransport([httpx.ReadTimeout("slow")] * 3)
    adapter = MistralChatAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_TIMEOUT


def test_a_retry_that_succeeds_reports_both_attempts() -> None:
    transport = RecordingTransport(
        [
            (429, error_body()),
            chat_body([tool_call("read_records", "{}")], model=PINNED_MODEL),
        ]
    )
    adapter = MistralChatAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    assert adapter.next_call(_turn_request(), _deadline()).action == "read_records"
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 2
    assert telemetry.fault_counts == {PROVIDER_FAULT_RATE_LIMITED: 1}
    assert telemetry.terminal_reason == "response"


def test_no_provider_text_reaches_a_durable_failure_detail() -> None:
    """This SDK builds its exception message out of the response body.

    Which makes it exactly the case the redaction rule exists for: the message
    in hand at the point the fault is raised contains the provider's own text
    verbatim, and none of it may reach a ledger row.
    """
    secret = "PROVIDER-PROSE-THAT-MUST-NOT-BE-DURABLE"
    transport = RecordingTransport([(401, error_body(message=secret))])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.client()).next_call(_turn_request(), _deadline())
    assert secret not in str(caught.value)


def test_the_credential_never_appears_in_the_repr_or_the_settings() -> None:
    _transport, client = _one_call()
    adapter = _adapter(client)
    assert FAKE_API_KEY not in repr(adapter)
    assert FAKE_API_KEY not in json.dumps(dict(adapter.settings))


# -- configuration refusals ---------------------------------------------------


def test_a_missing_credential_refuses_before_a_client_exists() -> None:
    with pytest.raises(MistralConfigurationError) as caught:
        build_mistral_adapter(
            model=MISTRAL_SMALL_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={},
        )
    assert "MISTRAL_API_KEY" in str(caught.value)


@pytest.mark.parametrize("variable", ["MISTRAL_BASE_URL", "MISTRAL_SERVER_URL"])
def test_an_environment_that_would_redirect_the_request_is_refused(variable: str) -> None:
    with pytest.raises(MistralConfigurationError) as caught:
        build_mistral_adapter(
            model=MISTRAL_SMALL_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={variable: "https://elsewhere.invalid", "MISTRAL_API_KEY": "k"},
        )
    assert variable in str(caught.value)


def test_an_unauthorised_configuration_is_refused_before_the_credential_is_read() -> None:
    with pytest.raises(MistralConfigurationError):
        build_mistral_adapter(
            model=MISTRAL_SMALL_MODEL,
            configuration_id="b" * 64,
            audited={"a" * 64},
            environ={"MISTRAL_API_KEY": "k"},
        )


def test_a_client_with_its_own_retry_loop_is_refused() -> None:
    """The adapter owns the retries, and this SDK carries a whole strategy object.

    The other two integrations state that as ``max_retries``; here it is a
    ``RetryConfig``, so the check is that the strategy is the one that does
    nothing. A second, hidden loop underneath this one would spend the episode's
    wall-clock budget on calls the runner never saw and hide a rate limit the run
    should have recorded.
    """
    from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

    transport = RecordingTransport([])
    client = Mistral(
        api_key=FAKE_API_KEY,
        server_url=MISTRAL_BASE_URL,
        client=httpx.Client(transport=httpx.MockTransport(transport.handler)),
        retry_config=RetryConfig("backoff", BackoffStrategy(1, 1, 1, 1), True),
    )
    with pytest.raises(MistralConfigurationError) as caught:
        MistralChatAdapter(model=PINNED_MODEL, client=client)
    assert "retry" in str(caught.value).lower()
    assert transport.calls == 0


def test_the_build_pre_authorises_nothing() -> None:
    with pytest.raises(MistralConfigurationError):
        build_mistral_adapter(
            model=MISTRAL_SMALL_MODEL,
            configuration_id="a" * 64,
            environ={"MISTRAL_API_KEY": "k"},
        )


# -- identity and settings ----------------------------------------------------


def test_identity_names_the_service_the_model_and_this_integration() -> None:
    identity = mistral_identity(MISTRAL_SMALL_MODEL)
    assert identity.provider == "mistral"
    assert identity.model == MISTRAL_SMALL_MODEL
    assert identity.implementation == "mistral_chat_completions"


def test_settings_state_every_result_affecting_request_setting() -> None:
    from importlib.metadata import version

    settings = mistral_settings(model=MISTRAL_SMALL_MODEL)
    assert settings["api"] == "chat_completions"
    assert settings["base_url"] == MISTRAL_BASE_URL
    assert settings["sdk"] == "mistralai"
    assert settings["sdk_version"] == version("mistralai")
    assert settings["sdk_max_retries"] == 0
    assert settings["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert settings["tool_choice"] == "required"
    assert settings["parallel_tool_calls"] is False
    assert settings["response_contract"] == "exactly_one_tool_call"
    assert settings["retry"]["max_attempts"] == 3
