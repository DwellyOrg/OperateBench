"""The OpenAI Responses adapter.

Every test here runs the real SDK over an in-process transport (see
``tests.openai_transport``). No test contacts a provider, and none of them is
evidence about a model: they are evidence about the integration.
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
from boundarybench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENAI_BASE_URL,
    OpenAIConfigurationError,
    OpenAIResponsesAdapter,
    build_openai_adapter,
    build_responses_request,
    openai_identity,
    openai_settings,
    request_profile_for,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.conftest import minimal_card
from tests.openai_transport import (
    FAKE_API_KEY,
    RecordingTransport,
    error_body,
    function_call_item,
    message_item,
    responses_body,
    scripted_client,
)

PINNED_MODEL = "gpt-test-20990101"


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
) -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(model=model, client=client)


def _one_call(action: str = "read_records", arguments: str = "{}", **kwargs: object):
    return scripted_client(
        responses_body(
            [function_call_item(action, arguments)], model=PINNED_MODEL, **kwargs
        )
    )


# -- what goes on the wire ----------------------------------------------------


def test_the_installed_sdk_serializes_exactly_the_body_the_adapter_built() -> None:
    """The real SDK's wire body *is* the mapping this build produced.

    Asserted as an equality against ``build_responses_request`` rather than as a
    list of expected keys. A body test that checks for known fields passes for
    every field nobody thought to check for; requiring the two to be the same
    object means neither the SDK nor a future edit can add, drop or rename one
    without this failing.
    """
    transport, client = _one_call()
    request = _turn_request()
    _adapter(client).next_call(request, _deadline())
    assert transport.calls == 1
    assert transport.bodies[0] == build_responses_request(request, model=PINNED_MODEL)


def test_the_body_carries_the_pinned_settings_and_the_required_omissions() -> None:
    """Every result-affecting field is on the wire, and the omitted ones are absent.

    ``store: false`` is checked as hard as the rest. A run that let the provider
    retain its prompts server-side would be a different experiment from one that
    did not, whatever the answers came back as.
    """
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    body = transport.bodies[0]
    assert body["model"] == PINNED_MODEL
    assert body["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["store"] is False
    profile = request_profile_for(PINNED_MODEL)
    assert tuple(sorted(body)) == profile.sent_fields()
    for field in profile.omitted_fields():
        assert field not in body


def test_the_tool_schema_is_the_flat_responses_shape_with_the_query_enum() -> None:
    """Responses tools are flat, and the acquisition argument carries its enum.

    A nested ``{"function": {...}}`` tool is the Chat Completions shape and this
    API rejects it, so the shape is part of what the request mapping means.
    """
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    tools = transport.bodies[0]["tools"]
    assert tools, "the case offers actions, so the body must declare tools"
    for tool in tools:
        assert tool["type"] == "function"
        # Flat: no "function" wrapper.
        assert "function" not in tool
        assert set(tool) == {"type", "name", "description", "parameters"}
        assert tool["parameters"]["additionalProperties"] is False


def test_the_request_shows_nothing_the_turn_request_does_not_carry() -> None:
    """The grading state is not reachable from here and must not appear.

    The variant's expected disposition, expected reason and content digest are
    not fields of ``TurnRequest``, so an equality against the projection already
    proves they cannot be present. This states it directly as well, because it
    is the property the whole boundary exists for.
    """
    variant = _variant()
    transport, client = _one_call()
    _adapter(client).next_call(_turn_request(), _deadline())
    wire = json.dumps(transport.bodies[0], sort_keys=True)
    # The identifiers, which are unique to the compiled variant and appear
    # nowhere a model is entitled to see.
    assert variant.variant_id not in wire
    assert variant.content_digest not in wire
    # The grading *fields*, by name. Checked as names rather than as values on
    # purpose: ``ACT`` is also the public disposition vocabulary the scaffold
    # offers, so asserting the value's absence would be asserting that the model
    # is not told what it may answer. What must never appear is the structure
    # that says which answer is right.
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


def test_the_client_is_pinned_to_the_documented_endpoint() -> None:
    """The base URL is where the credential goes, so it is not configurable."""
    assert OPENAI_BASE_URL == "https://api.openai.com/v1"


# -- the answer ---------------------------------------------------------------


def test_one_function_call_becomes_one_adapter_call() -> None:
    _transport, client = _one_call("ask_user", '{"fact": "boiler_age_years"}')
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.action == "ask_user"
    assert call.arguments == {"fact": "boiler_age_years"}


def test_prose_beside_the_call_is_ignored_and_never_recorded() -> None:
    """A message item is ordinary commentary; the action is still the action."""
    _transport, client = scripted_client(
        responses_body(
            [
                message_item("here is my reasoning, at length"),
                function_call_item("read_records", "{}"),
            ],
            model=PINNED_MODEL,
        )
    )
    call = _adapter(client).next_call(_turn_request(), _deadline())
    assert call.action == "read_records"


@pytest.mark.parametrize(
    "output",
    [
        pytest.param([], id="no-call"),
        pytest.param(
            [
                function_call_item("read_records", "{}", call_id="a"),
                function_call_item("ask_user", '{"fact": "x"}', call_id="b"),
            ],
            id="two-calls",
        ),
    ],
)
def test_anything_other_than_exactly_one_call_is_a_protocol_failure(output: list) -> None:
    _transport, client = scripted_client(responses_body(output, model=PINNED_MODEL))
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_an_unknown_output_item_type_is_refused_rather_than_read_as_commentary() -> None:
    """Content this build cannot interpret is not silently skipped.

    Reading past an item whose type this scaffold never asked for would mean
    guessing which part of the answer was the action.
    """
    _transport, client = scripted_client(
        responses_body(
            [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                function_call_item("read_records", "{}"),
            ],
            model=PINNED_MODEL,
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
        pytest.param('"a string"', id="scalar"),
    ],
)
def test_arguments_that_are_not_a_json_object_are_refused(arguments: str) -> None:
    """``arguments`` arrives as a *string* on this API and has to be parsed.

    Which means it can fail to be JSON at all, or be JSON that is not an object
    — two failure modes the Messages API's pre-parsed input never had.
    """
    _transport, client = _one_call("read_records", arguments)
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_a_lone_surrogate_in_the_arguments_is_refused() -> None:
    """A value this build cannot record in a ledger row is not accepted as one.

    ``\\ud800`` survives ``json.loads`` as a Python string and then cannot be
    encoded back to UTF-8, so a row carrying it could never be written or read.
    """
    _transport, client = _one_call("ask_user", '{"fact": "\\ud800"}')
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


def test_termination_at_the_output_ceiling_is_its_own_failure() -> None:
    """Truncation is the run's own limit, not the model failing a schema."""
    _transport, client = scripted_client(
        responses_body(
            [function_call_item("read_records", "{}")],
            status="incomplete",
            incomplete_reason="max_output_tokens",
            model=PINNED_MODEL,
        )
    )
    with pytest.raises(AdapterOutputLimitError) as caught:
        _adapter(client).next_call(_turn_request(), _deadline())
    assert str(MAX_OUTPUT_TOKENS) in str(caught.value)


def test_an_incomplete_response_for_another_reason_is_a_protocol_failure() -> None:
    """Cut short for a reason this scaffold does not model: not the action made."""
    _transport, client = scripted_client(
        responses_body(
            [function_call_item("read_records", "{}")],
            status="incomplete",
            incomplete_reason="content_filter",
            model=PINNED_MODEL,
        )
    )
    with pytest.raises(AdapterProtocolError):
        _adapter(client).next_call(_turn_request(), _deadline())


# -- provenance ---------------------------------------------------------------


def test_a_response_from_another_model_is_refused_before_the_action_or_usage() -> None:
    """The provider's own statement of who answered is checked, and it decides.

    Classified as a response-invalid provider fault rather than a protocol
    failure: nothing about the model's *behaviour* is wrong — the call may be
    flawless — so filing it under the model would put a service-side provenance
    fault in the bucket that means "the model answered badly".
    """
    _transport, client = scripted_client(
        responses_body(
            [function_call_item("read_records", "{}")], model="gpt-something-else"
        )
    )
    adapter = _adapter(client)
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    # The counts are not banked: they belong to a model this run did not pin.
    assert adapter.last_usage().input_tokens is None
    assert adapter.last_usage().output_tokens is None


def test_usage_is_reported_even_when_the_answer_fails_the_protocol() -> None:
    """A response that arrived was paid for, so it is measured either way."""
    _transport, client = _one_call("drop_database", "{}")
    adapter = _adapter(client)
    with pytest.raises(AdapterProtocolError):
        adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137
    assert adapter.last_usage().output_tokens == 29


def test_the_measurement_is_cleared_before_anything_else_the_turn_does() -> None:
    """A measurement that outlived the call it measured would be stale evidence."""
    transport = RecordingTransport(
        [
            responses_body(
                [function_call_item("read_records", "{}")], model=PINNED_MODEL
            ),
            httpx.ConnectError("down"),
        ]
    )
    adapter = _adapter(transport.client())
    adapter.next_call(_turn_request(), _deadline())
    assert adapter.last_usage().input_tokens == 137
    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline(seconds=0.001))
    assert adapter.last_usage().input_tokens is None


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
def test_terminal_statuses_map_to_faults_and_are_not_retried(
    status: int, fault: str
) -> None:
    transport = RecordingTransport([(status, error_body())])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.client()).next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 1, "a terminal fault is not retried"


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
    adapter = OpenAIResponsesAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == fault
    assert transport.calls == 3, "the bounded policy allows exactly three attempts"


def test_a_dropped_connection_is_a_network_fault() -> None:
    transport = RecordingTransport([httpx.ConnectError("refused")] * 3)
    adapter = OpenAIResponsesAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_NETWORK_ERROR


def test_a_retry_that_succeeds_reports_both_attempts() -> None:
    transport = RecordingTransport(
        [
            (429, error_body()),
            responses_body(
                [function_call_item("read_records", "{}")], model=PINNED_MODEL
            ),
        ]
    )
    adapter = OpenAIResponsesAdapter(
        model=PINNED_MODEL, client=transport.client(), sleep=lambda _s: None
    )
    call = adapter.next_call(_turn_request(), _deadline())
    assert call.action == "read_records"
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 2
    assert telemetry.fault_counts == {PROVIDER_FAULT_RATE_LIMITED: 1}
    assert telemetry.terminal_reason == "response"


# -- what may never become durable -------------------------------------------


def test_no_provider_text_reaches_a_durable_failure_detail() -> None:
    """Vendor- and attacker-controlled text does not enter a ledger row.

    The provider's message, the response body and the request id are all in hand
    at the point the fault is raised, and all deliberately dropped.
    """
    secret = "PROVIDER-PROSE-THAT-MUST-NOT-BE-DURABLE"
    transport = RecordingTransport([(401, error_body(message=secret))])
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(transport.client()).next_call(_turn_request(), _deadline())
    assert secret not in str(caught.value)


def test_no_model_string_from_the_response_reaches_the_mismatch_detail() -> None:
    """Neither identifier is quoted: one is provider text, the other is on every row."""
    impostor = "gpt-IMPOSTOR-MODEL-NAME"
    _transport, client = scripted_client(
        responses_body([function_call_item("read_records", "{}")], model=impostor)
    )
    with pytest.raises(AdapterProviderError) as caught:
        _adapter(client).next_call(_turn_request(), _deadline())
    assert impostor not in str(caught.value)


def test_the_credential_never_appears_in_the_repr_or_the_settings() -> None:
    _transport, client = _one_call()
    adapter = _adapter(client)
    assert FAKE_API_KEY not in repr(adapter)
    assert FAKE_API_KEY not in json.dumps(dict(adapter.settings))


# -- configuration refusals ---------------------------------------------------


def test_a_missing_credential_refuses_before_a_client_exists() -> None:
    with pytest.raises(OpenAIConfigurationError) as caught:
        build_openai_adapter(
            model=GPT_5_6_LUNA_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={},
        )
    assert "OPENAI_API_KEY" in str(caught.value)


@pytest.mark.parametrize("variable", ["OPENAI_BASE_URL"])
def test_an_environment_that_would_redirect_the_request_is_refused(variable: str) -> None:
    """An operator who set one meant it to take effect; ignoring it is worse.

    The refusal happens before the credential is read, so a redirected
    environment never handles the key at all.
    """
    with pytest.raises(OpenAIConfigurationError) as caught:
        build_openai_adapter(
            model=GPT_5_6_LUNA_MODEL,
            configuration_id="a" * 64,
            audited={"a" * 64},
            environ={variable: "https://elsewhere.invalid", "OPENAI_API_KEY": "k"},
        )
    assert variable in str(caught.value)


def test_an_unauthorised_configuration_is_refused_before_the_credential_is_read() -> None:
    """Order is load-bearing: nothing unauthorised ever handles the key."""
    with pytest.raises(OpenAIConfigurationError):
        build_openai_adapter(
            model=GPT_5_6_LUNA_MODEL,
            configuration_id="b" * 64,
            audited={"a" * 64},
            environ={"OPENAI_API_KEY": "k"},
        )


def test_the_build_pre_authorises_nothing() -> None:
    with pytest.raises(OpenAIConfigurationError):
        build_openai_adapter(
            model=GPT_5_6_LUNA_MODEL,
            configuration_id="a" * 64,
            environ={"OPENAI_API_KEY": "k"},
        )


def test_a_client_with_its_own_retry_loop_is_refused() -> None:
    """A hidden retry underneath this one would spend budget the runner never saw."""
    transport = RecordingTransport([])
    with pytest.raises(OpenAIConfigurationError):
        OpenAIResponsesAdapter(model=PINNED_MODEL, client=transport.client(max_retries=2))


# -- identity and settings ----------------------------------------------------


def test_identity_names_the_service_the_model_and_this_integration() -> None:
    identity = openai_identity(GPT_5_6_LUNA_MODEL)
    assert identity.provider == "openai"
    assert identity.model == GPT_5_6_LUNA_MODEL
    assert identity.implementation == "openai_responses"


def test_settings_state_every_result_affecting_request_setting() -> None:
    """The settings are hashed into run identity, so they are the answer to
    "what exactly was asked of the provider"."""
    settings = openai_settings(model=GPT_5_6_LUNA_MODEL)
    assert settings["api"] == "responses"
    assert settings["base_url"] == OPENAI_BASE_URL
    assert settings["sdk"] == "openai"
    assert settings["sdk_version"] == openai.__version__
    assert settings["sdk_max_retries"] == 0
    assert settings["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert settings["tool_choice"] == "required"
    assert settings["parallel_tool_calls"] is False
    assert settings["store"] is False
    assert settings["response_contract"] == "exactly_one_tool_call"
    assert "request_fields_sent" in settings and "request_fields_omitted" in settings
    assert settings["retry"]["max_attempts"] == 3


def test_a_body_the_settings_do_not_describe_is_not_dispatched() -> None:
    """The settings are this run's durable claim about what was asked."""
    from boundarybench.adapter import AdapterError

    transport = RecordingTransport([])
    adapter = _adapter(transport.client())
    profile = request_profile_for(PINNED_MODEL)
    payload = dict(build_responses_request(_turn_request(), model=PINNED_MODEL))
    payload["top_p"] = 0.9
    with pytest.raises(AdapterError):
        adapter._send_payload(payload, _deadline())
    assert transport.calls == 0, "a body the settings do not describe is not sent"
    assert "top_p" in profile.omitted_fields()
