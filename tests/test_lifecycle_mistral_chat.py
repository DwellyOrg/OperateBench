"""Lifecycle Mistral Chat Completions vertical, entirely offline."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import MethodType
from typing import Any, cast

import httpx
import pytest
from mistralai.client import Mistral
from mistralai.client._hooks.sdkhooks import SDKHooks
from mistralai.client.basesdk import BaseSDK
from mistralai.client.chat import Chat
from mistralai.client.sdkconfiguration import SDKConfiguration
from mistralai.client.utils.retries import RetryConfig

from operatebench.agents import mistral_chat as mistral_agent
from operatebench.agents.evidence import (
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    normalized_response_digest,
    provider_identity_for,
)
from operatebench.agents.lifecycle_contract import (
    ModelRequestRefused,
    UndecodableArguments,
)
from operatebench.agents.mistral_chat import (
    MistralChatCompletionsTransport,
    build_model_payload,
    mistral_outcome_tools,
)
from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    TRUNCATED_STOP_REASON,
    ModelAgent,
)
from operatebench.agents.pricing import LifecycleCostGuard, LifecyclePricingPolicy
from operatebench.agents.transport import ModelRequest, ToolCall
from operatebench.core.protocol import AgentObservation
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers import mistral_chat as mistral_provider
from operatebench.providers.faults import AdapterBusyError, AdapterProviderError
from operatebench.providers.mistral_chat import (
    MISTRAL_BASE_URL,
    MISTRAL_SMALL_MODEL,
    CapturedHttpClient,
    MistralConfigurationError,
    build_client,
    ensure_capturing_client,
)
from operatebench.providers.wire import (
    MAX_PROVIDER_RESPONSE_ID_CHARACTERS,
    MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES,
)
from tests.execution_ledger_fixtures import DIGEST_A, controls


def observation() -> AgentObservation:
    return AgentObservation(
        now="2025-01-01T09:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_0123456789abcdef0123456789abcdef",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=[],
        last_rejection=None,
    )


def model_request() -> ModelRequest:
    class Never:
        def send(self, request: ModelRequest) -> Any:
            raise AssertionError("request construction must not dispatch")

    return ModelAgent(
        Never(), model=MISTRAL_SMALL_MODEL, agent_id="diagnostic"
    ).build_request(observation())


def response_body(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "cmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": MISTRAL_SMALL_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "abcdefghi",
                            "type": "function",
                            "function": {
                                "name": "complete",
                                "arguments": json.dumps(
                                    {"reason": "done", "evidence_refs": []}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 137,
            "completion_tokens": 29,
            "total_tokens": 166,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    body.update(changes)
    return body


class Wire:
    def __init__(
        self, body: Any, *, response_headers: dict[str, str] | None = None
    ) -> None:
        self.body = body
        self.response_headers = response_headers
        self.requests: list[httpx.Request] = []
        self.http_client = httpx.Client(transport=httpx.MockTransport(self.handler))
        self.client: Mistral = build_client(
            api_key="offline-placeholder-not-a-credential",
            http_client=self.http_client,
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if callable(self.body):
            handler = cast(Callable[[httpx.Request], httpx.Response], self.body)
            return handler(request)
        if isinstance(self.body, BaseException):
            raise self.body
        if isinstance(self.body, tuple):
            status, body = self.body
            return httpx.Response(status, json=body, request=request)
        if isinstance(self.body, bytes):
            return httpx.Response(200, content=self.body, request=request)
        return httpx.Response(
            200, json=self.body, headers=self.response_headers, request=request
        )


def transport(
    wire: Wire,
    *,
    cost_guard: LifecycleCostGuard | None = None,
    before_dispatch: Any = None,
) -> MistralChatCompletionsTransport:
    return MistralChatCompletionsTransport(
        model=MISTRAL_SMALL_MODEL,
        client=wire.client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        cost_guard=cost_guard,
        before_dispatch=before_dispatch,
    )


def test_dispatch_observer_runs_at_final_sdk_boundary() -> None:
    wire = Wire(response_body())
    events: list[str] = []

    def observer() -> None:
        assert wire.requests == []
        events.append("observer")

    lane = transport(wire, before_dispatch=observer)

    lane.send(model_request())

    assert events == ["observer"]
    assert len(wire.requests) == 1


def test_dispatch_observer_counts_transport_failure_but_not_preflight_refusal() -> None:
    events: list[str] = []
    failing = transport(
        Wire(httpx.ConnectError("offline failure")),
        before_dispatch=lambda: events.append("dispatch"),
    )
    with pytest.raises(AdapterProviderError):
        failing.send(model_request())
    assert events == ["dispatch"]

    wire = Wire(response_body())
    refused = transport(wire, before_dispatch=lambda: events.append("unexpected"))
    with pytest.raises(ModelRequestRefused):
        refused.send(replace(model_request(), max_output_tokens=2048))
    assert events == ["dispatch"]


CLIENT_STATE_MUTATIONS = (
    "endpoint",
    "retry_strategy",
    "retry_backoff",
    "retry_connection_errors",
    "headers",
    "query",
    "auth",
    "cookies",
    "request_hook",
    "response_hook",
    "sdk_hook",
    "sdk_hook_state",
)


def mutate_client_state(wire: Wire, mutation: str) -> None:
    configuration = wire.client.sdk_configuration
    if mutation == "endpoint":
        configuration.server_url = "https://redirect.invalid"
    elif mutation == "retry_strategy":
        configuration.retry_config.strategy = "backoff"
    elif mutation == "retry_backoff":
        configuration.retry_config.backoff.max_elapsed_time = 2
    elif mutation == "retry_connection_errors":
        configuration.retry_config.retry_connection_errors = True
    elif mutation == "headers":
        wire.http_client.headers["x-route"] = "secret"
    elif mutation == "query":
        wire.http_client.params = {"tenant": "secret"}
    elif mutation == "auth":
        wire.http_client.auth = ("secret", "secret")
    elif mutation == "cookies":
        wire.http_client.cookies.set("session", "secret")
    elif mutation == "request_hook":
        wire.http_client.event_hooks["request"].append(lambda request: None)
    elif mutation == "response_hook":
        wire.http_client.event_hooks["response"].append(lambda response: None)
    elif mutation == "sdk_hook":
        configuration._hooks.before_request_hooks.append(object())
    else:
        tracing = next(
            hook
            for hook in configuration._hooks.before_request_hooks
            if type(hook).__name__ == "TracingHook"
        )
        tracing.tracing_enabled = True


EXECUTION_SURFACE_MUTATIONS = (
    "chat_complete",
    "inner_build_request",
    "inner_request",
    "inner_send",
    "capture_build_request",
    "capture_send",
    "configuration_get_server_details",
    "sdk_hooks_before_request",
    "hook_before_request",
    "hook_after_success",
    "hook_after_error",
)


def mutate_execution_surface(wire: Wire, mutation: str) -> None:
    configuration = wire.client.sdk_configuration
    capture = configuration.client
    if mutation == "chat_complete":
        target, name = wire.client.chat, "complete"
    elif mutation.startswith("inner_"):
        target, name = wire.http_client, mutation.removeprefix("inner_")
    elif mutation.startswith("capture_"):
        assert type(capture) is CapturedHttpClient
        target, name = capture, mutation.removeprefix("capture_")
    elif mutation == "configuration_get_server_details":
        target, name = configuration, "get_server_details"
    elif mutation == "sdk_hooks_before_request":
        target, name = configuration._hooks, "before_request"
    elif mutation == "hook_before_request":
        target = configuration._hooks.before_request_hooks[0]
        name = "before_request"
    elif mutation == "hook_after_success":
        target = configuration._hooks.after_success_hooks[0]
        name = "after_success"
    else:
        target = configuration._hooks.after_error_hooks[0]
        name = "after_error"

    def substituted(_self: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("substituted execution surface ran")

    setattr(target, name, MethodType(substituted, target))


def test_real_mistral_sdk_serializes_exact_native_request() -> None:
    wire = Wire(response_body())
    request = model_request()
    lane = transport(wire)

    answer = lane.send(request)

    assert answer.tool_calls == (
        ToolCall("complete", {"reason": "done", "evidence_refs": []}),
    )
    assert len(wire.requests) == 1
    sent = json.loads(wire.requests[0].content)
    expected = build_model_payload(request, model=MISTRAL_SMALL_MODEL)
    assert sent == {**expected, "stream": False}
    assert wire.requests[0].url == f"{MISTRAL_BASE_URL}/v1/chat/completions"
    assert sent["max_tokens"] == MAX_OUTPUT_TOKENS == 4096
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    assert sent["temperature"] == 0.0
    assert "top_p" not in sent and "random_seed" not in sent
    assert [tool["function"] for tool in sent["tools"]] == mistral_outcome_tools()
    assert wire.requests[0].headers["authorization"].startswith("Bearer ")
    assert lane.last_usage().input_tokens == 137
    assert lane.last_usage().output_tokens == answer.output_tokens == 29
    assert lane.provider == "mistral"
    assert lane.api == "chat_completions"
    assert lane.settings["sdk"] == "mistralai"
    assert lane.settings["sdk_max_retries"] == 0
    assert lane.settings["sdk_injected_fields"] == {"stream": False}
    assert (
        lane.settings["response_state_policy"] == "clear_owned_httpx_response_cookies_v1"
    )


def test_response_cookies_are_cleared_before_acceptance_and_never_replayed() -> None:
    body = response_body()
    body["choices"][0]["message"]["tool_calls"][0]["function"] = {
        "name": "retrieve",
        "arguments": json.dumps(
            {
                "requests": [
                    {"tool": "get_case_record", "arguments": {}},
                    {"tool": "list_checkpoints", "arguments": {}},
                    {"tool": "list_quotes", "arguments": {}},
                ]
            }
        ),
    }
    wire = Wire(body, response_headers={"set-cookie": "session=provider-secret"})
    lane = transport(wire)

    first = lane.send(model_request())
    assert first.tool_calls[0].name == "retrieve"
    assert not wire.http_client.cookies

    lane.send(model_request())

    assert len(wire.requests) == 2
    assert all("cookie" not in request.headers for request in wire.requests)
    assert not wire.http_client.cookies


@pytest.mark.parametrize(
    "mutation", ("headers", "query", "auth", "request_hook", "response_hook")
)
def test_non_cookie_response_induced_client_state_is_refused_on_that_response(
    mutation: str,
) -> None:
    wire: Wire

    def response(request: httpx.Request) -> httpx.Response:
        mutate_client_state(wire, mutation)
        return httpx.Response(200, json=response_body(), request=request)

    wire = Wire(response)
    lane = transport(wire)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())

    assert raised.value.fault == "provider_response_invalid"
    assert len(wire.requests) == 1
    attempt = lane.last_telemetry().attempts[0]
    assert attempt.response_received is True


def test_replacement_cookie_jar_after_response_is_refused_not_normalized() -> None:
    wire: Wire

    def response(request: httpx.Request) -> httpx.Response:
        wire.http_client._cookies = httpx.Cookies()
        return httpx.Response(200, json=response_body(), request=request)

    wire = Wire(response)
    lane = transport(wire)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())

    assert raised.value.fault == "provider_response_invalid"
    assert lane.last_telemetry().attempts[0].response_received is True


def test_response_cookie_clear_failure_is_a_received_response_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire: Wire

    def response(request: httpx.Request) -> httpx.Response:
        jar = wire.http_client.cookies.jar

        def fail_clear(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("provider-secret")

        monkeypatch.setattr(jar, "clear", fail_clear)
        return httpx.Response(
            200,
            json=response_body(),
            headers={"set-cookie": "session=provider-secret"},
            request=request,
        )

    wire = Wire(response)
    lane = transport(wire)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())

    assert raised.value.fault == "provider_response_invalid"
    assert "provider-secret" not in str(raised.value)
    assert lane.last_telemetry().attempts[0].response_received is True


@pytest.mark.parametrize("mutation", CLIENT_STATE_MUTATIONS)
def test_client_state_mutation_refuses_at_turn_entry_without_dispatch(
    mutation: str,
) -> None:
    wire = Wire(response_body())
    guard = LifecycleCostGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-entry-admission-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    mutate_client_state(wire, mutation)

    with pytest.raises(MistralConfigurationError) as raised:
        lane.send(model_request())

    assert "secret" not in str(raised.value)
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == Decimal(0)
    assert guard.committed_usd == Decimal(0)


@pytest.mark.parametrize("mutation", CLIENT_STATE_MUTATIONS)
def test_client_state_mismatch_refuses_at_construction(mutation: str) -> None:
    wire = Wire(response_body())
    mutate_client_state(wire, mutation)

    with pytest.raises(MistralConfigurationError) as raised:
        transport(wire)

    assert "secret" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize("mutation", CLIENT_STATE_MUTATIONS)
def test_final_pre_dispatch_mutation_cancels_reservation_without_attempt(
    mutation: str,
) -> None:
    wire = Wire(response_body())

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            mutate_client_state(wire, mutation)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-final-admission-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)
    assert guard.reserved_tokens == 0


@pytest.mark.parametrize("ceiling", [2048, 4096.0, True])
def test_noncanonical_output_ceiling_refuses_before_dispatch(ceiling: Any) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    with pytest.raises(ModelRequestRefused, match="output-token ceiling"):
        lane.send(replace(model_request(), max_output_tokens=ceiling))
    assert wire.requests == []


@pytest.mark.parametrize(
    "body",
    [
        response_body(unexpected="provider secret"),
        response_body(model="wrong-model"),
        response_body(choices=[]),
        response_body(choices=[response_body()["choices"][0]] * 2),
        response_body(
            usage={"prompt_tokens": True, "completion_tokens": 29, "total_tokens": 30}
        ),
        response_body(
            usage={"prompt_tokens": "137", "completion_tokens": 29, "total_tokens": 166}
        ),
        response_body(
            usage={"prompt_tokens": 137.0, "completion_tokens": 29, "total_tokens": 166}
        ),
        response_body(
            usage={"prompt_tokens": -1, "completion_tokens": 29, "total_tokens": 28}
        ),
        response_body(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "provider secret"},
                }
            ]
        ),
    ],
)
def test_raw_typed_identity_cardinality_usage_and_prose_fail_closed(
    body: dict[str, Any],
) -> None:
    wire = Wire(body)
    lane = transport(wire)
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    assert raised.value.fault == "provider_response_invalid"
    assert "provider secret" not in str(raised.value)
    assert lane.last_usage().input_tokens is None


@pytest.mark.parametrize(
    "details",
    [
        {},
        {"cached_tokens": True},
        {"cached_tokens": -1},
        {"cached_tokens": 0, "other": 1},
    ],
)
def test_response_extension_is_closed_exact_and_unsettled_on_failure(
    details: Any,
) -> None:
    body = response_body()
    body["usage"]["prompt_tokens_details"] = details
    lane = transport(Wire(body))
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    assert raised.value.fault == "provider_response_invalid"
    assert lane.last_usage().input_tokens is None


def test_undecodable_arguments_are_distinct_from_valid_empty_input() -> None:
    invalid = response_body()
    invalid["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        "not-json-secret"
    )
    empty = response_body()
    empty["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{}"
    invalid_answer = transport(Wire(invalid)).send(model_request())
    empty_answer = transport(Wire(empty)).send(model_request())
    assert isinstance(invalid_answer.tool_calls[0].arguments, UndecodableArguments)
    assert empty_answer.tool_calls[0].arguments == {}
    assert normalized_response_digest(
        model=invalid_answer.model,
        stop_classification=invalid_answer.stop_reason,
        tool_calls=invalid_answer.tool_calls,
    ) != normalized_response_digest(
        model=empty_answer.model,
        stop_classification=empty_answer.stop_reason,
        tool_calls=empty_answer.tool_calls,
    )


def test_output_limit_uses_canonical_truncation_and_settles_usage() -> None:
    body = response_body(
        choices=[{"index": 0, "finish_reason": "length", "message": None}]
    )
    lane = transport(Wire(body))
    answer = lane.send(model_request())
    assert answer.stop_reason == TRUNCATED_STOP_REASON == "max_tokens"
    assert answer.tool_calls == ()
    assert lane.last_usage().input_tokens == 137


@pytest.mark.parametrize(
    "case",
    [
        "maximum",
        "over",
        "utf8_maximum",
        "utf8_over",
        "lone_surrogate",
        "empty",
        "absent",
        "type",
    ],
)
def test_response_id_policy_precedes_usage_settlement(case: str) -> None:
    body = response_body()
    if case == "maximum":
        body["id"] = "a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS
    elif case == "over":
        body["id"] = "a" * (MAX_PROVIDER_RESPONSE_ID_CHARACTERS + 1)
    elif case == "utf8_maximum":
        body["id"] = "😀" * (MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES // 4)
    elif case == "utf8_over":
        body["id"] = "😀" * (MAX_PROVIDER_RESPONSE_ID_UTF8_BYTES // 4) + "a"
    elif case == "lone_surrogate":
        body["id"] = "\ud800"
    elif case == "empty":
        body["id"] = ""
    elif case == "absent":
        body.pop("id")
    else:
        body["id"] = 7
    wire_body: Any = body
    if case == "lone_surrogate":
        wire_body = json.dumps(body, ensure_ascii=True).encode("utf-8")
    lane = transport(Wire(wire_body))
    if case in {"maximum", "utf8_maximum", "empty"}:
        lane.send(model_request())
    else:
        with pytest.raises(AdapterProviderError):
            lane.send(model_request())
        assert lane.last_usage().input_tokens is None


@pytest.mark.parametrize(
    ("body", "fault"),
    [
        (httpx.ReadTimeout("offline timeout"), "provider_timeout"),
        (httpx.ConnectError("offline network"), "provider_network_error"),
        ((429, {"message": "secret"}), "provider_rate_limited"),
        (b"not-json-secret", "provider_response_invalid"),
    ],
)
def test_sdk_timeout_network_http_and_malformed_wire_are_classified(
    body: Any, fault: str
) -> None:
    lane = transport(Wire(body))
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    assert raised.value.fault == fault
    assert lane.last_usage().input_tokens is None


def test_mistral_ledger_identity_and_response_id_digest(tmp_path: Path) -> None:
    lane = transport(Wire(response_body(id="cmpl_test")))
    path = tmp_path / "mistral.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="mistral-small-2603",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)
    lane.send(model_request())
    recorder.abandon_open_call()
    attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert attempt.response_id_digest_sha256 is not None
    assert attempt.input_tokens == 137
    assert attempt.response_model == MISTRAL_SMALL_MODEL
    assert provider_identity_for(lane).implementation == "mistral_chat_completions"


def test_same_named_sdk_hook_is_refused_before_authorization_or_dispatch() -> None:
    wire = Wire(response_body())
    guard = LifecycleCostGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-hook-identity-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    class CustomUserAgentHook:
        def before_request(self, hook_ctx: Any, request: httpx.Request) -> httpx.Request:
            request.headers["x-spoofed-hook"] = "secret"
            return request

    wire.client.sdk_configuration._hooks.before_request_hooks[0] = CustomUserAgentHook()

    with pytest.raises(MistralConfigurationError) as raised:
        lane.send(model_request())

    assert "secret" not in str(raised.value)
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == Decimal(0)


def test_replacement_pristine_http_client_is_refused_at_entry() -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    replacement_requests: list[httpx.Request] = []

    def replacement_handler(request: httpx.Request) -> httpx.Response:
        replacement_requests.append(request)
        return httpx.Response(200, json=response_body(), request=request)

    replacement = httpx.Client(transport=httpx.MockTransport(replacement_handler))
    wire.client.sdk_configuration.client = replacement

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == replacement_requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_http_client_subclass_is_refused_during_construction() -> None:
    requests: list[httpx.Request] = []

    class SubstitutedClient(httpx.Client):
        def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
            request = super().build_request(*args, **kwargs)
            request.headers["x-provider-secret"] = "must-not-persist"
            return request

    inner = SubstitutedClient(
        transport=httpx.MockTransport(
            lambda request: (
                requests.append(request)
                or httpx.Response(200, json=response_body(), request=request)
            )
        )
    )
    client = build_client(
        api_key="offline-placeholder-not-a-credential", http_client=inner
    )

    with pytest.raises(MistralConfigurationError) as raised:
        MistralChatCompletionsTransport(
            model=MISTRAL_SMALL_MODEL,
            client=client,
            deadline_seconds=30.0,
            clock=lambda: 0.0,
        )

    assert "secret" not in str(raised.value)
    assert requests == []


def test_preinstalled_capture_subclass_is_refused_during_construction() -> None:
    wire = Wire(response_body())

    class SubstitutedCapture(CapturedHttpClient):
        pass

    wire.client.sdk_configuration.client = SubstitutedCapture(wire.http_client)

    with pytest.raises(MistralConfigurationError) as raised:
        transport(wire)

    assert "secret" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize("mutation", EXECUTION_SURFACE_MUTATIONS)
def test_execution_surface_instance_shadow_refuses_at_construction(
    mutation: str,
) -> None:
    wire = Wire(response_body())
    if mutation.startswith("capture_"):
        wire.client.sdk_configuration.client = CapturedHttpClient(wire.http_client)
    mutate_execution_surface(wire, mutation)

    with pytest.raises(MistralConfigurationError) as raised:
        transport(wire)

    assert "secret" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize("mutation", EXECUTION_SURFACE_MUTATIONS)
def test_execution_surface_instance_shadow_refuses_at_entry(mutation: str) -> None:
    wire = Wire(response_body())
    guard = LifecycleCostGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-surface-entry-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    mutate_execution_surface(wire, mutation)

    with pytest.raises(MistralConfigurationError) as raised:
        lane.send(model_request())

    assert "secret" not in str(raised.value)
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert lane.last_usage().input_tokens is None
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == guard.committed_usd == Decimal(0)


@pytest.mark.parametrize("mutation", EXECUTION_SURFACE_MUTATIONS)
def test_execution_surface_instance_shadow_after_authorization_cancels_exactly(
    mutation: str,
) -> None:
    wire = Wire(response_body())

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            mutate_execution_surface(wire, mutation)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-surface-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError) as raised:
        lane.send(model_request())

    assert "secret" not in str(raised.value)
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert lane.last_usage().input_tokens is None
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == Decimal(0)
    assert guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_capture_class_mutation_refuses_at_entry() -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    capture = wire.client.sdk_configuration.client
    assert type(capture) is CapturedHttpClient

    class SubstitutedCapture(CapturedHttpClient):
        pass

    capture.__class__ = SubstitutedCapture

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_capture_class_mutation_after_authorization_cancels_exactly() -> None:
    wire = Wire(response_body())

    class SubstitutedCapture(CapturedHttpClient):
        pass

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            capture = wire.client.sdk_configuration.client
            assert type(capture) is CapturedHttpClient
            capture.__class__ = SubstitutedCapture
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-capture-class-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_inner_http_client_class_mutation_refuses_at_entry() -> None:
    wire = Wire(response_body())
    lane = transport(wire)

    class SubstitutedClient(httpx.Client):
        pass

    wire.http_client.__class__ = SubstitutedClient

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_inner_http_client_class_mutation_after_authorization_cancels_exactly() -> None:
    wire = Wire(response_body())

    class SubstitutedClient(httpx.Client):
        pass

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            wire.http_client.__class__ = SubstitutedClient
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-inner-class-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_chat_complete_class_descriptor_mutation_refuses_at_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    original = Chat.complete

    def substituted(self: Chat, **kwargs: Any) -> Any:
        kwargs["temperature"] = 1.0
        return original(self, **kwargs)

    monkeypatch.setattr(Chat, "complete", substituted)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_chat_complete_class_descriptor_mutation_after_authorization_cancels_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    original = Chat.complete

    def substituted(self: Chat, **kwargs: Any) -> Any:
        kwargs["temperature"] = 1.0
        return original(self, **kwargs)

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            monkeypatch.setattr(Chat, "complete", substituted)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-class-descriptor-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_chat_complete_code_replacement_refuses_at_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)

    def substituted(_self: Chat, **_kwargs: Any) -> Any:
        raise AssertionError("substituted function body ran")

    monkeypatch.setattr(Chat.complete, "__code__", substituted.__code__)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_chat_complete_code_replacement_after_authorization_cancels_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())

    def substituted(_self: Chat, **_kwargs: Any) -> Any:
        raise AssertionError("substituted function body ran")

    class MutatingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            self.authorization_count += 1
            monkeypatch.setattr(Chat.complete, "__code__", substituted.__code__)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-function-code-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert guard.authorization_count == 1
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd > Decimal(0)
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_chat_complete_kwdefaults_in_place_drift_refuses_at_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    defaults = Chat.complete.__kwdefaults__
    assert defaults is not None
    monkeypatch.setitem(defaults, "temperature", 1.0)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert Chat.complete.__kwdefaults__ is defaults
    assert defaults["temperature"] == 1.0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_chat_do_request_defaults_identity_drift_refuses_at_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    defaults = Chat.do_request.__defaults__
    assert defaults is not None
    replacement = (*defaults,)
    assert replacement is not defaults
    monkeypatch.setattr(Chat.do_request, "__defaults__", replacement)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


class _StatefulDataDescriptor:
    def __init__(self, value: object) -> None:
        self.value = value
        self.get_count = 0

    def __get__(self, instance: object, owner: type[object]) -> object:
        self.get_count += 1
        raise AssertionError("stateful data descriptor was invoked")

    def __set__(self, instance: object, value: object) -> None:
        raise AssertionError("stateful data descriptor was assigned")


class _SubstitutingDataDescriptor(_StatefulDataDescriptor):
    def __get__(self, instance: object, owner: type[object]) -> object:
        self.get_count += 1
        return self.value


def _chat_configuration_mro_substitution(
    wire: Wire,
) -> tuple[_SubstitutingDataDescriptor, tuple[type[Any], ...], type[BaseSDK]]:
    original = wire.client.__dict__["sdk_configuration"]
    capture = original.__dict__["client"]
    security = original.__dict__["security"]
    assert type(capture) is CapturedHttpClient
    replacement = replace(
        original,
        security=type(security)(api_key="substituted-offline-placeholder"),
    )
    replacement.__dict__["_hooks"] = original.__dict__["_hooks"]
    descriptor = _SubstitutingDataDescriptor(replacement)

    class SubstitutedBaseSDK(BaseSDK):
        sdk_configuration = descriptor

    original_bases = type.__getattribute__(Chat, "__bases__")
    return descriptor, original_bases, SubstitutedBaseSDK


def test_chat_mro_descriptor_insertion_refuses_at_entry_without_invocation() -> None:
    wire = Wire(response_body())

    class CountingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            self.authorization_count += 1
            return super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)

    guard = CountingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-mro-entry-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    descriptor, original_bases, substituted_base = _chat_configuration_mro_substitution(
        wire
    )
    try:
        Chat.__bases__ = (substituted_base,)
        with pytest.raises(MistralConfigurationError):
            lane.send(model_request())
    finally:
        Chat.__bases__ = original_bases

    assert descriptor.get_count == 0
    assert guard.authorization_count == 0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == guard.committed_usd == Decimal(0)


def test_chat_mro_descriptor_insertion_after_authorization_cancels_exactly() -> None:
    wire = Wire(response_body())

    class MutatingGuard(LifecycleCostGuard):
        authorization_count = 0
        authorized_amount = Decimal(0)

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            self.authorization_count += 1
            self.authorized_amount = amount
            Chat.__bases__ = (substituted_base,)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-mro-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    descriptor, original_bases, substituted_base = _chat_configuration_mro_substitution(
        wire
    )
    try:
        with pytest.raises(MistralConfigurationError):
            lane.send(model_request())
    finally:
        Chat.__bases__ = original_bases

    assert descriptor.get_count == 0
    assert guard.authorization_count == 1
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == guard.authorized_amount > Decimal(0)
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


def test_sdk_security_data_descriptor_refuses_at_entry_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())

    class CountingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            self.authorization_count += 1
            return super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)

    guard = CountingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-security-descriptor-entry-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    original_security = wire.client.__dict__["sdk_configuration"].__dict__["security"]
    substituted_security = type(original_security)(
        api_key="substituted-offline-placeholder"
    )
    descriptor = _SubstitutingDataDescriptor(substituted_security)
    monkeypatch.setattr(SDKConfiguration, "security", descriptor, raising=False)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert guard.authorization_count == 0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == guard.committed_usd == Decimal(0)


def test_sdk_security_data_descriptor_after_authorization_cancels_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    original_security = wire.client.__dict__["sdk_configuration"].__dict__["security"]
    substituted_security = type(original_security)(
        api_key="substituted-offline-placeholder"
    )
    descriptor = _SubstitutingDataDescriptor(substituted_security)

    class MutatingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            self.authorization_count += 1
            monkeypatch.setattr(SDKConfiguration, "security", descriptor, raising=False)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-security-descriptor-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert guard.authorization_count == 1
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd > Decimal(0)
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


@pytest.mark.parametrize("name", ("user_agent", "debug_logger"))
def test_other_sdk_dispatch_field_descriptors_refuse_without_invocation(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    descriptor = _StatefulDataDescriptor(object())
    monkeypatch.setattr(SDKConfiguration, name, descriptor, raising=False)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_arbitrary_admitted_class_descriptor_addition_refuses_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    descriptor = _StatefulDataDescriptor(object())
    monkeypatch.setattr(
        RetryConfig, "previously_absent_arbitrary_field", descriptor, raising=False
    )

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


@pytest.mark.parametrize("mutation", ("remove", "replace"))
def test_raw_class_namespace_entry_removal_or_replacement_is_refused(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    if mutation == "remove":
        monkeypatch.delattr(RetryConfig, "__init__")
    else:
        monkeypatch.setattr(RetryConfig, "__init__", object())

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


@pytest.mark.parametrize(
    ("owner", "name", "value_of"),
    (
        (Mistral, "chat", lambda wire: wire.client.__dict__["chat"]),
        (
            SDKConfiguration,
            "client",
            lambda wire: wire.client.__dict__["sdk_configuration"].__dict__["client"],
        ),
        (Chat, "sdk_configuration", lambda wire: wire.client.__dict__["chat"]),
        (
            CapturedHttpClient,
            "inner",
            lambda wire: wire.client.__dict__["sdk_configuration"].__dict__["client"],
        ),
        (
            SDKHooks,
            "before_request_hooks",
            lambda wire: wire.client.__dict__["sdk_configuration"].__dict__["_hooks"],
        ),
    ),
)
def test_identity_state_data_descriptor_refuses_at_entry_without_invocation(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[Any],
    name: str,
    value_of: Any,
) -> None:
    wire = Wire(response_body())

    class CountingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            self.authorization_count += 1
            return super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)

    guard = CountingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-data-descriptor-entry-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)
    descriptor = _StatefulDataDescriptor(value_of(wire))
    monkeypatch.setattr(owner, name, descriptor, raising=False)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert guard.authorization_count == 0
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd == guard.committed_usd == Decimal(0)


@pytest.mark.parametrize(
    ("owner", "name"),
    ((Mistral, "chat"), (SDKConfiguration, "client")),
)
def test_identity_state_data_descriptor_after_authorization_cancels_exactly(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[Any],
    name: str,
) -> None:
    wire = Wire(response_body())
    chat = wire.client.chat
    descriptor = _StatefulDataDescriptor(chat)

    class MutatingGuard(LifecycleCostGuard):
        authorization_count = 0

        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            self.authorization_count += 1
            monkeypatch.setattr(owner, name, descriptor, raising=False)
            return amount

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-data-descriptor-final-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert descriptor.get_count == 0
    assert guard.authorization_count == 1
    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.reserved_usd > Decimal(0)
    assert guard.committed_usd == guard.measured_usd == guard.forfeited_usd == Decimal(0)


@pytest.mark.parametrize(
    "mutation",
    (
        "configuration",
        "chat",
        "chat_configuration",
        "capture_inner",
        "hooks_container",
        "hook_group_container",
        "hook_member_identity",
    ),
)
def test_sealed_sdk_object_identity_drift_is_refused_at_entry(mutation: str) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    other = Wire(response_body())
    configuration = wire.client.sdk_configuration
    capture = configuration.client
    assert isinstance(capture, CapturedHttpClient)

    if mutation == "configuration":
        wire.client.sdk_configuration = other.client.sdk_configuration
    elif mutation == "chat":
        wire.client.chat = other.client.chat
    elif mutation == "chat_configuration":
        wire.client.chat.sdk_configuration = other.client.sdk_configuration
    elif mutation == "capture_inner":
        capture.inner = other.http_client
    elif mutation == "hooks_container":
        configuration._hooks = other.client.sdk_configuration._hooks
    elif mutation == "hook_group_container":
        hooks = configuration._hooks
        hooks.before_request_hooks = list(hooks.before_request_hooks)
    else:
        configuration._hooks.before_request_hooks[0] = (
            other.client.sdk_configuration._hooks.before_request_hooks[0]
        )

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()


def test_replacement_pristine_http_client_after_authorization_cancels_exactly() -> None:
    wire = Wire(response_body())
    replacement = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response_body(), request=request)
        )
    )

    class ReplacingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            amount = super().authorize(input_tokens_upper_bound=input_tokens_upper_bound)
            wire.client.sdk_configuration.client = replacement
            return amount

    guard = ReplacingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="mistral-client-identity-v1",
            input_usd_per_mtok=Decimal("1"),
            output_usd_per_mtok=Decimal("2"),
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    lane = transport(wire, cost_guard=guard)

    with pytest.raises(MistralConfigurationError):
        lane.send(model_request())

    assert wire.requests == []
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"
    assert lane.last_telemetry().attempts == ()
    assert guard.reserved_tokens == 0
    assert guard.committed_usd == Decimal(0)


def test_single_flight_owns_projection_and_evidence_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_projection = threading.Event()
    release_projection = threading.Event()
    observed: list[tuple[int | None, int | None]] = []

    class Recorder:
        def observe_provider_turn(self, evidence: Any) -> None:
            observed.append(
                (evidence.usage.input_tokens, evidence.response.output_tokens)
            )

    body = response_body(
        usage={"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
    )
    wire = Wire(body)
    lane = MistralChatCompletionsTransport(
        model=MISTRAL_SMALL_MODEL,
        client=wire.client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        recorder=Recorder(),
    )
    original = mistral_agent._model_response_from_validated

    def paused_projection(response: Any) -> Any:
        entered_projection.set()
        assert release_projection.wait(10.0)
        return original(response)

    monkeypatch.setattr(
        mistral_agent, "_model_response_from_validated", paused_projection
    )
    outcomes: list[Any] = []

    def first_call() -> None:
        try:
            outcomes.append(lane.send(model_request()))
        except Exception as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=first_call)
    thread.start()
    assert entered_projection.wait(10.0)
    try:
        with pytest.raises(AdapterBusyError):
            lane.send(model_request())
        assert len(wire.requests) == 1
        assert observed == []
    finally:
        release_projection.set()
        thread.join(10.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and not isinstance(outcomes[0], Exception)
    assert observed == [(7, 1)]


def test_single_flight_owns_projection_exception_and_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observing = threading.Event()
    release_observer = threading.Event()

    class ProjectionError(RuntimeError):
        pass

    class Recorder:
        def observe_provider_turn(self, evidence: Any) -> None:
            observing.set()
            assert release_observer.wait(10.0)

    lane = MistralChatCompletionsTransport(
        model=MISTRAL_SMALL_MODEL,
        client=Wire(response_body()).client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        recorder=Recorder(),
    )

    def fail_projection(response: Any) -> Any:
        raise ProjectionError

    monkeypatch.setattr(mistral_agent, "_model_response_from_validated", fail_projection)
    outcomes: list[BaseException] = []

    def first_call() -> None:
        try:
            lane.send(model_request())
        except BaseException as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=first_call)
    thread.start()
    assert observing.wait(10.0)
    try:
        with pytest.raises(AdapterBusyError):
            lane.send(model_request())
    finally:
        release_observer.set()
        thread.join(10.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], ProjectionError)


def test_single_flight_owns_exception_mapping_until_evidence_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = threading.Event()
    release_mapping = threading.Event()
    observing = threading.Event()
    release_observer = threading.Event()
    observed_faults: list[str | None] = []

    class Recorder:
        def observe_provider_turn(self, evidence: Any) -> None:
            observed_faults.append(evidence.fault)
            observing.set()
            assert release_observer.wait(10.0)

    wire = Wire(httpx.ReadTimeout("offline timeout"))
    lane = MistralChatCompletionsTransport(
        model=MISTRAL_SMALL_MODEL,
        client=wire.client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        recorder=Recorder(),
    )
    original = mistral_provider.classify_exception

    def paused_mapping(exception: Exception) -> Any:
        mapping.set()
        assert release_mapping.wait(10.0)
        return original(exception)

    monkeypatch.setattr(mistral_provider, "classify_exception", paused_mapping)
    outcomes: list[BaseException] = []

    def first_call() -> None:
        try:
            lane.send(model_request())
        except BaseException as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=first_call)
    thread.start()
    assert mapping.wait(10.0)
    with pytest.raises(AdapterBusyError):
        lane.send(model_request())
    release_mapping.set()
    assert observing.wait(10.0)
    try:
        with pytest.raises(AdapterBusyError):
            lane.send(model_request())
    finally:
        release_observer.set()
        thread.join(10.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], AdapterProviderError)
    assert outcomes[0].fault == "provider_timeout"
    assert observed_faults == ["provider_timeout"]
    assert len(wire.requests) == 1


def test_capture_install_and_identity_claim_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = Wire(response_body())
    entered = threading.Event()
    second_waiting = threading.Event()
    release = threading.Event()
    wrappers: list[CapturedHttpClient] = []
    original_lock = mistral_provider._CAPTURE_INSTALL_LOCK

    class AnnouncingLock:
        def __enter__(self) -> AnnouncingLock:
            if threading.current_thread().name == "second":
                second_waiting.set()
            original_lock.acquire()
            return self

        def __exit__(self, *exc: object) -> None:
            original_lock.release()

    class GatedCapture(CapturedHttpClient):
        def __init__(self, inner: Any) -> None:
            super().__init__(inner)
            wrappers.append(self)
            if threading.current_thread().name == "first":
                entered.set()
                assert release.wait(10.0)

    monkeypatch.setattr(mistral_provider, "_CAPTURE_INSTALL_LOCK", AnnouncingLock())
    monkeypatch.setattr(mistral_provider, "CapturedHttpClient", GatedCapture)
    outcomes: list[Any] = []

    def claim() -> None:
        try:
            outcomes.append(ensure_capturing_client(wire.client, owner=object()))
        except Exception as exc:
            outcomes.append(exc)

    first = threading.Thread(target=claim, name="first")
    second = threading.Thread(target=claim, name="second")
    first.start()
    assert entered.wait(10.0)
    second.start()
    assert second_waiting.wait(10.0)
    release.set()
    first.join(10.0)
    second.join(10.0)
    assert not first.is_alive() and not second.is_alive()
    captures = [value for value in outcomes if isinstance(value, CapturedHttpClient)]
    assert len(captures) == 1
    assert (
        len([value for value in outcomes if isinstance(value, MistralConfigurationError)])
        == 1
    )
    assert len(wrappers) == 1
    assert wire.client.sdk_configuration.client is wrappers[0]
    assert wire.requests == []
