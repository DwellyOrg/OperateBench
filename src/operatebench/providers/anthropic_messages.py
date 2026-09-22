"""Anthropic Messages provider exchange shared by Lifecycle projections.

This module owns only the SDK wire boundary: model profiles, endpoint and retry
checks, strict raw/typed response validation, exact usage, and the common turn
executor.  It imports no Lifecycle or Boundary semantics.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import anthropic
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from operatebench.providers.config import (
    ProviderConfigurationError,
    check_endpoint,
    check_payload_fields,
)
from operatebench.providers.cost import CostGuard, request_input_token_bound
from operatebench.providers.executor import (
    SDK_MAX_RETRIES,
    Clock,
    ProviderRetryPolicy,
    Sleep,
    TurnExecutor,
    check_retry_policy,
)
from operatebench.providers.faults import (
    PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED,
    PROVIDER_FAULT_INVALID_REQUEST_OR_SPEND_LIMIT,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SPEND_LIMIT,
    PROVIDER_FAULT_TIMEOUT,
    AdapterProviderError,
    Fault,
    SingleFlight,
    classify_http_status,
)
from operatebench.providers.response import (
    check_response_collection,
    check_response_contract,
    check_response_object,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.usage import TokenUsage
from operatebench.providers.wire import (
    WireResponse,
    WireShape,
    checked_wire_list,
    checked_wire_mapping,
    checked_wire_object,
    checked_wire_string,
    declared_wire_fields,
    is_wire_token_count,
    wire_token_usage,
)

ANTHROPIC_PROVIDER = "anthropic"
ANTHROPIC_API = "messages"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
TEMPERATURE = 0.0
TOOL_CHOICE: Mapping[str, Any] = MappingProxyType(
    {"type": "any", "disable_parallel_tool_use": True}
)
PROHIBITED_REQUEST_FIELDS = ("temperature", "thinking", "top_k", "top_p")
_COMMON_REQUEST_FIELDS = (
    "max_tokens",
    "messages",
    "model",
    "system",
    "tool_choice",
    "tools",
)


class AnthropicConfigurationError(ProviderConfigurationError):
    """The injected Anthropic client or request profile is not the pinned one."""


class ProviderDispatchObserver(Protocol):
    """Observe one SDK request attempt immediately before it begins."""

    def __call__(self) -> None: ...


class _DispatchObserverFailure(AnthropicConfigurationError):
    """Carry a pre-dispatch observer exception through the shared executor."""

    def __init__(self, exception: Exception) -> None:
        super().__init__("the provider dispatch observer failed before SDK invocation")
        self.exception = exception


@dataclass(frozen=True)
class RequestProfile:
    profile_id: str
    temperature: float | None
    thinking: Mapping[str, Any] | None

    def sent_fields(self) -> tuple[str, ...]:
        fields = list(_COMMON_REQUEST_FIELDS)
        if self.temperature is not None:
            fields.append("temperature")
        if self.thinking is not None:
            fields.append("thinking")
        return tuple(sorted(fields))

    def omitted_fields(self) -> tuple[str, ...]:
        sent = set(self.sent_fields())
        return tuple(
            sorted(field for field in PROHIBITED_REQUEST_FIELDS if field not in sent)
        )

    def as_settings(self) -> dict[str, Any]:
        return {
            "request_profile": self.profile_id,
            "request_fields_sent": list(self.sent_fields()),
            "request_fields_omitted": list(self.omitted_fields()),
            "temperature": self.temperature,
            "thinking": None if self.thinking is None else dict(self.thinking),
        }


HAIKU_4_5_MODEL = "claude-haiku-4-5-20251001"
SONNET_5_MODEL = "claude-sonnet-5"
HAIKU_4_5_PROFILE = RequestProfile(
    "haiku45_temperature_zero_thinking_omitted_v1", TEMPERATURE, None
)
SONNET_5_PROFILE = RequestProfile(
    "sonnet5_sampling_omitted_thinking_disabled_v1",
    None,
    MappingProxyType({"type": "disabled"}),
)
MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {HAIKU_4_5_MODEL: HAIKU_4_5_PROFILE, SONNET_5_MODEL: SONNET_5_PROFILE}
)


def request_profile_for(model: str) -> RequestProfile:
    try:
        return MODEL_REQUEST_PROFILES[model]
    except KeyError as exc:
        raise AnthropicConfigurationError(
            "this Lifecycle lane has no reviewed Anthropic request profile for the "
            "named model"
        ) from exc


def check_request_profile(
    profile: RequestProfile | None, *, model: str
) -> RequestProfile:
    expected = request_profile_for(model)
    if profile is None or profile == expected:
        return expected
    raise AnthropicConfigurationError(
        "the request profile is not the pinned profile for this Anthropic model"
    )


def check_client_endpoint(client: anthropic.Anthropic) -> None:
    check_endpoint(
        client.base_url,
        expected=ANTHROPIC_BASE_URL,
        provider=ANTHROPIC_PROVIDER,
        setting="base_url",
        error=AnthropicConfigurationError,
    )


MESSAGE_SHAPE = WireShape(
    "Anthropic message",
    declared_wire_fields(Message),
    ("content", "model", "stop_reason", "usage"),
)
TOOL_USE_SHAPE = WireShape(
    "Anthropic tool-use block",
    declared_wire_fields(ToolUseBlock),
    ("id", "input", "name", "type"),
)
TEXT_SHAPE = WireShape(
    "Anthropic text block", declared_wire_fields(TextBlock), ("text", "type")
)
USAGE_FIELDS = ("input_tokens", "output_tokens")
CACHE_USAGE_FIELDS = ("cache_creation_input_tokens", "cache_read_input_tokens")
USAGE_SHAPE = WireShape("Anthropic usage", declared_wire_fields(Usage), USAGE_FIELDS)


def check_response_admissible(response: WireResponse[Message], *, model: str) -> None:
    wire = checked_wire_object(response.wire, MESSAGE_SHAPE)
    checked_wire_string(wire["model"], kind="response model")
    if wire["model"] != model:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the response identifies a model other than the pinned model; neither "
            "provider-controlled identifier is retained",
        )
    blocks = checked_wire_list(wire["content"], kind="content blocks")
    for raw in blocks:
        block = checked_wire_mapping(raw, kind="content block")
        declared = checked_wire_string(block.get("type"), kind="content block type")
        if declared == "tool_use":
            checked = checked_wire_object(block, TOOL_USE_SHAPE)
            checked_wire_string(checked["name"], kind="tool name")
            checked_wire_mapping(checked["input"], kind="tool input")
        elif declared == "text":
            checked_wire_object(block, TEXT_SHAPE)
        else:
            raise AdapterProviderError(
                PROVIDER_FAULT_RESPONSE_INVALID,
                "the response carries a content block outside the closed text and "
                "tool-use contract; its provider-controlled type is not retained",
            )
    usage = checked_wire_object(wire["usage"], USAGE_SHAPE)
    wire_token_usage(usage, fields=USAGE_FIELDS)
    for field in CACHE_USAGE_FIELDS:
        value = usage.get(field)
        if value is None:
            continue
        if not is_wire_token_count(value):
            raise AdapterProviderError(
                PROVIDER_FAULT_RESPONSE_INVALID,
                "the Anthropic usage reports a cache count that is not an exact "
                "bounded nonnegative integer",
            )
        if value != 0:
            raise AdapterProviderError(
                PROVIDER_FAULT_RESPONSE_INVALID,
                "the Anthropic usage reports nonzero cache tokens, but this build "
                "has no versioned cache-pricing and accounting policy",
            )
    parsed = response.parsed
    check_response_contract(parsed)
    check_response_collection(parsed.content, kind="content blocks")
    check_response_object(parsed.usage, fields=USAGE_FIELDS, kind="usage block")
    if parsed.model != model:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the SDK response identifies a model other than the pinned model",
        )


def response_usage(response: WireResponse[Message]) -> TokenUsage:
    usage = checked_wire_object(response.wire["usage"], USAGE_SHAPE)
    return wire_token_usage(usage, fields=USAGE_FIELDS)


def classify_exception(exception: Exception) -> Fault | None:
    if isinstance(exception, anthropic.APITimeoutError):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, anthropic.APIConnectionError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, anthropic.APIResponseValidationError):
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, anthropic.APIStatusError):
        if exception.status_code == 400:
            return _classify_bad_request(exception)
        return classify_http_status(exception.status_code)
    return None


# These are the lane's locally reviewed spend-limit sentences. Matching is exact
# after stripping surrounding ASCII whitespace: provider prose is otherwise
# neither interpreted nor retained, so near-misses and appended text cannot turn
# an unknown rejection into a stronger diagnosis. Anthropic's documented claim is
# only the broader semantic one: an organization or workspace spend limit can be
# HTTP 400 (https://platform.claude.com/docs/en/api/errors).
_SPEND_LIMIT_MESSAGES = frozenset(
    {
        "Your organization has reached its spend limit",
        "Your organization has reached its spend limit.",
        "Your workspace has reached its spend limit",
        "Your workspace has reached its spend limit.",
    }
)


def _classify_bad_request(exception: anthropic.APIStatusError) -> Fault:
    body = exception.body
    if not isinstance(body, Mapping):
        return Fault(
            PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED,
            False,
            400,
            response_received=True,
        )
    error = body.get("error")
    if not isinstance(error, Mapping) or error.get("type") != "invalid_request_error":
        return Fault(
            PROVIDER_FAULT_BAD_REQUEST_UNCLASSIFIED,
            False,
            400,
            response_received=True,
        )
    message = error.get("message")
    fault = (
        PROVIDER_FAULT_SPEND_LIMIT
        if isinstance(message, str) and message.strip(" \t\r\n") in _SPEND_LIMIT_MESSAGES
        else PROVIDER_FAULT_INVALID_REQUEST_OR_SPEND_LIMIT
    )
    return Fault(fault, False, 400, response_received=True)


class AnthropicMessagesExchange:
    """One pinned Anthropic client exchanging provider-neutral payload mappings."""

    def __init__(
        self,
        *,
        model: str,
        client: anthropic.Anthropic,
        single_flight_label: str,
        retry: ProviderRetryPolicy,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        self._profile = check_request_profile(profile, model=model)
        check_client_endpoint(client)
        if client.max_retries != SDK_MAX_RETRIES:
            raise AnthropicConfigurationError(
                "the Anthropic SDK retry loop must be disabled with max_retries=0"
            )
        self._model = model
        self._client = client
        self._retry = check_retry_policy(
            retry, provider=ANTHROPIC_PROVIDER, error=AnthropicConfigurationError
        )
        self._executor: TurnExecutor[WireResponse[Message]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        self._single_flight = SingleFlight(adapter=single_flight_label)
        if before_dispatch is not None and not callable(before_dispatch):
            raise AnthropicConfigurationError(
                "the provider dispatch observer must be a zero-argument callable"
            )
        self._before_dispatch = before_dispatch

    @property
    def model(self) -> str:
        return self._model

    @property
    def request_profile(self) -> RequestProfile:
        return self._profile

    @property
    def retry(self) -> ProviderRetryPolicy:
        return self._retry

    def turn(self) -> SingleFlight:
        return self._single_flight

    def clear(self) -> None:
        self._executor.clear()

    def last_usage(self) -> AdapterUsage:
        return self._executor.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._executor.last_telemetry()

    def exchange(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[Message]:
        check_client_endpoint(self._client)
        check_payload_fields(
            payload,
            profile_id=self._profile.profile_id,
            sent=self._profile.sent_fields(),
            omitted=self._profile.omitted_fields(),
        )

        def check_before_settlement(result: WireResponse[Message]) -> None:
            check_response_admissible(result, model=self._model)
            result.admit_response_id()

        try:
            return self._executor.run(
                payload=payload,
                deadline=deadline,
                token_bound=request_input_token_bound(payload, "Anthropic request"),
                dispatch=lambda timeout: self._dispatch(payload, timeout),
                check_before_dispatch=self._check_before_dispatch,
                classify=classify_exception,
                check_identity=check_before_settlement,
                usage_of=response_usage,
            )
        except _DispatchObserverFailure as exc:
            raise exc.exception from exc

    def _check_before_dispatch(self) -> None:
        check_client_endpoint(self._client)
        if self._before_dispatch is not None:
            try:
                self._before_dispatch()
            except Exception as exc:
                raise _DispatchObserverFailure(exc) from exc

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[Message]:
        raw = self._client.messages.with_raw_response.create(**payload, timeout=timeout)
        return WireResponse(raw.text, raw.parse, kind="Anthropic message")


__all__ = [
    "ANTHROPIC_API",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_PROVIDER",
    "HAIKU_4_5_MODEL",
    "HAIKU_4_5_PROFILE",
    "SONNET_5_MODEL",
    "SONNET_5_PROFILE",
    "TOOL_CHOICE",
    "AnthropicConfigurationError",
    "AnthropicMessagesExchange",
    "ProviderDispatchObserver",
    "RequestProfile",
    "check_request_profile",
    "request_profile_for",
]
