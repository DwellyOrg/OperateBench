"""Lifecycle projection over xAI's OpenAI-compatible Chat Completions API.

The SDK is the real pinned ``openai`` SDK. This is never represented as native
xAI: provider, compatibility API, implementation, endpoint, and SDK are all
recorded independently.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletion

from operatebench.agents.lifecycle_contract import (
    AGENT_TOOL_NAMES,
    INSTRUCTIONS_CONTRACT,
    LIFECYCLE_NULL_ELISION,
    MAX_TRANSIENT_TEXT_CHARACTERS,
    ToolArgumentsUndecodable,
    UndecodableArguments,
    check_model_request,
    elide_null_optionals,
    outcome_tools,
)
from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    TRUNCATED_STOP_REASON,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ProviderTurnEvidence,
    ProviderTurnObserver,
    ToolCall,
)
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.cost import CostGuard
from operatebench.providers.executor import Clock, ProviderRetryPolicy, Sleep
from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.toolcalls import decode_tool_arguments
from operatebench.providers.wire import WireResponse
from operatebench.providers.xai_openai_compat import (
    OUTPUT_LIMIT_FINISH_REASON,
    PARALLEL_TOOL_CALLS,
    REQUIRED_FINISH_REASON,
    RESPONSE_EXTENSION_DIGEST,
    TOOL_CHOICE,
    XAI_API,
    XAI_API_COMPATIBILITY,
    XAI_BASE_URL,
    XAI_COMPAT_RESPONSE_EXTENSIONS,
    XAI_IMPLEMENTATION,
    XAI_PROVIDER,
    XAI_RESPONSE_CAPTURE,
    RequestProfile,
    XAIOpenAICompatExchange,
    check_request_profile,
)

LIFECYCLE_XAI_REQUEST_MAPPING_VERSION_V1 = (
    "lifecycle_xai_openai_compat_chat_completions_model_request_v1"
)

# Compact action-schema projection successor; predecessor stays literal.
LIFECYCLE_XAI_REQUEST_MAPPING_VERSION_V2 = (
    "lifecycle_xai_openai_compat_chat_completions_model_request_v2"
)

LIFECYCLE_XAI_REQUEST_MAPPING_VERSION = (
    "lifecycle_xai_openai_compat_chat_completions_model_request_v3"
)
LIFECYCLE_XAI_RETRY_POLICY = ProviderRetryPolicy(
    max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
)
COMPLETED_STOP_REASON = "completed"


def xai_outcome_tools() -> list[dict[str, Any]]:
    """Chat Completions function definitions from the neutral Lifecycle authority."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": tool["strict"],
        }
        for tool in outcome_tools()
    ]


def request_instructions(request: ModelRequest) -> str:
    return (
        f"{INSTRUCTIONS_CONTRACT}\n"
        f"request_mapping: {LIFECYCLE_XAI_REQUEST_MAPPING_VERSION}\n"
        f"protocol_version: {request.protocol_version}\n"
        f"request_digest_sha256: {request.request_digest_sha256}"
    )


def build_model_payload(
    request: ModelRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    check_model_request(request, model=model)
    shape = check_request_profile(profile, model=model)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_output_tokens,
        "tools": [{"type": "function", "function": tool} for tool in xai_outcome_tools()],
        "tool_choice": TOOL_CHOICE,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "messages": [
            {"role": "system", "content": request_instructions(request)},
            {
                "role": "user",
                "content": canonical_json_text(
                    request.prompt, "Lifecycle xAI model request"
                ),
            },
        ],
    }
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    if shape.reasoning_effort is not None:
        payload["reasoning_effort"] = shape.reasoning_effort
    return payload


def lifecycle_xai_settings(
    *, model: str, deadline_seconds: float, profile: RequestProfile
) -> dict[str, Any]:
    return {
        "provider": XAI_PROVIDER,
        "implementation": XAI_IMPLEMENTATION,
        "api": XAI_API,
        "api_compatibility": XAI_API_COMPATIBILITY,
        "base_url": XAI_BASE_URL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "null_elision": LIFECYCLE_NULL_ELISION,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "request_mapping": LIFECYCLE_XAI_REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_capture": XAI_RESPONSE_CAPTURE,
        "response_contract": "closed_exactly_one_tool_call",
        "response_extensions": XAI_COMPAT_RESPONSE_EXTENSIONS,
        "response_extensions_digest": RESPONSE_EXTENSION_DIGEST,
        "retry": LIFECYCLE_XAI_RETRY_POLICY.as_dict(),
        "sdk": "openai",
        "sdk_max_retries": 0,
        "sdk_version": openai.__version__,
        "tool_choice": TOOL_CHOICE,
        "tool_names": list(AGENT_TOOL_NAMES),
        "turn_deadline_seconds": deadline_seconds,
    }


def _invalid(detail: str) -> AdapterProviderError:
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


def check_lifecycle_response(response: WireResponse[ChatCompletion]) -> None:
    """Validate Lifecycle response protocol before usage can be settled."""
    parsed = response.parsed
    if len(parsed.choices) != 1:
        raise _invalid(
            "the xAI-compatible answer must carry exactly one choice; response "
            "content is not retained"
        )
    choice = parsed.choices[0]
    if choice.finish_reason == OUTPUT_LIMIT_FINISH_REASON:
        return
    message = choice.message
    if message is None:
        raise _invalid(
            "the xAI-compatible answer carries a choice with no message, so it "
            "states no action; response content is not retained"
        )
    calls = message.tool_calls or []
    if choice.finish_reason != REQUIRED_FINISH_REASON or len(calls) != 1:
        raise _invalid(
            "the xAI-compatible answer must complete with exactly one function "
            "tool call; provider prose and call content are not retained"
        )
    call = calls[0]
    if call.type != "function" or getattr(call, "function", None) is None:
        raise _invalid(
            "the xAI-compatible answer carries a tool call outside the closed "
            "function contract"
        )


def model_response_from(response: WireResponse[ChatCompletion]) -> ModelResponse:
    """Validate and project one response for direct callers."""
    check_lifecycle_response(response)
    return _model_response_from_validated(response)


def _model_response_from_validated(
    response: WireResponse[ChatCompletion],
) -> ModelResponse:
    """Project a response already admitted before settlement, without provider faults."""
    parsed = response.parsed
    choice = parsed.choices[0]
    usage = parsed.usage
    output_tokens = 0 if usage is None else usage.completion_tokens
    if choice.finish_reason == OUTPUT_LIMIT_FINISH_REASON:
        return ModelResponse(
            model=parsed.model,
            stop_reason=TRUNCATED_STOP_REASON,
            output_tokens=output_tokens,
        )
    message = cast("Any", choice.message)
    calls = message.tool_calls
    call = calls[0]
    function = call.function
    try:
        arguments: Any = decode_tool_arguments(
            function.arguments,
            label="Lifecycle xAI tool arguments",
            error=ToolArgumentsUndecodable,
        )
    except ToolArgumentsUndecodable:
        arguments = UndecodableArguments()
    if isinstance(arguments, Mapping):
        arguments = elide_null_optionals(function.name, arguments)
    content = message.content if isinstance(message.content, str) else ""
    return ModelResponse(
        model=parsed.model,
        stop_reason=COMPLETED_STOP_REASON,
        tool_calls=(ToolCall(function.name, arguments),),
        text=content[:MAX_TRANSIENT_TEXT_CHARACTERS],
        output_tokens=output_tokens,
    )


def _response_id_digest(wire: WireResponse[ChatCompletion] | None) -> str | None:
    return None if wire is None else wire.response_id_digest_sha256


class XAIOpenAICompatChatCompletionsTransport:
    """A Lifecycle transport explicitly using xAI's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        deadline_seconds: float,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        recorder: ProviderTurnObserver | None = None,
    ) -> None:
        if (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ValueError("turn deadline must be a finite positive number")
        self._exchange = XAIOpenAICompatExchange(
            model=model,
            client=client,
            single_flight_label="XAIOpenAICompatChatCompletionsTransport",
            retry=LIFECYCLE_XAI_RETRY_POLICY,
            sleep=sleep,
            clock=clock,
            cost_guard=cost_guard,
            profile=profile,
        )
        self._model = model
        self._profile = self._exchange.request_profile
        self._deadline_seconds = float(deadline_seconds)
        self._clock = clock
        self._recorder = recorder

    @property
    def provider(self) -> str:
        return XAI_PROVIDER

    @property
    def api(self) -> str:
        return XAI_API

    @property
    def request_mapping(self) -> str:
        return LIFECYCLE_XAI_REQUEST_MAPPING_VERSION

    @property
    def model(self) -> str:
        return self._model

    @property
    def settings(self) -> Mapping[str, Any]:
        return lifecycle_xai_settings(
            model=self._model,
            deadline_seconds=self._deadline_seconds,
            profile=self._profile,
        )

    def last_usage(self) -> AdapterUsage:
        return self._exchange.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._exchange.last_telemetry()

    def attach_recorder(self, recorder: ProviderTurnObserver) -> None:
        if self._recorder is not None:
            raise ValueError("a provider evidence recorder is already attached")
        self._recorder = recorder

    def send(self, request: ModelRequest) -> ModelResponse:
        started = self._clock()
        check_model_request(request, model=self._model)
        payload = build_model_payload(request, model=self._model, profile=self._profile)
        answer: ModelResponse | None = None
        wire: WireResponse[ChatCompletion] | None = None
        fault: str | None = None
        try:
            with self._exchange.turn():
                self._exchange.clear()
                wire = self._exchange.exchange(
                    payload,
                    self._deadline(started),
                    response_validator=check_lifecycle_response,
                )
            answer = _model_response_from_validated(wire)
            return answer
        except AdapterProviderError as exc:
            fault = exc.fault
            raise
        finally:
            if self._recorder is not None:
                self._recorder.observe_provider_turn(
                    ProviderTurnEvidence(
                        request=request,
                        payload=payload,
                        settings=self.settings,
                        telemetry=self.last_telemetry(),
                        usage=self.last_usage(),
                        response=answer,
                        response_id_digest_sha256=_response_id_digest(wire),
                        fault=fault,
                    )
                )

    def _deadline(self, started: float) -> TurnDeadline:
        budget = self._deadline_seconds
        return TurnDeadline(
            remaining_seconds=budget,
            cancelled=lambda: (self._clock() - started) >= budget,
            started_at=started,
        )


__all__ = [
    "LIFECYCLE_XAI_REQUEST_MAPPING_VERSION",
    "LIFECYCLE_XAI_RETRY_POLICY",
    "XAIOpenAICompatChatCompletionsTransport",
    "build_model_payload",
    "check_lifecycle_response",
    "lifecycle_xai_settings",
    "model_response_from",
    "xai_outcome_tools",
]
