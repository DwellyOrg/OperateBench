"""Lifecycle projection over Mistral's native Chat Completions SDK."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, cast

from mistralai.client import Mistral
from mistralai.client.models import ChatCompletionResponse

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
from operatebench.providers.mistral_chat import (
    MISTRAL_API,
    MISTRAL_API_COMPATIBILITY,
    MISTRAL_BASE_URL,
    MISTRAL_IMPLEMENTATION,
    MISTRAL_PROVIDER,
    MISTRAL_RESPONSE_CAPTURE,
    MISTRAL_RESPONSE_EXTENSIONS,
    MISTRAL_RESPONSE_STATE_POLICY,
    OUTPUT_LIMIT_FINISH_REASON,
    PARALLEL_TOOL_CALLS,
    REQUIRED_FINISH_REASON,
    RESPONSE_EXTENSION_DIGEST,
    SDK_INJECTED_FIELDS,
    TOOL_CHOICE,
    MistralChatExchange,
    ProviderDispatchObserver,
    RequestProfile,
    check_request_profile,
    sdk_version,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.toolcalls import decode_tool_arguments
from operatebench.providers.wire import WireResponse

LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION_V1 = (
    "lifecycle_mistral_chat_completions_model_request_v1"
)

# Compact action-schema projection successor; predecessor stays literal.
LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION_V2 = (
    "lifecycle_mistral_chat_completions_model_request_v2"
)

LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION = (
    "lifecycle_mistral_chat_completions_model_request_v3"
)
LIFECYCLE_MISTRAL_RETRY_POLICY = ProviderRetryPolicy(
    max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
)
COMPLETED_STOP_REASON = "completed"


def mistral_outcome_tools() -> list[dict[str, Any]]:
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
        f"request_mapping: {LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION}\n"
        f"protocol_version: {request.protocol_version}\n"
        f"request_digest_sha256: {request.request_digest_sha256}"
    )


def build_model_payload(
    request: ModelRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    check_model_request(request, model=model)
    shape = check_request_profile(profile, model=model)
    return {
        "model": model,
        "max_tokens": request.max_output_tokens,
        "tools": [
            {"type": "function", "function": tool} for tool in mistral_outcome_tools()
        ],
        "tool_choice": TOOL_CHOICE,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "messages": [
            {"role": "system", "content": request_instructions(request)},
            {
                "role": "user",
                "content": canonical_json_text(
                    request.prompt, "Lifecycle Mistral model request"
                ),
            },
        ],
        "temperature": shape.temperature,
    }


def lifecycle_mistral_settings(
    *, model: str, deadline_seconds: float, profile: RequestProfile
) -> dict[str, Any]:
    return {
        "provider": MISTRAL_PROVIDER,
        "implementation": MISTRAL_IMPLEMENTATION,
        "api": MISTRAL_API,
        "api_compatibility": MISTRAL_API_COMPATIBILITY,
        "base_url": MISTRAL_BASE_URL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "null_elision": LIFECYCLE_NULL_ELISION,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "request_mapping": LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_capture": MISTRAL_RESPONSE_CAPTURE,
        "response_contract": "closed_exactly_one_tool_call",
        "response_extensions": MISTRAL_RESPONSE_EXTENSIONS,
        "response_extensions_digest": RESPONSE_EXTENSION_DIGEST,
        "response_state_policy": MISTRAL_RESPONSE_STATE_POLICY,
        "retry": LIFECYCLE_MISTRAL_RETRY_POLICY.as_dict(),
        "sdk": "mistralai",
        "sdk_injected_fields": dict(SDK_INJECTED_FIELDS),
        "sdk_max_retries": 0,
        "sdk_version": sdk_version(),
        "tool_choice": TOOL_CHOICE,
        "tool_names": list(AGENT_TOOL_NAMES),
        "turn_deadline_seconds": deadline_seconds,
    }


def _invalid(detail: str) -> AdapterProviderError:
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


def check_lifecycle_response(response: WireResponse[ChatCompletionResponse]) -> None:
    parsed = response.parsed
    choices = parsed.choices or []
    if len(choices) != 1:
        raise _invalid("the Mistral answer must carry exactly one choice")
    choice = choices[0]
    if choice.finish_reason == OUTPUT_LIMIT_FINISH_REASON:
        return
    message = choice.message
    if message is None:
        raise _invalid("the Mistral answer carries no message and states no action")
    calls = message.tool_calls or []
    if choice.finish_reason != REQUIRED_FINISH_REASON or len(calls) != 1:
        raise _invalid(
            "the Mistral answer must complete with exactly one function tool call"
        )
    call = calls[0]
    if call.type != "function" or getattr(call, "function", None) is None:
        raise _invalid(
            "the Mistral answer carries a tool call outside the function contract"
        )


def _model_response_from_validated(
    response: WireResponse[ChatCompletionResponse],
) -> ModelResponse:
    parsed = response.parsed
    choice = (parsed.choices or [])[0]
    usage = parsed.usage
    output_tokens = 0 if usage is None else (usage.completion_tokens or 0)
    if choice.finish_reason == OUTPUT_LIMIT_FINISH_REASON:
        return ModelResponse(
            model=parsed.model,
            stop_reason=TRUNCATED_STOP_REASON,
            output_tokens=output_tokens,
        )
    message = cast("Any", choice.message)
    call = message.tool_calls[0]
    function = call.function
    try:
        arguments: Any = decode_tool_arguments(
            function.arguments,
            label="Lifecycle Mistral tool arguments",
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


def model_response_from(response: WireResponse[ChatCompletionResponse]) -> ModelResponse:
    check_lifecycle_response(response)
    return _model_response_from_validated(response)


def _response_id_digest(wire: WireResponse[ChatCompletionResponse] | None) -> str | None:
    return None if wire is None else wire.response_id_digest_sha256


class MistralChatCompletionsTransport:
    """Lifecycle transport using ``mistralai.client.Mistral`` natively."""

    def __init__(
        self,
        *,
        model: str,
        client: Mistral,
        deadline_seconds: float,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        recorder: ProviderTurnObserver | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        if (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ValueError("turn deadline must be a finite positive number")
        self._exchange = MistralChatExchange(
            model=model,
            client=client,
            single_flight_label="MistralChatCompletionsTransport",
            retry=LIFECYCLE_MISTRAL_RETRY_POLICY,
            sleep=sleep,
            clock=clock,
            cost_guard=cost_guard,
            profile=profile,
            before_dispatch=before_dispatch,
        )
        self._model = model
        self._profile = self._exchange.request_profile
        self._deadline_seconds = float(deadline_seconds)
        self._clock = clock
        self._recorder = recorder

    @property
    def provider(self) -> str:
        return MISTRAL_PROVIDER

    @property
    def api(self) -> str:
        return MISTRAL_API

    @property
    def request_mapping(self) -> str:
        return LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION

    @property
    def model(self) -> str:
        return self._model

    @property
    def settings(self) -> Mapping[str, Any]:
        return lifecycle_mistral_settings(
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
        with self._exchange.turn():
            started = self._clock()
            check_model_request(request, model=self._model)
            payload = build_model_payload(
                request, model=self._model, profile=self._profile
            )
            answer: ModelResponse | None = None
            wire: WireResponse[ChatCompletionResponse] | None = None
            fault: str | None = None
            try:
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
    "LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION",
    "LIFECYCLE_MISTRAL_RETRY_POLICY",
    "MistralChatCompletionsTransport",
    "build_model_payload",
    "check_lifecycle_response",
    "lifecycle_mistral_settings",
    "mistral_outcome_tools",
    "model_response_from",
]
