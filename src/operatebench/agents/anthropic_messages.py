"""Lifecycle projection onto the shared Anthropic Messages exchange."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import anthropic
from anthropic.types import Message

from operatebench.agents.lifecycle_contract import (
    INSTRUCTIONS_CONTRACT,
    MAX_TRANSIENT_TEXT_CHARACTERS,
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
from operatebench.providers.anthropic_messages import (
    ANTHROPIC_API,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_PROVIDER,
    TOOL_CHOICE,
    AnthropicMessagesExchange,
    ProviderDispatchObserver,
    RequestProfile,
    check_request_profile,
)
from operatebench.providers.cost import CostGuard
from operatebench.providers.executor import Clock, ProviderRetryPolicy, Sleep
from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.wire import WireResponse

#: Historical mapping retained as an independent literal for readers of existing
#: evidence.  It projected the authoritative schema unchanged, including root
#: combinators that the Messages tool contract does not accept.
LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V1 = (
    "lifecycle_anthropic_messages_model_request_v1"
)
#: Current mapping.  It changes only Anthropic's wire projection: every tool has
#: a root object schema without a root combinator.  The provider-side choice stays
#: non-strict; local parsing and Core validation remain the semantic gate.
LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V2 = (
    "lifecycle_anthropic_messages_model_request_v2"
)

# Compact action-schema projection successor; predecessor stays literal.
LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V3 = (
    "lifecycle_anthropic_messages_model_request_v3"
)

LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION = (
    "lifecycle_anthropic_messages_model_request_v4"
)
LIFECYCLE_ANTHROPIC_RETRY_POLICY = ProviderRetryPolicy(
    max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
)
COMPLETED_STOP_REASON = "completed"

_ROOT_COMBINATORS = frozenset({"oneOf", "anyOf", "allOf"})
_SCHEMA_MAX_DEPTH = 32
_SCHEMA_MAX_NODES = 10_000


@dataclass
class _SchemaBudget:
    nodes: int = 0

    def visit(self, depth: int) -> None:
        self.nodes += 1
        if depth > _SCHEMA_MAX_DEPTH or self.nodes > _SCHEMA_MAX_NODES:
            raise ValueError(
                "Anthropic input schema exceeds the bounded projection limit"
            )


def _clone_schema(value: Any, budget: _SchemaBudget, depth: int, active: set[int]) -> Any:
    budget.visit(depth)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("Anthropic input schema contains a cyclic object graph")
        active.add(identity)
        try:
            return {
                key: _clone_schema(child, budget, depth + 1, active)
                for key, child in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise ValueError("Anthropic input schema contains a cyclic object graph")
        active.add(identity)
        try:
            return [_clone_schema(child, budget, depth + 1, active) for child in value]
        finally:
            active.remove(identity)
    return value


def _pointer(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError("Anthropic input schema uses a non-local root reference")
    current: Any = root
    for encoded in reference[2:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            raise ValueError(
                "Anthropic input schema contains an unresolved root reference"
            )
        current = current[token]
    if not isinstance(current, Mapping):
        raise ValueError(
            "Anthropic input schema root reference does not resolve to a schema"
        )
    return current


def _resolve_projection_schema(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    budget: _SchemaBudget,
    depth: int,
    references: set[str],
) -> Mapping[str, Any]:
    budget.visit(depth)
    reference = schema.get("$ref")
    if reference is None:
        return schema
    if not isinstance(reference, str):
        raise ValueError("Anthropic input schema root reference is not a string")
    unsupported = set(schema) - {"$ref", "$defs", "definitions"}
    if unsupported:
        raise ValueError("Anthropic input schema combines a root reference with siblings")
    if reference in references:
        raise ValueError("Anthropic input schema contains a cyclic root reference")
    references.add(reference)
    try:
        return _resolve_projection_schema(
            _pointer(root, reference),
            root=root,
            budget=budget,
            depth=depth + 1,
            references=references,
        )
    finally:
        references.remove(reference)


def _object_parts(schema: Mapping[str, Any]) -> tuple[Mapping[str, Any], set[str], bool]:
    declared_type = schema.get("type")
    if declared_type != "object":
        raise ValueError("Anthropic input schema combinator contains a non-object branch")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Mapping)
        for name, value in properties.items()
    ):
        raise ValueError("Anthropic input schema has malformed object properties")
    required = schema.get("required", [])
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not all(isinstance(name, str) for name in required)
        or not set(required) <= set(properties)
    ):
        raise ValueError("Anthropic input schema has malformed required properties")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        raise ValueError(
            "Anthropic input schema has an unsupported additionalProperties policy"
        )
    return properties, set(required), additional


def lower_anthropic_input_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project one authoritative tool schema into Anthropic's root-object contract.

    Nested schemas are copied byte-for-byte in meaning.  Only a root combinator is
    widened: object properties are censused across every branch, common required
    names remain required, and a closed additional-property policy is retained only
    when every branch is closed.  Conflicting property schemas, non-object branches,
    unsupported references, cycles and excessive inputs are refused rather than
    guessed at.  The local Lifecycle parser remains the exact semantic authority.
    """
    if not isinstance(schema, Mapping):
        raise ValueError("Anthropic input schema is not a mapping")
    budget = _SchemaBudget()
    resolved = _resolve_projection_schema(
        schema, root=schema, budget=budget, depth=0, references=set()
    )
    combinators = _ROOT_COMBINATORS & set(resolved)
    if len(combinators) > 1:
        raise ValueError("Anthropic input schema has multiple root combinators")
    if not combinators:
        if resolved.get("type") != "object":
            raise ValueError("Anthropic input schema root is not an object")
        cloned = cast(dict[str, Any], _clone_schema(resolved, budget, 0, set()))
        if resolved is not schema:
            for definition_key in ("$defs", "definitions"):
                if definition_key in schema:
                    cloned[definition_key] = _clone_schema(
                        schema[definition_key], budget, 1, set()
                    )
        return cloned

    combinator = next(iter(combinators))
    raw_branches = resolved[combinator]
    if not isinstance(raw_branches, list) or not raw_branches:
        raise ValueError("Anthropic input schema root combinator has no branches")
    base = {key: value for key, value in resolved.items() if key != combinator}
    definitions: dict[str, Any] = {
        key: value for key, value in base.items() if key in {"$defs", "definitions"}
    }
    if resolved is not schema:
        for key in ("$defs", "definitions"):
            if key in schema:
                definitions[key] = schema[key]
    object_base = {key: value for key, value in base.items() if key not in definitions}
    if "type" not in object_base:
        object_base["type"] = "object"
    base_properties, base_required, base_additional = _object_parts(object_base)

    branch_properties: dict[str, Mapping[str, Any]] = {}
    branch_required: list[set[str]] = []
    branch_additional: list[bool] = []
    for raw_branch in raw_branches:
        if not isinstance(raw_branch, Mapping):
            raise ValueError(
                "Anthropic input schema root combinator has a malformed branch"
            )
        branch = _resolve_projection_schema(
            raw_branch, root=schema, budget=budget, depth=1, references=set()
        )
        properties, required, additional = _object_parts(branch)
        branch_required.append(required)
        branch_additional.append(additional)
        for name, property_schema in properties.items():
            projected_property = _clone_schema(property_schema, budget, 2, set())
            if name in base_properties:
                continue
            previous = branch_properties.get(name)
            if previous is not None and previous != projected_property:
                raise ValueError(
                    "Anthropic input schema has conflicting schemas for one "
                    "branch property"
                )
            branch_properties[name] = projected_property

    properties = dict(base_properties)
    if base_additional:
        properties.update(branch_properties)
    common_branch_required = set.intersection(*branch_required)
    if not base_additional:
        common_branch_required &= set(base_properties)
    required = base_required | common_branch_required
    additional = base_additional and not all(
        value is False for value in branch_additional
    )

    lowered: dict[str, Any] = {"type": "object"}
    lowered["properties"] = {
        name: _clone_schema(properties[name], budget, 1, set())
        for name in sorted(properties)
    }
    lowered["required"] = sorted(required)
    if not additional:
        lowered["additionalProperties"] = False
    for key, value in object_base.items():
        if key not in {"type", "properties", "required", "additionalProperties"}:
            lowered[key] = _clone_schema(value, budget, 1, set())
    for key, value in definitions.items():
        lowered[key] = _clone_schema(value, budget, 1, set())
    return lowered


def anthropic_outcome_tools() -> list[dict[str, Any]]:
    """Anthropic tools projected from the same authored Lifecycle contract."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": lower_anthropic_input_schema(tool["parameters"]),
        }
        for tool in outcome_tools()
    ]


def request_instructions(request: ModelRequest) -> str:
    return (
        f"{INSTRUCTIONS_CONTRACT}\n"
        f"request_mapping: {LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION}\n"
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
        "system": request_instructions(request),
        "tools": anthropic_outcome_tools(),
        "tool_choice": dict(TOOL_CHOICE),
        "messages": [
            {
                "role": "user",
                "content": canonical_json_text(
                    request.prompt, "Lifecycle Anthropic model request"
                ),
            }
        ],
    }
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    if shape.thinking is not None:
        payload["thinking"] = dict(shape.thinking)
    return payload


def lifecycle_anthropic_settings(
    *, model: str, deadline_seconds: float, profile: RequestProfile
) -> dict[str, Any]:
    return {
        "api": ANTHROPIC_API,
        "base_url": ANTHROPIC_BASE_URL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "request_mapping": LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_contract": "closed_exactly_one_tool_call",
        "retry": LIFECYCLE_ANTHROPIC_RETRY_POLICY.as_dict(),
        "sdk": "anthropic",
        "sdk_max_retries": 0,
        "sdk_version": anthropic.__version__,
        "tool_choice": dict(TOOL_CHOICE),
        "tool_names": [tool["name"] for tool in anthropic_outcome_tools()],
        "turn_deadline_seconds": deadline_seconds,
    }


def _invalid(detail: str) -> AdapterProviderError:
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


def model_response_from(response: WireResponse[Message]) -> ModelResponse:
    parsed = response.parsed
    if parsed.stop_reason == "max_tokens":
        return ModelResponse(
            model=parsed.model,
            stop_reason=TRUNCATED_STOP_REASON,
            output_tokens=parsed.usage.output_tokens,
        )
    if parsed.stop_reason != "tool_use":
        raise _invalid(
            "the Anthropic answer did not complete with a tool use; the provider's "
            "stop text is not retained"
        )
    calls: list[ToolCall] = []
    prose: list[str] = []
    for block in parsed.content:
        if block.type == "tool_use":
            calls.append(
                ToolCall(block.name, elide_null_optionals(block.name, block.input))
            )
        elif block.type == "text":
            prose.append(block.text)
        else:  # defended again after raw and typed closure
            raise _invalid(
                "the Anthropic answer carries an undeclared content block; its "
                "provider-controlled type is not retained"
            )
    if len(calls) != 1:
        raise _invalid(
            "the Anthropic answer must carry exactly one tool call; provider prose "
            "and call content are not retained in the failure"
        )
    return ModelResponse(
        model=parsed.model,
        stop_reason=COMPLETED_STOP_REASON,
        tool_calls=tuple(calls),
        text="\n".join(prose)[:MAX_TRANSIENT_TEXT_CHARACTERS],
        output_tokens=parsed.usage.output_tokens,
    )


def _response_id_digest(wire: WireResponse[Message] | None) -> str | None:
    return None if wire is None else wire.response_id_digest_sha256


class AnthropicMessagesTransport:
    """A Lifecycle ModelTransport using Anthropic's real SDK wire contract."""

    def __init__(
        self,
        *,
        model: str,
        client: anthropic.Anthropic,
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
        self._exchange = AnthropicMessagesExchange(
            model=model,
            client=client,
            single_flight_label="AnthropicMessagesTransport",
            retry=LIFECYCLE_ANTHROPIC_RETRY_POLICY,
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
        return ANTHROPIC_PROVIDER

    @property
    def api(self) -> str:
        return ANTHROPIC_API

    @property
    def request_mapping(self) -> str:
        return LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION

    @property
    def model(self) -> str:
        return self._model

    @property
    def settings(self) -> Mapping[str, Any]:
        return lifecycle_anthropic_settings(
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
        wire: WireResponse[Message] | None = None
        fault: str | None = None
        try:
            with self._exchange.turn():
                self._exchange.clear()
                wire = self._exchange.exchange(payload, self._deadline(started))
            answer = model_response_from(wire)
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
    "LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION",
    "LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V1",
    "LIFECYCLE_ANTHROPIC_RETRY_POLICY",
    "AnthropicMessagesTransport",
    "anthropic_outcome_tools",
    "build_model_payload",
    "lifecycle_anthropic_settings",
    "lower_anthropic_input_schema",
    "model_response_from",
]
