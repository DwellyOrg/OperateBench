"""Provider-neutral Mistral Chat Completions kernel over the official SDK.

This module owns Mistral wire, client-state, response and usage admission.  It
contains no Lifecycle or Boundary task semantics; callers provide payloads and
response validators to the shared :class:`TurnExecutor`.
"""

from __future__ import annotations

import hashlib
import struct
import threading
import time
from abc import ABCMeta
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from types import FunctionType, MappingProxyType, WrapperDescriptorType
from typing import Any, Protocol, cast

import httpx
from mistralai.client import Mistral
from mistralai.client._hooks.custom_user_agent import CustomUserAgentHook
from mistralai.client._hooks.deprecation_warning import DeprecationWarningHook
from mistralai.client._hooks.sdkhooks import SDKHooks
from mistralai.client._hooks.stream_error_hook import WorkflowStreamErrorHook
from mistralai.client._hooks.traceparent import TraceparentInjectionHook
from mistralai.client._hooks.tracing import TracingHook
from mistralai.client._hooks.workflow_encoding_hook import WorkflowEncodingHook
from mistralai.client.chat import Chat
from mistralai.client.errors import MistralError, NoResponseError, ResponseValidationError
from mistralai.client.httpclient import HttpClient
from mistralai.client.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from mistralai.client.sdkconfiguration import SDKConfiguration
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from operatebench.jsonsafe import canonical_json_bytes
from operatebench.providers.config import (
    ProviderConfigurationError,
    check_endpoint,
    check_payload_fields,
)
from operatebench.providers.cost import CostGuard, request_input_token_bound
from operatebench.providers.executor import (
    Clock,
    ProviderRetryPolicy,
    Sleep,
    TurnExecutor,
    check_retry_policy,
)
from operatebench.providers.faults import (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    AdapterProviderError,
    Fault,
    SingleFlight,
    classify_http_status,
)
from operatebench.providers.response import (
    check_response_collection,
    check_response_contract,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.usage import (
    MAX_EXACT_TOKEN_COUNT,
    TokenUsage,
    checked_token_usage,
)
from operatebench.providers.wire import (
    WireResponse,
    WireShape,
    checked_wire_list,
    checked_wire_mapping,
    checked_wire_object,
    checked_wire_string,
    declared_wire_fields,
    is_wire_token_count,
    wire_invalid,
    wire_token_usage,
)

MISTRAL_PROVIDER = "mistral"
MISTRAL_IMPLEMENTATION = "mistral_chat_completions"
MISTRAL_API = "chat_completions"
MISTRAL_API_COMPATIBILITY = "native_mistral"
MISTRAL_BASE_URL = "https://api.mistral.ai"
MISTRAL_SMALL_MODEL = "mistral-small-2603"
MISTRAL_RESPONSE_CAPTURE = "mistral_sdk_captured_http_response_v1"
MISTRAL_RESPONSE_EXTENSIONS = "mistral_response_extensions_v1"
MISTRAL_RESPONSE_STATE_POLICY = "clear_owned_httpx_response_cookies_v1"
TOOL_CHOICE = "required"
PARALLEL_TOOL_CALLS = False
TEMPERATURE = 0.0
REQUIRED_FINISH_REASON = "tool_calls"
OUTPUT_LIMIT_FINISH_REASON = "length"
SDK_INJECTED_FIELDS: Mapping[str, Any] = MappingProxyType({"stream": False})
PROHIBITED_REQUEST_FIELDS = ("presence_penalty", "random_seed", "top_p")


class ProviderDispatchObserver(Protocol):
    """Observe one SDK request attempt immediately before it begins."""

    def __call__(self) -> None: ...


_COMMON_REQUEST_FIELDS = (
    "max_tokens",
    "messages",
    "model",
    "parallel_tool_calls",
    "temperature",
    "tool_choice",
    "tools",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens")


class MistralConfigurationError(ProviderConfigurationError):
    """The Mistral SDK client cannot satisfy its recorded configuration."""


@dataclass(frozen=True)
class RequestProfile:
    profile_id: str
    temperature: float
    contract_verified: bool = False

    def sent_fields(self) -> tuple[str, ...]:
        return tuple(sorted(_COMMON_REQUEST_FIELDS))

    def omitted_fields(self) -> tuple[str, ...]:
        return PROHIBITED_REQUEST_FIELDS

    def as_settings(self) -> dict[str, Any]:
        return {
            "request_profile": self.profile_id,
            "request_fields_sent": list(self.sent_fields()),
            "request_fields_omitted": list(self.omitted_fields()),
            "temperature": self.temperature,
            "model_contract_verified": self.contract_verified,
        }


MISTRAL_SMALL_PROFILE = RequestProfile(
    "mistralsmall2603_temperature_zero_v1", TEMPERATURE
)
MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {MISTRAL_SMALL_MODEL: MISTRAL_SMALL_PROFILE}
)


def request_profile_for(model: str) -> RequestProfile:
    try:
        return MODEL_REQUEST_PROFILES[model]
    except KeyError as exc:
        raise MistralConfigurationError(
            "this Lifecycle lane has no reviewed Mistral request profile "
            "for the named model"
        ) from exc


def check_request_profile(
    profile: RequestProfile | None, *, model: str
) -> RequestProfile:
    expected = request_profile_for(model)
    if profile is None or profile == expected:
        return expected
    raise MistralConfigurationError(
        "the supplied request profile is not the pinned model profile"
    )


EXTENSION_COUNT = "non_negative_bounded_integer"
RESPONSE_EXTENSION_SCHEMA: Mapping[str, Any] = MappingProxyType(
    {"prompt_tokens_details": MappingProxyType({"cached_tokens": EXTENSION_COUNT})}
)
RESPONSE_EXTENSION_CONTRACT = {
    "contract": MISTRAL_RESPONSE_EXTENSIONS,
    "extension_values_read": False,
    "max_count": MAX_EXACT_TOKEN_COUNT,
    "members": {"prompt_tokens_details": {"cached_tokens": EXTENSION_COUNT}},
    "object_members_required": True,
    "top_level_members_required": False,
}
RESPONSE_EXTENSION_DIGEST = hashlib.sha256(
    canonical_json_bytes(RESPONSE_EXTENSION_CONTRACT, "mistral response extensions")
).hexdigest()
USAGE_EXTENSION_NAMES = frozenset(RESPONSE_EXTENSION_SCHEMA)
TYPED_EXTENSION_EXTRAS: Mapping[type, frozenset[str]] = MappingProxyType(
    {UsageInfo: USAGE_EXTENSION_NAMES}
)


def check_response_extensions(node: Mapping[str, Any]) -> None:
    if "prompt_tokens_details" not in node:
        return
    details = checked_wire_mapping(
        node["prompt_tokens_details"], kind="named response extension"
    )
    if set(details) != {"cached_tokens"}:
        raise wire_invalid(
            "the Mistral response extension does not state its exact closed member set"
        )
    count = details["cached_tokens"]
    if not is_wire_token_count(count) or count > MAX_EXACT_TOKEN_COUNT:
        raise wire_invalid(
            "the Mistral response extension count is not an exact bounded integer"
        )


def _typed_extras(value: Any) -> Mapping[str, Any]:
    extra = getattr(value, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}


def check_response_extensions_agree(
    wire: Mapping[str, Any], parsed: ChatCompletionResponse
) -> None:
    raw_usage = wire.get("usage")
    raw = raw_usage if isinstance(raw_usage, Mapping) else {}
    typed = _typed_extras(parsed.usage)
    check_response_extensions(typed)
    raw_extension = {name: raw[name] for name in USAGE_EXTENSION_NAMES if name in raw}
    typed_extension = {
        name: typed[name] for name in USAGE_EXTENSION_NAMES if name in typed
    }
    if raw_extension != typed_extension:
        raise wire_invalid(
            "the Mistral wire and typed response extension readings disagree"
        )


RESPONSE_WIRE_SHAPE = WireShape(
    "completion", declared_wire_fields(ChatCompletionResponse), ("model", "usage")
)
CHOICE_WIRE_SHAPE = WireShape(
    "completion choice",
    declared_wire_fields(ChatCompletionChoice),
    ("finish_reason", "message"),
)
MESSAGE_WIRE_SHAPE = WireShape(
    "completion message", declared_wire_fields(AssistantMessage), ()
)
TOOL_CALL_WIRE_SHAPE = WireShape(
    "tool call", declared_wire_fields(ToolCall), ("function",)
)
FUNCTION_WIRE_SHAPE = WireShape(
    "tool call function object", declared_wire_fields(FunctionCall), ("name", "arguments")
)
USAGE_WIRE_SHAPE = WireShape(
    "usage block", declared_wire_fields(UsageInfo) | USAGE_EXTENSION_NAMES, USAGE_FIELDS
)


def check_wire_completion(raw: Mapping[str, Any]) -> None:
    body = checked_wire_object(raw, RESPONSE_WIRE_SHAPE)
    for entry in checked_wire_list(body.get("choices") or [], kind="completion choices"):
        choice = checked_wire_object(entry, CHOICE_WIRE_SHAPE)
        if choice["message"] is None:
            continue
        message = checked_wire_object(choice["message"], MESSAGE_WIRE_SHAPE)
        if message.get("tool_calls") is None:
            continue
        for raw_call in checked_wire_list(message["tool_calls"], kind="tool calls"):
            call = checked_wire_mapping(raw_call, kind="tool call")
            if call.get("type") not in (None, "function"):
                continue
            function = checked_wire_object(call, TOOL_CALL_WIRE_SHAPE)["function"]
            named = checked_wire_object(function, FUNCTION_WIRE_SHAPE)
            checked_wire_string(named["name"], kind="tool call name")
    usage = checked_wire_object(body["usage"], USAGE_WIRE_SHAPE)
    check_response_extensions(usage)
    wire_token_usage(usage, fields=USAGE_FIELDS)


def check_response_model(response: ChatCompletionResponse, *, model: str) -> None:
    if response.model != model:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the response states a model other than the pinned requested model",
        )


def check_response_admissible(
    response: WireResponse[ChatCompletionResponse], *, model: str
) -> None:
    check_wire_completion(response.wire)
    parsed = response.parsed
    check_response_model(parsed, model=model)
    check_response_contract(parsed, allowed_extras_by_model=TYPED_EXTENSION_EXTRAS)
    check_response_extensions_agree(response.wire, parsed)
    if parsed.choices is not None:
        check_response_collection(parsed.choices, kind="completion choices")
    if parsed.usage is not None:
        checked_token_usage(parsed.usage.prompt_tokens, parsed.usage.completion_tokens)


def response_usage(response: WireResponse[ChatCompletionResponse]) -> TokenUsage:
    usage = checked_wire_object(response.wire.get("usage"), USAGE_WIRE_SHAPE)
    return wire_token_usage(usage, fields=USAGE_FIELDS)


def classify_exception(exception: Exception) -> Fault | None:
    if isinstance(exception, httpx.TimeoutException):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, (httpx.TransportError, NoResponseError)):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, ResponseValidationError):
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, MistralError):
        if 200 <= exception.status_code < 300:
            return Fault(
                PROVIDER_FAULT_RESPONSE_INVALID,
                False,
                None,
                response_received=True,
            )
        return classify_http_status(exception.status_code)
    return None


def _no_sdk_retries() -> RetryConfig:
    """A fresh immutable-by-ownership zero-retry configuration per client."""
    return RetryConfig("none", BackoffStrategy(1, 1, 1, 1), False)


def build_client(*, api_key: str, http_client: httpx.Client | None = None) -> Mistral:
    return Mistral(
        api_key=api_key,
        server_url=MISTRAL_BASE_URL,
        client=http_client,
        retry_config=_no_sdk_retries(),
    )


class CapturedHttpClient:
    """SDK HttpClient delegate retaining exactly one in-flight raw response."""

    def __init__(self, inner: HttpClient) -> None:
        self.inner = inner
        self._captured: list[httpx.Response] = []
        self._owner: object | None = None
        self._claim_lock = threading.Lock()

    def claim(self, owner: object) -> None:
        """Bind the capture to one owner by object identity for its lifetime."""
        with self._claim_lock:
            if self._owner is None or self._owner is owner:
                self._owner = owner
                return
        raise MistralConfigurationError(
            "the Mistral response capture belongs to another exchange; "
            "separate exchanges require separate SDK clients"
        )

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        response = self.inner.send(request, **kwargs)
        self._captured.append(response)
        return response

    def build_request(self, method: str, url: Any, **kwargs: Any) -> httpx.Request:
        return self.inner.build_request(method, url, **kwargs)

    def close(self) -> None:
        self.inner.close()

    def clear(self) -> None:
        self._captured.clear()

    def take(self) -> httpx.Response:
        if len(self._captured) != 1:
            raise wire_invalid(
                "the Mistral SDK call did not capture exactly one HTTP response"
            )
        return self._captured[0]


_EXPECTED_SDK_HOOK_TYPES: Mapping[str, tuple[type[Any], ...]] = {
    "sdk_init_hooks": (),
    "before_request_hooks": (
        CustomUserAgentHook,
        TraceparentInjectionHook,
        TracingHook,
        WorkflowEncodingHook,
    ),
    "after_success_hooks": (
        DeprecationWarningHook,
        TracingHook,
        WorkflowEncodingHook,
        WorkflowStreamErrorHook,
    ),
    "after_error_hooks": (TracingHook,),
}


@dataclass(frozen=True, slots=True)
class _ClientIdentitySeal:
    client: Mistral
    configuration: SDKConfiguration
    chat: Chat
    capture: CapturedHttpClient
    inner: HttpClient
    cookies: httpx.Cookies
    hooks: SDKHooks
    hook_groups: tuple[tuple[str, list[Any], tuple[Any, ...]], ...]
    class_mros: tuple[_RawClassMroSeal, ...]
    class_namespaces: tuple[_RawClassNamespaceSeal, ...]
    execution_surfaces: tuple[_ExecutionSurfaceSeal, ...]


@dataclass(frozen=True, slots=True)
class _ExecutionSurfaceSeal:
    instance: object
    exact_type: type[Any]
    descriptors: tuple[tuple[str, type[Any], object, _FunctionBehaviorSeal | None], ...]


@dataclass(frozen=True, slots=True)
class _RawClassNamespaceSeal:
    owner: type[Any]
    keys: frozenset[str]
    values: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _RawClassMroSeal:
    exact_type: type[Any]
    mro: tuple[type[Any], ...]


@dataclass(frozen=True, slots=True)
class _ValueSnapshot:
    kind: str
    identity: object
    value: object | None = None
    items: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class _FunctionBehaviorSeal:
    function: FunctionType
    code: object
    defaults: tuple[Any, ...] | None
    defaults_snapshot: _ValueSnapshot
    kwdefaults: dict[str, Any] | None
    kwdefaults_snapshot: _ValueSnapshot
    closure: tuple[Any, ...] | None
    closure_cells: tuple[tuple[object, _ValueSnapshot], ...]


_CHAT_EXECUTION_METHODS = (
    "complete",
    "_get_url",
    "_build_request",
    "_build_request_with_client",
    "do_request",
)
_CAPTURE_EXECUTION_METHODS = ("claim", "build_request", "send", "clear", "take")
_HTTPX_EXECUTION_METHODS = (
    "build_request",
    "request",
    "send",
    "_merge_url",
    "_merge_headers",
    "_merge_cookies",
    "_merge_queryparams",
    "_set_timeout",
    "_build_request_auth",
    "_send_handling_auth",
    "_send_handling_redirects",
    "_send_single_request",
    "_transport_for_url",
    "_build_redirect_request",
    "_redirect_method",
    "_redirect_url",
    "_redirect_headers",
    "_redirect_stream",
)
_COOKIE_EXECUTION_METHODS = ("clear",)
_SDK_HOOK_EXECUTION_METHODS: Mapping[str, str] = {
    "before_request_hooks": "before_request",
    "after_success_hooks": "after_success",
    "after_error_hooks": "after_error",
}


def _instance_state(instance: object) -> dict[str, Any]:
    state = object.__getattribute__(instance, "__dict__")
    if type(state) is not dict:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return cast("dict[str, Any]", state)


def _raw_descriptor(
    exact_type: type[Any], name: str
) -> tuple[type[Any] | None, object | None]:
    for owner in type.__getattribute__(exact_type, "__mro__"):
        namespace = type.__getattribute__(owner, "__dict__")
        if name in namespace:
            return owner, namespace[name]
    return None, None


def _resolve_descriptor(exact_type: type[Any], name: str) -> tuple[type[Any], object]:
    owner, descriptor = _raw_descriptor(exact_type, name)
    if owner is None:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return owner, descriptor


def _raw_class_namespace(owner: type[Any]) -> _RawClassNamespaceSeal:
    """Take a stable raw snapshot without binding or inspecting any descriptor."""
    if type(owner) not in (type, ABCMeta):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    namespace = type.__getattribute__(owner, "__dict__")
    if type(namespace) is not MappingProxyType:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    try:
        keys = tuple(namespace)
    except RuntimeError as exc:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        ) from exc
    if any(type(key) is not str for key in keys):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    trusted_keys = cast("tuple[str, ...]", keys)
    try:
        values = tuple((key, namespace[key]) for key in trusted_keys)
        final_keys = tuple(namespace)
    except (KeyError, RuntimeError) as exc:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        ) from exc
    if any(type(key) is not str for key in final_keys):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    try:
        changed = frozenset(final_keys) != frozenset(trusted_keys) or any(
            namespace[key] is not value for key, value in values
        )
    except KeyError as exc:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        ) from exc
    if changed:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return _RawClassNamespaceSeal(
        owner=owner,
        keys=frozenset(trusted_keys),
        values=values,
    )


def _raw_class_mro(exact_type: type[Any]) -> tuple[type[Any], ...]:
    if type(exact_type) not in (type, ABCMeta):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    mro = type.__getattribute__(exact_type, "__mro__")
    if (
        type(mro) is not tuple
        or not mro
        or mro[0] is not exact_type
        or any(type(owner) not in (type, ABCMeta) for owner in mro)
    ):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return cast("tuple[type[Any], ...]", mro)


def _raw_class_seals(
    exact_types: tuple[type[Any], ...],
) -> tuple[tuple[_RawClassMroSeal, ...], tuple[_RawClassNamespaceSeal, ...]]:
    owners: list[type[Any]] = []
    mro_seals: list[_RawClassMroSeal] = []
    for exact_type in exact_types:
        mro = _raw_class_mro(exact_type)
        mro_seals.append(_RawClassMroSeal(exact_type=exact_type, mro=mro))
        for owner in mro:
            if not any(owner is admitted for admitted in owners):
                owners.append(owner)
    namespaces = tuple(_raw_class_namespace(owner) for owner in owners)
    _check_raw_class_mros(tuple(mro_seals))
    return tuple(mro_seals), namespaces


def _check_raw_class_mros(seals: tuple[_RawClassMroSeal, ...]) -> None:
    for seal in seals:
        current = _raw_class_mro(seal.exact_type)
        if len(current) != len(seal.mro) or any(
            owner is not expected
            for owner, expected in zip(current, seal.mro, strict=True)
        ):
            raise MistralConfigurationError(
                "the Mistral SDK client identity has changed since construction"
            )


def _check_raw_class_namespaces(
    seals: tuple[_RawClassNamespaceSeal, ...],
) -> None:
    for seal in seals:
        current = _raw_class_namespace(seal.owner)
        current_values = dict(current.values)
        if current.keys != seal.keys or any(
            current_values[name] is not expected_value
            for name, expected_value in seal.values
        ):
            raise MistralConfigurationError(
                "the Mistral SDK client identity has changed since construction"
            )


def _state_value(instance: object, name: str) -> Any:
    state = _instance_state(instance)
    if name not in state:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return state[name]


def _snapshot_value(value: object, *, budget: list[int] | None = None) -> _ValueSnapshot:
    remaining = [4096] if budget is None else budget
    remaining[0] -= 1
    if remaining[0] < 0:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    exact = type(value)
    if value is None or exact in (bool, int, str, bytes):
        return _ValueSnapshot("scalar", value, value)
    if exact is float:
        return _ValueSnapshot("scalar", value, struct.pack("!d", value))
    if exact in (tuple, list):
        sequence = cast("tuple[Any, ...] | list[Any]", value)
        return _ValueSnapshot(
            exact.__name__,
            value,
            items=tuple(_snapshot_value(item, budget=remaining) for item in sequence),
        )
    if exact is dict:
        mapping = cast("dict[Any, Any]", value)
        return _ValueSnapshot(
            "dict",
            value,
            items=tuple(
                (
                    _snapshot_value(key, budget=remaining),
                    _snapshot_value(item, budget=remaining),
                )
                for key, item in mapping.items()
            ),
        )
    if exact in (set, frozenset):
        values = cast("set[Any] | frozenset[Any]", value)
        return _ValueSnapshot(
            exact.__name__,
            value,
            items=tuple(_snapshot_value(item, budget=remaining) for item in values),
        )
    return _ValueSnapshot("identity", value)


def _snapshot_matches(
    value: object, snapshot: _ValueSnapshot, *, budget: list[int] | None = None
) -> bool:
    remaining = [4096] if budget is None else budget
    remaining[0] -= 1
    if remaining[0] < 0:
        return False
    if snapshot.kind == "identity":
        return value is snapshot.identity
    if snapshot.kind == "scalar":
        if type(value) is not type(snapshot.identity):
            return False
        if type(value) is float:
            return struct.pack("!d", value) == snapshot.value
        return value == snapshot.value
    if snapshot.kind in ("tuple", "list"):
        if type(value).__name__ != snapshot.kind or value is not snapshot.identity:
            return False
        sequence = cast("tuple[Any, ...] | list[Any]", value)
        if len(sequence) != len(snapshot.items):
            return False
        return all(
            _snapshot_matches(item, cast("_ValueSnapshot", expected), budget=remaining)
            for item, expected in zip(sequence, snapshot.items, strict=True)
        )
    if snapshot.kind == "dict":
        if type(value) is not dict or value is not snapshot.identity:
            return False
        current = tuple(value.items())
        if len(current) != len(snapshot.items):
            return False
        return all(
            _snapshot_matches(
                key,
                cast("tuple[_ValueSnapshot, _ValueSnapshot]", expected)[0],
                budget=remaining,
            )
            and _snapshot_matches(
                item,
                cast("tuple[_ValueSnapshot, _ValueSnapshot]", expected)[1],
                budget=remaining,
            )
            for (key, item), expected in zip(current, snapshot.items, strict=True)
        )
    if snapshot.kind in ("set", "frozenset"):
        if type(value).__name__ != snapshot.kind or value is not snapshot.identity:
            return False
        set_current = list(cast("set[Any] | frozenset[Any]", value))
        if len(set_current) != len(snapshot.items):
            return False
        unmatched = list(snapshot.items)
        for item in set_current:
            for index, expected in enumerate(unmatched):
                local_budget = [remaining[0]]
                if _snapshot_matches(
                    item, cast("_ValueSnapshot", expected), budget=local_budget
                ):
                    remaining[0] = local_budget[0]
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    return False


def _function_behavior(function: FunctionType) -> _FunctionBehaviorSeal:
    closure = function.__closure__
    cells: list[tuple[object, _ValueSnapshot]] = []
    if closure is not None:
        for cell in closure:
            try:
                content = cell.cell_contents
            except ValueError:
                content = cell
            cells.append((cell, _snapshot_value(content)))
    return _FunctionBehaviorSeal(
        function=function,
        code=function.__code__,
        defaults=function.__defaults__,
        defaults_snapshot=_snapshot_value(function.__defaults__),
        kwdefaults=function.__kwdefaults__,
        kwdefaults_snapshot=_snapshot_value(function.__kwdefaults__),
        closure=closure,
        closure_cells=tuple(cells),
    )


def _descriptor_behavior(descriptor: object) -> _FunctionBehaviorSeal | None:
    if type(descriptor) is FunctionType:
        return _function_behavior(descriptor)
    if type(descriptor) is WrapperDescriptorType:
        return None
    raise MistralConfigurationError(
        "the Mistral SDK execution surface cannot be verified"
    )


def _check_function_behavior(seal: _FunctionBehaviorSeal) -> bool:
    function = seal.function
    if (
        function.__code__ is not seal.code
        or function.__defaults__ is not seal.defaults
        or not _snapshot_matches(function.__defaults__, seal.defaults_snapshot)
        or function.__kwdefaults__ is not seal.kwdefaults
        or not _snapshot_matches(function.__kwdefaults__, seal.kwdefaults_snapshot)
        or function.__closure__ is not seal.closure
    ):
        return False
    closure = function.__closure__
    if closure is None:
        return not seal.closure_cells
    if len(closure) != len(seal.closure_cells):
        return False
    for cell, (expected_cell, expected_content) in zip(
        closure, seal.closure_cells, strict=True
    ):
        if cell is not expected_cell:
            return False
        try:
            content = cell.cell_contents
        except ValueError:
            content = cell
        if not _snapshot_matches(content, expected_content):
            return False
    return True


def _execution_surface(
    instance: object, exact_type: type[Any], names: tuple[str, ...]
) -> _ExecutionSurfaceSeal:
    if type(instance) is not exact_type:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    names = ("__getattribute__", *names)
    state = _instance_state(instance)
    if any(name in state for name in names):
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    return _ExecutionSurfaceSeal(
        instance=instance,
        exact_type=exact_type,
        descriptors=tuple(
            (name, owner, descriptor, _descriptor_behavior(descriptor))
            for name in names
            for owner, descriptor in (_resolve_descriptor(exact_type, name),)
        ),
    )


def _check_execution_surface(seal: _ExecutionSurfaceSeal) -> None:
    if type(seal.instance) is not seal.exact_type:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface has changed since construction"
        )
    state = _instance_state(seal.instance)
    for name, expected_owner, expected_descriptor, behavior in seal.descriptors:
        owner, descriptor = _resolve_descriptor(seal.exact_type, name)
        if (
            name in state
            or owner is not expected_owner
            or descriptor is not expected_descriptor
            or (behavior is not None and not _check_function_behavior(behavior))
        ):
            raise MistralConfigurationError(
                "the Mistral SDK execution surface has changed since construction"
            )


def _execution_surfaces(
    *,
    client: Mistral,
    configuration: SDKConfiguration,
    chat: Chat,
    capture: CapturedHttpClient | None,
    inner: httpx.Client,
    cookies: httpx.Cookies,
    hooks: SDKHooks,
    hook_groups: Mapping[str, list[Any]],
) -> tuple[_ExecutionSurfaceSeal, ...]:
    surfaces = [
        _execution_surface(client, Mistral, ()),
        _execution_surface(chat, Chat, _CHAT_EXECUTION_METHODS),
        _execution_surface(configuration, SDKConfiguration, ("get_server_details",)),
        _execution_surface(
            hooks, SDKHooks, ("before_request", "after_success", "after_error")
        ),
        _execution_surface(inner, httpx.Client, _HTTPX_EXECUTION_METHODS),
        _execution_surface(cookies, httpx.Cookies, _COOKIE_EXECUTION_METHODS),
    ]
    if capture is not None:
        surfaces.append(
            _execution_surface(capture, CapturedHttpClient, _CAPTURE_EXECUTION_METHODS)
        )
    for group_name, method_name in _SDK_HOOK_EXECUTION_METHODS.items():
        for hook in hook_groups[group_name]:
            surfaces.append(_execution_surface(hook, type(hook), (method_name,)))
    return tuple(surfaces)


def _http_client(client: Mistral) -> httpx.Client:
    if type(client) is not Mistral:
        raise MistralConfigurationError(
            "the Mistral client has no auditable synchronous httpx client"
        )
    configuration = _state_value(client, "sdk_configuration")
    if type(configuration) is not SDKConfiguration:
        raise MistralConfigurationError(
            "the Mistral client has no auditable synchronous httpx client"
        )
    configured = _state_value(configuration, "client")
    if type(configured) is CapturedHttpClient:
        configured = _state_value(configured, "inner")
    if type(configured) is not httpx.Client:
        raise MistralConfigurationError(
            "the Mistral client has no auditable synchronous httpx client"
        )
    return configured


def _client_class_seals(
    *,
    configuration: SDKConfiguration,
    hook_groups: Mapping[str, list[Any]],
) -> tuple[tuple[_RawClassMroSeal, ...], tuple[_RawClassNamespaceSeal, ...]]:
    retry = _state_value(configuration, "retry_config")
    if type(retry) is not RetryConfig:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    backoff = _state_value(retry, "backoff")
    if type(backoff) is not BackoffStrategy:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    hook_types = tuple(type(hook) for group in hook_groups.values() for hook in group)
    return _raw_class_seals(
        (
            Mistral,
            Chat,
            SDKConfiguration,
            SDKHooks,
            CapturedHttpClient,
            httpx.Client,
            httpx.Cookies,
            RetryConfig,
            BackoffStrategy,
            *hook_types,
        )
    )


def check_client_state(
    client: Mistral,
    *,
    seal: _ClientIdentitySeal | None = None,
    allow_response_cookies: bool = False,
) -> None:
    if type(client) is not Mistral:
        raise MistralConfigurationError("the Mistral client exposes no SDK configuration")
    if seal is not None:
        _check_raw_class_mros(seal.class_mros)
        _check_raw_class_namespaces(seal.class_namespaces)
        for execution_surface in seal.execution_surfaces:
            _check_execution_surface(execution_surface)
    configuration = _state_value(client, "sdk_configuration")
    client_state = _instance_state(client)
    if "chat" not in client_state:
        if seal is not None or _raw_descriptor(Mistral, "chat") != (None, None):
            raise MistralConfigurationError(
                "the Mistral client exposes no SDK configuration"
            )
        created_chat = client.chat
        del created_chat
    chat = _state_value(client, "chat")
    if (
        type(configuration) is not SDKConfiguration
        or type(chat) is not Chat
        or _state_value(chat, "sdk_configuration") is not configuration
        or _state_value(chat, "parent_ref") is not client
    ):
        raise MistralConfigurationError("the Mistral client exposes no SDK configuration")
    _execution_surface(chat, Chat, _CHAT_EXECUTION_METHODS)
    _execution_surface(configuration, SDKConfiguration, ("get_server_details",))
    capture = _state_value(configuration, "client")
    hooks = _state_value(configuration, "_hooks")
    if seal is not None and (
        client is not seal.client
        or configuration is not seal.configuration
        or chat is not seal.chat
        or type(seal.capture) is not CapturedHttpClient
        or type(seal.inner) is not httpx.Client
        or capture is not seal.capture
        or _state_value(seal.capture, "inner") is not seal.inner
        or hooks is not seal.hooks
        or any(
            _state_value(seal.hooks, name) is not group
            or len(group) != len(members)
            or any(
                current is not expected
                for current, expected in zip(group, members, strict=True)
            )
            for name, group, members in seal.hook_groups
        )
    ):
        raise MistralConfigurationError(
            "the Mistral SDK client identity has changed since construction"
        )
    owner, details_descriptor = _resolve_descriptor(
        SDKConfiguration, "get_server_details"
    )
    del owner
    if type(details_descriptor) is not FunctionType:
        raise MistralConfigurationError(
            "the Mistral client exposes no effective endpoint"
        )
    details = details_descriptor.__get__(configuration, SDKConfiguration)
    check_endpoint(
        details()[0],
        expected=MISTRAL_BASE_URL,
        provider=MISTRAL_PROVIDER,
        setting="base_url",
        error=MistralConfigurationError,
    )
    retry = _state_value(configuration, "retry_config")
    if type(retry) is not RetryConfig:
        raise MistralConfigurationError(
            "the Mistral SDK retry configuration must be exactly disabled"
        )
    backoff = _state_value(retry, "backoff")
    if (
        _state_value(retry, "strategy") != "none"
        or _state_value(retry, "retry_connection_errors") is not False
        or type(backoff) is not BackoffStrategy
        or type(_state_value(backoff, "initial_interval")) is not int
        or _state_value(backoff, "initial_interval") != 1
        or type(_state_value(backoff, "max_interval")) is not int
        or _state_value(backoff, "max_interval") != 1
        or type(_state_value(backoff, "exponent")) is not int
        or _state_value(backoff, "exponent") != 1
        or type(_state_value(backoff, "max_elapsed_time")) is not int
        or _state_value(backoff, "max_elapsed_time") != 1
    ):
        raise MistralConfigurationError(
            "the Mistral SDK retry configuration must be exactly disabled"
        )
    if type(hooks) is not SDKHooks:
        raise MistralConfigurationError(
            "the Mistral SDK carries non-default mutable hooks"
        )
    hook_groups: dict[str, list[Any]] = {}
    for name, expected in _EXPECTED_SDK_HOOK_TYPES.items():
        values = _state_value(hooks, name)
        if type(values) is not list or tuple(type(value) for value in values) != expected:
            raise MistralConfigurationError(
                "the Mistral SDK carries non-default mutable hooks"
            )
        hook_groups[name] = values
    before = hook_groups["before_request_hooks"]
    after_success = hook_groups["after_success_hooks"]
    after_error = hook_groups["after_error_hooks"]
    tracing = before[2]
    empty_state_hooks = (
        before[0],
        before[1],
        before[3],
        after_success[0],
        after_success[3],
    )
    tracing_state = _instance_state(tracing)
    if (
        after_success[1] is not tracing
        or after_error[0] is not tracing
        or after_success[2] is not before[3]
        or any(_instance_state(hook) for hook in empty_state_hooks)
        or set(tracing_state)
        != {
            "tracer_provider",
            "_auto_telemetry_provider",
            "_telemetry_finalizer",
            "_telemetry_auto_disabled",
            "_telemetry_use_global_provider",
            "tracing_enabled",
            "tracer",
        }
        or tracing_state["tracer_provider"] is not None
        or tracing_state["_auto_telemetry_provider"] is not None
        or tracing_state["_telemetry_finalizer"] is not None
        or type(tracing_state["_telemetry_auto_disabled"]) is not bool
        or tracing_state["_telemetry_use_global_provider"] is not False
        or tracing_state["tracing_enabled"] is not False
        or type(tracing_state["tracer"]).__name__ != "ProxyTracer"
        or type(tracing_state["tracer"]).__module__ != "opentelemetry.trace"
    ):
        raise MistralConfigurationError(
            "the Mistral SDK carries non-default mutable hook state"
        )
    http_client = _http_client(client)
    cookies = _state_value(http_client, "_cookies")
    if type(cookies) is not httpx.Cookies or (
        seal is not None and cookies is not seal.cookies
    ):
        raise MistralConfigurationError(
            "the Mistral HTTP client cookie jar identity has changed"
        )
    _execution_surfaces(
        client=client,
        configuration=configuration,
        chat=chat,
        capture=capture if type(capture) is CapturedHttpClient else None,
        inner=http_client,
        cookies=cookies,
        hooks=hooks,
        hook_groups=hook_groups,
    )
    with httpx.Client(trust_env=False) as pristine:
        if _state_value(http_client, "_headers") != _state_value(pristine, "_headers"):
            raise MistralConfigurationError(
                "the Mistral HTTP client carries non-default headers"
            )
        if _state_value(http_client, "_event_hooks") != _state_value(
            pristine, "_event_hooks"
        ):
            raise MistralConfigurationError(
                "the Mistral HTTP client carries non-default event hooks"
            )
    if (
        _state_value(http_client, "_params") != httpx.QueryParams()
        or _state_value(http_client, "_auth") is not None
        or (cookies and not allow_response_cookies)
    ):
        raise MistralConfigurationError(
            "the Mistral HTTP client carries mutable request state"
        )


_CAPTURE_INSTALL_LOCK = threading.Lock()


def ensure_capturing_client(client: Mistral, *, owner: object) -> CapturedHttpClient:
    """Atomically wrap, identity-claim, and install the client's HTTP capture."""
    with _CAPTURE_INSTALL_LOCK:
        if type(client) is not Mistral:
            raise MistralConfigurationError(
                "the Mistral client has no capturable HTTP client"
            )
        configuration = _state_value(client, "sdk_configuration")
        if type(configuration) is not SDKConfiguration:
            raise MistralConfigurationError(
                "the Mistral client has no capturable HTTP client"
            )
        current = _state_value(configuration, "client")
        if type(current) is CapturedHttpClient:
            if type(_state_value(current, "inner")) is not httpx.Client:
                raise MistralConfigurationError(
                    "the Mistral client has no capturable HTTP client"
                )
            current.claim(owner)
            return current
        if type(current) is not httpx.Client:
            raise MistralConfigurationError(
                "the Mistral client has no capturable HTTP client"
            )
        wrapper = CapturedHttpClient(current)
        wrapper.claim(owner)
        _instance_state(configuration)["client"] = wrapper
        return wrapper


def _seal_client_identity(
    client: Mistral, capture: CapturedHttpClient
) -> _ClientIdentitySeal:
    if type(capture) is not CapturedHttpClient:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    inner = _state_value(capture, "inner")
    if type(inner) is not httpx.Client:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    cookies = _state_value(inner, "_cookies")
    if type(cookies) is not httpx.Cookies:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    configuration = _state_value(client, "sdk_configuration")
    chat = _state_value(client, "chat")
    if type(configuration) is not SDKConfiguration or type(chat) is not Chat:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    hooks = _state_value(configuration, "_hooks")
    if type(hooks) is not SDKHooks:
        raise MistralConfigurationError(
            "the Mistral SDK execution surface cannot be verified"
        )
    hook_groups = {
        name: cast("list[Any]", _state_value(hooks, name))
        for name in _EXPECTED_SDK_HOOK_TYPES
    }
    class_mros, class_namespaces = _client_class_seals(
        configuration=configuration,
        hook_groups=hook_groups,
    )
    return _ClientIdentitySeal(
        client=client,
        configuration=configuration,
        chat=chat,
        capture=capture,
        inner=inner,
        cookies=cookies,
        hooks=hooks,
        hook_groups=tuple(
            (name, group, tuple(group)) for name, group in hook_groups.items()
        ),
        class_mros=class_mros,
        class_namespaces=class_namespaces,
        execution_surfaces=_execution_surfaces(
            client=client,
            configuration=configuration,
            chat=chat,
            capture=capture,
            inner=inner,
            cookies=cookies,
            hooks=hooks,
            hook_groups=hook_groups,
        ),
    )


class MistralChatExchange:
    """One pinned native Mistral SDK exchange."""

    def __init__(
        self,
        *,
        model: str,
        client: Mistral,
        single_flight_label: str,
        retry: ProviderRetryPolicy,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        self._profile = check_request_profile(profile, model=model)
        check_client_state(client)
        self._retry = check_retry_policy(
            retry, provider=MISTRAL_PROVIDER, error=MistralConfigurationError
        )
        self._model = model
        self._client = client
        self._capture = ensure_capturing_client(client, owner=self)
        self._identity_seal = _seal_client_identity(client, self._capture)
        check_client_state(client, seal=self._identity_seal)
        self._executor: TurnExecutor[WireResponse[ChatCompletionResponse]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        self._clock = clock
        self._single_flight = SingleFlight(adapter=single_flight_label)
        if before_dispatch is not None and not callable(before_dispatch):
            raise MistralConfigurationError(
                "the provider dispatch observer must be a zero-argument callable"
            )
        self._before_dispatch = before_dispatch

    @property
    def request_profile(self) -> RequestProfile:
        return self._profile

    def turn(self) -> SingleFlight:
        return self._single_flight

    def clear(self) -> None:
        self._executor.clear()

    def last_usage(self) -> AdapterUsage:
        return self._executor.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._executor.last_telemetry()

    def exchange(
        self,
        payload: Mapping[str, Any],
        deadline: TurnDeadline,
        *,
        response_validator: Callable[[WireResponse[ChatCompletionResponse]], None]
        | None = None,
    ) -> WireResponse[ChatCompletionResponse]:
        admission_started = self._clock()
        try:
            check_client_state(self._client, seal=self._identity_seal)
            self._capture = ensure_capturing_client(self._client, owner=self)
        except MistralConfigurationError:
            self._executor.record_pre_dispatch_refusal(started_at=admission_started)
            raise
        check_payload_fields(
            payload,
            profile_id=self._profile.profile_id,
            sent=self._profile.sent_fields(),
            omitted=self._profile.omitted_fields(),
        )

        def admit(result: WireResponse[ChatCompletionResponse]) -> None:
            self._reconcile_post_response_state()
            check_response_admissible(result, model=self._model)
            result.admit_response_id()
            if response_validator is not None:
                response_validator(result)

        return self._executor.run(
            payload=payload,
            deadline=deadline,
            token_bound=request_input_token_bound(payload, "Mistral request"),
            dispatch=lambda timeout: self._dispatch(payload, timeout),
            check_before_dispatch=self._check_before_dispatch,
            classify=classify_exception,
            check_identity=admit,
            usage_of=response_usage,
        )

    def _reconcile_post_response_state(self) -> None:
        """Erase only response cookies, then revalidate every sealed client field."""
        try:
            # Only the exact cookie jar sealed while empty may acquire provider
            # response state. All other mutable request surfaces remain immutable.
            check_client_state(
                self._client,
                seal=self._identity_seal,
                allow_response_cookies=True,
            )
            self._identity_seal.cookies.clear()
            check_client_state(self._client, seal=self._identity_seal)
        except Exception as exc:
            raise AdapterProviderError(
                PROVIDER_FAULT_RESPONSE_INVALID,
                "the Mistral response left client request state outside the pinned "
                "post-response policy",
            ) from exc

    def _check_before_dispatch(self) -> None:
        # This is the adapter's final mutation checkpoint. A compromised-runtime
        # race beginning after it is outside the synchronous adapter boundary.
        check_client_state(self._client, seal=self._identity_seal)
        self._capture = ensure_capturing_client(self._client, owner=self)
        if self._before_dispatch is not None:
            self._before_dispatch()

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[ChatCompletionResponse]:
        self._capture.clear()
        try:
            response = self._client.chat.complete(
                **payload, timeout_ms=max(1, int(timeout * 1000))
            )
            if response is None:
                raise NoResponseError("the SDK returned no response object")
            text = self._capture.take().text
        finally:
            self._capture.clear()
        return WireResponse(text, lambda: response, kind="Mistral completion")


def sdk_version() -> str:
    return distribution_version("mistralai")


__all__ = [
    "MISTRAL_API",
    "MISTRAL_API_COMPATIBILITY",
    "MISTRAL_BASE_URL",
    "MISTRAL_IMPLEMENTATION",
    "MISTRAL_PROVIDER",
    "MISTRAL_RESPONSE_CAPTURE",
    "MISTRAL_RESPONSE_EXTENSIONS",
    "MISTRAL_RESPONSE_STATE_POLICY",
    "MISTRAL_SMALL_MODEL",
    "MISTRAL_SMALL_PROFILE",
    "OUTPUT_LIMIT_FINISH_REASON",
    "PARALLEL_TOOL_CALLS",
    "REQUIRED_FINISH_REASON",
    "RESPONSE_EXTENSION_DIGEST",
    "SDK_INJECTED_FIELDS",
    "TOOL_CHOICE",
    "MistralChatExchange",
    "MistralConfigurationError",
    "ProviderDispatchObserver",
    "RequestProfile",
    "build_client",
    "check_request_profile",
    "request_profile_for",
    "sdk_version",
]
