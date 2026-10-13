"""Meta Standard Responses exchange using the common executor and SDK reader.

Independent snapshots only. No server state, built-in tools or continuity.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
import openai
from openai.types.responses import Response

from operatebench.providers.config import (
    ProviderConfigurationError,
    check_endpoint,
    check_environment,
    resolve_api_key,
)
from operatebench.providers.cost import CostGuard, request_input_token_bound
from operatebench.providers.executor import (
    Clock,
    ProviderRetryPolicy,
    Sleep,
    TurnExecutor,
)
from operatebench.providers.faults import SingleFlight
from operatebench.providers.openai_responses import (
    ProviderDispatchObserver,
    classify_exception,
    response_usage,
)
from operatebench.providers.openai_responses import (
    check_response_admissible as check_sdk_response,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.wire import WireResponse, is_wire_token_count, wire_invalid

PROVIDER = "meta"
BASE_URL = "https://api.meta.ai/v1"
MODELS = ("muse-wonder-1.0", "muse-spark-1.3")
API_KEY_VARIABLE = "MODEL_API_KEY"
REJECTED_VARIABLES = ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")


class MetaConfigurationError(ProviderConfigurationError):
    """A Meta configuration refused before dispatch; never quotes caller values."""


def check_model(model: str) -> str:
    if not isinstance(model, str) or model not in MODELS:
        raise MetaConfigurationError("Meta requires an exact supported Standard model ID")
    return model


def check_key(key: object) -> None:
    # Bound a visible-ASCII header value without assuming a credential prefix.
    # Account authorization is checked separately via GET /v1/models.
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= 4096
        or any(not "!" <= char <= "~" for char in key)
    ):
        raise MetaConfigurationError(
            "MODEL_API_KEY must be 1-4096 visible ASCII characters without whitespace"
        )


def build_client(
    *,
    model: str,
    environ: Mapping[str, str] | None = None,
    http_client: httpx.Client | None = None,
) -> openai.OpenAI:
    """Resolve only MODEL_API_KEY, with no SDK credential fallback."""
    check_model(model)
    check_environment(
        REJECTED_VARIABLES,
        provider=PROVIDER,
        base_url=BASE_URL,
        error=MetaConfigurationError,
        environ=environ,
    )
    key = resolve_api_key(
        API_KEY_VARIABLE, provider=PROVIDER, error=MetaConfigurationError, environ=environ
    )
    check_key(key)
    client = openai.OpenAI(
        api_key=key,
        base_url=BASE_URL,
        organization=None,
        project=None,
        max_retries=0,
        http_client=http_client,
    )
    check_client(client)
    return client


def check_client(client: openai.OpenAI) -> None:
    check_endpoint(
        client.base_url,
        expected=BASE_URL,
        provider=PROVIDER,
        setting="base_url",
        error=MetaConfigurationError,
    )
    if (
        client.organization is not None
        or client.project is not None
        or client.max_retries != 0
    ):
        raise MetaConfigurationError(
            "Meta requires no organization/project and SDK retries=0"
        )
    check_key(client.api_key)
    headers = httpx.Headers(
        {
            name: value
            for name, value in client.default_headers.items()
            if isinstance(value, str)
        }
    )
    if (
        "authorization" in headers
        or httpx.Headers(client.auth_headers).get("authorization")
        != f"Bearer {client.api_key}"
        or "openai-organization" in headers
        or "openai-project" in headers
        or client.default_query
    ):
        raise MetaConfigurationError(
            "Meta refuses bearer/account header or query overrides"
        )


def check_response(
    response: WireResponse[Response], *, model: str
) -> WireResponse[Response]:
    """Validate a Meta-only SDK projection, retaining the original wire untouched.

    The inserted cache-write zero is a schema placeholder, never provider usage.
    Return the strictly validated typed projection for action parsing; callers
    must continue to read measured usage and capture bytes from the original.
    """
    body = response.wire
    if not set(body) <= set(Response.model_fields) | {"store"}:
        raise wire_invalid("Meta response has unsupported server fields")
    if "store" in body and body["store"] is not False:
        raise wire_invalid("Meta response has invalid store field")
    projected = dict(body)
    projected.pop("store", None)
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        raise wire_invalid("Meta response has invalid usage block")
    details = usage.get("input_tokens_details")
    if details is not None:
        if (
            not isinstance(details, Mapping)
            or not {"cached_tokens"} <= set(details)
            or not set(details) <= {"cached_tokens", "cache_write_tokens"}
            or any(not is_wire_token_count(value) for value in details.values())
            or ("cache_write_tokens" in details and details["cache_write_tokens"] != 0)
        ):
            raise wire_invalid("Meta response has invalid input usage details")
        projected["usage"] = dict(usage)
        projected_details = dict(details)
        if "cache_write_tokens" not in projected_details:
            projected_details["cache_write_tokens"] = 0
        projected["usage"]["input_tokens_details"] = projected_details
    projection = WireResponse(
        json.dumps(projected),
        lambda: Response.model_validate(projected, strict=True),
        kind="Meta validation projection",
    )
    check_sdk_response(projection, model=model)
    usage = response.wire["usage"]
    for field, count, ceiling in (
        ("input_tokens_details", "cached_tokens", "input_tokens"),
        ("output_tokens_details", "reasoning_tokens", "output_tokens"),
    ):
        details = usage.get(field)
        if details is not None:
            for value in details.values():
                if value is not None and not is_wire_token_count(value):
                    raise wire_invalid("Meta response has invalid usage detail counts")
            value = details.get(count)
            if value is not None and (
                not is_wire_token_count(value) or value > usage[ceiling]
            ):
                raise wire_invalid("Meta response has invalid usage details")
    total = usage.get("total_tokens")
    if total is not None and (
        not is_wire_token_count(total)
        or total != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise wire_invalid("Meta response has inconsistent total usage")
    response.admit_response_id()
    return projection


class MetaResponsesExchange:
    """Small provider-specific send path; retry/budget/telemetry stay common."""

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        self.model = check_model(model)
        check_client(client)
        self._client = client
        self.retry = ProviderRetryPolicy(
            max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
        )
        self._executor: TurnExecutor[WireResponse[Response]] = TurnExecutor(
            retry=self.retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        self._flight = SingleFlight(adapter="MetaResponsesTransport")
        if before_dispatch is not None and not callable(before_dispatch):
            raise MetaConfigurationError("Meta dispatch observer must be callable")
        self._before_dispatch = before_dispatch

    def __repr__(self) -> str:
        return f"<MetaResponsesExchange provider=meta model={self.model}>"

    def turn(self) -> SingleFlight:
        return self._flight

    def clear(self) -> None:
        self._executor.clear()

    def last_usage(self) -> AdapterUsage:
        return self._executor.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._executor.last_telemetry()

    def _preflight(self) -> None:
        check_client(self._client)
        if self._before_dispatch is not None:
            try:
                self._before_dispatch()
            except Exception:
                raise MetaConfigurationError(
                    "Meta dispatch observer refused preflight"
                ) from None
        check_client(self._client)

    def exchange(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[Response]:
        check_client(self._client)
        if (
            set(payload)
            != {
                "model",
                "max_output_tokens",
                "instructions",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "store",
                "input",
            }
            or payload["model"] != self.model
            or payload["store"] is not False
            or payload["max_output_tokens"] != 4096
            or payload["tool_choice"] != "auto"
            or payload["parallel_tool_calls"] is not False
        ):
            raise MetaConfigurationError("Meta request differs from the snapshot profile")
        tools = payload["tools"]
        inputs = payload["input"]
        if (
            not isinstance(tools, list)
            or not tools
            or any(
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or tool.get("strict") is not False
                for tool in tools
            )
            or not isinstance(inputs, list)
            or len(inputs) != 1
            or not isinstance(inputs[0], dict)
            or set(inputs[0]) != {"role", "content"}
            or inputs[0]["role"] != "user"
            or not isinstance(inputs[0]["content"], str)
        ):
            raise MetaConfigurationError(
                "Meta requires non-strict functions and one user snapshot"
            )
        return self._executor.run(
            payload=payload,
            deadline=deadline,
            token_bound=request_input_token_bound(payload, "meta request bound"),
            dispatch=lambda timeout: self._dispatch(payload, timeout),
            check_before_dispatch=self._preflight,
            classify=classify_exception,
            check_identity=self._check_response,
            usage_of=response_usage,
        )

    def _check_response(self, response: WireResponse[Response]) -> None:
        check_response(response, model=self.model)

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[Response]:
        raw = self._client.responses.with_raw_response.create(**payload, timeout=timeout)
        return WireResponse(raw.text, raw.parse, kind="meta response")
