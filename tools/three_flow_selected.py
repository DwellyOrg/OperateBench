"""Finite selected-six HTTP contracts; reuses campaign admission and accounting.

Public context bounds are diagnostic, never verified endpoint guarantees.
Raw reasoning and all usage categories remain in the existing wire capture.
"""

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from operatebench.agents.transport import ModelResponse, ToolCall
from operatebench.providers.wire import WireResponse, wire_invalid
from tools.three_flow_profiles import ENDPOINTS, SELECTED_SIX


def mode_fields(provider: str, model: str) -> dict[str, Any]:
    if (provider, model) not in SELECTED_SIX:
        raise ValueError("outside selected-six scope")
    controls: dict[str, dict[str, Any]] = {
        "openai": {"reasoning": {"effort": "none"}},
        "anthropic": {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        },
        "mistral": {"reasoning_effort": "none"},
        "xai": {"reasoning": {"effort": "low"}},
        "river_http": {"reasoning_effort": "low" if model == "GLM-5.3" else "none"},
    }
    return controls[provider]


def selected_registry(transport_policy: str) -> list[dict[str, Any]]:
    if transport_policy not in ("legacy-v1", "paced-safe-errors-v1"):
        raise ValueError("unknown transport policy")
    return [
        {
            "provider": p,
            "model": m,
            "endpoint": ENDPOINTS[p],
            "mapping": "three-flow-"
            + p
            + "-selected-six-snapshot-v1"
            + (
                "-paced-safe-errors-v1"
                if p == "mistral" and transport_policy != "legacy-v1"
                else ""
            ),
            "request_settings": {
                **mode_fields(p, m),
                "no_native_continuation": True,
                "tool_choice": {"type": "any"} if p == "anthropic" else "required",
            },
            "settings_source": "tools/three_flow_http.py",
            "retries": 0,
            "admitted": False,
        }
        for p, m in SELECTED_SIX
    ]


def validate_selected_profile(
    p: dict[str, Any], provider: str, model: str | None
) -> None:
    if (provider, model) not in SELECTED_SIX:
        raise ValueError("selected-six exact provider/model required")
    if p.get("execution_policy") != "three-flow-optional-limits-v1":
        raise ValueError("selected-six requires null monetary policy")
    if provider == "river_http" and p.get("owner_output_cap_tokens") != 30000:
        raise ValueError("River HTTP owner output cap must be 30000")


def river_response(
    wire: WireResponse[Any], model: str
) -> tuple[SimpleNamespace, ModelResponse]:
    """Use native SDK parsing and raw JSON so string arguments are never coerced."""
    from operatebench.agents.lifecycle_contract import (
        ToolArgumentsUndecodable,
        UndecodableArguments,
    )
    from operatebench.providers.toolcalls import decode_tool_arguments

    body = wire.wire
    check_river_wire(body)
    parsed = wire.parsed
    _agree(body, parsed.model_dump(mode="json", exclude_unset=True))
    if body.get("model") != model or parsed.model != model:
        raise wire_invalid("River HTTP response model differs")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise wire_invalid("River HTTP usage missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise wire_invalid("invalid River HTTP usage")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise wire_invalid("River HTTP total usage differs")
    for key, total in (
        ("prompt_tokens_details", "prompt_tokens"),
        ("completion_tokens_details", "completion_tokens"),
    ):
        details = usage.get(key)
        if details is not None and (
            not isinstance(details, dict)
            or any(
                type(v) is not int or not 0 <= v <= usage[total] for v in details.values()
            )
        ):
            raise wire_invalid("invalid River HTTP usage categories")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise wire_invalid("River HTTP requires one choice")
    choice = choices[0]
    message = choice.get("message", {})
    if message.get("role") != "assistant" or choice.get("finish_reason") not in (
        "stop",
        "tool_calls",
        "length",
    ):
        raise wire_invalid("invalid River HTTP completion")
    calls = []
    for call in message.get("tool_calls") or []:
        if call.get("type") != "function" or not isinstance(call.get("function"), dict):
            raise wire_invalid("invalid River HTTP tool call")
        f = call["function"]
        if not isinstance(f.get("name"), str) or not isinstance(f.get("arguments"), str):
            raise wire_invalid("invalid River HTTP function")
        arguments: Any
        try:
            arguments = decode_tool_arguments(
                f["arguments"],
                label="River HTTP tool arguments",
                error=ToolArgumentsUndecodable,
            )
        except ToolArgumentsUndecodable:
            arguments = UndecodableArguments()
        calls.append(ToolCall(f["name"], arguments))
    return SimpleNamespace(
        input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"]
    ), ModelResponse(
        model=model,
        stop_reason="max_output_tokens"
        if choice["finish_reason"] == "length"
        else "completed",
        output_tokens=usage["completion_tokens"],
        tool_calls=tuple(calls),
    )


def _object(
    value: Any, allowed: Iterable[str], required: Iterable[str] = ()
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise wire_invalid("River HTTP object fields or container differ")
    return value


def _string(value: Any, nullable: bool = False) -> None:
    if not (isinstance(value, str) or (nullable and value is None)):
        raise wire_invalid("River HTTP string type differs")


def _count(value: Any) -> None:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise wire_invalid("River HTTP token/count type differs")


def _agree(raw: Any, typed: Any) -> None:
    if type(raw) is not type(typed):
        raise wire_invalid("River HTTP raw and typed types differ")
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key not in typed:
                raise wire_invalid("River HTTP typed field missing")
            _agree(value, typed[key])
    elif isinstance(raw, list):
        if len(raw) != len(typed):
            raise wire_invalid("River HTTP typed container differs")
        for a, b in zip(raw, typed, strict=True):
            _agree(a, b)
    elif raw != typed:
        raise wire_invalid("River HTTP raw and typed values differ")


def check_river_wire(body: Any) -> None:
    _object(
        body,
        {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "system_fingerprint",
            "service_tier",
        },
        {"id", "object", "created", "model", "choices", "usage"},
    )
    for key in ("id", "object", "model"):
        _string(body[key])
    if not body["id"] or body["object"] != "chat.completion":
        raise wire_invalid("River HTTP completion identity differs")
    _count(body["created"])
    for key in ("system_fingerprint", "service_tier"):
        if key in body:
            _string(body[key], nullable=True)
    choices = body["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise wire_invalid("River HTTP choices container differs")
    choice = _object(
        choices[0],
        {"index", "finish_reason", "message", "logprobs"},
        {"index", "finish_reason", "message"},
    )
    if (
        type(choice["index"]) is not int
        or choice["index"] != 0
        or choice.get("logprobs") is not None
    ):
        raise wire_invalid("River HTTP choice contract differs")
    _string(choice["finish_reason"])
    message = _object(
        choice["message"],
        {"role", "content", "reasoning_content", "refusal", "tool_calls"},
        {"role"},
    )
    if message["role"] != "assistant":
        raise wire_invalid("River HTTP message role differs")
    for key in ("content", "reasoning_content", "refusal"):
        if key in message:
            _string(message[key], nullable=True)
    calls = message.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list):
            raise wire_invalid("River HTTP tool calls container differs")
        ids = set()
        for call in calls:
            _object(call, {"id", "type", "function"}, {"id", "type", "function"})
            _string(call["id"])
            if not call["id"] or call["id"] in ids or call["type"] != "function":
                raise wire_invalid("River HTTP tool identity differs")
            ids.add(call["id"])
            f = _object(call["function"], {"name", "arguments"}, {"name", "arguments"})
            _string(f["name"])
            _string(f["arguments"])
    usage = _object(
        body["usage"],
        {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        },
        {"prompt_tokens", "completion_tokens", "total_tokens"},
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _count(usage[key])
    for key, allowed in (
        ("prompt_tokens_details", {"cached_tokens"}),
        ("completion_tokens_details", {"reasoning_tokens"}),
    ):
        if usage.get(key) is not None:
            for value in _object(usage[key], allowed).values():
                _count(value)


# Published category maxima for the selected endpoint/tier envelope, USD/MTok.
# Unknown fees are never assigned an invented token rate or called zero.
TARIFFS = {
    "gpt-6.1-terra": ("4.4", "0.22", "11", "16.5"),
    "claude-haiku-5-5": ("0.50", "0.05", "1", "2.50"),
    "mistral-large-4": ("1.36", "0.14", None, "4.18"),
    "grok-4.7": ("4", "1", None, "12"),
    "GLM-5.3": ("1.12", "0.208", None, "3.52"),
    "moonshotai/Kimi-K3": ("3", "0.3", None, "15"),
}
BOUNDS = dict(
    zip(
        (m for _, m in SELECTED_SIX),
        (
            (1050000, 922000, 128000),
            (1000000, None, 128000),
            (1000000, None, None),
            (500000, None, None),
            (1048576, None, 128000),
            (1048576, None, 32768),
        ),
        strict=True,
    )
)
PRICING_POLICY = "selected-six-published-token-envelope-v1"
ACCOUNTING_POLICY = {
    "mode": "reported-usage-estimate-v1",
    "additional_fees": "unknown",
    "invoice_guaranteed": False,
}


def validate_published_profile(p: Any, provider: str, model: str | None) -> None:
    import re
    from decimal import Decimal

    from tools.three_flow_admission import GUARANTEES, digest

    fields = {
        "endpoint",
        "input_max",
        "output_max",
        "context_max",
        "input_usd_per_million",
        "output_usd_per_million",
        "network_timeout_seconds",
        "rate_evidence_sha256",
        "bounds_evidence_sha256",
        "guarantees",
        "pricing_policy",
        "execution_policy",
        "bounds_policy",
        "accounting_policy",
        "tariff_evidence",
    }
    if provider == "river_http":
        fields.add("owner_output_cap_tokens")
    if (
        not isinstance(p, dict)
        or set(p) != fields
        or (provider, model) not in SELECTED_SIX
    ):
        raise ValueError("closed selected-six published profile required")
    if (
        p["pricing_policy"] != PRICING_POLICY
        or p["accounting_policy"] != ACCOUNTING_POLICY
        or p["guarantees"] != dict.fromkeys(GUARANTEES, False)
    ):
        raise ValueError("selected-six estimate cannot assert billing guarantees")
    validate_selected_profile(p, provider, model)
    assert model is not None
    context, input_max, output = BOUNDS[model]
    if (
        (p["context_max"], p["input_max"], p["output_max"])
        != (context, input_max, output)
        or p["network_timeout_seconds"] is not None
        or p["endpoint"] != ENDPOINTS[provider]
    ):
        raise ValueError("selected-six published route/bounds differ")
    rates = TARIFFS[model]
    if (
        p["input_usd_per_million"]
        != str(max(Decimal(v) for v in rates[:3] if v is not None))
        or p["output_usd_per_million"] != rates[3]
    ):
        raise ValueError("selected-six published envelope differs")
    evidence = p["tariff_evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"category_rates", "source", "sha256", "interpretation"}
        or evidence["category_rates"]
        != dict(zip(("input", "cache_read", "cache_write", "output"), rates, strict=True))
        or evidence["interpretation"]
        != (
            "published-category-maxima; estimated reported-token cost; "
            "unknown additional fees"
        )
    ):
        raise ValueError("selected-six tariff evidence differs")
    if (
        not isinstance(evidence["source"], str)
        or not evidence["source"].startswith("https://")
        or not re.fullmatch(r"[a-f0-9]{64}", evidence["sha256"])
        or p["rate_evidence_sha256"] != digest(evidence)
        or not re.fullmatch(r"[a-f0-9]{64}", p["bounds_evidence_sha256"])
    ):
        raise ValueError("selected-six source evidence required")
    b = p["bounds_policy"]
    if (
        not isinstance(b, dict)
        or set(b) != {"mode", "source", "scope", "interpretation"}
        or b["mode"] != "published-context-diagnostic-v1"
        or b["scope"] != "endpoint"
        or b["interpretation"]
        not in ("published-integer", "conservative-decimal-shorthand")
        or not b["source"].startswith("https://")
    ):
        raise ValueError("selected-six diagnostic bounds required")
