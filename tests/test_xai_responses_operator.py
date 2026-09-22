# SPDX-License-Identifier: Apache-2.0
"""Focused contract tests for the private xAI Responses lane (offline only)."""

from __future__ import annotations

import json

import httpx
import openai
import pytest


def test_xai_responses_profile_and_endpoint_are_distinct() -> None:
    from operatebench.providers.xai_responses import (
        GROK_4_5_MODEL,
        GROK_4_5_PROFILE,
        XAI_BASE_URL,
        XAI_PROVIDER,
        XAI_RESPONSES_API,
        build_client,
    )

    assert (XAI_PROVIDER, XAI_RESPONSES_API, XAI_BASE_URL, GROK_4_5_MODEL) == (
        "xai",
        "responses",
        "https://api.x.ai/v1",
        "grok-4.5",
    )
    assert GROK_4_5_PROFILE.reasoning == {"effort": "low"}
    assert GROK_4_5_PROFILE.temperature is None
    client = build_client(api_key="offline-not-a-credential")
    assert str(client.base_url) == "https://api.x.ai/v1/"
    assert client.max_retries == 0


def test_lifecycle_xai_request_has_only_six_client_function_tools() -> None:
    from operatebench.agents.model import MAX_OUTPUT_TOKENS
    from operatebench.agents.transport import ModelRequest
    from operatebench.agents.xai_responses import build_model_payload
    from operatebench.providers.xai_responses import GROK_4_5_MODEL

    request = ModelRequest(
        model=GROK_4_5_MODEL,
        protocol_version="operatebench.model.v4",
        prompt={"probe": True},
        tool_names=("act", "ask", "wait", "complete", "escalate", "retrieve"),
        observation_digest_sha256="0" * 64,
        prompt_digest_sha256="1" * 64,
        invocation_index=0,
        turn_index=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    body = build_model_payload(request, model=GROK_4_5_MODEL)
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 4096
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["store"] is False
    assert "temperature" not in body and "top_p" not in body
    assert len(body["tools"]) == 6
    assert {tool["type"] for tool in body["tools"]} == {"function"}


def test_xai_sdk_serialises_the_exact_responses_url_and_controls() -> None:
    from operatebench.agents.model import MAX_OUTPUT_TOKENS
    from operatebench.agents.transport import ModelRequest
    from operatebench.agents.xai_responses import build_model_payload
    from operatebench.providers.xai_responses import GROK_4_5_MODEL

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 4096,
                "model": GROK_4_5_MODEL,
                "output": [],
                "parallel_tool_calls": False,
                "previous_response_id": None,
                "reasoning": {"effort": "low", "summary": None},
                "store": False,
                "temperature": None,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "required",
                "tools": [],
                "top_p": None,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 2,
                },
                "user": None,
                "metadata": {},
            },
        )

    client = openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ModelRequest(
        model=GROK_4_5_MODEL,
        protocol_version="operatebench.model.v4",
        prompt={"probe": True},
        tool_names=("act", "ask", "wait", "complete", "escalate", "retrieve"),
        observation_digest_sha256="0" * 64,
        prompt_digest_sha256="1" * 64,
        invocation_index=0,
        turn_index=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    payload = build_model_payload(request, model=GROK_4_5_MODEL)
    client.responses.with_raw_response.create(**payload, timeout=30.0)
    assert str(seen[0].url) == "https://api.x.ai/v1/responses"
    sent = json.loads(seen[0].content)
    assert sent == payload


def test_raw_tool_argument_byte_bound_accepts_exact_limit_and_refuses_plus_one() -> None:
    from operatebench.providers.faults import AdapterProviderError
    from operatebench.providers.xai_responses import check_tool_argument_wire_bytes

    def body(arguments: str) -> str:
        return '{"output":[{"type":"function_call","arguments":"' + arguments + '"}]}'

    check_tool_argument_wire_bytes(body("x" * 17), max_bytes=17)
    with pytest.raises(AdapterProviderError) as caught:
        check_tool_argument_wire_bytes(body("x" * 18), max_bytes=17)
    assert caught.value.fault == "provider_response_invalid"
    assert "xxxxxxxx" not in str(caught.value)


def test_invalid_huge_arguments_are_refused_before_json_or_sdk_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from operatebench.providers.wire import WireResponse
    from operatebench.providers.xai_responses import check_tool_argument_wire_bytes

    text = '{"output":[{"type":"function_call","arguments":"' + ("{" * 18)
    parsed = False

    def sdk_parse() -> object:
        nonlocal parsed
        parsed = True
        raise AssertionError("SDK parser reached")

    def json_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads reached")

    monkeypatch.setattr("operatebench.providers.wire.json.loads", json_parse)
    response: WireResponse[object] = WireResponse(
        text,
        sdk_parse,
        kind="xAI response",
        pre_json_check=lambda payload: check_tool_argument_wire_bytes(
            payload, max_bytes=17
        ),
    )
    with pytest.raises(Exception, match="byte limit"):
        _ = response.wire
    assert parsed is False


def test_oversized_arguments_record_a_received_response_without_usage() -> None:
    from operatebench.agents.model import MAX_OUTPUT_TOKENS
    from operatebench.agents.transport import ModelRequest
    from operatebench.agents.xai_responses import build_model_payload
    from operatebench.providers.faults import AdapterProviderError
    from operatebench.providers.telemetry import TurnDeadline
    from operatebench.providers.xai_responses import (
        GROK_4_5_MODEL,
        XAIResponsesExchange,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "resp_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 4096,
                "model": GROK_4_5_MODEL,
                "output": [
                    {
                        "id": "call_offline",
                        "type": "function_call",
                        "call_id": "call_offline",
                        "name": "complete",
                        "arguments": "x" * 18,
                        "status": "completed",
                    }
                ],
                "parallel_tool_calls": False,
                "previous_response_id": None,
                "reasoning": {"effort": "low", "summary": None},
                "store": False,
                "temperature": None,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "required",
                "tools": [],
                "top_p": None,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
                "user": None,
                "metadata": {},
            },
        )

    client = openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ModelRequest(
        model=GROK_4_5_MODEL,
        protocol_version="operatebench.model.v4",
        prompt={"probe": True},
        tool_names=("act", "ask", "wait", "complete", "escalate", "retrieve"),
        observation_digest_sha256="0" * 64,
        prompt_digest_sha256="1" * 64,
        invocation_index=0,
        turn_index=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    exchange = XAIResponsesExchange(
        model=GROK_4_5_MODEL,
        client=client,
        single_flight_label="test",
        max_tool_argument_wire_bytes=17,
    )
    with exchange.turn(), pytest.raises(AdapterProviderError):
        exchange.clear()
        exchange.exchange(
            build_model_payload(request, model=GROK_4_5_MODEL),
            TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False),
        )
    attempt = exchange.last_telemetry().attempts[0]
    assert attempt.response_received is True
    assert attempt.usage_reported is False


def test_xai_responses_identity_is_separate_from_chat_completions() -> None:
    from operatebench.agents.evidence import provider_identity_for
    from operatebench.agents.xai_responses import XAIResponsesTransport
    from operatebench.providers.xai_openai_compat import XAI_API as CHAT_API
    from operatebench.providers.xai_responses import (
        GROK_4_5_MODEL,
        XAI_RESPONSES_API,
    )

    client = openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )
    lane = XAIResponsesTransport(
        model=GROK_4_5_MODEL, client=client, deadline_seconds=30.0
    )
    identity = provider_identity_for(lane)
    assert (identity.provider, identity.api, identity.model) == (
        "xai",
        XAI_RESPONSES_API,
        GROK_4_5_MODEL,
    )
    assert identity.api != CHAT_API
    assert identity.implementation == "XAIResponsesTransport"
