"""Benign SDK fixtures for the two observed, root-only null metadata fields."""

import json
import socket

import anthropic
import httpx
import openai
import pytest

from operatebench.agents.model import ModelResponse
from operatebench.providers import anthropic_messages as ap
from operatebench.providers import openai_responses as op
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tools.three_flow_http import (
    ENDPOINTS,
    MODELS,
    HTTPCampaignTransport,
    explicit_sdk_client,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)


def response_body(provider):
    # Independently valid old contract; neither metadata extension is baked in.
    if provider == "openai":
        return {
            "id": "resp_offline",
            "object": "response",
            "created_at": 1,
            "model": MODELS[provider],
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "parallel_tool_calls": False,
            "temperature": 1,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_offline",
                    "call_id": "call_offline",
                    "name": "wait",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": MODELS[provider],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "content": [
            {"type": "tool_use", "id": "tool_offline", "name": "wait", "input": {}}
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }


def extension(provider):
    return "access_programs" if provider == "openai" else "diagnostics"


def dispatch(provider, body):
    calls = []

    def handle(request):
        calls.append(1)
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        client = explicit_sdk_client(
            openai.OpenAI if provider == "openai" else anthropic.Anthropic,
            api_key="offline-placeholder",
            max_retries=0,
            http_client=http,
            base_url=ENDPOINTS[provider].rsplit("/", 1)[0],
        )
        transport = object.__new__(HTTPCampaignTransport)
        transport.provider = provider
        transport.model = MODELS[provider]
        transport.client = client
        transport.network_timeout = None
        transport.admitted_cache_envelope = True
        payload = {"model": MODELS[provider]}
        if provider == "openai":
            payload["input"] = "offline"
        else:
            payload.update(
                max_tokens=10, messages=[{"role": "user", "content": "offline"}]
            )
        result = transport._dispatch(payload)
        assert calls == [1]
        return result


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_complete_sdk_response_counts_usage(provider, with_metadata):
    body = response_body(provider)
    if with_metadata:
        body[extension(provider)] = None
    usage, response = dispatch(provider, body)
    assert (usage.input_tokens, usage.output_tokens) == (7, 3)
    assert isinstance(response, ModelResponse)
    assert response.model == MODELS[provider]
    assert response.output_tokens == 3


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("value", [False, 0, "", [], {}, {"nested": None}])
def test_nonnull_metadata_is_not_a_contract(provider, value):
    body = response_body(provider)
    body[extension(provider)] = value
    with pytest.raises(AdapterProviderError) as error:
        dispatch(provider, body)
    assert error.value.fault == "provider_response_invalid"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "fault", ["other_root", "wrong_provider", "nested", "wrong_usage"]
)
def test_null_metadata_does_not_open_other_shapes(provider, fault):
    body = response_body(provider)
    key = extension(provider)
    body[key] = None
    if fault == "other_root":
        body["unknown_metadata"] = None
    elif fault == "wrong_provider":
        body[extension("anthropic" if provider == "openai" else "openai")] = None
    elif fault == "nested":
        body["usage"][key] = None
    else:
        body["usage"]["input_tokens"] = True
    with pytest.raises(AdapterProviderError) as error:
        dispatch(provider, body)
    assert error.value.fault == "provider_response_invalid"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_null_metadata_combined_http_cli_and_zero_call_replay(
    tmp_path, monkeypatch, provider
):
    from tests.test_three_flow_context import (
        test_context_consuming_cli_sdk_dynamic_identity_replay as combined_roundtrip,
    )
    from tools.three_flow_mock import SDKMockTransport

    original = SDKMockTransport.handle

    def handle(mock, request):
        body = json.loads(original(mock, request).content)
        key = extension(provider)
        assert key not in body
        body[key] = None
        return httpx.Response(200, json=body)

    monkeypatch.setattr(SDKMockTransport, "handle", handle)
    # Existing consuming-CLI assertions cover usage, dynamic identity, replay,
    # and tamper rejection; all custody/budget fixtures are fresh synthetic ones.
    combined_roundtrip(tmp_path, provider)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("change", ["drop", "add", "nonnull"])
def test_raw_typed_metadata_disagreement_rejected(provider, change):
    body = response_body(provider)
    key = extension(provider)
    body[key] = None
    model = (
        openai.types.responses.Response
        if provider == "openai"
        else anthropic.types.Message
    )
    typed = model.model_validate(body)
    if change == "drop":
        typed.model_extra.pop(key)
    elif change == "add":
        body.pop(key)
    else:
        typed.model_extra[key] = {}
    wire = WireResponse(json.dumps(body), lambda: typed, kind="offline metadata test")
    adapter = op if provider == "openai" else ap
    with pytest.raises(AdapterProviderError) as error:
        adapter.check_response_admissible(wire, model=MODELS[provider])
    assert error.value.fault == "provider_response_invalid"
