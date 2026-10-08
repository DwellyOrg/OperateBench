"""SDK 3 transport qualification without sockets or provider credentials."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version

import httpx
import httpx2
import pytest
from mistralai.client import Mistral

from boundarybench.providers import mistral_chat as boundary
from operatebench.agents.mistral_chat import (
    MistralChatCompletionsTransport,
    build_model_payload,
)
from operatebench.providers import mistral_chat as lifecycle
from operatebench.providers.faults import AdapterProviderError
from tests.mistral_transport import chat_body, tool_call
from tests.test_lifecycle_mistral_chat import model_request, response_body
from tests.test_mistral_adapter import _adapter, _deadline, _turn_request


def test_sdk_async_native_transport_auth_payload_and_usage() -> None:
    async def exercise() -> None:
        seen = []

        def handler(request):
            seen.append(request)
            return httpx2.Response(200, json=response_body())

        with httpx2.Client(transport=httpx2.MockTransport(handler)) as sync_client:
            async with httpx2.AsyncClient(
                transport=httpx2.MockTransport(handler)
            ) as async_client:
                client = Mistral(
                    api_key="offline-placeholder",
                    client=sync_client,
                    async_client=async_client,
                )
                payload = build_model_payload(
                    model_request(), model=lifecycle.MISTRAL_SMALL_MODEL
                )
                response = await client.chat.complete_async(**payload)
                assert len(seen) == 1
                assert seen[0].headers["authorization"] == "Bearer offline-placeholder"
                assert json.loads(seen[0].content) == {
                    **payload,
                    **lifecycle.SDK_INJECTED_FIELDS,
                }
                assert response.usage.prompt_tokens == 137
                assert response.usage.completion_tokens == 29

    asyncio.run(exercise())


def test_default_sdk_clients_are_admitted_without_dispatch() -> None:
    assert version("mistralai") == "3.0.0"
    client = lifecycle.build_client(api_key="offline-placeholder")
    try:
        assert type(client.sdk_configuration.client) is httpx2.Client
        assert type(client.sdk_configuration.async_client) is httpx2.AsyncClient
        lifecycle.check_client_state(client)
    finally:
        client.sdk_configuration.client.close()
        asyncio.run(client.sdk_configuration.async_client.aclose())


@pytest.mark.parametrize("http", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_lifecycle_capture_preserves_payload_usage_and_cookie_policy(http) -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return http.Response(
            200, json=response_body(), headers={"set-cookie": "session=discard; Path=/"}
        )

    with http.Client(transport=http.MockTransport(handler)) as inner:
        client = lifecycle.build_client(api_key="offline-placeholder", http_client=inner)
        lane = MistralChatCompletionsTransport(
            model=lifecycle.MISTRAL_SMALL_MODEL, client=client, deadline_seconds=30
        )
        request = model_request()
        answer = lane.send(request)
        lane.send(request)
        assert len(seen) == 2
        assert json.loads(seen[0].content) == {
            **build_model_payload(request, model=lifecycle.MISTRAL_SMALL_MODEL),
            **lifecycle.SDK_INJECTED_FIELDS,
        }
        assert lane.last_usage().input_tokens == 137
        assert lane.last_usage().output_tokens == answer.output_tokens == 29
        assert not inner.cookies
        assert "cookie" not in seen[1].headers


def test_boundary_native_httpx2_wire_and_usage() -> None:
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(
            200,
            json=chat_body([tool_call("read_records", "{}")]),
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as inner:
        client = boundary.build_client(api_key="offline-placeholder", http_client=inner)
        lane = _adapter(client)
        request = _turn_request()
        lane.next_call(request, _deadline())
        assert json.loads(seen[0].content) == {
            **boundary.build_mistral_request(request, model="mistral-test-20990101"),
            **boundary.SDK_INJECTED_FIELDS,
        }
        assert lane.last_usage().input_tokens == 137
        assert lane.last_usage().output_tokens == 29


@pytest.mark.parametrize("mutation", ["unknown_field", "boolean_usage"])
def test_native_sdk_response_keeps_closed_world_validation(mutation) -> None:
    body = response_body()
    if mutation == "unknown_field":
        body["unreviewed"] = "extension"
    else:
        body["usage"]["prompt_tokens"] = True

    def handler(request):
        return httpx2.Response(200, json=body)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as inner:
        client = lifecycle.build_client(api_key="offline-placeholder", http_client=inner)
        lane = MistralChatCompletionsTransport(
            model=lifecycle.MISTRAL_SMALL_MODEL, client=client, deadline_seconds=30
        )
        with pytest.raises(AdapterProviderError) as caught:
            lane.send(model_request())
        assert caught.value.fault == "provider_response_invalid"


@pytest.mark.parametrize(
    ("error", "fault"),
    [
        (httpx2.ReadTimeout, "provider_timeout"),
        (httpx2.ConnectError, "provider_network_error"),
    ],
)
def test_native_transport_faults_are_classified_by_both_adapters(error, fault) -> None:
    def handler(request):
        raise error("offline failure", request=request)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as inner:
        client = lifecycle.build_client(api_key="offline-placeholder", http_client=inner)
        lane = MistralChatCompletionsTransport(
            model=lifecycle.MISTRAL_SMALL_MODEL, client=client, deadline_seconds=30
        )
        with pytest.raises(AdapterProviderError) as caught:
            lane.send(model_request())
        assert caught.value.fault == fault
        assert boundary.classify_exception(error("offline failure")).fault == fault


@pytest.mark.parametrize("mutation", ["state", "method"])
def test_service_account_hook_mutation_is_refused_before_dispatch(
    monkeypatch, mutation
) -> None:
    def forbidden(request):
        raise AssertionError("mutated client dispatched")

    with httpx2.Client(transport=httpx2.MockTransport(forbidden)) as inner:
        client = lifecycle.build_client(api_key="offline-placeholder", http_client=inner)
        lane = MistralChatCompletionsTransport(
            model=lifecycle.MISTRAL_SMALL_MODEL, client=client, deadline_seconds=30
        )
        hook = client.sdk_configuration._hooks.before_request_hooks[0]
        assert type(hook).__name__ == "ServiceAccountAuthHook"
        if mutation == "state":
            hook.unreviewed = True
        else:
            monkeypatch.setattr(type(hook), "before_request", lambda *args: args[-1])
        with pytest.raises(lifecycle.MistralConfigurationError):
            lane.send(model_request())
