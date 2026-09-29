"""Anthropic 1.8 qualification through the actual SDK and offline HTTP boundary."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator

import anthropic
import httpx
import httpx2
import pytest

from operatebench.providers import anthropic_messages as provider
from tests.anthropic_transport import message_body, tool_use_block


def test_sdk_accepts_explicit_bridge_and_preserves_request_bytes() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=message_body([tool_use_block("done", {})]))

    with provider.anthropic_http_client(
        transport=httpx.MockTransport(handler), timeout=12.0
    ) as http:
        assert isinstance(http, httpx2.Client)
        with anthropic.Anthropic(
            api_key="offline-placeholder", max_retries=0, http_client=http
        ) as client:
            result = client.messages.create(
                model="claude-test-20990101",
                max_tokens=100,
                messages=[{"role": "user", "content": "offline"}],
            )
    assert result.content[0].name == "done"
    assert result.usage.input_tokens == 137
    assert len(seen) == 1
    assert seen[0].content == (
        b'{"max_tokens":100,"messages":[{"role":"user","content":"offline"}],'
        b'"model":"claude-test-20990101"}'
    )
    assert json.loads(seen[0].content)["model"] == result.model
    assert httpx.Client is not httpx2.Client


class ResponseStream(httpx.SyncByteStream):
    def __init__(self, body: bytes, failure: Exception | None = None) -> None:
        self.body, self.failure = body, failure
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        if self.failure is not None:
            raise self.failure
        yield self.body

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("buffered", [False, True])
def test_sdk_gzip_response_decodes_once_with_original_headers(buffered: bool) -> None:
    body = message_body([tool_use_block("done", {})])
    encoded = gzip.compress(json.dumps(body).encode())
    stream = ResponseStream(encoded)

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            stream=stream,
            headers=[
                ("content-type", "application/json"),
                ("content-encoding", "gzip"),
                ("content-length", str(len(encoded))),
                ("request-id", "req_offline"),
                ("x-duplicate", "first"),
                ("x-duplicate", "second"),
            ],
            extensions={"http_version": b"HTTP/1.1"},
        )
        if buffered:
            response.read()  # BudgetWire also reads before the SDK sees it.
        return response

    with anthropic.Anthropic(
        api_key="offline-placeholder",
        max_retries=0,
        http_client=provider.anthropic_http_client(
            transport=httpx.MockTransport(handler)
        ),
    ) as client:
        raw = client.messages.with_raw_response.create(
            model="claude-test-20990101", max_tokens=100, messages=[]
        )
        assert json.loads(raw.text()) == body
        assert raw.parse().usage.output_tokens == 29
        assert raw.headers["content-encoding"] == "gzip"
        assert raw.headers.get_list("x-duplicate") == ["first", "second"]
        assert raw.http_response.extensions["http_version"] == b"HTTP/1.1"
        assert isinstance(raw.http_response, httpx2.Response)
    assert stream.closed


@pytest.mark.parametrize("during_read", [False, True])
@pytest.mark.parametrize(
    "error_name",
    [
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
    ],
)
def test_sdk_transport_errors_keep_types_and_close(
    during_read: bool, error_name: str
) -> None:
    failure = getattr(httpx, error_name)("offline failure")
    stream = ResponseStream(b"", failure)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if during_read:
            return httpx.Response(200, stream=stream)
        raise failure

    with anthropic.Anthropic(
        api_key="offline-placeholder",
        max_retries=0,
        http_client=provider.anthropic_http_client(
            transport=httpx.MockTransport(handler)
        ),
    ) as client:
        expected = (
            anthropic.APITimeoutError
            if "Timeout" in error_name
            else anthropic.APIConnectionError
        )
        with pytest.raises(expected) as caught:
            client.messages.create(
                model="claude-test-20990101", max_tokens=100, messages=[]
            )
    assert len(calls) == 1
    cause = caught.value.__cause__
    assert isinstance(cause, getattr(httpx2, error_name))
    assert cause.__cause__ is failure
    assert isinstance(cause.request, httpx2.Request)
    if during_read:
        assert stream.closed


def test_bridge_preserves_extensions_identity_headers_url_and_close() -> None:
    from operatebench.agents.evidence import WireCaptureTransport

    class Inner(httpx.BaseTransport):
        closed = 0

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            assert request.extensions is extensions
            assert request.headers.raw == headers
            assert str(request.url) == "https://provider.invalid/v1/messages?x=a%2Fb"
            assert request.content == b"exact bytes"
            return httpx.Response(200, json={})

        def close(self) -> None:
            self.closed += 1

    inner = Inner()
    capture = WireCaptureTransport()
    capture.attach(inner)
    with provider.anthropic_http_client(transport=capture, timeout=None) as client:
        request = client.build_request(
            "POST",
            "https://provider.invalid/v1/messages?x=a%2Fb",
            content=b"exact bytes",
            headers=[("x-duplicate", "first"), ("x-duplicate", "second")],
        )
        extensions, headers = request.extensions, request.headers.raw
        client.send(request)
    assert inner.closed == 1
    assert capture.calls == 1


def test_capture_close_for_other_sdk_transport() -> None:
    from operatebench.agents.evidence import WireCaptureTransport

    class Inner(httpx.MockTransport):
        closed = 0

        def close(self) -> None:
            self.closed += 1

    inner = Inner(lambda _: httpx.Response(200))
    capture = WireCaptureTransport()
    capture.attach(inner)
    with httpx.Client(transport=capture) as client:
        client.get("https://provider.invalid")
    assert inner.closed == 1


@pytest.mark.parametrize("temperature", [0, 0.0, 0.7, None])
def test_sdk_kwargs_preserve_present_temperature_without_mutating_payload(temperature):
    payload = {
        "model": "claude-test-20990101",
        "temperature": temperature,
        "thinking": {"type": "disabled"},
    }
    original = dict(payload)
    result = provider.anthropic_sdk_kwargs(payload)
    assert result == {
        "model": payload["model"],
        "thinking": payload["thinking"],
        "extra_body": {"temperature": temperature},
    }
    assert payload == original


def test_sdk_kwargs_keep_absent_temperature_absent():
    payload = {"model": "claude-test-20990101", "thinking": {"type": "disabled"}}
    assert provider.anthropic_sdk_kwargs(payload) == payload


@pytest.mark.parametrize("extra", [{}, {"temperature": 1}, None])
def test_sdk_kwargs_reject_preexisting_extra_body(extra):
    with pytest.raises(provider.AnthropicConfigurationError, match="extra_body"):
        provider.anthropic_sdk_kwargs({"temperature": 0, "extra_body": extra})


def test_sdk_kwargs_send_temperature_as_semantically_equal_json():
    seen = []
    payload = {
        "model": "claude-test-20990101",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "offline"}],
        "temperature": 0.0,
    }

    def handler(request):
        seen.append(request.content)
        return httpx.Response(200, json=message_body([tool_use_block("done", {})]))

    with anthropic.Anthropic(
        api_key="offline-placeholder",
        max_retries=0,
        http_client=provider.anthropic_http_client(
            transport=httpx.MockTransport(handler)
        ),
    ) as client:
        client.messages.create(**provider.anthropic_sdk_kwargs(payload))
    assert json.loads(seen[0]) == payload
    assert seen[0] == (
        b'{"max_tokens":100,"messages":[{"role":"user","content":"offline"}],'
        b'"model":"claude-test-20990101","temperature":0.0}'
    )


@pytest.mark.parametrize("during_read", [False, True])
def test_bridge_preserves_non_httpx_errors_and_closes_failed_stream(during_read):
    failure = ValueError("offline non-network failure")
    stream = ResponseStream(b"", failure)

    def handler(request):
        if during_read:
            return httpx.Response(200, stream=stream)
        raise failure

    with (
        provider.anthropic_http_client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError) as caught,
    ):
        client.get("https://provider.invalid")
    assert caught.value is failure
    if during_read:
        assert stream.closed


@pytest.mark.parametrize(
    "cell", ["haiku45", "sonnet5", "luna56", "grok45", "mistralsmall"]
)
def test_aggregate_client_has_one_transport_owner(cell):
    from tools.diagnose_aggregate_budget import client_for

    class Inner(httpx.MockTransport):
        closed = 0

        def close(self):
            self.closed += 1

    inner = Inner(lambda _: httpx.Response(200, json={}))
    client = client_for(cell, "offline-placeholder", inner)
    if cell == "mistralsmall":
        client.sdk_configuration.client.close()
    else:
        client.close()
    assert inner.closed == 1


@pytest.mark.parametrize("provider_name", ["anthropic", "openai", "mistral"])
def test_campaign_client_closes_shared_transport_once(tmp_path, provider_name):
    from tests.test_three_flow_providers import fixture

    budget, mock, transport, _ = fixture(tmp_path, provider_name)
    closed = []
    mock.close = lambda: closed.append(True)
    try:
        assert transport.http.timeout.read is None
        assert not transport.http.trust_env
        assert not transport.http.follow_redirects
        transport.close()
        assert closed == [True]
    finally:
        budget.close()
