"""An in-process Mistral transport, wired under the real SDK.

The double sits at the *network* boundary, not the SDK boundary: every test that
uses it builds a genuine ``mistralai.client.Mistral`` over an
``httpx.MockTransport``, so the request the adapter produces is serialised by
the real SDK, the response is parsed into real response models, and an HTTP
status is mapped to the real SDK exception class. A hand-written stub client
would prove none of that: it would only prove that the adapter agrees with the
stub.

Nothing here opens a socket. ``MockTransport`` answers every request from the
handler this module installs, so a test that accidentally reached the network
would fail rather than call a provider.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from mistralai.client import Mistral

#: Not a credential. The transport never leaves the process, and the SDK
#: requires *some* key to build a client.
FAKE_API_KEY = "test-key-not-a-credential"

#: Mistral validates tool-call ids as exactly nine alphanumeric characters, so a
#: body that used a longer placeholder would be rejected by the SDK's own model
#: parsing rather than by the adapter under test.
VALID_CALL_ID = "abcdefghi"


def tool_call(
    name: str, arguments: str, *, call_id: str = VALID_CALL_ID
) -> dict[str, Any]:
    """One tool call. ``arguments`` may be a JSON string or an object."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def chat_body(
    tool_calls: Sequence[Any] | None,
    *,
    finish_reason: str = "tool_calls",
    content: str | None = None,
    prompt_tokens: int = 137,
    completion_tokens: int = 29,
    model: str = "mistral-test-20990101",
) -> dict[str, Any]:
    """One chat-completion body, as the wire carries it."""
    return {
        "id": "cmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None if tool_calls is None else list(tool_calls),
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def error_body(message: str = "provider said no") -> Any:
    return {"message": message, "type": "invalid_request_error", "code": None}


class RecordingTransport:
    """Answers each request from a scripted sequence, recording what arrived.

    A step is a response body (answered ``200``), an ``(int, body)`` pair, raw
    ``bytes``, or an exception instance to raise from the transport itself.
    """

    def __init__(self, steps: Sequence[Any], *, default: Any = None) -> None:
        self._steps = list(steps)
        self._default = default
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content.decode("utf-8")))
        if not self._steps:
            if self._default is None:
                raise AssertionError(
                    f"the transport was called {self.calls} time(s) but only "
                    f"{self.calls - 1} step(s) were scripted"
                )
            return httpx.Response(200, json=self._default, request=request)
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, bytes):
            return httpx.Response(
                200,
                content=step,
                headers={"content-type": "application/json"},
                request=request,
            )
        if isinstance(step, tuple):
            status, body = step
            return httpx.Response(status, json=body, request=request)
        return httpx.Response(200, json=step, request=request)

    def client(self, **kwargs: Any) -> Mistral:
        from boundarybench.providers.mistral_chat import build_client

        return build_client(
            api_key=FAKE_API_KEY,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            **kwargs,
        )


def scripted_client(*steps: Any, **kwargs: Any) -> tuple[RecordingTransport, Mistral]:
    """A transport and a real SDK client bound to it, in that order."""
    transport = RecordingTransport(steps)
    return transport, transport.client(**kwargs)
