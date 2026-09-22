"""In-process OpenAI-SDK transports, wired under the real SDK.

The double sits at the *network* boundary, not the SDK boundary: every test that
uses it builds a genuine :class:`openai.OpenAI` client over an
``httpx.MockTransport``, so the request an adapter produces is serialised by the
real SDK, the response is parsed into real response models, and an HTTP status
is mapped to the real SDK exception class. A hand-written stub client would
prove none of that: it would only prove that the adapter agrees with the stub.

Two surfaces live here because two adapters use this SDK. The OpenAI adapter
speaks the Responses API; the xAI adapter speaks xAI's documented
OpenAI-compatible Chat Completions surface through the same client. Their
request and response shapes are genuinely different, so the bodies are built
separately rather than by one helper with a flag.

Nothing here opens a socket. ``MockTransport`` answers every request from the
handler this module installs, so a test that accidentally reached the network
would fail rather than call a provider.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import openai

from boundarybench.providers.openai_responses import OPENAI_BASE_URL
from boundarybench.providers.xai_openai_compat import XAI_BASE_URL

#: Not a credential. The transport never leaves the process, and the SDK
#: requires *some* key to build a client, so this is the placeholder used to
#: prove that the key an adapter resolves is the one the header carries.
FAKE_API_KEY = "test-key-not-a-credential"

#: An endpoint that is not either adapter's, for the tests that prove a client
#: pointed somewhere else is refused rather than used. Nothing answers on it,
#: and nothing here would reach it if it did: ``MockTransport`` intercepts the
#: request before a socket exists.
FAKE_BASE_URL = "https://provider.invalid/v1"


# -- Responses API bodies -----------------------------------------------------


def function_call_item(
    name: str, arguments: str, *, call_id: str = "call_test", item_id: str = "fc_test"
) -> dict[str, Any]:
    """One ``function_call`` output item. ``arguments`` is a JSON *string*."""
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def message_item(text: str, *, item_id: str = "msg_test") -> dict[str, Any]:
    """One assistant message output item: prose beside an action."""
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def responses_body(
    output: Sequence[Any],
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
    input_tokens: int = 137,
    output_tokens: int = 29,
    model: str = "gpt-test-20990101",
) -> dict[str, Any]:
    """One Responses API body, as the wire carries it."""
    body: dict[str, Any] = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": status,
        "output": list(output),
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "tools": [],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    body["incomplete_details"] = (
        None if incomplete_reason is None else {"reason": incomplete_reason}
    )
    return body


# -- Chat Completions bodies (the xAI-compatible surface) ---------------------


def tool_call(name: str, arguments: str, *, call_id: str = "call_test") -> dict[str, Any]:
    """One Chat Completions tool call. ``arguments`` is a JSON *string*."""
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
    model: str = "grok-test-20990101",
) -> dict[str, Any]:
    """One Chat Completions body, as the wire carries it."""
    return {
        "id": "chatcmpl_test",
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


def error_body(kind: str = "api_error", message: str = "provider said no") -> Any:
    return {"error": {"type": kind, "message": message, "code": None, "param": None}}


class RecordingTransport:
    """Answers each request from a scripted sequence, recording what arrived.

    A step is either a response body (answered ``200``), an ``(int, body)`` pair
    (answered with that status, which is what makes the SDK raise its own typed
    error), raw ``bytes`` (so a body can carry a JSON escape no Python value
    survives being encoded from — a lone surrogate, for instance), or an
    exception instance to raise from the transport itself, which is the only way
    to exercise a connection failure or a read timeout honestly.
    """

    def __init__(self, steps: Sequence[Any], *, default: Any = None) -> None:
        self._steps = list(steps)
        self._default = default
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.timeouts: list[Any] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content.decode("utf-8")))
        self.timeouts.append(request.extensions.get("timeout"))
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

    def client(
        self, *, max_retries: int = 0, base_url: str = OPENAI_BASE_URL, **kwargs: Any
    ) -> openai.OpenAI:
        """A real SDK client on this transport, pinned to OpenAI's endpoint.

        The base URL defaults to the *official* one rather than to a placeholder
        because the adapters now refuse a client configured for anywhere else —
        which is the point of :data:`FAKE_BASE_URL` still existing. No request
        leaves the process either way: ``MockTransport`` answers every one of
        them, so what the URL decides here is only what the adapter is entitled
        to believe about where it would have gone.
        """
        return openai.OpenAI(
            api_key=FAKE_API_KEY,
            base_url=base_url,
            max_retries=max_retries,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            **kwargs,
        )

    def xai_client(self, **kwargs: Any) -> openai.OpenAI:
        """The same client, pinned to xAI's OpenAI-compatible endpoint."""
        kwargs.setdefault("base_url", XAI_BASE_URL)
        return self.client(**kwargs)


def scripted_client(
    *steps: Any, **kwargs: Any
) -> tuple[RecordingTransport, openai.OpenAI]:
    """A transport and a real SDK client bound to it, in that order."""
    transport = RecordingTransport(steps)
    return transport, transport.client(**kwargs)


def xai_scripted_client(
    *steps: Any, **kwargs: Any
) -> tuple[RecordingTransport, openai.OpenAI]:
    """The same pair, for the xAI-compatible surface."""
    transport = RecordingTransport(steps)
    return transport, transport.xai_client(**kwargs)
