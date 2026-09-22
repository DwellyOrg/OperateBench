"""An in-process Anthropic transport, wired under the real SDK.

The double sits at the *network* boundary, not the SDK boundary: every test
that uses it builds a genuine :class:`anthropic.Anthropic` client over an
``httpx.MockTransport``, so the request the adapter produces is serialised by
the real SDK, the response is parsed into real ``Message``/``Usage`` models, and
an HTTP status is mapped to the real SDK exception class. A hand-written stub
client would prove none of that: it would only prove that the adapter agrees
with the stub.

Nothing here opens a socket. ``MockTransport`` answers every request from the
handler this module installs, so a test that accidentally reached the network
would fail rather than call a provider.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import anthropic
import httpx

#: Not a credential. The transport never leaves the process, and the SDK
#: requires *some* key to build a client, so this is the placeholder used to
#: prove that the key the adapter resolves is the one the header carries.
FAKE_API_KEY = "test-key-not-a-credential"

#: What the fake provider answers on, so a base-URL assertion has something to
#: compare against without naming the real service.
FAKE_BASE_URL = "https://provider.invalid"


def tool_use_block(name: str, arguments: Any, *, block_id: str = "toolu_test") -> Any:
    return {"type": "tool_use", "id": block_id, "name": name, "input": arguments}


def text_block(text: str) -> Any:
    return {"type": "text", "text": text}


def message_body(
    content: Sequence[Any],
    *,
    stop_reason: str = "tool_use",
    input_tokens: int = 137,
    output_tokens: int = 29,
    model: str = "claude-test-20990101",
) -> dict[str, Any]:
    """One Messages API response body, as the wire carries it."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(content),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class RecordingTransport:
    """Answers each request from a scripted sequence, recording what arrived.

    A step is either a response body (answered ``200``), an ``(int, body)`` pair
    (answered with that status, which is what makes the SDK raise its own typed
    error), or an exception instance to raise from the transport itself — the
    only way to exercise a connection failure or a read timeout honestly.
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
            # Raw bytes, so a body can carry a JSON escape that no Python value
            # survives being encoded from — a lone surrogate, for instance.
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

    def client(self, *, max_retries: int = 0) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key=FAKE_API_KEY,
            base_url=FAKE_BASE_URL,
            max_retries=max_retries,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


def scripted_client(*steps: Any) -> tuple[RecordingTransport, anthropic.Anthropic]:
    """A transport and a real SDK client bound to it, in that order."""
    transport = RecordingTransport(steps)
    return transport, transport.client()


def one_call_client(
    action: str, arguments: Any, **kwargs: Any
) -> tuple[RecordingTransport, anthropic.Anthropic]:
    """The common case: a provider that answers one well-formed tool call."""
    return scripted_client(message_body([tool_use_block(action, arguments)], **kwargs))


def error_body(kind: str = "api_error", message: str = "provider said no") -> Any:
    return {"type": "error", "error": {"type": kind, "message": message}}
