"""Explicit synchronous httpx transport interop for Anthropic's httpx2 client.

The other SDKs and the shared evidence/budget transports still use httpx. Keep
that boundary local: never alias either package globally in a mixed SDK process.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import httpx2


def _request_error(
    exception: httpx.RequestError, request: httpx2.Request
) -> httpx2.RequestError:
    # Preserve the nearest public class even for an operator-defined subclass.
    for cls in type(exception).__mro__:
        counterpart = getattr(httpx2, cls.__name__, None)
        if (
            getattr(httpx, cls.__name__, None) is cls
            and isinstance(counterpart, type)
            and issubclass(counterpart, httpx2.RequestError)
        ):
            return counterpart(str(exception), request=request)
    return httpx2.RequestError(str(exception), request=request)


class _ResponseStream(httpx2.SyncByteStream):
    def __init__(self, response: httpx.Response, request: httpx2.Request) -> None:
        self._response, self._request = response, request

    def __iter__(self) -> Iterator[bytes]:
        try:
            # The original encoded stream, not iter_bytes() or read(): httpx2
            # must be the only decoder, including for compressed live replies.
            assert isinstance(self._response.stream, httpx.SyncByteStream)
            yield from self._response.stream
        except httpx.RequestError as exc:
            self.close()
            raise _request_error(exc, self._request) from exc
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._response.close()


class _AnthropicTransport(httpx2.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        forwarded = httpx.Request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            content=request.read(),
            extensions=request.extensions,
        )
        forwarded.extensions = request.extensions
        try:
            response = self._inner.handle_request(forwarded)
        except httpx.RequestError as exc:
            raise _request_error(exc, request) from exc
        try:
            content = response.content
        except httpx.ResponseNotRead:
            return httpx2.Response(
                response.status_code,
                headers=response.headers.raw,
                stream=_ResponseStream(response, request),
                extensions=response.extensions,
            )
        # MockTransport and BudgetWire can return already-decoded content.
        # Buffer before restoring headers so Content-Encoding is not applied
        # twice; like httpx, the buffered object retains the original headers.
        try:
            result = httpx2.Response(
                response.status_code, content=content, extensions=response.extensions
            )
            result.headers = httpx2.Headers(response.headers.raw)
            return result
        finally:
            response.close()

    def close(self) -> None:
        self._inner.close()


def anthropic_http_client(
    *, transport: httpx.BaseTransport, timeout: float | None = 5.0
) -> httpx2.Client:
    """Own one explicit transport chain, without environment proxy discovery."""
    return httpx2.Client(
        transport=_AnthropicTransport(transport),
        timeout=timeout,
        trust_env=False,
        follow_redirects=False,
    )
