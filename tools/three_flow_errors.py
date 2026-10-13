"""Closed, body-free diagnostics. Unknown reasons are not retry authority."""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable
from typing import Any, cast

import httpx


class ProviderPacer:
    """One process-wide cooperative Mistral lane; no automatic resubmission.

    A 429 plus Retry-After alone cannot prove generation was not accepted or
    distinguish quota from rate windows. The delay gates *subsequent* work,
    never refunds an attempt or retries an uncertain paid query.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def call(self, send: Callable[[], httpx.Response]) -> httpx.Response:
        with self._lock:
            delay = self._next - self._clock()
            if delay > 0:
                self._sleep(delay)
            try:
                response = send()
            finally:
                self._next = self._clock() + 1.0
            if response.status_code == 429:
                delay_seconds = retry_after_seconds(response.headers.get("Retry-After"))
                if delay_seconds is not None:
                    self._next = max(self._next, self._clock() + delay_seconds)
            return response


class MetaDispatchStopped(RuntimeError):
    """Local fail-stop; the blocked request was never dispatched."""


class MetaRatePacer:
    """Standard rolling quotas, independent model windows, shared 429 tripwire.

    Reservations use complete serialized input plus requested output. They are
    never discounted by actual usage. No retry or generation deadline is added.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        enabled: bool = True,
    ) -> None:
        self.clock, self.sleep, self.enabled = clock, sleep, enabled
        self.lock = threading.Lock()
        self.windows: dict[str, list[tuple[float, int]]] = {}
        self.model_locks: dict[str, Any] = {}
        self.stopped = threading.Event()

    def call(
        self, model: str, tokens: int, send: Callable[[], httpx.Response]
    ) -> httpx.Response:
        if tokens > 4_000_000 or tokens <= 0:
            raise ValueError("request reservation cannot fit Meta Standard TPM window")
        with self.lock:
            model_lock = self.model_locks.setdefault(model, threading.Lock())
        while True:
            with model_lock:
                if self.stopped.is_set():
                    raise MetaDispatchStopped("Meta fleet stopped after provider failure")
                now = self.clock()
                window = self.windows.setdefault(model, [])
                window[:] = [(t, n) for t, n in window if t + 60 > now]
                if not self.enabled or (
                    len(window) < 3000 and sum(n for _, n in window) + tokens <= 4_000_000
                ):
                    # send's local preparation runs before commit(); failures there
                    # consume no quota. The caller commits at the inner boundary.
                    return send()
                delay = window[0][0] + 60 - now
            # Do not hold the shared lock while waiting for an independent model.
            self.sleep(delay)

    def commit(self, model: str, tokens: int) -> None:
        """Called under the model lock immediately before actual dispatch."""
        self.windows[model].append((self.clock(), tokens))


def grpc_error(exception: BaseException, operation: str) -> dict[str, Any]:
    grpc = importlib.import_module("grpc")
    code_method = getattr(exception, "code", None)
    status = (
        code_method()
        if isinstance(exception, grpc.RpcError) and callable(code_method)
        else None
    )
    # Identity against the installed enum; no str(), details(), header/trailer,
    # SDK wrapper text or unconstrained category is ever copied.
    name = next(
        (cast(Any, code).name for code in grpc.StatusCode if status is code),
        "UNKNOWN",
    )
    return {
        "representation": "sanitized_grpc_error_v2",
        "operation": operation if operation in OPERATIONS else "unknown",
        "grpc_status": name,
        "grpc_status_code": next(
            (cast(Any, code).value[0] for code in grpc.StatusCode if status is code),
            None,
        ),
        "reason": "unknown",
        "automatic_resubmission": False,
    }


def river_failed(message: Any) -> dict[str, Any]:
    # Installed River 0.12 RequestFailedResponse uses free-text category and
    # protobuf Struct details, not a documented reason enum. Do not guess one.
    return {
        "representation": "sanitized_river_failure_v1",
        "operation": "RetrieveFuture",
        "error_category": "unknown",
        "reason": "unknown",
        "structured_details_present": message.HasField("details"),
        "automatic_resubmission": False,
    }


OPERATIONS = (
    "responses",
    "messages",
    "chat_completions",
    "InferenceGenerate",
    "RetrieveFuture",
)


def retry_after_seconds(value: object) -> int | None:
    # Delta-seconds only; dates, fractions, signs, huge integers and booleans
    # are not accepted. This is a diagnostic bound, not a generation deadline.
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 5
    ):
        return None
    seconds = int(value)
    return seconds if 0 <= seconds <= 86400 else None


def http_error(status: object, operation: str, retry_after: object) -> dict[str, Any]:
    return {
        "representation": "sanitized_http_error_v2",
        "diagnostic": "http_non_success",
        "body_retained": False,
        "operation": operation if operation in OPERATIONS else "unknown",
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
        "reason": "unknown",
        "retry_after_seconds": retry_after_seconds(retry_after),
        "automatic_resubmission": False,
    }
