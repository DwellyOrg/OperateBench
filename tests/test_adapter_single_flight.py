"""One turn at a time, per adapter instance.

Every provider adapter in this build holds mutable per-instance state that a
turn owns while it runs: the last usage, the last attempt evidence, and — for
Mistral — the captured HTTP response the wire checks are taken from. None of it
is per-call, so two ``next_call`` invocations overlapping on one adapter would
interleave into one set of numbers: the second call's ``clear()`` erases the
first call's telemetry, and the capture holds two bodies so both turns refuse
to identify their own.

The product decision is that an adapter instance is **single-flight**, and that
the second caller is refused immediately rather than queued: waiting would make
a benchmark's wall-clock depend on contention it does not record, and could
deadlock a turn that re-entered its own adapter. So the second invocation fails
with a named adapter-state error, makes no provider call, and leaves the first
invocation's measurement exactly as it was.

Every test here runs the real SDKs over in-process transports. No socket is
opened and nothing here is evidence about a model.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest

from boundarybench.adapter import (
    AdapterBusyError,
    AdapterError,
    AdapterProviderError,
    SingleFlight,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant
from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter
from boundarybench.providers.mistral_chat import MistralChatAdapter
from boundarybench.providers.openai_responses import OpenAIResponsesAdapter
from boundarybench.providers.xai_openai_compat import XAIOpenAICompatAdapter
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests import mistral_transport, openai_transport
from tests.anthropic_transport import (
    RecordingTransport as AnthropicTransport,
)
from tests.anthropic_transport import (
    message_body,
    tool_use_block,
)
from tests.conftest import minimal_card

ANTHROPIC_MODEL = "claude-test-20990101"
OPENAI_MODEL = "gpt-test-20990101"
XAI_MODEL = "grok-test-20990101"
MISTRAL_MODEL = "mistral-test-20990101"

ACTION = "read_records"
ARGUMENTS = "{}"


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    from boundarybench.compiler import compile_cube
    from boundarybench.schema import ConstructCard

    return compile_cube(ConstructCard.from_dict(minimal_card())).variants[0]


def _turn_request() -> TurnRequest:
    from boundarybench.environment import Environment

    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(_variant()), turns_remaining=12
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


class _Reentrant:
    """A transport handler that re-enters the adapter mid-request.

    The re-entrant call happens on the same thread, from inside the request the
    adapter is currently dispatching, which is the sharpest form of the overlap
    this guard exists to refuse: the adapter is provably mid-turn. Whatever the
    nested call raises is recorded and swallowed here, so the outer turn
    completes normally and the test can assert on both halves separately.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.adapter: Any = None
        self.raised: list[BaseException] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.adapter is not None:
            try:
                self.adapter.next_call(_turn_request(), _deadline())
            except Exception as exc:  # recorded here, asserted on by the caller
                self.raised.append(exc)
        return self._inner(request)


def _anthropic() -> tuple[Any, Any, _Reentrant]:
    import anthropic

    from tests.anthropic_transport import FAKE_API_KEY, FAKE_BASE_URL

    transport = AnthropicTransport(
        [],
        default=message_body(
            [tool_use_block(ACTION, {})],
            model=ANTHROPIC_MODEL,
        ),
    )
    nested = _Reentrant(transport.handler)
    client = anthropic.Anthropic(
        api_key=FAKE_API_KEY,
        base_url=FAKE_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(nested.handler)),
    )
    adapter = AnthropicMessagesAdapter(model=ANTHROPIC_MODEL, client=client)
    nested.adapter = adapter
    return adapter, transport, nested


def _openai() -> tuple[Any, Any, _Reentrant]:
    transport = openai_transport.RecordingTransport(
        [],
        default=openai_transport.responses_body(
            [openai_transport.function_call_item(ACTION, ARGUMENTS)],
            model=OPENAI_MODEL,
        ),
    )
    nested = _Reentrant(transport.handler)
    client = openai_transport.openai.OpenAI(
        api_key=openai_transport.FAKE_API_KEY,
        base_url=openai_transport.OPENAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(nested.handler)),
    )
    adapter = OpenAIResponsesAdapter(model=OPENAI_MODEL, client=client)
    nested.adapter = adapter
    return adapter, transport, nested


def _xai() -> tuple[Any, Any, _Reentrant]:
    transport = openai_transport.RecordingTransport(
        [],
        default=openai_transport.chat_body(
            [openai_transport.tool_call(ACTION, ARGUMENTS)],
            model=XAI_MODEL,
        ),
    )
    nested = _Reentrant(transport.handler)
    client = openai_transport.openai.OpenAI(
        api_key=openai_transport.FAKE_API_KEY,
        base_url=openai_transport.XAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(nested.handler)),
    )
    adapter = XAIOpenAICompatAdapter(model=XAI_MODEL, client=client)
    nested.adapter = adapter
    return adapter, transport, nested


def _mistral() -> tuple[Any, Any, _Reentrant]:
    transport = mistral_transport.RecordingTransport(
        [],
        default=mistral_transport.chat_body(
            [mistral_transport.tool_call(ACTION, ARGUMENTS)],
            model=MISTRAL_MODEL,
        ),
    )
    nested = _Reentrant(transport.handler)
    from boundarybench.providers.mistral_chat import build_client

    client = build_client(
        api_key=mistral_transport.FAKE_API_KEY,
        http_client=httpx.Client(transport=httpx.MockTransport(nested.handler)),
    )
    adapter = MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    nested.adapter = adapter
    return adapter, transport, nested


ADAPTERS = (
    ("anthropic", _anthropic),
    ("openai", _openai),
    ("xai", _xai),
    ("mistral", _mistral),
)


# -- the refusal --------------------------------------------------------------


@pytest.mark.parametrize(("name", "build"), ADAPTERS)
def test_a_second_overlapping_turn_is_refused_by_name(name: str, build: Any) -> None:
    adapter, _transport, nested = build()

    adapter.next_call(_turn_request(), _deadline())

    assert len(nested.raised) == 1, name
    refusal = nested.raised[0]
    assert isinstance(refusal, AdapterBusyError), refusal
    assert isinstance(refusal, AdapterError)


@pytest.mark.parametrize(("name", "build"), ADAPTERS)
def test_the_refused_turn_makes_no_provider_call(name: str, build: Any) -> None:
    """Zero transport calls from the second caller: it is refused before dispatch."""
    adapter, transport, nested = build()

    adapter.next_call(_turn_request(), _deadline())

    assert nested.raised, name
    assert transport.calls == 1


@pytest.mark.parametrize(("name", "build"), ADAPTERS)
def test_the_refused_turn_does_not_disturb_the_first_ones_measurement(
    name: str, build: Any
) -> None:
    """The first turn's telemetry and usage are exactly its own.

    The failure this guards is silent: the second call's reset runs before the
    first call has published anything, so the first turn's evidence would be a
    cleared object and its attempt would have vanished from the record.
    """
    adapter, _transport, nested = build()

    adapter.next_call(_turn_request(), _deadline())

    assert nested.raised, name
    telemetry = adapter.last_telemetry()
    assert len(telemetry.attempts) == 1
    assert telemetry.attempts[0].index == 1
    usage = adapter.last_usage()
    assert usage.input_tokens == 137
    assert usage.output_tokens == 29


@pytest.mark.parametrize(("name", "build"), ADAPTERS)
def test_sequential_reuse_of_one_adapter_is_still_supported(
    name: str, build: Any
) -> None:
    """The guard is per turn, not per adapter lifetime."""
    adapter, transport, nested = build()
    nested.adapter = None

    adapter.next_call(_turn_request(), _deadline())
    adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 2, name
    assert len(adapter.last_telemetry().attempts) == 1


@pytest.mark.parametrize(("name", "build"), ADAPTERS)
def test_a_turn_that_raised_still_releases_the_guard(name: str, build: Any) -> None:
    """Released in a ``finally``: a failed turn must not wedge the adapter shut."""
    adapter, transport, nested = build()
    nested.adapter = None

    with pytest.raises(AdapterProviderError):
        # A deadline with no budget left raises out of the executor before any
        # request is dispatched, which is the earliest failure a turn has.
        adapter.next_call(
            _turn_request(), TurnDeadline(remaining_seconds=0.0, cancelled=lambda: False)
        )
    adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 1, name


# -- genuinely concurrent callers ---------------------------------------------


def test_two_threads_cannot_both_hold_one_mistral_adapter() -> None:
    """The reported blocker, exactly: two overlapping calls on one capture.

    Both threads are real, the first is held inside the request the transport is
    answering, and the second arrives while it is. The second must fail
    immediately — not wait — so the first is released by an event the test sets,
    and every wait here is bounded so a regression to blocking acquisition fails
    the test rather than hanging the suite.
    """
    entered = threading.Event()
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        assert release.wait(timeout=10.0), "the first turn was never released"
        return inner(request)

    transport = mistral_transport.RecordingTransport(
        [],
        default=mistral_transport.chat_body(
            [mistral_transport.tool_call(ACTION, ARGUMENTS)],
            model=MISTRAL_MODEL,
        ),
    )
    inner = transport.handler
    from boundarybench.providers.mistral_chat import build_client

    adapter = MistralChatAdapter(
        model=MISTRAL_MODEL,
        client=build_client(
            api_key=mistral_transport.FAKE_API_KEY,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )

    first: list[Any] = []

    def run_first() -> None:
        first.append(adapter.next_call(_turn_request(), _deadline()))

    thread = threading.Thread(target=run_first)
    thread.start()
    try:
        assert entered.wait(timeout=10.0), "the first turn never reached the transport"
        # Counted around the refused call rather than absolutely: the first
        # turn's own request is not recorded until the handler it is held inside
        # returns, so what this states is that the refused caller added nothing.
        before = transport.calls
        with pytest.raises(AdapterBusyError):
            adapter.next_call(_turn_request(), _deadline())
        assert transport.calls == before
    finally:
        release.set()
        thread.join(timeout=10.0)

    assert not thread.is_alive()
    assert len(first) == 1
    assert first[0].action == ACTION
    assert transport.calls == 1
    assert len(adapter.last_telemetry().attempts) == 1


# -- the primitive itself -----------------------------------------------------


def test_the_guard_names_the_adapter_that_refused() -> None:
    guard = SingleFlight(adapter="ExampleAdapter")
    with guard, pytest.raises(AdapterBusyError) as raised, guard:
        pass
    assert "ExampleAdapter" in str(raised.value)


def test_the_guard_is_reusable_after_a_failure_inside_it() -> None:
    guard = SingleFlight(adapter="ExampleAdapter")
    with pytest.raises(ValueError), guard:
        raise ValueError("the turn failed")
    with guard:
        pass


def test_the_guard_never_waits_for_a_holder() -> None:
    """Non-blocking acquisition, proven by a holder that is never released."""
    guard = SingleFlight(adapter="ExampleAdapter")
    held = threading.Event()
    done = threading.Event()

    def hold() -> None:
        with guard:
            held.set()
            assert done.wait(timeout=10.0)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert held.wait(timeout=10.0)
        with pytest.raises(AdapterBusyError), guard:
            pass
    finally:
        done.set()
        thread.join(timeout=10.0)
    assert not thread.is_alive()
