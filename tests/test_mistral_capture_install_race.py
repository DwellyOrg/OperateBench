"""Two adapters, one unwrapped Mistral SDK client, one capture that gets installed.

The Mistral capture is claimed per wrapper: ``CapturedHttpClient.claim`` holds a
lock, so two adapters that meet on *one* wrapper are decided cleanly and the
second is refused at construction. That decision is only reachable when the
wrapper both adapters find is the same object.

``ensure_capturing_client`` reads ``sdk_configuration.client``, builds a wrapper
around what it read, and installs it — three steps with nothing holding them
together. Two constructors that both read the same *unwrapped* client each build
their own wrapper and each claim the one they built, so both claims succeed on
two different objects and both constructors return. Only one wrapper survives
the install; the other adapter now holds a capture that is attached to nothing,
and the SDK client it will send through belongs to the adapter that installed
last. Nothing refuses it until its first turn, where the response it is measured
from was never captured at all.

That is a construction-time configuration error being reported as a turn-time
one, on an adapter the caller was told was built. This module holds it open with
a gate rather than hoping for it: the first constructor is parked inside the
install window — provably there, because the only way to reach the gate is to
have found the client unwrapped — and the second is let in behind it.

Every test here is two real threads over a real SDK client bound to an
in-process transport. Nothing opens a socket, and the whole point of the first
test is that nothing dispatches at all.
"""

from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest

from boundarybench.adapter import TurnDeadline, TurnRequest, build_turn_request
from boundarybench.compiler import Variant
from boundarybench.providers import mistral_chat
from boundarybench.providers.mistral_chat import (
    CapturedHttpClient,
    MistralChatAdapter,
    MistralConfigurationError,
    ensure_capturing_client,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests import mistral_transport
from tests.conftest import minimal_card

MISTRAL_MODEL = "mistral-test-20990101"
ACTION = "read_records"
ARGUMENTS = "{}"

#: Every wait here is bounded, so a regression that blocks fails this module
#: rather than hanging the suite. No test proves anything by waiting: the
#: interleaving is decided by events, and the bounds are only an upper limit on
#: how long a broken build is allowed to look busy.
WAIT = 10.0

#: Thread names are how the gate tells the two constructors apart. A name rather
#: than an identity because the gate is entered from inside the code under test.
FIRST = "mistral-constructor-first"
SECOND = "mistral-constructor-second"

#: What a module-level ``threading.Lock`` is, as a type. Used to find the fix's
#: serialisation point by shape instead of by name — see :func:`_arm`.
_LOCK_TYPE = type(threading.Lock())


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


def _transport() -> mistral_transport.RecordingTransport:
    """A transport that would answer a turn, and records every request it sees."""
    return mistral_transport.RecordingTransport(
        [],
        default=mistral_transport.chat_body(
            [mistral_transport.tool_call(ACTION, ARGUMENTS)], model=MISTRAL_MODEL
        ),
    )


class _InstallGate:
    """Holds the first constructor open inside the install window.

    A thread only reaches :meth:`arrive` by having found the SDK client
    unwrapped and built a wrapper for it, which is exactly the state the second
    constructor must not be able to reach at the same time. So the first
    constructor parks there and the second is started behind it.

    The second constructor announces itself in one of two places, and which one
    is the whole question this module asks. It reaches the same window — in
    which case it announces from :meth:`arrive`, both wrappers exist at once and
    the race is open — or it is queued at the serialisation point that keeps it
    out, in which case it announces from :class:`_AnnouncingLock` before it
    blocks. Either way the announcement is an event and not an interval, so
    nothing here is timing-dependent in the direction that matters: the first
    constructor is released only once the second has committed.
    """

    def __init__(self) -> None:
        self.first_inside = threading.Event()
        self.second_committed = threading.Event()
        self.release_first = threading.Event()
        self._lock = threading.Lock()
        #: Every wrapper built while the gate is armed, in construction order.
        self.wrappers: list[CapturedHttpClient] = []

    def arrive(self, wrapper: CapturedHttpClient) -> None:
        name = threading.current_thread().name
        if name not in (FIRST, SECOND):
            # A wrapper built by the test's own setup rather than by one of the
            # two constructors under test. Not part of the race, so not recorded
            # and not gated.
            return
        with self._lock:
            self.wrappers.append(wrapper)
        if name == FIRST:
            self.first_inside.set()
            assert self.release_first.wait(WAIT), (
                "the first constructor was never released"
            )
        else:
            self.second_committed.set()

    def announce_queued(self) -> None:
        if threading.current_thread().name == SECOND:
            self.second_committed.set()


class _AnnouncingLock:
    """A module-level lock, wrapped so a queued acquirer says so before it blocks.

    Only meaningful once the source has a serialisation point to wrap. On a
    build without one this is never installed and the gate hears from the second
    constructor directly, which is the reported failure.
    """

    def __init__(self, inner: Any, gate: _InstallGate) -> None:
        self._inner = inner
        self._gate = gate

    def __enter__(self) -> _AnnouncingLock:
        self._gate.announce_queued()
        self._inner.acquire()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self._inner.release()
        return False

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        self._gate.announce_queued()
        return bool(self._inner.acquire(*args, **kwargs))

    def release(self) -> None:
        self._inner.release()


def _arm(monkeypatch: pytest.MonkeyPatch, gate: _InstallGate) -> None:
    """Instrument the install window, without naming anything private.

    The wrapper class is replaced by a subclass, so that building a wrapper is
    observable and every ``isinstance`` in the module still answers "wrapped"
    for one. It has to be armed *before* any wrapper a test wants the module to
    recognise is built, because a wrapper of the original class is not an
    instance of the subclass and the module would read that client as unwrapped.

    The serialisation point is found by shape — a lock held at module level — so
    this does not encode where the fix chose to put it or what it is called, and
    a build that has none is instrumented exactly as far as it can be.
    """

    class _GatedCapturedHttpClient(CapturedHttpClient):
        def __init__(self, inner: Any) -> None:
            super().__init__(inner)
            gate.arrive(self)

    monkeypatch.setattr(mistral_chat, "CapturedHttpClient", _GatedCapturedHttpClient)
    for name, value in list(vars(mistral_chat).items()):
        if isinstance(value, _LOCK_TYPE):
            monkeypatch.setattr(mistral_chat, name, _AnnouncingLock(value, gate))


def _construct(client: Any, into: dict[str, Any], name: str) -> None:
    """One constructor, with whatever it produced recorded under its own name."""
    try:
        into[name] = MistralChatAdapter(model=MISTRAL_MODEL, client=client)
    except Exception as exc:  # recorded here, asserted on by the caller
        into[name] = exc


def _built(outcomes: dict[str, Any]) -> list[MistralChatAdapter]:
    return [value for value in outcomes.values() if isinstance(value, MistralChatAdapter)]


def _refused(outcomes: dict[str, Any]) -> list[MistralConfigurationError]:
    return [
        value
        for value in outcomes.values()
        if isinstance(value, MistralConfigurationError)
    ]


def test_two_racing_constructors_over_one_unwrapped_client_leave_one_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported race: both constructors return, and one of them is a lie.

    Two adapters read one unwrapped client, build one wrapper each and claim the
    one they built. Both claims succeed because they are claims on two different
    objects, so both constructors return an adapter the caller is entitled to
    use — while only the last wrapper installed is the one the SDK client will
    send through. The loser is refused at its first turn instead, after a
    reservation, which is a configuration error charged as a turn.

    What must happen instead is decided here, at construction and before any
    transport: exactly one adapter is built, exactly one caller is told why it
    was not, one wrapper is installed, and the adapter that was built works.
    """
    gate = _InstallGate()
    transport = _transport()
    client = transport.client()
    raw = client.sdk_configuration.client
    assert not isinstance(raw, CapturedHttpClient)
    _arm(monkeypatch, gate)

    outcomes: dict[str, Any] = {}
    first = threading.Thread(
        target=_construct, args=(client, outcomes, FIRST), name=FIRST
    )
    first.start()
    try:
        assert gate.first_inside.wait(WAIT), (
            "the first constructor never reached the window"
        )
        second = threading.Thread(
            target=_construct, args=(client, outcomes, SECOND), name=SECOND
        )
        second.start()
        # The second constructor has committed: it is either inside the same
        # window or queued outside it. Only then is the first released.
        assert gate.second_committed.wait(WAIT), "the second constructor never committed"
    finally:
        gate.release_first.set()
        first.join(WAIT)
    second.join(WAIT)
    assert not first.is_alive()
    assert not second.is_alive()

    # One adapter, one refusal, and nothing else — an unexpected exception type
    # would otherwise satisfy "exactly one was built" while meaning something
    # entirely different happened.
    assert len(outcomes) == 2
    assert len(_built(outcomes)) == 1
    assert len(_refused(outcomes)) == 1
    assert len(_built(outcomes)) + len(_refused(outcomes)) == len(outcomes)
    # The refusal names the remedy and nothing about a provider.
    assert "separate SDK client" in str(_refused(outcomes)[0])

    # Exactly one wrapper was ever built, so the loser holds no capture that is
    # attached to nothing, and the installed one wraps the raw client rather
    # than another wrapper.
    assert len(gate.wrappers) == 1
    installed = client.sdk_configuration.client
    assert installed is gate.wrappers[0]
    assert isinstance(installed, CapturedHttpClient)
    assert installed is not raw

    # Refused before any transport, which is the point of refusing at
    # construction at all.
    assert transport.calls == 0

    winner = _built(outcomes)[0]
    # The capture belongs to the adapter that was built, and to nobody else:
    # re-establishing it for the winner is the idempotent no-op the dispatch
    # path relies on, and anyone else is still turned away.
    assert ensure_capturing_client(client, owner=winner) is installed
    assert client.sdk_configuration.client is installed
    with pytest.raises(MistralConfigurationError):
        ensure_capturing_client(client, owner=object())
    assert client.sdk_configuration.client is installed

    # And the adapter that was built is the one that works.
    call = winner.next_call(_turn_request(), _deadline())
    assert call.action == ACTION
    assert transport.calls == 1
    assert installed.captured_count == 0


def test_racing_constructors_over_separate_clients_both_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialising the install must not turn two clients into one refusal.

    The remedy the refusal names is one SDK client per adapter, so that
    arrangement has to keep working while the first constructor is held open
    inside the window. Both adapters are built, each owns the capture on its own
    client, and each takes a turn through its own transport.
    """
    gate = _InstallGate()
    first_transport, second_transport = _transport(), _transport()
    first_client, second_client = first_transport.client(), second_transport.client()
    _arm(monkeypatch, gate)

    outcomes: dict[str, Any] = {}
    first = threading.Thread(
        target=_construct, args=(first_client, outcomes, FIRST), name=FIRST
    )
    first.start()
    try:
        assert gate.first_inside.wait(WAIT), (
            "the first constructor never reached the window"
        )
        second = threading.Thread(
            target=_construct, args=(second_client, outcomes, SECOND), name=SECOND
        )
        second.start()
        assert gate.second_committed.wait(WAIT), "the second constructor never committed"
    finally:
        gate.release_first.set()
        first.join(WAIT)
    second.join(WAIT)
    assert not first.is_alive()
    assert not second.is_alive()

    adapters = _built(outcomes)
    assert len(adapters) == 2, outcomes
    # Two clients, two wrappers, and neither client is holding the other's.
    assert len(gate.wrappers) == 2
    first_capture = first_client.sdk_configuration.client
    second_capture = second_client.sdk_configuration.client
    assert isinstance(first_capture, CapturedHttpClient)
    assert isinstance(second_capture, CapturedHttpClient)
    assert first_capture is not second_capture
    assert {id(first_capture), id(second_capture)} == {id(w) for w in gate.wrappers}
    for adapter in adapters:
        assert adapter.next_call(_turn_request(), _deadline()).action == ACTION
    assert first_transport.calls == 1
    assert second_transport.calls == 1


def test_racing_constructors_over_a_prewrapped_client_still_refuse_the_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim already decided this case, and it must decide it the same way.

    A client wrapped before either adapter exists is the arrangement
    ``ensure_capturing_client`` already handles: both constructors find the same
    wrapper, and the wrapper's own claim picks one. Whatever serialises the
    install must leave that outcome exactly as it is — one owner, one refusal,
    the same wrapper as before, and no second wrapper built over the first.
    """
    gate = _InstallGate()
    transport = _transport()
    client = transport.client()
    # Armed before the client is wrapped, so the wrapper both constructors find
    # is the instrumented class they would have built themselves — otherwise
    # every ``isinstance`` in the module would answer "not wrapped" and the
    # arrangement under test would never be reached.
    _arm(monkeypatch, gate)
    wrapper = ensure_capturing_client(client)
    assert client.sdk_configuration.client is wrapper

    started = threading.Barrier(2, timeout=WAIT)
    outcomes: dict[str, Any] = {}

    def construct(name: str) -> None:
        started.wait()
        _construct(client, outcomes, name)

    threads = [
        threading.Thread(target=construct, args=(name,), name=name)
        for name in (FIRST, SECOND)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT)
        assert not thread.is_alive()

    assert len(_built(outcomes)) == 1
    assert len(_refused(outcomes)) == 1
    # No wrapper was built at all: there was one to find, and it is still the
    # one installed.
    assert gate.wrappers == []
    assert client.sdk_configuration.client is wrapper
    assert _built(outcomes)[0].next_call(_turn_request(), _deadline()).action == ACTION
    assert transport.calls == 1


def test_a_refused_constructor_leaves_no_half_installed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loser changes nothing: not the wrapper, not the client, not the capture.

    A constructor that fails must fail whole. The client it was handed keeps the
    winner's wrapper, that wrapper holds no response, and the raw HTTP client it
    wraps is not reachable as the client the SDK would send through — a
    half-rewired client would send a request whose response nothing captured,
    which is the one thing this build refuses to pay for.
    """
    gate = _InstallGate()
    transport = _transport()
    client = transport.client()
    raw = client.sdk_configuration.client
    _arm(monkeypatch, gate)

    outcomes: dict[str, Any] = {}
    first = threading.Thread(
        target=_construct, args=(client, outcomes, FIRST), name=FIRST
    )
    first.start()
    try:
        assert gate.first_inside.wait(WAIT)
        second = threading.Thread(
            target=_construct, args=(client, outcomes, SECOND), name=SECOND
        )
        second.start()
        assert gate.second_committed.wait(WAIT)
    finally:
        gate.release_first.set()
        first.join(WAIT)
    second.join(WAIT)

    installed = client.sdk_configuration.client
    assert isinstance(installed, CapturedHttpClient)
    assert installed is not raw
    assert installed.captured_count == 0
    assert len(_refused(outcomes)) == 1
    # The refusal is a configuration error, not a fault attributed to a
    # provider that was never called.
    assert isinstance(_refused(outcomes)[0], MistralConfigurationError)
    assert transport.calls == 0
    # Whatever the losing constructor built is not what the SDK will send
    # through, and the winner's capture is undisturbed by it.
    assert len(gate.wrappers) == 1
    assert isinstance(raw, httpx.Client)
