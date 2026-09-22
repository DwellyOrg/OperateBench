"""The integers a provider-reported token count is allowed to be.

Two readers answer "is this a token count": the typed one in
:mod:`boundarybench.providers.common`, which reads what an SDK handed back, and
the wire one in :mod:`boundarybench.providers.wire`, which reads what the
provider actually sent. Both bounded the domain from below and neither bounded
it from above, so a body reporting ``2**53`` — or an integer of five thousand
digits — was a count both of them accepted.

What that bought is the whole subject of this module. Nothing downstream is
built for such a number:

* :data:`~boundarybench.pricing.MAX_EXACT_TOKEN_COUNT` is the largest integer a
  JSON number carries exactly, so a count past it is one a ledger row could not
  be read back as itself even if every other step succeeded.
* An accepted count is multiplied by the run's pinned rates and committed
  against its cost cap, and a count no reader can round-trip is a cost nobody
  can re-derive.
* Python refuses to convert an integer of more than 4,300 digits to text, so a
  large enough count turns the *serialiser* into the thing that fails —
  ``ValueError`` from inside ``json.dumps``, at the moment a durable row was
  being written, in a type no caller of the JSON boundary catches.

So the bound belongs where the count is read, and it is the pricing constant
rather than a second number that means the same thing. The refusal is
``provider_response_invalid``: a response *did* arrive, so the attempt says it
did, the reservation stays as exposure rather than settling to a zero nobody
measured, and the durable detail is this build's own fixed sentence that never
quotes what the provider sent.

The last section covers the layer beneath: canonical JSON turns an integer it
cannot render into this build's named boundary error rather than the
interpreter's ``ValueError``. That is defence in depth, not the control — the
control is that no such count is ever put into a payload.

Every provider test here runs the real installed SDK over an in-process
``httpx.MockTransport``. Nothing opens a socket.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pytest

from boundarybench.adapter import (
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import (
    CostControls,
    CostReservationBreachedError,
    RunCostGuard,
)
from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.jsonsafe import (
    MAX_JSON_INTEGER_DIGITS,
    JsonSafetyError,
    NonJsonValueError,
    UnrenderableValueError,
    canonical_json_bytes,
    canonical_json_text,
    ensure_json_safe,
)
from boundarybench.pricing import MAX_EXACT_TOKEN_COUNT, price_for
from boundarybench.providers import common, wire
from boundarybench.providers.common import checked_token_usage, is_token_count
from boundarybench.providers.mistral_chat import MistralChatAdapter
from boundarybench.providers.openai_responses import OpenAIResponsesAdapter
from boundarybench.providers.wire import is_wire_token_count, wire_token_usage
from boundarybench.providers.xai_openai_compat import XAIOpenAICompatAdapter
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from tests import mistral_transport, openai_transport
from tests.conftest import minimal_card

OPENAI_MODEL = "gpt-5.6-luna"
XAI_MODEL = "grok-4.5"
MISTRAL_MODEL = "mistral-small-2603"

#: One past the bound: the smallest integer this build refuses, and the one a
#: reader that thought "non-negative" was the whole rule would accept.
PAST_THE_BOUND = MAX_EXACT_TOKEN_COUNT + 1

#: An integer far past it but still renderable, so a test can prove the refusal
#: is the boundary's rather than the serialiser's.
ABSURD_COUNT = 10**30

#: An integer Python itself will not convert to text. Five thousand and one
#: digits, against an interpreter limit of 4,300.
UNRENDERABLE_COUNT = 10**5000

#: The two integers either side of the limit, one apart. The first has exactly
#: :data:`~boundarybench.jsonsafe.MAX_JSON_INTEGER_DIGITS` digits and renders;
#: the second has one more and does not. They share a bit length, which is the
#: point: a boundary stated only in bits cannot tell them apart, and the one it
#: gets wrong is the one that has to be refused.
LAST_RENDERABLE = 10**MAX_JSON_INTEGER_DIGITS - 1
FIRST_UNRENDERABLE = 10**MAX_JSON_INTEGER_DIGITS

#: The three of them as parameters. Named ids rather than generated ones: pytest
#: builds an id by rendering the value, and rendering the last of these is the
#: very thing the interpreter refuses.
PAST_THE_DOMAIN = [
    pytest.param(PAST_THE_BOUND, id="one-past-the-bound"),
    pytest.param(ABSURD_COUNT, id="absurd"),
    pytest.param(UNRENDERABLE_COUNT, id="unrenderable"),
]


def _request() -> TurnRequest:
    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=Environment(cube.variants[0]),
        turns_remaining=12,
    )


def _deadline() -> TurnDeadline:
    return TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False)


def _guard(provider: str, model: str) -> RunCostGuard:
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("100"),
            max_episodes=1,
            price=price_for(provider=provider, model=model),
        ),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )


# -- the one bound, stated once ------------------------------------------------


def test_both_readers_bound_the_domain_by_the_one_pricing_constant() -> None:
    """One number, imported, not two numbers that happen to agree today.

    The bound is a property of what a JSON number carries exactly, and it is
    already stated — and already reasoned about — in the pricing module. A
    second literal in a provider module would be a copy that can drift from the
    thing it was copied from, and the drift would show up as a count one layer
    accepts and the next cannot price.
    """
    assert common.MAX_EXACT_TOKEN_COUNT is MAX_EXACT_TOKEN_COUNT
    assert wire.MAX_EXACT_TOKEN_COUNT is MAX_EXACT_TOKEN_COUNT
    for module in (common, wire):
        assert str(MAX_EXACT_TOKEN_COUNT) not in _module_source(module)
        assert "2**53" not in _module_source(module)


def _module_source(module: Any) -> str:
    assert module.__file__ is not None
    with open(module.__file__, encoding="utf-8") as handle:
        return handle.read()


# -- the two readers -----------------------------------------------------------


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(137, id="ordinary"),
        pytest.param(MAX_EXACT_TOKEN_COUNT - 1, id="one-below-the-bound"),
        pytest.param(MAX_EXACT_TOKEN_COUNT, id="at-the-bound"),
    ],
)
def test_every_exactly_representable_count_is_accepted_by_both_readers(
    count: int,
) -> None:
    """The bound is inclusive, and nothing inside it is refused.

    The largest published context window is some nine orders of magnitude below
    this bound, so a build that refused at it would refuse nothing real — but a
    boundary that is one off is a boundary that is wrong, and the count exactly
    at it is the one a reader gets wrong.
    """
    assert is_token_count(count) is True
    assert is_wire_token_count(count) is True
    assert checked_token_usage(count, count) == common.TokenUsage(count, count)
    usage = wire_token_usage(
        {"input_tokens": count, "output_tokens": count},
        fields=("input_tokens", "output_tokens"),
    )
    assert usage == common.TokenUsage(count, count)


@pytest.mark.parametrize("count", PAST_THE_DOMAIN)
def test_a_count_past_the_exact_domain_is_not_a_count_on_either_path(
    count: int,
) -> None:
    """Refused at the read, which is the only place it can be refused safely.

    Past this point the value is arithmetic input: it is multiplied by a rate,
    committed against a cap, and written into a row. Every one of those is a
    worse place to discover the number than the function whose whole job is to
    say whether it is a measurement.
    """
    assert is_token_count(count) is False
    assert is_wire_token_count(count) is False


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(1.0, id="float"),
        pytest.param(Decimal(5), id="decimal"),
        pytest.param("5", id="string"),
    ],
)
def test_the_shapes_that_were_never_counts_are_still_refused(value: Any) -> None:
    """The new upper bound does not loosen anything the readers already refused."""
    assert is_token_count(value) is False
    assert is_wire_token_count(value) is False


def test_absence_still_means_absence_on_the_typed_path_only() -> None:
    """``None`` is an honest "the SDK reported nothing", and only there.

    On the wire it is a documented field the body did not send, which is a
    different fact and a refusal rather than a null.
    """
    assert is_token_count(None) is True
    assert is_wire_token_count(None) is False


# -- what the refusal says -----------------------------------------------------


@pytest.mark.parametrize("count", PAST_THE_DOMAIN)
@pytest.mark.parametrize("position", [0, 1])
def test_the_typed_path_refuses_the_whole_response_without_quoting_it(
    count: int, position: int
) -> None:
    """``provider_response_invalid``, and a detail that carries no provider value.

    The classification is closed-set and this build's own. The number is not:
    it arrived in a body, it can be of any size, and a durable failure row that
    quoted it would be a row whose contents a provider chose — including, at
    five thousand digits, a row that cannot be rendered at all.
    """
    counts: list[Any] = [11, 11]
    counts[position] = count
    with pytest.raises(AdapterProviderError) as raised:
        checked_token_usage(counts[0], counts[1])
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_no_number_leaked(str(raised.value), count)


@pytest.mark.parametrize("count", PAST_THE_DOMAIN)
def test_the_wire_path_refuses_the_whole_response_without_quoting_it(
    count: int,
) -> None:
    """The same refusal, one layer earlier, on the bytes the provider sent."""
    with pytest.raises(AdapterProviderError) as raised:
        wire_token_usage(
            {"input_tokens": count, "output_tokens": 11},
            fields=("input_tokens", "output_tokens"),
        )
    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_no_number_leaked(str(raised.value), count)


def test_the_detail_is_fixed_whatever_the_count_was() -> None:
    """One sentence for the whole class of refusal.

    A detail that varied with the value would be a channel for the value, which
    is the thing a durable row must not carry from a provider.
    """
    details = set()
    for count in (PAST_THE_BOUND, ABSURD_COUNT, UNRENDERABLE_COUNT, -1):
        with pytest.raises(AdapterProviderError) as raised:
            checked_token_usage(count, 11)
        details.add(str(raised.value))
    assert len(details) == 1


def _assert_no_number_leaked(detail: str, count: int) -> None:
    """No rendering of the provider's integer appears in this build's sentence."""
    try:
        rendered = str(count)
    except ValueError:
        # An integer Python will not render cannot appear in text at all, which
        # is the strongest form of the property this asserts.
        return
    assert rendered not in detail
    # Nor the leading digits of it: a truncated count is still a provider value.
    assert rendered[:12] not in detail


# -- the three surfaces, end to end --------------------------------------------


def _openai_body(count: int) -> dict[str, Any]:
    body = openai_transport.responses_body(
        [openai_transport.function_call_item("read_records", "{}")],
        model=OPENAI_MODEL,
    )
    body["usage"]["input_tokens"] = count
    # Left equal to the count rather than summed: ``total_tokens`` is a third
    # number this build never prices on, and a sum that overflowed the bound
    # would make it ambiguous which field the refusal was about.
    body["usage"]["total_tokens"] = count
    return body


def _xai_body(count: int) -> dict[str, Any]:
    body = openai_transport.chat_body(
        [openai_transport.tool_call("read_records", "{}")], model=XAI_MODEL
    )
    body["usage"]["prompt_tokens"] = count
    body["usage"]["total_tokens"] = count
    return body


def _mistral_body(count: int) -> dict[str, Any]:
    body = mistral_transport.chat_body(
        [mistral_transport.tool_call("read_records", "{}")], model=MISTRAL_MODEL
    )
    body["usage"]["prompt_tokens"] = count
    body["usage"]["total_tokens"] = count
    return body


def _turn(provider: str, count: int) -> tuple[Any, Any, RunCostGuard]:
    if provider == "openai":
        guard = _guard("openai", OPENAI_MODEL)
        transport, client = openai_transport.scripted_client(_openai_body(count))
        return (
            transport,
            OpenAIResponsesAdapter(
                model=OPENAI_MODEL,
                client=client,
                cost_guard=guard,
                sleep=lambda _s: None,
            ),
            guard,
        )
    if provider == "xai":
        guard = _guard("xai", XAI_MODEL)
        transport, client = openai_transport.xai_scripted_client(_xai_body(count))
        return (
            transport,
            XAIOpenAICompatAdapter(
                model=XAI_MODEL, client=client, cost_guard=guard, sleep=lambda _s: None
            ),
            guard,
        )
    guard = _guard("mistral", MISTRAL_MODEL)
    transport, client = mistral_transport.scripted_client(_mistral_body(count))
    return (
        transport,
        MistralChatAdapter(
            model=MISTRAL_MODEL, client=client, cost_guard=guard, sleep=lambda _s: None
        ),
        guard,
    )


@pytest.mark.parametrize("provider", ["openai", "xai", "mistral"])
@pytest.mark.parametrize(
    "count",
    [
        pytest.param(PAST_THE_BOUND, id="one-past-the-bound"),
        pytest.param(ABSURD_COUNT, id="absurd"),
    ],
)
def test_a_body_reporting_an_unpriceable_count_is_refused_by_every_adapter(
    provider: str, count: int
) -> None:
    """The vertical each defect actually travelled: a real body, a real SDK.

    The economics are the load-bearing part. A refusal that let the reservation
    settle would make a body stating ``2**53`` tokens the cheapest turn a run
    can buy, and a refusal that retried would send the same request again on a
    budget the first one already spent.
    """
    transport, adapter, guard = _turn(provider, count)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_request(), _deadline())

    assert raised.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    _assert_no_number_leaked(str(raised.value), count)
    # One request. A body this build cannot read is not made readable by asking
    # for it again.
    assert transport.calls == 1

    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["attempt_count"] == 1
    assert telemetry["response_received"] is True
    assert telemetry["usage_reported"] is False
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_FORFEITED

    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["cost_usd"] is None

    assert guard.exposure_usd > 0
    assert guard.measured_usd == Decimal(0)


@pytest.mark.parametrize("provider", ["openai", "xai", "mistral"])
def test_a_body_reporting_the_largest_exact_count_is_still_read(provider: str) -> None:
    """The bound is inclusive end to end, not only in the predicate.

    A count at the bound is priceable, and pricing it is exact — so the turn is
    *measured*. It is also, at these rates, several billion dollars, so what
    stops the run is the cost control doing its job rather than the read
    refusing a number it could price: the count reaches the usage row, the
    settlement is measured, and the guard raises the breach it exists to raise.
    """
    transport, adapter, guard = _turn(provider, MAX_EXACT_TOKEN_COUNT)

    with pytest.raises(CostReservationBreachedError):
        adapter.next_call(_request(), _deadline())

    assert transport.calls == 1
    telemetry = adapter.last_telemetry().as_dict()
    assert telemetry["usage_reported"] is True
    assert telemetry["attempts"][0]["cost_settlement"] == ATTEMPT_SETTLEMENT_MEASURED
    usage = adapter.last_usage().as_dict()
    assert usage["input_tokens"] == MAX_EXACT_TOKEN_COUNT
    assert guard.measured_usd > 0


# -- defence in depth: the JSON boundary ---------------------------------------


def test_canonical_json_turns_an_unrenderable_integer_into_a_named_refusal() -> None:
    """``json.dumps`` raising ``ValueError`` is not a failure a caller can act on.

    Every caller of this boundary catches :class:`JsonSafetyError` and rewrites
    it into its own domain error — "this ledger row", "this scaffold". A bare
    ``ValueError`` from the interpreter's integer-to-text limit walks past all
    of them and surfaces as a traceback from whichever writer happened to be
    running.
    """
    with pytest.raises(JsonSafetyError) as raised:
        canonical_json_text({"input_tokens": UNRENDERABLE_COUNT}, "a durable row")
    assert "a durable row" in str(raised.value)
    # This build's own bounded detail: the context it was given and the limit it
    # applies. Never the value, which is why the number is not rendered even to
    # say how large it was.
    assert "input_tokens" in str(raised.value)


def test_the_same_refusal_reaches_the_bytes_and_the_walk() -> None:
    """One rule, whichever of the three entry points a caller uses."""
    payload = [UNRENDERABLE_COUNT]
    for call in (ensure_json_safe, canonical_json_text, canonical_json_bytes):
        with pytest.raises(JsonSafetyError):
            call(payload, "a durable row")


def test_an_integer_the_interpreter_can_render_is_still_serialised() -> None:
    """The refusal is a limit, not a distrust of large numbers.

    Nothing this build puts in a payload is anywhere near it — the token domain
    stops sixteen digits in — but the boundary is a general one and refusing a
    perfectly renderable integer would be a second bug in the other direction.
    """
    value = 10**4000
    assert json.loads(canonical_json_text({"n": value}, "a row"))["n"] == value


def test_a_programmer_type_error_is_not_dressed_up_as_unsafe_json() -> None:
    """The wrapper catches the serialiser's ``ValueError`` and nothing else.

    A value of a type JSON cannot carry is already refused by name before the
    encoder runs. What is left for ``json.dumps`` to raise ``TypeError`` about
    is a mistake in this build — an object shaped like a sequence that no
    encoder was taught to write — and swallowing it as a boundary refusal would
    turn a bug here into a report that a payload was untrustworthy.
    """

    class SequenceOfNothing(Sequence[Any]):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: Any) -> Any:  # pragma: no cover - never indexed
            raise IndexError(index)

    with pytest.raises(TypeError) as raised:
        canonical_json_text(SequenceOfNothing(), "a row")
    assert not isinstance(raised.value, JsonSafetyError)


def test_a_value_json_cannot_carry_is_still_the_named_refusal() -> None:
    """The pre-existing shapes keep their own error classes."""
    with pytest.raises(NonJsonValueError):
        canonical_json_text({"s": {1, 2}}, "a row")


# -- the digit limit, at the digit ---------------------------------------------
#
# The walk's promise is that a value it accepts can be serialised. A bound
# expressed only in bits cannot keep it: bits are a coarser measure than digits,
# so integers on both sides of the digit limit share a bit length and a
# bits-only test has to answer the same way for both. Whichever way it answers,
# it is wrong about one of them — and being wrong in the accepting direction
# means the refusal arrives from the encoder, at the moment a durable artefact
# was being written, which is the failure this whole boundary exists to prevent.


def test_the_last_renderable_integer_is_carried() -> None:
    """Exactly :data:`MAX_JSON_INTEGER_DIGITS` digits is inside the limit.

    The limit is on conversion, and an integer of exactly that many digits
    converts. Refusing it would be the same bug pointed the other way.
    """
    assert ensure_json_safe({"n": LAST_RENDERABLE}, "a row") is not None
    assert json.loads(canonical_json_text({"n": LAST_RENDERABLE}, "a row"))["n"] == (
        LAST_RENDERABLE
    )


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_the_first_unrenderable_integer_is_refused_by_the_walk(sign: int) -> None:
    """One more digit, and the walk itself says no — not the encoder behind it.

    The negative case is not symmetry for its own sake: magnitude is what the
    interpreter converts, so a check that read the sign as part of the size
    would let the mirror image of a refused value through.
    """
    value = sign * FIRST_UNRENDERABLE
    for call in (ensure_json_safe, canonical_json_text, canonical_json_bytes):
        with pytest.raises(UnrenderableValueError):
            call({"n": value}, "a durable row")


def test_two_integers_of_the_same_bit_length_are_told_apart() -> None:
    """The measure has to be the magnitude, not an approximation of it.

    These two differ by less than a bit and by a decimal digit, so any bound
    that rounds bits to digits accepts both or refuses both.
    """
    accepted = 1 << (FIRST_UNRENDERABLE.bit_length() - 1)
    assert accepted.bit_length() == FIRST_UNRENDERABLE.bit_length()
    assert ensure_json_safe(accepted, "a row") == accepted
    with pytest.raises(UnrenderableValueError):
        ensure_json_safe(FIRST_UNRENDERABLE, "a row")


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_an_integer_of_five_thousand_and_one_digits_is_refused(sign: int) -> None:
    """Far past the limit answers the same way as one past it."""
    with pytest.raises(UnrenderableValueError):
        ensure_json_safe({"n": sign * UNRENDERABLE_COUNT}, "a durable row")


def test_the_walk_never_renders_the_integer_it_is_judging() -> None:
    """Deciding whether a value can be rendered must not render it.

    Rendering is the operation the interpreter refuses, so a check that reached
    for the decimal form to count its digits would raise the very failure it
    exists to report — and would raise it as the interpreter's ``ValueError``,
    which no caller of this boundary catches.
    """

    class RefusesToBeRendered(int):
        def __str__(self) -> str:
            raise AssertionError("the walk rendered the integer it was judging")

        def __repr__(self) -> str:
            raise AssertionError("the walk rendered the integer it was judging")

    assert ensure_json_safe(RefusesToBeRendered(LAST_RENDERABLE), "a row") is not None
    with pytest.raises(UnrenderableValueError):
        ensure_json_safe(RefusesToBeRendered(FIRST_UNRENDERABLE), "a row")


def test_booleans_are_still_booleans_and_not_narrow_integers() -> None:
    """``bool`` is an ``int`` subclass, and it is answered as ``bool`` first."""
    carried = json.loads(canonical_json_text({"t": True, "f": False}, "a row"))
    assert carried == {"t": True, "f": False}
    assert isinstance(carried["t"], bool)


# -- what the encoder's ValueError actually was --------------------------------
#
# ``canonical_json_text`` translates the interpreter's integer-to-text refusal
# into a named boundary error, because a bare ``ValueError`` walks past every
# caller's ``except JsonSafetyError``. That translation has to be about the one
# failure it names. ``json.dumps`` raises ``ValueError`` for other reasons — a
# structure that became circular after the walk proved it acyclic, most plainly
# — and relabelling those reports the wrong cause in a durable failure row.


class _ChangesAfterTheWalk(dict[str, Any]):
    """A mapping that is safe when the walk reads it and something else after.

    Not a hypothetical: any mapping whose contents are computed rather than
    stored can differ between the two reads, and the walk and the encoder are
    two reads. It is built here as a dict subclass because that is the only
    mapping ``json.dumps`` will encode at all.
    """

    def __init__(self, later: Any) -> None:
        # Real storage stays non-empty so the encoder asks for the items rather
        # than short-circuiting on an empty object.
        super().__init__({"n": 1})
        self._later = later
        self._reads = 0

    def items(self) -> Any:
        self._reads += 1
        if self._reads == 1:
            return (("n", 1),)
        return self._later(self)


def test_a_late_circular_reference_is_not_relabelled_as_a_huge_integer() -> None:
    """The encoder's own words, unchanged, for a failure that is not the limit.

    A row refused as "an integer too wide to render" when what actually
    happened was a cycle is a diagnosis nobody can act on: the operator looks
    for a number that is not there.
    """
    payload = _ChangesAfterTheWalk(lambda mapping: (("self", mapping),))
    with pytest.raises(ValueError) as raised:
        canonical_json_text(payload, "a durable row")
    assert not isinstance(raised.value, JsonSafetyError)
    assert type(raised.value) is ValueError
    assert "Circular reference detected" in str(raised.value)


def test_an_unrelated_encoder_value_error_propagates_as_the_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not re-raised, not wrapped: the same exception instance leaves the call."""
    unrelated = ValueError("the encoder had a problem of its own")

    def explode(*args: Any, **kwargs: Any) -> str:
        raise unrelated

    monkeypatch.setattr(json, "dumps", explode)
    with pytest.raises(ValueError) as raised:
        canonical_json_text({"n": 1}, "a durable row")
    assert raised.value is unrelated


def test_an_integer_that_only_appears_at_encode_time_is_the_named_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one failure the translation is for still gets translated.

    This is the path the walk cannot cover — the payload the encoder saw is not
    the payload the walk proved — so it is the reason the translation exists at
    all rather than dead code.
    """
    payload = _ChangesAfterTheWalk(lambda mapping: (("n", UNRENDERABLE_COUNT),))
    with pytest.raises(UnrenderableValueError) as raised:
        canonical_json_text(payload, "a durable row")
    assert "a durable row" in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)


def test_the_translation_recognises_this_interpreter_own_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned to the running interpreter rather than to a transcribed sentence.

    The exception raised here is the one CPython itself produced, and its exact
    wording is a detail of the version under test — 3.11 through 3.14 all ship
    this build. A supported version whose wording moved fails this test rather
    than silently falling through to the propagating branch.
    """
    try:
        str(UNRENDERABLE_COUNT)
    except ValueError as exc:
        interpreter_refusal: ValueError = exc
    else:  # pragma: no cover - the interpreter under test has no digit limit
        pytest.fail("this interpreter renders a five-thousand-digit integer")

    def explode(*args: Any, **kwargs: Any) -> str:
        raise interpreter_refusal

    monkeypatch.setattr(json, "dumps", explode)
    with pytest.raises(UnrenderableValueError) as raised:
        canonical_json_text({"n": 1}, "a durable row")
    assert raised.value.__cause__ is interpreter_refusal
