"""The OpenAI response's named server-extension block, checked as an exact shape.

The OpenAI service returns a small number of top-level fields on a Responses
body that the pinned SDK's ``Response`` model does not declare, and that the
2.53 and 3.0 model files do not declare either. This build used to refuse every
such body whole, which is the right default and the wrong answer for a field set
that is stable, named, and read by nothing here.

So the lane now carries one *named* response-extension contract —
``openai_response_server_extensions_v1`` — and accepts exactly it:

* five top-level names and no others, each of them independently optional,
  because a server that stops sending one is not a server that broke the
  contract;
* every named object that *does* arrive stated whole: each child the contract
  fixes under a present object is required there. A half-stated object was never
  observed and is not what this contract describes, so accepting one would make
  ``exact`` mean "some subset of the observed shape";
* every object inside them closed recursively, so an unknown key at any depth is
  still ``provider_response_invalid``;
* every scalar checked for its exact wire type — counts are exact non-negative
  integers and never booleans, the two penalties are finite JSON numbers inside
  the published ``[-2, 2]`` range, the flag is a JSON boolean, and the payer is a
  non-empty bounded string that is never written anywhere.

The contract is **provider-observed**, not vendor-declared: it transcribes the
name and type shape of what the service returned, and the pinned SDK declares
none of it. That is why it has a name and a digest of its own in adapter
settings rather than being read off ``openai.types``, and why accepting it moved
the adapter version.

Nothing in the block is read for an action, for the model's identity or for the
usage a turn is priced on. It is validated so that a body carrying it can be
accepted at all, and then it is dropped.

Every request here is answered by ``httpx.MockTransport`` inside this process.
No socket is opened, no credential is read, and nothing here is evidence about
any model: it is evidence about the integration.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from itertools import combinations
from typing import Any

import pytest
from openai.types.responses import Response

from boundarybench.adapter import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment
from boundarybench.jsonsafe import canonical_json_bytes, canonical_json_text
from boundarybench.pricing import price_for
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENAI_ADAPTER_VERSION,
    OpenAIResponsesAdapter,
    build_responses_request,
    openai_identity,
    openai_settings,
)
from boundarybench.providers.wire import WireResponse
from boundarybench.runmanifest import RunLimits, build_run_manifest
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST, minimal_card
from tests.openai_transport import (
    RecordingTransport,
    chat_body,
    function_call_item,
    responses_body,
    scripted_client,
    tool_call,
)

#: The contract's own name, written out rather than imported: it is inside
#: ``configuration_id``, so an identifier asserted against itself would follow
#: any rename and prove nothing about the run identity a reader compares.
EXTENSIONS_CONTRACT = "openai_response_server_extensions_v2"

#: The name the superseded reading of the same field set carried. ``v1``
#: transcribed these names and types and then made every one of them optional
#: under its own parent, so it accepted bodies — ``billing`` stating no payer,
#: ``tool_usage`` stating neither tool — that nothing observed and nothing
#: justified. It is named here because a run made under it is not a run made
#: under this one, and the identity assertions below have to say so.
SUPERSEDED_EXTENSIONS_CONTRACT = "openai_response_server_extensions_v1"

#: The exact name and type shape the two authorised diagnostics established,
#: hand-written here rather than imported from the module under test. The
#: published schema is asserted against *this* literal, and the digest in
#: settings is asserted against a SHA-256 over its canonical bytes, so a silent
#: widening of the accepted contract fails here rather than in production.
EXPECTED_SCHEMA: dict[str, Any] = {
    "billing": {"payer": "bounded_text"},
    "frequency_penalty": "penalty_number",
    "presence_penalty": "penalty_number",
    "store": "boolean",
    "tool_usage": {
        "image_gen": {
            "input_tokens": "non_negative_integer",
            "input_tokens_details": {
                "image_tokens": "non_negative_integer",
                "text_tokens": "non_negative_integer",
            },
            "output_tokens": "non_negative_integer",
            "output_tokens_details": {
                "image_tokens": "non_negative_integer",
                "text_tokens": "non_negative_integer",
            },
            "total_tokens": "non_negative_integer",
        },
        "web_search": {"num_requests": "non_negative_integer"},
    },
}

#: The whole contract as it is hashed: the name, the two rules that decide what
#: a *present* object has to state, and the shape above.
#:
#: The rules are inside the digest deliberately. The transcribed names and types
#: did not move between ``v1`` and ``v2`` — what moved is that a present object
#: must now state every child fixed for it — so a digest taken over the shape
#: alone would be identical across two builds that read the same body
#: differently, and every manifest would agree while the runs did not.
EXPECTED_CONTRACT: dict[str, Any] = {
    "contract": EXTENSIONS_CONTRACT,
    "members": EXPECTED_SCHEMA,
    "object_members_required": True,
    "top_level_members_required": False,
}

#: One valid block of the observed shape: every name present, every scalar of
#: the observed type. This is what a real HTTP 200 from the pinned model carries
#: beside the response the SDK models, and accepting it is the whole amendment.
OBSERVED_EXTENSIONS: dict[str, Any] = {
    "billing": {"payer": "org-observed"},
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "store": False,
    "tool_usage": {
        "image_gen": {
            "input_tokens": 0,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
            "total_tokens": 0,
        },
        "web_search": {"num_requests": 0},
    },
}

#: Every string this build must never write into a durable failure detail: the
#: contract's own field names at every level, and the values a body carried.
EXTENSION_FIELD_NAMES: tuple[str, ...] = (
    "billing",
    "payer",
    "frequency_penalty",
    "presence_penalty",
    "store",
    "tool_usage",
    "image_gen",
    "web_search",
    "num_requests",
    "input_tokens_details",
    "output_tokens_details",
    "image_tokens",
    "text_tokens",
)


def _plain(node: Any) -> Any:
    """One published schema node as plain data, so it can be compared and hashed.

    The shipped contract is deeply immutable — a schema shared by reference and
    mutable is not a fixed contract — and a ``mappingproxy`` is neither equal to
    a ``dict`` for JSON's encoder nor serialisable by it.
    """
    if isinstance(node, Mapping):
        return {key: _plain(value) for key, value in node.items()}
    return node


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    return compile_cube(ConstructCard.from_dict(minimal_card())).variants[0]


def _turn_request() -> TurnRequest:
    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(_variant()), turns_remaining=12
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _guard() -> RunCostGuard:
    """A real cost guard, so a refusal's effect on the money can be asserted."""
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("10.00"),
            max_episodes=36,
            price=price_for(provider="openai", model=GPT_5_6_LUNA_MODEL),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=ProviderRetryPolicy().max_attempts,
    )


def _body(extensions: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    """One Responses body for the pinned model, with a server-extension block."""
    body = responses_body(
        [function_call_item("read_records", "{}")], model=GPT_5_6_LUNA_MODEL
    )
    if extensions is not None:
        body.update(copy.deepcopy(extensions))
    body.update(copy.deepcopy(overrides))
    return body


def _observed(**changes: Any) -> dict[str, Any]:
    """The observed block, with named top-level members replaced or added."""
    block = copy.deepcopy(OBSERVED_EXTENSIONS)
    block.update(copy.deepcopy(changes))
    return block


def _at(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """The observed block with one nested key set to ``value``."""
    block = copy.deepcopy(OBSERVED_EXTENSIONS)
    node: Any = block
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    return block


def _without(path: tuple[str, ...]) -> dict[str, Any]:
    """The observed block with one nested key removed entirely."""
    block = copy.deepcopy(OBSERVED_EXTENSIONS)
    node: Any = block
    for step in path[:-1]:
        node = node[step]
    del node[path[-1]]
    return block


def _adapter(*steps: Any) -> tuple[OpenAIResponsesAdapter, RunCostGuard]:
    transport = RecordingTransport(list(steps))
    guard = _guard()
    return (
        OpenAIResponsesAdapter(
            model=GPT_5_6_LUNA_MODEL, client=transport.client(), cost_guard=guard
        ),
        guard,
    )


def _accepted(*steps: Any) -> None:
    """One turn that must be read as the action and the counts the body states.

    The whole point of naming an extension contract is that a body carrying it
    is an ordinary turn: the action is parsed, the two counts are measured off
    the wire, and the reservation settles at the measured cost.
    """
    adapter, guard = _adapter(*steps)

    call = adapter.next_call(_turn_request(), _deadline())

    assert call.action == "read_records"
    assert call.arguments == {}
    usage = adapter.last_usage()
    assert usage.input_tokens == 137
    assert usage.output_tokens == 29
    assert adapter.last_telemetry().usage_reported is True
    assert guard.measured_usd > Decimal(0)
    assert guard.exposure_usd == Decimal(0)


def _refused(adapter: Any, guard: RunCostGuard) -> AdapterProviderError:
    """Run one turn, demand a named response-invalid refusal, and check the money."""
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    error = caught.value
    assert error.fault == PROVIDER_FAULT_RESPONSE_INVALID
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.response_received is True
    assert telemetry.usage_reported is False
    assert adapter.last_usage().input_tokens is None
    assert adapter.last_usage().cost_usd is None
    assert guard.measured_usd == Decimal(0)
    assert guard.exposure_usd > Decimal(0)
    return error


def _refuses(body: Any) -> AdapterProviderError:
    adapter, guard = _adapter(body)
    return _refused(adapter, guard)


# -- 1. the observed block is the contract, and it is accepted -----------------


def test_the_observed_server_extension_block_is_accepted_whole() -> None:
    """The blocker: a real HTTP 200 from the pinned model is refused today.

    Every one of the five names is a top-level field the SDK's ``Response`` does
    not declare, so the response-contract check counted five undeclared fields
    and refused the body — with the turn's reservation held as exposure and no
    action read — although nothing in the block is read for an action, for the
    model's identity or for the counts the turn is priced on.
    """
    _accepted(_body(OBSERVED_EXTENSIONS))


def test_a_body_with_no_extension_block_at_all_is_still_an_ordinary_turn() -> None:
    """Backwards compatibility: the contract adds a permission, not a demand."""
    _accepted(_body())


@pytest.mark.parametrize("absent", sorted(OBSERVED_EXTENSIONS))
def test_any_one_top_level_extension_may_be_absent(absent: str) -> None:
    """A server that stops sending one field has not broken the contract.

    Required-ness is the direction that fails closed for nothing: this build
    reads none of these fields, so demanding them would refuse a body it can
    read perfectly well the day the service stops emitting one.
    """
    block = copy.deepcopy(OBSERVED_EXTENSIONS)
    del block[absent]

    _accepted(_body(block))


@pytest.mark.parametrize("absent", list(combinations(sorted(OBSERVED_EXTENSIONS), 2)))
def test_the_five_top_level_extensions_are_independently_optional(
    absent: tuple[str, str],
) -> None:
    """Optional *independently*: one going missing does not make another required."""
    block = copy.deepcopy(OBSERVED_EXTENSIONS)
    for name in absent:
        del block[name]

    _accepted(_body(block))


#: Every object this contract fixes, addressed by where it sits in the block.
#: A present one has to state the whole of itself; the list is exhaustive so a
#: new object cannot be added to the contract without a decision about it here.
OBJECT_PATHS: tuple[tuple[str, ...], ...] = (
    ("billing",),
    ("tool_usage",),
    ("tool_usage", "image_gen"),
    ("tool_usage", "image_gen", "input_tokens_details"),
    ("tool_usage", "image_gen", "output_tokens_details"),
    ("tool_usage", "web_search"),
)

#: Every child the contract fixes under one of those objects. Exhaustive, and
#: each one is required *where it sits*: the five top-level names above may be
#: absent, and nothing below them may.
REQUIRED_CHILD_PATHS: tuple[tuple[str, ...], ...] = (
    ("billing", "payer"),
    ("tool_usage", "image_gen"),
    ("tool_usage", "web_search"),
    ("tool_usage", "image_gen", "input_tokens"),
    ("tool_usage", "image_gen", "input_tokens_details"),
    ("tool_usage", "image_gen", "input_tokens_details", "image_tokens"),
    ("tool_usage", "image_gen", "input_tokens_details", "text_tokens"),
    ("tool_usage", "image_gen", "output_tokens"),
    ("tool_usage", "image_gen", "output_tokens_details"),
    ("tool_usage", "image_gen", "output_tokens_details", "image_tokens"),
    ("tool_usage", "image_gen", "output_tokens_details", "text_tokens"),
    ("tool_usage", "image_gen", "total_tokens"),
    ("tool_usage", "web_search", "num_requests"),
)


@pytest.mark.parametrize("path", REQUIRED_CHILD_PATHS)
def test_a_present_object_missing_a_child_the_contract_fixes_is_refused(
    path: tuple[str, ...],
) -> None:
    """Optional at the top, required underneath, and that asymmetry is the point.

    A top-level name is optional because a service that stops sending a whole
    block has stopped making a statement this build never read. A *child* is
    different: its parent is still making the statement, and a parent that
    states half of itself is not the shape this contract transcribed. Accepting
    it would make an exact contract mean "any subset of the observed shape",
    which is a claim nothing here ever observed and nothing here can justify.
    """
    _refuses(_body(_without(path)))


@pytest.mark.parametrize("path", OBJECT_PATHS)
def test_a_present_but_vacuous_extension_object_is_refused(
    path: tuple[str, ...],
) -> None:
    """``{}`` is a present object that states nothing, and it was never observed.

    This is the assertion this change exists to invert. Reading ``{}`` as "every
    child is absent, and every child is optional, so this is fine" was the
    permissive step: it accepted a body no diagnostic ever saw, under a contract
    that calls itself exact.
    """
    _refuses(_body(_at(path, {})))


# -- 2. the contract is closed, at the top level and at every depth ------------


def test_an_unknown_top_level_field_beside_the_block_is_still_refused() -> None:
    """The amendment names five fields. It does not open the top level."""
    _refuses(_body(OBSERVED_EXTENSIONS, unexpected_extension=1))


@pytest.mark.parametrize(
    "path",
    [
        ("billing", "unexpected"),
        ("tool_usage", "unexpected"),
        ("tool_usage", "image_gen", "unexpected"),
        ("tool_usage", "image_gen", "input_tokens_details", "unexpected"),
        ("tool_usage", "image_gen", "output_tokens_details", "unexpected"),
        ("tool_usage", "web_search", "unexpected"),
    ],
)
def test_an_unknown_key_inside_the_block_is_refused_at_every_depth(
    path: tuple[str, ...],
) -> None:
    """Recursively closed: a field nobody can describe is refused wherever it is."""
    _refuses(_body(_at(path, 1)))


@pytest.mark.parametrize(
    "path",
    [
        ("billing",),
        ("tool_usage",),
        ("tool_usage", "image_gen"),
        ("tool_usage", "image_gen", "input_tokens_details"),
        ("tool_usage", "web_search"),
    ],
)
@pytest.mark.parametrize("value", [[], "object", 7, True, None, 1.5])
def test_an_extension_object_that_is_not_a_json_object_is_refused(
    path: tuple[str, ...], value: Any
) -> None:
    """A declared object arriving as a scalar, a list or a null is not the shape."""
    _refuses(_body(_at(path, value)))


# -- 3. every scalar is checked for its exact wire type ------------------------

#: The count paths this contract fixes: every one of them exact, non-negative
#: and never a boolean, on the same argument the token counts are checked on.
COUNT_PATHS: tuple[tuple[str, ...], ...] = (
    ("tool_usage", "image_gen", "input_tokens"),
    ("tool_usage", "image_gen", "input_tokens_details", "image_tokens"),
    ("tool_usage", "image_gen", "input_tokens_details", "text_tokens"),
    ("tool_usage", "image_gen", "output_tokens"),
    ("tool_usage", "image_gen", "output_tokens_details", "image_tokens"),
    ("tool_usage", "image_gen", "output_tokens_details", "text_tokens"),
    ("tool_usage", "image_gen", "total_tokens"),
    ("tool_usage", "web_search", "num_requests"),
)

#: What a count may never be. ``True`` is an ``int`` to Python and ``1`` to a
#: lenient reader; ``"3"`` is coerced; a float is not a count of anything
#: discrete; a negative one is not a count at all; a null is an absent field
#: stated as a present one.
UNCOUNTABLE: tuple[Any, ...] = (True, False, "3", 3.0, 0.5, -1, None, [], {})


@pytest.mark.parametrize("path", COUNT_PATHS)
def test_a_nested_count_is_an_exact_non_negative_integer(path: tuple[str, ...]) -> None:
    _accepted(_body(_at(path, 0)))
    _accepted(_body(_at(path, 4096)))


@pytest.mark.parametrize("value", UNCOUNTABLE)
def test_a_nested_count_of_any_other_shape_is_refused(value: Any) -> None:
    for path in COUNT_PATHS:
        _refuses(_body(_at(path, value)))


#: The two penalty fields, and the range the vendor publishes for them.
PENALTY_PATHS: tuple[tuple[str, ...], ...] = (
    ("frequency_penalty",),
    ("presence_penalty",),
)


@pytest.mark.parametrize("path", PENALTY_PATHS)
@pytest.mark.parametrize("value", [-2, -2.0, -1.5, 0, 0.0, 0.25, 1.999, 2, 2.0])
def test_a_penalty_inside_the_published_range_is_accepted(
    path: tuple[str, ...], value: Any
) -> None:
    """Both bounds are inclusive, and an integral JSON number is a JSON number."""
    _accepted(_body(_at(path, value)))


@pytest.mark.parametrize("path", PENALTY_PATHS)
@pytest.mark.parametrize(
    "value", [-2.0001, 2.0001, -3, 3, 1e9, True, False, "0.5", None, [], {}]
)
def test_a_penalty_outside_the_range_or_of_another_type_is_refused(
    path: tuple[str, ...], value: Any
) -> None:
    _refuses(_body(_at(path, value)))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_penalty_is_refused_as_the_number_it_is_not(literal: str) -> None:
    """``NaN`` compares false against every bound, so it must be refused by name.

    Sent as raw bytes, because Python's own encoder emits these literals happily
    and both readers — this build's strict decode and the SDK's — accept them.
    A range check alone would let ``NaN`` through every comparison it is asked.
    """
    text = json.dumps(_body(OBSERVED_EXTENSIONS))
    body = text.replace('"frequency_penalty": 0.0', f'"frequency_penalty": {literal}')
    assert body != text

    _refuses(body.encode("utf-8"))


@pytest.mark.parametrize("value", [True, False, 0, 1, "false", None, [], {}])
def test_the_flag_must_be_a_json_boolean(value: Any) -> None:
    if isinstance(value, bool):
        _accepted(_body(_observed(store=value)))
    else:
        _refuses(_body(_observed(store=value)))


@pytest.mark.parametrize("value", ["o", "org-observed", "x" * 128])
def test_the_payer_is_a_non_empty_bounded_string(value: str) -> None:
    _accepted(_body(_at(("billing", "payer"), value)))


@pytest.mark.parametrize(
    "value", ["", "   ", "x" * 129, "x" * 4096, 7, True, None, [], {}]
)
def test_a_payer_that_is_empty_oversized_or_not_a_string_is_refused(value: Any) -> None:
    """Bounded because it is provider-controlled text of otherwise unbounded size,
    and non-empty because a stated identity of nothing states nothing."""
    _refuses(_body(_at(("billing", "payer"), value)))


def test_the_payer_never_reaches_a_durable_row_accepted_or_refused() -> None:
    """It is validated and dropped: no usage, telemetry or detail carries it."""
    secret = "payer-value-that-must-not-be-recorded"
    adapter, _ = _adapter(_body(_at(("billing", "payer"), secret)))

    adapter.next_call(_turn_request(), _deadline())

    assert secret not in canonical_json_text(
        adapter.last_telemetry().as_dict(), "telemetry"
    )
    assert secret not in canonical_json_text(adapter.last_usage().as_dict(), "usage")


# -- 4. a refusal records this build's own detail and nothing else -------------


def test_no_extension_name_or_value_reaches_a_durable_failure_detail() -> None:
    """A durable row carries fixed sentences and closed-set values, as ever.

    The field names are this build's own, but the row does not need them and the
    values beside them are provider-controlled — so a refusal states what kind of
    shape was wrong, not which field carried it or what it held.
    """
    # Long enough to be an invalid payer as well as an invalid count, flag and
    # penalty, so one marker exercises every kind of refusal the block raises.
    marker = "provider-authored-marker-" * 8
    for body in (
        _body(_at(("billing", "payer"), marker)),
        _body(_at(("tool_usage", "web_search", "num_requests"), marker)),
        _body(_at(("tool_usage", "image_gen", "unexpected"), marker)),
        _body(_observed(store=marker)),
        _body(_at(("frequency_penalty",), marker)),
        # The refusal this change adds. Which child was missing is not recorded
        # either: the names are this build's own, but a row that named one would
        # be reporting which fields a provider left out, turn after turn.
        _body(_without(("billing", "payer"))),
        _body(_at(("tool_usage", "web_search"), {})),
    ):
        detail = str(_refuses(body))
        assert marker not in detail
        for name in EXTENSION_FIELD_NAMES:
            assert name not in detail


def test_an_incomplete_object_is_refused_with_one_fixed_closed_detail() -> None:
    """One sentence for every missing child, at every depth, of every parent.

    Not "the same kind of message with the name filled in", and not a count of
    how many children were missing either: which fields a body left out is a
    provider-controlled fact, and a durable row that varied with it would be a
    per-turn record of the provider's own shape drift.
    """
    details = {str(_refuses(_body(_without(path)))) for path in REQUIRED_CHILD_PATHS}
    details |= {str(_refuses(_body(_at(path, {})))) for path in OBJECT_PATHS}

    assert len(details) == 1
    detail = details.pop()
    for name in EXTENSION_FIELD_NAMES:
        assert name not in detail


# -- 5. the wire and the SDK's own reading must agree --------------------------


def _parsed(body: dict[str, Any]) -> Response:
    """The real SDK's typed reading of one body, produced by the real client."""
    _, client = scripted_client(body)
    return client.responses.with_raw_response.create(
        model=GPT_5_6_LUNA_MODEL, input=[]
    ).parse()


def test_the_typed_extras_and_the_raw_body_are_both_validated_and_must_agree() -> None:
    """Two readings of one body, and a disagreement is refused rather than resolved.

    The SDK parses unknown top-level fields into ``model_extra`` rather than
    dropping them, so there are two places this block can be stated and they can
    differ. Neither reading is preferred: a body whose typed extras are not the
    extras on the wire is not one contract, and this build refuses it.
    """
    from boundarybench.providers.openai_responses import check_response_admissible

    agreeing = _body(OBSERVED_EXTENSIONS)
    disagreeing = _body(_observed(store=True))

    check_response_admissible(
        WireResponse(
            json.dumps(agreeing), lambda: _parsed(agreeing), kind="openai response"
        ),
        model=GPT_5_6_LUNA_MODEL,
    )
    with pytest.raises(AdapterProviderError) as caught:
        check_response_admissible(
            WireResponse(
                json.dumps(agreeing),
                lambda: _parsed(disagreeing),
                kind="openai response",
            ),
            model=GPT_5_6_LUNA_MODEL,
        )
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_the_typed_extras_are_validated_rather_than_cleared_to_hide_them() -> None:
    """Emptying ``model_extra`` would make the SDK's reading agree by force.

    That is the repair this build must not make: the typed parse stays active,
    the extras stay on the object exactly as the SDK read them, and what changed
    is only that this contract can describe them.
    """
    from boundarybench.providers.openai_responses import check_response_admissible

    body = _body(OBSERVED_EXTENSIONS)
    response: WireResponse[Response] = WireResponse(
        json.dumps(body), lambda: _parsed(body), kind="openai response"
    )

    check_response_admissible(response, model=GPT_5_6_LUNA_MODEL)

    extra = response.parsed.model_extra
    assert extra is not None
    assert extra == OBSERVED_EXTENSIONS
    assert response.parsed.status == "completed"
    assert response.parsed.usage is not None


def test_a_typed_extra_the_wire_never_stated_is_refused() -> None:
    """The check runs on both readings, so neither alone can smuggle a field in."""
    from boundarybench.providers.openai_responses import check_response_admissible

    wire = _body()
    typed = _body(OBSERVED_EXTENSIONS)

    response: WireResponse[Response] = WireResponse(
        json.dumps(wire), lambda: _parsed(typed), kind="openai response"
    )

    with pytest.raises(AdapterProviderError) as caught:
        check_response_admissible(response, model=GPT_5_6_LUNA_MODEL)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


# -- 6. the published contract, its digest, and what settings claim ------------


def test_the_published_schema_is_the_observed_shape_and_nothing_wider() -> None:
    """The accepted contract, pinned against a hand-written transcription."""
    from boundarybench.providers.openai_responses import (
        SERVER_EXTENSION_CONTRACT,
        SERVER_EXTENSION_NAMES,
        SERVER_EXTENSION_SCHEMA,
    )

    assert _plain(SERVER_EXTENSION_SCHEMA) == EXPECTED_SCHEMA
    assert set(SERVER_EXTENSION_NAMES) == set(EXPECTED_SCHEMA)
    assert _plain(SERVER_EXTENSION_CONTRACT) == EXPECTED_CONTRACT


def test_the_contract_is_named_and_hashed_in_the_settings_run_identity_covers() -> None:
    """A named contract nobody can re-derive is a label; the digest is the claim."""
    from boundarybench.providers.openai_responses import (
        OPENAI_RESPONSE_SERVER_EXTENSIONS,
        SERVER_EXTENSION_DIGEST,
    )

    digest = hashlib.sha256(
        canonical_json_bytes(EXPECTED_CONTRACT, "openai response server extensions")
    ).hexdigest()
    settings = openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL)

    assert OPENAI_RESPONSE_SERVER_EXTENSIONS == EXTENSIONS_CONTRACT
    assert digest == SERVER_EXTENSION_DIGEST
    assert settings["response_server_extensions"] == EXTENSIONS_CONTRACT
    assert settings["response_server_extensions_digest"] == digest


def test_the_digest_moved_although_the_transcribed_names_and_types_did_not() -> None:
    """The digest has to cover the reading, not only the shape.

    ``v1`` hashed the name-and-type transcription alone, and this change does not
    touch a single name or type in it: what moved is which bodies that shape
    admits. A digest that still covered only the transcription would be equal
    across the two builds, and two manifests would claim one configuration while
    the runs behind them read the same body differently.
    """
    from boundarybench.providers.openai_responses import SERVER_EXTENSION_DIGEST

    superseded = hashlib.sha256(
        canonical_json_bytes(EXPECTED_SCHEMA, "openai response server extensions")
    ).hexdigest()

    assert EXPECTED_CONTRACT["members"] == EXPECTED_SCHEMA
    assert superseded != SERVER_EXTENSION_DIGEST
    assert EXTENSIONS_CONTRACT != SUPERSEDED_EXTENSIONS_CONTRACT


def test_the_contract_is_provider_observed_and_says_so_by_not_claiming_the_sdk() -> None:
    """Honesty about provenance: the pinned SDK declares none of these fields.

    If it declared them there would be nothing to amend — ``declared_wire_fields``
    would already allow them and the response-contract check would already count
    zero. The contract exists precisely because the vendor's own model does not
    state them, so it is a transcription of what the service returned.
    """
    from boundarybench.providers.openai_responses import SERVER_EXTENSION_NAMES
    from boundarybench.providers.wire import declared_wire_fields

    declared = declared_wire_fields(Response)

    assert set(SERVER_EXTENSION_NAMES).isdisjoint(declared)
    assert len(SERVER_EXTENSION_NAMES) == 5


# -- 7. what moved, and what did not -------------------------------------------


def test_the_openai_adapter_version_moved_because_what_it_accepts_moved() -> None:
    """``0.5.0``: narrowing what a run will read as an answer is a version move.

    The same rule that took this lane to ``0.4.0`` for widening it. A body that
    ``0.4.0`` read as an answer — an object stating half of itself — is refused
    here, so a run made under either number is not comparable with one made under
    the other, and the identity a row carries has to say which one it was.
    """
    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert openai_identity(GPT_5_6_LUNA_MODEL).version == "0.5.0"


def _manifest(settings: Any, *, version: str) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider="openai",
        model=GPT_5_6_LUNA_MODEL,
        implementation="openai_responses",
        adapter_version=version,
        adapter_settings=settings,
        trials=1,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
    )


def test_the_configuration_identity_is_not_the_one_that_refused_the_block() -> None:
    """A run that would have refused this body is not a run that accepts it."""
    current = openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL)
    superseded = {
        key: value
        for key, value in current.items()
        if not key.startswith("response_server_extensions")
    }

    assert _manifest(current, version=OPENAI_ADAPTER_VERSION).configuration_id not in {
        _manifest(superseded, version="0.3.0").configuration_id,
        _manifest(superseded, version=OPENAI_ADAPTER_VERSION).configuration_id,
        _manifest(current, version="0.3.0").configuration_id,
    }


def test_the_configuration_identity_is_not_the_one_that_read_a_half_stated_block() -> (
    None
):
    """The identity has to move on both halves of what ``v1`` recorded.

    A build that accepted a vacuous object recorded a contract *name* and a
    contract *digest*, and this change moves both — so neither the name alone nor
    the digest alone can carry a run across the two readings, and neither can the
    adapter version. Each of the three is checked while the other two are held at
    the superseded value, because an identity that moved for only one of them
    would still let a resume cross the change on the other two.
    """
    current = openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL)
    permissive = dict(current)
    permissive["response_server_extensions"] = SUPERSEDED_EXTENSIONS_CONTRACT
    permissive["response_server_extensions_digest"] = hashlib.sha256(
        canonical_json_bytes(EXPECTED_SCHEMA, "openai response server extensions")
    ).hexdigest()
    only_name_moved = dict(permissive) | {
        "response_server_extensions": EXTENSIONS_CONTRACT
    }
    only_digest_moved = dict(permissive) | {
        "response_server_extensions_digest": current["response_server_extensions_digest"]
    }

    assert _manifest(current, version=OPENAI_ADAPTER_VERSION).configuration_id not in {
        _manifest(permissive, version="0.4.0").configuration_id,
        _manifest(permissive, version=OPENAI_ADAPTER_VERSION).configuration_id,
        _manifest(only_name_moved, version=OPENAI_ADAPTER_VERSION).configuration_id,
        _manifest(only_digest_moved, version=OPENAI_ADAPTER_VERSION).configuration_id,
        _manifest(current, version="0.4.0").configuration_id,
    }


def test_the_request_this_turn_sends_is_byte_identical_to_the_superseded_one() -> None:
    """What this build *accepts* moved. What it *asks* must not have.

    Asserted against the body ``httpx.MockTransport`` actually received after the
    real SDK serialised it, and against this build's own request mapping, so a
    silent change to the question asked of the model would fail here.
    """
    transport = RecordingTransport([_body(OBSERVED_EXTENSIONS)])
    adapter = OpenAIResponsesAdapter(model=GPT_5_6_LUNA_MODEL, client=transport.client())

    adapter.next_call(_turn_request(), _deadline())

    body = transport.bodies[0]
    assert canonical_json_text(body, "sent") == canonical_json_text(
        build_responses_request(_turn_request(), model=GPT_5_6_LUNA_MODEL), "built"
    )
    assert body["reasoning"] == {"effort": "none"}
    assert "temperature" not in body
    assert "top_p" not in body
    assert sorted(body) == [
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "store",
        "tool_choice",
        "tools",
    ]


def test_no_other_lane_accepts_these_fields_or_records_this_contract() -> None:
    """The amendment is one lane's. Everywhere else, *these* extras are zero.

    Two other lanes have since named response-extension contracts of their own:
    the xAI one at ``0.3.0``, for fields its compatibility endpoint states on the
    message and the usage block, and the Mistral one at ``0.3.0``, for the
    ``prompt_tokens_details`` object that service states on its usage block. Both
    are different contracts over different names at different sites, and both are
    recorded under a different settings key — which is exactly what the assertion
    below checks: no lane but this one records a ``response_server_extensions``
    claim, and none of them accepts *these* five names on a response root.
    """
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_ADAPTER_VERSION,
        SONNET_5_MODEL,
        AnthropicRetryPolicy,
        anthropic_settings,
    )
    from boundarybench.providers.mistral_chat import (
        MISTRAL_ADAPTER_VERSION,
        MISTRAL_SMALL_MODEL,
        mistral_settings,
    )
    from boundarybench.providers.xai_openai_compat import (
        GROK_4_5_MODEL,
        XAI_ADAPTER_VERSION,
        xai_settings,
    )

    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert MISTRAL_ADAPTER_VERSION == "0.3.0"
    assert ANTHROPIC_ADAPTER_VERSION == "0.8.0"
    for settings in (
        xai_settings(model=GROK_4_5_MODEL),
        mistral_settings(model=MISTRAL_SMALL_MODEL),
        anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL),
    ):
        assert not [key for key in settings if key.startswith("response_server")]


def test_the_xai_lane_still_refuses_the_same_block_on_its_own_surface() -> None:
    """Sharing an SDK is not sharing a response contract."""
    from boundarybench.providers.xai_openai_compat import XAIOpenAICompatAdapter

    body = chat_body([tool_call("read_records", "{}")], model="grok-test-20990101")
    body.update(copy.deepcopy(OBSERVED_EXTENSIONS))
    transport = RecordingTransport([body])
    guard = _guard()
    adapter = XAIOpenAICompatAdapter(
        model="grok-test-20990101", client=transport.xai_client(), cost_guard=guard
    )

    _refused(adapter, guard)


def test_the_common_response_contract_check_still_defaults_to_zero_extras() -> None:
    """The shared checker was widened by an argument, not by a new default."""
    from boundarybench.providers.common import check_response_contract

    parsed = _parsed(_body(OBSERVED_EXTENSIONS))

    with pytest.raises(AdapterProviderError) as caught:
        check_response_contract(parsed)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    check_response_contract(parsed, allowed_root_extras=frozenset(EXPECTED_SCHEMA))
