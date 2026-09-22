"""The xAI compatibility layer's response extensions, checked as an exact shape.

xAI's OpenAI-compatible Chat Completions endpoint returns, on an otherwise
ordinary HTTP 200 from ``grok-4.5``, a small set of fields the pinned OpenAI SDK
2.53 does not declare: one on the assistant message, two on the usage block, and
one pair inside the SDK-declared ``prompt_tokens_details`` object. Under this
build's response contract every one of them is an undeclared field, so the whole
body was refused — the action went unparsed, the usage went unmeasured, and the
turn's reservation was held as exposure over fields nothing here reads.

So this lane names **one exact response-extension contract** —
``xai_compat_response_extensions_v1`` — and accepts precisely it:

* ``choices[].message.reasoning_content``: optional, and an exact JSON string
  when present — valid Unicode scalars only, and bounded by a conservative limit
  tied to this run's pinned output ceiling. It is never read for an action, never
  persisted, and never quoted in a failure detail;
* ``usage.cost_in_usd_ticks`` and ``usage.num_sources_used``: each optional, and
  an exact non-negative bounded integer when present — never a boolean, a
  string, a fractional number or a null. Neither is used for the cost of a turn:
  the run's pinned price policy remains the only source of a measured cost;
* ``usage.prompt_tokens_details.image_tokens`` and ``.text_tokens``: a pair
  inside an object the SDK *does* declare. Both may be absent, and if either
  arrives both are required, because a half-stated pair was never observed.

Everything outside that contract is refused exactly as it was before: an
undeclared field at the root, on a message, on the usage block, inside a detail
object, or an extension name at a site the contract does not fix for it.

Each of those fields is stated twice — in the exact JSON the provider sent and
in the typed reading this SDK parks undeclared fields on — so the two readings
are compared over the **union** of what each of them states: the objects the
sites live on have to be stated by both readings or by neither before any value
is compared, and an extension one reading states and the other does not is a
disagreement wherever it appears. Neither reading is preferred, neither is
edited to agree with the other, and ``model_extra`` is never cleared.

The contract is **provider-observed**, not vendor-declared. Two bounded
schema-only diagnostics established these names and types against the exact
``grok-4.5`` compatibility endpoint; the raw bodies were not persisted, and the
official model page publishes no schema for them. A provider-observed contract
can move without notice, so it is named *and* hashed in adapter settings, and
both are inside ``configuration_id``.

The model page publishes ``Reasoning: Yes`` and no control that disables it, so
this build sends no reasoning field and invents none. What it does instead is
describe what comes back.

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
from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

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
from boundarybench.pricing import MAX_EXACT_TOKEN_COUNT, price_for
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.providers.wire import WireResponse, declared_wire_fields
from boundarybench.providers.xai_openai_compat import (
    GROK_4_5_MODEL,
    MAX_OUTPUT_TOKENS,
    XAI_ADAPTER_VERSION,
    XAIOpenAICompatAdapter,
    build_chat_request,
    check_response_admissible,
    check_response_extensions_agree,
    request_profile_for,
    xai_identity,
    xai_settings,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST, minimal_card
from tests.openai_transport import (
    RecordingTransport,
    chat_body,
    tool_call,
    xai_scripted_client,
)

#: The contract's own name, written out rather than imported: it is inside
#: ``configuration_id``, so an identifier asserted against itself would follow
#: any rename and prove nothing about the run identity a reader compares.
EXTENSIONS_CONTRACT = "xai_compat_response_extensions_v1"

#: The exact name and type shape the two bounded schema-only diagnostics
#: established, hand-written here rather than imported from the module under
#: test. The published schema is asserted against *this* literal and the digest
#: in settings against a SHA-256 over the canonical bytes of the whole contract,
#: so a silent widening of what a run will read as an answer fails here.
EXPECTED_SCHEMA: dict[str, Any] = {
    "completion_message": {"reasoning_content": "bounded_reasoning_text"},
    "prompt_token_details": {
        "image_tokens": "non_negative_bounded_integer",
        "text_tokens": "non_negative_bounded_integer",
    },
    "usage_block": {
        "cost_in_usd_ticks": "non_negative_bounded_integer",
        "num_sources_used": "non_negative_bounded_integer",
    },
}

#: The rules the shape is read under, hashed beside it. The shape alone is not
#: the contract: a build that transcribed these same names and types and then
#: accepted a half-stated detail pair would read a body this one refuses, and a
#: digest over the transcription alone would be equal across the two.
EXPECTED_CONTRACT: dict[str, Any] = {
    "contract": EXTENSIONS_CONTRACT,
    "extension_values_read": False,
    "max_count": MAX_EXACT_TOKEN_COUNT,
    "max_reasoning_characters": MAX_OUTPUT_TOKENS * 16,
    "members": EXPECTED_SCHEMA,
    # ``prompt_token_details`` is all-or-nothing; the other two sites carry
    # independently optional members.
    "members_required_together": {
        "completion_message": False,
        "prompt_token_details": True,
        "usage_block": False,
    },
}

#: Every field name this contract fixes, at any site. No refusal may name one.
EXTENSION_FIELD_NAMES: tuple[str, ...] = (
    "cost_in_usd_ticks",
    "image_tokens",
    "num_sources_used",
    "reasoning_content",
    "text_tokens",
)

#: The observed reasoning string's *length*, which is the only thing about it
#: this repository records. The text a model produced is not transcribed here or
#: anywhere else: it is model-authored content, this build reads none of it, and
#: the contract's question about it is how long it may be.
OBSERVED_REASONING_LENGTH = 95

#: A stand-in of exactly that length, built rather than quoted.
OBSERVED_REASONING = "x" * OBSERVED_REASONING_LENGTH

#: The longest reasoning string this contract accepts, and one character past it.
MAX_REASONING_CHARACTERS = MAX_OUTPUT_TOKENS * 16

#: The counts the diagnostics observed beside the standard usage block. Their
#: values are per-turn facts; what the contract fixes is their type.
OBSERVED_COST_TICKS = 3129
OBSERVED_SOURCES = 4
OBSERVED_IMAGE_TOKENS = 0
OBSERVED_TEXT_TOKENS = 137


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
    """A real cost guard at this model's pinned price, so money can be asserted."""
    return RunCostGuard(
        controls=CostControls(
            max_cost_usd=Decimal("10.00"),
            max_episodes=36,
            price=price_for(provider="xai", model=GROK_4_5_MODEL),
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=ProviderRetryPolicy().max_attempts,
    )


def _bare_body() -> dict[str, Any]:
    """One ordinary completion for the pinned model, carrying no extension."""
    return chat_body([tool_call("read_records", "{}")], model=GROK_4_5_MODEL)


def _body(
    *,
    message: Mapping[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One completion body with the named extension members merged in at each site.

    ``details`` is merged into ``usage.prompt_tokens_details``, which is created
    only when something is put in it: an absent detail object is the ordinary
    case, not an extension.
    """
    body = _bare_body()
    if message is not None:
        body["choices"][0]["message"].update(copy.deepcopy(dict(message)))
    if usage is not None:
        body["usage"].update(copy.deepcopy(dict(usage)))
    if details is not None:
        body["usage"].setdefault("prompt_tokens_details", {})
        body["usage"]["prompt_tokens_details"].update(copy.deepcopy(dict(details)))
    return body


def _observed() -> dict[str, Any]:
    """The whole observed shape: every extension this contract fixes, present."""
    return _body(
        message={"reasoning_content": OBSERVED_REASONING},
        usage={
            "cost_in_usd_ticks": OBSERVED_COST_TICKS,
            "num_sources_used": OBSERVED_SOURCES,
        },
        details={
            "image_tokens": OBSERVED_IMAGE_TOKENS,
            "text_tokens": OBSERVED_TEXT_TOKENS,
        },
    )


def _adapter(*steps: Any) -> tuple[XAIOpenAICompatAdapter, RunCostGuard]:
    transport = RecordingTransport(list(steps))
    guard = _guard()
    return (
        XAIOpenAICompatAdapter(
            model=GROK_4_5_MODEL, client=transport.xai_client(), cost_guard=guard
        ),
        guard,
    )


def _accepted(body: Any) -> RunCostGuard:
    """One turn that must be read as the action and the counts the body states.

    The whole point of naming an extension contract is that a body carrying it
    is an ordinary turn: the action is parsed, the two counts are measured off
    the wire, and the reservation settles at the measured cost.
    """
    adapter, guard = _adapter(body)

    call = adapter.next_call(_turn_request(), _deadline())

    assert call.action == "read_records"
    assert call.arguments == {}
    usage = adapter.last_usage()
    assert usage.input_tokens == 137
    assert usage.output_tokens == 29
    assert adapter.last_telemetry().usage_reported is True
    assert guard.measured_usd > Decimal(0)
    assert guard.exposure_usd == Decimal(0)
    return guard


def _refused(
    adapter: XAIOpenAICompatAdapter, guard: RunCostGuard
) -> AdapterProviderError:
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


# -- 1. the observed shape is the contract, and it is accepted -----------------


def test_the_observed_extension_shape_is_accepted_whole() -> None:
    """The blocker: a real HTTP 200 from the pinned model is refused today.

    One undeclared field on the message, two on the usage block and two inside
    the declared detail object — five fields the pinned SDK does not model — so
    the response-contract check counted five and refused the body, with the
    turn's reservation held as exposure and no action read, although nothing in
    any of them is read for an action, for the model's identity or for the cost.
    """
    _accepted(_observed())


def test_a_body_with_no_extension_at_all_is_still_an_ordinary_turn() -> None:
    """Naming a contract for what may arrive does not require it to arrive."""
    _accepted(_bare_body())


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            _body(message={"reasoning_content": OBSERVED_REASONING}), id="reasoning"
        ),
        pytest.param(_body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}), id="ticks"),
        pytest.param(_body(usage={"num_sources_used": OBSERVED_SOURCES}), id="sources"),
        pytest.param(
            _body(
                details={
                    "image_tokens": OBSERVED_IMAGE_TOKENS,
                    "text_tokens": OBSERVED_TEXT_TOKENS,
                }
            ),
            id="detail-pair",
        ),
    ],
)
def test_each_extension_site_is_independently_optional(body: dict[str, Any]) -> None:
    """A service that states one of these and not the others has not broken a run.

    This build reads none of them, so requiring any would refuse a body it can
    read perfectly well. The one grouping the contract does fix is the detail
    pair, which is all-or-nothing rather than optional member by member.
    """
    _accepted(body)


def test_the_declared_detail_object_may_still_arrive_with_no_extension() -> None:
    """``prompt_tokens_details`` is the SDK's own object; the pair inside is not."""
    _accepted(_body(details={"cached_tokens": 0}))


def test_the_completion_detail_object_carries_no_extension_of_this_contract() -> None:
    """The other detail object is the SDK's own, whole, and stays that way.

    Worth its own case on a reasoning model: ``reasoning_tokens`` is a field the
    SDK *declares*, so it is read under the vendor's schema and not under this
    contract, and this contract fixes no name inside that object at all.
    """
    body = _bare_body()
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 12}

    _accepted(body)


# -- 2. the reasoning string: exact, bounded, encodable, and read for nothing --


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param(OBSERVED_REASONING, id="observed-length"),
        pytest.param("y" * MAX_REASONING_CHARACTERS, id="at-the-bound"),
    ],
)
def test_a_reasoning_string_within_the_bound_is_accepted(text: str) -> None:
    """Empty is a statement a model may make; the bound is where refusal starts.

    The limit is tied to this run's own pinned output ceiling rather than
    guessed: :data:`MAX_OUTPUT_TOKENS` times sixteen characters a token is far
    above what any tokenizer emits per token, and far above the length observed.
    """
    _accepted(_body(message={"reasoning_content": text}))


def test_a_reasoning_string_one_character_past_the_bound_is_refused() -> None:
    """A body held in memory for a turn is bounded, or it is not bounded."""
    _refuses(_body(message={"reasoning_content": "y" * (MAX_REASONING_CHARACTERS + 1)}))


def test_a_reasoning_string_carrying_a_lone_surrogate_is_refused() -> None:
    """UTF-8 cannot encode it, so it is not a string this build will accept.

    Sent as raw bytes, because no Python string this test could build would
    survive being encoded into the body: the escape has to reach the decoder as
    the provider would have written it.
    """
    body = _body(message={"reasoning_content": "placeholder"})
    raw = json.dumps(body).replace('"placeholder"', '"lone \\ud800 surrogate"')

    _refuses(raw.encode("utf-8"))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(95, id="integer"),
        pytest.param(True, id="boolean"),
        pytest.param(9.5, id="float"),
        pytest.param(None, id="null"),
        pytest.param(["reasoned"], id="array"),
        pytest.param({"text": "reasoned"}, id="object"),
    ],
)
def test_a_reasoning_field_that_is_not_a_json_string_is_refused(value: Any) -> None:
    """The contract fixes a string there. A null is an absence stated as a value."""
    _refuses(_body(message={"reasoning_content": value}))


def test_the_reasoning_text_never_reaches_a_durable_row_or_a_failure_detail() -> None:
    """Validated, then dropped: no usage, telemetry, action or detail carries it."""
    marker = "reasoning-text-that-must-not-be-recorded"
    adapter, _ = _adapter(_body(message={"reasoning_content": marker}))

    call = adapter.next_call(_turn_request(), _deadline())

    assert marker not in canonical_json_text(call.arguments, "arguments")
    assert marker not in canonical_json_text(
        adapter.last_telemetry().as_dict(), "telemetry"
    )
    assert marker not in canonical_json_text(adapter.last_usage().as_dict(), "usage")
    # ...and the same text in a body that is refused for another reason.
    detail = str(
        _refuses(
            _body(
                message={"reasoning_content": marker},
                details={"image_tokens": OBSERVED_IMAGE_TOKENS},
            )
        )
    )
    assert marker not in detail


# -- 3. the counts: exact, non-negative, bounded, and never the cost -----------

#: Every place the contract fixes a count, as the keyword ``_body`` takes it
#: under. Each one is checked against the same closed set of wrong shapes.
COUNT_SITES: tuple[tuple[str, str], ...] = (
    ("usage", "cost_in_usd_ticks"),
    ("usage", "num_sources_used"),
    ("details", "image_tokens"),
    ("details", "text_tokens"),
)


def _with_count(site: str, name: str, value: Any) -> dict[str, Any]:
    """One body stating ``value`` at one count site, with the pair kept whole."""
    if site == "details":
        other = "text_tokens" if name == "image_tokens" else "image_tokens"
        return _body(details={name: value, other: 0})
    return _body(usage={name: value})


@pytest.mark.parametrize(("site", "name"), COUNT_SITES)
@pytest.mark.parametrize(
    "value",
    [pytest.param(0, id="zero"), pytest.param(MAX_EXACT_TOKEN_COUNT, id="at-the-bound")],
)
def test_a_count_that_is_an_exact_bounded_integer_is_accepted(
    site: str, name: str, value: int
) -> None:
    """Zero is a count. The bound is the largest integer that stays exact."""
    _accepted(_with_count(site, name, value))


@pytest.mark.parametrize(("site", "name"), COUNT_SITES)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(MAX_EXACT_TOKEN_COUNT + 1, id="past-the-bound"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param("7", id="string"),
        pytest.param(7.0, id="float"),
        pytest.param(None, id="null"),
    ],
)
def test_a_count_of_any_other_shape_is_refused(site: str, name: str, value: Any) -> None:
    """Coercion is what makes this worth checking on the wire.

    ``true`` reaches a lenient reader as one and ``"7"`` as seven, and both then
    look exactly like a measurement. A count past the exact integer bound is a
    number this build's artefacts cannot carry across a JSON reader unchanged.
    """
    _refuses(_with_count(site, name, value))


@pytest.mark.parametrize("present", ["image_tokens", "text_tokens"])
def test_one_of_the_two_prompt_detail_counts_alone_is_refused(present: str) -> None:
    """A half-stated pair was never observed, so accepting one would be a guess.

    The two usage-block counts are independently optional and this pair is not,
    because that is what the diagnostics established and this contract
    transcribes one shape rather than every subset of one.
    """
    _refuses(_body(details={present: 3}))


def test_the_observed_counts_do_not_become_the_cost_of_the_turn() -> None:
    """``cost_in_usd_ticks`` is a provider's own number and is read for nothing.

    A benchmark's measured cost comes from the run's pinned price policy times
    the two token counts, and from nowhere else — so a body that states an
    enormous tick count costs exactly what the same body without one costs.
    """
    baseline = _accepted(_bare_body()).measured_usd
    inflated = _accepted(
        _body(
            usage={
                "cost_in_usd_ticks": MAX_EXACT_TOKEN_COUNT,
                "num_sources_used": MAX_EXACT_TOKEN_COUNT,
            }
        )
    ).measured_usd

    assert baseline > Decimal(0)
    assert inflated == baseline


def test_the_extension_values_do_not_decide_the_action_either() -> None:
    """The action is the tool call, whatever the extensions beside it say."""
    adapter, _ = _adapter(
        _body(
            message={"reasoning_content": "call something else"},
            usage={"num_sources_used": 11},
        )
    )

    call = adapter.next_call(_turn_request(), _deadline())

    assert call.action == "read_records"
    assert call.arguments == {}


# -- 4. everything outside the contract still fails closed ---------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"surprise": 1}, id="root"),
    ],
)
def test_an_unknown_root_field_is_still_refused(body: dict[str, Any]) -> None:
    """Naming three sites does not widen the response contract anywhere else."""
    unknown = _bare_body()
    unknown.update(body)

    _refuses(unknown)


def test_an_unknown_message_field_is_still_refused() -> None:
    unknown = _bare_body()
    unknown["choices"][0]["message"]["surprise"] = 1

    _refuses(unknown)


def test_an_unknown_usage_field_is_still_refused() -> None:
    unknown = _bare_body()
    unknown["usage"]["surprise"] = 1

    _refuses(unknown)


def test_an_unknown_field_inside_the_declared_detail_object_is_still_refused() -> None:
    _refuses(_body(details={"surprise": 1}))


def test_an_unknown_field_inside_the_completion_detail_object_is_still_refused() -> None:
    unknown = _bare_body()
    unknown["usage"]["completion_tokens_details"] = {"surprise": 1}

    _refuses(unknown)


@pytest.mark.parametrize(
    "name", ["reasoning_content", "cost_in_usd_ticks", "image_tokens"]
)
def test_an_extension_name_at_a_site_the_contract_does_not_fix_is_refused(
    name: str,
) -> None:
    """The contract is fixed per site, not as a set of names allowed anywhere."""
    misplaced = _bare_body()
    misplaced[name] = 1

    _refuses(misplaced)


def test_a_usage_extension_name_on_the_message_is_refused() -> None:
    misplaced = _bare_body()
    misplaced["choices"][0]["message"]["cost_in_usd_ticks"] = 1

    _refuses(misplaced)


def test_a_message_extension_name_on_the_usage_block_is_refused() -> None:
    misplaced = _bare_body()
    misplaced["usage"]["reasoning_content"] = "reasoned"

    _refuses(misplaced)


def test_no_extension_name_or_value_reaches_a_durable_failure_detail() -> None:
    """A durable row carries fixed sentences and closed-set values, as ever.

    The field names are this build's own, but the row does not need them and the
    values beside them are provider-controlled — so a refusal states what kind of
    shape was wrong, not which field carried it or what it held.
    """
    marker = "provider-authored-marker-" * 8
    for body in (
        # Long enough to be past the reasoning bound as well as being no count at
        # all, so one marker exercises every kind of refusal this contract raises.
        _body(message={"reasoning_content": marker * 128}),
        _body(usage={"cost_in_usd_ticks": marker}),
        _body(usage={"num_sources_used": marker}),
        _body(details={"image_tokens": marker, "text_tokens": 1}),
        _body(details={"text_tokens": 1}),
    ):
        detail = str(_refuses(body))
        assert marker not in detail
        for name in EXTENSION_FIELD_NAMES:
            assert name not in detail


# -- 5. the wire and the SDK's own reading must agree --------------------------


def _parsed(body: dict[str, Any]) -> ChatCompletion:
    """The real SDK's typed reading of one body, produced by the real client."""
    _, client = xai_scripted_client(body)
    return client.chat.completions.with_raw_response.create(
        model=GROK_4_5_MODEL, messages=[]
    ).parse()


def _wire(wire: dict[str, Any], typed: dict[str, Any]) -> WireResponse[ChatCompletion]:
    return WireResponse(json.dumps(wire), lambda: _parsed(typed), kind="xai completion")


def test_the_typed_extras_and_the_raw_body_are_both_validated_and_must_agree() -> None:
    """Two readings of one body, and a disagreement is refused rather than resolved.

    This SDK parks a field its models do not declare in ``model_extra`` rather
    than dropping it, at each of the three nested sites, so there are two places
    this block can be stated and they can differ. Neither reading is preferred: a
    body whose typed extras are not the extras on the wire is not one contract.
    """
    agreeing = _observed()
    disagreeing = _body(
        message={"reasoning_content": OBSERVED_REASONING},
        usage={
            "cost_in_usd_ticks": OBSERVED_COST_TICKS + 1,
            "num_sources_used": OBSERVED_SOURCES,
        },
        details={
            "image_tokens": OBSERVED_IMAGE_TOKENS,
            "text_tokens": OBSERVED_TEXT_TOKENS,
        },
    )

    check_response_admissible(_wire(agreeing, agreeing), model=GROK_4_5_MODEL)

    with pytest.raises(AdapterProviderError) as caught:
        check_response_admissible(_wire(agreeing, disagreeing), model=GROK_4_5_MODEL)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


@pytest.mark.parametrize(
    "typed",
    [
        pytest.param(
            _body(message={"reasoning_content": OBSERVED_REASONING}), id="message"
        ),
        pytest.param(_body(usage={"num_sources_used": OBSERVED_SOURCES}), id="usage"),
        pytest.param(_body(details={"image_tokens": 1, "text_tokens": 2}), id="details"),
    ],
)
def test_a_typed_extra_the_wire_never_stated_is_refused(typed: dict[str, Any]) -> None:
    """The check runs on both readings, so neither alone can smuggle a field in."""
    with pytest.raises(AdapterProviderError) as caught:
        check_response_admissible(_wire(_bare_body(), typed), model=GROK_4_5_MODEL)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_a_typed_reading_that_states_no_usage_has_no_usage_extension_to_compare() -> None:
    """The agreement check is driven from the typed side, and reaches for nothing.

    A body with no usage block never reaches here through the adapter — the wire
    contract requires one — but this is a public checker over a field the SDK
    types as optional, and the alternative to the guard is an attribute error
    dressed as an adapter bug. Absent on *both* readings is the case this admits:
    the message half is still compared, and a usage block only one of the two
    readings states is the disagreement below.
    """
    body = _body(message={"reasoning_content": OBSERVED_REASONING})
    del body["usage"]
    parsed = _parsed(body)

    assert parsed.usage is None
    check_response_extensions_agree(body, parsed)


def test_the_typed_extras_are_validated_rather_than_cleared_to_hide_them() -> None:
    """Emptying ``model_extra`` would make the SDK's reading agree by force.

    That is the repair this build must not make: the typed parse stays active,
    the extras stay on the objects exactly as the SDK read them at all three
    sites, and what changed is only that this contract can describe them.
    """
    body = _observed()
    response = _wire(body, body)

    check_response_admissible(response, model=GROK_4_5_MODEL)

    parsed = response.parsed
    message = parsed.choices[0].message
    usage = parsed.usage
    assert usage is not None
    details = usage.prompt_tokens_details
    assert details is not None
    assert message.model_extra == {"reasoning_content": OBSERVED_REASONING}
    assert usage.model_extra == {
        "cost_in_usd_ticks": OBSERVED_COST_TICKS,
        "num_sources_used": OBSERVED_SOURCES,
    }
    assert details.model_extra == {
        "image_tokens": OBSERVED_IMAGE_TOKENS,
        "text_tokens": OBSERVED_TEXT_TOKENS,
    }
    # ...and the typed parse is still the reading an action comes out of.
    assert message.tool_calls is not None
    assert usage.prompt_tokens == 137


# -- 5b. the comparison is over the union of what the two readings state -------
#
# The two readings are compared site by site, and which sites those are is the
# question this section pins. Deriving them from the typed reading alone makes
# every site the typed reading resolved to nothing invisible: a raw body stating
# a whole extension group under an object the typed reading never produced is
# never compared with anything, and is accepted although only one of the two
# readings ever stated it. So the sites come from the *union* of the two, the
# objects they live on must be stated by both readings or by neither, and only
# then are the values compared.


def _without_usage(body: dict[str, Any]) -> dict[str, Any]:
    """One body whose usage block is absent; the SDK types the field optional."""
    without = copy.deepcopy(body)
    del without["usage"]
    return without


def _without_message(body: dict[str, Any]) -> dict[str, Any]:
    """One body whose single choice states no message, which this API may do."""
    without = copy.deepcopy(body)
    without["choices"][0]["message"] = None
    return without


def _with_second_choice(body: dict[str, Any]) -> dict[str, Any]:
    """One body carrying a second choice, so a reading can state one site more."""
    extended = copy.deepcopy(body)
    extra = copy.deepcopy(extended["choices"][0])
    extra["index"] = 1
    extended["choices"].append(extra)
    return extended


def _reasoning_body() -> dict[str, Any]:
    return _body(message={"reasoning_content": OBSERVED_REASONING})


def _refuses_agreement(
    wire: dict[str, Any], typed: dict[str, Any]
) -> AdapterProviderError:
    """Two readings of one site set that do not agree, refused rather than resolved."""
    with pytest.raises(AdapterProviderError) as caught:
        check_response_extensions_agree(wire, _parsed(typed))
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    return caught.value


#: One reading states an object this contract fixes names on and the other does
#: not, at each of the sites — and in both directions, because a raw-only object
#: and a typed-only object are the same failure seen from two sides.
PRESENCE_DISAGREEMENTS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    (
        "raw-usage-only",
        _body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}),
        _without_usage(_bare_body()),
    ),
    (
        "typed-usage-only",
        _without_usage(_bare_body()),
        _body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}),
    ),
    (
        "raw-prompt-details-only",
        _body(
            details={
                "image_tokens": OBSERVED_IMAGE_TOKENS,
                "text_tokens": OBSERVED_TEXT_TOKENS,
            }
        ),
        _bare_body(),
    ),
    (
        "typed-prompt-details-only",
        _bare_body(),
        _body(
            details={
                "image_tokens": OBSERVED_IMAGE_TOKENS,
                "text_tokens": OBSERVED_TEXT_TOKENS,
            }
        ),
    ),
    (
        "raw-message-only",
        _reasoning_body(),
        _without_message(_bare_body()),
    ),
    (
        "typed-message-only",
        _without_message(_bare_body()),
        _reasoning_body(),
    ),
    (
        "raw-choice-only",
        _with_second_choice(_reasoning_body()),
        _reasoning_body(),
    ),
    (
        "typed-choice-only",
        _reasoning_body(),
        _with_second_choice(_reasoning_body()),
    ),
)


@pytest.mark.parametrize(
    ("wire", "typed"),
    [pytest.param(wire, typed, id=name) for name, wire, typed in PRESENCE_DISAGREEMENTS],
)
def test_an_object_only_one_reading_states_is_refused(
    wire: dict[str, Any], typed: dict[str, Any]
) -> None:
    """A site one reading has and the other has not is not one body to check.

    Presence is the question before the values are, and it is asked in both
    directions: the reading that states the object could have stated extensions
    under it that the other reading had nowhere to state, so comparing only the
    sites both readings happen to carry proves the contract about a body neither
    of them is.
    """
    _refuses_agreement(wire, typed)


@pytest.mark.parametrize(
    "wire",
    [
        pytest.param(
            _body(
                details={
                    "image_tokens": OBSERVED_IMAGE_TOKENS,
                    "text_tokens": OBSERVED_TEXT_TOKENS,
                }
            ),
            id="prompt-details-pair",
        ),
        pytest.param(
            _body(
                usage={
                    "cost_in_usd_ticks": OBSERVED_COST_TICKS,
                    "num_sources_used": OBSERVED_SOURCES,
                }
            ),
            id="usage-counts",
        ),
    ],
)
def test_a_raw_only_extension_site_is_refused_by_the_admissibility_check(
    wire: dict[str, Any],
) -> None:
    """The blocker, at the door a turn actually goes through.

    Both of these wire bodies pass the wire contract whole — a complete detail
    pair under a well-formed usage block, and two well-formed usage counts — so
    nothing before this refuses them. Against a typed reading that resolved the
    object they sit on to nothing, the comparison had no site to make and the
    body was admitted with an extension only one of the two readings ever stated.
    """
    typed = (
        _bare_body()
        if "prompt_tokens_details" in wire["usage"]
        else _without_usage(_bare_body())
    )

    with pytest.raises(AdapterProviderError) as caught:
        check_response_admissible(_wire(wire, typed), model=GROK_4_5_MODEL)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


@pytest.mark.parametrize(
    ("wire", "typed"),
    [
        pytest.param(
            _body(usage={"num_sources_used": OBSERVED_SOURCES}),
            _bare_body(),
            id="raw-only-count",
        ),
        pytest.param(
            _bare_body(),
            _body(usage={"num_sources_used": OBSERVED_SOURCES}),
            id="typed-only-count",
        ),
        pytest.param(
            _reasoning_body(),
            _bare_body(),
            id="raw-only-reasoning",
        ),
        pytest.param(
            _bare_body(),
            _reasoning_body(),
            id="typed-only-reasoning",
        ),
        pytest.param(
            _body(
                usage={
                    "cost_in_usd_ticks": OBSERVED_COST_TICKS,
                    "num_sources_used": OBSERVED_SOURCES,
                }
            ),
            _body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}),
            id="one-count-of-two",
        ),
    ],
)
def test_an_extension_only_one_reading_states_inside_a_shared_object_is_refused(
    wire: dict[str, Any], typed: dict[str, Any]
) -> None:
    """Presence disagreement inside an object *both* readings state, either way.

    The site is present on both readings here, so this is the other half of the
    question: which members it states. A member one reading has and the other has
    not is a disagreement exactly as a differing value is — the last case, where
    the two readings agree about one count and differ about the second, is the
    one a comparison that stopped at the first equal member would miss.
    """
    _refuses_agreement(wire, typed)


def test_two_readings_that_state_the_same_sites_and_values_are_accepted() -> None:
    """Absent on both is one body; present and equal on both is one body.

    The union is what is compared, not a demand that anything be there: a run
    reads none of these fields, so a body that states none of them at either
    site is the ordinary case and stays one.
    """
    check_response_extensions_agree(_bare_body(), _parsed(_bare_body()))
    check_response_extensions_agree(_observed(), _parsed(_observed()))
    check_response_admissible(_wire(_observed(), _observed()), model=GROK_4_5_MODEL)
    check_response_admissible(_wire(_bare_body(), _bare_body()), model=GROK_4_5_MODEL)


def test_a_choice_stating_no_message_on_either_reading_is_not_a_disagreement() -> None:
    """``null`` there is this API's own absence, and both readings read it as one."""
    absent = _without_message(_bare_body())

    check_response_extensions_agree(absent, _parsed(absent))


def test_a_presence_disagreement_does_not_clear_or_rewrite_the_typed_extras() -> None:
    """The refusal is a refusal, not a repair: the typed parse is left as it was.

    Emptying ``model_extra`` would make the two readings agree by force, and
    doing it on the way out of a refusal would leave a body the provider never
    sent behind for anything that looked at the object afterwards.
    """
    typed = _body(usage={"num_sources_used": OBSERVED_SOURCES})
    parsed = _parsed(typed)

    with pytest.raises(AdapterProviderError):
        check_response_extensions_agree(_bare_body(), parsed)

    assert parsed.usage is not None
    assert parsed.usage.model_extra == {"num_sources_used": OBSERVED_SOURCES}


class _TwoReadingAdapter(XAIOpenAICompatAdapter):
    """One adapter whose transport hands back a response with two readings.

    The readings a real turn holds come from one string of bytes, so this is the
    one thing an in-process HTTP transport cannot stage: it exists to run the
    refusal through the executor that owns a turn's money, rather than to claim
    a provider sends two bodies.
    """

    def __init__(self, response: WireResponse[ChatCompletion], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._response = response

    def _dispatch(self, payload: Any, timeout: float) -> WireResponse[ChatCompletion]:
        return self._response


def test_a_presence_disagreement_holds_the_turns_money_as_exposure() -> None:
    """Refused before anything is read, and paid for as an unmeasured turn.

    ``provider_response_invalid`` and nothing else: the attempt records that a
    response was received, the usage is left unmeasured rather than zero, and the
    reservation that authorised the request stays as exposure instead of settling
    at a cost nothing measured.
    """
    transport = RecordingTransport([])
    guard = _guard()
    adapter = _TwoReadingAdapter(
        _wire(
            _body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}),
            _without_usage(_bare_body()),
        ),
        model=GROK_4_5_MODEL,
        client=transport.xai_client(),
        cost_guard=guard,
    )

    _refused(adapter, guard)


def test_no_field_name_or_value_reaches_the_presence_disagreement_detail() -> None:
    """Which objects each reading stated is provider-controlled, so it is not said."""
    marker = "provider-authored-marker-" * 8
    details = [
        str(_refuses_agreement(wire, typed)) for _, wire, typed in PRESENCE_DISAGREEMENTS
    ]
    details.append(
        str(
            _refuses_agreement(
                _body(message={"reasoning_content": marker}),
                _without_message(_bare_body()),
            )
        )
    )

    for detail in details:
        assert marker not in detail
        assert OBSERVED_REASONING not in detail
        for name in EXTENSION_FIELD_NAMES:
            assert name not in detail
        for site in EXPECTED_SCHEMA:
            assert site not in detail


# -- 6. the published contract, its digest, and what settings claim ------------


def test_the_published_schema_is_the_observed_shape_and_nothing_wider() -> None:
    """The accepted contract, pinned against a hand-written transcription."""
    from boundarybench.providers.xai_openai_compat import (
        MAX_EXTENSION_COUNT,
        MAX_REASONING_CONTENT_CHARACTERS,
        RESPONSE_EXTENSION_CONTRACT,
        RESPONSE_EXTENSION_SCHEMA,
    )

    assert _plain(RESPONSE_EXTENSION_SCHEMA) == EXPECTED_SCHEMA
    assert _plain(RESPONSE_EXTENSION_CONTRACT) == EXPECTED_CONTRACT
    assert MAX_EXTENSION_COUNT == MAX_EXACT_TOKEN_COUNT
    assert MAX_REASONING_CONTENT_CHARACTERS == MAX_OUTPUT_TOKENS * 16


def test_the_contract_is_named_and_hashed_in_the_settings_run_identity_covers() -> None:
    """A named contract nobody can re-derive is a label; the digest is the claim."""
    from boundarybench.providers.xai_openai_compat import (
        RESPONSE_EXTENSION_DIGEST,
        XAI_COMPAT_RESPONSE_EXTENSIONS,
    )

    digest = hashlib.sha256(
        canonical_json_bytes(EXPECTED_CONTRACT, "xai compat response extensions")
    ).hexdigest()
    settings = xai_settings(model=GROK_4_5_MODEL)

    assert XAI_COMPAT_RESPONSE_EXTENSIONS == EXTENSIONS_CONTRACT
    assert digest == RESPONSE_EXTENSION_DIGEST
    assert settings["response_extensions"] == EXTENSIONS_CONTRACT
    assert settings["response_extensions_digest"] == digest


def test_the_digest_covers_the_reading_and_not_only_the_transcribed_names() -> None:
    """Two builds can share every field name and still read one body differently.

    A digest over the name-and-type transcription alone would be equal across a
    build that requires the detail pair whole and one that accepts half of it, so
    the rules and the bounds are hashed beside the shape.
    """
    from boundarybench.providers.xai_openai_compat import RESPONSE_EXTENSION_DIGEST

    shape_only = hashlib.sha256(
        canonical_json_bytes(EXPECTED_SCHEMA, "xai compat response extensions")
    ).hexdigest()
    permissive = dict(EXPECTED_CONTRACT) | {
        "members_required_together": {
            "completion_message": False,
            "prompt_token_details": False,
            "usage_block": False,
        }
    }
    permissive_digest = hashlib.sha256(
        canonical_json_bytes(permissive, "xai compat response extensions")
    ).hexdigest()

    assert EXPECTED_CONTRACT["members"] == EXPECTED_SCHEMA
    assert shape_only != RESPONSE_EXTENSION_DIGEST
    assert permissive_digest != RESPONSE_EXTENSION_DIGEST


def test_the_contract_is_provider_observed_and_says_so_by_not_claiming_the_sdk() -> None:
    """Honesty about provenance: the pinned SDK declares none of these fields.

    If it declared them there would be nothing to amend — ``declared_wire_fields``
    would already allow them and the response-contract check would already count
    zero. The contract exists precisely because the vendor's own model does not
    state them, so it is a transcription of what this repository observed.
    """
    from boundarybench.providers.xai_openai_compat import RESPONSE_EXTENSION_SCHEMA

    declared = {
        "completion_message": declared_wire_fields(ChatCompletionMessage),
        "usage_block": declared_wire_fields(CompletionUsage),
        "prompt_token_details": declared_wire_fields(PromptTokensDetails),
    }

    for site, members in RESPONSE_EXTENSION_SCHEMA.items():
        assert set(members).isdisjoint(declared[site])


# -- 7. what moved, and what did not -------------------------------------------


def test_the_xai_adapter_version_moved_because_what_it_accepts_moved() -> None:
    """``0.3.0``: what a run will read as an answer is part of the integration."""
    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert xai_identity(GROK_4_5_MODEL).version == "0.3.0"


#: Every shape of body this contract has an opinion about, each one a single
#: string of bytes: the whole observed shape, none of it, each site alone, the
#: SDK's own detail object without the pair, a choice stating no message, and a
#: body with no usage block at all.
SINGLE_SOURCE_BODIES: tuple[dict[str, Any], ...] = (
    _observed(),
    _bare_body(),
    _reasoning_body(),
    _body(usage={"cost_in_usd_ticks": OBSERVED_COST_TICKS}),
    _body(details={"image_tokens": OBSERVED_IMAGE_TOKENS, "text_tokens": 1}),
    _body(details={"cached_tokens": 0}),
    _without_message(_bare_body()),
    _without_usage(_reasoning_body()),
    _with_second_choice(_reasoning_body()),
)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(body, id=str(index))
        for index, body in enumerate(SINGLE_SOURCE_BODIES)
    ],
)
def test_one_string_of_bytes_states_the_same_sites_on_both_readings(
    body: dict[str, Any],
) -> None:
    """Why comparing the union of the two readings did not move run identity.

    A turn holds two readings of *one* string of bytes: the exact JSON, and this
    pinned SDK's typed reading of the same bytes. Every object this contract
    fixes names on is therefore stated by both readings or by neither — the SDK
    resolves a stated object to an object and an absent or ``null`` one to
    ``None`` — so no body a provider can send is admitted under one rule and
    refused under the other. The contract's name, its digest, the adapter version
    and so ``configuration_id`` are unchanged, because the set of bodies this run
    will read as an answer is unchanged.
    """
    check_response_extensions_agree(body, _parsed(body))


def test_the_identity_this_lane_records_did_not_move_with_the_comparison() -> None:
    """The deliberate choice, stated where a reader compares run identity.

    The three carriers of this lane's response-extension identity are the
    contract's name, the digest over the contract document, and the adapter
    version. None of them moves here: the document is the same document — the
    same sites, the same members, the same bounds, the same all-or-nothing rule —
    and, as the case above shows, no single body a provider can send changes
    which side of the contract it falls on. Moving any of the three would claim a
    reading changed for runs that read every real body exactly as before, and
    would strand a resume across a change that is not one.
    """
    from boundarybench.providers.xai_openai_compat import (
        RESPONSE_EXTENSION_DIGEST,
        XAI_COMPAT_RESPONSE_EXTENSIONS,
    )

    settings = xai_settings(model=GROK_4_5_MODEL)

    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert XAI_COMPAT_RESPONSE_EXTENSIONS == EXTENSIONS_CONTRACT
    assert (
        hashlib.sha256(
            canonical_json_bytes(EXPECTED_CONTRACT, "xai compat response extensions")
        ).hexdigest()
        == RESPONSE_EXTENSION_DIGEST
    )
    assert settings["response_extensions"] == EXTENSIONS_CONTRACT
    assert settings["response_extensions_digest"] == RESPONSE_EXTENSION_DIGEST


def _manifest(settings: Any, *, version: str) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider="xai",
        model=GROK_4_5_MODEL,
        implementation="xai_openai_compat_chat_completions",
        adapter_version=version,
        adapter_settings=settings,
        trials=1,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
    )


def test_the_configuration_identity_is_not_the_one_that_refused_these_fields() -> None:
    """A run that would have refused this body is not a run that accepts it.

    Each of the three carriers — the version, the contract name and the digest —
    is checked while the other two are held at the superseded value, because an
    identity that moved for only one of them would still let a resume cross the
    change on the other two.
    """
    current = xai_settings(model=GROK_4_5_MODEL)
    superseded = {
        key: value
        for key, value in current.items()
        if not key.startswith("response_extensions")
    }
    only_name_moved = dict(superseded) | {
        "response_extensions": EXTENSIONS_CONTRACT,
    }
    only_digest_moved = dict(superseded) | {
        "response_extensions_digest": current["response_extensions_digest"],
    }

    assert _manifest(current, version=XAI_ADAPTER_VERSION).configuration_id not in {
        _manifest(superseded, version="0.2.0").configuration_id,
        _manifest(superseded, version=XAI_ADAPTER_VERSION).configuration_id,
        _manifest(only_name_moved, version=XAI_ADAPTER_VERSION).configuration_id,
        _manifest(only_digest_moved, version=XAI_ADAPTER_VERSION).configuration_id,
        _manifest(current, version="0.2.0").configuration_id,
    }


def test_the_request_this_turn_sends_is_byte_identical_to_the_superseded_one() -> None:
    """What this build *accepts* moved. What it *asks* must not have.

    Asserted against the body ``httpx.MockTransport`` actually received after the
    real SDK serialised it, and against this build's own request mapping, so a
    silent change to the question asked of the model would fail here. There is no
    reasoning field in it and this change did not invent one: the model page
    publishes reasoning as always on and no control that disables it.
    """
    transport = RecordingTransport([_observed()])
    adapter = XAIOpenAICompatAdapter(model=GROK_4_5_MODEL, client=transport.xai_client())

    adapter.next_call(_turn_request(), _deadline())

    body = transport.bodies[0]
    profile = request_profile_for(GROK_4_5_MODEL)
    assert canonical_json_text(body, "sent") == canonical_json_text(
        build_chat_request(_turn_request(), model=GROK_4_5_MODEL), "built"
    )
    assert body["temperature"] == 0.0
    assert "reasoning_effort" not in body
    assert "top_p" not in body
    assert tuple(sorted(body)) == profile.sent_fields()
    assert tuple(sorted(body)) == (
        "max_tokens",
        "messages",
        "model",
        "parallel_tool_calls",
        "temperature",
        "tool_choice",
        "tools",
    )


def test_the_request_profile_and_pricing_claims_are_untouched() -> None:
    """The profile still records an unverified contract and no reasoning control."""
    settings = xai_settings(model=GROK_4_5_MODEL)

    assert settings["model_contract_verified"] is False
    assert settings["reasoning_effort"] is None
    assert settings["temperature"] == 0.0
    assert settings["request_profile"] == "grok45_temperature_zero_reasoning_omitted_v1"


def test_no_other_lane_accepts_these_fields_or_records_this_contract() -> None:
    """The amendment is one lane's. Everywhere else, *these* extras are still zero.

    The Mistral lane has since named a response-extension contract of its own, at
    ``0.3.0``, for the ``prompt_tokens_details`` object that service states on its
    usage block. It shares this lane's settings *key* — both are a lane's own
    reading of its own service's extensions — so the assertion below is that no
    other lane records *this* contract, by name and by digest, rather than that
    no other lane records the key at all. Sharing a key name is not sharing a
    contract: the two fix different names at different sites, and neither's
    allowance reaches the other's surface.
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
    from boundarybench.providers.openai_responses import (
        GPT_5_6_LUNA_MODEL,
        OPENAI_ADAPTER_VERSION,
        openai_settings,
    )
    from boundarybench.providers.xai_openai_compat import RESPONSE_EXTENSION_DIGEST

    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert MISTRAL_ADAPTER_VERSION == "0.3.0"
    assert ANTHROPIC_ADAPTER_VERSION == "0.8.0"
    for settings in (
        openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL),
        mistral_settings(model=MISTRAL_SMALL_MODEL),
        anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL),
    ):
        assert settings.get("response_extensions") != EXTENSIONS_CONTRACT
        assert settings.get("response_extensions_digest") != RESPONSE_EXTENSION_DIGEST


def test_the_openai_lane_still_refuses_the_same_fields_on_its_own_surface() -> None:
    """Sharing an SDK is not sharing a response contract.

    The OpenAI lane speaks the Responses API, where none of these names is
    declared and none of them is named by its own extension contract, so a body
    carrying them is refused there exactly as it was before.
    """
    from boundarybench.providers.openai_responses import (
        GPT_5_6_LUNA_MODEL,
        OpenAIResponsesAdapter,
    )
    from tests.openai_transport import function_call_item, responses_body

    body = responses_body(
        [function_call_item("read_records", "{}")], model=GPT_5_6_LUNA_MODEL
    )
    body["usage"]["cost_in_usd_ticks"] = OBSERVED_COST_TICKS
    transport = RecordingTransport([body])
    adapter = OpenAIResponsesAdapter(
        model=GPT_5_6_LUNA_MODEL, client=transport.client(), cost_guard=None
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_the_common_response_contract_check_still_defaults_to_zero_extras() -> None:
    """The shared checker was widened by an argument, not by a new default."""
    from boundarybench.providers.common import check_response_contract
    from boundarybench.providers.xai_openai_compat import TYPED_EXTENSION_EXTRAS

    parsed = _parsed(_observed())

    with pytest.raises(AdapterProviderError) as caught:
        check_response_contract(parsed)

    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    check_response_contract(parsed, allowed_extras_by_model=TYPED_EXTENSION_EXTRAS)
