"""The Mistral lane's own response-extension contract.

``mistral-small-2603`` returns, on an otherwise ordinary HTTP 200, one object on
the usage block that the pinned ``mistralai`` SDK's ``UsageInfo`` does not
declare: ``prompt_tokens_details``, carrying a single ``cached_tokens`` count.
Under the response contract that object is an undeclared field, so the whole
body was refused: the action went unparsed, the usage went unmeasured, and the
turn's reservation was held as exposure over a field nothing here reads.

So this lane names **one exact contract** for it —
``mistral_response_extensions_v1`` — and accepts precisely it: closed, complete
where it is present, typed scalar by scalar, bounded, and read for nothing.

Two bounded schema-only diagnostics established the name and the type shape
against this exact model. The raw bodies were not persisted and no value from
them is transcribed here; what the contract fixes is the *shape*, and the
per-turn numbers below are stand-ins of the right type.

**What the pinned SDK does with the object.** It does not declare it, and it does
not drop it either: ``UsageInfo`` is configured ``extra="allow"``, so the SDK's
own unmarshal path parks the object verbatim in ``usage.model_extra``. Those are
two different things, and the difference decides this contract's shape. Because
the object survives into the typed reading, it is stated **twice** — once in the
exact JSON the provider sent and once on the typed object — so it is checked on
both readings and the two must state the same thing, exactly as the xAI lane
checks its own. Neither reading is preferred, neither is edited to agree with the
other, and ``model_extra`` is never cleared. A raw-only contract would also be
non-functional here: ``check_response_contract`` counts the typed extra and
refuses the body on its own, so accepting this body requires naming the typed
allowance as well as the wire one.

**The count is read for nothing, and priced into nothing.** ``cached_tokens``
says a provider cached part of a prompt and charged less for it. This build does
not model that discount: the pinned Mistral price charges every
provider-reported ``prompt_token`` at the uncached input rate, which is
conservative in the only direction that matters — a run never under-reports what
it spent. The tests below prove that moving ``cached_tokens`` across its whole
domain changes neither the accepted action, nor the standard usage, nor the cost.

Every test here runs the real SDK over an in-process transport (see
``tests.mistral_transport``). No test contacts a provider, and none of them is
evidence about a model: they are evidence about the integration.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from boundarybench.adapter import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment
from boundarybench.jsonsafe import canonical_json_bytes
from boundarybench.pricing import MAX_EXACT_TOKEN_COUNT, MISTRAL_PRICING_MODELS
from boundarybench.providers.common import ProviderRetryPolicy
from boundarybench.providers.mistral_chat import (
    MISTRAL_ADAPTER_VERSION,
    MISTRAL_SMALL_MODEL,
    SDK_INJECTED_FIELDS,
    USAGE_WIRE_SHAPE,
    MistralChatAdapter,
    build_mistral_request,
    check_response_extensions,
    check_response_extensions_agree,
    mistral_settings,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST, minimal_card
from tests.mistral_transport import RecordingTransport, scripted_client, tool_call

#: The contract's own name, written out rather than imported: it is inside
#: ``configuration_id``, so an identifier asserted against itself would follow
#: any rename and prove nothing about the run identity a reader compares.
EXTENSIONS_CONTRACT = "mistral_response_extensions_v1"

#: Where the object hangs, as the contract names the site.
USAGE_SITE = "usage_block"

#: The exact name and type shape the diagnostics established, hand-written here
#: rather than imported from the module under test. The published schema is
#: asserted against *this* literal and the digest in settings against a SHA-256
#: over the canonical bytes of the whole contract, so a silent widening of what a
#: run will read as an answer fails here.
EXPECTED_SCHEMA: dict[str, Any] = {
    "prompt_tokens_details": {"cached_tokens": "non_negative_bounded_integer"}
}

#: The rules the shape is read under, hashed beside it. The shape alone is not
#: the contract: a build that transcribed this same name and type and then
#: accepted a vacuous ``{}`` would read a body this one refuses, and a digest
#: over the transcription alone would be equal across the two.
EXPECTED_CONTRACT: dict[str, Any] = {
    "contract": EXTENSIONS_CONTRACT,
    # Stated in the document rather than left to prose: this build reads no value
    # in it, which is why the object as a whole is allowed to be absent.
    "extension_values_read": False,
    "max_count": MAX_EXACT_TOKEN_COUNT,
    "members": EXPECTED_SCHEMA,
    # A present object states every member fixed for it...
    "object_members_required": True,
    # ...and the object itself may be absent.
    "top_level_members_required": False,
}

#: Every field name this contract fixes. No refusal may name one.
EXTENSION_FIELD_NAMES: tuple[str, ...] = ("cached_tokens", "prompt_tokens_details")

#: The standard counts every body below states. Per-turn facts; what the contract
#: fixes is their type, and what this file proves is that they do not move.
OBSERVED_PROMPT_TOKENS = 137
OBSERVED_COMPLETION_TOKENS = 29

#: A cached count of the right type. Its value is a per-turn fact.
OBSERVED_CACHED_TOKENS = 64

#: The action every accepted body below carries.
OBSERVED_ACTION = "read_records"


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


def _body(
    *, details: Any = ..., usage: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One chat-completion body from the pinned model, carrying one tool call.

    ``details`` defaults to the sentinel, which states no ``prompt_tokens_details``
    at all — the body this lane read perfectly well before this contract existed.
    Anything else is placed in that field exactly as given, including ``None``.
    """
    body: dict[str, Any] = {
        "id": "cmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": MISTRAL_SMALL_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call(OBSERVED_ACTION, "{}")],
                },
            }
        ],
        "usage": {
            "prompt_tokens": OBSERVED_PROMPT_TOKENS,
            "completion_tokens": OBSERVED_COMPLETION_TOKENS,
            "total_tokens": OBSERVED_PROMPT_TOKENS + OBSERVED_COMPLETION_TOKENS,
        },
    }
    if details is not ...:
        body["usage"]["prompt_tokens_details"] = details
    if usage:
        body["usage"].update(usage)
    return body


def _observed() -> dict[str, Any]:
    """The body as the diagnostics observed it: the object, whole and well-typed."""
    return _body(details={"cached_tokens": OBSERVED_CACHED_TOKENS})


def _run(body: Mapping[str, Any]) -> tuple[RecordingTransport, MistralChatAdapter, Any]:
    """One turn against one scripted body, through the real SDK and the adapter."""
    transport, client = scripted_client(copy.deepcopy(body))
    adapter = MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)
    call = adapter.next_call(_turn_request(), _deadline())
    return transport, adapter, call


def _refused(body: Mapping[str, Any]) -> AdapterProviderError:
    """One body this contract does not describe, refused whole rather than read past."""
    _, client = scripted_client(copy.deepcopy(body))
    adapter = MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)
    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID
    # The capture is empty on the way out of a refusal exactly as it is on the
    # way out of a success; see the section on it below.
    assert adapter._capture.captured_count == 0
    return caught.value


# -- 1. the published contract ------------------------------------------------


def test_the_published_schema_is_exactly_the_observed_shape() -> None:
    """The transcription, asserted against a literal rather than against itself."""
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_SCHEMA

    assert _plain(RESPONSE_EXTENSION_SCHEMA) == EXPECTED_SCHEMA


def test_the_published_contract_states_the_rules_beside_the_shape() -> None:
    """The shape alone is not the contract; the rules it is read under are in it."""
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_CONTRACT

    assert _plain(RESPONSE_EXTENSION_CONTRACT) == EXPECTED_CONTRACT


def test_the_digest_is_a_sha256_over_the_whole_contract_document() -> None:
    """Re-derivable from the published document, not merely recorded beside it."""
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_DIGEST

    assert (
        hashlib.sha256(
            canonical_json_bytes(EXPECTED_CONTRACT, "mistral response extensions")
        ).hexdigest()
        == RESPONSE_EXTENSION_DIGEST
    )


def test_the_contract_is_deeply_immutable() -> None:
    """A schema shared by reference and mutable is not a fixed contract."""
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_SCHEMA

    with pytest.raises(TypeError):
        RESPONSE_EXTENSION_SCHEMA["prompt_tokens_details"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        RESPONSE_EXTENSION_SCHEMA["prompt_tokens_details"]["cached_tokens"] = "x"  # type: ignore[index]


def test_the_digest_moves_when_the_accepted_shape_would() -> None:
    """The digest answers "which bodies will this run read", not "which names".

    A build that transcribed this same name and type and then accepted a vacuous
    object is a different reading of the same wire, and its digest must not be
    equal to this one's.
    """
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_DIGEST

    widened = dict(EXPECTED_CONTRACT) | {"object_members_required": False}
    assert (
        hashlib.sha256(
            canonical_json_bytes(widened, "mistral response extensions")
        ).hexdigest()
        != RESPONSE_EXTENSION_DIGEST
    )


# -- 2. the object is optional, and exact when it is here ---------------------


def test_a_body_that_states_no_details_object_is_accepted_unchanged() -> None:
    """Absence is not a failure: this build reads nothing out of the object.

    The body this lane read perfectly well before the contract existed still
    reads exactly the same way, which is what makes the amendment an addition
    rather than a new requirement placed on the provider.
    """
    _, adapter, call = _run(_body())

    assert call.action == OBSERVED_ACTION
    usage = adapter.last_usage()
    assert usage.input_tokens == OBSERVED_PROMPT_TOKENS
    assert usage.output_tokens == OBSERVED_COMPLETION_TOKENS


def test_the_observed_object_is_accepted_and_the_turn_reads_normally() -> None:
    """The blocker, at the door a turn actually goes through.

    This is the body a real HTTP 200 from ``mistral-small-2603`` carried and that
    this build refused whole. It now yields the action it always stated and the
    two counts it always stated.
    """
    _, adapter, call = _run(_observed())

    assert call.action == OBSERVED_ACTION
    usage = adapter.last_usage()
    assert usage.input_tokens == OBSERVED_PROMPT_TOKENS
    assert usage.output_tokens == OBSERVED_COMPLETION_TOKENS


@pytest.mark.parametrize(
    "cached",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(OBSERVED_CACHED_TOKENS, id="observed"),
        pytest.param(MAX_EXACT_TOKEN_COUNT, id="max-exact"),
    ],
)
def test_every_count_inside_the_bound_is_accepted(cached: int) -> None:
    """The whole accepted domain, ends included.

    Zero is a real answer — a prompt nothing of which was cached — and the bound
    is the largest integer that survives a JSON number in every reader this
    build's artefacts cross.
    """
    _, _, call = _run(_body(details={"cached_tokens": cached}))

    assert call.action == OBSERVED_ACTION


@pytest.mark.parametrize(
    "cached",
    [
        pytest.param(MAX_EXACT_TOKEN_COUNT + 1, id="one-past-the-bound"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean-true"),
        pytest.param(False, id="boolean-false"),
        pytest.param("64", id="string-of-digits"),
        pytest.param(64.0, id="float-that-is-whole"),
        pytest.param(0.5, id="float"),
        pytest.param(None, id="null"),
        pytest.param([64], id="list"),
        pytest.param({"cached_tokens": 64}, id="object"),
    ],
)
def test_a_count_outside_the_fixed_type_is_refused(cached: Any) -> None:
    """Typed on the wire, not after coercion.

    ``True`` is an ``int`` in Python and ``"64"`` is what a lenient reader
    renders as sixty-four; neither is a count anybody measured. A float is
    refused even when it is whole, because a JSON fraction is not a JSON integer
    and the contract fixes the wire type rather than the value.
    """
    _refused(_body(details={"cached_tokens": cached}))


# -- 3. a present object is a whole object ------------------------------------


def test_a_vacuous_object_is_refused() -> None:
    """``{}`` states nothing, and a present object is a statement.

    The whole object may be absent — this build reads none of it — but an object
    that arrives states every member this contract fixes for it. A vacuous one
    was never observed, and accepting it would widen an exact contract into a
    guess.
    """
    _refused(_body(details={}))


def test_an_object_that_omits_the_count_is_refused() -> None:
    """The same rule, seen through an object that states something else instead.

    Refused twice over, and both refusals are this contract's: ``cached_tokens``
    is required, and ``other`` is a name the contract does not fix.
    """
    _refused(_body(details={"other": 1}))


@pytest.mark.parametrize(
    "details",
    [
        pytest.param(None, id="null"),
        pytest.param(0, id="number"),
        pytest.param("cached", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param([{"cached_tokens": 1}], id="list"),
    ],
)
def test_a_details_field_that_is_not_an_object_is_refused(details: Any) -> None:
    """The contract fixes an object there; anything else is not the shape it names.

    ``None`` is refused with the rest rather than read as an absence: the way to
    state that this object is absent is not to send the field, and a null is a
    value the SDK would park in ``model_extra`` exactly as it parks the object.
    """
    _refused(_body(details=details))


# -- 4. everything outside the contract is still refused ----------------------


def test_an_unknown_field_inside_the_details_object_is_refused() -> None:
    """Closed at the depth the contract reaches, not only at the top of it."""
    _refused(
        _body(details={"cached_tokens": OBSERVED_CACHED_TOKENS, "cached_ratio": 0.5})
    )


def test_an_unknown_field_on_the_usage_block_is_refused() -> None:
    """The allowance is one name on one object, not "extras on the usage block"."""
    _refused(_body(usage={"cached_tokens": OBSERVED_CACHED_TOKENS}))


def test_an_unknown_field_at_the_root_is_refused() -> None:
    """A name proven for the usage block buys nothing at the root."""
    body = _observed()
    body["prompt_tokens_details"] = {"cached_tokens": OBSERVED_CACHED_TOKENS}
    _refused(body)


def test_the_contract_name_is_not_a_licence_on_another_object() -> None:
    """The same object, on a message, is still an undeclared field."""
    body = _observed()
    body["choices"][0]["message"]["prompt_tokens_details"] = {"cached_tokens": 1}
    _refused(body)


def test_no_refusal_quotes_a_field_name_or_a_value() -> None:
    """A durable failure row carries this build's own detail and nothing else.

    The contract's field names are this build's, but the *values* beside them are
    provider-controlled, and a row that named the field would tell a reader which
    provider-controlled value had been in it. Neither is recorded.
    """
    refusals = [
        _refused(_body(details={"cached_tokens": -1})),
        _refused(_body(details={})),
        _refused(_body(details={"cached_tokens": 1, "cached_ratio": 0.5})),
        _refused(_body(usage={"cached_tokens": 1})),
    ]
    for refusal in refusals:
        detail = str(refusal)
        for name in EXTENSION_FIELD_NAMES:
            assert name not in detail
        assert "0.5" not in detail
        assert str(OBSERVED_CACHED_TOKENS) not in detail


# -- 5. what the pinned SDK actually does with the object ---------------------
#
# The premise this contract was designed against was that the SDK drops the
# object before a typed response exists, which would make raw-vs-typed agreement
# impossible and the contract raw-only. It is not what this SDK version does, and
# the difference decides the contract's shape — so it is pinned here rather than
# assumed, and it will fail loudly if a future SDK changes its mind.


def test_the_pinned_sdk_does_not_declare_the_field() -> None:
    """The reason this needed a contract of its own rather than a schema entry."""
    from mistralai.client.models import UsageInfo

    assert "prompt_tokens_details" not in UsageInfo.model_fields


def test_the_pinned_sdk_preserves_the_object_rather_than_dropping_it() -> None:
    """Not declared and not dropped are two different things; this SDK does the first.

    ``UsageInfo`` is configured ``extra="allow"``, so the object survives the
    SDK's own unmarshal path verbatim. That is what makes the typed reading a
    real second statement of the same field, and therefore what makes the
    agreement check below possible rather than fabricated.
    """
    from mistralai.client.models import UsageInfo

    assert UsageInfo.model_config.get("extra") == "allow"

    _, _, _ = _run(_observed())

    import json

    from mistralai.client import models
    from mistralai.client.utils.serializers import unmarshal_json

    parsed = unmarshal_json(json.dumps(_observed()), models.ChatCompletionResponse)
    assert parsed.usage.model_extra == {
        "prompt_tokens_details": {"cached_tokens": OBSERVED_CACHED_TOKENS}
    }


def test_the_typed_extras_are_validated_rather_than_cleared_to_hide_them() -> None:
    """Emptying ``model_extra`` would make the two readings agree by force.

    It would also hide the very field this contract exists to describe and leave
    the SDK's typed object in a state the provider never sent. The typed parse
    stays exactly as the SDK produced it, and the standard counts beside it are
    untouched.
    """
    import json

    from mistralai.client import models
    from mistralai.client.utils.serializers import unmarshal_json

    body = _observed()
    parsed = unmarshal_json(json.dumps(body), models.ChatCompletionResponse)
    check_response_extensions_agree(body, parsed)

    assert parsed.usage.model_extra == {
        "prompt_tokens_details": {"cached_tokens": OBSERVED_CACHED_TOKENS}
    }
    assert parsed.usage.prompt_tokens == OBSERVED_PROMPT_TOKENS


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(None, id="null"),
        pytest.param(0, id="number"),
        pytest.param("usage", id="string"),
        pytest.param([], id="list"),
    ],
)
def test_a_raw_usage_block_that_is_not_an_object_states_no_extensions(
    usage: Any,
) -> None:
    """A body with no readable usage block states no extension to compare with.

    A body like this never reaches here through the adapter — the wire contract
    proves the usage block is an object first — but this is a public checker, and
    the alternative to the guard is an attribute error dressed as an adapter bug.
    It states nothing rather than being read as stating something, so it agrees
    with a typed reading that also states nothing and disagrees with one that
    does. Both directions are asserted, because a guard that swallowed the
    disagreement would be worse than the crash it replaced.
    """
    import json

    from mistralai.client import models
    from mistralai.client.utils.serializers import unmarshal_json

    bare = unmarshal_json(json.dumps(_body()), models.ChatCompletionResponse)
    check_response_extensions_agree({"usage": usage}, bare)

    stated = unmarshal_json(json.dumps(_observed()), models.ChatCompletionResponse)
    with pytest.raises(AdapterProviderError) as caught:
        check_response_extensions_agree({"usage": usage}, stated)
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


@pytest.mark.parametrize(
    ("raw_details", "typed_details"),
    [
        pytest.param({"cached_tokens": 1}, {"cached_tokens": 2}, id="different-count"),
        pytest.param({"cached_tokens": 1}, ..., id="raw-only"),
        pytest.param(..., {"cached_tokens": 1}, id="typed-only"),
    ],
)
def test_two_readings_that_state_different_things_are_refused(
    raw_details: Any, typed_details: Any
) -> None:
    """Neither reading is preferred, and neither is edited to agree with the other.

    A body whose typed extras are not the extras on the wire is not one body this
    build can describe, so it is refused whole. The raw-only and typed-only cases
    are the same failure seen from two sides.
    """
    import json

    from mistralai.client import models
    from mistralai.client.utils.serializers import unmarshal_json

    wire = _body(details=raw_details)
    parsed = unmarshal_json(
        json.dumps(_body(details=typed_details)), models.ChatCompletionResponse
    )
    with pytest.raises(AdapterProviderError) as caught:
        check_response_extensions_agree(wire, parsed)
    assert caught.value.fault == PROVIDER_FAULT_RESPONSE_INVALID


def test_the_checker_accepts_the_two_bodies_it_should() -> None:
    """Both readings stating the object, and neither stating it."""
    import json

    from mistralai.client import models
    from mistralai.client.utils.serializers import unmarshal_json

    for body in (_body(), _observed()):
        parsed = unmarshal_json(json.dumps(body), models.ChatCompletionResponse)
        check_response_extensions_agree(body, parsed)


def test_the_site_checker_is_public_and_takes_the_whole_object() -> None:
    """Given the whole object, so a caller cannot prove the contract about a copy."""
    check_response_extensions(
        {
            "prompt_tokens": OBSERVED_PROMPT_TOKENS,
            "prompt_tokens_details": {"cached_tokens": OBSERVED_CACHED_TOKENS},
        }
    )
    with pytest.raises(AdapterProviderError):
        check_response_extensions({"prompt_tokens_details": {}})


# -- 6. the count is read for nothing, and priced into nothing ----------------


CACHED_VALUES: tuple[Any, ...] = (0, 1, OBSERVED_CACHED_TOKENS, MAX_EXACT_TOKEN_COUNT)


@pytest.mark.parametrize("cached", CACHED_VALUES)
def test_moving_the_cached_count_changes_neither_the_action_nor_the_usage(
    cached: int,
) -> None:
    """The whole accepted domain, against the two things a turn is measured by."""
    _, adapter, call = _run(_body(details={"cached_tokens": cached}))

    assert call.action == OBSERVED_ACTION
    usage = adapter.last_usage()
    assert usage.input_tokens == OBSERVED_PROMPT_TOKENS
    assert usage.output_tokens == OBSERVED_COMPLETION_TOKENS


def test_every_cached_count_settles_the_identical_usage_as_no_object_at_all() -> None:
    """The baseline is the body that states no object, and nothing moves off it."""
    _, baseline_adapter, baseline_call = _run(_body())
    baseline = baseline_adapter.last_usage()

    for cached in CACHED_VALUES:
        _, adapter, call = _run(_body(details={"cached_tokens": cached}))
        usage = adapter.last_usage()
        assert (usage.input_tokens, usage.output_tokens) == (
            baseline.input_tokens,
            baseline.output_tokens,
        )
        assert call.action == baseline_call.action


def test_the_pinned_price_charges_every_prompt_token_at_the_uncached_rate() -> None:
    """Conservative on purpose: this build does not model the provider's discount.

    ``cached_tokens`` says the provider charged less for part of the prompt. This
    build prices the whole of ``prompt_tokens`` at the uncached input rate, so a
    run never under-reports what it spent. The assertion that this is
    *conservative* rather than merely *unchanged* is the second one: pricing the
    discount would produce a strictly smaller number, and that is the number this
    build declines to claim.
    """
    price = MISTRAL_PRICING_MODELS[MISTRAL_SMALL_MODEL]
    charged = price.cost(
        input_tokens=OBSERVED_PROMPT_TOKENS, output_tokens=OBSERVED_COMPLETION_TOKENS
    )

    for cached in CACHED_VALUES:
        _, adapter, _ = _run(_body(details={"cached_tokens": cached}))
        usage = adapter.last_usage()
        assert usage.input_tokens is not None
        assert usage.output_tokens is not None
        assert (
            price.cost(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
            == charged
        )

    discounted = price.cost(
        input_tokens=OBSERVED_PROMPT_TOKENS - OBSERVED_CACHED_TOKENS,
        output_tokens=OBSERVED_COMPLETION_TOKENS,
    )
    assert discounted < charged


def test_the_price_states_no_cached_rate_for_anything_to_apply() -> None:
    """Nothing downstream can apply a discount this build never recorded.

    Stronger than "the cost did not move": a price that carried a cached rate
    could be applied by some later reader even though this contract passes the
    count to nobody. There is no such field to apply — the only tier this model
    price can state is the long-context one, and this model states none — so the
    conservative reading is a property of the pinned table rather than of the
    call site that happens not to use it.
    """
    from dataclasses import fields

    price = MISTRAL_PRICING_MODELS[MISTRAL_SMALL_MODEL]

    assert not [field.name for field in fields(price) if "cach" in field.name.lower()]
    assert price.long_context_tier is None
    # ...so the rate is the same one at any prompt length, including one that
    # would have been mostly cache hits.
    assert price.rates_for(OBSERVED_PROMPT_TOKENS) == price.rates_for(
        MAX_EXACT_TOKEN_COUNT
    )


# -- 7. what this build sends is untouched ------------------------------------


def test_the_request_this_turn_sends_is_byte_identical_to_the_superseded_one() -> None:
    """What this build *accepts* moved. What it *asks* must not have.

    The observed object is a property of the service's response, not of anything
    this build requested, and no parameter that would turn it off is invented
    here. So the body on the wire is exactly the adapter's own payload plus the
    SDK's injected field, as it was before.
    """
    transport, _, _ = _run(_observed())

    request = _turn_request()
    payload = build_mistral_request(request, model=MISTRAL_SMALL_MODEL)
    sent = transport.bodies[0]
    assert set(sent) == set(payload) | set(SDK_INJECTED_FIELDS)
    assert sent["model"] == MISTRAL_SMALL_MODEL
    assert sent["max_tokens"] == payload["max_tokens"]
    assert sent["tool_choice"] == payload["tool_choice"]
    assert sent["parallel_tool_calls"] == payload["parallel_tool_calls"]
    assert sent["temperature"] == payload["temperature"]
    assert "prompt_tokens_details" not in sent
    assert "cached_tokens" not in sent


def test_the_request_profile_and_pricing_settings_did_not_move() -> None:
    """Only what this lane accepts moved; the profile it sends is the same one."""
    settings = mistral_settings(model=MISTRAL_SMALL_MODEL)

    assert settings["request_profile"] == "mistralsmall2603_temperature_zero_v1"
    assert settings["temperature"] == 0.0
    assert settings["max_output_tokens"] == 1024
    assert settings["tool_choice"] == "required"


# -- 8. the contract is inside run identity -----------------------------------


def test_the_settings_record_the_contract_and_its_digest() -> None:
    """A run's own claim about which bodies it would read as answers."""
    from boundarybench.providers.mistral_chat import RESPONSE_EXTENSION_DIGEST

    settings = mistral_settings(model=MISTRAL_SMALL_MODEL)

    assert settings["response_extensions"] == EXTENSIONS_CONTRACT
    assert settings["response_extensions_digest"] == RESPONSE_EXTENSION_DIGEST


def test_the_adapter_version_moved_for_this_amendment() -> None:
    """What an integration accepts is as much its version as what it sends."""
    assert MISTRAL_ADAPTER_VERSION == "0.3.0"


def _manifest(settings: Any, *, version: str) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider="mistral",
        model=MISTRAL_SMALL_MODEL,
        implementation="mistral_chat_completions",
        adapter_version=version,
        adapter_settings=settings,
        trials=1,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
    )


def test_the_configuration_identity_is_not_the_one_that_refused_this_body() -> None:
    """A run that would have refused this body is not a run that accepts it.

    Each of the three carriers — the version, the contract name and the digest —
    is checked while the other two are held at the superseded value, because an
    identity that moved for only one of them would still let a resume cross the
    change on the other two.
    """
    current = mistral_settings(model=MISTRAL_SMALL_MODEL)
    superseded = {
        key: value
        for key, value in current.items()
        if not key.startswith("response_extensions")
    }
    only_name_moved = dict(superseded) | {"response_extensions": EXTENSIONS_CONTRACT}
    only_digest_moved = dict(superseded) | {
        "response_extensions_digest": current["response_extensions_digest"]
    }

    assert _manifest(current, version=MISTRAL_ADAPTER_VERSION).configuration_id not in {
        _manifest(superseded, version="0.2.0").configuration_id,
        _manifest(superseded, version=MISTRAL_ADAPTER_VERSION).configuration_id,
        _manifest(only_name_moved, version=MISTRAL_ADAPTER_VERSION).configuration_id,
        _manifest(only_digest_moved, version=MISTRAL_ADAPTER_VERSION).configuration_id,
        _manifest(current, version="0.2.0").configuration_id,
    }


def test_no_other_lane_accepts_this_field_or_records_this_contract() -> None:
    """The amendment is one lane's. Everywhere else, extras are still zero."""
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_ADAPTER_VERSION,
        SONNET_5_MODEL,
        AnthropicRetryPolicy,
        anthropic_settings,
    )
    from boundarybench.providers.openai_responses import (
        GPT_5_6_LUNA_MODEL,
        OPENAI_ADAPTER_VERSION,
        openai_settings,
    )
    from boundarybench.providers.xai_openai_compat import (
        GROK_4_5_MODEL,
        XAI_ADAPTER_VERSION,
        xai_settings,
    )

    # The other three lanes' versions are unmoved by this amendment.
    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert ANTHROPIC_ADAPTER_VERSION == "0.8.0"

    # ...and none of them names *this* contract.
    for settings in (
        openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL),
        xai_settings(model=GROK_4_5_MODEL),
        anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL),
    ):
        assert settings.get("response_extensions") != EXTENSIONS_CONTRACT


def test_the_usage_wire_shape_widened_by_exactly_one_name() -> None:
    """The allowance is auditable as a set difference, not as prose."""
    from mistralai.client.models import UsageInfo

    from boundarybench.providers.wire import declared_wire_fields

    assert USAGE_WIRE_SHAPE.allowed - declared_wire_fields(UsageInfo) == {
        "prompt_tokens_details"
    }


# -- 9. the capture is still owned, single-flight and short-lived -------------


def test_the_capture_is_empty_after_a_turn_that_accepted_the_object() -> None:
    """The response is held for one turn and is never written anywhere.

    The contract added a check; it did not add a reason to keep a provider body
    alive past the turn it belongs to.
    """
    _, adapter, _ = _run(_observed())

    assert adapter._capture.captured_count == 0


def test_the_capture_is_empty_after_a_turn_this_contract_refused() -> None:
    """The same, on the way out that raises rather than returns."""
    transport, client = scripted_client(_body(details={}))
    adapter = MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())

    assert adapter._capture.captured_count == 0
    assert transport.calls == 1


def test_two_adapters_over_one_client_are_still_refused() -> None:
    """The ownership rule the capture is built on is untouched by this contract."""
    from boundarybench.providers.mistral_chat import MistralConfigurationError

    _, client = scripted_client(_observed())
    MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)

    with pytest.raises(MistralConfigurationError):
        MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)


def test_a_second_turn_reads_its_own_body_rather_than_the_first_turns() -> None:
    """Single-flight and cleared per turn: two turns, two independent readings."""
    first = _body(details={"cached_tokens": 1})
    second = _body(details={"cached_tokens": 2})
    transport, client = scripted_client(first, second)
    adapter = MistralChatAdapter(model=MISTRAL_SMALL_MODEL, client=client)

    assert adapter.next_call(_turn_request(), _deadline()).action == OBSERVED_ACTION
    assert adapter._capture.captured_count == 0
    assert adapter.next_call(_turn_request(), _deadline()).action == OBSERVED_ACTION
    assert adapter._capture.captured_count == 0
    assert transport.calls == 2
