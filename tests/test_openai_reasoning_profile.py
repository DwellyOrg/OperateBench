"""``gpt-5.6-luna``'s request profile, against its published model contract.

The model page this build read states three things this profile depends on: the
Responses API is supported, function calling is supported, and ``reasoning.effort``
accepts ``none``, ``low``, ``medium`` (the default), ``high``, ``xhigh`` and
``max``. It does *not* establish that the model accepts a sampling parameter.

So the profile sends an explicit ``reasoning: {"effort": "none"}`` and omits
``temperature`` altogether, and these tests hold that:

* the constants that transcribe the contract name the page they came from and
  the exact effort vocabulary it publishes, including that the default is
  ``medium`` and that this build never relies on it;
* what the *real* SDK serialises for this model carries the reasoning object and
  no sampling parameter — asserted against the body ``httpx.MockTransport``
  actually received, not against a helper's return value;
* ``adapter.settings`` — which is hashed into ``configuration_id`` — describes
  that body exactly, and says the model contract is verified;
* the adapter version and the resulting configuration identity both moved,
  because the request semantics did;
* the Anthropic, xAI and Mistral lanes are untouched by any of it.

Every request here is answered by ``httpx.MockTransport`` inside this process.
No socket is opened, no credential is read, and nothing here is evidence about
any model: it is evidence about the integration.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import openai
import pytest

from boundarybench.adapter import TurnDeadline, TurnRequest, build_turn_request
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    HAIKU_4_5_MODEL,
    SONNET_5_MODEL,
)
from boundarybench.providers.anthropic_messages import (
    request_profile_for as anthropic_profile_for,
)
from boundarybench.providers.common import SDK_MAX_RETRIES, ProviderRetryPolicy
from boundarybench.providers.mistral_chat import (
    MISTRAL_ADAPTER_VERSION,
    MISTRAL_SMALL_MODEL,
)
from boundarybench.providers.mistral_chat import (
    request_profile_for as mistral_profile_for,
)
from boundarybench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    MAX_OUTPUT_TOKENS,
    OPENAI_ADAPTER_VERSION,
    OPENAI_BASE_URL,
    OPENAI_RESPONSE_CAPTURE,
    OPENAI_RESPONSE_SERVER_EXTENSIONS,
    PROHIBITED_REQUEST_FIELDS,
    REQUEST_MAPPING_VERSION,
    SERVER_EXTENSION_DIGEST,
    OpenAIResponsesAdapter,
    openai_identity,
    openai_settings,
    request_profile_for,
)
from boundarybench.providers.xai_openai_compat import (
    GROK_4_5_MODEL,
    XAI_ADAPTER_VERSION,
)
from boundarybench.providers.xai_openai_compat import (
    request_profile_for as xai_profile_for,
)
from boundarybench.runmanifest import RunLimits, build_run_manifest
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST, minimal_card
from tests.openai_transport import (
    RecordingTransport,
    function_call_item,
    responses_body,
    scripted_client,
)

#: The profile identity this build ships for the pinned OpenAI model. Written
#: out rather than read off the profile: an identifier that is asserted against
#: itself would follow any rename, and this one is inside ``configuration_id``.
LUNA_PROFILE_ID = "gpt56luna_reasoning_none_sampling_omitted_v2"


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


def _wire_body() -> dict[str, Any]:
    """The body the real SDK put on the wire for one turn of the pinned model."""
    transport, client = scripted_client(
        responses_body(
            [function_call_item("read_records", "{}")], model=GPT_5_6_LUNA_MODEL
        )
    )
    adapter = OpenAIResponsesAdapter(model=GPT_5_6_LUNA_MODEL, client=client)

    adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 1
    return transport.bodies[0]


# -- the published contract, transcribed --------------------------------------


def test_the_profile_names_the_model_page_its_contract_was_read_from() -> None:
    """The claim ``model_contract_verified: true`` has to be checkable.

    A boolean saying a vendor contract was read is worth exactly as much as the
    reader's ability to go and read the same page, so the source is a constant in
    the module that makes the claim rather than a sentence in a commit message.

    Imported here rather than at the top of this module on purpose: the
    constants below are the transcription of the contract, and a module-level
    import of them would turn every other test in this file into a collection
    error rather than the specific failure it is meant to be.
    """
    from boundarybench.providers.openai_responses import GPT_5_6_LUNA_CONTRACT_SOURCE

    assert GPT_5_6_LUNA_CONTRACT_SOURCE == (
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna.md"
    )


def test_the_documented_effort_vocabulary_is_the_page_s_and_not_the_sdk_s() -> None:
    """The published values for this model, in the order the page states them.

    Pinned against the page rather than against ``openai.types``: the SDK's
    ``ReasoningEffort`` alias is the union over every model the SDK serves and
    carries a value this model's page does not publish. A profile that widened
    itself to the SDK's union would be claiming a contract it never read.
    """
    from boundarybench.providers.openai_responses import (
        REASONING_EFFORT_DEFAULT,
        REASONING_EFFORT_NONE,
        REASONING_EFFORT_VALUES,
    )

    assert REASONING_EFFORT_VALUES == ("none", "low", "medium", "high", "xhigh", "max")
    assert REASONING_EFFORT_NONE == "none"
    assert REASONING_EFFORT_DEFAULT == "medium"
    assert REASONING_EFFORT_NONE in REASONING_EFFORT_VALUES
    assert REASONING_EFFORT_DEFAULT in REASONING_EFFORT_VALUES


def test_the_reasoning_object_this_build_sends_is_the_documented_none_one() -> None:
    """One documented value, sent explicitly, and never the default.

    ``medium`` is what this model does when ``reasoning`` is absent. A
    standardized no-reasoning track that left the field out would be running
    under that default while recording nothing about it, so the field is sent and
    its value is the published ``none``.
    """
    from boundarybench.providers.openai_responses import (
        NO_REASONING,
        REASONING_EFFORT_DEFAULT,
        REASONING_EFFORT_NONE,
    )

    assert dict(NO_REASONING) == {"effort": REASONING_EFFORT_NONE}
    assert GPT_5_6_LUNA_PROFILE.reasoning is not None
    assert dict(GPT_5_6_LUNA_PROFILE.reasoning) == {"effort": "none"}
    assert dict(GPT_5_6_LUNA_PROFILE.reasoning)["effort"] != REASONING_EFFORT_DEFAULT


def test_the_profile_is_the_v2_identity_with_a_verified_contract() -> None:
    """The profile id is inside ``configuration_id``, so it is a durable claim."""
    assert GPT_5_6_LUNA_PROFILE.profile_id == LUNA_PROFILE_ID
    assert GPT_5_6_LUNA_PROFILE.contract_verified is True
    assert GPT_5_6_LUNA_PROFILE.temperature is None
    assert request_profile_for(GPT_5_6_LUNA_MODEL) == GPT_5_6_LUNA_PROFILE


def test_the_profile_accounts_for_every_field_a_profile_decides() -> None:
    """Each profile-controlled field is either sent or named as omitted."""
    profile = GPT_5_6_LUNA_PROFILE

    assert profile.sent_fields() == (
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "reasoning",
        "store",
        "tool_choice",
        "tools",
    )
    assert profile.omitted_fields() == ("temperature", "top_p")
    assert set(PROHIBITED_REQUEST_FIELDS) == {"reasoning", "temperature", "top_p"}


# -- what the real SDK actually serialises ------------------------------------


def test_the_serialized_body_carries_reasoning_none_and_no_sampling_field() -> None:
    """The wire body, key by key: the profile as the provider would receive it.

    Asserted against what ``httpx.MockTransport`` received after the real SDK
    serialised it, because the SDK decides what of a mapping reaches the wire —
    a helper agreeing with itself proves nothing about the request.
    """
    body = _wire_body()

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


def test_the_profile_moved_the_request_and_nothing_else_about_the_turn() -> None:
    """A profile that also moved the ceiling or the tool contract would be two changes."""
    body = _wire_body()

    assert body["model"] == GPT_5_6_LUNA_MODEL
    assert body["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["store"] is False
    assert [message["role"] for message in body["input"]] == ["user"]
    assert [tool["name"] for tool in body["tools"]] == [
        action.name for action in _turn_request().actions
    ]


def test_the_settings_describe_exactly_the_body_the_transport_saw() -> None:
    """``adapter.settings`` is hashed into run identity, so it is the claim."""
    transport, client = scripted_client(
        responses_body(
            [function_call_item("read_records", "{}")], model=GPT_5_6_LUNA_MODEL
        )
    )
    adapter = OpenAIResponsesAdapter(model=GPT_5_6_LUNA_MODEL, client=client)

    adapter.next_call(_turn_request(), _deadline())

    body = transport.bodies[0]
    settings = adapter.settings
    assert sorted(body) == list(settings["request_fields_sent"])
    assert settings["reasoning"] == body["reasoning"]
    assert settings["temperature"] is None
    for field in settings["request_fields_omitted"]:
        assert field not in body


def test_the_recorded_luna_settings_are_exactly_this_build_s() -> None:
    """The stored schema, pinned whole rather than spot-checked."""
    assert openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL) == {
        "action_surface": "cube_allowed_actions_exact_v1",
        "api": "responses",
        "base_url": OPENAI_BASE_URL,
        "fact_affordance": "cube_fact_key_enum_v1",
        "max_output_tokens": 1024,
        "model_contract_verified": True,
        "parallel_tool_calls": False,
        "query_resolution": "cube_query_resolution_registry_v1",
        "reasoning": {"effort": "none"},
        "request_fields_omitted": ["temperature", "top_p"],
        "request_fields_sent": [
            "input",
            "instructions",
            "max_output_tokens",
            "model",
            "parallel_tool_calls",
            "reasoning",
            "store",
            "tool_choice",
            "tools",
        ],
        "request_mapping": REQUEST_MAPPING_VERSION,
        "request_profile": LUNA_PROFILE_ID,
        "response_capture": OPENAI_RESPONSE_CAPTURE,
        "response_contract": "exactly_one_tool_call",
        # The named response-extension contract this lane accepts, and its
        # digest. Neither is a request setting — what this build *sends* is
        # unchanged — but both are result-affecting all the same: they decide
        # which bodies a run will read as answers at all.
        "response_server_extensions": OPENAI_RESPONSE_SERVER_EXTENSIONS,
        "response_server_extensions_digest": SERVER_EXTENSION_DIGEST,
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.5,
            "backoff_multiplier": 2.0,
        },
        "sdk": "openai",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_version": openai.__version__,
        "store": False,
        "temperature": None,
        "tool_choice": "required",
    }


# -- run identity moved with the request --------------------------------------


def test_the_openai_adapter_version_moved_with_the_request_semantics() -> None:
    """A changed request is a changed integration, and says so in its version.

    ``0.5.0`` rather than the ``0.3.0`` this profile arrived at: what the lane
    *accepts* moved twice afterwards, for reasons that have nothing to do with
    this profile — see ``tests/test_openai_response_extensions.py``. The
    profile's own claim is the one below it: the identity a run records is this
    integration's, and it is not the superseded one.
    """
    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert openai_identity(GPT_5_6_LUNA_MODEL).version == OPENAI_ADAPTER_VERSION


#: The request this build used to send to this model: a pinned sampling
#: parameter and no reasoning field. Synthesised here so the identity comparison
#: below is against a real settings shape rather than an edited copy of the
#: current one.
SUPERSEDED_SETTINGS: dict[str, Any] = {
    "action_surface": "cube_allowed_actions_exact_v1",
    "api": "responses",
    "base_url": OPENAI_BASE_URL,
    "fact_affordance": "cube_fact_key_enum_v1",
    "max_output_tokens": 1024,
    "model_contract_verified": False,
    "parallel_tool_calls": False,
    "query_resolution": "cube_query_resolution_registry_v1",
    "reasoning": None,
    "request_fields_omitted": ["reasoning", "top_p"],
    "request_fields_sent": [
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "store",
        "temperature",
        "tool_choice",
        "tools",
    ],
    "request_mapping": REQUEST_MAPPING_VERSION,
    "request_profile": "gpt56luna_temperature_zero_reasoning_omitted_v1",
    "response_capture": OPENAI_RESPONSE_CAPTURE,
    "response_contract": "exactly_one_tool_call",
    "retry": {
        "max_attempts": 3,
        "initial_backoff_seconds": 0.5,
        "backoff_multiplier": 2.0,
    },
    "sdk": "openai",
    "sdk_max_retries": SDK_MAX_RETRIES,
    "sdk_version": openai.__version__,
    "store": False,
    "temperature": 0.0,
    "tool_choice": "required",
}


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


def test_the_configuration_identity_is_not_the_superseded_one() -> None:
    """Two runs that asked the provider for different things are two configurations.

    Both halves of the change are enough on their own — the settings carry the
    profile and the manifest carries the adapter version — so the assertion is
    made against the superseded settings *and* against the superseded version.
    """
    current = _manifest(
        openai_settings(ProviderRetryPolicy(), model=GPT_5_6_LUNA_MODEL),
        version=OPENAI_ADAPTER_VERSION,
    )
    superseded = _manifest(SUPERSEDED_SETTINGS, version="0.2.0")
    version_only = _manifest(SUPERSEDED_SETTINGS, version=OPENAI_ADAPTER_VERSION)

    assert current.configuration_id != superseded.configuration_id
    assert current.configuration_id != version_only.configuration_id
    assert superseded.configuration_id != version_only.configuration_id


def test_a_body_the_superseded_profile_would_have_sent_is_not_dispatched() -> None:
    """The settings are enforced against the payload, not merely written beside it.

    The mapping this build serialises is replaced with the one the superseded
    profile produced — sampling parameter present, reasoning absent. The run's
    recorded settings would then describe a request that was not made, so the
    request is not made.
    """
    from boundarybench.adapter import AdapterError
    from boundarybench.providers.openai_responses import build_responses_request

    transport = RecordingTransport([])
    adapter = OpenAIResponsesAdapter(model=GPT_5_6_LUNA_MODEL, client=transport.client())
    payload = dict(build_responses_request(_turn_request(), model=GPT_5_6_LUNA_MODEL))
    payload.pop("reasoning")
    payload["temperature"] = 0.0

    with pytest.raises(AdapterError) as raised:
        adapter._send_payload(payload, _deadline())

    assert transport.calls == 0
    assert "temperature" in str(raised.value)


# -- what a plan against this model now states --------------------------------


def test_the_plan_states_reasoning_is_off_as_a_fact_for_this_model() -> None:
    """A verified "no reasoning" instruction is a fact, not an open question.

    This model's plan used to report ``optional_extended_thinking: null`` beside
    a note saying this build sent no reasoning field and had verified nothing.
    It now sends one, and the note's own words would be false about the request
    the plan describes — so the plan states the answer instead.
    """
    from boundarybench.cli import _preflight_scope
    from boundarybench.providers.registry import provider_choice

    choice = provider_choice("openai")
    scope = _preflight_scope(choice, choice.settings(GPT_5_6_LUNA_MODEL), _scaffold())

    assert scope["optional_extended_thinking"] is False
    assert scope["reasoning"] == {"effort": "none"}
    assert scope["sampling_parameters_sent"] == {}
    assert "temperature" in scope["sampling_parameters_omitted"]
    assert "optional_extended_thinking_note" not in scope


@pytest.mark.parametrize(
    ("adapter", "model"),
    (
        ("openai", "gpt-test-20990101"),
        ("xai", "grok-4.5"),
        ("mistral", "mistral-small-2603"),
    ),
)
def test_a_lane_that_still_sends_no_reasoning_field_still_says_so(
    adapter: str, model: str
) -> None:
    """The other requests are unverified in exactly the way they were.

    Including the OpenAI baseline profile: what changed is one model's contract,
    not this build's willingness to claim one it has not read.
    """
    from boundarybench.cli import REASONING_DEFAULT_UNVERIFIED, _preflight_scope
    from boundarybench.providers.registry import provider_choice

    choice = provider_choice(adapter)
    scope = _preflight_scope(choice, choice.settings(model), _scaffold())

    assert scope["optional_extended_thinking"] is None
    assert scope["optional_extended_thinking_note"] == REASONING_DEFAULT_UNVERIFIED


# -- the other three lanes are untouched --------------------------------------


def test_the_other_providers_request_shapes_did_not_move() -> None:
    """One lane changed. A matrix where a second one moved silently is not one.

    The xAI and Mistral lanes' *versions* have since moved to ``0.3.0``, and this
    test still holds: each moved for what that lane now **accepts** — the
    compatibility endpoint's own response extensions, and the
    ``prompt_tokens_details`` object Mistral states on its usage block — and the
    profiles asserted below are what they **send**, which is unchanged. The two
    are asserted together here rather than one standing in for the other.
    """
    assert XAI_ADAPTER_VERSION == "0.3.0"
    assert MISTRAL_ADAPTER_VERSION == "0.3.0"
    assert ANTHROPIC_ADAPTER_VERSION == "0.8.0"

    xai = xai_profile_for(GROK_4_5_MODEL)
    assert xai.profile_id == "grok45_temperature_zero_reasoning_omitted_v1"
    assert xai.temperature == 0.0
    assert xai.reasoning_effort is None
    assert xai.contract_verified is False

    mistral = mistral_profile_for(MISTRAL_SMALL_MODEL)
    assert mistral.profile_id == "mistralsmall2603_temperature_zero_v1"
    assert mistral.temperature == 0.0
    assert mistral.contract_verified is False

    assert anthropic_profile_for(SONNET_5_MODEL).thinking == {"type": "disabled"}
    assert anthropic_profile_for(HAIKU_4_5_MODEL).temperature == 0.0


def test_the_baseline_openai_profile_still_covers_an_unencoded_model() -> None:
    """The amendment is this model's, not a change to what every model is sent.

    A model this build ships no profile for still has to be sent something, and
    what it is sent is unchanged: this repository read no contract for it, so it
    can claim none.
    """
    baseline = request_profile_for("gpt-test-20990101")

    assert baseline.profile_id == "openai_baseline_temperature_zero_reasoning_omitted_v1"
    assert baseline.temperature == 0.0
    assert baseline.reasoning is None
    assert baseline.contract_verified is False


def test_the_profile_is_immutable_and_so_is_its_reasoning_object() -> None:
    """A shared mutable default would let one turn rewrite every later one's request."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        GPT_5_6_LUNA_PROFILE.temperature = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        GPT_5_6_LUNA_PROFILE.reasoning["effort"] = "max"  # type: ignore[index]
