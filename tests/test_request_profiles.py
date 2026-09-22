"""Model-specific immutable request profiles.

Request shape is a property of *this build and the model it is pinned to*.
``temperature: 0.0`` with no ``thinking`` field is not a neutral shape:
Anthropic's Sonnet 5 documentation says adaptive thinking is on by default when
``thinking`` is omitted, non-default sampling parameters are rejected with HTTP
400, and disabling thinking requires ``thinking: {"type": "disabled"}``.
This build therefore freezes one profile per model. These tests hold that:

* what the real SDK serialises for each model, key by key and value by value —
  not what a helper returns;
* that the fields each profile prohibits are absent from the wire body;
* that ``adapter.settings`` — which is hashed into ``configuration_id`` — is an
  exact description of that body rather than a nearby one;
* that a profile which is not the pinned model's is refused before a client, a
  request or a run directory exists;
* that the per-model configuration identities differ from a single-shape one
  and from each other;
* that a single-shape configuration cannot be resumed or read under this
  build.

Every request here is answered by ``httpx.MockTransport`` inside this process.
No socket is opened, no credential is read, and nothing here is evidence about
any model: it is evidence about the integration.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import anthropic
import pytest

from boundarybench.adapter import (
    AdapterError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.compiler import Variant, compile_cube
from boundarybench.environment import Environment
from boundarybench.ledger import LedgerError, read_ledger
from boundarybench.pricing import price_for
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    ANTHROPIC_IMPLEMENTATION,
    ANTHROPIC_PROVIDER,
    DISABLED_THINKING,
    HAIKU_4_5_MODEL,
    HAIKU_4_5_PROFILE,
    MAX_OUTPUT_TOKENS,
    PROHIBITED_REQUEST_FIELDS,
    REQUEST_MAPPING_VERSION,
    SONNET_5_MODEL,
    SONNET_5_PROFILE,
    AnthropicConfigurationError,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
    anthropic_settings,
    build_messages_request,
    request_input_token_bound,
    request_profile_for,
)
from boundarybench.runmanifest import (
    RunLimits,
    RunManifestMismatchError,
    build_run_manifest,
    open_run_session,
)
from boundarybench.runner import compiled_variants, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from boundarybench.schema import ConstructCard
from boundarybench.suite import validate_suite
from tests.anthropic_transport import (
    RecordingTransport,
    message_body,
    one_call_client,
    scripted_client,
    tool_use_block,
)
from tests.conftest import SUITE_MANIFEST, minimal_card

#: The only two models this build holds a model-specific profile for.
MODELS = (SONNET_5_MODEL, HAIKU_4_5_MODEL)

STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    return compile_cube(ConstructCard.from_dict(minimal_card())).variants[0]


def _turn_request() -> TurnRequest:
    environment = Environment(_variant())
    environment.read_records()
    return build_turn_request(
        scaffold=_scaffold(), environment=environment, turns_remaining=6
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _wire_body(model: str) -> dict[str, Any]:
    """The body the real SDK put on the wire for one turn of this model."""
    transport, client = one_call_client(
        "complete_case",
        STOP_CALL["input"],
        model=model,
    )
    adapter = AnthropicMessagesAdapter(model=model, client=client)

    adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 1
    return transport.bodies[0]


# -- what each model's request actually is -----------------------------------


def test_the_sonnet_request_omits_sampling_and_disables_thinking() -> None:
    """The profile, asserted against the serialised body.

    Not against ``build_messages_request``'s return value: the SDK decides what
    of a mapping reaches the wire, so a helper agreeing with itself proves
    nothing about the request a provider receives.
    """
    body = _wire_body(SONNET_5_MODEL)

    assert body["thinking"] == {"type": "disabled"}
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert sorted(body) == [
        "max_tokens",
        "messages",
        "model",
        "system",
        "thinking",
        "tool_choice",
        "tools",
    ]


def test_the_haiku_request_keeps_temperature_and_omits_thinking() -> None:
    """The shipped, tested Haiku-specific profile keeps its declared fields."""
    body = _wire_body(HAIKU_4_5_MODEL)

    assert body["temperature"] == 0.0
    assert "thinking" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert sorted(body) == [
        "max_tokens",
        "messages",
        "model",
        "system",
        "temperature",
        "tool_choice",
        "tools",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_the_two_profiles_agree_on_everything_that_is_not_the_profile(
    model: str,
) -> None:
    """One scaffold, one mapping, two sampling/thinking profiles.

    A profile that also moved the output ceiling, the tool contract or the
    message projection would be a second experiment smuggled in beside the
    profile change.
    """
    body = _wire_body(model)

    assert body["model"] == model
    assert body["max_tokens"] == 1024
    assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert [message["role"] for message in body["messages"]] == ["user"]
    assert [tool["name"] for tool in body["tools"]] == [
        action.name for action in _turn_request().actions
    ]


@pytest.mark.parametrize("model", MODELS)
def test_no_profile_ever_sends_a_prohibited_sampling_field(model: str) -> None:
    """``top_p`` and ``top_k`` are sent by no profile this build ships.

    Sonnet rejects them; Haiku does not need them. Neither is reachable through
    a setting, so the assertion is that they are absent from the body and from
    every profile's own description of itself.
    """
    body = _wire_body(model)
    profile = request_profile_for(model)

    for field in ("top_p", "top_k"):
        assert field not in body
        assert field in profile.omitted_fields()
    assert set(PROHIBITED_REQUEST_FIELDS) == {"temperature", "thinking", "top_k", "top_p"}


# -- settings describe the request, exactly ----------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_settings_describe_exactly_what_the_transport_saw(model: str) -> None:
    """``adapter.settings`` is hashed into run identity, so it is the claim.

    The settings must exactly describe the serialised model-specific request,
    including sampling and thinking fields.
    """
    transport, client = one_call_client("complete_case", STOP_CALL["input"], model=model)
    adapter = AnthropicMessagesAdapter(model=model, client=client)

    adapter.next_call(_turn_request(), _deadline())

    body = transport.bodies[0]
    settings = adapter.settings
    assert sorted(body) == list(settings["request_fields_sent"])
    assert settings["temperature"] == body.get("temperature")
    assert settings["thinking"] == body.get("thinking")
    for field in settings["request_fields_omitted"]:
        assert field not in body


@pytest.mark.parametrize("model", MODELS)
def test_the_adapter_reports_the_settings_the_module_states_for_its_model(
    model: str,
) -> None:
    _transport, client = one_call_client("complete_case", STOP_CALL["input"], model=model)
    adapter = AnthropicMessagesAdapter(model=model, client=client)

    assert dict(adapter.settings) == anthropic_settings(
        AnthropicRetryPolicy(), model=model
    )


def test_the_recorded_sonnet_settings_are_exactly_this_build_s() -> None:
    """The stored schema, pinned whole rather than spot-checked."""
    assert anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL) == {
        "action_surface": "cube_allowed_actions_exact_v1",
        "api": "messages",
        "base_url": "https://api.anthropic.com",
        "fact_affordance": "cube_fact_key_enum_v1",
        "max_output_tokens": 1024,
        "query_resolution": "cube_query_resolution_registry_v1",
        "request_fields_omitted": ["temperature", "top_k", "top_p"],
        "request_fields_sent": [
            "max_tokens",
            "messages",
            "model",
            "system",
            "thinking",
            "tool_choice",
            "tools",
        ],
        "request_mapping": REQUEST_MAPPING_VERSION,
        "request_profile": SONNET_5_PROFILE.profile_id,
        "response_contract": "exactly_one_tool_call",
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.5,
            "backoff_multiplier": 2.0,
        },
        "sdk": "anthropic",
        "sdk_max_retries": 0,
        "sdk_version": anthropic.__version__,
        "temperature": None,
        "thinking": {"type": "disabled"},
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    }


def test_the_recorded_haiku_settings_are_exactly_this_build_s() -> None:
    assert anthropic_settings(AnthropicRetryPolicy(), model=HAIKU_4_5_MODEL) == {
        "action_surface": "cube_allowed_actions_exact_v1",
        "api": "messages",
        "base_url": "https://api.anthropic.com",
        "fact_affordance": "cube_fact_key_enum_v1",
        "max_output_tokens": 1024,
        "query_resolution": "cube_query_resolution_registry_v1",
        "request_fields_omitted": ["thinking", "top_k", "top_p"],
        "request_fields_sent": [
            "max_tokens",
            "messages",
            "model",
            "system",
            "temperature",
            "tool_choice",
            "tools",
        ],
        "request_mapping": REQUEST_MAPPING_VERSION,
        "request_profile": HAIKU_4_5_PROFILE.profile_id,
        "response_contract": "exactly_one_tool_call",
        "retry": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.5,
            "backoff_multiplier": 2.0,
        },
        "sdk": "anthropic",
        "sdk_max_retries": 0,
        "sdk_version": anthropic.__version__,
        "temperature": 0.0,
        "thinking": None,
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    }


# -- a profile that is not the model's -----------------------------------------


def test_a_profile_that_is_not_the_models_is_refused_before_any_request() -> None:
    """A model/profile mismatch is refused at construction before any request."""
    transport, client = one_call_client("complete_case", STOP_CALL["input"])

    with pytest.raises(AnthropicConfigurationError) as raised:
        AnthropicMessagesAdapter(
            model=SONNET_5_MODEL, client=client, profile=HAIKU_4_5_PROFILE
        )

    assert transport.calls == 0
    assert HAIKU_4_5_PROFILE.profile_id in str(raised.value)
    assert SONNET_5_MODEL in str(raised.value)


def test_a_mismatched_profile_is_refused_when_a_request_is_built() -> None:
    request = _turn_request()

    with pytest.raises(AnthropicConfigurationError):
        build_messages_request(request, model=HAIKU_4_5_MODEL, profile=SONNET_5_PROFILE)


def test_a_payload_that_disagrees_with_the_settings_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings are enforced against the payload, not merely written beside it.

    The mapping this build serialises is replaced with one carrying the field
    Sonnet rejects. The run's recorded settings would then describe a request
    that was not made, so the request is not made: the refusal happens before the
    client is reached and before any output exists.
    """
    from boundarybench.providers import anthropic_messages

    def sampled(request: TurnRequest, **kwargs: Any) -> dict[str, Any]:
        payload = build_messages_request(request, model=SONNET_5_MODEL)
        payload["temperature"] = 0.0
        return payload

    monkeypatch.setattr(anthropic_messages, "build_messages_request", sampled)
    transport, client = one_call_client("complete_case", STOP_CALL["input"])
    adapter = AnthropicMessagesAdapter(model=SONNET_5_MODEL, client=client)

    with pytest.raises(AdapterError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 0
    assert "temperature" in str(raised.value)


def test_a_payload_missing_the_disabled_thinking_field_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction: the field this model's profile requires, dropped."""
    from boundarybench.providers import anthropic_messages

    def thinkless(request: TurnRequest, **kwargs: Any) -> dict[str, Any]:
        payload = build_messages_request(request, model=SONNET_5_MODEL)
        payload.pop("thinking")
        return payload

    monkeypatch.setattr(anthropic_messages, "build_messages_request", thinkless)
    transport, client = one_call_client("complete_case", STOP_CALL["input"])
    adapter = AnthropicMessagesAdapter(model=SONNET_5_MODEL, client=client)

    with pytest.raises(AdapterError):
        adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 0


@pytest.mark.parametrize(
    ("model", "field", "value"),
    (
        (SONNET_5_MODEL, "thinking", {"type": "enabled", "budget_tokens": 1024}),
        (HAIKU_4_5_MODEL, "temperature", 0.7),
    ),
)
def test_a_payload_whose_value_is_not_the_recorded_one_is_never_sent(
    model: str, field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same field, a different value: still a request the settings do not describe.

    A field-set check alone would pass this. Adaptive thinking switched on under
    a profile recorded as disabled, or a sampling temperature that is not the
    pinned one, changes what the run measured while leaving the manifest saying
    otherwise — so the value is checked, not only the key.
    """
    from boundarybench.providers import anthropic_messages

    def altered(request: TurnRequest, **kwargs: Any) -> dict[str, Any]:
        payload = build_messages_request(request, model=model)
        payload[field] = value
        return payload

    monkeypatch.setattr(anthropic_messages, "build_messages_request", altered)
    transport, client = one_call_client("complete_case", STOP_CALL["input"], model=model)
    adapter = AnthropicMessagesAdapter(model=model, client=client)

    with pytest.raises(AdapterError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert transport.calls == 0
    assert field in str(raised.value)


@pytest.mark.parametrize("model", MODELS)
def test_the_adapter_states_which_profile_it_is_running(model: str) -> None:
    """The runner and a test can both ask; neither has to infer it from a body."""
    _transport, client = one_call_client("complete_case", STOP_CALL["input"], model=model)

    adapter = AnthropicMessagesAdapter(model=model, client=client)

    assert adapter.request_profile == request_profile_for(model)
    assert adapter.settings["request_profile"] == adapter.request_profile.profile_id


# -- run identity ------------------------------------------------------------


def _manifest(model: str, settings: Any, *, version: str = ANTHROPIC_ADAPTER_VERSION):
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider=ANTHROPIC_PROVIDER,
        model=model,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=version,
        adapter_settings=settings,
        trials=1,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
    )


#: Adversarial single-shape settings: one temperature for every model, no
#: thinking field, and request mapping v1. A run configured this way is a
#: different run that this build refuses to continue. Everything these tests
#: touch is synthesised inside ``tmp_path``.
SINGLE_SHAPE_SETTINGS: dict[str, Any] = {
    "api": "messages",
    "base_url": "https://api.anthropic.com",
    "max_output_tokens": 1024,
    "request_mapping": "turn_request_json_v1",
    "response_contract": "exactly_one_tool_call",
    "retry": {
        "max_attempts": 3,
        "initial_backoff_seconds": 0.5,
        "backoff_multiplier": 2.0,
    },
    "sdk": "anthropic",
    "sdk_max_retries": 0,
    "sdk_version": anthropic.__version__,
    "temperature": 0.0,
    "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
}


def test_the_profiled_sonnet_configuration_is_not_the_single_shape_one() -> None:
    """A changed request is a changed experiment, and says so in its identity."""
    rejected = _manifest(SONNET_5_MODEL, SINGLE_SHAPE_SETTINGS, version="0.5.0")
    amended = _manifest(
        SONNET_5_MODEL, anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL)
    )

    assert amended.configuration_id != rejected.configuration_id


def test_the_two_models_differ_by_profile_and_not_only_by_name() -> None:
    """Two runs whose only recorded difference was the model name would be a lie.

    They send different bodies, so their settings differ before the model field
    is even reached.
    """
    sonnet = anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL)
    haiku = anthropic_settings(AnthropicRetryPolicy(), model=HAIKU_4_5_MODEL)

    assert sonnet["request_profile"] != haiku["request_profile"]
    assert sonnet["temperature"] != haiku["temperature"]
    assert sonnet["thinking"] != haiku["thinking"]
    assert sonnet["request_fields_sent"] != haiku["request_fields_sent"]
    assert (
        _manifest(SONNET_5_MODEL, sonnet).configuration_id
        != _manifest(HAIKU_4_5_MODEL, haiku).configuration_id
    )


def test_a_run_configured_with_a_single_shape_cannot_be_resumed(
    tmp_path: Path,
) -> None:
    """That configuration, synthesised here, is refused a resume.

    Synthetic on purpose and local on purpose: both manifests are built in this
    process and written under ``tmp_path``. What is under test is the rule, and
    the rule is a property of the configuration rather than of any particular
    directory.
    """
    root = tmp_path / "synthetic-single-shape"
    rejected = _manifest(SONNET_5_MODEL, SINGLE_SHAPE_SETTINGS, version="0.5.0")
    with open_run_session(root, rejected) as session:
        assert session.resumed is False

    amended = _manifest(
        SONNET_5_MODEL, anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL)
    )
    with (
        pytest.raises(RunManifestMismatchError) as raised,
        open_run_session(root, amended),
    ):
        pass

    assert rejected.configuration_id in str(raised.value)
    assert amended.configuration_id in str(raised.value)


def test_a_capped_ledger_pinning_the_rejected_profile_is_not_read(
    tmp_path: Path,
) -> None:
    """A row this build cannot rebuild the request of is not read as evidence.

    The rows are real ones, written by this build through the real adapter. The
    manifest they are then read under is the synthetic single-shape one, so what
    the reader is asked to do is exactly what resuming such a run would ask:
    rederive reservations from a request profile this build does not send.
    """
    root = tmp_path / "run"
    manifest = _capped_manifest()
    guard = RunCostGuard(
        controls=manifest.cost_controls,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
    )
    transport = RecordingTransport(
        [], default=message_body([STOP_CALL], model=SONNET_5_MODEL)
    )
    with open_run_session(root, manifest) as session:
        execute_run(
            suite=validate_suite(SUITE_MANIFEST),
            scaffold=_scaffold(),
            adapter=AnthropicMessagesAdapter(
                model=SONNET_5_MODEL,
                client=transport.client(),
                cost_guard=guard,
            ),
            session=session,
            cost_guard=guard,
            stop_after=1,
        )

    variants = compiled_variants(validate_suite(SUITE_MANIFEST))
    ledger = root / "episodes.jsonl"
    assert read_ledger(ledger, manifest, variants, scaffold=_scaffold())

    stale = _rehashed(manifest, SINGLE_SHAPE_SETTINGS)
    with pytest.raises(LedgerError) as raised:
        read_ledger(ledger, stale, variants, scaffold=_scaffold())

    assert "request" in str(raised.value)


def _capped_manifest() -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider=ANTHROPIC_PROVIDER,
        model=SONNET_5_MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL),
        trials=1,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
        cost_controls=CostControls(
            max_cost_usd=Decimal("66"),
            max_episodes=36,
            price=price_for(provider="anthropic", model=SONNET_5_MODEL),
        ),
    )


def _rehashed(manifest: Any, settings: Any) -> Any:
    """The same run with the single-shape settings, self-consistently hashed.

    A manifest whose recorded ``configuration_id`` did not follow from its own
    payload is refused long before the request profile is looked at, and that is
    a different refusal from the one under test.
    """
    from boundarybench.runmanifest import (
        configuration_digest,
        execution_digest_of,
        normalized_adapter_settings,
    )

    stale = dataclasses.replace(
        manifest,
        adapter_settings=normalized_adapter_settings(settings),
        configuration_id="",
    )
    identity = configuration_digest(stale.configuration_payload())
    return dataclasses.replace(
        stale,
        configuration_id=identity,
        execution_id=execution_digest_of(
            configuration_id=identity,
            execution_id=manifest.execution_id,
            created_at_utc=manifest.created_at_utc,
        ),
    )


# -- the cost bound follows the model-specific payload -----------------------


@pytest.mark.parametrize("model", MODELS)
def test_the_request_bound_is_derived_from_that_model_s_own_payload(
    model: str,
) -> None:
    """What a request may cost is bounded by the request that is actually sent.

    Not by a shared template. Sonnet's body carries a thinking field Haiku's
    does not and omits a temperature Haiku sends, so the two bodies have
    different lengths and the guard authorises different amounts for them.
    """
    from boundarybench.budget import conservative_input_token_bound
    from boundarybench.jsonsafe import canonical_json_text

    payload = build_messages_request(_turn_request(), model=model)
    encoded = canonical_json_text(payload, "bound").encode("utf-8")

    assert request_input_token_bound(payload) == conservative_input_token_bound(
        len(encoded)
    )


def test_the_two_models_bounds_are_not_the_same_number() -> None:
    request = _turn_request()

    sonnet = request_input_token_bound(
        build_messages_request(request, model=SONNET_5_MODEL)
    )
    haiku = request_input_token_bound(
        build_messages_request(request, model=HAIKU_4_5_MODEL)
    )

    assert sonnet != haiku


@pytest.mark.parametrize("model", MODELS)
def test_the_real_payload_stays_inside_the_planning_assumption(model: str) -> None:
    """The preflight's planning size is still an over-statement of both profiles.

    The preflight cannot serialise a request — it must not build one at all — so
    it prices the plan at a stated per-request planning size. That number is only
    honest while it is above what the profiles actually send, and per-model
    profiles changed what they send, so it is checked here rather than assumed.
    """
    from boundarybench.cli import DEFAULT_PLANNING_INPUT_TOKENS

    bound = request_input_token_bound(
        build_messages_request(_turn_request(), model=model)
    )

    assert bound < DEFAULT_PLANNING_INPUT_TOKENS


# -- the authorised plans still fit, offline ---------------------------------


def _preflight_argv(model: str, cap: str, output: Path) -> list[str]:
    return [
        "preflight",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        model,
        "--output-dir",
        str(output),
        "--max-cost-usd",
        cap,
        "--max-episodes",
        "36",
        "--trials",
        "3",
        "--max-turns",
        "6",
        "--json",
    ]


@pytest.mark.parametrize(
    ("model", "cap"), ((SONNET_5_MODEL, "66"), (HAIKU_4_5_MODEL, "34"))
)
def test_the_authorised_allocation_still_passes_offline(
    model: str,
    cap: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-model profiles changed the request, not what was authorised.

    36 episodes, six turns, USD 66 for Sonnet and USD 34 for Haiku — each plan's
    conservative upper bound is still inside its own allocation under the
    model-specific profile, and the preflight still contacts nothing.
    """
    from boundarybench.cli import main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)

    code = main(_preflight_argv(model, cap, root))

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cost_bound"]["fits_within_cap"] is True
    assert float(payload["cost_bound"]["upper_bound_usd"]) <= float(cap)
    assert payload["cost_bound"]["max_requests"] == 36 * 6 * 3
    # The bound is stated for a named profile, so a reader can tell which
    # request shape the plan was priced for.
    assert (
        payload["cost_bound"]["request_profile"] == request_profile_for(model).profile_id
    )
    assert payload["settings"]["request_profile"] == request_profile_for(model).profile_id


@pytest.mark.parametrize("model", MODELS)
def test_the_plan_report_states_the_amended_scope(
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled thinking is not optional extended thinking, and is not reported as it.

    A run may declare "no optional extended/adaptive thinking" out of scope.
    Sonnet carries an explicit ``thinking`` field whose whole content is that the
    feature is off, so a report that read the presence of the key as the feature
    being on would contradict the declared scope while the request obeyed it.
    """
    from boundarybench.cli import main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "plan"
    root.mkdir(mode=0o700)

    assert main(_preflight_argv(model, "66", root)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["scope"]["optional_extended_thinking"] is False
    assert payload["scope"]["thinking"] == payload["settings"]["thinking"]
    assert payload["scope"]["sampling_parameters_sent"] == (
        {} if model == SONNET_5_MODEL else {"temperature": 0.0}
    )


# -- an unencoded model still has a stated profile ---------------------------


def test_a_model_this_build_has_no_profile_for_gets_the_named_baseline() -> None:
    """A shape-checked model identifier is not an allowlisted one.

    ``check_model`` deliberately accepts any well-formed identifier, so a model
    this build ships no documented profile for still has to send *something*.
    What it sends is the baseline profile, and its identity says that is what it
    is, so a run under it cannot be mistaken for one under a profile that was
    checked against a model's published contract.
    """
    profile = request_profile_for("claude-test-20990101")

    assert profile.profile_id != SONNET_5_PROFILE.profile_id
    assert profile.profile_id != HAIKU_4_5_PROFILE.profile_id
    assert profile.temperature == 0.0
    assert profile.thinking is None


def test_the_thinking_block_this_build_sends_is_the_documented_disabled_one() -> None:
    assert DISABLED_THINKING == {"type": "disabled"}
    assert SONNET_5_PROFILE.thinking == DISABLED_THINKING


def test_a_scripted_provider_error_still_classifies_under_the_new_profile() -> None:
    """The profile moved the request, not the fault taxonomy.

    A 400 under the model-specific Sonnet profile is still one episode's request
    rejection, recorded without the provider's text.
    """
    from boundarybench.adapter import (
        PROVIDER_FAULT_REQUEST_REJECTED,
        AdapterProviderError,
    )
    from tests.anthropic_transport import error_body

    transport, client = scripted_client((400, error_body("invalid_request_error", "no")))
    adapter = AnthropicMessagesAdapter(model=SONNET_5_MODEL, client=client)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(_turn_request(), _deadline())

    assert raised.value.fault == PROVIDER_FAULT_REQUEST_REJECTED
    assert transport.calls == 1
    assert "no" not in str(raised.value).split("HTTP")[0]
