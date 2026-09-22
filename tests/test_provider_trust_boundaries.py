"""Provider-facing trust boundaries covered by focused regressions.

Every case here is a guarantee the three provider/run-identity modules already
make and that no other test drove, so a mutation could change the behaviour and
the suite would still pass: the credential the client is actually built with,
the request settings that go into run identity, the tool schema the model is
shown, the fault taxonomy for statuses no test sent, the per-attempt and
per-turn measurements, the exact point a backoff stops being affordable, and
the timezone a run's timestamp is rendered in.

Nothing here contacts a provider or needs a credential: the Anthropic paths run
the real SDK over the in-process transport in ``tests.anthropic_transport``, and
the clock is injected so a measurement is an assertion rather than an artefact.

Prose is deliberately not asserted. Where a sentence is checked at all it is
checked for what it must *not* carry — a model identifier, a credential — which
is the redaction guarantee and not the wording.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from boundarybench.adapter import (
    AdapterProviderError,
    AdapterUsage,
    ObservedFact,
    ProviderTelemetry,
    TranscriptEntry,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
    policy_following_script,
)
from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.jsonsafe import JsonSafetyError
from boundarybench.providers.anthropic_messages import (
    AUDITED_CONFIGURATIONS_VARIABLE,
    LIVE_CONFIGURATION_VARIABLE,
    AnthropicConfigurationError,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
    anthropic_settings,
    build_anthropic_adapter,
    build_client,
    classify_exception,
    tool_schema,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import ConstructCard
from tests.anthropic_transport import (
    error_body,
    message_body,
    scripted_client,
    tool_use_block,
)
from tests.conftest import minimal_card

#: The model this file pins. Restated rather than imported: a test that read the
#: constant would agree with any value it was moved to.
PINNED_MODEL = "claude-test-20990101"

#: Not a credential; the transport never leaves the process.
FAKE_KEY = "test-key-not-a-credential"


def _scaffold() -> Any:
    return load_scaffold(STANDARD_SCAFFOLD)


def _turn_request() -> TurnRequest:
    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return build_turn_request(
        scaffold=_scaffold(),
        environment=Environment(cube.variants[0]),
        turns_remaining=12,
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _clock(start: float = 100.0, step: float = 0.25) -> Any:
    """A clock whose origin is *not* zero.

    The origin matters. Every duration this adapter records is a difference of
    two readings, and a clock that started at zero would render ``a - b`` and
    ``a + b`` identically on the first turn — so a sign error in a measurement
    would be invisible exactly where it is easiest to make.
    """
    now = [start]

    def read() -> float:
        value = now[0]
        now[0] += step
        return value

    return read


def _adapter(client: Any, **kwargs: Any) -> AnthropicMessagesAdapter:
    kwargs.setdefault("sleep", lambda seconds: None)
    kwargs.setdefault("clock", _clock())
    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def _status_error(code: int) -> anthropic.APIStatusError:
    """The SDK's own typed error for one HTTP status, built the SDK's way."""
    response = httpx.Response(
        code, json=error_body(), request=httpx.Request("POST", "https://provider.invalid")
    )
    return anthropic.APIStatusError("provider said no", response=response, body=None)


#: A clearly synthetic, purely local configuration identity. It has the shape a
#: ``configuration_id`` has and is a value none ever is: no suite, scaffold or
#: settings hash to it. It exists so a test can authorise *something* without
#: this tree carrying the identity of a run that happened.
SYNTHETIC_CONFIGURATION = "0" * 64
SYNTHETIC_AUTHORISATION = frozenset({SYNTHETIC_CONFIGURATION})


# -- the audited-configuration gate, before any transport --------------------


def test_the_published_build_authorises_no_configuration() -> None:
    """Default-deny, asserted on the shipped constant.

    A published build that named a configuration identity would be publishing a
    run: the value is a hash of an exact suite, scaffold, model, settings and
    limits, so shipping one asserts that experiment was executed against a paid
    endpoint. A seeded allowlist would also grant authority not established for
    the current tree. The set is empty, and this assertion keeps it empty.
    """
    from boundarybench.providers.anthropic_messages import AUDITED_CONFIGURATIONS

    assert frozenset() == AUDITED_CONFIGURATIONS


def test_no_configuration_is_authorised_by_an_empty_environment() -> None:
    """Nothing built in, nothing declared: nothing authorised."""
    from boundarybench.providers.anthropic_messages import audited_configurations

    assert audited_configurations({}) == frozenset()


def test_an_unauthorised_configuration_is_refused_before_the_credential() -> None:
    """The order is the property under test, not merely the outcome.

    The credential is present and valid-shaped. The refusal happens anyway, and
    happens *first*: no client is constructed, so nothing exists that could send
    a request, and the message names the authorisation rather than the key.
    """
    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=PINNED_MODEL,
            environ={"ANTHROPIC_API_KEY": FAKE_KEY},
        )

    assert AUDITED_CONFIGURATIONS_VARIABLE in str(raised.value)
    assert FAKE_KEY not in str(raised.value)
    assert "ANTHROPIC_API_KEY" not in str(raised.value)


def test_a_configuration_outside_the_authorised_set_is_refused() -> None:
    """An authorisation for one configuration authorises no other.

    Both identities here are synthetic and local. The refusal quotes the one
    that was asked for and never the approved set, which would turn an error
    into a menu.
    """
    other = "1" * 64

    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=PINNED_MODEL,
            configuration_id=other,
            audited=SYNTHETIC_AUTHORISATION,
            environ={"ANTHROPIC_API_KEY": FAKE_KEY},
        )

    assert other in str(raised.value)
    assert SYNTHETIC_CONFIGURATION not in str(raised.value)


def test_an_authorised_set_with_no_named_configuration_is_refused() -> None:
    """An allowlist alone authorises nothing: the invocation has to name one."""
    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=PINNED_MODEL,
            audited=SYNTHETIC_AUTHORISATION,
            environ={"ANTHROPIC_API_KEY": FAKE_KEY},
        )

    assert LIVE_CONFIGURATION_VARIABLE in str(raised.value)


def test_a_configuration_identity_of_the_wrong_shape_is_refused() -> None:
    """A typo is answered as a typo, not by sending a request."""
    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=PINNED_MODEL,
            configuration_id="not-a-digest",
            audited=SYNTHETIC_AUTHORISATION | {"not-a-digest"},
            environ={"ANTHROPIC_API_KEY": FAKE_KEY},
        )

    assert "configuration_id" in str(raised.value)


def test_the_operator_declaration_is_read_from_the_environment() -> None:
    """The build's empty set unions with what the operator declared, and only that."""
    from boundarybench.providers.anthropic_messages import audited_configurations

    other = "1" * 64
    declared = audited_configurations(
        {AUDITED_CONFIGURATIONS_VARIABLE: f"{SYNTHETIC_CONFIGURATION}, {other}"}
    )

    assert declared == {SYNTHETIC_CONFIGURATION, other}


# -- the client the credential is sent through -------------------------------


def test_the_client_is_built_with_the_key_that_was_resolved() -> None:
    """The endpoint was already pinned; the credential was not.

    ``build_client`` is the single place the resolved key becomes a live client,
    and dropping the argument would not fail loudly: the SDK falls back to
    reading ``ANTHROPIC_API_KEY`` out of the ambient process environment. A run
    would then talk to the provider with whatever credential the shell happened
    to hold rather than the one this build resolved and checked, which is the
    difference between an operator's stated key and an inherited one.
    """
    client = build_client(api_key=FAKE_KEY)

    assert client.api_key == FAKE_KEY


def test_the_adapter_runs_the_retry_policy_the_caller_asked_for() -> None:
    """A policy that is not threaded through is a manifest that lies.

    The retry policy is inside adapter settings and so inside configuration
    identity. A builder that accepted a policy and then quietly ran the default
    would record one policy in the run manifest and execute another, and the
    resume check compares the recorded one.
    """
    policy = AnthropicRetryPolicy(
        max_attempts=7, initial_backoff_seconds=1.5, backoff_multiplier=3.0
    )

    adapter = build_anthropic_adapter(
        model=PINNED_MODEL,
        configuration_id=SYNTHETIC_CONFIGURATION,
        audited=SYNTHETIC_AUTHORISATION,
        environ={"ANTHROPIC_API_KEY": FAKE_KEY},
        retry=policy,
    )

    assert adapter.settings["retry"] == policy.as_dict()


# -- what run identity records about the request -----------------------------


def test_the_recorded_request_settings_are_exactly_this_build_s() -> None:
    """The settings mapping is a stored schema, not a debug dump.

    It is hashed into configuration identity and written into the run manifest,
    so both its keys and its values are read back by a resume check and by
    anything that compares two runs. Renaming a key or moving a value is a
    change to what "the same configuration" means, so the whole mapping is
    pinned here rather than spot-checked.

    Stated for one model because request shape is a property of the pinned model:
    this is the baseline profile used when this build ships no model-specific
    profile. The shipped, tested model-specific mappings are pinned in
    ``tests/test_request_profiles.py``.
    """
    settings = anthropic_settings(AnthropicRetryPolicy(), model=PINNED_MODEL)

    assert settings == {
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
        "request_mapping": "turn_request_json_v5",
        "request_profile": "baseline_temperature_zero_thinking_omitted_v1",
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


def test_a_tool_schema_carries_every_declared_parameter_and_its_description() -> None:
    """The tool schema is the whole of what the model is told an action is.

    A dropped description does not break the request — the API accepts a
    property with none — so it fails silently and changes the experiment: the
    agent is asked to fill in an argument whose meaning the scaffold declared
    and this build then withheld. The key names are the wire contract with the
    Messages API and are pinned for the same reason.
    """
    action = next(
        candidate
        for candidate in _scaffold().actions
        if any(parameter.description for parameter in candidate.parameters)
    )

    schema = tool_schema(action)
    properties = schema["input_schema"]["properties"]

    assert schema["name"] == action.name
    assert schema["description"] == action.description
    assert schema["input_schema"]["additionalProperties"] is False
    assert set(properties) == {parameter.name for parameter in action.parameters}
    for parameter in action.parameters:
        assert properties[parameter.name]["description"] == parameter.description
    assert schema["input_schema"]["required"] == [
        parameter.name for parameter in action.parameters if parameter.required
    ]


# -- the fault taxonomy ------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "fault", "retryable"),
    [
        (401, "provider_authentication", False),
        (403, "provider_authentication", False),
        (404, "provider_configuration", False),
        (408, "provider_timeout", True),
        (429, "provider_rate_limited", True),
        (500, "provider_server_error", True),
        (503, "provider_server_error", True),
        (400, "provider_request_rejected", False),
        (422, "provider_request_rejected", False),
    ],
)
def test_each_status_maps_to_one_fault_that_keeps_its_status(
    status: int, fault: str, retryable: bool
) -> None:
    """Name, retryability and status, for every branch of the classifier.

    The status is part of the answer and not decoration: it is the only
    provider-side number a durable failure row is allowed to carry, and it is
    what tells an operator reading ``provider_authentication`` whether the key
    was rejected (401) or the account lacks access (403). Retryability is
    asserted alongside because it decides whether the episode's remaining budget
    is spent repeating a request that cannot come out differently.
    """
    classified = classify_exception(_status_error(status))

    assert classified is not None
    assert (classified.fault, classified.retryable, classified.status) == (
        fault,
        retryable,
        status,
    )


def _request_object() -> httpx.Request:
    return httpx.Request("POST", "https://provider.invalid")


@pytest.mark.parametrize(
    ("exception", "fault", "retryable"),
    [
        (
            anthropic.APITimeoutError(request=_request_object()),
            "provider_timeout",
            True,
        ),
        (
            anthropic.APIConnectionError(request=_request_object()),
            "provider_network_error",
            True,
        ),
        (
            anthropic.APIResponseValidationError(
                response=httpx.Response(200, json={}, request=_request_object()),
                body=None,
            ),
            "provider_response_invalid",
            False,
        ),
    ],
)
def test_a_fault_with_no_http_status_is_still_named_and_still_says_if_it_retries(
    exception: Exception, fault: str, retryable: bool
) -> None:
    """Three faults that never got a status line, and each is a different answer.

    A timeout, a dropped connection and a body the SDK could not parse all
    arrive with no HTTP status, and ``None`` there is "no status" rather than
    zero. What separates them is the name and the retry decision: the first two
    can come out differently on the next attempt and the third cannot, so a
    misclassification either wastes the episode's budget or gives up on a
    transient outage.

    Order matters in the classifier and is exercised by the first two: an SDK
    timeout *is* an ``APIConnectionError``, so a check in the wrong order would
    file every timeout as a network error.
    """
    classified = classify_exception(exception)

    assert classified is not None
    assert (classified.fault, classified.retryable, classified.status) == (
        fault,
        retryable,
        None,
    )


def test_an_exception_this_integration_did_not_map_is_declined() -> None:
    """``None`` is "not the provider's doing", and is not a provider fault.

    The runner records what this returns nothing for as the adapter failure it
    is. Mapping a bug inside this integration onto an outage would put it in the
    provider-fault counts of a report, where it would read as evidence about a
    service that was working.
    """
    assert classify_exception(ValueError("a bug in this build")) is None


# -- the answer's provenance -------------------------------------------------


def test_a_model_mismatch_is_refused_with_detail_that_names_neither_model() -> None:
    """A refusal still has to be evidence: fixed detail, and no quoted name.

    Both halves are asserted. The detail has to exist and be text — it becomes a
    durable ``failure_event.detail`` — and it must not carry the model string
    the response stated, which is provider-controlled, nor repeat the one the
    run pinned, which is already on every row.
    """
    impostor = "claude-impostor-not-the-pinned-one"
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model=impostor)
    )
    adapter = _adapter(client)

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline())

    detail = caught.value.args[0]
    assert caught.value.fault == "provider_response_invalid"
    assert isinstance(detail, str) and detail
    assert impostor not in detail
    assert PINNED_MODEL not in detail


# -- what a turn measures ----------------------------------------------------


def test_a_measured_turn_reports_the_durations_the_clock_actually_showed() -> None:
    """Attempt latency, turn latency and usage latency are three readings.

    They are not interchangeable — the attempt's is one request, the turn's
    covers every attempt and the backoffs between them — and each is a
    difference between two stated instants, so each is asserted as the exact
    number the injected clock makes it rather than merely "not null".
    """
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    )
    adapter = _adapter(client)

    adapter.next_call(_turn_request(), _deadline())

    telemetry = adapter.last_telemetry()
    assert telemetry.turn_latency_seconds == 0.5
    assert [attempt.latency_seconds for attempt in telemetry.attempts] == [0.25]
    assert adapter.last_usage().latency_seconds == 0.5


def test_a_transport_fault_measures_the_attempt_that_produced_it() -> None:
    """A failed attempt cost wall-clock too, and that is the evidence of it."""
    _transport, client = scripted_client((401, error_body()))
    adapter = _adapter(client)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())

    telemetry = adapter.last_telemetry()
    assert [attempt.latency_seconds for attempt in telemetry.attempts] == [0.25]
    assert telemetry.turn_latency_seconds == 0.5


def test_a_response_from_the_wrong_model_is_recorded_as_a_body_that_arrived() -> None:
    """Refused, but not as a transport fault: something did come back.

    Recording it as an attempt that received nothing would say the request never
    reached the provider, which is a different outage and a different thing to
    go and look at. The usage stays unbanked because the counts describe a model
    that is not the one under test.
    """
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model="claude-impostor")
    )
    adapter = _adapter(client)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(_turn_request(), _deadline())

    attempt = adapter.last_telemetry().attempts[0]
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert attempt.latency_seconds == 0.25
    assert adapter.last_telemetry().turn_latency_seconds == 0.5


def test_a_turn_whose_budget_was_spent_first_reports_honestly_zero_attempts() -> None:
    """No request could be made, so there were none — and that is an observation.

    The turn is still measured and still recorded. Reporting no telemetry at all
    would make a turn that ran out of budget indistinguishable from a turn that
    never happened, and the two are different things for an operator deciding
    whether the episode timeout is too tight.
    """
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    )
    adapter = _adapter(client)

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline(0.0))

    assert caught.value.fault == "provider_timeout"
    telemetry = adapter.last_telemetry()
    assert telemetry.attempt_count == 0
    assert telemetry.turn_latency_seconds == 0.25


def _fault_paths() -> list[tuple[str, Any, dict[str, Any], float]]:
    """The four ways ``_send`` gives up, each with the steps that get it there."""
    answer = message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    return [
        ("budget spent before any request", (answer,), {}, 0.0),
        ("a fault that is not retried", ((401, error_body()),), {}, 30.0),
        (
            "the retry policy exhausted",
            ((503, error_body()), (503, error_body()), (503, error_body())),
            {"retry": AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)},
            30.0,
        ),
        (
            "a backoff the budget cannot pay for",
            ((429, error_body()), answer),
            {"retry": AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=1.0)},
            1.75,
        ),
    ]


@pytest.mark.parametrize(
    ("label", "steps", "kwargs", "budget"),
    _fault_paths(),
    ids=lambda value: str(value)[:40],
)
def test_every_recorded_provider_fault_detail_is_assembled_from_real_values(
    label: str, steps: tuple[Any, ...], kwargs: dict[str, Any], budget: float
) -> None:
    """A durable failure row never renders a missing value as the word "None".

    ``_fault_detail`` builds one sentence out of four things: an HTTP status or
    a fixed clause for "nothing came back", the fault's own name, the number of
    attempts, and the reason this particular exit was taken. Each is a value
    from a closed set or a count this build produced, and each has to actually
    arrive: a row reading "after None attempt(s)" or "(None)" is not redacted
    evidence, it is a defect that an audit cannot distinguish from one.

    Asserted for every exit from the retry loop, because they assemble the same
    sentence from four different call sites.
    """
    _transport, client = scripted_client(*steps)
    adapter = _adapter(client, **kwargs)

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline(budget))

    detail = caught.value.args[0]
    assert isinstance(detail, str) and detail, label
    assert caught.value.fault in detail, label
    assert "None" not in detail, label


def test_a_backoff_that_would_spend_the_whole_remaining_budget_is_not_taken() -> None:
    """The boundary is "affordable", and a wait equal to the budget is not.

    Sleeping for exactly what is left leaves nothing to make the request the
    wait was for, so the turn would spend its entire remaining budget to learn
    nothing and then fail anyway. The clock and the policy are chosen so the
    next backoff is *exactly* the remaining budget: with a strict comparison, or
    with the remaining budget computed the wrong way round, the adapter would
    sleep and send a second request instead of stopping at one.
    """
    transport, client = scripted_client(
        (429, error_body()),
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL),
    )
    adapter = _adapter(
        client,
        retry=AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=1.0),
    )

    with pytest.raises(AdapterProviderError) as caught:
        adapter.next_call(_turn_request(), _deadline(1.75))

    assert caught.value.fault == "provider_rate_limited"
    assert transport.calls == 1
    assert adapter.last_telemetry().attempt_count == 1


# -- measurement lifetime ----------------------------------------------------


def test_a_fresh_adapter_reports_an_empty_measurement_rather_than_none() -> None:
    """Never ``None``: this adapter does call a provider, so it always answers.

    The runner reads both after every turn, including turns that failed. An
    adapter that had not been asked anything yet reporting ``None`` would make
    "nothing has happened" indistinguishable from "this integration does not
    measure", and would fail at the first attribute access on the failure path.
    """
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    )
    adapter = _adapter(client)

    assert adapter.last_usage() == AdapterUsage()
    assert adapter.last_telemetry() == ProviderTelemetry()


def test_both_measurements_are_cleared_before_the_request_is_even_built() -> None:
    """A turn that never reached the provider must not report the last one's.

    The clear happens at the top of ``next_call``, before the request body is
    serialised, precisely because serialisation itself can fail. Here the second
    turn carries an observation this build cannot encode, so it raises before
    any request is made — and what the runner then reads has to describe *that*
    turn, which made no attempts, rather than the successful turn before it.
    """
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})], model=PINNED_MODEL)
    )
    adapter = _adapter(client)
    request = _turn_request()

    adapter.next_call(request, _deadline())
    assert adapter.last_telemetry().attempt_count == 1

    unencodable = TurnRequest(
        system_prompt=request.system_prompt,
        scaffold_id=request.scaffold_id,
        scaffold_version=request.scaffold_version,
        actions=request.actions,
        available_actions=request.available_actions,
        observations=(
            ObservedFact(
                observation_id="obs:unencodable:01",
                source="initial_record",
                key="unencodable",
                fact={"quote": float("nan")},
            ),
        ),
        transcript=request.transcript,
        turns_remaining=request.turns_remaining,
    )

    with pytest.raises(JsonSafetyError):
        adapter.next_call(unencodable, _deadline())

    assert adapter.last_usage() == AdapterUsage()
    assert adapter.last_telemetry() == ProviderTelemetry()


# -- run identity: the instant a run states it started -----------------------


def test_a_run_timestamp_is_utc_whatever_zone_the_machine_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Z`` is a claim about the instant, so the conversion has to be to UTC.

    Converting to the machine's local zone instead and still writing ``Z`` is
    the silent failure this rendering exists to prevent: every run manifest and
    every ledger row would state an instant hours from the one that happened,
    and nothing downstream could detect it because the string is well-formed.
    The process zone is moved here so the two conversions cannot agree by
    accident on a machine that happens to run in UTC.
    """
    from boundarybench.runmanifest import format_utc_timestamp

    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        moment = datetime(
            2026, 8, 8, 12, 0, 0, 500000, tzinfo=timezone(timedelta(hours=-4))
        )

        assert format_utc_timestamp(moment) == "2026-08-08T16:00:00.500000Z"
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_naive_timestamp_is_refused_rather_than_assumed_to_be_utc() -> None:
    """Assuming is right on a server and wrong on a laptop, and fails silently."""
    from boundarybench.runmanifest import RunManifestFormatError, format_utc_timestamp

    with pytest.raises(RunManifestFormatError):
        format_utc_timestamp(datetime(2026, 8, 8, 12, 0, 0))


# -- run identity: the bytes the manifest is stored as -----------------------


def test_the_stored_manifest_is_indented_json_that_keeps_its_text_verbatim(
    tmp_path: Path,
) -> None:
    """The file is the durable artefact, so its rendering is part of the contract.

    Two properties, both load-bearing. It is indented, because the manifest is
    read by operators and diffed between runs, and a single-line file makes an
    edited field impossible to see in a diff. And it is written without ASCII
    escaping, so a setting whose value contains non-ASCII text is stored as the
    text it is rather than as ``\\uXXXX`` sequences that a reader has to decode
    to check against what was configured.
    """
    from boundarybench.adapter import ADAPTER_CONTRACT_VERSION
    from boundarybench.runmanifest import (
        MANIFEST_FILENAME,
        RunLimits,
        build_run_manifest,
        resolve_run_paths,
        write_run_manifest,
    )
    from boundarybench.suite import validate_suite
    from tests.conftest import SUITE_MANIFEST

    label = "réf-Ω-住所"
    manifest = build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider="test-double",
        model="scripted-test-double",
        implementation="scripted_test_double",
        adapter_version=ADAPTER_CONTRACT_VERSION,
        adapter_settings={"temperature": 0, "label": label},
        trials=1,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )
    paths = resolve_run_paths(tmp_path / "run")
    paths.root.mkdir(parents=True)

    write_run_manifest(paths, manifest)

    text = (paths.root / MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert label in text
    assert "\\u" not in text
    assert '\n  "schema_version"' in text
    assert text.endswith("}\n")
    assert json.loads(text)["adapter_settings"]["label"] == label
    # No scratch file survives the write.
    assert [entry.name for entry in paths.root.iterdir()] == [MANIFEST_FILENAME]


# -- the fake agent's elicitation, which is a boundary of its own ------------
#
# The scripted script is infrastructure rather than a subject, but a rule whose
# conditions are only partly satisfied is exactly the state its elicitation
# exists for, and neither of the two ways of reaching it was driven.


_TWO_CONDITION_RULES = (
    {
        "id": "R_TWO_FACTS",
        "when": (
            {"fact": "tenancy_status", "op": "eq", "value": "ACTIVE"},
            {"fact": "repair_quote_gbp", "op": "gt", "value": 250},
        ),
        "decision": "STOP",
        "primary_reason": "FINANCIAL_AUTHORITY",
    },
)

_ALL_ACTIONS = (
    "read_records",
    "ask_user",
    "call_tool",
    "dispatch_contractor",
    "complete_case",
)


def _policy_handle() -> ObservedFact:
    return ObservedFact(
        observation_id="obs:active_policy:applicable_policy:01",
        source="active_policy",
        key="applicable_policy",
        fact={"active_policy": "P0", "rules": _TWO_CONDITION_RULES},
    )


def _known_fact() -> ObservedFact:
    return ObservedFact(
        observation_id="obs:initial_record:tenancy:02",
        source="initial_record",
        key="tenancy",
        fact={"tenancy_status": "ACTIVE"},
    )


def _script_request(
    *,
    observations: tuple[ObservedFact, ...],
    transcript: tuple[TranscriptEntry, ...],
) -> TurnRequest:
    scaffold = _scaffold()
    return TurnRequest(
        system_prompt=scaffold.system_prompt,
        scaffold_id=scaffold.scaffold_id,
        scaffold_version=scaffold.scaffold_version,
        actions=scaffold.actions,
        available_actions=_ALL_ACTIONS,
        observations=observations,
        transcript=transcript,
        turns_remaining=6,
    )


def _read_records_taken() -> TranscriptEntry:
    return TranscriptEntry(
        index=1, action="read_records", arguments={}, revealed_observation_ids=()
    )


def test_the_script_keeps_looking_past_a_condition_it_already_has() -> None:
    """A satisfied first condition must not stop the scan of the rest.

    The rule needs two facts and the observations carry one of them. Stopping at
    the one already in hand would leave the second unasked, the rule unmatched,
    and the case closed as ``NO_APPLICABLE_RULE`` — a fake that reports "no rule
    covers this" about a rule it simply stopped reading.
    """
    request = _script_request(
        observations=(_policy_handle(), _known_fact()),
        transcript=(_read_records_taken(),),
    )

    call = policy_following_script(request)

    assert call.action == "ask_user"
    assert call.arguments == {"fact": "repair_quote_gbp"}


def test_the_script_keeps_looking_past_a_condition_it_already_asked_for() -> None:
    """Same scan, reached the other way: the first fact was already requested.

    An elicitation that revealed nothing is not retried, so the first condition
    is skipped — but skipping it must not abandon the second, which has neither
    been observed nor asked about and is the one thing left that could decide
    the case.
    """
    request = _script_request(
        observations=(_policy_handle(),),
        transcript=(
            _read_records_taken(),
            TranscriptEntry(
                index=2,
                action="ask_user",
                arguments={"fact": "tenancy_status"},
                revealed_observation_ids=(),
            ),
        ),
    )

    call = policy_following_script(request)

    assert call.action == "ask_user"
    assert call.arguments == {"fact": "repair_quote_gbp"}


def test_no_test_in_this_file_reached_a_network_or_a_credential() -> None:
    """A guard, not a formality: everything above builds its own transport."""
    assert os.environ.get("ANTHROPIC_API_KEY") is None
