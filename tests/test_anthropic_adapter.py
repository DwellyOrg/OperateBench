"""The Anthropic Messages adapter.

Every test here runs the real SDK over an in-process transport (see
``tests.anthropic_transport``). No test contacts a provider, and none of them is
evidence about a model: they are evidence about the integration.
"""

from __future__ import annotations

import json

import anthropic
import pytest

from boundarybench.adapter import (
    AdapterProtocolError,
    TurnDeadline,
    TurnRequest,
    build_turn_request,
)
from boundarybench.compiler import Variant
from boundarybench.environment import Environment
from boundarybench.scaffold import STANDARD_SCAFFOLD, Scaffold, load_scaffold
from tests.anthropic_transport import (
    FAKE_API_KEY,
    error_body,
    message_body,
    one_call_client,
    scripted_client,
    text_block,
    tool_use_block,
)
from tests.conftest import minimal_card


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _variant() -> Variant:
    from boundarybench.compiler import compile_cube
    from boundarybench.schema import ConstructCard

    cube = compile_cube(ConstructCard.from_dict(minimal_card()))
    return cube.variants[0]


def _turn_request(*, read: bool = False) -> tuple[TurnRequest, Variant]:
    variant = _variant()
    environment = Environment(variant)
    if read:
        environment.read_records()
    return (
        build_turn_request(
            scaffold=_scaffold(), environment=environment, turns_remaining=12
        ),
        variant,
    )


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


PINNED_MODEL = "claude-test-20990101"


def test_the_serialized_request_shows_only_what_the_turn_request_carries() -> None:
    """The wire body is a projection of ``TurnRequest`` and of nothing else.

    Asserted as an equality rather than a list of absent strings. A leak test
    that only greps for known answer-key names passes for every field nobody
    thought to grep for; requiring the body to *be* the projection means a new
    field cannot appear in it at all.
    """
    from boundarybench.providers.anthropic_messages import (
        AnthropicMessagesAdapter,
        build_messages_request,
    )

    request, variant = _turn_request(read=True)
    transport, client = one_call_client(
        "complete_case",
        {
            "disposition": "STOP",
            "primary_reason_code": "FINANCIAL_AUTHORITY",
            "secondary_reason_codes": [],
            "evidence_refs": [],
        },
    )
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)

    adapter.next_call(request, _deadline())

    assert transport.calls == 1
    body = transport.bodies[0]
    assert body == build_messages_request(request, model=PINNED_MODEL)

    rendered = json.dumps(body, sort_keys=True)
    assert variant.variant_id not in rendered
    assert variant.content_digest not in rendered
    for forbidden in (
        "expected_disposition",
        "expected_primary_reason",
        "required_evidence",
        "expected_mutations",
        "matched_rule",
        "evidence_relevant",
        "answer_key",
    ):
        assert forbidden not in rendered


def test_the_tools_are_the_scaffold_action_schemas() -> None:
    """One scaffold across providers means the tool list *is* the scaffold."""
    from boundarybench.providers.anthropic_messages import build_messages_request

    request, _ = _turn_request()
    body = build_messages_request(request, model=PINNED_MODEL)

    assert [tool["name"] for tool in body["tools"]] == [
        action.name for action in request.actions
    ]
    complete = next(tool for tool in body["tools"] if tool["name"] == "complete_case")
    assert complete["input_schema"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "disposition": {
                "type": "string",
                "description": complete["input_schema"]["properties"]["disposition"][
                    "description"
                ],
            },
            "primary_reason_code": {
                "type": "string",
                "description": complete["input_schema"]["properties"][
                    "primary_reason_code"
                ]["description"],
            },
            "secondary_reason_codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": complete["input_schema"]["properties"][
                    "secondary_reason_codes"
                ]["description"],
            },
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": complete["input_schema"]["properties"]["evidence_refs"][
                    "description"
                ],
            },
        },
        "required": [
            "disposition",
            "primary_reason_code",
            "secondary_reason_codes",
            "evidence_refs",
        ],
    }
    assert body["system"] == request.system_prompt


def test_the_observed_transcript_is_the_model_visible_history() -> None:
    """History comes from the environment's record, not from provider state."""
    from boundarybench.providers.anthropic_messages import build_messages_request

    request, _ = _turn_request(read=True)
    body = build_messages_request(request, model=PINNED_MODEL)

    assert [message["role"] for message in body["messages"]] == ["user"]
    carried = json.loads(body["messages"][0]["content"])
    assert carried == {
        "available_actions": list(request.available_actions),
        "resolvable_queries": {"ask_user": ["repair_quote_gbp"]},
        "observations": [entry.as_dict() for entry in request.observations],
        "transcript": [entry.as_dict() for entry in request.transcript],
        "turns_remaining": request.turns_remaining,
        "scaffold_id": request.scaffold_id,
        "scaffold_version": request.scaffold_version,
    }
    assert carried["transcript"][0]["action"] == "read_records"
    assert carried["observations"]


def test_one_tool_call_becomes_one_adapter_call() -> None:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    request, _ = _turn_request()
    _transport, client = one_call_client("ask_user", {"fact": "repair_quote_gbp"})
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)

    call = adapter.next_call(request, _deadline())

    assert call.action == "ask_user"
    assert call.arguments == {"fact": "repair_quote_gbp"}


@pytest.mark.parametrize("block", [tool_use_block("read_records", {})])
def test_a_read_call_carries_an_empty_argument_mapping(block: object) -> None:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    request, _ = _turn_request()
    _transport, client = one_call_client("read_records", {})
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)

    assert adapter.next_call(request, _deadline()).arguments == {}


# -- the response contract ---------------------------------------------------


def _call(*content: object, **kwargs: object):
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    request, _ = _turn_request()
    _transport, client = scripted_client(message_body(list(content), **kwargs))
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)
    return adapter.next_call(request, _deadline())


def test_a_text_only_answer_is_a_protocol_failure() -> None:
    """Prose is not an action, and must not be read as one."""
    with pytest.raises(AdapterProtocolError) as raised:
        _call(text_block("I think we should stop here."), stop_reason="end_turn")

    assert "exactly one" in str(raised.value)


def test_two_tool_calls_are_a_protocol_failure() -> None:
    """One turn is one action; two answers make the action ambiguous."""
    with pytest.raises(AdapterProtocolError) as raised:
        _call(
            tool_use_block("read_records", {}, block_id="toolu_a"),
            tool_use_block("ask_user", {"fact": "x"}, block_id="toolu_b"),
        )

    assert "exactly one" in str(raised.value)


def test_a_tool_the_scaffold_does_not_declare_is_a_protocol_failure() -> None:
    with pytest.raises(AdapterProtocolError) as raised:
        _call(tool_use_block("exfiltrate_answer_key", {}))

    assert "does not declare" in str(raised.value)
    # The name the model produced is model text, so it is diagnosed without
    # being quoted into a message that ends up in a durable row.
    assert "exfiltrate_answer_key" not in str(raised.value)


def test_arguments_that_are_not_an_object_are_a_protocol_failure() -> None:
    with pytest.raises(AdapterProtocolError) as raised:
        _call(tool_use_block("read_records", ["not", "an", "object"]))

    assert "JSON object" in str(raised.value)


def test_arguments_that_cannot_be_recorded_are_a_protocol_failure() -> None:
    """A lone surrogate parses into a Python string and cannot be written back.

    Sent as raw bytes, because the escape is what the provider would put on the
    wire: ``"\\ud800"`` is ASCII JSON that parses into a Python string UTF-8
    cannot encode, which is exactly the shape the ledger has to refuse.
    """
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    request, _ = _turn_request()
    body = message_body([tool_use_block("ask_user", {"fact": "\ud800"})])
    _transport, client = scripted_client(json.dumps(body).encode("ascii"))
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)

    with pytest.raises(AdapterProtocolError) as raised:
        adapter.next_call(request, _deadline())

    assert "cannot record" in str(raised.value)


def test_a_block_that_is_neither_text_nor_a_tool_call_is_a_protocol_failure() -> None:
    """Extra provider content makes the answer ambiguous rather than richer."""
    with pytest.raises(AdapterProtocolError) as raised:
        _call(
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            tool_use_block("read_records", {}),
        )

    assert "content block" in str(raised.value)


def test_a_truncated_answer_is_its_own_output_limit_failure() -> None:
    """``max_tokens`` means the call was cut off by *this run's* own ceiling.

    It has its own failure class rather than the protocol-error class that means
    the model cannot follow a schema; see ``tests/test_provider_methodology.py``
    for the taxonomy and eligibility rule.

    The refusal still names nothing the response stated: the stop reason it
    carried is provider-controlled text and this message becomes a durable
    ledger detail.
    """
    from boundarybench.adapter import AdapterOutputLimitError

    with pytest.raises(AdapterOutputLimitError) as raised:
        _call(tool_use_block("read_records", {}), stop_reason="max_tokens")

    assert not isinstance(raised.value, AdapterProtocolError)
    assert "max_tokens" not in str(raised.value)
    assert "1024" in str(raised.value)


def test_an_unrecognised_stop_reason_is_still_a_redacted_protocol_failure() -> None:
    """Only the reason this build controls moved; the rest keep their old home."""
    with pytest.raises(AdapterProtocolError) as raised:
        _call(tool_use_block("read_records", {}), stop_reason="end_turn")

    assert "end_turn" not in str(raised.value)
    assert "tool_use" in str(raised.value)


def test_commentary_beside_one_tool_call_is_accepted() -> None:
    """Text alongside the action is normal, and does not make it ambiguous."""
    call = _call(
        text_block("Reading the case record first."),
        tool_use_block("read_records", {}),
    )

    assert call.action == "read_records"


# -- the credential ----------------------------------------------------------

SECRET = "sk-ant-not-a-real-key-0123456789"

#: A clearly synthetic, purely local configuration identity. It has the shape a
#: ``configuration_id`` has and is a value none ever is: no suite, scaffold or
#: settings hash to it. It exists so a test can authorise *something* without
#: this tree carrying the identity of a run that happened.
SYNTHETIC_CONFIGURATION = "0" * 64
SYNTHETIC_AUTHORISATION = frozenset({SYNTHETIC_CONFIGURATION})


def test_a_missing_credential_is_refused_by_name() -> None:
    from boundarybench.providers.anthropic_messages import (
        AnthropicConfigurationError,
        resolve_api_key,
    )

    with pytest.raises(AnthropicConfigurationError) as raised:
        resolve_api_key({})

    assert "ANTHROPIC_API_KEY" in str(raised.value)


def test_a_blank_credential_is_a_missing_one() -> None:
    from boundarybench.providers.anthropic_messages import (
        AnthropicConfigurationError,
        resolve_api_key,
    )

    with pytest.raises(AnthropicConfigurationError):
        resolve_api_key({"ANTHROPIC_API_KEY": "   "})


@pytest.mark.parametrize("variable", ["ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"])
def test_an_environment_that_redirects_the_request_is_refused(variable: str) -> None:
    """A base URL is where the credential is sent, so it is never inferred."""
    from boundarybench.providers.anthropic_messages import (
        AnthropicConfigurationError,
        build_anthropic_adapter,
    )

    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=PINNED_MODEL,
            configuration_id=SYNTHETIC_CONFIGURATION,
            audited=SYNTHETIC_AUTHORISATION,
            environ={"ANTHROPIC_API_KEY": SECRET, variable: "https://elsewhere.invalid"},
        )

    assert variable in str(raised.value)
    assert SECRET not in str(raised.value)


def test_the_credential_reaches_the_provider_and_nothing_else() -> None:
    """It is a request header, and it is not identity, settings or a repr."""
    from boundarybench.providers.anthropic_messages import build_anthropic_adapter

    adapter = build_anthropic_adapter(
        model=PINNED_MODEL,
        configuration_id=SYNTHETIC_CONFIGURATION,
        audited=SYNTHETIC_AUTHORISATION,
        environ={"ANTHROPIC_API_KEY": SECRET},
    )

    rendered = "\n".join(
        [
            repr(adapter),
            json.dumps(dict(adapter.settings), sort_keys=True),
            json.dumps(adapter.identity.as_dict(), sort_keys=True),
        ]
    )
    assert SECRET not in rendered
    assert adapter.identity.provider == "anthropic"
    assert adapter.identity.model == PINNED_MODEL


def test_the_client_disables_the_sdk_retry_loop_and_pins_the_endpoint() -> None:
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_BASE_URL,
        build_client,
    )

    client = build_client(api_key=SECRET)

    assert client.max_retries == 0
    assert str(client.base_url).rstrip("/") == ANTHROPIC_BASE_URL


def test_the_credential_travels_in_the_authentication_header() -> None:
    """Proof the resolved key is actually used, without leaving the process."""
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    request, _ = _turn_request()
    transport, _client = one_call_client("read_records", {})
    client = transport.client()
    adapter = AnthropicMessagesAdapter(model=PINNED_MODEL, client=client)

    adapter.next_call(request, _deadline())

    assert transport.requests[0].headers["x-api-key"] == FAKE_API_KEY


@pytest.mark.parametrize("model", ["", "   ", "claude test", "claude/test"])
def test_a_model_identifier_that_is_not_one_is_refused(model: str) -> None:
    from boundarybench.providers.anthropic_messages import (
        AnthropicConfigurationError,
        build_anthropic_adapter,
    )

    with pytest.raises(AnthropicConfigurationError) as raised:
        build_anthropic_adapter(
            model=model,
            configuration_id=SYNTHETIC_CONFIGURATION,
            audited=SYNTHETIC_AUTHORISATION,
            environ={"ANTHROPIC_API_KEY": SECRET},
        )

    assert "model" in str(raised.value)


# -- the retry policy --------------------------------------------------------


class Sleeps:
    """Records what the retry loop would have waited, without waiting."""

    def __init__(self) -> None:
        self.waited: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


def _adapter(client, **kwargs):
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    return AnthropicMessagesAdapter(model=PINNED_MODEL, client=client, **kwargs)


def _ok_body():
    return message_body([tool_use_block("read_records", {})])


def test_a_rate_limit_is_retried_and_the_next_attempt_answers() -> None:
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    request, _ = _turn_request()
    transport, client = scripted_client((429, error_body("rate_limit_error")), _ok_body())
    sleeps = Sleeps()
    adapter = _adapter(client, retry=AnthropicRetryPolicy(max_attempts=3), sleep=sleeps)

    call = adapter.next_call(request, _deadline())

    assert call.action == "read_records"
    assert transport.calls == 2
    assert sleeps.waited == [0.5]


def test_retries_are_bounded_and_exhaustion_is_a_classified_fault() -> None:
    from boundarybench.adapter import AdapterProviderError
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    request, _ = _turn_request()
    transport, client = scripted_client(
        *[(429, error_body("rate_limit_error", "slow down please"))] * 3
    )
    sleeps = Sleeps()
    adapter = _adapter(client, retry=AnthropicRetryPolicy(max_attempts=3), sleep=sleeps)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 3
    assert sleeps.waited == [0.5, 1.0]
    assert raised.value.fault == "provider_rate_limited"
    # The detail is stable and closed-set: a status, a count, a fixed sentence.
    assert "429" in str(raised.value)
    assert "3 attempt(s)" in str(raised.value)
    assert "slow down please" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "fault"),
    [
        (401, "provider_authentication"),
        (403, "provider_authentication"),
        (400, "provider_request_rejected"),
        # 404 is the resource this run pins, not the payload this turn sent, so
        # it is the configuration fault rather than a request rejection.
        (404, "provider_configuration"),
        (413, "provider_request_rejected"),
    ],
)
def test_a_non_retryable_provider_refusal_is_recorded_at_once(
    status: int, fault: str
) -> None:
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client((status, error_body()), _ok_body())
    sleeps = Sleeps()
    adapter = _adapter(client, sleep=sleeps)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 1
    assert sleeps.waited == []
    assert raised.value.fault == fault
    assert adapter.last_telemetry().attempts[0].response_received is True


@pytest.mark.parametrize("status", [500, 502, 529])
def test_a_server_error_is_retried_then_recorded(status: int) -> None:
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client(*[(status, error_body())] * 3)
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 3
    assert raised.value.fault == "provider_server_error"
    assert all(attempt.response_received for attempt in adapter.last_telemetry().attempts)


def test_a_connection_failure_is_retried_then_recorded() -> None:
    import httpx

    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    failures = [httpx.ConnectError("no route to host") for _ in range(3)]
    transport, client = scripted_client(*failures)
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 3
    assert raised.value.fault == "provider_network_error"
    assert not any(
        attempt.response_received for attempt in adapter.last_telemetry().attempts
    )
    assert "no route to host" not in str(raised.value)


def test_a_request_timeout_is_retried_then_recorded() -> None:
    import httpx

    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client(*[httpx.ReadTimeout("too slow")] * 3)
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 3
    assert raised.value.fault == "provider_timeout"


def test_a_client_whose_own_retry_loop_is_live_is_refused() -> None:
    """The adapter owns the policy, so a hidden second one is a configuration bug."""
    from boundarybench.providers.anthropic_messages import AnthropicConfigurationError
    from tests.anthropic_transport import RecordingTransport

    client = RecordingTransport([]).client(max_retries=2)

    with pytest.raises(AnthropicConfigurationError) as raised:
        _adapter(client)

    assert "max_retries" in str(raised.value)


def test_an_interrupt_is_never_swallowed_by_the_retry_loop() -> None:
    """``BaseException`` is not a provider fault and is not retried."""
    request, _ = _turn_request()
    transport, client = scripted_client(KeyboardInterrupt(), _ok_body())
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(KeyboardInterrupt):
        adapter.next_call(request, _deadline())

    assert transport.calls == 1


# -- the deadline ------------------------------------------------------------


class TickingClock:
    """A monotonic clock that advances a fixed step on every reading."""

    def __init__(self, step: float = 0.1) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        now = self._now
        self._now += self._step
        return now


def test_each_attempt_is_given_the_turns_remaining_budget() -> None:
    """The SDK is told the ceiling, so a hung request cannot outlast the turn."""
    request, _ = _turn_request()
    transport, client = scripted_client(_ok_body())
    adapter = _adapter(client, clock=lambda: 0.0)

    adapter.next_call(
        request, TurnDeadline(remaining_seconds=7.5, cancelled=lambda: False)
    )

    assert transport.timeouts[0]["read"] == pytest.approx(7.5)


def test_a_retry_is_given_less_budget_than_the_attempt_before_it() -> None:
    """Retrying inside a fixed budget means the budget shrinks, not resets."""
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    request, _ = _turn_request()
    transport, client = scripted_client((429, error_body()), _ok_body())
    adapter = _adapter(
        client,
        retry=AnthropicRetryPolicy(max_attempts=3, initial_backoff_seconds=0.0),
        sleep=Sleeps(),
        clock=TickingClock(0.1),
    )

    adapter.next_call(request, _deadline(30.0))

    assert transport.timeouts[1]["read"] < transport.timeouts[0]["read"]


def test_a_spent_budget_stops_before_the_provider_is_called() -> None:
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client(_ok_body())
    adapter = _adapter(client, clock=lambda: 0.0)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(
            request, TurnDeadline(remaining_seconds=0.0, cancelled=lambda: False)
        )

    assert transport.calls == 0
    assert raised.value.fault == "provider_timeout"
    assert "0 attempt(s)" in str(raised.value)


def test_a_cancelled_turn_is_not_retried() -> None:
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client((429, error_body()), _ok_body())
    sleeps = Sleeps()
    adapter = _adapter(client, sleep=sleeps)

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(
            request,
            TurnDeadline(remaining_seconds=30.0, cancelled=lambda: transport.calls >= 1),
        )

    assert transport.calls == 1
    assert sleeps.waited == []
    assert raised.value.fault == "provider_rate_limited"
    assert "cancelled" in str(raised.value) or "abandoned" in str(raised.value)


def test_a_backoff_the_budget_cannot_pay_for_is_never_slept() -> None:
    """Sleeping through the rest of the turn to then fail helps nobody."""
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    transport, client = scripted_client((429, error_body()), _ok_body())
    sleeps = Sleeps()
    adapter = _adapter(client, sleep=sleeps, clock=lambda: 0.0)

    with pytest.raises(AdapterProviderError):
        adapter.next_call(
            request, TurnDeadline(remaining_seconds=0.2, cancelled=lambda: False)
        )

    assert transport.calls == 1
    assert sleeps.waited == []


# -- telemetry ---------------------------------------------------------------


def test_tokens_come_from_the_provider_and_latency_is_measured_locally() -> None:
    request, _ = _turn_request()
    _transport, client = scripted_client(
        message_body(
            [tool_use_block("read_records", {})], input_tokens=1234, output_tokens=56
        )
    )
    adapter = _adapter(client, clock=TickingClock(0.25))

    adapter.next_call(request, _deadline())
    usage = adapter.last_usage()

    assert usage.input_tokens == 1234
    assert usage.output_tokens == 56
    assert usage.latency_seconds == pytest.approx(0.5)
    # No versioned, reviewed pricing source exists, so a cost would be invented.
    assert usage.cost_usd is None


def test_measured_latency_covers_the_retries_the_turn_actually_took() -> None:
    """One turn's wall-clock is what the turn spent, retries and waits included."""
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    request, _ = _turn_request()
    _transport, client = scripted_client((429, error_body()), _ok_body())
    clock = TickingClock(0.25)
    adapter = _adapter(
        client, retry=AnthropicRetryPolicy(max_attempts=3), sleep=Sleeps(), clock=clock
    )

    adapter.next_call(request, _deadline())

    first_attempt_only = 0.5
    assert adapter.last_usage().latency_seconds > first_attempt_only


def test_a_turn_that_never_completed_reports_no_measurement() -> None:
    """A failed call has no tokens, so it reports none — not zero, and not stale."""
    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    _transport, client = scripted_client(
        message_body([tool_use_block("read_records", {})]),
        (401, error_body()),
    )
    adapter = _adapter(client, sleep=Sleeps())

    adapter.next_call(request, _deadline())
    assert adapter.last_usage().input_tokens == 137

    with pytest.raises(AdapterProviderError):
        adapter.next_call(request, _deadline())

    stale = adapter.last_usage()
    assert stale.input_tokens is None
    assert stale.output_tokens is None
    assert stale.latency_seconds is None
    assert stale.cost_usd is None


def test_a_response_the_sdk_cannot_read_is_a_provider_fault_and_is_not_retried() -> None:
    """An unparseable answer is the provider's, and repeating it does not help."""
    import httpx

    from boundarybench.adapter import AdapterProviderError

    request, _ = _turn_request()
    invalid = anthropic.APIResponseValidationError(
        response=httpx.Response(
            200, request=httpx.Request("POST", "https://provider.invalid/v1/messages")
        ),
        body=None,
    )
    transport, client = scripted_client(invalid, _ok_body())
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(AdapterProviderError) as raised:
        adapter.next_call(request, _deadline())

    assert transport.calls == 1
    assert raised.value.fault == "provider_response_invalid"


def test_a_request_timeout_status_is_retried_like_a_transport_timeout() -> None:
    request, _ = _turn_request()
    transport, client = scripted_client((408, error_body()), _ok_body())
    adapter = _adapter(client, sleep=Sleeps())

    assert adapter.next_call(request, _deadline()).action == "read_records"
    assert transport.calls == 2


def test_a_failure_that_is_not_the_providers_is_left_for_the_runner() -> None:
    """This integration breaking is an adapter failure, not an outage.

    Reported by *not* classifying it: the runner already records an exception
    out of ``next_call`` as an adapter failure, and dressing an internal fault
    up as a provider fault would put a bug in this build and a service outage in
    the same bucket, which is the confusion the taxonomy exists to remove.
    """
    request, _ = _turn_request()
    transport, client = scripted_client(anthropic.AnthropicError("internal"))
    adapter = _adapter(client, sleep=Sleeps())

    with pytest.raises(anthropic.AnthropicError):
        adapter.next_call(request, _deadline())

    assert transport.calls == 1
