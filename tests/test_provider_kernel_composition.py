"""The Boundary OpenAI adapter is a projection over the shared exchange.

The extraction this file guards is a *composition* change and nothing else: the
Boundary Track keeps owning ``TurnRequest -> body`` and ``response ->
AdapterCall``, and everything between them — the endpoint check, the profile
check, single-flight, the retry loop, the cost guard, the raw-wire capture and
the response contract — moves to one shared exchange under
``operatebench.providers`` that neither track owns.

That is only true if two things hold, and both are pinned here rather than
argued.

**Nothing observable moved.** The exact request bytes, the settings mapping
hashed into ``configuration_id``, the adapter identity and the attempt evidence
are byte-for-byte what the pre-extraction implementation recorded. The two
digests below lock that recording. If either one moves, the extraction has
changed what a run *is* — every stored ledger row is hashed against it and a
resume across it is refused by design — so they are stated as literals and a
failure here is a stop, not a number to update.

**There is one send path, not two.** A duplicated executor would pass every
golden test above while quietly drifting on the next change, so the adapter is
also checked for holding no retry loop of its own and for reaching the provider
through the shared exchange when a real turn runs.

Nothing here opens a socket: every client is a real ``openai.OpenAI`` over
``httpx.MockTransport``. No credential is read and no provider is called.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pytest

from boundarybench.adapter import (
    ObservedFact,
    TranscriptEntry,
    TurnDeadline,
    TurnRequest,
)
from boundarybench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    OPENAI_ADAPTER_VERSION,
    OpenAIResponsesAdapter,
    build_responses_request,
    openai_identity,
    openai_settings,
    parse_response,
    request_token_bound,
)
from boundarybench.scaffold import ActionParameter, ActionSchema
from tests.openai_transport import (
    function_call_item,
    responses_body,
    scripted_client,
)

MODEL = GPT_5_6_LUNA_MODEL

#: The canonical SHA-256 of the exact request body this fixture puts on the
#: wire, recorded from the shipped adapter before the kernel existed.
BASELINE_REQUEST_DIGEST = (
    "4452f2d6467a1ee883eddf32bfab00980980334e16b5589c61c25e6d4c9343a9"
)

#: The canonical SHA-256 of ``openai_settings(model=GPT_5_6_LUNA_MODEL)``, from
#: the same recording. This mapping is hashed into ``configuration_id``.
BASELINE_SETTINGS_DIGEST = (
    "02f7334b1a586ac6631700436e43fd0dedac709936a1035b1434df1d95b26164"
)

#: Provider, model, implementation, version — the identity every ledger row
#: carries.
BASELINE_IDENTITY = ("openai", "gpt-5.6-luna", "openai_responses", "0.5.0")

#: One attempt, on the recorded fields: index, outcome, fault, http status,
#: response received, usage reported, reservation, settlement.
BASELINE_ATTEMPT = (1, "response", None, None, True, True, None, None)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _turn_request() -> TurnRequest:
    """One fixed Boundary turn, so the body below is reproducible."""
    return TurnRequest(
        system_prompt="You are under test.",
        scaffold_id="scaffold-probe",
        scaffold_version="1.0.0",
        actions=(
            ActionSchema(
                name="read_records",
                description="Retrieve a fact.",
                terminal=False,
                parameters=(
                    ActionParameter(
                        name="fact",
                        type="string",
                        required=True,
                        description="Which fact.",
                        enum=("tenancy_start", "deposit_scheme"),
                    ),
                ),
            ),
            ActionSchema(
                name="complete_case",
                description="Finish.",
                terminal=True,
                parameters=(
                    ActionParameter(
                        name="disposition",
                        type="string",
                        required=True,
                        description="The call.",
                    ),
                ),
            ),
        ),
        available_actions=("read_records", "complete_case"),
        observations=(
            ObservedFact(
                observation_id="obs-1",
                source="records",
                key="tenancy_start",
                fact={"value": "2024-01-05"},
                outcome="resolved",
            ),
        ),
        transcript=(
            TranscriptEntry(
                index=0,
                action="read_records",
                arguments={"fact": "tenancy_start"},
                revealed_observation_ids=("obs-1",),
            ),
        ),
        turns_remaining=3,
    )


def _deadline() -> TurnDeadline:
    return TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False)


def _body() -> dict[str, Any]:
    return responses_body(
        [function_call_item("read_records", '{"fact": "deposit_scheme"}')],
        model=MODEL,
    )


def _attempt_fields(attempt: Any) -> tuple[Any, ...]:
    return (
        attempt.index,
        attempt.outcome,
        attempt.fault,
        attempt.http_status,
        attempt.response_received,
        attempt.usage_reported,
        attempt.cost_reservation_usd,
        attempt.cost_settlement,
    )


# -- the goldens --------------------------------------------------------------


def test_the_request_body_on_the_wire_is_byte_for_byte_the_recorded_one() -> None:
    transport, client = scripted_client(_body())
    adapter = OpenAIResponsesAdapter(model=MODEL, client=client)

    adapter.next_call(_turn_request(), _deadline())

    assert _digest(transport.bodies[0]) == BASELINE_REQUEST_DIGEST


def test_the_projection_builds_exactly_the_body_the_adapter_sends() -> None:
    transport, client = scripted_client(_body())
    adapter = OpenAIResponsesAdapter(model=MODEL, client=client)

    adapter.next_call(_turn_request(), _deadline())
    payload = build_responses_request(_turn_request(), model=MODEL)

    assert json.loads(json.dumps(payload)) == transport.bodies[0]


def test_the_settings_hashed_into_run_identity_are_unchanged() -> None:
    assert _digest(openai_settings(model=MODEL)) == BASELINE_SETTINGS_DIGEST


def test_the_adapter_settings_are_the_settings_function_settings() -> None:
    _transport, client = scripted_client(_body())
    adapter = OpenAIResponsesAdapter(model=MODEL, client=client)

    assert _digest(dict(adapter.settings)) == BASELINE_SETTINGS_DIGEST


def test_the_adapter_identity_and_version_are_unchanged() -> None:
    identity = openai_identity(MODEL)

    assert OPENAI_ADAPTER_VERSION == "0.5.0"
    assert (
        identity.provider,
        identity.model,
        identity.implementation,
        identity.version,
    ) == BASELINE_IDENTITY


# -- the composition ----------------------------------------------------------


def test_the_shared_exchange_takes_a_payload_and_a_deadline_and_nothing_else() -> None:
    """The seam is payload-in, wire-out: no ``TurnRequest`` crosses it."""
    from operatebench.providers.openai_responses import OpenAIResponsesExchange

    annotations = OpenAIResponsesExchange.exchange.__annotations__
    assert "TurnRequest" not in str(annotations)
    assert set(annotations) == {"payload", "deadline", "return"}


def test_a_caller_that_owns_the_projection_puts_identical_bytes_on_the_wire() -> None:
    """The whole claim, end to end.

    Path one is the shipped adapter. Path two builds the body itself, hands it
    to the shared exchange and parses the answer with the Boundary parser — which
    is exactly what a second track would do. The bytes, the action, the token
    counts and the attempt evidence must be the same, or the exchange is not the
    thing the adapter uses.
    """
    from operatebench.providers.openai_responses import OpenAIResponsesExchange

    request = _turn_request()

    whole_transport, whole_client = scripted_client(_body())
    whole = OpenAIResponsesAdapter(model=MODEL, client=whole_client)
    whole_call = whole.next_call(request, _deadline())

    split_transport, split_client = scripted_client(_body())
    exchange = OpenAIResponsesExchange(
        model=MODEL, client=split_client, single_flight_label="test caller"
    )
    payload = build_responses_request(request, model=MODEL)
    with exchange.turn():
        exchange.clear()
        response = exchange.exchange(payload, _deadline())
    split_call = parse_response(response.parsed, request)

    assert split_transport.bodies[0] == whole_transport.bodies[0]
    assert _digest(split_transport.bodies[0]) == BASELINE_REQUEST_DIGEST
    assert split_call == whole_call

    whole_usage, split_usage = whole.last_usage(), exchange.last_usage()

    def counts(usage: Any) -> tuple[Any, ...]:
        return (usage.input_tokens, usage.output_tokens, usage.cost_usd)

    assert counts(whole_usage) == counts(split_usage)

    whole_telemetry = whole.last_telemetry()
    split_telemetry = exchange.last_telemetry()
    assert [_attempt_fields(a) for a in whole_telemetry.attempts] == [
        _attempt_fields(a) for a in split_telemetry.attempts
    ]
    assert _attempt_fields(whole_telemetry.attempts[0]) == BASELINE_ATTEMPT
    assert whole_telemetry.terminal_reason == split_telemetry.terminal_reason


def test_a_real_turn_reaches_the_provider_through_the_shared_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioural, not a source scan: the one send path is observed in use."""
    from operatebench.providers.openai_responses import OpenAIResponsesExchange

    seen: list[Mapping[str, Any]] = []
    original = OpenAIResponsesExchange.exchange

    def recording(
        self: OpenAIResponsesExchange,
        payload: Mapping[str, Any],
        deadline: TurnDeadline,
    ) -> Any:
        seen.append(dict(payload))
        return original(self, payload, deadline)

    monkeypatch.setattr(OpenAIResponsesExchange, "exchange", recording)

    transport, client = scripted_client(_body())
    adapter = OpenAIResponsesAdapter(model=MODEL, client=client)
    request = _turn_request()
    adapter.next_call(request, _deadline())

    assert len(seen) == 1
    assert seen[0] == build_responses_request(request, model=MODEL)
    assert transport.calls == 1


def test_the_adapter_keeps_no_retry_loop_of_its_own() -> None:
    """One executor per turn, and it belongs to the exchange."""
    from operatebench.providers.executor import TurnExecutor
    from operatebench.providers.openai_responses import OpenAIResponsesExchange

    _transport, client = scripted_client(_body())
    adapter = OpenAIResponsesAdapter(model=MODEL, client=client)

    held_by_adapter = [
        value for value in vars(adapter).values() if isinstance(value, TurnExecutor)
    ]
    exchanges = [
        value
        for value in vars(adapter).values()
        if isinstance(value, OpenAIResponsesExchange)
    ]
    assert held_by_adapter == []
    assert len(exchanges) == 1
    assert any(isinstance(value, TurnExecutor) for value in vars(exchanges[0]).values())


def test_the_token_bound_has_one_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The writer and the ledger reader compute the same number, from one place."""
    import boundarybench.providers.openai_responses as boundary
    import operatebench.providers.openai_responses as kernel

    assert boundary.request_token_bound is kernel.request_token_bound

    payload = build_responses_request(_turn_request(), model=MODEL)
    assert request_token_bound(payload) == kernel.request_token_bound(payload)


def test_the_exchange_names_the_run_and_never_the_credential() -> None:
    """The client it holds carries an API key, so the repr is written out."""
    from operatebench.providers.openai_responses import OpenAIResponsesExchange
    from tests.openai_transport import FAKE_API_KEY

    _transport, client = scripted_client(_body())
    exchange = OpenAIResponsesExchange(
        model=MODEL, client=client, single_flight_label="OpenAIResponsesAdapter"
    )

    rendered = repr(exchange)

    assert "OpenAIResponsesExchange" in rendered
    assert MODEL in rendered
    assert FAKE_API_KEY not in rendered
    assert "api_key" not in rendered


def _take_the_turn(exchange: Any) -> None:
    """Enter the exchange's single-flight guard, and leave again."""
    with exchange.turn():
        pass


def test_the_exchange_still_refuses_a_second_overlapping_turn() -> None:
    from boundarybench.adapter import AdapterBusyError
    from operatebench.providers.openai_responses import OpenAIResponsesExchange

    _transport, client = scripted_client(_body(), _body())
    exchange = OpenAIResponsesExchange(
        model=MODEL, client=client, single_flight_label="OpenAIResponsesAdapter"
    )
    with exchange.turn(), pytest.raises(AdapterBusyError):
        _take_the_turn(exchange)
