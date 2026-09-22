"""Lifecycle Anthropic Messages vertical over the real SDK, entirely in process."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest
from jsonschema import Draft202012Validator

from operatebench.agents.anthropic_messages import (
    LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
    LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V1,
    AnthropicMessagesTransport,
    anthropic_outcome_tools,
    build_model_payload,
    lower_anthropic_input_schema,
)
from operatebench.agents.evidence import (
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    provider_identity_for,
)
from operatebench.agents.lifecycle_contract import ModelRequestRefused, outcome_tools
from operatebench.agents.model import (
    AGENT_TOOL_NAMES,
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    TRUNCATED_STOP_REASON,
    ModelAgent,
    tool_schema,
)
from operatebench.agents.transport import ModelRequest, ToolCall
from operatebench.core.protocol import AgentObservation
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.anthropic_messages import (
    ANTHROPIC_BASE_URL,
    HAIKU_4_5_MODEL,
    HAIKU_4_5_PROFILE,
    SONNET_5_MODEL,
    SONNET_5_PROFILE,
    AnthropicConfigurationError,
)
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import MAX_PROVIDER_RESPONSE_ID_CHARACTERS
from tests.execution_ledger_fixtures import DIGEST_A, controls

SYNTHETIC_PROVIDER_SECRET = "synthetic-secret-provider-prose-7f3a"
SYNTHETIC_REQUEST_ID = "req_synthetic_observability_7f3a"


def observation() -> AgentObservation:
    return AgentObservation(
        now="2025-01-01T09:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_0123456789abcdef0123456789abcdef",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=[],
        last_rejection=None,
    )


def model_request(model: str) -> ModelRequest:
    # Use the public builder so protocol and tool identity are exactly Lifecycle's.
    class Never:
        def send(self, request: ModelRequest) -> Any:
            raise AssertionError("request construction must not dispatch")

    return ModelAgent(Never(), model=model, agent_id="diagnostic").build_request(
        observation()
    )


def response_body(
    model: str,
    *,
    content: Sequence[Any] | None = None,
    input_tokens: Any = 137,
    output_tokens: Any = 29,
    **extra: Any,
) -> dict[str, Any]:
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(
            content
            or [
                {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": "complete",
                    "input": {"reason": "done", "evidence_refs": []},
                }
            ]
        ),
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    body.update(extra)
    return body


class Wire:
    def __init__(self, *bodies: dict[str, Any]) -> None:
        self.bodies = list(bodies)
        self.requests: list[httpx.Request] = []
        self.sent: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.sent.append(json.loads(request.content))
        raw = json.dumps(self.bodies.pop(0)).encode("ascii")
        return httpx.Response(200, content=raw, request=request)

    def client(
        self, *, max_retries: int = 0, base_url: str = ANTHROPIC_BASE_URL
    ) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=base_url,
            max_retries=max_retries,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


@pytest.mark.parametrize(
    ("model", "profile", "sent", "omitted"),
    [
        (HAIKU_4_5_MODEL, HAIKU_4_5_PROFILE, {"temperature": 0.0}, {"thinking"}),
        (
            SONNET_5_MODEL,
            SONNET_5_PROFILE,
            {"thinking": {"type": "disabled"}},
            {"temperature"},
        ),
    ],
)
def test_real_sdk_serializes_each_exact_anthropic_profile(
    model: str, profile: Any, sent: dict[str, Any], omitted: set[str]
) -> None:
    wire = Wire(response_body(model))
    transport = AnthropicMessagesTransport(
        model=model,
        client=wire.client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    request = model_request(model)

    decision = transport.send(request)

    assert decision.tool_calls == (
        ToolCall("complete", {"reason": "done", "evidence_refs": []}),
    )
    assert wire.sent == [build_model_payload(request, model=model, profile=profile)]
    assert wire.requests[0].url == f"{ANTHROPIC_BASE_URL}/v1/messages"
    for field, value in sent.items():
        assert wire.sent[0][field] == value
    for field in omitted:
        assert field not in wire.sent[0]
    assert transport.settings["request_mapping"] == (
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION
    )
    assert transport.settings["retry"]["max_attempts"] == 1
    assert (
        transport.settings["max_output_tokens"]
        == wire.sent[0]["max_tokens"]
        == request.max_output_tokens
        == MAX_OUTPUT_TOKENS
        == 4096
    )
    assert transport.settings["protocol_version"] == MODEL_PROTOCOL_VERSION
    assert transport.last_usage().input_tokens == 137
    assert transport.last_usage().output_tokens == decision.output_tokens == 29


@pytest.mark.parametrize(
    ("message", "surrounding_whitespace"),
    [
        (sentence, whitespace)
        for sentence in (
            "Your organization has reached its spend limit",
            "Your organization has reached its spend limit.",
            "Your workspace has reached its spend limit",
            "Your workspace has reached its spend limit.",
        )
        for whitespace in ("", " \t\r\n")
    ],
)
def test_anthropic_spend_limit_fault_records_a_received_response_without_raw_data(
    tmp_path: Path, message: str, surrounding_whitespace: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"{surrounding_whitespace}{message}{surrounding_whitespace}"
                    ),
                },
            },
            headers={
                "request-id": SYNTHETIC_REQUEST_ID,
                "x-synthetic-secret": SYNTHETIC_PROVIDER_SECRET,
            },
            request=request,
        )

    client = anthropic.Anthropic(
        api_key="test-key-not-a-credential",
        base_url=ANTHROPIC_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    path = tmp_path / "anthropic-spend-limit.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="anthropic-haiku-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "7" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))
    recorder.abandon_open_call()
    recorder.finalize_excluded("provider_transport")
    recorder.close()

    audit = read_execution_ledger(path, require_complete=True)
    attempt = audit.calls[0].attempts[0]
    sdk_error = raised.value.__cause__
    assert isinstance(sdk_error, anthropic.BadRequestError)
    assert sdk_error.status_code == 400
    assert sdk_error.type == "invalid_request_error"
    assert sdk_error.request_id == SYNTHETIC_REQUEST_ID
    assert raised.value.fault == "provider_spend_limit"
    assert attempt.fault == "provider_spend_limit"
    assert attempt.http_status == 400
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert attempt.response_model is None
    assert attempt.response_id_digest_sha256 is None
    assert attempt.response_stop_classification is None
    assert attempt.response_normalized_digest_sha256 is None
    assert len(requests) == 1
    persisted = path.read_text(encoding="utf-8")
    assert SYNTHETIC_PROVIDER_SECRET not in persisted
    assert SYNTHETIC_REQUEST_ID not in persisted


def test_anthropic_invalid_request_stays_ambiguous_and_records_the_response() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": SYNTHETIC_PROVIDER_SECRET,
                },
            },
            headers={"request-id": SYNTHETIC_REQUEST_ID},
            request=request,
        )

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))

    attempt = lane.last_telemetry().attempts[0]
    assert raised.value.fault == "provider_invalid_request_or_spend_limit"
    assert attempt.fault == "provider_invalid_request_or_spend_limit"
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert calls == 1
    assert SYNTHETIC_PROVIDER_SECRET not in str(raised.value)
    assert SYNTHETIC_REQUEST_ID not in str(raised.value)


@pytest.mark.parametrize(
    ("body", "raw_content", "request_id"),
    [
        (
            {"type": "error", "error": {"type": "future_error", "message": "x"}},
            None,
            None,
        ),
        ({"type": "error"}, None, "malformed request id"),
        (None, b"not-json-synthetic-secret-91c2", "r" * 1024),
        (
            {"type": "error", "error": {"type": 7, "message": "x"}},
            None,
            "credential-shaped-synthetic-secret-91c2",
        ),
    ],
)
def test_anthropic_unrecognized_bad_requests_fail_closed_without_raw_fallback(
    tmp_path: Path, body: Any, raw_content: bytes | None, request_id: str | None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {} if request_id is None else {"request-id": request_id}
        if raw_content is not None:
            return httpx.Response(
                400,
                content=raw_content,
                headers=headers,
                request=request,
            )
        return httpx.Response(400, json=body, headers=headers, request=request)

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    path = tmp_path / "anthropic-unclassified.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="anthropic-haiku-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "8" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))
    recorder.abandon_open_call()
    recorder.finalize_excluded("provider_transport")
    recorder.close()

    attempt = read_execution_ledger(path, require_complete=True).calls[0].attempts[0]
    assert raised.value.fault == "provider_bad_request_unclassified"
    assert attempt.fault == "provider_bad_request_unclassified"
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert raw_content is None or raw_content.decode("ascii") not in str(raised.value)
    assert request_id is None or request_id not in str(raised.value)
    persisted = path.read_text(encoding="utf-8")
    assert raw_content is None or raw_content.decode("ascii") not in persisted
    assert request_id is None or request_id not in persisted


@pytest.mark.parametrize(
    "message",
    [
        "your organization has reached its spend limit.",
        "Your organization has reached its spend limit today.",
        "Your organization has reached its spend limit. synthetic-secret-91c2",
        "Your organization has almost reached its spend limit.",
    ],
)
def test_anthropic_spend_limit_allowlist_rejects_near_misses(message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": message},
            },
            request=request,
        )

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))

    assert raised.value.fault == "provider_invalid_request_or_spend_limit"
    assert message not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "fault"),
    [
        (401, "provider_authentication"),
        (403, "provider_authentication"),
        (404, "provider_configuration"),
        (413, "provider_request_rejected"),
        (429, "provider_rate_limited"),
        (500, "provider_server_error"),
        (503, "provider_server_error"),
    ],
)
def test_every_anthropic_http_error_records_that_a_response_arrived(
    status: int, fault: str
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={"type": "error", "error": {"type": "api_error", "message": "x"}},
            request=request,
        )

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))

    attempt = lane.last_telemetry().attempts[0]
    assert raised.value.fault == fault
    assert attempt.fault == fault
    assert attempt.http_status == status
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert calls == 1


def test_anthropic_transport_failure_still_records_no_received_response() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("synthetic transport failure", request=request)

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))

    attempt = lane.last_telemetry().attempts[0]
    assert raised.value.fault == "provider_network_error"
    assert attempt.response_received is False
    assert attempt.usage_reported is False
    assert calls == 1


def test_dispatch_observer_runs_once_at_the_real_sdk_boundary() -> None:
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        events.append("sdk_create")
        return httpx.Response(200, json=response_body(HAIKU_4_5_MODEL), request=request)

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        before_dispatch=lambda: events.append("observer"),
    )

    lane.send(model_request(HAIKU_4_5_MODEL))

    assert events == ["observer", "sdk_create"]
    assert lane.last_telemetry().attempt_count == 1


def test_dispatch_observer_exception_is_a_zero_attempt_pre_dispatch_failure() -> None:
    sdk_calls = 0
    sentinel = RuntimeError("synthetic observer failure")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sdk_calls
        sdk_calls += 1
        return httpx.Response(200, json=response_body(HAIKU_4_5_MODEL), request=request)

    def observer() -> None:
        raise sentinel

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        before_dispatch=observer,
    )

    with pytest.raises(RuntimeError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))

    assert raised.value is sentinel
    assert sdk_calls == 0
    assert lane.last_telemetry().attempt_count == 0
    assert lane.last_usage().input_tokens is None
    assert lane.last_usage().output_tokens is None


@pytest.mark.parametrize("invalid", [False, True, 0, -1, object()])
def test_dispatch_observer_must_be_an_exact_zero_argument_callable(invalid: Any) -> None:
    with pytest.raises(AnthropicConfigurationError):
        AnthropicMessagesTransport(
            model=HAIKU_4_5_MODEL,
            client=Wire(response_body(HAIKU_4_5_MODEL)).client(),
            deadline_seconds=30.0,
            before_dispatch=invalid,
        )


def test_dispatch_observer_can_stop_at_fifty_without_beginning_call_fifty_one() -> None:
    sdk_calls = 0
    observed = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sdk_calls
        sdk_calls += 1
        return httpx.Response(200, json=response_body(HAIKU_4_5_MODEL), request=request)

    def observer() -> None:
        nonlocal observed
        if observed >= 50:
            raise RuntimeError("synthetic fifty-call bound")
        observed += 1

    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=anthropic.Anthropic(
            api_key="test-key-not-a-credential",
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        before_dispatch=observer,
    )
    request = model_request(HAIKU_4_5_MODEL)

    for _ in range(50):
        lane.send(request)
    with pytest.raises(RuntimeError, match="fifty-call bound"):
        lane.send(request)

    assert observed == sdk_calls == 50
    assert lane.last_telemetry().attempt_count == 0


@pytest.mark.parametrize(
    "body",
    [
        response_body(HAIKU_4_5_MODEL, unexpected="provider prose"),
        response_body(HAIKU_4_5_MODEL, input_tokens="137"),
        response_body(HAIKU_4_5_MODEL, output_tokens=True),
        response_body("another-model"),
        response_body(
            HAIKU_4_5_MODEL,
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": "complete",
                    "input": ["not", "an", "object"],
                }
            ],
        ),
    ],
)
def test_anthropic_malformed_extra_coerced_or_wrong_identity_fails_closed(
    body: dict[str, Any],
) -> None:
    wire = Wire(body)
    transport = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=wire.client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError) as raised:
        transport.send(model_request(HAIKU_4_5_MODEL))

    assert raised.value.fault == "provider_response_invalid"
    assert "provider prose" not in str(raised.value)
    assert transport.last_telemetry().attempt_count == 1


@pytest.mark.parametrize(
    "cache_field", ["cache_creation_input_tokens", "cache_read_input_tokens"]
)
def test_nonzero_anthropic_cache_usage_fails_closed_without_a_pricing_policy(
    cache_field: str,
) -> None:
    body = response_body(HAIKU_4_5_MODEL)
    body["usage"][cache_field] = 1
    wire = Wire(body)
    transport = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=wire.client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )

    with pytest.raises(AdapterProviderError, match="cache") as raised:
        transport.send(model_request(HAIKU_4_5_MODEL))

    assert raised.value.fault == "provider_response_invalid"


@pytest.mark.parametrize("value", [True, False, 0.0, "0", -1, 2**53])
@pytest.mark.parametrize(
    "cache_field", ["cache_creation_input_tokens", "cache_read_input_tokens"]
)
def test_anthropic_cache_usage_requires_an_exact_bounded_integer(
    cache_field: str, value: Any
) -> None:
    body = response_body(HAIKU_4_5_MODEL)
    body["usage"][cache_field] = value
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=Wire(body).client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request(HAIKU_4_5_MODEL))
    assert raised.value.fault == "provider_response_invalid"


@pytest.mark.parametrize(
    "cache_field", ["cache_creation_input_tokens", "cache_read_input_tokens"]
)
def test_anthropic_zero_exact_cache_usage_is_admitted(cache_field: str) -> None:
    body = response_body(HAIKU_4_5_MODEL)
    body["usage"][cache_field] = 0
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=Wire(body).client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    assert lane.send(model_request(HAIKU_4_5_MODEL)).output_tokens == 29


@pytest.mark.parametrize("ceiling", [2048, 4096.0, True])
def test_anthropic_noncanonical_output_ceiling_refuses_before_dispatch(
    ceiling: Any,
) -> None:
    wire = Wire(response_body(HAIKU_4_5_MODEL))
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=wire.client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    request = replace(model_request(HAIKU_4_5_MODEL), max_output_tokens=ceiling)

    with pytest.raises(ModelRequestRefused, match="output-token ceiling"):
        lane.send(request)

    assert wire.requests == []


def test_anthropic_output_limit_returns_canonical_response_and_flushes_coherent_evidence(
    tmp_path: Path,
) -> None:
    body = response_body(
        HAIKU_4_5_MODEL,
        content=[{"type": "text", "text": "partial provider output"}],
        stop_reason="max_tokens",
    )
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=Wire(body).client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    path = tmp_path / "anthropic-output-limit.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="anthropic-haiku-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)

    answer = lane.send(model_request(HAIKU_4_5_MODEL))
    recorder.abandon_open_call()

    assert answer.stop_reason == TRUNCATED_STOP_REASON == "max_tokens"
    assert answer.tool_calls == ()
    assert answer.output_tokens == 29
    attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert attempt.outcome == "response"
    assert attempt.fault is None
    assert attempt.response_received is True
    assert attempt.usage_reported is True
    assert attempt.input_tokens == 137
    assert attempt.output_tokens == 29
    assert attempt.response_model == HAIKU_4_5_MODEL
    assert attempt.response_stop_classification == TRUNCATED_STOP_REASON
    assert attempt.response_normalized_digest_sha256 is not None


@pytest.mark.parametrize(
    ("case", "valid", "expected_digest"),
    [
        ("lone-surrogate", False, None),
        (
            "maximum",
            True,
            hashlib.sha256(b"a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS).hexdigest(),
        ),
        ("over-maximum", False, None),
        ("empty", True, None),
        ("absent", True, None),
        ("invalid-type", False, None),
    ],
)
def test_anthropic_response_id_is_admitted_before_settlement_and_recording(
    tmp_path: Path, case: str, valid: bool, expected_digest: str | None
) -> None:
    body = response_body(HAIKU_4_5_MODEL)
    if case == "lone-surrogate":
        body["id"] = "\ud800"
    elif case == "maximum":
        body["id"] = "a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS
    elif case == "over-maximum":
        body["id"] = "a" * (MAX_PROVIDER_RESPONSE_ID_CHARACTERS + 1)
    elif case == "empty":
        body["id"] = ""
    elif case == "absent":
        body.pop("id")
    else:
        body["id"] = 7
    wire = Wire(body)
    lane = AnthropicMessagesTransport(
        model=HAIKU_4_5_MODEL,
        client=wire.client(),
        deadline_seconds=30.0,
        clock=lambda: 0.0,
    )
    path = tmp_path / f"anthropic-response-id-{case}.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="anthropic-haiku-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "3" * 32)
    lane.attach_recorder(recorder)

    if valid:
        lane.send(model_request(HAIKU_4_5_MODEL))
    else:
        with pytest.raises(AdapterProviderError) as raised:
            lane.send(model_request(HAIKU_4_5_MODEL))
        assert raised.value.fault == "provider_response_invalid"
        assert lane.last_usage().input_tokens is None
        assert lane.last_telemetry().attempts[0].usage_reported is False
    recorder.abandon_open_call()

    attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert attempt.response_id_digest_sha256 == expected_digest
    if not valid:
        assert attempt.response_model is None
        assert attempt.response_stop_classification is None
        assert attempt.response_normalized_digest_sha256 is None
        persisted = path.read_text(encoding="utf-8")
        assert "ud800" not in persisted
        assert "UnicodeEncodeError" not in persisted


def test_anthropic_provider_identity_is_read_from_the_transport() -> None:
    model = HAIKU_4_5_MODEL
    transport = AnthropicMessagesTransport(
        model=model,
        client=Wire(response_body(model)).client(),
        deadline_seconds=30.0,
    )

    identity = provider_identity_for(transport)

    assert identity.provider == "anthropic"
    assert identity.api == "messages"
    assert identity.model == model
    assert identity.request_mapping == LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION


def test_anthropic_historical_mapping_identity_is_not_a_live_alias() -> None:
    assert (
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V1
        == "lifecycle_anthropic_messages_model_request_v1"
    )
    assert (
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION
        == "lifecycle_anthropic_messages_model_request_v4"
    )
    assert (
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION_V1
        != LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION
    )


def test_anthropic_action_and_retrieval_tools_are_exact_lifecycle_projections() -> None:
    """Anthropic receives a root-safe projection of the authoritative schemas."""
    openai_tools = outcome_tools()
    anthropic_tools = anthropic_outcome_tools()

    assert [tool["name"] for tool in anthropic_tools] == list(AGENT_TOOL_NAMES)
    assert [tool["name"] for tool in anthropic_tools] == [
        tool["name"] for tool in openai_tools
    ]
    assert len(anthropic_tools) == 6
    for authored, projected in zip(openai_tools, anthropic_tools, strict=True):
        schema = projected["input_schema"]
        assert schema["type"] == "object"
        assert not ({"oneOf", "anyOf", "allOf"} & schema.keys())
        if not ({"oneOf", "anyOf", "allOf"} & authored["parameters"].keys()):
            assert schema == authored["parameters"]
    assert set(tool_schema()) == set(AGENT_TOOL_NAMES)
    assert anthropic_tools[1]["name"] == "wait"
    assert "anyOf" in openai_tools[1]["parameters"]
    assert anthropic_tools[-1]["name"] == "retrieve"


def test_lowered_anthropic_schemas_admit_representative_valid_core_calls() -> None:
    samples = {
        "act": {"action_type": "dispatch", "payload": {"id": 1}},
        "wait": [
            {"reason": "await reply", "wake_on": ["message"]},
            {"reason": "deadline", "fallback_after_minutes": 5},
            {
                "reason": "either",
                "wake_on": ["message"],
                "fallback_after_minutes": 5,
            },
        ],
        "ask": {
            "recipient_actor_id": "tenant",
            "message_fixture_id": "fixture",
            "wait": {"reason": "await reply", "wake_on": ["message"]},
        },
        "escalate": {"checkpoint_id": "approval", "exception_type": "policy"},
        "complete": {},
        "retrieve": {"requests": [{"tool": "read_record", "arguments": {"id": 1}}]},
    }
    schemas = {tool["name"]: tool["input_schema"] for tool in anthropic_outcome_tools()}

    for name, values in samples.items():
        for value in values if isinstance(values, list) else [values]:
            Draft202012Validator(schemas[name]).validate(value)


def test_local_parser_remains_the_exact_gate_after_wait_schema_lowering() -> None:
    from operatebench.agents.model import ModelToolArgumentsMalformed, parse_tool_call

    lowered = anthropic_outcome_tools()[1]["input_schema"]
    invalid = {"reason": "never ends"}
    unknown = {"reason": "event", "wake_on": ["message"], "surprise": True}

    Draft202012Validator(lowered).validate(invalid)
    assert isinstance(
        parse_tool_call(ToolCall("wait", invalid)), ModelToolArgumentsMalformed
    )
    assert isinstance(
        parse_tool_call(ToolCall("wait", unknown)), ModelToolArgumentsMalformed
    )


def test_schema_lowering_unions_branch_properties_and_intersects_required() -> None:
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"shared": {"type": "string"}, "left": {"type": "integer"}},
                "required": ["shared", "left"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "shared": {"type": "string"},
                    "right": {"type": "boolean"},
                },
                "required": ["shared", "right"],
                "additionalProperties": False,
            },
        ]
    }

    assert lower_anthropic_input_schema(schema) == {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "boolean"},
            "shared": {"type": "string"},
        },
        "required": ["shared"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "schema",
    [
        {
            "oneOf": [
                {"type": "object", "properties": {"x": {"type": "boolean"}}},
                {"type": "object", "properties": {"x": {"type": "integer"}}},
            ]
        },
        {"anyOf": [{"type": "object"}, {"type": "string"}]},
        {
            "allOf": [
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {"type": "object", "properties": {"x": {"type": "number"}}},
            ]
        },
    ],
)
def test_schema_lowering_fails_closed_on_unrepresentable_branches(
    schema: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Anthropic input schema"):
        lower_anthropic_input_schema(schema)


def test_schema_lowering_preserves_nested_combinators_and_does_not_alias_input() -> None:
    nested = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"value": nested},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"other": {"type": "integer"}},
                "additionalProperties": False,
            },
        ]
    }
    before = json.loads(json.dumps(schema))

    lowered = lower_anthropic_input_schema(schema)
    lowered["properties"]["value"]["anyOf"][0]["type"] = "number"

    assert schema == before
    assert schema["oneOf"][0]["properties"]["value"] == nested


def test_schema_lowering_resolves_local_root_refs_and_is_branch_order_deterministic() -> (
    None
):
    branches = [
        {"$ref": "#/$defs/left"},
        {"$ref": "#/$defs/right"},
    ]
    schema = {
        "$defs": {
            "left": {"type": "object", "properties": {"a": {"type": "string"}}},
            "right": {"type": "object", "properties": {"b": {"type": "integer"}}},
        },
        "anyOf": branches,
    }
    reversed_schema = {**schema, "anyOf": list(reversed(branches))}

    assert lower_anthropic_input_schema(schema) == lower_anthropic_input_schema(
        reversed_schema
    )


def test_schema_lowering_refuses_cyclic_and_overdeep_refs() -> None:
    cyclic = {"$defs": {"loop": {"$ref": "#/$defs/loop"}}, "$ref": "#/$defs/loop"}
    deep: dict[str, Any] = {"type": "object"}
    for _ in range(40):
        deep = {"$defs": {"next": deep}, "$ref": "#/$defs/next"}

    with pytest.raises(ValueError, match="Anthropic input schema"):
        lower_anthropic_input_schema(cyclic)
    with pytest.raises(ValueError, match="Anthropic input schema"):
        lower_anthropic_input_schema(deep)


@pytest.mark.parametrize(
    "client",
    [
        Wire(response_body(HAIKU_4_5_MODEL)).client(max_retries=1),
        Wire(response_body(HAIKU_4_5_MODEL)).client(base_url="https://provider.invalid"),
    ],
)
def test_anthropic_injected_client_mismatch_is_refused(
    client: anthropic.Anthropic,
) -> None:
    with pytest.raises(AnthropicConfigurationError):
        AnthropicMessagesTransport(
            model=HAIKU_4_5_MODEL,
            client=client,
            deadline_seconds=30.0,
        )
