"""Lifecycle xAI OpenAI-compatible Chat Completions vertical, offline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from jsonschema import Draft202012Validator

from operatebench.agents.evidence import (
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    normalized_response_digest,
    provider_identity_for,
)
from operatebench.agents.lifecycle_contract import (
    ModelRequestRefused,
    UndecodableArguments,
)
from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    TRUNCATED_STOP_REASON,
    ModelAgent,
    ModelToolArgumentsMalformed,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import ModelRequest, ToolCall
from operatebench.agents.xai_chat_completions import (
    LIFECYCLE_XAI_REQUEST_MAPPING_VERSION,
    XAIOpenAICompatChatCompletionsTransport,
    build_model_payload,
    xai_outcome_tools,
)
from operatebench.core.protocol import AgentObservation
from operatebench.execution_ledger import (
    ABORTED_CODE,
    TERMINAL_ABORTED,
    read_execution_ledger,
)
from operatebench.providers.cost import CostGuard
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import MAX_PROVIDER_RESPONSE_ID_CHARACTERS
from operatebench.providers.xai_openai_compat import (
    GROK_4_5_MODEL,
    XAI_API_COMPATIBILITY,
    XAI_BASE_URL,
    XAI_IMPLEMENTATION,
    XAIConfigurationError,
)
from tests.execution_ledger_fixtures import DIGEST_A, controls


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


def model_request() -> ModelRequest:
    class Never:
        def send(self, request: ModelRequest) -> Any:
            raise AssertionError("request construction must not dispatch")

    return ModelAgent(Never(), model=GROK_4_5_MODEL, agent_id="diagnostic").build_request(
        observation()
    )


def response_body(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": GROK_4_5_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": "complete",
                                "arguments": json.dumps(
                                    {"reason": "done", "evidence_refs": []}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 137, "completion_tokens": 29, "total_tokens": 166},
    }
    body.update(changes)
    return body


class Wire:
    def __init__(
        self,
        body: dict[str, Any],
        *,
        base_url: str = XAI_BASE_URL,
        max_retries: Any = 0,
        organization: str | None = None,
        project: str | None = None,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, str] | None = None,
        http_client_options: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
    ) -> None:
        self.body = body
        self.raw_body = raw_body
        self.requests: list[httpx.Request] = []
        self.http_client = httpx.Client(
            transport=httpx.MockTransport(self.handler),
            **(http_client_options or {}),
        )
        self.client = openai.OpenAI(
            api_key="offline-placeholder-not-a-credential",
            base_url=base_url,
            max_retries=max_retries,
            organization=organization,
            project=project,
            default_headers=default_headers,
            default_query=default_query,
            http_client=self.http_client,
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raw_body is not None:
            return httpx.Response(200, content=self.raw_body, request=request)
        return httpx.Response(200, json=self.body, request=request)


def mutating_event_hook(wire: Wire, hook_type: str, secret: str = "event-hook-secret"):
    if hook_type == "request":

        def request_hook(request: httpx.Request) -> None:
            request.url = httpx.URL("https://redirect.invalid/v1/chat/completions")
            request.headers["x-hook-state"] = secret

        return request_hook

    def response_hook(response: httpx.Response) -> None:
        response._content = json.dumps(response_body(model="hook-mutated-model")).encode()

    return response_hook


def transport(
    wire: Wire, *, cost_guard: CostGuard | None = None
) -> XAIOpenAICompatChatCompletionsTransport:
    return XAIOpenAICompatChatCompletionsTransport(
        model=GROK_4_5_MODEL,
        client=wire.client,
        deadline_seconds=30.0,
        clock=lambda: 0.0,
        cost_guard=cost_guard,
    )


def test_real_openai_sdk_serializes_exact_xai_compat_request() -> None:
    wire = Wire(response_body())
    request = model_request()
    lane = transport(wire)

    answer = lane.send(request)

    assert answer.tool_calls == (
        ToolCall("complete", {"reason": "done", "evidence_refs": []}),
    )
    assert len(wire.requests) == 1
    sent = json.loads(wire.requests[0].content)
    assert sent == build_model_payload(request, model=GROK_4_5_MODEL)
    assert wire.requests[0].url == f"{XAI_BASE_URL}/chat/completions"
    assert sent["model"] == GROK_4_5_MODEL
    assert sent["max_tokens"] == MAX_OUTPUT_TOKENS == 4096
    assert sent["tool_choice"] == "required"
    assert sent["parallel_tool_calls"] is False
    assert sent["temperature"] == 0.0
    assert "reasoning_effort" not in sent and "top_p" not in sent
    assert [tool["function"] for tool in sent["tools"]] == xai_outcome_tools()
    assert wire.requests[0].headers["authorization"].startswith("Bearer ")
    assert wire.requests[0].headers["x-stainless-raw-response"] == "true"
    assert lane.last_usage().input_tokens == 137
    assert lane.last_usage().output_tokens == answer.output_tokens == 29
    assert lane.settings["api_compatibility"] == XAI_API_COMPATIBILITY
    assert lane.settings["sdk"] == "openai"
    assert lane.settings["sdk_max_retries"] == 0
    assert lane.settings["request_mapping"] == LIFECYCLE_XAI_REQUEST_MAPPING_VERSION
    identity = provider_identity_for(lane)
    assert (identity.provider, identity.api, identity.model) == (
        "xai",
        "chat_completions",
        GROK_4_5_MODEL,
    )
    assert lane.settings["implementation"] == XAI_IMPLEMENTATION
    assert identity.implementation == XAI_IMPLEMENTATION


def test_xai_ledger_header_binds_the_canonical_implementation(tmp_path: Path) -> None:
    lane = transport(Wire(response_body()))
    path = tmp_path / "xai-identity.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)

    header = read_execution_ledger(path).header
    assert header.provider.implementation == XAI_IMPLEMENTATION
    assert header.provider.settings["implementation"] == XAI_IMPLEMENTATION


@pytest.mark.parametrize(
    "wire",
    [
        Wire(response_body(), base_url="https://api.openai.com/v1"),
        Wire(response_body(), base_url="https://proxy.invalid/v1"),
        Wire(response_body(), max_retries=1),
        Wire(response_body(), organization="org-test"),
        Wire(response_body(), project="proj-test"),
    ],
)
def test_injected_client_identity_mismatch_refuses_before_dispatch(wire: Wire) -> None:
    with pytest.raises(XAIConfigurationError):
        transport(wire)
    assert wire.requests == []


@pytest.mark.parametrize(
    "client_options",
    [
        {"default_headers": {"Authorization": "Bearer alternate-placeholder"}},
        {"default_headers": {"x-account-scope": "alternate-placeholder"}},
        {"default_query": {"tenant": "alternate-placeholder"}},
    ],
)
def test_custom_client_headers_and_query_refuse_at_construction_without_dispatch(
    client_options: dict[str, Any],
) -> None:
    wire = Wire(response_body(), **client_options)
    with pytest.raises(XAIConfigurationError) as raised:
        transport(wire)
    assert "alternate-placeholder" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize(
    "http_client_options",
    [
        {"headers": {"x-smuggled-route": "tenant-b"}},
        {"params": {"tenant": "tenant-b"}},
        {"auth": ("tenant-b", "not-a-credential")},
        {"cookies": {"session": "tenant-b"}},
    ],
)
def test_underlying_httpx_client_defaults_refuse_at_construction_without_dispatch(
    http_client_options: dict[str, Any],
) -> None:
    wire = Wire(response_body(), http_client_options=http_client_options)

    with pytest.raises(XAIConfigurationError) as raised:
        transport(wire)

    assert "tenant-b" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize("hook_type", ["request", "response"])
def test_httpx_event_hooks_refuse_at_construction_without_dispatch(
    hook_type: str,
) -> None:
    wire = Wire(response_body())
    wire.http_client.event_hooks[hook_type].append(mutating_event_hook(wire, hook_type))

    with pytest.raises(XAIConfigurationError) as raised:
        transport(wire)

    assert "event-hook-secret" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        {"request": [], "response": [], "unexpected": []},
        {"request": []},
        {"request": (), "response": []},
        {"request": [], "response": ()},
    ],
)
def test_malformed_httpx_event_hook_state_refuses_without_reading_values(
    malformed: Any,
) -> None:
    wire = Wire(response_body())
    wire.http_client._event_hooks = malformed

    with pytest.raises(XAIConfigurationError):
        transport(wire)

    assert wire.requests == []


def test_missing_httpx_event_hook_state_refuses() -> None:
    wire = Wire(response_body())
    del wire.http_client._event_hooks

    with pytest.raises(XAIConfigurationError):
        transport(wire)

    assert wire.requests == []


@pytest.mark.parametrize("mutation", ["headers", "query"])
def test_post_construction_client_customization_refuses_before_dispatch(
    mutation: str,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    if mutation == "headers":
        wire.client._custom_headers = {"unrecorded": "post-construction-secret"}
    else:
        wire.client._custom_query = {"unrecorded": "post-construction-secret"}

    with pytest.raises(XAIConfigurationError) as raised:
        lane.send(model_request())

    assert "post-construction-secret" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize("mutation", ["headers", "query", "auth", "cookies"])
def test_post_construction_underlying_httpx_mutation_refuses_before_dispatch(
    mutation: str,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    if mutation == "headers":
        wire.http_client.headers["x-smuggled-route"] = "post-construction-secret"
    elif mutation == "query":
        wire.http_client.params = {"tenant": "post-construction-secret"}
    elif mutation == "auth":
        wire.http_client.auth = ("post-construction-secret", "not-a-credential")
    else:
        wire.http_client.cookies.set("session", "post-construction-secret")

    with pytest.raises(XAIConfigurationError) as raised:
        lane.send(model_request())

    assert "post-construction-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert wire.requests == []


@pytest.mark.parametrize(
    "mutation", ["headers", "query", "auth", "cookies", "retry", "request", "response"]
)
def test_entry_client_mutation_records_an_aborted_local_refusal(
    tmp_path: Path, mutation: str
) -> None:
    wire = Wire(response_body())
    guard = LifecycleCostGuard(
        policy=LifecyclePricingPolicy(
            policy_id="xai-entry-refusal-test-v1",
            input_usd_per_mtok=Decimal("1.25"),
            output_usd_per_mtok=Decimal("10"),
            rate_source=RATE_SOURCE_OPERATOR,
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=100_000,
    )
    lane = transport(wire, cost_guard=guard)
    path = tmp_path / f"entry-{mutation}.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
        guard=guard,
    )
    recorder.begin(operation_instance_id="opinst_" + "3" * 32)
    lane.attach_recorder(recorder)
    if mutation == "headers":
        wire.http_client.headers["x-smuggled-route"] = "entry-refusal-secret"
    elif mutation == "query":
        wire.http_client.params = {"tenant": "entry-refusal-secret"}
    elif mutation == "auth":
        wire.http_client.auth = ("entry-refusal-secret", "not-a-credential")
    elif mutation == "cookies":
        wire.http_client.cookies.set("session", "entry-refusal-secret")
    elif mutation in {"request", "response"}:
        wire.http_client.event_hooks[mutation].append(
            mutating_event_hook(wire, mutation, "entry-refusal-secret")
        )
    else:
        wire.client.max_retries = 1

    with pytest.raises(XAIConfigurationError) as raised:
        lane.send(model_request())
    recorder.finalize_excluded("provider_transport")

    assert "entry-refusal-secret" not in str(raised.value)
    assert "not-a-credential" not in str(raised.value)
    assert wire.requests == []
    assert lane.last_usage().as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }
    telemetry = lane.last_telemetry()
    assert telemetry.attempts == ()
    assert telemetry.turn_latency_seconds == 0.0
    assert telemetry.terminal_reason == "pre_dispatch_refused"
    ledger = read_execution_ledger(path, require_complete=True)
    assert ledger.calls == ()
    assert ledger.status == TERMINAL_ABORTED
    assert ledger.terminal is not None
    assert ledger.terminal.kind == TERMINAL_ABORTED
    assert ledger.terminal.exclusion_code == ABORTED_CODE
    assert ledger.totals.provider_calls == 0
    assert ledger.totals.attempts == 0
    assert ledger.totals.input_tokens == 0
    assert ledger.totals.output_tokens == 0
    assert ledger.totals.fault_counts == {}
    assert guard.committed_usd == Decimal(0)
    assert guard.measured_usd == Decimal(0)
    assert guard.forfeited_usd == Decimal(0)
    assert guard.reserved_tokens == 0
    assert guard.take_events() == ()
    persisted = path.read_text(encoding="utf-8")
    assert "entry-refusal-secret" not in persisted
    assert "not-a-credential" not in persisted
    assert "provider_transport" not in persisted
    assert "response_model" not in persisted
    assert "response_stop_classification" not in persisted
    assert "response_normalized_digest_sha256" not in persisted


@pytest.mark.parametrize("mutation", ["retry", "request_hook"])
def test_entry_refusal_after_success_does_not_retain_usage_or_attempts(
    mutation: str,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    lane.send(model_request())
    assert lane.last_usage().input_tokens == 137
    assert lane.last_telemetry().attempt_count == 1
    if mutation == "retry":
        wire.client.max_retries = 1
    else:
        wire.http_client.event_hooks["request"].append(
            mutating_event_hook(wire, "request", "stale-usage-secret")
        )

    with pytest.raises(XAIConfigurationError):
        lane.send(model_request())

    assert len(wire.requests) == 1
    assert lane.last_usage().input_tokens is None
    assert lane.last_usage().output_tokens is None
    assert lane.last_telemetry().attempts == ()
    assert lane.last_telemetry().terminal_reason == "pre_dispatch_refused"


@pytest.mark.parametrize("mutation", ["headers", "cookies", "request", "response"])
def test_httpx_mutation_after_authorize_cancels_without_a_provider_attempt(
    tmp_path: Path, mutation: str
) -> None:
    wire = Wire(response_body())

    class MutatingGuard(LifecycleCostGuard):
        def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
            reservation = super().authorize(
                input_tokens_upper_bound=input_tokens_upper_bound
            )
            if mutation == "headers":
                wire.http_client.headers["x-smuggled-route"] = "final-dispatch-secret"
            elif mutation == "cookies":
                wire.http_client.cookies.set("session", "final-dispatch-secret")
            else:
                wire.http_client.event_hooks[mutation].append(
                    mutating_event_hook(wire, mutation, "final-dispatch-secret")
                )
            return reservation

    guard = MutatingGuard(
        policy=LifecyclePricingPolicy(
            policy_id="xai-pre-dispatch-test-v1",
            input_usd_per_mtok=Decimal("1.25"),
            output_usd_per_mtok=Decimal("10"),
            rate_source=RATE_SOURCE_OPERATOR,
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=100_000,
    )
    lane = transport(wire, cost_guard=guard)
    path = tmp_path / "pre-dispatch-refusal.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
        guard=guard,
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(XAIConfigurationError) as raised:
        lane.send(model_request())
    recorder.finalize_excluded("provider_transport")

    assert "final-dispatch-secret" not in str(raised.value)
    assert wire.requests == []
    telemetry = lane.last_telemetry()
    assert telemetry.attempt_count == 0
    assert telemetry.terminal_reason == "pre_dispatch_refused"
    ledger = read_execution_ledger(path, require_complete=True)
    assert ledger.calls == ()
    assert ledger.status == TERMINAL_ABORTED
    assert ledger.terminal is not None
    assert ledger.terminal.kind == TERMINAL_ABORTED
    assert ledger.terminal.exclusion_code == ABORTED_CODE
    assert ledger.totals.provider_calls == 0
    assert ledger.totals.attempts == 0
    assert ledger.totals.fault_counts == {}
    assert guard.committed_usd == Decimal(0)
    assert guard.measured_usd == Decimal(0)
    assert guard.forfeited_usd == Decimal(0)
    assert guard.reserved_tokens == 0
    assert guard.take_events() == ()
    assert "final-dispatch-secret" not in path.read_text(encoding="utf-8")
    assert "provider_deadline" not in path.read_text(encoding="utf-8")
    assert "provider_transport" not in path.read_text(encoding="utf-8")


def test_lone_surrogate_response_id_is_refused_before_usage_is_banked(
    tmp_path: Path,
) -> None:
    body = response_body(id="\ud800")
    wire = Wire(body, raw_body=json.dumps(body).encode("ascii"))
    guard = LifecycleCostGuard(
        policy=LifecyclePricingPolicy(
            policy_id="xai-response-id-test-v1",
            input_usd_per_mtok=Decimal("1.25"),
            output_usd_per_mtok=Decimal("10"),
            rate_source=RATE_SOURCE_OPERATOR,
        ),
        cap_usd=Decimal("5"),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=100_000,
    )
    lane = transport(wire, cost_guard=guard)
    path = tmp_path / "invalid-response-id.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
        guard=guard,
    )
    recorder.begin(operation_instance_id="opinst_" + "2" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    recorder.abandon_open_call()

    assert raised.value.fault == "provider_response_invalid"
    assert len(wire.requests) == 1
    assert lane.last_usage().as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }
    telemetry = lane.last_telemetry()
    assert telemetry.attempt_count == 1
    attempt = telemetry.attempts[0]
    assert attempt.fault == "provider_response_invalid"
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert attempt.cost_settlement == "forfeited"
    ledger_attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert ledger_attempt.input_tokens is None
    assert ledger_attempt.output_tokens is None
    assert ledger_attempt.response_model is None
    assert ledger_attempt.response_id_digest_sha256 is None
    assert ledger_attempt.response_stop_classification is None
    assert ledger_attempt.response_normalized_digest_sha256 is None
    assert ledger_attempt.cost_measured_usd is None
    persisted = path.read_text(encoding="utf-8")
    assert "ud800" not in persisted
    assert "UnicodeEncodeError" not in persisted


@pytest.mark.parametrize(
    ("case", "valid", "expected_digest"),
    [
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
def test_xai_response_id_boundaries_are_admitted_before_durable_evidence(
    tmp_path: Path, case: str, valid: bool, expected_digest: str | None
) -> None:
    body = response_body()
    if case == "maximum":
        body["id"] = "a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS
    elif case == "over-maximum":
        body["id"] = "a" * (MAX_PROVIDER_RESPONSE_ID_CHARACTERS + 1)
    elif case == "empty":
        body["id"] = ""
    elif case == "absent":
        body.pop("id")
    else:
        body["id"] = 7
    wire = Wire(body, raw_body=json.dumps(body).encode("ascii"))
    lane = transport(wire)
    path = tmp_path / f"xai-response-id-{case}.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "4" * 32)
    lane.attach_recorder(recorder)

    if valid:
        lane.send(model_request())
    else:
        with pytest.raises(AdapterProviderError) as raised:
            lane.send(model_request())
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


@pytest.mark.parametrize("max_retries", [False, 0.0])
def test_sdk_retry_count_requires_exact_zero_integer_at_construction(
    max_retries: Any,
) -> None:
    wire = Wire(response_body(), max_retries=max_retries)
    with pytest.raises(XAIConfigurationError):
        transport(wire)
    assert wire.requests == []


@pytest.mark.parametrize("max_retries", [False, 0.0])
def test_post_construction_retry_mutation_refuses_before_dispatch(
    max_retries: Any,
) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    wire.client.max_retries = max_retries
    with pytest.raises(XAIConfigurationError):
        lane.send(model_request())
    assert wire.requests == []


@pytest.mark.parametrize("ceiling", [2048, 4096.0, True])
def test_noncanonical_xai_output_ceiling_refuses_before_dispatch(ceiling: Any) -> None:
    wire = Wire(response_body())
    lane = transport(wire)
    request = replace(model_request(), max_output_tokens=ceiling)

    with pytest.raises(ModelRequestRefused, match="output-token ceiling"):
        lane.send(request)

    assert wire.requests == []


def fault_evidence(path: Path, body: dict[str, Any]) -> Any:
    lane = transport(Wire(body))
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    recorder.abandon_open_call()
    assert raised.value.fault == "provider_response_invalid"
    return read_execution_ledger(path).calls[0].attempts[0]


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [response_body()["choices"][0], response_body()["choices"][0]],
        [{**response_body()["choices"][0], "finish_reason": "stop"}],
        [
            {
                **response_body()["choices"][0],
                "message": {"role": "assistant", "content": None, "tool_calls": None},
            }
        ],
        [
            {
                **response_body()["choices"][0],
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        response_body()["choices"][0]["message"]["tool_calls"][0],
                        response_body()["choices"][0]["message"]["tool_calls"][0],
                    ],
                },
            }
        ],
        [
            {
                **response_body()["choices"][0],
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "wrong", "type": "custom"}],
                },
            }
        ],
        [
            {
                **response_body()["choices"][0],
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "missing", "type": "function"}],
                },
            }
        ],
    ],
    ids=(
        "zero-choices",
        "two-choices",
        "finish-reason",
        "no-call",
        "two-calls",
        "wrong-call",
        "missing-function",
    ),
)
def test_xai_protocol_faults_are_refused_before_usage_settlement(
    tmp_path: Path, choices: list[dict[str, Any]]
) -> None:
    attempt = fault_evidence(tmp_path / "fault.ndjson", response_body(choices=choices))
    assert attempt.outcome == "fault"
    assert attempt.fault == "provider_response_invalid"
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert attempt.input_tokens is None
    assert attempt.output_tokens is None
    assert attempt.response_model is None
    assert attempt.response_stop_classification is None
    assert attempt.response_normalized_digest_sha256 is None


def test_undecodable_arguments_are_distinct_from_valid_empty_input() -> None:
    invalid_text = "not-json-secret-provider-text"
    invalid = response_body()
    invalid["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        invalid_text
    )
    empty = response_body()
    empty["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{}"

    invalid_answer = transport(Wire(invalid)).send(model_request())
    empty_answer = transport(Wire(empty)).send(model_request())

    assert isinstance(invalid_answer.tool_calls[0].arguments, UndecodableArguments)
    assert empty_answer.tool_calls[0].arguments == {}
    invalid_digest = normalized_response_digest(
        model=invalid_answer.model,
        stop_classification=invalid_answer.stop_reason,
        tool_calls=invalid_answer.tool_calls,
    )
    empty_digest = normalized_response_digest(
        model=empty_answer.model,
        stop_classification=empty_answer.stop_reason,
        tool_calls=empty_answer.tool_calls,
    )
    assert invalid_digest != empty_digest

    decision = ModelAgent(
        transport(Wire(invalid)), model=GROK_4_5_MODEL, agent_id="diagnostic"
    ).decide(observation())
    assert isinstance(decision, ModelToolArgumentsMalformed)
    assert invalid_text not in repr(decision)


@pytest.mark.parametrize(
    "body",
    [
        response_body(unexpected="raw provider prose"),
        response_body(model="wrong-model"),
        response_body(choices=[]),
        response_body(
            choices=[response_body()["choices"][0], response_body()["choices"][0]]
        ),
        response_body(
            usage={"prompt_tokens": True, "completion_tokens": 29, "total_tokens": 30}
        ),
        response_body(
            usage={"prompt_tokens": 137.0, "completion_tokens": 29, "total_tokens": 166}
        ),
        response_body(
            usage={"prompt_tokens": "137", "completion_tokens": 29, "total_tokens": 166}
        ),
        response_body(
            usage={"prompt_tokens": -1, "completion_tokens": 29, "total_tokens": 28}
        ),
        response_body(
            usage={
                "prompt_tokens": 2**53,
                "completion_tokens": 29,
                "total_tokens": 2**53 + 29,
            }
        ),
        response_body(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "SECRET_PROVIDER_TEXT"},
                }
            ]
        ),
        response_body(choices=[{"index": 0, "finish_reason": "stop", "message": None}]),
        response_body(
            choices=[{"index": 0, "finish_reason": "tool_calls", "message": None}]
        ),
    ],
)
def test_raw_typed_identity_cardinality_and_usage_fail_closed(
    body: dict[str, Any],
) -> None:
    wire = Wire(body)
    lane = transport(wire)
    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    assert raised.value.fault == "provider_response_invalid"
    assert "raw provider prose" not in str(raised.value)
    assert "SECRET_PROVIDER_TEXT" not in str(raised.value)


@pytest.mark.parametrize(
    "message",
    [None, {"role": "assistant", "content": "partial provider output"}],
)
def test_output_limit_returns_canonical_response_and_flushes_coherent_evidence(
    tmp_path: Path, message: dict[str, Any] | None
) -> None:
    body = response_body(
        choices=[{"index": 0, "finish_reason": "length", "message": message}]
    )
    lane = transport(Wire(body))
    path = tmp_path / "xai-output-limit.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)

    answer = lane.send(model_request())
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
    assert attempt.response_model == GROK_4_5_MODEL
    assert attempt.response_stop_classification == TRUNCATED_STOP_REASON
    assert attempt.response_normalized_digest_sha256 is not None


def test_null_message_is_a_provider_fault_in_durable_evidence(tmp_path: Path) -> None:
    body = response_body(
        choices=[{"index": 0, "finish_reason": "tool_calls", "message": None}]
    )
    lane = transport(Wire(body))
    path = tmp_path / "xai-null-message.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="xai-grok-4-5",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as raised:
        lane.send(model_request())
    recorder.abandon_open_call()

    assert raised.value.fault == "provider_response_invalid"
    attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert attempt.outcome == "fault"
    assert attempt.fault == "provider_response_invalid"
    assert attempt.response_received is True
    assert attempt.usage_reported is False
    assert attempt.input_tokens is None
    assert attempt.output_tokens is None
    assert attempt.response_model is None
    assert attempt.response_stop_classification is None
    assert attempt.response_normalized_digest_sha256 is None


def test_shared_lifecycle_schema_has_one_neutral_authority() -> None:
    from operatebench.agents import lifecycle_contract, openai_responses
    from operatebench.agents.anthropic_messages import (
        anthropic_outcome_tools,
        lower_anthropic_input_schema,
    )
    from operatebench.agents.model import parse_tool_call

    neutral = lifecycle_contract.outcome_tools()
    canonical_before_projection = json.loads(json.dumps(neutral))
    assert openai_responses.outcome_tools is lifecycle_contract.outcome_tools
    assert xai_outcome_tools() == [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "strict": False,
        }
        for tool in neutral
    ]

    anthropic = anthropic_outcome_tools()
    assert lifecycle_contract.outcome_tools() == canonical_before_projection
    assert [tool["name"] for tool in anthropic] == [tool["name"] for tool in neutral]
    for canonical, projected in zip(neutral, anthropic, strict=True):
        schema = projected["input_schema"]
        assert schema == lower_anthropic_input_schema(canonical["parameters"])
        assert schema.get("type") == "object"
        assert not ({"oneOf", "anyOf", "allOf"} & schema.keys())

    def schema_bytes(schema: dict[str, Any]) -> bytes:
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()

    canonical_by_name = {tool["name"]: tool["parameters"] for tool in neutral}
    anthropic_by_name = {tool["name"]: tool["input_schema"] for tool in anthropic}
    assert "anyOf" in canonical_by_name["wait"]
    assert {
        name
        for name in canonical_by_name
        if schema_bytes(anthropic_by_name[name]) != schema_bytes(canonical_by_name[name])
    } == {"wait"}

    canonical_wait = canonical_by_name["wait"]
    projected_wait = anthropic_by_name["wait"]
    branches = canonical_wait["anyOf"]
    assert projected_wait["properties"] == canonical_wait["properties"]
    assert set(projected_wait["required"]) == set.intersection(
        *(set(branch["required"]) for branch in branches)
    )
    assert projected_wait["additionalProperties"] is False
    for valid_wait in (
        {"reason": "event", "wake_on": ["message"]},
        {"reason": "deadline", "fallback_after_minutes": 5},
    ):
        Draft202012Validator(canonical_wait).validate(valid_wait)
        Draft202012Validator(projected_wait).validate(valid_wait)

    widened_only = {"reason": "no termination condition"}
    Draft202012Validator(projected_wait).validate(widened_only)
    assert isinstance(
        parse_tool_call(ToolCall("wait", widened_only)), ModelToolArgumentsMalformed
    )


def test_all_lifecycle_lanes_consume_the_canonical_truncation_stop_reason() -> None:
    from operatebench.agents import (
        anthropic_messages,
        openai_responses,
        xai_chat_completions,
    )

    assert openai_responses.TRUNCATED_STOP_REASON == TRUNCATED_STOP_REASON
    assert anthropic_messages.TRUNCATED_STOP_REASON == TRUNCATED_STOP_REASON
    assert xai_chat_completions.TRUNCATED_STOP_REASON == TRUNCATED_STOP_REASON
