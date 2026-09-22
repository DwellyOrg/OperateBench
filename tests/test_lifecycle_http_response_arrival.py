"""Cross-provider HTTP response-arrival semantics over each real SDK."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest
from mistralai.client.errors import MistralError

from operatebench.agents.anthropic_messages import AnthropicMessagesTransport
from operatebench.agents.evidence import (
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    provider_identity_for,
)
from operatebench.agents.model import ModelAgent
from operatebench.agents.transport import ModelRequest
from operatebench.core.protocol import AgentObservation
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.anthropic_messages import ANTHROPIC_BASE_URL, HAIKU_4_5_MODEL
from operatebench.providers.faults import AdapterProviderError
from tests.execution_ledger_fixtures import DIGEST_A, controls
from tests.openai_transport import RecordingTransport, error_body
from tests.test_lifecycle_mistral_chat import Wire as MistralWire
from tests.test_lifecycle_mistral_chat import model_request as mistral_request
from tests.test_lifecycle_mistral_chat import transport as mistral_transport
from tests.test_lifecycle_openai_bridge import MODEL as OPENAI_MODEL
from tests.test_lifecycle_openai_bridge import transport_for as openai_transport
from tests.test_lifecycle_xai_chat_completions import Wire as XAIWire
from tests.test_lifecycle_xai_chat_completions import model_request as xai_request
from tests.test_lifecycle_xai_chat_completions import transport as xai_transport

_STATUS_CASES = (
    (400, "provider_request_rejected"),
    (401, "provider_authentication"),
    (403, "provider_authentication"),
    (404, "provider_configuration"),
    (408, "provider_timeout"),
    (429, "provider_rate_limited"),
    (500, "provider_server_error"),
    (503, "provider_server_error"),
)


def _request(model: str) -> ModelRequest:
    class Never:
        def send(self, request: ModelRequest) -> Any:
            raise AssertionError("request construction must not dispatch")

    observation = AgentObservation(
        now="2025-01-01T09:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_" + "1" * 32,
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=[],
        last_rejection=None,
    )
    return ModelAgent(Never(), model=model, agent_id="diagnostic").build_request(
        observation
    )


def _anthropic_lane(
    step: int | BaseException,
) -> tuple[Any, ModelRequest, Callable[[], int]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(step, BaseException):
            raise step
        return httpx.Response(
            step,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "synthetic refusal",
                },
            },
            request=request,
        )

    client = anthropic.Anthropic(
        api_key="offline-placeholder-not-a-credential",
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
    return lane, _request(HAIKU_4_5_MODEL), lambda: len(requests)


def _openai_lane(
    step: int | BaseException,
) -> tuple[Any, ModelRequest, Callable[[], int]]:
    scripted = step if isinstance(step, BaseException) else (step, error_body())
    wire = RecordingTransport([scripted])
    lane = openai_transport(wire.client())
    return lane, _request(OPENAI_MODEL), lambda: wire.calls


def _xai_lane(step: int | BaseException) -> tuple[Any, ModelRequest, Callable[[], int]]:
    scripted = step if isinstance(step, BaseException) else (step, error_body())
    wire = RecordingTransport([scripted])
    seed = XAIWire({})
    seed.client.close()
    seed.http_client.close()
    seed.client = wire.xai_client()
    lane = xai_transport(seed)
    return lane, xai_request(), lambda: wire.calls


def _mistral_lane(
    step: int | BaseException,
) -> tuple[Any, ModelRequest, Callable[[], int]]:
    scripted = (
        step if isinstance(step, BaseException) else (step, {"message": "synthetic"})
    )
    wire = MistralWire(scripted)
    return mistral_transport(wire), mistral_request(), lambda: len(wire.requests)


_LANES = {
    "anthropic": _anthropic_lane,
    "openai": _openai_lane,
    "xai": _xai_lane,
    "mistral": _mistral_lane,
}


@pytest.mark.parametrize("provider", tuple(_LANES))
@pytest.mark.parametrize(("status", "common_fault"), _STATUS_CASES)
def test_every_sdk_status_response_is_durable_as_received_without_becoming_accepted(
    tmp_path: Path, provider: str, status: int, common_fault: str
) -> None:
    lane, request, calls = _LANES[provider](status)
    path = tmp_path / f"{provider}-{status}.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id=f"{provider}-diagnostic",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "2" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as caught:
        lane.send(request)
    recorder.abandon_open_call()
    recorder.finalize_excluded("provider_transport")
    recorder.close()

    expected_fault = (
        "provider_invalid_request_or_spend_limit"
        if provider == "anthropic" and status == 400
        else common_fault
    )
    assert caught.value.fault == expected_fault
    sdk_error = caught.value.__cause__
    if provider == "anthropic":
        assert isinstance(sdk_error, anthropic.APIStatusError)
    elif provider in {"openai", "xai"}:
        assert isinstance(sdk_error, openai.APIStatusError)
    else:
        assert isinstance(sdk_error, MistralError)
    assert calls() == 1

    telemetry = lane.last_telemetry()
    assert telemetry.attempt_count == 1
    assert telemetry.terminal_reason == "fault_not_retryable" or status in {
        408,
        429,
        500,
        503,
    }
    live = telemetry.attempts[0]
    assert live.fault == expected_fault
    assert live.http_status == status
    assert live.response_received is True
    assert live.usage_reported is False
    assert live.cost_reservation_usd is None
    assert live.cost_settlement is None
    assert lane.last_usage().input_tokens is None
    assert lane.last_usage().output_tokens is None

    audit = read_execution_ledger(path, require_complete=True)
    durable = audit.calls[0].attempts[0]
    assert durable.response_received is True
    assert durable.usage_reported is False
    assert durable.input_tokens is None
    assert durable.output_tokens is None
    assert durable.response_model is None
    assert durable.response_id_digest_sha256 is None
    assert durable.response_stop_classification is None
    assert durable.response_normalized_digest_sha256 is None
    assert durable.cost_reservation_usd is None
    assert durable.cost_settlement is None


@pytest.mark.parametrize("provider", tuple(_LANES))
def test_every_sdk_transport_exception_remains_durable_as_no_response(
    tmp_path: Path, provider: str
) -> None:
    lane, request, calls = _LANES[provider](
        httpx.ConnectError("synthetic offline failure")
    )
    path = tmp_path / f"{provider}-transport.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id=f"{provider}-diagnostic",
        ),
        provider=provider_identity_for(lane),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "3" * 32)
    lane.attach_recorder(recorder)

    with pytest.raises(AdapterProviderError) as caught:
        lane.send(request)
    recorder.abandon_open_call()
    recorder.finalize_excluded("provider_transport")
    recorder.close()

    assert caught.value.fault == "provider_network_error"
    assert calls() == 1
    live = lane.last_telemetry().attempts[0]
    assert live.response_received is False
    assert live.http_status is None
    durable = read_execution_ledger(path, require_complete=True).calls[0].attempts[0]
    assert durable.response_received is False
    assert durable.http_status is None
    assert durable.usage_reported is False
    assert durable.response_model is None
    assert durable.response_id_digest_sha256 is None
    assert durable.response_stop_classification is None
    assert durable.response_normalized_digest_sha256 is None
