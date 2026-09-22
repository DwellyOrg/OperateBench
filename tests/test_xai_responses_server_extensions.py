# SPDX-License-Identifier: Apache-2.0
"""Exact offline contract for xAI Responses server-ahead fields."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import httpx
import openai
import pytest

from operatebench.agents.model import MAX_OUTPUT_TOKENS
from operatebench.agents.transport import ModelRequest
from operatebench.agents.xai_responses import (
    LIFECYCLE_XAI_REQUEST_MAPPING_VERSION,
    build_model_payload,
    model_response_from,
)
from operatebench.jsonsafe import canonical_json_bytes
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from operatebench.providers.xai_responses import GROK_4_5_MODEL

CONTRACT_NAME = "xai_responses_server_extensions_v2"
V1_CONTRACT_NAME = "xai_responses_server_extensions_v1"
V1_SCHEMA = {
    "response_root": {
        "frequency_penalty": "finite_penalty_number",
        "presence_penalty": "finite_penalty_number",
        "store": "boolean",
    },
    "usage_block": {
        "context_details": {"image_tokens": "exact_count", "text_tokens": "exact_count"},
        "cost_in_usd_ticks": "exact_count",
        "num_server_side_tools_used": "exact_zero",
        "num_sources_used": "exact_zero",
    },
}
SCHEMA = {
    "response_root": {
        "frequency_penalty": "finite_penalty_number",
        "presence_penalty": "finite_penalty_number",
        "store": "boolean",
    },
    "usage_block": {
        "context_details": {
            "one_of": [
                {"image_tokens": "exact_count", "text_tokens": "exact_count"},
                {"input_tokens": "exact_count", "output_tokens": "exact_count"},
            ]
        },
        "cost_in_usd_ticks": "exact_count",
        "num_server_side_tools_used": "exact_zero",
        "num_sources_used": "exact_zero",
    },
}
EXTENSIONS = {
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "store": False,
}
USAGE_EXTENSIONS = {
    "context_details": {"image_tokens": 0, "text_tokens": 4385},
    "cost_in_usd_ticks": 17,
    "num_server_side_tools_used": 0,
    "num_sources_used": 0,
}
CONTEXT_VARIANT_B = {"input_tokens": 4385, "output_tokens": 82}


def plain(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {key: plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [plain(value) for value in node]
    return node


def request() -> ModelRequest:
    return ModelRequest(
        model=GROK_4_5_MODEL,
        protocol_version="operatebench.model.v4",
        prompt={"observation": {"probe": True}},
        tool_names=("act", "ask", "wait", "complete", "escalate", "retrieve"),
        observation_digest_sha256="0" * 64,
        prompt_digest_sha256="1" * 64,
        invocation_index=0,
        turn_index=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def body(*, context_details: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "completed_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 4096,
        "model": GROK_4_5_MODEL,
        "output": [
            {
                "id": "rs_offline",
                "type": "reasoning",
                "summary": [],
                "status": "completed",
            },
            {
                "id": "fc_offline",
                "type": "function_call",
                "call_id": "call_offline",
                "name": "complete",
                "arguments": json.dumps(
                    {"reason": "done", "evidence_refs": []}, separators=(",", ":")
                ),
                "status": "completed",
            },
        ],
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": "low", "summary": "auto"},
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "required",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 4385,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 82,
            "output_tokens_details": {"reasoning_tokens": 40},
            "total_tokens": 4467,
            **copy.deepcopy(USAGE_EXTENSIONS),
        },
        "user": None,
        "metadata": {},
        **copy.deepcopy(EXTENSIONS),
    }
    if context_details is not None:
        result["usage"]["context_details"] = copy.deepcopy(context_details)
    return result


def client_for(response_body: dict[str, Any]) -> openai.OpenAI:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=req, json=response_body)

    return openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def exchange_for(response_body: dict[str, Any]) -> Any:
    from operatebench.providers.xai_responses import XAIResponsesExchange

    return XAIResponsesExchange(
        model=GROK_4_5_MODEL,
        client=client_for(response_body),
        single_flight_label="test",
        max_tool_argument_wire_bytes=65536,
    )


def dispatch(exchange: Any) -> Any:
    from operatebench.providers.telemetry import TurnDeadline

    with exchange.turn():
        exchange.clear()
        return exchange.exchange(
            build_model_payload(request(), model=GROK_4_5_MODEL),
            TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False),
        )


def run_exchange(response_body: dict[str, Any]) -> tuple[Any, Any]:
    exchange = exchange_for(response_body)
    return exchange, dispatch(exchange)


def test_exact_shape_with_reasoning_and_canonical_local_arguments_is_accepted() -> None:
    exchange, wire = run_exchange(body())
    response = model_response_from(wire)
    assert response.tool_calls[0].name == "complete"
    assert response.tool_calls[0].arguments == {"reason": "done", "evidence_refs": []}
    assert exchange.last_usage().input_tokens == 4385
    assert exchange.last_usage().output_tokens == 82
    assert exchange.last_telemetry().usage_reported is True


def test_second_observed_context_details_variant_is_accepted() -> None:
    exchange, wire = run_exchange(body(context_details=CONTEXT_VARIANT_B))
    response = model_response_from(wire)
    assert response.tool_calls[0].name == "complete"
    assert exchange.last_usage().input_tokens == 4385
    assert exchange.last_usage().output_tokens == 82
    assert exchange.last_telemetry().usage_reported is True


def test_response_contract_is_digest_bound_without_aliasing_request_mapping() -> None:
    from operatebench.agents.xai_responses import lifecycle_xai_settings
    from operatebench.providers.xai_responses import (
        RESPONSE_EXTENSION_CONTRACT,
        RESPONSE_EXTENSION_DIGEST,
        XAI_RESPONSE_SERVER_EXTENSIONS,
    )

    contract = {
        "contract": CONTRACT_NAME,
        "members": SCHEMA,
        "sites_optional": True,
        "structured_members_required": True,
    }
    digest = hashlib.sha256(
        canonical_json_bytes(contract, "xAI Responses extensions")
    ).hexdigest()
    assert plain(RESPONSE_EXTENSION_CONTRACT) == contract
    assert XAI_RESPONSE_SERVER_EXTENSIONS == CONTRACT_NAME
    assert digest == RESPONSE_EXTENSION_DIGEST
    settings = lifecycle_xai_settings(model=GROK_4_5_MODEL, deadline_seconds=30.0)
    assert settings["response_server_extensions"] == CONTRACT_NAME
    assert settings["response_server_extensions_digest"] == digest
    assert (
        LIFECYCLE_XAI_REQUEST_MAPPING_VERSION
        == "lifecycle_xai_responses_model_request_v6"
    )
    assert LIFECYCLE_XAI_REQUEST_MAPPING_VERSION != XAI_RESPONSE_SERVER_EXTENSIONS


def test_v1_contract_and_validator_remain_exactly_historical() -> None:
    from operatebench.providers.xai_responses import (
        RESPONSE_EXTENSION_CONTRACT_V1,
        RESPONSE_EXTENSION_DIGEST_V1,
        check_response_extensions_v1,
    )

    contract = {
        "contract": V1_CONTRACT_NAME,
        "members": V1_SCHEMA,
        "sites_optional": True,
        "structured_members_required": True,
    }
    assert plain(RESPONSE_EXTENSION_CONTRACT_V1) == contract
    assert RESPONSE_EXTENSION_DIGEST_V1 == (
        "ca5fabda902d3d3ccb985ebc04389d3e2c895d54d90c7b810ba8ec85d1fe8f05"
    )
    check_response_extensions_v1(
        {"context_details": {"image_tokens": 0, "text_tokens": 4385}},
        site="usage_block",
    )
    with pytest.raises(AdapterProviderError):
        check_response_extensions_v1(
            {"context_details": CONTEXT_VARIANT_B}, site="usage_block"
        )


@pytest.mark.parametrize(
    "site,key",
    [
        ("root", "frequency_penalty"),
        ("root", "store"),
        ("usage", "cost_in_usd_ticks"),
        ("usage", "context_details"),
    ],
)
def test_raw_only_and_typed_only_extensions_are_refused(site: str, key: str) -> None:
    from operatebench.providers.xai_responses import check_response_admissible

    wire = body()
    typed = body()
    target = wire if site == "root" else wire["usage"]
    del target[key]
    parsed = (
        client_for(typed)
        .responses.with_raw_response.create(model=GROK_4_5_MODEL, input=[])
        .parse()
    )
    with pytest.raises(AdapterProviderError):
        check_response_admissible(
            WireResponse(json.dumps(wire), lambda: parsed, kind="xAI response"),
            model=GROK_4_5_MODEL,
        )


@pytest.mark.parametrize(
    "path,value",
    [
        (("frequency_penalty",), None),
        (("presence_penalty",), True),
        (("store",), 0),
        (("usage", "cost_in_usd_ticks"), True),
        (("usage", "cost_in_usd_ticks"), 2**53),
        (("usage", "num_server_side_tools_used"), 1),
        (("usage", "num_sources_used"), 1),
        (("usage", "context_details"), {}),
        (("usage", "context_details", "text_tokens"), False),
    ],
)
def test_wrong_null_boolean_huge_vacuous_and_nonzero_values_fail_before_usage(
    path: tuple[str, ...], value: Any
) -> None:
    mutated = body()
    node: Any = mutated
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    exchange = exchange_for(mutated)
    with pytest.raises(AdapterProviderError):
        dispatch(exchange)
    assert exchange.last_telemetry().response_received is True
    assert exchange.last_telemetry().usage_reported is False
    assert exchange.last_usage().input_tokens is None


@pytest.mark.parametrize(
    "context_details",
    [
        {"image_tokens": 0, "output_tokens": 82},
        {"input_tokens": 4385},
        {"input_tokens": 4385, "output_tokens": 82, "unexpected": 0},
        {"input_tokens": None, "output_tokens": 82},
        {"input_tokens": False, "output_tokens": 82},
        {"input_tokens": 4385, "output_tokens": -1},
        {"input_tokens": 4385, "output_tokens": "82"},
        {"input_tokens": 4385, "output_tokens": 2**53},
    ],
)
def test_second_context_variant_is_closed_complete_and_exactly_bounded(
    context_details: dict[str, Any],
) -> None:
    exchange = exchange_for(body(context_details=context_details))
    with pytest.raises(AdapterProviderError):
        dispatch(exchange)
    assert exchange.last_telemetry().response_received is True
    assert exchange.last_telemetry().usage_reported is False


@pytest.mark.parametrize(
    "path",
    [
        ("unexpected",),
        ("usage", "unexpected"),
        ("usage", "context_details", "unexpected"),
    ],
)
def test_unknown_extension_keys_fail_closed(path: tuple[str, ...]) -> None:
    mutated = body()
    node: Any = mutated
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = 0
    exchange = exchange_for(mutated)
    with pytest.raises(AdapterProviderError):
        dispatch(exchange)
    assert exchange.last_telemetry().usage_reported is False


def test_typed_context_or_ticks_mismatch_is_refused() -> None:
    from operatebench.providers.xai_responses import check_response_admissible

    wire = body()
    typed = body()
    typed["usage"]["cost_in_usd_ticks"] += 1
    parsed = (
        client_for(typed)
        .responses.with_raw_response.create(model=GROK_4_5_MODEL, input=[])
        .parse()
    )
    with pytest.raises(AdapterProviderError):
        check_response_admissible(
            WireResponse(json.dumps(wire), lambda: parsed, kind="xAI response"),
            model=GROK_4_5_MODEL,
        )


@pytest.mark.parametrize(
    "typed_context",
    [
        {"input_tokens": 4386, "output_tokens": 82},
        {"input_tokens": 4385.0, "output_tokens": 82},
        {"input_tokens": 4385},
        {"image_tokens": 0, "text_tokens": 4385},
    ],
)
def test_second_context_variant_requires_raw_typed_value_type_and_variant_parity(
    typed_context: dict[str, Any],
) -> None:
    from operatebench.providers.xai_responses import check_response_admissible

    wire = body(context_details=CONTEXT_VARIANT_B)
    typed = body(context_details=typed_context)
    parsed = (
        client_for(typed)
        .responses.with_raw_response.create(model=GROK_4_5_MODEL, input=[])
        .parse()
    )
    with pytest.raises(AdapterProviderError):
        check_response_admissible(
            WireResponse(json.dumps(wire), lambda: parsed, kind="xAI response"),
            model=GROK_4_5_MODEL,
        )


def test_full_46_call_xai_episode_writes_artifact8_audits_bundle_and_replays(
    tmp_path: Any,
) -> None:
    from operatebench.agents.evidence import (
        EvidenceRecordingModelAgent,
        EvidenceRunIdentity,
        ProviderEvidenceRecorder,
        WireCaptureTransport,
        provider_identity_for,
    )
    from operatebench.agents.pricing import LifecycleCostGuard
    from operatebench.agents.xai_responses import XAIResponsesTransport
    from operatebench.artifact import (
        build_artifact,
        read_artifact,
        replay_artifact,
        write_artifact,
    )
    from operatebench.domains.lettings.maintenance.agents import build_agent
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.execution_bundle import audit_execution_bundle_files
    from operatebench.execution_ledger import read_execution_ledger
    from operatebench.runner import run_episode
    from tests.model_transport import observation_from_dict, tool_call_for
    from tests.test_provider_evidence_capture import controls, fixed_clock, policy

    spec = load_spec("examples/operatebench/maintenance_v0_1.yaml")
    reference = build_agent("reference")
    capture = WireCaptureTransport()
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        prompt = json.loads(req.content)["input"][0]["content"]
        observation = observation_from_dict(json.loads(prompt)["observation"])
        call = tool_call_for(reference.decide(observation))
        answer = body(context_details=CONTEXT_VARIANT_B)
        answer["output"][1]["name"] = call.name
        answer["output"][1]["arguments"] = json.dumps(
            call.arguments, separators=(",", ":"), sort_keys=True
        )
        return httpx.Response(200, request=req, json=answer)

    capture.attach(httpx.MockTransport(handler))
    client = openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url="https://api.x.ai/v1",
        max_retries=0,
        http_client=httpx.Client(transport=capture),
    )
    cost_guard = LifecycleCostGuard(
        policy=policy(), cap_usd=Decimal("5"), max_output_tokens=MAX_OUTPUT_TOKENS
    )
    transport = XAIResponsesTransport(
        model=GROK_4_5_MODEL,
        client=client,
        deadline_seconds=30.0,
        clock=fixed_clock(),
        cost_guard=cost_guard,
    )
    ledger = tmp_path / "execution.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=ledger,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id="V1",
            agent_id="xai-grok-4-5-responses-offline",
        ),
        provider=provider_identity_for(transport),
        controls=controls(),
        wire=capture,
        guard=cost_guard,
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GROK_4_5_MODEL,
        agent_id="xai-grok-4-5-responses-offline",
        recorder=recorder,
        max_transport_calls=60,
    )
    run = run_episode(
        spec,
        "V1",
        "xai-grok-4-5-responses-offline",
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    audit = read_execution_ledger(ledger, require_complete=True)
    assert calls == audit.totals.provider_calls == 46
    assert audit.totals.input_tokens == 46 * 4385
    assert audit.totals.output_tokens == 46 * 82
    artifact = tmp_path / "artifact.json"
    write_artifact(run, artifact)
    assert build_artifact(run)["artifact_version"] == 8
    assert audit_execution_bundle_files(artifact, ledger).ok is True
    replay = replay_artifact(spec, read_artifact(artifact))
    assert replay.ok is True and replay.reproduction == "playback"
    assert calls == 46
