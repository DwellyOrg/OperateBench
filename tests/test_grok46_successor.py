# SPDX-License-Identifier: Apache-2.0
"""The Grok 4.6 high-effort cell is distinct from historical low-effort cells."""

from dataclasses import replace
from typing import Any

import httpx
import openai
import pytest

from operatebench.agents.lifecycle_contract import ModelRequestRefused
from operatebench.agents.model import MAX_OUTPUT_TOKENS, ModelAgent
from operatebench.agents.transport import ModelRequest
from operatebench.agents.xai_responses import XAIResponsesTransport, build_model_payload
from operatebench.core.protocol import AgentObservation
from operatebench.providers import xai_responses as exchange


def request_for(model: str, ceiling: int) -> ModelRequest:
    class Never:
        def send(self, request: ModelRequest) -> Any:
            raise AssertionError("construction must not dispatch")

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
    return ModelAgent(
        Never(), model=model, agent_id="diagnostic", max_output_tokens=ceiling
    ).build_request(observation)


@pytest.mark.parametrize("model,ceiling", [("grok-4.5", 8192), ("grok-4.6", 4096)])
def test_payload_refuses_other_profiles_ceiling(model: str, ceiling: int) -> None:
    with pytest.raises(ModelRequestRefused):
        build_model_payload(request_for(model, ceiling), model=model)


@pytest.mark.parametrize(
    "model,ceiling,version",
    [("grok-4.5", 4096, "v6"), ("grok-4.6", 8192, "v7"), ("grok-4.6-latest", 4096, "v6")],
)
def test_profile_binding_and_early_transport_refusal(
    model: str, ceiling: int, version: str
) -> None:
    assert MAX_OUTPUT_TOKENS == 4096
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("refused request reached dispatch")

    with openai.OpenAI(
        api_key="offline-not-a-credential",
        base_url=exchange.XAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        lane = XAIResponsesTransport(model=model, client=client, deadline_seconds=30)
        request = request_for(model, ceiling)
        payload = build_model_payload(request, model=model)
        mapping = "lifecycle_xai_responses_model_request_" + version
        assert (
            lane.settings["max_output_tokens"] == ceiling == payload["max_output_tokens"]
        )
        assert lane.request_mapping == mapping == lane.settings["request_mapping"]
        assert "request_mapping: " + mapping + "\n" in payload["instructions"]
        assert request.request_digest_sha256 in payload["instructions"]
        for wrong in (4096 if ceiling == 8192 else 8192, ceiling - 1, ceiling + 1):
            with pytest.raises(ModelRequestRefused):
                lane.send(replace(request, max_output_tokens=wrong))
        with pytest.raises(ModelRequestRefused):
            lane.send(replace(request, model="another-model"))
        assert calls == []


def test_registered_high_profile_does_not_relabel_predecessor() -> None:
    profile = exchange.request_profile_for("grok-4.6")
    assert profile.profile_id == "grok46_reasoning_high_sampling_omitted_v1"
    assert profile.reasoning == {"effort": "high"}
    assert profile.omitted_fields() == ("temperature", "top_p")
    assert profile.contract_verified is True
    old = exchange.request_profile_for("grok-4.5")
    assert old.profile_id == "grok45_reasoning_low_sampling_omitted_v1"
    assert old.reasoning == {"effort": "low"}
    assert old.contract_verified is False
    with pytest.raises(exchange.XAIResponsesConfigurationError):
        exchange.check_request_profile(old, model="grok-4.6")
    assert exchange.request_profile_for("grok-4.6-latest") is exchange.BASELINE_PROFILE


@pytest.mark.parametrize(
    "model",
    ["grok-4.5", "gpt-4.1", "claude-haiku-4-5", "mistral-large-latest", "grok-4.6"],
)
def test_shared_historical_contract_stays_exact4096(model: str) -> None:
    from operatebench.agents.lifecycle_contract import check_model_request

    request = request_for(model, 4096)
    check_model_request(request, model=model)
    with pytest.raises(ModelRequestRefused):
        check_model_request(replace(request, max_output_tokens=8192), model=model)


def test_successor_contract_cannot_be_selected_for_other_model() -> None:
    from operatebench.agents.lifecycle_contract import (
        LifecycleRequestContract,
        check_model_request,
    )

    for model in ("grok-4.5", "grok-4.6-latest"):
        with pytest.raises(ModelRequestRefused):
            check_model_request(
                request_for(model, 8192),
                model=model,
                contract=LifecycleRequestContract.XAI_GROK46,
            )
    with pytest.raises(ModelRequestRefused):
        build_model_payload(request_for("grok-4.5", 8192), model="grok-4.6")


def test_successor_is_a_separate_fixed_operator() -> None:
    from importlib.util import find_spec

    assert find_spec("tools.run_lifecycle_v1_xai_grok_4_6_responses_canary") is not None


def test_external_consumer_entrypoint_exists() -> None:
    from importlib.util import find_spec

    assert find_spec("tools.consume_grok46_external_authorization") is not None
