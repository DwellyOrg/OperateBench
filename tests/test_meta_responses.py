"""Meta Standard snapshot contract over the real SDK; no network or credentials."""

import hashlib
import json

import httpx
import pytest

from operatebench.agents.meta_responses import (
    MAPPING_VERSION,
    MetaResponsesTransport,
    build_model_payload,
    pricing_profile,
)
from operatebench.agents.model import ModelAgent
from operatebench.providers.meta_responses import (
    MetaConfigurationError,
    build_client,
    check_key,
)
from tests.openai_transport import function_call_item, responses_body
from tests.test_lifecycle_openai_bridge import observation

MODELS = ("muse-wonder-1.0", "muse-spark-1.3")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("key", ["LLM|0|offline-meta", "opaque:synthetic-token+/="])
def test_sdk_snapshot_and_replay(model, key):
    captured = []
    evidence = []
    body = responses_body(
        [function_call_item("complete", '{"summary":"done"}')],
        model=model,
        output_tokens=29,
    )
    body["usage"]["input_tokens_details"] = {"cached_tokens": 7}
    body["usage"]["output_tokens_details"] = {"reasoning_tokens": 11}
    body["output"].insert(0, {"type": "reasoning", "id": "rs_test", "summary": []})

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=body)

    class Recorder:
        def observe_provider_turn(self, turn):
            evidence.append(turn)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = build_client(
            model=model, environ={"MODEL_API_KEY": key}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=model, client=client, deadline_seconds=10, recorder=Recorder()
        )
        assert captured == []
        request = ModelAgent(None, model=model, agent_id="test").build_request(
            observation()
        )
        answer = transport.send(request)
        repeated = transport.send(request)
    payload = build_model_payload(request, model=model)
    # SDK 2.53.0 declares this order; comparing the bytes catches drift.
    sdk_order = (
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "store",
        "tool_choice",
        "tools",
    )
    expected = json.dumps(
        {k: payload[k] for k in sdk_order}, ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert [r.content for r in captured] == [expected, expected]
    assert str(captured[0].url) == "https://api.meta.ai/v1/responses"
    assert captured[0].headers["authorization"] == f"Bearer {key}"
    assert answer == repeated
    assert answer.tool_calls[0].name == "complete"
    assert answer.output_tokens == 29
    assert transport.last_usage().output_tokens == 29
    assert transport.last_usage().input_tokens == 137
    assert evidence[0].request.identity() == request.identity()
    assert evidence[0].response == answer
    assert evidence[0].payload == payload
    assert (
        evidence[0].response_id_digest_sha256 == hashlib.sha256(b"resp_test").hexdigest()
    )
    assert transport.provider == "meta"
    assert transport.request_mapping == MAPPING_VERSION
    assert transport.settings["comparison"] == "equal_scaffold_stateless"
    assert transport.settings["max_output_tokens_policy"] == "scaffold_ceiling"
    assert payload["max_output_tokens"] == 4096
    assert payload["store"] is False
    assert set(payload) == {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "store",
        "max_output_tokens",
    }
    assert all(t["type"] == "function" and t["strict"] is False for t in payload["tools"])
    assert pricing_profile(model).output_usd_per_mtok > 0


@pytest.mark.parametrize(
    "model", ["muse-spark-1.3-contributor", "muse-spark-1.2", "unknown"]
)
def test_model_preflight(model):
    with pytest.raises(MetaConfigurationError):
        build_client(model=model, environ={"MODEL_API_KEY": "LLM|0|offline"})


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"OPENAI_API_KEY": "secret-marker"},
        {"MODEL_API_KEY": " "},
        {"MODEL_API_KEY": "unsafe secret-marker"},
        {"MODEL_API_KEY": "LLM|0|offline", "OPENAI_BASE_URL": "secret-marker"},
    ],
)
def test_credential_preflight(environ):
    captured = []
    with (
        httpx.Client(transport=httpx.MockTransport(lambda r: captured.append(r))) as http,
        pytest.raises(MetaConfigurationError) as exc,
    ):
        build_client(model=MODELS[0], environ=environ, http_client=http)
    assert captured == []
    assert "secret-marker" not in str(exc.value)


@pytest.mark.parametrize("key", ["!", "~" * 4096, "opaque:synthetic-token+/="])
def test_safe_opaque_key(key):
    check_key(key)


@pytest.mark.parametrize(
    "key",
    [
        None,
        123,
        b"opaque",
        [],
        {},
        "",
        " ",
        "opaque token",
        " token",
        "token ",
        "token\tvalue",
        "token\nvalue",
        "token\rvalue",
        "token\r\nvalue",
        "token\x00value",
        "token\x1fvalue",
        "token\x7fvalue",
        "tokené",
        "x" * 4097,
    ],
)
def test_unsafe_key_is_sanitized(key):
    with pytest.raises(MetaConfigurationError) as exc:
        check_key(key)
    assert str(exc.value) == (
        "MODEL_API_KEY must be 1-4096 visible ASCII characters without whitespace"
    )


@pytest.mark.parametrize("mutation", ["unknown", "boolean_usage", "wrong_model"])
def test_invalid_wire_is_sanitized(mutation):
    body = responses_body([], model=MODELS[0])
    if mutation == "unknown":
        body["secret-marker"] = "secret-marker"
    elif mutation == "boolean_usage":
        body["usage"]["output_tokens_details"] = {"reasoning_tokens": True}
    else:
        body["model"] = "secret-marker"
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as http:
        client = build_client(
            model=MODELS[0], environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=MODELS[0], client=client, deadline_seconds=10
        )
        request = ModelAgent(None, model=MODELS[0], agent_id="test").build_request(
            observation()
        )
        with pytest.raises(Exception) as exc:
            transport.send(request)
    assert "secret-marker" not in str(exc.value)


@pytest.mark.parametrize("model", MODELS)
def test_full_core_episode_ledger_and_playback(model, tmp_path):
    from dataclasses import replace
    from decimal import Decimal
    from pathlib import Path

    from operatebench.agents.evidence import (
        EvidenceRecordingModelAgent,
        WireCaptureTransport,
    )
    from operatebench.agents.pricing import LifecycleCostGuard
    from operatebench.artifact import read_artifact, write_artifact
    from operatebench.domains.lettings.maintenance.agents import build_agent
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.execution_ledger import read_execution_ledger
    from operatebench.providers.cost import usd_text
    from operatebench.runner import run_episode
    from tests.model_transport import observation_from_dict, tool_call_for
    from tests.test_provider_evidence_capture import controls, recorder_for

    reference = build_agent("reference")
    bodies = []

    def handler(request):
        bodies.append(request.content)
        payload = json.loads(request.content)
        obs = observation_from_dict(
            json.loads(payload["input"][0]["content"])["observation"]
        )
        call = tool_call_for(reference.decide(obs))
        body = responses_body(
            [function_call_item(call.name, json.dumps(call.arguments))],
            model=model,
            input_tokens=4211,
            output_tokens=118,
        )
        body["usage"]["input_tokens_details"] = {"cached_tokens": 0}
        body["usage"]["output_tokens_details"] = {"reasoning_tokens": 100}
        return httpx.Response(200, json=body)

    spec = load_spec(Path("examples/operatebench/maintenance_v0_1.yaml"))
    policy = pricing_profile(model)
    guard = LifecycleCostGuard(
        policy=policy, cap_usd=Decimal("5"), max_output_tokens=4096
    )
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(handler))
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    path = directory / "execution.ndjson"
    with httpx.Client(transport=capture) as http:
        client = build_client(
            model=model, environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=model, client=client, deadline_seconds=30, cost_guard=guard
        )
        recorder = recorder_for(
            path,
            transport,
            spec=spec,
            wire=capture,
            cost_guard=guard,
            ledger_controls=replace(controls(), pricing=policy.as_dict()),
            agent_id="meta-test",
        )
        transport.attach_recorder(recorder)
        agent = EvidenceRecordingModelAgent(
            transport,
            model=model,
            agent_id="meta-test",
            recorder=recorder,
            max_transport_calls=60,
        )
        run = run_episode(
            spec,
            "V1",
            "meta-test",
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.scored and run.reliable
    assert audit.header.provider.provider == "meta"
    assert audit.header.provider.request_mapping == MAPPING_VERSION
    assert audit.totals.input_tokens == len(bodies) * 4211
    assert audit.totals.output_tokens == len(bodies) * 118
    assert audit.totals.measured_cost_usd == usd_text(
        policy.price(input_tokens=len(bodies) * 4211, output_tokens=len(bodies) * 118)
    )
    assert [c.wire_request_digest_sha256 for c in audit.calls] == [
        hashlib.sha256(b).hexdigest() for b in bodies
    ]
    target = directory / "artifact.json"
    write_artifact(run, target)
    # The frozen artifact binding excludes Meta; preserve that contract.
    from operatebench.core.errors import ArtifactError

    with pytest.raises(ArtifactError, match="not a provider this build binds"):
        read_artifact(target)
    assert run.evaluation.dimension("deterministic_replay").ok


@pytest.mark.parametrize("model", MODELS)
def test_refusal_is_paid_but_has_no_tool_call(model):
    from operatebench.agents.model import ModelReturnedNoToolCall

    body = responses_body(
        [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "secret-marker"}],
            }
        ],
        model=model,
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as http:
        client = build_client(
            model=model, environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=model, client=client, deadline_seconds=10
        )
        agent = ModelAgent(transport, model=model, agent_id="test")
        decision = agent.decide(observation())
        assert isinstance(decision, ModelReturnedNoToolCall)
        assert transport.last_usage().output_tokens == 29
        assert "secret-marker" not in repr(decision)


@pytest.mark.parametrize(
    "mutation", ["endpoint", "empty_key", "unsafe_key", "request_model", "digest"]
)
def test_mutated_configuration_dispatches_nothing(mutation):
    from dataclasses import replace

    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as http:
        client = build_client(
            model=MODELS[0], environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=MODELS[0], client=client, deadline_seconds=10
        )
        request = ModelAgent(None, model=MODELS[0], agent_id="test").build_request(
            observation()
        )
        if mutation == "endpoint":
            client.base_url = "https://secret-marker.invalid/v1"
        elif mutation == "empty_key":
            client.api_key = " "
        elif mutation == "unsafe_key":
            client.api_key = "unsafe\nsecret-marker"
        elif mutation == "request_model":
            request = replace(request, model="secret-marker")
        else:
            request = replace(request, prompt_digest_sha256="0" * 64)
        with pytest.raises(Exception) as exc:
            transport.send(request)
    assert calls == []
    assert "secret-marker" not in str(exc.value)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_sdk_errors_are_sanitized(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "secret-marker"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = build_client(
            model=MODELS[0], environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        transport = MetaResponsesTransport(
            model=MODELS[0], client=client, deadline_seconds=10
        )
        request = ModelAgent(None, model=MODELS[0], agent_id="test").build_request(
            observation()
        )
        with pytest.raises(Exception) as exc:
            transport.send(request)
    assert len(calls) == 1
    assert "secret-marker" not in str(exc.value)
    assert "secret-marker" not in repr(transport)


@pytest.mark.parametrize(
    "mutation", ["web_tool", "strict", "previous_response_id", "input"]
)
def test_exchange_refuses_non_snapshot_requests(mutation):
    from operatebench.providers.meta_responses import MetaResponsesExchange
    from operatebench.providers.telemetry import TurnDeadline

    captured = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: captured.append(r))
    ) as http:
        client = build_client(
            model=MODELS[0], environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )
        exchange = MetaResponsesExchange(model=MODELS[0], client=client)
        request = ModelAgent(None, model=MODELS[0], agent_id="test").build_request(
            observation()
        )
        payload = build_model_payload(request, model=MODELS[0])
        if mutation == "web_tool":
            payload["tools"] = [{"type": "web_search"}]
        elif mutation == "strict":
            payload["tools"][0]["strict"] = True
        elif mutation == "previous_response_id":
            payload["previous_response_id"] = "secret-marker"
        else:
            payload["input"].append(
                {"type": "function_call_output", "call_id": "test", "output": "test"}
            )
        with pytest.raises(MetaConfigurationError):
            exchange.exchange(
                payload, TurnDeadline(remaining_seconds=10, cancelled=lambda: False)
            )
    assert captured == []


def test_final_observer_cannot_redirect_a_request():
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as http:
        client = build_client(
            model=MODELS[0], environ={"MODEL_API_KEY": "LLM|0|offline"}, http_client=http
        )

        def redirect():
            client.base_url = "https://secret-marker.invalid/v1"

        transport = MetaResponsesTransport(
            model=MODELS[0], client=client, deadline_seconds=10, before_dispatch=redirect
        )
        request = ModelAgent(None, model=MODELS[0], agent_id="test").build_request(
            observation()
        )
        with pytest.raises(MetaConfigurationError) as exc:
            transport.send(request)
    assert calls == []
    assert "secret-marker" not in str(exc.value)
