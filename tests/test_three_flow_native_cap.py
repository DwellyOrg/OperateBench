"""Request-local River response bounds; synthetic shims and one offline SDK trial."""

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from operatebench.agents.model import ModelAgent
from tools.three_flow_river import RiverCampaignTransport


class Response:
    def __init__(self, count=3):
        self.usage = SimpleNamespace(
            prompt_tokens=2,
            completion_tokens=count,
            total_tokens=2 + count,
            training_tokens=0,
        )
        self.results = [
            SimpleNamespace(
                token_ids=list(range(20, 20 + count)),
                text="fragment",
                retained_kv=False,
                cached_prompt_tokens=0,
                prompt_tokens=2,
                prompt_token_ids=[1, 2],
            )
        ]

    def HasField(self, field):
        return field == "usage"


def transport_fixture():
    transport = object.__new__(RiverCampaignTransport)
    transport.mode_profile = "legacy-v1"
    transport.reasoning_prefilled = True
    transport.pb = SimpleNamespace(InferenceResponse=Response)
    transport.model = "nvidia/Kimi-K2.6-NVFP4"
    transport.tokenizer = SimpleNamespace(decode=lambda ids: "fragment")
    transport.max_output_tokens = 100
    return transport


@pytest.mark.parametrize(
    "count,stop,decision",
    [
        (2, "completed", "ModelReturnedNoToolCall"),
        (3, "max_tokens", "ModelOutputTruncated"),
    ],
)
def test_request_local_response_cap(count, stop, decision):
    transport = transport_fixture()
    response = transport.response(Response(count), [1, 2], max_output_tokens=3)
    agent = object.__new__(ModelAgent)
    agent.max_output_tokens = 3
    assert response.stop_reason == stop
    assert type(agent._parse(response)).__name__ == decision
    assert transport.max_output_tokens == 100
    # A later, larger request must not inherit the earlier small cap.
    assert (
        transport.response(Response(3), [1, 2], max_output_tokens=4).stop_reason
        == "completed"
    )


def test_request_local_response_rejects_over_cap():
    from operatebench.providers.faults import AdapterProviderError

    transport = transport_fixture()
    with pytest.raises(AdapterProviderError):
        transport.response(Response(4), [1, 2], max_output_tokens=3)
    assert transport.max_output_tokens == 100


def test_legacy_response_call_keeps_construction_cap():
    transport = transport_fixture()
    assert transport.response(Response(), [1, 2]).stop_reason == "completed"
    transport.max_output_tokens = 3
    assert transport.response(Response(), [1, 2]).stop_reason == "max_tokens"


def test_native_capped_send_metadata_and_replay(tmp_path, monkeypatch):
    if "THREE_FLOW_RIVER_ASSETS" not in os.environ:
        pytest.skip("explicit native environment/assets required")
    import socket

    import grpc

    from tests.test_three_flow_context import bind_combined, diagnostic
    from tests.test_three_flow_live import Network, cli, write
    from tests.test_three_flow_owner_pricing import assumed_fixture
    from tools.three_flow_river import sdk
    from tools.three_flow_river_mock import RiverMockChannel
    from tools.three_flow_runtime import Trial, replay_trial

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(grpc, "secure_channel", forbidden)
    monkeypatch.setattr(grpc, "insecure_channel", forbidden)
    original = RiverMockChannel.generate

    def capped_generate(self, request):
        ack = original(self, request)
        response = self.pending[ack.request_id]
        # Exact submitted cap, smaller than the construction maximum. Keep actual
        # protobuf/tokenizer/SDK/guard/send/recording paths; replace only mock output.
        count = request.prompts[0].max_tokens
        token = self.tokenizer.encode("fragment")[0]
        ids = [token] * count
        response.results[0].token_ids[:] = ids
        response.results[0].text = self.tokenizer.decode(ids)
        response.usage.completion_tokens = count
        response.usage.total_tokens = response.usage.prompt_tokens + count
        return ack

    monkeypatch.setattr(RiverMockChannel, "generate", capped_generate)
    path, admission, slot = assumed_fixture(tmp_path, "river")
    profile = diagnostic(admission["profiles"][slot["model"]])
    profile["context_max"] = 32000
    profile["bounds_policy"]["source"] = "https://docs.river.ai/"
    bind_combined(profile)
    write(path, admission)
    network = Network()
    assert cli(path, admission, network) == 0
    root = Path(admission["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
    pb, _ = sdk()
    assert record["provider_turns"]
    for bound, turn in zip(
        record["request_output_bounds"], record["provider_turns"], strict=True
    ):
        wire = next(row for row in turn["wire"] if row["model_submission"])
        proto = pb.InferenceGenerateRequest.FromString(
            base64.b64decode(wire["request_base64"])
        )
        assert 0 < bound < 32768
        assert proto.prompts[0].max_tokens == bound == turn["requested_output_bound"]
        assert turn["output_tokens"] == bound
        assert turn["local_stop_classification"] == "max_tokens"
        assert turn["provider_stop_reason"] is None
    assert "ModelOutputTruncated" in json.dumps(record)
    assert "ModelReturnedNoToolCall" not in json.dumps(record)
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    before = list(network.calls)
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    assert network.calls == before
