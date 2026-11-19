"""Offline Ripple protobuf, official renderer and pinned checkpoint assets."""

import os
import socket
from decimal import Decimal
from pathlib import Path

import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_ripple import MODEL
from tools.three_flow_river import GRPC_OPTIONS, RiverCampaignTransport
from tools.three_flow_river_mock import RiverMockChannel
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("fault", [None, "usage"])
def test_ripple_native_protobuf_record_replay(tmp_path, fault):
    asset_path = os.environ.get("THREE_FLOW_RIPPLE_ASSETS")
    if not asset_path:
        pytest.skip("explicit pinned public asset directory required")
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    p = {
        "endpoint": "api.river.ai:443",
        "input_max": None,
        "output_max": 30000,
        "context_max": 262144,
        "owner_output_cap_tokens": 30000,
        "bounds_policy": {"mode": "test"},
    }
    guard = AdmittedGuard(
        profile=p,
        budget=budget,
        cell="trial",
        max_output_tokens=30000,
        policy=LifecyclePricingPolicy("synthetic", Decimal("1"), Decimal("3")),
    )
    channel = RiverMockChannel(MODEL, "commerce")
    transport = RiverCampaignTransport(
        model=MODEL,
        assets=Path(asset_path),
        channel=channel,
        guard=guard,
        max_output_tokens=30000,
        channel_options=GRPC_OPTIONS,
        network_timeout=None,
        mode_profile="ripple-default-v1",
    )
    channel.tokenizer = transport.tokenizer
    if fault == "usage":
        generate = channel.generate

        def invalid_usage(request):
            ack = generate(request)
            channel.pending[ack.request_id].usage.total_tokens += 1
            return ack

        channel.generate = invalid_usage
    transport.settings.update(
        bounds_policy=p["bounds_policy"], context_request_bound=262144
    )
    trial = Trial(
        "offline",
        "trial",
        "commerce",
        "UK_NORMAL",
        MODEL,
        30000,
        provider_identity(transport),
    )
    try:
        if fault:
            with pytest.raises(ProviderFailure):
                run_trial(trial, transport, output=tmp_path / "trial")
            assert len(channel.submissions) == 1
            assert all(r["state"] == "unknown" for r in budget.records.values())
            return
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        import base64
        import copy
        import hashlib

        from operatebench.agents.transport import content_digest
        from operatebench.providers.faults import AdapterProviderError

        bad = copy.deepcopy(record)
        row = bad["provider_turns"][0]["wire"][-1]
        reply = transport.pb.RetrieveFutureResponse.FromString(
            base64.b64decode(row["response_base64"])
        )
        reply.inference.usage.total_tokens += 1
        raw = reply.SerializeToString(deterministic=True)
        row["response_base64"] = base64.b64encode(raw).decode()
        row["response_sha256"] = hashlib.sha256(raw).hexdigest()
        bad["record_digest"] = content_digest(
            {k: v for k, v in bad.items() if k != "record_digest"}
        )
        with pytest.raises(AdapterProviderError):
            replay_trial(
                trial,
                bad,
                expected_record_digest=bad["record_digest"],
                ripple_assets=Path(asset_path),
            )
        assert channel.submissions
        assert all(p.max_tokens <= 30000 for s in channel.submissions for p in s.prompts)
        assert "thinking" not in transport.settings
        assert (
            replay_trial(
                trial,
                record,
                expected_record_digest=record["record_digest"],
                ripple_assets=Path(asset_path),
            )["provider_calls"]
            == 0
        )
    finally:
        transport.close()
        budget.close()


def test_four_model_shared_operator(tmp_path):
    from tools.three_flow_campaign import assignments, run_campaign
    from tools.three_flow_late_november_offline import factory
    from tools.three_flow_profiles import LATE_NOVEMBER_MODELS

    assets = os.environ.get("THREE_FLOW_RIPPLE_ASSETS")
    if not assets:
        pytest.skip("explicit pinned public asset directory required")
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    budget = CampaignBudget(
        root,
        campaign_id="offline-four",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    all_slots = assignments("offline-four", LATE_NOVEMBER_PROFILE)
    slots = [
        next(
            s
            for s in all_slots
            if (s["provider"], s["model"]) == pair and s["flow"] == "commerce"
        )
        for pair in LATE_NOVEMBER_MODELS[:4]
    ]
    rows = run_campaign(
        root,
        campaign_id="offline-four",
        slots=slots,
        workers=4,
        campaign_budget=budget,
        transport_factory=factory(Path(assets)),
        evidence_origin="SDK_MOCK_NOT_LLM",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    assert len(rows) == 4
    assert all(row["classification"] == "scored" for row in rows)


def test_ripple_parser_strict_native_blocks():
    from river_client.renderers.ripple import parse_ripple_content_blocks

    for body in (
        '{"name":"act","arguments":{"x":NaN}}',
        '{"name":"act","name":"act","arguments":{}}',
        '{"name":"act","arguments":{},"extra":1}',
    ):
        _, calls = parse_ripple_content_blocks("<tool_call>" + body + "</tool_call>")
        assert "error" in calls[0]
    _, calls = parse_ripple_content_blocks(
        '<tool_call>{"name":"act","arguments":{"x":"</tool_call>"}}</tool_call>'
    )
    assert calls[0]["function"]["name"] == "act"
    _, calls = parse_ripple_content_blocks(
        '<tool_call>{"name":"act","arguments":{}}</tool_call>'
        '<tool_call>{"name":"wait","arguments":{}}</tool_call>'
    )
    assert len(calls) == 2
