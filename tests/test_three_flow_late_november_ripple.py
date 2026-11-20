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


@pytest.mark.parametrize(
    "fault", [None, "usage", "stripped", "wrong_last", "interior", "stripped_usage"]
)
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
    if fault is not None:
        generate = channel.generate

        def shaped_response(request):
            ack = generate(request)
            result = channel.pending[ack.request_id]
            if fault in ("stripped", "wrong_last", "interior", "stripped_usage"):
                sample = result.results[0]
                assert transport.tokenizer.decode([100265]) == "<|im_end|>"
                sample.token_ids.append(100265)
                result.usage.completion_tokens += 1
                result.usage.total_tokens += 1
                if fault == "wrong_last":
                    sample.token_ids[-1] = transport.tokenizer.encode("!")[0]
                elif fault == "interior":
                    sample.token_ids[0] = 100265
                elif fault == "stripped_usage":
                    result.usage.total_tokens += 1
            else:
                result.usage.total_tokens += 1
            return ack

        channel.generate = shaped_response
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
        if fault in ("usage", "wrong_last", "interior", "stripped_usage"):
            with pytest.raises(ProviderFailure) as failure:
                run_trial(trial, transport, output=tmp_path / "trial")
            assert failure.value.fault == "provider_response_invalid"
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

        if fault == "stripped":
            for turn in record["provider_turns"]:
                row = turn["wire"][-1]
                reply = transport.pb.RetrieveFutureResponse.FromString(
                    base64.b64decode(row["response_base64"])
                )
                sample = reply.inference.results[0]
                output_ids = list(sample.token_ids)
                assert output_ids[-1] == 100265
                assert transport.tokenizer.decode(output_ids[:-1]) == sample.text
                assert transport.tokenizer.decode(output_ids) != sample.text
                assert reply.inference.usage.completion_tokens == len(output_ids)
                assert turn["output_tokens"] == len(output_ids)

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


def test_ripple_verified_assets_rejects_symlinks_and_mutated_manifest(
    tmp_path, monkeypatch
):
    import shutil
    from pathlib import Path

    from tools import three_flow_ripple as ripple

    root = Path(os.environ["THREE_FLOW_RIPPLE_ASSETS"])
    data, metadata = ripple.read_verified_assets(root)
    assert set(data) == set(ripple.ASSETS)
    assert metadata["repo"] == ripple.MODEL
    copy = tmp_path / "assets"
    shutil.copytree(root, copy)
    target = copy / "tokenizer_config.json"
    target.unlink()
    target.symlink_to(root / "tokenizer_config.json")
    with pytest.raises((ValueError, OSError)):
        ripple.read_verified_assets(copy)
    target.unlink()
    shutil.copyfile(root / "tokenizer_config.json", target)
    linked = tmp_path / "linked"
    linked.symlink_to(copy, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        ripple.read_verified_assets(linked)
    target.unlink()
    os.mkfifo(target)
    with pytest.raises((ValueError, OSError)):
        ripple.read_verified_assets(copy)
    target.unlink()
    shutil.copyfile(root / "tokenizer_config.json", target)
    original_open = os.open
    moved = tmp_path / "moved"
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    switched = []

    def rename_before_member(name, flags, *args, **kwargs):
        if name == "provenance.json" and not switched:
            copy.rename(moved)
            copy.symlink_to(sentinel, target_is_directory=True)
            switched.append(True)
        return original_open(name, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", rename_before_member)
        anchored, _ = ripple.read_verified_assets(copy)
    assert switched and anchored == data
    with monkeypatch.context() as patch:
        patch.setitem(ripple.ASSETS, "../escape", "0" * 64)
        patch.setattr(
            os, "open", lambda *a, **k: pytest.fail("asset I/O before validation")
        )
        with pytest.raises(ValueError):
            ripple.read_verified_assets(root)
    with monkeypatch.context() as patch:
        patch.setitem(ripple.ASSETS, "tokenizer.json", "0" * 64)
        patch.setattr(
            os, "open", lambda *a, **k: pytest.fail("asset I/O before validation")
        )
        with pytest.raises(ValueError):
            ripple.read_verified_assets(root)


def test_ripple_catalog_rejects_duplicate_and_traversal_before_asset_io(monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    from tools import three_flow_ripple as ripple

    real_read = Path.read_bytes
    root = Path(os.environ["THREE_FLOW_RIPPLE_ASSETS"])
    original = json.loads(
        Path(ripple.__file__)
        .with_name("three_flow_river_ripple_assets.json")
        .read_bytes()
    )
    for bad_name in (original["assets"][0]["filename"], "../escape"):
        catalog = json.loads(json.dumps(original))
        catalog["assets"][1]["filename"] = bad_name
        raw = json.dumps(catalog).encode()
        with monkeypatch.context() as patch:
            patch.setattr(ripple, "CATALOG_SHA256", hashlib.sha256(raw).hexdigest())
            patch.setattr(
                Path,
                "read_bytes",
                lambda path, raw=raw: (
                    raw
                    if path.name == "three_flow_river_ripple_assets.json"
                    else real_read(path)
                ),
            )
            patch.setattr(
                os,
                "open",
                lambda *a, **k: pytest.fail("asset I/O before catalog validation"),
            )
            with pytest.raises(ValueError):
                ripple.read_verified_assets(root)


def test_ripple_asset_root_rejects_double_slash_before_directory_io(monkeypatch):
    from pathlib import Path

    from tools import three_flow_ripple as ripple
    from tools.three_flow_admission import digest, verify_ripple_assets

    root = Path(os.environ["THREE_FLOW_RIPPLE_ASSETS"])
    binding = digest(
        {
            "model": ripple.MODEL,
            "provenance_sha256": ripple.PROVENANCE_SHA256,
            "assets": ripple.ASSETS,
        }
    )
    data, metadata = ripple.read_verified_assets(root)
    assert set(data) == set(ripple.ASSETS) and metadata["repo"] == ripple.MODEL
    verify_ripple_assets({"root": str(root), "catalog_sha256": binding}, True)
    alias = "//" + str(root).lstrip("/")
    assert Path(alias).anchor == "//"
    with monkeypatch.context() as patch:
        patch.setattr(
            os,
            "open",
            lambda *a, **k: pytest.fail("directory opened before canonical check"),
        )
        for invalid in (alias, "relative/assets", "/safe/../assets"):
            with pytest.raises(ValueError):
                ripple.read_verified_assets(Path(invalid))
            with pytest.raises(ValueError):
                verify_ripple_assets({"root": invalid, "catalog_sha256": binding}, True)
