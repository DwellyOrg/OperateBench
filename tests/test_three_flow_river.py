"""Optional Python >=3.12 actual River SDK/channel offline qualification."""

import copy
import json
import os
import socket
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from operatebench.agents.playback import PlaybackError
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure, content_digest
from tools.aggregate_budget import SharedGuard
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import CONFIGURATIONS, RIVER_MODELS
from tools.three_flow_river import GRPC_OPTIONS, RiverCampaignTransport
from tools.three_flow_river_mock import RiverMockChannel
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="River SDK requires Python >=3.12; separate matrix"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    if sys.version_info >= (3, 12):
        grpc = pytest.importorskip("grpc")
        monkeypatch.setattr(grpc, "secure_channel", forbidden)
        monkeypatch.setattr(grpc, "insecure_channel", forbidden)


@pytest.fixture
def assets():
    value = os.environ.get("THREE_FLOW_RIVER_ASSETS")
    if value is None:
        pytest.skip("explicit pinned River asset cache required")
    return Path(value)


def fixture(
    tmp_path,
    assets,
    model=RIVER_MODELS[0],
    flow="commerce",
    scenario="UK_NORMAL",
    *,
    polls=0,
    mode_profile="legacy-v1",
):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="river-test", cap=Decimal("1000"), create=True
    )
    guard = SharedGuard(
        budget=budget,
        cell="trial",
        policy=LifecyclePricingPolicy(
            "OFFLINE_SYNTHETIC_NOT_CURRENT_PRICES", Decimal("1"), Decimal("2")
        ),
        max_output_tokens=32768,
    )
    channel = RiverMockChannel(model, flow, pending_polls=polls)
    transport = RiverCampaignTransport(
        model=model,
        assets=assets,
        channel=channel,
        guard=guard,
        max_output_tokens=32768,
        channel_options=GRPC_OPTIONS,
        **({"mode_profile": mode_profile} if mode_profile != "legacy-v1" else {}),
    )
    channel.tokenizer = transport.tokenizer
    trial = Trial(
        "river-test", "trial", flow, scenario, model, 32768, provider_identity(transport)
    )
    return budget, channel, transport, trial


@pytest.mark.parametrize("model", RIVER_MODELS)
def test_each_exact_model_sdk_native_roundtrip(tmp_path, assets, model):
    budget, channel, transport, trial = fixture(tmp_path, assets, model, polls=1)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        count = len(channel.submissions)
        assert count > 0 and len(channel.polls) == 2 * count
        assert len(budget.records) == count
        assert all(r["state"] == "settled" for r in budget.records.values())
        assert all(
            t["network_rpcs"] == 3 and t["model_submissions"] == 1
            for t in record["provider_turns"]
        )
        assert all(t["provider_stop_reason"] is None for t in record["provider_turns"])
        assert all(r.base_model == model for r in channel.submissions)
        assert all(
            r.prompts[0].max_tokens == 32768
            and r.prompts[0].top_k == -1
            and r.prompts[0].top_p == 1
            for r in channel.submissions
        )
        assert record["provider"]["settings"]["admission"] is False
        assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(channel.submissions) == count
        assert budget.totals()["unknown"] == 0
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("flow,scenario", CONFIGURATIONS)
def test_river_44_real_domain_roundtrip(tmp_path, assets, flow, scenario):
    budget, channel, transport, trial = fixture(
        tmp_path, assets, flow=flow, scenario=scenario
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        count = len(channel.submissions)
        assert count > 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        assert len(channel.submissions) == count
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    "fault",
    [
        "usage_missing",
        "total",
        "input",
        "output",
        "text",
        "sample_count",
        "training",
        "cache",
        "retained",
        "prompt_ids",
        "wrong_poll",
        "empty_ack",
        "network",
        "typed_shape",
        "unknown_fields",
    ],
)
@pytest.mark.parametrize("model", RIVER_MODELS)
def test_invalid_rpc_retains_liability(tmp_path, assets, fault, model):
    budget, channel, transport, trial = fixture(tmp_path, assets, model)
    original = channel.generate

    def generate(request):
        ack = original(request)
        result = channel.pending[ack.request_id]
        if fault == "usage_missing":
            result.ClearField("usage")
        elif fault == "total":
            result.usage.total_tokens += 1
        elif fault == "input":
            result.usage.prompt_tokens += 1
            result.usage.total_tokens += 1
        elif fault == "output":
            result.usage.completion_tokens += 1
            result.usage.total_tokens += 1
        elif fault == "text":
            result.results[0].text = "foreign"
        elif fault == "sample_count":
            result.results.add()
        elif fault == "training":
            result.usage.training_tokens = 1
        elif fault == "cache":
            result.results[0].cached_prompt_tokens = result.usage.prompt_tokens + 1
        elif fault == "retained":
            result.results[0].retained_kv = True
        elif fault == "prompt_ids":
            result.results[0].prompt_token_ids.append(7)
        elif fault == "empty_ack":
            ack.request_id = ""
        elif fault == "unknown_fields":
            result.MergeFromString(b"\xa0\x06\x01")
        elif fault == "network":
            raise TimeoutError("ambiguous submission")
        return ack

    channel.generate = generate
    if fault == "wrong_poll":
        channel.retrieve = lambda r: channel.pb.RetrieveFutureResponse(
            try_again=channel.pb.TryAgainResponse(request_id="foreign")
        )
    if fault == "typed_shape":
        original_factory = channel.unary_unary

        def factory(path, **kw):
            rpc = original_factory(path, **kw)

            def invoke(request, **kwargs):
                rpc(request, **kwargs)
                return object()

            return invoke

        channel.unary_unary = factory
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(channel.submissions) == 1
        assert transport.local_failure is None
        assert transport.last_turn["classification"] == (
            "provider_timeout" if fault == "network" else "provider_response_invalid"
        )
        assert budget.totals()["unknown"] > 0
        assert budget.totals()["pending"] == 0
    finally:
        channel.__dict__.pop("generate", None)
        channel.__dict__.pop("retrieve", None)
        channel.__dict__.pop("unary_unary", None)
        try:
            transport.close()
        finally:
            budget.close()


def test_failed_reservation_never_reaches_sdk_channel(tmp_path, assets, monkeypatch):
    budget, channel, transport, trial = fixture(tmp_path, assets)

    def fail(*args, **kwargs):
        raise OSError("journal unavailable")

    monkeypatch.setattr(budget, "_append", fail)
    try:
        with pytest.raises(RuntimeError, match="local campaign infrastructure failed"):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert transport.local_failure == "budget_journal_error"
        assert channel.submissions == []
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize(
    "field", ["model", "settings", "source", "decisions", "request", "evaluation"]
)
def test_resealed_record_cannot_rebind_trial_or_closure(tmp_path, assets, field):
    budget, _channel, transport, trial = fixture(tmp_path, assets)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        forged = copy.deepcopy(record)
        if field == "model":
            forged["trial"]["model"] = RIVER_MODELS[1]
        elif field == "settings":
            forged["provider"]["settings"]["temperature"] = 1
        elif field == "source":
            forged["binding"]["campaign_runtime_digest"] = "0" * 64
        elif field == "decisions":
            forged["decisions"] = []
        elif field == "request":
            forged["request_digests"][0] = "0" * 64
        elif field == "evaluation":
            forged["evaluation"]["reliable"] = False
        forged["record_digest"] = content_digest(
            {k: v for k, v in forged.items() if k != "record_digest"}
        )
        with pytest.raises((ValueError, PlaybackError)):
            replay_trial(trial, forged, expected_record_digest=record["record_digest"])
        with pytest.raises((ValueError, PlaybackError)):
            replay_trial(trial, forged, expected_record_digest=forged["record_digest"])
    finally:
        transport.close()
        budget.close()


def test_explicit_channel_no_auth_lookup_or_network(tmp_path, assets, monkeypatch):
    import builtins
    import io

    from tools.three_flow_river import ExplicitRiverChannel

    opened = []
    original_open = builtins.open
    original_io_open = io.open

    def check(file):
        if isinstance(file, (str, os.PathLike)):
            name = str(file)
            opened.append(name)
            assert Path(name).name not in {
                ".env",
                ".netrc",
                "credentials",
                "token",
                "stored_tokens",
                "auth.json",
            }

    def safe_open(file, *args, **kwargs):
        check(file)
        return original_open(file, *args, **kwargs)

    def safe_io_open(file, *args, **kwargs):
        check(file)
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", safe_open)
    monkeypatch.setattr(io, "open", safe_io_open)
    budget, mock, transport, trial = fixture(tmp_path, assets)
    calls = []

    class Channel:
        def unary_unary(self, path, **kwargs):
            rpc = mock.unary_unary(path, **kwargs)

            def invoke(request, *, timeout, metadata):
                assert metadata == (("x-api-key", "offline-not-a-credential"),)
                return rpc(request, timeout=timeout)

            return invoke

        def close(self):
            pass

    def factory(target, credentials, *, options):
        calls.append(target)
        assert target == "api.river.ai:443"
        assert credentials == "mock-tls" and options == (("grpc.enable_retries", 0),)
        return Channel()

    explicit = ExplicitRiverChannel(
        api_key="offline-not-a-credential",
        channel_factory=factory,
        credentials_factory=lambda: "mock-tls",
    )
    transport.wire.inner = explicit
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"] and calls == ["api.river.ai:443"]
        assert len(mock.submissions) > 0
    finally:
        transport.close()
        budget.close()


def test_every_submission_has_durable_journal_and_ack_before_poll(
    tmp_path, assets, monkeypatch
):
    budget, channel, transport, trial = fixture(tmp_path, assets, polls=2)
    generate, retrieve = channel.generate, channel.retrieve
    append = budget._append
    durable = []

    def persisted(event):
        append(event)  # Both journal and directory fsync must return before RPC.
        durable.append((event["kind"], event.get("id")))

    monkeypatch.setattr(budget, "_append", persisted)

    def submit(request):
        rows = [
            json.loads(line)
            for line in (tmp_path / "campaign-budget.jsonl").read_text().splitlines()
        ]
        dispatch, rpc = rows[-2:]
        assert (dispatch["kind"], rpc["kind"]) == ("dispatch", "raw_rpc")
        assert dispatch["id"] == rpc["id"]
        assert durable[-2:] == [("dispatch", dispatch["id"]), ("raw_rpc", rpc["id"])]
        assert dispatch["seq"] + 1 == rpc["seq"]
        assert rpc["ordinal"] == 1
        assert rpc["endpoint"].endswith("/InferenceGenerate")
        assert rpc["request_sha256"] == transport.wire.captures[-1]["request_sha256"]
        sent = [ident for ident, row in budget.records.items() if row["state"] == "sent"]
        assert sent == [dispatch["id"]]
        return generate(request)

    def poll(request):
        partial = (tmp_path / "trial" / "partial.ndjson").read_text()
        import base64

        raw = channel.pb.AsyncResponse(request_id=request.request_id).SerializeToString()
        assert base64.b64encode(raw).decode() in partial
        return retrieve(request)

    channel.generate, channel.retrieve = submit, poll
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert all(t["network_rpcs"] == 4 for t in record["provider_turns"])
        assert len(budget.records) == len(channel.submissions)
    finally:
        channel.__dict__.pop("generate", None)
        channel.__dict__.pop("retrieve", None)
        try:
            transport.close()
        finally:
            budget.close()


def test_dispatch_fsync_failure_no_submission(tmp_path, assets, monkeypatch):
    budget, channel, transport, trial = fixture(tmp_path, assets)
    append = budget._append

    def fail(event):
        if event["kind"] == "dispatch":
            raise OSError("dispatch journal fsync failed")
        return append(event)

    monkeypatch.setattr(budget, "_append", fail)
    try:
        with pytest.raises(RuntimeError, match="local campaign infrastructure failed"):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert transport.local_failure == "dispatch_journal_error"
        assert transport.last_turn["classification"] == "infrastructure_failure"
        assert channel.submissions == []
        assert budget.totals()["unknown"] > 0
    finally:
        transport.close()
        budget.close()


def test_unknown_survives_restart_then_http_uses_same_budget(tmp_path, assets):
    from tools.three_flow_http import HTTPCampaignTransport
    from tools.three_flow_mock import SDKMockTransport

    budget, channel, transport, trial = fixture(tmp_path, assets)
    original = channel.generate

    def ambiguous(request):
        original(request)
        raise TimeoutError("generation may be running")

    channel.generate = ambiguous
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        unknown = budget.totals()["unknown"]
        assert unknown > 0 and len(channel.submissions) == 1
    finally:
        channel.__dict__.pop("generate", None)
        try:
            transport.close()
        finally:
            budget.close()
    recovered = CampaignBudget(tmp_path, campaign_id="river-test", cap=Decimal("1000"))
    guard = SharedGuard(
        budget=recovered,
        cell="http",
        policy=LifecyclePricingPolicy(
            "OFFLINE_SYNTHETIC_NOT_CURRENT_PRICES", Decimal("1"), Decimal("2")
        ),
        max_output_tokens=128000,
    )
    http = HTTPCampaignTransport(
        provider="openai",
        api_key="offline-not-a-credential",
        inner=SDKMockTransport("openai", "commerce"),
        guard=guard,
        max_output_tokens=128000,
    )
    try:
        trial = Trial(
            "river-test",
            "http",
            "commerce",
            "UK_NORMAL",
            http.model,
            128000,
            provider_identity(http),
        )
        record = run_trial(trial, http, output=tmp_path / "http")
        assert record["evaluation"]["reliable"]
        assert recovered.totals()["unknown"] == unknown
        assert recovered.totals()["pending"] == 0
        assert {r["cell"] for r in recovered.records.values()} == {"trial", "http"}
        assert len(channel.submissions) == 1
    finally:
        http.close()
        recovered.close()


@pytest.mark.parametrize(
    "message", ["offline ambiguous generation", "dispatch_journal_error internal_error"]
)
def test_scheduler_river_correction_and_resume_share_journal(
    tmp_path, assets, monkeypatch, message
):
    from tools import three_flow_campaign as campaign

    original = RiverMockChannel.generate
    count = []

    def fault(self, request):
        count.append(1)
        original(self, request)
        raise TimeoutError(message)

    monkeypatch.setattr(RiverMockChannel, "generate", fault)
    slot = next(
        r
        for r in campaign.assignments("correct")
        if r["provider"] == "river" and r["flow"] == "commerce"
    )
    root = tmp_path / "campaign"
    rows = campaign.run_offline(
        root, campaign_id="correct", slots=[slot], river_assets=assets
    )
    assert rows[0]["classification"] == "excluded" and len(count) == 1
    assert rows[0]["fault"] == "provider_timeout"
    assert "local_failure" not in rows[0]
    partial = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_text().splitlines()
    ]
    assert partial[-1]["classification"] == "excluded"
    assert partial[-1]["exclusion_code"] == "provider_timeout"
    assert Decimal(rows[0]["unknown_liability_usd"]) > 0
    original_closure = (root / (slot["trial_id"] + ".closure.json")).read_bytes()
    monkeypatch.setattr(RiverMockChannel, "generate", original)
    # Simulate a separately identified source correction without changing fixtures.
    monkeypatch.setattr(campaign, "source_binding", lambda: "corrected-source")
    fixes = campaign.run_offline(
        root,
        campaign_id="correct",
        providers=("river",),
        resume=True,
        corrections={slot["trial_id"]: "provider_fix"},
        river_assets=assets,
    )
    assert fixes[0]["classification"] == "scored"
    assert (
        fixes[0]["trial_id"] != slot["trial_id"]
        and fixes[0]["rerun_of"] == slot["trial_id"]
    )
    assert (root / (slot["trial_id"] + ".closure.json")).read_bytes() == original_closure
    before = (root / "campaign-budget.jsonl").read_bytes()
    resumed = campaign.run_offline(
        root, campaign_id="correct", slots=[slot], resume=True, river_assets=assets
    )
    assert resumed == rows
    assert (root / "campaign-budget.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "stage,marker,submissions,polls",
    [
        ("reserve", "budget_journal_error", 0, 0),
        ("dispatch", "dispatch_journal_error", 0, 0),
        ("settle", "budget_journal_error", 1, 1),
        ("unknown", "budget_journal_error", 1, 0),
        ("wire_request", "wire_evidence_error", 0, 0),
        ("wire_response", "wire_evidence_error", 1, 0),
    ],
)
def test_scheduler_river_local_persistence_failure(
    tmp_path, assets, monkeypatch, stage, marker, submissions, polls
):
    from unittest.mock import patch

    from tools import three_flow_campaign as campaign
    from tools.three_flow_runtime import PartialExecutionEvidence

    append = CampaignBudget._append
    evidence_append = PartialExecutionEvidence.append
    generate = RiverMockChannel.generate
    retrieve = RiverMockChannel.retrieve
    calls, poll_calls = [], []

    def journal(self, event):
        if event["kind"] == stage:
            with patch("tools.aggregate_budget.os.fsync", side_effect=OSError("local")):
                return append(self, event)
        return append(self, event)

    def evidence(self, event):
        if event["kind"] == stage:
            raise OSError("local evidence persistence")
        return evidence_append(self, event)

    def submit(self, request):
        calls.append(1)
        ack = generate(self, request)
        if stage == "unknown":
            raise TimeoutError("provider unavailable")
        return ack

    def poll(self, request):
        poll_calls.append(1)
        return retrieve(self, request)

    monkeypatch.setattr(CampaignBudget, "_append", journal)
    monkeypatch.setattr(PartialExecutionEvidence, "append", evidence)
    monkeypatch.setattr(RiverMockChannel, "generate", submit)
    monkeypatch.setattr(RiverMockChannel, "retrieve", poll)
    slot = next(
        r
        for r in campaign.assignments("local")
        if r["provider"] == "river" and r["flow"] == "commerce"
    )
    root = tmp_path / "campaign"
    row = campaign.run_offline(
        root, campaign_id="local", slots=[slot], river_assets=assets
    )[0]
    assert len(calls) == submissions and len(poll_calls) == polls
    assert row["classification"] == "aborted"
    assert row["fault"] == "internal_error"
    assert row["local_failure"] == marker
    partial = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_text().splitlines()
    ]
    assert partial[-1]["classification"] == "aborted"
    assert partial[-2]["local_failure"] == marker
    closure = root / (slot["trial_id"] + ".closure.json")
    retained = closure.read_bytes()
    monkeypatch.setattr(CampaignBudget, "_append", append)
    resumed = campaign.run_offline(
        root, campaign_id="local", slots=[slot], resume=True, river_assets=assets
    )[0]
    assert resumed["classification"] == "aborted"
    assert resumed["local_failure"] == marker
    assert closure.read_bytes() == retained
    assert len(calls) == submissions and len(poll_calls) == polls
    # A durable settle row may be recovered after its fsync failed; other
    # allocated records must stay unknown, never be dispatched on resume.
    if stage != "settle":
        assert Decimal(resumed["unknown_liability_usd"]) > 0


@pytest.mark.parametrize("asset_state", ["missing", "digest"])
@pytest.mark.parametrize("stage", ["render", "response"])
def test_scheduler_native_asset_failure(
    tmp_path, assets, monkeypatch, asset_state, stage
):
    from tools import three_flow_campaign as campaign
    from tools import three_flow_river as river
    from tools.three_flow_river_native import _asset

    slot = next(
        r
        for r in campaign.assignments("asset")
        if r["provider"] == "river" and r["flow"] == "commerce" and "Qwen" in r["model"]
    )
    local_assets = tmp_path / "assets"
    if asset_state == "digest":
        target = local_assets / slot["model"] / "chat_template.jinja"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"invalid local template")
    render = river.family_render
    response = RiverCampaignTransport.response
    send = RiverCampaignTransport.send
    observed = []

    def render_fault(model, prompt, asset_root, **kwargs):
        return render(model, prompt, local_assets, **kwargs)

    def response_fault(self, *args, **kwargs):
        _asset(self.model, local_assets, "chat_template.jinja")
        return response(self, *args, **kwargs)

    def observe(self, request):
        try:
            return send(self, request)
        finally:
            observed.append((self.local_failure, self.last_turn))

    monkeypatch.setattr(RiverCampaignTransport, "send", observe)
    if stage == "render":
        monkeypatch.setattr(river, "family_render", render_fault)
    else:
        monkeypatch.setattr(RiverCampaignTransport, "response", response_fault)
    root = tmp_path / "campaign"
    row = campaign.run_offline(
        root, campaign_id="asset", slots=[slot], river_assets=assets
    )[0]
    assert row["classification"] == "aborted"
    assert row["fault"] == "internal_error"
    assert row["local_failure"] == "native_asset_error"
    marker, turn = observed[0]
    assert marker == "native_asset_error"
    assert turn["classification"] == "infrastructure_failure"
    assert turn["network_rpcs"] == (0 if stage == "render" else 2)
    partial = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_text().splitlines()
    ]
    assert partial[-1]["classification"] == "aborted"
    assert partial[-2]["local_failure"] == "native_asset_error"
    assert not (root / slot["trial_id"] / "record.json").exists()
    unknown = Decimal(row["unknown_liability_usd"])
    assert unknown == 0 if stage == "render" else unknown > 0
    budget = CampaignBudget(root, campaign_id="asset", cap=Decimal("1000"))
    try:
        assert budget.totals()["pending"] == 0
        assert budget.totals()["unknown"] == unknown
        if stage == "render":
            assert not budget.records
    finally:
        budget.close()
    closure = (root / (slot["trial_id"] + ".closure.json")).read_bytes()
    journal = (root / "campaign-budget.jsonl").read_bytes()
    resumed = campaign.run_offline(
        root, campaign_id="asset", slots=[slot], resume=True, river_assets=assets
    )[0]
    assert resumed == row
    assert (root / (slot["trial_id"] + ".closure.json")).read_bytes() == closure
    assert (root / "campaign-budget.jsonl").read_bytes() == journal
    assert len(observed) == 1


@pytest.mark.parametrize("model", RIVER_MODELS)
def test_semantic_typed_native_controls(model):
    from operatebench.agents.lifecycle_contract import outcome_tools
    from operatebench.agents.transport import ToolCall
    from tools.three_flow_river_assets import frame, synthetic_frame
    from tools.three_flow_river_kimi import parse_kimi
    from tools.three_flow_river_native import decode_response

    calls = [
        ToolCall(
            "act",
            {
                "action_type": "domain.tool",
                "payload": {
                    "amount": 1.5,
                    "flag": False,
                    "nothing": None,
                    "code": "007",
                    "nested": [1, True, {"text": "<tool_call>"}],
                },
            },
        ),
        ToolCall("wait", {"fallback_after_minutes": 30, "reason": "007"}),
        ToolCall("complete", {"reason": "complete"}),
        ToolCall("retrieve", {"requests": [{"source": "public", "id": "007"}]}),
    ]
    for call in calls:
        text = frame(call) if "Kimi" in model else synthetic_frame(model, call)
        parsed = (
            parse_kimi(text)
            if "Kimi" in model
            else decode_response(
                model, text, outcome_tools(), reasoning_prefilled=True
            ).tool_calls
        )
        assert parsed == (call,)


@pytest.mark.parametrize("model", RIVER_MODELS)
def test_malformed_native_text_is_model_error_with_known_usage(tmp_path, assets, model):
    budget, channel, transport, trial = fixture(tmp_path, assets, model)
    original = channel.generate

    def malformed(request):
        ack = original(request)
        result = channel.pending[ack.request_id]
        text = "</think><broken native tool output>"
        tokens = channel.tokenizer.encode(text)
        result.results[0].text = text
        del result.results[0].token_ids[:]
        result.results[0].token_ids.extend(tokens)
        result.usage.completion_tokens = len(tokens)
        result.usage.total_tokens = result.usage.prompt_tokens + len(tokens)
        return ack

    channel.generate = malformed
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"] is False
        assert budget.totals()["unknown"] == 0
        assert all(r["state"] == "settled" for r in budget.records.values())
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        channel.__dict__.pop("generate", None)
        try:
            transport.close()
        finally:
            budget.close()


@pytest.mark.parametrize("fault", [None, "usage_missing", "wrong_poll", "typed_shape"])
def test_native_resources_released_without_gc(tmp_path, assets, monkeypatch, fault):
    import gc
    import weakref

    refs = []
    original_fixture = fixture

    def tracked_fixture(*args, **kwargs):
        result = original_fixture(*args, **kwargs)
        budget, channel, transport, _trial = result
        refs.extend(
            weakref.ref(obj) for obj in (budget, channel, transport, transport.tokenizer)
        )
        return result

    monkeypatch.setitem(globals(), "fixture", tracked_fixture)
    enabled = gc.isenabled()
    gc.disable()
    try:
        if fault is None:
            test_each_exact_model_sdk_native_roundtrip(tmp_path, assets, RIVER_MODELS[0])
        else:
            test_invalid_rpc_retains_liability(tmp_path, assets, fault, RIVER_MODELS[0])
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
