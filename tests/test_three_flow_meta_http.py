"""Meta campaign bridge through real SDK/runtime, with synthetic capabilities only."""

import hashlib
import json
import socket
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.transport import ProviderFailure
from tests.test_three_flow_meta import profile
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import META_MODELS, META_ROSTER_PROFILE
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def bridge(tmp_path, model, fault=None, *, uncapped=False):
    tmp_path.chmod(0o700)
    from tests.test_three_flow_meta import uncapped_admission

    limit_policy = uncapped_admission()["limit_policy"] if uncapped else None
    budget = CampaignBudget(
        tmp_path,
        campaign_id="test",
        cap=None if uncapped else Decimal("990"),
        limit_policy=limit_policy,
        roster_profile=META_ROSTER_PROFILE,
        create=True,
    )
    p = profile(model)
    if uncapped:
        p["execution_policy"] = limit_policy["mode"]
    maximum = p["context_max"]
    guard = AdmittedGuard(
        profile=p,
        budget=budget,
        cell="trial",
        policy=LifecyclePricingPolicy(
            "synthetic",
            Decimal(p["input_usd_per_million"]),
            Decimal(p["output_usd_per_million"]),
        ),
        max_output_tokens=maximum,
    )
    mock = SDKMockTransport("openai", "commerce", model=model)
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == "https://api.meta.ai/v1/responses"
        assert request.headers["authorization"] == "Bearer synthetic-meta-key"
        assert "openai-organization" not in request.headers
        assert "openai-project" not in request.headers
        result = mock.handle(request)
        body = json.loads(result.content)
        body["store"] = False
        body["usage"].update(
            input_tokens_details={"cached_tokens": 1},
            output_tokens_details={"reasoning_tokens": 1},
        )
        if fault == "prose":
            body["output"] = [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ]
        if fault == "model":
            body["model"] = "muse-wonder-1.0-contributor"
        if fault == "usage":
            body["usage"]["output_tokens"] = True
        if fault == "wire":
            return httpx.Response(200, content=b"{")
        if fault == "redirect":
            return httpx.Response(
                307, headers={"location": "https://api.openai.com/v1/responses"}
            )
        response = httpx.Response(200, json=body)
        request.extensions["synthetic_response_bytes"] = response.content
        return response

    transport = HTTPCampaignTransport(
        provider="meta",
        model=model,
        api_key="synthetic-meta-key",
        inner=httpx.MockTransport(handle),
        guard=guard,
        max_output_tokens=maximum,
        context_tokens=maximum,
        mode_profile="off-or-minimum-v1",
    )
    transport.settings.update(
        bounds_policy=p["bounds_policy"], context_request_bound=maximum
    )
    trial = Trial(
        "test",
        "trial",
        "commerce",
        "UK_NORMAL",
        model,
        maximum,
        provider_identity(transport),
    )
    return budget, mock, transport, trial, requests


@pytest.mark.parametrize("model", META_MODELS)
def test_meta_runtime_sdk_bytes_partial_and_replay(tmp_path, model):
    budget, mock, transport, trial, requests = bridge(tmp_path, model)
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert record["provider"]["provider"] == "meta"
        assert (
            record["provider"]["request_mapping"]
            == "three-flow-meta-snapshot-minimal-auto-v3"
        )
        assert len(requests) == len(record["provider_turns"]) > 1
        for request, body, turn in zip(
            requests, mock.calls, record["provider_turns"], strict=True
        ):
            assert set(body) == {
                "model",
                "max_output_tokens",
                "instructions",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "store",
                "input",
                "reasoning",
            }
            assert body["model"] == model
            assert body["tool_choice"] == transport.settings["tool_choice"] == "auto"
            assert body["reasoning"] == {"effort": "minimal"}
            assert body["store"] is body["parallel_tool_calls"] is False
            assert all(
                t["type"] == "function" and t["strict"] is False for t in body["tools"]
            )
            assert "three-flow-meta-snapshot-minimal-auto-v3" in body["instructions"]
            assert "openai" not in body["instructions"]
            assert turn["wire"]["request_utf8"] == request.content.decode()
            actual_response = request.extensions["synthetic_response_bytes"]
            assert turn["wire"]["response_utf8"].encode() == actual_response
            assert (
                turn["wire"]["response_sha256"]
                == hashlib.sha256(actual_response).hexdigest()
            )
            raw_body = json.loads(actual_response)
            assert raw_body["store"] is False
            assert "cache_write_tokens" not in raw_body["usage"]["input_tokens_details"]
            assert turn["output_tokens"] == turn["input_tokens"] == 1
            assert body["max_output_tokens"] > 4096
            assert request.extensions["timeout"] == {
                "connect": 10,
                "pool": 10,
                "read": None,
                "write": None,
            }
        assert (tmp_path / "trial" / "partial.ndjson").exists()
        assert budget.totals()["unknown"] == 0
        expected = (
            (
                Decimal(profile(model)["input_usd_per_million"])
                + Decimal(profile(model)["output_usd_per_million"])
            )
            * len(requests)
            / 1000000
        )
        assert budget.totals()["known"] == expected
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        events = [
            json.loads(line)
            for line in (tmp_path / "campaign-budget.jsonl").read_text().splitlines()
        ]
        assert sum(e["kind"] == "raw_rpc" for e in events) == len(requests)
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", META_MODELS)
@pytest.mark.parametrize("fault", ["model", "usage", "wire", "redirect"])
def test_meta_faults_keep_unknown_and_partial(tmp_path, model, fault):
    budget, _, transport, trial, requests = bridge(tmp_path, model, fault)
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(requests) == 1
        assert budget.totals()["unknown"] > 0
        assert (tmp_path / "trial" / "partial.ndjson").exists()
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", META_MODELS)
def test_meta_plain_text_is_paid_and_excluded(tmp_path, model):
    budget, _, transport, trial, requests = bridge(tmp_path, model, "prose")
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert len(requests) == 1
        assert budget.totals()["known"] > 0
        assert budget.totals()["unknown"] == 0
        assert not (tmp_path / "trial" / "record.json").exists()
        events = [
            json.loads(line)
            for line in (tmp_path / "trial" / "partial.ndjson").read_text().splitlines()
        ]
        assert events[-1]["classification"] == "excluded"
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", META_MODELS)
@pytest.mark.parametrize("mutation", ["temperature", "required"])
def test_sdk_unexpected_field_refused_before_rpc(tmp_path, model, mutation):
    budget, _, transport, trial, requests = bridge(tmp_path, model)
    create = transport.client.responses.with_raw_response.create

    def changed(**kwargs):
        if mutation == "required":
            kwargs["tool_choice"] = "required"
        else:
            kwargs["temperature"] = 0
        return create(**kwargs)

    transport.client.responses.with_raw_response.create = changed
    try:
        with pytest.raises(ProviderFailure):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert requests == []
        events = (tmp_path / "campaign-budget.jsonl").read_text()
        assert '"raw_rpc"' not in events
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", [None, "muse-wonder-1.0-contributor", "gpt-6-astra"])
def test_invalid_meta_model_no_key_fallback(tmp_path, model):
    from types import SimpleNamespace

    from operatebench.providers.meta_responses import MetaConfigurationError

    with pytest.raises(MetaConfigurationError):
        HTTPCampaignTransport(
            provider="meta",
            model=model,
            api_key="",
            inner=httpx.MockTransport(lambda _: None),
            guard=SimpleNamespace(),
            max_output_tokens=1048576,
        )


@pytest.mark.parametrize("model", META_MODELS)
@pytest.mark.parametrize("mutation", ["endpoint", "account"])
def test_mutated_sdk_capability_never_reaches_inner(tmp_path, model, mutation):
    from operatebench.providers.meta_responses import MetaConfigurationError

    budget, _, transport, trial, requests = bridge(tmp_path, model)
    if mutation == "endpoint":
        transport.client.base_url = "https://api.openai.com/v1"
    else:
        transport.client.project = "synthetic-project"
    try:
        with pytest.raises(MetaConfigurationError):
            run_trial(trial, transport, output=tmp_path / "trial")
        assert requests == []
        assert '"raw_rpc"' not in (tmp_path / "campaign-budget.jsonl").read_text()
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("uncapped", [False, True])
def test_source_bound_meta_consume_all_88(tmp_path, uncapped):
    """Run after commit: deliberately requires the production clean-source gate."""
    from pathlib import Path

    from tests.test_three_flow_latest_http import admission
    from tests.test_three_flow_live import write
    from tools import three_flow_live as live
    from tools.three_flow_admission import GUARANTEES
    from tools.three_flow_campaign import assignments
    from tools.three_flow_profiles import META_ROSTER_PROFILE, registry

    sha = live.source_identity()
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    credential = custody / "dummy.env"
    credential.write_text("MODEL_API_KEY=synthetic-meta-key\n")
    credential.chmod(0o600)
    a = admission()
    if uncapped:
        a["campaign_id"] = "meta-distinct-uncapped-series"
    slots = assignments(a["campaign_id"], META_ROSTER_PROFILE)
    a.update(
        source_sha=sha,
        cap_usd="990",
        roster_profile=META_ROSTER_PROFILE,
        registry=registry(
            roster_profile=META_ROSTER_PROFILE,
            mode_profile="off-or-minimum-v1",
        ),
        campaign_root=str(tmp_path / "campaign"),
        custody_root=str(custody),
        credential_file=str(credential),
        selected=[s["trial_id"] for s in slots],
        pending=[s["trial_id"] for s in slots],
        profiles={},
    )
    for model in META_MODELS:
        a["profiles"][model] = dict(
            profile(model),
            rate_evidence_sha256="a" * 64,
            bounds_evidence_sha256="b" * 64,
            pricing_policy="conservative-all-tier-envelope-v1",
            guarantees={
                **dict.fromkeys(GUARANTEES, True),
                "provider_bounds_verified": False,
            },
        )
    if uncapped:
        from tools.three_flow_budget import META_UNCAPPED_PROFILE

        a["cap_usd"] = None
        a["limit_policy"] = {
            "mode": META_UNCAPPED_PROFILE,
            "owner_accepted": True,
            "owner_evidence_sha256": "a" * 64,
            "source_sha": sha,
            "historical_liability_manifest_sha256": "b" * 64,
        }
        for p in a["profiles"].values():
            p["execution_policy"] = META_UNCAPPED_PROFILE
    path = custody / "admission.json"
    write(path, a)

    class Network:
        def __init__(self):
            self.mocks = {}

        def http(self, slot):
            mock = SDKMockTransport("openai", slot["flow"], model=slot["model"])
            self.mocks[slot["trial_id"]] = mock
            return mock

    network = Network()
    rows = live.consume(path, sha, network=network)
    assert len(rows) == 88
    assert all(row["classification"] == "scored" for row in rows)
    root = Path(a["campaign_root"])
    assert len(network.mocks) == 88
    expected = Decimal(0)
    for slot in slots:
        ident = slot["trial_id"]
        mock = network.mocks[ident]
        record = json.loads((root / ident / "record.json").read_text())
        trial = Trial(**json.loads((root / (ident + ".binding.json")).read_text()))
        assert isinstance(record["evaluation"]["reliable"], bool)
        assert (root / ident / "partial.ndjson").exists()
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
        row = next(r for r in a["registry"] if r["model"] == slot["model"])
        assert record["provider"]["request_mapping"] == row["mapping"]
        assert row["mapping"] == "three-flow-meta-snapshot-minimal-auto-v3"
        assert record["provider"]["settings"]["tool_choice"] == "auto"
        assert all(call["tool_choice"] == "auto" for call in mock.calls)
        for key, value in row["request_settings"].items():
            assert record["provider"]["settings"][key] == value
        p = profile(slot["model"])
        expected += (
            (Decimal(p["input_usd_per_million"]) + Decimal(p["output_usd_per_million"]))
            * len(mock.calls)
            / 1000000
        )
    budget = CampaignBudget(
        root,
        campaign_id=a["campaign_id"],
        cap=None if uncapped else Decimal("990"),
        limit_policy=a.get("limit_policy"),
        roster_profile=META_ROSTER_PROFILE,
    )
    try:
        assert budget.totals()["known"] == expected
        assert budget.totals()["unknown"] == 0
    finally:
        budget.close()
    with pytest.raises(ValueError):
        live.consume(path, sha, network=network)
    assert len(network.mocks) == 88
    scorecard = json.loads((root / "scorecard.json").read_text())
    if uncapped:
        assert scorecard["global_budget"]["remaining"] is None
        assert scorecard["global_budget"]["cap"] is None


@pytest.mark.parametrize("model", META_MODELS)
@pytest.mark.parametrize(
    "flow,scenario",
    [("maintenance", "V1"), ("commerce", "UK_NORMAL"), ("compliance", "EN_normal")],
)
def test_meta_admitted_factory_each_flow(tmp_path, model, flow, scenario):
    from tests.test_three_flow_latest_http import admission
    from tools.three_flow_admission import GUARANTEES, validate
    from tools.three_flow_campaign import assignments
    from tools.three_flow_live_factory import live_factory
    from tools.three_flow_profiles import META_ROSTER_PROFILE, registry

    a = admission()
    slot = next(
        s
        for s in assignments(a["campaign_id"], META_ROSTER_PROFILE)
        if s["model"] == model and s["flow"] == flow and s["scenario"] == scenario
    )
    p = dict(
        profile(model),
        rate_evidence_sha256="a" * 64,
        bounds_evidence_sha256="b" * 64,
        pricing_policy="conservative-all-tier-envelope-v1",
        guarantees={**dict.fromkeys(GUARANTEES, True), "provider_bounds_verified": False},
    )
    a.update(
        cap_usd="990",
        roster_profile=META_ROSTER_PROFILE,
        registry=registry(
            roster_profile=META_ROSTER_PROFILE,
            mode_profile="off-or-minimum-v1",
        ),
        selected=[slot["trial_id"]],
        pending=[slot["trial_id"]],
        profiles={model: p},
    )
    validate(a, a["source_sha"], synthetic=True)
    mock = SDKMockTransport("openai", flow, model=model)

    class Network:
        def http(self, selected):
            assert selected == slot
            return mock

    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id=a["campaign_id"],
        cap=Decimal("990"),
        roster_profile=META_ROSTER_PROFILE,
        create=True,
    )
    transport = live_factory(
        a, {"MODEL_API_KEY": "synthetic-meta-key"}, network=Network()
    )(slot, budget)
    trial = Trial(
        a["campaign_id"],
        slot["trial_id"],
        flow,
        scenario,
        model,
        p["context_max"],
        provider_identity(transport),
    )
    try:
        record = run_trial(trial, transport, output=tmp_path / "trial")
        assert record["evaluation"]["reliable"]
        assert record["provider"]["settings"]["admission"] is True
        assert record["provider"]["request_mapping"] == a["registry"][0]["mapping"]
        expected = (
            (Decimal(p["input_usd_per_million"]) + Decimal(p["output_usd_per_million"]))
            * len(mock.calls)
            / 1000000
        )
        assert budget.totals()["known"] == expected
        assert budget.totals()["unknown"] == 0
        assert (
            replay_trial(trial, record, expected_record_digest=record["record_digest"])[
                "provider_calls"
            ]
            == 0
        )
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", META_MODELS)
def test_guard_uses_shared_campaign_exposure(tmp_path, model):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path,
        campaign_id="shared-meta",
        cap=Decimal("990"),
        roster_profile=META_ROSTER_PROFILE,
        create=True,
    )
    try:
        # Another model's irreversible unknown exposure leaves only USD 1.
        first = profile(META_MODELS[0])
        reservation = budget.reserve(
            META_MODELS[0],
            input_bound=0,
            output_bound=49450000,
            policy=LifecyclePricingPolicy("synthetic", Decimal("4"), Decimal("20")),
        )
        budget.forfeit(reservation)
        p = profile(model)
        guard = AdmittedGuard(
            profile=p,
            budget=budget,
            cell=model,
            policy=LifecyclePricingPolicy(
                "synthetic",
                Decimal(p["input_usd_per_million"]),
                Decimal(p["output_usd_per_million"]),
            ),
            max_output_tokens=first["context_max"],
        )
        from operatebench.providers.cost import CostCapExceededError

        assert guard.output_bound(0) == int(
            Decimal("1000000") / Decimal(p["output_usd_per_million"])
        )
        with pytest.raises(CostCapExceededError):
            guard.authorize(input_tokens_upper_bound=0)
        with pytest.raises(FileExistsError):
            CampaignBudget(
                tmp_path,
                campaign_id=model,
                cap=Decimal("990"),
                roster_profile=META_ROSTER_PROFILE,
                create=True,
            )
        assert budget.totals()["unknown"] == 989
    finally:
        budget.close()


def test_meta_rate_windows_and_predispatch_failure():
    from tools.three_flow_errors import MetaDispatchStopped, MetaRatePacer

    now = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    pacer = MetaRatePacer(clock=lambda: now[0], sleep=sleep)
    sent = []

    def dispatch(model, tokens, status=200):
        def send():
            pacer.commit(model, tokens)
            sent.append((model, now[0]))
            if status == 429:
                pacer.stopped.set()
            return httpx.Response(status)

        return pacer.call(model, tokens, send)

    for _ in range(3):
        dispatch("wonder", 1_030_000)
    dispatch("spark", 1_030_000)
    assert sleeps == []
    dispatch("wonder", 1_030_000)
    assert sleeps == [60]
    assert sent[-1] == ("wonder", 60)
    before = dict(pacer.windows)
    with pytest.raises(ValueError):
        pacer.call("wonder", 1, lambda: (_ for _ in ()).throw(ValueError()))
    assert pacer.windows == before
    dispatch("spark", 1, 429)
    with pytest.raises(MetaDispatchStopped):
        dispatch("wonder", 1)
    assert len(sent) == 6


def test_meta_rpm_boundary():
    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    pacer = MetaRatePacer(
        clock=lambda: now[0], sleep=lambda d: now.__setitem__(0, now[0] + d)
    )

    def send():
        pacer.commit("wonder", 1)
        return httpx.Response(200)

    for _ in range(3001):
        pacer.call("wonder", 1, send)
    assert now[0] == 60
    assert len(pacer.windows["wonder"]) == 1


def test_meta_real_wire_boundary_fake_clock(tmp_path):
    from tools.three_flow_errors import MetaRatePacer

    budget, _, transport, trial, requests = bridge(tmp_path, META_MODELS[0])
    now = [0.0]
    transport.wire.meta_pacer = MetaRatePacer(
        clock=lambda: now[0], sleep=lambda d: now.__setitem__(0, now[0] + d)
    )
    record = run_trial(trial, transport, output=tmp_path / "trial")
    assert requests
    assert record["provider_turns"]
    window = transport.wire.meta_pacer.windows[META_MODELS[0]]
    assert sum(n for _, n in window) <= 4_000_000
    assert transport.settings["max_retries"] == 0
    transport.close()
    budget.close()


def test_meta_local_wire_tripwire_consumes_no_provider_quota(tmp_path):
    from tools.three_flow_errors import MetaRatePacer

    budget, _, transport, _, requests = bridge(tmp_path, META_MODELS[0])
    pacer = MetaRatePacer()
    transport.wire.meta_pacer = pacer
    transport.wire.expected_payload = {"different": True}
    request = httpx.Request(
        "POST",
        transport.settings["endpoint"],
        json={"model": META_MODELS[0], "max_output_tokens": 1},
    )
    with pytest.raises(ValueError, match="SDK wire differs"):
        transport.wire.handle_request(request)
    assert pacer.windows[META_MODELS[0]] == []
    assert not pacer.stopped.is_set()
    assert requests == []
    transport.close()
    budget.close()


def test_meta_threads_share_model_window():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    clock_lock = threading.Lock()

    def sleep(delay):
        with clock_lock:
            now[0] += delay

    pacer = MetaRatePacer(clock=lambda: now[0], sleep=sleep)
    sent = []

    def send():
        pacer.commit("wonder", 1_000_000)
        sent.append(now[0])
        return httpx.Response(200)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: pacer.call("wonder", 1_000_000, send), range(12))
        )
    assert len(results) == 12
    for timestamp in sent:
        assert sum(timestamp <= t < timestamp + 60 for t in sent) <= 4


@pytest.mark.parametrize("fault", ["stopped", "captured", "429", "inflight"])
def test_meta_stop_cancels_only_proven_unsent(tmp_path, fault):
    from tools.three_flow_errors import MetaRatePacer

    budget, _, transport, trial, requests = bridge(tmp_path, META_MODELS[0])
    pacer = MetaRatePacer(enabled=False)
    transport.wire.meta_pacer = pacer
    if fault in ("stopped", "captured"):
        pacer.stopped.set()
        if fault == "captured":
            transport.wire.captures.append({"status_code": 200})
    else:

        def fail(request):
            requests.append(request)
            if fault == "inflight":
                raise httpx.ReadError("synthetic in-flight failure")
            return httpx.Response(429, json={"error": {"message": "synthetic"}})

        transport.wire.inner = httpx.MockTransport(fail)
    try:
        with pytest.raises((ProviderFailure, RuntimeError)):
            run_trial(trial, transport, output=tmp_path / "trial")
        events = [
            json.loads(line)
            for line in (tmp_path / "campaign-budget.jsonl").read_text().splitlines()
        ]
        assert len(requests) == (0 if fault in ("stopped", "captured") else 1)
        assert sum(e["kind"] == "raw_rpc" for e in events) == len(requests)
        assert (
            budget.totals()["unknown"] == 0
            if fault == "stopped"
            else budget.totals()["unknown"] > 0
        )
        assert not (tmp_path / "trial" / "record.json").exists()
        assert pacer.stopped.is_set()
    finally:
        transport.close()
        budget.close()


@pytest.mark.parametrize("model", META_MODELS)
@pytest.mark.parametrize("pacing", [False, True])
def test_uncapped_meta_live_shaped_timeout(tmp_path, model, pacing):
    from tools.three_flow_errors import MetaRatePacer

    budget, _, transport, trial, requests = bridge(tmp_path, model, uncapped=True)
    now = [0.0]
    transport.wire.meta_pacer = (
        MetaRatePacer(
            clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
        )
        if pacing
        else None
    )
    try:
        run_trial(trial, transport, output=tmp_path / "trial")
        assert requests[0].extensions["timeout"] == {
            "connect": 10,
            "pool": 10,
            "read": None,
            "write": None,
        }
    finally:
        transport.close()
        budget.close()


def test_meta_two_models_progress_independently():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tools.three_flow_errors import MetaRatePacer

    pacer = MetaRatePacer()
    entered = threading.Event()
    release = threading.Event()

    def wonder():
        pacer.commit(META_MODELS[0], 1_000_000)
        entered.set()
        assert release.wait(5)
        return httpx.Response(200)

    def spark():
        pacer.commit(META_MODELS[1], 1_000_000)
        return httpx.Response(200)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(pacer.call, META_MODELS[0], 1_000_000, wonder)
        try:
            assert entered.wait(5)
            second = pool.submit(pacer.call, META_MODELS[1], 1_000_000, spark)
            assert second.result(timeout=5).status_code == 200
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert set(pacer.windows) == set(META_MODELS)


@pytest.mark.parametrize("mutation", ["source", "policy"])
def test_real_shaped_uncapped_refusal_precredential(tmp_path, monkeypatch, mutation):
    from tests.test_three_flow_live import write
    from tests.test_three_flow_meta import uncapped_admission
    from tools import three_flow_live as live
    from tools.three_flow_admission import validate

    a = uncapped_admission()
    a.update(approved=True, mode="paid")
    sha = a["source_sha"]
    validate(a, sha, synthetic=False)
    if mutation == "source":
        a["limit_policy"]["source_sha"] = "0" * 40
    else:
        a["limit_policy"]["mode"] = "three-flow-optional-limits-v1"
    path = tmp_path / "admission.json"
    write(path, a)

    def forbidden(*args, **kwargs):
        raise AssertionError("refused admission reached credential custody")

    monkeypatch.setattr(live, "private_directory", forbidden)
    original_fd = live.private_fd

    def admission_only(value):
        if value != str(path):
            forbidden()
        return original_fd(value)

    monkeypatch.setattr(live, "private_fd", admission_only)
    with pytest.raises(ValueError):
        live.consume(path, sha)


def test_meta_preflight_optional_policy_is_description_only():
    from tools.three_flow_admission import proposal
    from tools.three_flow_campaign import preflight
    from tools.three_flow_profiles import registry

    report = proposal("c" * 40, META_ROSTER_PROFILE)
    assert report["cap_usd"] == "990"
    assert report["diagnostic_ceiling_usd"] == "10"
    assert "integration is not ready" not in " ".join(report["blockers"])
    for value in (report, preflight(META_ROSTER_PROFILE)):
        assert value["admitted"] is False
        assert value["optional_successor"]["cap_usd"] is None
        assert value["optional_successor"]["admitted"] is False
        assert (
            value["optional_successor"]["limit_policy"] == "three-flow-meta-uncapped-v1"
        )
    for row in registry(
        roster_profile=META_ROSTER_PROFILE, mode_profile="off-or-minimum-v1"
    ):
        assert "pacing_enabled" not in row["request_settings"]
        assert row["request_settings"]["tokens_per_minute"] == 4_000_000
        assert row["request_settings"]["requests_per_minute"] == 3000
        assert row["request_settings"]["automatic_resubmission"] is False


def quota_headers(tokens="3999999", requests="2999"):
    return {
        "x-ratelimit-limit-tokens": "4000000",
        "x-ratelimit-remaining-tokens": tokens,
        "x-ratelimit-limit-requests": "3000",
        "x-ratelimit-remaining-requests": requests,
    }


def test_meta_header_authority_fast_serialized_requests():
    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    sleeps = []
    pacer = MetaRatePacer(clock=lambda: now[0], sleep=sleeps.append)
    sent = []

    def send():
        pacer.commit("wonder", 1_030_000)
        sent.append(now[0])
        now[0] += 0.1
        return httpx.Response(200, headers=quota_headers())

    for _ in range(44):
        pacer.call("wonder", 1_030_000, send)
    assert len(sent) == 44
    assert sleeps == []
    assert sum(n for _, n in pacer.windows["wonder"]) > 4_000_000


@pytest.mark.parametrize("tokens,requests", [("100", "2999"), ("3999999", "0")])
def test_meta_low_headers_wait_and_models_independent(tokens, requests):
    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    pacer = MetaRatePacer(clock=lambda: now[0], sleep=sleep)

    def send(model, headers):
        pacer.commit(model, 1_030_000)
        return httpx.Response(200, headers=headers)

    pacer.call(
        "wonder", 1_030_000, lambda: send("wonder", quota_headers(tokens, requests))
    )
    pacer.call("spark", 1_030_000, lambda: send("spark", quota_headers()))
    assert sleeps == []
    pacer.call("wonder", 1_030_000, lambda: send("wonder", {}))
    assert sleeps == [60]
    assert "wonder" not in pacer.remaining


@pytest.mark.parametrize(
    "field,value",
    [
        ("remaining-tokens", None),
        ("remaining-tokens", "-1"),
        ("remaining-tokens", "1.0"),
        ("remaining-tokens", "+1"),
        ("remaining-tokens", "4000001"),
        ("remaining-requests", "3001"),
        ("limit-tokens", "100"),
        ("limit-requests", "0"),
        ("limit-tokens", "4000001"),
    ],
)
def test_meta_invalid_headers_clear_authority_and_fallback(field, value):
    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    pacer = MetaRatePacer(
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
    )
    headers = quota_headers()
    if value is None:
        del headers["x-ratelimit-" + field]
    else:
        headers["x-ratelimit-" + field] = value

    def send(response_headers):
        pacer.commit("wonder", 1_030_000)
        return httpx.Response(200, headers=response_headers)

    pacer.call("wonder", 1_030_000, lambda: send(quota_headers()))
    pacer.call("wonder", 1_030_000, lambda: send(headers))
    assert "wonder" not in pacer.remaining
    pacer.call("wonder", 1_030_000, lambda: send({}))
    pacer.call("wonder", 1_030_000, lambda: send({}))
    assert now[0] == 60


def test_meta_stale_headers_restore_conservative_window():
    from tools.three_flow_errors import MetaRatePacer

    now = [0.0]
    pacer = MetaRatePacer(
        clock=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)
    )

    def send(headers):
        pacer.commit("wonder", 1_030_000)
        return httpx.Response(200, headers=headers)

    pacer.call("wonder", 1_030_000, lambda: send(quota_headers()))
    now[0] = 60
    for _ in range(4):
        pacer.call("wonder", 1_030_000, lambda: send({}))
    assert now[0] == 120
