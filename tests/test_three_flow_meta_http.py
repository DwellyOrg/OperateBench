"""Meta campaign bridge through real SDK/runtime, with synthetic capabilities only."""

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
from tools.three_flow_profiles import META_MODELS
from tools.three_flow_runtime import Trial, provider_identity, replay_trial, run_trial


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def bridge(tmp_path, model, fault=None):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="test", cap=Decimal("1000"), create=True
    )
    p = profile(model)
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
        body["usage"].update(
            input_tokens_details={"cached_tokens": 1},
            output_tokens_details={"reasoning_tokens": 1},
        )
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
        return httpx.Response(200, json=body)

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
            record["provider"]["request_mapping"] == "three-flow-meta-snapshot-minimal-v1"
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
            assert body["reasoning"] == {"effort": "minimal"}
            assert body["store"] is body["parallel_tool_calls"] is False
            assert all(
                t["type"] == "function" and t["strict"] is False for t in body["tools"]
            )
            assert "three-flow-meta-snapshot-minimal-v1" in body["instructions"]
            assert "openai" not in body["instructions"]
            assert turn["wire"]["request_utf8"] == request.content.decode()
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
def test_sdk_unexpected_field_refused_before_rpc(tmp_path, model):
    budget, _, transport, trial, requests = bridge(tmp_path, model)
    create = transport.client.responses.with_raw_response.create

    def changed(**kwargs):
        return create(**kwargs, temperature=0)

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


def test_source_bound_meta_consume_all_88(tmp_path):
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
    slots = assignments(a["campaign_id"], META_ROSTER_PROFILE)
    a.update(
        source_sha=sha,
        roster_profile=META_ROSTER_PROFILE,
        registry=registry(
            roster_profile=META_ROSTER_PROFILE, mode_profile="off-or-minimum-v1"
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
        for key, value in row["request_settings"].items():
            assert record["provider"]["settings"][key] == value
        p = profile(slot["model"])
        expected += (
            (Decimal(p["input_usd_per_million"]) + Decimal(p["output_usd_per_million"]))
            * len(mock.calls)
            / 1000000
        )
    budget = CampaignBudget(root, campaign_id=a["campaign_id"], cap=Decimal("1000"))
    try:
        assert budget.totals()["known"] == expected
        assert budget.totals()["unknown"] == 0
    finally:
        budget.close()
    with pytest.raises(ValueError):
        live.consume(path, sha, network=network)
    assert len(network.mocks) == 88


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
        roster_profile=META_ROSTER_PROFILE,
        registry=registry(
            roster_profile=META_ROSTER_PROFILE, mode_profile="off-or-minimum-v1"
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
        tmp_path, campaign_id=a["campaign_id"], cap=Decimal("1000"), create=True
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
