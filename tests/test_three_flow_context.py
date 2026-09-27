"""Explicit diagnostic policy, real consuming CLI/SDK; no paid authority."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.test_three_flow_live import Network, cli, fixture, no_network, write
from tests.test_three_flow_owner_pricing import assumed_fixture
from tools.three_flow_admission import digest, validate_profile
from tools.three_flow_runtime import Trial, replay_trial

# Reuse the scoped socket tripwire, not a global environment mutation.
__all__ = ["no_network"]


def diagnostic(p):
    p.update(input_max=None, output_max=None, context_max=256000)
    p["guarantees"]["provider_bounds_verified"] = False
    p["bounds_policy"] = {
        "mode": "published-context-diagnostic-v1",
        "source": "https://docs.mistral.ai/models/mistral-medium-3-5-26-04",
        "scope": "endpoint",
        "interpretation": "conservative-decimal-shorthand",
    }
    return p


def bind_combined(p):
    # Rebind synthetic owner consent to the complete combined diagnostic profile.
    p["owner_pricing"]["profile_sha256"] = digest(
        {k: v for k, v in p.items() if k not in ("owner_pricing", "rate_evidence_sha256")}
    )
    p["rate_evidence_sha256"] = digest(p["owner_pricing"])


def assert_combined(record, p):
    settings = record["provider"]["settings"]
    assert settings["bounds_policy"] == p["bounds_policy"]
    assert settings["owner_pricing"] == p["owner_pricing"]
    assert settings["pricing_basis"]["cost_is_estimate_under_owner_assumption"] is True
    assert settings["pricing_basis"]["actual_invoice_guaranteed"] is False
    assert settings["model_contract_verified"] is False
    assert p["guarantees"]["provider_bounds_verified"] is False
    assert p["guarantees"]["all_output_tiers_covered"] is False
    assert p["input_usd_per_million"] == "12"
    assert p["output_usd_per_million"] == "14.4375"
    assert p["owner_pricing"]["unknown_usd_per_million"] == "10"


class ObservedNetwork(Network):
    def __init__(self, status=None):
        super().__init__()
        self.requests = []
        self.status = status

    def http(self, slot):
        inner = super().http(slot)

        def respond(request):
            self.requests.append(json.loads(request.read()))
            if self.status:
                return httpx.Response(
                    self.status, json={"error": {"message": "invalid max_tokens"}}
                )
            return inner.handle_request(request)

        return httpx.MockTransport(respond)


@pytest.mark.parametrize("provider", ["mistral", "openai", "anthropic"])
def test_context_consuming_cli_sdk_dynamic_identity_replay(tmp_path, provider):
    from operatebench.agents.transport import content_digest
    from operatebench.providers.cost import request_input_token_bound

    path, a, slot = assumed_fixture(tmp_path, provider)
    p = diagnostic(a["profiles"][slot["model"]])
    if provider != "mistral":
        p.update(output_max=128000, context_max=140000)
    bind_combined(p)
    write(path, a)
    network = ObservedNetwork()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    assert len(network.requests) > 1
    assert len(set(record["request_output_bounds"])) > 1
    assert_combined(record, p)
    assert record["provider"]["settings"]["dedicated_output_max"] == p["output_max"]
    for body, bound in zip(
        network.requests, record["request_output_bounds"], strict=True
    ):
        if provider == "mistral":
            assert body["stream"] is False
        assert body.get("max_output_tokens", body.get("max_tokens")) == bound
        assert (
            bound + request_input_token_bound(body, "test actual SDK") <= p["context_max"]
        )
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    calls = len(network.requests)
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    assert len(network.requests) == calls
    changed = copy.deepcopy(record)
    changed["request_output_bounds"][0] -= 1
    changed["record_digest"] = content_digest(
        {k: v for k, v in changed.items() if k != "record_digest"}
    )
    with pytest.raises(ValueError, match="differs"):
        replay_trial(trial, changed, expected_record_digest=changed["record_digest"])
    assert cli(path, a, network) == 2
    assert len(network.requests) == calls


@pytest.mark.parametrize("status", [400, 422])
def test_context_provider_refusal_one_call_unknown_liability(tmp_path, status):
    path, a, slot = assumed_fixture(tmp_path, "mistral")
    p = diagnostic(a["profiles"][slot["model"]])
    bind_combined(p)
    write(path, a)
    network = ObservedNetwork(status)
    assert cli(path, a, network) == 2
    root = Path(a["campaign_root"])
    closure = json.loads((root / (slot["trial_id"] + ".closure.json")).read_text())
    assert len(network.requests) == 1
    assert closure["classification"] == "excluded"
    assert closure["evaluation"] is None
    assert closure["diagnostic_category"] == "provider_request_refusal"
    assert closure["safe_status"] == status
    assert closure["automatic_resubmission"] is False
    assert Decimal(closure["unknown_liability_usd"]) > 0
    partial = (root / slot["trial_id"] / "partial.ndjson").read_text()
    assert '"body_retained":false' in partial
    assert "invalid max_tokens" not in partial


def test_strict_unknown_refused_context_policy_does_not_remove_other_guards(tmp_path):
    _, a, slot = fixture(tmp_path, "mistral")
    p = a["profiles"][slot["model"]]
    p["output_max"] = None
    with pytest.raises(ValueError):
        validate_profile(p, "mistral")
    diagnostic(p)
    validate_profile(p, "mistral")
    for key, value in [
        ("context_max", None),
        ("context_max", True),
        ("context_max", 0),
        ("network_timeout_seconds", None),
    ]:
        changed = copy.deepcopy(p)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_profile(changed, "mistral")
    p["guarantees"]["hidden_reasoning_in_output_usage_and_limit"] = False
    with pytest.raises(ValueError):
        validate_profile(p, "mistral")


def test_known_output_limit_and_shared_remaining_intersection(tmp_path):
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_live_factory import live_factory

    _, a, slot = fixture(tmp_path, "mistral")
    p = diagnostic(a["profiles"][slot["model"]])
    p["output_max"] = 32768
    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    budget = CampaignBudget(root, campaign_id="test", cap=Decimal("1000"), create=True)
    transport = live_factory(a, {"MISTRAL_API_KEY": "dummy"}, network=Network())(
        slot, budget
    )
    try:
        assert transport.guard.output_bound(1000) == 32768
        budget.cap = budget.cap / 100000
        output = transport.guard.output_bound(1000)
        assert 0 < output < 32768
        assert (Decimal(1000) + Decimal(output) * 2) / 1000000 <= Decimal("0.01")
        with pytest.raises(ValueError):
            transport.guard.check_context(256000, 1)
        with pytest.raises(ValueError):
            transport.guard.check_context(10, 32769)
    finally:
        transport.close()
        budget.close()


def test_sdk_added_defaults_exceed_context_refused_before_wire(tmp_path, monkeypatch):
    from tools.three_flow_http import HTTPCampaignTransport

    path, a, slot = fixture(tmp_path, "mistral")
    diagnostic(a["profiles"][slot["model"]])
    write(path, a)
    # Fault injection: omit the SDK default in pre-projection only. Final actual
    # body guard must independently refuse rather than spend beyond context.
    monkeypatch.setattr(HTTPCampaignTransport, "wire_payload", lambda self, p: p)
    network = ObservedNetwork()
    assert cli(path, a, network) == 2
    assert network.requests == []
    events = [
        json.loads(line)
        for line in (Path(a["campaign_root"]) / "campaign-budget.jsonl")
        .read_text()
        .splitlines()
    ]
    assert not any(e["kind"] == "raw_rpc" for e in events)


def test_model_valid_no_tool_response_remains_scored(tmp_path):
    path, a, slot = fixture(tmp_path, "mistral")
    diagnostic(a["profiles"][slot["model"]])
    write(path, a)

    class ModelNoAction(Network):
        def http(self, slot):
            inner = super().http(slot)

            def respond(request):
                result = inner.handle_request(request)
                body = json.loads(result.read())
                body["choices"][0]["message"]["tool_calls"] = []
                body["choices"][0]["finish_reason"] = "stop"
                return httpx.Response(200, json=body)

            return httpx.MockTransport(respond)

    assert cli(path, a, ModelNoAction()) == 0
    closure = json.loads(
        (Path(a["campaign_root"]) / (slot["trial_id"] + ".closure.json")).read_text()
    )
    assert closure["classification"] == "scored"
    assert "diagnostic_category" not in closure


def test_native_context_consuming_cli_dynamic_identity_replay(tmp_path, monkeypatch):
    import base64
    import hashlib
    import os

    from operatebench.agents.transport import content_digest
    from tools.three_flow_river import sdk

    if "THREE_FLOW_RIVER_ASSETS" not in os.environ:
        pytest.skip("native diagnostic runs alone under qualified River interpreter")
    import grpc

    def forbidden(*args, **kwargs):
        raise AssertionError("native network forbidden")

    monkeypatch.setattr(grpc, "secure_channel", forbidden)
    monkeypatch.setattr(grpc, "insecure_channel", forbidden)
    path, a, slot = assumed_fixture(tmp_path, "river")
    p = diagnostic(a["profiles"][slot["model"]])
    p["context_max"] = 32000
    p["bounds_policy"]["source"] = "https://docs.river.ai/"
    from tests.test_three_flow_reported_usage import reported

    reported(p)
    bind_combined(p)
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    assert record["evidence_origin"] == "SDK_MOCK_NOT_LLM"
    assert_combined(record, p)
    assert record["provider"]["settings"]["dedicated_output_max"] is None
    assert record["provider"]["settings"]["input_measure"] == "native-tokenizer"
    assert record["provider"]["settings"]["accounting_policy"] == p["accounting_policy"]
    assert record["provider"]["settings"]["provider_guarantees"] == p["guarantees"]
    assert len(set(record["request_output_bounds"])) > 1
    pb, _ = sdk()
    for bound, turn in zip(
        record["request_output_bounds"], record["provider_turns"], strict=True
    ):
        submission = next(row for row in turn["wire"] if row["model_submission"])
        proto = pb.InferenceGenerateRequest.FromString(
            base64.b64decode(submission["request_base64"])
        )
        assert proto.prompts[0].max_tokens == bound
        assert len(proto.prompts[0].input_ids) == turn["input_bound"]
        assert turn["input_bound"] + bound == p["context_max"]
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    before = list(network.calls)
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    assert network.calls == before
    changed = copy.deepcopy(record)
    submission = next(
        row for row in changed["provider_turns"][0]["wire"] if row["model_submission"]
    )
    proto = pb.InferenceGenerateRequest.FromString(
        base64.b64decode(submission["request_base64"])
    )
    proto.prompts[0].max_tokens -= 1
    raw = proto.SerializeToString()
    submission["request_base64"] = base64.b64encode(raw).decode()
    submission["request_sha256"] = hashlib.sha256(raw).hexdigest()
    changed["record_digest"] = content_digest(
        {k: v for k, v in changed.items() if k != "record_digest"}
    )
    with pytest.raises(ValueError, match="native wire output/input differs"):
        replay_trial(trial, changed, expected_record_digest=changed["record_digest"])
    assert cli(path, a, network) == 2
    assert network.calls == before


def test_native_preparation_uses_actual_tokenizer_not_http_bytes(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from operatebench.agents.transport import ModelRequest
    from tools import three_flow_river as river

    seen = []
    prompt = {"complete": "observation"}
    monkeypatch.setattr(
        river,
        "family_render",
        lambda model, p, assets, **kwargs: "full-native-frame:" + p["complete"],
    )
    tokenizer = SimpleNamespace(encode=lambda text: seen.append(text) or list(range(123)))
    transport = object.__new__(river.RiverCampaignTransport)
    transport.mode_profile = "legacy-v1"
    transport.model, transport.assets, transport.tokenizer = (
        "dummy",
        Path("/unused"),
        tokenizer,
    )
    transport.guard = SimpleNamespace(
        diagnostic=True, output_bound=lambda bound: 32000 - bound
    )
    request = ModelRequest(
        model="dummy",
        protocol_version="dummy",
        max_output_tokens=32000,
        tool_names=(),
        observation_digest_sha256="a" * 64,
        prompt_digest_sha256="b" * 64,
        invocation_index=0,
        turn_index=0,
        prompt=prompt,
    )
    prepared = transport.prepare_request(request)
    assert prepared == replace(request, max_output_tokens=31877)
    assert seen == ["full-native-frame:observation"]
