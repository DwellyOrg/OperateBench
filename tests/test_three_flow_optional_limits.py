"""Explicit successor only; synthetic accounting and SDKs, no paid authority."""

import json
import os
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.test_three_flow_budget import policy
from tests.test_three_flow_context import diagnostic
from tests.test_three_flow_live import cli, no_network, write
from tests.test_three_flow_owner_pricing import assumed_fixture
from tests.test_three_flow_reported_usage import reported
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_runtime import Trial, replay_trial

__all__ = ["no_network"]
POLICY = "three-flow-optional-limits-v1"


class UnlimitedNetwork:
    def __init__(self):
        self.requests = []
        self.rpcs = []

    def http(self, slot):
        inner = SDKMockTransport(slot["provider"], slot["flow"])

        def respond(request):
            assert request.extensions["timeout"] == dict.fromkeys(
                ("connect", "read", "write", "pool"), None
            )
            self.requests.append(json.loads(request.read()))
            return inner.handle_request(request)

        return httpx.MockTransport(respond)

    def river(self, slot):
        from tools.three_flow_river_mock import RiverMockChannel

        network = self

        class Channel(RiverMockChannel):
            def unary_unary(self, path, **kwargs):
                def invoke(request, *, timeout):
                    assert timeout is None
                    network.rpcs.append((path, request.SerializeToString()))
                    # Real generated protobuf round trip, no network. Reuse
                    # fixture response generator without its historical 120 check.
                    request = type(request).FromString(
                        kwargs["request_serializer"](request)
                    )
                    response = (
                        self.generate(request)
                        if path.endswith("InferenceGenerate")
                        else self.retrieve(request)
                    )
                    return kwargs["response_deserializer"](response.SerializeToString())

                return invoke

        return Channel(slot["model"], slot["flow"], pending_polls=1)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral", "river"])
@pytest.mark.parametrize("mode_profile", ["legacy-v1", "off-or-minimum-v1"])
def test_optional_consuming_cli_real_sdk_nullable_controls_and_replay(
    tmp_path, provider, mode_profile, monkeypatch
):
    if provider == "river" and "THREE_FLOW_RIVER_ASSETS" not in os.environ:
        pytest.skip("separate existing native interpreter")
    path, a, slot = assumed_fixture(tmp_path, provider)
    if mode_profile != "legacy-v1":
        from tools.three_flow_admission import digest, registry

        a.update(mode_profile=mode_profile, registry=registry(mode_profile=mode_profile))
        a["profiles"][slot["model"]]["owner_pricing"]["registry_entry_sha256"] = digest(
            next(r for r in a["registry"] if r["model"] == slot["model"])
        )
    p = diagnostic(a["profiles"][slot["model"]])
    if provider in ("openai", "anthropic"):
        p.update(output_max=128000, context_max=140000)
    p.update(network_timeout_seconds=None, execution_policy=POLICY)
    reported(p)
    a.update(cap_usd=None, limit_policy={**consent(), "source_sha": a["source_sha"]})
    write(path, a)
    if provider == "river" and mode_profile != "legacy-v1":
        import tools.three_flow_river_mock as mock

        for name in ("frame", "synthetic_frame"):
            original = getattr(mock, name)

            def selected(*args, original=original):
                return original(*args).removeprefix("</think>")

            monkeypatch.setattr(mock, name, selected)
    network = UnlimitedNetwork()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    settings = record["provider"]["settings"]
    if mode_profile != "legacy-v1":
        assert settings["mode_profile"] == mode_profile
        assert record["provider"]["request_mapping"].endswith("-" + mode_profile)
    assert settings["limit_policy"] == a["limit_policy"]
    assert settings["execution_controls"] == {
        "monetary_cap_usd": None,
        "model_calls": None,
        "cumulative_tokens": None,
        "generation_seconds": None,
        "network_timeout_seconds": None,
    }
    assert settings["dedicated_output_max"] == p["output_max"]
    assert not any(settings["provider_guarantees"].values())
    report = json.loads((root / "scorecard.json").read_text())
    assert report["global_budget"]["remaining"] is None
    assert report["global_budget"]["cap"] is None
    assert Decimal(report["global_budget"]["known"]) > 0
    assert report["limit_policy"] == a["limit_policy"]
    assert "no monetary ceiling" in (root / "scorecard.md").read_text()
    assert (
        json.loads((Path(a["custody_root"]) / "SERIES-CONSUMED").read_text())[
            "limit_policy"
        ]
        == a["limit_policy"]
    )
    assert len(record["request_digests"]) >= 2
    if provider == "river":
        submissions = [r for r in network.rpcs if r[0].endswith("InferenceGenerate")]
        assert len(submissions) == len(record["request_digests"])
        assert len(network.rpcs) == 3 * len(submissions)
    else:
        assert len(network.requests) == len(record["request_digests"])
    for turn, output in zip(
        record["provider_turns"], record["request_output_bounds"], strict=True
    ):
        assert (
            type(output) is int and 0 < output <= p["context_max"] - turn["input_bound"]
        )
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    before = (len(network.requests), len(network.rpcs))
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    assert (len(network.requests), len(network.rpcs)) == before
    assert cli(path, a, network) == 2
    assert (len(network.requests), len(network.rpcs)) == before


def consent():
    return {
        "mode": POLICY,
        "owner_accepted": True,
        "owner_evidence_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "historical_liability_manifest_sha256": "c" * 64,
    }


def test_optional_money_retains_known_unknown_pending_and_reopen(tmp_path):
    tmp_path.chmod(0o700)
    options = {"campaign_id": "new-series", "cap": None, "limit_policy": consent()}
    budget = CampaignBudget(tmp_path, create=True, **options)
    try:
        unknown = budget.reserve(
            "unknown", input_bound=100, output_bound=2000, policy=policy()
        )
        budget.forfeit(unknown)
        measured = budget.reserve(
            "measured", input_bound=100, output_bound=3000, policy=policy()
        )
        budget.dispatch("measured", {"max_output_tokens": 3000})
        budget.settle(measured, input_tokens=0, output_tokens=1500)
        budget.reserve("pending", input_bound=100, output_bound=4000, policy=policy())
        assert budget.cap is None
        assert budget.totals() == {
            "known": 1500,
            "unknown": 2000,
            "pending": 4000,
            "exposure": 7500,
            "remaining": None,
        }
    finally:
        budget.close()
    genesis = json.loads((tmp_path / "campaign-budget.jsonl").read_text().splitlines()[0])
    assert genesis["cap"] is None
    assert genesis["limit_policy"] == consent()
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, campaign_id="new-series", cap=Decimal("1000"))
    budget = CampaignBudget(tmp_path, **options)
    try:
        assert budget.totals() == {
            "known": 1500,
            "unknown": 6000,
            "pending": 0,
            "exposure": 7500,
            "remaining": None,
        }
        budget.reserve("next", input_bound=100, output_bound=5000, policy=policy())
        assert budget.exposure == 12500
    finally:
        budget.close()


@pytest.mark.parametrize(
    "mutation", ["owner", "source", "history", "cap", "policy", "timeout", "context"]
)
def test_optional_admission_fail_closed_before_credentials(
    tmp_path, monkeypatch, mutation
):
    from tools import three_flow_live as live

    path, a, slot = assumed_fixture(tmp_path, "mistral")
    p = diagnostic(a["profiles"][slot["model"]])
    p.update(network_timeout_seconds=None, execution_policy=POLICY)
    reported(p)
    a.update(cap_usd=None, limit_policy={**consent(), "source_sha": a["source_sha"]})
    if mutation == "owner":
        a["limit_policy"]["owner_accepted"] = False
    elif mutation == "source":
        a["limit_policy"]["source_sha"] = "0" * 40
    elif mutation == "history":
        del a["limit_policy"]["historical_liability_manifest_sha256"]
    elif mutation == "cap":
        a["cap_usd"] = "1000"
    elif mutation == "policy":
        del a["limit_policy"]
    elif mutation == "timeout":
        p["network_timeout_seconds"] = 120
        reported(p)
    else:
        p["context_max"] = None
        reported(p)
    write(path, a)
    original = live.private_fd
    credential_pins = []

    def custody_tripwire(value, **kwargs):
        if value == a["credential_file"]:
            credential_pins.append(value)
            raise AssertionError("credential pin forbidden")
        return original(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", custody_tripwire)
    assert cli(path, a, UnlimitedNetwork()) == 2
    assert credential_pins == []
    assert not (Path(a["custody_root"]) / "SERIES-CONSUMED").exists()
    assert not Path(a["campaign_root"]).exists()


def test_optional_cli_missing_usage_retains_unknown_and_other_cells(tmp_path):
    from tools.three_flow_campaign import assignments

    path, a, slot = assumed_fixture(tmp_path, "mistral")
    p = diagnostic(a["profiles"][slot["model"]])
    p.update(network_timeout_seconds=None, execution_policy=POLICY)
    reported(p)
    a.update(cap_usd=None, limit_policy={**consent(), "source_sha": a["source_sha"]})
    slots = [s for s in assignments(a["campaign_id"]) if s["model"] == slot["model"]][:2]
    a["selected"] = a["pending"] = [s["trial_id"] for s in slots]
    write(path, a)

    class Mixed(UnlimitedNetwork):
        def http(self, selected):
            inner = super().http(selected)

            def respond(request):
                result = inner.handle_request(request)
                if selected == slots[0]:
                    body = json.loads(result.read())
                    body.pop("usage")
                    return httpx.Response(200, json=body)
                return result

            return httpx.MockTransport(respond)

    network = Mixed()
    assert cli(path, a, network) == 2
    root = Path(a["campaign_root"])
    report = json.loads((root / "scorecard.json").read_text())
    assert [r["classification"] for r in report["rows"]] == ["excluded", "scored"]
    totals = report["global_budget"]
    assert totals["cap"] is None and totals["remaining"] is None
    assert Decimal(totals["known"]) > 0 and Decimal(totals["unknown"]) > 0
    budget = CampaignBudget(
        root, campaign_id=a["campaign_id"], cap=None, limit_policy=a["limit_policy"]
    )
    try:
        from tools.aggregate_budget import decimal

        assert str(decimal(budget.totals()["known"])) == totals["known"]
        assert str(decimal(budget.totals()["unknown"])) == totals["unknown"]
        assert budget.totals()["remaining"] is None
    finally:
        budget.close()


def test_null_requires_explicit_successor_and_historical_reference(tmp_path):
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError):
        CampaignBudget(tmp_path, campaign_id="test", cap=None, create=True)
    assert not (tmp_path / "campaign-budget.jsonl").exists()
