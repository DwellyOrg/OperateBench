"""Explicit synthetic owner assumptions: no real authority, credentials or network."""

from __future__ import annotations

import copy
import csv
import json
import socket
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from operatebench.agents.pricing import LifecyclePricingPolicy
from tests.test_three_flow_live import DUMMY, Network, cli, fixture, write
from tools.three_flow_admission import digest, registry, validate
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_live import source_identity
from tools.three_flow_runtime import Trial, replay_trial


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


POLICY = "owner-assumed-token-envelope-v1"
CATEGORIES = ("input", "cache_read", "cache_write", "output", "reasoning")


def assumed_fixture(tmp_path, provider="anthropic"):
    path, a, slot = fixture(tmp_path, provider)
    p = a["profiles"][slot["model"]]
    p.update(
        pricing_policy=POLICY,
        input_usd_per_million="12",
        output_usd_per_million="14.4375",
    )
    p["guarantees"]["all_input_cache_write_read_tiers_covered"] = False
    p["guarantees"]["all_output_tiers_covered"] = False
    p["owner_pricing"] = {
        "owner_accepted": True,
        "currency": "USD",
        "unit_tokens": 1000000,
        "unknown_usd_per_million": "10",
        "owner_evidence_sha256": "c" * 64,
        "source_sha": a["source_sha"],
        "registry_entry_sha256": digest(
            next(r for r in registry() if r["model"] == slot["model"])
        ),
        "profile_sha256": digest(
            {k: v for k, v in p.items() if k != "rate_evidence_sha256"}
        ),
        "verified_rate_evidence_sha256": p["rate_evidence_sha256"],
        "rates": {
            c: {
                "verified_usd_per_million": (
                    ["1", "12"] if c == "input" else ["14.4375"] if c == "output" else []
                ),
                "assume_unknown": True,
            }
            for c in CATEGORIES
        },
    }
    p["rate_evidence_sha256"] = digest(p["owner_pricing"])
    return path, a, slot


def test_explicit_owner_profile_accepts_without_claiming_provider_verification(tmp_path):
    _, a, slot = assumed_fixture(tmp_path)
    assert validate(a, source_identity(), synthetic=True) == a
    p = a["profiles"][slot["model"]]
    assert p["input_usd_per_million"] == "12"
    assert p["output_usd_per_million"] == "14.4375"
    assert p["guarantees"]["all_input_cache_write_read_tiers_covered"] is False


class AggregateCacheNetwork(Network):
    def http(self, slot):
        inner = super().http(slot)

        def response(request):
            result = inner.handle_request(request)
            body = json.loads(result.read())
            usage = body["usage"]
            key = (
                "input_tokens_details"
                if slot["provider"] == "openai"
                else "prompt_tokens_details"
            )
            usage[key] = {"cached_tokens": 1}
            return httpx.Response(200, json=body)

        return httpx.MockTransport(response)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "mistral"])
def test_owner_cli_journal_scorecard_usage_and_replay(tmp_path, capsys, provider):
    path, a, slot = assumed_fixture(tmp_path, provider)
    p = a["profiles"][slot["model"]]
    write(path, a)
    network = Network("cache") if provider == "anthropic" else AggregateCacheNetwork()
    assert cli(path, a, network) == 0
    root = Path(a["campaign_root"])
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    settings = record["provider"]["settings"]
    assert settings["model_contract_verified"] is False
    assert settings["owner_pricing_coverage_accepted"] is True
    assert settings["owner_pricing"] == p["owner_pricing"]
    assert (
        settings["accounting"]
        == "owner-assumed envelope estimate; not guaranteed invoice"
    )
    row = json.loads((root / "scorecard.json").read_text())["rows"][0]
    assert row["pricing_basis"] == {
        "policy": POLICY,
        "owner_record_sha256": digest(p["owner_pricing"]),
        "profile_sha256": digest(p),
        "cost_is_estimate_under_owner_assumption": True,
        "actual_invoice_guaranteed": False,
    }
    markdown = (root / "scorecard.md").read_text()
    details = json.loads(markdown.split("```json\n")[1].split("```")[0])
    for key in ("evidence_origin", "eligible_for_live_results", "pricing_basis"):
        assert details[key] == row[key]
    with (root / "scorecard.csv").open() as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["evidence_origin"] == details["evidence_origin"]
    assert csv_row["eligible_for_live_results"] == "False"
    assert json.loads(csv_row["pricing_basis"]) == details["pricing_basis"]
    assert details["evidence_origin"] == "SDK_MOCK_NOT_LLM"
    assert details["eligible_for_live_results"] is False
    assert details["pricing_basis"]["actual_invoice_guaranteed"] is False
    assert row["classification"] == "scored"
    assert row["eligible_for_live_results"] is False
    expected = Decimal(0)
    for turn in record["provider_turns"]:
        u = json.loads(turn["wire"]["response_utf8"])["usage"]
        if provider == "anthropic":
            assert turn["input_tokens"] == u["input_tokens"] + 5 + 7
        else:
            key = "input_tokens" if provider == "openai" else "prompt_tokens"
            assert turn["input_tokens"] == u[key]  # Cached subset already included.
        expected += (
            Decimal(turn["input_tokens"]) * 12
            + Decimal(turn["output_tokens"]) * Decimal("14.4375")
        ) / 1000000
    assert Decimal(row["estimated_usd"]) == expected
    assert Decimal(row["known_cost_usd"]) == expected
    events = [
        json.loads(line)
        for line in (root / "campaign-budget.jsonl").read_text().splitlines()
    ]
    reserves = [e for e in events if e["kind"] == "reserve"]
    policy = LifecyclePricingPolicy(
        "admitted-envelope-" + digest(p), Decimal("12"), Decimal("14.4375")
    )
    assert reserves and all(e["digest"] == policy.digest_sha256 for e in reserves)
    assert all(e["ir"] == "12" and e["or"] == "14.4375" for e in reserves)
    assert sum(e["kind"] == "genesis" for e in events) == 1
    assert events[0]["cap"] == "1000"
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    replay_trial(trial, record, expected_record_digest=record["record_digest"])
    with_budget = CampaignBudget(root, campaign_id=a["campaign_id"], cap=Decimal("1000"))
    try:
        assert len(with_budget.records) == len(reserves)
    finally:
        with_budget.close()
    before = list(network.calls)
    assert cli(path, a, network) == 2
    assert network.calls == before
    output = capsys.readouterr()
    assert DUMMY not in output.out + output.err
    for file in root.rglob("*"):
        if file.is_file():
            assert DUMMY.encode() not in file.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_consent",
        "false_consent",
        "units",
        "currency",
        "unknown_rate",
        "changed_known_rate",
        "changed_nonmax_rate",
        "envelope",
        "source",
        "settings",
        "evidence",
        "profile",
        "missing_category",
        "provider_claim",
        "bounds",
        "access",
        "usage",
        "reasoning",
        "fees",
    ],
)
def test_owner_admission_rejects_incomplete_or_changed_record(
    tmp_path, mutation, monkeypatch
):
    path, a, slot = assumed_fixture(tmp_path)
    p = a["profiles"][slot["model"]]
    o = p["owner_pricing"]
    if mutation == "missing_consent":
        del o["owner_accepted"]
    elif mutation == "false_consent":
        o["owner_accepted"] = False
    elif mutation == "units":
        o["unit_tokens"] = 1000
    elif mutation == "currency":
        o["currency"] = "EUR"
    elif mutation == "unknown_rate":
        o["unknown_usd_per_million"] = "1"
    elif mutation == "changed_nonmax_rate":
        o["rates"]["input"]["verified_usd_per_million"][0] = "2"
    elif mutation == "changed_known_rate":
        o["rates"]["input"]["verified_usd_per_million"] = ["1", "11"]
    elif mutation == "envelope":
        p["output_usd_per_million"] = "10"
        o["profile_sha256"] = digest(
            {
                k: v
                for k, v in p.items()
                if k not in ("owner_pricing", "rate_evidence_sha256")
            }
        )
    elif mutation == "source":
        o["source_sha"] = "0" * 40
    elif mutation == "settings":
        o["registry_entry_sha256"] = "0" * 64
    elif mutation == "evidence":
        o["owner_evidence_sha256"] = ""
    elif mutation == "profile":
        p["input_max"] -= 1
    elif mutation == "missing_category":
        del o["rates"]["cache_write"]
    else:
        key = {
            "provider_claim": "all_input_cache_write_read_tiers_covered",
            "bounds": "provider_bounds_verified",
            "access": "account_access_verified",
            "usage": "all_billable_input_in_usage",
            "reasoning": "hidden_reasoning_in_output_usage_and_limit",
            "fees": "no_other_fees",
        }[mutation]
        p["guarantees"][key] = mutation == "provider_claim"
        o["profile_sha256"] = digest(
            {
                k: v
                for k, v in p.items()
                if k not in ("owner_pricing", "rate_evidence_sha256")
            }
        )
    if mutation in (
        "envelope",
        "provider_claim",
        "bounds",
        "access",
        "usage",
        "reasoning",
        "fees",
    ):
        p["rate_evidence_sha256"] = digest(o)
    write(path, a)
    from tools import three_flow_live as live

    original = live.private_fd
    pinned = []

    def pin(value, **kwargs):
        pinned.append(value)
        assert value != a["credential_file"], "credential pin before rejected admission"
        return original(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", pin)
    with pytest.raises(ValueError):
        validate(a, source_identity(), synthetic=True)
    assert cli(path, a, Network()) == 2
    assert a["credential_file"] not in pinned
    assert not Path(a["campaign_root"]).exists()
    assert not (path.parent / "SERIES-CONSUMED").exists()


@pytest.mark.parametrize("usage_shape", ["missing", None, [], "invalid"])
def test_owner_failed_usage_retains_conditional_pricing_label(tmp_path, usage_shape):
    class InvalidUsageNetwork(Network):
        def http(self, slot):
            inner = super().http(slot)

            def response(request):
                result = inner.handle_request(request)
                body = json.loads(result.read())
                if usage_shape != "missing":
                    body["usage"] = usage_shape
                return httpx.Response(200, json=body)

            return httpx.MockTransport(response)

    path, a, _slot = assumed_fixture(tmp_path)
    write(path, a)
    assert cli(path, a, InvalidUsageNetwork("missing_usage")) == 2
    root = Path(a["campaign_root"])
    row = json.loads((root / "scorecard.json").read_text())["rows"][0]
    details = json.loads(
        (root / "scorecard.md").read_text().split("```json\n")[1].split("```")[0]
    )
    for key in ("evidence_origin", "eligible_for_live_results", "pricing_basis"):
        assert details[key] == row.get(key)
    assert details["evidence_origin"] is None  # Exclusion retained no model record.
    assert details["eligible_for_live_results"] is False
    assert row["classification"] == "excluded"
    assert row["fault"] == "provider_response_invalid"
    assert row["raw_rpc_attempts"] == 1
    assert Decimal(row["unknown_liability_usd"]) > 0
    assert row["pricing_basis"]["actual_invoice_guaranteed"] is False
    assert row["pricing_basis"]["cost_is_estimate_under_owner_assumption"] is True


def test_default_verified_only_does_not_implicitly_assume_unknowns(tmp_path):
    _, a, slot = fixture(tmp_path)
    p = a["profiles"][slot["model"]]
    original = copy.deepcopy(a)
    validate(a, source_identity(), synthetic=True)
    assert a == original
    p["guarantees"]["all_input_cache_write_read_tiers_covered"] = False
    with pytest.raises(ValueError, match="unknown rate or token coverage"):
        validate(a, source_identity(), synthetic=True)
