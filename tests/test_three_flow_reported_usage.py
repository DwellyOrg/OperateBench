"""Owner-consented usage estimates, never invoice assurance or live authority."""

from __future__ import annotations

import copy
import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from tests.test_three_flow_context import bind_combined, diagnostic
from tests.test_three_flow_live import Network, cli, no_network, write
from tests.test_three_flow_owner_pricing import assumed_fixture
from tools.three_flow_admission import validate
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import assignments, score_row
from tools.three_flow_runtime import Trial, replay_trial

__all__ = ["no_network"]
POLICY = "reported-usage-diagnostic-v1"
UNVERIFIED = (
    "account_access_verified",
    "all_billable_input_in_usage",
    "hidden_reasoning_in_output_usage_and_limit",
    "no_other_fees",
)


def reported(p, *, documented_reasoning=False):
    p["accounting_policy"] = {
        "mode": POLICY,
        "owner_accepted": True,
        "owner_evidence_sha256": "d" * 64,
    }
    for key in UNVERIFIED:
        p["guarantees"][key] = False
    p["guarantees"]["hidden_reasoning_in_output_usage_and_limit"] = documented_reasoning
    bind_combined(p)
    return p


@pytest.mark.parametrize("reasoning", [False, True])
def test_explicit_accounting_accepts_honest_booleans(tmp_path, reasoning):
    _, a, slot = assumed_fixture(tmp_path)
    p = reported(a["profiles"][slot["model"]], documented_reasoning=reasoning)
    before = copy.deepcopy(p)
    assert validate(a, a["source_sha"], synthetic=True) == a
    assert p == before


@pytest.mark.parametrize(
    "mutation", ["absent", "consent", "evidence", "profile", "source", "boolean"]
)
def test_accounting_requires_consent_and_existing_bindings(tmp_path, mutation):
    _, a, slot = assumed_fixture(tmp_path)
    p = reported(a["profiles"][slot["model"]])
    if mutation == "absent":
        del p["accounting_policy"]
    elif mutation == "consent":
        p["accounting_policy"]["owner_accepted"] = False
    elif mutation == "evidence":
        p["accounting_policy"]["owner_evidence_sha256"] = ""
    elif mutation == "profile":
        p["network_timeout_seconds"] += 1
    elif mutation == "source":
        p["owner_pricing"]["source_sha"] = "0" * 40
    else:
        p["guarantees"]["no_other_fees"] = 0
    if mutation != "profile":
        bind_combined(p)
    with pytest.raises(ValueError):
        validate(a, a["source_sha"], synthetic=True)


def test_reported_usage_cli_settlement_exclusion_same_journal(tmp_path):
    path, a, slot = assumed_fixture(tmp_path, "mistral")
    p = reported(diagnostic(a["profiles"][slot["model"]]))
    slots = [s for s in assignments(a["campaign_id"]) if s["model"] == slot["model"]][:2]
    a["selected"] = a["pending"] = [s["trial_id"] for s in slots]
    write(path, a)

    class Mixed(Network):
        def http(self, selected):
            lane = Network("missing_usage" if selected == slots[1] else None)
            self.calls.append(selected["trial_id"])
            return lane.http(selected)

    network = Mixed()
    assert cli(path, a, network) == 2
    root = Path(a["campaign_root"])
    report = json.loads((root / "scorecard.json").read_text())
    rows = report["rows"]
    assert [r["classification"] for r in rows] == ["scored", "excluded"]
    assert Decimal(rows[0]["known_cost_usd"]) > 0
    assert Decimal(rows[1]["unknown_liability_usd"]) > 0
    with (root / "scorecard.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    markdown = (root / "scorecard.md").read_text()
    for row, csv_row in zip(rows, csv_rows, strict=True):
        basis = row["pricing_basis"]
        assert basis["accounting_policy"] == p["accounting_policy"]
        assert basis["ceiling_basis"] == "estimated-usage-not-invoice"
        assert basis["additional_fees"] == "unknown-not-included"
        assert basis["actual_invoice_guaranteed"] is False
        assert json.loads(csv_row["pricing_basis"]) == basis
        details = [
            json.loads(part.split("```")[0]) for part in markdown.split("```json\n")[1:]
        ]
        assert details[rows.index(row)]["pricing_basis"] == basis
        assert POLICY in markdown and "estimated-usage-not-invoice" in markdown
    record = json.loads((root / slot["trial_id"] / "record.json").read_text())
    assert record["provider"]["settings"]["model_contract_verified"] is False
    trial = Trial(**json.loads((root / (slot["trial_id"] + ".binding.json")).read_text()))
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    # Scoring eligibility follows actual origin, not contract verification.
    candidate = copy.deepcopy(record)
    candidate["evidence_origin"] = "PROVIDER_CANDIDATE"
    assert score_row(slot, candidate)["eligible_for_live_results"] is True
    events = [
        json.loads(line)
        for line in (root / "campaign-budget.jsonl").read_text().splitlines()
    ]
    assert sum(e["kind"] == "genesis" for e in events) == 1
    budget = CampaignBudget(root, campaign_id=a["campaign_id"], cap=Decimal("1000"))
    try:
        totals = budget.totals()
        assert totals["known"] > 0 and totals["unknown"] > 0
        assert totals["pending"] == 0
        assert totals["known"] + totals["pending"] + totals["unknown"] <= 1000
        assert Decimal(rows[0]["known_cost_usd"]) == totals["known"]
    finally:
        budget.close()
    calls = list(network.calls)
    assert cli(path, a, network) == 2
    assert calls == network.calls
