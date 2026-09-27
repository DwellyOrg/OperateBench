"""Actual shared scheduler, refusal, source and private-report closure tests."""

import json
import socket
import threading
from decimal import Decimal

import pytest

from tools.three_flow_budget import PROFILE, CampaignBudget
from tools.three_flow_campaign import assignments, preflight, run_offline
from tools.three_flow_mock import SDKMockTransport


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_exact_inventory_and_closed_preflight():
    rows = assignments("test")
    assert len(rows) == 616
    assert len({r["trial_id"] for r in rows}) == 616
    assert len({r["model"] for r in rows}) == 14
    assert {
        f: len({r["scenario"] for r in rows if r["flow"] == f})
        for f in ("maintenance", "commerce", "compliance")
    } == {"maintenance": 3, "commerce": 16, "compliance": 25}
    report = preflight()
    assert not report["admitted"] and not report["authority_created"]
    assert all(not p["admitted"] for p in report["providers"])
    assert (
        "current official price unknown"
        in next(
            p
            for p in report["providers"]
            if p["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        )["blockers"]
    )


def test_four_worker_sdk_dispatch_overlap_and_reports(tmp_path, monkeypatch):
    barrier = threading.Barrier(4, timeout=30)
    original = SDKMockTransport.handle
    seen = set()
    lock = threading.Lock()
    calls = []

    def handle(self, request):
        with lock:
            first = id(self) not in seen
            seen.add(id(self))
            calls.append(threading.get_ident())
        if first:
            barrier.wait()
        return original(self, request)

    monkeypatch.setattr(SDKMockTransport, "handle", handle)
    slots = [
        s
        for s in assignments("test")
        if s["provider"] == "openai" and s["flow"] == "commerce"
    ][:4]
    root = tmp_path / "campaign"
    rows = run_offline(root, campaign_id="test", slots=slots)
    assert len(set(calls)) == 4
    assert all(r["classification"] == "scored" for r in rows), rows
    assert all(r["evidence_origin"] == "SDK_MOCK_NOT_LLM" for r in rows)
    assert all(not r["eligible_for_live_results"] for r in rows)
    assert all(r["milestones"] for r in rows)
    assert all(Decimal(r["known_cost_usd"]) > 0 for r in rows)
    markdown = (root / "scorecard.md").read_text()
    assert '"milestones"' in markdown and '"caused_by"' in markdown
    assert len(list(root.glob("*.closure.json"))) == 4
    assert all((root / f"scorecard.{ext}").exists() for ext in ("json", "csv", "md"))
    assert (root.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in root.glob("*.json"))
    budget = CampaignBudget(root, campaign_id="test", cap=Decimal("1000"))
    try:
        assert budget.totals()["known"] > 0
        assert budget.totals()["unknown"] == 0
        assert (
            budget.cell_accounting(slots[0]["trial_id"])["controller_profile"] == PROFILE
        )
    finally:
        budget.close()
    with pytest.raises(FileExistsError):
        run_offline(root, campaign_id="test", slots=slots)


def test_duplicate_slot_refused_before_output(tmp_path):
    slot = assignments("test")[-1]
    with pytest.raises(ValueError, match="duplicate"):
        run_offline(tmp_path / "campaign", campaign_id="test", slots=[slot, slot])
    assert not (tmp_path / "campaign").exists()


def test_river_missing_assets_refused_without_dispatch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no River substitute")

    monkeypatch.setattr(SDKMockTransport, "__init__", forbidden)
    with pytest.raises(ValueError, match="explicit pinned assets"):
        run_offline(
            tmp_path / "campaign", campaign_id="test", slots=assignments("test")[:1]
        )
    assert not (tmp_path / "campaign").exists()


def test_boolean_journal_sequence_refused(tmp_path):
    tmp_path.chmod(0o700)
    budget = CampaignBudget(
        tmp_path, campaign_id="test", cap=Decimal("1000"), create=True
    )
    budget.close()
    path = tmp_path / "campaign-budget.jsonl"
    row = json.loads(path.read_text())
    row["seq"] = False
    from tools.aggregate_budget import canonical

    path.write_bytes(canonical(row))
    with pytest.raises(ValueError, match="torn or reordered"):
        CampaignBudget(tmp_path, campaign_id="test", cap=Decimal("1000"))


def test_restart_never_resubmits_allocated_slot(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    slots = [s for s in assignments("test") if s["provider"] == "openai"][:2]
    rows = run_offline(root, campaign_id="test", slots=slots[:1])
    from operatebench.agents.pricing import LifecyclePricingPolicy
    from tools.aggregate_budget import canonical

    budget = CampaignBudget(root, campaign_id="test", cap=Decimal("1000"))
    ticket = budget.reserve(
        slots[1]["trial_id"],
        input_bound=1,
        output_bound=1,
        policy=LifecyclePricingPolicy("offline", Decimal("1"), Decimal("2")),
    )
    budget.close()
    (root / (slots[1]["trial_id"] + ".assigned.json")).write_bytes(canonical(slots[1]))

    def forbidden(*args, **kwargs):
        raise AssertionError("allocated slots cannot resubmit")

    monkeypatch.setattr(SDKMockTransport, "__init__", forbidden)
    resumed = run_offline(root, campaign_id="test", slots=slots, resume=True)
    assert resumed[0] == rows[0]
    assert resumed[1]["fault"] == "interrupted_no_resubmit"
    assert Decimal(resumed[1]["unknown_liability_usd"]) > 0
    budget = CampaignBudget(root, campaign_id="test", cap=Decimal("1000"))
    try:
        assert budget.records[ticket]["state"] == "unknown"
    finally:
        budget.close()
    assert len(list(root.glob("resume-*/scorecard.json"))) == 1


def test_linked_correction_shares_original_budget(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    original = next(s for s in assignments("test") if s["provider"] == "openai")
    handler = SDKMockTransport.handle

    def interrupted(self, request):
        raise TimeoutError("offline transport interruption")

    monkeypatch.setattr(SDKMockTransport, "handle", interrupted)
    first = run_offline(root, campaign_id="test", slots=[original])
    assert first[0]["classification"] == "excluded"
    monkeypatch.setattr(SDKMockTransport, "handle", handler)
    corrected = run_offline(
        root,
        campaign_id="test",
        resume=True,
        corrections={original["trial_id"]: "provider_fix"},
    )
    assert corrected[0]["rerun_of"] == original["trial_id"]
    assert corrected[0]["trial_id"] != original["trial_id"]
    assert len(list(root.glob("campaign-budget.jsonl"))) == 1
    assert (
        json.loads((root / (original["trial_id"] + ".closure.json")).read_text())
        == first[0]
    )


def _http_slot(flow="commerce", provider="openai"):
    return next(
        s for s in assignments("test") if s["provider"] == provider and s["flow"] == flow
    )


def _forbid_resume_sdk(monkeypatch):
    from tools.three_flow_http import HTTPCampaignTransport

    def forbidden(*args, **kwargs):
        pytest.fail("allocated resume must not construct SDK or resubmit")

    monkeypatch.setattr(SDKMockTransport, "__init__", forbidden)
    monkeypatch.setattr(HTTPCampaignTransport, "__init__", forbidden)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing_record", "retained_evidence_missing"),
        ("missing_binding", "retained_evidence_missing"),
        ("corrupt_record", "retained_evidence_invalid"),
        ("resealed_record", "retained_evidence_invalid"),
        ("binding_mismatch", "retained_evidence_invalid"),
        ("resealed_record_and_binding", "retained_evidence_invalid"),
        ("historical_source", "frozen_source_unavailable"),
        ("unchanged", None),
    ],
)
def test_resume_reconciles_retained_evidence(tmp_path, monkeypatch, damage, reason):
    from operatebench.agents.transport import content_digest
    from tools.three_flow_runtime import replay_trial

    slot = _http_slot()
    root = tmp_path / "campaign"
    original = run_offline(root, campaign_id="test", slots=[slot])[0]
    record_path = root / slot["trial_id"] / "record.json"
    binding_path = root / (slot["trial_id"] + ".binding.json")
    closure_path = root / (slot["trial_id"] + ".closure.json")
    closure_bytes = closure_path.read_bytes()
    record = json.loads(record_path.read_text())
    if damage == "missing_record":
        record_path.rename(record_path.with_suffix(".withheld"))
    elif damage == "missing_binding":
        binding_path.rename(binding_path.with_suffix(".withheld"))
    elif damage == "corrupt_record":
        record_path.write_text("{")
    elif damage in ("resealed_record", "resealed_record_and_binding"):
        record["trial"]["max_output_tokens"] = 1
        record["record_digest"] = content_digest(
            {k: v for k, v in record.items() if k != "record_digest"}
        )
        record_path.write_text(json.dumps(record))
        if damage == "resealed_record_and_binding":
            binding_path.write_text(json.dumps(record["trial"]))
    elif damage == "binding_mismatch":
        binding = json.loads(binding_path.read_text())
        binding["scenario"] = "US_NORMAL"
        binding_path.write_text(json.dumps(binding))
    elif damage == "historical_source":
        # A different executing source cannot replay the retained historical one.
        monkeypatch.setattr("tools.three_flow_campaign.source_binding", lambda: "new")
    retained = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    replays = []

    def replay(*args, **kwargs):
        result = replay_trial(*args, **kwargs)
        replays.append(result)
        return result

    monkeypatch.setattr("tools.three_flow_campaign.replay_trial", replay)
    _forbid_resume_sdk(monkeypatch)
    resumed = run_offline(root, campaign_id="test", slots=[slot], resume=True)[0]
    if reason is None:
        assert resumed == original
        assert resumed["evaluation"]["reliable"] is True
        assert len(replays) == 1 and replays[0]["provider_calls"] == 0
    else:
        assert resumed["classification"] == "aborted"
        assert resumed["fault"] == "evidence_reconciliation"
        assert resumed["evidence_reason"] == reason
        assert resumed["evaluation"] is None
        assert resumed["completion"] is None
    reconciliation = json.loads(
        next(root.glob("resume-*/*.reconciliation.json")).read_text()
    )
    assert reconciliation["evidence_status"] == (
        "verified"
        if reason is None
        else "unavailable"
        if "unavailable" in reason or "missing" in reason
        else "invalid"
    )
    assert closure_path.read_bytes() == closure_bytes
    assert all(p.read_bytes() == data for p, data in retained.items())


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_local_dispatch_fsync_failure_reconciliation(tmp_path, monkeypatch, provider):
    from unittest.mock import patch

    from tools.three_flow_http import HTTPCampaignTransport

    original_dispatch = HTTPCampaignTransport._dispatch
    calls = []

    def forbidden(self, request):
        calls.append(1)
        pytest.fail("local journal failure must precede wire")

    def dispatch(self, payload):
        with patch("tools.aggregate_budget.os.fsync", side_effect=OSError("synthetic")):
            return original_dispatch(self, payload)

    monkeypatch.setattr(SDKMockTransport, "handle", forbidden)
    monkeypatch.setattr(HTTPCampaignTransport, "_dispatch", dispatch)
    slot = _http_slot(provider=provider)
    root = tmp_path / "campaign"
    row = run_offline(root, campaign_id="test", slots=[slot])[0]
    closure = root / (slot["trial_id"] + ".closure.json")
    original_closure = closure.read_bytes()
    partial = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_text().splitlines()
    ]
    _forbid_resume_sdk(monkeypatch)
    resumed = run_offline(root, campaign_id="test", slots=[slot], resume=True)[0]
    journal = [
        json.loads(line)
        for line in (root / "campaign-budget.jsonl").read_text().splitlines()
    ]
    reserve = next(r for r in journal if r["kind"] == "reserve")
    dispatch_row = next(r for r in journal if r["kind"] == "dispatch")
    exact_liability = (
        Decimal(dispatch_row["ib"]) * Decimal(reserve["ir"])
        + Decimal(reserve["ob"]) * Decimal(reserve["or"])
    ) / 1_000_000
    assert Decimal(resumed["unknown_liability_usd"]) == exact_liability
    assert dispatch_row["id"] == reserve["id"]
    assert dispatch_row["ib"] >= reserve["ib"]
    assert 0 < reserve["ob"] <= (256000 if provider == "mistral" else 128000)
    assert (reserve["ir"], reserve["or"]) == ("1", "2")
    # Before recovery the failed append leaves the reservation pending, not
    # released. Recovery converts that retained exposure to unknown liability.
    totals = json.loads((root / "scorecard.json").read_text())["global_budget"]
    assert Decimal(totals["exposure"]) == exact_liability
    assert Decimal(totals["pending"]) + Decimal(totals["unknown"]) == exact_liability
    assert not any(r["kind"] == "raw_rpc" for r in journal)
    budget = CampaignBudget(root, campaign_id="test", cap=Decimal("1000"))
    try:
        accounting = budget.cell_accounting(slot["trial_id"])
        expected = accounting["full_unknown_usd"]
        assert Decimal(expected) > 0
        assert resumed["unknown_liability_usd"] == expected
        assert budget.totals()["pending"] == 0
        assert budget.totals()["unknown"] == budget.exposure
        assert all(r["state"] == "unknown" for r in accounting["records"].values())
    finally:
        budget.close()
    assert calls == []
    assert row["classification"] == resumed["classification"] == "aborted"
    assert row["fault"] == resumed["fault"] == "internal_error"
    assert row["local_failure"] == resumed["local_failure"] == "dispatch_journal_error"
    assert partial[-1]["classification"] == "aborted"
    assert partial[-2]["local_failure"] == "dispatch_journal_error"
    assert closure.read_bytes() == original_closure


@pytest.mark.parametrize(
    "flow,profile",
    [("commerce", "UK"), ("compliance", "EN_PRIVATE_ELECTRICAL"), ("maintenance", None)],
)
def test_literal_profile_report_projections(tmp_path, flow, profile):
    import csv

    slot = _http_slot(flow)
    root = tmp_path / "campaign"
    row = run_offline(root, campaign_id="test", slots=[slot])[0]
    assert row["classification"] == "scored"
    assert row["profile"] == profile
    if profile is not None:
        record = json.loads((root / slot["trial_id"] / "record.json").read_text())
        assert record["binding"]["profile"]["profile_id"] == profile
    assert (
        json.loads((root / "scorecard.json").read_text())["rows"][0]["profile"] == profile
    )
    with (root / "scorecard.csv").open() as stream:
        assert next(csv.DictReader(stream))["profile"] == (profile or "")
    assert '"profile": ' + json.dumps(profile) in (root / "scorecard.md").read_text()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral"])
def test_network_failure_remains_provider_exclusion_on_resume(
    tmp_path, monkeypatch, provider
):
    import httpx

    calls = []

    def network_failure(self, request):
        calls.append(1)
        # Untrusted text resembling a local failure must not change attribution.
        raise httpx.ReadError("dispatch_journal_error internal_error", request=request)

    monkeypatch.setattr(SDKMockTransport, "handle", network_failure)
    slot = _http_slot(provider=provider)
    root = tmp_path / "campaign"
    row = run_offline(root, campaign_id="test", slots=[slot])[0]
    _forbid_resume_sdk(monkeypatch)
    resumed = run_offline(root, campaign_id="test", slots=[slot], resume=True)[0]
    assert calls == [1]
    assert resumed == row
    assert row["classification"] == "excluded"
    assert row["fault"] == "provider_network_error"
    assert "local_failure" not in row
    assert Decimal(row["unknown_liability_usd"]) > 0
    partial = [
        json.loads(line)
        for line in (root / slot["trial_id"] / "partial.ndjson").read_text().splitlines()
    ]
    assert partial[-1]["classification"] == "excluded"
    assert partial[-1]["exclusion_code"] == "provider_network_error"
