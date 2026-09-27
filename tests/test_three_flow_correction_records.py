"""Synthetic-only independent probes; never constructs paid admission."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_three_flow_correction_binding import OLD, transition
from tests.test_three_flow_live import Network, cli, no_network, write
from tools import three_flow_live as live
from tools.three_flow_admission import digest, validate

__all__ = ["no_network"]


@pytest.mark.parametrize(
    "mutation",
    ["receipt_operation", "receipt_authorizing", "prior_schema", "prior_approval"],
)
def test_malformed_original_authority_rejected_before_recovery(
    tmp_path, monkeypatch, mutation
):
    path, a, _slot, before, _options = transition(tmp_path, monkeypatch, capped=True)
    receipt_path = path.parent / "SERIES-CONSUMED"
    receipt = json.loads(receipt_path.read_text())
    prior = a["correction_binding"]["original_admission"]
    if mutation == "receipt_operation":
        receipt["operation"] = "correction"
    elif mutation == "receipt_authorizing":
        receipt["authorizing"] = True
    else:
        if mutation == "prior_schema":
            prior["schema"] = "not-an-admission"
        else:
            del prior["approved"]
        # A rehashed object does not establish that this was an admitted shape.
        with pytest.raises(ValueError):
            validate(copy.deepcopy(prior), OLD, synthetic=True)
        receipt["admission_sha256"] = digest(prior)
    write(receipt_path, receipt)
    a["original_claim_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    write(path, a)
    pins = []
    original = live.private_fd

    def guard(value, **kw):
        if value == a["credential_file"]:
            pins.append(value)
            raise ValueError("independent sentinel: credential boundary reached")
        return original(value, **kw)

    monkeypatch.setattr(live, "private_fd", guard)
    network = Network()
    assert cli(path, a, network) == 2
    assert pins == [], "malformed original authority crossed credential boundary"
    assert not network.calls
    after = (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes()
    assert after == before


def test_unchanged_consumed_receipt_authenticates_original(tmp_path, monkeypatch):
    path, a, _slot, before, _options = transition(tmp_path, monkeypatch, capped=True)
    a["correction_binding"]["original_admission"]["schema"] = "not-an-admission"
    write(path, a)
    original = live.private_fd
    pins = []

    def guard(value, **kw):
        if value == a["credential_file"]:
            pins.append(value)
            raise AssertionError("credential boundary forbidden")
        return original(value, **kw)

    monkeypatch.setattr(live, "private_fd", guard)
    network = Network()
    assert cli(path, a, network) == 2
    assert pins == [] and network.calls == []
    assert (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes() == before


def reject_without_recovery(path, a, before, monkeypatch):
    write(path, a)
    original = live.private_fd
    pins = []

    def guard(value, **kwargs):
        if value == a["credential_file"]:
            pins.append(value)
            raise AssertionError("credential boundary forbidden")
        return original(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", guard)
    network = Network()
    assert cli(path, a, network) == 2
    assert not pins and not network.calls
    assert (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes() == before


def rebind_original(path, a):
    prior = a["correction_binding"]["original_admission"]
    receipt_path = path.parent / "SERIES-CONSUMED"
    receipt = json.loads(receipt_path.read_text())
    receipt["admission_sha256"] = digest(prior)
    write(receipt_path, receipt)
    write(path.parent / ("RECEIPT-" + prior["attempt_id"]), receipt)
    a["original_claim_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("target", ["SERIES-CONSUMED", "RECEIPT-first"])
@pytest.mark.parametrize(
    "field",
    [
        "campaign_id",
        "campaign_root",
        "custody_root",
        "attempt_id",
        "selected",
        "mode",
        "source_sha",
        "operation",
        "original_claim_sha256",
        "journal_sha256",
        "authorizing",
        "admission_sha256",
        "cap_usd",
        "unknown",
        "missing",
        "boolint",
    ],
)
def test_receipt_closed_contract(tmp_path, monkeypatch, target, field):
    path, a, _, before, _ = transition(tmp_path, monkeypatch, capped=True)
    record_path = path.parent / target
    record = json.loads(record_path.read_text())
    if field == "missing":
        del record["authorizing"]
    elif field == "boolint":
        record["authorizing"] = 0
    else:
        record[field] = True if field == "authorizing" else "changed"
    write(record_path, record)
    a["original_claim_sha256"] = hashlib.sha256(
        (path.parent / "SERIES-CONSUMED").read_bytes()
    ).hexdigest()
    reject_without_recovery(path, a, before, monkeypatch)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "missing_approved",
        "approved_int",
        "mode",
        "unknown",
        "operation",
        "recursive",
        "pending",
        "original_claim_sha256",
        "journal_sha256",
        "corrections",
        "parent_gates",
        "no_intervening_spend",
        "runtime",
        "registry",
        "registry_boolint",
    ],
)
def test_coherent_receipts_cannot_admit_malformed_original(
    tmp_path, monkeypatch, mutation
):
    path, a, _, before, _ = transition(tmp_path, monkeypatch, capped=True)
    prior = a["correction_binding"]["original_admission"]
    if mutation == "missing_approved":
        del prior["approved"]
    elif mutation == "approved_int":
        prior["approved"] = 0
    elif mutation == "recursive":
        prior["correction_binding"] = copy.deepcopy(a["correction_binding"])
    elif mutation == "pending":
        prior["pending"] = []
    elif mutation == "corrections":
        prior["corrections"] = {"unexpected": {}}
    elif mutation in ("parent_gates", "no_intervening_spend"):
        prior[mutation] = 1
    elif mutation == "registry_boolint":
        prior["registry"][0]["retries"] = False
    else:
        prior[mutation] = "changed"
    rebind_original(path, a)
    reject_without_recovery(path, a, before, monkeypatch)


@pytest.mark.parametrize("counterpart", ["missing", "renamed_attempt"])
def test_authentic_initial_counterpart_required(tmp_path, monkeypatch, counterpart):
    path, a, _, before, _ = transition(tmp_path, monkeypatch, capped=True)
    if counterpart == "missing":
        (path.parent / "RECEIPT-first").unlink()
    else:
        receipt_path = path.parent / "SERIES-CONSUMED"
        receipt = json.loads(receipt_path.read_text())
        receipt["attempt_id"] = "other"
        write(receipt_path, receipt)
        write(path.parent / "RECEIPT-other", receipt)
        a["original_claim_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    reject_without_recovery(path, a, before, monkeypatch)


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "legacy-v1", "temperature": 0.0, "thinking": True},
        {"mode": "off-minimum", "thinking": False, "effort": "minimum", "limit": None},
    ],
)
def test_historical_identity_is_not_revalidated_against_today(
    tmp_path, monkeypatch, settings
):
    path, a, _, before, _ = transition(tmp_path, monkeypatch, capped=True)
    prior = a["correction_binding"]["original_admission"]
    prior["runtime"].update(
        python="3.11.0",
        executable="/historical/python",
        dependencies=[["old-sdk", "1.0"]],
        executable_sha256="1" * 64,
        runtime_digest="2" * 64,
        lock_sha256="3" * 64,
    )
    for row in prior["registry"]:
        row["mapping"] = "legacy-v1"
        row["request_settings"] = settings
    rebind_original(path, a)
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 0
    assert network.calls
    assert (
        (Path(a["campaign_root"]) / "campaign-budget.jsonl")
        .read_bytes()
        .startswith(before)
    )
