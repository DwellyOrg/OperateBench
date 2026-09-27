"""Private synthetic source transition; actual SDK mock transport, no credentials."""

import copy
import hashlib
import os
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.test_three_flow_budget import policy
from tests.test_three_flow_context import diagnostic
from tests.test_three_flow_live import cli, continuation, fixture, no_network, write
from tests.test_three_flow_optional_limits import UnlimitedNetwork, consent
from tests.test_three_flow_owner_pricing import assumed_fixture
from tests.test_three_flow_reported_usage import reported
from tools import three_flow_live as live
from tools.three_flow_admission import digest, validate
from tools.three_flow_budget import CampaignBudget

__all__ = ["no_network"]
OLD = "a" * 40
NEW = "b" * 40


def transition(tmp_path, monkeypatch, capped=False):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(live, "source_identity", lambda expected=None: OLD)
    path, a, slot = (fixture if capped else assumed_fixture)(tmp_path, "mistral")
    if not capped:
        p = diagnostic(a["profiles"][slot["model"]])
        p.update(network_timeout_seconds=None, execution_policy=consent()["mode"])
        reported(p)
        a.update(cap_usd=None, limit_policy={**consent(), "source_sha": OLD})
    original = copy.deepcopy(a)
    write(path, a)
    from tests.test_three_flow_live import Network

    class Failed(UnlimitedNetwork):
        def http(self, selected):
            inner = super().http(selected)

            def response(request):
                import json

                result = inner.handle_request(request)
                body = json.loads(result.read())
                body.pop("usage", None)
                return httpx.Response(200, json=body)

            return httpx.MockTransport(response)

    assert cli(path, a, Network("missing_usage") if capped else Failed()) == 2
    root = Path(a["campaign_root"])
    options = {
        "campaign_id": a["campaign_id"],
        "cap": Decimal("1000") if capped else None,
        "limit_policy": a.get("limit_policy"),
    }
    b = CampaignBudget(root, **options)
    try:
        known = b.reserve("settled", input_bound=0, output_bound=10, policy=policy())
        b.dispatch("settled", {"max_output_tokens": 10})
        b.settle(known, input_tokens=0, output_tokens=3)
        unknown = b.reserve("unknown", input_bound=0, output_bound=7, policy=policy())
        b.forfeit(unknown)
        b.reserve("pending", input_bound=0, output_bound=11, policy=policy())
    finally:
        b.close()
    continuation(path, a, operation="correction")
    a.update(
        source_sha=NEW,
        corrections={
            slot["trial_id"]: {"reason": "runtime_fix", "evidence_sha256": "c" * 64}
        },
    )
    journal = (root / "campaign-budget.jsonl").read_bytes()
    a["correction_binding"] = {
        "schema": "three-flow-correction-binding-v1",
        "owner_accepted": True,
        "owner_evidence_sha256": "d" * 64,
        "original_source_sha": OLD,
        "execution_source_sha": NEW,
        "original_admission": original,
        "financial_policy_sha256": digest(
            {"cap_usd": a["cap_usd"], "limit_policy": a.get("limit_policy")}
        ),
        "genesis_sha256": hashlib.sha256(
            journal.splitlines(keepends=True)[0]
        ).hexdigest(),
        "original_trial_artifacts_sha256": {
            slot["trial_id"]: {
                kind: hashlib.sha256(
                    (root / (slot["trial_id"] + "." + kind + ".json")).read_bytes()
                ).hexdigest()
                for kind in ("assigned", "binding", "closure")
            }
        },
        "manifest_sha256": hashlib.sha256(
            (root / "assignment.json").read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(live, "source_identity", lambda expected=None: NEW)
    write(path, a)
    return path, a, slot, journal, options


@pytest.mark.parametrize("capped", [False, True])
def test_new_source_uses_original_financial_genesis(tmp_path, monkeypatch, capped):
    from tests.test_three_flow_live import Network

    path, a, slot, before, options = transition(tmp_path, monkeypatch, capped)
    root = Path(a["campaign_root"])
    retained = {
        p: p.read_bytes()
        for p in [
            root / "assignment.json",
            path.parent / "SERIES-CONSUMED",
            root / (slot["trial_id"] + ".closure.json"),
            root / (slot["trial_id"] + ".assigned.json"),
            root / (slot["trial_id"] + ".binding.json"),
        ]
    }
    network = Network() if capped else UnlimitedNetwork()
    assert cli(path, a, network) == 0
    assert (root / "campaign-budget.jsonl").read_bytes().startswith(before)
    assert all(p.read_bytes() == raw for p, raw in retained.items())
    b = CampaignBudget(root, **options)
    try:
        assert b.totals()["known"] >= 3
        assert b.totals()["unknown"] > 18
        assert b.totals()["pending"] == 0
        assert b.cap == options["cap"]
        assert len([r for r in b.records.values() if r["cell"] == "pending"]) == 1
    finally:
        b.close()
    assert (root / (slot["trial_id"] + ".assigned.json")).exists()
    assert (root / (slot["trial_id"] + "-fix-second.assigned.json")).exists()
    # Even refreshing the journal witness must not reuse the consumed claim.
    a["journal_sha256"] = hashlib.sha256(
        (root / "campaign-budget.jsonl").read_bytes()
    ).hexdigest()
    write(path, a)
    original_fd = live.private_fd

    def no_credentials(value, **kwargs):
        assert value != a["credential_file"]
        return original_fd(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", no_credentials)
    assert cli(path, a, network) == 2
    for p in tmp_path.rglob("*"):
        assert p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600)


@pytest.mark.parametrize(
    "mutation",
    [
        "execution_source_sha",
        "original_source_sha",
        "financial_policy_sha256",
        "genesis_sha256",
        "manifest_sha256",
        "owner_accepted",
        "policy",
        "history",
        "namespace",
        "root",
        "journal",
        "claim",
        "price",
        "original_profile",
        "downgrade",
        "initial",
        "reused",
        "runtime",
        "registry",
        "assigned_artifact",
        "binding_artifact",
        "closure_artifact",
    ],
)
def test_transition_refused_before_credentials_and_recovery(
    tmp_path, monkeypatch, mutation
):
    path, a, slot, before, _ = transition(tmp_path, monkeypatch)
    b = a["correction_binding"]
    if mutation.endswith("_artifact"):
        b["original_trial_artifacts_sha256"][slot["trial_id"]][mutation.split("_")[0]] = (
            "0" * 64
        )
    elif mutation in ("execution_source_sha", "original_source_sha"):
        b[mutation] = "0" * 40
    elif mutation in ("financial_policy_sha256", "genesis_sha256", "manifest_sha256"):
        b[mutation] = "0" * 64
    elif mutation == "owner_accepted":
        b[mutation] = False
    elif mutation in ("policy", "history"):
        a["limit_policy"][
            "source_sha"
            if mutation == "policy"
            else "historical_liability_manifest_sha256"
        ] = "0" * (40 if mutation == "policy" else 64)
    elif mutation == "namespace":
        a["campaign_id"] = "foreign"
    elif mutation == "root":
        a["campaign_root"] = str(tmp_path)
    elif mutation == "journal":
        a["journal_sha256"] = "0" * 64
    elif mutation == "claim":
        a["original_claim_sha256"] = "0" * 64
    elif mutation in ("price", "original_profile"):
        a["profiles"][slot["model"]]["input_usd_per_million"] = "99"
        if mutation == "original_profile":
            b["original_admission"]["profiles"] = copy.deepcopy(a["profiles"])
    elif mutation == "downgrade":
        del a["correction_binding"]
    elif mutation == "initial":
        a["operation"] = "initial"
    elif mutation == "runtime":
        a["runtime"]["python"] = "changed"
    elif mutation == "registry":
        a["registry"][0]["retries"] = 99
    else:
        write(path.parent / "ATTEMPT-second-CONSUMED", {"synthetic": True})
    write(path, a)
    original_fd = live.private_fd
    pins = []

    def guarded(value, **kwargs):
        if value == a["credential_file"]:
            pins.append(value)
            raise AssertionError("credential pin forbidden")
        return original_fd(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", guarded)
    network = UnlimitedNetwork()
    assert cli(path, a, network) == 2
    assert not pins and not network.requests
    assert (tmp_path / "campaign" / "campaign-budget.jsonl").read_bytes() == before


def test_original_conflict_remains_closed_without_transition(tmp_path, monkeypatch):
    _path, a, _, before, options = transition(tmp_path, monkeypatch)
    del a["correction_binding"]
    with pytest.raises(ValueError, match="optional limits source binding"):
        validate(a, NEW, synthetic=True)
    a["limit_policy"]["source_sha"] = NEW
    p = next(iter(a["profiles"].values()))
    p["owner_pricing"]["source_sha"] = NEW
    p["rate_evidence_sha256"] = digest(p["owner_pricing"])
    validate(a, NEW, synthetic=True)
    options["limit_policy"] = a["limit_policy"]
    with pytest.raises(ValueError, match="foreign campaign or cap"):
        CampaignBudget(Path(a["campaign_root"]), **options)
    assert (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes() == before


def test_exhausted_original_cap_does_not_gain_fresh_wallet(tmp_path, monkeypatch):
    from tests.test_three_flow_live import Network

    path, a, _, _, options = transition(tmp_path, monkeypatch, capped=True)
    root = Path(a["campaign_root"])
    b = CampaignBudget(root, **options)
    try:
        from operatebench.agents.pricing import LifecyclePricingPolicy
        from tools.aggregate_budget import decimal

        remaining = b.totals()["remaining"]
        price = LifecyclePricingPolicy(
            policy_id="synthetic-exhaust",
            input_usd_per_mtok=Decimal("0"),
            output_usd_per_mtok=decimal(remaining) * 1000000,
            rate_source="operator_supplied_pinned_rates",
        )
        ident = b.reserve("exhaust", input_bound=0, output_bound=1, policy=price)
        b.forfeit(ident)
        assert b.totals()["remaining"] == 0
    finally:
        b.close()
    a["journal_sha256"] = hashlib.sha256(
        (root / "campaign-budget.jsonl").read_bytes()
    ).hexdigest()
    write(path, a)
    network = Network()
    assert cli(path, a, network) == 2
    b = CampaignBudget(root, **options)
    try:
        assert b.exposure == 1000
        assert not any("-fix-" in r["cell"] for r in b.records.values())
    finally:
        b.close()


@pytest.fixture(autouse=True)
def private_synthetic_umask():
    # Exercise explicit private custody under the ordinary base umask.
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)
