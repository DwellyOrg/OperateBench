"""Synthetic custody only: integrated mode, pacing and financial source transition."""

import copy
import json
from pathlib import Path

import pytest

from tests import test_three_flow_correction_binding as binding
from tests.test_three_flow_correction_records import reject_without_recovery
from tests.test_three_flow_live import cli, no_network, write
from tests.test_three_flow_optional_limits import UnlimitedNetwork
from tools import three_flow_live as live
from tools.three_flow_admission import digest, registry
from tools.three_flow_runtime import Trial, replay_trial

__all__ = ["no_network"]
MODE = "off-or-minimum-v1"
PACE = "paced-safe-errors-v1"


def integrated_transition(tmp_path, monkeypatch, *, legacy=False):
    original_fixture = binding.assumed_fixture
    source_identity = live.source_identity
    current_sha = source_identity()
    read = live.read_pinned
    reads = []

    def guarded_read(fd, cap):
        if cap == 65536:
            custody = tmp_path / "custody"
            claim = "SERIES-CONSUMED" if not reads else "ATTEMPT-second-CONSUMED"
            assert (custody / claim).is_file()
            reads.append(claim)
        return read(fd, cap)

    monkeypatch.setattr(live, "read_pinned", guarded_read)

    def configured_fixture(root, provider):
        path, a, slot = original_fixture(root, provider)
        if not legacy:
            a.update(mode_profile=MODE, transport_policy=PACE)
            a["registry"] = registry(mode_profile=MODE, transport_policy=PACE)
        p = a["profiles"][slot["model"]]
        entry = next(r for r in a["registry"] if r["model"] == slot["model"])
        p["owner_pricing"]["registry_entry_sha256"] = digest(entry)
        # The transition helper subsequently binds the complete diagnostic profile.
        return path, a, slot

    monkeypatch.setattr(binding, "assumed_fixture", configured_fixture)
    path, a, slot, before, options = binding.transition(tmp_path, monkeypatch)
    a.update(mode_profile=MODE, transport_policy=PACE)
    a["registry"] = registry(mode_profile=MODE, transport_policy=PACE)
    a["source_sha"] = current_sha
    a["correction_binding"]["execution_source_sha"] = current_sha
    monkeypatch.setattr(live, "source_identity", source_identity)
    write(path, a)
    return path, a, slot, before, options, reads


@pytest.mark.parametrize("legacy", [False, True])
def test_consumed_mode_pacing_correction_actual_source_and_replay(
    tmp_path, monkeypatch, legacy
):
    from tools import three_flow_http as http
    from tools.three_flow_live_factory import HTTPCampaignTransport

    # Retain the real process-wide pacing object, but avoid wall-clock waiting.
    # Its dedicated tests prove spacing; here we prove the factory selects it.
    paced = []
    monkeypatch.setattr(http.MISTRAL_PACER, "call", lambda fn: (paced.append(1), fn())[1])
    constructed = []
    original_init = HTTPCampaignTransport.__init__

    def observed_init(self, **kwargs):
        original_init(self, **kwargs)
        if constructed or not legacy:
            assert self.wire.pacer is http.MISTRAL_PACER
            assert self.settings["mode_profile"] == MODE
            assert self.settings["transport_policy"] == PACE
        else:
            assert self.wire.pacer is None
            assert "mode_profile" not in self.settings
            assert "transport_policy" not in self.settings
        constructed.append(self)

    monkeypatch.setattr(HTTPCampaignTransport, "__init__", observed_init)
    path, a, slot, before, _, reads = integrated_transition(
        tmp_path, monkeypatch, legacy=legacy
    )
    root = Path(a["campaign_root"])
    original = copy.deepcopy(a["correction_binding"]["original_admission"])
    assert original.get("mode_profile", "legacy-v1") == ("legacy-v1" if legacy else MODE)
    assert original.get("transport_policy", "legacy-v1") == (
        "legacy-v1" if legacy else PACE
    )
    retained = {p: p.read_bytes() for p in root.glob(slot["trial_id"] + ".*.json")}
    retained[path.parent / "SERIES-CONSUMED"] = (
        path.parent / "SERIES-CONSUMED"
    ).read_bytes()
    assert len(constructed) == 1
    network = UnlimitedNetwork()
    assert cli(path, a, network) == 0
    assert len(constructed) == 2  # One real SDK transport node per consumed attempt.
    assert reads == ["SERIES-CONSUMED", "ATTEMPT-second-CONSUMED"]
    assert paced and network.requests
    assert all(p.read_bytes() == raw for p, raw in retained.items())
    assert (root / "campaign-budget.jsonl").read_bytes().startswith(before)
    assert a["profiles"] == original["profiles"]
    ident = slot["trial_id"] + "-fix-second"
    record = json.loads((root / ident / "record.json").read_text())
    trial = Trial(**json.loads((root / (ident + ".binding.json")).read_text()))
    entry = next(r for r in a["registry"] if r["model"] == slot["model"])
    assert record["provider"]["request_mapping"] == entry["mapping"]
    assert all(
        record["provider"]["settings"][k] == v
        for k, v in entry["request_settings"].items()
    )
    assert (
        record["identity"]["operation_id"] == "lettings_maintenance_delivery_recovery_v2"
    )
    assert record["evaluation"]["reliable"]
    for turn in record["provider_turns"]:
        payload = json.loads(turn["wire"]["request_utf8"])
        assert payload["reasoning_effort"] == "none"
    calls = len(network.requests)
    assert (
        replay_trial(trial, record, expected_record_digest=record["record_digest"])[
            "provider_calls"
        ]
        == 0
    )
    assert len(network.requests) == calls and len(constructed) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "mode",
        "transport",
        "registry",
        "history_mode",
        "history_transport",
        "price",
        "new_series_price",
    ],
)
def test_integrated_correction_refuses_tampering(tmp_path, monkeypatch, mutation):
    from tools import three_flow_http as http

    monkeypatch.setattr(http.MISTRAL_PACER, "call", lambda fn: fn())
    path, a, slot, before, _, _ = integrated_transition(tmp_path, monkeypatch)
    prior = a["correction_binding"]["original_admission"]
    if mutation == "mode":
        del a["mode_profile"]
    elif mutation == "transport":
        del a["transport_policy"]
    elif mutation == "registry":
        next(r for r in a["registry"] if r["model"] == slot["model"])["request_settings"][
            "minimum_spacing_seconds"
        ] = 0
    elif mutation == "history_mode":
        del prior["mode_profile"]
    elif mutation == "history_transport":
        del prior["transport_policy"]
    elif mutation == "price":
        a["profiles"][slot["model"]]["input_usd_per_million"] = "99"
    else:
        # A new-series source price binding cannot replace immutable correction prices.
        p = a["profiles"][slot["model"]]
        p["owner_pricing"]["source_sha"] = a["source_sha"]
        p["rate_evidence_sha256"] = digest(p["owner_pricing"])
    reject_without_recovery(path, a, before, monkeypatch)
