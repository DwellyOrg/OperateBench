"""Selected-six historical admission regression; synthetic SDK custody only."""

import copy
import json
from pathlib import Path

import pytest

from tests import test_three_flow_correction_binding as binding
from tests.test_selected_six_bounds import mistral_profile
from tests.test_selected_six_http import Network
from tests.test_three_flow_live import cli, fixture, no_network, write
from tools import three_flow_live as live
from tools.three_flow_admission import (
    digest,
    registry,
    validate,
    validate_historical_identity,
)
from tools.three_flow_campaign import assignments
from tools.three_flow_profiles import ENDPOINTS, SELECTED_SIX_PROFILE
from tools.three_flow_selected import TARIFFS

__all__ = ["no_network"]


def selected_transition(tmp_path, monkeypatch):
    def configured(root, _provider):
        path, a, _ = fixture(root, "anthropic")
        slot = next(
            s
            for s in assignments(a["campaign_id"], SELECTED_SIX_PROFILE)
            if s["provider"] == "anthropic" and s["flow"] == "commerce"
        )
        model = slot["model"]
        p = mistral_profile()
        rates = TARIFFS[model]
        p.update(
            endpoint=ENDPOINTS["anthropic"],
            context_max=1000000,
            output_max=128000,
            input_usd_per_million=max(rates[:3]),
            output_usd_per_million=rates[3],
        )
        p["tariff_evidence"]["category_rates"] = dict(
            zip(("input", "cache_read", "cache_write", "output"), rates, strict=True)
        )
        p["rate_evidence_sha256"] = digest(p["tariff_evidence"])
        a.update(
            roster_profile=SELECTED_SIX_PROFILE,
            mode_profile="off-or-minimum-v1",
            transport_policy="paced-safe-errors-v1",
            registry=registry(
                roster_profile=SELECTED_SIX_PROFILE,
                mode_profile="off-or-minimum-v1",
                transport_policy="paced-safe-errors-v1",
            ),
            selected=[slot["trial_id"]],
            pending=[slot["trial_id"]],
            profiles={model: p},
        )
        return path, a, slot

    monkeypatch.setattr(binding, "assumed_fixture", configured)
    monkeypatch.setattr(binding, "diagnostic", lambda p: p)
    monkeypatch.setattr(binding, "reported", lambda p: None)
    monkeypatch.setattr(binding, "UnlimitedNetwork", Network)
    return binding.transition(tmp_path, monkeypatch)


def test_selected_source_transition_consumes_sdk_correction(tmp_path, monkeypatch):
    path, a, slot, before, _ = selected_transition(tmp_path, monkeypatch)
    prior = copy.deepcopy(a["correction_binding"]["original_admission"])
    assert all("output_limit" not in row for row in prior["registry"])
    assert validate(a, binding.NEW, synthetic=True) == a
    root = Path(a["campaign_root"])
    retained = {p: p.read_bytes() for p in root.glob(slot["trial_id"] + ".*.json")}
    retained[path.parent / "SERIES-CONSUMED"] = (
        path.parent / "SERIES-CONSUMED"
    ).read_bytes()
    network = Network()
    assert cli(path, a, network) == 0
    assert all(p.read_bytes() == raw for p, raw in retained.items())
    assert a["profiles"] == prior["profiles"]
    assert a["correction_binding"]["original_admission"] == prior
    root = Path(a["campaign_root"])
    assert (root / "campaign-budget.jsonl").read_bytes().startswith(before)
    assert network.transports and all(
        call["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
        for t in network.transports
        for call in t.calls
    )
    record = json.loads(
        (root / (slot["trial_id"] + "-fix-second") / "record.json").read_text()
    )
    assert record["evaluation"]["reliable"]


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "profile",
        "model",
        "provider",
        "retries",
        "output",
        "admitted",
        "receipt",
    ],
)
def test_selected_history_refused_before_secrets(tmp_path, monkeypatch, mutation):
    path, a, _, before, _ = selected_transition(tmp_path, monkeypatch)
    prior = a["correction_binding"]["original_admission"]
    row = prior["registry"][1]
    if mutation == "extra":
        row["extra"] = False
    elif mutation == "missing":
        del row["mapping"]
    elif mutation == "profile":
        prior["roster_profile"] = "legacy-v1"
    elif mutation in ("model", "provider"):
        row[mutation] = "wrong"
    elif mutation == "retries":
        row["retries"] = False
    elif mutation == "output":
        row["output_limit"] = False
    elif mutation == "admitted":
        row["admitted"] = 0
    else:
        # Valid opaque historical values still cannot disagree with consumed receipt.
        row["mapping"] = "altered-prior-mapping"
    write(path, a)
    original_fd = live.private_fd

    def guarded(value, **kwargs):
        assert value != a["credential_file"], "credential access forbidden"
        return original_fd(value, **kwargs)

    monkeypatch.setattr(live, "private_fd", guarded)
    network = Network()
    assert cli(path, a, network) == 2
    assert not network.transports
    assert (Path(a["campaign_root"]) / "campaign-budget.jsonl").read_bytes() == before


def test_original_selected_emitted_shape_and_legacy_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "source_identity", lambda expected=None: binding.OLD)
    _, a, _ = fixture(tmp_path, "anthropic")
    a.update(
        roster_profile=SELECTED_SIX_PROFILE,
        registry=registry(
            roster_profile=SELECTED_SIX_PROFILE,
            mode_profile="off-or-minimum-v1",
            transport_policy="paced-safe-errors-v1",
        ),
    )
    row = a["registry"][1]
    # Actual f3bd Haiku emitted metadata; historical settings stay opaque.
    row["mapping"] = "three-flow-anthropic-selected-six-snapshot-v1"
    row["request_settings"]["tool_choice"] = {"type": "any"}
    assert all(len(r) == 8 for r in a["registry"])
    validate_historical_identity(a)
    a.pop("roster_profile")
    a["registry"] = registry()
    validate_historical_identity(a)
    del a["registry"][0]["output_limit"]
    with pytest.raises(ValueError, match="closed original registry"):
        validate_historical_identity(a)


@pytest.mark.parametrize("field", ["retries", "output_limit"])
def test_legacy_history_rejects_boolean_fields(tmp_path, monkeypatch, field):
    monkeypatch.setattr(live, "source_identity", lambda expected=None: binding.OLD)
    _, a, _ = fixture(tmp_path, "anthropic")
    a["registry"][0][field] = False
    with pytest.raises(ValueError, match="closed original registry"):
        validate_historical_identity(a)
