"""Synthetic network through the production factory; no admission authority."""

import copy
import json
import os
import socket
from decimal import Decimal

import httpx
import pytest

from tests.test_selected_six_http import SelectedMock
from tools.three_flow_budget import CampaignBudget
from tools.three_flow_campaign import assignments, run_campaign
from tools.three_flow_gemini import LATE_NOVEMBER_MODELS, GeminiMockTransport
from tools.three_flow_live_factory import live_factory
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import LATE_NOVEMBER_PROFILE
from tools.three_flow_river_mock import RiverMockChannel


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_five_routes_production_factory_synthetic(tmp_path):
    assets = os.environ.get("THREE_FLOW_RIPPLE_ASSETS")
    if not assets:
        pytest.skip("explicit Ripple assets required")
    root = tmp_path / "campaign"
    root.mkdir(mode=0o700)
    slots = assignments("offline", LATE_NOVEMBER_PROFILE)
    slots = [s for s in slots if s["flow"] == "commerce" and s["scenario"] == "UK_NORMAL"]
    assert len(slots) == 5
    bounds = {
        "mode": "published-context-diagnostic-v1",
        "source": "https://dev.meta.ai/docs",
        "scope": "endpoint",
        "interpretation": "published-integer",
    }

    from tools.three_flow_admission import (
        GUARANTEES,
        RESOURCE_POLICY,
        digest,
        runtime_identity,
        validate,
    )
    from tools.three_flow_profiles import registry
    from tools.three_flow_ripple import ASSETS, MODEL, PROVENANCE_SHA256

    def p(endpoint, context, output, ir, out):
        return {
            "endpoint": endpoint,
            "input_max": None,
            "output_max": output,
            "context_max": context,
            "input_usd_per_million": ir,
            "output_usd_per_million": out,
            "network_timeout_seconds": None,
            "bounds_policy": bounds,
            "rate_evidence_sha256": "a" * 64,
            "bounds_evidence_sha256": "b" * 64,
            "guarantees": {
                **dict.fromkeys(GUARANTEES, True),
                "provider_bounds_verified": False,
            },
            "pricing_policy": "conservative-all-tier-envelope-v1",
        }

    profiles = {
        name: profile.settings(65536) for name, profile in LATE_NOVEMBER_MODELS.items()
    }
    profiles["muse-ember-1.0"] = p(
        "https://api.meta.ai/v1/responses", 1048576, 524288, "0.30", "1.50"
    )
    profiles["ricursiveAI/Ripple-65B-Preview-NVFP4"] = {
        **p("api.river.ai:443", 262144, None, "1", "3"),
        "bounds_policy": {**bounds, "source": "https://docs.river.ai/guides/models.md"},
        "owner_output_cap_tokens": 30000,
    }
    profiles["moonshotai/Kimi-K3.1-Pro"] = {
        **p("https://api.river.ai/v1/chat/completions", 1048576, 1048575, "2", "10"),
        "bounds_policy": {
            **bounds,
            "source": "https://docs.river.ai/guides/inference-models/",
        },
        "owner_output_cap_tokens": 30000,
    }
    admission = {
        "roster_profile": LATE_NOVEMBER_PROFILE,
        "mode_profile": "late-november-v1",
        "campaign_id": "offline",
        "selected": [s["trial_id"] for s in slots],
        "profiles": profiles,
        "assets": {
            "root": assets,
            "catalog_sha256": digest(
                {"model": MODEL, "provenance_sha256": PROVENANCE_SHA256, "assets": ASSETS}
            ),
        },
        "cap_usd": "1000",
        "schema": "three-flow-admission-v1",
        "approved": False,
        "mode": "synthetic",
        "source_sha": "d" * 40,
        "runtime": runtime_identity(),
        "registry": registry(
            roster_profile=LATE_NOVEMBER_PROFILE, mode_profile="late-november-v1"
        ),
        "attempt_id": "first",
        "campaign_root": str(root),
        "custody_root": str(tmp_path / "custody"),
        "credential_file": str(tmp_path / "dummy.env"),
        "resource_policy": RESOURCE_POLICY,
        "operation": "initial",
        "original_claim_sha256": None,
        "journal_sha256": None,
        "pending": [s["trial_id"] for s in slots],
        "corrections": {},
        "parent_gates": True,
        "no_intervening_spend": True,
    }
    assert validate(admission, "d" * 40, synthetic=True) == admission
    for mutation in ("extra", "selection", "cap", "mode", "runtime"):
        bad = copy.deepcopy(admission)
        if mutation == "extra":
            bad["unexpected"] = True
        elif mutation == "selection":
            bad["selected"] = ["initial-999-99"]
        elif mutation == "cap":
            bad["cap_usd"] = "0"
        elif mutation == "mode":
            bad["mode_profile"] = "off-or-minimum-v1"
        else:
            bad["runtime"]["python"] = "unexpected"
        with pytest.raises(ValueError):
            validate(bad, "d" * 40, synthetic=True)
    uncapped = copy.deepcopy(admission)
    uncapped["cap_usd"] = None
    uncapped["limit_policy"] = {
        "mode": "three-flow-optional-limits-v1",
        "owner_accepted": True,
        "owner_evidence_sha256": "a" * 64,
        "source_sha": "d" * 40,
        "historical_liability_manifest_sha256": "b" * 64,
    }
    for model, profile in uncapped["profiles"].items():
        if model not in LATE_NOVEMBER_MODELS:
            profile["execution_policy"] = "three-flow-optional-limits-v1"
    assert validate(uncapped, "d" * 40, synthetic=True) == uncapped
    for original in (admission, uncapped):
        correction = copy.deepcopy(original)
        correction.update(
            source_sha="e" * 40,
            operation="correction",
            pending=[],
            original_claim_sha256="a" * 64,
            journal_sha256="b" * 64,
            corrections={
                ident: {"reason": "runtime_fix", "evidence_sha256": "c" * 64}
                for ident in original["selected"]
            },
        )
        correction["correction_binding"] = {
            "schema": "three-flow-correction-binding-v1",
            "owner_accepted": True,
            "owner_evidence_sha256": "d" * 64,
            "original_source_sha": "d" * 40,
            "execution_source_sha": "e" * 40,
            "original_admission": copy.deepcopy(original),
            "financial_policy_sha256": digest(
                {
                    "cap_usd": original["cap_usd"],
                    "limit_policy": original.get("limit_policy"),
                }
            ),
            "genesis_sha256": "e" * 64,
            "manifest_sha256": "f" * 64,
            "original_trial_artifacts_sha256": {
                ident: {"assigned": "a" * 64, "binding": "b" * 64, "closure": "c" * 64}
                for ident in original["selected"]
            },
        }
        assert validate(correction, "e" * 40, synthetic=True) == correction
        bad = copy.deepcopy(correction)
        bad["correction_binding"]["financial_policy_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            validate(bad, "e" * 40, synthetic=True)

    class Network:
        def http(self, slot):
            if slot["provider"] == "gemini":
                return GeminiMockTransport(slot["model"], slot["flow"])
            if slot["provider"] == "river_http":
                return SelectedMock("river_http", slot["model"], slot["flow"])
            mock = SDKMockTransport("openai", slot["flow"], model=slot["model"])

            def reply(request):
                body = json.loads(mock.handle(request).content)
                body.update(
                    store=False, parallel_tool_calls=False, tool_choice="auto", tools=[]
                )
                body["usage"].update(
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                )
                return httpx.Response(200, json=body)

            return httpx.MockTransport(reply)

        def river(self, slot):
            return RiverMockChannel(slot["model"], slot["flow"])

    budget = CampaignBudget(
        root,
        campaign_id="offline",
        cap=Decimal("1000"),
        roster_profile=LATE_NOVEMBER_PROFILE,
        create=True,
    )
    keys = {
        "GEMINI_API_KEY": "dummy-token-noncredential-g9z",
        "MODEL_API_KEY": "dummy-token-noncredential-g9z",
        "RIVER_API_KEY": "dummy-token-noncredential-g9z",
    }
    errors = []
    make = live_factory(admission, keys, network=Network())

    def traced(slot, shared):
        transport = make(slot, shared)
        send = transport.send

        def capture(request):
            try:
                return send(request)
            except Exception as exc:
                errors.append((slot["model"], type(exc).__name__, str(exc)))
                raise

        transport.send = capture
        return transport

    rows = run_campaign(
        root,
        campaign_id="offline",
        slots=slots,
        workers=4,
        campaign_budget=budget,
        transport_factory=traced,
        evidence_origin="SDK_MOCK_NOT_LLM",
        roster_profile=LATE_NOVEMBER_PROFILE,
    )
    assert len(rows) == 5
    assert all(row["classification"] == "scored" for row in rows), (rows, errors)
