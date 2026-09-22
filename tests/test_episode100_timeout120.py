# SPDX-License-Identifier: Apache-2.0
"""Closed diagnostic profile; real SDK, no network or wall-clock sleeps."""

import json
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.evidence import WireCaptureTransport
from tools import diagnose_aggregate_budget as runner
from tools import run_episode100 as launch
from tools.aggregate_budget import Budget


@pytest.mark.parametrize("timeout120", [False, True])
@pytest.mark.parametrize("case", ["full", "latency", "prewire", "loop", "503"])
def test_real_sdk_deadline_and_evidence(tmp_path, monkeypatch, timeout120, case):
    tmp_path.chmod(0o700)
    m = runner.module("grok46")
    clock = [0.0]
    cls = runner.XAIResponsesTransport
    transports = []

    def transport(**kwargs):
        obj = cls(**kwargs, clock=lambda: clock[0])
        transports.append(obj)
        return obj

    monkeypatch.setattr(runner, "XAIResponsesTransport", transport)
    script = m.ScriptedProvider(max_calls=100)
    requests = []
    first = []

    def handler(req):
        requests.append(req)
        timeout = req.extensions["timeout"]["read"]
        assert timeout == (120 if timeout120 else 30) - (40 if case == "prewire" else 0)
        payload = json.loads(req.content)
        assert payload["max_output_tokens"] == 8192
        assert payload["reasoning"]["effort"] == "high"
        if case == "503":
            return httpx.Response(503, json={"error": {"message": "offline"}})
        if case == "latency":
            clock[0] += 40
            if timeout < 40:
                raise httpx.ReadTimeout("simulated latency", request=req)
        if not first:
            first.append(script(req))
            return first[0]
        if case == "loop":
            return httpx.Response(200, content=first[0].content)
        return script(req)

    budget = Budget(
        tmp_path, cap=Decimal("18"), namespace="offline-test-120", create=True
    )
    if case == "prewire":
        from operatebench.agents import xai_responses

        original = xai_responses.build_model_payload

        def delayed(*args, **kwargs):
            value = original(*args, **kwargs)
            clock[0] += 40
            return value

        monkeypatch.setattr(xai_responses, "build_model_payload", delayed)

    wire = WireCaptureTransport()
    wire.attach(runner.DispatchTransport(httpx.MockTransport(handler), budget, "grok46"))
    client = runner.client_for("grok46", runner.PLACEHOLDER, wire)
    try:
        result = runner.execute_cell(
            "grok46",
            tmp_path,
            budget,
            client,
            wire,
            episode100=True,
            grok46_timeout120=timeout120,
        )
        ledger = m.read_execution_ledger(
            tmp_path / "grok46/execution_ledger.ndjson", require_complete=True
        )
        assert ledger.header.controls.turn_deadline_seconds == (120 if timeout120 else 30)
        assert transports[0].turn_deadline_seconds == (120 if timeout120 else 30)
        if case == "full" or (timeout120 and case in ("latency", "prewire")):
            assert result["bundle_ok"] and result["replay_ok"], result
            assert result["api_requests"] == result["decision_turns"] == 46
            assert ledger.header.agent_id == (
                "model-synthetic-demo-episode100-grok46-timeout120-v1"
                if timeout120
                else "synthetic-demo-model-episode100-v1"
            )
        else:
            assert ledger.status == "excluded", result
            assert len(requests) == (
                100 if case == "loop" else 0 if case == "prewire" else 1
            )
            assert result["termination_cause"] == (
                "episode_decision_limit" if case == "loop" else "provider_transport"
            )
            rows = [
                json.loads(line)
                for line in (tmp_path / "grok46/execution_ledger.partial.ndjson")
                .read_text()
                .splitlines()
            ]
            assert rows[-1]["ledger_terminal"] == ledger.terminal.as_dict()
        assert result["retries"] == 0
        assert budget.totals()["pending"] == 0
    finally:
        client.close()
        budget.close()


@pytest.mark.parametrize(
    "field,value",
    [("committed_exposure_usd", "4"), ("prior_evidence_sha256", ["a" * 64])],
)
def test_timeout_series_pins_prior(tmp_path, field, value):
    from tools.episode100_grok46_timeout120 import SERIES

    series = dict(SERIES)
    series[field] = value
    path = tmp_path / "series.json"
    path.write_text(json.dumps(series))
    path.chmod(0o600)
    with pytest.raises(ValueError):
        launch.load_series(path, Decimal("18"), ("grok46",))


def test_timeout_authority_and_single_consumption(tmp_path, monkeypatch):
    from tools.episode100_grok46_timeout120 import SERIES

    monkeypatch.setattr(launch, "source_sha", lambda: "a" * 40)
    tmp_path.chmod(0o700)
    credential = tmp_path / "credential.json"
    credential.write_text(json.dumps({"grok46": runner.PLACEHOLDER}))
    credential.chmod(0o600)
    approval = {
        "profile": launch.TIMEOUT_PROFILE,
        "source_sha": "a" * 40,
        "output": str(tmp_path / "out"),
        "cap_usd": "18",
        "cells": ["grok46"],
        "max_episode_decisions": 100,
        "retries": 0,
        "approved": True,
        "historical_authorities_retired": True,
        "mode": "dummy",
        "credential_file": str(credential),
        "series_manifest_sha256": launch.series_digest(SERIES),
        "turn_deadline_seconds": 30,
    }
    authority = tmp_path / "approval.json"
    authority.write_text(json.dumps(approval))
    authority.chmod(0o600)
    kwargs = {
        "output": tmp_path / "out",
        "cap": Decimal("18"),
        "mock": True,
        "cells": ("grok46",),
        "series": SERIES,
    }
    with pytest.raises(ValueError, match="exactly match"):
        launch.consume(authority, **kwargs)
    assert not (tmp_path / "EPISODE100-CONSUMED").exists()
    approval["turn_deadline_seconds"] = 120
    authority.write_text(json.dumps(approval))
    keys, sha = launch.consume(authority, **kwargs)
    assert sha == "a" * 40
    assert keys == {"grok46": runner.PLACEHOLDER}
    receipt = json.loads((tmp_path / "EPISODE100-CONSUMED").read_text())
    assert receipt["turn_deadline_seconds"] == 120
    with pytest.raises(FileExistsError):
        launch.consume(authority, **kwargs)


def test_closed_timeout_scope():
    assert launch.scope(("grok46",)) == "synthetic-demo-episode100-grok46-timeout120-v1"


def test_timeout_profile_controls():
    from operatebench.agents.pricing import LifecyclePricingPolicy
    from tools import episode100_grok46_timeout120 as profile
    from tools.diagnose_aggregate_budget import module

    m = module("grok46")
    policy = LifecyclePricingPolicy(
        policy_id=m.FIXED_CANARY_POLICY_ID,
        input_usd_per_mtok=m.FIXED_CANARY_INPUT_USD_PER_MTOK,
        output_usd_per_mtok=m.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
        rate_source=m.FIXED_CANARY_RATE_SOURCE,
    )
    controls = profile.controls_for(m, policy, Decimal("18"))
    assert controls.turn_deadline_seconds == 120
    assert controls.wall_clock_deadline_seconds == 12000
    assert controls.max_provider_calls == 100
    assert controls.max_attempts_per_call == 1
    assert controls.max_output_tokens == 8192
    spec = profile.load_profile_spec(m)
    assert spec.policy.horizon_minutes == 43200
    assert spec.policy.max_turns_per_invocation == 101
    assert m.TURN_DEADLINE_SECONDS == 30
