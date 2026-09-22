import json
from decimal import Decimal

import httpx
import pytest

from operatebench.agents.evidence import WireCaptureTransport
from tools import diagnose_aggregate_budget as runner
from tools.aggregate_budget import CELLS, Budget


@pytest.mark.parametrize("cell", ["haiku45", "grok45", "grok46"])
@pytest.mark.parametrize("prefix,decisions", [(0, 99), (0, 100), (0, 101), (20, 101)])
def test_retrieve_loops_episode_boundary(cell, prefix, decisions, tmp_path):
    tmp_path.chmod(0o700)
    m = runner.module(cell)
    budget = Budget(
        tmp_path, cap=Decimal("10"), namespace="offline-test-loop", create=True
    )
    script = m.ScriptedProvider(max_calls=101)
    seen = []
    first = []

    def handler(req):
        seen.append(req)
        if len(seen) > decisions:
            return httpx.Response(
                503, json={"error": {"type": "server_error", "message": "dummy"}}
            )
        if not first:
            first.append(script(req))
        if len(seen) <= prefix:
            return first[0] if len(seen) == 1 else script(req)
        return httpx.Response(200, content=first[0].content, headers=first[0].headers)

    wire = WireCaptureTransport()
    wire.attach(runner.DispatchTransport(httpx.MockTransport(handler), budget, cell))
    client = runner.client_for(cell, runner.PLACEHOLDER, wire)
    try:
        result = runner.execute_cell(
            cell, tmp_path, budget, client, wire, episode100=True
        )
        ledger = m.read_execution_ledger(
            tmp_path / cell / "execution_ledger.ndjson", require_complete=True
        )
        assert ledger.status == "excluded", result
        assert result["decision_turns"] == min(decisions, 100)
        if not prefix:
            assert result["retrieve_decisions"] == min(decisions, 100)
            assert result["invocations"] == 1
        assert len(seen) == 100
        assert result["termination_cause"] == (
            "episode_decision_limit" if decisions >= 100 else "provider_transport"
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / cell / "execution_ledger.partial.ndjson")
            .read_text()
            .splitlines()
        ]
        assert rows[-1]["classification"] == "excluded"
        assert rows[-1]["ledger_terminal"] == ledger.terminal.as_dict()
        assert not (tmp_path / cell / "episode_artifact.json").exists()
        if prefix:
            assert len({c.invocation_index for c in ledger.calls}) > 1
    finally:
        client.close()
        budget.close()


@pytest.mark.parametrize("cell", CELLS)
def test_new_profile_real_sdk(cell, tmp_path):
    tmp_path.chmod(0o700)
    m = runner.module(cell)
    budget = Budget(
        tmp_path, cap=Decimal("10"), namespace="offline-test-100", create=True
    )
    wire = WireCaptureTransport()
    wire.attach(
        runner.DispatchTransport(
            httpx.MockTransport(m.ScriptedProvider(max_calls=100)), budget, cell
        )
    )
    client = runner.client_for(cell, runner.PLACEHOLDER, wire)
    try:
        result = runner.execute_cell(
            cell, tmp_path, budget, client, wire, episode100=True
        )
        assert result["bundle_ok"] and result["replay_ok"], result
        assert result["controller_profile"] == "synthetic-demo-episode100-v1"
        assert result["decision_turns"] == 46
        ledger = m.read_execution_ledger(
            tmp_path / cell / "execution_ledger.ndjson", require_complete=True
        )
        assert ledger.header.controls.max_provider_calls == 100
        assert ledger.header.agent_id == "synthetic-demo-model-episode100-v1"
    finally:
        getattr(client, "close", wire.close)()
        budget.close()
