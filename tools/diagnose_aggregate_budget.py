"""Offline settled-budget diagnostic on the bounded corrected engine.

This is not the historical fixed-canary operator. It retains its exact model
profiles and Maintenance V1 fixture, with a distinct budget-controller profile.
The engine and legacy ledger identities remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import anthropic
import httpx
import openai

from operatebench.agents.anthropic_messages import AnthropicMessagesTransport
from operatebench.agents.evidence import (
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    provider_identity_for,
)
from operatebench.agents.mistral_chat import MistralChatCompletionsTransport
from operatebench.agents.openai_responses import OpenAIResponsesTransport
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.xai_responses import XAIResponsesTransport
from operatebench.providers.mistral_chat import build_client as build_mistral_client
from tools.aggregate_budget import CELLS, Budget, FairTransport, SharedGuard

MODULES = (
    "anthropic_haiku",
    "anthropic_sonnet",
    "openai_luna",
    "xai_grok_4_5_responses",
    "xai_grok_4_6_responses",
    "mistral_small_2603",
)
PLACEHOLDER = "offline-placeholder-not-a-real-key"


def module(cell: str) -> ModuleType:
    return importlib.import_module(
        "tools.run_lifecycle_v1_" + MODULES[CELLS.index(cell)] + "_canary"
    )


class DispatchTransport(httpx.BaseTransport):
    """One durable reservation per HTTP dispatch; SDK retries cannot bypass it."""

    def __init__(self, inner: httpx.BaseTransport, budget: Budget, cell: str) -> None:
        self.inner, self.budget, self.cell = inner, budget, cell

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.budget.dispatch(self.cell, payload)
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


def client_for(cell: str, key: str, wire: httpx.BaseTransport) -> Any:
    http = httpx.Client(
        transport=wire, timeout=None, follow_redirects=False, trust_env=False
    )
    if cell.startswith("grok"):
        return openai.OpenAI(
            api_key=key,
            base_url="https://api.x.ai/v1",
            max_retries=0,
            timeout=None,
            http_client=http,
        )
    if cell == "luna56":
        return openai.OpenAI(
            api_key=key,
            base_url="https://api.openai.com/v1",
            max_retries=0,
            timeout=None,
            http_client=http,
        )
    if cell == "mistralsmall":
        return build_mistral_client(api_key=key, http_client=http)
    return anthropic.Anthropic(
        api_key=key,
        base_url="https://api.anthropic.com",
        max_retries=0,
        timeout=None,
        http_client=http,
    )


def execute_cell(
    cell: str,
    root: Path,
    budget: Budget,
    client: Any,
    wire: WireCaptureTransport,
    *,
    offline_debug: bool = False,
    episode100: bool = False,
    grok46_timeout120: bool = False,
) -> dict[str, Any]:
    from tools import episode100 as default_profile

    profile: ModuleType = default_profile
    if grok46_timeout120:
        if cell != "grok46" or not episode100:
            raise ValueError("120s profile requires episode100 Grok46 exclusively")
        from tools import episode100_grok46_timeout120

        profile = episode100_grok46_timeout120

    m = module(cell)
    agent_id = profile.AGENT_ID if episode100 else m.CANARY_AGENT_ID
    result: dict[str, Any]
    recorder = None
    ledger_path = root / cell / "execution_ledger.ndjson"
    try:
        directory = root / cell
        directory.mkdir(mode=0o700)
        spec = profile.load_profile_spec(m) if episode100 else m.load_spec(m.FIXTURE)
        if not episode100:
            m.check_spec_identity(spec)
        policy = LifecyclePricingPolicy(
            policy_id=m.FIXED_CANARY_POLICY_ID,
            input_usd_per_mtok=m.FIXED_CANARY_INPUT_USD_PER_MTOK,
            output_usd_per_mtok=m.FIXED_CANARY_OUTPUT_USD_PER_MTOK,
            rate_source=m.FIXED_CANARY_RATE_SOURCE,
        )
        from tools.aggregate_budget import decimal

        cap = decimal(budget.cap)
        controls = (
            profile.controls_for(m, policy, cap)
            if episode100
            else m.controls_for(policy, cap=cap)
        )
        guard = SharedGuard(
            budget=budget,
            cell=cell,
            policy=policy,
            max_output_tokens=m.MAX_OUTPUT_TOKENS,
            token_hard_cap=controls.token_hard_cap,
        )
        cls = (
            AnthropicMessagesTransport
            if cell in CELLS[:2]
            else OpenAIResponsesTransport
            if cell == "luna56"
            else MistralChatCompletionsTransport
            if cell == "mistralsmall"
            else XAIResponsesTransport
        )
        transport = cls(
            model=m.CANARY_MODEL,
            client=client,
            deadline_seconds=controls.turn_deadline_seconds,
            cost_guard=guard,
        )
        ledger_path = directory / "execution_ledger.ndjson"
        recorder = ProviderEvidenceRecorder(
            path=ledger_path,
            identity=EvidenceRunIdentity(
                operation_id=spec.operation_id,
                spec_digest_sha256=spec.spec_digest_sha256,
                scenario_id=m.CANARY_SCENARIO_ID,
                agent_id=agent_id,
            ),
            provider=provider_identity_for(transport),
            controls=controls,
            wire=wire,
            guard=guard,
        )
        transport.attach_recorder(recorder)
        agent = m.DeadlinedModelAgent(
            FairTransport(transport, budget, cell),
            model=m.CANARY_MODEL,
            agent_id=agent_id,
            recorder=recorder,
            max_transport_calls=controls.max_provider_calls,
            deadline_at=time.monotonic() + controls.wall_clock_deadline_seconds,
            max_output_tokens=m.MAX_OUTPUT_TOKENS,
        )
        run = m.run_episode(
            spec,
            "V1",
            agent_id,
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
        artifact_path = m.write_artifact(run, directory / "episode_artifact.json")
        artifact = m.read_artifact(artifact_path)
        ledger = m.read_execution_ledger(ledger_path, require_complete=True)
        bundle = m.audit_execution_bundle_files(artifact_path, ledger_path)
        replay = m.replay_artifact(spec, artifact)
        result = {
            "cell": cell,
            "model": m.CANARY_MODEL,
            "status": run.outcome.status,
            "reliable": run.reliable,
            "provider_calls": ledger.totals.provider_calls,
            "ledger_status": ledger.status,
            "bundle_ok": bundle.ok,
            "replay_ok": replay.ok,
            "summary": run.summary(),
            "cost": guard.as_dict(),
        }
        if episode100:
            result.update(
                controller_profile=profile.PROFILE,
                decision_turns=sum(c.decision is not None for c in ledger.calls),
                retrieve_decisions=sum(
                    c.decision is not None and c.decision.kind == "RETRIEVE"
                    for c in ledger.calls
                ),
                invocations=len({c.invocation_index for c in ledger.calls}),
                api_requests=ledger.totals.attempts,
                retries=0,
            )
        if not bundle.ok or not replay.ok:
            result["audit_error"] = {"bundle": str(bundle), "replay": str(replay)}
    except BaseException as exc:
        # Error text can contain provider/credential material: persist only class.
        result = {
            "cell": cell,
            "status": "failed-or-excluded-audit-required",
            "exception_type": type(exc).__name__,
        }
        if episode100 and recorder is not None:
            ledger = m.read_execution_ledger(ledger_path, require_complete=True)
            result.update(
                controller_profile=profile.PROFILE,
                ledger_status=ledger.status,
                exclusion_code=ledger.terminal.exclusion_code,
                decision_turns=sum(c.decision is not None for c in ledger.calls),
                retrieve_decisions=sum(
                    c.decision is not None and c.decision.kind == "RETRIEVE"
                    for c in ledger.calls
                ),
                invocations=len({c.invocation_index for c in ledger.calls}),
                api_requests=ledger.totals.attempts,
                retries=0,
                termination_cause="episode_decision_limit"
                if sum(c.decision is not None for c in ledger.calls) == 100
                and getattr(exc, "fault", None) == "provider_budget"
                else getattr(exc, "fault", type(exc).__name__),
            )
        if offline_debug:
            import traceback

            traceback.print_exc()
    finally:
        if recorder is not None:
            recorder.close()
        budget.finish(cell)
    result["budget_accounting"] = budget.cell_accounting(cell)
    (root / (cell + "-outcome.json")).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """New offline identity only; never opens a live provider transport."""
    import socket
    import sys

    from tools.aggregate_budget import CONTROLLER_PROFILE, consume_offline_once, decimal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def deny_network(event: str, arguments: tuple[Any, ...]) -> None:
        if event == "socket.connect" and arguments[0].family in (
            socket.AF_INET,
            socket.AF_INET6,
        ):
            raise RuntimeError("offline diagnostic forbids internet sockets")

    sys.addaudithook(deny_network)
    root = args.output.absolute()
    root.mkdir(mode=0o700)  # Never adopt/reuse an existing run or its journal.
    consume_offline_once(root)
    budget = Budget(
        root, cap=Decimal("10"), namespace="offline-test-six-sdk", create=True
    )
    prepared = []
    try:
        for cell in CELLS:
            m = module(cell)
            handler = m.ScriptedProvider(max_calls=m.EXPECTED_PROVIDER_CALLS)
            wire = WireCaptureTransport()
            wire.attach(DispatchTransport(httpx.MockTransport(handler), budget, cell))
            prepared.append((cell, client_for(cell, PLACEHOLDER, wire), wire))
        with ThreadPoolExecutor(max_workers=len(CELLS)) as pool:
            futures = [
                pool.submit(execute_cell, cell, root, budget, client, wire)
                for cell, client, wire in prepared
            ]
            results = [future.result() for future in futures]
        summary = {
            "controller_profile": CONTROLLER_PROFILE,
            "authority": "offline-dummy-only-not-live-authorization",
            "cells": results,
            "aggregate": {
                key: str(decimal(value)) for key, value in budget.totals().items()
            },
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return (
            0
            if all(r.get("bundle_ok") and r.get("replay_ok") for r in results)
            and budget.totals()["pending"] == 0
            and budget.totals()["unknown"] == 0
            else 1
        )
    finally:
        for _, client, wire in prepared:
            close = getattr(client, "close", None)
            if close is not None:
                close()
            else:
                wire.close()
        budget.close()


if __name__ == "__main__":
    raise SystemExit(main())
