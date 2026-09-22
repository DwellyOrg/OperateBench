import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import httpx

from operatebench.agents.evidence import WireCaptureTransport
from operatebench.agents.transport import ProviderFailure
from operatebench.providers.cost import CostReservationBreachedError
from tools import diagnose_aggregate_budget as runner
from tools.aggregate_budget import Budget


class SDKFault(unittest.TestCase):
    def test_real_sdk_blocked_first_other_500_retains_full_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            b = Budget(
                root, cap=Decimal("10"), namespace="offline-test-sdk-fault", create=True
            )
            entered = threading.Event()
            release = threading.Event()
            seen = []
            clients = []
            m = runner.module("haiku45")
            script = m.ScriptedProvider(max_calls=m.EXPECTED_PROVIDER_CALLS)

            def blocked(req):
                if not entered.is_set():
                    entered.set()
                    release.wait()
                return script(req)

            def unknown(req):
                seen.append(req)
                return httpx.Response(
                    500,
                    json={
                        "error": {
                            "type": "server_error",
                            "message": "synthetic offline fault",
                        }
                    },
                )

            for cell, handler in [("haiku45", blocked), ("sonnet5", unknown)]:
                wire = WireCaptureTransport()
                wire.attach(
                    runner.DispatchTransport(httpx.MockTransport(handler), b, cell)
                )
                clients.append(
                    (cell, runner.client_for(cell, runner.PLACEHOLDER, wire), wire)
                )
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    cell, client, wire = clients[0]
                    first = pool.submit(runner.execute_cell, cell, root, b, client, wire)
                    try:
                        self.assertTrue(entered.wait(5))
                        cell, client, wire = clients[1]
                        other = pool.submit(
                            runner.execute_cell, cell, root, b, client, wire
                        )
                        result = other.result(
                            timeout=10
                        )  # test bound, not runtime policy
                        self.assertEqual(
                            result["status"], "failed-or-excluded-audit-required"
                        )
                        self.assertEqual(len(seen), 1)
                        self.assertGreater(b.totals()["unknown"], 0)
                        self.assertGreater(b.totals()["pending"], 0)
                        self.assertTrue((root / "sonnet5-outcome.json").exists())
                        self.assertFalse(first.done())
                    finally:
                        release.set()
                    self.assertTrue(first.result(timeout=30)["bundle_ok"])
                self.assertEqual(b.totals()["pending"], 0)
                self.assertGreater(b.totals()["unknown"], 0)
            finally:
                release.set()
                for _, client, _wire in clients:
                    client.close()
                b.close()

    def test_overbound_second_response_finalizes_budget_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            b = Budget(
                root, cap=Decimal("10"), namespace="offline-test-prefix", create=True
            )
            m = runner.module("haiku45")
            script = m.ScriptedProvider(max_calls=m.EXPECTED_PROVIDER_CALLS)
            calls = []

            def handler(req):
                calls.append(req)
                response = script(req)
                if len(calls) == 2:
                    body = json.loads(response.content)
                    body["usage"]["output_tokens"] = m.MAX_OUTPUT_TOKENS + 1
                    return httpx.Response(200, json=body, headers=dict(response.headers))
                return response

            wire = WireCaptureTransport()
            wire.attach(
                runner.DispatchTransport(httpx.MockTransport(handler), b, "haiku45")
            )
            client = runner.client_for("haiku45", runner.PLACEHOLDER, wire)
            try:
                failures = []
                original = m.run_episode

                def observe(*args, **kwargs):
                    try:
                        return original(*args, **kwargs)
                    except ProviderFailure as exc:
                        failures.append(exc)
                        raise

                with patch.object(m, "run_episode", observe):
                    result = runner.execute_cell("haiku45", root, b, client, wire)
                self.assertEqual(result["exception_type"], "ProviderFailure")
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0].fault, "provider_budget")
                breach = failures[0].__cause__
                self.assertIsInstance(breach, CostReservationBreachedError)
                self.assertIsNone(breach.measured_usd)
                self.assertIsNotNone(breach.__cause__)
                self.assertEqual(result["status"], "failed-or-excluded-audit-required")
                self.assertEqual(len(calls), 2)
                self.assertEqual(b.totals()["pending"], 0)
                self.assertGreater(b.totals()["known"], 0)
                self.assertGreater(b.totals()["unknown"], 0)
                ledger = m.read_execution_ledger(
                    root / "haiku45" / "execution_ledger.ndjson", require_complete=True
                )
                self.assertEqual(ledger.status, "excluded")
                self.assertEqual(ledger.terminal.exclusion_code, "provider_budget")
                self.assertEqual(ledger.rows_verified, 4)
                self.assertEqual(ledger.totals.attempts, 2)
                self.assertEqual(len(ledger.calls), 2)
                self.assertIsNone(ledger.calls[1].decision)
                self.assertIsNone(ledger.calls[1].attempts[0].fault)
                account = result["budget_accounting"]
                self.assertEqual(
                    Decimal(ledger.totals.forfeited_reservation_usd),
                    Decimal(account["legacy_forfeited_usd"]),
                )
                self.assertEqual(
                    Fraction(Decimal(account["legacy_forfeited_usd"]))
                    + Fraction(Decimal(account["wire_supplement_forfeited_usd"])),
                    b.totals()["unknown"],
                )
                journal = [
                    json.loads(line)
                    for line in (root / "successor-events.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [row["kind"] for row in journal],
                    [
                        "genesis",
                        "reserve",
                        "dispatch",
                        "settle",
                        "reserve",
                        "dispatch",
                        "unknown",
                    ],
                )
                self.assertEqual(ledger.calls[0].attempts[0].cost_settlement, "measured")
                self.assertEqual(ledger.calls[1].attempts[0].cost_settlement, "forfeited")
                self.assertFalse((root / "haiku45" / "episode_artifact.json").exists())
                rows = [
                    json.loads(line)
                    for line in (root / "haiku45" / "execution_ledger.partial.ndjson")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(rows[-1]["classification"], "excluded")
                self.assertEqual(rows[-1]["exclusion_code"], "provider_budget")
                self.assertEqual(rows[-1]["ledger_terminal"], ledger.terminal.as_dict())
                self.assertEqual(
                    rows[-1]["ledger_digest_sha256"], ledger.ledger_digest_sha256
                )
                self.assertEqual(
                    rows[0]["identity"]["execution_run_id"],
                    ledger.header.execution_run_id,
                )
                self.assertEqual(rows[0]["engine_version"], "0.12.0")
                self.assertTrue(any(row["kind"] == "decision" for row in rows))
                self.assertTrue(any(row["kind"] == "state" for row in rows))
            finally:
                client.close()
                b.close()
