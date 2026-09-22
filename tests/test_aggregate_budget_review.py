import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path

import httpx

from operatebench.agents.evidence import WireCaptureTransport
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.providers.cost import CostCapExceededError
from tools import aggregate_budget as s
from tools import diagnose_aggregate_budget as runner

ROOT = Path(__file__).parent
P = LifecyclePricingPolicy("dummy-review", D("2"), D("5"))


class Review(unittest.TestCase):
    def test_racing_capacity_and_same_cell(self):
        for same in (False, True):
            with tempfile.TemporaryDirectory() as t:
                b = s.Budget(
                    Path(t), cap=D(".009"), namespace="offline-test-race", create=True
                )
                barrier = threading.Barrier(8)

                def admit(n, barrier=barrier, b=b, same=same):
                    barrier.wait(3)
                    try:
                        return b.reserve(
                            "same" if same else str(n),
                            input_bound=2000,
                            output_bound=100,
                            policy=P,
                        )
                    except (ValueError, CostCapExceededError):
                        return None

                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(admit, range(8)))
                self.assertEqual(sum(x is not None for x in results), 1 if same else 2)
                self.assertEqual(b.exposure, F(".0045") if same else F(".009"))
                b.close()

    def test_real_sdk_rejects_raw_claims_before_release(self):
        for variant in (
            "negative",
            "bool",
            "string",
            "over_output",
            "wrong_model_zero",
            "bad_protocol_zero",
            "cache_zero",
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                b = s.Budget(root, cap=D("10"), namespace="offline-test-raw", create=True)
                m = runner.module("haiku45")
                script = m.ScriptedProvider(max_calls=m.EXPECTED_PROVIDER_CALLS)
                calls = []

                def handler(req, calls=calls, script=script, variant=variant, m=m):
                    calls.append(1)
                    resp = script(req)
                    body = json.loads(resp.content)
                    if variant in ("negative", "bool", "string"):
                        body["usage"]["input_tokens"] = {
                            "negative": -1,
                            "bool": True,
                            "string": "0",
                        }[variant]
                    elif variant == "over_output":
                        body["usage"]["output_tokens"] = m.MAX_OUTPUT_TOKENS + 1
                    else:
                        body["usage"]["input_tokens"] = body["usage"]["output_tokens"] = 0
                        if variant == "wrong_model_zero":
                            body["model"] = "wrong-model"
                        elif variant == "bad_protocol_zero":
                            body["content"] = [{"type": "bogus"}]
                        else:
                            body["usage"]["cache_read_input_tokens"] = 1
                    return httpx.Response(200, json=body, headers=dict(resp.headers))

                wire = WireCaptureTransport()
                wire.attach(
                    runner.DispatchTransport(httpx.MockTransport(handler), b, "haiku45")
                )
                client = runner.client_for("haiku45", runner.PLACEHOLDER, wire)
                try:
                    outcome = runner.execute_cell("haiku45", root, b, client, wire)
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(
                        any(x["state"] == "settled" for x in b.records.values())
                    )
                    self.assertGreater(b.exposure, 0)
                    self.assertEqual(b.totals()["known"], 0)
                    self.assertEqual(b.totals()["pending"], 0)
                    self.assertTrue(
                        all(x["state"] == "unknown" for x in b.records.values())
                    )
                    ledger = m.read_execution_ledger(
                        root / "haiku45" / "execution_ledger.ndjson",
                        require_complete=False,
                    )
                    self.assertEqual(
                        ledger.calls[-1].attempts[-1].cost_settlement, "forfeited"
                    )
                    self.assertTrue(
                        (root / "haiku45" / "execution_ledger.partial.ndjson").exists()
                    )
                    accounting = outcome["budget_accounting"]
                    self.assertEqual(
                        D(ledger.totals.forfeited_reservation_usd),
                        D(accounting["legacy_forfeited_usd"]),
                    )
                    self.assertEqual(
                        D(accounting["legacy_forfeited_usd"])
                        + D(accounting["wire_supplement_forfeited_usd"]),
                        s.decimal(b.totals()["unknown"]),
                    )
                    print(
                        json.dumps(
                            {
                                "case": variant,
                                "states": [x["state"] for x in b.records.values()],
                                "exposure": str(b.exposure),
                                "outcome": outcome["status"],
                            }
                        )
                    )
                finally:
                    client.close()
                    b.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
