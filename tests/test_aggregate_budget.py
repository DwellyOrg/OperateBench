import tempfile
import unittest
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path


class Settlement(unittest.TestCase):
    def test_known_releases(self):
        from operatebench.agents.pricing import LifecyclePricingPolicy
        from tools import aggregate_budget as s

        with tempfile.TemporaryDirectory() as tmp:
            b = s.Budget(Path(tmp), cap=D("1"), namespace="offline-test-new", create=True)
            p = LifecyclePricingPolicy("dummy", D("2"), D("5"))
            r = b.reserve("a", input_bound=1000, output_bound=100, policy=p)
            self.assertEqual(b.exposure, F("0.0025"))
            b.dispatch("a", {"max_tokens": 100})
            b.settle(r, input_tokens=10, output_tokens=2)
            self.assertEqual(b.exposure, F("0.000030"))
            b.close()

    def test_only_typed_bound_refusal_closes_unknown(self):
        from operatebench.agents.pricing import LifecyclePricingPolicy
        from operatebench.providers.cost import CostReservationBreachedError
        from tools import aggregate_budget as s

        with tempfile.TemporaryDirectory() as tmp:
            b = s.Budget(
                Path(tmp), cap=D("1"), namespace="offline-test-breach", create=True
            )
            try:
                g = s.SharedGuard(
                    budget=b,
                    cell="a",
                    policy=LifecyclePricingPolicy("dummy", D("2"), D("5")),
                    max_output_tokens=100,
                )
                r = g.authorize(input_tokens_upper_bound=0)
                b.dispatch("a", {"max_tokens": 100, "wire_extra": "x" * 100})
                full = b.total
                # Malformed internal numeric input is NOT a classified provider breach.
                with self.assertRaises(ValueError) as internal:
                    g.settle(r, input_tokens=-1, output_tokens=0)
                self.assertNotIsInstance(internal.exception, CostReservationBreachedError)
                self.assertEqual(b.totals()["pending"], F(full))
                with self.assertRaises(CostReservationBreachedError) as caught:
                    g.settle(r, input_tokens=0, output_tokens=101)
                self.assertIsNone(caught.exception.measured_usd)
                self.assertEqual(caught.exception.reservation_usd, full)
                self.assertEqual(g.forfeited_usd, full)
                self.assertEqual(b.totals()["unknown"], F(full))
                self.assertEqual(b.totals()["pending"], 0)
                self.assertIsNone(g.ticket)
                report = b.cell_accounting("a")
                self.assertEqual(
                    D(report["legacy_forfeited_usd"])
                    + D(report["wire_supplement_forfeited_usd"]),
                    full,
                )
                with self.assertRaises(ValueError):
                    g.forfeit(r)
            finally:
                b.close()

    def test_fixed_token_ceiling_still_applies(self):
        from operatebench.agents.pricing import LifecyclePricingPolicy
        from operatebench.providers.cost import CostCapExceededError
        from tools import aggregate_budget as s

        with tempfile.TemporaryDirectory() as tmp:
            b = s.Budget(
                Path(tmp), cap=D("1"), namespace="offline-test-tokens", create=True
            )
            try:
                g = s.SharedGuard(
                    budget=b,
                    cell="a",
                    policy=LifecyclePricingPolicy("dummy", D("2"), D("5")),
                    max_output_tokens=100,
                    token_hard_cap=1099,
                )
                with self.assertRaises(CostCapExceededError):
                    g.authorize(input_tokens_upper_bound=1000)
                self.assertEqual(b.records, {})
            finally:
                b.close()


if __name__ == "__main__":
    unittest.main()
