import os
import random
import tempfile
import threading
import unittest
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.providers.cost import CostCapExceededError
from tools import aggregate_budget as s

ROOT = Path(__file__).parent
P = LifecyclePricingPolicy("dummy-nonzero", D("2"), D("5"))


class Properties(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.b = s.Budget(
            self.root, cap=D(".02"), namespace="offline-test-properties", create=True
        )

    def tearDown(self):
        self.b.close()
        self.tmp.cleanup()

    def reserve(self, cell="a", i=2000, o=100):
        return self.b.reserve(cell, input_bound=i, output_bound=o, policy=P)

    def test_exact_known_pending_unknown_cap(self):
        a = self.reserve()
        self.b.dispatch("a", {"max_tokens": 100})
        self.b.settle(a, input_tokens=10, output_tokens=2)
        u = self.reserve("u")
        self.b.dispatch("u", {"max_tokens": 100})
        self.b.forfeit(u)
        self.reserve("p", i=7485)
        self.assertEqual(self.b.totals()["known"], F("0.000030"))
        self.assertEqual(self.b.totals()["unknown"], F("0.0045"))
        self.assertEqual(self.b.totals()["pending"], F("0.01547"))
        self.assertEqual(self.b.exposure, F(".02"))
        with self.assertRaises(CostCapExceededError):
            self.reserve("next", i=0, o=1)

    def test_cancel_proof_one_use_and_invalid_usage(self):
        a = self.reserve()
        self.b.cancel(a)
        with self.assertRaises(ValueError):
            self.b.cancel(a)
        a = self.reserve()
        self.b.dispatch("a", {"max_tokens": 100})
        for action in [
            lambda: self.b.cancel(a),
            lambda: self.b.dispatch("a", {"max_tokens": 100}),
            lambda: self.b.settle(a, input_tokens=-1, output_tokens=0),
            lambda: self.b.settle(a, input_tokens=True, output_tokens=0),
            lambda: self.b.settle(a, input_tokens=2001, output_tokens=0),
            lambda: self.b.settle(a, input_tokens=0, output_tokens=101),
        ]:
            with self.assertRaises(ValueError):
                action()
            self.assertEqual(self.b.exposure, F(".0045"))
        self.b.settle(a, input_tokens=0, output_tokens=0)
        with self.assertRaises(ValueError):
            self.b.settle(a, input_tokens=0, output_tokens=0)

    def test_overhead_and_refusal_before_inner(self):
        a = self.reserve(i=0)
        self.b.dispatch("a", {"max_tokens": 100, "sdk_extra": "x" * 100})
        self.assertGreater(self.b.exposure, F(".0005"))
        self.b.forfeit(a)
        b = self.reserve("b", i=0)
        with self.assertRaises(CostCapExceededError):
            self.b.dispatch("b", {"max_tokens": 100, "sdk_extra": "x" * 10000})
        self.assertEqual(self.b.records[b]["state"], "reserved")
        self.b.cancel(b)

    def test_unknown_full_and_recovery_frozen(self):
        a = self.reserve()
        self.b.dispatch("a", {"max_tokens": 100})
        self.b.settle(a, input_tokens=None, output_tokens=1)
        self.b.close()
        self.b = s.Budget(self.root, cap=D(".02"), namespace="offline-test-properties")
        self.assertEqual(self.b.totals()["unknown"], F(".0045"))
        with self.assertRaises(ValueError):
            self.reserve()
        with self.assertRaises(FileExistsError):
            s.Budget(
                self.root,
                cap=D(".02"),
                namespace="offline-test-properties",
                create=True,
            )

    def test_durable_before_release_and_io_failure(self):
        a = self.reserve()
        self.b.dispatch("a", {"max_tokens": 100})
        real = os.fsync
        observations = []

        def observe(fd):
            observations.append(self.b.exposure)
            return real(fd)

        with patch.object(s.os, "fsync", side_effect=observe):
            self.b.settle(a, input_tokens=10, output_tokens=2)
        self.assertEqual(observations, [F(".0045"), F(".0045")])
        a = self.reserve()
        self.b.dispatch("a", {"max_tokens": 100})
        old = self.b.exposure
        with (
            patch.object(s.os, "fsync", side_effect=OSError("synthetic")),
            self.assertRaises(OSError),
        ):
            self.b.settle(a, input_tokens=1, output_tokens=1)
        self.assertTrue(self.b.broken)
        self.assertEqual(self.b.exposure, old)
        with self.assertRaises(ValueError):
            self.reserve("z")

    def test_torn_reordered_duplicate_foreign_fail_closed(self):
        self.reserve()
        self.b.close()
        path = self.root / "successor-events.jsonl"
        raw = path.read_bytes()
        rows = raw.splitlines(keepends=True)
        for damaged in [
            raw[:-1],
            rows[1] + rows[0],
            raw + rows[1],
            b"",
            raw.replace(b"offline-test-properties", b"offline-test-foreign"),
        ]:
            path.write_bytes(damaged)
            with self.assertRaises((ValueError, KeyError)):
                s.Budget(self.root, cap=D(".02"), namespace="offline-test-properties")
        path.write_bytes(raw)

    def test_independent_blocked_transport(self):
        first = threading.Event()
        release = threading.Event()
        second = threading.Event()
        errors = []

        class Inner:
            def __init__(self, cell):
                self.cell = cell

            def send(inner, request):
                try:
                    r = self.reserve(inner.cell)
                    self.b.dispatch(inner.cell, {"max_tokens": 100})
                    if inner.cell == "a":
                        first.set()
                        release.wait()
                    else:
                        second.set()
                    self.b.settle(r, input_tokens=1, output_tokens=1)
                except BaseException as e:
                    errors.append(e)

        a = threading.Thread(
            target=s.FairTransport(Inner("a"), self.b, "a").send, args=({},)
        )
        b = threading.Thread(
            target=s.FairTransport(Inner("b"), self.b, "b").send, args=({},)
        )
        a.start()
        try:
            self.assertTrue(first.wait(3))
            b.start()
            self.assertTrue(second.wait(3))
            journal = (self.root / "successor-events.jsonl").read_text()
            self.assertIn('"cell":"b"', journal)
        finally:
            release.set()
            a.join(3)
            if b.ident is not None:
                b.join(3)
        self.assertFalse(a.is_alive())
        self.assertFalse(b.is_alive())
        self.assertEqual(errors, [])

    def test_100_random_sequences_independent_integer_oracle(self):
        # Integer microUSD: rate fixture makes reservation=2*i+5*o.
        self.b.close()
        for seed in range(100):
            with tempfile.TemporaryDirectory() as temp:
                b = s.Budget(
                    Path(temp),
                    cap=D(".02"),
                    namespace=f"offline-test-random-{seed}",
                    create=True,
                )
                rng = random.Random(seed)
                known = unknown = 0
                pending = {}
                for n in range(80):
                    if not pending or rng.random() < 0.55:
                        i = rng.randrange(1100, 2000)
                        o = rng.randrange(1, 80)
                        amount = 2 * i + 5 * o
                        try:
                            r = b.reserve(str(n), input_bound=i, output_bound=o, policy=P)
                        except CostCapExceededError:
                            self.assertGreater(
                                known
                                + unknown
                                + sum(x[1] for x in pending.values())
                                + amount,
                                20000,
                            )
                        else:
                            pending[r] = (str(n), amount, i, o)
                    else:
                        r = rng.choice(list(pending))
                        cell, amount, i, o = pending.pop(r)
                        choice = rng.randrange(3)
                        if choice == 0:
                            b.cancel(r)
                        else:
                            b.dispatch(cell, {"max_tokens": o})
                            if choice == 1:
                                b.forfeit(r)
                                unknown += amount
                            else:
                                ui = rng.randrange(i + 1)
                                uo = rng.randrange(o + 1)
                                b.settle(r, input_tokens=ui, output_tokens=uo)
                                known += 2 * ui + 5 * uo
                    expected = F(
                        known + unknown + sum(x[1] for x in pending.values()), 1000000
                    )
                    self.assertEqual(b.exposure, expected)
                    self.assertLessEqual(expected, F(".02"))
                b.close()

    def test_adapter_same_price_stale_token_refused(self):
        g = s.SharedGuard(budget=self.b, cell="a", policy=P, max_output_tokens=100)
        old = g.authorize(input_tokens_upper_bound=1000)
        g.cancel(old, input_tokens_upper_bound=1000)
        fresh = g.authorize(input_tokens_upper_bound=1000)
        with self.assertRaises(ValueError):
            g.cancel(old, input_tokens_upper_bound=1000)
        self.b.dispatch("a", {"max_tokens": 100})
        g.settle(fresh, input_tokens=10, output_tokens=2)
        with self.assertRaises(ValueError):
            g.forfeit(fresh)


if __name__ == "__main__":
    unittest.main()
