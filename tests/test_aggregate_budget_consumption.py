import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import aggregate_budget as s


class Consumption(unittest.TestCase):
    def test_permanent_consume_before_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            order = []
            real = s.os.fsync

            def sync(fd):
                order.append("sync")
                return real(fd)

            with patch.object(s.os, "fsync", side_effect=sync):
                s.consume_offline_once(root)
            order.append("dummy-client-construction")
            self.assertEqual(order, ["sync", "sync", "dummy-client-construction"])
            with self.assertRaises(FileExistsError):
                s.consume_offline_once(root)

    def test_failed_consume_never_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(s.os, "fsync", side_effect=OSError("synthetic")),
                self.assertRaises(OSError),
            ):
                s.consume_offline_once(root)
            with self.assertRaises(FileExistsError):
                s.consume_offline_once(root)
