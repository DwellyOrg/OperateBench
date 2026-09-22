"""Collection must not change permission semantics for unrelated tests."""

import os
import subprocess
import sys
from pathlib import Path


def test_aggregate_review_import_preserves_process_umask() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; os.umask(0o022); "
            "import tests.test_aggregate_budget_review; "
            "actual = os.umask(0o022); assert actual == 0o022, oct(actual)",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
