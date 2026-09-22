# SPDX-License-Identifier: Apache-2.0
"""Public Grok 4.6 authority consumption is disabled.

Always refuses without credentials, record access, marker creation or dispatch.
Mock-only durable record tests use the operator's explicit injected transport;
this command cannot enable that test path and cannot consume live authority.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.parse_args(argv)
    print(json.dumps({"ok": False, "credential_read": False, "network_access": False}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
