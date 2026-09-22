"""Six-model episode100 controller. No paid authority is shipped.

--offline uses real SDKs over MockTransport. --mock-authority exercises the
same one-shot launch gate with dummy files. --live requires a new external
owner approval, explicit ceiling, clean exact source, and sibling retirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from operatebench.agents.evidence import WireCaptureTransport
from tools.aggregate_budget import (
    CELLS,
    Budget,
    canonical,
    decimal,
    money,
)
from tools.diagnose_aggregate_budget import (
    PLACEHOLDER,
    DispatchTransport,
    client_for,
    execute_cell,
    module,
)
from tools.episode100 import PROFILE

SOURCE = Path(__file__).resolve().parents[1]
GROK_CELLS = ("grok45", "grok46")
GROK_PROFILE = "synthetic-demo-episode100-grok-pair-v1"
TIMEOUT_CELLS = ("grok46",)
TIMEOUT_PROFILE = "synthetic-demo-episode100-grok46-timeout120-v1"


def scope(cells: tuple[str, ...]) -> str:
    if cells == CELLS:
        return PROFILE
    if cells == GROK_CELLS:
        return GROK_PROFILE
    if cells == TIMEOUT_CELLS:
        return TIMEOUT_PROFILE
    raise ValueError("unsupported closed cell scope")


def load_series(
    path: Path | None, cap: Decimal, cells: tuple[str, ...]
) -> dict[str, Any] | None:
    scope(cells)
    if cells == CELLS:
        if path is not None:
            raise ValueError("series manifest is only supported for the Grok pair")
        return None
    if path is None:
        raise ValueError("Grok pair requires an external series manifest")
    fd = private_fd(path)
    try:
        value = read_json(fd)
    finally:
        os.close(fd)
    fields = {
        "schema",
        "series_id",
        "prior_evidence_sha256",
        "authorized_total_usd",
        "committed_exposure_usd",
        "new_cap_usd",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("invalid series manifest")
    if (
        value["schema"] != "episode100-series-v1"
        or type(value["series_id"]) is not str
        or not value["series_id"].strip()
        or type(value["prior_evidence_sha256"]) is not list
        or not value["prior_evidence_sha256"]
        or any(
            type(v) is not str
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for v in value["prior_evidence_sha256"]
        )
    ):
        raise ValueError("invalid series identity")
    for field in ("authorized_total_usd", "committed_exposure_usd", "new_cap_usd"):
        if type(value[field]) is not str:
            raise ValueError("series money must be decimal strings")
    total, carry, new = (
        Decimal(value[k])
        for k in ("authorized_total_usd", "committed_exposure_usd", "new_cap_usd")
    )
    if any(not x.is_finite() or x <= 0 for x in (total, carry, new)):
        raise ValueError("positive finite series amounts required")
    if money(total) != money(carry) + money(new) or new != cap:
        raise ValueError("series cap must retain committed exposure")
    if cells == TIMEOUT_CELLS:
        from tools.episode100_grok46_timeout120 import SERIES

        if value != SERIES:
            raise ValueError("120s diagnostic requires pinned prior series and carry")
    return dict(value)


def series_digest(series: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(series)).hexdigest()


def source_sha() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE):
        raise ValueError("clean exact source required")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()


def private_fd(path: str | Path) -> int:
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise ValueError("canonical absolute private file required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        if (
            not stat.S_ISREG(st.st_mode)
            or st.st_uid != os.getuid()
            or stat.S_IMODE(st.st_mode) != 0o600
            or st.st_nlink != 1
            or st.st_size > 65536
        ):
            raise ValueError("unsafe private file custody")
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_json(fd: int) -> Any:
    raw = os.read(fd, 65537)
    if len(raw) > 65536:
        raise ValueError("oversized private JSON")
    return json.loads(raw)


def consume(
    authority: Path | None,
    *,
    output: Path,
    cap: Decimal,
    mock: bool,
    cells: tuple[str, ...] = CELLS,
    series: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    """Pin credential descriptor without reading; durably consume first."""
    if authority is None:
        raise ValueError(
            "new external authority required; old authorities are not reusable"
        )
    profile = scope(cells)
    if (cells != CELLS) != (series is not None):
        raise ValueError("series required exactly for closed Grok scopes")
    sha = source_sha()
    afd = private_fd(authority)
    try:
        approval = read_json(afd)
    finally:
        os.close(afd)
    expected = {
        "profile": profile,
        "source_sha": sha,
        "output": str(output),
        "cap_usd": str(cap),
        "cells": list(cells),
        "max_episode_decisions": 100,
        "retries": 0,
        "approved": True,
        "historical_authorities_retired": True,
        "mode": "dummy" if mock else "paid",
    }
    if cells == TIMEOUT_CELLS:
        expected["turn_deadline_seconds"] = 120
    if series is not None:
        expected["series_manifest_sha256"] = series_digest(series)
    if (
        type(approval) is not dict
        or set(approval) != set(expected) | {"credential_file"}
        or any(
            type(approval[k]) is not type(v) or approval[k] != v
            for k, v in expected.items()
        )
    ):
        raise ValueError("new authority does not exactly match this launch")
    directory = Path(authority).parent
    if (
        directory != directory.resolve(strict=True)
        or stat.S_IMODE(directory.stat().st_mode) != 0o700
        or directory.stat().st_uid != os.getuid()
    ):
        raise ValueError("private authority directory required")
    credential = private_fd(approval["credential_file"])
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Directory-wide marker, not output-local: changing output cannot reuse it.
        fd = os.open(
            "EPISODE100-CONSUMED",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=dfd,
        )
        try:
            receipt = canonical(
                {
                    "profile": profile,
                    "source_sha": sha,
                    "output": str(output),
                    "cap_usd": str(cap),
                    "mode": expected["mode"],
                    "authorizing": False,
                    "cells": list(cells),
                    **({"turn_deadline_seconds": 120} if cells == TIMEOUT_CELLS else {}),
                    **(
                        {"series_manifest_sha256": series_digest(series)}
                        if series is not None
                        else {}
                    ),
                }
            )
            if os.write(fd, receipt) != len(receipt):
                raise OSError("short consumption receipt")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(dfd)
        keys = read_json(credential)
        if (
            type(keys) is not dict
            or set(keys) != set(cells)
            or any(type(v) is not str or not v.strip() for v in keys.values())
        ):
            raise ValueError(
                "invalid cell credential mapping; authority remains consumed"
            )
        if mock and any(v != PLACEHOLDER for v in keys.values()):
            raise ValueError("dummy mode accepts only the public placeholder")
        return keys, sha
    finally:
        os.close(credential)
        os.close(dfd)


def run(
    root: Path,
    cap: Decimal,
    keys: dict[str, str],
    *,
    mock: bool,
    sha: str,
    cells: tuple[str, ...] = CELLS,
    series: dict[str, Any] | None = None,
) -> int:
    profile = scope(cells)
    if set(keys) != set(cells) or (cells != CELLS) != (series is not None):
        raise ValueError("exact scope and series required")
    root.mkdir(mode=0o700)
    if series is not None:
        with (root / "series-manifest.json").open("xb") as stream:
            stream.write(canonical(series))
            stream.flush()
            os.fsync(stream.fileno())
        dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    budget = Budget(
        root,
        cap=cap,
        namespace=("offline-test-" if mock else "episode100-paid-") + profile,
        create=True,
    )
    prepared = []
    try:
        for cell in cells:
            wire = WireCaptureTransport()
            native = (
                httpx.MockTransport(module(cell).ScriptedProvider(max_calls=100))
                if mock
                else httpx.HTTPTransport(retries=0, trust_env=False)
            )
            wire.attach(DispatchTransport(native, budget, cell))
            prepared.append((cell, client_for(cell, keys[cell], wire), wire))
        keys.clear()
        with ThreadPoolExecutor(max_workers=len(cells)) as pool:
            futures = [
                pool.submit(
                    execute_cell,
                    cell,
                    root,
                    budget,
                    client,
                    wire,
                    episode100=True,
                    grok46_timeout120=cells == TIMEOUT_CELLS,
                )
                for cell, client, wire in prepared
            ]
            results = [f.result() for f in futures]
        summary = {
            "controller_profile": profile,
            **({"turn_deadline_seconds": 120} if cells == TIMEOUT_CELLS else {}),
            **(
                {
                    "series_manifest_sha256": series_digest(series),
                    "committed_exposure_usd": series["committed_exposure_usd"],
                }
                if series is not None
                else {}
            ),
            "source_sha": sha,
            "mock": mock,
            "max_decisions_per_episode": 100,
            "cells": results,
            "aggregate": {k: str(decimal(v)) for k, v in budget.totals().items()},
        }
        (root / "summary.json").write_bytes(canonical(summary))
        print(json.dumps(summary, indent=2))
        return 0 if all(r.get("bundle_ok") and r.get("replay_ok") for r in results) else 1
    finally:
        keys.clear()
        for _, client, wire in prepared:
            getattr(client, "close", wire.close)()
        budget.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--mock-authority", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cap-usd", required=True)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--cells", choices=("grok45,grok46", "grok46"))
    parser.add_argument("--grok46-timeout120", action="store_true")
    parser.add_argument("--series-manifest", type=Path)
    args = parser.parse_args(argv)
    if (args.cells == "grok46") != args.grok46_timeout120:
        parser.error("--cells grok46 requires --grok46-timeout120 exclusively")
    cells = (
        TIMEOUT_CELLS if args.grok46_timeout120 else GROK_CELLS if args.cells else CELLS
    )
    cap = Decimal(args.cap_usd)
    if money(cap) <= 0:
        raise ValueError("positive newly approved cap required")
    output = args.output.absolute()
    if output.exists() or output != output.resolve():
        raise ValueError("new canonical output required")
    series = load_series(args.series_manifest, cap, cells)
    if args.offline:
        if args.authority:
            raise ValueError("offline preparation must not consume real authority")
        keys, sha = dict.fromkeys(cells, PLACEHOLDER), source_sha()
    else:
        keys, sha = consume(
            args.authority,
            output=output,
            cap=cap,
            mock=args.mock_authority,
            cells=cells,
            series=series,
        )
    return run(output, cap, keys, mock=not args.live, sha=sha, cells=cells, series=series)


if __name__ == "__main__":
    raise SystemExit(main())
