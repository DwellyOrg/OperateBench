"""Consuming three-flow CLI. Default preflight is inert; no approval writer.

Run this file with a reviewed interpreter and --expected-source for consumption.
Only the owner supplies external admission. No credential paths are defaulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAP = 1048576


def source_identity(expected: str | None = None) -> str:
    # Stdlib only until this gate. Ignore ambient Git routing/configuration.
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(ROOT), *args],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all") or (
        expected is not None and expected != sha
    ):
        raise ValueError("clean expected source required")
    return sha


def normal_path(value: str, *, exists: bool = True) -> Path:
    if not isinstance(value, str) or "\0" in value or value != os.path.normpath(value):
        raise ValueError("normal absolute path required")
    p = Path(value)
    if not p.is_absolute() or p.resolve(strict=exists) != p:
        raise ValueError("canonical path required")
    for component in (p, *p.parents):
        if component.exists() and component.is_symlink():
            raise ValueError("symlink custody forbidden")
    return p


def private_directory(value: str) -> int:
    p = normal_path(value)
    fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        s, named = os.fstat(fd), p.lstat()
        if (
            s.st_uid != os.getuid()
            or stat.S_IMODE(s.st_mode) != 0o700
            or (s.st_dev, s.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("owner-only mode-0700 directory required")
        return fd
    except BaseException:
        os.close(fd)
        raise


def private_fd(value: str, *, cap: int = CAP) -> int:
    p = normal_path(value)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        s, named = os.fstat(fd), p.lstat()
        if (
            not stat.S_ISREG(s.st_mode)
            or s.st_uid != os.getuid()
            or stat.S_IMODE(s.st_mode) != 0o600
            or s.st_nlink != 1
            or not 0 < s.st_size <= cap
            or (s.st_dev, s.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("unsafe private file custody")
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_pinned(fd: int, cap: int) -> bytes:
    raw = bytearray()
    while len(raw) <= cap:
        part = os.read(fd, min(65536, cap + 1 - len(raw)))
        if not part:
            return bytes(raw)
        raw.extend(part)
    raise ValueError("private file exceeded bound")


def read_private(path: str) -> bytes:
    fd = private_fd(path)
    try:
        return read_pinned(fd, CAP)
    finally:
        os.close(fd)


def decode(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate admission field")
            value[key] = item
        return value

    return json.loads(raw, object_pairs_hook=pairs)


def publish(dfd: int, name: str, value: Any) -> None:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=dfd,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short claim write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(dfd)


def parse_env(raw: bytes, wanted: set[str]) -> dict[str, str]:
    # Parse literal API-key assignments with the strict shared grammar; never shell.
    values: dict[str, str] = {}
    seen: set[str] = set()
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export )?([A-Z_][A-Z0-9_]*)=(.*)", line)
        if match is None:
            raise ValueError("invalid credential grammar")
        name, value = match.groups()
        if name in seen:
            raise ValueError("duplicate credential name")
        seen.add(name)
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("invalid credential quoting")
            value = value[1:-1]
        if re.fullmatch(r"[A-Za-z0-9_./:+,=@%!?-]+", value) is None:
            raise ValueError("invalid credential value")
        if name in wanted:
            values[name] = value
    if set(values) != wanted:
        raise ValueError("required provider credential absent")
    return values


def consume(path: Path, sha: str, *, network: Any = None) -> list[dict[str, Any]]:
    # The CLI already checked source before importing these executable modules.
    from decimal import Decimal

    from tools.aggregate_budget import canonical
    from tools.three_flow_admission import (
        VARIABLES,
        digest,
        validate,
        validate_initial_receipt,
    )
    from tools.three_flow_budget import CampaignBudget
    from tools.three_flow_campaign import (
        assignments,
        campaign_manifest,
        correction_slot,
        run_campaign,
        write_private,
    )
    from tools.three_flow_live_factory import live_factory

    admission_path = normal_path(str(path))
    if admission_path.is_relative_to(ROOT):
        raise ValueError("external admission required")
    a = validate(decode(read_private(str(path))), sha, synthetic=network is not None)
    cap = None if a["cap_usd"] is None else Decimal(a["cap_usd"])
    limit_policy = a.get("limit_policy")
    custody = normal_path(a["custody_root"])
    if admission_path.parent != custody or custody.is_relative_to(ROOT):
        raise ValueError("admission must be in external series custody directory")
    root = normal_path(a["campaign_root"], exists=a["operation"] != "initial")
    if root.is_relative_to(ROOT) or root == custody:
        raise ValueError("external distinct campaign root required")
    if a["operation"] == "initial" and root.exists():
        raise ValueError("initial root already exists")
    dfd = private_directory(str(custody))
    budget = None
    credential = None
    keys: dict[str, str] = {}
    try:
        original = custody / "SERIES-CONSUMED"
        slots = [
            s
            for s in assignments(a["campaign_id"], a.get("roster_profile", "legacy-v1"))
            if s["trial_id"] in a["selected"]
        ]
        corrections = None
        if a["operation"] != "initial":
            raw = read_private(str(original))
            old = decode(raw)
            if hashlib.sha256(raw).hexdigest() != a["original_claim_sha256"] or any(
                old[k] != a[k]
                for k in ("campaign_id", "campaign_root", "custody_root", "mode")
            ):
                raise ValueError("foreign original series claim")
            if a["attempt_id"] == old["attempt_id"]:
                raise ValueError("distinct continuation attempt required")
            if any(s["trial_id"] not in old["selected"] for s in slots):
                raise ValueError("continuation exceeds original scope")
            binding = a.get("correction_binding")
            if (
                old["source_sha"] != sha
                and a["operation"] == "correction"
                and binding is None
            ):
                raise ValueError("source transition requires correction binding")
            if binding is not None:
                prior = binding["original_admission"]
                validate_initial_receipt(old, prior)
                # Bind to the initial attempt from the admitted original, not an
                # attempt name supplied by a mutated SERIES-CONSUMED record.
                counterpart = decode(
                    read_private(str(custody / ("RECEIPT-" + prior["attempt_id"])))
                )
                validate_initial_receipt(counterpart, prior)
                if (
                    digest(prior) != old["admission_sha256"]
                    or old["source_sha"] != binding["original_source_sha"]
                    or old["cap_usd"] != a["cap_usd"]
                    or old.get("limit_policy") != limit_policy
                    or hashlib.sha256(
                        read_private(str(root / "assignment.json"))
                    ).hexdigest()
                    != binding["manifest_sha256"]
                ):
                    raise ValueError("correction original authority or manifest mismatch")
                for ident, artifacts in binding[
                    "original_trial_artifacts_sha256"
                ].items():
                    for kind, expected in artifacts.items():
                        if (
                            hashlib.sha256(
                                read_private(str(root / (ident + "." + kind + ".json")))
                            ).hexdigest()
                            != expected
                        ):
                            raise ValueError("original correction artifact changed")
            if (custody / ("ATTEMPT-" + a["attempt_id"] + "-CONSUMED")).exists():
                raise ValueError("continuation authority already consumed")
            # Lock, hash-check, then replay/reconcile before any credential read.
            budget = CampaignBudget(
                root,
                campaign_id=a["campaign_id"],
                cap=cap,
                limit_policy=limit_policy,
                create=False,
                expected_journal_sha256=a["journal_sha256"],
                expected_genesis_sha256=binding["genesis_sha256"] if binding else None,
            )
            if a["operation"] == "resume":
                actual_pending = [
                    s["trial_id"]
                    for s in slots
                    if not (root / (s["trial_id"] + ".assigned.json")).exists()
                ]
                if actual_pending != a["pending"]:
                    raise ValueError("pending subset changed; no resubmission")
            else:
                corrections = []
                for slot in slots:
                    ident = slot["trial_id"]
                    closure = decode(read_private(str(root / (ident + ".closure.json"))))
                    fix = a["corrections"][ident]
                    if closure["classification"] not in ("aborted", "excluded"):
                        raise ValueError("scored model behavior is not a technical rerun")
                    corrections.append(
                        correction_slot(
                            slot, a["attempt_id"], fix["reason"], fix["evidence_sha256"]
                        )
                    )
        # Pin descriptor only after all nonsecret admission checks.
        credential_parent = private_directory(
            str(normal_path(a["credential_file"]).parent)
        )
        os.close(credential_parent)
        credential = private_fd(a["credential_file"], cap=65536)
        source_identity(sha)
        receipt = {
            k: a[k]
            for k in (
                "campaign_id",
                "campaign_root",
                "custody_root",
                "attempt_id",
                "selected",
                "mode",
                "source_sha",
                "operation",
                "original_claim_sha256",
                "journal_sha256",
            )
        }
        receipt.update(
            authorizing=False, admission_sha256=digest(a), cap_usd=a["cap_usd"]
        )
        if limit_policy is not None:
            receipt["limit_policy"] = limit_policy
        if "correction_binding" in a:
            receipt["correction_binding"] = a["correction_binding"]
        publish(
            dfd,
            "SERIES-CONSUMED"
            if a["operation"] == "initial"
            else "ATTEMPT-" + a["attempt_id"] + "-CONSUMED",
            receipt,
        )
        publish(dfd, "RECEIPT-" + a["attempt_id"], receipt)
        if budget is None:
            root.mkdir(mode=0o700)
            budget = CampaignBudget(
                root,
                campaign_id=a["campaign_id"],
                cap=cap,
                limit_policy=limit_policy,
                create=True,
            )
        if a["operation"] == "initial":
            write_private(
                root / "assignment.json",
                canonical(
                    campaign_manifest(
                        a["campaign_id"],
                        slots,
                        a["mode"],
                        limit_policy=limit_policy,
                        roster_profile=a.get("roster_profile", "legacy-v1"),
                    )
                ),
            )
        wanted = {VARIABLES[s["provider"]] for s in slots}
        keys = parse_env(read_pinned(credential, 65536), wanted)
        factory = live_factory(a, keys, network=network)
        return run_campaign(
            root,
            campaign_id=a["campaign_id"],
            slots=slots,
            resume=a["operation"] != "initial",
            campaign_budget=budget,
            transport_factory=factory,
            mode=a["mode"],
            attempt_id=a["attempt_id"],
            selected_corrections=corrections,
            assignment_prepared=a["operation"] == "initial",
            roster_profile=a.get("roster_profile", "legacy-v1"),
        )
    finally:
        keys.clear()
        if credential is not None:
            os.close(credential)
        if budget is not None:
            budget.close()
        os.close(dfd)


def main(argv: list[str] | None = None, *, network: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "run"), default="preflight", nargs="?"
    )
    parser.add_argument("--expected-source")
    parser.add_argument("--admission")
    args = parser.parse_args(argv)
    try:
        if args.mode == "run" and (not args.expected_source or args.admission is None):
            raise ValueError("expected source and external admission required")
        sha = source_identity(args.expected_source)
        sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
        if args.mode == "preflight":
            from tools.three_flow_admission import proposal

            print(json.dumps(proposal(sha), indent=2))
            return 0
        rows = consume(normal_path(args.admission), sha, network=network)
        print(
            json.dumps(
                {
                    "assigned": len(rows),
                    "scored": sum(r["classification"] == "scored" for r in rows),
                    "origin": "synthetic"
                    if network is not None
                    else "provider_candidate",
                }
            )
        )
        return 0 if all(r["classification"] == "scored" for r in rows) else 2
    except Exception:
        # No provider exceptions, credential values or arbitrary admission prose.
        print(
            json.dumps(
                {
                    "status": "refused_or_incomplete",
                    "authority_may_be_consumed": args.mode == "run",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
