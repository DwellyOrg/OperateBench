"""Global USD accounting successor, NOT owner authority or a paid entrypoint.

One process owns the exclusive journal lock; independent worker threads share
this object. A fresh process replays the journal and durably converts unresolved
attempts to unknown exposure before admitting anything new. Unknown liability
has no timeout or automatic release. Uses the existing aggregate reducer,
wire supplement, SharedGuard and exact rational arithmetic without changing
historical controllers or their audit-only recovery policy.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

from tools.aggregate_budget import Budget, canonical, money

PROFILE = "three-flow-global-usd-v1"
OPTIONAL_PROFILE = "three-flow-optional-limits-v1"


def validate_limit_policy(value: Any) -> None:
    """Explicit next-series consent and immutable historical-liability reference.

    Digests bind externally reviewed evidence; they do not reconcile old spend
    or authorize dispatch. Never reopen a capped journal under this identity.
    """
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "owner_accepted",
        "owner_evidence_sha256",
        "source_sha",
        "historical_liability_manifest_sha256",
    }:
        raise ValueError(
            "explicit optional limits and historical liability record required"
        )
    if value["mode"] != OPTIONAL_PROFILE or value["owner_accepted"] is not True:
        raise ValueError("explicit owner optional limits consent required")
    for key, size in (
        ("source_sha", 40),
        ("owner_evidence_sha256", 64),
        ("historical_liability_manifest_sha256", 64),
    ):
        if not isinstance(value[key], str) or not re.fullmatch(
            rf"[a-f0-9]{{{size}}}", value[key]
        ):
            raise ValueError("owner/source/historical liability binding required")


class CampaignBudget(Budget):
    """Trusted controller primitive: caller must separately admit prices/authority.

    The campaign ID survives source corrections. All requests, retries and
    linked reruns must use this SAME root and campaign ID. No per-model pools.
    Integrity hashes are corruption checks, not signatures or anti-rollback.
    """

    def __init__(
        self,
        root: Path,
        *,
        campaign_id: str,
        cap: Decimal | None,
        limit_policy: dict[str, Any] | None = None,
        create: bool = False,
        expected_journal_sha256: str | None = None,
        expected_genesis_sha256: str | None = None,
    ) -> None:
        if type(campaign_id) is not str or not campaign_id or len(campaign_id) > 200:
            raise ValueError("nonempty bounded campaign ID required")
        if cap is None:
            validate_limit_policy(limit_policy)
        elif limit_policy is not None:
            raise ValueError("historical capped policy cannot opt into optional limits")
        self.limit_policy = copy.deepcopy(limit_policy)
        self.controller_profile = OPTIONAL_PROFILE if cap is None else PROFILE
        self.cap = None if cap is None else money(cap)
        if self.cap is not None and self.cap != 1000:
            raise ValueError("this successor requires the global USD 1000 cap")
        if root != root.resolve(strict=True) or root.is_symlink():
            raise ValueError("canonical private root required")
        st = root.stat()
        if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
            raise ValueError("owner-only directory required")
        self.root = root
        self.namespace = campaign_id
        self.lock = threading.RLock()
        self.broken = False
        self.records = {}
        self.seq = 0
        self.previous = None
        self.fd = None
        self.dirfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        genesis: dict[str, Any] = {
            "kind": "genesis",
            "namespace": campaign_id,
            "cap": None if cap is None else "1000",
            "currency": "USD",
            "controller_profile": self.controller_profile,
        }
        if self.limit_policy is not None:
            genesis["limit_policy"] = self.limit_policy
        try:
            self.fd = os.open(
                "campaign-budget.jsonl",
                os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else 0),
                0o600,
                dir_fd=self.dirfd,
            )
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st = os.fstat(self.fd)
            if (
                st.st_uid != os.getuid()
                or stat.S_IMODE(st.st_mode) != 0o600
                or st.st_nlink != 1
                or not stat.S_ISREG(st.st_mode)
            ):
                raise ValueError("unsafe campaign journal")
            if expected_journal_sha256 is not None:
                with os.fdopen(os.dup(self.fd), "rb") as check:
                    actual_digest = hashlib.sha256(check.read()).hexdigest()
                os.lseek(self.fd, 0, os.SEEK_SET)
                if create or actual_digest != expected_journal_sha256:
                    raise ValueError("journal changed since admission")
            if create:
                self._append(genesis)
            else:
                with os.fdopen(os.dup(self.fd), "rb") as stream:
                    for line in stream:
                        event = json.loads(line)
                        if (
                            not line.endswith(b"\n")
                            or canonical(event) != line
                            or type(event["seq"]) is not int
                            or event["seq"] != self.seq
                            or event["prev"] != self.previous
                        ):
                            raise ValueError("torn or reordered campaign journal")
                        if self.seq == 0:
                            if (
                                expected_genesis_sha256 is not None
                                and hashlib.sha256(line).hexdigest()
                                != expected_genesis_sha256
                            ):
                                raise ValueError("financial genesis changed")
                            if event != dict(genesis, seq=0, prev=None):
                                raise ValueError("foreign campaign or cap")
                        else:
                            self.records = self._reduce(self.records, event)
                        self.seq += 1
                        self.previous = hashlib.sha256(line).hexdigest()
                if not self.seq:
                    raise ValueError("empty campaign journal")
                # No evidence that an unresolved attempt did not reach the wire.
                for ident, row in list(self.records.items()):
                    if row["state"] in ("reserved", "sent"):
                        self.forfeit(ident)
        except BaseException:
            self.close()
            raise

    def _reduce(
        self, old: dict[str, dict[str, Any]], e: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        if e["kind"] != "raw_rpc":
            return super()._reduce(old, e)
        rows = copy.deepcopy(old)
        row = rows[e["id"]]
        if (
            row["state"] != "sent"
            or type(e["ordinal"]) is not int
            or e["ordinal"] != row.get("raw_rpcs", 0) + 1
        ):
            raise ValueError("invalid raw RPC sequence")
        row["raw_rpcs"] = e["ordinal"]
        return rows

    def raw_rpc(self, cell: str, endpoint: str, request_sha256: str) -> None:
        """Every wire RPC, including non-resubmitting polls, durably precedes wire."""
        with self.lock:
            if self.broken:
                raise ValueError("broken journal")
            matches = [
                (i, r)
                for i, r in self.records.items()
                if r["cell"] == cell and r["state"] == "sent"
            ]
            if len(matches) != 1:
                raise ValueError("RPC requires one dispatched reservation")
            ident, row = matches[0]
            event = {
                "kind": "raw_rpc",
                "id": ident,
                "ordinal": row.get("raw_rpcs", 0) + 1,
                "endpoint": endpoint,
                "request_sha256": request_sha256,
            }
            reduced = self._reduce(self.records, event)
            self._append(event)
            self.records = reduced

    def cell_accounting(self, cell: str) -> dict[str, Any]:
        projection = super().cell_accounting(cell)
        projection["controller_profile"] = self.controller_profile
        return projection
