"""Optional settled aggregate controller. Trusted in-process API, not launch authority."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import threading
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
from typing import Any

from operatebench.agents.pricing import (
    GuardEvent,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.providers.cost import (
    CostCapExceededError,
    CostReservationBreachedError,
    request_input_token_bound,
)

CONTROLLER_PROFILE = "settled-aggregate-bounded-v1"


class SettlementBoundsError(ValueError):
    """Validated integer usage exceeds this request's admitted token bounds."""


CELLS = ("haiku45", "sonnet5", "luna56", "grok45", "grok46", "mistralsmall")


def canonical(x: Any) -> bytes:
    return (
        json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def count(x: Any, positive: bool = False) -> int:
    if type(x) is not int or x < int(positive):
        raise ValueError("invalid token count")
    return x


def money(x: Any) -> F:
    if type(x) not in (str, Decimal, int) or isinstance(x, bool):
        raise ValueError("exact amount required")
    d = Decimal(x)
    if not d.is_finite() or d < 0:
        raise ValueError("invalid amount")
    return F(d)


def decimal(x: F) -> Decimal:
    # Rates are finite decimal; denominator factors only 2 and 5.
    from decimal import localcontext

    with localcontext() as c:
        c.prec = max(100, len(str(abs(x.numerator))) + len(str(x.denominator)) + 10)
        return Decimal(x.numerator) / Decimal(x.denominator)


def price(r: dict[str, Any], i: int, o: int) -> F:
    return (count(i) * money(r["ir"]) + count(o) * money(r["or"])) / 1000000


def consume_offline_once(root: Path) -> None:
    """Permanent dummy marker before SDK construction; NOT paid authorization.

    Failure retains the marker. No credential source, inheritance, or reset API.
    """
    if (
        root != root.resolve(strict=True)
        or root.stat().st_uid != os.getuid()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
    ):
        raise ValueError("private canonical root required")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        fd = os.open(
            "OFFLINE-ONLY-CONSUMED",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        raw = b"OFFLINE DUMMY ONLY; no paid authority\n"
        if os.write(fd, raw) != len(raw):
            raise OSError("partial consumption marker")
        os.fsync(fd)
        os.fsync(directory)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


class Budget:
    def __init__(
        self, root: Path, *, cap: Decimal, namespace: str, create: bool = False
    ) -> None:
        # Namespace is accounting identity, never launch authorization. The new
        # CLI consumes external one-shot authority before selecting live HTTP.
        if not namespace.startswith(
            ("offline-test-", "episode100-paid-", "campaign-paid-")
        ):
            raise ValueError("unsupported controller namespace")
        if root != root.resolve(strict=True) or root.is_symlink():
            raise ValueError("canonical root required")
        st = root.stat()
        if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
            raise ValueError("root-only directory required")
        self.cap = money(cap)
        if self.cap <= 0:
            raise ValueError("positive cap required")
        self.root = root
        self.namespace = namespace
        self.lock = threading.RLock()
        self.broken = False
        self.records: dict[str, dict[str, Any]] = {}
        self.seq = 0
        self.previous: str | None = None
        self.fd: int | None = None
        self.dirfd: int | None = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            # Stable lifetime lock; append journal is never replaced or reset.
            self.fd = os.open(
                root / "successor-events.jsonl",
                (os.O_RDWR | os.O_CREAT | os.O_EXCL if create else os.O_RDONLY)
                | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st = os.fstat(self.fd)
            if (
                st.st_uid != os.getuid()
                or stat.S_IMODE(st.st_mode) != 0o600
                or st.st_nlink != 1
            ):
                raise ValueError("unsafe journal")
            if create:
                self._append(
                    {
                        "kind": "genesis",
                        "namespace": namespace,
                        "cap": str(cap),
                        "controller_profile": CONTROLLER_PROFILE,
                    }
                )
            else:
                raw = b""
                while block := os.read(self.fd, 65536):
                    raw += block
                if not raw or not raw.endswith(b"\n"):
                    raise ValueError("empty/torn journal")
                for line in raw.splitlines(keepends=True):
                    e = json.loads(line)
                    if (
                        canonical(e) != line
                        or e["seq"] != self.seq
                        or e["prev"] != self.previous
                    ):
                        raise ValueError("noncanonical/reordered/duplicate journal")
                    if self.seq == 0:
                        if e != {
                            "kind": "genesis",
                            "controller_profile": CONTROLLER_PROFILE,
                            "namespace": namespace,
                            "cap": str(cap),
                            "seq": 0,
                            "prev": None,
                        }:
                            raise ValueError("foreign authority")
                    else:
                        self.records = self._reduce(self.records, e)
                    self.seq += 1
                    self.previous = hashlib.sha256(line).hexdigest()
                # Recovery is audit-only: cannot trust absence of dispatch after crash.
                self.broken = True
        except BaseException:
            self.close()
            raise

    @property
    def exposure(self) -> F:
        return self.totals()["exposure"]

    @property
    def total(self) -> Decimal:
        return decimal(self.exposure)

    def totals(self) -> dict[str, F]:
        with self.lock:
            known = sum(
                (
                    F(r["actual"])
                    for r in self.records.values()
                    if r["state"] == "settled"
                ),
                F(0),
            )
            pending = sum(
                (
                    F(r["bound"])
                    for r in self.records.values()
                    if r["state"] in ("reserved", "sent")
                ),
                F(0),
            )
            unknown = sum(
                (F(r["bound"]) for r in self.records.values() if r["state"] == "unknown"),
                F(0),
            )
            return {
                "known": known,
                "pending": pending,
                "unknown": unknown,
                "exposure": known + pending + unknown,
                "remaining": self.cap - known - pending - unknown,
            }

    def _reduce(
        self, old: dict[str, dict[str, Any]], e: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        rows = copy.deepcopy(old)
        k = e["kind"]
        ident = e["id"]
        if k == "reserve":
            if ident in rows or any(
                r["cell"] == e["cell"] and r["state"] in ("reserved", "sent")
                for r in rows.values()
            ):
                raise ValueError("duplicate admission")
            count(e["ib"])
            count(e["ob"], True)
            r = {key: e[key] for key in ("cell", "ib", "ob", "ir", "or", "digest")}
            r.update(state="reserved", bound=str(price(r, r["ib"], r["ob"])))
            r["initial_bound"] = r["bound"]
            r["initial_input_bound"] = r["ib"]
            rows[ident] = r
        else:
            r = rows[ident]
            if k == "dispatch":
                if r["state"] != "reserved":
                    raise ValueError("dispatch consumed")
                count(e["ib"])
                if e["ib"] < r["ib"]:
                    raise ValueError("shrinking wire bound")
                r["ib"] = e["ib"]
                r["bound"] = str(price(r, r["ib"], r["ob"]))
                r["state"] = "sent"
            elif k == "cancel":
                if r["state"] != "reserved" or e["proof"] != "guarded-wire-not-consumed":
                    raise ValueError("no prewire proof")
                r["state"] = "cancelled"
            elif k == "unknown":
                if r["state"] not in ("reserved", "sent"):
                    raise ValueError("already resolved")
                r["state"] = "unknown"
            elif k == "settle":
                if r["state"] != "sent":
                    raise ValueError("not sent or already resolved")
                i = count(e["input"])
                o = count(e["output"])
                if i > r["ib"] or o > r["ob"]:
                    raise SettlementBoundsError("usage exceeds admitted bounds")
                actual = price(r, i, o)
                if actual > F(r["bound"]):
                    raise SettlementBoundsError("over settlement")
                r.update(state="settled", actual=str(actual))
            else:
                raise ValueError("unknown event")
        exposure = sum(
            (
                F(r.get("actual", "0"))
                if r["state"] == "settled"
                else F(r["bound"])
                if r["state"] in ("reserved", "sent", "unknown")
                else F(0)
                for r in rows.values()
            ),
            F(0),
        )
        if exposure > self.cap:
            raise CostCapExceededError("aggregate exposure exhausted before wire")
        return rows

    def _append(self, e: dict[str, Any]) -> None:
        assert self.fd is not None and self.dirfd is not None
        e = dict(e, seq=self.seq, prev=self.previous)
        raw = canonical(e)
        try:
            view = memoryview(raw)
            while view:
                n = os.write(self.fd, view)
                if n <= 0:
                    raise OSError("short journal write")
                view = view[n:]
            os.fsync(self.fd)
            os.fsync(self.dirfd)
        except BaseException:
            self.broken = True
            raise
        self.seq += 1
        self.previous = hashlib.sha256(raw).hexdigest()

    def _event(self, e: dict[str, Any]) -> None:
        with self.lock:
            if self.broken or self.fd is None:
                raise ValueError("frozen/broken/closed journal")
            nxt = self._reduce(self.records, e)
            self._append(e)  # durability BEFORE publishing released delta
            self.records = nxt

    def reserve(
        self,
        cell: str,
        *,
        input_bound: int,
        output_bound: int,
        policy: LifecyclePricingPolicy,
    ) -> str:
        with self.lock:
            ident = f"{self.namespace}:{self.seq}"
            self._event(
                dict(
                    kind="reserve",
                    id=ident,
                    cell=cell,
                    ib=input_bound,
                    ob=output_bound,
                    ir=str(policy.input_usd_per_mtok),
                    digest=policy.digest_sha256,
                    **{"or": str(policy.output_usd_per_mtok)},
                )
            )
            return ident

    def dispatch(self, cell: str, payload: dict[str, Any]) -> None:
        with self.lock:
            candidates = [
                (i, r)
                for i, r in self.records.items()
                if r["cell"] == cell and r["state"] == "reserved"
            ]
            if len(candidates) != 1:
                raise ValueError("no one-use admission")
            ident, r = candidates[0]
            if (
                type(payload.get("max_output_tokens", payload.get("max_tokens")))
                is not int
                or payload.get("max_output_tokens", payload.get("max_tokens")) != r["ob"]
            ):
                raise ValueError("wire output changed")
            ib = max(
                r["ib"], request_input_token_bound(payload, "offline successor wire")
            )
            self._event(
                {
                    "kind": "dispatch",
                    "id": ident,
                    "ib": ib,
                    "payload_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                }
            )

    def settle(
        self, ident: str, *, input_tokens: int | None, output_tokens: int | None
    ) -> Decimal | None:
        if input_tokens is None or output_tokens is None:
            self.forfeit(ident)
            return None
        with self.lock:
            self._event(
                {
                    "kind": "settle",
                    "id": ident,
                    "input": input_tokens,
                    "output": output_tokens,
                }
            )
            return decimal(F(self.records[ident]["actual"]))

    def forfeit(self, ident: str) -> None:
        self._event({"kind": "unknown", "id": ident})

    def cancel(self, ident: str) -> None:
        self._event({"kind": "cancel", "id": ident, "proof": "guarded-wire-not-consumed"})

    def cell_accounting(self, cell: str) -> dict[str, Any]:
        """Full settled accounting, explicitly separated from ledger-3 projection."""
        with self.lock:
            rows = {i: dict(r) for i, r in self.records.items() if r["cell"] == cell}
            initial_unknown = sum(
                (F(r["initial_bound"]) for r in rows.values() if r["state"] == "unknown"),
                F(0),
            )
            full_unknown = sum(
                (F(r["bound"]) for r in rows.values() if r["state"] == "unknown"), F(0)
            )
            return {
                "controller_profile": CONTROLLER_PROFILE,
                "legacy_ledger_accounting": "initial-reservation-projection-only",
                "legacy_forfeited_usd": str(decimal(initial_unknown)),
                "wire_supplement_forfeited_usd": str(
                    decimal(full_unknown - initial_unknown)
                ),
                "full_unknown_usd": str(decimal(full_unknown)),
                "records": rows,
            }

    def enter(self, cell: str) -> None:
        if self.broken:
            raise ValueError("frozen")

    def leave(self, cell: str) -> None:
        pass

    def finish(self, cell: str) -> None:
        pass

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.dirfd is not None:
            os.close(self.dirfd)
            self.dirfd = None


class SharedGuard(LifecycleCostGuard):
    """Legacy CostGuard signature; each cell serial, fleet independent.

    Admission IDs are internal: Decimal return is not a reusable capability.
    Trusted executor owns cancellation; journal proves wire not consumed.
    """

    def __init__(
        self,
        *,
        budget: Budget,
        cell: str,
        policy: LifecyclePricingPolicy,
        max_output_tokens: int,
        token_hard_cap: int | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            cap_usd=decimal(budget.cap),
            max_output_tokens=max_output_tokens,
            token_hard_cap=token_hard_cap,
        )
        self.budget = budget
        self.cell = cell
        self.ticket: tuple[str, Decimal, int] | None = None

    def authorize(self, *, input_tokens_upper_bound: int) -> Decimal:
        if self.ticket is not None:
            raise ValueError("cell already pending")
        exposure = input_tokens_upper_bound + self._max_output_tokens
        if (
            self.token_hard_cap is not None
            and self.reserved_tokens + exposure > self.token_hard_cap
        ):
            raise CostCapExceededError("fixed cell token ceiling exhausted")
        ident = self.budget.reserve(
            self.cell,
            input_bound=input_tokens_upper_bound,
            output_bound=self._max_output_tokens,
            policy=self.policy,
        )
        amount = decimal(F(self.budget.records[ident]["bound"]))
        self.ticket = (ident, amount, input_tokens_upper_bound)
        self._outstanding = amount
        self._reserved += amount
        self._reserved_tokens += input_tokens_upper_bound + self._max_output_tokens
        return amount

    def _check(self, reservation: Decimal) -> str:
        if self.ticket is None or reservation is not self.ticket[1]:
            raise ValueError("not current one-use returned reservation")
        return self.ticket[0]

    def settle(
        self, reservation: Decimal, *, input_tokens: int | None, output_tokens: int | None
    ) -> Decimal | None:
        ident = self._check(reservation)
        try:
            actual = self.budget.settle(
                ident, input_tokens=input_tokens, output_tokens=output_tokens
            )
        except SettlementBoundsError as exc:
            # Only the explicit numeric admission refusal is classified here.
            # Internal ValueError bugs retain exposure and propagate unchanged.
            self.forfeit(reservation)
            raise CostReservationBreachedError(
                "usage exceeds admitted bounds; full wire bound retained as unknown",
                measured_usd=None,
                reservation_usd=decimal(F(self.budget.records[ident]["bound"])),
                cap_usd=self.cap_usd,
            ) from exc
        if actual is None:
            self._forfeited += decimal(F(self.budget.records[ident]["bound"]))
        else:
            self._measured += actual
        self._last = GuardEvent(
            reservation_usd=reservation,
            measured_usd=actual,
            forfeited_usd=reservation if actual is None else None,
        )
        self._pending_events.append(self._last)
        self._outstanding = Decimal(0)
        self.ticket = None
        return actual

    def forfeit(self, reservation: Decimal) -> None:
        self.settle(reservation, input_tokens=None, output_tokens=None)

    def cancel(self, reservation: Decimal, *, input_tokens_upper_bound: int) -> None:
        ident = self._check(reservation)
        assert self.ticket is not None
        if input_tokens_upper_bound != self.ticket[2]:
            raise ValueError("cancel bound mismatch")
        self.budget.cancel(ident)
        self._outstanding = Decimal(0)
        self.ticket = None
        self._reserved_tokens -= input_tokens_upper_bound + self._max_output_tokens


class FairTransport:
    """Compatibility name only: no round-robin/network-global gate."""

    def __init__(self, inner: Any, budget: Budget, cell: str) -> None:
        self.inner = inner
        self.budget = budget
        self.cell = cell

    def send(self, request: Any) -> Any:
        self.budget.enter(self.cell)
        return self.inner.send(request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
