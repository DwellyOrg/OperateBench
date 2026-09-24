"""Small offline regressions for cross-file accounting integrity."""

import json
from decimal import Decimal

import pytest

from tools import campaign as c


def stopped_campaign(tmp_path):
    config = json.loads(c.EXAMPLE.read_text())
    root = tmp_path / "run"

    scripted = c.module("haiku45").ScriptedProvider(max_calls=50)

    def interrupt(request):
        import httpx

        body = json.loads(scripted(request).content)
        body["usage"] = None
        return httpx.Response(200, json=body)

    c.run_campaign(config, root, handlers={"offline-two-sdk-r001-haiku45": interrupt})
    assert len((root / "successor-events.jsonl").read_bytes().splitlines()) > 1
    return root


@pytest.mark.parametrize("damage", ["genesis", "last", "rewrite"])
def test_terminal_rejects_valid_accounting_prefix(tmp_path, damage):
    root = stopped_campaign(tmp_path)
    journal = root / "successor-events.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    if damage == "rewrite":
        import hashlib

        previous = None
        for i, line in enumerate(lines):
            event = json.loads(line)
            event["prev"] = previous
            if event["kind"] == "reserve":
                event["digest"] = "0" * 64
            lines[i] = c.canonical(event)
            previous = hashlib.sha256(lines[i]).hexdigest()
    else:
        lines = lines[:1] if damage == "genesis" else lines[:-1]
    journal.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="accounting checkpoint"):
        c.rebuild_report(root)


@pytest.mark.parametrize("tail", ["sent", "unknown", "genesis"])
def test_unfinished_finances_unknown_and_counters_lower_bound(tmp_path, tail):
    root = stopped_campaign(tmp_path)
    sorted((root / "status").iterdir())[-1].unlink()
    journal = root / "successor-events.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    if tail == "sent":
        end = next(
            i for i, line in enumerate(lines) if json.loads(line)["kind"] == "dispatch"
        )
        journal.write_bytes(b"".join(lines[: end + 1]))
    elif tail == "genesis":
        journal.write_bytes(lines[0])
    ledger = next(root.glob("*/execution_ledger.ndjson"))
    ledger.write_bytes(ledger.read_bytes().splitlines(keepends=True)[0])
    report = c.rebuild_report(root)
    assert report["financial_complete"] is False
    assert report["accounting"]["known"] is None
    assert report["accounting"]["exposure"] is None
    trial = report["trials"][0]
    assert trial["financial_status"] == "unknown_incomplete_start"
    assert trial["measured_usd"] is None
    assert trial["requests"] == 0
    assert trial["counter_status"] == "lower_bound"
    assert report["totals"]["incomplete_counter_trials"] == 1
    assert report["totals"]["counter_status"] == "lower_bound"
    assert report["totals"]["measured_usd"] is None
    if tail != "genesis":
        assert Decimal(report["observed_accounting"]["exposure"]) > 0
        field = "pending_exposure_usd" if tail == "sent" else "unknown_exposure_usd"
        assert Decimal(trial[field]) > 0


def test_terminal_checkpoint_accepts_later_partial_liability(tmp_path):
    import hashlib

    root = stopped_campaign(tmp_path)
    envelope = json.loads((root / "plan.json").read_text())
    second = envelope["plan"]["trials"][1]["trial_id"]
    c.append_status(root, envelope["sha256"], second, "started")
    journal = root / "successor-events.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    reserve = next(
        json.loads(line) for line in lines if json.loads(line)["kind"] == "reserve"
    )
    reserve.update(
        id="second-reservation",
        cell=second,
        seq=len(lines),
        prev=hashlib.sha256(lines[-1]).hexdigest(),
    )
    journal.write_bytes(b"".join(lines) + c.canonical(reserve))
    report = c.rebuild_report(root)
    assert report["financial_complete"] is False
    assert Decimal(report["observed_accounting"]["unknown"]) > 0
    assert Decimal(report["observed_accounting"]["pending"]) > 0
    assert Decimal(report["trials"][1]["pending_exposure_usd"]) > 0


def test_audit_opens_read_only_and_cannot_resume(tmp_path, monkeypatch):
    import os

    from tools.aggregate_budget import Budget

    root = tmp_path / "budget"
    root.mkdir(mode=0o700)
    budget = Budget(root, cap=Decimal("1"), namespace="offline-test-audit", create=True)
    budget.close()
    original = os.open
    modes = []

    def audit_open(path, flags, *args, **kwargs):
        if str(path).endswith("successor-events.jsonl"):
            modes.append(flags & os.O_ACCMODE)
            assert flags & os.O_ACCMODE == os.O_RDONLY
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", audit_open)
    budget = Budget(root, cap=Decimal("1"), namespace="offline-test-audit")
    try:
        assert budget.broken is True
        with pytest.raises(ValueError, match="frozen/broken/closed"):
            budget.forfeit("nonexistent")
    finally:
        budget.close()
    assert modes == [os.O_RDONLY]


def test_legacy_evidence_schema_is_explicitly_refused(tmp_path):
    root = stopped_campaign(tmp_path)
    path = root / "plan.json"
    envelope = json.loads(path.read_text())
    envelope["plan"]["schema"] = "operatebench.campaign.v1"
    envelope["sha256"] = c.digest(envelope["plan"])
    path.write_bytes(c.canonical(envelope))
    with pytest.raises(ValueError, match="unsupported campaign evidence schema"):
        c.rebuild_report(root)


def test_socket_tripwire_blocks_direct_connect():
    import socket

    c.network_tripwire()
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match="offline internet access forbidden"),
    ):
        sock.connect(("127.0.0.1", 9))
