"""Where the bytes land, and what happens when they do not land at all.

Two failures are covered here, and they are the two that make a crash-evidence
journal worth having or worthless.

**The location.** Validating a directory by path and then opening a file by path
leaves a window between the check and the use. Anything that can rename that
directory in the window chooses where the evidence goes, and the check reports
success about a directory the bytes never reached. So every component is opened
from the root, refusing symbolic links as it goes, and the final file is created
against the *descriptor* of the directory that was validated — the same inode,
whatever its name says afterwards.

**The write.** ``os.write`` may write fewer bytes than it was given, and
``fsync`` may fail after it. Treating either as success leaves a partial row that
the next row is appended onto, and a reader then meets a line that is two halves
of two rows. Worse, a terminal row whose ``fsync`` failed can still be visible,
and a visible terminal row says ``scored``. So a row is written in full or rolled
back to the offset before it, the writer is poisoned permanently on any failure,
and nothing — including the runner's own finaliser — appends to it afterwards.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest

from operatebench.execution_ledger import (
    TERMINAL_SCORED,
    ExecutionLedgerJournal,
    LedgerPathError,
    LedgerStateError,
    LedgerWriteError,
    read_execution_ledger,
)
from tests.execution_ledger_fixtures import (
    LATER,
    call,
    header,
)


def _is_whole_row(payload: bytes) -> bool:
    """A complete ledger row rather than the remainder of a partial write."""
    return (
        payload.startswith(b'{"')
        and payload.endswith(b"\n")
        and b'"ledger_version"' in payload
    )


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


# -- C: the directory a row lands in ------------------------------------------


def test_an_ancestor_swapped_for_a_symlink_at_the_seam_cannot_redirect_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check/use race, run as an attacker would: rename, then symlink.

    The swap happens inside ``os.open`` itself — that is, after every check a
    path-based implementation could have made and before the file is created.
    Either the ledger is refused or it lands in the directory that was
    validated. It never lands in the attacker's.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    moved = tmp_path / "evidence-as-validated"
    real_open = os.open
    state = {"swapped": False}

    def attacking_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if flags & os.O_CREAT and not state["swapped"]:
            state["swapped"] = True
            os.rename(evidence, moved)
            os.symlink(attacker, evidence)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", attacking_open)
    try:
        journal = ExecutionLedgerJournal.open(evidence / "run.ndjson", header())
    except LedgerPathError:
        pass
    else:
        journal.close()
    monkeypatch.undo()
    assert state["swapped"]
    assert not (attacker / "run.ndjson").exists()


def test_an_anchor_open_failure_is_reported_as_a_ledger_path_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "evidence"
    path = directory / "run.ndjson"
    real_open = os.open
    real_close = os.close
    anchor = Path(os.path.abspath(directory)).anchor or os.sep
    failure = PermissionError(errno.EACCES, "untrusted kernel prose")
    closed: list[int] = []

    def failing_open(
        opened_path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if opened_path == anchor and dir_fd is None:
            raise failure
        return real_open(opened_path, flags, mode, dir_fd=dir_fd)

    def recording_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "open", failing_open)
    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(LedgerPathError) as raised:
        ExecutionLedgerJournal.open(path, header())

    assert raised.value.__cause__ is failure
    assert "untrusted kernel prose" not in str(raised.value)
    assert closed == []
    assert not path.exists()


def test_a_ledger_directory_reached_through_a_symlink_is_refused(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(link / "run.ndjson", header())


def test_a_ledger_directory_whose_ancestor_is_a_symlink_is_refused(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    (real / "inner").mkdir(mode=0o700, parents=True)
    os.chmod(real, 0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(link / "inner" / "run.ndjson", header())


def test_a_ledger_name_that_is_a_symlink_is_refused(ledger_dir: Path) -> None:
    elsewhere = ledger_dir.parent / "elsewhere.ndjson"
    elsewhere.write_text("")
    (ledger_dir / "run.ndjson").symlink_to(elsewhere)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(ledger_dir / "run.ndjson", header())


def test_a_ledger_name_that_is_a_dangling_symlink_is_refused(ledger_dir: Path) -> None:
    (ledger_dir / "run.ndjson").symlink_to(ledger_dir / "nothing-here.ndjson")
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(ledger_dir / "run.ndjson", header())


def test_a_ledger_directory_that_is_not_private_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    with pytest.raises(LedgerPathError):
        ExecutionLedgerJournal.open(directory / "run.ndjson", header())


def test_the_ledger_is_created_private_in_the_directory_that_was_validated(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    with ExecutionLedgerJournal.open(path, header()) as journal:
        journal.record_call(call(0))
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_execution_ledger(path, require_complete=True).scored


# -- D: a row is written in full, or not at all -------------------------------


class RowFailure:
    """Break the durable write of the Nth row written while this is installed.

    Rows are recognised by their first field rather than by a file descriptor
    this test would have to reach into the journal for, so the interception
    stays outside the writer's own state. Counting starts at installation, so
    ``row=1`` is the next row the journal writes.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, row: int, mode: str) -> None:
        self._row = row
        self._mode = mode
        self._seen = 0
        self._armed = False
        self._partial = False
        self.interrupted = 0
        self._write = os.write
        self._fsync = os.fsync
        self._ftruncate = os.ftruncate
        monkeypatch.setattr(os, "write", self.write)
        monkeypatch.setattr(os, "fsync", self.fsync)
        monkeypatch.setattr(os, "ftruncate", self.ftruncate)

    def write(self, fd: int, data: Any) -> int:
        payload = bytes(data)
        if _is_whole_row(payload):
            self._seen += 1
            if self._seen == self._row:
                if self._mode in ("short", "truncated"):
                    self._partial = True
                    return self._write(fd, payload[: len(payload) // 2])
                if self._mode == "zero":
                    return 0
                if self._mode == "error":
                    raise OSError(28, "no space left on device")
                if self._mode == "interrupt":
                    self.interrupted += 1
                    raise InterruptedError(4, "interrupted system call")
                self._armed = True
        elif self._partial and self._mode == "truncated":
            # The remainder of a row this device has stopped accepting.
            raise OSError(28, "no space left on device")
        return self._write(fd, payload)

    def fsync(self, fd: int) -> None:
        if self._armed and self._mode in ("fsync", "rollback"):
            self._armed = False
            raise OSError(5, "input/output error")
        self._fsync(fd)

    def ftruncate(self, fd: int, length: int) -> None:
        if self._mode == "rollback":
            raise OSError(5, "input/output error")
        self._ftruncate(fd, length)


def _header_only_bytes(directory: Path) -> int:
    path = directory / "measure.ndjson"
    with ExecutionLedgerJournal.open(path, header()):
        pass
    size = path.stat().st_size
    path.unlink()
    return size


def test_a_short_write_is_completed_rather_than_treated_as_a_written_row(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's first finding: half a row was reported as a whole one."""
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    RowFailure(monkeypatch, row=1, mode="short")
    journal.record_call(call(0))
    journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    monkeypatch.undo()
    assert not journal.poisoned
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.scored
    assert audit.totals.provider_calls == 1
    assert len(path.read_text().splitlines()) == 3


@pytest.mark.parametrize("mode", ["truncated", "zero", "error"])
def test_a_call_row_that_could_not_be_written_in_full_is_rolled_back(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    expected = _header_only_bytes(ledger_dir)
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    RowFailure(monkeypatch, row=1, mode=mode)
    with pytest.raises(LedgerWriteError):
        journal.record_call(call(0))
    monkeypatch.undo()
    assert path.stat().st_size == expected
    assert journal.poisoned
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert not audit.scored
    assert audit.calls == ()


def test_a_call_row_whose_fsync_failed_poisons_the_writer(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _header_only_bytes(ledger_dir)
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    RowFailure(monkeypatch, row=1, mode="fsync")
    with pytest.raises(LedgerWriteError):
        journal.record_call(call(0))
    monkeypatch.undo()
    assert journal.poisoned
    assert path.stat().st_size == expected
    with pytest.raises(LedgerStateError):
        journal.record_call(call(0))
    with pytest.raises(LedgerStateError):
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert not audit.scored


def test_a_terminal_row_whose_fsync_failed_never_reads_as_scored(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the reviewer found: the write raised and the ledger read scored."""
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    journal.record_call(call(0))
    size_before = path.stat().st_size
    RowFailure(monkeypatch, row=1, mode="fsync")
    with pytest.raises(LedgerWriteError):
        journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    monkeypatch.undo()
    assert journal.poisoned
    assert not journal.finalized
    assert path.stat().st_size == size_before
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert not audit.scored
    assert audit.totals.provider_calls == 1


def test_a_rollback_that_itself_failed_is_named_rather_than_hidden(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    RowFailure(monkeypatch, row=1, mode="rollback")
    with pytest.raises(LedgerWriteError) as raised:
        journal.record_call(call(0))
    monkeypatch.undo()
    assert not raised.value.rolled_back
    assert journal.poisoned
    audit = read_execution_ledger(path)
    assert not audit.scored
    assert audit.status == "incomplete"


def test_an_interrupted_write_is_retried_rather_than_lost(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    failure = RowFailure(monkeypatch, row=1, mode="interrupt")
    journal.record_call(call(0))
    journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    monkeypatch.undo()
    assert failure.interrupted == 1
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.scored
    assert audit.totals.provider_calls == 1


def test_a_finaliser_does_not_append_to_a_poisoned_journal(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the runner does on the way out of a failed run, against a dead writer.

    The finaliser runs while an exception is unwinding. It must not append —
    the writer's last row was rolled back and anything after it would sit on an
    offset nothing verified — and it must not raise, because the failure the run
    is already carrying is the one worth reporting.
    """
    from operatebench.agents.evidence import (
        EvidenceRunIdentity,
        ProviderEvidenceRecorder,
    )
    from tests.execution_ledger_fixtures import controls, provider_identity

    path = ledger_dir / "run.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256="a" * 64,
            scenario_id="V1",
            agent_id="openai-gpt-5-6-luna",
        ),
        provider=provider_identity(),
        controls=controls(),
    )
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    size_before = path.stat().st_size
    RowFailure(monkeypatch, row=1, mode="fsync")
    with pytest.raises(LedgerWriteError):
        recorder.journal.record_call(call(0))
    monkeypatch.undo()
    recorder.finalize_aborted()
    recorder.finalize_scored()
    assert path.stat().st_size == size_before
    audit = read_execution_ledger(path)
    assert audit.status == "incomplete"
    assert not audit.scored


def test_a_poisoned_journal_refuses_every_later_append(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ledger_dir / "run.ndjson"
    journal = ExecutionLedgerJournal.open(path, header())
    RowFailure(monkeypatch, row=1, mode="error")
    with pytest.raises(LedgerWriteError):
        journal.record_call(call(0))
    monkeypatch.undo()
    for attempt in range(2):
        with pytest.raises(LedgerStateError):
            journal.record_call(call(attempt))
        with pytest.raises(LedgerStateError):
            journal.finalize(TERMINAL_SCORED, ended_at=LATER)
    journal.close()
    assert journal.poisoned
