"""Portability checks for the shared open-descriptor counter."""

from __future__ import annotations

import ctypes
import os
import platform
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from operatebench import _write_once


class DomainWriteError(Exception):
    def __init__(self, failure: _write_once._WriteOnceFailure) -> None:
        super().__init__(failure.reason)


def test_open_fd_count_falls_back_to_dev_fd_when_procfs_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    calls: list[str] = []

    def listdir(path: str) -> list[str]:
        calls.append(path)
        if path == "/proc/self/fd":
            raise FileNotFoundError(path)
        assert path == "/dev/fd"
        return ["0", "1", "2", "3"]

    monkeypatch.setattr(os, "listdir", listdir)

    current_open_fd_count = request.getfixturevalue("open_fd_count")

    assert isinstance(current_open_fd_count, Callable)
    assert current_open_fd_count() == 4
    assert calls == ["/proc/self/fd", "/dev/fd", "/dev/fd"]


def test_unsupported_openat2_contract_runs_without_platform_success_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-closed contract remains runnable on unsupported platforms."""
    target = tmp_path / "record.bin"
    monkeypatch.setattr(_write_once, "_OPENAT2", None)

    with pytest.raises(DomainWriteError, match="unsupported"):
        _write_once._write_once_bytes(b"evidence", target, error_adapter=DomainWriteError)

    assert not target.exists()


@pytest.mark.parametrize(
    ("missing_prerequisite", "available_prerequisite"),
    [
        ("_DIR_FD_AVAILABLE", "_NOFOLLOW_STAT_AVAILABLE"),
        ("_NOFOLLOW_STAT_AVAILABLE", "_DIR_FD_AVAILABLE"),
    ],
)
def test_capability_rejects_missing_production_prerequisite_without_opening_fd(
    monkeypatch: pytest.MonkeyPatch,
    missing_prerequisite: str,
    available_prerequisite: str,
) -> None:
    monkeypatch.setattr(_write_once, missing_prerequisite, False)
    monkeypatch.setattr(_write_once, available_prerequisite, True)
    monkeypatch.setattr(_write_once, "_OPENAT2", lambda *args: 0)

    def unexpected_open(*args: object, **kwargs: object) -> int:
        raise AssertionError("descriptor open was not expected")

    monkeypatch.setattr(_write_once.os, "open", unexpected_open)

    assert not _write_once._openat2_capable()


@pytest.mark.parametrize(
    ("missing_prerequisite", "available_prerequisite"),
    [
        ("_DIR_FD_AVAILABLE", "_NOFOLLOW_STAT_AVAILABLE"),
        ("_NOFOLLOW_STAT_AVAILABLE", "_DIR_FD_AVAILABLE"),
    ],
)
def test_success_fixture_skips_when_production_prerequisite_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    missing_prerequisite: str,
    available_prerequisite: str,
) -> None:
    monkeypatch.setattr(_write_once, missing_prerequisite, False)
    monkeypatch.setattr(_write_once, available_prerequisite, True)
    monkeypatch.setattr(_write_once, "_OPENAT2", lambda *args: 0)

    def unexpected_open(*args: object, **kwargs: object) -> int:
        raise AssertionError("descriptor open was not expected")

    monkeypatch.setattr(_write_once.os, "open", unexpected_open)

    with pytest.raises(pytest.skip.Exception):
        request.getfixturevalue("require_openat2")


@pytest.mark.parametrize(
    ("missing_prerequisite", "available_prerequisite"),
    [
        ("_DIR_FD_AVAILABLE", "_NOFOLLOW_STAT_AVAILABLE"),
        ("_NOFOLLOW_STAT_AVAILABLE", "_DIR_FD_AVAILABLE"),
    ],
)
def test_missing_production_prerequisite_fails_closed_without_open_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_prerequisite: str,
    available_prerequisite: str,
) -> None:
    target = tmp_path / "record.bin"
    monkeypatch.setattr(_write_once, missing_prerequisite, False)
    monkeypatch.setattr(_write_once, available_prerequisite, True)

    def unexpected_open(*args: object, **kwargs: object) -> int:
        raise AssertionError("descriptor open was not expected")

    monkeypatch.setattr(_write_once.os, "open", unexpected_open)

    with pytest.raises(DomainWriteError, match="unsupported"):
        _write_once._write_once_bytes(b"evidence", target, error_adapter=DomainWriteError)

    assert not target.exists()


def test_success_fixture_skips_when_exact_production_capability_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(_write_once, "_OPENAT2", None)

    with pytest.raises(pytest.skip.Exception):
        request.getfixturevalue("require_openat2")


def test_x86_64_syscall_fallback_uses_audited_real_openat2_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-symbol fallback reaches the real kernel with the audited ABI."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("the audited syscall regression executes only on x86-64")

    assert ctypes.sizeof(_write_once._OpenHow) == 24
    assert _write_once._OpenHow._fields_ == [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]
    assert (
        _write_once._OpenHow.flags.offset,
        _write_once._OpenHow.mode.offset,
        _write_once._OpenHow.resolve.offset,
    ) == (0, 8, 16)

    real_libc = ctypes.CDLL(None, use_errno=True)
    real_syscall = real_libc.syscall
    original_argtypes = real_syscall.argtypes
    original_restype = real_syscall.restype
    observed: list[tuple[int, int, int, int]] = []

    class IsolatedVariadicSyscall:
        """Keep loader metadata assignments off the process-global binding."""

        restype: object = None

        def __call__(self, *arguments: object) -> int:
            number = arguments[0]
            size = arguments[4]
            assert isinstance(number, ctypes.c_long)
            assert isinstance(size, ctypes.c_size_t)
            how = ctypes.cast(arguments[3], ctypes.POINTER(_write_once._OpenHow)).contents
            observed.append((number.value, how.flags, how.resolve, size.value))
            return int(real_syscall(*arguments))

    class SyscallOnlyLibc:
        syscall = IsolatedVariadicSyscall()

    monkeypatch.setattr(
        _write_once.ctypes, "CDLL", lambda *args, **kwargs: SyscallOnlyLibc()
    )
    monkeypatch.setattr(_write_once.platform, "machine", lambda: "x86_64")

    fallback = _write_once._load_openat2()
    assert fallback is not None
    rootfd = os.open(os.sep, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    opened: int | None = None
    try:
        opened = fallback(
            rootfd,
            b".",
            os.O_PATH | os.O_CLOEXEC,
            0,
            _write_once._OPENAT2_RESOLVE,
        )
        root_info = os.fstat(rootfd)
        opened_info = os.fstat(opened)
        assert stat.S_ISDIR(opened_info.st_mode)
        assert (opened_info.st_dev, opened_info.st_ino) == (
            root_info.st_dev,
            root_info.st_ino,
        )
        assert not os.get_inheritable(opened)
    finally:
        if opened is not None:
            os.close(opened)
        os.close(rootfd)

    assert observed == [(437, os.O_PATH | os.O_CLOEXEC, 0x0E, 24)]
    assert _write_once._AUDITED_OPENAT2_SYSCALLS == {"x86_64": 437, "amd64": 437}
    assert real_syscall.argtypes is original_argtypes
    assert real_syscall.restype is original_restype
