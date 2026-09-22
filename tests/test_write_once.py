"""Focused durability and fail-closed tests for the private write-once primitive."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from operatebench import _write_once

pytestmark = pytest.mark.usefixtures("require_openat2")


class DomainWriteError(Exception):
    def __init__(self, failure: _write_once._WriteOnceFailure) -> None:
        super().__init__(failure.reason)
        self.failure = failure


def write_bytes(payload: bytes, path: Path) -> Path:
    return _write_once._write_once_bytes(
        payload,
        path,
        error_adapter=DomainWriteError,
    )


def test_nested_parents_are_created_and_exact_bytes_are_private(tmp_path: Path) -> None:
    target = tmp_path / "one" / "two" / "record.bin"

    assert write_bytes(b"\x00exact\xff", target) == target

    assert target.read_bytes() == b"\x00exact\xff"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "target", [Path("new/../record.bin"), Path("jump/../record.bin")]
)
def test_explicit_parent_components_fail_before_filesystem_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (tmp_path / "jump").symlink_to(attacker, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DomainWriteError, match="invalid"):
        write_bytes(b"evidence", target)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["attacker", "jump"]
    assert list(attacker.iterdir()) == []


@pytest.mark.parametrize(
    "relative_target",
    [Path("parent\x00suffix/record.bin"), Path("parent/record\x00suffix.bin")],
)
def test_nul_component_is_invalid_before_descriptor_open_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_target: Path,
    open_fd_count: Callable[[], int],
) -> None:
    real_open = os.open
    open_calls = 0

    def recording_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal open_calls
        open_calls += 1
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", recording_open)
    target = tmp_path / relative_target
    descriptors_before = open_fd_count()

    for _ in range(20):
        with pytest.raises(DomainWriteError, match="invalid"):
            write_bytes(b"evidence", target)

    assert open_calls == 0
    assert open_fd_count() == descriptors_before
    assert list(tmp_path.iterdir()) == []


def test_ordinary_relative_path_keeps_exact_bytes_mode_and_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = Path("nested/record.bin")

    assert write_bytes(b"\x00exact\xff", target) == target
    assert target.read_bytes() == b"\x00exact\xff"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_relative_target_cwd_prefix_change_after_durability_fails_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
) -> None:
    prefix = tmp_path / "start"
    anchored_parent = prefix / "deeper"
    anchored_parent.mkdir(parents=True)
    anchored_target = anchored_parent / "record.bin"
    shifted_target = prefix / "record.bin"
    target = Path("record.bin")
    parent_identity = anchored_parent.stat()
    real_fstat = os.fstat
    real_fsync = os.fsync
    cwd_changed = False

    def changing_cwd_after_parent_fsync(fd: int) -> None:
        nonlocal cwd_changed
        real_fsync(fd)
        info = real_fstat(fd)
        if not cwd_changed and (info.st_dev, info.st_ino) == (
            parent_identity.st_dev,
            parent_identity.st_ino,
        ):
            cwd_changed = True
            os.chdir(prefix)

    monkeypatch.chdir(anchored_parent)
    monkeypatch.setattr(os, "fsync", changing_cwd_after_parent_fsync)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError) as caught:
        write_bytes(b"complete", target)

    assert caught.value.failure.reason == "identity"
    assert cwd_changed
    assert open_fd_count() == descriptors_before
    assert anchored_target.read_bytes() == b"complete"
    assert not shifted_target.exists()
    monkeypatch.chdir(anchored_parent)
    with pytest.raises(DomainWriteError, match="exists"):
        write_bytes(b"replacement", target)
    assert anchored_target.read_bytes() == b"complete"


def test_revalidation_abspath_failure_fails_identity_without_fd_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
) -> None:
    target = tmp_path / "record.bin"
    real_abspath = os.path.abspath
    calls = 0

    def failing_second_abspath(path: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOENT, "simulated unavailable cwd")
        return real_abspath(path)

    monkeypatch.setattr(os.path, "abspath", failing_second_abspath)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert calls == 2
    assert open_fd_count() == descriptors_before
    assert target.read_bytes() == b"complete"


def test_deep_existing_parent_succeeds_with_four_open_descriptor_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
) -> None:
    parent = tmp_path.joinpath(*(f"level-{index}" for index in range(48)))
    parent.mkdir(parents=True)
    target = parent / "record.bin"
    real_open = os.open
    real_close = os.close
    opened: set[int] = set()
    peak_opened = 0

    def budgeted_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal peak_opened
        if len(opened) >= 4:
            raise OSError(errno.EMFILE, "simulated descriptor budget")
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.add(fd)
        peak_opened = max(peak_opened, len(opened))
        return fd

    def recording_close(fd: int) -> None:
        real_close(fd)
        opened.discard(fd)

    monkeypatch.setattr(os, "open", budgeted_open)
    monkeypatch.setattr(os, "close", recording_close)
    descriptors_before = open_fd_count()

    assert write_bytes(b"deep evidence", target) == target

    assert peak_opened == 4
    assert opened == set()
    assert open_fd_count() == descriptors_before
    assert target.read_bytes() == b"deep evidence"


@pytest.mark.parametrize("failure", ["fstat", "close"])
def test_parent_walk_identity_or_close_failure_is_bounded_and_leak_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    open_fd_count: Callable[[], int],
) -> None:
    parent = tmp_path / "existing"
    parent.mkdir()
    target = parent / "record.bin"
    real_open = os.open
    real_dup = os.dup
    real_fstat = os.fstat
    real_close = os.close
    anchor_fd: int | None = None
    walk_root_fd: int | None = None
    injected_fd: int | None = None
    injected_component = Path(os.path.abspath(target)).parent.parts[1]
    injection_calls = 0

    def recording_dup(fd: int) -> int:
        nonlocal walk_root_fd
        duplicated = real_dup(fd)
        if fd == anchor_fd and walk_root_fd is None:
            walk_root_fd = duplicated
        return duplicated

    def recording_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal anchor_fd, injected_fd
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == os.sep:
            anchor_fd = fd
        elif path == injected_component and dir_fd == walk_root_fd:
            injected_fd = fd
        return fd

    def failing_fstat(fd: int) -> os.stat_result:
        nonlocal injection_calls
        if failure == "fstat" and fd == injected_fd:
            injection_calls += 1
            raise OSError(errno.EIO, "simulated parent identity failure")
        return real_fstat(fd)

    def close_then_eio(fd: int) -> None:
        nonlocal injection_calls
        if failure == "close" and fd == walk_root_fd:
            injection_calls += 1
            real_close(fd)
            raise OSError(errno.EIO, "simulated parent close failure")
        real_close(fd)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "dup", recording_dup)
    monkeypatch.setattr(os, "fstat", failing_fstat)
    monkeypatch.setattr(os, "close", close_then_eio)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError, match="parent_open"):
        write_bytes(b"evidence", target)

    assert injection_calls == 1
    assert open_fd_count() == descriptors_before
    assert not target.exists()


def test_interrupted_and_short_writes_are_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    real_write = os.write
    calls = 0

    def interrupted_short_write(fd: int, data: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError(errno.EINTR, "interrupted")
        return real_write(fd, bytes(data)[:2])

    monkeypatch.setattr(os, "write", interrupted_short_write)

    write_bytes(b"abcdefgh", target)

    assert calls == 5
    assert target.read_bytes() == b"abcdefgh"


@pytest.mark.parametrize("failed_call", [1, 2])
def test_file_or_parent_fsync_failure_leaves_non_authoritative_write_once_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_call: int
) -> None:
    target = tmp_path / "record.bin"
    real_fsync = os.fsync
    calls = 0

    def failing_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise OSError(errno.EIO, "untrusted device prose")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"complete", target)

    monkeypatch.undo()
    assert target.read_bytes() == b"complete"
    with pytest.raises(DomainWriteError, match="exists"):
        write_bytes(b"replacement", target)


@pytest.mark.parametrize("failed_identity_call", ["fstat", "named_stat"])
def test_post_create_identity_eio_leaves_non_authoritative_write_once_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_identity_call: str,
) -> None:
    target = tmp_path / "record.bin"
    if failed_identity_call == "fstat":
        real_fstat = os.fstat
        real_open = os.open
        created_fd: int | None = None
        armed = True

        def recording_open(
            path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            nonlocal created_fd
            fd = real_open(path, flags, mode, dir_fd=dir_fd)
            if flags & os.O_CREAT:
                created_fd = fd
            return fd

        def failing_fstat(fd: int) -> os.stat_result:
            nonlocal armed
            if fd == created_fd and armed:
                armed = False
                raise OSError(errno.EIO, "untrusted identity prose")
            return real_fstat(fd)

        monkeypatch.setattr(os, "open", recording_open)
        monkeypatch.setattr(os, "fstat", failing_fstat)
    else:
        real_stat = os.stat
        armed = True

        def failing_named_stat(
            path: Any,
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            nonlocal armed
            if (
                path == target.name
                and dir_fd is not None
                and not follow_symlinks
                and armed
            ):
                armed = False
                raise OSError(errno.EIO, "untrusted identity prose")
            return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(os, "stat", failing_named_stat)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"complete", target)

    monkeypatch.undo()
    assert target.read_bytes() == b""
    with pytest.raises(DomainWriteError, match="exists"):
        write_bytes(b"replacement", target)


def test_persistent_named_identity_eio_is_translated_without_deleting_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"attacker")
    real_stat = os.stat
    failed_lookups = 0

    def persistently_failing_named_stat(
        path: Any,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal failed_lookups
        if path == target.name and dir_fd is not None and not follow_symlinks:
            if failed_lookups == 0:
                os.replace(attacker, target)
            failed_lookups += 1
            raise OSError(errno.EIO, "untrusted persistent identity prose")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", persistently_failing_named_stat)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"complete", target)

    assert failed_lookups == 1
    assert target.read_bytes() == b"attacker"


@pytest.mark.parametrize("replacement_timing", ["before", "after"])
def test_final_name_replacement_during_parent_fsync_fails_without_deleting_attacker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_timing: str,
) -> None:
    target = tmp_path / "record.bin"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"attacker")
    parent_identity = tmp_path.stat()
    real_fstat = os.fstat
    real_fsync = os.fsync
    replaced = False

    def replacing_parent_fsync(fd: int) -> None:
        nonlocal replaced
        identity = real_fstat(fd)
        is_parent = (
            identity.st_dev == parent_identity.st_dev
            and identity.st_ino == parent_identity.st_ino
        )
        if is_parent and not replaced:
            replaced = True
            if replacement_timing == "before":
                os.replace(attacker, target)
            real_fsync(fd)
            if replacement_timing == "after":
                os.replace(attacker, target)
            return
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", replacing_parent_fsync)

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert replaced
    assert target.read_bytes() == b"attacker"


def test_final_name_replacement_after_namespace_rewalk_fails_without_deleting_attacker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"attacker")
    real_namespace_matches = _write_once._namespace_matches
    replaced = False

    def replacing_after_rewalk(path: Path, descriptors: list[int]) -> bool:
        nonlocal replaced
        matches = real_namespace_matches(path, descriptors)
        os.replace(attacker, target)
        replaced = True
        return matches

    monkeypatch.setattr(_write_once, "_namespace_matches", replacing_after_rewalk)

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert replaced
    assert target.read_bytes() == b"attacker"


def test_ancestor_replacement_after_rewalk_closes_it_fails_at_final_openat2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
) -> None:
    output = tmp_path / "output"
    parent = output / "deep"
    parent.mkdir(parents=True)
    target = parent / "record.bin"
    held_output = tmp_path / "output-held"
    attacker = tmp_path / "attacker"
    attacker_parent = attacker / "deep"
    attacker_parent.mkdir(parents=True)
    attacker_target = attacker_parent / target.name
    attacker_target.write_bytes(b"attacker")
    output_identity = output.stat()
    real_close = os.close
    real_fstat = os.fstat
    output_closes = 0
    replaced = False

    def replacing_after_validated_output_close(fd: int) -> None:
        nonlocal output_closes, replaced
        info = real_fstat(fd)
        is_output = (info.st_dev, info.st_ino) == (
            output_identity.st_dev,
            output_identity.st_ino,
        )
        real_close(fd)
        if is_output:
            output_closes += 1
            if output_closes == 2:
                output.rename(held_output)
                output.symlink_to(attacker, target_is_directory=True)
                replaced = True

    monkeypatch.setattr(os, "close", replacing_after_validated_output_close)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert replaced
    assert output_closes == 2
    assert open_fd_count() == descriptors_before
    assert target.read_bytes() == b"attacker"
    assert (held_output / "deep" / target.name).read_bytes() == b"complete"


def test_symlink_back_after_rewalk_never_returns_success_for_requested_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
) -> None:
    output = tmp_path / "output"
    parent = output / "deep"
    parent.mkdir(parents=True)
    target = parent / "record.bin"
    held_output = tmp_path / "output-held"
    output_identity = output.stat()
    real_close = os.close
    real_fstat = os.fstat
    output_closes = 0

    def replacing_with_symlink_back(fd: int) -> None:
        nonlocal output_closes
        info = real_fstat(fd)
        is_output = (info.st_dev, info.st_ino) == (
            output_identity.st_dev,
            output_identity.st_ino,
        )
        real_close(fd)
        if is_output:
            output_closes += 1
            if output_closes == 2:
                output.rename(held_output)
                output.symlink_to(held_output, target_is_directory=True)

    monkeypatch.setattr(os, "close", replacing_with_symlink_back)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert output_closes == 2
    assert open_fd_count() == descriptors_before
    assert target.read_bytes() == b"complete"
    assert (held_output / "deep" / target.name).read_bytes() == b"complete"


@pytest.mark.parametrize(
    "replacement_kind", ["unrelated_ancestor", "final_symlink", "magic_symlink"]
)
def test_final_openat2_rejects_symlinked_frozen_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    target = output / "record.bin"
    held_output = tmp_path / "output-held"
    held_file = tmp_path / "held-record.bin"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    unrelated_target = unrelated / target.name
    unrelated_target.write_bytes(b"attacker")
    real_namespace_matches = _write_once._namespace_matches
    real_open = os.open
    real_close = os.close
    magic_fd: int | None = None

    def replacing_after_rewalk(path: Path, chain: _write_once._ParentChain) -> bool:
        nonlocal magic_fd
        matches = real_namespace_matches(path, chain)
        if replacement_kind == "unrelated_ancestor":
            output.rename(held_output)
            output.symlink_to(unrelated, target_is_directory=True)
        else:
            os.rename(target, held_file)
            if replacement_kind == "final_symlink":
                target.symlink_to(unrelated_target)
            else:
                magic_fd = real_open(unrelated_target, os.O_PATH | os.O_CLOEXEC)
                target.symlink_to(f"/proc/self/fd/{magic_fd}")
        return matches

    monkeypatch.setattr(_write_once, "_namespace_matches", replacing_after_rewalk)
    try:
        with pytest.raises(DomainWriteError, match="identity"):
            write_bytes(b"complete", target)
    finally:
        if magic_fd is not None:
            real_close(magic_fd)

    if replacement_kind == "magic_symlink":
        assert target.is_symlink()
        assert unrelated_target.read_bytes() == b"attacker"
    else:
        assert target.read_bytes() == b"attacker"


@pytest.mark.parametrize(
    "error_number", [errno.ENOSYS, errno.EINVAL, errno.E2BIG, errno.EPERM]
)
def test_unavailable_openat2_policy_fails_before_mutation_without_fd_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
    error_number: int,
) -> None:
    target = tmp_path / "new" / "record.bin"

    def unavailable(*args: Any, **kwargs: Any) -> int:
        raise OSError(error_number, "simulated unavailable openat2 policy")

    monkeypatch.setattr(_write_once, "_openat2", unavailable)
    descriptors_before = open_fd_count()

    for _ in range(20):
        with pytest.raises(DomainWriteError, match="unsupported"):
            write_bytes(b"evidence", target)

    assert open_fd_count() == descriptors_before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["syscall", "fstat", "mismatch", "close"])
def test_final_openat2_descriptor_failures_are_identity_and_owned_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_fd_count: Callable[[], int],
    failure: str,
) -> None:
    target = tmp_path / "record.bin"
    other = tmp_path / "other.bin"
    other.write_bytes(b"other")
    real_openat2 = _write_once._openat2
    real_fstat = os.fstat
    real_close = os.close
    finalfd: int | None = None
    final_close_calls = 0

    def controlled_openat2(
        dirfd: int, path: bytes, flags: int, mode: int = 0, resolve: int = 0x0E
    ) -> int:
        nonlocal finalfd
        if path == b".":
            return real_openat2(dirfd, path, flags, mode, resolve)
        if failure == "syscall":
            raise OSError(errno.EIO, "simulated final resolver failure")
        if failure == "mismatch":
            finalfd = os.open(other, os.O_PATH | os.O_CLOEXEC)
        else:
            finalfd = real_openat2(dirfd, path, flags, mode, resolve)
        return finalfd

    def controlled_fstat(fd: int) -> os.stat_result:
        if failure == "fstat" and fd == finalfd:
            raise OSError(errno.EIO, "simulated final fstat failure")
        return real_fstat(fd)

    def controlled_close(fd: int) -> None:
        nonlocal final_close_calls
        if fd == finalfd:
            final_close_calls += 1
            real_close(fd)
            if failure == "close":
                raise OSError(errno.EIO, "simulated final close failure")
            return
        real_close(fd)

    monkeypatch.setattr(_write_once, "_openat2", controlled_openat2)
    monkeypatch.setattr(os, "fstat", controlled_fstat)
    monkeypatch.setattr(os, "close", controlled_close)
    descriptors_before = open_fd_count()

    with pytest.raises(DomainWriteError, match="identity"):
        write_bytes(b"complete", target)

    assert final_close_calls == (0 if failure == "syscall" else 1)
    assert open_fd_count() == descriptors_before


def test_repeated_success_keeps_zero_descriptor_delta(
    tmp_path: Path, open_fd_count: Callable[[], int]
) -> None:
    descriptors_before = open_fd_count()

    for index in range(100):
        write_bytes(b"complete", tmp_path / f"record-{index}.bin")

    assert open_fd_count() == descriptors_before


def test_namespace_mutation_after_final_openat2_is_allowed_post_linearization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"attacker")
    held = tmp_path / "held.bin"
    real_openat2 = _write_once._openat2

    def mutate_after_openat2(
        dirfd: int, path: bytes, flags: int, mode: int = 0, resolve: int = 0x0E
    ) -> int:
        descriptor = real_openat2(dirfd, path, flags, mode, resolve)
        if path != b".":
            target.rename(held)
            attacker.rename(target)
        return descriptor

    monkeypatch.setattr(_write_once, "_openat2", mutate_after_openat2)

    assert write_bytes(b"complete", target) == target
    assert held.read_bytes() == b"complete"
    assert target.read_bytes() == b"attacker"


def test_each_created_entry_and_final_file_and_parent_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "one" / "two" / "record.bin"
    real_fstat = os.fstat
    real_fsync = os.fsync
    real_mkdir = os.mkdir
    mkdir_parent_identities: list[tuple[int, int]] = []
    fsynced_identities: list[tuple[int, int]] = []

    def identity(fd: int) -> tuple[int, int]:
        info = real_fstat(fd)
        return info.st_dev, info.st_ino

    def recording_mkdir(
        path: Any, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        assert dir_fd is not None
        mkdir_parent_identities.append(identity(dir_fd))
        real_mkdir(path, mode, dir_fd=dir_fd)

    def recording_fsync(fd: int) -> None:
        fsynced_identities.append(identity(fd))
        real_fsync(fd)

    monkeypatch.setattr(os, "mkdir", recording_mkdir)
    monkeypatch.setattr(os, "fsync", recording_fsync)

    write_bytes(b"complete", target)

    assert all(item in fsynced_identities for item in mkdir_parent_identities)
    final_identity = target.stat()
    final_parent_identity = target.parent.stat()
    assert (final_identity.st_dev, final_identity.st_ino) in fsynced_identities
    assert (
        final_parent_identity.st_dev,
        final_parent_identity.st_ino,
    ) in fsynced_identities


def test_created_directory_parent_fsync_failure_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "one" / "two" / "record.bin"
    root_identity = tmp_path.stat()
    real_fstat = os.fstat
    real_fsync = os.fsync

    def failing_created_entry_fsync(fd: int) -> None:
        info = real_fstat(fd)
        if (info.st_dev, info.st_ino) == (root_identity.st_dev, root_identity.st_ino):
            raise OSError(errno.EIO, "untrusted directory prose")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_created_entry_fsync)

    with pytest.raises(DomainWriteError, match="parent_create"):
        write_bytes(b"complete", target)

    assert not target.exists()


def test_failed_write_removes_only_its_inode_and_not_an_attacker_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    created = tmp_path / "created.bin"
    real_open = os.open
    real_write = os.write
    armed = True

    def replacing_failed_write(fd: int, data: Any) -> int:
        nonlocal armed
        if armed:
            armed = False
            real_write(fd, bytes(data)[:1])
            os.rename(target, created)
            attacker_fd = real_open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                real_write(attacker_fd, b"attacker")
            finally:
                os.close(attacker_fd)
            raise OSError(errno.ENOSPC, "untrusted device prose")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", replacing_failed_write)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"evidence", target)

    assert target.read_bytes() == b"attacker"
    assert created.read_bytes() == b"e"


def test_success_path_close_eio_is_write_failure_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    real_open = os.open
    real_close = os.close
    created_fd: int | None = None
    created_fd_close_calls = 0

    def recording_open(
        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal created_fd
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT:
            created_fd = fd
        return fd

    def close_then_eio(fd: int) -> None:
        nonlocal created_fd_close_calls
        if fd == created_fd:
            created_fd_close_calls += 1
            if created_fd_close_calls == 1:
                real_close(fd)
                raise OSError(errno.EIO, "untrusted close prose")
        real_close(fd)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "close", close_then_eio)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"complete", target)

    assert created_fd is not None
    assert created_fd_close_calls == 1
    monkeypatch.undo()
    assert target.read_bytes() == b"complete"
    with pytest.raises(DomainWriteError, match="exists"):
        write_bytes(b"replacement", target)


def test_replacement_after_cleanup_identity_observation_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.bin"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"attacker")
    real_stat = os.stat
    real_close = os.close
    named_lookups = 0
    replacement_observed = False

    def replacing_after_identity_observation(
        path: Any,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal named_lookups, replacement_observed
        info = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == target.name and dir_fd is not None and not follow_symlinks:
            named_lookups += 1
            if named_lookups == 2:
                os.replace(attacker, target)
                replacement_observed = True
        return info

    def failing_write(fd: int, data: Any) -> int:
        raise OSError(errno.ENOSPC, "untrusted device prose")

    def replacing_on_close_if_cleanup_was_not_attempted(fd: int) -> None:
        nonlocal replacement_observed
        if not replacement_observed and target.exists() and attacker.exists():
            os.replace(attacker, target)
            replacement_observed = True
        real_close(fd)

    monkeypatch.setattr(os, "stat", replacing_after_identity_observation)
    monkeypatch.setattr(os, "write", failing_write)
    monkeypatch.setattr(os, "close", replacing_on_close_if_cleanup_was_not_attempted)

    with pytest.raises(DomainWriteError, match="write"):
        write_bytes(b"evidence", target)

    assert replacement_observed
    assert target.read_bytes() == b"attacker"
