"""Private descriptor-anchored creation of one durable bytes file."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Never


@dataclass(frozen=True)
class _WriteOnceFailure:
    """Bounded facts a caller can translate into its own domain error."""

    reason: str
    path: Path
    component: str | None = None


_ErrorAdapter = Callable[[_WriteOnceFailure], Exception]
_DirectoryIdentity = tuple[int, int]


@dataclass(frozen=True)
class _ParentChain:
    """Named directory identities plus the held final parent descriptor."""

    absolute_target: Path
    anchor: str
    parent_components: tuple[str, ...]
    identities: tuple[_DirectoryIdentity, ...]
    parentfd: int
    anchorfd: int
    relative_target: bytes


_REQUIRED_FLAGS = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_PATH")
_DIR_FD_AVAILABLE = all(
    function in os.supports_dir_fd for function in (os.open, os.mkdir, os.stat)
)
_NOFOLLOW_STAT_AVAILABLE = os.stat in os.supports_follow_symlinks
_OPENAT2_RESOLVE = 0x0E
_OPENAT2_UNSUPPORTED_ERRNOS = frozenset(
    {errno.ENOSYS, errno.EINVAL, errno.E2BIG, errno.EPERM}
)
_AUDITED_OPENAT2_SYSCALLS = {"x86_64": 437, "amd64": 437}


class _OpenHow(ctypes.Structure):
    """Linux ``struct open_how`` at its stable 24-byte v0 size."""

    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


assert ctypes.sizeof(_OpenHow) == 24


def _load_openat2() -> Callable[[int, bytes, int, int, int], int] | None:
    """Bind exported ``openat2`` or one explicitly audited syscall ABI."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except (OSError, AttributeError):
        return None
    exported = getattr(libc, "openat2", None)
    syscall_number = _AUDITED_OPENAT2_SYSCALLS.get(platform.machine().lower())
    syscall = getattr(libc, "syscall", None)
    if exported is not None:
        exported.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(_OpenHow),
            ctypes.c_size_t,
        ]
        exported.restype = ctypes.c_int
    elif syscall_number is None or syscall is None:
        return None
    else:
        syscall.restype = ctypes.c_long

    def call(dirfd: int, path: bytes, flags: int, mode: int, resolve: int) -> int:
        how = _OpenHow(flags=flags, mode=mode, resolve=resolve)
        ctypes.set_errno(0)
        if exported is not None:
            result = exported(dirfd, path, ctypes.byref(how), ctypes.sizeof(how))
        else:
            assert syscall is not None and syscall_number is not None
            result = syscall(
                ctypes.c_long(syscall_number),
                ctypes.c_int(dirfd),
                ctypes.c_char_p(path),
                ctypes.byref(how),
                ctypes.c_size_t(ctypes.sizeof(how)),
            )
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), os.fsdecode(path))
        return int(result)

    return call


_OPENAT2 = _load_openat2()


def _openat2(
    dirfd: int, path: bytes, flags: int, mode: int = 0, resolve: int = _OPENAT2_RESOLVE
) -> int:
    if _OPENAT2 is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), os.fsdecode(path))
    return _OPENAT2(dirfd, path, flags, mode, resolve)


def _probe_openat2(anchorfd: int) -> None:
    """Exercise the exact fixed resolution policy without namespace mutation."""
    descriptor = _openat2(anchorfd, b".", os.O_PATH | os.O_CLOEXEC)
    os.close(descriptor)


def _primitives_available() -> bool:
    return (
        all(hasattr(os, name) for name in _REQUIRED_FLAGS)
        and _DIR_FD_AVAILABLE
        and _NOFOLLOW_STAT_AVAILABLE
    )


def _openat2_capable() -> bool:
    """Side-effect-bounded predicate for the exact production capability."""
    if not _primitives_available() or _OPENAT2 is None:
        return False
    anchorfd: int | None = None
    try:
        anchorfd = os.open(
            os.sep, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        _probe_openat2(anchorfd)
    except OSError:
        return False
    finally:
        if anchorfd is not None:
            _close_ignoring_error(anchorfd)
    return True


def _raise(
    adapter: _ErrorAdapter,
    reason: str,
    path: Path,
    *,
    component: str | None = None,
    cause: BaseException | None = None,
) -> Never:
    error = adapter(_WriteOnceFailure(reason, path, component))
    if cause is None:
        raise error
    raise error from cause


def _require_primitives(path: Path, adapter: _ErrorAdapter) -> None:
    if not _primitives_available():
        _raise(adapter, "unsupported", path)


def _component_is_symlink(dirfd: int, component: str) -> bool:
    try:
        info = os.stat(component, dir_fd=dirfd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode)


def _identity(info: os.stat_result) -> _DirectoryIdentity:
    return info.st_dev, info.st_ino


def _close_ignoring_error(descriptor: int) -> None:
    """Relinquish one descriptor without ever retrying an indeterminate close."""
    with suppress(OSError):
        os.close(descriptor)


def _open_parent_chain(
    path: Path, absolute: Path, anchorfd: int, adapter: _ErrorAdapter
) -> _ParentChain:
    anchor = absolute.anchor or os.sep
    parent_components = absolute.parent.parts[1:]
    relative_target = os.fsencode(str(absolute.relative_to(anchor)))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        currentfd = os.dup(anchorfd)
    except OSError as exc:
        _raise(adapter, "anchor", path, cause=exc)

    try:
        identities = [_identity(os.fstat(currentfd))]
    except OSError as exc:
        _close_ignoring_error(currentfd)
        _raise(adapter, "anchor", path, cause=exc)

    for component in parent_components:
        try:
            childfd = os.open(component, directory_flags, dir_fd=currentfd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                created_directory = False
                try:
                    os.mkdir(component, 0o700, dir_fd=currentfd)
                    created_directory = True
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    _close_ignoring_error(currentfd)
                    _raise(
                        adapter,
                        "parent_create",
                        path,
                        component=component,
                        cause=mkdir_exc,
                    )
                if created_directory:
                    try:
                        os.fsync(currentfd)
                    except OSError as sync_exc:
                        _close_ignoring_error(currentfd)
                        _raise(
                            adapter,
                            "parent_create",
                            path,
                            component=component,
                            cause=sync_exc,
                        )
                try:
                    childfd = os.open(component, directory_flags, dir_fd=currentfd)
                except OSError as open_exc:
                    reason = (
                        "parent_symlink"
                        if _component_is_symlink(currentfd, component)
                        else "parent_open"
                    )
                    _close_ignoring_error(currentfd)
                    _raise(adapter, reason, path, component=component, cause=open_exc)
            else:
                reason = (
                    "parent_symlink"
                    if _component_is_symlink(currentfd, component)
                    else "parent_open"
                )
                _close_ignoring_error(currentfd)
                _raise(adapter, reason, path, component=component, cause=exc)
        try:
            child_identity = _identity(os.fstat(childfd))
        except OSError as exc:
            _close_ignoring_error(childfd)
            _close_ignoring_error(currentfd)
            _raise(adapter, "parent_open", path, component=component, cause=exc)
        identities.append(child_identity)
        previousfd = currentfd
        currentfd = childfd
        try:
            os.close(previousfd)
        except OSError as exc:
            _close_ignoring_error(currentfd)
            _raise(adapter, "parent_open", path, component=component, cause=exc)
    return _ParentChain(
        absolute,
        anchor,
        parent_components,
        tuple(identities),
        currentfd,
        anchorfd,
        relative_target,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _named_inode(dirfd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _namespace_matches(path: Path, chain: _ParentChain) -> bool:
    """Rewalk the named parent chain against recorded immutable identities."""
    try:
        absolute = Path(os.path.abspath(path))
    except OSError:
        return False
    if absolute != chain.absolute_target:
        return False
    if len(chain.identities) != len(chain.parent_components) + 1:
        return False
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        currentfd = os.dup(chain.anchorfd)
    except OSError:
        return False
    try:
        try:
            if _identity(os.fstat(currentfd)) != chain.identities[0]:
                return False
        except OSError:
            return False
        for component, expected in zip(
            chain.parent_components, chain.identities[1:], strict=True
        ):
            try:
                childfd = os.open(component, directory_flags, dir_fd=currentfd)
            except OSError:
                return False
            try:
                child_identity = _identity(os.fstat(childfd))
            except OSError:
                _close_ignoring_error(childfd)
                return False
            if child_identity != expected:
                _close_ignoring_error(childfd)
                return False
            previousfd = currentfd
            currentfd = childfd
            try:
                os.close(previousfd)
            except OSError:
                return False
        finalfd = currentfd
        currentfd = -1
        try:
            os.close(finalfd)
        except OSError:
            return False
        return True
    finally:
        if currentfd >= 0:
            _close_ignoring_error(currentfd)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        try:
            count = os.write(descriptor, view[written:])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError(errno.EIO, "write accepted no bytes")
        written += count


def _final_identity_matches(
    chain: _ParentChain,
    created: os.stat_result,
    target: Path,
    adapter: _ErrorAdapter,
) -> None:
    """Linearize final no-symlink resolution, then verify the opened object."""
    try:
        finalfd = _openat2(
            chain.anchorfd,
            chain.relative_target,
            os.O_PATH | os.O_CLOEXEC,
        )
    except OSError as exc:
        _raise(adapter, "identity", target, cause=exc)

    observed: os.stat_result | None = None
    observation_error: OSError | None = None
    try:
        observed = os.fstat(finalfd)
    except OSError as exc:
        observation_error = exc
    try:
        os.close(finalfd)
    except OSError as exc:
        if observation_error is None:
            observation_error = exc
    if observation_error is not None:
        _raise(adapter, "identity", target, cause=observation_error)
    assert observed is not None
    if not stat.S_ISREG(observed.st_mode) or not _same_inode(created, observed):
        _raise(adapter, "identity", target)


def _write_once_bytes(
    payload: bytes,
    path: str | Path,
    *,
    error_adapter: _ErrorAdapter,
) -> Path:
    """Create ``path`` once through held dirfds and make ``payload`` durable."""
    target = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("write-once payload must be bytes")
    if os.pardir in target.parts or any(
        "\x00" in component for component in target.parts
    ):
        _raise(error_adapter, "invalid", target)
    if not target.name or target.name in (os.curdir, os.pardir):
        _raise(error_adapter, "invalid", target)
    try:
        absolute_target = Path(os.path.abspath(target))
    except OSError as exc:
        _raise(error_adapter, "invalid", target, cause=exc)
    _require_primitives(target, error_adapter)

    anchorfd: int | None = None
    try:
        anchorfd = os.open(
            os.sep, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as exc:
        _raise(error_adapter, "anchor", target, cause=exc)
    try:
        _probe_openat2(anchorfd)
    except OSError as exc:
        _close_ignoring_error(anchorfd)
        if exc.errno in _OPENAT2_UNSUPPORTED_ERRNOS:
            _raise(error_adapter, "unsupported", target, cause=exc)
        _raise(error_adapter, "anchor", target, cause=exc)

    try:
        chain = _open_parent_chain(target, absolute_target, anchorfd, error_adapter)
    except BaseException:
        _close_ignoring_error(anchorfd)
        raise
    parentfd = chain.parentfd
    descriptor_to_close: int | None = None
    try:
        try:
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parentfd,
            )
            descriptor_to_close = descriptor
        except FileExistsError as exc:
            reason = (
                "final_symlink"
                if _component_is_symlink(parentfd, target.name)
                else "exists"
            )
            _raise(error_adapter, reason, target, cause=exc)
        except OSError as exc:
            reason = (
                "final_symlink"
                if _component_is_symlink(parentfd, target.name)
                else "create"
            )
            _raise(error_adapter, reason, target, cause=exc)

        created: os.stat_result | None = None
        try:
            created = os.fstat(descriptor)
            named = _named_inode(parentfd, target.name)
            if (
                not stat.S_ISREG(created.st_mode)
                or named is None
                or not stat.S_ISREG(named.st_mode)
                or not _same_inode(created, named)
            ):
                _raise(error_adapter, "identity", target)
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            named = _named_inode(parentfd, target.name)
            if named is None or not _same_inode(created, named):
                _raise(error_adapter, "identity", target)
            os.fsync(parentfd)
            descriptor_to_close = None
            os.close(descriptor)
            if not _namespace_matches(target, chain):
                _raise(error_adapter, "identity", target)
            named = _named_inode(parentfd, target.name)
            if named is None or not _same_inode(created, named):
                _raise(error_adapter, "identity", target)
            # This openat2 is the final namespace observation. It atomically
            # rejects every symlink in the frozen path and identifies the exact
            # created inode. The pathname may still mutate after linearization;
            # successful return does not promise a permanent pathname lease.
            _final_identity_matches(chain, created, target, error_adapter)
        except OSError as exc:
            _raise(error_adapter, "write", target, cause=exc)
    finally:
        if descriptor_to_close is not None:
            with suppress(OSError):
                os.close(descriptor_to_close)
        _close_ignoring_error(parentfd)
        _close_ignoring_error(chain.anchorfd)
    return target


def _write_once_bytes_at(
    payload: bytes,
    name: str,
    *,
    dir_fd: int,
    display_path: str | Path,
    error_adapter: _ErrorAdapter,
) -> Path:
    """Create one durable file relative to an already trusted directory fd.

    The caller retains ownership of ``dir_fd``.  This deliberately never reopens
    its directory by pathname: a rename can make the display path stale, but it
    cannot choose the inode that receives these bytes.
    """
    target = Path(display_path)
    if not isinstance(payload, bytes):
        raise TypeError("write-once payload must be bytes")
    if (
        not isinstance(name, str)
        or not name
        or name in (os.curdir, os.pardir)
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        _raise(error_adapter, "invalid", target)
    if not (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_CLOEXEC")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    ):
        _raise(error_adapter, "unsupported", target)

    descriptor_to_close: int | None = None
    try:
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=dir_fd,
            )
            descriptor_to_close = descriptor
        except FileExistsError as exc:
            reason = "final_symlink" if _component_is_symlink(dir_fd, name) else "exists"
            _raise(error_adapter, reason, target, cause=exc)
        except OSError as exc:
            reason = "final_symlink" if _component_is_symlink(dir_fd, name) else "create"
            _raise(error_adapter, reason, target, cause=exc)

        try:
            created = os.fstat(descriptor)
            named = _named_inode(dir_fd, name)
            if (
                not stat.S_ISREG(created.st_mode)
                or named is None
                or not stat.S_ISREG(named.st_mode)
                or not _same_inode(created, named)
            ):
                _raise(error_adapter, "identity", target)
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            named = _named_inode(dir_fd, name)
            if named is None or not _same_inode(created, named):
                _raise(error_adapter, "identity", target)
            os.fsync(dir_fd)
            descriptor_to_close = None
            os.close(descriptor)
            named = _named_inode(dir_fd, name)
            if named is None or not _same_inode(created, named):
                _raise(error_adapter, "identity", target)
        except OSError as exc:
            _raise(error_adapter, "write", target, cause=exc)
    finally:
        if descriptor_to_close is not None:
            _close_ignoring_error(descriptor_to_close)
    return target
