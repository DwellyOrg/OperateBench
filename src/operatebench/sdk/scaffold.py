"""Deterministically create one unregistered operation-pack starter directory."""

from __future__ import annotations

import json
import keyword
import os
import re
import stat
from contextlib import suppress
from importlib.resources import files
from pathlib import Path

from operatebench.sdk._validation import is_safe_owner_display_name
from operatebench.sdk.errors import OperationScaffoldError

_TEMPLATE_PACKAGE = "operatebench.resources.operation_pack_template"
_TEMPLATE_NAMES: tuple[str, ...] = (
    "README.md.tmpl",
    "__init__.py.tmpl",
    "agents.py.tmpl",
    "evaluator.py.tmpl",
    "negative_control_oracle.yaml.tmpl",
    "operation.py.tmpl",
    "operation.yaml.tmpl",
    "pack.py.tmpl",
    "spec.py.tmpl",
    "test_pack.py.tmpl",
)
_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)


def _validate_id(value: str, where: str) -> None:
    if type(value) is not str or not _ID_RE.fullmatch(value) or len(value) > 128:
        raise OperationScaffoldError(
            f"{where} must be 1-128 lowercase ASCII letters/digits separated by "
            "'.', '_' or '-', starting with a letter"
        )


def _module_name(pack_id: str) -> str:
    parts = pack_id.split(".")
    candidate = parts[-2] if len(parts) > 1 and parts[-1] == "synthetic" else parts[-1]
    candidate = candidate.replace("-", "_")
    if (
        not candidate.isascii()
        or not candidate.isidentifier()
        or keyword.iskeyword(candidate)
        or candidate.casefold() in _WINDOWS_RESERVED
        or len(candidate) > 63
    ):
        raise OperationScaffoldError(
            f"pack id {pack_id!r} derives unsafe Python module name {candidate!r}; "
            "use a final operation segment that is a lowercase ASCII identifier"
        )
    return candidate


def _validate_destination_name(destination: Path) -> None:
    name = destination.name
    if (
        not name.isascii()
        or not name.isidentifier()
        or keyword.iskeyword(name)
        or name.casefold() in _WINDOWS_RESERVED
        or len(name) > 63
    ):
        raise OperationScaffoldError(
            f"destination directory name {name!r} is not a safe Python package "
            "name; use a lowercase identifier such as 'return_refund'"
        )


def _refuse_symlink_ancestors(destination: Path) -> None:
    absolute = destination if destination.is_absolute() else Path.cwd() / destination
    for ancestor in absolute.parents:
        if ancestor.is_symlink():
            raise OperationScaffoldError(
                f"{ancestor} is a symlink on the scaffold path; generated code is "
                "created only through real directories"
            )


def _render(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{{" + name + "}}", value)
    if "{{" in rendered or "}}" in rendered:
        raise OperationScaffoldError("an operation scaffold template is incomplete")
    return rendered


_SECURE_DIRECTORY_FDS_SUPPORTED = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "fchmod")
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    )
    and os.stat in os.supports_follow_symlinks
)


def _supports_secure_directory_fds() -> bool:
    return _SECURE_DIRECTORY_FDS_SUPPORTED


def _identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_real_directory(path: Path) -> int:
    """Open every absolute path component without following a symlink."""
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    descriptor: int | None = None
    try:
        descriptor = os.open(parts[0], _directory_flags())
        for component in parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        raise OperationScaffoldError(
            f"parent directory {path} must exist and resolve entirely through "
            "real directories; symlinks and path replacement are refused"
        ) from exc
    if descriptor is None:  # pragma: no cover - absolute paths always have a root
        raise AssertionError("an absolute directory path has no root component")
    return descriptor


def _entry_stat(directory_fd: int, name: str, context: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise OperationScaffoldError(
            f"{context} changed while the scaffold was being created; path "
            "replacement is refused"
        ) from exc


def _verify_created_files(
    target_fd: int,
    created_files: dict[str, tuple[int, int]],
) -> None:
    for filename, file_identity in created_files.items():
        file_stat = _entry_stat(
            target_fd,
            filename,
            f"generated file {filename!r}",
        )
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or (file_stat.st_dev, file_stat.st_ino) != file_identity
        ):
            raise OperationScaffoldError(
                f"generated file {filename!r} was replaced while the scaffold "
                "was being created"
            )


def _verify_named_destination(
    *,
    parent: Path,
    parent_fd: int,
    target_name: str,
    target_identity: tuple[int, int],
) -> None:
    """Re-walk the public path and prove it still names the anchored target."""
    verification_fd = _open_real_directory(parent)
    try:
        if _identity(os.fstat(verification_fd)) != _identity(os.fstat(parent_fd)):
            raise OperationScaffoldError(
                f"parent directory {parent} was replaced while the scaffold was "
                "being created"
            )
        target_stat = _entry_stat(
            verification_fd,
            target_name,
            f"destination {parent / target_name}",
        )
        if (
            not stat.S_ISDIR(target_stat.st_mode)
            or _identity(target_stat) != target_identity
        ):
            raise OperationScaffoldError(
                f"destination {parent / target_name} was replaced while the "
                "scaffold was being created"
            )
    finally:
        os.close(verification_fd)


def _remove_owned_tree(
    *,
    parent_fd: int,
    target_fd: int | None,
    target_name: str,
    target_identity: tuple[int, int] | None,
    created_files: dict[str, tuple[int, int]],
) -> None:
    """Remove only descriptor-anchored entries whose inode we created."""
    if target_fd is not None:
        for filename, file_identity in reversed(tuple(created_files.items())):
            try:
                file_stat = os.stat(
                    filename,
                    dir_fd=target_fd,
                    follow_symlinks=False,
                )
            except OSError:
                continue
            if (
                stat.S_ISREG(file_stat.st_mode)
                and _identity(file_stat) == file_identity
                and file_stat.st_nlink == 1
            ):
                with suppress(OSError):
                    os.unlink(filename, dir_fd=target_fd)
    if target_identity is None:
        return
    try:
        target_stat = os.stat(
            target_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return
    if stat.S_ISDIR(target_stat.st_mode) and _identity(target_stat) == target_identity:
        with suppress(OSError):
            os.rmdir(target_name, dir_fd=parent_fd)


def _create_anchored_scaffold(
    *,
    target: Path,
    rendered: list[tuple[str, str]],
) -> tuple[Path, ...]:
    parent = target.parent
    parent_fd = _open_real_directory(parent)
    target_fd: int | None = None
    target_identity: tuple[int, int] | None = None
    created_files: dict[str, tuple[int, int]] = {}
    try:
        try:
            os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise OperationScaffoldError(
                f"{target} appeared while the scaffold was being prepared; it is "
                "refused rather than replaced"
            ) from exc

        target_stat = _entry_stat(parent_fd, target.name, f"destination {target}")
        if not stat.S_ISDIR(target_stat.st_mode):
            raise OperationScaffoldError(
                f"destination {target} was replaced by a non-directory while the "
                "scaffold was being created"
            )
        mkdir_identity = _identity(target_stat)
        try:
            target_fd = os.open(
                target.name,
                _directory_flags(),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise OperationScaffoldError(
                f"destination {target} could not be anchored without following a "
                "symlink; path replacement is refused"
            ) from exc
        opened_stat = os.fstat(target_fd)
        target_identity = _identity(opened_stat)
        if not stat.S_ISDIR(opened_stat.st_mode) or target_identity != mkdir_identity:
            raise OperationScaffoldError(
                f"destination {target} was replaced between creation and opening"
            )

        for filename, content in rendered:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            try:
                descriptor = os.open(
                    filename,
                    flags,
                    0o600,
                    dir_fd=target_fd,
                )
            except FileExistsError as exc:
                raise OperationScaffoldError(
                    f"{target / filename} appeared while the scaffold was being "
                    "created; it is refused rather than replaced"
                ) from exc
            try:
                file_stat = os.fstat(descriptor)
                created_files[filename] = _identity(file_stat)
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content.encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.fchmod(handle.fileno(), 0o644)
            finally:
                os.close(descriptor)

        _verify_created_files(target_fd, created_files)
        _verify_named_destination(
            parent=parent,
            parent_fd=parent_fd,
            target_name=target.name,
            target_identity=target_identity,
        )
        os.fchmod(target_fd, 0o755)
        _verify_named_destination(
            parent=parent,
            parent_fd=parent_fd,
            target_name=target.name,
            target_identity=target_identity,
        )
    except BaseException:
        _remove_owned_tree(
            parent_fd=parent_fd,
            target_fd=target_fd,
            target_name=target.name,
            target_identity=target_identity,
            created_files=created_files,
        )
        raise
    finally:
        if target_fd is not None:
            os.close(target_fd)
        os.close(parent_fd)
    return tuple(target / filename for filename, _content in rendered)


def scaffold_operation(
    *,
    pack_id: str,
    operation_id: str,
    owner_id: str,
    owner_display_name: str,
    contribution_kind: str,
    destination: str | Path,
) -> tuple[Path, ...]:
    """Create a byte-deterministic, unregistered incubator scaffold.

    The parent must already exist. The destination directory is created with an
    exclusive mkdir and every fixed file with exclusive, no-follow creation; an
    existing path is never merged or overwritten. Creating a scaffold does not
    import it or mutate the registry.
    """
    _validate_id(pack_id, "pack_id")
    _validate_id(operation_id, "operation_id")
    if pack_id == operation_id:
        raise OperationScaffoldError(
            "pack_id and operation_id must be distinct operation identities"
        )
    if type(owner_id) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", owner_id):
        raise OperationScaffoldError(
            "owner_id must be a lowercase ASCII Python-safe identifier"
        )
    if keyword.iskeyword(owner_id) or owner_id in _WINDOWS_RESERVED:
        raise OperationScaffoldError("owner_id is a reserved Python or Windows name")
    if not is_safe_owner_display_name(owner_display_name):
        raise OperationScaffoldError(
            "owner_display_name must be non-empty safe text of at most 128 characters"
        )
    if type(contribution_kind) is not str or contribution_kind not in (
        "maintainer",
        "partner",
    ):
        raise OperationScaffoldError(
            "contribution_kind must be 'maintainer' or 'partner'"
        )
    if contribution_kind == "partner" and not pack_id.startswith(f"partner.{owner_id}."):
        raise OperationScaffoldError(
            f"partner pack_id must start with 'partner.{owner_id}.'"
        )
    module_name = _module_name(pack_id)
    target = Path(destination)
    _validate_destination_name(target)
    _refuse_symlink_ancestors(target)
    if not _supports_secure_directory_fds():
        raise OperationScaffoldError(
            "safe scaffold creation requires directory-descriptor relative mkdir, "
            "stat, open, unlink and rmdir plus no-follow directory opens; this "
            "platform does not expose that boundary, so no files were created"
        )
    if target.exists() or target.is_symlink():
        raise OperationScaffoldError(
            f"{target} already exists; init-operation never merges, overwrites or "
            "follows an existing path"
        )
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise OperationScaffoldError(
            f"parent directory {parent} does not exist as a real directory; create "
            "it explicitly before generating a scaffold"
        )
    values = {
        "PACK_ID": pack_id,
        "OPERATION_TYPE": pack_id,
        "OPERATION_ID": operation_id,
        "MODULE_NAME": module_name,
        "OWNER_ID": owner_id,
        "OWNER_DISPLAY_NAME_LITERAL": json.dumps(owner_display_name, ensure_ascii=False),
        "CONTRIBUTION_KIND": contribution_kind,
    }
    resources = files(_TEMPLATE_PACKAGE)
    rendered: list[tuple[str, str]] = []
    for template_name in _TEMPLATE_NAMES:
        template = resources.joinpath(template_name).read_text(encoding="utf-8")
        rendered.append((template_name.removesuffix(".tmpl"), _render(template, values)))
    return _create_anchored_scaffold(target=target, rendered=rendered)


__all__ = ["scaffold_operation"]
