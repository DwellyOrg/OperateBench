# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Policy gate for contribution ownership and imports; this is not a sandbox."""

from __future__ import annotations

import ast
import os
import selectors
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import yaml

from operatebench._contribution_structure import (
    LITERAL_NODE_LIMIT as _LITERAL_NODE_LIMIT,
)
from operatebench._contribution_structure import (
    PUBLIC_RESOURCE_SUFFIXES,
    ContributionStructureError,
    bounded_directory_entries,
    claim_operation_identity,
    invalid_cache_entries,
    is_inert_python_init,
    is_literal_tree,
    is_safe_python_component,
    validate_public_tree,
    validate_reserved_registry,
    yaml_event_violation,
)
from operatebench.sdk._validation import (
    canonical_owner_display_name_key,
    is_operation_identity,
    is_pack_version,
    is_safe_display_name,
    is_safe_owner_display_name,
)

OWNER_FIELDS = (
    "schema_version",
    "owner_id",
    "owner_display_name",
    "contribution_kind",
    "code_license",
    "content_license",
)
LITERAL_NODE_LIMIT = _LITERAL_NODE_LIMIT
OPERATION_FIELDS = (
    "schema_version",
    "owner_id",
    "pack_id",
    "operation_id",
    "operation_type",
    "aliases",
    "module",
    "source_path",
    "test_path",
    "fixture_path",
    "documentation_path",
)
RESERVED_FIELDS = ("schema_version", "operations")
MANIFEST_LIMIT = 16 * 1024
PYTHON_LIMIT = 1024 * 1024
DECLARATION_LIMIT = 1024 * 1024
GIT_OUTPUT_LIMIT = 1024 * 1024
GIT_STDERR_LIMIT = 64 * 1024
GIT_TIMEOUT_SECONDS = 10
GIT_READ_CHUNK = 64 * 1024
GIT_STOP_GRACE_SECONDS = 0.25
READ_QUIESCENCE_SECONDS = 0.01
METADATA_FIELDS = {
    "pack_id",
    "operation_type",
    "operation_id",
    "pack_version",
    "display_name",
    "owner_id",
    "owner_display_name",
    "contribution_kind",
    "status",
    "privacy_status",
    "default_spec",
    "agent_ids",
    "aliases",
    "evidence_eligible",
}
DYNAMIC_MODULES = ("importlib", "runpy", "pkgutil")
DYNAMIC_CALLS = {"__import__", "exec", "eval", "compile"}
NAMESPACE_ESCAPE_CALLS = {"globals", "locals", "vars"}
REFLECTION_PRIMITIVES = {"getattr", "setattr", "delattr"}
REFLECTION_ATTRIBUTES = {"__getattribute__", "__setattr__", "__delattr__"}
CONTRIBUTION_SOURCE_OPERATEBENCH_MODULES = frozenset(
    {
        "operatebench._resource_access",
        "operatebench._write_once",
        "operatebench.core.errors",
        "operatebench.core.evaluation",
        "operatebench.core.outcomes",
        "operatebench.core.protocol",
        "operatebench.core.read_contract",
        "operatebench.core.retrieval",
        "operatebench.sdk",
        "operatebench.sdk.errors",
    }
)
CONTRIBUTION_TEST_OPERATEBENCH_MODULES = frozenset(
    {
        "operatebench.cli",
        "operatebench.core.errors",
        "operatebench.sdk",
    }
)
# ImportFrom can bind a package's real submodules as attributes, so approving only
# the module on the left of ``import`` is not a closed boundary.  Keep the public
# names finite and static; never import contribution targets while checking them.
CONTRIBUTION_OPERATEBENCH_SYMBOLS: dict[str, frozenset[str]] = {
    "operatebench.cli": frozenset({"main"}),
    "operatebench.core.errors": frozenset(
        {"AgentRegistryError", "ArtifactError", "OracleManifestError", "SpecSchemaError"}
    ),
    "operatebench.core.evaluation": frozenset({"OperationEvaluation"}),
    "operatebench.core.outcomes": frozenset({"AgentOutcome"}),
    "operatebench.core.protocol": frozenset(
        {"AgentObservation", "EnvironmentContext", "EpisodePlan", "Verdict"}
    ),
    "operatebench.core.read_contract": frozenset({"ReadRequirementContract"}),
    "operatebench.core.retrieval": frozenset(
        {"RetrievalRequest", "RetrieveBatch", "ToolResult"}
    ),
    "operatebench.sdk": frozenset(
        {
            "CheckRequest",
            "CommandResult",
            "OperationPackError",
            "OperationPackMetadata",
            "ReplayRequest",
            "RunRequest",
            "ValidateRequest",
        }
    ),
    "operatebench.sdk.errors": frozenset({"OperationPackMismatchError"}),
}
DICT_MUTATING_METHODS = {
    "__delitem__",
    "__ior__",
    "__setitem__",
    "clear",
    "pop",
    "popitem",
    "setdefault",
    "update",
}


class ContributionError(ValueError):
    """One fail-closed static contribution violation."""


def check_tracked_contribution_artifacts(root: Path) -> None:
    """Reject tracked bytecode/cache paths in the executable contribution trees."""
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ContributionError(f"cannot resolve Git repository root: {exc}") from exc
    command = [
        "git",
        "ls-files",
        "-z",
        "--",
        "src/operatebench/contributions",
        "tests/contributions",
    ]
    try:
        returncode, output, diagnostic = _run_git_bounded(
            command,
            root,
        )
    except FileNotFoundError as exc:
        raise ContributionError("Git executable is unavailable") from exc
    except OSError as exc:
        raise ContributionError(
            f"Git tracked-path discovery could not run: {exc}"
        ) from exc
    if returncode != 0:
        detail = _git_diagnostic(diagnostic)
        suffix = f": {detail}" if detail else ""
        raise ContributionError(f"Git tracked-path discovery failed{suffix}")
    if output and not output.endswith(b"\0"):
        raise ContributionError("Git tracked-path output is malformed")
    try:
        decoded = output.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ContributionError(
            "Git tracked-path output contains non-UTF-8 data"
        ) from exc
    prefixes = (
        ("src", "operatebench", "contributions"),
        ("tests", "contributions"),
    )
    for raw_path in decoded.removesuffix("\0").split("\0") if decoded else ():
        path = PurePosixPath(raw_path)
        parts = path.parts
        if (
            not raw_path
            or path.is_absolute()
            or path.as_posix() != raw_path
            or any(part in ("", ".", "..") for part in parts)
            or not any(parts[: len(prefix)] == prefix for prefix in prefixes)
        ):
            raise ContributionError("Git tracked-path output contains an invalid path")
        if path.suffix in (".pyc", ".pyo") or "__pycache__" in parts:
            raise ContributionError(f"tracked contribution artifact is forbidden: {path}")


def _run_git_bounded(command: list[str], root: Path) -> tuple[int, bytes, bytes]:
    """Collect Git pipes concurrently, enforcing byte and wall-clock limits."""
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        _stop_process(process)
        raise ContributionError("Git tracked-path discovery pipes are unavailable")

    streams = {
        process.stdout: (bytearray(), GIT_OUTPUT_LIMIT, "output exceeded 1 MiB"),
        process.stderr: (bytearray(), GIT_STDERR_LIMIT, "stderr exceeded 64 KiB"),
    }
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, stream)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ContributionError("Git tracked-path discovery timed out")
            events = selector.select(remaining)
            if not events:
                raise ContributionError("Git tracked-path discovery timed out")
            for key, _ in events:
                stream = cast(BinaryIO, key.data)
                retained, limit, overflow_message = streams[stream]
                try:
                    chunk = os.read(stream.fileno(), GIT_READ_CHUNK)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                if len(retained) + len(chunk) > limit:
                    raise ContributionError(f"Git tracked-path {overflow_message}")
                retained.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContributionError("Git tracked-path discovery timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise ContributionError("Git tracked-path discovery timed out") from exc
        return (
            returncode,
            bytes(streams[process.stdout][0]),
            bytes(streams[process.stderr][0]),
        )
    finally:
        selector.close()
        for stream in streams:
            if not stream.closed:
                stream.close()
        _stop_process(process)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Reap a child promptly, escalating after a short bounded grace period."""
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=GIT_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=GIT_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - OS failure
            raise ContributionError(
                "Git tracked-path discovery child could not be reaped"
            ) from exc


def _git_diagnostic(raw: bytes) -> str:
    """Return a bounded, printable diagnostic even for invalid UTF-8 bytes."""
    decoded = raw.decode("utf-8", errors="replace").strip()
    return "".join(character if character.isprintable() else " " for character in decoded)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if type(key) is not str:
            raise ContributionError(f"manifest key {key!r} must be text")
        if key in result:
            raise ContributionError(f"duplicate manifest key {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read_bounded(path: Path, limit: int) -> bytes:
    flags = _descriptor_flags()
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ContributionError(
            f"cannot open regular non-symlink file {path}: {exc}"
        ) from exc
    try:
        expected: tuple[int, ...] | None = None
        first: bytearray | None = None
        for pass_number in range(3):
            if pass_number:
                time.sleep(READ_QUIESCENCE_SECONDS)
                try:
                    offset = os.lseek(fd, 0, os.SEEK_SET)
                except OSError as exc:
                    raise ContributionError(
                        f"cannot seek open file {path}: {exc}"
                    ) from exc
                if offset != 0:
                    raise ContributionError(
                        f"cannot seek open file {path} to offset zero"
                    )
            before = _descriptor_snapshot(fd, path)
            _validate_snapshot(path, before, limit)
            if expected is None:
                expected = before
            elif before != expected:
                raise ContributionError(f"{path} changed during bounded read")
            current = _read_exact(fd, path, before[4], limit)
            after = _descriptor_snapshot(fd, path)
            _validate_snapshot(path, after, limit)
            if after != before or (first is not None and current != first):
                current.clear()
                raise ContributionError(f"{path} changed during bounded read")
            if first is None:
                first = current
            else:
                current.clear()
        assert first is not None
        stable = bytes(first)
        first.clear()
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            raise ContributionError(f"cannot close open file {path}: {exc}") from exc
    return stable


def _descriptor_snapshot(fd: int, path: Path) -> tuple[int, ...]:
    try:
        observed = os.fstat(fd)
    except OSError as exc:
        raise ContributionError(f"cannot inspect open file {path}: {exc}") from exc
    fields = (
        "st_mode",
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    values: list[int] = []
    for field in fields:
        value = getattr(observed, field, None)
        if type(value) is not int:
            raise ContributionError(f"required descriptor statistic {field} unavailable")
        values.append(value)
    return tuple(values)


def _validate_snapshot(path: Path, snapshot: tuple[int, ...], limit: int) -> None:
    if not stat.S_ISREG(snapshot[0]):
        raise ContributionError(f"{path} must be a regular file")
    if snapshot[4] < 0 or snapshot[4] > limit:
        raise ContributionError(f"{path} is larger than limit {limit}")


def _read_exact(fd: int, path: Path, expected: int, limit: int) -> bytearray:
    data = bytearray()
    while len(data) < expected:
        remaining = expected - len(data)
        try:
            chunk = os.read(fd, min(64 * 1024, remaining))
        except OSError as exc:
            raise ContributionError(f"cannot read {path}: {exc}") from exc
        if not chunk:
            break
        if len(chunk) > remaining:
            raise ContributionError(f"{path} changed during bounded read")
        data.extend(chunk)
    if len(data) < expected:
        raise ContributionError(f"short read from {path}")
    try:
        extra = os.read(fd, 1)
    except OSError as exc:
        raise ContributionError(f"cannot read {path}: {exc}") from exc
    if extra:
        if expected == limit:
            raise ContributionError(f"{path} is larger than limit {limit}")
        raise ContributionError(f"{path} changed during bounded read")
    return data


def _descriptor_flags() -> int:
    for name in ("O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK"):
        value = getattr(os, name, None)
        if type(value) is not int or (name != "O_RDONLY" and value == 0):
            raise ContributionError(f"required descriptor capability {name} unavailable")
    seek_set = getattr(os, "SEEK_SET", None)
    if type(seek_set) is not int:
        raise ContributionError("required descriptor capability SEEK_SET unavailable")
    for name in ("open", "read", "fstat", "lseek", "close"):
        if not callable(getattr(os, name, None)):
            raise ContributionError(
                f"required descriptor primitive os.{name} unavailable"
            )
    if not callable(getattr(stat, "S_ISREG", None)):
        raise ContributionError(
            "required regular descriptor primitive stat.S_ISREG unavailable"
        )
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _decode(path: Path, limit: int) -> str:
    try:
        return _read_bounded(path, limit).decode("utf-8")
    except UnicodeError as exc:
        raise ContributionError(f"cannot parse {path} as UTF-8: {exc}") from exc


def _load(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    try:
        text = _decode(path, MANIFEST_LIMIT)
        for event in yaml.parse(text):
            if violation := yaml_event_violation(event):
                raise ContributionError(f"manifest {path} contains a YAML {violation}")
        value = yaml.load(text, Loader=_UniqueKeyLoader)
    except ContributionError:
        raise
    except (yaml.YAMLError, RecursionError) as exc:
        raise ContributionError(f"cannot parse manifest {path}: {exc}") from exc
    if type(value) is not dict:
        raise ContributionError(f"manifest {path} must be a mapping")
    if tuple(value) != fields:
        raise ContributionError(
            f"manifest {path} fields must be exactly {list(fields)} in order"
        )
    if not is_literal_tree(value):
        raise ContributionError(f"manifest {path} is not a plain literal tree")
    return value


def _is_literal_tree(value: object) -> bool:
    return is_literal_tree(value)


def _text(value: object, field: str, maximum: int = 256) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ContributionError(f"{field} must be non-empty bounded text")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ContributionError(f"{field} contains unsafe text")
    return value


def _owner_display_text(value: object, field: str) -> str:
    if not is_safe_owner_display_name(value):
        raise ContributionError(
            f"{field} must be non-empty safe text of at most 128 characters"
        )
    assert type(value) is str
    return value


def _operation_identity(value: object, field: str) -> str:
    if not is_operation_identity(value):
        raise ContributionError(f"{field} must be a lowercase bounded operation identity")
    assert type(value) is str
    return value


def load_reserved_operation_identities(root: Path) -> tuple[dict[str, object], ...]:
    """Strictly parse the maintainer-owned static registration authority."""
    path = root / "src/operatebench/resources/operation_pack_registry.yaml"
    manifest = _load(path, RESERVED_FIELDS)
    try:
        return tuple(dict(row) for row in validate_reserved_registry(manifest))
    except ContributionStructureError as exc:
        raise ContributionError(str(exc)) from exc


def _claim_operation_identity(
    owners: dict[str, tuple[str, str]], value: str, pack_id: str, kind: str
) -> None:
    try:
        claim_operation_identity(owners, value, pack_id, kind)
    except ContributionStructureError as exc:
        raise ContributionError(str(exc)) from exc


def _safe_python_component(value: object, field: str) -> str:
    component = _text(value, field)
    if not is_safe_python_component(component):
        raise ContributionError(f"{field} is not a safe lowercase Python identifier")
    return component


def _safe_owner_id(value: object, field: str = "owner_id") -> str:
    return _safe_python_component(_text(value, field, 63), field)


def _declared_path(
    root: Path, value: object, field: str, prefix: str, *, directory: bool
) -> tuple[Path, PurePosixPath]:
    text = _text(value, field)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in text
        or pure.as_posix() != text
    ):
        raise ContributionError(f"{field} must be a normalized relative path")
    if not (text == prefix or text.startswith(prefix + "/")):
        raise ContributionError(f"{field} must remain inside {prefix}")
    path = root
    for index, part in enumerate(pure.parts):
        path /= part
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise ContributionError(f"{field} does not exist: {text}") from exc
        if stat.S_ISLNK(mode):
            raise ContributionError(f"{field} traverses a symlink: {path}")
        final = index == len(pure.parts) - 1
        if not final and not stat.S_ISDIR(mode):
            raise ContributionError(f"{field} has a non-directory ancestor: {path}")
        if final and directory and not stat.S_ISDIR(mode):
            raise ContributionError(f"{field} must be a real directory")
        if final and not directory and not stat.S_ISREG(mode):
            raise ContributionError(f"{field} must be a real regular file")
    return path, pure


def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(_decode(path, PYTHON_LIMIT), filename=str(path))
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ContributionError(f"cannot parse Python {path}: {exc}") from exc


def _enumerate(path: Path, label: str) -> list[Path]:
    try:
        return list(bounded_directory_entries(path))
    except ContributionStructureError as exc:
        raise ContributionError(f"cannot enumerate {label} {path}: {exc}") from exc


def _check_bytecode_cache(path: Path, label: str) -> None:
    """Allow only immediate, canonical CPython cache files in a real cache dir."""
    entries = sorted(_enumerate(path, f"{label} bytecode cache"))
    invalid = invalid_cache_entries(entries)
    if invalid:
        raise ContributionError(
            f"non-canonical entry is forbidden in {label} bytecode cache: {invalid[0]}"
        )


def _scan_public_tree(
    path: Path, label: str, *, allow_canonical_cache: bool = False
) -> dict[Path, ast.Module]:
    """Validate authored entries and parse Python, excluding real bytecode caches."""
    trees: dict[Path, ast.Module] = {}
    try:
        files = validate_public_tree(path, allow_canonical_cache=allow_canonical_cache)
    except ContributionStructureError as exc:
        raise ContributionError(f"invalid {label}: {exc}") from exc
    for item in files:
        if item.suffix == ".py":
            trees[item] = _parse_python(item)
        else:
            _decode(item, DECLARATION_LIMIT)
    return trees


def _module_for_path(root: Path, path: Path, root_module: str) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((root_module, *parts))


def _mode(path: Path, label: str) -> int:
    try:
        return path.lstat().st_mode
    except OSError as exc:
        raise ContributionError(f"cannot inspect {label} {path}: {exc}") from exc


def _absolute_import(node: ast.ImportFrom, current_module: str, is_package: bool) -> str:
    if not node.level:
        return node.module or ""
    package = current_module if is_package else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove >= len(parts):
        raise ContributionError("relative import beyond top-level package")
    base = parts[: len(parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _check_import_target(
    path: Path,
    name: str,
    owner_id: str,
    operation_module: str,
    *,
    is_test: bool,
) -> None:
    if name == "sys" or name.startswith("sys."):
        raise ContributionError(f"{path}: interpreter-global module 'sys' is forbidden")
    if name == "src.operatebench" or name.startswith("src.operatebench."):
        raise ContributionError(
            "non-canonical import spelling 'src.operatebench' is forbidden"
        )
    if name == "tests" or name.startswith("tests."):
        test_operation = "tests.contributions." + operation_module.removeprefix(
            "operatebench.contributions."
        )
        if not is_test:
            raise ContributionError(f"{path}: import from tests namespace {name!r}")
        if name == test_operation or name.startswith(test_operation + "."):
            return
        test_prefix = "tests.contributions."
        if name.startswith(test_prefix):
            name = "operatebench.contributions." + name.removeprefix(test_prefix)
        elif name == "tests.contributions":
            raise ContributionError(f"{path}: cross-owner import {name!r}")
        else:
            raise ContributionError(
                f"{path}: import outside own contribution test namespace {name!r}"
            )
    owner_prefix = f"operatebench.contributions.{owner_id}."
    if name == "operatebench._resource_access" or name.startswith(
        "operatebench._resource_access."
    ):
        raise ContributionError(f"{path}: resource access import is not admitted")
    if name == "builtins" or name.startswith("builtins."):
        raise ContributionError(f"{path}: dynamic execution module {name!r}")
    if name == "operatebench.domains" or name.startswith("operatebench.domains."):
        raise ContributionError(f"{path}: import from domain pack {name!r}")
    if name == "operatebench.contributions" or name.startswith(
        "operatebench.contributions."
    ):
        if not (name == owner_prefix[:-1] or name.startswith(owner_prefix)):
            raise ContributionError(f"{path}: cross-owner import {name!r}")
        if name.startswith(owner_prefix) and not (
            name == operation_module or name.startswith(operation_module + ".")
        ):
            raise ContributionError(f"{path}: cross-operation import {name!r}")
    if name in DYNAMIC_MODULES or name.startswith(
        tuple(item + "." for item in DYNAMIC_MODULES)
    ):
        raise ContributionError(f"{path}: dynamic loading module {name!r}")


def _check_operatebench_module_import(
    path: Path,
    name: str,
    operation_module: str,
    *,
    is_test: bool,
) -> None:
    if not (name == "operatebench" or name.startswith("operatebench.")):
        return
    if name == operation_module or name.startswith(operation_module + "."):
        return
    supported = (
        CONTRIBUTION_TEST_OPERATEBENCH_MODULES
        if is_test
        else CONTRIBUTION_SOURCE_OPERATEBENCH_MODULES
    )
    if name not in supported:
        context = "test" if is_test else "source"
        raise ContributionError(
            f"{path}: operatebench module {name!r} is not supported in "
            f"contribution {context}"
        )


def _contains_dunder_dict_attribute(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.Attribute) and candidate.attr == "__dict__"
        for candidate in ast.walk(node)
    )


def _is_exact_as_file_call(call: ast.AST | None, argument: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "as_file"
        and call.args == [argument]
        and not call.keywords
    )


def _has_non_name_binding(tree: ast.Module, name: str) -> bool:
    """Recognise bindings whose identifier is not represented by an AST Name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return True
        if (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".")[0]) == name
        ):
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            return True
    return False


def _has_binding_other_than_required_import(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.alias):
            bound = node.asname or node.name.split(".")[0]
            if bound == name and not (node.name == name and node.asname is None):
                return True
            continue
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            return True
    return False


def _check_resource_call(
    path: Path,
    tree: ast.Module,
    call: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> None:
    """Require ``files(__package__)`` to anchor one direct packaged file."""
    attribute = parents.get(call)
    joinpath = parents.get(attribute) if attribute is not None else None
    if not (
        isinstance(attribute, ast.Attribute)
        and attribute.value is call
        and attribute.attr == "joinpath"
        and isinstance(joinpath, ast.Call)
        and joinpath.func is attribute
        and len(joinpath.args) == 1
        and not joinpath.keywords
        and isinstance(joinpath.args[0], ast.Constant)
        and type(joinpath.args[0].value) is str
    ):
        raise ContributionError(
            f"{path}: resource access must be an immediate single-argument "
            "files(__package__).joinpath() call"
        )
    name = joinpath.args[0].value
    if (
        not is_safe_display_name(name)
        or name in {".", ".."}
        or name.startswith(".")
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise ContributionError(
            f"{path}: resource name must be a safe literal direct-child filename"
        )
    if Path(name).suffix not in PUBLIC_RESOURCE_SUFFIXES:
        raise ContributionError(f"{path}: resource suffix is forbidden: {name!r}")
    resource = path.parent / name
    mode = _mode(resource, "resource direct child")
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ContributionError(
            f"{path}: resource direct child must be a regular non-symlink file: "
            f"{resource}"
        )
    use = parents.get(joinpath)
    if _is_exact_as_file_call(use, joinpath):
        return
    if not (
        isinstance(use, ast.Assign)
        and use.value is joinpath
        and len(use.targets) == 1
        and isinstance(use.targets[0], ast.Name)
    ):
        raise ContributionError(
            f"{path}: resource result must be passed directly to as_file or assigned "
            "to one simple name"
        )
    target = use.targets[0]
    resource_name = target.id
    if _has_non_name_binding(tree, resource_name):
        raise ContributionError(
            f"{path}: resource name {resource_name!r} has an ambiguous binding"
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != resource_name:
            continue
        if node is target:
            continue
        parent = parents.get(node)
        if isinstance(node.ctx, ast.Load) and _is_exact_as_file_call(parent, node):
            continue
        raise ContributionError(
            f"{path}: resource name {resource_name!r} may only be loaded as the sole "
            "positional argument to as_file"
        )


def _check_ast(
    path: Path,
    tree: ast.Module,
    owner_id: str,
    operation_module: str,
    current_module: str,
    *,
    is_test: bool,
) -> None:
    aliases: set[str] = set()
    loader_aliases: set[str] = set()
    operatebench_modules: set[str] = set()
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    resource_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and _absolute_import(node, current_module, path.name == "__init__.py")
        == "operatebench._resource_access"
    ]
    top_level_resource_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and _absolute_import(node, current_module, path.name == "__init__.py")
        == "operatebench._resource_access"
    ]
    if resource_imports != top_level_resource_imports or len(resource_imports) > 1:
        raise ContributionError(
            f"{path}: resource access import must be a single top-level statement"
        )
    resource_file_names = {"files"} if resource_imports else set()
    allowed_resource_name_uses = {
        id(candidate.func)
        for candidate in ast.walk(tree)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "files"
        and len(candidate.args) == 1
        and isinstance(candidate.args[0], ast.Name)
        and candidate.args[0].id == "__package__"
        and not candidate.keywords
    }
    resource_calls = [
        candidate
        for candidate in ast.walk(tree)
        if (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id in resource_file_names
            and id(candidate.func) in allowed_resource_name_uses
        )
    ]
    if resource_imports:
        if any(
            _has_binding_other_than_required_import(tree, name)
            for name in ("files", "as_file")
        ):
            raise ContributionError(f"{path}: resource access name is shadowed")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "as_file":
                parent = parents.get(node)
                if not (
                    isinstance(node.ctx, ast.Load)
                    and isinstance(parent, ast.Call)
                    and parent.func is node
                    and len(parent.args) == 1
                    and not parent.keywords
                ):
                    raise ContributionError(
                        f"{path}: resource access as_file must be called exactly"
                    )
    allowed_pack_targets = {
        id(node.targets[0])
        for node in tree.body
        if current_module == operation_module + ".pack"
        and isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PACK"
    }
    if any(
        isinstance(node, ast.Name)
        and node.id == "__builtins__"
        and isinstance(node.ctx, (ast.Load, ast.Store, ast.Del))
        for node in ast.walk(tree)
    ):
        raise ContributionError(f"{path}: __builtins__ reference is forbidden")
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            if any(
                alias.name == "operatebench" or alias.name.startswith("operatebench.")
                for alias in node.names
            ):
                raise ContributionError(
                    f"{path}: bare operatebench import is forbidden; use an explicit "
                    "from import"
                )
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = _absolute_import(node, current_module, path.name == "__init__.py")
            if name == "operatebench" or name.startswith("operatebench."):
                operatebench_modules.add(name)
            if name == "operatebench._resource_access":
                if [(item.name, item.asname) for item in node.names] != [
                    ("files", None),
                    ("as_file", None),
                ]:
                    raise ContributionError(
                        f"{path}: resource access import must be exact "
                        "'from operatebench._resource_access import files, as_file'"
                    )
                names = []
                # The exact helper import is the sole exception to the general
                # resource-module import rejection below.
                continue
            if name == "operatebench._write_once" and [
                (item.name, item.asname) for item in node.names
            ] != [
                ("_write_once_bytes", None),
                ("_WriteOnceFailure", None),
            ]:
                raise ContributionError(
                    f"{path}: write-once helper import must be exact "
                    "'from operatebench._write_once import "
                    "_write_once_bytes, _WriteOnceFailure'"
                )
            if any(alias.name in {"files", "as_file"} for alias in node.names):
                raise ContributionError(
                    f"{path}: resource access names may only be imported directly "
                    "from operatebench._resource_access"
                )
            if any(alias.name == "sys" for alias in node.names):
                raise ContributionError(
                    f"{path}: interpreter-global module 'sys' is forbidden"
                )
            if name in CONTRIBUTION_OPERATEBENCH_SYMBOLS:
                approved = CONTRIBUTION_OPERATEBENCH_SYMBOLS[name]
                unknown = [
                    alias.name for alias in node.names if alias.name not in approved
                ]
                if unknown:
                    raise ContributionError(
                        f"{path}: symbol {unknown[0]!r} is not supported from {name!r}"
                    )
            names.append(name)
            for alias in node.names:
                if alias.name == "*":
                    raise ContributionError(f"{path}: star import is forbidden")
                names.append(f"{name}.{alias.name}" if name else alias.name)
                if name == "builtins" and alias.name in DYNAMIC_CALLS:
                    aliases.add(alias.asname or alias.name)
                if name.startswith(DYNAMIC_MODULES) and alias.name in {
                    "import_module",
                    "run_module",
                    "run_path",
                }:
                    loader_aliases.add(alias.asname or alias.name)
            if node.level and any(alias.name in DYNAMIC_MODULES for alias in node.names):
                raise ContributionError(f"{path}: dynamic loading relative import")
            if (
                node.level
                and node.module
                and node.module.split(".", 1)[0] in DYNAMIC_MODULES
            ):
                raise ContributionError(f"{path}: dynamic loading relative import")
        for name in names:
            _check_import_target(
                path,
                name,
                owner_id,
                operation_module,
                is_test=is_test,
            )
        if (
            isinstance(node, ast.Name)
            and node.id in resource_file_names
            and (
                isinstance(node.ctx, (ast.Store, ast.Del))
                or (
                    isinstance(node.ctx, ast.Load)
                    and id(node) not in allowed_resource_name_uses
                )
            )
        ):
            raise ContributionError(
                f"{path}: resource access files must be called directly as "
                "files(__package__)"
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in NAMESPACE_ESCAPE_CALLS
        ):
            raise ContributionError(
                f"{path}: namespace escape primitive {node.id!r} is forbidden"
            )
        if (
            isinstance(node, ast.expr)
            and isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
            and _contains_dunder_dict_attribute(node)
        ):
            raise ContributionError(
                f"{path}: object namespace mutation through .__dict__ is forbidden"
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in DYNAMIC_CALLS
        ):
            raise ContributionError(f"{path}: dynamic execution primitive {node.id}")
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in {"__module__", "__class__", "__name__", "__package__", "PACK"}
            and id(node) not in allowed_pack_targets
        ):
            raise ContributionError(f"{path}: forbidden binding mutation {node.id}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr == "metadata"
        ):
            raise ContributionError(f"{path}: reserved metadata mutation is forbidden")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr
            in {"__module__", "__class__", "__name__", "__package__", "PACK"}
        ):
            raise ContributionError(f"{path}: forbidden binding mutation {node.attr}")
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in REFLECTION_PRIMITIVES
        ):
            raise ContributionError(
                f"{path}: dynamic reflection primitive {node.id!r} is forbidden"
            )
        if isinstance(node, ast.Attribute) and node.attr in REFLECTION_ATTRIBUTES:
            raise ContributionError(
                f"{path}: dynamic reflection primitive {node.attr!r} is forbidden"
            )
        if isinstance(node, ast.Attribute) and node.attr == "sys":
            raise ContributionError(
                f"{path}: interpreter-global module 'sys' is forbidden"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in loader_aliases:
                raise ContributionError(
                    f"{path}: dynamic loading primitive {node.func.id}"
                )
            if node.func.id in DYNAMIC_CALLS or node.func.id in aliases:
                raise ContributionError(
                    f"{path}: dynamic execution primitive {node.func.id}"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in DICT_MUTATING_METHODS
            and _contains_dunder_dict_attribute(node.func.value)
        ):
            raise ContributionError(
                f"{path}: object namespace mutation through .__dict__ is forbidden"
            )
    for name in sorted(operatebench_modules):
        _check_operatebench_module_import(path, name, operation_module, is_test=is_test)
    for resource_call in resource_calls:
        _check_resource_call(path, tree, resource_call, parents)


def _check_inert_init(path: Path, tree: ast.Module) -> None:
    if not is_inert_python_init(tree):
        raise ContributionError(f"owner root {path} must be inert")


def _check_package_docstring_init(path: Path, tree: ast.Module) -> None:
    if not (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and type(tree.body[0].value.value) is str
    ):
        raise ContributionError(
            f"contribution root {path} must contain only a package docstring"
        )


def _metadata_contract(
    tree: ast.Module, operation_dir: Path, expected: dict[str, object]
) -> ast.ClassDef:
    """Validate and return the exact top-level class bound to ``PACK``."""
    imports = [
        alias
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == "operatebench.sdk"
        for alias in node.names
        if alias.name == "OperationPackMetadata" and alias.asname is None
    ]
    rebound = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "OperationPackMetadata"
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == "OperationPackMetadata"
        )
    ]
    if len(imports) != 1 or rebound:
        raise ContributionError(
            f"{operation_dir}: metadata constructor must be the direct SDK import"
        )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OperationPackMetadata"
    ]
    metadata_assignments: list[ast.Assign] = []
    authoritative: list[tuple[ast.ClassDef, ast.Call]] = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for item in class_node.body:
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == "metadata"
            ):
                metadata_assignments.append(item)
                if (
                    isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "OperationPackMetadata"
                ):
                    authoritative.append((class_node, item.value))
    if (
        len(calls) != 1
        or len(metadata_assignments) != 1
        or len(authoritative) != 1
        or calls[0] is not authoritative[0][1]
    ):
        raise ContributionError(
            f"{operation_dir}: exactly one authoritative metadata call is required"
        )
    owner_class, call = authoritative[0]
    if owner_class.decorator_list:
        raise ContributionError(
            f"{operation_dir}: authoritative pack class decorators are forbidden"
        )
    if owner_class.bases or owner_class.keywords:
        raise ContributionError(
            f"{operation_dir}: authoritative pack class may have no base classes "
            "or class keywords/metaclass"
        )
    class_body = list(owner_class.body)
    if (
        class_body
        and isinstance(class_body[0], ast.Expr)
        and isinstance(class_body[0].value, ast.Constant)
        and type(class_body[0].value.value) is str
    ):
        class_body.pop(0)
    for item in class_body:
        if item is metadata_assignments[0]:
            continue
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ContributionError(
                f"{operation_dir}: pack class body is limited to metadata and methods"
            )
        if item.name.startswith("__") and item.name.endswith("__"):
            raise ContributionError(
                f"{operation_dir}: pack construction/metadata interception method "
                f"{item.name!r} is forbidden"
            )
        if item.name == "metadata" or any(
            not isinstance(decorator, ast.Name)
            or decorator.id not in {"staticmethod", "classmethod"}
            for decorator in item.decorator_list
        ):
            raise ContributionError(
                f"{operation_dir}: pack methods may not replace metadata and "
                "may use only inert built-in method decorators"
            )
        for nested in ast.walk(item):
            if (
                isinstance(nested, ast.Attribute)
                and isinstance(nested.ctx, (ast.Store, ast.Del))
                and nested.attr == "metadata"
            ):
                raise ContributionError(
                    f"{operation_dir}: pack methods may not mutate metadata"
                )
            if (
                isinstance(nested, ast.Call)
                and isinstance(nested.func, ast.Name)
                and nested.func.id in {"setattr", "delattr"}
                and len(nested.args) >= 2
                and isinstance(nested.args[1], ast.Constant)
                and nested.args[1].value == "metadata"
            ):
                raise ContributionError(
                    f"{operation_dir}: pack methods may not intercept metadata"
                )
    if call.args or any(item.arg is None for item in call.keywords):
        raise ContributionError(
            f"{operation_dir}: metadata rejects positional arguments and **kwargs"
        )
    pack_bindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.id == "PACK"
    ]
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PACK"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == owner_class.name
        and not node.value.args
        and not node.value.keywords
    ]
    if len(pack_bindings) != 1 or len(declarations) != 1:
        raise ContributionError(
            f"{operation_dir}: exactly one top-level PACK = {owner_class.name}() "
            "binding is required"
        )
    declaration = declarations[0]
    if tree.body[-1] is not declaration:
        raise ContributionError(
            f"{operation_dir}: PACK constructor binding must be the exact final statement"
        )
    forbidden_module_statements = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Match,
        ast.Delete,
        ast.AugAssign,
    )
    for item in tree.body:
        if item is declaration:
            continue
        if isinstance(item, forbidden_module_statements):
            raise ContributionError(
                f"{operation_dir}: forbidden pack module execution or mutation statement"
            )
        if (
            isinstance(item, ast.Expr)
            and item.lineno > owner_class.lineno
            and not (
                isinstance(item.value, ast.Constant) and type(item.value.value) is str
            )
        ):
            raise ContributionError(
                f"{operation_dir}: module-level calls/expressions are forbidden"
            )
    protected = {
        "PACK",
        "metadata",
        owner_class.name,
        "__module__",
        "__class__",
        "__name__",
        "__package__",
    }
    for item in tree.body:
        if item is owner_class or item is declaration:
            continue
        direct_targets: list[ast.expr] = []
        if isinstance(item, ast.Assign):
            direct_targets.extend(item.targets)
        elif isinstance(item, (ast.AnnAssign, ast.AugAssign)):
            direct_targets.append(item.target)
        elif isinstance(item, ast.Delete):
            direct_targets.extend(item.targets)
        if any(
            isinstance(target, ast.Name) and target.id in protected
            for target in direct_targets
        ) or (
            isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name in protected
        ):
            raise ContributionError(
                f"{operation_dir}: forbidden pack class/metadata binding mutation"
            )
        for node in ast.walk(item):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.attr in protected
            ):
                raise ContributionError(
                    f"{operation_dir}: forbidden pack binding/metadata mutation"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"setattr", "delattr"}
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in {owner_class.name, "PACK"}
            ):
                raise ContributionError(
                    f"{operation_dir}: forbidden statically identifiable "
                    "metadata mutation"
                )
    values = {item.arg: item.value for item in call.keywords if item.arg is not None}
    allowed = METADATA_FIELDS | {"sdk_api_version"}
    if set(values) not in (METADATA_FIELDS, allowed):
        raise ContributionError(f"{operation_dir}: metadata keyword fields are not exact")
    for field, wanted in expected.items():
        field_node = values.get(field)
        if field == "aliases":
            try:
                actual = ast.literal_eval(field_node) if field_node is not None else None
            except (ValueError, TypeError):
                actual = None
            if type(actual) is not tuple or actual != wanted:
                raise ContributionError(
                    f"{operation_dir}: metadata aliases must be direct exact {wanted!r}"
                )
            continue
        if (
            not isinstance(field_node, ast.Constant)
            or type(field_node.value) is not type(wanted)
            or field_node.value != wanted
        ):
            raise ContributionError(
                f"{operation_dir}: metadata {field} must be direct exact {wanted!r}"
            )
    sdk = values.get("sdk_api_version")
    if sdk is not None and (
        not isinstance(sdk, ast.Constant) or type(sdk.value) is not int or sdk.value != 2
    ):
        raise ContributionError(
            f"{operation_dir}: metadata sdk_api_version must be direct integer 2 "
            "or use the v2 default"
        )
    pack_version = values["pack_version"]
    if (
        not isinstance(pack_version, ast.Constant)
        or type(pack_version.value) is not str
        or not is_pack_version(pack_version.value)
    ):
        raise ContributionError(
            f"{operation_dir}: metadata pack_version must be a direct exact "
            "three-part version string"
        )
    display_name = values["display_name"]
    if (
        not isinstance(display_name, ast.Constant)
        or type(display_name.value) is not str
        or not is_safe_display_name(display_name.value)
    ):
        raise ContributionError(
            f"{operation_dir}: metadata display_name must be a direct non-empty "
            "safe string of at most 128 characters"
        )
    agent_ids = values["agent_ids"]
    if not isinstance(agent_ids, ast.Tuple) or not agent_ids.elts:
        raise ContributionError(
            f"{operation_dir}: metadata agent_ids must be a direct non-empty tuple "
            "of direct exact strings"
        )
    agents: list[str] = []
    for element in agent_ids.elts:
        if (
            not isinstance(element, ast.Constant)
            or type(element.value) is not str
            or not is_operation_identity(element.value)
        ):
            raise ContributionError(
                f"{operation_dir}: metadata agent_ids must contain only direct "
                "exact registry IDs"
            )
        agents.append(element.value)
    if len(set(agents)) != len(agents) or agents != sorted(agents):
        raise ContributionError(
            f"{operation_dir}: metadata agent_ids must be sorted and unique"
        )
    return owner_class


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    count = min(len(left.parts), len(right.parts))
    return left.parts[:count] == right.parts[:count]


def _check_owner_test_root(root: Path, owner_id: str, operation_names: set[str]) -> None:
    test_root, _ = _declared_path(
        root,
        f"tests/contributions/{owner_id}",
        "owner test root",
        f"tests/contributions/{owner_id}",
        directory=True,
    )
    entries = _enumerate(test_root, "owner test root")
    found_operations: set[str] = set()
    for entry in entries:
        mode = _mode(entry, "owner test root entry")
        if stat.S_ISLNK(mode):
            raise ContributionError(
                f"owner test root symlink entry is forbidden: {entry}"
            )
        if entry.name == "__pycache__" and stat.S_ISDIR(mode):
            _check_bytecode_cache(entry, "owner test root")
            continue
        if stat.S_ISDIR(mode):
            if entry.name not in operation_names:
                raise ContributionError(
                    f"owner test root has undeclared operation directory: {entry}"
                )
            found_operations.add(entry.name)
            continue
        if not stat.S_ISREG(mode) or entry.name not in {"conftest.py", "__init__.py"}:
            raise ContributionError(
                f"owner test root entry {entry} is not an allowed regular file or "
                "declared operation directory"
            )
        tree = _parse_python(entry)
        if entry.name == "__init__.py":
            _check_inert_init(entry, tree)
        current = f"tests.contributions.{owner_id}." + entry.stem
        _check_ast(
            entry,
            tree,
            owner_id,
            f"operatebench.contributions.{owner_id}.__owner_neutral__",
            current,
            is_test=True,
        )
    if found_operations != operation_names:
        missing = sorted(operation_names - found_operations)
        raise ContributionError(
            "owner test root operation directories do not match source; "
            f"missing {missing}"
        )


def _check_shared_roots(
    contribution_root: Path, contribution_entries: list[Path], root: Path
) -> list[Path]:
    """Validate both shared namespace roots before any owner-owned file is parsed."""
    owner_dirs: list[Path] = []
    init_path = contribution_root / "__init__.py"
    for entry in contribution_entries:
        mode = _mode(entry, "contribution root entry")
        if stat.S_ISLNK(mode):
            raise ContributionError(f"contribution root symlink is forbidden: {entry}")
        if entry.name == "__pycache__" and stat.S_ISDIR(mode):
            _check_bytecode_cache(entry, "contribution root")
            continue
        if entry.name == "__init__.py" and stat.S_ISREG(mode):
            continue
        if stat.S_ISDIR(mode) and not entry.name.startswith("."):
            owner_dirs.append(entry)
            continue
        raise ContributionError(f"contribution root entry is forbidden: {entry}")
    if init_path not in contribution_entries:
        raise ContributionError("contribution root requires regular __init__.py")
    _check_package_docstring_init(init_path, _parse_python(init_path))

    test_root = root / "tests/contributions"
    if not stat.S_ISDIR(_mode(test_root, "shared test root")):
        raise ContributionError("shared test root must be a real directory")
    test_owner_dirs: set[str] = set()
    test_owner_paths: list[Path] = []
    for entry in _enumerate(test_root, "shared test root"):
        mode = _mode(entry, "shared test root entry")
        if stat.S_ISLNK(mode):
            raise ContributionError(f"shared test root symlink is forbidden: {entry}")
        if entry.name == "__pycache__" and stat.S_ISDIR(mode):
            _check_bytecode_cache(entry, "shared test root")
            continue
        if entry.name == "__init__.py" and stat.S_ISREG(mode):
            _check_inert_init(entry, _parse_python(entry))
            continue
        if stat.S_ISDIR(mode) and not entry.name.startswith("."):
            test_owner_dirs.add(entry.name)
            test_owner_paths.append(entry)
            continue
        raise ContributionError(f"shared test root entry is forbidden: {entry}")
    source_owner_names = {entry.name for entry in owner_dirs}
    if test_owner_dirs != source_owner_names:
        raise ContributionError(
            "shared test root owner directories must exactly match source owners; "
            f"source={sorted(source_owner_names)}, tests={sorted(test_owner_dirs)}"
        )
    for public_kind, public_root in (
        ("example", root / "examples/contributions"),
        ("documentation", root / "docs/contributions"),
    ):
        if not stat.S_ISDIR(_mode(public_root, f"shared {public_kind} root")):
            raise ContributionError(f"shared {public_kind} root must be a real directory")
        public_owners: dict[str, Path] = {}
        for entry in _enumerate(public_root, f"shared {public_kind} root"):
            mode = _mode(entry, f"shared {public_kind} root entry")
            if (
                stat.S_ISLNK(mode)
                or not stat.S_ISDIR(mode)
                or not is_safe_python_component(entry.name)
            ):
                raise ContributionError(
                    f"shared {public_kind} root entry is forbidden: {entry}"
                )
            public_owners[entry.name] = entry
        if set(public_owners) != source_owner_names:
            raise ContributionError(
                f"shared {public_kind} root owners must exactly match source owners"
            )
        for owner_path in public_owners.values():
            _scan_public_tree(owner_path, f"owner {public_kind} root")
    for test_owner in test_owner_paths:
        owner_init = test_owner / "__init__.py"
        if not stat.S_ISREG(_mode(owner_init, "owner test initializer")):
            raise ContributionError("owner test initializer must be a regular file")
        _check_inert_init(owner_init, _parse_python(owner_init))
    return sorted(owner_dirs)


def check_contributions(root: Path) -> tuple[str, ...]:
    _descriptor_flags()
    try:
        root = root.resolve()
    except OSError as exc:
        raise ContributionError(f"cannot resolve trusted repository root: {exc}") from exc
    reserved_rows = load_reserved_operation_identities(root)
    contribution_root = root / "src/operatebench/contributions"
    try:
        contribution_mode = contribution_root.lstat().st_mode
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ContributionError(f"cannot inspect contribution root: {exc}") from exc
    if not stat.S_ISDIR(contribution_mode):
        raise ContributionError("contribution root must be a real directory")
    reserved_by_identity: dict[str, dict[str, object]] = {}
    reserved_owner_authority: dict[str, tuple[str, str]] = {}
    identity_owners: dict[str, tuple[str, str]] = {}
    for row in reserved_rows:
        authority = (str(row["owner_display_name"]), str(row["contribution_kind"]))
        prior_authority = reserved_owner_authority.setdefault(
            str(row["owner_id"]), authority
        )
        if prior_authority != authority:
            raise ContributionError(
                f"reserved owner {row['owner_id']!r} has inconsistent authority rows"
            )
        row_aliases = row["aliases"]
        if type(row_aliases) is not tuple:
            raise ContributionError("reserved aliases parser invariant failed")
        for identity in (
            str(row["pack_id"]),
            str(row["operation_type"]),
            str(row["operation_id"]),
            *row_aliases,
        ):
            reserved_by_identity[str(identity)] = row
        for value, kind in (
            (str(row["pack_id"]), "pack_id"),
            (str(row["operation_type"]), "operation_type"),
            *((str(alias), "alias") for alias in row_aliases),
            (str(row["operation_id"]), "operation_id"),
        ):
            _claim_operation_identity(identity_owners, value, str(row["pack_id"]), kind)
    owners: set[str] = set()
    displays = {
        canonical_owner_display_name_key(display): owner_id
        for owner_id, (display, _kind) in reserved_owner_authority.items()
    }
    claimed: list[PurePosixPath] = []
    checked: list[str] = []
    contribution_entries = _enumerate(contribution_root, "contribution root")
    owner_dirs = _check_shared_roots(contribution_root, contribution_entries, root)
    for owner_dir in owner_dirs:
        if owner_dir.is_symlink():
            raise ContributionError(f"symlinked owner root is forbidden: {owner_dir}")
        owner = _load(owner_dir / "OWNER.yaml", OWNER_FIELDS)
        owner_id = _safe_owner_id(owner["owner_id"])
        if type(owner["schema_version"]) is not int or owner["schema_version"] != 1:
            raise ContributionError("owner schema_version must be integer 1")
        if owner_dir.name != owner_id or owner_id in owners:
            raise ContributionError(
                f"owner directory/owner_id mismatch or duplicate: {owner_id}"
            )
        display = _owner_display_text(owner["owner_display_name"], "owner_display_name")
        folded = canonical_owner_display_name_key(display)
        if folded in displays and displays[folded] != owner_id:
            raise ContributionError(
                f"owner display name {display!r} is claimed by different owner IDs"
            )
        displays[folded] = owner_id
        kind = owner["contribution_kind"]
        if type(kind) is not str or kind not in ("maintainer", "partner"):
            raise ContributionError("contribution_kind must be maintainer or partner")
        owner_authority = reserved_owner_authority.get(owner_id)
        if owner_authority is not None and owner_authority != (display, kind):
            raise ContributionError(
                "reserved owner identity requires exact display and contribution kind"
            )
        if owner_authority is None and kind == "maintainer":
            raise ContributionError(
                "contribution_kind maintainer requires the same owner display and "
                "maintainer kind in the reserved registry"
            )
        if (
            type(owner["code_license"]) is not str
            or owner["code_license"] != "Apache-2.0"
        ):
            raise ContributionError("code_license must be exact string Apache-2.0")
        if (
            type(owner["content_license"]) is not str
            or owner["content_license"] != "CC-BY-4.0"
        ):
            raise ContributionError("content_license must be exact string CC-BY-4.0")
        entries = _enumerate(owner_dir, "owner root")
        allowed_files = {"OWNER.yaml", "__init__.py"}
        for entry in entries:
            mode = _mode(entry, "owner root entry")
            if entry.name == "__pycache__" and stat.S_ISDIR(mode):
                _check_bytecode_cache(entry, "owner root")
                continue
            if stat.S_ISLNK(mode) or (
                not stat.S_ISDIR(mode) and entry.name not in allowed_files
            ):
                raise ContributionError(
                    f"owner root entry {entry} must be an allowed regular file "
                    "or an operation real directory"
                )
        init_tree = _parse_python(owner_dir / "__init__.py")
        _check_inert_init(owner_dir / "__init__.py", init_tree)
        owners.add(owner_id)
        operation_names: set[str] = set()
        for operation_dir in sorted(
            entry
            for entry in entries
            if stat.S_ISDIR(_mode(entry, "owner root entry"))
            and entry.name != "__pycache__"
        ):
            if operation_dir.is_symlink():
                raise ContributionError(
                    f"symlinked operation package is forbidden: {operation_dir}"
                )
            operation_name = _safe_python_component(
                operation_dir.name, "operation directory name"
            )
            operation_names.add(operation_name)
            manifest = _load(operation_dir / "CONTRIBUTION.yaml", OPERATION_FIELDS)
            if (
                type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != 1
                or manifest["owner_id"] != owner_id
            ):
                raise ContributionError(
                    f"{operation_dir}: operation owner/schema mismatch"
                )
            pack_id = _operation_identity(
                manifest["pack_id"], "pack_id operation identity"
            )
            operation_id = _operation_identity(
                manifest["operation_id"], "operation_id operation identity"
            )
            operation_type = _operation_identity(
                manifest["operation_type"], "operation_type operation identity"
            )
            raw_aliases = manifest["aliases"]
            if type(raw_aliases) is not list:
                raise ContributionError("aliases must be a list")
            aliases = tuple(
                _operation_identity(alias, "alias operation identity")
                for alias in raw_aliases
            )
            if len(set(aliases)) != len(aliases) or tuple(sorted(aliases)) != aliases:
                raise ContributionError("aliases must be sorted and unique")
            module = _text(manifest["module"], "module")
            expected_module = (
                f"operatebench.contributions.{owner_id}.{operation_dir.name}"
            )
            if module != expected_module:
                raise ContributionError(
                    f"{operation_dir}: module mismatch; expected {expected_module}"
                )
            source_path_text = _text(manifest["source_path"], "source_path")
            candidates = {
                id(candidate_row): candidate_row
                for identity in (pack_id, operation_type, operation_id, *aliases)
                if (candidate_row := reserved_by_identity.get(identity)) is not None
            }
            if candidates:
                exact = {
                    "owner_id": owner_id,
                    "owner_display_name": display,
                    "contribution_kind": kind,
                    "pack_id": pack_id,
                    "operation_type": operation_type,
                    "operation_id": operation_id,
                    "aliases": aliases,
                    "source_module": module + ".pack",
                    "source_path": source_path_text,
                }
                if len(candidates) != 1 or exact != next(iter(candidates.values())):
                    raise ContributionError(
                        f"{operation_dir}: reserved operation identity requires an "
                        "exact registered row"
                    )
            else:
                for value, identity_kind in (
                    (pack_id, "pack_id"),
                    (operation_type, "operation_type"),
                    *((alias, "alias") for alias in aliases),
                    (operation_id, "operation_id"),
                ):
                    _claim_operation_identity(
                        identity_owners, value, pack_id, identity_kind
                    )
            if kind == "partner" and not pack_id.startswith(f"partner.{owner_id}."):
                raise ContributionError(
                    f"partner pack_id must start with partner.{owner_id}."
                )
            prefixes = {
                "source_path": (
                    f"src/operatebench/contributions/{owner_id}/{operation_dir.name}",
                    True,
                ),
                "test_path": (
                    f"tests/contributions/{owner_id}/{operation_dir.name}",
                    False,
                ),
                "fixture_path": (
                    f"examples/contributions/{owner_id}/{operation_dir.name}",
                    False,
                ),
                "documentation_path": (f"docs/contributions/{owner_id}", False),
            }
            paths: dict[str, Path] = {}
            for field, (prefix, directory) in prefixes.items():
                path, pure = _declared_path(
                    root, manifest[field], field, prefix, directory=directory
                )
                if any(_overlaps(pure, other) for other in claimed):
                    raise ContributionError(
                        f"declared contribution path {pure!s} overlaps another claim"
                    )
                claimed.append(pure)
                paths[field] = path
            if paths["source_path"] != operation_dir:
                raise ContributionError(f"{operation_dir}: source_path mismatch")
            test_root, _test_root_pure = _declared_path(
                root,
                f"tests/contributions/{owner_id}/{operation_dir.name}",
                "canonical test root",
                f"tests/contributions/{owner_id}/{operation_dir.name}",
                directory=True,
            )
            if paths["test_path"].suffix != ".py":
                raise ContributionError("test_path must be a regular .py file")
            trees = _scan_public_tree(
                operation_dir, "operation package", allow_canonical_cache=True
            )
            test_trees = _scan_public_tree(
                test_root, "operation test root", allow_canonical_cache=True
            )
            test_init = test_root / "__init__.py"
            if test_init not in test_trees:
                raise ContributionError(
                    "operation test root requires regular __init__.py"
                )
            _check_inert_init(test_init, test_trees[test_init])
            if paths["test_path"] not in test_trees:
                raise ContributionError("test_path must be a scanned regular .py file")
            trees.update(test_trees)
            _read_bounded(paths["fixture_path"], DECLARATION_LIMIT)
            _read_bounded(paths["documentation_path"], DECLARATION_LIMIT)
            for path, tree in trees.items():
                if path.is_relative_to(test_root):
                    current = _module_for_path(
                        test_root,
                        path,
                        f"tests.contributions.{owner_id}.{operation_dir.name}",
                    )
                else:
                    current = _module_for_path(operation_dir, path, module)
                _check_ast(
                    path,
                    tree,
                    owner_id,
                    module,
                    current,
                    is_test=path.is_relative_to(test_root),
                )
            pack_tree = trees.get(operation_dir / "pack.py")
            if pack_tree is None:
                raise ContributionError(
                    f"{operation_dir}: declared pack module pack.py is missing"
                )
            _metadata_contract(
                pack_tree,
                operation_dir,
                {
                    "owner_id": owner_id,
                    "owner_display_name": display,
                    "contribution_kind": kind,
                    "pack_id": pack_id,
                    "operation_type": operation_type,
                    "operation_id": operation_id,
                    "aliases": aliases,
                    "default_spec": manifest["fixture_path"],
                    "status": "incubator",
                    "privacy_status": "SYNTHETIC_ONLY",
                    "evidence_eligible": False,
                },
            )
            checked.append(f"{owner_id}:{operation_id}")
        _check_owner_test_root(root, owner_id, operation_names)
    return tuple(checked)


def main() -> int:
    try:
        root = Path.cwd()
        check_tracked_contribution_artifacts(root)
        checked = check_contributions(root)
    except ContributionError as exc:
        print(f"contribution isolation check failed: {exc}", file=sys.stderr)
        return 1
    print(f"contribution isolation check passed: {len(checked)} operation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
