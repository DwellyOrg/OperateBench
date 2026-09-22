# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Create one complete, repository-owned partner contribution transaction."""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from operatebench._contribution_structure import (
    LITERAL_NODE_LIMIT,
    ContributionStructureError,
    DirectoryEntryLimitError,
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
    is_safe_owner_display_name,
)
from operatebench.sdk.errors import OperationScaffoldError
from operatebench.sdk.scaffold import (
    _TEMPLATE_NAMES,
    _TEMPLATE_PACKAGE,
    _directory_flags,
    _identity,
    _open_real_directory,
    _supports_secure_directory_fds,
)

_MANIFEST_LIMIT = 16 * 1024
_PYTHON_SUPPORT_LIMIT = 1024 * 1024
_OWNER_FIELDS = (
    "schema_version",
    "owner_id",
    "owner_display_name",
    "contribution_kind",
    "code_license",
    "content_license",
)
_CONTRIBUTION_FIELDS = (
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
_RESERVED_FIELDS = ("schema_version", "operations")
_LICENSES = ("Apache-2.0", "CC-BY-4.0")
_SPDX_LICENSE_TAG = "SPDX-License-" + "Identifier:"
# A 16 KiB manifest cannot usefully contain this many YAML nodes.  Keep the
# direct helper bounded too, because callers can pass already-constructed values.
_LITERAL_NODE_LIMIT = LITERAL_NODE_LIMIT
_READ_QUIESCENCE_SECONDS = 0.01


@dataclass(frozen=True)
class CompanyScaffoldResult:
    """Machine-readable result containing repository-relative paths only."""

    created_paths: tuple[str, ...]
    reused_owner_declaration: str | None
    registered: bool = False
    runnable: bool = False
    evidence_eligible: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "created_paths": list(self.created_paths),
            "reused_owner_declaration": self.reused_owner_declaration,
            "registered": self.registered,
            "runnable": self.runnable,
            "evidence_eligible": self.evidence_eligible,
        }


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if type(key) is not str:
            raise OperationScaffoldError(f"manifest key {key!r} must be text")
        if key in result:
            raise OperationScaffoldError(f"duplicate manifest key {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _is_literal_tree(value: object) -> bool:
    return is_literal_tree(value)


def _bounded_stable_regular_file(path: Path, limit: int, context: str) -> bytes:
    """Read an untrusted regular file three times through one bounded descriptor."""
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OperationScaffoldError(
            f"cannot open bounded regular non-symlink {context}: {path}"
        ) from exc
    try:
        try:
            stable_fields = (
                "st_mode",
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )

            def snapshot() -> tuple[int, ...]:
                observed = os.fstat(descriptor)
                values: list[int] = []
                for name in stable_fields:
                    value = getattr(observed, name, None)
                    if type(value) is not int:
                        raise OperationScaffoldError(
                            f"{context} {path} lacks required descriptor statistic {name}"
                        )
                    values.append(value)
                if not stat.S_ISREG(values[0]) or values[4] < 0 or values[4] > limit:
                    raise OperationScaffoldError(
                        f"{context} {path} must be a regular file no larger than {limit}"
                    )
                return tuple(values)

            def read_pass(size: int) -> bytes:
                body = bytearray()
                while len(body) < size:
                    chunk = os.read(
                        descriptor,
                        min(65536, size - len(body)),
                    )
                    if not chunk:
                        break
                    body.extend(chunk)
                return bytes(body)

            expected: tuple[int, ...] | None = None
            body: bytes | None = None
            for pass_number in range(3):
                if pass_number:
                    time.sleep(_READ_QUIESCENCE_SECONDS)
                    if os.lseek(descriptor, 0, os.SEEK_SET) != 0:
                        raise OperationScaffoldError(f"cannot seek {context}: {path}")
                before = snapshot()
                if expected is None:
                    expected = before
                elif before != expected:
                    raise OperationScaffoldError(
                        f"{context} {path} changed during stable read"
                    )
                current = read_pass(before[4])
                after = snapshot()
                if len(current) != before[4] or after != before:
                    phase = "bounded" if pass_number == 0 else "stable"
                    raise OperationScaffoldError(
                        f"{context} {path} changed during {phase} read"
                    )
                if body is None:
                    body = current
                elif current != body:
                    raise OperationScaffoldError(
                        f"{context} {path} changed during stable read"
                    )
            assert body is not None
            return body
        except OSError as exc:
            raise OperationScaffoldError(
                f"cannot complete bounded stable read of {context}: {path}"
            ) from exc
    finally:
        os.close(descriptor)


def _stable_manifest(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    body = _bounded_stable_regular_file(path, _MANIFEST_LIMIT, "manifest")
    try:
        text = body.decode("utf-8")
        for event in yaml.parse(text):
            if violation := yaml_event_violation(event):
                raise OperationScaffoldError(
                    f"manifest {path} contains a YAML {violation}"
                )
        value = yaml.load(text, Loader=_UniqueLoader)
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise OperationScaffoldError(f"cannot parse manifest {path}: {exc}") from exc
    if type(value) is not dict or tuple(value) != fields:
        raise OperationScaffoldError(
            f"manifest {path} fields must be exactly {list(fields)} in order"
        )
    if not _is_literal_tree(value):
        raise OperationScaffoldError(f"manifest {path} is not a plain literal tree")
    return value


def _authority_manifest_snapshot(root: Path) -> dict[str, bytes]:
    """Capture bounded bytes for authority manifests validated by the scanner."""
    paths = [root / "src/operatebench/resources/operation_pack_registry.yaml"]
    contribution_root = root / "src/operatebench/contributions"
    for owner_dir in _entries(contribution_root, "contribution root snapshot"):
        if owner_dir.name in {"__init__.py", "__pycache__"}:
            continue
        paths.append(owner_dir / "OWNER.yaml")
        for operation_dir in _entries(owner_dir, "source owner root snapshot"):
            if operation_dir.name in {"OWNER.yaml", "__init__.py", "__pycache__"}:
                continue
            paths.append(operation_dir / "CONTRIBUTION.yaml")
    return {
        path.relative_to(root).as_posix(): _bounded_stable_regular_file(
            path, _MANIFEST_LIMIT, "authority manifest snapshot"
        )
        for path in paths
    }


def _component(value: object, field: str) -> str:
    if not is_safe_python_component(value):
        raise OperationScaffoldError(
            f"{field} must be a safe lowercase ASCII Python component"
        )
    assert isinstance(value, str)
    return value


def _entries(path: Path, context: str) -> list[Path]:
    try:
        return list(bounded_directory_entries(path))
    except ContributionStructureError as exc:
        if isinstance(exc, DirectoryEntryLimitError):
            raise OperationScaffoldError(
                f"too many entries in {context}: {path}"
            ) from exc
        raise OperationScaffoldError(f"cannot enumerate {context}: {path}") from exc


def _mode(path: Path, context: str) -> int:
    try:
        return path.lstat().st_mode
    except OSError as exc:
        raise OperationScaffoldError(f"cannot inspect {context}: {path}") from exc


def _check_cache(path: Path, context: str) -> None:
    if not stat.S_ISDIR(_mode(path, context)):
        raise OperationScaffoldError(f"{context} cache must be a real directory")
    invalid = invalid_cache_entries(_entries(path, f"{context} cache"))
    if invalid:
        raise OperationScaffoldError(
            f"non-canonical entry in {context} bytecode cache: {invalid[0]}"
        )


def _regular(path: Path, context: str) -> None:
    if not stat.S_ISREG(_mode(path, context)):
        raise OperationScaffoldError(f"{context} must be a regular non-symlink file")


def _scan_public_tree(
    path: Path, context: str, *, allow_canonical_cache: bool = False
) -> set[Path]:
    """Apply the trusted checker's public resource shape rules."""
    try:
        return set(
            validate_public_tree(path, allow_canonical_cache=allow_canonical_cache)
        )
    except ContributionStructureError as exc:
        raise OperationScaffoldError(f"invalid {context}: {exc}") from exc


def _inert_init(path: Path, context: str) -> None:
    _regular(path, context)
    try:
        tree = ast.parse(
            _bounded_stable_regular_file(path, _PYTHON_SUPPORT_LIMIT, context),
            filename=str(path),
        )
    except (SyntaxError, ValueError) as exc:
        raise OperationScaffoldError(f"invalid {context}: {path}") from exc
    if not is_inert_python_init(tree):
        raise OperationScaffoldError(f"{context} must be inert")


def _validate_existing_layout(root: Path, operations: dict[str, set[str]]) -> None:
    """Validate the complete four-root physical layout before resource access."""
    roots = {
        "source": root / "src/operatebench/contributions",
        "test": root / "tests/contributions",
        "example": root / "examples/contributions",
        "documentation": root / "docs/contributions",
    }
    owner_sets: dict[str, set[str]] = {}
    for kind, shared in roots.items():
        owners: set[str] = set()
        for entry in _entries(shared, f"shared {kind} root"):
            mode = _mode(entry, f"shared {kind} root entry")
            if kind == "source" and entry.name == "__init__.py" and stat.S_ISREG(mode):
                continue
            if kind == "test" and entry.name == "__init__.py" and stat.S_ISREG(mode):
                _inert_init(entry, "shared test initializer")
                continue
            if entry.name == "__pycache__" and kind in {"source", "test"}:
                _check_cache(entry, f"shared {kind} root")
                continue
            if entry.name == "__pycache__":
                raise OperationScaffoldError(
                    f"bytecode cache is forbidden in shared {kind} root: {entry}"
                )
            if not stat.S_ISDIR(mode) or not is_safe_python_component(entry.name):
                raise OperationScaffoldError(
                    f"foreign or symlinked shared {kind} root entry: {entry}"
                )
            owners.add(entry.name)
        owner_sets[kind] = owners
    expected = set(operations)
    for kind, owners in owner_sets.items():
        if owners != expected:
            raise OperationScaffoldError(
                f"partial or ambiguous owner roots: source={sorted(expected)}, "
                f"{kind}={sorted(owners)}"
            )

    for owner_id, declared in operations.items():
        source_owner = roots["source"] / owner_id
        test_owner = roots["test"] / owner_id
        example_owner = roots["example"] / owner_id
        docs_owner = roots["documentation"] / owner_id
        init_path = source_owner / "__init__.py"
        _regular(init_path, "owner namespace initializer")
        try:
            init_body = _bounded_stable_regular_file(
                init_path,
                _PYTHON_SUPPORT_LIMIT,
                "owner namespace initializer",
            )
            init_tree = ast.parse(init_body, filename=str(init_path))
        except (OSError, SyntaxError, ValueError) as exc:
            raise OperationScaffoldError(
                "owner namespace initializer is invalid"
            ) from exc
        if not is_inert_python_init(init_tree):
            raise OperationScaffoldError("owner namespace initializer must be inert")

        source_ops: set[str] = set()
        for entry in _entries(source_owner, "source owner root"):
            mode = _mode(entry, "source owner root entry")
            if entry.name in {"OWNER.yaml", "__init__.py"} and stat.S_ISREG(mode):
                continue
            if entry.name == "__pycache__" and stat.S_ISDIR(mode):
                _check_cache(entry, "source owner root")
                continue
            if not stat.S_ISDIR(mode) or not is_safe_python_component(entry.name):
                raise OperationScaffoldError(
                    f"foreign or symlinked source owner entry: {entry}"
                )
            source_ops.add(entry.name)
        if source_ops != declared:
            raise OperationScaffoldError(
                "source owner operations disagree with declarations"
            )

        _inert_init(test_owner / "__init__.py", "owner test initializer")
        test_ops: set[str] = set()
        for entry in _entries(test_owner, "owner test root"):
            mode = _mode(entry, "owner test root entry")
            if entry.name == "__pycache__" and stat.S_ISDIR(mode):
                _check_cache(entry, "owner test root")
                continue
            if entry.name == "__init__.py" and stat.S_ISREG(mode):
                continue
            if entry.name == "conftest.py" and stat.S_ISREG(mode):
                try:
                    body = _bounded_stable_regular_file(
                        entry,
                        _PYTHON_SUPPORT_LIMIT,
                        "owner test support file",
                    )
                    compile(body, str(entry), "exec")
                except (OSError, SyntaxError, ValueError) as exc:
                    raise OperationScaffoldError(
                        f"invalid owner test support file: {entry}"
                    ) from exc
                continue
            if not stat.S_ISDIR(mode) or entry.name not in declared:
                raise OperationScaffoldError(
                    f"owner test root has foreign entry: {entry}"
                )
            test_ops.add(entry.name)
            _scan_public_tree(entry, "operation test root", allow_canonical_cache=True)
            _regular(entry / "test_pack.py", "declared operation test")
            _inert_init(entry / "__init__.py", "operation test initializer")
        if test_ops != declared:
            raise OperationScaffoldError(
                "owner test root operations disagree with source"
            )

        example_ops: set[str] = set()
        for entry in _entries(example_owner, "owner example root"):
            if (
                not stat.S_ISDIR(_mode(entry, "owner example entry"))
                or entry.name not in declared
            ):
                raise OperationScaffoldError(f"foreign owner example entry: {entry}")
            example_ops.add(entry.name)
            _scan_public_tree(entry, "operation example root")
            _regular(entry / "operation.yaml", "operation fixture")
        if example_ops != declared:
            raise OperationScaffoldError("owner example operations disagree with source")

        _scan_public_tree(docs_owner, "owner documentation root")
        for operation in declared:
            _regular(docs_owner / f"{operation}.md", "operation documentation")


def _real_directory(path: Path, context: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise OperationScaffoldError(
            f"{context} is missing or inaccessible: {path}"
        ) from exc
    if not stat.S_ISDIR(mode):
        raise OperationScaffoldError(f"{context} must be a real non-symlink directory")


def _claim(
    claims: dict[str, tuple[str, str]], identity: str, pack_id: str, kind: str
) -> None:
    try:
        claim_operation_identity(claims, identity, pack_id, kind)
    except ContributionStructureError as exc:
        raise OperationScaffoldError(str(exc)) from exc


def _validate_owner(owner: dict[str, Any], owner_id: str, display: str) -> None:
    expected = (1, owner_id, display, "partner", *_LICENSES)
    actual = tuple(owner[field] for field in _OWNER_FIELDS)
    if actual != expected:
        raise OperationScaffoldError(
            "existing OWNER declaration does not exactly match owner_id, "
            "owner_display_name, partner kind and required licenses"
        )


def _scan_authority(
    root: Path,
    owner_id: str,
    display: str,
    pack_id: str,
    operation_id: str,
    *,
    claim_requested: bool = True,
) -> bool:
    registry = _stable_manifest(
        root / "src/operatebench/resources/operation_pack_registry.yaml",
        _RESERVED_FIELDS,
    )
    try:
        reserved_rows = validate_reserved_registry(registry)
    except ContributionStructureError as exc:
        raise OperationScaffoldError(str(exc)) from exc
    claims: dict[str, tuple[str, str]] = {}
    displays: dict[str, str] = {}
    reserved_owners: dict[str, tuple[object, object]] = {}
    reserved_packs: dict[str, Mapping[str, object]] = {}
    for row in reserved_rows:
        scanned_reserved_owner = str(row["owner_id"])
        authority = (row["owner_display_name"], row["contribution_kind"])
        prior_authority = reserved_owners.get(scanned_reserved_owner)
        if prior_authority is not None and prior_authority != authority:
            raise OperationScaffoldError("inconsistent reserved owner authority rows")
        scanned_reserved_pack = str(row["pack_id"])
        reserved_owners[scanned_reserved_owner] = authority
        reserved_packs[scanned_reserved_pack] = row
        key = canonical_owner_display_name_key(str(row["owner_display_name"]))
        prior = displays.setdefault(key, str(row["owner_id"]))
        if prior != row["owner_id"]:
            raise OperationScaffoldError("reserved owner display identity is ambiguous")
        row_aliases = row["aliases"]
        if type(row_aliases) is not tuple:  # shared-validator invariant
            raise OperationScaffoldError("reserved aliases parser invariant failed")
        for identity, kind in (
            (row["pack_id"], "pack_id"),
            (row["operation_type"], "operation_type"),
            *((alias, "alias") for alias in row_aliases),
            (row["operation_id"], "operation_id"),
        ):
            _claim(claims, str(identity), str(row["pack_id"]), kind)
    if owner_id in reserved_owners:
        raise OperationScaffoldError("reserved owner IDs cannot use the partner scaffold")
    display_key = canonical_owner_display_name_key(display)
    if display_key in displays and displays[display_key] != owner_id:
        raise OperationScaffoldError(
            "owner display name is already claimed by another owner"
        )

    contribution_root = root / "src/operatebench/contributions"
    existing_owner = False
    operations: dict[str, set[str]] = {}
    for owner_dir in _entries(contribution_root, "contribution root"):
        mode = _mode(owner_dir, "contribution root entry")
        if owner_dir.name == "__pycache__":
            _check_cache(owner_dir, "contribution root")
            continue
        if owner_dir.name == "__init__.py" and stat.S_ISREG(mode):
            continue
        if not stat.S_ISDIR(mode):
            raise OperationScaffoldError(f"foreign contribution-root entry: {owner_dir}")
        owner = _stable_manifest(owner_dir / "OWNER.yaml", _OWNER_FIELDS)
        try:
            init_mode = _mode(owner_dir / "__init__.py", "owner namespace initializer")
        except OSError as exc:
            raise OperationScaffoldError(
                f"owner namespace initializer is missing: {owner_dir}"
            ) from exc
        if not stat.S_ISREG(init_mode):
            raise OperationScaffoldError(
                f"owner namespace initializer must be a regular file: {owner_dir}"
            )
        scanned_id = _component(owner["owner_id"], "owner_id")
        if scanned_id != owner_dir.name:
            raise OperationScaffoldError("owner directory and declaration disagree")
        scanned_display = owner["owner_display_name"]
        if not is_safe_owner_display_name(scanned_display):
            raise OperationScaffoldError("existing owner display name is unsafe")
        key = canonical_owner_display_name_key(scanned_display)
        prior = displays.setdefault(key, scanned_id)
        if prior != scanned_id:
            raise OperationScaffoldError("owner display name is claimed by different IDs")
        if scanned_id == owner_id:
            _validate_owner(owner, owner_id, display)
            existing_owner = True
        if (
            owner["schema_version"] != 1
            or owner["code_license"] != _LICENSES[0]
            or owner["content_license"] != _LICENSES[1]
            or owner["contribution_kind"] not in {"maintainer", "partner"}
        ):
            raise OperationScaffoldError("existing OWNER declaration is invalid")
        reserved_authority = reserved_owners.get(scanned_id)
        if reserved_authority is None and owner["contribution_kind"] != "partner":
            raise OperationScaffoldError("unreserved owners must be partner owners")
        if reserved_authority is not None and reserved_authority != (
            scanned_display,
            owner["contribution_kind"],
        ):
            raise OperationScaffoldError("reserved owner authority does not match")
        operation_names: set[str] = set()
        for operation_dir in _entries(owner_dir, "source owner root"):
            entry_mode = _mode(operation_dir, "source owner root entry")
            if operation_dir.name in {"OWNER.yaml", "__init__.py"} and stat.S_ISREG(
                entry_mode
            ):
                continue
            if operation_dir.name == "__pycache__" and stat.S_ISDIR(entry_mode):
                _check_cache(operation_dir, "source owner root")
                continue
            if not stat.S_ISDIR(entry_mode):
                raise OperationScaffoldError(
                    f"foreign or symlinked owner-root entry: {operation_dir}"
                )
            operation_name = _component(operation_dir.name, "operation directory")
            operation_names.add(operation_name)
            manifest_path = operation_dir / "CONTRIBUTION.yaml"
            if not manifest_path.exists() and not manifest_path.is_symlink():
                raise OperationScaffoldError(f"partial operation tree: {operation_dir}")
            manifest = _stable_manifest(manifest_path, _CONTRIBUTION_FIELDS)
            _scan_public_tree(
                operation_dir, "operation package", allow_canonical_cache=True
            )
            aliases = manifest["aliases"]
            if (
                type(aliases) is not list
                or any(type(alias) is not str for alias in aliases)
                or len(set(aliases)) != len(aliases)
                or aliases != sorted(aliases)
            ):
                raise OperationScaffoldError(
                    "contribution aliases must be a sorted unique list"
                )
            expected_paths = {
                "module": f"operatebench.contributions.{scanned_id}.{operation_name}",
                "source_path": (
                    f"src/operatebench/contributions/{scanned_id}/{operation_name}"
                ),
                "test_path": (
                    f"tests/contributions/{scanned_id}/{operation_name}/test_pack.py"
                ),
                "fixture_path": (
                    f"examples/contributions/{scanned_id}/{operation_name}/operation.yaml"
                ),
                "documentation_path": (
                    f"docs/contributions/{scanned_id}/{operation_name}.md"
                ),
            }
            if manifest["owner_id"] != scanned_id or any(
                manifest[field] != expected for field, expected in expected_paths.items()
            ):
                raise OperationScaffoldError(
                    f"contribution paths or owner are not canonical: {operation_dir}"
                )
            if manifest["schema_version"] != 1:
                raise OperationScaffoldError("contribution schema_version must be 1")
            reserved_row = reserved_packs.get(str(manifest["pack_id"]))
            if reserved_row is not None:
                exact_reserved = {
                    "owner_id": scanned_id,
                    "owner_display_name": scanned_display,
                    "contribution_kind": owner["contribution_kind"],
                    "pack_id": manifest["pack_id"],
                    "operation_type": manifest["operation_type"],
                    "operation_id": manifest["operation_id"],
                    "aliases": tuple(aliases),
                    "source_module": str(manifest["module"]) + ".pack",
                    "source_path": manifest["source_path"],
                }
                expected_reserved = dict(reserved_row)
                if exact_reserved != expected_reserved:
                    raise OperationScaffoldError(
                        "reserved contribution must exactly match its registry row"
                    )
                continue
            for identity, kind in (
                (manifest["pack_id"], "pack_id"),
                (manifest["operation_type"], "operation_type"),
                *((alias, "alias") for alias in aliases),
                (manifest["operation_id"], "operation_id"),
            ):
                if not is_operation_identity(identity):
                    raise OperationScaffoldError("contribution has an invalid identity")
                _claim(claims, str(identity), str(manifest["pack_id"]), kind)
        operations[scanned_id] = operation_names
    if display_key in displays and displays[display_key] != owner_id:
        raise OperationScaffoldError(
            "owner display name is already claimed by another owner"
        )
    _validate_existing_layout(root, operations)
    if claim_requested:
        _claim(claims, pack_id, pack_id, "pack_id")
        _claim(claims, pack_id, pack_id, "operation_type")
        _claim(claims, operation_id, pack_id, "operation_id")
    return existing_owner


def _yaml_document(payload: dict[str, object], copyright_text: str) -> str:
    return (
        f"# SPDX-FileCopyrightText: {copyright_text}\n"
        f"# {_SPDX_LICENSE_TAG} CC-BY-4.0\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    )


def _rendered_files(
    *, owner_id: str, display: str, operation_name: str, pack_id: str, operation_id: str
) -> dict[str, bytes]:
    values = {
        "PACK_ID": pack_id,
        "OPERATION_TYPE": pack_id,
        "OPERATION_ID": operation_id,
        "MODULE_NAME": operation_name,
        "OWNER_ID": owner_id,
        "OWNER_DISPLAY_NAME_LITERAL": json.dumps(display, ensure_ascii=False),
        "CONTRIBUTION_KIND": "partner",
    }
    resources = files(_TEMPLATE_PACKAGE)
    flat: dict[str, str] = {}
    token = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
    for template_name in _TEMPLATE_NAMES:
        template = resources.joinpath(template_name).read_text(encoding="utf-8")
        names = set(token.findall(template))
        if "{{" in token.sub("", template) or "}}" in token.sub("", template):
            raise OperationScaffoldError("an operation scaffold template is incomplete")
        try:
            rendered_template = token.sub(lambda match: values[match.group(1)], template)
        except KeyError as exc:
            raise OperationScaffoldError(
                "an operation scaffold template is incomplete"
            ) from exc
        if names - values.keys():
            raise OperationScaffoldError("an operation scaffold template is incomplete")
        flat[template_name.removesuffix(".tmpl")] = rendered_template
    source = f"src/operatebench/contributions/{owner_id}/{operation_name}"
    test = f"tests/contributions/{owner_id}/{operation_name}/test_pack.py"
    test_owner_init = f"tests/contributions/{owner_id}/__init__.py"
    test_operation_init = f"tests/contributions/{owner_id}/{operation_name}/__init__.py"
    fixture = f"examples/contributions/{owner_id}/{operation_name}/operation.yaml"
    documentation = f"docs/contributions/{owner_id}/{operation_name}.md"
    owner_payload: dict[str, object] = {
        "schema_version": 1,
        "owner_id": owner_id,
        "owner_display_name": display,
        "contribution_kind": "partner",
        "code_license": _LICENSES[0],
        "content_license": _LICENSES[1],
    }
    contribution: dict[str, object] = {
        "schema_version": 1,
        "owner_id": owner_id,
        "pack_id": pack_id,
        "operation_id": operation_id,
        "operation_type": pack_id,
        "aliases": [],
        "module": f"operatebench.contributions.{owner_id}.{operation_name}",
        "source_path": source,
        "test_path": test,
        "fixture_path": fixture,
        "documentation_path": documentation,
    }
    rendered = {
        f"src/operatebench/contributions/{owner_id}/OWNER.yaml": _yaml_document(
            owner_payload, "the contributor"
        ),
        f"src/operatebench/contributions/{owner_id}/__init__.py": (
            "# SPDX-FileCopyrightText: the contributor\n"
            f"# {_SPDX_LICENSE_TAG} Apache-2.0\n"
            '"""Inert partner contribution namespace."""\n'
        ),
        f"{source}/CONTRIBUTION.yaml": _yaml_document(contribution, "the contributor"),
        f"{source}/README.md": flat.pop("README.md"),
        f"{source}/__init__.py": flat.pop("__init__.py"),
        f"{source}/agents.py": flat.pop("agents.py"),
        f"{source}/evaluator.py": flat.pop("evaluator.py"),
        f"{source}/negative_control_oracle.yaml": flat.pop(
            "negative_control_oracle.yaml"
        ),
        f"{source}/operation.py": flat.pop("operation.py"),
        f"{source}/pack.py": flat.pop("pack.py"),
        f"{source}/spec.py": flat.pop("spec.py"),
        test_owner_init: (
            "# SPDX-FileCopyrightText: the contributor\n"
            f"# {_SPDX_LICENSE_TAG} Apache-2.0\n"
        ),
        test_operation_init: (
            "# SPDX-FileCopyrightText: the contributor\n"
            f"# {_SPDX_LICENSE_TAG} Apache-2.0\n"
        ),
        test: flat.pop("test_pack.py").replace(
            "from .pack import PACK",
            "from operatebench.contributions."
            f"{owner_id}.{operation_name}.pack import PACK",
        ),
        fixture: flat.pop("operation.yaml"),
        documentation: (
            "<!-- SPDX-FileCopyrightText: the contributor -->\n"
            f"<!-- {_SPDX_LICENSE_TAG} CC-BY-4.0 -->\n\n"
            f"# {pack_id}\n\nGuided, pre-beta partner operation scaffold. It is "
            "unregistered, unrunnable, and not eligible as benchmark evidence.\n"
        ),
    }
    if flat:
        raise OperationScaffoldError("unexpected operation scaffold template set")
    return {path: content.encode("utf-8") for path, content in rendered.items()}


def _write_file(parent_fd: int, name: str, content: bytes) -> tuple[int, int]:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    identity: tuple[int, int] | None = None
    try:
        identity = _identity(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, 0o644)
        if _identity(os.fstat(descriptor)) != identity:
            raise OperationScaffoldError(f"created file descriptor changed: {name}")
        return identity
    except BaseException:
        if identity is None:
            # Recover only through the still-open descriptor. Under persistent
            # descriptor blindness, retaining the entry is safer than unlinking
            # a public name whose ownership cannot be proved.
            with suppress(OSError):
                identity = _identity(os.fstat(descriptor))
        if identity is not None:
            with suppress(OSError):
                observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    stat.S_ISREG(observed.st_mode)
                    and _identity(observed) == identity
                    and observed.st_nlink == 1
                ):
                    os.unlink(name, dir_fd=parent_fd)
        raise
    finally:
        with suppress(OSError):
            os.close(descriptor)


@dataclass
class _CreatedDirectory:
    parent_fd: int
    child_fd: int
    name: str
    identity: tuple[int, int]
    display_path: Path


@dataclass
class _CreatedFile:
    parent_fd: int
    name: str
    identity: tuple[int, int]
    display_path: Path


def _open_relative_directory(root_fd: int, relative: str | Path) -> int:
    """Open a fixed relative directory without consulting an absolute pathname."""
    parts = Path(relative).parts
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            if part in ("", ".", ".."):
                raise OperationScaffoldError("transaction path is not canonical")
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _create_transaction(
    root: Path,
    root_fd: int,
    rendered: dict[str, bytes],
    verify_public_authority: Callable[[], None],
    verify_final_authority: Callable[[], None] | None = None,
) -> None:
    created_dirs: list[_CreatedDirectory] = []
    created_files: list[_CreatedFile] = []
    try:
        directories_needed: set[str] = set()
        for path in rendered:
            parent = Path(path).parent
            while parent != Path(".") and parent != Path(""):
                directories_needed.add(str(parent))
                parent = parent.parent
        directories = sorted(
            directories_needed,
            key=lambda value: (len(Path(value).parts), value),
        )
        for relative in directories:
            verify_public_authority()
            target = root / relative
            parent_relative = str(Path(relative).parent)
            parent_fd = _open_relative_directory(root_fd, parent_relative)
            child_fd: int | None = None
            try:
                try:
                    observed = os.stat(
                        target.name, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    observed = None
                if observed is not None:
                    if not stat.S_ISDIR(observed.st_mode):
                        raise OperationScaffoldError(
                            "existing scaffold directory is not a real directory: "
                            f"{target}"
                        )
                    continue
                verify_public_authority()
                os.mkdir(target.name, 0o700, dir_fd=parent_fd)
                parent_after_mkdir = os.fstat(parent_fd)
                child_fd = os.open(target.name, _directory_flags(), dir_fd=parent_fd)
                identity = _identity(os.fstat(child_fd))
                parent_after_open = os.fstat(parent_fd)
                parent_fields = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
                if tuple(
                    getattr(parent_after_mkdir, field) for field in parent_fields
                ) != tuple(getattr(parent_after_open, field) for field in parent_fields):
                    raise OperationScaffoldError(
                        "directory parent changed before descriptor proof: "
                        f"{target.parent}"
                    )
                created_dirs.append(
                    _CreatedDirectory(parent_fd, child_fd, target.name, identity, target)
                )
                parent_fd = -1
                child_fd = None
            finally:
                if child_fd is not None:
                    with suppress(OSError):
                        os.close(child_fd)
                if parent_fd >= 0:
                    with suppress(OSError):
                        os.close(parent_fd)
        for relative in sorted(rendered):
            verify_public_authority()
            target = root / relative
            parent_fd = _open_relative_directory(root_fd, Path(relative).parent)
            try:
                verify_public_authority()
                identity = _write_file(parent_fd, target.name, rendered[relative])
                created_files.append(
                    _CreatedFile(parent_fd, target.name, identity, target)
                )
                parent_fd = -1
            finally:
                if parent_fd >= 0:
                    with suppress(OSError):
                        os.close(parent_fd)

        def verify_created_entries(*, set_directory_modes: bool = False) -> None:
            for file_entry in created_files:
                observed = os.stat(
                    file_entry.name,
                    dir_fd=file_entry.parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(observed.st_mode)
                    or _identity(observed) != file_entry.identity
                    or observed.st_nlink != 1
                ):
                    raise OperationScaffoldError(
                        "created file was replaced or hardlinked: "
                        f"{file_entry.display_path}"
                    )
            for directory_entry in created_dirs:
                observed = os.stat(
                    directory_entry.name,
                    dir_fd=directory_entry.parent_fd,
                    follow_symlinks=False,
                )
                held = os.fstat(directory_entry.child_fd)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or not stat.S_ISDIR(held.st_mode)
                    or _identity(observed) != directory_entry.identity
                    or _identity(held) != directory_entry.identity
                ):
                    raise OperationScaffoldError(
                        f"created directory was replaced: {directory_entry.display_path}"
                    )
                if set_directory_modes:
                    os.fchmod(directory_entry.child_fd, 0o755)

        verify_created_entries(set_directory_modes=True)
        (verify_final_authority or verify_public_authority)()
        verify_created_entries()
    except BaseException:
        for file_entry in reversed(created_files):
            with suppress(OSError):
                observed = os.stat(
                    file_entry.name,
                    dir_fd=file_entry.parent_fd,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISREG(observed.st_mode)
                    and _identity(observed) == file_entry.identity
                    and observed.st_nlink == 1
                ):
                    os.unlink(file_entry.name, dir_fd=file_entry.parent_fd)
        for directory_entry in reversed(created_dirs):
            with suppress(OSError):
                observed = os.stat(
                    directory_entry.name,
                    dir_fd=directory_entry.parent_fd,
                    follow_symlinks=False,
                )
                held = os.fstat(directory_entry.child_fd)
                if (
                    stat.S_ISDIR(observed.st_mode)
                    and stat.S_ISDIR(held.st_mode)
                    and _identity(observed) == directory_entry.identity
                    and _identity(held) == directory_entry.identity
                ):
                    os.rmdir(directory_entry.name, dir_fd=directory_entry.parent_fd)
        raise
    finally:
        for file_entry in created_files:
            with suppress(OSError):
                os.close(file_entry.parent_fd)
        for directory_entry in created_dirs:
            with suppress(OSError):
                os.close(directory_entry.child_fd)
            with suppress(OSError):
                os.close(directory_entry.parent_fd)


def scaffold_company_operation(
    *,
    repository_root: str | Path,
    owner_id: str,
    owner_display_name: str,
    pack_id: str,
    operation_id: str,
    operation_name: str,
) -> CompanyScaffoldResult:
    """Create a complete partner contribution at canonical repository paths."""
    owner_id = _component(owner_id, "owner_id")
    operation_name = _component(operation_name, "operation_name")
    if not is_safe_owner_display_name(owner_display_name):
        raise OperationScaffoldError(
            "owner_display_name must be non-empty bounded NFC safe text"
        )
    if ("{{" in owner_display_name) != ("}}" in owner_display_name):
        raise OperationScaffoldError(
            "owner_display_name contains an unmatched template delimiter"
        )
    if not is_operation_identity(pack_id) or not is_operation_identity(operation_id):
        raise OperationScaffoldError(
            "pack_id and operation_id must be lowercase bounded identities"
        )
    if pack_id == operation_id:
        raise OperationScaffoldError(
            "pack_id and operation_id must be distinct operation identities"
        )
    if not pack_id.startswith(f"partner.{owner_id}."):
        raise OperationScaffoldError(
            f"partner pack_id must start with 'partner.{owner_id}.'"
        )
    if not operation_id.startswith(f"{owner_id}_"):
        raise OperationScaffoldError(
            f"partner operation_id must start with '{owner_id}_'"
        )
    if not _supports_secure_directory_fds():
        raise OperationScaffoldError(
            "safe scaffold creation requires secure directory-fd support"
        )
    supplied_root = os.fspath(repository_root)
    lexical_root = os.path.abspath(supplied_root)
    try:
        resolved_root = Path(os.path.realpath(supplied_root))
    except OSError as exc:
        raise OperationScaffoldError(
            f"cannot resolve repository_root: {lexical_root}"
        ) from exc
    root = Path(lexical_root)
    if resolved_root != root:
        raise OperationScaffoldError(
            "repository_root must not contain symlinks or lexical/realpath divergence"
        )
    _real_directory(root, "repository_root")
    root_fd = _open_real_directory(root)
    canonical_relatives = (
        "src/operatebench/contributions",
        "src/operatebench/resources",
        "tests/contributions",
        "examples/contributions",
        "docs/contributions",
    )
    canonical_fds: dict[str, int] = {}
    canonical_identities: dict[str, tuple[int, int]] = {}
    owner_relatives = (
        f"src/operatebench/contributions/{owner_id}",
        f"tests/contributions/{owner_id}",
        f"examples/contributions/{owner_id}",
        f"docs/contributions/{owner_id}",
    )
    owner_fds: dict[str, int] = {}
    owner_identities: dict[str, tuple[int, int]] = {}
    try:
        root_identity = _identity(os.fstat(root_fd))
        for relative in canonical_relatives:
            try:
                descriptor = _open_relative_directory(root_fd, relative)
            except OSError as exc:
                raise OperationScaffoldError(
                    f"cannot securely open canonical repository root: {relative}"
                ) from exc
            canonical_fds[relative] = descriptor
            canonical_identities[relative] = _identity(os.fstat(descriptor))

        def verify_root_authority() -> None:
            try:
                public = os.stat(root, follow_symlinks=False)
            except OSError as exc:
                raise OperationScaffoldError(
                    "repository_root no longer names the validated directory"
                ) from exc
            if not stat.S_ISDIR(public.st_mode) or _identity(public) != root_identity:
                raise OperationScaffoldError(
                    "repository_root no longer names the validated directory"
                )

        def verify_public_authority() -> None:
            verify_root_authority()
            expected_roots = canonical_identities | owner_identities
            for relative, expected_identity in expected_roots.items():
                try:
                    descriptor = _open_relative_directory(root_fd, relative)
                except OSError as exc:
                    raise OperationScaffoldError(
                        f"validated repository directory changed: {relative}"
                    ) from exc
                try:
                    observed = os.fstat(descriptor)
                    held_fd = (
                        owner_fds[relative]
                        if relative in owner_fds
                        else canonical_fds[relative]
                    )
                    held = os.fstat(held_fd)
                    if (
                        not stat.S_ISDIR(observed.st_mode)
                        or not stat.S_ISDIR(held.st_mode)
                        or _identity(observed) != expected_identity
                        or _identity(held) != expected_identity
                    ):
                        raise OperationScaffoldError(
                            f"validated repository directory changed: {relative}"
                        )
                finally:
                    os.close(descriptor)

        verify_public_authority()
        for relative in owner_relatives:
            try:
                descriptor = _open_relative_directory(root_fd, relative)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise OperationScaffoldError(
                    f"cannot securely inspect owner root: {relative}"
                ) from exc
            owner_fds[relative] = descriptor
            observed = os.fstat(descriptor)
            if not stat.S_ISDIR(observed.st_mode):
                raise OperationScaffoldError(
                    f"owner root must be a real directory: {relative}"
                )
            owner_identities[relative] = _identity(observed)
        if owner_fds and len(owner_fds) != len(owner_relatives):
            raise OperationScaffoldError("partial or ambiguous owner roots already exist")
        verify_public_authority()
        existing_owner = _scan_authority(
            root, owner_id, owner_display_name, pack_id, operation_id
        )
        preflight_authority = _authority_manifest_snapshot(root)
        verify_public_authority()
        owner_roots = (
            root / f"src/operatebench/contributions/{owner_id}",
            root / f"tests/contributions/{owner_id}",
            root / f"examples/contributions/{owner_id}",
            root / f"docs/contributions/{owner_id}",
        )
        if existing_owner != bool(owner_fds):
            raise OperationScaffoldError("partial or ambiguous owner roots already exist")
        operation_roots = (
            owner_roots[0] / operation_name,
            owner_roots[1] / operation_name,
            owner_roots[2] / operation_name,
        )
        if any(path.exists() or path.is_symlink() for path in operation_roots):
            raise OperationScaffoldError("partial or existing operation tree conflict")
        verify_public_authority()
        rendered = _rendered_files(
            owner_id=owner_id,
            display=owner_display_name,
            operation_name=operation_name,
            pack_id=pack_id,
            operation_id=operation_id,
        )
        verify_public_authority()
        owner_paths = {
            f"src/operatebench/contributions/{owner_id}/OWNER.yaml",
            f"src/operatebench/contributions/{owner_id}/__init__.py",
            f"tests/contributions/{owner_id}/__init__.py",
        }
        to_create = {
            path: body
            for path, body in rendered.items()
            if not (existing_owner and path in owner_paths)
        }
        for relative in to_create:
            target = root / relative
            if target.exists() or target.is_symlink():
                raise OperationScaffoldError(
                    f"target path conflict: {relative} already exists"
                )
        verify_public_authority()

        def verify_final_authority() -> None:
            verify_public_authority()
            _scan_authority(
                root,
                owner_id,
                owner_display_name,
                pack_id,
                operation_id,
                claim_requested=False,
            )
            final_authority = _authority_manifest_snapshot(root)
            if any(
                final_authority.get(relative) != body
                for relative, body in preflight_authority.items()
            ):
                raise OperationScaffoldError(
                    "authority manifest changed after preflight validation"
                )
            verify_public_authority()

        _create_transaction(
            root,
            root_fd,
            to_create,
            verify_public_authority,
            verify_final_authority,
        )
        return CompanyScaffoldResult(
            created_paths=tuple(sorted(to_create)),
            reused_owner_declaration=(
                f"src/operatebench/contributions/{owner_id}/OWNER.yaml"
                if existing_owner
                else None
            ),
        )
    finally:
        for descriptor in owner_fds.values():
            with suppress(OSError):
                os.close(descriptor)
        for descriptor in canonical_fds.values():
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.close(root_fd)


__all__ = ["CompanyScaffoldResult", "scaffold_company_operation"]
