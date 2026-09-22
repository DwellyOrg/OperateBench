# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Git-independent physical contribution-tree primitives."""

from __future__ import annotations

import ast
import keyword
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from yaml.events import AliasEvent, NodeEvent

PYTHON_COMPONENT_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
OPERATION_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
PACK_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
WINDOWS_RESERVED_COMPONENTS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
CPYTHON_CACHE_FILE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.cpython-[0-9]{2,3}"
    r"(?:\.opt-[12]|-pytest-[0-9]+\.[0-9]+\.[0-9]+)?\.pyc\Z"
)
LITERAL_NODE_LIMIT = 16_384
DIRECTORY_ENTRY_LIMIT = 4096
PUBLIC_TREE_ENTRY_LIMIT = 16_384
PUBLIC_TREE_DEPTH_LIMIT = 64
PUBLIC_RESOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".md", ".txt"})
RESERVED_ROW_FIELDS = (
    "owner_id",
    "owner_display_name",
    "contribution_kind",
    "pack_id",
    "operation_type",
    "operation_id",
    "aliases",
    "source_module",
    "source_path",
)


class ContributionStructureError(ValueError):
    """A semantic violation in already-loaded contribution data."""


class DirectoryEntryLimitError(ContributionStructureError):
    """A directory exceeded the bounded enumeration limit."""


class PublicTreeLimitError(ContributionStructureError):
    """A public tree exceeded its aggregate entry or depth limit."""


def is_operation_identity(value: object) -> bool:
    return (
        type(value) is str
        and len(value) <= 128
        and OPERATION_ID_RE.fullmatch(value) is not None
    )


def is_pack_version(value: object) -> bool:
    return type(value) is str and PACK_VERSION_RE.fullmatch(value) is not None


def is_safe_display_name(value: object) -> bool:
    return (
        type(value) is str
        and bool(value.strip())
        and value == value.strip()
        and len(value) <= 128
        and not any(unicodedata.category(char).startswith("C") for char in value)
    )


def is_safe_owner_display_name(value: object) -> bool:
    return (
        is_safe_display_name(value)
        and type(value) is str
        and unicodedata.normalize("NFC", value) == value
    )


def canonical_owner_display_name_key(value: str) -> str:
    if type(value) is not str:
        raise TypeError("owner display identity requires an exact str")
    return unicodedata.normalize("NFC", value).casefold()


def is_literal_tree(value: object) -> bool:
    """Recognise an acyclic exact plain-literal tree within a fixed node budget."""
    pending = [value]
    scheduled = 1
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) in (str, int, float, bool, type(None)):
            continue
        if type(item) is list:
            children = len(item)
        elif type(item) is dict:
            children = len(item) * 2
        else:
            return False
        if id(item) in seen:
            return False
        seen.add(id(item))
        if scheduled + children > LITERAL_NODE_LIMIT:
            return False
        scheduled += children
        if type(item) is list:
            pending.extend(item)
        else:
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                return False
            pending.extend(mapping.values())
    return True


def yaml_event_violation(event: object) -> str | None:
    """Describe a forbidden YAML graph feature, if present."""
    if isinstance(event, AliasEvent):
        return "alias"
    if isinstance(event, NodeEvent) and (
        event.anchor is not None or getattr(event, "tag", None) is not None
    ):
        return "anchor or explicit tag"
    return None


def claim_operation_identity(
    owners: dict[str, tuple[str, str]], value: str, pack_id: str, kind: str
) -> None:
    """Apply the single compatibility rule for operation identity claims."""
    prior = owners.get(value)
    if prior is None:
        owners[value] = (pack_id, kind)
        return
    compatible = {prior[1], kind} in (
        {"pack_id", "operation_type"},
        {"alias", "operation_type"},
    )
    if prior[0] == pack_id and compatible:
        return
    if prior[1] == kind:
        raise ContributionStructureError(
            f"duplicate {kind} {value!r}: claimed by {prior[0]!r} and "
            f"{pack_id!r} as {kind}"
        )
    raise ContributionStructureError(
        f"operation identity {value!r} is {prior[1]} for {prior[0]!r} "
        f"and {kind} for {pack_id!r}"
    )


def validate_reserved_registry(manifest: object) -> tuple[Mapping[str, object], ...]:
    """Validate loaded reserved-registry data and return immutable normalized rows."""
    if type(manifest) is not dict or tuple(manifest) != ("schema_version", "operations"):
        raise ContributionStructureError("reserved registry fields are invalid")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ContributionStructureError(
            "reserved registry schema_version must be integer 1"
        )
    operations = manifest["operations"]
    if type(operations) is not list or not operations:
        raise ContributionStructureError(
            "reserved registry operations must be a non-empty list"
        )
    rows: list[Mapping[str, object]] = []
    identities: dict[str, tuple[str, str]] = {}
    displays: dict[str, str] = {}
    authorities: dict[str, tuple[str, str]] = {}
    modules: set[str] = set()
    paths: set[str] = set()
    packs: set[str] = set()
    for index, raw in enumerate(operations):
        prefix = f"operations[{index}]"
        if type(raw) is not dict or tuple(raw) != RESERVED_ROW_FIELDS:
            raise ContributionStructureError(
                f"reserved registry row {index} fields must be exactly "
                f"{list(RESERVED_ROW_FIELDS)} in order"
            )
        owner = raw["owner_id"]
        display = raw["owner_display_name"]
        kind = raw["contribution_kind"]
        if not is_safe_python_component(owner):
            raise ContributionStructureError(
                f"{prefix}.owner_id is not a safe lowercase Python identifier"
            )
        if not is_safe_owner_display_name(display):
            raise ContributionStructureError(
                f"{prefix}.owner_display_name must be non-empty safe text of at "
                "most 128 characters"
            )
        if type(kind) is not str or kind not in ("maintainer", "partner"):
            raise ContributionStructureError(
                f"{prefix}.contribution_kind must be maintainer or partner"
            )
        display_key = canonical_owner_display_name_key(display)
        if displays.setdefault(display_key, owner) != owner:
            raise ContributionStructureError(
                f"reserved owner display name {display!r} is claimed by different "
                "owner IDs"
            )
        authority = (display, kind)
        if authorities.setdefault(owner, authority) != authority:
            raise ContributionStructureError("inconsistent reserved owner authority rows")
        normalized: dict[str, object] = {name: raw[name] for name in RESERVED_ROW_FIELDS}
        for field in ("pack_id", "operation_type", "operation_id"):
            value = raw[field]
            if not is_operation_identity(value):
                raise ContributionStructureError(
                    f"{prefix}.{field} must be a lowercase bounded operation identity"
                )
        pack_id = raw["pack_id"]
        if pack_id in packs:
            raise ContributionStructureError("duplicate reserved pack row")
        packs.add(pack_id)
        aliases = raw["aliases"]
        if type(aliases) is not list or any(
            not is_operation_identity(alias) for alias in aliases
        ):
            raise ContributionStructureError(
                f"{prefix}.aliases must be a list of operation identity values"
            )
        alias_tuple = tuple(aliases)
        if alias_tuple != tuple(sorted(set(alias_tuple))):
            raise ContributionStructureError(
                f"{prefix}.aliases must be sorted and unique"
            )
        normalized["aliases"] = alias_tuple
        source_module = raw["source_module"]
        source_path = raw["source_path"]
        if (
            type(source_module) is not str
            or not source_module
            or source_module != source_module.strip()
            or len(source_module) > 256
        ):
            raise ContributionStructureError(
                f"{prefix}.source_module must be non-empty bounded text"
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in source_module):
            raise ContributionStructureError(
                f"{prefix}.source_module contains unsafe text"
            )
        if (
            type(source_path) is not str
            or not source_path
            or source_path != source_path.strip()
            or len(source_path) > 256
        ):
            raise ContributionStructureError(
                f"{prefix}.source_path must be non-empty bounded text"
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in source_path):
            raise ContributionStructureError(f"{prefix}.source_path contains unsafe text")
        pure = PurePosixPath(source_path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in source_path
            or pure.as_posix() != source_path
        ):
            raise ContributionStructureError(f"{prefix}.source_path must be normalized")
        expected = "src/" + source_module.removesuffix(".pack").replace(".", "/")
        if not source_module.endswith(".pack") or source_path != expected:
            raise ContributionStructureError(
                f"operations[{index}] source module/path mapping is not canonical"
            )
        if source_module in modules or source_path in paths:
            raise ContributionStructureError(f"duplicate reserved source in row {index}")
        modules.add(source_module)
        paths.add(source_path)
        for value, identity_kind in (
            (pack_id, "pack_id"),
            (raw["operation_type"], "operation_type"),
            *((alias, "alias") for alias in alias_tuple),
            (raw["operation_id"], "operation_id"),
        ):
            claim_operation_identity(identities, value, pack_id, identity_kind)
        rows.append(MappingProxyType(normalized))
    pack_ids = tuple(cast(str, row["pack_id"]) for row in rows)
    if pack_ids != tuple(sorted(pack_ids)):
        raise ContributionStructureError(
            "reserved registry operations must be sorted by pack_id"
        )
    return tuple(rows)


def is_safe_python_component(value: object) -> bool:
    """Return whether *value* is one canonical physical Python component."""
    return (
        type(value) is str
        and PYTHON_COMPONENT_RE.fullmatch(value) is not None
        and not keyword.iskeyword(value)
        and value not in WINDOWS_RESERVED_COMPONENTS
    )


def bounded_directory_entries(path: Path) -> tuple[Path, ...]:
    """Enumerate at most the shared cap plus one without unbounded allocation."""
    entries: list[Path] = []
    try:
        for entry in path.iterdir():
            entries.append(entry)
            if len(entries) > DIRECTORY_ENTRY_LIMIT:
                raise DirectoryEntryLimitError(f"too many entries in directory: {path}")
    except ContributionStructureError:
        raise
    except OSError as exc:
        raise ContributionStructureError(f"cannot enumerate directory: {path}") from exc
    return tuple(sorted(entries))


def validate_public_tree(
    path: Path, *, allow_canonical_cache: bool = False
) -> tuple[Path, ...]:
    """Enforce the finite pre-beta text-resource policy on one owner-scoped tree."""
    files: list[Path] = []
    pending = [(path, 0)]
    retained = 0

    def entries_within_remaining(directory: Path) -> tuple[Path, ...]:
        entries: list[Path] = []
        try:
            for entry in directory.iterdir():
                entries.append(entry)
                if len(entries) > DIRECTORY_ENTRY_LIMIT:
                    raise DirectoryEntryLimitError(
                        f"too many entries in directory: {directory}"
                    )
                if len(entries) > PUBLIC_TREE_ENTRY_LIMIT - retained:
                    raise PublicTreeLimitError(
                        f"too many aggregate entries in public tree: {path}"
                    )
        except ContributionStructureError:
            raise
        except OSError as exc:
            raise ContributionStructureError(
                f"cannot enumerate directory: {directory}"
            ) from exc
        return tuple(sorted(entries))

    def consume(entry: Path) -> None:
        nonlocal retained
        if retained >= PUBLIC_TREE_ENTRY_LIMIT:
            raise PublicTreeLimitError(
                f"too many aggregate entries in public tree: {path}"
            )
        retained += 1

    while pending:
        directory, depth = pending.pop()
        for entry in entries_within_remaining(directory):
            consume(entry)
            try:
                mode = entry.lstat().st_mode
            except OSError as exc:
                raise ContributionStructureError(
                    f"cannot inspect public entry: {entry}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise ContributionStructureError(f"nested symlink is forbidden: {entry}")
            if entry.name == "__pycache__" and stat.S_ISDIR(mode):
                if not allow_canonical_cache:
                    raise ContributionStructureError(
                        f"bytecode cache is forbidden in public tree: {entry}"
                    )
                cache_entries = entries_within_remaining(entry)
                for cache_entry in cache_entries:
                    consume(cache_entry)
                invalid = invalid_cache_entries(cache_entries)
                if invalid:
                    raise ContributionStructureError(
                        f"non-canonical bytecode cache entry is forbidden: {invalid[0]}"
                    )
                continue
            if entry.name.startswith("."):
                raise ContributionStructureError(f"hidden entry is forbidden: {entry}")
            if stat.S_ISDIR(mode):
                child_depth = depth + 1
                if child_depth > PUBLIC_TREE_DEPTH_LIMIT:
                    raise PublicTreeLimitError(f"public tree is too deep: {entry}")
                pending.append((entry, child_depth))
                continue
            if not stat.S_ISREG(mode):
                raise ContributionStructureError(
                    f"non-regular entry is forbidden: {entry}"
                )
            if entry.suffix not in PUBLIC_RESOURCE_SUFFIXES:
                raise ContributionStructureError(f"resource suffix is forbidden: {entry}")
            files.append(entry)
    return tuple(files)


def invalid_cache_entries(entries: Iterable[Path]) -> tuple[Path, ...]:
    """Return non-canonical immediate entries from a CPython bytecode cache."""
    invalid: list[Path] = []
    for entry in entries:
        try:
            mode = entry.lstat().st_mode
        except OSError:
            invalid.append(entry)
            continue
        if not stat.S_ISREG(mode) or CPYTHON_CACHE_FILE_RE.fullmatch(entry.name) is None:
            invalid.append(entry)
    return tuple(invalid)


def is_inert_python_init(tree: ast.Module) -> bool:
    """Return whether an owner initializer is empty apart from one docstring."""
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        body.pop(0)
    return not body


__all__ = [
    "CPYTHON_CACHE_FILE_RE",
    "DIRECTORY_ENTRY_LIMIT",
    "LITERAL_NODE_LIMIT",
    "PUBLIC_RESOURCE_SUFFIXES",
    "PUBLIC_TREE_DEPTH_LIMIT",
    "PUBLIC_TREE_ENTRY_LIMIT",
    "PYTHON_COMPONENT_RE",
    "RESERVED_ROW_FIELDS",
    "WINDOWS_RESERVED_COMPONENTS",
    "ContributionStructureError",
    "DirectoryEntryLimitError",
    "PublicTreeLimitError",
    "bounded_directory_entries",
    "canonical_owner_display_name_key",
    "claim_operation_identity",
    "invalid_cache_entries",
    "is_inert_python_init",
    "is_literal_tree",
    "is_operation_identity",
    "is_pack_version",
    "is_safe_display_name",
    "is_safe_owner_display_name",
    "is_safe_python_component",
    "validate_public_tree",
    "validate_reserved_registry",
    "yaml_event_violation",
]
