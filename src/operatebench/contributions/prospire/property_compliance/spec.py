# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Strict, bounded loader for the invented property-compliance fixture."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from operatebench.core.errors import SpecSchemaError

from ._bounded_io import read_bounded

PACK_ID = "lettings.property_compliance.synthetic"
OPERATION_ID = "lettings_property_compliance_synthetic_v1"
VERSION = "0.1.0"
SEMANTIC_SCENARIO_ID = "property_compliance_certificate_renewal_synthetic_v1"
MAX_SPEC_BYTES = 64 * 1024
ROLES = {
    "tenant_synthetic": "tenant",
    "inspection_supplier_synthetic": "inspection_supplier",
    "compliance_reviewer_synthetic": "compliance_reviewer",
    "property_record_system_synthetic": "property_record_system",
    "certificate_registry_synthetic": "certificate_registry",
    "scheduler_synthetic": "scheduler",
    "agent_synthetic": "agent",
}
EXPECTED_SCENARIOS = {
    "V1": ("normal_renewal", "certificate_renewed_final"),
    "V2": ("remediation_reinspection", "certificate_renewed_final"),
    "V3": ("no_access_expiry_exception", "human_exception_final"),
}
TOP_FIELDS = {
    "schema_version",
    "operation_id",
    "operation_type",
    "operation_version",
    "semantic_scenario_id",
    "privacy_status",
    "disclaimer",
    "data_provenance",
    "actors",
    "scenarios",
}


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader: _Loader, node: yaml.MappingNode, deep: bool = False) -> Any:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            mark = key_node.start_mark
            raise SpecSchemaError(
                "spec mapping field names must be strings; "
                f"line {mark.line + 1}, column {mark.column + 1} names "
                f"{type(key).__name__}"
            )
        if key in result:
            raise SpecSchemaError(f"spec repeats field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _fields(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    non_text = sorted(repr(key) for key in value if type(key) is not str)
    if non_text:
        raise SpecSchemaError(
            f"{where}: mapping field names must be strings, got {non_text}"
        )
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise SpecSchemaError(f"{where}: unknown field(s) {unknown}")
    if missing:
        raise SpecSchemaError(f"{where}: missing field(s) {missing}")


@dataclass(frozen=True)
class OperationSpec:
    source: str
    operation_id: str
    operation_type: str
    operation_version: str
    semantic_scenario_id: str
    scenario_ids: tuple[str, ...]
    actors: Mapping[str, str]
    spec_digest_sha256: str


def load_spec(path: str | Path) -> OperationSpec:
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        if source.is_symlink():
            raise SpecSchemaError(f"{source}: spec symlinks are refused") from exc
        raise SpecSchemaError(f"{source}: cannot read spec: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SpecSchemaError(f"{source}: spec must be a regular file")
        if metadata.st_size > MAX_SPEC_BYTES:
            raise SpecSchemaError(f"{source}: spec exceeds {MAX_SPEC_BYTES}-byte limit")
        body = read_bounded(descriptor, MAX_SPEC_BYTES + 1)
    except OSError as exc:
        raise SpecSchemaError(f"{source}: cannot read spec: {exc}") from exc
    finally:
        os.close(descriptor)
    if len(body) > MAX_SPEC_BYTES:
        raise SpecSchemaError(f"{source}: spec exceeds {MAX_SPEC_BYTES}-byte limit")
    try:
        text = body.decode("utf-8")
        raw = yaml.load(text, Loader=_Loader)
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise SpecSchemaError(f"{source}: malformed YAML: {exc}") from exc
    if not isinstance(raw, Mapping) or any(type(k) is not str for k in raw):
        raise SpecSchemaError(f"{source}: spec must be a string-keyed mapping")
    _fields(raw, TOP_FIELDS, str(source))
    fixed = {
        "schema_version": 1,
        "operation_id": OPERATION_ID,
        "operation_type": PACK_ID,
        "operation_version": VERSION,
        "semantic_scenario_id": SEMANTIC_SCENARIO_ID,
        "privacy_status": "SYNTHETIC_ONLY",
    }
    for name, expected in fixed.items():
        if raw[name] != expected or type(raw[name]) is not type(expected):
            raise SpecSchemaError(f"{source}: {name} must be {expected!r}")
    disclaimer = raw["disclaimer"]
    provenance = raw["data_provenance"]
    if type(disclaimer) is not str or not all(
        term in disclaimer.casefold()
        for term in (
            "fully synthetic",
            "not policy",
            "legal advice",
            "benchmark evidence",
        )
    ):
        raise SpecSchemaError(
            f"{source}: disclaimer must preserve all synthetic exclusions"
        )
    if type(provenance) is not str or not all(
        term in provenance.casefold()
        for term in (
            "entirely invented",
            "no production data",
            "real people",
            "statutory names",
        )
    ):
        raise SpecSchemaError(
            f"{source}: data_provenance must preserve invented provenance"
        )
    actors = raw["actors"]
    if not isinstance(actors, Mapping):
        raise SpecSchemaError(
            f"{source}: actors must be exactly the invented authority roster"
        )
    _fields(actors, set(ROLES), f"{source}: actors")
    if dict(actors) != ROLES:
        raise SpecSchemaError(
            f"{source}: actors must be exactly the invented authority roster"
        )
    scenarios = raw["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise SpecSchemaError(f"{source}: scenarios must contain exactly V1, V2 and V3")
    seen: list[str] = []
    for index, item in enumerate(scenarios):
        if not isinstance(item, Mapping):
            raise SpecSchemaError(f"{source}: scenarios[{index}] must be a mapping")
        _fields(
            item,
            {"scenario_id", "path", "expected_terminal"},
            f"{source}: scenarios[{index}]",
        )
        scenario_id = item["scenario_id"]
        if scenario_id not in EXPECTED_SCENARIOS or scenario_id in seen:
            raise SpecSchemaError(
                f"{source}: invalid or repeated scenario_id {scenario_id!r}"
            )
        if (item["path"], item["expected_terminal"]) != EXPECTED_SCENARIOS[scenario_id]:
            raise SpecSchemaError(f"{source}: unsafe values for scenario {scenario_id}")
        seen.append(scenario_id)
    if tuple(seen) != tuple(EXPECTED_SCENARIOS):
        raise SpecSchemaError(f"{source}: scenarios must be ordered V1, V2, V3")
    return OperationSpec(
        source=str(source),
        operation_id=OPERATION_ID,
        operation_type=PACK_ID,
        operation_version=VERSION,
        semantic_scenario_id=SEMANTIC_SCENARIO_ID,
        scenario_ids=tuple(seen),
        actors=MappingProxyType(dict(ROLES)),
        spec_digest_sha256=hashlib.sha256(body).hexdigest(),
    )
