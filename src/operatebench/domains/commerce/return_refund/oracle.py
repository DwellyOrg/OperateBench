"""Strict loader for the separately authored return/refund control oracle."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from boundarybench.jsonsafe import JsonSafetyError, canonical_json_bytes
from operatebench.core.errors import OracleManifestError

ORACLE_RESOURCE_PACKAGE = "operatebench.domains.commerce.return_refund"
ORACLE_RESOURCE_NAME = "return_refund_negative_controls_v0_1.yaml"
ORACLE_RESOURCE_PARTS = ("resources", "oracles", ORACLE_RESOURCE_NAME)
MAX_ORACLE_BYTES = 1024 * 1024

_TOP_LEVEL_FIELDS = (
    "oracle_id",
    "oracle_version",
    "operation_type",
    "authored_against_pack_version",
    "provenance",
    "result_vector_dimensions",
    "controls",
)
_PROVENANCE_FIELDS = (
    "authored_by",
    "independence",
    "evidence_class",
    "review_status",
)
_CONTROL_FIELDS = (
    "agent_id",
    "scenario_id",
    "intervention",
    "guarantee_attacked",
    "expected_failed_dimensions",
    "required_finding_codes",
    "must_pass_dimensions",
    "rationale",
)


class _StrictOracleLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_mapping(
    loader: _StrictOracleLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> Any:
    mapping: dict[Any, Any] = {}
    yield mapping
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            mark = key_node.start_mark
            raise OracleManifestError(
                f"oracle mapping key at line {mark.line + 1}, "
                f"column {mark.column + 1} is not usable: {exc}"
            ) from exc
        if duplicate:
            mark = key_node.start_mark
            raise OracleManifestError(
                f"oracle repeats mapping key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictOracleLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OracleManifestError(
            f"{where}: expected a mapping, got {type(value).__name__}"
        )
    return value


def _fields(payload: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    non_text = sorted(repr(name) for name in payload if type(name) is not str)
    if non_text:
        raise OracleManifestError(
            f"{where}: mapping field names must be strings, got {non_text}"
        )
    unknown = sorted(set(payload) - set(allowed))
    missing = sorted(set(allowed) - set(payload))
    if unknown:
        raise OracleManifestError(f"{where}: unknown field(s) {unknown}")
    if missing:
        raise OracleManifestError(f"{where}: missing field(s) {missing}")


def _text(payload: Mapping[str, Any], name: str, where: str) -> str:
    value = payload[name]
    if type(value) is not str or not value.strip():
        raise OracleManifestError(f"{where}: {name} must be a non-empty string")
    return value.strip()


def _names(payload: Mapping[str, Any], name: str, where: str) -> tuple[str, ...]:
    value = payload[name]
    if not isinstance(value, list) or not value:
        raise OracleManifestError(f"{where}: {name} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if type(item) is not str or not item.strip():
            raise OracleManifestError(f"{where}: {name} contains a non-name")
        if item in result:
            raise OracleManifestError(f"{where}: {name} repeats {item!r}")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class NegativeControl:
    agent_id: str
    scenario_id: str
    intervention: str
    guarantee_attacked: str
    expected_failed_dimensions: tuple[str, ...]
    required_finding_codes: tuple[str, ...]
    must_pass_dimensions: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "intervention": self.intervention,
            "guarantee_attacked": self.guarantee_attacked,
            "expected_failed_dimensions": list(self.expected_failed_dimensions),
            "required_finding_codes": list(self.required_finding_codes),
            "must_pass_dimensions": list(self.must_pass_dimensions),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class NegativeControlOracle:
    oracle_id: str
    oracle_version: str
    operation_type: str
    authored_against_pack_version: str
    provenance: Mapping[str, str]
    result_vector_dimensions: tuple[str, ...]
    controls: tuple[NegativeControl, ...]
    digest_sha256: str
    source: str

    @property
    def oracle_digest_sha256(self) -> str:
        return self.digest_sha256

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(control.agent_id for control in self.controls)

    def control(self, agent_id: str) -> NegativeControl:
        for control in self.controls:
            if control.agent_id == agent_id:
                return control
        raise OracleManifestError(
            f"{self.source}: no control for {agent_id!r}; declares {list(self.agent_ids)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "operation_type": self.operation_type,
            "authored_against_pack_version": self.authored_against_pack_version,
            "provenance": dict(self.provenance),
            "result_vector_dimensions": list(self.result_vector_dimensions),
            "controls": [control.as_dict() for control in self.controls],
            "digest_sha256": self.digest_sha256,
        }


def _digest(payload: Mapping[str, Any], source: str) -> str:
    try:
        body = canonical_json_bytes(dict(payload), f"{source}: return/refund oracle")
    except JsonSafetyError as exc:
        raise OracleManifestError(f"{source}: oracle is not JSON-safe: {exc}") from exc
    return hashlib.sha256(body).hexdigest()


def _parse_yaml(text: str, source: str) -> Mapping[str, Any]:
    try:
        raw = yaml.load(text, Loader=_StrictOracleLoader)
    except OracleManifestError:
        raise
    except yaml.YAMLError as exc:
        raise OracleManifestError(f"{source}: invalid YAML: {exc}") from exc
    except RecursionError as exc:
        raise OracleManifestError(f"{source}: oracle is nested too deeply") from exc
    return _mapping(raw, source)


def _decode_bounded(body: bytes, source: str) -> str:
    if len(body) > MAX_ORACLE_BYTES:
        raise OracleManifestError(
            f"{source}: oracle exceeds the {MAX_ORACLE_BYTES}-byte limit"
        )
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OracleManifestError(f"{source}: oracle is not valid UTF-8: {exc}") from exc


def _parse_control(
    raw: Any,
    vocabulary: tuple[str, ...],
    where: str,
) -> NegativeControl:
    payload = _mapping(raw, where)
    _fields(payload, _CONTROL_FIELDS, where)
    agent_id = _text(payload, "agent_id", where)
    named = f"{where} ({agent_id})"
    failed = _names(payload, "expected_failed_dimensions", named)
    passing = _names(payload, "must_pass_dimensions", named)
    unknown = sorted((set(failed) | set(passing)) - set(vocabulary))
    if unknown:
        raise OracleManifestError(f"{named}: unknown result dimension(s) {unknown}")
    overlap = sorted(set(failed) & set(passing))
    if overlap:
        raise OracleManifestError(f"{named}: dimensions both fail and pass: {overlap}")
    unjudged = sorted(set(vocabulary) - set(failed) - set(passing))
    if unjudged:
        raise OracleManifestError(f"{named}: unjudged dimension(s) {unjudged}")
    if tuple(name for name in vocabulary if name in failed) != failed:
        raise OracleManifestError(
            f"{named}: failed dimensions must follow result-vector order"
        )
    if tuple(name for name in vocabulary if name in passing) != passing:
        raise OracleManifestError(
            f"{named}: must-pass dimensions must follow result-vector order"
        )
    return NegativeControl(
        agent_id=agent_id,
        scenario_id=_text(payload, "scenario_id", named),
        intervention=_text(payload, "intervention", named),
        guarantee_attacked=_text(payload, "guarantee_attacked", named),
        expected_failed_dimensions=failed,
        required_finding_codes=_names(payload, "required_finding_codes", named),
        must_pass_dimensions=passing,
        rationale=_text(payload, "rationale", named),
    )


def _build(payload: Mapping[str, Any], source: str) -> NegativeControlOracle:
    _fields(payload, _TOP_LEVEL_FIELDS, source)
    provenance_raw = _mapping(payload["provenance"], f"{source}: provenance")
    _fields(provenance_raw, _PROVENANCE_FIELDS, f"{source}: provenance")
    provenance = {
        name: _text(provenance_raw, name, f"{source}: provenance")
        for name in _PROVENANCE_FIELDS
    }
    vocabulary = _names(payload, "result_vector_dimensions", source)
    raw_controls = payload["controls"]
    if not isinstance(raw_controls, list) or not raw_controls:
        raise OracleManifestError(f"{source}: controls must be a non-empty list")
    controls = tuple(
        _parse_control(raw, vocabulary, f"{source}: controls[{index}]")
        for index, raw in enumerate(raw_controls)
    )
    agent_ids = tuple(control.agent_id for control in controls)
    if len(set(agent_ids)) != len(agent_ids):
        raise OracleManifestError(f"{source}: control agent ids must be unique")
    signatures = tuple(
        (control.expected_failed_dimensions, control.required_finding_codes)
        for control in controls
    )
    if len(set(signatures)) != len(signatures):
        raise OracleManifestError(
            f"{source}: controls must have distinct failure/code signatures"
        )
    return NegativeControlOracle(
        oracle_id=_text(payload, "oracle_id", source),
        oracle_version=_text(payload, "oracle_version", source),
        operation_type=_text(payload, "operation_type", source),
        authored_against_pack_version=_text(
            payload, "authored_against_pack_version", source
        ),
        provenance=MappingProxyType(provenance),
        result_vector_dimensions=vocabulary,
        controls=controls,
        digest_sha256=_digest(payload, source),
        source=source,
    )


def load_negative_control_oracle(path: str | Path) -> NegativeControlOracle:
    source = str(path)
    try:
        with Path(path).open("rb") as handle:
            body = handle.read(MAX_ORACLE_BYTES + 1)
    except OSError as exc:
        raise OracleManifestError(f"cannot read oracle {source}: {exc}") from exc
    return _build(_parse_yaml(_decode_bounded(body, source), source), source)


_CACHE: NegativeControlOracle | None = None


def negative_control_oracle() -> NegativeControlOracle:
    global _CACHE
    if _CACHE is None:
        resource = resources.files(ORACLE_RESOURCE_PACKAGE)
        for part in ORACLE_RESOURCE_PARTS:
            resource = resource.joinpath(part)
        source = f"{ORACLE_RESOURCE_PACKAGE}:{'/'.join(ORACLE_RESOURCE_PARTS)}"
        try:
            with resource.open("rb") as handle:
                body = handle.read(MAX_ORACLE_BYTES + 1)
        except (OSError, UnicodeError) as exc:
            raise OracleManifestError(
                f"cannot read packaged oracle {source}: {exc}"
            ) from exc
        text = _decode_bounded(body, source)
        _CACHE = _build(_parse_yaml(text, source), source)
    return _CACHE


def control_expectation(agent_id: str) -> NegativeControl | None:
    oracle = negative_control_oracle()
    return next(
        (control for control in oracle.controls if control.agent_id == agent_id),
        None,
    )


__all__ = [
    "MAX_ORACLE_BYTES",
    "ORACLE_RESOURCE_NAME",
    "ORACLE_RESOURCE_PACKAGE",
    "NegativeControl",
    "NegativeControlOracle",
    "control_expectation",
    "load_negative_control_oracle",
    "negative_control_oracle",
]
