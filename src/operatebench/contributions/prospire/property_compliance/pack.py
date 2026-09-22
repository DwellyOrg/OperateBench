# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001 -- checker policy requires this exact finite import spelling.
"""Offline SDK adapter with pack-owned run and replay formats."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from operatebench._resource_access import files, as_file
from operatebench._write_once import _write_once_bytes, _WriteOnceFailure
from operatebench.core.errors import (
    AgentRegistryError,
    ArtifactError,
    OracleManifestError,
    SpecSchemaError,
)
from operatebench.sdk import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.errors import OperationPackMismatchError

from ._bounded_io import read_bounded
from .agents import (
    NEGATIVE_AGENT_IDS,
    NEGATIVE_AGENT_SCENARIOS,
    REFERENCE_AGENT_ID,
    events_for,
)
from .evaluator import DIMENSIONS, evaluate
from .operation import PropertyComplianceOperation
from .spec import (
    EXPECTED_SCENARIOS,
    OPERATION_ID,
    PACK_ID,
    VERSION,
    OperationSpec,
    load_spec,
)

RUN_FORMAT = "operatebench.property-compliance-incubator-run.v1"
MAX_RUN_BYTES = 1024 * 1024
MAX_ORACLE_BYTES = 64 * 1024
SCOPE = (
    "Fully synthetic maintainer-authored incubator reference; non-evidence, "
    "not policy, legal advice, a Dwelly workflow, or a public benchmark case."
)
RUN_FIELDS = {
    "format",
    "pack_id",
    "pack_version",
    "operation_type",
    "operation_id",
    "spec_digest_sha256",
    "scope",
    "self_check",
    "result",
}
RESULT_FIELDS = {
    "scenario_id",
    "agent_id",
    "events",
    "final_state",
    "findings",
    "evaluation",
}
ORACLE_ID = "property_compliance_negative_controls_v1"


def _bound(path: Path) -> OperationSpec:
    spec = load_spec(path)
    if spec.operation_type != PACK_ID:
        raise OperationPackMismatchError(
            f"pack {PACK_ID!r} refuses operation_type {spec.operation_type!r}"
        )
    return spec


def _execute(scenario: str, agent: str, *, self_check: bool) -> dict[str, Any]:
    if agent not in (REFERENCE_AGENT_ID, *NEGATIVE_AGENT_IDS):
        raise AgentRegistryError(f"unknown property-compliance agent {agent!r}")
    bound_scenario = NEGATIVE_AGENT_SCENARIOS.get(agent)
    if bound_scenario is not None and scenario != bound_scenario:
        raise AgentRegistryError(
            f"property-compliance agent {agent!r} is bound to scenario "
            f"{bound_scenario!r}, not {scenario!r}"
        )
    events = events_for(agent, scenario)
    state, findings = PropertyComplianceOperation().execute(events)
    replay_ok = False
    if self_check:
        state2, findings2 = PropertyComplianceOperation().execute(events)
        replay_ok = state.canonical() == state2.canonical() and findings == findings2
    return {
        "scenario_id": scenario,
        "agent_id": agent,
        "events": events,
        "final_state": state.canonical(),
        "findings": list(findings),
        "evaluation": evaluate(
            scenario=scenario,
            state=state,
            findings=findings,
            replay_ok=replay_ok,
        ),
    }


def _payload(spec: Any, scenario: str, agent: str, *, self_check: bool) -> dict[str, Any]:
    return {
        "format": RUN_FORMAT,
        "pack_id": PACK_ID,
        "pack_version": VERSION,
        "operation_type": PACK_ID,
        "operation_id": OPERATION_ID,
        "spec_digest_sha256": spec.spec_digest_sha256,
        "scope": SCOPE,
        "self_check": True if self_check else None,
        "result": _execute(scenario, agent, self_check=self_check),
    }


def _write_error(failure: _WriteOnceFailure) -> ArtifactError:
    return ArtifactError(
        f"incubator run output {failure.path} cannot be written "
        f"({failure.reason}); outputs are exclusive and symlinks are refused"
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    if len(body) > MAX_RUN_BYTES:
        raise ArtifactError("property-compliance run exceeds bounded size")
    _write_once_bytes(body, path, error_adapter=_write_error)


def _read(path: Path) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        if path.is_symlink():
            raise ArtifactError("incubator run symlinks are refused") from exc
        raise ArtifactError(f"cannot read incubator run {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactError("incubator run must be a regular file")
        if metadata.st_size > MAX_RUN_BYTES:
            raise ArtifactError("incubator run exceeds bounded size")
        body = read_bounded(descriptor, MAX_RUN_BYTES + 1)
    except OSError as exc:
        raise ArtifactError(f"cannot read incubator run {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    if len(body) > MAX_RUN_BYTES:
        raise ArtifactError("incubator run exceeds bounded size")
    try:
        raw = json.loads(body, object_pairs_hook=_unique)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ArtifactError(f"malformed incubator run: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != RUN_FIELDS:
        raise ArtifactError("incubator run has unexpected fields")
    return raw


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON field {key!r}")
        out[key] = value
    return out


def _validate_run(spec: OperationSpec, recorded: dict[str, Any]) -> dict[str, Any]:
    fixed = {
        "format": RUN_FORMAT,
        "pack_id": PACK_ID,
        "pack_version": VERSION,
        "operation_type": PACK_ID,
        "operation_id": OPERATION_ID,
        "spec_digest_sha256": spec.spec_digest_sha256,
        "scope": SCOPE,
    }
    if any(
        type(recorded[name]) is not str or recorded[name] != expected
        for name, expected in fixed.items()
    ):
        raise ArtifactError("incubator run identity does not match this pack and spec")
    if recorded["self_check"] is not None and type(recorded["self_check"]) is not bool:
        raise ArtifactError("incubator run self_check must be true or null")
    if recorded["self_check"] is False:
        raise ArtifactError("incubator run self_check must be true or null")
    result = recorded["result"]
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        raise ArtifactError("incubator run result has unexpected fields")
    scenario = result["scenario_id"]
    agent = result["agent_id"]
    if type(scenario) is not str or scenario not in spec.scenario_ids:
        raise ArtifactError(f"incubator run names unknown scenario {scenario!r}")
    if type(agent) is not str or agent not in (REFERENCE_AGENT_ID, *NEGATIVE_AGENT_IDS):
        raise ArtifactError(f"incubator run names unknown agent {agent!r}")
    if NEGATIVE_AGENT_SCENARIOS.get(agent) not in (None, scenario):
        raise ArtifactError("incubator run agent/scenario mismatch")
    return result


def _validate_oracle(raw: object) -> list[dict[str, Any]]:
    """Validate the fixed oracle without performing resource I/O."""
    envelope_fields = {
        "oracle_id",
        "oracle_version",
        "operation_type",
        "controls",
    }
    if not isinstance(raw, dict):
        raise OracleManifestError("property-compliance oracle envelope invalid")
    _oracle_fields(raw, envelope_fields, "property-compliance oracle")
    fixed = {
        "oracle_id": ORACLE_ID,
        "oracle_version": VERSION,
        "operation_type": PACK_ID,
    }
    if any(
        type(raw[name]) is not str or raw[name] != value for name, value in fixed.items()
    ):
        raise OracleManifestError("property-compliance oracle identity invalid")
    controls = raw["controls"]
    if not isinstance(controls, list) or len(controls) != len(NEGATIVE_AGENT_IDS):
        raise OracleManifestError(
            "property-compliance oracle must contain seven controls"
        )
    for index, row in enumerate(controls):
        control_fields = {
            "agent_id",
            "scenario_id",
            "required_fail",
            "required_pass",
        }
        if not isinstance(row, dict):
            raise OracleManifestError(
                f"property-compliance oracle control {index} fields invalid"
            )
        _oracle_fields(row, control_fields, f"property-compliance oracle control {index}")
        agent = row["agent_id"]
        scenario = row["scenario_id"]
        failed = row["required_fail"]
        passed = row["required_pass"]
        if type(agent) is not str or agent != NEGATIVE_AGENT_IDS[index]:
            raise OracleManifestError(
                "property-compliance oracle agents invalid or unordered"
            )
        if type(scenario) is not str or scenario not in EXPECTED_SCENARIOS:
            raise OracleManifestError(
                f"property-compliance oracle scenario {scenario!r} invalid"
            )
        if NEGATIVE_AGENT_SCENARIOS[agent] != scenario:
            raise OracleManifestError(
                "property-compliance oracle agent/scenario binding invalid"
            )
        if (
            not isinstance(failed, list)
            or not isinstance(passed, list)
            or not failed
            or not passed
        ):
            raise OracleManifestError(
                "property-compliance oracle dimension partitions must be non-empty lists"
            )
        if any(type(name) is not str for name in (*failed, *passed)):
            raise OracleManifestError(
                "property-compliance oracle dimensions must be strings"
            )
        if len(set(failed)) != len(failed) or len(set(passed)) != len(passed):
            raise OracleManifestError(
                "property-compliance oracle dimensions must be unique"
            )
        if set(failed) & set(passed) or set(failed) | set(passed) != set(DIMENSIONS):
            raise OracleManifestError(
                "property-compliance oracle dimensions must exactly partition "
                "the result vector"
            )
        if failed != [name for name in DIMENSIONS if name in failed] or passed != [
            name for name in DIMENSIONS if name in passed
        ]:
            raise OracleManifestError(
                "property-compliance oracle dimensions must follow result-vector order"
            )
    return controls


class _OracleLoader(yaml.SafeLoader):
    pass


def _oracle_mapping(
    loader: _OracleLoader, node: yaml.MappingNode, deep: bool = False
) -> Any:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise OracleManifestError("oracle mapping field names must be strings")
        if key in result:
            raise OracleManifestError(f"oracle repeats mapping key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_OracleLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _oracle_mapping
)


def _oracle_fields(value: Mapping[Any, Any], expected: set[str], where: str) -> None:
    non_text = sorted(repr(key) for key in value if type(key) is not str)
    if non_text:
        raise OracleManifestError(
            f"{where}: mapping field names must be strings, got {non_text}"
        )
    if set(value) != expected:
        raise OracleManifestError(f"{where} fields invalid")


def _parse_oracle(body: bytes, source: str) -> list[dict[str, Any]]:
    if len(body) > MAX_ORACLE_BYTES:
        raise OracleManifestError(
            f"{source}: oracle exceeds {MAX_ORACLE_BYTES}-byte limit"
        )
    try:
        raw = yaml.load(body.decode("utf-8"), Loader=_OracleLoader)
    except OracleManifestError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise OracleManifestError(f"invalid {source}: {exc}") from exc
    return _validate_oracle(raw)


def _read_oracle_resource() -> bytes:
    resource = files(__package__).joinpath("negative_control_oracle.yaml")
    try:
        with as_file(resource) as concrete:
            descriptor = os.open(concrete, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise OracleManifestError(
                        "property-compliance oracle cannot read bounded resource"
                    )
                if metadata.st_size > MAX_ORACLE_BYTES:
                    raise OracleManifestError(
                        "property-compliance oracle cannot read bounded resource"
                    )
                body = read_bounded(descriptor, MAX_ORACLE_BYTES + 1)
            finally:
                os.close(descriptor)
    except OracleManifestError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise OracleManifestError(
            "property-compliance oracle cannot read bounded resource"
        ) from exc
    if len(body) > MAX_ORACLE_BYTES:
        raise OracleManifestError(
            "property-compliance oracle cannot read bounded resource"
        )
    return body


def _oracle() -> list[dict[str, Any]]:
    return _parse_oracle(_read_oracle_resource(), "property-compliance oracle")


def _oracle_row_passed(
    dimensions: Mapping[str, object], control: Mapping[str, Any]
) -> bool:
    return all(not dimensions[name] for name in control["required_fail"]) and all(
        dimensions[name] for name in control["required_pass"]
    )


class PropertyCompliancePack:
    metadata = OperationPackMetadata(
        pack_id="lettings.property_compliance.synthetic",
        operation_type="lettings.property_compliance.synthetic",
        operation_id="lettings_property_compliance_synthetic_v1",
        pack_version="0.1.0",
        display_name="Synthetic Property Safety Certificate renewal",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/contributions/prospire/property_compliance/operation.yaml",
        agent_ids=(
            "act_after_finality",
            "complete_before_registry_validation",
            "ignore_certificate_rejection",
            "reference",
            "skip_remediation_approval",
            "trust_supplier_certificate_claim",
            "unnecessary_human_review",
            "wait_past_expiry_deadline",
        ),
        aliases=(),
        evidence_eligible=False,
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        spec = _bound(request.spec_path)
        return CommandResult(
            True,
            {
                "valid": True,
                "pack": self.metadata.as_dict(),
                "operation_id": spec.operation_id,
                "scenario_ids": list(spec.scenario_ids),
            },
            (f"valid {PACK_ID} synthetic fixture",),
        )

    def run(self, request: RunRequest) -> CommandResult:
        spec = _bound(request.spec_path)
        if request.scenario_id not in spec.scenario_ids:
            raise SpecSchemaError(f"unknown scenario {request.scenario_id!r}")
        payload = _payload(
            spec,
            request.scenario_id,
            request.agent_id,
            self_check=request.self_check,
        )
        _write(request.output_path, payload)
        passed = bool(payload["result"]["evaluation"]["passed"])
        return CommandResult(
            passed,
            {"written": str(request.output_path), "run": payload},
            (f"wrote {request.output_path}",),
        )

    def replay(self, request: ReplayRequest) -> CommandResult:
        spec = _bound(request.spec_path)
        recorded = _read(request.run_path)
        result = _validate_run(spec, recorded)
        recomputed = _payload(
            spec,
            result["scenario_id"],
            result["agent_id"],
            self_check=recorded["self_check"] is True,
        )
        matched = recorded == recomputed
        return CommandResult(
            matched,
            {"replay_matched": matched, "recomputed": recomputed},
            (
                (
                    "replay matched"
                    if matched
                    else "replay mismatch: recorded run was not trusted"
                ),
            ),
        )

    def check(self, request: CheckRequest) -> CommandResult:
        spec = _bound(request.spec_path)
        selected = set(request.agent_ids or (REFERENCE_AGENT_ID, *NEGATIVE_AGENT_IDS))
        unknown = selected - set(self.metadata.agent_ids)
        if unknown:
            raise AgentRegistryError(
                f"unknown property-compliance agent(s) {sorted(unknown)}"
            )
        references = []
        if REFERENCE_AGENT_ID in selected:
            references = [
                _execute(s, REFERENCE_AGENT_ID, self_check=True)
                for s in spec.scenario_ids
            ]
        controls = []
        for control in _oracle():
            if control["agent_id"] not in selected:
                continue
            result = _execute(
                control["scenario_id"], control["agent_id"], self_check=True
            )
            dims = result["evaluation"]["dimensions"]
            oracle_passed = _oracle_row_passed(dims, control)
            controls.append(
                {
                    "agent_id": control["agent_id"],
                    "scenario_id": control["scenario_id"],
                    "oracle_passed": oracle_passed,
                    "required_fail": control["required_fail"],
                    "required_pass": control["required_pass"],
                    "evaluation": result["evaluation"],
                }
            )
        passed = all(r["evaluation"]["passed"] for r in references) and all(
            c["oracle_passed"] for c in controls
        )
        return CommandResult(
            passed,
            {
                "passed": passed,
                "reference_successes": references,
                "negative_controls": controls,
                "counts": {
                    "reference_successes": len(references),
                    "negative_controls": len(controls),
                },
            },
            (
                f"property compliance check: {len(references)} references, "
                f"{len(controls)} controls, "
                f"{'passed' if passed else 'failed'}",
            ),
        )


PACK = PropertyCompliancePack()
