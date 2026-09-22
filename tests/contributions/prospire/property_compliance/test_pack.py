# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from operatebench.cli import main
from operatebench.contributions.prospire.property_compliance.agents import (
    NEGATIVE_AGENT_IDS,
    NEGATIVE_AGENT_SCENARIOS,
    reference_events,
)
from operatebench.contributions.prospire.property_compliance.evaluator import (
    DIMENSIONS,
    evaluate,
)
from operatebench.contributions.prospire.property_compliance.operation import (
    PropertyComplianceOperation,
)
from operatebench.contributions.prospire.property_compliance.pack import (
    MAX_ORACLE_BYTES,
    PACK,
    _oracle,
    _oracle_row_passed,
    _parse_oracle,
    _read,
    _read_oracle_resource,
    _validate_oracle,
)
from operatebench.contributions.prospire.property_compliance.spec import (
    EXPECTED_SCENARIOS,
    MAX_SPEC_BYTES,
    load_spec,
)
from operatebench.core.errors import (
    AgentRegistryError,
    ArtifactError,
    OracleManifestError,
    SpecSchemaError,
)
from operatebench.sdk import CheckRequest, ReplayRequest, RunRequest

FIXTURE = Path("examples/contributions/prospire/property_compliance/operation.yaml")


def _python_executable() -> str:
    executable = shutil.which("python")
    assert executable is not None
    return executable


def test_strict_synthetic_spec_loads() -> None:
    spec = load_spec(FIXTURE)
    assert spec.operation_type == "lettings.property_compliance.synthetic"
    assert spec.scenario_ids == ("V1", "V2", "V3")
    assert spec.actors["certificate_registry_synthetic"] == "certificate_registry"


@pytest.mark.parametrize(
    "body",
    [
        "? [a]\n: b\n",
        FIXTURE.read_text(encoding="utf-8").replace(
            "    path: normal_renewal",
            "    1: a\n    zz: b\n    path: normal_renewal",
        ),
    ],
)
def test_spec_loader_names_every_non_string_mapping_key(
    tmp_path: Path, body: str
) -> None:
    target = tmp_path / "bad-key.yaml"
    target.write_text(body, encoding="utf-8")
    with pytest.raises(SpecSchemaError, match="mapping field names must be strings"):
        load_spec(target)


@pytest.mark.parametrize(
    "body",
    [
        "? [a]\n: b\n",
        FIXTURE.read_text(encoding="utf-8").replace(
            "    path: normal_renewal",
            "    1: a\n    zz: b\n    path: normal_renewal",
        ),
    ],
)
def test_real_cli_refuses_non_string_mapping_keys_without_traceback(
    tmp_path: Path, body: str
) -> None:
    target = tmp_path / "bad-key.yaml"
    target.write_text(body, encoding="utf-8")
    completed = subprocess.run(
        [
            _python_executable(),
            "-m",
            "operatebench.cli",
            "validate",
            "--pack",
            "lettings.property_compliance.synthetic",
            str(target),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "mapping field names must be strings" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_main_maps_complex_yaml_key_to_domain_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "complex.yaml"
    target.write_text("? [a]\n: b\n", encoding="utf-8")
    assert (
        main(
            [
                "validate",
                "--pack",
                "lettings.property_compliance.synthetic",
                str(target),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "mapping field names must be strings" in captured.err
    assert "Traceback" not in captured.err


def test_string_mapping_key_control_reaches_named_schema_validation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "string-key.yaml"
    target.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "    path: normal_renewal",
            "    zz: b\n    path: normal_renewal",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecSchemaError, match=r"unknown field.*zz"):
        load_spec(target)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("schema_version: 1", "schema_version: 1\nunknown: unsafe", "unknown field"),
        ("SYNTHETIC_ONLY", "PRIVATE", "privacy_status"),
        ("Entirely invented", "Derived", "data_provenance"),
        ("lettings.property_compliance.synthetic", "wrong.operation", "operation_type"),
    ],
)
def test_strict_synthetic_spec_refuses_unsafe_values(
    tmp_path: Path, needle: str, replacement: str, message: str
) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text(FIXTURE.read_text().replace(needle, replacement), encoding="utf-8")
    with pytest.raises(SpecSchemaError, match=message):
        load_spec(target)


@pytest.mark.parametrize("scenario", ["V1", "V2", "V3"])
def test_reference_paths_are_final_and_replay(tmp_path: Path, scenario: str) -> None:
    output = tmp_path / f"{scenario}.json"
    result = PACK.run(RunRequest(FIXTURE, scenario, "reference", output))
    assert result.contract_passed
    assert PACK.replay(ReplayRequest(FIXTURE, output)).contract_passed


def test_every_reference_result_dimension_is_true(tmp_path: Path) -> None:
    for scenario in EXPECTED_SCENARIOS:
        output = tmp_path / f"reference-{scenario}.json"
        result = PACK.run(RunRequest(FIXTURE, scenario, "reference", output))
        dimensions = result.payload["run"]["result"]["evaluation"]["dimensions"]
        assert dimensions == dict.fromkeys(DIMENSIONS, True)


def test_refused_action_inserted_into_reference_fails_action_validity() -> None:
    events = reference_events("V2")
    repair_index = next(
        index for index, event in enumerate(events) if event["type"] == "repair_evidence"
    )
    events.insert(
        repair_index,
        _event("reinspection_clear", "inspection_supplier_synthetic"),
    )
    state, findings = PropertyComplianceOperation().execute(events)
    result = evaluate(scenario="V2", state=state, findings=findings, replay_ok=True)
    dimensions = result["dimensions"]
    assert isinstance(dimensions, Mapping)
    assert findings == ("REPAIR_EVIDENCE_REQUIRED",)
    assert dimensions["action_validity"] is False
    assert all(value for name, value in dimensions.items() if name != "action_validity")


def test_full_oracle_matrix_has_pass_and_fail_expectations() -> None:
    result = PACK.check(CheckRequest(FIXTURE))
    assert result.contract_passed
    assert result.payload["counts"] == {"reference_successes": 3, "negative_controls": 7}
    for control in result.payload["negative_controls"]:
        assert control["required_fail"]
        assert control["required_pass"]
        assert control["oracle_passed"]
        partition = (
            control["required_pass"]
            if control["agent_id"] == "wait_past_expiry_deadline"
            else control["required_fail"]
        )
        assert "action_validity" in partition


def test_run_is_exclusive_and_replay_detects_tamper(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    PACK.run(RunRequest(FIXTURE, "V1", "reference", output))
    with pytest.raises(ArtifactError, match="exclusive"):
        PACK.run(RunRequest(FIXTURE, "V1", "reference", output))
    raw = output.read_text(encoding="utf-8").replace(
        '"terminal": "certificate_renewed_final"', '"terminal": "tampered"'
    )
    output.write_text(raw, encoding="utf-8")
    assert not PACK.replay(ReplayRequest(FIXTURE, output)).contract_passed


def _written_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    output = tmp_path / "run.json"
    PACK.run(RunRequest(FIXTURE, "V1", "reference", output))
    return output, json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "wrong-format"),
        ("pack_id", "wrong-pack"),
        ("pack_version", "9.9.9"),
        ("operation_type", "wrong-operation-type"),
        ("operation_id", "wrong-operation-id"),
        ("scope", "wrong-scope"),
    ],
)
def test_replay_refuses_changed_fixed_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    output, payload = _written_run(tmp_path)
    payload[field] = value
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        PACK.replay(ReplayRequest(FIXTURE, output))


@pytest.mark.parametrize(
    ("field", "value"),
    [("agent_id", "unknown-agent"), ("scenario_id", "unknown-scenario")],
)
def test_replay_wraps_unknown_nested_dispatch_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    output, payload = _written_run(tmp_path)
    payload["result"][field] = value
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        PACK.replay(ReplayRequest(FIXTURE, output))


@pytest.mark.parametrize("result", [None, [], "result", True])
def test_replay_refuses_non_mapping_result(tmp_path: Path, result: object) -> None:
    output, payload = _written_run(tmp_path)
    payload["result"] = result
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        PACK.replay(ReplayRequest(FIXTURE, output))


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_replay_refuses_inexact_result_fields(tmp_path: Path, mutation: str) -> None:
    output, payload = _written_run(tmp_path)
    if mutation == "extra":
        payload["result"]["unexpected"] = True
    else:
        del payload["result"]["events"]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError):
        PACK.replay(ReplayRequest(FIXTURE, output))


def test_replay_refuses_duplicate_nested_json_field(tmp_path: Path) -> None:
    output, _ = _written_run(tmp_path)
    body = output.read_text(encoding="utf-8").replace(
        '"agent_id": "reference",',
        '"agent_id": "reference", "agent_id": "reference",',
        1,
    )
    output.write_text(body, encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate JSON field"):
        PACK.replay(ReplayRequest(FIXTURE, output))


def test_runs_are_byte_deterministic_and_wrong_type_writes_nothing(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    PACK.run(RunRequest(FIXTURE, "V2", "reference", first))
    PACK.run(RunRequest(FIXTURE, "V2", "reference", second))
    assert first.read_bytes() == second.read_bytes()

    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "lettings.property_compliance.synthetic", "wrong.operation"
        ),
        encoding="utf-8",
    )
    refused_output = tmp_path / "must-not-exist.json"
    with pytest.raises(SpecSchemaError, match="operation_type"):
        PACK.run(RunRequest(wrong, "V1", "reference", refused_output))
    assert not refused_output.exists()


@pytest.mark.parametrize(("agent_id", "bound_scenario"), NEGATIVE_AGENT_SCENARIOS.items())
def test_negative_agents_are_bound_to_one_scenario_before_write(
    tmp_path: Path, agent_id: str, bound_scenario: str
) -> None:
    valid = tmp_path / f"{agent_id}-valid.json"
    PACK.run(RunRequest(FIXTURE, bound_scenario, agent_id, valid))
    assert valid.exists()
    for scenario in set(EXPECTED_SCENARIOS) - {bound_scenario}:
        refused = tmp_path / f"{agent_id}-{scenario}.json"
        with pytest.raises(AgentRegistryError, match="bound to scenario"):
            PACK.run(RunRequest(FIXTURE, scenario, agent_id, refused))
        assert not refused.exists()


def test_negative_agent_scenario_metadata_is_immutable() -> None:
    with pytest.raises(TypeError):
        NEGATIVE_AGENT_SCENARIOS["act_after_finality"] = "V2"  # type: ignore[index]


def test_replay_wraps_negative_agent_scenario_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "negative.json"
    PACK.run(
        RunRequest(
            FIXTURE,
            NEGATIVE_AGENT_SCENARIOS["skip_remediation_approval"],
            "skip_remediation_approval",
            output,
        )
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["result"]["scenario_id"] = "V1"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="agent/scenario mismatch"):
        PACK.replay(ReplayRequest(FIXTURE, output))


@pytest.mark.parametrize(("self_check", "calls"), [(True, 2), (False, 1)])
def test_run_honours_self_check_execution_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    self_check: bool,
    calls: int,
) -> None:
    real_execute = PropertyComplianceOperation.execute
    observed = 0

    def instrumented(self: PropertyComplianceOperation, events: Any) -> Any:
        nonlocal observed
        observed += 1
        return real_execute(self, events)

    monkeypatch.setattr(PropertyComplianceOperation, "execute", instrumented)
    result = PACK.run(
        RunRequest(FIXTURE, "V1", "reference", tmp_path / "run.json", self_check)
    )
    assert observed == calls
    dimensions = result.payload["run"]["result"]["evaluation"]["dimensions"]
    assert dimensions["deterministic_replay"] is self_check
    assert result.contract_passed is self_check


def test_cli_no_self_check_executes_once_and_records_unverified_dimension(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.json"
    completed = subprocess.run(
        [
            _python_executable(),
            "-m",
            "operatebench.cli",
            "run",
            "--pack",
            "lettings.property_compliance.synthetic",
            "--spec",
            str(FIXTURE),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(output),
            "--no-self-check",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["self_check"] is None
    assert payload["result"]["evaluation"]["dimensions"]["deterministic_replay"] is False
    assert PACK.replay(ReplayRequest(FIXTURE, output)).contract_passed


def test_partner_doc_separates_historical_and_runnable_scaffold_commands() -> None:
    text = Path("docs/contributions/prospire/property_compliance.md").read_text(
        encoding="utf-8"
    )
    assert "Historical record only" in text
    assert "partner_name.new_operation.synthetic" in text
    assert "partner_name_new_operation_synthetic_v1" in text
    assert "--operation-name new_operation" in text
    historical = (
        "operatebench init-operation --pack-id lettings.property_compliance.synthetic"
    )
    guided = "operatebench init-company-operation --repository-root ."
    assert historical in text
    assert guided in text
    assert text.index(historical) < text.index(guided)


def test_sdk_doc_links_canonical_property_compliance_reference() -> None:
    text = Path("docs/OPERATION_PACK_SDK.md").read_text(encoding="utf-8")
    prose = " ".join(text.split())
    link = (
        "[worked property-compliance reference]"
        "(contributions/prospire/property_compliance.md)"
    )
    assert link in text
    for phrase in (
        "maintainer-authored",
        "isolated",
        "synthetic",
        "non-evidence",
        "canonical scaffold-to-static-registration example",
    ):
        assert phrase in prose


def test_pack_metadata_is_fail_closed() -> None:
    metadata = PACK.metadata.as_dict()
    assert metadata["status"] == "incubator"
    assert metadata["evidence_eligible"] is False
    assert metadata["privacy_status"] == "SYNTHETIC_ONLY"


def test_package_is_isolated_and_registry_is_only_dispatch_integration() -> None:
    package = Path("src/operatebench/contributions/prospire/property_compliance")
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))
    assert "operatebench.domains.commerce" not in source
    assert "operatebench.domains.lettings.maintenance" not in source
    integrations = []
    for path in Path("src/operatebench").rglob("*.py"):
        if package not in path.parents and "property_compliance" in path.read_text(
            encoding="utf-8"
        ):
            integrations.append(path.as_posix())
    assert integrations == ["src/operatebench/sdk/builtins.py"]


def _event(kind: str, actor: str) -> dict[str, str]:
    return {"type": kind, "actor": actor}


def test_complete_shortcut_reports_every_missing_certificate_precondition() -> None:
    events = [
        _event("registry_validated", "certificate_registry_synthetic"),
        _event("complete", "agent_synthetic"),
    ]
    state, findings = PropertyComplianceOperation().execute(events)
    assert not state.final
    assert state.terminal is None
    assert {
        "DUE_NOTICE_REQUIRED",
        "AUTHORITATIVE_ACCESS_REQUIRED",
        "CLEAR_INSPECTION_REQUIRED",
        "CERTIFICATE_ISSUANCE_REQUIRED",
        "REGISTRY_VALIDATION_REQUIRED",
    } <= set(findings)


@pytest.mark.parametrize(
    "missing",
    [
        "due_notice",
        "access_recorded",
        "inspection_clear",
        "certificate_issued",
        "registry_validated",
    ],
)
def test_v1_requires_ordered_evidence(missing: str) -> None:
    events = [e for e in reference_events("V1") if e["type"] != missing]
    state, findings = PropertyComplianceOperation().execute(events)
    result = evaluate(scenario="V1", state=state, findings=findings, replay_ok=True)
    assert not result["passed"]
    dimensions = result["dimensions"]
    assert isinstance(dimensions, Mapping)
    assert not dimensions["certificate_path"]


def test_v2_requires_its_remediation_topology_not_a_v1_trajectory() -> None:
    state, findings = PropertyComplianceOperation().execute(reference_events("V1"))
    result = evaluate(scenario="V2", state=state, findings=findings, replay_ok=True)
    assert not result["passed"]
    dimensions = result["dimensions"]
    assert isinstance(dimensions, Mapping)
    assert not dimensions["remediation_checkpoint"]


@pytest.mark.parametrize(
    ("events", "finding"),
    [
        ([], "DUE_NOTICE_REQUIRED"),
        (["due_notice"], "ACCESS_ATTEMPT_REQUIRED"),
        (["due_notice", "access_attempt"], "ACCESS_REMINDER_REQUIRED"),
        (
            ["due_notice", "access_attempt", "access_reminder"],
            "EXPIRY_DEADLINE_REQUIRED",
        ),
    ],
)
def test_v3_exception_refuses_each_missing_predecessor(
    events: list[str], finding: str
) -> None:
    actors = {
        "due_notice": "property_record_system_synthetic",
        "access_attempt": "agent_synthetic",
        "access_reminder": "scheduler_synthetic",
    }
    trajectory = [_event(kind, actors[kind]) for kind in events]
    trajectory.append(_event("human_exception", "compliance_reviewer_synthetic"))
    state, findings = PropertyComplianceOperation().execute(trajectory)
    assert not state.final
    assert finding in findings


def test_actor_claims_never_create_authoritative_facts() -> None:
    state, _ = PropertyComplianceOperation().execute(
        [
            _event("due_notice", "property_record_system_synthetic"),
            _event("tenant_consent_claim", "tenant_synthetic"),
            _event("certificate_issued", "tenant_synthetic"),
        ]
    )
    assert not state.access_authoritative
    assert not state.certificate_issued


def test_unauthorised_attempt_is_audited_without_operational_mutation() -> None:
    state = PropertyComplianceOperation().execute([])[0]
    before = state.operational_canonical()
    finding = PropertyComplianceOperation().reduce(
        state, _event("certificate_issued", "tenant_synthetic")
    )
    assert finding == "UNAUTHORISED_EVENT"
    assert state.operational_canonical() == before
    assert state.audit == [
        {
            "type": "certificate_issued",
            "actor": "tenant_synthetic",
            "disposition": "unauthorised",
        }
    ]


def test_registry_rejection_before_issuance_is_refused_and_audited() -> None:
    state, findings = PropertyComplianceOperation().execute(
        [_event("registry_rejected", "certificate_registry_synthetic")]
    )
    assert findings == ("CERTIFICATE_ISSUANCE_REQUIRED_FOR_REJECTION",)
    assert not state.registry_rejected
    assert state.audit[-1]["disposition"] == "refused"


def test_registry_rejection_requires_reissue_then_revalidation() -> None:
    events = reference_events("V1")[:-1]
    events += [
        _event("registry_rejected", "certificate_registry_synthetic"),
        _event("complete", "agent_synthetic"),
    ]
    state, findings = PropertyComplianceOperation().execute(events)
    assert not state.final
    assert "CERTIFICATE_ISSUANCE_REQUIRED" in findings
    assert "REGISTRY_VALIDATION_REQUIRED" in findings
    recovered, recovered_findings = PropertyComplianceOperation().execute(
        [
            *events[:-1],
            _event("certificate_issued", "inspection_supplier_synthetic"),
            _event("registry_validated", "certificate_registry_synthetic"),
            _event("complete", "agent_synthetic"),
        ]
    )
    assert recovered.final
    assert not recovered_findings


@pytest.mark.parametrize(
    ("scenario", "events"),
    [
        ("V1", reference_events("V3")),
        ("V2", reference_events("V3")),
        ("V3", reference_events("V1")),
    ],
)
def test_scenarios_cannot_cross_terminal_paths(
    scenario: str, events: list[dict[str, str]]
) -> None:
    state, findings = PropertyComplianceOperation().execute(events)
    result = evaluate(scenario=scenario, state=state, findings=findings, replay_ok=True)
    assert not result["passed"]
    dimensions = result["dimensions"]
    assert isinstance(dimensions, Mapping)
    assert dimensions["terminal_outcome"] is False


def test_external_late_event_is_audit_only_but_agent_action_is_negative() -> None:
    operation = PropertyComplianceOperation()
    reference, findings = operation.execute(reference_events("V1"))
    before = reference.operational_canonical()
    assert (
        operation.reduce(
            reference, _event("registry_rejected", "certificate_registry_synthetic")
        )
        is None
    )
    assert reference.operational_canonical() == before
    assert reference.audit[-1]["disposition"] == "audit_only_after_finality"
    correct = evaluate(scenario="V1", state=reference, findings=findings, replay_ok=True)
    assert correct["passed"]

    attempted, attempted_findings = operation.execute(
        [*reference_events("V1"), _event("complete", "agent_synthetic")]
    )
    negative = evaluate(
        scenario="V1", state=attempted, findings=attempted_findings, replay_ok=True
    )
    dimensions = negative["dimensions"]
    assert isinstance(dimensions, Mapping)
    assert not dimensions["guarded_finality"]
    assert not dimensions["action_validity"]
    assert all(
        dimensions[name]
        for name in dimensions
        if name not in ("guarded_finality", "action_validity")
    )
    assert "ACTION_AFTER_FINALITY" in attempted_findings


def test_each_oracle_row_rejects_constant_dimension_mutations() -> None:
    controls = _oracle()
    assert len(controls) == 7
    for control in controls:
        assert control["required_fail"]
        assert control["required_pass"]
        assert not set(control["required_fail"]) & set(control["required_pass"])
        assert set(control["required_fail"]) | set(control["required_pass"]) == set(
            DIMENSIONS
        )
        for constant in (False, True):
            dimensions = dict.fromkeys(DIMENSIONS, constant)
            assert not _oracle_row_passed(dimensions, control)


def _valid_oracle_document() -> dict[str, Any]:
    return {
        "oracle_id": "property_compliance_negative_controls_v1",
        "oracle_version": "0.1.0",
        "operation_type": "lettings.property_compliance.synthetic",
        "controls": [dict(row) for row in _oracle()],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.__setitem__("extra", True),
        lambda doc: doc.__setitem__("oracle_id", "wrong"),
        lambda doc: doc.__setitem__("oracle_version", "9.9.9"),
        lambda doc: doc.__setitem__("operation_type", "wrong"),
        lambda doc: doc.__setitem__("controls", "not-a-list"),
        lambda doc: doc["controls"].append(dict(doc["controls"][0])),
        lambda doc: doc["controls"][0].__setitem__("extra", True),
        lambda doc: doc["controls"][0].__setitem__("agent_id", True),
        lambda doc: doc["controls"][0].__setitem__("agent_id", "unknown"),
        lambda doc: doc["controls"][0].__setitem__("scenario_id", "V9"),
        lambda doc: doc["controls"][0].__setitem__("required_fail", []),
        lambda doc: doc["controls"][0].__setitem__("required_pass", []),
        lambda doc: doc["controls"][0].__setitem__("required_fail", [True]),
        lambda doc: doc["controls"][0]["required_pass"].append("unknown"),
        lambda doc: doc["controls"][0]["required_pass"].append(
            doc["controls"][0]["required_fail"][0]
        ),
        lambda doc: doc["controls"][0]["required_pass"].pop(),
        lambda doc: doc["controls"].reverse(),
    ],
)
def test_oracle_validator_refuses_every_schema_and_identity_violation(
    mutate: Any,
) -> None:
    document = _valid_oracle_document()
    mutate(document)
    with pytest.raises(OracleManifestError):
        _validate_oracle(document)


def test_oracle_validator_accepts_exact_fixed_manifest() -> None:
    controls = _validate_oracle(_valid_oracle_document())
    assert tuple(row["agent_id"] for row in controls) == NEGATIVE_AGENT_IDS


def test_oracle_parser_refuses_duplicate_keys_and_oversize_before_yaml() -> None:
    with pytest.raises(OracleManifestError, match="repeats mapping key"):
        _parse_oracle(b"oracle_id: one\noracle_id: two\n", "test oracle")
    with pytest.raises(OracleManifestError, match="exceeds"):
        _parse_oracle(b" " * (MAX_ORACLE_BYTES + 1), "test oracle")


def test_oracle_resource_reader_reads_normal_resource() -> None:
    assert len(_read_oracle_resource()) <= MAX_ORACLE_BYTES
    assert _oracle()


def test_oracle_resource_descriptor_is_closed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    closed: list[int] = []

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "close", tracked_close)
    _read_oracle_resource()
    assert len(closed) == 1


def test_oracle_resource_reader_completes_short_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read = os.read

    def short_read(descriptor: int, count: int) -> bytes:
        return real_read(descriptor, min(count, 7))

    monkeypatch.setattr(os, "read", short_read)
    assert _oracle()


@pytest.mark.parametrize("kind", ["symlink", "non_regular", "oversized_sparse"])
def test_oracle_resource_reader_refuses_unsafe_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from operatebench.contributions.prospire.property_compliance import (
        pack as pack_module,
    )

    target = tmp_path / "oracle.yaml"
    if kind == "non_regular":
        target.mkdir()
    else:
        with target.open("wb") as stream:
            if kind == "oversized_sparse":
                stream.truncate(MAX_ORACLE_BYTES + 1)
            else:
                stream.write(b"oracle_id: harmless\n")
    resource = target
    if kind == "symlink":
        resource = tmp_path / "oracle-link.yaml"
        resource.symlink_to(target)

    class ResourceRoot:
        def joinpath(self, name: str) -> Path:
            assert name == "negative_control_oracle.yaml"
            return resource

    monkeypatch.setattr(pack_module, "files", lambda package: ResourceRoot())
    with pytest.raises(OracleManifestError, match="cannot read bounded resource"):
        _read_oracle_resource()


def test_full_check_rejects_mutated_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from operatebench.contributions.prospire.property_compliance import (
        pack as pack_module,
    )

    for constant in (False, True):

        def mutated(*, value: bool = constant, **kwargs: object) -> dict[str, object]:
            del kwargs
            dimensions = dict.fromkeys(DIMENSIONS, value)
            return {
                "passed": all(dimensions.values()),
                "dimensions": dimensions,
                "finding_codes": [],
            }

        monkeypatch.setattr(pack_module, "evaluate", mutated)
        assert not PACK.check(CheckRequest(FIXTURE)).contract_passed


@pytest.mark.parametrize("reader", ["spec", "run"])
def test_bounded_readers_never_use_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
    if reader == "spec":
        load_spec(FIXTURE)
    else:
        output = tmp_path / "run.json"
        output.write_text("{}", encoding="utf-8")
        with pytest.raises(ArtifactError):
            _read(output)


@pytest.mark.parametrize("reader", ["spec", "run"])
def test_bounded_readers_complete_short_regular_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    real_read = os.read

    def short_read(descriptor: int, count: int) -> bytes:
        return real_read(descriptor, min(count, 7))

    monkeypatch.setattr(os, "read", short_read)
    if reader == "spec":
        assert load_spec(FIXTURE).scenario_ids == ("V1", "V2", "V3")
    else:
        output = tmp_path / "run.json"
        PACK.run(RunRequest(FIXTURE, "V1", "reference", output))
        assert _read(output)["result"]["agent_id"] == "reference"


@pytest.mark.parametrize("reader", ["spec", "run"])
def test_bounded_readers_reject_sparse_and_symlink_inputs(
    tmp_path: Path, reader: str
) -> None:
    limit = MAX_SPEC_BYTES if reader == "spec" else 1024 * 1024
    sparse = tmp_path / "sparse"
    with sparse.open("wb") as stream:
        stream.truncate(limit + 1)
    link = tmp_path / "link"
    link.symlink_to(sparse)
    error = SpecSchemaError if reader == "spec" else ArtifactError
    loader = load_spec if reader == "spec" else _read
    with pytest.raises(error, match="exceeds"):
        loader(sparse)
    with pytest.raises(error, match=r"symlink|regular"):
        loader(link)


def test_new_surfaces_name_authoritative_spdx_owner() -> None:
    paths = [
        *Path("src/operatebench/contributions/prospire/property_compliance").rglob("*"),
        Path("examples/contributions/prospire/property_compliance/operation.yaml"),
        Path("docs/contributions/prospire/property_compliance.md"),
        Path(__file__),
    ]
    for path in paths:
        if path.is_file() and "__pycache__" not in path.parts:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            assert "SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD" in first
            assert "Dwelly Ltd" not in first
