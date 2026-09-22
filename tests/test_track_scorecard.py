"""Offline synthetic controls; these are not model trial results."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/track_scorecard.py"


def test_two_success_controls(tmp_path):
    from operatebench.artifact import write_artifact
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode

    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    rows = []
    spec = load_spec(ROOT / "examples/operatebench/maintenance_v0_1.yaml")
    for index, scenario in enumerate(("V1", "V3")):
        artifact = tmp_path / f"control-{index}.json"
        write_artifact(run_episode(spec, scenario, "reference"), artifact)
        a = json.loads(artifact.read_text())
        assert a["evaluation"]["reliable"] is True
        report = tmp_path / f"report-{index}.json"
        report.write_text(json.dumps({"evaluation": a["evaluation"]}))
        rows.append(
            {
                "trial_id": f"synthetic-{index}",
                "execution_run_id": None,
                "source_commit": source,
                "reader_root": str(ROOT),
                "model": "reference",
                "profile": None,
                "scenario": scenario,
                "rubric": "original-evaluation",
                "artifact": str(artifact),
                "metrics": str(report),
                "ledger": None,
                "partial": None,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": rows})
    )
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(TOOL), str(manifest), str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    scorecard = json.loads((out / "scorecard.json").read_text())
    assert scorecard["summary"]["assigned"] == 2
    assert scorecard["summary"]["verified_completed"] == 2
    assert all(r["milestones_retrospective"] is None for r in scorecard["runs"])
    assert all(r["costs"] is None for r in scorecard["runs"])
    assert (out / "scorecard.csv").exists()
    assert (out / "scorecard.md").exists()
    assert scorecard["runs"][0]["provenance"]["checks"] == {
        "artifact_reader": True,
        "ledger_reader": None,
        "bundle_audit": None,
        "original_metrics_equal": True,
    }
    assert "artifact.py" in scorecard["runs"][0]["provenance"]["reader_files_sha256"]
    import pytest

    from tools.track_scorecard import build

    copied = tmp_path / "copied.json"
    a = json.loads(Path(rows[0]["artifact"]).read_text())
    a["note"] = "different wrapper, same operation identity"
    copied.write_text(json.dumps(a))
    duplicate = dict(rows[0], trial_id="renamed", artifact=str(copied))
    manifest.write_text(
        json.dumps(
            {"schema": "operatebench.scorecard-manifest.v1", "runs": [rows[0], duplicate]}
        )
    )
    with pytest.raises(ValueError, match="duplicate operation_instance_id"):
        build(manifest)


def excluded_manifest(tmp_path):
    from operatebench.execution_ledger import TERMINAL_EXCLUDED, ExecutionLedgerJournal
    from tests.execution_ledger_fixtures import LATER, faulted_call, header

    ledger = tmp_path / "ledger.ndjson"
    h = header()
    with ExecutionLedgerJournal.open(ledger, h) as journal:
        journal.record_call(faulted_call())
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    row = {
        "trial_id": "assigned-1",
        "execution_run_id": h.execution_run_id,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "reader_root": str(ROOT),
        "model": h.provider.model,
        "profile": None,
        "scenario": h.scenario_id,
        "rubric": "original-evaluation",
        "ledger": str(ledger),
        "artifact": None,
        "metrics": None,
        "partial": None,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [row]})
    )
    return manifest, row, h


def test_excluded_partial_and_missing_stay_unknown(tmp_path):
    from operatebench.partial_evidence import PartialExecutionEvidence
    from tools.track_scorecard import build

    manifest, row, h = excluded_manifest(tmp_path)
    partial = tmp_path / "partial.ndjson"
    writer = PartialExecutionEvidence(
        {"operation_instance_id": h.operation_instance_id, "scenario_id": h.scenario_id},
        path=partial,
    )
    writer.state({"status": "RUNNING"}, at="before_agent")
    writer.close()
    with partial.open("ab") as file:
        file.write(b'{"kind":')
    row["partial"] = str(partial)
    missing = dict(
        row, trial_id="assigned-2", execution_run_id=None, ledger=None, partial=None
    )
    manifest.write_text(
        json.dumps(
            {"schema": "operatebench.scorecard-manifest.v1", "runs": [row, missing]}
        )
    )
    hashes = {p: p.read_bytes() for p in (manifest, partial, Path(row["ledger"]))}
    result = build(manifest)
    assert result["summary"] == {
        "assigned": 2,
        "verified_completed": 0,
        "completion_unknown": 2,
        "quality_evaluable": 0,
        "quality_passed": 0,
    }
    r = result["runs"][0]
    assert r["reliable"] is None and r["dimensions"] is None
    assert r["completion_observed"] is None and r["completion_verified"] is False
    assert r["partial_diagnostics"]["scoring"] == "NON-SCORED"
    assert r["partial_diagnostics"]["truncated_tail"] is True
    assert r["execution_details"]["attempt_faults"] == ["provider_server_error"]
    assert r["costs"]["measured_cost_usd"] == "0"
    assert r["costs"]["forfeited_reservation_usd"] != "0"
    assert r["wall_time_seconds"] is None
    assert result["runs"][1]["provider_attempt_latency_seconds_sum"] is None
    assert all(p.read_bytes() == before for p, before in hashes.items())


def test_duplicate_bundle_and_trial_rejected(tmp_path):
    import pytest

    from tools.track_scorecard import build

    manifest, row, _ = excluded_manifest(tmp_path)
    for second in (
        dict(row),
        dict(row, trial_id="other"),
        dict(row, trial_id="other", execution_run_id="forged"),
    ):
        manifest.write_text(
            json.dumps(
                {"schema": "operatebench.scorecard-manifest.v1", "runs": [row, second]}
            )
        )
        with pytest.raises(ValueError, match="duplicate"):
            build(manifest)


def test_original_milestone_vector_retained(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    from operatebench.artifact import write_artifact
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode
    from tools.track_scorecard import worker

    artifact = tmp_path / "a.json"
    write_artifact(
        run_episode(
            load_spec(ROOT / "examples/operatebench/maintenance_v0_1.yaml"),
            "V1",
            "reference",
        ),
        artifact,
    )
    evaluation = json.loads(artifact.read_text())["evaluation"]
    vector = {"example_report_predicate": {"earned": True, "receipts": [{"index": 1}]}}
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"evaluation": evaluation, "milestones_retrospective": vector})
    )
    result = worker(
        {
            "artifact": str(artifact),
            "ledger": None,
            "partial": None,
            "metrics": str(report),
            "scenario": "V1",
            "model": "reference",
            "execution_run_id": None,
            "preserve_report_milestones": True,
        }
    )
    assert result["milestones_retrospective"] == vector


def test_budget_vs_provider_and_unknown_latency(tmp_path):
    from operatebench.execution_ledger import TERMINAL_EXCLUDED, ExecutionLedgerJournal
    from tests.execution_ledger_fixtures import (
        LATER,
        call,
        controls,
        faulted_attempt,
        faulted_call,
        header,
    )
    from tools.track_scorecard import build

    manifest, row, _ = excluded_manifest(tmp_path)
    budget = tmp_path / "budget.ndjson"
    h = header(controls=controls(max_provider_calls=1, expected_provider_calls=1))
    with ExecutionLedgerJournal.open(budget, h) as journal:
        journal.record_call(call())
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_budget"
        )
    row["ledger"] = str(budget)
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [row]})
    )
    r = build(manifest)["runs"][0]
    assert r["execution_cause"] == "provider_budget"
    assert r["cost_scope"] == "complete_ledger"
    assert r["reliable"] is None
    assert r["execution_details"]["max_provider_calls"] == 1
    unknown = tmp_path / "unknown.ndjson"
    with ExecutionLedgerJournal.open(unknown, header()) as journal:
        journal.record_call(
            faulted_call(attempts=(faulted_attempt(latency_seconds=None),))
        )
        journal.finalize(
            TERMINAL_EXCLUDED, ended_at=LATER, exclusion_code="provider_transport"
        )
    row["ledger"] = str(unknown)
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [row]})
    )
    r = build(manifest)["runs"][0]
    assert r["provider_attempt_latency_seconds_sum"] is None
    assert r["wall_time_seconds"] is None


def test_refusals_no_outputs(tmp_path):
    import pytest

    from tools.track_scorecard import build, partial_prefix, write_outputs

    manifest, row, _ = excluded_manifest(tmp_path)
    for key, value, message in (
        ("source_commit", "bad", "commit mismatch"),
        ("model", "wrong", "identity mismatch"),
        ("trial_id", None, "identity"),
    ):
        bad = dict(row)
        bad[key] = value
        manifest.write_text(
            json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [bad]})
        )
        with pytest.raises(ValueError, match=message):
            build(manifest)
    partial = tmp_path / "bad.ndjson"
    partial.write_text(
        '{"kind":"header","schema":"operatebench.partial-execution.v1","scoring":"SCORED"}\n'
    )
    with pytest.raises(ValueError, match="header invalid"):
        partial_prefix(partial, {})
    with pytest.raises(FileExistsError):
        write_outputs({}, tmp_path)


def test_v2_completed_transfer_is_not_reliable(tmp_path):
    from operatebench.artifact import write_artifact
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode
    from tools.track_scorecard import build

    manifest, row, _ = excluded_manifest(tmp_path)
    artifact = tmp_path / "v2.json"
    run = run_episode(
        load_spec(ROOT / "examples/operatebench/maintenance_v0_1.yaml"),
        "V2",
        "reference",
    )
    assert run.reliable is False
    assert run.evaluation.failed_dimensions == ("recovery", "obligations")
    write_artifact(run, artifact)
    evaluation = json.loads(artifact.read_text())["evaluation"]
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"evaluation": evaluation}))
    row.update(
        artifact=str(artifact),
        metrics=str(metrics),
        ledger=None,
        execution_run_id=None,
        model="reference",
        scenario="V2",
    )
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [row]})
    )
    scorecard = build(manifest)
    assert scorecard["summary"] == {
        "assigned": 1,
        "verified_completed": 1,
        "completion_unknown": 0,
        "quality_evaluable": 1,
        "quality_passed": 0,
    }
    result = scorecard["runs"][0]
    assert result["completion_observed"] is True
    assert result["completion_verified"] is True
    assert result["reliable"] is False
    assert result["dimensions"] == evaluation["dimensions"]
    assert result["provenance"]["checks"]["original_metrics_equal"] is True


def test_observed_noncompletion_is_false_not_unknown(tmp_path):
    import pytest

    from operatebench.artifact import write_artifact
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from operatebench.runner import run_episode
    from tools.track_scorecard import build

    manifest, row, _ = excluded_manifest(tmp_path)
    artifact = tmp_path / "wait.json"
    write_artifact(
        run_episode(
            load_spec(ROOT / "examples/operatebench/maintenance_v0_1.yaml"),
            "V1",
            "always_wait",
        ),
        artifact,
    )
    evaluation = json.loads(artifact.read_text())["evaluation"]
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"evaluation": evaluation}))
    row.update(
        artifact=str(artifact),
        metrics=str(metrics),
        ledger=None,
        execution_run_id=None,
        model="always_wait",
    )
    manifest.write_text(
        json.dumps({"schema": "operatebench.scorecard-manifest.v1", "runs": [row]})
    )
    r = build(manifest)["runs"][0]
    assert r["completion_observed"] is False
    assert r["completion_verified"] is False
    assert r["reliable"] is False
    assert r["dimensions"] is not None
    evaluation["reliable"] = True
    metrics.write_text(json.dumps({"evaluation": evaluation}))
    with pytest.raises(ValueError, match="original metrics mismatch"):
        build(manifest)


def test_incomplete_ledger_costs_are_prefix_only(tmp_path):
    from tools.track_scorecard import build

    manifest, row, _ = excluded_manifest(tmp_path)
    ledger = Path(row["ledger"])
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:-1]) + b'{"row')
    r = build(manifest)["runs"][0]
    assert r["execution_status"] == "incomplete"
    assert r["cost_scope"] == "validated_prefix_only"
    assert r["reliable"] is None
    assert r["completion_observed"] is None
