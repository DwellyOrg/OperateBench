"""Read-only retrospective tracking, not execution, replay or a new grader."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "operatebench.scorecard.v1"
IDENTITY = (
    "trial_id",
    "execution_run_id",
    "source_commit",
    "model",
    "profile",
    "scenario",
    "rubric",
)


def load(path: Path) -> Any:
    return json.loads(
        path.read_text(), parse_constant=lambda x: fail(f"invalid JSON {x}")
    )


def fail(message: str) -> Any:
    raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def partial_prefix(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Validate v1 framing/shape/identity, NOT causal replay or business state."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    truncated = bool(lines and not lines[-1].endswith(b"\n"))
    if truncated:
        lines.pop()
    records = [json.loads(line) for line in lines]
    if not records:
        fail("partial header missing")
    header = records[0]
    if (header.get("kind"), header.get("schema"), header.get("scoring")) != (
        "header",
        "operatebench.partial-execution.v1",
        "NON-SCORED",
    ):
        fail("partial header invalid")
    if any(header["identity"].get(k) != v for k, v in identity.items()):
        fail("partial identity mismatch")
    shapes = {
        "decision": ("decision", dict),
        "receipt": ("receipt", dict),
        "state": ("last_state", dict),
        "closure": ("classification", str),
    }
    closed = False
    counts: dict[str, int] = {}
    for record in records[1:]:
        kind = record.get("kind")
        if closed or kind not in shapes:
            fail("partial record order/kind invalid")
        key, expected = shapes[kind]
        if not isinstance(record.get(key), expected):
            fail("partial record shape invalid")
        if kind == "closure":
            if record[key] not in ("scored", "excluded", "aborted"):
                fail("partial closure invalid")
            closed = True
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "schema": "operatebench.partial-prefix-diagnostic.v1",
        "scoring": "NON-SCORED",
        "validation": (
            "v1 framing, record shapes, order and identity only; NOT causal validation"
        ),
        "records_validated": len(records),
        "record_counts": counts,
        "closure_present": closed,
        "truncated_tail": truncated,
        "business_completion": None,
        "reliable": None,
        "dimensions": None,
    }


def worker(row: dict[str, Any]) -> dict[str, Any]:
    # Imports deliberately occur only in the isolated, version-pinned child.
    import socket

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("scorecard network forbidden")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    import operatebench.artifact as artifact_module
    from operatebench.artifact import read_artifact
    from operatebench.execution_bundle import audit_execution_bundle_files
    from operatebench.execution_ledger import read_execution_ledger
    from operatebench.version import OPERATEBENCH_VERSION

    a = read_artifact(row["artifact"]) if row["artifact"] else None
    ledger = read_execution_ledger(row["ledger"]) if row["ledger"] else None
    if a is not None and a["scenario_id"] != row["scenario"]:
        fail("artifact scenario mismatch")
    if ledger is not None:
        h = ledger.header
        if (h.execution_run_id, h.scenario_id, h.provider.model) != (
            row["execution_run_id"],
            row["scenario"],
            row["model"],
        ):
            fail("ledger identity mismatch")
        if h.provider.settings.get("request_profile") != row["profile"]:
            fail("ledger profile mismatch")
    milestones = None
    if a is not None:
        if a.get("provider_execution") is not None:
            if ledger is None:
                fail("provider artifact requires ledger")
            audit_execution_bundle_files(row["artifact"], row["ledger"])
        elif (
            ledger is not None
            or row["execution_run_id"] is not None
            or a["agent_id"] != row["model"]
        ):
            fail("deterministic control identity mismatch")
        metrics = load(Path(row["metrics"]))
        for key in row.get("metrics_keys", []):
            metrics = metrics[key]
        if metrics["evaluation"] != a["evaluation"]:
            fail("original metrics mismatch")
        if row.get("preserve_report_milestones"):
            milestones = metrics.get("milestones_retrospective")
            if milestones is not None:
                if not isinstance(milestones, dict):
                    fail("milestone vector must be an object")
                for value in milestones.values():
                    if (
                        not isinstance(value, dict)
                        or type(value.get("earned")) is not bool
                        or not isinstance(value.get("receipts"), list)
                        or value["earned"] != bool(value["receipts"])
                    ):
                        fail("milestone receipt vector invalid")
    summary = ledger.summary() if ledger else None
    latencies = (
        [at.latency_seconds for call in ledger.calls for at in call.attempts]
        if ledger
        else None
    )
    partial = None
    if row["partial"]:
        identity = {"scenario_id": row["scenario"]}
        if ledger:
            identity["operation_instance_id"] = ledger.header.operation_instance_id
        elif a:
            identity["operation_instance_id"] = a["operation_instance_id"]
        else:
            fail("partial diagnostics require a ledger or artifact identity anchor")
        partial = partial_prefix(Path(row["partial"]), identity)
    details = (
        {
            "max_provider_calls": ledger.header.controls.max_provider_calls,
            "call_terminal_reasons": [c.terminal_reason for c in ledger.calls],
            "attempt_faults": [at.fault for c in ledger.calls for at in c.attempts],
        }
        if ledger
        else None
    )
    return {
        "checks": {
            "artifact_reader": True if a else None,
            "ledger_reader": True if ledger else None,
            "bundle_audit": True if a and ledger else None,
            "original_metrics_equal": True if a else None,
        },
        "operation_instance_id": a["operation_instance_id"]
        if a
        else ledger.header.operation_instance_id
        if ledger
        else None,
        "reader_engine_version": OPERATEBENCH_VERSION,
        "reader_files_sha256": {
            name: digest(Path(artifact_module.__file__).parent / name)
            for name in ("artifact.py", "execution_ledger.py", "execution_bundle.py")
        },
        "milestones_retrospective": milestones,
        "partial_diagnostics": partial,
        "execution_details": details,
        "evaluation": a["evaluation"] if a else None,
        "business_outcome": a["status"] if a else None,
        "ledger": summary,
        "provider_attempt_latency_seconds_sum": (
            sum(x for x in latencies if x is not None)
            if latencies is not None and all(x is not None for x in latencies)
            else None
        ),
    }


def build(manifest: Path) -> dict[str, Any]:
    manifest = manifest.resolve()
    inputs = {str(manifest): digest(manifest)}
    data = load(manifest)
    if data["schema"] != "operatebench.scorecard-manifest.v1":
        fail("unsupported manifest schema")
    rows = []
    operations: set[str] = set()
    seen: dict[str, set[str]] = {
        k: set() for k in ("trial_id", "execution_run_id", "artifact", "ledger")
    }
    for entry in data["runs"]:
        row = dict(entry)
        for key in IDENTITY:
            if key not in row:
                fail(f"missing identity: {key}")
            if key not in ("profile", "execution_run_id") and (
                not isinstance(row[key], str) or not row[key]
            ):
                fail(f"invalid identity: {key}")
        root = (manifest.parent / row["reader_root"]).resolve()
        if git(root, "rev-parse", "HEAD") != row["source_commit"]:
            fail("reader source commit mismatch")
        if git(root, "status", "--porcelain", "--", "src"):
            fail("reader source dirty")
        for key in ("artifact", "ledger", "metrics", "partial"):
            ref = row.get(key)
            row[key] = str((manifest.parent / ref).resolve()) if ref else None
            if row[key]:
                inputs[row[key]] = digest(Path(row[key]))
        for key, values in seen.items():
            value = (
                inputs[row[key]]
                if key in ("artifact", "ledger") and row[key]
                else row[key]
            )
            if value is not None:
                if value in values:
                    fail(f"duplicate {key}")
                values.add(value)
        env = {
            "PATH": os.defpath,
            "HOME": str(manifest.parent),
            "PYTHONPATH": str(root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        child = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(row),
            text=True,
            capture_output=True,
            env=env,
            cwd=root,
            timeout=120,
        )
        if child.returncode:
            fail("original reader refused: " + child.stderr)
        observed = json.loads(child.stdout)
        operation_id = observed["operation_instance_id"]
        if operation_id is not None:
            if operation_id in operations:
                fail("duplicate operation_instance_id")
            operations.add(operation_id)
        evaluation = observed["evaluation"]
        ledger = observed.pop("ledger")
        eligible = evaluation is not None
        result = {key: row[key] for key in IDENTITY}
        result.update(
            assigned=True,
            operation_instance_id=operation_id,
            evidence_eligible=eligible,
            completion_observed=evaluation["legitimate_completion"] if eligible else None,
            completion_verified=evaluation["legitimate_completion"]
            if eligible
            else False,
            reliable=evaluation["reliable"] if eligible else None,
            dimensions=evaluation["dimensions"] if eligible else None,
            business_outcome=observed["business_outcome"],
            execution_status=ledger["status"]
            if ledger
            else "missing_output"
            if not eligible
            else "deterministic_control",
            execution_cause=ledger["exclusion_code"] if ledger else None,
            costs=ledger["totals"] if ledger else None,
            cost_scope=(
                "complete_ledger" if ledger["complete"] else "validated_prefix_only"
            )
            if ledger
            else None,
            provider_attempt_latency_seconds_sum=observed[
                "provider_attempt_latency_seconds_sum"
            ],
            wall_time_seconds=None,
            milestones_retrospective=observed["milestones_retrospective"],
            partial_diagnostics=observed["partial_diagnostics"],
            execution_details=observed["execution_details"],
            provenance={
                "reader_source_commit": row["source_commit"],
                "reader_engine_version": observed["reader_engine_version"],
                "reader_files_sha256": observed["reader_files_sha256"],
                "checks": observed["checks"],
                "assignment_labels": (
                    "trial_id and rubric supplied by manifest; "
                    "not independently authenticated"
                ),
                "validation": (
                    "original artifact/ledger readers; bundle checked when provider-bound"
                ),
                "metrics": (
                    "existing report equality; not replay or independent preregistration"
                ),
                "milestones": (
                    "report-supplied retrospective receipt vector; "
                    "not recomputed or causally revalidated"
                ),
                "inputs_sha256": {
                    k: inputs[row[k]] if row[k] else None
                    for k in ("artifact", "ledger", "metrics", "partial")
                },
            },
        )
        rows.append(result)
    if any(digest(Path(p)) != h for p, h in inputs.items()):
        fail("input changed during validation")
    return {
        "schema": SCHEMA,
        "status": "retrospective tracking; no ranking or aggregate score",
        "summary": {
            "assigned": len(rows),
            "verified_completed": sum(r["completion_verified"] is True for r in rows),
            "completion_unknown": sum(r["completion_observed"] is None for r in rows),
            "quality_evaluable": sum(r["evidence_eligible"] for r in rows),
            "quality_passed": sum(r["reliable"] is True for r in rows),
        },
        "runs": rows,
        "source_inputs_sha256": inputs,
        "tracking_tool_sha256": digest(Path(__file__)),
    }


def write_outputs(result: dict[str, Any], destination: Path) -> None:
    # A new directory only: frozen inputs and existing outputs cannot be overwritten.
    destination.mkdir(mode=0o700)
    (destination / "scorecard.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        *IDENTITY,
        "assigned",
        "operation_instance_id",
        "business_outcome",
        "execution_details",
        "cost_scope",
        "evidence_eligible",
        "completion_observed",
        "completion_verified",
        "reliable",
        "execution_status",
        "execution_cause",
        "costs",
        "dimensions",
        "milestones_retrospective",
        "partial_diagnostics",
        "provider_attempt_latency_seconds_sum",
        "wall_time_seconds",
        "provenance",
    ]
    with (destination / "scorecard.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: json.dumps(row[key], ensure_ascii=True) for key in fields}
            for row in result["runs"]
        )
    text = (
        "# Retrospective scorecard tracking\n\n"
        "No weighted score, ranking, replay or regrading. null means unknown.\n\n"
    )
    text += "```json\n" + json.dumps(result["summary"], indent=2) + "\n```\n\n"
    text += (
        "Each assigned row (separate dimensions, immutable ledger cost categories):\n\n"
    )
    for row in result["runs"]:
        text += (
            "```json\n"
            + json.dumps(row, indent=2, ensure_ascii=True).replace("`", "\\u0060")
            + "\n```\n\n"
        )
    (destination / "scorecard.md").write_text(text)


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path, help="new private output directory")
    args = parser.parse_args()
    write_outputs(build(args.manifest), args.destination)


if __name__ == "__main__":
    main()
