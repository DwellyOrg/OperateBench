"""Static SDK adapter for the existing synthetic Maintenance vertical.

The adapter changes no execution, grading or artefact semantics. It loads the
same strict spec and calls the same runner, writer, replay and causal gate the
legacy CLI called directly.
"""

from __future__ import annotations

from pathlib import Path

from operatebench.artifact import replay_file, write_artifact
from operatebench.domains.lettings.maintenance.agents import agent_ids
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.runner import check_maintenance, run_episode
from operatebench.sdk.api import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.errors import OperationPackMismatchError

MAINTENANCE_PACK_ID = "lettings.maintenance.synthetic"
MAINTENANCE_OPERATION_TYPE = "lettings.maintenance.synthetic"

SCOPE_NOTE = (
    "Technical preview of the OperateBench Lifecycle construct on a fully "
    "synthetic fixture. Not a leaderboard, not a model ranking, not Dwelly "
    "production policy, and not a production operations engine."
)


def _bound_spec(path: str | Path) -> OperationSpec:
    spec = load_spec(path)
    if spec.operation_type != MAINTENANCE_OPERATION_TYPE:
        raise OperationPackMismatchError(
            f"pack {MAINTENANCE_PACK_ID!r} implements operation_type "
            f"{MAINTENANCE_OPERATION_TYPE!r}, but {spec.source} declares "
            f"{spec.operation_type!r}; a pack selection is not persisted in "
            "Artifact 8, so a wrong-pack run is refused before execution"
        )
    return spec


def _validate_lines(spec: OperationSpec) -> tuple[str, ...]:
    lines = [f"{spec.operation_id}: spec OK (static validation; no scenario executed)"]
    lines.extend(
        (
            f"  operation type    {spec.operation_type}",
            f"  operation version {spec.operation_version}",
            f"  semantic scenario {spec.semantic_scenario_id}",
            f"  privacy status    {spec.privacy_status}",
            f"  spec digest       {spec.spec_digest_sha256}",
        )
    )
    for scenario_id in spec.scenario_ids:
        scenario = spec.scenario(scenario_id)
        lines.append(
            f"    {scenario_id}  {len(scenario.events)} authored event(s), "
            f"declares {scenario.expected_terminal}, human budget "
            f"{scenario.human_checkpoint_budget}"
        )
        lines.append(f"        {scenario.label}")
    lines.extend(
        (
            f"  actors            {sorted(spec.actors)}",
            (
                "  note              declared terminals are not checked here; run "
                "run-maintenance and replay per variant to establish reachability"
            ),
            f"  disclaimer        {spec.disclaimer}",
            f"  scope             {SCOPE_NOTE}",
        )
    )
    return tuple(lines)


class MaintenancePack:
    """The existing vertical behind the SDK v2 command contract."""

    metadata = OperationPackMetadata(
        pack_id=MAINTENANCE_PACK_ID,
        operation_type=MAINTENANCE_OPERATION_TYPE,
        operation_id="lettings_maintenance_synthetic_v1",
        pack_version="0.1.0",
        display_name="Synthetic lettings maintenance",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="development",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/operatebench/maintenance_v0_1.yaml",
        agent_ids=tuple(sorted(agent_ids())),
        aliases=("maintenance",),
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        return CommandResult(
            contract_passed=True,
            payload=spec.identity_payload(),
            text_lines=_validate_lines(spec),
        )

    def run(self, request: RunRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        run = run_episode(
            spec,
            request.scenario_id,
            request.agent_id,
            self_check=request.self_check,
        )
        path = write_artifact(run, request.output_path)
        payload = {
            **run.summary(),
            "artifact_path": str(path),
            "evaluation": run.evaluation.as_dict(),
        }
        lines = [
            f"{spec.operation_id} {request.scenario_id} / {request.agent_id}",
            f"  status             {run.outcome.status}",
            (
                "  expected terminal  "
                f"{spec.scenario(request.scenario_id).expected_terminal}"
            ),
            f"  reliable           {run.reliable}",
            (
                f"  simulated time     {run.outcome.simulated_minutes} minute(s) over "
                f"{run.outcome.invocations} agent invocation(s)"
            ),
        ]
        for dimension in run.evaluation.dimensions:
            status = "OK" if dimension.ok else "FAILED"
            lines.append(f"  {dimension.name:<22} {status}")
            for finding in dimension.findings:
                lines.append(f"      ! {finding.code}: {finding.detail}")
        lines.extend(
            (
                f"  final state digest {run.outcome.final_state_digest_sha256}",
                f"  trajectory digest  {run.outcome.trajectory_digest_sha256}",
                f"  artefact           {path}",
                f"  scope              {SCOPE_NOTE}",
            )
        )
        return CommandResult(
            contract_passed=run.reliable,
            payload=payload,
            text_lines=tuple(lines),
        )

    def replay(self, request: ReplayRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        report = replay_file(spec, request.run_path)
        headline = "replay OK" if report.ok else "replay DIVERGED"
        lines = [
            f"{request.run_path}: {headline}",
            f"  scenario / agent   {report.scenario_id} / {report.agent_id}",
            f"  reproduced by      {report.reproduction}",
        ]
        if not report.total_comparison:
            lines.extend(
                (
                    f"  carried forward    {list(report.carried_forward)}",
                    (
                        "  note               the carried fields were taken from the "
                        "record rather than re-derived, so the comparison below is "
                        "not total"
                    ),
                )
            )
        for section, fields in report.sections.items():
            matched = all(fields.values())
            lines.append(f"  {section:<18} {'matches' if matched else 'DIFFERS'}")
            for name, field_matched in fields.items():
                if not field_matched:
                    lines.append(f"      - {name}")
        lines.extend(
            (
                f"  recorded digest    {report.recorded_final_state_digest}",
                f"  replayed digest    {report.replayed_final_state_digest}",
            )
        )
        lines.extend(f"      ! {difference}" for difference in report.differences)
        return CommandResult(
            contract_passed=report.ok,
            payload=report.as_dict(),
            text_lines=tuple(lines),
        )

    def check(self, request: CheckRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        report = check_maintenance(spec, agent_ids=request.agent_ids)
        lines = [
            f"{spec.operation_id}: causal acceptance contract",
            (
                f"  oracle             {report.oracle_id} {report.oracle_version} "
                f"({report.oracle_digest_sha256})"
            ),
        ]
        for check in report.checks:
            status = "OK" if check.ok else "FAILED"
            lines.append(
                f"  {check.agent_id:<20} {check.scenario_id:<3} {check.kind:<9} "
                f"reliable={check.reliable!s:<5} {status}"
            )
            if check.reference_expectation_id:
                lines.append(f"      expectation {check.reference_expectation_id}")
            if check.failed_dimensions:
                lines.append(f"      failed    {list(check.failed_dimensions)}")
            if check.finding_codes:
                lines.append(f"      findings  {list(check.finding_codes)}")
            lines.extend(f"      ! {problem}" for problem in check.problems)
        lines.extend(
            (
                f"causal acceptance: {'OK' if report.ok else 'FAILED'}",
                f"scope: {SCOPE_NOTE}",
            )
        )
        return CommandResult(
            contract_passed=report.ok,
            payload=report.as_dict(),
            text_lines=tuple(lines),
        )


MAINTENANCE_PACK = MaintenancePack()

__all__ = [
    "MAINTENANCE_OPERATION_TYPE",
    "MAINTENANCE_PACK",
    "MAINTENANCE_PACK_ID",
    "SCOPE_NOTE",
    "MaintenancePack",
]
