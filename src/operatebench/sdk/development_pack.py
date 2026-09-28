"""SDK commands over the shared, non-evidence development runtime.

Only reviewed Python factories are accepted. Negative expectations are authored
inputs, never derived from the execution being checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from operatebench.sdk.api import (
    CheckRequest,
    CommandResult,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.development_record import read_record
from operatebench.sdk.development_runtime import (
    SCOPE,
    DevelopmentFactories,
    DevelopmentRuntimeError,
    replay,
    run,
    run_mock_model,
)
from operatebench.sdk.errors import OperationPackError


@dataclass(frozen=True)
class DevelopmentCommands:
    factories: DevelopmentFactories
    agent_ids: tuple[str, ...]
    reference_agents: tuple[str, ...]
    reference_scenarios: tuple[str, ...]
    negative_controls: tuple[Mapping[str, Any], ...]

    def validate(self, request: ValidateRequest) -> CommandResult:
        spec = self.factories.load_spec(request.spec_path)
        return CommandResult(
            True,
            {
                "scope": SCOPE,
                "pack_id": self.factories.pack_id,
                "operation_id": spec.operation_id,
                "spec_digest": spec.content_digest,
                "scenarios": list(spec.scenarios),
                "profiles": {
                    s: dict(self.factories.profile_identity(spec, s))
                    for s in spec.scenarios
                },
            },
        )

    def run(self, request: RunRequest) -> CommandResult:
        if request.agent_id not in (*self.agent_ids, "reference-mock"):
            raise OperationPackError(f"unknown development agent {request.agent_id!r}")
        try:
            if request.agent_id == "reference-mock":
                record = run_mock_model(
                    self.factories,
                    request.spec_path,
                    request.scenario_id,
                    output=request.output_path,
                )
            else:
                record = run(
                    self.factories,
                    request.spec_path,
                    request.scenario_id,
                    request.agent_id,
                    output=request.output_path,
                )
        except DevelopmentRuntimeError as exc:
            raise OperationPackError(str(exc)) from exc
        # Exact replay is mandatory in this lane, including --no-self-check.
        return CommandResult(record["evaluation"]["reliable"], record)

    def replay(self, request: ReplayRequest) -> CommandResult:
        try:
            result = replay(
                self.factories, request.spec_path, read_record(request.run_path)
            )
        except DevelopmentRuntimeError as exc:
            raise OperationPackError(str(exc)) from exc
        return CommandResult(result["consistent"], result)

    def check(self, request: CheckRequest) -> CommandResult:
        spec = self.factories.load_spec(request.spec_path)
        if set(spec.scenarios) != set(self.reference_scenarios):
            raise OperationPackError(
                "check requires the complete authored scenario matrix"
            )
        selected = request.agent_ids if request.agent_ids is not None else self.agent_ids
        if not selected or any(a not in self.agent_ids for a in selected):
            raise OperationPackError(
                "check requires known fixed agents with authored oracles"
            )
        rows = []
        try:
            for agent in self.reference_agents:
                if agent not in selected:
                    continue
                for scenario in self.reference_scenarios:
                    grade = run(self.factories, request.spec_path, scenario, agent)[
                        "evaluation"
                    ]
                    passed = (
                        grade["reliable"]
                        and all(grade["dimensions"].values())
                        and not grade["findings"]
                    )
                    rows.append(
                        {
                            "agent_id": agent,
                            "scenario_id": scenario,
                            "passed": passed,
                            "evaluation": grade,
                        }
                    )
            for oracle in self.negative_controls:
                if oracle["agent_id"] not in selected:
                    continue
                grade = run(
                    self.factories,
                    request.spec_path,
                    oracle["scenario_id"],
                    oracle["agent_id"],
                )["evaluation"]
                failures = {k for k, v in grade["dimensions"].items() if not v}
                codes = {f["code"] for f in grade["findings"]}
                passed = (
                    not grade["reliable"]
                    and failures == set(oracle["failed_dimensions"])
                    and codes == set(oracle["finding_codes"])
                    and all(
                        grade["dimensions"].get(k) is True for k in oracle["must_pass"]
                    )
                )
                rows.append(
                    {
                        "agent_id": oracle["agent_id"],
                        "scenario_id": oracle["scenario_id"],
                        "passed": passed,
                        "expected": dict(oracle),
                        "evaluation": grade,
                    }
                )
        except DevelopmentRuntimeError as exc:
            raise OperationPackError(str(exc)) from exc
        covered = {r["agent_id"] for r in rows}
        passed = (
            bool(rows) and set(selected) <= covered and all(r["passed"] for r in rows)
        )
        return CommandResult(
            passed,
            {
                "scope": SCOPE,
                "pack_id": self.factories.pack_id,
                "passed": passed,
                "rows": rows,
            },
        )
