"""SDK adapter for the synthetic commerce return/refund development pack."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boundarybench.jsonsafe import canonical_json_bytes, ensure_json_safe
from operatebench._write_once import _write_once_bytes, _WriteOnceFailure
from operatebench.core.engine import Engine, EpisodeOutcome
from operatebench.core.errors import (
    AgentRegistryError,
    ArtifactError,
    OracleManifestError,
    SpecSchemaError,
)
from operatebench.core.evaluation import OperationEvaluation
from operatebench.core.instance import (
    new_operation_instance_id,
    require_operation_instance_id,
)
from operatebench.domains.commerce.return_refund.agents import (
    REFERENCE_AGENT_ID,
    agent_ids,
    build_agent,
)
from operatebench.domains.commerce.return_refund.evaluator import DIMENSIONS, evaluate
from operatebench.domains.commerce.return_refund.operation import ReturnRefundOperation
from operatebench.domains.commerce.return_refund.oracle import (
    NegativeControl,
    NegativeControlOracle,
    negative_control_oracle,
)
from operatebench.domains.commerce.return_refund.spec import (
    ReturnRefundSpec,
    ScenarioSpec,
    load_spec,
)
from operatebench.sdk.api import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)
from operatebench.sdk.errors import OperationPackMismatchError

PACK_ID = "commerce.return_refund.synthetic.v1"
OPERATION_TYPE = "commerce.return_refund.synthetic.v1"
PACK_VERSION = "0.1.0"
RUN_FORMAT = "operatebench.incubator-run.v1"
MAX_RUN_BYTES = 16 * 1024 * 1024
EXPECTED_TERMINALS = {
    "V1": "completed_refund_settled",
    "V2": "closed_refund_denied",
    "V3": "closed_return_expired",
}
EXPECTED_CHECKPOINTS = {
    "V1": ("high_value_refund_approval",),
    "V2": ("high_value_refund_approval",),
    "V3": (),
}

SCOPE_NOTE = (
    "Fully synthetic development/incubator operation pack. Its run record is "
    "not Artifact 8, is not evidence-eligible, and supports no official model, "
    "leaderboard or production-policy claim."
)


def _bound_spec(path: Path) -> ReturnRefundSpec:
    spec = load_spec(path)
    if spec.operation_type != OPERATION_TYPE:
        raise OperationPackMismatchError(
            f"pack {PACK_ID!r} implements operation_type {OPERATION_TYPE!r}, but "
            f"{spec.source} declares {spec.operation_type!r}"
        )
    if spec.operation_version != PACK_VERSION:
        raise OperationPackMismatchError(
            f"pack {PACK_ID!r} is version {PACK_VERSION!r}, but {spec.source} "
            f"declares operation_version {spec.operation_version!r}"
        )
    if spec.scenario_ids() != tuple(EXPECTED_TERMINALS):
        raise SpecSchemaError(
            f"{spec.source}: pack {PACK_ID!r} requires variants "
            f"{list(EXPECTED_TERMINALS)}, got {list(spec.scenario_ids())}"
        )
    for scenario_id, terminal in EXPECTED_TERMINALS.items():
        scenario = spec.scenario(scenario_id)
        declared = scenario.expected_terminal
        if declared != terminal:
            raise SpecSchemaError(
                f"{spec.source}: {scenario_id} must declare terminal {terminal!r}, "
                f"got {declared!r}"
            )
        required_checkpoints = EXPECTED_CHECKPOINTS[scenario_id]
        if scenario.required_checkpoint_types != required_checkpoints:
            raise SpecSchemaError(
                f"{spec.source}: {scenario_id} must require checkpoint types "
                f"{list(required_checkpoints)}, got "
                f"{list(scenario.required_checkpoint_types)}"
            )
        if scenario.human_checkpoint_budget != len(required_checkpoints):
            raise SpecSchemaError(
                f"{spec.source}: {scenario_id} human checkpoint budget must be "
                f"{len(required_checkpoints)}, got {scenario.human_checkpoint_budget}"
            )
    return spec


def _scenario(spec: ReturnRefundSpec, scenario_id: str) -> ScenarioSpec:
    try:
        return spec.scenario(scenario_id)
    except KeyError as exc:
        raise SpecSchemaError(
            f"unknown scenario {scenario_id!r}; expected one of "
            f"{list(spec.scenario_ids())}"
        ) from exc


def _identity(
    spec: ReturnRefundSpec,
    scenario_id: str,
    agent_id: str,
    operation_instance_id: str,
) -> dict[str, str]:
    return {
        "operation_id": spec.operation_id,
        "operation_instance_id": operation_instance_id,
        "scenario_id": scenario_id,
        "agent_id": agent_id,
        "spec_digest_sha256": spec.spec_digest_sha256,
    }


def _execute(
    spec: ReturnRefundSpec,
    scenario_id: str,
    agent_id: str,
    operation_instance_id: str,
) -> EpisodeOutcome:
    scenario = _scenario(spec, scenario_id)
    if agent_id not in agent_ids():
        raise AgentRegistryError(
            f"unknown return/refund agent {agent_id!r}; expected one of "
            f"{list(agent_ids())}"
        )
    domain = ReturnRefundOperation(spec, scenario_id)
    agent = build_agent(agent_id)
    return Engine(
        domain,
        agent,
        identity=_identity(spec, scenario_id, agent_id, operation_instance_id),
        dispatch_failures=frozenset(scenario.dispatch_failures),
    ).run()


def _same_episode(first: EpisodeOutcome, second: EpisodeOutcome) -> bool:
    return canonical_json_bytes(first.as_dict(), "first incubator episode") == (
        canonical_json_bytes(second.as_dict(), "second incubator episode")
    )


def _run_write_error(failure: _WriteOnceFailure) -> ArtifactError:
    target = failure.path
    if failure.reason == "parent_symlink":
        return ArtifactError(
            f"{failure.component} is a symlink on the path to incubator run {target}; "
            "the output cannot be redirected"
        )
    if failure.reason == "final_symlink":
        return ArtifactError(
            f"incubator run output {target} is a symlink; it is not followed"
        )
    if failure.reason == "exists":
        return ArtifactError(
            f"incubator run output {target} already exists; it is not overwritten"
        )
    if failure.reason == "unsupported":
        return ArtifactError(
            "cannot write an incubator run safely on this platform: "
            "descriptor-anchored no-follow filesystem primitives are unavailable"
        )
    if failure.reason in {"parent_create", "parent_open", "anchor"}:
        return ArtifactError(f"cannot create the directory for incubator run {target}")
    return ArtifactError(f"cannot write incubator run {target}")


def _write_run(path: Path, payload: Mapping[str, Any]) -> Path:
    detached = dict(payload)
    ensure_json_safe(detached, "return/refund incubator run")
    body = (json.dumps(detached, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(body) > MAX_RUN_BYTES:
        raise ArtifactError(
            "return/refund incubator run is "
            f"{len(body)} bytes; the replayable limit is {MAX_RUN_BYTES} bytes"
        )
    return _write_once_bytes(body, path, error_adapter=_run_write_error)


def _strict_json_object(pairs: list[tuple[str, Any]], source: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ArtifactError(f"incubator run {source} repeats JSON field {name!r}")
        result[name] = value
    return result


def _read_run(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            body = handle.read(MAX_RUN_BYTES + 1)
    except OSError as exc:
        raise ArtifactError(f"cannot read incubator run {path}: {exc}") from exc
    if len(body) > MAX_RUN_BYTES:
        raise ArtifactError(
            f"incubator run {path} exceeds the {MAX_RUN_BYTES}-byte replay limit"
        )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"incubator run {path} is not valid UTF-8: {exc}") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=lambda pairs: _strict_json_object(pairs, path),
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ArtifactError(f"cannot parse incubator run {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ArtifactError(f"incubator run {path} must contain one JSON object")
    required = {
        "format",
        "pack_id",
        "pack_version",
        "operation_type",
        "operation_id",
        "spec_digest_sha256",
        "scenario_id",
        "agent_id",
        "operation_instance_id",
        "self_check",
        "outcome",
        "evaluation",
        "scope",
    }
    if set(raw) != required:
        raise ArtifactError(
            f"incubator run fields are {sorted(raw)}, expected {sorted(required)}"
        )
    ensure_json_safe(dict(raw), f"incubator run {path}")
    return raw


def _run_payload(
    spec: ReturnRefundSpec,
    scenario_id: str,
    agent_id: str,
    operation_instance_id: str,
    outcome: EpisodeOutcome,
    evaluation: OperationEvaluation,
    self_check: bool | None,
) -> dict[str, Any]:
    return {
        "format": RUN_FORMAT,
        "pack_id": PACK_ID,
        "pack_version": PACK_VERSION,
        "operation_type": spec.operation_type,
        "operation_id": spec.operation_id,
        "spec_digest_sha256": spec.spec_digest_sha256,
        "scenario_id": scenario_id,
        "agent_id": agent_id,
        "operation_instance_id": operation_instance_id,
        "self_check": self_check,
        "outcome": outcome.as_dict(),
        "evaluation": evaluation.as_dict(),
        "scope": SCOPE_NOTE,
    }


def _run_once(
    spec: ReturnRefundSpec,
    scenario_id: str,
    agent_id: str,
    *,
    self_check: bool,
    operation_instance_id: str | None = None,
) -> tuple[EpisodeOutcome, OperationEvaluation, bool | None, str]:
    instance_id = operation_instance_id or new_operation_instance_id()
    outcome = _execute(spec, scenario_id, agent_id, instance_id)
    replay_ok: bool | None = None
    if self_check:
        reproduced = _execute(spec, scenario_id, agent_id, instance_id)
        replay_ok = _same_episode(outcome, reproduced)
    evaluation = evaluate(outcome, replay_ok, _scenario(spec, scenario_id))
    return outcome, evaluation, replay_ok, instance_id


def _evaluation_lines(
    spec: ReturnRefundSpec,
    scenario_id: str,
    agent_id: str,
    outcome: EpisodeOutcome,
    evaluation: OperationEvaluation,
) -> tuple[str, ...]:
    lines = [
        f"{spec.operation_id} {scenario_id} / {agent_id}",
        f"  status             {outcome.status}",
        f"  expected terminal  {_scenario(spec, scenario_id).expected_terminal}",
        f"  reliable           {evaluation.reliable}",
        f"  simulated time     {outcome.simulated_minutes} minute(s)",
    ]
    for dimension in evaluation.dimensions:
        lines.append(f"  {dimension.name:<22} {'OK' if dimension.ok else 'FAILED'}")
        lines.extend(
            f"      ! {finding.code}: {finding.detail}" for finding in dimension.findings
        )
    lines.extend(
        (
            f"  final state digest {outcome.final_state_digest_sha256}",
            f"  trajectory digest  {outcome.trajectory_digest_sha256}",
            f"  scope              {SCOPE_NOTE}",
        )
    )
    return tuple(lines)


def _control_fields(
    control: NegativeControl,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        str(control.agent_id),
        str(control.scenario_id),
        tuple(control.expected_failed_dimensions),
        tuple(control.required_finding_codes),
        tuple(control.must_pass_dimensions),
    )


def _bound_oracle(spec: ReturnRefundSpec) -> NegativeControlOracle:
    oracle = negative_control_oracle()
    problems: list[str] = []
    if oracle.operation_type != OPERATION_TYPE:
        problems.append(
            f"operation_type is {oracle.operation_type!r}, expected {OPERATION_TYPE!r}"
        )
    if oracle.authored_against_pack_version != PACK_VERSION:
        problems.append(
            "authored_against_pack_version is "
            f"{oracle.authored_against_pack_version!r}, expected {PACK_VERSION!r}"
        )
    if oracle.result_vector_dimensions != DIMENSIONS:
        problems.append("result_vector_dimensions do not match the evaluator")
    expected_agents = set(agent_ids()) - {REFERENCE_AGENT_ID}
    if set(oracle.agent_ids) != expected_agents:
        problems.append(
            f"control agents are {sorted(oracle.agent_ids)}, expected "
            f"{sorted(expected_agents)}"
        )
    unknown_scenarios = sorted(
        {
            control.scenario_id
            for control in oracle.controls
            if control.scenario_id not in spec.scenario_ids()
        }
    )
    if unknown_scenarios:
        problems.append(f"controls name unknown scenarios {unknown_scenarios}")
    if problems:
        raise OracleManifestError(
            "return/refund control oracle is not bound to this pack: "
            + "; ".join(problems)
        )
    return oracle


class ReturnRefundPack:
    """The commerce pilot behind the SDK v2 command contract."""

    metadata = OperationPackMetadata(
        pack_id=PACK_ID,
        operation_type=OPERATION_TYPE,
        operation_id="commerce_return_refund_synthetic_v1",
        pack_version=PACK_VERSION,
        display_name="Synthetic commerce return and refund",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec="examples/operatebench/commerce_return_refund_v0_1.yaml",
        agent_ids=tuple(sorted(agent_ids())),
        aliases=("return-refund",),
        evidence_eligible=False,
    )

    def validate(self, request: ValidateRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        lines = [
            f"{spec.operation_id}: spec OK (static validation; no scenario executed)",
            f"  operation type    {spec.operation_type}",
            f"  operation version {spec.operation_version}",
            f"  semantic scenario {spec.semantic_scenario_id}",
            f"  privacy status    {spec.privacy_status}",
            f"  spec digest       {spec.spec_digest_sha256}",
        ]
        for scenario_id in spec.scenario_ids():
            scenario = spec.scenario(scenario_id)
            lines.append(
                f"    {scenario_id}  {len(scenario.events)} authored event(s), "
                f"declares {scenario.expected_terminal}, human budget "
                f"{scenario.human_checkpoint_budget}"
            )
        lines.extend(
            (
                f"  disclaimer        {spec.disclaimer}",
                f"  scope             {SCOPE_NOTE}",
            )
        )
        return CommandResult(
            contract_passed=True,
            payload=spec.identity_payload(),
            text_lines=tuple(lines),
        )

    def run(self, request: RunRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        outcome, evaluation, replay_ok, instance_id = _run_once(
            spec,
            request.scenario_id,
            request.agent_id,
            self_check=request.self_check,
        )
        payload = _run_payload(
            spec,
            request.scenario_id,
            request.agent_id,
            instance_id,
            outcome,
            evaluation,
            replay_ok,
        )
        path = _write_run(request.output_path, payload)
        result_payload = {
            **payload,
            "artifact_path": str(path),
            "reliable": evaluation.reliable,
        }
        lines = list(
            _evaluation_lines(
                spec, request.scenario_id, request.agent_id, outcome, evaluation
            )
        )
        lines.insert(-1, f"  incubator run      {path}")
        return CommandResult(
            contract_passed=evaluation.reliable,
            payload=result_payload,
            text_lines=tuple(lines),
        )

    def replay(self, request: ReplayRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        recorded = _read_run(request.run_path)
        expected_bindings = {
            "format": RUN_FORMAT,
            "pack_id": PACK_ID,
            "pack_version": self.metadata.pack_version,
            "operation_type": spec.operation_type,
            "operation_id": spec.operation_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
            "scope": SCOPE_NOTE,
        }
        binding_mismatches = [
            name
            for name, expected in expected_bindings.items()
            if recorded.get(name) != expected
        ]
        if binding_mismatches:
            raise ArtifactError(
                f"incubator run {request.run_path} has invalid binding field(s) "
                f"{binding_mismatches}; replay requires exact pack, version, "
                "operation, fixture and scope bindings"
            )
        scenario_id = recorded.get("scenario_id")
        agent_id = recorded.get("agent_id")
        if type(scenario_id) is not str or type(agent_id) is not str:
            raise ArtifactError("incubator run scenario_id and agent_id must be text")
        instance_id = require_operation_instance_id(
            recorded.get("operation_instance_id"), "incubator run operation_instance_id"
        )
        outcome = _execute(spec, scenario_id, agent_id, instance_id)
        reproduced = _execute(spec, scenario_id, agent_id, instance_id)
        live_replay_ok = _same_episode(outcome, reproduced)
        recorded_self_check = recorded.get("self_check")
        if recorded_self_check is not None and type(recorded_self_check) is not bool:
            raise ArtifactError("incubator run self_check must be true, false or null")
        # Preserve the evaluation shape of a run whose check was deliberately
        # skipped. A recorded boolean is compared below, but only the two fresh
        # executions can supply determinism evidence to a new evaluation.
        evaluation_replay_ok = None if recorded_self_check is None else live_replay_ok
        evaluation = evaluate(
            outcome,
            evaluation_replay_ok,
            _scenario(spec, scenario_id),
        )
        comparisons = {
            "deterministic_replay": live_replay_ok,
            "self_check": (
                recorded_self_check is None or recorded_self_check is live_replay_ok
            ),
            "outcome": canonical_json_bytes(
                outcome.as_dict(), "replayed incubator outcome"
            )
            == canonical_json_bytes(
                recorded.get("outcome"), "recorded incubator outcome"
            ),
            "evaluation": canonical_json_bytes(
                evaluation.as_dict(), "replayed incubator evaluation"
            )
            == canonical_json_bytes(
                recorded.get("evaluation"), "recorded incubator evaluation"
            ),
        }
        ok = all(comparisons.values())
        replay_payload = {
            "ok": ok,
            "pack_id": PACK_ID,
            "scenario_id": scenario_id,
            "agent_id": agent_id,
            "comparisons": comparisons,
            "recorded_self_check": recorded_self_check,
            "live_self_check": live_replay_ok,
            "recorded_final_state_digest": (
                recorded.get("outcome", {}).get("final_state_digest_sha256")
                if isinstance(recorded.get("outcome"), Mapping)
                else None
            ),
            "replayed_final_state_digest": outcome.final_state_digest_sha256,
            "scope": SCOPE_NOTE,
        }
        return CommandResult(
            contract_passed=ok,
            payload=replay_payload,
            text_lines=(
                f"{request.run_path}: replay {'OK' if ok else 'DIVERGED'}",
                f"  scenario / agent   {scenario_id} / {agent_id}",
                (
                    "  live determinism   "
                    f"{'matches' if comparisons['deterministic_replay'] else 'DIFFERS'}"
                ),
                (
                    "  recorded selfcheck "
                    f"{'matches' if comparisons['self_check'] else 'DIFFERS'}"
                ),
                (
                    "  outcome            "
                    f"{'matches' if comparisons['outcome'] else 'DIFFERS'}"
                ),
                (
                    "  evaluation         "
                    f"{'matches' if comparisons['evaluation'] else 'DIFFERS'}"
                ),
                f"  scope              {SCOPE_NOTE}",
            ),
        )

    def check(self, request: CheckRequest) -> CommandResult:
        spec = _bound_spec(request.spec_path)
        selected = tuple(request.agent_ids or self.metadata.agent_ids)
        unknown = sorted(set(selected) - set(self.metadata.agent_ids))
        if unknown:
            raise AgentRegistryError(
                f"unknown return/refund check agent(s) {unknown}; expected "
                f"{list(self.metadata.agent_ids)}"
            )
        oracle = _bound_oracle(spec)
        controls = {str(control.agent_id): control for control in oracle.controls}
        checks: list[dict[str, Any]] = []
        lines = [
            f"{spec.operation_id}: incubator causal acceptance contract",
            (
                f"  oracle             {oracle.oracle_id} {oracle.oracle_version} "
                f"({oracle.digest_sha256})"
            ),
        ]
        if REFERENCE_AGENT_ID in selected:
            for scenario_id in spec.scenario_ids():
                _outcome, evaluation, replay_ok, _instance = _run_once(
                    spec,
                    scenario_id,
                    REFERENCE_AGENT_ID,
                    self_check=True,
                )
                reference_problems = (
                    []
                    if evaluation.reliable and replay_ok
                    else ["reference agent was not reliable under deterministic replay"]
                )
                checks.append(
                    {
                        "agent_id": REFERENCE_AGENT_ID,
                        "scenario_id": scenario_id,
                        "kind": "reference",
                        "reliable": evaluation.reliable,
                        "failed_dimensions": list(evaluation.failed_dimensions),
                        "finding_codes": list(evaluation.finding_codes),
                        "problems": reference_problems,
                    }
                )
        for agent_id in selected:
            if agent_id == REFERENCE_AGENT_ID:
                continue
            control = controls.get(agent_id)
            if control is None:
                checks.append(
                    {
                        "agent_id": agent_id,
                        "scenario_id": "",
                        "kind": "negative",
                        "reliable": False,
                        "failed_dimensions": [],
                        "finding_codes": [],
                        "problems": ["negative agent has no separate oracle control"],
                    }
                )
                continue
            (
                _declared_agent,
                scenario_id,
                expected_failed,
                required_codes,
                must_pass,
            ) = _control_fields(control)
            _outcome, evaluation, replay_ok, _instance = _run_once(
                spec, scenario_id, agent_id, self_check=True
            )
            failed = tuple(evaluation.failed_dimensions)
            codes = set(evaluation.finding_codes)
            passed = set(evaluation.passed_dimensions)
            control_problems: list[str] = []
            if evaluation.reliable:
                control_problems.append("negative control unexpectedly passed")
            if failed != expected_failed:
                control_problems.append(
                    f"failed dimensions {list(failed)} != oracle {list(expected_failed)}"
                )
            missing_codes = sorted(set(required_codes) - codes)
            if missing_codes:
                control_problems.append(f"required finding codes absent: {missing_codes}")
            missing_passes = sorted(set(must_pass) - passed)
            if missing_passes:
                control_problems.append(f"must-pass dimensions failed: {missing_passes}")
            if replay_ok is not True:
                control_problems.append(
                    "negative control did not reproduce deterministically"
                )
            checks.append(
                {
                    "agent_id": agent_id,
                    "scenario_id": scenario_id,
                    "kind": "negative",
                    "reliable": evaluation.reliable,
                    "failed_dimensions": list(failed),
                    "finding_codes": list(evaluation.finding_codes),
                    "problems": control_problems,
                }
            )
        ok = bool(checks) and all(not check["problems"] for check in checks)
        for check in checks:
            lines.append(
                f"  {check['agent_id']:<30} {check['scenario_id']:<3} "
                f"{check['kind']:<9} {'OK' if not check['problems'] else 'FAILED'}"
            )
            lines.extend(f"      ! {problem}" for problem in check["problems"])
        lines.extend(
            (
                f"incubator causal acceptance: {'OK' if ok else 'FAILED'}",
                f"scope: {SCOPE_NOTE}",
            )
        )
        payload = {
            "ok": ok,
            "pack_id": PACK_ID,
            "oracle": oracle.as_dict(),
            "checks": checks,
            "scope": SCOPE_NOTE,
        }
        return CommandResult(
            contract_passed=ok,
            payload=payload,
            text_lines=tuple(lines),
        )


RETURN_REFUND_PACK = ReturnRefundPack()

__all__ = [
    "EXPECTED_CHECKPOINTS",
    "EXPECTED_TERMINALS",
    "MAX_RUN_BYTES",
    "OPERATION_TYPE",
    "PACK_ID",
    "PACK_VERSION",
    "RETURN_REFUND_PACK",
    "RUN_FORMAT",
    "SCOPE_NOTE",
    "ReturnRefundPack",
]
