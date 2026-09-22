"""Executable convergence proofs for the canonical Phase-1 evidence contract."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import ModelAgent
from operatebench.agents.openai_responses import outcome_tools
from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Escalate, Wait
from operatebench.core.read_contract import (
    ACTION_WITHOUT_RETRIEVAL,
    REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED,
    ActionEvidenceContract,
)
from operatebench.domains.lettings.maintenance import operation as operation_module
from operatebench.domains.lettings.maintenance import spec as spec_module
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import (
    action_schema_view,
    load_spec,
    maintenance_action_evidence_contract,
)
from tests.model_transport import TEST_MODEL, ReferenceDrivenTransport

SPEC = "examples/operatebench/maintenance_v0_1.yaml"
TARGETS = {
    "authorise_supplier_work": ("V1", ("checkpoint_1",), "quote_1:v1"),
    "request_invoice_validation": ("V1", ("evidence_1",), "invoice_1"),
    "request_payment": ("V1", ("invoice_1", "evidence_1"), "quote_1:v1"),
    "request_exception_resolution": (
        "V2",
        ("checkpoint_1", "quote_1:v1"),
        "work_cycle_2",
    ),
}


def _engine(spec: Any, scenario_id: str, agent: Any) -> Engine:
    return Engine(
        MaintenanceOperation(spec, scenario_id),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opinst_evidence_matrix_00000000",
            "scenario_id": scenario_id,
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=spec.scenario(scenario_id).dispatch_failures,
    )


def _evaluate(spec: Any, scenario_id: str, outcome: Any, trajectory: Any) -> Any:
    return evaluate(
        status=outcome.status,
        replay_final=outcome.replay_final,
        simulated_minutes=outcome.simulated_minutes,
        invocations=outcome.invocations,
        final_state=outcome.final_state,
        trajectory=trajectory,
        events=outcome.events,
        scenario=spec.scenario(scenario_id),
        replay_ok=None,
    )


def _discipline_codes(result: Any) -> set[str]:
    dimension = next(
        item for item in result.dimensions if item.name == "retrieval_discipline"
    )
    return {finding.code for finding in dimension.findings}


def _matrix_cases() -> list[tuple[str, str, tuple[str, ...], str, bool]]:
    cases: list[tuple[str, str, tuple[str, ...], str, bool]] = []
    for action, (_scenario, required, extra) in TARGETS.items():
        missing_code = (
            "INCOMPLETE_CHECKPOINT_CONTEXT"
            if action == "request_exception_resolution"
            else "MISSING_AUTHORITY_EVIDENCE"
        )
        cases.append((action, "read_only_no_citation", (), missing_code, False))
        cases.append(
            (
                action,
                "citation_without_reads",
                required,
                ACTION_WITHOUT_RETRIEVAL,
                False,
            )
        )
        for omitted in required:
            supplied = tuple(ref for ref in required if ref != omitted)
            cases.append(
                (
                    action,
                    f"omitted_{omitted}",
                    supplied,
                    missing_code,
                    False,
                )
            )
        cases.append(
            (
                action,
                "wrong_ref",
                ("not_an_operation_record",),
                "UNRESOLVED_EVIDENCE",
                False,
            )
        )
        cases.append((action, "required_plus_valid_extra", (*required, extra), "", True))
    return cases


class EvidenceMatrixAgent:
    """Run the real reference path and perturb exactly one target proposal."""

    agent_id = "phase1_evidence_matrix"

    def __init__(
        self,
        action_type: str,
        mode: str,
        evidence_refs: tuple[str, ...],
    ) -> None:
        self.reference = RetrievingReferenceAgent()
        self.action_type = action_type
        self.mode = mode
        self.evidence_refs = evidence_refs
        self.engine: Engine | None = None
        self.attacked = False
        self.before: dict[str, Any] | None = None
        self.immutability_checked = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.reference.begin_episode(identity)

    def _snapshot(self) -> dict[str, Any]:
        assert self.engine is not None
        return {
            "state": deepcopy(self.engine.domain.canonical_state(self.engine.state)),
            "queue": deepcopy(vars(self.engine.queue)),
            "conditionals": deepcopy(self.engine._conditionals),
            "timer_counter": self.engine._timer_counter,
            "delivered": deepcopy(self.engine._delivered),
        }

    def decide(self, observation: Any) -> Any:
        assert self.engine is not None
        if self.attacked and not self.immutability_checked and self.before is not None:
            assert self._snapshot() == self.before
            self.immutability_checked = True
            return Wait(reason="matrix refusal observed", fallback_after_minutes=1)

        outcome = self.reference.decide(observation)
        is_target = (
            isinstance(outcome, Act) and outcome.action_type == self.action_type
        ) or (
            isinstance(outcome, Escalate)
            and self.action_type == "request_exception_resolution"
        )
        if is_target and not self.attacked:
            self.attacked = True
            if self.mode != "required_plus_valid_extra":
                self.before = self._snapshot()
            if self.mode == "citation_without_reads":
                tools = maintenance_action_evidence_contract().required_tools_for(
                    self.action_type
                )
                for tool in tools:
                    self.engine._served.pop(tool, None)
                    self.engine._read_history.pop(tool, None)
            if isinstance(outcome, Escalate):
                return Escalate(
                    checkpoint_id=outcome.checkpoint_id,
                    exception_type=outcome.exception_type,
                    evidence_refs=self.evidence_refs,
                    deadline_after_minutes=outcome.deadline_after_minutes,
                    rationale="evidence matrix perturbation",
                )
            assert isinstance(outcome, Act)
            return Act(
                action_type=outcome.action_type,
                payload=outcome.payload,
                evidence_refs=self.evidence_refs,
                rationale="evidence matrix perturbation",
            )
        return outcome


@pytest.mark.parametrize(
    "action_type,mode,evidence_refs,expected_code,accepted",
    _matrix_cases(),
    ids=lambda value: str(value),
)
def test_real_engine_all_four_citation_matrix(
    action_type: str,
    mode: str,
    evidence_refs: tuple[str, ...],
    expected_code: str,
    accepted: bool,
) -> None:
    spec = load_spec(SPEC)
    scenario_id = TARGETS[action_type][0]
    agent = EvidenceMatrixAgent(action_type, mode, evidence_refs)
    engine = _engine(spec, scenario_id, agent)
    agent.engine = engine
    outcome = engine.run()

    proposals = [
        row
        for row in outcome.trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == action_type
    ]
    assert proposals
    proposal_id = proposals[0]["proposal_id"]
    refusals = [
        row
        for row in outcome.trajectory
        if row.get("record_type") == "action_rejected"
        and row.get("proposal_id") == proposal_id
    ]
    effects = [
        row
        for row in outcome.trajectory
        if row.get("record_type") == "effect_accepted"
        and row.get("proposal_id") == proposal_id
    ]
    if accepted:
        assert refusals == []
        assert len(effects) == 1
    else:
        assert [row["code"] for row in refusals] == [expected_code]
        assert effects == []
        assert agent.immutability_checked


def test_actual_model_request_boundary_has_nested_obligations_and_wire_refs() -> None:
    spec = load_spec(SPEC)
    transport = ReferenceDrivenTransport()
    engine = _engine(
        spec,
        "V2",
        ModelAgent(transport, model=TEST_MODEL, agent_id="model_boundary_matrix"),
    )
    engine.run()

    schemas = action_schema_view()
    request_schema = schemas["request_exception_resolution"]
    visible = [
        request.prompt["observation"]["action_schemas"]["request_exception_resolution"]
        for request in transport.requests
    ]
    actionable = next(schema for schema in visible if "evidence_refs" in schema)
    assert actionable == {
        name: request_schema[name] for name in ("required", "evidence_refs")
    }
    selector = actionable["evidence_refs"]["required"]["quote_record_key"]["select"]
    assert selector == (
        "key == approval.quote_id + ':v' + approval.quote_version where "
        "approval=list_checkpoints.approvals[payload.approval_checkpoint_id]"
    )
    act = next(item for item in outcome_tools() if item["name"] == "act")
    evidence_wire = act["parameters"]["properties"]["evidence_refs"]
    assert "array" in evidence_wire["type"]
    assert evidence_wire["items"]["type"] == "string"


@pytest.mark.parametrize("action_type", TARGETS)
def test_evaluator_all_four_reconstruct_missing_citations_without_trusting_refusals(
    action_type: str,
) -> None:
    spec = load_spec(SPEC)
    scenario_id = TARGETS[action_type][0]
    outcome = _engine(spec, scenario_id, RetrievingReferenceAgent()).run()
    trajectory = [dict(row) for row in outcome.trajectory]
    proposal = next(
        row
        for row in trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == action_type
    )
    proposal["evidence_refs"] = []
    trajectory.append(
        {
            **proposal,
            "record_type": "action_rejected",
            "code": "FORGED_CLEAN_REFUSAL",
            "detail": "untrusted",
        }
    )

    assert "REQUIRED_EVIDENCE_REF_NOT_CITED" in _discipline_codes(
        _evaluate(spec, scenario_id, outcome, trajectory)
    )


SELECTOR_TOOLS = {
    "authorise_supplier_work": ("list_checkpoints",),
    "request_invoice_validation": (
        "list_billing",
        "get_case_record",
        "list_authoritative_records",
    ),
    "request_payment": (
        "list_billing",
        "get_case_record",
        "list_authoritative_records",
    ),
    "request_exception_resolution": ("list_checkpoints", "list_quotes"),
}


@pytest.mark.parametrize(
    "action_type,tool",
    [
        (action_type, tool)
        for action_type, tools in SELECTOR_TOOLS.items()
        for tool in tools
    ],
)
def test_evaluator_selector_records_are_prior_individual_and_payload_gated(
    action_type: str, tool: str
) -> None:
    spec = load_spec(SPEC)
    scenario_id = TARGETS[action_type][0]
    outcome = _engine(spec, scenario_id, RetrievingReferenceAgent()).run()
    trajectory = [deepcopy(row) for row in outcome.trajectory]
    proposal_index = next(
        index
        for index, row in enumerate(trajectory)
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == action_type
    )
    served_index = next(
        index
        for index in range(proposal_index - 1, -1, -1)
        if trajectory[index].get("record_type") == "retrieval_served"
        and trajectory[index].get("tool") == tool
    )
    original_records = trajectory[served_index]["records"]
    trajectory[served_index]["records"] = {}
    later = deepcopy(trajectory[served_index])
    later["records"] = original_records
    trajectory.insert(proposal_index + 1, later)

    codes = _discipline_codes(_evaluate(spec, scenario_id, outcome, trajectory))
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in codes

    malformed = deepcopy(trajectory)
    malformed[proposal_index]["payload"] = {}
    malformed_codes = _discipline_codes(_evaluate(spec, scenario_id, outcome, malformed))
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED not in malformed_codes


def _without_payment_work_selector() -> ActionEvidenceContract:
    original = maintenance_action_evidence_contract()
    declarations: dict[str, dict[str, Any]] = {}
    for key in original.outcome_keys():
        selectors = original.evidence_for(key)
        if key == "request_payment":
            selectors = selectors[:1]
        declarations[key] = {
            "reads": dict(original.requirements.get(key, {})),
            "evidence_refs": selectors,
        }
    return ActionEvidenceContract(declarations)


def test_one_selector_perturbation_moves_model_runtime_and_independent_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(SPEC)
    original_schema = action_schema_view()["request_payment"]["evidence_refs"]
    reference_outcome = _engine(spec, "V1", RetrievingReferenceAgent()).run()
    forged = [deepcopy(row) for row in reference_outcome.trajectory]
    payment = next(
        row
        for row in forged
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_payment"
    )
    payment["evidence_refs"] = ["invoice_1"]
    assert "REQUIRED_EVIDENCE_REF_NOT_CITED" in _discipline_codes(
        _evaluate(spec, "V1", reference_outcome, forged)
    )

    perturbed = _without_payment_work_selector()
    monkeypatch.setattr(
        spec_module, "maintenance_action_evidence_contract", lambda: perturbed
    )
    changed_schema = action_schema_view()["request_payment"]["evidence_refs"]
    assert changed_schema != original_schema
    assert set(changed_schema["required"]) == {"invoice_id"}

    agent = EvidenceMatrixAgent(
        "request_payment", "required_plus_valid_extra", ("invoice_1",)
    )
    runtime = _engine(spec, "V1", agent)
    agent.engine = runtime
    runtime_outcome = runtime.run()
    payment_effects = [
        row
        for row in runtime_outcome.trajectory
        if row.get("record_type") == "effect_accepted"
        and row.get("action_type") == "request_payment"
    ]
    assert payment_effects

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("evaluator used the runtime selector resolver")

    monkeypatch.setattr(operation_module, "resolve_required_evidence", poisoned)
    assert "REQUIRED_EVIDENCE_REF_NOT_CITED" not in _discipline_codes(
        _evaluate(spec, "V1", reference_outcome, forged)
    )


def test_runtime_guard_census_binds_each_canonical_action_exactly_once() -> None:
    source = Path(operation_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "_required_evidence_verdict"
        ):
            continue
        assert len(node.args) >= 2
        action = node.args[1]
        assert isinstance(action, ast.Constant) and isinstance(action.value, str)
        guarded.append(action.value)
    assert sorted(guarded) == sorted(TARGETS)
    assert all(guarded.count(action) == 1 for action in TARGETS)

    handler_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_handle_")
    }
    forbidden_local_logic: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in handler_names:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Sub):
                forbidden_local_logic.append((node.name, "set subtraction"))
            if isinstance(child, ast.Compare) and any(
                isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
            ):
                names = {
                    name.id for name in ast.walk(child) if isinstance(name, ast.Name)
                }
                if names & {"evidence_refs", "required", "missing"}:
                    forbidden_local_logic.append((node.name, "evidence membership"))
    assert forbidden_local_logic == []
