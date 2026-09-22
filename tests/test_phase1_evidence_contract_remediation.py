"""Adversarial regression proofs for the Phase-1 evidence contract remediation."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from operatebench.core.clock import shift_minutes
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.read_contract import (
    REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED,
    ActionEvidenceContract,
    EvidenceResolution,
    PayloadRegistryKey,
    QuoteForSelectedApproval,
    WorkEvidenceForInvoice,
    action_evidence_contract_problem,
    resolve_required_evidence,
)
from operatebench.core.retrieval import RetrievalRequest
from operatebench.core.retrieval_evidence import public_record_version
from operatebench.domains.lettings.maintenance import evaluator as evaluator_module
from operatebench.domains.lettings.maintenance import operation as operation_module
from operatebench.domains.lettings.maintenance.evaluator import (
    RETRIEVAL_DISCIPLINE_CODES,
    evaluate,
)
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    maintenance_retrieval_catalogue,
    maintenance_retrieval_record_contract,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    action_schema_view,
    load_spec,
    maintenance_action_evidence_contract,
)
from operatebench.domains.lettings.maintenance.state import Approval, Cycle, Quote
from operatebench.runner import run_episode

SPEC = "examples/operatebench/maintenance_v0_1.yaml"


def _problem(contract: ActionEvidenceContract) -> str | None:
    return action_evidence_contract_problem(
        contract,
        catalogue=maintenance_retrieval_catalogue(),
        retrieval_tools=MAINTENANCE_RETRIEVAL_TOOLS,
        payload_schemas=MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
        outcome_keys=(*MAINTENANCE_ACTION_PAYLOAD_SCHEMAS, "complete"),
    )


def _reevaluate(run: Any, trajectory: list[dict[str, Any]]) -> Any:
    spec = load_spec(SPEC)
    return evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=run.outcome.final_state,
        trajectory=trajectory,
        events=run.outcome.events,
        scenario=spec.scenario("V1"),
        replay_ok=None,
    )


def _discipline(result: Any) -> Any:
    return next(d for d in result.dimensions if d.name == "retrieval_discipline")


def test_selector_algebra_is_closed_and_deeply_immutable() -> None:
    import operatebench.core.read_contract as module

    assert not hasattr(module, "ForeignRecordField")
    assert not hasattr(module, "CompositeRecordKey")
    selector = QuoteForSelectedApproval(
        "list_checkpoints",
        "approvals",
        "approval_checkpoint_id",
        "list_quotes",
        "quotes",
    )
    with pytest.raises(FrozenInstanceError):
        selector.root = "invoices"  # type: ignore[misc]
    assert all(not isinstance(value, list) for value in vars(selector).values())


def test_malformed_external_lists_are_detached_and_validation_never_raises() -> None:
    malformed_selector: list[object] = ["mutable"]
    raw_refs: list[object] = [malformed_selector]
    contract = ActionEvidenceContract({"request_payment": {"evidence_refs": raw_refs}})
    malformed_selector.append("later mutation")
    raw_refs.clear()

    assert contract.evidence_for("request_payment") == (("mutable",),)
    problem = _problem(contract)
    assert problem is not None and "unknown evidence selector" in problem


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"request_payment": []},
        {"request_payment": {"reads": []}},
        {"request_payment": {"evidence_refs": 1}},
    ],
)
def test_malformed_contract_construction_and_validation_fail_by_name(raw: object) -> None:
    contract = ActionEvidenceContract(raw)  # type: ignore[arg-type]
    problem = _problem(contract)
    assert isinstance(problem, str) and problem


def test_projection_discloses_fixed_semantics_without_resolved_values() -> None:
    schemas = action_schema_view()
    work = schemas["request_invoice_validation"]["evidence_refs"]["required"]
    assert work == {
        "work_evidence_id": {
            "from": "list_authoritative_records.authoritative_records",
            "select": (
                "key == cycle.work_evidence_id; kind == work_evidence; "
                "cycle_id == cycle.cycle_id where "
                "invoice=list_billing.invoices[payload.invoice_id], "
                "cycle=get_case_record.cycles[invoice.cycle_id]"
            ),
        }
    }
    quote = schemas["request_exception_resolution"]["evidence_refs"]["required"]
    assert quote == {
        "approval_checkpoint_id": {
            "from": "list_checkpoints.approvals",
            "select": "key == payload.approval_checkpoint_id",
        },
        "quote_record_key": {
            "from": "list_quotes.quotes",
            "select": (
                "key == approval.quote_id + ':v' + approval.quote_version where "
                "approval=list_checkpoints.approvals[payload.approval_checkpoint_id]"
            ),
        },
    }
    assert "checkpoint_1" not in repr(schemas)
    assert "quote_1:v1" not in repr(schemas)


def test_projection_is_detached_from_the_internal_typed_selector_contract() -> None:
    contract = maintenance_action_evidence_contract()
    internal = contract.evidence_for("request_payment")
    assert isinstance(internal[0], PayloadRegistryKey)
    assert isinstance(internal[1], WorkEvidenceForInvoice)

    projected = action_schema_view()
    projected["request_payment"]["evidence_refs"]["required"].clear()

    assert isinstance(contract.evidence_for("request_payment")[0], PayloadRegistryKey)
    assert action_schema_view()["request_payment"]["evidence_refs"]["required"]


def test_compact_evidence_projection_has_an_independent_byte_bound() -> None:
    evidence_projection = {
        action: schema["evidence_refs"] for action, schema in action_schema_view().items()
    }
    assert len(json.dumps(evidence_projection, sort_keys=True)) < 1_800


def test_model_projection_progresses_each_action_only_after_its_own_reads() -> None:
    schemas = action_schema_view()

    def projected(
        served: object,
        record_contract: object = maintenance_retrieval_record_contract(),
    ) -> dict[str, object]:
        observation = AgentObservation(
            now="2025-01-01T00:00:00Z",
            operation_id="op",
            operation_instance_id="opinst_x",
            invocation_index=1,
            turn_index=0,
            policy={},
            actors={},
            message_fixture_ids=(),
            action_schemas=schemas,
            retrieval={
                "catalogue": maintenance_retrieval_catalogue(),
                "record_contract": record_contract,
                "served": served,
            },
        )
        return model_projection(observation)["action_schemas"]

    def envelope(tool: str, *, ok: bool = True, error_code: str | None = None):
        entry = maintenance_retrieval_catalogue()[tool]
        suffix, context, _algorithm, length = maintenance_retrieval_record_contract()
        records: dict[str, object] = {}
        return {
            "tool": tool,
            "source": entry["source"],
            "authority": entry["authority"],
            "record_id": f"{entry['source']}:{suffix}",
            "record_version": public_record_version(
                records,
                context=context,
                length=length,
            ),
            "as_of": "2025-01-01T00:00:00Z",
            "schema_id": entry["schema_id"],
            "records": records,
            "ok": ok,
            "error_code": error_code,
        }

    before_reads = projected({})
    assert before_reads == {
        action: {"reads": schema["reads"]} for action, schema in schemas.items()
    }

    after_case = projected({"get_case_record": envelope("get_case_record")})
    assert after_case["request_supplier_visit"] == {
        name: schemas["request_supplier_visit"][name] for name in ("required", "optional")
    }
    assert after_case["request_payment"] == {
        "reads": {
            "list_authoritative_records": "required",
            "list_billing": "required",
        }
    }

    payment_reads = {
        tool: envelope(tool)
        for tool in ("get_case_record", "list_authoritative_records", "list_billing")
    }
    after_payment_reads = projected(payment_reads)
    assert after_payment_reads["request_payment"] == {
        name: schemas["request_payment"][name] for name in ("required", "evidence_refs")
    }
    assert after_payment_reads["authorise_supplier_work"] == {
        "reads": {"list_checkpoints": "required", "list_quotes": "required"}
    }

    for field, forged in (
        ("record_id", "canonical-looking:forged"),
        ("record_version", "0" * 16),
        ("records", {"unread": 1}),
        ("as_of", "not-an-instant"),
    ):
        forgery = deepcopy(payment_reads)
        forgery["list_billing"][field] = forged
        assert projected(forgery)["request_payment"] == {
            "reads": schemas["request_payment"]["reads"]
        }

    synthetic = deepcopy(payment_reads)
    synthetic_records = {"synthetic": {"status": "OPEN"}}
    _suffix, context, _algorithm, length = maintenance_retrieval_record_contract()
    synthetic["list_billing"]["records"] = synthetic_records
    synthetic["list_billing"]["record_version"] = public_record_version(
        synthetic_records,
        context=context,
        length=length,
    )
    assert after_payment_reads == projected(synthetic)

    for malformed_contract in (
        None,
        (),
        ("maintenance_case", "maintenance retrieval record", "unsupported", 16),
        (
            "maintenance_case",
            "maintenance retrieval record",
            "canonical-json-sha256-prefix-v1",
            0,
        ),
        (
            "maintenance_case",
            "maintenance retrieval record",
            "canonical-json-sha256-prefix-v1",
            65,
        ),
        (
            "maintenance_case",
            "maintenance retrieval record",
            "canonical-json-sha256-prefix-v1",
            16,
            "extra",
        ),
    ):
        assert projected(payment_reads, malformed_contract)["request_payment"] == {
            "reads": schemas["request_payment"]["reads"]
        }

    action = "request_supplier_visit"
    schemas[action]["reads"] = {
        "get_case_record": "required",
        "list_quotes": "optional",
    }
    assert projected({"get_case_record": envelope("get_case_record")})[action] == {
        "required": schemas[action]["required"],
        "optional": schemas[action]["optional"],
        "reads": {"list_quotes": "optional"},
    }


def test_optional_read_does_not_create_an_evaluator_retrieval_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ActionEvidenceContract(
        {
            "request_supplier_visit": {
                "reads": {
                    "get_case_record": "required",
                    "list_quotes": "optional",
                }
            }
        }
    )
    monkeypatch.setattr(
        evaluator_module.maintenance_spec,
        "maintenance_action_evidence_contract",
        lambda: contract,
    )
    authority = maintenance_retrieval_catalogue()["get_case_record"]["authority"]
    trajectory = (
        {
            "record_type": "agent_invoked",
            "invocation_index": 1,
            "trigger_claim_values": [],
            "trigger_is_authoritative": False,
        },
        {
            "record_type": "retrieval_served",
            "invocation_index": 1,
            "tool": "get_case_record",
            "authority": authority,
            "record_version": "version-1",
            "records": {},
            "ok": True,
        },
        {
            "record_type": "action_proposed",
            "action_type": "request_supplier_visit",
            "proposal_id": "proposal-1",
            "payload": {},
            "evidence_refs": [],
            "at": "2025-01-01T00:00:00Z",
        },
    )

    dimension = evaluator_module._retrieval_discipline(trajectory)

    assert dimension.ok
    assert dimension.findings == ()


def test_model_projection_does_not_count_error_envelopes_as_served_reads() -> None:
    schemas = action_schema_view()
    catalogue = maintenance_retrieval_catalogue()
    entry = catalogue["get_case_record"]
    envelope = {
        "tool": "get_case_record",
        "source": entry["source"],
        "authority": entry["authority"],
        "record_id": "record",
        "record_version": "version",
        "as_of": "2025-01-01T00:00:00Z",
        "schema_id": entry["schema_id"],
        "records": {},
        "ok": False,
        "error_code": "FAILED",
    }
    observation = AgentObservation(
        now="2025-01-01T00:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_x",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=(),
        action_schemas=schemas,
        retrieval={
            "catalogue": catalogue,
            "served": {"get_case_record": envelope},
        },
    )
    assert model_projection(observation)["action_schemas"] == {
        action: {"reads": schema["reads"]} for action, schema in schemas.items()
    }


@pytest.mark.parametrize(
    "served",
    [
        True,
        "served",
        {"fabricated": None},
        {"fabricated": {"records": {}}},
    ],
)
def test_model_projection_rejects_malformed_served_boundary(served: object) -> None:
    schemas = action_schema_view()
    observation = AgentObservation(
        now="2025-01-01T00:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_x",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=(),
        action_schemas=schemas,
        retrieval={"served": served},
    )

    assert model_projection(observation)["action_schemas"] == {
        action: {"reads": schema["reads"]} for action, schema in schemas.items()
    }


def test_closed_selectors_validate_tool_root_and_payload_ownership() -> None:
    selectors = [
        PayloadRegistryKey("list_quotes", "approvals", "approval_checkpoint_id"),
        WorkEvidenceForInvoice(
            "list_billing",
            "invoices",
            "moon_id",
            "get_case_record",
            "cycles",
            "list_authoritative_records",
            "authoritative_records",
        ),
        QuoteForSelectedApproval(
            "list_checkpoints",
            "approvals",
            "approval_checkpoint_id",
            "list_billing",
            "quotes",
        ),
    ]
    for selector in selectors:
        contract = ActionEvidenceContract(
            {"request_exception_resolution": {"evidence_refs": (selector,)}}
        )
        assert _problem(contract) is not None


def test_evaluator_names_unreconstructable_obligated_selector_and_ignores_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V1", "reference")
    forged: list[dict[str, Any]] = []
    for original in run.outcome.trajectory:
        row = dict(original)
        if row.get("record_type") == "retrieval_served":
            row["records"] = {}
        if row.get("record_type") == "action_rejected":
            row["code"] = "FORGED_CLEAN"
        forged.append(row)
    forged = [row for row in forged if row.get("record_type") != "action_rejected"]

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("evaluator called the runtime evidence resolver")

    monkeypatch.setattr(operation_module, "resolve_required_evidence", poisoned)
    result = _reevaluate(run, forged)
    codes = {finding.code for finding in _discipline(result).findings}
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in codes
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in RETRIEVAL_DISCIPLINE_CODES


def test_valid_but_unresolved_exception_selector_has_an_evidence_obligation() -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V1", "always_escalate")

    proposals = [
        row
        for row in run.outcome.trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_exception_resolution"
    ]
    assert proposals
    assert all(
        row.get("payload", {}).get("approval_checkpoint_id")
        == "requested_exception_checkpoint"
        for row in proposals
    )
    codes = {finding.code for finding in _discipline(run.evaluation).findings}
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in codes


@pytest.mark.parametrize("selector", [pytest.param(None, id="missing"), 7, ""])
def test_genuinely_malformed_selector_payload_has_no_evidence_obligation(
    selector: object,
) -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V2", "reference")
    trajectory = [deepcopy(row) for row in run.outcome.trajectory]
    proposal = next(
        row
        for row in trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_exception_resolution"
    )
    if selector is None:
        proposal["payload"].pop("approval_checkpoint_id")
    else:
        proposal["payload"]["approval_checkpoint_id"] = selector

    codes = {
        finding.code for finding in _discipline(_reevaluate(run, trajectory)).findings
    }
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED not in codes


def test_later_retrieval_cannot_cure_an_unreconstructable_proposal() -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V1", "reference")
    trajectory = [dict(row) for row in run.outcome.trajectory]
    proposal_index = next(
        index
        for index, row in enumerate(trajectory)
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "authorise_supplier_work"
    )
    prior = next(
        row
        for row in reversed(trajectory[:proposal_index])
        if row.get("record_type") == "retrieval_served"
        and row.get("tool") == "list_checkpoints"
    )
    saved_records = prior["records"]
    prior["records"] = {}
    later = dict(prior)
    later["records"] = saved_records
    trajectory.insert(proposal_index + 1, later)

    codes = {
        finding.code for finding in _discipline(_reevaluate(run, trajectory)).findings
    }
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in codes


def test_additional_valid_citable_reference_is_intentionally_accepted() -> None:
    """Required citations are a subset obligation, never exact-set equality."""
    from operatebench.core.read_contract import resolve_required_evidence

    selector = PayloadRegistryKey("list_billing", "invoices", "invoice_id")
    contract = ActionEvidenceContract({"request_payment": {"evidence_refs": (selector,)}})
    resolution = resolve_required_evidence(
        contract,
        "request_payment",
        {"invoice_id": "invoice_1"},
        {"invoices": {"invoice_1": {"invoice_id": "invoice_1"}}},
    )
    supplied = {"invoice_1", "work_evidence_1"}
    assert resolution.problem is None
    assert set(resolution.required) <= supplied


class _RuntimeContext:
    def __init__(self) -> None:
        self.now = "2031-03-03T09:00:00Z"
        self.records: list[tuple[str, dict[str, Any]]] = []

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None:
        self.records.append((record_type, dict(payload)))

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None:
        del event_id, event_type, actor_id, delay_minutes, payload, triggers_agent

    def cancel_timer(self, event_id: str) -> bool:
        del event_id
        return False

    def dispatch_fails(self, message_fixture_id: str) -> bool:
        del message_fixture_id
        return False


def _two_eligible_approval_runtime_fixture() -> tuple[Any, Any, _RuntimeContext]:
    spec = load_spec(SPEC)
    domain = operation_module.MaintenanceOperation(spec, "V1")
    state = domain.initial_state()
    context = _RuntimeContext()
    for number in (1, 2):
        cycle_id = f"work_cycle_{number}"
        quote_id = f"quote_{number}"
        checkpoint_id = f"checkpoint_{number}"
        state.cycles[cycle_id] = Cycle(cycle_id=cycle_id, kind="APPROVED_WORK")
        state.quotes[f"{quote_id}:v{number}"] = Quote(
            quote_id=quote_id,
            version=number,
            cycle_id=cycle_id,
            amount_minor=60_000 + number,
            currency="GBP",
            scope_digest=f"scope_{number}",
        )
        state.approvals[checkpoint_id] = Approval(
            checkpoint_id=checkpoint_id,
            quote_id=quote_id,
            quote_version=number,
            cycle_id=cycle_id,
            amount_minor=60_000 + number,
            currency="GBP",
            scope_digest=f"scope_{number}",
            opened_at=context.now,
            deadline_at=shift_minutes(context.now, 60),
            status="RESOLVED_REJECTED",
        )
    return domain, state, context


def _assert_runtime_selects_non_first_approval() -> tuple[Any, list[Any]]:
    payload = {
        "exception_type": "APPROVAL_REJECTED",
        "approval_checkpoint_id": "checkpoint_2",
        "deadline_after_minutes": 1_440,
    }
    first_refs = ("checkpoint_1", "quote_1:v1")
    selected_refs = ("checkpoint_2", "quote_2:v2", "work_cycle_1")

    domain, state, context = _two_eligible_approval_runtime_fixture()
    before = deepcopy(state.canonical())
    refused = domain.apply_action(
        state, "request_exception_resolution", payload, first_refs, context
    )
    assert refused.accepted is False
    assert refused.code == "INCOMPLETE_CHECKPOINT_CONTEXT"
    assert state.canonical() == before

    domain, state, context = _two_eligible_approval_runtime_fixture()
    retrievals = domain.serve_retrieval(
        state,
        tuple(
            RetrievalRequest(tool=tool)
            for tool in ("get_case_record", "list_checkpoints", "list_quotes")
        ),
        context.now,
    )
    checkpoint_records = next(
        result.records["approvals"]
        for result in retrievals
        if result.tool == "list_checkpoints"
    )
    assert tuple(checkpoint_records) == ("checkpoint_1", "checkpoint_2")
    accepted = domain.apply_action(
        state, "request_exception_resolution", payload, selected_refs, context
    )
    assert accepted.accepted is True
    exception = next(iter(state.exceptions.values()))
    assert exception.cycle_id == "work_cycle_2"
    assert exception.evidence_refs == selected_refs
    return payload, list(retrievals)


def test_two_eligible_approvals_bind_runtime_and_evaluator_to_non_first_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, retrievals = _assert_runtime_selects_non_first_approval()
    records: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = [
        {
            "record_type": "agent_invoked",
            "invocation_index": 1,
            "at": "2031-03-03T09:00:00Z",
        }
    ]
    for result in retrievals:
        records.update(result.records)
        trajectory.append(
            {
                "record_type": "retrieval_served",
                "invocation_index": 1,
                "tool": result.tool,
                "authority": result.authority,
                "record_version": result.record_version,
                "records": result.records,
                "ok": result.ok,
                "at": result.as_of,
            }
        )
    proposal = {
        "record_type": "action_proposed",
        "proposal_id": "proposal_selected_second",
        "action_type": "request_exception_resolution",
        "payload": payload,
        "evidence_refs": ["checkpoint_2", "quote_2:v2", "work_cycle_1"],
        "at": "2031-03-03T09:00:00Z",
    }
    trajectory.append(proposal)

    reconstructed = evaluator_module._required_citations_from_prior_records(
        "request_exception_resolution", payload, records
    )
    assert reconstructed.required == ("checkpoint_2", "quote_2:v2")

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("evaluator called the runtime resolver")

    monkeypatch.setattr(operation_module, "resolve_required_evidence", poisoned)
    correct = evaluator_module._retrieval_discipline(trajectory)
    assert not correct.findings
    wrong = deepcopy(trajectory)
    wrong[-1]["evidence_refs"] = ["checkpoint_1", "quote_1:v1", "work_cycle_1"]
    assert "REQUIRED_EVIDENCE_REF_NOT_CITED" in {
        finding.code for finding in evaluator_module._retrieval_discipline(wrong).findings
    }


def test_two_approval_runtime_proof_kills_first_record_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def first_record_mutant(*args: Any, **kwargs: Any) -> EvidenceResolution:
        del args, kwargs
        return EvidenceResolution(("checkpoint_1", "quote_1:v1"), None)

    monkeypatch.setattr(
        operation_module, "resolve_required_evidence", first_record_mutant
    )
    with pytest.raises(AssertionError):
        _assert_runtime_selects_non_first_approval()


_CANONICAL_GUARDED_HANDLERS = {
    "_act_authorise_supplier_work",
    "_act_request_invoice_validation",
    "_act_request_payment",
    "_act_request_exception_resolution",
}


def _assert_action_guard_census(source: str) -> None:
    tree = ast.parse(source)
    operation = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MaintenanceOperation"
    )
    handlers = {
        node.name: node
        for node in operation.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_act_")
    }
    assert handlers.keys() >= _CANONICAL_GUARDED_HANDLERS
    for name, handler in handlers.items():
        guard_calls = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_required_evidence_verdict"
        ]
        if name in _CANONICAL_GUARDED_HANDLERS:
            assert len(guard_calls) == 1, name
            continue
        assert not guard_calls, name
        loaded_refs = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Name)
            and node.id == "evidence_refs"
            and isinstance(node.ctx, ast.Load)
        ]
        assert not loaded_refs, (
            f"{name} contains handler-local evidence_refs logic outside the canonical "
            "guarded action set"
        )


def test_runtime_has_exactly_four_canonical_evidence_guard_call_sites() -> None:
    source = Path(operation_module.__file__).read_text(encoding="utf-8")
    _assert_action_guard_census(source)
    assert "next(iter(" not in source
    assert "first eligible" not in source.lower()


def test_action_guard_census_rejects_a_shadow_guard_source_mutation() -> None:
    source = Path(operation_module.__file__).read_text(encoding="utf-8")
    marker = '        recipient = str(payload["recipient_actor_id"])\n'
    injected = (
        "        if evidence_refs and 'shadow_guard' not in evidence_refs:\n"
        "            return Verdict.refused('SHADOW_GUARD', 'missing shadow citation')\n"
    )
    assert marker in source
    mutant = source.replace(marker, injected + marker, 1)
    with pytest.raises(AssertionError, match="handler-local evidence_refs"):
        _assert_action_guard_census(mutant)


def test_all_consumers_call_the_live_canonical_accessor() -> None:
    operation_source = Path(operation_module.__file__).read_text(encoding="utf-8")
    evaluator_source = Path(evaluator_module.__file__).read_text(encoding="utf-8")
    assert "maintenance_spec.maintenance_action_evidence_contract()" in operation_source
    assert "maintenance_spec.maintenance_action_evidence_contract()" in evaluator_source


class _HostileMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        raise TypeError("hostile lookup")

    def __iter__(self) -> Iterator[object]:
        yield []

    def __len__(self) -> int:
        return 1


def test_selector_subclasses_are_rejected_without_invoking_overrides() -> None:
    called = False

    class Divergent(PayloadRegistryKey):
        def projection(self) -> tuple[str, dict[str, str]]:
            nonlocal called
            called = True
            raise AssertionError("untrusted selector override was invoked")

    contract = ActionEvidenceContract(
        {
            "request_payment": {
                "evidence_refs": (Divergent("list_billing", "invoices", "invoice_id"),)
            }
        }
    )
    assert "unknown evidence selector" in str(_problem(contract))
    assert (
        resolve_required_evidence(
            contract,
            "request_payment",
            {"invoice_id": "invoice_1"},
            {"invoices": {"invoice_1": {}}},
        ).problem
        == "the evidence selector declaration is unknown"
    )
    with pytest.raises(ValueError, match="unknown evidence selector"):
        contract.evidence_schema_part("request_payment")
    assert called is False


def test_hostile_and_recursive_contract_inputs_fail_with_stable_problem() -> None:
    recursive: list[object] = []
    recursive.append(recursive)
    for raw in (
        _HostileMapping(),
        {"request_payment": {"evidence_refs": recursive}},
    ):
        contract = ActionEvidenceContract(raw)  # type: ignore[arg-type]
        assert _problem(contract) == "action evidence contract construction failed"


def test_declarations_reject_mutable_text_surrogates_without_live_views() -> None:
    class MutableText:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return self.value

        def __eq__(self, other: object) -> bool:
            return self.value == other

        def __hash__(self) -> int:
            return hash(self.value)

    tool = MutableText("list_billing")
    strength = MutableText("required")
    field = MutableText("invoice_id")
    contract = ActionEvidenceContract(
        {
            "request_payment": {
                "reads": {tool: strength},
                "evidence_refs": (
                    PayloadRegistryKey(
                        "list_billing",
                        "invoices",
                        field,  # type: ignore[arg-type]
                    ),
                ),
            }
        }
    )
    before = (
        contract.schema_part("request_payment"),
        contract.tools_for("request_payment"),
        contract.evidence_for("request_payment"),
    )
    tool.value = "list_quotes"
    strength.value = "optional"
    field.value = "approval_checkpoint_id"
    assert (
        contract.schema_part("request_payment"),
        contract.tools_for("request_payment"),
        contract.evidence_for("request_payment"),
    ) == before
    assert isinstance(_problem(contract), str)


@pytest.mark.parametrize("field", ["source", "authority", "schema_id"])
def test_model_projection_rejects_forged_served_provenance(field: str) -> None:
    schemas = action_schema_view()
    catalogue = maintenance_retrieval_catalogue()
    entry = catalogue["get_case_record"]
    envelope = {
        "tool": "get_case_record",
        "source": entry["source"],
        "authority": entry["authority"],
        "record_id": "record",
        "record_version": "version",
        "as_of": "2025-01-01T00:00:00Z",
        "schema_id": entry["schema_id"],
        "records": {},
        "ok": True,
        "error_code": None,
    }
    envelope[field] = "forged"
    observation = AgentObservation(
        now="2025-01-01T00:00:00Z",
        operation_id="op",
        operation_instance_id="opinst_x",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=(),
        action_schemas=schemas,
        retrieval={"catalogue": catalogue, "served": {"get_case_record": envelope}},
    )
    assert model_projection(observation)["action_schemas"] == {
        action: {"reads": schema["reads"]} for action, schema in schemas.items()
    }


def test_valid_but_unobserved_exception_selector_is_unresolved() -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V2", "reference")
    trajectory = [deepcopy(row) for row in run.outcome.trajectory]
    proposal = next(
        row
        for row in trajectory
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_exception_resolution"
    )
    proposal["payload"]["approval_checkpoint_id"] = "absent_checkpoint"
    codes = {
        finding.code for finding in _discipline(_reevaluate(run, trajectory)).findings
    }
    assert REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED in codes


def test_non_mapping_trajectory_row_is_named_without_changing_result_vector() -> None:
    spec = load_spec(SPEC)
    run = run_episode(spec, "V1", "reference")
    trajectory: list[Any] = [*run.outcome.trajectory]
    trajectory.insert(1, None)
    result = _reevaluate(run, trajectory)
    assert tuple(d.name for d in result.dimensions) == evaluator_module.DIMENSIONS
    assert "MALFORMED_TRAJECTORY_ROW" in {
        finding.code for finding in _discipline(result).findings
    }
