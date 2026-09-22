"""Canonical Maintenance V1 SemanticScenario and its four matched arm projections.

This is deliberately a Maintenance compiler, not a workflow language.  It freezes
five reviewed decision boundaries from the executable V1 state machine and
projects one object into three point controls plus the unchanged Lifecycle Engine.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any, cast

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.core.engine import (
    STATUS_DEADLOCK,
    STATUS_HORIZON,
    Engine,
    EpisodeOutcome,
    has_canonical_episode_outcome_provenance,
)
from operatebench.core.errors import OperateBenchError
from operatebench.core.outcomes import Act, AgentOutcome, Wait
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.read_contract import resolve_required_evidence
from operatebench.core.retrieval import (
    RetrievalRequest,
    RetrieveBatch,
    ToolResult,
    canonical_requests,
    retrieval_batch_problem,
)
from operatebench.decision_points import (
    QuantifiedClause,
    RecordCondition,
    ScalarClause,
    StatePredicate,
    observation_facts,
    reach_decision_point,
)
from operatebench.domains.lettings.maintenance.agents import ReferenceAgent, build_agent
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.retrieval import (
    maintenance_retrieval_catalogue,
    record_version,
    serve_maintenance_retrieval,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    OperationSpec,
    action_schema_view,
    maintenance_action_evidence_contract,
    validate_action_payload,
)
from operatebench.runner import run_episode

ARM_STATIC = "static"
ARM_BOUNDARY = "boundary_only"
ARM_ORDERED = "ordered_snapshot_control"
ARM_FULL = "full_lifecycle"
ARM_NAMES = (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED, ARM_FULL)

ADMISSIBLE = "ADMISSIBLE"
UNREACHED = "UNREACHED"
NOT_ESTABLISHED = "NOT_ESTABLISHED"
ERROR = "ERROR"

COMMON_ESTIMAND = "fixed_five_point_ordered_boundary_resolution_status"
ORDERED_TIME_TRANSFORM_ID = "maintenance_v1_ordered_anchor_time_v1"
ORDERED_TIME_ANCHOR_RULE = "first_canonical_decision_point_reached_now"
ACCOUNTING_CONTROL_CLAIM = (
    "accounting/order-presentation control only; it removes elapsed time and "
    "timer/wake ownership while retaining causal business order and admissibility. "
    "WAIT remains in the frozen protocol vocabulary, but is not a lifecycle-owned "
    "control obligation and a local WAIT is graded NOT_ESTABLISHED. It does not test "
    "lifecycle memory or temporal memory."
)

_ALLOWED_CLAIMS: Mapping[str, str] = {
    ARM_STATIC: "fresh local decision-point admissibility with no future-state access",
    ARM_BOUNDARY: (
        "local boundary resolution and its required terminal obligation, without "
        "authored progression outside the boundary"
    ),
    ARM_ORDERED: ACCOUNTING_CONTROL_CLAIM,
    ARM_FULL: (
        "ownership of one operation through waits, wakes, external events, state "
        "changes, reopen and late-event handling in the canonical Engine"
    ),
}


def _control_transform_declarations() -> dict[str, dict[str, Any]]:
    return {
        ARM_ORDERED: {
            "transform_id": ORDERED_TIME_TRANSFORM_ID,
            "anchor_rule": ORDERED_TIME_ANCHOR_RULE,
            "now_rule": "anchor_for_every_point_and_turn",
            "tool_result_as_of_rule": "anchor_for_every_served_result",
            "wake_event_types": [],
            "removed_time_fields": [
                "at",
                "elapsed_minutes",
                "timers",
                "*_at",
            ],
        }
    }


class CompilationError(OperateBenchError):
    """The canonical scenario and executable Maintenance V1 no longer agree."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _action_payloads_equivalent(
    action_type: str, actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare payloads after filling only schema-declared nullable omissions."""
    schema = MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.get(action_type)
    if schema is None:
        return False
    _required, optional = schema

    def canonical(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = _plain(payload)
        for field_name, kind in optional.items():
            if kind == "optional_string" and field_name not in result:
                result[field_name] = None
        return cast(dict[str, Any], result)

    return canonical(actual) == canonical(expected)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _digest(payload: Mapping[str, Any], context: str) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload), context)).hexdigest()


def _q(
    collection: str,
    field_name: str,
    value: Any,
    *,
    bind: str | None = None,
    quantifier: str = "exactly_one",
) -> QuantifiedClause:
    return QuantifiedClause(
        collection=collection,
        quantifier=quantifier,
        bind=bind,
        where=(RecordCondition(field=field_name, op="eq", value=value),),
    )


@dataclass(frozen=True)
class SemanticDecisionPoint:
    """One authored V1 boundary, ordered inside one correlated scenario."""

    point_id: str
    order: int
    label: str
    reach_event_type: str
    required_prior_marker: str | None
    predicate: StatePredicate
    expected_obligation: str
    prior_mechanisms: tuple[str, ...]
    terminal_obligation: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "order": self.order,
            "label": self.label,
            "reach_event_type": self.reach_event_type,
            "required_prior_marker": self.required_prior_marker,
            "predicate": self.predicate.as_dict(),
            "expected_obligation": self.expected_obligation,
            "prior_mechanisms": list(self.prior_mechanisms),
            "terminal_obligation": self.terminal_obligation,
        }


@dataclass(frozen=True)
class SemanticScenario:
    """The immutable common denominator from which every arm is compiled."""

    scenario_id: str
    operation_id: str
    operation_spec_digest_sha256: str
    points: tuple[SemanticDecisionPoint, ...]
    estimand: str
    digest_sha256: str

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "operation_id": self.operation_id,
            "operation_spec_digest_sha256": self.operation_spec_digest_sha256,
            "estimand": self.estimand,
            "points": [point.semantic_payload() for point in self.points],
            "control_transforms": _control_transform_declarations(),
        }

    def verify_identity(self) -> None:
        if (
            _digest(self.semantic_payload(), "maintenance V1 semantic scenario")
            != self.digest_sha256
        ):
            raise CompilationError("semantic scenario digest does not match its content")


def _point_definitions() -> tuple[SemanticDecisionPoint, ...]:
    """Reviewed state predicates from V1 semantics, not from any model trajectory."""
    return (
        SemanticDecisionPoint(
            "initial_report",
            0,
            "The reported diagnostic cycle exists before any visit is requested.",
            "customer_issue_reported",
            None,
            StatePredicate(
                (
                    _q("cycles", "visit_status", "NONE", bind="cycle"),
                    ScalarClause(
                        path="state.current_cycle_id",
                        op="eq",
                        binding="cycle",
                        binding_field="cycle_id",
                    ),
                    ScalarClause(path="state.reopen_count", op="eq", value=0),
                )
            ),
            "ACT:request_supplier_visit",
            ("customer_issue_reported",),
            "A diagnostic supplier visit must be requested for the current cycle.",
        ),
        SemanticDecisionPoint(
            "approval_granted",
            1,
            "The exact quote is approved and its approved-work visit is not booked.",
            "approver_decision_received",
            None,
            StatePredicate(
                (
                    _q("approvals", "status", "RESOLVED_APPROVED", bind="approval"),
                    _q("cycles", "kind", "APPROVED_WORK", bind="cycle"),
                    ScalarClause(
                        path="state.current_cycle_id",
                        op="eq",
                        binding="cycle",
                        binding_field="cycle_id",
                    ),
                )
            ),
            "ACT:authorise_supplier_work",
            (
                "wait_declared",
                "approver_decision_received",
                "derived_context_invalidated",
            ),
            "Only work for the exact approved quote/checkpoint/cycle may be authorised.",
        ),
        SemanticDecisionPoint(
            "work_verified",
            2,
            "Authoritative work evidence exists and the matching invoice is unvalidated.",
            "work_evidence_verified",
            None,
            StatePredicate(
                (
                    _q("cycles", "visit_status", "WORK_VERIFIED", bind="cycle"),
                    _q("invoices", "status", "RECEIVED_UNVALIDATED", bind="invoice"),
                    ScalarClause(
                        path="state.current_cycle_id",
                        op="eq",
                        binding="cycle",
                        binding_field="cycle_id",
                    ),
                )
            ),
            "ACT:request_invoice_validation",
            ("wait_declared", "work_evidence_verified", "derived_context_invalidated"),
            (
                "The matching invoice must be sent for validation using verified-work "
                "evidence."
            ),
        ),
        SemanticDecisionPoint(
            "invoice_validated",
            3,
            "The matching invoice is validated and settlement has not been requested.",
            "invoice_validation_completed",
            None,
            StatePredicate(
                (
                    _q("invoices", "status", "VALIDATED", bind="invoice"),
                    ScalarClause(
                        path="state.payment.status", op="eq", value="NOT_REQUESTED"
                    ),
                )
            ),
            "ACT:request_payment",
            (
                "wait_declared",
                "invoice_validation_completed",
                "derived_context_invalidated",
            ),
            (
                "Payment must bind the exact validated invoice, amount, currency and "
                "work evidence."
            ),
        ),
        SemanticDecisionPoint(
            "warranty_reopened",
            4,
            "A persistence report reopened the operation onto its warranty cycle.",
            "customer_resolution_reported",
            "operation_reopened",
            StatePredicate(
                (
                    ScalarClause(path="state.reopen_count", op="eq", value=1),
                    _q("cycles", "kind", "WARRANTY_REVISIT", bind="cycle"),
                    ScalarClause(
                        path="state.current_cycle_id",
                        op="eq",
                        binding="cycle",
                        binding_field="cycle_id",
                    ),
                    _q("cycles", "visit_status", "NONE", bind="unbooked"),
                )
            ),
            "ACT:request_supplier_visit",
            (
                "payment_settlement_confirmed",
                "provisional_close_entered",
                "operation_reopened",
            ),
            (
                "A warranty revisit, not another payment, must be requested for the "
                "reopened cycle."
            ),
        ),
    )


def maintenance_v1_semantic_scenario(spec: OperationSpec) -> SemanticScenario:
    """Build the sole canonical V1 semantic object from the executable spec identity."""
    points = _point_definitions()
    payload = {
        "scenario_id": spec.semantic_scenario_id,
        "operation_id": spec.operation_id,
        "operation_spec_digest_sha256": spec.spec_digest_sha256,
        "estimand": COMMON_ESTIMAND,
        "points": [point.semantic_payload() for point in points],
        "control_transforms": _control_transform_declarations(),
    }
    return SemanticScenario(
        scenario_id=spec.semantic_scenario_id,
        operation_id=spec.operation_id,
        operation_spec_digest_sha256=spec.spec_digest_sha256,
        points=points,
        estimand=COMMON_ESTIMAND,
        digest_sha256=_digest(payload, "maintenance V1 semantic scenario"),
    )


@dataclass
class FrozenRetrievalSession:
    """Subject-owned reads over one frozen boundary using Engine-equivalent serving."""

    observable: Mapping[str, Any]
    as_of: str
    remaining_batches: int
    observation_payload: Mapping[str, Any]
    remove_time: bool = False
    turn_index: int = 0
    served: dict[str, ToolResult] = field(default_factory=dict)
    last_rejection: Mapping[str, Any] | None = None

    def current_observation(self) -> AgentObservation:
        """Return this turn as a detached, transport-safe Core observation."""
        payload = _plain(self.observation_payload)
        payload["turn_index"] = self.turn_index
        payload["last_rejection"] = _plain(self.last_rejection)
        payload["retrieval"]["served"] = {
            tool: _plain(result.as_dict()) for tool, result in sorted(self.served.items())
        }
        payload["retrieval"]["batch_budget_remaining"] = self.remaining_batches
        return AgentObservation(**payload)

    @property
    def model_projection(self) -> Mapping[str, Any]:
        projected = model_projection(self.current_observation())
        return cast(Mapping[str, Any], _freeze(projected))

    def submit(self, decision: Any) -> FrozenTurn:
        """Advance one subject call with Core's budget/refusal/projection semantics."""
        results: tuple[ToolResult, ...] = ()
        refusal: str | None = None
        if isinstance(decision, RetrieveBatch):
            problem = retrieval_batch_problem(
                decision,
                catalogue=maintenance_retrieval_catalogue(),
                batch_budget_remaining=self.remaining_batches,
            )
            if problem is not None:
                refusal, detail = problem
                self.last_rejection = {
                    "action_type": "retrieve",
                    "code": refusal,
                    "detail": detail,
                }
            else:
                self.remaining_batches -= 1
                raw = serve_maintenance_retrieval(
                    self.observable, canonical_requests(decision), self.as_of
                )
                results = tuple(
                    _ordered_tool_result(result, self.as_of)
                    if self.remove_time
                    else result
                    for result in raw
                )
                for result in results:
                    if result.ok:
                        self.served[result.tool] = result
        self.turn_index += 1
        return FrozenTurn(self.model_projection, results, refusal)

    def serve(self, tools: Sequence[str]) -> tuple[Any, ...]:
        turn = self.submit(
            RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in tools))
        )
        if turn.refusal_code is not None:
            raise CompilationError(f"subject retrieval refused: {turn.refusal_code}")
        return turn.results

    def establishes(self, point: CompiledPoint, outcome: AgentOutcome) -> bool:
        if point.establish(outcome) != ADMISSIBLE or not isinstance(outcome, Act):
            return False
        try:
            validate_action_payload(
                outcome.action_type, outcome.payload, "static proposal"
            )
        except OperateBenchError:
            return False
        required = point.action_schemas[outcome.action_type].get("reads", {})
        if any(
            tool not in self.served
            for tool, strength in required.items()
            if strength == "required"
        ):
            return False
        records: dict[str, Any] = {}
        for result in self.served.values():
            if (
                not result.ok
                or result.error_code is not None
                or result.as_of != self.as_of
                or not all(
                    isinstance(value, str) and value
                    for value in (
                        result.source,
                        result.authority,
                        result.record_id,
                        result.record_version,
                        result.schema_id,
                    )
                )
            ):
                continue
            for root, value in result.records.items():
                if root in records and _plain(records[root]) != _plain(value):
                    return False
                records[str(root)] = value
        resolution = resolve_required_evidence(
            maintenance_action_evidence_contract(),
            outcome.action_type,
            outcome.payload,
            records,
        )
        return resolution.problem is None and set(resolution.required).issubset(
            outcome.evidence_refs
        )


@dataclass(frozen=True)
class FrozenTurn:
    model_projection: Mapping[str, Any]
    results: tuple[ToolResult, ...]
    refusal_code: str | None


def _ordered_tool_result(result: ToolResult, anchor: str) -> ToolResult:
    """Apply the declared ordered transform before versioning the served record."""
    records = _remove_time_fields(result.records)
    return ToolResult(
        tool=result.tool,
        source=result.source,
        authority=result.authority,
        record_id=result.record_id,
        records=records,
        record_version=record_version(records),
        as_of=anchor,
        schema_id=result.schema_id,
        ok=result.ok,
        error_code=result.error_code,
    )


@dataclass(frozen=True)
class CompiledPoint:
    point_id: str
    order: int
    reach_event_type: str
    required_prior_marker: str | None
    model_input: Mapping[str, Any]
    backend_facts: Mapping[str, Any]
    action_schemas: Mapping[str, Any]
    retrieval_catalogue: Mapping[str, Any]
    subject_retrieval_budget: int
    driver_read_count: int
    expected_obligation: str
    expected_payload: Mapping[str, Any]
    expected_evidence_refs: tuple[str, ...]
    boundary_resolution: tuple[str, ...]
    _as_of: str = field(repr=False)
    _remove_time: bool = field(default=False, repr=False)
    _observation_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def new_retrieval_session(self) -> FrozenRetrievalSession:
        return FrozenRetrievalSession(
            observable=_plain(self.backend_facts),
            as_of=self._as_of,
            remaining_batches=self.subject_retrieval_budget,
            observation_payload=_plain(self._observation_payload),
            remove_time=self._remove_time,
        )

    def establish(self, outcome: AgentOutcome) -> str:
        if not isinstance(outcome, Act):
            return NOT_ESTABLISHED
        obligation = f"ACT:{outcome.action_type}"
        if obligation != self.expected_obligation:
            return NOT_ESTABLISHED
        if not isinstance(outcome.payload, Mapping):
            return NOT_ESTABLISHED
        try:
            validate_action_payload(
                outcome.action_type, outcome.payload, "compiled point proposal"
            )
        except OperateBenchError:
            return NOT_ESTABLISHED
        if not _action_payloads_equivalent(
            outcome.action_type, outcome.payload, self.expected_payload
        ):
            return NOT_ESTABLISHED
        if tuple(outcome.evidence_refs) != self.expected_evidence_refs:
            return NOT_ESTABLISHED
        return ADMISSIBLE


@dataclass(frozen=True)
class CompiledArm:
    treatment_label: str
    allowed_claim: str
    semantic_scenario_id: str
    semantic_scenario_digest_sha256: str
    source_scenario: SemanticScenario
    points: tuple[CompiledPoint, ...]
    common_expected_vector: tuple[str, ...]
    denominator: int
    point_ids: tuple[str, ...]
    subject_continuity: str
    harness_metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    engine_class: str | None = None
    full_outcome: EpisodeOutcome | None = None


def validate_point_correlations(point_id: str, facts: Mapping[str, Any]) -> None:
    """Fail closed when a target status belongs to another cycle or payment chain."""
    current_id = facts.get("current_cycle_id")
    cycles = facts.get("cycles") or {}
    current = cycles.get(current_id) if isinstance(cycles, Mapping) else None
    if not isinstance(current, Mapping):
        raise CompilationError(f"{point_id} has no correlated current cycle")
    approvals = facts.get("approvals") or {}
    invoices = facts.get("invoices") or {}
    if point_id == "approval_granted":
        matched = [
            row
            for row in approvals.values()
            if row.get("status") == "RESOLVED_APPROVED"
            and row.get("cycle_id") == current_id
        ]
        if len(matched) != 1:
            raise CompilationError(
                "approval_granted requires exactly one approved record correlated "
                "to the current cycle"
            )
    if point_id in {"work_verified", "invoice_validated"}:
        wanted = "RECEIVED_UNVALIDATED" if point_id == "work_verified" else "VALIDATED"
        matched = [
            row
            for row in invoices.values()
            if row.get("status") == wanted and row.get("cycle_id") == current_id
        ]
        if len(matched) != 1:
            raise CompilationError(
                f"{point_id} requires exactly one {wanted} invoice correlated to "
                "the current cycle"
            )
        invoice = matched[0]
        linked = [
            row
            for row in approvals.values()
            if row.get("cycle_id") == current_id
            and row.get("quote_id") == invoice.get("quote_id")
            and row.get("quote_version") == invoice.get("quote_version")
            and row.get("status") == "RESOLVED_APPROVED"
        ]
        if len(linked) != 1:
            raise CompilationError(
                f"{point_id} invoice is not correlated to its approval"
            )
    if point_id == "warranty_reopened":
        payment = facts.get("payment") or {}
        parent = current.get("parent_cycle_id")
        matched = [
            row
            for row in invoices.values()
            if row.get("invoice_id") == payment.get("invoice_id")
            and row.get("cycle_id") == parent
            and row.get("status") == "VALIDATED"
        ]
        if (
            payment.get("status") != "SETTLED"
            or payment.get("cycle_id") != parent
            or len(matched) != 1
            or current.get("visit_status") != "NONE"
        ):
            raise CompilationError(
                "warranty_reopened current unbooked cycle is not correlated to the "
                "settled payment"
            )


def _expected_action(
    point_id: str, facts: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    cycles = facts["cycles"]
    current = cycles[facts["current_cycle_id"]]
    if point_id in {"initial_report", "warranty_reopened"}:
        return (
            {
                "visit_type": current["kind"],
                "cycle_id": current["cycle_id"],
                "scope_digest": current["scope_digest"],
            },
            (),
        )
    if point_id == "approval_granted":
        approval = next(
            item
            for item in facts["approvals"].values()
            if item["status"] == "RESOLVED_APPROVED"
            and item["cycle_id"] == facts["current_cycle_id"]
        )
        return (
            {
                "quote_id": approval["quote_id"],
                "quote_version": approval["quote_version"],
                "approval_checkpoint_id": approval["checkpoint_id"],
                "cycle_id": approval["cycle_id"],
            },
            (str(approval["checkpoint_id"]),),
        )
    invoice = next(
        item
        for item in facts["invoices"].values()
        if item["cycle_id"] == current["cycle_id"]
        and item["status"]
        == ("RECEIVED_UNVALIDATED" if point_id == "work_verified" else "VALIDATED")
    )
    if point_id == "work_verified":
        return (
            {"invoice_id": invoice["invoice_id"]},
            (str(current["work_evidence_id"]),),
        )
    return (
        {
            "invoice_id": invoice["invoice_id"],
            "amount_minor": invoice["amount_minor"],
            "currency": invoice["currency"],
        },
        (str(invoice["invoice_id"]), str(current["work_evidence_id"])),
    )


def _initial_projection(
    payload: Mapping[str, Any], *, retrieval_budget: int
) -> dict[str, Any]:
    projection = cast(dict[str, Any], _plain(payload))
    projection["operation_instance_id"] = "<opaque-run-id>"
    projection["turn_index"] = 0
    retrieval = projection["retrieval"]
    retrieval["served"] = {}
    retrieval["batch_budget_remaining"] = retrieval_budget
    # Re-run Core's allowlisted projection *after* resetting retrieval.  Projecting
    # the driver's already-served turn and then emptying ``served`` would leave the
    # action schemas unlocked by private reads, which is itself oracle leakage.
    projection = model_projection(AgentObservation(**projection))
    return projection


def _remove_time_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        removed = {}
        for key, item in value.items():
            name = str(key)
            if name in {
                "at",
                "now",
                "wake_event_types",
                "elapsed_minutes",
                "timers",
            }:
                continue
            if name.endswith("_at"):
                continue
            removed[name] = _remove_time_fields(item)
        return removed
    if isinstance(value, (list, tuple)):
        return [_remove_time_fields(item) for item in value]
    return copy.deepcopy(value)


def _ordered_observation_payload(
    payload: Mapping[str, Any], anchor: str
) -> dict[str, Any]:
    """Neutralise lifecycle clock ownership while retaining protocol validity."""
    transformed = cast(dict[str, Any], _remove_time_fields(payload))
    transformed["now"] = anchor
    transformed["wake_event_types"] = []
    return transformed


def _compile_points(
    spec: OperationSpec, scenario: SemanticScenario, treatment: str
) -> tuple[CompiledPoint, ...]:
    compiled: list[CompiledPoint] = []
    ordered = treatment == ARM_ORDERED
    encounters = tuple(
        (
            point,
            reach_decision_point(
                spec,
                # The established probe consumes only predicate and ids; this
                # narrow adapter avoids a second engine implementation.
                _as_decision_point(point),
            ),
        )
        for point in scenario.points
    )
    anchor = encounters[0][1].now
    for point, encounter in encounters:
        raw_payload = encounter.observation_payload()
        payload = (
            _ordered_observation_payload(raw_payload, anchor) if ordered else raw_payload
        )
        facts = observation_facts(raw_payload)
        validate_point_correlations(point.point_id, facts)
        expected_payload, evidence_refs = _expected_action(point.point_id, facts)
        compiled.append(
            CompiledPoint(
                point_id=point.point_id,
                order=point.order,
                reach_event_type=point.reach_event_type,
                required_prior_marker=point.required_prior_marker,
                model_input=_freeze(
                    _initial_projection(
                        payload,
                        retrieval_budget=spec.policy.max_retrieval_batches_per_invocation,
                    )
                ),
                backend_facts=_freeze(facts),
                action_schemas=_freeze(action_schema_view()),
                retrieval_catalogue=_freeze(maintenance_retrieval_catalogue()),
                subject_retrieval_budget=(
                    spec.policy.max_retrieval_batches_per_invocation
                ),
                driver_read_count=len(payload["retrieval"]["served"]),
                expected_obligation=point.expected_obligation,
                expected_payload=_freeze(expected_payload),
                expected_evidence_refs=evidence_refs,
                boundary_resolution=(
                    point.expected_obligation,
                    point.terminal_obligation,
                ),
                _as_of=anchor if ordered else encounter.now,
                _remove_time=ordered,
                _observation_payload=_freeze(payload),
            )
        )
    return tuple(compiled)


def _as_decision_point(point: SemanticDecisionPoint) -> Any:
    """Supply the established probe with only the fields it reads."""
    from operatebench.decision_points import DecisionPoint

    return DecisionPoint(
        decision_point_id=point.point_id,
        scenario_id="V1",
        label=point.label,
        state_predicate=point.predicate,
        observable_facts=(),
        evidence_provenance=(),
        admissible_outcomes=(),
    )


@dataclass(frozen=True)
class PointRun:
    vector_status: str
    trajectory: tuple[Mapping[str, Any], ...] = ()
    effect_applied: bool = False
    proposal_id: str | None = None
    accepted_effect: Mapping[str, Any] | None = None


def run_static(point: CompiledPoint, decisions: Sequence[Any]) -> PointRun:
    """Grade a local proposal and its subject-owned reads without applying it."""
    session = point.new_retrieval_session()
    for decision in decisions:
        if isinstance(decision, RetrieveBatch):
            session.submit(decision)
            continue
        session.submit(decision)
        established = session.establishes(point, decision)
        return PointRun(ADMISSIBLE if established else NOT_ESTABLISHED)
    return PointRun(NOT_ESTABLISHED)


class _BoundaryAgent:
    """Let the reference drive outside one timestamp, and the subject own it inside."""

    agent_id = "semantic_boundary_subject"

    def __init__(self, at: str, decisions: Sequence[Any]) -> None:
        self.at = at
        self.decisions = list(decisions)
        self.reference = build_agent("reference")
        self.started = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.reference.begin_episode(identity)

    def decide(self, observation: AgentObservation) -> Any:
        if observation.now == self.at and (self.decisions or self.started):
            self.started = True
            if self.decisions:
                return self.decisions.pop(0)
            return Wait(
                reason="the subject-owned boundary has no decision remaining",
                fallback_after_minutes=1,
            )
        return self.reference.decide(observation)


def _exact_proposal(point: CompiledPoint, row: Mapping[str, Any]) -> bool:
    action_type = row.get("action_type")
    payload = row.get("payload")
    return (
        row.get("record_type") == "action_proposed"
        and isinstance(action_type, str)
        and action_type == point.expected_obligation.removeprefix("ACT:")
        and isinstance(payload, Mapping)
        and _action_payloads_equivalent(action_type, payload, point.expected_payload)
        and tuple(row.get("evidence_refs") or ()) == point.expected_evidence_refs
    )


def _effect_binding_matches(
    effect: Mapping[str, Any], proposal: Mapping[str, Any], point: CompiledPoint
) -> bool:
    """Validate the identities the production Maintenance handler commits."""
    action_type = proposal.get("action_type")
    payload = proposal.get("payload")
    bindings = effect.get("bindings")
    if not isinstance(action_type, str) or not isinstance(payload, Mapping):
        return False
    if not isinstance(bindings, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in bindings.items()
    ):
        return False
    expected_cycle = payload.get("cycle_id")
    if action_type in {"request_invoice_validation", "request_payment"}:
        invoice_id = payload.get("invoice_id")
        invoices = point.backend_facts.get("invoices")
        invoice = invoices.get(invoice_id) if isinstance(invoices, Mapping) else None
        if not isinstance(invoice, Mapping):
            return False
        expected_cycle = invoice.get("cycle_id")
    if not isinstance(expected_cycle, str) or effect.get("cycle_id") != expected_cycle:
        return False
    if action_type in {"request_supplier_visit", "authorise_supplier_work"}:
        return (
            set(bindings) == {"visit_id", "related_visit_id"}
            and bindings["visit_id"] == bindings["related_visit_id"]
        )
    if action_type == "request_invoice_validation":
        return bindings == {"invoice_id": payload.get("invoice_id")}
    if action_type == "request_payment":
        return set(bindings) == {"payment_request_id"}
    return False


def _effect_binding_is_well_formed(effect: Mapping[str, Any], action_type: str) -> bool:
    bindings = effect.get("bindings")
    if not isinstance(bindings, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in bindings.items()
    ):
        return False
    expected_keys = {
        "request_supplier_visit": {"visit_id", "related_visit_id"},
        "authorise_supplier_work": {"visit_id", "related_visit_id"},
        "request_invoice_validation": {"invoice_id"},
        "request_payment": {"payment_request_id"},
    }.get(action_type)
    return expected_keys is not None and set(bindings) == expected_keys


def _effect_is_bound(
    effect: Mapping[str, Any], proposal: Mapping[str, Any], point: CompiledPoint
) -> bool:
    proposal_index = proposal.get("index")
    effect_index = effect.get("index")
    return (
        effect.get("record_type") == "effect_accepted"
        and effect.get("proposal_id") == proposal.get("proposal_id")
        and effect.get("action_type") == proposal.get("action_type")
        and effect.get("code") == "ACCEPTED"
        and effect.get("at") == proposal.get("at")
        and type(proposal_index) is int
        and type(effect_index) is int
        and effect_index > proposal_index
        and _effect_binding_matches(effect, proposal, point)
    )


def run_boundary(
    spec: OperationSpec, point: CompiledPoint, decisions: Sequence[Any]
) -> PointRun:
    """Execute a subject's local turns through Engine's real read/action boundary."""
    try:
        scenario = spec.scenario("V1")
        agent = _BoundaryAgent(point._as_of, decisions)
        engine = Engine(
            MaintenanceOperation(spec, "V1"),
            agent,
            identity={
                "operation_id": spec.operation_id,
                "operation_instance_id": "semantic-boundary-offline",
                "scenario_id": "V1",
                "agent_id": agent.agent_id,
                "spec_digest_sha256": spec.spec_digest_sha256,
            },
            dispatch_failures=scenario.dispatch_failures,
        )
        outcome = engine.run()
    except Exception:  # Infrastructure/mapping failures are data, never model failure.
        return PointRun(ERROR)
    rows = tuple(outcome.trajectory)
    proposals = [
        row
        for row in rows
        if row.get("at") == point._as_of and _exact_proposal(point, row)
    ]
    for proposal in proposals:
        proposal_id = str(proposal.get("proposal_id"))
        effect = next(
            (
                row
                for row in rows
                if row.get("proposal_id") == proposal_id
                and _effect_is_bound(row, proposal, point)
            ),
            None,
        )
        if effect is not None:
            return PointRun(ADMISSIBLE, rows, True, proposal_id, effect)
    return PointRun(NOT_ESTABLISHED, rows)


def map_full_outcome(
    points: Sequence[CompiledPoint],
    outcome: EpisodeOutcome | Sequence[Mapping[str, Any]] | None,
) -> tuple[str, ...]:
    """Map actual point reach and proposal/effect bindings onto the fixed denominator.

    A terminal suffix is authoritative only on an authenticated outcome minted
    by this Engine process (including the outcome released by an exact successful
    replay).  Serialized rows and caller-built outcomes remain diagnostic input:
    they can establish reached points, but cannot license ``UNREACHED``.
    """
    if outcome is None:
        return tuple(ERROR for _ in points)
    rows = list(outcome.trajectory if isinstance(outcome, EpisodeOutcome) else outcome)
    if any(not isinstance(row, Mapping) for row in rows):
        return tuple(ERROR for _ in points)
    indices = [row.get("index") for row in rows]
    if any(type(index) is not int for index in indices) or any(
        cast(int, after) <= cast(int, before) for before, after in pairwise(indices)
    ):
        return tuple(ERROR for _ in points)

    boundary_positions: list[int | None] = []
    boundary_marker_errors: list[bool] = []
    cursor = 0
    for point in points:
        search_from = cursor
        marker_error = False
        if point.required_prior_marker is not None:
            prior = next(
                (
                    position
                    for position in range(cursor, len(rows))
                    if rows[position].get("record_type") == point.required_prior_marker
                ),
                None,
            )
            if prior is None:
                boundary_positions.append(None)
                boundary_marker_errors.append(False)
                continue
            marker_error = any(
                rows[position].get("record_type") == "event_observed"
                and rows[position].get("event_type") == point.reach_event_type
                for position in range(cursor, prior)
            )
            search_from = prior + 1
        position = next(
            (
                candidate
                for candidate in range(search_from, len(rows))
                if rows[candidate].get("record_type") == "event_observed"
                and rows[candidate].get("event_type") == point.reach_event_type
            ),
            None,
        )
        boundary_positions.append(position)
        boundary_marker_errors.append(marker_error)
        if position is not None:
            cursor = position + 1

    trusted_terminal = (
        isinstance(outcome, EpisodeOutcome)
        and outcome.status in {STATUS_DEADLOCK, STATUS_HORIZON}
        and has_canonical_episode_outcome_provenance(outcome)
    )
    statuses: list[str] = []
    remainder_status: str | None = None
    for point_index, point in enumerate(points):
        if remainder_status is not None:
            statuses.append(remainder_status)
            continue
        if boundary_marker_errors[point_index]:
            statuses.append(ERROR)
            remainder_status = ERROR
            continue
        start = boundary_positions[point_index]
        if start is None:
            is_true_suffix = all(
                position is None for position in boundary_positions[point_index + 1 :]
            )
            remainder_status = UNREACHED if trusted_terminal and is_true_suffix else ERROR
            statuses.append(remainder_status)
            continue
        end = next(
            (
                position
                for position in boundary_positions[point_index + 1 :]
                if position is not None
            ),
            len(rows),
        )
        interval = rows[start:end]
        marker = rows[start]
        marker_id = marker.get("event_id")
        invocations = [
            row
            for row in interval
            if row.get("record_type") == "agent_invoked"
            and row.get("trigger_event_id") == marker_id
            and row.get("trigger_event_type") == point.reach_event_type
        ]
        if not isinstance(marker_id, str) or not marker_id or len(invocations) != 1:
            statuses.append(ERROR)
            remainder_status = ERROR
            continue
        if any(
            row.get("record_type") in {"execution_error", "mapping_error"}
            for row in interval
        ):
            statuses.append(ERROR)
            remainder_status = ERROR
            continue
        exact = [row for row in interval if _exact_proposal(point, row)]
        exact_ids = {proposal.get("proposal_id") for proposal in exact}
        relevant_effects = [
            row
            for row in interval
            if row.get("record_type") == "effect_accepted"
            and row.get("proposal_id") in exact_ids
        ]
        if any(
            not _effect_binding_is_well_formed(
                effect, point.expected_obligation.removeprefix("ACT:")
            )
            for effect in relevant_effects
        ):
            statuses.append(ERROR)
            remainder_status = ERROR
            continue
        accepted = [
            proposal
            for proposal in exact
            if any(_effect_is_bound(effect, proposal, point) for effect in interval)
        ]
        if len(accepted) > 1:
            statuses.append(ERROR)
            remainder_status = ERROR
        else:
            statuses.append(ADMISSIBLE if accepted else NOT_ESTABLISHED)
    return tuple(statuses)


def compile_maintenance_v1(
    spec: OperationSpec,
    scenario: SemanticScenario,
    *,
    semantic_override: Mapping[str, Any] | None = None,
) -> Mapping[str, CompiledArm]:
    """Compile all four named arms, failing closed on any identity or parity drift."""
    if spec.compute_digest() != spec.spec_digest_sha256:
        raise CompilationError("operation spec identity is stale")
    scenario.verify_identity()
    if (
        semantic_override is not None
        and _plain(semantic_override) != scenario.semantic_payload()
    ):
        raise CompilationError(
            "semantic override changes the canonical digest or obligation"
        )
    if scenario.operation_spec_digest_sha256 != spec.spec_digest_sha256:
        raise CompilationError(
            "semantic scenario was authored against a different spec identity"
        )
    if tuple(point.order for point in scenario.points) != tuple(
        range(len(scenario.points))
    ):
        raise CompilationError("decision-point order must be contiguous and canonical")

    controls = {
        treatment: _compile_points(spec, scenario, treatment)
        for treatment in (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED, ARM_FULL)
    }
    full_run = run_episode(spec, "V1", "reference")
    if not full_run.reliable:
        raise CompilationError(
            "reference Full Lifecycle execution did not establish parity"
        )
    vector = map_full_outcome(controls[ARM_FULL], full_run.outcome)
    if vector != (ADMISSIBLE,) * len(scenario.points):
        raise CompilationError(
            f"reference Full mapping did not establish parity: {vector}"
        )
    point_ids = tuple(point.point_id for point in scenario.points)
    arms: dict[str, CompiledArm] = {}
    for treatment in (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED):
        harness_metadata = (
            _freeze(
                {
                    "control_transform_id": ORDERED_TIME_TRANSFORM_ID,
                    "anchor_rule": ORDERED_TIME_ANCHOR_RULE,
                    "anchor_now": controls[treatment][0]._as_of,
                }
            )
            if treatment == ARM_ORDERED
            else _freeze({})
        )
        arms[treatment] = CompiledArm(
            treatment_label=treatment,
            allowed_claim=_ALLOWED_CLAIMS[treatment],
            semantic_scenario_id=scenario.scenario_id,
            semantic_scenario_digest_sha256=scenario.digest_sha256,
            source_scenario=scenario,
            points=controls[treatment],
            common_expected_vector=vector,
            denominator=len(scenario.points),
            point_ids=point_ids,
            # Observed contract: model invocations are separate provider calls and
            # the deterministic reference explicitly keeps no memory between
            # invocations. Ordered-vs-static is therefore not interpretable as a
            # temporal-memory test.
            subject_continuity="stateless_across_operation_invocations",
            harness_metadata=harness_metadata,
        )
    arms[ARM_FULL] = CompiledArm(
        treatment_label=ARM_FULL,
        allowed_claim=_ALLOWED_CLAIMS[ARM_FULL],
        semantic_scenario_id=scenario.scenario_id,
        semantic_scenario_digest_sha256=scenario.digest_sha256,
        source_scenario=scenario,
        points=controls[ARM_FULL],
        common_expected_vector=vector,
        denominator=len(scenario.points),
        point_ids=point_ids,
        subject_continuity="stateless_across_operation_invocations",
        engine_class="operatebench.core.engine.Engine",
        full_outcome=full_run.outcome,
    )
    return MappingProxyType(arms)


def verify_reference_vector(
    arms: Mapping[str, CompiledArm], reference: ReferenceAgent
) -> tuple[str, ...]:
    """Independently execute the reference rules at every common boundary."""
    established: list[str] = []
    for point in arms[ARM_STATIC].points:
        outcome = reference.decide_from(point.backend_facts, point.model_input["policy"])
        reads = RetrieveBatch(
            tuple(RetrievalRequest(tool=tool) for tool in point.retrieval_catalogue)
        )
        result = run_static(point, (reads, outcome)).vector_status
        if result != ADMISSIBLE:
            raise CompilationError(
                f"reference did not establish {point.point_id}: {result}"
            )
        established.append(result)
    vector = tuple(established)
    if vector != arms[ARM_FULL].common_expected_vector:
        raise CompilationError("reference vector differs from Full common vector")
    return vector


__all__ = [
    "ACCOUNTING_CONTROL_CLAIM",
    "ADMISSIBLE",
    "ARM_BOUNDARY",
    "ARM_FULL",
    "ARM_NAMES",
    "ARM_ORDERED",
    "ARM_STATIC",
    "COMMON_ESTIMAND",
    "ERROR",
    "NOT_ESTABLISHED",
    "ORDERED_TIME_ANCHOR_RULE",
    "ORDERED_TIME_TRANSFORM_ID",
    "UNREACHED",
    "CompilationError",
    "CompiledArm",
    "CompiledPoint",
    "FrozenRetrievalSession",
    "FrozenTurn",
    "PointRun",
    "SemanticDecisionPoint",
    "SemanticScenario",
    "compile_maintenance_v1",
    "maintenance_v1_semantic_scenario",
    "map_full_outcome",
    "run_boundary",
    "run_static",
    "validate_point_correlations",
    "verify_reference_vector",
]
