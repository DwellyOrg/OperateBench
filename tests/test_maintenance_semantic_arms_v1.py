"""Executable anti-confound tests for the Maintenance V1 four-arm compiler."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import operatebench.artifact as artifact_module
import operatebench.core.engine as engine_module
from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.agents.model import ModelAgent
from operatebench.agents.openai_responses import build_model_payload
from operatebench.artifact import build_artifact, replay_artifact
from operatebench.core.engine import (
    STATUS_DEADLOCK,
    STATUS_HORIZON,
    Engine,
    EpisodeOutcome,
    has_canonical_episode_outcome_provenance,
)
from operatebench.core.outcomes import Act, Wait
from operatebench.core.protocol import MODEL_VISIBLE_FIELDS
from operatebench.core.read_contract import resolve_required_evidence
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import ReferenceAgent, build_agent
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.retrieval import record_version
from operatebench.domains.lettings.maintenance.semantic_arms_v1 import (
    ACCOUNTING_CONTROL_CLAIM,
    ARM_BOUNDARY,
    ARM_FULL,
    ARM_ORDERED,
    ARM_STATIC,
    ERROR,
    NOT_ESTABLISHED,
    ORDERED_TIME_ANCHOR_RULE,
    ORDERED_TIME_TRANSFORM_ID,
    UNREACHED,
    CompilationError,
    compile_maintenance_v1,
    maintenance_v1_semantic_scenario,
    map_full_outcome,
    run_boundary,
    run_static,
    validate_point_correlations,
    verify_reference_vector,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_EVIDENCE_CONTRACT,
    load_spec,
)
from operatebench.runner import _run_and_evaluate, run_episode

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")


class _NeverCalled:
    def send(self, _request):
        raise AssertionError("control request construction must not dispatch")


@pytest.fixture(scope="module")
def compiled():
    spec = load_spec(FIXTURE)
    scenario = maintenance_v1_semantic_scenario(spec)
    return spec, scenario, compile_maintenance_v1(spec, scenario)


def test_all_four_arms_derive_from_one_immutable_canonical_identity(compiled) -> None:
    _spec, scenario, arms = compiled
    assert scenario.points == tuple(scenario.points)
    assert tuple(arms) == (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED, ARM_FULL)
    assert {arm.semantic_scenario_digest_sha256 for arm in arms.values()} == {
        scenario.digest_sha256
    }
    assert {arm.semantic_scenario_id for arm in arms.values()} == {scenario.scenario_id}
    assert all(arm.source_scenario is scenario for arm in arms.values())
    assert [point.reach_event_type for point in scenario.points] == [
        "customer_issue_reported",
        "approver_decision_received",
        "work_evidence_verified",
        "invoice_validation_completed",
        "customer_resolution_reported",
    ]
    assert scenario.points[-1].required_prior_marker == "operation_reopened"


def test_order_and_denominator_are_common_and_never_shrink(compiled) -> None:
    _spec, scenario, arms = compiled
    expected = tuple(point.point_id for point in scenario.points)
    assert len(expected) == 5
    assert all(arm.point_ids == expected for arm in arms.values())
    assert all(arm.denominator == len(expected) for arm in arms.values())
    assert (
        arms[ARM_FULL].common_expected_vector == arms[ARM_STATIC].common_expected_vector
    )


def test_static_is_fresh_unserved_and_cannot_leak_later_or_final_facts(compiled) -> None:
    _spec, _scenario, arms = compiled
    first = arms[ARM_STATIC].points[0]
    projection = first.model_input
    assert projection["retrieval"]["served"] == {}
    assert projection["retrieval"]["batch_budget_remaining"] == 6
    assert projection["operation_instance_id"] == "<opaque-run-id>"
    assert set(projection) == set(MODEL_VISIBLE_FIELDS)
    assert set(projection["action_schemas"]["request_payment"]) == {"reads"}
    text = repr(projection)
    for forbidden in (
        "quote_1",
        "checkpoint_1",
        "invoice_1",
        "settlement_1",
        "FINALIZED",
    ):
        assert forbidden not in text
    assert first.backend_facts["reopen_count"] == 0
    assert first.backend_facts["terminal"] is None


def test_control_point_surface_and_retrieval_start_match(compiled) -> None:
    _spec, _scenario, arms = compiled
    controls = (arms[ARM_STATIC], arms[ARM_BOUNDARY], arms[ARM_ORDERED])
    for index in range(5):
        points = [arm.points[index] for arm in controls]
        assert len({repr(point.backend_facts) for point in points}) == 1
        assert len({repr(point.action_schemas) for point in points}) == 1
        assert len({repr(point.retrieval_catalogue) for point in points}) == 1
        assert {point.subject_retrieval_budget for point in points} == {6}
        assert all(point.model_input["retrieval"]["served"] == {} for point in points)


def test_private_driver_reads_never_consume_subject_budget(compiled) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[2]
    session = point.new_retrieval_session()
    assert session.remaining_batches == 6
    assert point.driver_read_count > 0
    result = session.serve(("get_case_record",))
    assert result[0].tool == "get_case_record"
    assert session.remaining_batches == 5
    fresh = point.new_retrieval_session()
    assert fresh.remaining_batches == 6


def test_every_control_turn_builds_the_frozen_lifecycle_request_without_dispatch(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    agent = ModelAgent(_NeverCalled(), model="test-model")
    for arm_name in (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED):
        for point in arms[arm_name].points:
            session = point.new_retrieval_session()
            for decision in (None, _all_reads(point)):
                if decision is not None:
                    session.submit(decision)
                observation = session.current_observation()
                request = agent.build_request(observation)
                payload = build_model_payload(request, model="test-model")
                assert json.loads(json.dumps(payload)) == payload
                assert set(request.prompt["observation"]) == set(MODEL_VISIBLE_FIELDS)


def test_ordered_snapshot_is_only_an_accounting_order_presentation_control(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    ordered = arms[ARM_ORDERED]
    assert ordered.allowed_claim == ACCOUNTING_CONTROL_CLAIM
    assert ordered.subject_continuity == "stateless_across_operation_invocations"
    assert [point.order for point in ordered.points] == list(range(5))
    anchor = arms[ARM_STATIC].points[0].model_input["now"]
    for point in ordered.points:
        observation = point.new_retrieval_session().current_observation()
        assert observation.now == anchor
        assert observation.wake_event_types == []
        assert point.model_input["now"] == anchor
        assert point.model_input["wake_event_types"] == ()
        assert "elapsed_minutes" not in point.model_input
        assert "timers" not in repr(point.model_input).lower()
        assert point.backend_facts["phase"]
        assert point.expected_obligation


def test_ordered_preserves_static_policy_contract_at_every_point_and_turn(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    agent = ModelAgent(_NeverCalled(), model="test-model")
    duration_policy_keys = {
        "approval_deadline_after_minutes",
        "approval_reminder_after_minutes",
        "approval_validity_minutes",
        "completion_notice_within_minutes",
        "provisional_close_minutes",
        "visit_followup_after_minutes",
    }

    for static_point, ordered_point in zip(
        arms[ARM_STATIC].points, arms[ARM_ORDERED].points, strict=True
    ):
        static_session = static_point.new_retrieval_session()
        ordered_session = ordered_point.new_retrieval_session()
        for decision in (None, "all_reads"):
            if decision is not None:
                static_session.submit(_all_reads(static_point))
                ordered_session.submit(_all_reads(ordered_point))

            static_observation = static_session.current_observation()
            ordered_observation = ordered_session.current_observation()
            assert duration_policy_keys <= set(static_observation.policy)
            assert ordered_observation.policy == static_observation.policy
            assert (
                ordered_session.model_projection["policy"]
                == (static_session.model_projection["policy"])
            )

            static_request = agent.build_request(static_observation)
            ordered_request = agent.build_request(ordered_observation)
            static_wire_prompt = json.loads(
                build_model_payload(static_request, model="test-model")["input"][0][
                    "content"
                ]
            )
            ordered_wire_prompt = json.loads(
                build_model_payload(ordered_request, model="test-model")["input"][0][
                    "content"
                ]
            )
            assert (
                ordered_wire_prompt["observation"]["policy"]
                == (static_wire_prompt["observation"]["policy"])
            )


def test_ordered_retrieval_applies_only_the_declared_time_transformation(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    static_result = (
        arms[ARM_STATIC].points[1].new_retrieval_session().serve(("list_checkpoints",))[0]
    )
    ordered_result = (
        arms[ARM_ORDERED]
        .points[1]
        .new_retrieval_session()
        .serve(("list_checkpoints",))[0]
    )
    anchor = arms[ARM_STATIC].points[0].model_input["now"]
    assert ordered_result.as_of == anchor
    assert "deadline_at" in repr(static_result.records)
    assert "deadline_at" not in repr(ordered_result.records)
    assert ordered_result.records["approvals"]["checkpoint_1"]["status"] == (
        "RESOLVED_APPROVED"
    )
    assert ordered_result.record_version == record_version(ordered_result.records)


def test_full_reuses_engine_and_retains_wait_time_wakes_and_late_event(compiled) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    assert full.engine_class == "operatebench.core.engine.Engine"
    assert full.full_outcome is not None
    rows = full.full_outcome.trajectory
    assert any(row["record_type"] == "wait_declared" for row in rows)
    assert any(row["record_type"] == "timer_scheduled" for row in rows)
    assert full.full_outcome.simulated_minutes > 0
    assert any(event["event_id"] == "v1_e18" for event in full.full_outcome.events)


def test_later_points_are_reached_after_their_authored_lifecycle_mechanisms(
    compiled,
) -> None:
    _spec, scenario, arms = compiled
    full = arms[ARM_FULL]
    rows = full.full_outcome.trajectory
    required = {point.point_id: point.prior_mechanisms for point in scenario.points}
    assert "wait_declared" in required["approval_granted"]
    assert "approver_decision_received" in required["approval_granted"]
    assert "work_evidence_verified" in required["work_verified"]
    assert "invoice_validation_completed" in required["invoice_validated"]
    assert "operation_reopened" in required["warranty_reopened"]
    compiled_points = {point.point_id: point for point in arms[ARM_STATIC].points}
    for point_id, mechanisms in required.items():
        reached_at = compiled_points[point_id].model_input["now"]
        for mechanism in mechanisms:
            before_boundary = [
                row
                for row in rows
                if mechanism in {row.get("record_type"), row.get("event_type")}
                and row["at"] <= reached_at
            ]
            assert before_boundary, (point_id, mechanism, reached_at)


def test_exact_same_type_action_from_earlier_cycle_does_not_establish_reopen(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    initial, *_, reopened = arms[ARM_STATIC].points
    earlier = Act(
        action_type="request_supplier_visit",
        payload=dict(initial.expected_payload),
        evidence_refs=initial.expected_evidence_refs,
        rationale="earlier cycle",
    )
    assert initial.establish(earlier) == "ADMISSIBLE"
    assert reopened.establish(earlier) == NOT_ESTABLISHED
    correct = Act(
        action_type="request_supplier_visit",
        payload=dict(reopened.expected_payload),
        evidence_refs=reopened.expected_evidence_refs,
        rationale="exact reopened cycle",
    )
    assert reopened.establish(correct) == "ADMISSIBLE"


def test_direct_establish_rejects_boolean_for_integer_payload_field(compiled) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[1]
    assert point.point_id == "approval_granted"
    assert type(point.expected_payload["quote_version"]) is int
    proposal = Act(
        action_type=point.expected_obligation.removeprefix("ACT:"),
        payload={**dict(point.expected_payload), "quote_version": True},
        evidence_refs=point.expected_evidence_refs,
        rationale="boolean is not an integer",
    )

    assert point.establish(proposal) == NOT_ESTABLISHED


@pytest.mark.parametrize("payload", ([{"x": 1}], 1))
def test_direct_establish_rejects_non_mapping_payload_without_exception(
    compiled, payload
) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[0]
    proposal = Act(
        action_type=point.expected_obligation.removeprefix("ACT:"),
        payload=payload,
        evidence_refs=point.expected_evidence_refs,
        rationale="malformed payload",
    )

    assert point.establish(proposal) == NOT_ESTABLISHED


@pytest.mark.parametrize("scope_representation", ("omitted", "explicit_null"))
def test_initial_report_optional_null_payload_equivalence_is_admissible(
    compiled, scope_representation: str
) -> None:
    spec, _scenario, arms = compiled
    for arm_name, runner in (
        (ARM_STATIC, lambda point, decisions: run_static(point, decisions)),
        (ARM_BOUNDARY, lambda point, decisions: run_boundary(spec, point, decisions)),
    ):
        point = arms[arm_name].points[0]
        assert point.expected_payload["scope_digest"] is None
        payload = dict(point.expected_payload)
        if scope_representation == "omitted":
            payload.pop("scope_digest")
        else:
            payload["scope_digest"] = None
        proposal = Act(
            action_type="request_supplier_visit",
            payload=payload,
            evidence_refs=point.expected_evidence_refs,
            rationale=f"{scope_representation} optional scope",
        )

        result = runner(point, (_all_reads(point), proposal))

        assert result.vector_status == "ADMISSIBLE"
        if arm_name == ARM_BOUNDARY:
            assert result.effect_applied


def test_ordered_initial_report_wrong_case_visit_type_remains_not_established(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_ORDERED].points[0]
    proposal = Act(
        action_type="request_supplier_visit",
        payload={**dict(point.expected_payload), "visit_type": "diagnostic"},
        evidence_refs=point.expected_evidence_refs,
        rationale="wrong-case visit type",
    )

    result = run_static(point, (_all_reads(point), proposal))

    assert result.vector_status == NOT_ESTABLISHED
    assert not result.effect_applied


def test_initial_report_material_non_null_scope_mismatch_remains_not_established(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    for arm_name, runner in (
        (ARM_STATIC, lambda point, decisions: run_static(point, decisions)),
        (ARM_BOUNDARY, lambda point, decisions: run_boundary(spec, point, decisions)),
    ):
        point = arms[arm_name].points[0]
        proposal = Act(
            action_type="request_supplier_visit",
            payload={
                **dict(point.expected_payload),
                "scope_digest": "sha256:materially-different",
            },
            evidence_refs=point.expected_evidence_refs,
            rationale="wrong non-null scope",
        )

        result = runner(point, (_all_reads(point), proposal))

        assert result.vector_status == NOT_ESTABLISHED
        assert not result.effect_applied


@pytest.mark.parametrize(
    ("case", "action_type", "payload_change", "evidence_refs"),
    (
        ("unknown field", "request_supplier_visit", {"unknown": "field"}, ()),
        ("missing required", "request_supplier_visit", {"cycle_id": None}, ()),
        ("wrong type", "request_supplier_visit", {"cycle_id": 7}, ()),
        ("wrong value", "request_supplier_visit", {"visit_type": "diagnostic"}, ()),
        (
            "wrong action",
            "request_payment",
            {"invoice_id": "invoice_1", "amount_minor": 47500, "currency": "GBP"},
            (),
        ),
        ("wrong evidence", "request_supplier_visit", {}, ("not-required",)),
    ),
)
def test_boundary_never_credits_reference_fallback_after_subject_proposal(
    compiled,
    case: str,
    action_type: str,
    payload_change: dict[str, object],
    evidence_refs: tuple[str, ...],
) -> None:
    spec, _scenario, arms = compiled
    point = arms[ARM_BOUNDARY].points[0]
    payload = dict(point.expected_payload)
    if case == "missing required":
        payload.pop("cycle_id")
    elif case == "wrong action":
        payload = payload_change
    else:
        payload.update(payload_change)
    proposal = Act(
        action_type=action_type,
        payload=payload,
        evidence_refs=evidence_refs,
        rationale=f"invalid subject proposal: {case}",
    )

    result = run_boundary(spec, point, (_all_reads(point), proposal))

    assert result.vector_status == NOT_ESTABLISHED
    assert not result.effect_applied


def test_reference_solver_establishes_the_common_vector(compiled) -> None:
    _spec, _scenario, arms = compiled
    assert (
        verify_reference_vector(arms, ReferenceAgent())
        == arms[ARM_FULL].common_expected_vector
    )


def test_targeted_mutations_fail_independent_recomputation(compiled) -> None:
    spec, scenario, _arms = compiled
    changed = copy.deepcopy(scenario.semantic_payload())
    changed["points"][0]["expected_obligation"] = "ACT:request_payment"
    with pytest.raises(CompilationError, match=r"digest|obligation"):
        compile_maintenance_v1(spec, scenario, semantic_override=changed)
    wrong_spec = copy.copy(spec)
    object.__setattr__(wrong_spec, "spec_digest_sha256", "0" * 64)
    with pytest.raises(CompilationError, match="spec identity"):
        compile_maintenance_v1(wrong_spec, scenario)


def test_non_scores_are_explicit_and_not_booleans() -> None:
    assert (UNREACHED, NOT_ESTABLISHED, ERROR) == (
        "UNREACHED",
        "NOT_ESTABLISHED",
        "ERROR",
    )
    assert not any(
        isinstance(value, bool) for value in (UNREACHED, NOT_ESTABLISHED, ERROR)
    )


def _exact(point):
    return Act(
        action_type=point.expected_obligation.removeprefix("ACT:"),
        payload=dict(point.expected_payload),
        evidence_refs=point.expected_evidence_refs,
        rationale="exact bound proposal",
    )


def _all_reads(point):
    return RetrieveBatch(
        tuple(RetrievalRequest(tool=tool) for tool in point.retrieval_catalogue)
    )


def _mutable(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable(item) for item in value]
    return value


def _assert_plain_json_tree(value):
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for item in value.values():
            _assert_plain_json_tree(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain_json_tree(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool))


def test_runtime_observations_are_fresh_deep_mutable_json_trees(compiled) -> None:
    _spec, _scenario, arms = compiled
    for arm_name in (ARM_STATIC, ARM_BOUNDARY, ARM_ORDERED):
        session = arms[arm_name].points[2].new_retrieval_session()
        session.submit(_all_reads(arms[arm_name].points[2]))
        first = session.current_observation()
        detached = copy.deepcopy(first)
        _assert_plain_json_tree(first.as_dict())
        _assert_plain_json_tree(detached.as_dict())
        first.retrieval["served"]["get_case_record"]["records"]["cycles"].clear()
        second = session.current_observation()
        assert second.retrieval["served"]["get_case_record"]["records"]["cycles"]
        _assert_plain_json_tree(
            ModelAgent(_NeverCalled(), model="test-model").build_request(second).prompt
        )


def test_static_and_boundary_runtime_clocks_and_wakes_preserve_actual_points(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    for index in range(5):
        static = arms[ARM_STATIC].points[index]
        boundary = arms[ARM_BOUNDARY].points[index]
        for point in (static, boundary):
            observation = point.new_retrieval_session().current_observation()
            assert observation.now == point.model_input["now"] == point._as_of
            assert observation.wake_event_types == list(
                point.model_input["wake_event_types"]
            )
        assert static.model_input == boundary.model_input


def test_ordered_projection_has_no_model_visible_arm_cue(compiled) -> None:
    _spec, _scenario, arms = compiled
    for point in arms[ARM_ORDERED].points:
        assert "treatment" not in point.model_input
        assert "arm" not in point.model_input


def test_ordered_transform_identity_is_harness_metadata_and_digest_input(
    compiled,
) -> None:
    _spec, scenario, arms = compiled
    declared = scenario.semantic_payload()["control_transforms"][ARM_ORDERED]
    assert declared["transform_id"] == ORDERED_TIME_TRANSFORM_ID
    assert declared["anchor_rule"] == ORDERED_TIME_ANCHOR_RULE
    metadata = arms[ARM_ORDERED].harness_metadata
    assert metadata["control_transform_id"] == ORDERED_TIME_TRANSFORM_ID
    assert metadata["anchor_rule"] == ORDERED_TIME_ANCHOR_RULE
    assert metadata["anchor_now"] == arms[ARM_STATIC].points[0].model_input["now"]
    assert ORDERED_TIME_TRANSFORM_ID not in repr(arms[ARM_ORDERED].points)


def test_ordered_turns_never_reintroduce_lifecycle_time_or_wait_ownership(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    anchor = arms[ARM_STATIC].points[0]._as_of
    later_actual_times = {point._as_of for point in arms[ARM_STATIC].points[1:]}
    for point in arms[ARM_ORDERED].points:
        session = point.new_retrieval_session()
        turn = session.submit(_all_reads(point))
        observation = session.current_observation()
        rendered = json.dumps(observation.as_dict())
        assert observation.now == anchor
        assert observation.wake_event_types == []
        assert all(result.as_of == anchor for result in turn.results)
        assert all(
            result.record_version == record_version(result.records)
            for result in turn.results
        )
        assert not (later_actual_times & set(rendered.split('"')))
        assert "elapsed_minutes" not in rendered
        assert "timers" not in rendered.lower()
        action_type = point.expected_obligation.removeprefix("ACT:")
        assert "required" in observation.action_schemas[action_type]
        assert (
            run_static(
                point,
                (Wait(reason="local control wait", wake_on=("anything",)),),
            ).vector_status
            == NOT_ESTABLISHED
        )


def test_frozen_session_evolves_like_an_engine_invocation(compiled) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[3]
    session = point.new_retrieval_session()
    before = session.model_projection
    assert before["turn_index"] == 0
    assert set(before["action_schemas"]["request_payment"]) == {"reads"}
    turn = session.submit(_all_reads(point))
    after = turn.model_projection
    assert after["turn_index"] == 1
    assert after["retrieval"]["batch_budget_remaining"] == 5
    assert set(after["retrieval"]["served"]) == set(point.retrieval_catalogue)
    assert "required" in after["action_schemas"]["request_payment"]
    assert turn.refusal_code is None
    refused = session.submit(RetrieveBatch((RetrievalRequest("not_a_tool"),)))
    assert refused.refusal_code == "UNKNOWN_RETRIEVAL_TOOL"
    assert refused.model_projection["turn_index"] == 2
    assert refused.model_projection["retrieval"]["batch_budget_remaining"] == 5
    assert refused.model_projection["last_rejection"]["action_type"] == "retrieve"


def test_valid_frozen_retrieval_preserves_prior_refusal_signal(compiled) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[3]
    session = point.new_retrieval_session()
    assert session.current_observation().last_rejection is None
    assert session.model_projection["last_rejection"] is None

    refused = session.submit(RetrieveBatch((RetrievalRequest("not_a_tool"),)))
    original_rejection = refused.model_projection["last_rejection"]
    assert original_rejection["action_type"] == "retrieve"
    assert original_rejection["code"] == "UNKNOWN_RETRIEVAL_TOOL"

    served = session.submit(_all_reads(point))
    observation = session.current_observation()
    assert served.refusal_code is None
    assert served.results
    assert observation.retrieval["batch_budget_remaining"] == 5
    assert set(observation.retrieval["served"]) == set(point.retrieval_catalogue)
    assert "required" in observation.action_schemas["request_payment"]
    assert observation.last_rejection == original_rejection
    assert served.model_projection["last_rejection"] == original_rejection


def test_static_grades_reads_and_provenance_without_applying_effect(compiled) -> None:
    _spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[3]
    unread = run_static(point, (_exact(point),))
    assert unread.vector_status == NOT_ESTABLISHED
    assert not unread.effect_applied
    established = run_static(point, (_all_reads(point), _exact(point)))
    assert established.vector_status == "ADMISSIBLE"
    assert not established.effect_applied
    absent = Act(
        action_type="request_payment",
        payload=dict(point.expected_payload),
        evidence_refs=("invoice_absent", "evidence_absent"),
        rationale="invented citations",
    )
    assert run_static(point, (_all_reads(point), absent)).vector_status == NOT_ESTABLISHED


def test_static_uses_the_production_relational_evidence_resolver(compiled) -> None:
    spec, _scenario, arms = compiled
    point = arms[ARM_STATIC].points[2]
    session = point.new_retrieval_session()
    session.submit(_all_reads(point))
    canonical_records = {
        root: value
        for result in session.served.values()
        for root, value in result.records.items()
    }
    resolution = resolve_required_evidence(
        MAINTENANCE_ACTION_EVIDENCE_CONTRACT,
        "request_invoice_validation",
        point.expected_payload,
        canonical_records,
    )
    assert resolution.problem is None
    assert resolution.required == point.expected_evidence_refs
    assert session.establishes(point, _exact(point))

    authoritative = session.served["list_authoritative_records"]
    forged_records = _mutable(authoritative.records)
    forged_records["authoritative_records"].pop(point.expected_evidence_refs[0])
    forged_records["unrelated"] = {"misleading_key": point.expected_evidence_refs[0]}
    session.served[authoritative.tool] = replace(
        authoritative,
        records=forged_records,
    )
    forged_merged = {
        root: value
        for result in session.served.values()
        for root, value in result.records.items()
    }
    rejected = resolve_required_evidence(
        MAINTENANCE_ACTION_EVIDENCE_CONTRACT,
        "request_invoice_validation",
        point.expected_payload,
        forged_merged,
    )
    assert rejected.problem == "the authoritative work evidence is missing"
    assert not session.establishes(point, _exact(point))

    session.served[authoritative.tool] = replace(
        authoritative,
        as_of="2031-03-09T09:29:00Z",
    )
    assert not session.establishes(point, _exact(point))
    session.served[authoritative.tool] = replace(authoritative, ok=False)
    assert not session.establishes(point, _exact(point))

    boundary = run_boundary(
        spec, arms[ARM_BOUNDARY].points[2], (_all_reads(point), _exact(point))
    )
    assert boundary.vector_status == "ADMISSIBLE"


def test_boundary_executes_exact_proposal_through_production_effect(compiled) -> None:
    spec, _scenario, arms = compiled
    point = arms[ARM_BOUNDARY].points[2]
    result = run_boundary(spec, point, (_all_reads(point), _exact(point)))
    assert result.vector_status == "ADMISSIBLE"
    assert result.effect_applied
    assert result.accepted_effect is not None
    assert result.accepted_effect["action_type"] == "request_invoice_validation"
    assert result.accepted_effect["proposal_id"] == result.proposal_id
    assert any(row["record_type"] == "effect_accepted" for row in result.trajectory)


def test_full_mapping_uses_bound_proposals_and_real_effects(compiled) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    expected = ("ADMISSIBLE",) * 5
    assert map_full_outcome(full.points, full.full_outcome) == expected
    assert full.common_expected_vector == expected
    rows = [dict(row) for row in full.full_outcome.trajectory]
    proposal = next(
        row
        for row in rows
        if row["record_type"] == "action_proposed"
        and row["action_type"] == "authorise_supplier_work"
    )
    proposal["action_type"] = "request_payment"
    assert map_full_outcome(full.points, rows) == (
        "ADMISSIBLE",
        NOT_ESTABLISHED,
        "ADMISSIBLE",
        "ADMISSIBLE",
        "ADMISSIBLE",
    )
    cutoff = full.points[3]._as_of
    truncated = [row for row in rows if row["at"] < cutoff]
    assert map_full_outcome(full.points, truncated) == (
        "ADMISSIBLE",
        NOT_ESTABLISHED,
        "ADMISSIBLE",
        ERROR,
        ERROR,
    )
    broken = [dict(row) for row in full.full_outcome.trajectory]
    broken_row = next(
        row
        for row in broken
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_invoice_validation"
    )
    broken_row.clear()
    broken_row.update(
        {
            "index": 145,
            "record_type": "execution_error",
            "at": full.points[2]._as_of,
        }
    )
    assert map_full_outcome(full.points, broken)[2:] == (ERROR, ERROR, ERROR)

    ambiguous = [dict(row) for row in full.full_outcome.trajectory]
    original = next(
        row
        for row in ambiguous
        if row["record_type"] == "action_proposed"
        and row["action_type"] == "request_invoice_validation"
    )
    duplicate = dict(original)
    duplicate["proposal_id"] = "proposal_ambiguous"
    effect = next(
        dict(row)
        for row in ambiguous
        if row["record_type"] == "effect_accepted"
        and row["proposal_id"] == original["proposal_id"]
    )
    effect["proposal_id"] = "proposal_ambiguous"
    insertion = ambiguous.index(original)
    ambiguous[insertion:insertion] = [duplicate, effect]
    for index, row in enumerate(ambiguous):
        row["index"] = index
    assert map_full_outcome(full.points, ambiguous)[2] == ERROR

    wrong_binding = [dict(row) for row in full.full_outcome.trajectory]
    validation_effect = next(
        row
        for row in wrong_binding
        if row.get("record_type") == "effect_accepted"
        and row.get("action_type") == "request_invoice_validation"
    )
    validation_effect["bindings"] = {"invoice_id": "invoice_unrelated"}
    assert map_full_outcome(full.points, wrong_binding)[2] == NOT_ESTABLISHED

    missing_binding = [dict(row) for row in full.full_outcome.trajectory]
    next(
        row
        for row in missing_binding
        if row.get("record_type") == "effect_accepted"
        and row.get("action_type") == "request_invoice_validation"
    ).pop("bindings")
    assert map_full_outcome(full.points, missing_binding)[2:] == (
        ERROR,
        ERROR,
        ERROR,
    )


def test_full_mapping_keeps_untrusted_missing_first_boundary_error_through_denominator(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    first_event_type = full.points[0].reach_event_type
    rows = [
        dict(row)
        for row in full.full_outcome.trajectory
        if not (
            row.get("record_type") == "event_observed"
            and row.get("event_type") == first_event_type
        )
    ]
    for index, row in enumerate(rows):
        row["index"] = index

    assert map_full_outcome(full.points, rows) == (ERROR,) * 5


class _DelayedInitialReference:
    agent_id = "delayed_initial_reference"

    def __init__(self) -> None:
        self.reference = build_agent("reference")
        self.delayed = False

    def begin_episode(self, identity):
        self.reference.begin_episode(identity)

    def decide(self, observation):
        if not self.delayed:
            self.delayed = True
            return Wait(
                reason="exercise the smallest semantic-time shift",
                wake_on=(),
                fallback_after_minutes=1,
            )
        return self.reference.decide(observation)


class _DeadlockAfterAcceptedAction:
    agent_id = "deadlock_after_accepted_action"

    def __init__(self, action_type: str) -> None:
        self.reference = build_agent("reference")
        self.action_type = action_type
        self.accepted_action_proposed = False

    def begin_episode(self, identity):
        self.reference.begin_episode(identity)

    def decide(self, observation):
        if self.accepted_action_proposed:
            return object()
        outcome = self.reference.decide(observation)
        if isinstance(outcome, Act) and outcome.action_type == self.action_type:
            self.accepted_action_proposed = True
        return outcome


class _HorizonAfterAcceptedAction:
    agent_id = "horizon_after_accepted_action"

    def __init__(self, action_type: str) -> None:
        self.reference = build_agent("reference")
        self.action_type = action_type
        self.accepted_action_proposed = False

    def begin_episode(self, identity):
        self.reference.begin_episode(identity)

    def decide(self, observation):
        if self.accepted_action_proposed:
            return Wait(
                reason="exercise a trusted horizon terminal before the suffix",
                wake_on=("operator_exception_resolved",),
            )
        outcome = self.reference.decide(observation)
        if isinstance(outcome, Act) and outcome.action_type == self.action_type:
            self.accepted_action_proposed = True
        return outcome


def test_full_mapping_marks_terminal_deadlock_suffix_unreached_after_admissible(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _DeadlockAfterAcceptedAction("request_supplier_visit")
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "full-terminal-deadlock",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()

    assert outcome.status == STATUS_DEADLOCK
    assert has_canonical_episode_outcome_provenance(outcome)
    assert "_provenance" not in outcome.as_dict()
    assert "authentication" not in json.dumps(outcome.as_dict())
    assert map_full_outcome(arms[ARM_FULL].points, outcome) == (
        "ADMISSIBLE",
        UNREACHED,
        UNREACHED,
        UNREACHED,
        UNREACHED,
    )
    raw_rows = [dict(row) for row in outcome.trajectory]
    assert map_full_outcome(arms[ARM_FULL].points, raw_rows)[1] == ERROR
    mismatched = replace(outcome, status=STATUS_HORIZON)
    assert map_full_outcome(arms[ARM_FULL].points, mismatched)[1] == ERROR

    exact_caller_copy = EpisodeOutcome(**outcome.as_dict())
    exact_replacement = replace(outcome)
    assert not has_canonical_episode_outcome_provenance(exact_caller_copy)
    assert not has_canonical_episode_outcome_provenance(exact_replacement)
    assert map_full_outcome(arms[ARM_FULL].points, exact_caller_copy)[1] == ERROR
    assert map_full_outcome(arms[ARM_FULL].points, exact_replacement)[1] == ERROR

    non_monotonic = [dict(row) for row in raw_rows]
    non_monotonic[1]["index"] = non_monotonic[0]["index"]
    assert map_full_outcome(arms[ARM_FULL].points, non_monotonic) == (ERROR,) * 5

    mapping_error_rows = [dict(row) for row in raw_rows]
    mapping_error_row = next(
        row for row in mapping_error_rows if row["record_type"] == "outcome_rejected"
    )
    mapping_error_row["record_type"] = "mapping_error"
    mapping_error_digest = hashlib.sha256(
        canonical_json_bytes(mapping_error_rows, "mapping-error trajectory")
    ).hexdigest()
    mapping_error = replace(
        outcome,
        trajectory=mapping_error_rows,
        trajectory_digest_sha256=mapping_error_digest,
    )
    assert map_full_outcome(arms[ARM_FULL].points, mapping_error)[0] == ERROR

    forged_rows = [dict(row) for row in outcome.trajectory]
    forged_rows[-1]["status"] = STATUS_HORIZON
    forged_digest = hashlib.sha256(
        canonical_json_bytes(forged_rows, "forged trajectory")
    ).hexdigest()
    forged = replace(
        outcome,
        status=STATUS_HORIZON,
        trajectory=forged_rows,
        trajectory_digest_sha256=forged_digest,
    )
    assert map_full_outcome(arms[ARM_FULL].points, forged)[1] == ERROR


def test_full_mapping_rejects_a_missing_boundary_when_a_later_boundary_exists(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    outcome = full.full_outcome
    assert outcome is not None
    missing_event_type = full.points[1].reach_event_type
    rows = [
        dict(row)
        for row in outcome.trajectory
        if not (
            row.get("record_type") == "event_observed"
            and row.get("event_type") == missing_event_type
        )
    ]
    for index, row in enumerate(rows):
        row["index"] = index
    rows[-1]["status"] = STATUS_HORIZON
    digest = hashlib.sha256(
        canonical_json_bytes(rows, "missing-middle-boundary trajectory")
    ).hexdigest()
    terminal = replace(
        outcome,
        status=STATUS_HORIZON,
        trajectory=rows,
        trajectory_digest_sha256=digest,
    )

    mapped = map_full_outcome(full.points, terminal)

    assert mapped == ("ADMISSIBLE", ERROR, ERROR, ERROR, ERROR)
    assert any(
        row.get("record_type") == "event_observed"
        and row.get("event_type") == full.points[2].reach_event_type
        for row in rows
    )


def test_full_mapping_marks_terminal_horizon_suffix_unreached_after_admissible(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _HorizonAfterAcceptedAction("request_payment")
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "full-terminal-horizon",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()

    assert outcome.status == STATUS_HORIZON
    assert map_full_outcome(arms[ARM_FULL].points, outcome) == (
        "ADMISSIBLE",
        "ADMISSIBLE",
        "ADMISSIBLE",
        "ADMISSIBLE",
        UNREACHED,
    )


def _redigest_outcome(outcome, **changes):
    forged = replace(outcome, **changes)
    if "trajectory" in changes and "trajectory_digest_sha256" not in changes:
        forged = replace(
            forged,
            trajectory_digest_sha256=hashlib.sha256(
                canonical_json_bytes(
                    [dict(row) for row in forged.trajectory], "forged trajectory"
                )
            ).hexdigest(),
        )
    if "final_state" in changes and "final_state_digest_sha256" not in changes:
        forged = replace(
            forged,
            final_state_digest_sha256=hashlib.sha256(
                canonical_json_bytes(dict(forged.final_state), "forged final state")
            ).hexdigest(),
        )
    return forged


@pytest.mark.parametrize(
    "structural_mutation",
    ["marker", "invocation", "effect", "binding"],
)
def test_full_mapping_keeps_authenticated_structural_error_through_remainder(
    compiled, structural_mutation
) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    outcome = full.full_outcome
    assert outcome is not None
    rows = [dict(row) for row in outcome.trajectory]
    point = full.points[2]
    marker = next(
        row
        for row in rows
        if row.get("record_type") == "event_observed"
        and row.get("event_type") == point.reach_event_type
    )
    invocation = next(
        row
        for row in rows
        if row.get("record_type") == "agent_invoked"
        and row.get("trigger_event_id") == marker["event_id"]
    )
    proposal = next(
        row
        for row in rows
        if row.get("record_type") == "action_proposed"
        and row.get("action_type") == "request_invoice_validation"
    )
    effect = next(
        row
        for row in rows
        if row.get("record_type") == "effect_accepted"
        and row.get("proposal_id") == proposal["proposal_id"]
    )

    if structural_mutation == "marker":
        marker["event_id"] = ""
    elif structural_mutation == "invocation":
        rows.insert(rows.index(invocation) + 1, dict(invocation))
    elif structural_mutation == "effect":
        duplicate_proposal = dict(proposal, proposal_id="proposal_ambiguous")
        duplicate_effect = dict(effect, proposal_id="proposal_ambiguous")
        rows[rows.index(proposal) : rows.index(proposal)] = [
            duplicate_proposal,
            duplicate_effect,
        ]
    else:
        effect.pop("bindings")
    for index, row in enumerate(rows):
        row["index"] = index
    authenticated = engine_module._attest_episode_outcome(
        _redigest_outcome(outcome, trajectory=rows)
    )
    assert has_canonical_episode_outcome_provenance(authenticated)

    assert map_full_outcome(full.points, authenticated) == (
        "ADMISSIBLE",
        "ADMISSIBLE",
        ERROR,
        ERROR,
        ERROR,
    )


def test_full_mapping_rejects_digest_consistent_forged_deadlock_provenance(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _DeadlockAfterAcceptedAction("request_supplier_visit")
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "forged-terminal-deadlock",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()
    critical = next(
        index
        for index, row in enumerate(outcome.trajectory)
        if row["record_type"] == "critical_violation"
    )

    without_critical = [
        dict(row) for index, row in enumerate(outcome.trajectory) if index != critical
    ]
    for index, row in enumerate(without_critical):
        row["index"] = index
    wrong_code = [dict(row) for row in outcome.trajectory]
    wrong_code[critical]["code"] = "NOT_THE_ENGINE_DEADLOCK_CODE"
    backwards = [dict(row) for row in outcome.trajectory]
    backwards[critical]["at"] = "2000-01-01T00:00:00Z"
    changed_state = dict(outcome.final_state)
    changed_state["forged_terminal_fact"] = True
    changed_events = [dict(event) for event in outcome.events]
    changed_events.pop()

    for forged in (
        _redigest_outcome(outcome, trajectory=without_critical),
        _redigest_outcome(outcome, trajectory=wrong_code),
        _redigest_outcome(outcome, trajectory=backwards),
        replace(outcome, terminal_outcome="transferred_to_human_ownership"),
        replace(outcome, replay_final=True),
        replace(outcome, simulated_minutes=outcome.simulated_minutes + 1),
        _redigest_outcome(outcome, final_state=changed_state),
        replace(outcome, events=changed_events),
    ):
        assert map_full_outcome(arms[ARM_FULL].points, forged)[1] == ERROR


@pytest.mark.parametrize("nested_field", ["trajectory", "final_state", "events"])
def test_nested_episode_outcome_mutation_invalidates_authentication(
    compiled, nested_field
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _DeadlockAfterAcceptedAction("request_supplier_visit")
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": f"nested-mutation-{nested_field}",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()
    assert has_canonical_episode_outcome_provenance(outcome)

    if nested_field == "trajectory":
        outcome.trajectory[-1]["status"] = STATUS_HORIZON
    elif nested_field == "final_state":
        outcome.final_state["forged_terminal_fact"] = True
    else:
        outcome.events[-1]["disposition"] = "forged"

    assert not has_canonical_episode_outcome_provenance(outcome)
    assert map_full_outcome(arms[ARM_FULL].points, outcome)[1] == ERROR


def test_successful_replay_releases_authenticated_outcome_for_valid_suffix(
    compiled, monkeypatch
) -> None:
    spec, _scenario, arms = compiled

    def agent_factory():
        return _DeadlockAfterAcceptedAction("request_supplier_visit")

    run = run_episode(spec, "V1", "reference", agent_factory=agent_factory)
    record = build_artifact(run)

    def reproduce(
        replay_spec,
        scenario_id,
        agent_id,
        *,
        operation_instance_id,
        **_kwargs,
    ):
        return _run_and_evaluate(
            replay_spec,
            scenario_id,
            agent_id,
            operation_instance_id=operation_instance_id,
            agent_factory=agent_factory,
        )

    monkeypatch.setattr(artifact_module, "_reproduce_episode", reproduce)
    report = replay_artifact(spec, record)

    assert report.ok
    assert report.validated_episode_outcome is not None
    assert has_canonical_episode_outcome_provenance(report.validated_episode_outcome)
    assert map_full_outcome(arms[ARM_FULL].points, report.validated_episode_outcome) == (
        "ADMISSIBLE",
        UNREACHED,
        UNREACHED,
        UNREACHED,
        UNREACHED,
    )


def test_full_mapping_rejects_horizon_with_a_contradictory_terminal_acceptance(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _HorizonAfterAcceptedAction("request_payment")
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "forged-terminal-horizon",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()
    rows = [dict(row) for row in outcome.trajectory]
    rows.insert(
        -1,
        {
            "index": len(rows) - 1,
            "at": outcome.ended_at,
            "record_type": "terminal_accepted",
            "outcome": "transferred_to_human_ownership",
            "replay_final": True,
            "cycle_id": None,
        },
    )
    rows[-1]["index"] = len(rows) - 1
    forged = _redigest_outcome(outcome, trajectory=rows)

    assert map_full_outcome(arms[ARM_FULL].points, forged)[4] == ERROR


def test_full_mapping_follows_semantic_boundaries_after_real_engine_time_shift(
    compiled,
) -> None:
    spec, _scenario, arms = compiled
    scenario = spec.scenario("V1")
    agent = _DelayedInitialReference()
    outcome = Engine(
        MaintenanceOperation(spec, "V1"),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "semantic-time-shift",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()
    first_proposal = next(
        row
        for row in outcome.trajectory
        if row["record_type"] == "action_proposed"
        and row["action_type"] == "request_supplier_visit"
    )
    assert first_proposal["at"] != arms[ARM_FULL].points[0]._as_of
    assert map_full_outcome(arms[ARM_FULL].points, outcome) == ("ADMISSIBLE",) * 5


def test_full_mapping_fails_closed_on_ambiguous_semantic_reach(compiled) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    rows = [dict(row) for row in full.full_outcome.trajectory]
    invocation = next(
        row
        for row in rows
        if row.get("record_type") == "agent_invoked"
        and row.get("trigger_event_type") == "work_evidence_verified"
    )
    rows.insert(rows.index(invocation) + 1, dict(invocation))
    for index, row in enumerate(rows):
        row["index"] = index
    assert map_full_outcome(full.points, rows)[2:] == (ERROR, ERROR, ERROR)


def test_full_mapping_fails_closed_on_reordered_required_marker(compiled) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    rows = [dict(row) for row in full.full_outcome.trajectory]
    reopened = next(row for row in rows if row["record_type"] == "operation_reopened")
    rows.remove(reopened)
    first_resolution = next(
        row
        for row in rows
        if row.get("record_type") == "event_observed"
        and row.get("event_type") == "customer_resolution_reported"
    )
    rows.insert(rows.index(first_resolution) + 1, reopened)
    for index, row in enumerate(rows):
        row["index"] = index

    assert map_full_outcome(full.points, rows)[4] == ERROR


def test_full_mapping_requires_trusted_outcome_for_missing_suffix(compiled) -> None:
    _spec, _scenario, arms = compiled
    full = arms[ARM_FULL]
    rows = [
        dict(row)
        for row in full.full_outcome.trajectory
        if row["index"] < 26 and row.get("record_type") != "effect_accepted"
    ]
    assert map_full_outcome(full.points, rows) == (
        NOT_ESTABLISHED,
        ERROR,
        ERROR,
        ERROR,
        ERROR,
    )


def test_every_point_projection_excludes_authored_later_and_final_markers(
    compiled,
) -> None:
    _spec, _scenario, arms = compiled
    forbidden_by_point = {
        "initial_report": ("quote_1", "checkpoint_1", "invoice_1", "settlement_1"),
        "approval_granted": ("evidence_1", "validation_1", "payment_1", "settlement_1"),
        "work_verified": ("validation_1", "payment_1", "settlement_1", "work_cycle_3"),
        "invoice_validated": ("payment_1", "settlement_1", "work_cycle_3", "FINALIZED"),
        "warranty_reopened": ("visit_3", "FINALIZED", "v1_e18"),
    }
    for point in arms[ARM_STATIC].points:
        visible = repr(point.model_input)
        assert all(marker not in visible for marker in forbidden_by_point[point.point_id])


def test_correlation_rejects_unrelated_target_status_records(compiled) -> None:
    _spec, _scenario, arms = compiled
    approval = _mutable(arms[ARM_STATIC].points[1].backend_facts)
    approval["approvals"] = {
        "unrelated": {
            **dict(next(iter(approval["approvals"].values()))),
            "checkpoint_id": "unrelated",
            "cycle_id": "old_cycle",
        }
    }
    with pytest.raises(CompilationError, match=r"correlat|current cycle"):
        validate_point_correlations("approval_granted", approval)
    invoice = _mutable(arms[ARM_STATIC].points[3].backend_facts)
    original = dict(next(iter(invoice["invoices"].values())))
    invoice["invoices"] = {
        "old_invoice": {**original, "invoice_id": "old_invoice", "cycle_id": "old_cycle"},
        "current_invoice": {
            **original,
            "invoice_id": "current_invoice",
            "status": "RECEIVED_UNVALIDATED",
        },
    }
    with pytest.raises(CompilationError, match=r"correlat|current cycle"):
        validate_point_correlations("invoice_validated", invoice)
