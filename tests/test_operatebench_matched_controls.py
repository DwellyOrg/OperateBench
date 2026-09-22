"""The matched decision-point control, and whether its match can be falsified.

A "matched control" is only worth the name if the thing it is matched *to* can
move out from under it and be noticed. The shipped draft is now immutable,
superseded history. Generic validator tests therefore use a clearly labelled,
test-only copy updated from the live spec, change exactly one thing — a fact
handle, an evidence handle, an action affordance, a member of the admissible set
— and require the validator to refuse it **before any episode runs**. This
synthetic grammar is not a benchmark control or evidence and is never shipped.

What is deliberately *not* asserted here: that the control agrees with the
reference agent, or that any particular answer is the right one. The admissible
sets in the shipped manifest are a clearly-labelled project-authored draft, and
the loader refuses to present them as an independent review they have not had.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from operatebench import decision_points as decision_points_module
from operatebench.controls import (
    CONTROL_MANIFEST_NAME,
    INDEPENDENCE_DRAFT,
    INDEPENDENCE_INDEPENDENT,
    MatchManifestError,
    build_match_manifest,
    load_match_manifest,
    match_manifest_path,
    require_match,
    review_target_digest,
    validate_match,
)
from operatebench.core.outcomes import OUTCOME_KINDS
from operatebench.core.read_contract import READ_OPTIONAL, ActionEvidenceContract
from operatebench.decision_points import (
    GRADED_OUTCOMES_PER_DECISION_POINT,
    GRADED_UNIT,
    SHARED_ESTIMAND,
    action_read_requirement_surface,
    observation_projection_fields,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    MAINTENANCE_READ_CONTRACT,
    load_spec,
)
from tests.matched_control_fixtures import (
    test_only_current_validator_payload as _test_only_current_validator_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def raw() -> Mapping[str, Any]:
    return yaml.safe_load(match_manifest_path().read_text(encoding="utf-8"))


@pytest.fixture()
def payload(raw: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    """A private test-only copy, so one sham can never weaken the next test."""
    return _test_only_current_validator_payload(raw, spec)


def _decision_point(payload: Mapping[str, Any], decision_point_id: str) -> dict[str, Any]:
    for entry in payload["decision_points"]:
        if entry["decision_point_id"] == decision_point_id:
            return entry
    raise AssertionError(
        f"the shipped manifest has no decision point {decision_point_id!r}"
    )


def _first(payload: Mapping[str, Any]) -> dict[str, Any]:
    return payload["decision_points"][0]


def _codes(payload: Mapping[str, Any], spec) -> tuple[str, ...]:
    """Build the (structurally valid) manifest and return its match problem codes."""
    manifest = build_match_manifest(payload, "sham")
    return tuple(problem.code for problem in validate_match(manifest, spec).problems)


#: What this file calls the refusal a sham meets before the operator grammar
#: sees it at all. A ``set`` is not JSON-safe, so the manifest cannot be built
#: from it and no match problem is ever reached — a different layer saying the
#: same no, and named here so a matrix can hold both answers to one standard.
BUILD_REFUSED = "MANIFEST_STRUCTURALLY_REFUSED"


def _codes_or_refusal(payload: Mapping[str, Any], spec) -> tuple[str, ...]:
    """The match problem codes this sham raises, or the structural refusal first."""
    try:
        manifest = build_match_manifest(payload, "sham")
    except MatchManifestError:
        return (BUILD_REFUSED,)
    return tuple(problem.code for problem in validate_match(manifest, spec).problems)


def _ceiling(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The first decision point's numeric-ceiling constraint, whatever it is called."""
    outcome = _first(payload)["admissible_outcomes"][0]
    return next(
        value
        for value in outcome["outcome_constraints"].values()
        if value["kind"] == "integer_range"
    )


def _clauses(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = _first(payload)["state_predicate"]["clauses"]
    return clauses


def _scalar_clause(payload: Mapping[str, Any]) -> dict[str, Any]:
    return next(clause for clause in _clauses(payload) if clause["kind"] == "scalar")


def _none_clause(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The first decision point's ``none`` clause: the guard a sham hides in."""
    return next(
        clause
        for clause in _clauses(payload)
        if clause["kind"] == "quantified" and clause["quantifier"] == "none"
    )


def _detail(payload: Mapping[str, Any], spec, code: str) -> str:
    """The single problem of this code the sham raises, so its wording can be held."""
    manifest = build_match_manifest(payload, "sham")
    raised = [
        problem
        for problem in validate_match(manifest, spec).problems
        if problem.code == code
    ]
    assert len(raised) == 1, [problem.as_dict() for problem in raised]
    return raised[0].detail


class TestTheShippedDraftIsSuperseded:
    def test_it_parses_but_does_not_match_the_live_operation(self, spec) -> None:
        manifest = load_match_manifest(match_manifest_path())
        report = validate_match(manifest, spec)
        assert report.ok is False
        assert tuple(problem.code for problem in report.problems) == (
            "ACTION_READ_REQUIREMENT_MISMATCH",
            "ACTION_SCHEMA_MISMATCH",
            "OPERATION_IDENTITY_MISMATCH",
        )

    def test_it_covers_v1_and_v3_and_nothing_else(self, spec) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert {point.scenario_id for point in manifest.decision_points} == {"V1", "V3"}

    def test_its_digest_is_content_addressed_not_byte_addressed(self, raw, spec) -> None:
        manifest = build_match_manifest(raw, "a")
        restated = build_match_manifest(copy.deepcopy(dict(raw)), "b")
        assert manifest.manifest_digest_sha256 == restated.manifest_digest_sha256
        moved = copy.deepcopy(dict(raw))
        _first(moved)["label"] = "a different label for the same state"
        assert (
            build_match_manifest(moved, "c").manifest_digest_sha256
            != manifest.manifest_digest_sha256
        )


class TestAChangedFactHandleBreaksTheMatch:
    def test_a_predicate_path_the_projection_does_not_carry(self, payload, spec) -> None:
        clauses = _first(payload)["state_predicate"]["clauses"]
        scalar = next(clause for clause in clauses if clause["kind"] == "scalar")
        scalar["path"] = "state.current_work_cycle_id"
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)

    def test_a_record_field_the_live_record_does_not_carry(self, payload, spec) -> None:
        clauses = _first(payload)["state_predicate"]["clauses"]
        quantified = next(clause for clause in clauses if clause["kind"] == "quantified")
        quantified["where"][0]["field"] = "state_of_the_quote"
        assert "UNKNOWN_RECORD_FIELD" in _codes(payload, spec)

    def test_a_collection_the_projection_does_not_carry(self, payload, spec) -> None:
        clauses = _first(payload)["state_predicate"]["clauses"]
        quantified = next(clause for clause in clauses if clause["kind"] == "quantified")
        quantified["collection"] = "estimates"
        assert "UNKNOWN_OBSERVATION_COLLECTION" in _codes(payload, spec)

    def test_a_declared_observable_fact_that_moved(self, payload, spec) -> None:
        _first(payload)["observable_facts"][0]["path"] = "state.quotes.*.price_minor"
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)

    def test_a_policy_handle_the_live_policy_view_does_not_carry(
        self, payload, spec
    ) -> None:
        _first(payload)["observable_facts"][0]["path"] = "policy.approval_ceiling_minor"
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)

    def test_hidden_environment_truth_is_not_a_fact_handle(self, payload, spec) -> None:
        _first(payload)["observable_facts"][0]["path"] = (
            "state.hidden.root_cause_fixture_id"
        )
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)


class TestAChangedEvidenceHandleBreaksTheMatch:
    def test_an_actor_that_does_not_hold_the_authority_it_claims(
        self, payload, spec
    ) -> None:
        provenance = _first(payload)["evidence_provenance"][0]
        provenance["produced_by_actor"] = "supplier_1"
        provenance["produced_by_authority"] = "verify_work"
        assert "EVIDENCE_ACTOR_LACKS_AUTHORITY" in _codes(payload, spec)

    def test_an_actor_that_is_not_in_the_registry(self, payload, spec) -> None:
        _first(payload)["evidence_provenance"][0]["produced_by_actor"] = "inspector_1"
        assert "EVIDENCE_ACTOR_NOT_IN_REGISTRY" in _codes(payload, spec)

    def test_a_collection_that_holds_no_citable_evidence(self, payload, spec) -> None:
        _first(payload)["evidence_provenance"][0]["collection"] = "communications"
        assert "UNKNOWN_EVIDENCE_COLLECTION" in _codes(payload, spec)

    def test_an_actor_claim_relabelled_as_an_authoritative_record(
        self, payload, spec
    ) -> None:
        provenance = _first(payload)["evidence_provenance"]
        provenance.append(
            {
                "collection": "assertions",
                "evidence_class": "authoritative_record",
                "citable_as_evidence": True,
                "produced_by_actor": "supplier_1",
                "produced_by_authority": "assert_status",
                "note": "a supplier claim, dressed as a verification",
            }
        )
        codes = _codes(payload, spec)
        assert "EVIDENCE_CLASS_MISMATCH" in codes


class TestAChangedActionAffordanceBreaksTheMatch:
    def test_the_current_all_required_surface_keeps_its_exact_bytes(self) -> None:
        assert action_read_requirement_surface() == (
            "authorise_supplier_work:get_case_record,list_checkpoints,list_quotes",
            "complete:get_case_record,list_billing,list_checkpoints,list_obligations",
            "request_approval:get_case_record,list_checkpoints,list_quotes",
            "request_exception_resolution:get_case_record,list_checkpoints,list_quotes",
            "request_invoice_validation:get_case_record,list_authoritative_records,list_billing",
            "request_payment:get_case_record,list_authoritative_records,list_billing",
            "request_supplier_visit:get_case_record",
            "send_message:get_case_record,list_communications",
        )

    def test_required_literal_optional_suffix_cannot_collide_with_optional_tool(
        self, monkeypatch
    ) -> None:
        contract = ActionEvidenceContract(
            {
                "required_case": {"reads": {"foo=optional": "required"}},
                "optional_case": {"reads": {"foo": "optional"}},
            }
        )
        monkeypatch.setattr(decision_points_module, "MAINTENANCE_READ_CONTRACT", contract)

        surface = action_read_requirement_surface()

        assert surface == (
            "optional_case:foo=optional",
            "required_case:foo%3Doptional",
        )
        assert len(set(surface)) == 2

    def test_read_codec_quotes_every_delimiter_percent_and_non_ascii_component(
        self, monkeypatch
    ) -> None:
        contract = ActionEvidenceContract(
            {
                "out:come,%✓": {
                    "reads": {
                        "z:comma,percent%✓": "required",
                        "a=optional": "optional",
                    }
                }
            }
        )
        monkeypatch.setattr(decision_points_module, "MAINTENANCE_READ_CONTRACT", contract)

        assert action_read_requirement_surface() == (
            "out%3Acome%2C%25%E2%9C%93:a%3Doptional=optional,"
            "z%3Acomma%2Cpercent%25%E2%9C%93",
        )

    def test_encoded_requirements_are_sorted_deterministically(self, monkeypatch) -> None:
        contract = ActionEvidenceContract(
            {"outcome": {"reads": {"z": "optional", "a": "required"}}}
        )
        monkeypatch.setattr(decision_points_module, "MAINTENANCE_READ_CONTRACT", contract)

        assert action_read_requirement_surface() == ("outcome:a,z=optional",)

    def test_an_extra_payload_field_the_operation_does_not_accept(
        self, payload, spec
    ) -> None:
        schema = payload["matched_surface"]["action_schemas"]["request_approval"]
        schema["required"]["approver_actor_id"] = "string"
        assert "ACTION_SCHEMA_MISMATCH" in _codes(payload, spec)

    def test_a_dropped_payload_field_the_operation_requires(self, payload, spec) -> None:
        schema = payload["matched_surface"]["action_schemas"]["request_approval"]
        del schema["required"]["scope_digest"]
        assert "ACTION_SCHEMA_MISMATCH" in _codes(payload, spec)

    def test_a_retyped_payload_field(self, payload, spec) -> None:
        schema = payload["matched_surface"]["action_schemas"]["request_approval"]
        schema["required"]["quote_version"] = "string"
        assert "ACTION_SCHEMA_MISMATCH" in _codes(payload, spec)

    def test_an_action_this_operation_does_not_have(self, payload, spec) -> None:
        payload["matched_surface"]["action_schemas"]["cancel_supplier_visit"] = {
            "required": {"visit_id": "string"},
            "optional": {},
        }
        assert "UNKNOWN_ACTION_TYPE" in _codes(payload, spec)

    def test_a_control_may_not_offer_a_convenient_subset_of_the_actions(
        self, payload, spec
    ) -> None:
        del payload["matched_surface"]["action_schemas"]["request_payment"]
        assert "ACTION_VOCABULARY_MISMATCH" in _codes(payload, spec)

    def test_the_outcome_vocabulary_must_be_cores_own(self, payload, spec) -> None:
        payload["matched_surface"]["outcome_kinds"] = [*OUTCOME_KINDS, "DEFER"]
        assert "OUTCOME_VOCABULARY_MISMATCH" in _codes(payload, spec)

    def test_the_observation_projection_must_be_cores_own(self, payload, spec) -> None:
        fields = payload["matched_surface"]["observation_projection_fields"]
        payload["matched_surface"]["observation_projection_fields"] = [
            *fields,
            "pending_events",
        ]
        assert "OBSERVATION_PROJECTION_MISMATCH" in _codes(payload, spec)

    def test_the_retrieval_catalogue_must_be_the_operations_own(
        self, payload, spec
    ) -> None:
        reads = payload["matched_surface"]["retrieval_catalogue"]
        payload["matched_surface"]["retrieval_catalogue"] = [
            *reads,
            "read_the_hidden_record:oracle_service:system_of_record",
        ]
        assert "RETRIEVAL_CATALOGUE_MISMATCH" in _codes(payload, spec)

    def test_the_declared_read_requirements_must_be_the_canonical_contract(
        self, payload, spec
    ) -> None:
        payload["matched_surface"]["action_read_requirements"] = [
            "request_payment:get_case_record"
        ]
        assert "ACTION_READ_REQUIREMENT_MISMATCH" in _codes(payload, spec)

    def test_required_to_optional_strength_drift_is_a_named_mismatch(
        self, payload, spec, monkeypatch
    ) -> None:
        changed = {
            key: {"reads": dict(MAINTENANCE_READ_CONTRACT.requirements.get(key, {}))}
            for key in MAINTENANCE_READ_CONTRACT.outcome_keys()
        }
        changed["request_supplier_visit"]["reads"]["get_case_record"] = READ_OPTIONAL
        monkeypatch.setattr(
            decision_points_module,
            "MAINTENANCE_READ_CONTRACT",
            ActionEvidenceContract(changed),
        )

        assert "ACTION_READ_REQUIREMENT_MISMATCH" in _codes(payload, spec)

    def test_a_delimiter_bearing_live_tool_is_a_named_mismatch(
        self, payload, spec, monkeypatch
    ) -> None:
        changed = {
            key: {"reads": dict(MAINTENANCE_READ_CONTRACT.requirements.get(key, {}))}
            for key in MAINTENANCE_READ_CONTRACT.outcome_keys()
        }
        changed["request_supplier_visit"]["reads"] = {"foo=optional": "required"}
        monkeypatch.setattr(
            decision_points_module,
            "MAINTENANCE_READ_CONTRACT",
            ActionEvidenceContract(changed),
        )

        detail = _detail(payload, spec, "ACTION_READ_REQUIREMENT_MISMATCH")
        assert "foo%3Doptional" in detail

    def test_the_declared_read_budget_must_be_the_one_an_arm_is_offered(
        self, payload, spec
    ) -> None:
        payload["matched_surface"]["retrieval_batch_budget"] = [99, 8]
        assert "RETRIEVAL_BUDGET_MISMATCH" in _codes(payload, spec)


class TestAChangedAdmissibleSetBreaksTheMatch:
    def test_an_outcome_naming_an_action_the_manifest_never_declared(
        self, payload, spec
    ) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        del payload["matched_surface"]["action_schemas"][outcome["action_type"]]
        assert "ADMISSIBLE_OUTCOME_UNDECLARED_ACTION" in _codes(payload, spec)

    def test_a_payload_constraint_outside_the_live_action_schema(
        self, payload, spec
    ) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        outcome["outcome_constraints"]["urgency"] = {"kind": "free_identifier"}
        assert "ADMISSIBLE_OUTCOME_UNKNOWN_FIELD" in _codes(payload, spec)

    def test_a_required_payload_field_left_unconstrained(self, payload, spec) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        del outcome["outcome_constraints"]["cycle_id"]
        assert "ADMISSIBLE_OUTCOME_UNCONSTRAINED_FIELD" in _codes(payload, spec)

    def test_a_binding_no_clause_in_the_predicate_binds(self, payload, spec) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        constraint = next(
            value
            for value in outcome["outcome_constraints"].values()
            if value["kind"] == "bound_field"
        )
        constraint["binding"] = "the_quote_we_wish_we_had"
        assert "UNKNOWN_BINDING" in _codes(payload, spec)

    def test_a_bound_field_the_bound_record_does_not_carry(self, payload, spec) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        constraint = next(
            value
            for value in outcome["outcome_constraints"].values()
            if value["kind"] == "bound_field"
        )
        constraint["field"] = "price_minor"
        assert "UNKNOWN_RECORD_FIELD" in _codes(payload, spec)

    def test_an_outcome_kind_outside_the_core_vocabulary(self, payload, spec) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        outcome["outcome_kind"] = "DEFER"
        with pytest.raises(MatchManifestError, match="outcome_kind"):
            build_match_manifest(payload, "sham")

    def test_an_empty_admissible_set_is_not_a_closed_set(self, payload, spec) -> None:
        _first(payload)["admissible_outcomes"] = []
        with pytest.raises(MatchManifestError, match="admissible"):
            build_match_manifest(payload, "sham")

    def test_a_maximum_reference_the_policy_view_does_not_carry(
        self, payload, spec
    ) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        constraint = next(
            value
            for value in outcome["outcome_constraints"].values()
            if value["kind"] == "integer_range"
        )
        constraint["maximum_ref"] = "policy.approval_grace_minutes"
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)


class TestTheControlStaysBoundToOneSemanticScenario:
    def test_a_scenario_the_spec_does_not_declare(self, payload, spec) -> None:
        _first(payload)["scenario_id"] = "V9"
        assert "UNKNOWN_SCENARIO" in _codes(payload, spec)

    def test_a_scenario_outside_the_stage_one_arms(self, payload, spec) -> None:
        _first(payload)["scenario_id"] = "V2"
        assert "SCENARIO_OUTSIDE_MATCHED_ARMS" in _codes(payload, spec)

    def test_another_operations_semantics(self, payload, spec) -> None:
        payload["semantic_scenario_id"] = "maintenance_emergency_flood_synthetic_v1"
        assert "SEMANTIC_SCENARIO_MISMATCH" in _codes(payload, spec)

    def test_another_operation_id(self, payload, spec) -> None:
        payload["operation_id"] = "lettings_arrears_synthetic_v1"
        assert "OPERATION_IDENTITY_MISMATCH" in _codes(payload, spec)

    def test_the_denominator_is_one_semantic_scenario(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.denominator.semantic_scenarios == 1
        assert (
            manifest.denominator.decision_points_are_independent_semantic_samples is False
        )
        assert manifest.shared_estimand == SHARED_ESTIMAND

    def test_decision_points_may_not_be_claimed_as_independent_samples(
        self, payload
    ) -> None:
        payload["denominator"]["decision_points_are_independent_semantic_samples"] = True
        with pytest.raises(MatchManifestError, match="independent"):
            build_match_manifest(payload, "sham")

    def test_the_denominator_may_not_be_inflated_to_the_decision_point_count(
        self, payload
    ) -> None:
        payload["denominator"]["semantic_scenarios"] = len(payload["decision_points"])
        with pytest.raises(MatchManifestError, match="semantic_scenarios"):
            build_match_manifest(payload, "sham")


class TestStructuralStrictness:
    def test_an_unknown_top_level_field(self, payload) -> None:
        payload["expected_terminal"] = "completed_successfully"
        with pytest.raises(MatchManifestError, match="unknown field"):
            build_match_manifest(payload, "sham")

    def test_a_missing_top_level_field(self, payload) -> None:
        del payload["shared_estimand"]
        with pytest.raises(MatchManifestError, match="missing"):
            build_match_manifest(payload, "sham")

    def test_an_unknown_decision_point_field(self, payload) -> None:
        _first(payload)["expected_outcome"] = "ACT"
        with pytest.raises(MatchManifestError, match="unknown field"):
            build_match_manifest(payload, "sham")

    def test_a_duplicate_decision_point_id(self, payload) -> None:
        payload["decision_points"].append(copy.deepcopy(_first(payload)))
        with pytest.raises(MatchManifestError, match="twice"):
            build_match_manifest(payload, "sham")

    def test_a_duplicate_admissible_outcome_id(self, payload) -> None:
        outcomes = _first(payload)["admissible_outcomes"]
        outcomes.append(copy.deepcopy(outcomes[0]))
        with pytest.raises(MatchManifestError, match="twice"):
            build_match_manifest(payload, "sham")

    def test_a_boolean_where_a_count_belongs(self, payload) -> None:
        payload["denominator"]["semantic_scenarios"] = True
        with pytest.raises(MatchManifestError, match="integer"):
            build_match_manifest(payload, "sham")

    def test_a_boolean_where_a_deadline_belongs(self, payload) -> None:
        outcome = _first(payload)["admissible_outcomes"][0]
        constraint = next(
            value
            for value in outcome["outcome_constraints"].values()
            if value["kind"] == "integer_range"
        )
        constraint["minimum"] = True
        with pytest.raises(MatchManifestError, match="integer"):
            build_match_manifest(payload, "sham")

    def test_a_container_where_an_identifier_belongs(self, payload) -> None:
        _first(payload)["decision_point_id"] = ["v1", "quote"]
        with pytest.raises(MatchManifestError, match="non-empty string"):
            build_match_manifest(payload, "sham")

    def test_a_clause_kind_this_build_does_not_know(self, payload) -> None:
        _first(payload)["state_predicate"]["clauses"][0]["kind"] = "regex"
        with pytest.raises(MatchManifestError, match="clause kind"):
            build_match_manifest(payload, "sham")

    def test_a_scalar_clause_that_states_two_comparands(self, payload) -> None:
        clauses = _first(payload)["state_predicate"]["clauses"]
        scalar = next(clause for clause in clauses if clause["kind"] == "scalar")
        scalar["value"] = "work_cycle_2"
        with pytest.raises(MatchManifestError, match="exactly one"):
            build_match_manifest(payload, "sham")

    def test_a_manifest_path_that_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(MatchManifestError, match="cannot read"):
            load_match_manifest(tmp_path / "nowhere" / "control.yaml")

    def test_a_manifest_that_is_not_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "control.yaml"
        path.write_text("- a list is not a control manifest\n", encoding="utf-8")
        with pytest.raises(MatchManifestError, match="mapping"):
            load_match_manifest(path)

    def test_a_manifest_that_is_not_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "control.yaml"
        path.write_text("control_id: [unclosed\n", encoding="utf-8")
        with pytest.raises(MatchManifestError, match="YAML"):
            load_match_manifest(path)


class TestNothingLeaksThroughTheManifest:
    def test_it_names_no_terminal_no_finding_and_no_hidden_fixture(self, raw) -> None:
        from operatebench.domains.lettings.maintenance.state import PRODUCIBLE_TERMINALS

        text = yaml.safe_dump(dict(raw), sort_keys=True)
        for terminal in PRODUCIBLE_TERMINALS:
            assert terminal not in text, (
                f"the control names the terminal {terminal!r}; a decision-point "
                "control that carries the episode's ending has leaked it"
            )
        for leaked in (
            "hidden_fixture_perished_seal",
            "hidden_fixture_reliability_band_2",
            "expected_terminal",
            "trajectory",
        ):
            assert leaked not in text

    def test_a_manifest_may_not_declare_an_expected_answer(self, payload) -> None:
        _first(payload)["reference_outcome"] = {"outcome_kind": "ACT"}
        with pytest.raises(MatchManifestError, match="unknown field"):
            build_match_manifest(payload, "sham")


class TestTheIndependentReviewSlotCannotBeSelfCompleted:
    def test_the_shipped_manifest_is_a_labelled_draft(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.provenance.independence == INDEPENDENCE_DRAFT
        assert manifest.provenance.independent_review is None
        assert manifest.is_independent is False

    def test_the_draft_cannot_be_loaded_as_independent_approval(self, payload) -> None:
        payload["provenance"]["independence"] = INDEPENDENCE_INDEPENDENT
        with pytest.raises(MatchManifestError, match="independent_review"):
            build_match_manifest(payload, "sham")

    def test_a_review_the_authoring_party_signed_itself(self, payload) -> None:
        payload["provenance"]["independence"] = INDEPENDENCE_INDEPENDENT
        payload["provenance"]["review_status"] = "INDEPENDENTLY_REVIEWED"
        payload["provenance"]["independent_review"] = {
            "reviewer_id": "operatebench_project",
            "reviewer_affiliation": payload["provenance"]["authored_by"],
            "reviewed_at": "2026-08-11T09:00:00Z",
            "attestation_digest_sha256": "0" * 64,
            "reviewed_manifest_digest_sha256": "0" * 64,
        }
        with pytest.raises(MatchManifestError, match="self-review"):
            build_match_manifest(payload, "sham")

    def test_a_draft_may_not_also_claim_a_review(self, payload) -> None:
        payload["provenance"]["independent_review"] = {
            "reviewer_id": "someone",
            "reviewer_affiliation": "elsewhere",
            "reviewed_at": "2026-08-11T09:00:00Z",
            "attestation_digest_sha256": "0" * 64,
            "reviewed_manifest_digest_sha256": "0" * 64,
        }
        with pytest.raises(MatchManifestError, match="draft"):
            build_match_manifest(payload, "sham")

    def test_no_public_call_can_grant_independence(self) -> None:
        import operatebench.controls as controls

        forbidden = [
            name
            for name in dir(controls)
            if not name.startswith("_")
            and any(
                word in name.lower() for word in ("approve", "attest", "sign", "grant")
            )
        ]
        assert forbidden == [], (
            f"{forbidden} would let the authoring party complete its own independent "
            "review slot; that gate is a human one and has no code path"
        )

    def test_the_evidence_class_says_what_this_is(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.provenance.evidence_class == "DRAFT_TEMPLATE_NOT_VALIDATION"
        assert manifest.provenance.review_status == "AWAITING_INDEPENDENT_REVIEW"


class TestAComparandMustResolveToExactlyOneScalar:
    """A handle a comparison is made *against* is not the same as a fact handle.

    ``state.quotes`` is a perfectly good thing for a control to declare as an
    observable fact and a nonsense thing to bound a deadline by: nothing in the
    registry is a number, so the ceiling would resolve to no value at all and an
    apparently enforced maximum would silently stop applying at grading time. The
    match validator refuses every such handle by name, before anything executes.
    """

    def test_a_registry_root_as_a_numeric_ceiling(self, payload, spec) -> None:
        _ceiling(payload)["maximum_ref"] = "state.quotes"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_wildcard_record_path_as_a_numeric_ceiling(self, payload, spec) -> None:
        # A legal fact handle — it names a field of every quote — and still not a
        # single value: "the amounts" is not a ceiling.
        _ceiling(payload)["maximum_ref"] = "state.quotes.*.amount_minor"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_fixed_record_root_as_a_numeric_ceiling(self, payload, spec) -> None:
        _ceiling(payload)["maximum_ref"] = "state.payment"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_string_policy_handle_as_a_numeric_ceiling(self, payload, spec) -> None:
        _ceiling(payload)["maximum_ref"] = "policy.currency"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_boolean_state_handle_as_a_numeric_ceiling(self, payload, spec) -> None:
        # ``True`` is an ``int`` in Python, so a bound taken from a yes/no field
        # would quietly become a ceiling of one minute.
        _ceiling(payload)["maximum_ref"] = "state.transfer_notice_sent"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_an_untyped_scalar_handle_as_a_numeric_ceiling(self, payload, spec) -> None:
        _ceiling(payload)["maximum_ref"] = "state.payment.status"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_an_unresolvable_handle_is_still_an_unknown_path(self, payload, spec) -> None:
        _ceiling(payload)["maximum_ref"] = "state.approval_ceiling_minutes"
        assert "UNKNOWN_OBSERVATION_PATH" in _codes(payload, spec)

    def test_a_scalar_clause_that_compares_a_whole_registry(self, payload, spec) -> None:
        scalar = _scalar_clause(payload)
        scalar["path"] = "state.quotes"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_scalar_clause_comparand_that_is_a_whole_registry(
        self, payload, spec
    ) -> None:
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["value_ref"] = "state.approvals"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_record_condition_comparand_that_is_a_whole_registry(
        self, payload, spec
    ) -> None:
        quantified = next(
            clause for clause in _clauses(payload) if clause["kind"] == "quantified"
        )
        condition = next(
            item for item in quantified["where"] if item["value_ref"] is not None
        )
        condition["value_ref"] = "state.approvals"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_an_ordering_comparison_against_a_yes_no_literal(self, payload, spec) -> None:
        quantified = next(
            clause for clause in _clauses(payload) if clause["kind"] == "quantified"
        )
        condition = next(
            item for item in quantified["where"] if item["value_ref"] is not None
        )
        condition["value_ref"] = None
        condition["value"] = True
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_the_test_only_current_comparands_are_all_typed_scalars(
        self, payload, spec
    ) -> None:
        manifest = build_match_manifest(payload, "test-only")
        report = validate_match(manifest, spec)
        assert report.ok, [problem.as_dict() for problem in report.problems]


class TestAComparisonNothingCouldEverDecideIsRefused:
    """A clause whose answer is fixed before the episode starts enforces nothing.

    A comparison the runtime can never make true is not a strict rule; under a
    ``none`` quantifier it is a guard that holds in every state the operation
    could ever reach, so the control reads as a refusal and grades nobody. Each
    sham here is one such comparison, and the match validator has to name it
    before anything runs.
    """

    def test_a_record_ordering_comparison_against_a_string_handle(
        self, payload, spec
    ) -> None:
        # 'no approval has a status greater than the policy currency': ``_compare``
        # refuses to order anything that is not a number, so the clause is false
        # for every record and the ``none`` quantifier is satisfied by every state.
        none_clause = next(
            clause
            for clause in _clauses(payload)
            if clause["kind"] == "quantified" and clause["quantifier"] == "none"
        )
        none_clause["where"] = [
            {"field": "status", "op": "gt", "value": None, "value_ref": "policy.currency"}
        ]
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_record_membership_test_against_a_single_handle(
        self, payload, spec
    ) -> None:
        # ``in`` is answered against a list. Every handle a comparison may name
        # resolves to exactly one value — that is the rule the comparand check
        # already enforces — so membership in one of them holds for no record.
        none_clause = next(
            clause
            for clause in _clauses(payload)
            if clause["kind"] == "quantified" and clause["quantifier"] == "none"
        )
        none_clause["where"] = [
            {
                "field": "status",
                "op": "in",
                "value": None,
                "value_ref": "policy.currency",
            }
        ]
        assert "MEMBERSHIP_COMPARAND_NOT_A_SET" in _codes(payload, spec)

    def test_a_record_membership_test_against_a_bare_literal(self, payload, spec) -> None:
        # The shipped clause lists the approval statuses that block a fresh one.
        # Dropping the list and stating one status reads like a tightening and is
        # the same vacuous guard: ``_compare`` answers a membership test against a
        # non-list with ``False``, whatever the record says.
        condition = next(
            item
            for clause in _clauses(payload)
            if clause["kind"] == "quantified"
            for item in clause["where"]
            if item["op"] == "in"
        )
        condition["value"] = "OPEN"
        assert "MEMBERSHIP_COMPARAND_NOT_A_SET" in _codes(payload, spec)

    def test_a_scalar_membership_test_against_a_single_handle(
        self, payload, spec
    ) -> None:
        # The same impossibility one layer up. A scalar clause is a conjunct, so
        # one that can never hold does not weaken the predicate — it stops it
        # holding anywhere, and the decision point is never reached at all.
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["op"] = "in"
        scalar["value_ref"] = "policy.currency"
        assert "MEMBERSHIP_COMPARAND_NOT_A_SET" in _codes(payload, spec)

    def test_a_scalar_membership_test_against_a_bare_literal(self, payload, spec) -> None:
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["op"] = "in"
        scalar["value"] = "CYC-1"
        assert "MEMBERSHIP_COMPARAND_NOT_A_SET" in _codes(payload, spec)

    def test_a_scalar_equality_between_kinds_that_can_never_be_equal(
        self, payload, spec
    ) -> None:
        # A cycle identifier is a string and an approval threshold is an amount.
        # Both handles resolve, both are single values, and no state exists in
        # which one equals the other — so the conjunct never holds.
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["op"] = "eq"
        scalar["value_ref"] = "policy.approval_threshold_minor"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_scalar_membership_test_against_a_bound_record_field(
        self, payload, spec
    ) -> None:
        # A binding reads one field of one record, which is one value and never a
        # list — so membership against it is the same fixed answer as membership
        # against a handle, and the layer that resolves the binding has to say so.
        scalar = _scalar_clause(payload)
        scalar["op"] = "in"
        assert "MEMBERSHIP_COMPARAND_NOT_A_SET" in _codes(payload, spec)

    def test_a_scalar_equality_against_a_bound_field_of_another_kind(
        self, payload, spec
    ) -> None:
        # The shipped clause asks whether the cycle under way is the quote's own
        # cycle. Pointing it at the quote's amount keeps a resolvable field and a
        # legal operator, and asks whether an identifier equals a sum of money —
        # which no state answers yes.
        scalar = _scalar_clause(payload)
        scalar["binding_field"] = "amount_minor"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_record_ordering_comparison_on_a_field_that_is_not_a_number(
        self, payload, spec
    ) -> None:
        # The same sham from the other side: a real threshold on the right, and a
        # field that holds a status word on the left. ``_compare`` orders nothing
        # that is not a number, so which side is wrong makes no difference to the
        # clause — it is false for every quote there will ever be.
        condition = next(
            item
            for clause in _clauses(payload)
            if clause["kind"] == "quantified"
            for item in clause["where"]
            if item["op"] == "gt"
        )
        condition["field"] = "status"
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_record_membership_test_against_no_members_at_all(
        self, payload, spec
    ) -> None:
        # Emptying the list is the quietest edit of the lot: the clause keeps its
        # shape, its field and its operator, and stops being able to hold.
        condition = next(
            item
            for clause in _clauses(payload)
            if clause["kind"] == "quantified"
            for item in clause["where"]
            if item["op"] == "in"
        )
        condition["value"] = []
        assert "MEMBERSHIP_COMPARAND_EMPTY_SET" in _codes(payload, spec)

    def test_a_record_membership_test_against_members_of_another_kind(
        self, payload, spec
    ) -> None:
        # A list with things in it, and no approval status that could be one of
        # them: ``same_value`` will not read a status word as a number, so the
        # clause is false for every approval and the ``none`` around it holds
        # wherever the operation goes.
        condition = next(
            item
            for clause in _clauses(payload)
            if clause["kind"] == "quantified"
            for item in clause["where"]
            if item["op"] == "in"
        )
        condition["value"] = [1, 2, 3]
        assert "COMPARAND_TYPE_MISMATCH" in _codes(payload, spec)

    def test_a_leaf_the_operation_leaves_untyped_is_still_comparable(
        self, payload, spec
    ) -> None:
        # The limit of the rule, held in place. A fixed record's leaves are typed
        # ``Any`` by the operation, so nothing here disagrees with a status word —
        # and a kind nobody derived is not a kind that conflicts. Refusing this
        # would take a comparison a control may legitimately make.
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["path"] = "state.payment.status"
        scalar["op"] = "eq"
        scalar["value"] = "SETTLED"
        assert _codes(payload, spec) == ()


class TestAComparisonAgainstSomethingThatIsNotOneValueIsRefused:
    """The remaining way a clause is answered before any state reaches it.

    A comparison holds between two single values. These shams put something else
    on one side of it: a field the operation declares as a *collection*, and a
    member of a membership list that is itself a list or a mapping. ``_compare``
    orders nothing that is not a whole number and reads no list as equal to a
    scalar, so each of these is false for every record there will ever be — and
    under the ``none`` quantifier they hide in, true in every state the operation
    could reach. A guard that cannot fail grades nobody, so it is refused by name
    before anything runs.
    """

    def test_an_ordering_comparison_against_a_declared_collection_field(
        self, payload, spec
    ) -> None:
        # Reads as 'no exception carries evidence': ``evidence_refs`` is declared
        # ``tuple[str, ...]``, ordering answers ``False`` for it whatever it
        # holds, and the guard is satisfied by every exception ever opened.
        clause = _none_clause(payload)
        clause["collection"] = "exceptions"
        clause["where"] = [
            {"field": "evidence_refs", "op": "gt", "value": 0, "value_ref": None}
        ]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_an_equality_comparison_against_a_declared_collection_field(
        self, payload, spec
    ) -> None:
        clause = _none_clause(payload)
        clause["collection"] = "exceptions"
        clause["where"] = [
            {"field": "evidence_refs", "op": "eq", "value": "EV-1", "value_ref": None}
        ]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_membership_test_against_a_declared_collection_field(
        self, payload, spec
    ) -> None:
        # 'no exception's evidence is one of these' asks whether a tuple is one
        # of a list of names, which no tuple is.
        clause = _none_clause(payload)
        clause["collection"] = "exceptions"
        clause["where"] = [
            {"field": "evidence_refs", "op": "in", "value": ["EV-1"], "value_ref": None}
        ]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_bound_field_that_is_a_declared_collection(self, payload, spec) -> None:
        # The same field read through a binding instead of a condition. The
        # binder becomes an exceptions record and the scalar clause compares an
        # identifier against that record's list of evidence handles.
        binder = _clauses(payload)[0]
        binder["collection"] = "exceptions"
        binder["where"] = [
            {"field": "status", "op": "eq", "value": "OPEN", "value_ref": None}
        ]
        _scalar_clause(payload)["binding_field"] = "evidence_refs"
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_record_membership_list_whose_member_is_a_list(self, payload, spec) -> None:
        # The quietest of the lot: the clause keeps its field, its operator and a
        # non-empty list, and every member of that list is a thing no status is.
        _none_clause(payload)["where"][0]["value"] = [[1]]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_record_membership_list_with_one_member_that_is_a_mapping(
        self, payload, spec
    ) -> None:
        # One good member does not rescue it: a member that is not one value is
        # a member no record can ever equal, so the list overstates what it tests.
        _none_clause(payload)["where"][0]["value"] = ["OPEN", {"status": "OPEN"}]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_scalar_membership_list_whose_member_is_a_list(self, payload, spec) -> None:
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["op"] = "in"
        scalar["value"] = [["CYC-1"]]
        assert "NON_SCALAR_COMPARAND" in _codes(payload, spec)

    def test_a_membership_test_that_is_constant_says_so_rather_than_never(
        self, payload, spec
    ) -> None:
        # ``not_in`` against members of another kind is not a comparison that
        # never holds — it is one that holds everywhere, which is the same
        # vacuity read from the other end. The diagnostic has to say which.
        condition = _none_clause(payload)["where"][0]
        condition["op"] = "not_in"
        condition["value"] = [1, 2, 3]
        detail = _detail(payload, spec, "COMPARAND_TYPE_MISMATCH")
        assert "could never hold" not in detail
        assert "holds in every state" in detail

    def test_an_unknown_field_is_named_once_and_not_twice(self, payload, spec) -> None:
        # A field the record does not carry has no kind to compare against, so
        # the comparand check has nothing to add: reporting it twice would bury
        # the one thing the author has to fix.
        condition = _none_clause(payload)["where"][0]
        condition["field"] = "state_of_the_approval"
        condition["value"] = "OPEN"
        assert _codes(payload, spec) == ("UNKNOWN_RECORD_FIELD",)

    def test_a_scalar_field_of_the_same_record_is_still_comparable(
        self, payload, spec
    ) -> None:
        # The limit of the rule: one collection field does not make the record
        # it belongs to uncomparable.
        clause = _none_clause(payload)
        clause["collection"] = "exceptions"
        clause["where"] = [
            {"field": "status", "op": "eq", "value": "OPEN", "value_ref": None}
        ]
        assert _codes(payload, spec) == ()

    def test_a_membership_list_may_still_name_the_absence_of_a_value(
        self, payload, spec
    ) -> None:
        # ``null`` is a value a record field genuinely holds — ``resolution`` is
        # ``str | None`` — so a list that names it is a rule some state decides.
        _none_clause(payload)["where"][0]["value"] = ["OPEN", None]
        assert _codes(payload, spec) == ()

    def test_an_untyped_leaf_may_still_be_tested_for_membership(
        self, payload, spec
    ) -> None:
        scalar = _scalar_clause(payload)
        scalar["binding"] = None
        scalar["binding_field"] = None
        scalar["path"] = "state.payment.status"
        scalar["op"] = "in"
        scalar["value"] = ["SETTLED", "PENDING"]
        assert _codes(payload, spec) == ()


#: Every shape a stated literal can take that is not one value, and the name the
#: manifest refuses it under. One table, so the next shape somebody writes is a
#: row rather than a hole: the defect is the *class* — a comparison whose right
#: hand side is not one value — and not the list literal that exposed it.
_NOT_ONE_VALUE: tuple[tuple[Any, str], ...] = (
    (["OPEN"], "NON_SCALAR_COMPARAND"),
    (("OPEN",), "NON_SCALAR_COMPARAND"),
    ({"status": "OPEN"}, "NON_SCALAR_COMPARAND"),
    ([], "NON_SCALAR_COMPARAND"),
    ({}, "NON_SCALAR_COMPARAND"),
    ([["OPEN"]], "NON_SCALAR_COMPARAND"),
    ({"OPEN"}, BUILD_REFUSED),
    (frozenset({"OPEN"}), BUILD_REFUSED),
)
_NOT_ONE_VALUE_IDS = (
    "list",
    "tuple",
    "mapping",
    "empty_list",
    "empty_mapping",
    "nested_list",
    "set",
    "frozenset",
)

#: The same shapes one level down, as *members* of a membership list. A member
#: that is not one value is a member no record can equal, so the list tests less
#: than it reads as testing. ``1.5`` is in the table because the defect is not
#: containers specifically: no field of the operation holds a fractional number
#: either, so a member that is one is the same fixed answer in a different shape.
_MEMBERS_THAT_ARE_NOT_VALUES: tuple[Any, ...] = (["X"], ("X",), {"k": "v"}, 1.5)
_MEMBER_IDS = ("list", "tuple", "mapping", "float")

#: What a membership operator is handed when it is handed something that is not
#: a list of members at all.
_NOT_A_MEMBERSHIP_LIST: tuple[Any, ...] = ({"status": "OPEN"}, "OPEN", 3, True)
_NOT_A_MEMBERSHIP_LIST_IDS = ("mapping", "string", "number", "yes_no")


class TestAComparandLiteralIsExactlyOneValue:
    """A comparison takes one value on each side, whatever the operator is.

    The parent repro states ``approvals.status eq ['OPEN']`` and
    ``eq {'status': 'OPEN'}``: both read as a tightening of the guard that keeps
    a second approval from being opened, and neither is one. ``same_value`` never
    reads a status word as equal to a list or a mapping, so the clause is false
    for every approval — and under the ``none`` quantifier it sits in, *true in
    every state the operation could ever reach*. ``ne`` is the same defect read
    from the other end, and an ordering operator is the same defect again.

    So the matrix below is the point of this class. Closing ``eq`` against a list
    would leave ``ne`` against a list, ``eq`` against a mapping and ``lt``
    against a tuple all open, and each of them is the same vacuous guard wearing
    a different literal. Every scalar operator meets every shape here.
    """

    @pytest.mark.parametrize("op", ("eq", "ne", "gt", "gte", "lt", "lte"))
    @pytest.mark.parametrize(("value", "code"), _NOT_ONE_VALUE, ids=_NOT_ONE_VALUE_IDS)
    def test_a_record_condition_against_a_literal_that_is_not_one_value(
        self, payload, spec, op: str, value: Any, code: str
    ) -> None:
        condition = _none_clause(payload)["where"][0]
        condition.update({"op": op, "value": value, "value_ref": None})
        assert _codes_or_refusal(payload, spec) == (code,)

    @pytest.mark.parametrize("op", ("eq", "ne"))
    @pytest.mark.parametrize(("value", "code"), _NOT_ONE_VALUE, ids=_NOT_ONE_VALUE_IDS)
    def test_a_scalar_clause_against_a_literal_that_is_not_one_value(
        self, payload, spec, op: str, value: Any, code: str
    ) -> None:
        # The same grammar one layer up. A scalar clause is a conjunct of the
        # predicate itself, so a constant one does not weaken the predicate — it
        # decides it, and the decision point is reached always or never.
        scalar = _scalar_clause(payload)
        scalar.update(
            {
                "binding": None,
                "binding_field": None,
                "op": op,
                "value": value,
                "value_ref": None,
            }
        )
        assert _codes_or_refusal(payload, spec) == (code,)

    @pytest.mark.parametrize("op", ("in", "not_in"))
    @pytest.mark.parametrize("member", _MEMBERS_THAT_ARE_NOT_VALUES, ids=_MEMBER_IDS)
    def test_a_membership_list_with_a_member_that_is_not_one_value(
        self, payload, spec, op: str, member: Any
    ) -> None:
        condition = _none_clause(payload)["where"][0]
        condition.update({"op": op, "value": ["OPEN", member], "value_ref": None})
        assert _codes(payload, spec) == ("NON_SCALAR_COMPARAND",)

    @pytest.mark.parametrize("op", ("in", "not_in"))
    @pytest.mark.parametrize(
        "value", _NOT_A_MEMBERSHIP_LIST, ids=_NOT_A_MEMBERSHIP_LIST_IDS
    )
    def test_a_membership_comparand_that_is_not_a_list_of_members(
        self, payload, spec, op: str, value: Any
    ) -> None:
        condition = _none_clause(payload)["where"][0]
        condition.update({"op": op, "value": value, "value_ref": None})
        assert _codes(payload, spec) == ("MEMBERSHIP_COMPARAND_NOT_A_SET",)

    def test_the_list_literal_the_parent_repro_states(self, payload, spec) -> None:
        # The parent repro, verbatim: the whole ``where`` of the guard clause
        # replaced by one equality against a one-member list.
        clause = _clauses(payload)[1]
        clause["where"] = [
            {"field": "status", "op": "eq", "value": ["OPEN"], "value_ref": None}
        ]
        assert _codes(payload, spec) == ("NON_SCALAR_COMPARAND",)

    def test_the_mapping_literal_the_parent_repro_states(self, payload, spec) -> None:
        clause = _clauses(payload)[1]
        clause["where"] = [
            {
                "field": "status",
                "op": "eq",
                "value": {"status": "OPEN"},
                "value_ref": None,
            }
        ]
        assert _codes(payload, spec) == ("NON_SCALAR_COMPARAND",)

    def test_an_equality_against_a_container_says_it_could_never_hold(
        self, payload, spec
    ) -> None:
        # Operator-correct, because the two diagnostics send an author to
        # different places: ``eq`` against a list is a guard that never fires.
        _none_clause(payload)["where"][0].update({"op": "eq", "value": ["OPEN"]})
        detail = _detail(payload, spec, "NON_SCALAR_COMPARAND")
        assert "could never hold" in detail
        assert "holds in every state" not in detail

    def test_an_inequality_against_a_container_says_it_holds_everywhere(
        self, payload, spec
    ) -> None:
        # ``ne`` against the same list is not a comparison that never holds; it
        # is one that holds for every record there will ever be, which is the
        # same vacuity from the other end and a different thing to go and fix.
        _none_clause(payload)["where"][0].update({"op": "ne", "value": ["OPEN"]})
        detail = _detail(payload, spec, "NON_SCALAR_COMPARAND")
        assert "holds in every state" in detail
        assert "could never hold" not in detail

    @pytest.mark.parametrize(
        ("op", "value"),
        (
            ("eq", "OPEN"),
            ("ne", "OPEN"),
            ("in", ["OPEN", "EXPIRED"]),
            ("not_in", ["OPEN"]),
        ),
    )
    def test_a_literal_that_is_one_value_is_still_comparable(
        self, payload, spec, op: str, value: Any
    ) -> None:
        # The limit of the rule. Every operator this class refuses a container
        # for keeps the comparison it is for.
        condition = _none_clause(payload)["where"][0]
        condition.update({"op": op, "value": value, "value_ref": None})
        assert _codes(payload, spec) == ()

    def test_an_ordering_comparison_against_a_number_is_still_comparable(
        self, payload, spec
    ) -> None:
        condition = _none_clause(payload)["where"][0]
        condition.update({"field": "amount_minor", "op": "gt", "value": 1000})
        assert _codes(payload, spec) == ()

    def test_a_yes_no_field_takes_a_yes_no_literal_and_not_a_number(
        self, payload, spec
    ) -> None:
        # ``True == 1`` in Python and the two are still not the same answer, so
        # tightening the literal check must not quietly let a count stand where
        # a flag is asked for.
        clause = _none_clause(payload)
        clause["collection"] = "cycles"
        clause["where"] = [
            {"field": "requires_payment", "op": "eq", "value": True, "value_ref": None}
        ]
        assert _codes(payload, spec) == ()
        clause["where"][0]["value"] = 1
        assert _codes(payload, spec) == ("COMPARAND_TYPE_MISMATCH",)

    def test_a_field_the_operation_declares_as_nullable_may_still_be_null(
        self, payload, spec
    ) -> None:
        # ``cycles.parent_cycle_id`` is ``str | None``: null is a value those
        # records genuinely hold, so a membership list that names it states a
        # rule some state decides and is not a shape this class refuses.
        clause = _none_clause(payload)
        clause["collection"] = "cycles"
        clause["where"] = [
            {
                "field": "parent_cycle_id",
                "op": "in",
                "value": [None, "CYC-1"],
                "value_ref": None,
            }
        ]
        assert _codes(payload, spec) == ()

    def test_a_leaf_the_operation_leaves_untyped_is_not_over_refused(
        self, payload, spec
    ) -> None:
        # An ``Any`` leaf is a value like any other and a kind nobody derived is
        # not a kind that disagrees; only the *shape* of the literal is at issue
        # here, so a name compared against one is still a comparison a control
        # may make — and a list compared against one still is not.
        scalar = _scalar_clause(payload)
        scalar.update(
            {
                "binding": None,
                "binding_field": None,
                "path": "state.payment.status",
                "op": "eq",
                "value": "SETTLED",
                "value_ref": None,
            }
        )
        assert _codes(payload, spec) == ()
        scalar["value"] = ["SETTLED"]
        assert _codes(payload, spec) == ("NON_SCALAR_COMPARAND",)


class TestABindingIsSingularAndIsDeclaredBeforeItIsRead:
    def test_a_quantifier_that_selects_nothing_may_not_bind(self, payload, spec) -> None:
        quantified = next(
            clause
            for clause in _clauses(payload)
            if clause["kind"] == "quantified" and clause["quantifier"] == "none"
        )
        quantified["bind"] = "the_approval_that_is_not_there"
        assert "BINDING_QUANTIFIER_NOT_SINGULAR" in _codes(payload, spec)

    def test_an_existential_quantifier_may_not_bind(self, payload, spec) -> None:
        # ``exists`` holds for one match or for nine; a binding taken from it is
        # "whichever record sorted first", which is not a semantic selection.
        quantified = next(
            clause for clause in _clauses(payload) if clause["kind"] == "quantified"
        )
        quantified["quantifier"] = "exists"
        assert "BINDING_QUANTIFIER_NOT_SINGULAR" in _codes(payload, spec)

    def test_a_consumer_that_runs_before_its_binder(self, payload, spec) -> None:
        clauses = _clauses(payload)
        scalar = next(clause for clause in clauses if clause["kind"] == "scalar")
        clauses.remove(scalar)
        clauses.insert(0, scalar)
        assert "BINDING_USED_BEFORE_BOUND" in _codes(payload, spec)

    def test_two_clauses_that_bind_the_same_name(self, payload, spec) -> None:
        clauses = _clauses(payload)
        binder = next(clause for clause in clauses if clause.get("bind"))
        twin = copy.deepcopy(binder)
        clauses.insert(1, twin)
        assert "DUPLICATE_BINDING" in _codes(payload, spec)


class TestADuplicateKeyIsAmbiguityNotLastWins:
    def test_a_duplicate_top_level_key(self, tmp_path: Path) -> None:
        text = match_manifest_path().read_text(encoding="utf-8")
        path = tmp_path / "control.yaml"
        path.write_text(f"{text}\ncontrol_id: a_second_answer\n", encoding="utf-8")
        with pytest.raises(MatchManifestError, match="twice"):
            load_match_manifest(path)

    def test_a_duplicate_key_nested_inside_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "control.yaml"
        path.write_text(
            "schema_version: 1\n"
            "denominator:\n"
            "  semantic_scenarios: 1\n"
            "  semantic_scenarios: 3\n",
            encoding="utf-8",
        )
        with pytest.raises(MatchManifestError, match="twice"):
            load_match_manifest(path)


class TestAReviewIsBoundToWhatItReviewed:
    """An independent review names the document it read, and is checked against it.

    The review target is deliberately *not* the manifest digest: the manifest
    digest covers the provenance block, and the provenance block is where the
    review lives, so a reviewer would have to sign a digest that changes when
    their signature is added to it. The target is the control's semantic content
    — everything except who authored it and who reviewed it — so it can be
    computed by the reviewer before the review exists, and cannot be moved by
    editing the review.
    """

    @staticmethod
    def _reviewed(payload: dict[str, Any], digest: str) -> dict[str, Any]:
        payload["provenance"] = {
            "authored_by": "operatebench_project",
            "independence": INDEPENDENCE_INDEPENDENT,
            "evidence_class": "INDEPENDENT_ADMISSIBLE_OUTCOME_ORACLE",
            "review_status": "INDEPENDENTLY_REVIEWED",
            "independent_review": {
                "reviewer_id": "reviewer_example",
                "reviewer_affiliation": "independent_example",
                "reviewed_at": "2030-01-01T00:00:00Z",
                "attestation_digest_sha256": "a" * 64,
                "reviewed_manifest_digest_sha256": digest,
            },
        }
        return payload

    def test_a_review_naming_a_digest_of_its_own_choosing_is_refused(
        self, payload
    ) -> None:
        self._reviewed(payload, "b" * 64)
        with pytest.raises(MatchManifestError, match="review target"):
            build_match_manifest(payload, "sham")

    def test_a_review_naming_the_computed_target_is_independent(
        self, payload, spec
    ) -> None:
        self._reviewed(payload, review_target_digest(payload, "sham"))
        manifest = build_match_manifest(payload, "sham")
        assert manifest.is_independent is True
        assert manifest.review_target_digest_sha256 == review_target_digest(
            payload, "sham"
        )
        assert validate_match(manifest, spec).is_independent is True

    def test_a_semantic_edit_after_the_review_invalidates_it(self, payload) -> None:
        self._reviewed(payload, review_target_digest(payload, "sham"))
        # One member of one admissible set, widened after the reviewer read it.
        _ceiling(payload)["minimum"] = 0
        with pytest.raises(MatchManifestError, match="review target"):
            build_match_manifest(payload, "sham")

    def test_a_predicate_edit_after_the_review_invalidates_it(self, payload) -> None:
        self._reviewed(payload, review_target_digest(payload, "sham"))
        quantified = next(
            clause for clause in _clauses(payload) if clause["kind"] == "quantified"
        )
        quantified["where"][0]["value"] = "SUPERSEDED"
        with pytest.raises(MatchManifestError, match="review target"):
            build_match_manifest(payload, "sham")

    def test_editing_the_review_does_not_move_what_was_reviewed(self, payload) -> None:
        target = review_target_digest(payload, "sham")
        self._reviewed(payload, target)
        first = build_match_manifest(payload, "sham")
        review = payload["provenance"]["independent_review"]
        review["reviewer_id"] = "reviewer_example_2"
        review["reviewed_at"] = "2031-06-30T12:00:00Z"
        review["attestation_digest_sha256"] = "c" * 64
        payload["provenance"]["review_status"] = "INDEPENDENTLY_REVIEWED_AGAIN"
        second = build_match_manifest(payload, "sham")
        assert second.review_target_digest_sha256 == target
        assert first.review_target_digest_sha256 == target
        assert second.is_independent is True
        # The document identity *did* move, which is the point of keeping the two
        # digests apart: one says which bytes, the other says which semantics.
        assert second.manifest_digest_sha256 != first.manifest_digest_sha256

    def test_the_target_is_not_the_manifest_digest(self, payload) -> None:
        manifest = build_match_manifest(payload, "sham")
        assert manifest.review_target_digest_sha256 != manifest.manifest_digest_sha256, (
            "a review target that is the whole document has to include the review"
        )

    def test_the_draft_states_a_target_a_reviewer_could_sign(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.is_independent is False
        assert len(manifest.review_target_digest_sha256) == 64
        assert manifest.identity()["review_target_digest_sha256"] == (
            manifest.review_target_digest_sha256
        )


class TestTheGradedUnitIsMachineReadable:
    def test_the_manifest_identity_states_one_graded_outcome(self) -> None:
        identity = load_match_manifest(match_manifest_path()).identity()
        assert identity["graded_unit"] == GRADED_UNIT
        assert identity["graded_outcomes_per_decision_point"] == 1
        assert GRADED_OUTCOMES_PER_DECISION_POINT == 1

    def test_the_identity_keeps_the_denominator_and_the_draft_posture(self) -> None:
        identity = load_match_manifest(match_manifest_path()).identity()
        assert identity["semantic_scenarios"] == 1
        assert identity["decision_points_are_independent_semantic_samples"] is False
        assert identity["independence"] == INDEPENDENCE_DRAFT
        assert identity["is_independent"] is False

    def test_the_match_report_states_it_too(self, spec) -> None:
        report = validate_match(load_match_manifest(match_manifest_path()), spec)
        payload = report.as_dict()
        assert payload["graded_unit"] == GRADED_UNIT
        assert payload["graded_outcomes_per_decision_point"] == 1
        assert payload["shared_estimand"] == SHARED_ESTIMAND
        assert payload["is_independent"] is False


class TestTheGateRefusesRatherThanReports:
    def test_require_match_raises_on_a_broken_match(self, payload, spec) -> None:
        payload["operation_id"] = "lettings_arrears_synthetic_v1"
        manifest = build_match_manifest(payload, "sham")
        with pytest.raises(MatchManifestError, match="OPERATION_IDENTITY_MISMATCH"):
            require_match(manifest, spec)

    def test_require_match_returns_the_report_on_a_good_test_only_match(
        self, payload, spec
    ) -> None:
        manifest = build_match_manifest(payload, "test-only")
        assert require_match(manifest, spec).ok

    def test_every_live_action_the_test_only_grammar_offers_is_schema_matched(
        self, payload, spec
    ) -> None:
        manifest = build_match_manifest(payload, "test-only")
        for action_type, schema in manifest.matched_surface.action_schemas.items():
            required, optional = MAINTENANCE_ACTION_PAYLOAD_SCHEMAS[action_type]
            assert schema.required == dict(required)
            assert schema.optional == dict(optional)


class TestTheDraftRetainsItsSupersededOperationIdentity:
    """The immutable draft's identity, held apart from the current operation.

    A control that says which operation version it was written for, and is not
    checked against the version that is running, is a control that can drift
    silently. These tests state the current numbers so a future move of the
    operation shows up here as a named failure rather than as a manifest that
    quietly grades the wrong surface.
    """

    def test_the_shipped_draft_is_the_v0_3_draft(self) -> None:
        assert CONTROL_MANIFEST_NAME == "matched_decision_points_draft_v0_3.yaml"
        assert match_manifest_path().name == CONTROL_MANIFEST_NAME
        assert match_manifest_path().is_file()

    def test_no_superseded_draft_is_left_where_the_loader_could_read_it(self) -> None:
        superseded = (
            match_manifest_path().parent / "matched_decision_points_draft_v0_1.yaml"
        )
        assert not superseded.exists(), (
            "a manifest authored against operation 0.1 sitting beside the current "
            "one is a document a reader can mistake for the current control"
        )

    def test_it_states_the_historical_operation_version_not_the_live_one(
        self, spec
    ) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.authored_against_operation_version == "0.4.0"
        assert spec.operation_version == "0.6.0"
        assert manifest.authored_against_operation_version != spec.operation_version
        assert manifest.control_version == "0.3.0-draft"

    def test_its_identity_does_not_still_name_the_version_it_left_behind(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert "v0_1" not in manifest.control_id
        assert "v0_2" not in manifest.control_id
        assert manifest.control_id == "matched_decision_point_control_maintenance_v0_3"

    def test_its_projection_is_the_live_one_field_for_field_in_order(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        assert manifest.matched_surface.observation_projection_fields == (
            observation_projection_fields()
        )
        assert manifest.matched_surface.observation_projection_fields == (
            "now",
            "operation_id",
            "operation_instance_id",
            "invocation_index",
            "turn_index",
            "policy",
            "actors",
            "message_fixture_ids",
            "action_schemas",
            "wake_event_types",
            "last_rejection",
            "phase",
            "event",
            "retrieval",
        )

    def test_its_historical_action_schemas_remain_the_v0_4_ones(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        declared = {
            name: (schema.required, schema.optional)
            for name, schema in manifest.matched_surface.action_schemas.items()
        }
        live = {
            name: (dict(required), dict(optional))
            for name, (required, optional) in MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.items()
        }
        assert declared != live
        assert declared == {
            "request_supplier_visit": (
                {"visit_type": "string", "cycle_id": "string"},
                {"scope_digest": "optional_string"},
            ),
            "request_approval": (
                {
                    "quote_id": "string",
                    "quote_version": "integer",
                    "amount_minor": "integer",
                    "currency": "string",
                    "scope_digest": "string",
                    "cycle_id": "string",
                    "deadline_after_minutes": "integer",
                },
                {},
            ),
            "send_message": (
                {"recipient_actor_id": "string", "message_fixture_id": "string"},
                {"correlation_id": "optional_string"},
            ),
            "authorise_supplier_work": (
                {
                    "quote_id": "string",
                    "quote_version": "integer",
                    "approval_checkpoint_id": "string",
                    "cycle_id": "string",
                },
                {},
            ),
            "request_invoice_validation": ({"invoice_id": "string"}, {}),
            "request_payment": (
                {
                    "invoice_id": "string",
                    "amount_minor": "integer",
                    "currency": "string",
                },
                {},
            ),
            "request_exception_resolution": (
                {"exception_type": "string", "deadline_after_minutes": "integer"},
                {},
            ),
        }


class TestTheControlDoesNotGradeIdentitiesTheAgentCannotSupply:
    """Identifiers the environment allocates are not the agent's answer.

    Operation 0.2 took ``visit_id``, the approval and exception ``checkpoint_id``
    and ``payment_request_id`` out of the agent's payloads: an agent that could
    name a record's identity could desynchronise every authored event that
    answers it. A control that still constrained them would be grading a field
    nobody submits — vacuously, and in a way that reads as coverage.
    """

    ENVIRONMENT_OWNED = ("visit_id", "checkpoint_id", "payment_request_id")

    def test_no_declared_action_offers_an_environment_owned_identifier(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        for action_type, schema in manifest.matched_surface.action_schemas.items():
            offered = set(schema.required) | set(schema.optional)
            assert offered.isdisjoint(self.ENVIRONMENT_OWNED), (action_type, offered)

    def test_no_admissible_outcome_constrains_an_environment_owned_identifier(
        self,
    ) -> None:
        manifest = load_match_manifest(match_manifest_path())
        for point in manifest.decision_points:
            for outcome in point.admissible_outcomes:
                named = set(outcome.constraints)
                assert named.isdisjoint(self.ENVIRONMENT_OWNED), (
                    point.decision_point_id,
                    outcome.outcome_id,
                    sorted(named),
                )

    def test_the_agent_controlled_fields_are_still_constrained(self) -> None:
        manifest = load_match_manifest(match_manifest_path())
        point = manifest.decision_point("over_threshold_quote_with_no_approval_on_record")
        (outcome,) = point.admissible_outcomes
        assert set(outcome.constraints) == {
            "quote_id",
            "quote_version",
            "amount_minor",
            "currency",
            "scope_digest",
            "cycle_id",
            "deadline_after_minutes",
        }
        bound = {
            name
            for name, constraint in outcome.constraints.items()
            if constraint.kind == "bound_field"
        }
        assert bound == {
            "quote_id",
            "quote_version",
            "amount_minor",
            "currency",
            "scope_digest",
            "cycle_id",
        }

    def test_an_environment_owned_identifier_is_not_smuggled_back_as_a_free_one(
        self,
    ) -> None:
        raw = yaml.safe_load(match_manifest_path().read_text(encoding="utf-8"))
        for point in raw["decision_points"]:
            for outcome in point["admissible_outcomes"]:
                for name, constraint in outcome["outcome_constraints"].items():
                    assert not (
                        name in self.ENVIRONMENT_OWNED
                        and constraint.get("kind") == "free_identifier"
                    ), (point["decision_point_id"], outcome["outcome_id"], name)
