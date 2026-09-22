"""Phase 2, part 5: the matched-control seam follows the surface that moved.

The shipped draft compared a declared ``observation_state_roots`` against the
live projection. That projection no longer exists on the agent path, so the seam
has to name what does: the published retrieval catalogue, the batch budget, and
each outcome's read requirements. One seam, declared and validated now, so that
Static / Boundary / Time-Removed / Full can later be *the same surface minus one
difference*. No arm is built here.
"""

from __future__ import annotations

import pytest

from operatebench import controls as controls_module
from operatebench import decision_points as dp
from operatebench.controls import (
    CONTROL_MANIFEST_NAME,
    MatchedSurface,
    load_match_manifest,
    match_manifest_path,
    validate_match,
)
from operatebench.core.read_contract import (
    READ_OPTIONAL,
    READ_REQUIRED,
    ActionEvidenceContract,
)
from operatebench.domains.lettings.maintenance.spec import load_spec

SPEC = "examples/operatebench/maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


class TestTheSurfaceNamesReads:
    def test_the_matched_surface_carries_the_retrieval_seam(self) -> None:
        fields = MatchedSurface.__dataclass_fields__
        assert "retrieval_catalogue" in fields
        assert "retrieval_batch_budget" in fields
        assert "action_read_requirements" in fields
        assert "observation_state_roots" not in fields

    def test_the_old_live_derivation_is_gone(self) -> None:
        assert not hasattr(dp, "observation_state_roots")

    def test_decision_points_publish_the_catalogue(self) -> None:
        assert dp.retrieval_catalogue_surface()
        assert dp.action_read_requirement_surface()

    def test_decision_points_include_required_and_optional_read_affordances(
        self, monkeypatch
    ) -> None:
        contract = ActionEvidenceContract(
            {
                "request_supplier_visit": {
                    "reads": {
                        "get_case_record": READ_REQUIRED,
                        "list_quotes": READ_OPTIONAL,
                    }
                }
            }
        )
        monkeypatch.setattr(dp, "MAINTENANCE_READ_CONTRACT", contract)

        assert dp.action_read_requirement_surface() == (
            "request_supplier_visit:get_case_record,list_quotes=optional",
        )

    def test_the_all_required_maintenance_surface_keeps_its_exact_codec(self) -> None:
        assert dp.action_read_requirement_surface() == (
            "authorise_supplier_work:get_case_record,list_checkpoints,list_quotes",
            "complete:get_case_record,list_billing,list_checkpoints,list_obligations",
            "request_approval:get_case_record,list_checkpoints,list_quotes",
            "request_exception_resolution:get_case_record,list_checkpoints,list_quotes",
            "request_invoice_validation:get_case_record,list_authoritative_records,list_billing",
            "request_payment:get_case_record,list_authoritative_records,list_billing",
            "request_supplier_visit:get_case_record",
            "send_message:get_case_record,list_communications",
        )

    def test_the_manifest_moved_to_the_next_draft(self) -> None:
        assert CONTROL_MANIFEST_NAME == "matched_decision_points_draft_v0_3.yaml"

    def test_the_shipped_draft_is_superseded_by_the_moved_action_surface(
        self, spec
    ) -> None:
        manifest = load_match_manifest(match_manifest_path())
        report = validate_match(manifest, spec)
        assert not report.ok
        assert [problem.code for problem in report.problems] == [
            "ACTION_READ_REQUIREMENT_MISMATCH",
            "ACTION_SCHEMA_MISMATCH",
            "OPERATION_IDENTITY_MISMATCH",
        ]

    def test_a_drifted_catalogue_is_a_named_problem(self, spec, monkeypatch) -> None:
        manifest = load_match_manifest(match_manifest_path())
        monkeypatch.setattr(
            controls_module,
            "retrieval_catalogue_surface",
            lambda: ("not_a_published_tool",),
        )
        report = validate_match(manifest, spec)
        assert not report.ok
