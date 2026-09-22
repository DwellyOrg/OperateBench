"""Strict loading and deterministic identity for the Maintenance OperationSpec.

Everything a spec claims is checked when it loads, not when a run trips over it:
an unknown field, an unknown event type, an actor emitting an event its
authority does not cover, a duplicate identity or a malformed instant are all
refused by name at the door. A benchmark artefact that only fails halfway
through episode seven is not a frozen artefact.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from operatebench.core.errors import (
    AuthorityError,
    DuplicateIdentityError,
    MalformedTimestampError,
    SpecFormatError,
    SpecSchemaError,
    UnknownFieldError,
    UnknownTypeError,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_EVENT_TYPES,
    POLICY_FIELDS,
    OperationSpec,
    PolicySpec,
    load_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


def raw_spec() -> dict[str, Any]:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def build(raw: dict[str, Any]) -> OperationSpec:
    return OperationSpec.from_mapping(raw, source="<fixture under test>")


class TestShippedFixture:
    def test_the_shipped_fixture_loads(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.operation_id == "lettings_maintenance_synthetic_v1"
        assert spec.operation_type == "lettings.maintenance.synthetic"
        assert spec.privacy_status == "SYNTHETIC_ONLY"

    def test_the_three_variants_cluster_under_one_semantic_scenario(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.scenario_ids == ("V1", "V2", "V3")
        assert {
            spec.scenario(name).semantic_scenario_id for name in spec.scenario_ids
        } == {"maintenance_recurring_leak_synthetic_v1"}

    def test_authored_events_get_a_stable_sequence_from_authoring_order(self) -> None:
        scenario = load_spec(FIXTURE).scenario("V1")
        sequences = [event.sequence for event in scenario.events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)

    def test_hidden_state_is_declared_and_non_empty(self) -> None:
        assert "root_cause_fixture_id" in load_spec(FIXTURE).hidden_state

    def test_expected_terminals_differ_across_variants(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.scenario("V1").expected_terminal == "completed_successfully"
        assert spec.scenario("V2").expected_terminal == "transferred_to_human_ownership"
        assert spec.scenario("V3").expected_terminal == "transferred_to_human_ownership"


class TestSpecIdentity:
    def test_identity_is_deterministic_for_identical_content(self) -> None:
        assert (
            build(raw_spec()).spec_digest_sha256 == build(raw_spec()).spec_digest_sha256
        )

    def test_identity_ignores_authoring_key_order(self) -> None:
        reordered = dict(reversed(list(raw_spec().items())))
        assert build(reordered).spec_digest_sha256 == build(raw_spec()).spec_digest_sha256

    def test_identity_changes_when_an_authored_amount_changes(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][4]["payload"]["amount_minor"] = 64001
        assert build(mutated).spec_digest_sha256 != build(raw_spec()).spec_digest_sha256

    def test_identity_changes_when_a_policy_threshold_changes(self) -> None:
        mutated = raw_spec()
        mutated["policy"]["approval_threshold_minor"] = 25001
        assert build(mutated).spec_digest_sha256 != build(raw_spec()).spec_digest_sha256

    @pytest.mark.parametrize("field", POLICY_FIELDS)
    def test_every_policy_field_reaches_the_identity(self, field: str) -> None:
        """Change any one policy field and the digest must move.

        Checking one threshold proves one threshold. This asks the question of
        the whole policy, one field at a time, because the way the digest went
        blind was not a broken hash — it was a hand-written projection that
        listed eleven of twelve fields, so a spec with a different
        ``max_retrieval_batches_per_invocation`` hashed to the identical value
        and one published digest named two different executable policies. A
        field that the runtime reads and the identity does not is the defect
        this parametrisation exists to fail on, for whichever field acquires it
        next.
        """
        original = raw_spec()
        mutated = raw_spec()
        current = mutated["policy"][field]
        mutated["policy"][field] = (
            f"{current}_ALTERED" if isinstance(current, str) else current + 1
        )
        assert mutated["policy"][field] != original["policy"][field]
        assert build(mutated).spec_digest_sha256 != build(original).spec_digest_sha256

    def test_the_policy_projection_names_exactly_the_executable_policy(self) -> None:
        """The projected keys are the dataclass's fields — no more, no fewer.

        The behavioural parametrisation above is the one that matters, but it
        can only ask about fields the fixture happens to author. This asks the
        structural question directly, so a field that is added to
        :class:`PolicySpec` and never written into the shipped fixture is still
        covered.

        ``fields()`` rather than ``__dataclass_fields__``, matching the module
        under test: the latter also reports ``ClassVar`` and ``InitVar``
        pseudo-fields, and asserting against those would demand the identity
        cover values that are not instance data.
        """
        spec = load_spec(FIXTURE)
        executable = {entry.name for entry in fields(PolicySpec)}
        assert set(spec.semantic_payload()["policy"]) == executable
        assert set(POLICY_FIELDS) == executable


class TestStrictness:
    def test_unknown_top_level_field_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["leaderboard_weighting"] = {"speed": 0.4}
        with pytest.raises(UnknownFieldError) as excinfo:
            build(mutated)
        assert "leaderboard_weighting" in str(excinfo.value)

    def test_unknown_scenario_field_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["difficulty"] = "hard"
        with pytest.raises(UnknownFieldError):
            build(mutated)

    def test_unknown_event_field_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][0]["priority"] = 3
        with pytest.raises(UnknownFieldError):
            build(mutated)

    def test_unknown_event_type_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][0]["type"] = "landlord_changed_mind"
        with pytest.raises(UnknownTypeError) as excinfo:
            build(mutated)
        assert "landlord_changed_mind" in str(excinfo.value)

    def test_unknown_actor_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][0]["actor"] = "customer_9"
        with pytest.raises(UnknownTypeError):
            build(mutated)

    def test_actor_without_the_required_authority_is_refused(self) -> None:
        mutated = raw_spec()
        # The supplier cannot verify its own work: that authority belongs to the
        # authoritative maintenance system.
        mutated["scenarios"]["V1"]["events"][9]["actor"] = "supplier_1"
        with pytest.raises(AuthorityError) as excinfo:
            build(mutated)
        assert "verify_work" in str(excinfo.value)

    def test_duplicate_event_identity_within_a_scenario_is_refused(self) -> None:
        mutated = raw_spec()
        events = mutated["scenarios"]["V1"]["events"]
        events[1]["event_id"] = events[0]["event_id"]
        with pytest.raises(DuplicateIdentityError):
            build(mutated)

    def test_duplicate_actor_authority_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["actors"]["supplier_1"]["authority"].append("report_work")
        with pytest.raises(DuplicateIdentityError):
            build(mutated)

    def test_malformed_timestamp_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][0]["at"] = "3 March, 9am"
        with pytest.raises(MalformedTimestampError):
            build(mutated)

    def test_event_with_both_absolute_and_conditional_timing_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][0]["on_action"] = {
            "type": "request_supplier_visit"
        }
        with pytest.raises(SpecSchemaError):
            build(mutated)

    def test_event_with_no_timing_at_all_is_refused(self) -> None:
        mutated = raw_spec()
        del mutated["scenarios"]["V1"]["events"][0]["at"]
        with pytest.raises(SpecSchemaError):
            build(mutated)

    def test_conditional_event_on_an_unknown_action_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][1]["on_action"]["type"] = "bribe_supplier"
        with pytest.raises(UnknownTypeError):
            build(mutated)

    def test_unknown_message_fixture_in_dispatch_failures_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V2"]["dispatch_failures"] = ["msg_unknown"]
        with pytest.raises(UnknownTypeError):
            build(mutated)

    def test_scenario_without_events_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"] = []
        with pytest.raises(SpecSchemaError):
            build(mutated)

    def test_negative_delay_is_refused(self) -> None:
        mutated = raw_spec()
        mutated["scenarios"]["V1"]["events"][1]["delay_minutes"] = -30
        with pytest.raises(SpecSchemaError):
            build(mutated)

    def test_unknown_scenario_name_is_refused_by_name(self) -> None:
        with pytest.raises(UnknownTypeError) as excinfo:
            load_spec(FIXTURE).scenario("V9")
        assert "V9" in str(excinfo.value)

    def test_every_authored_event_type_is_a_known_maintenance_type(self) -> None:
        spec = load_spec(FIXTURE)
        authored = {
            event.event_type
            for name in spec.scenario_ids
            for event in spec.scenario(name).events
        }
        assert authored <= set(MAINTENANCE_EVENT_TYPES)


class TestFileLevelFailures:
    def test_missing_file_is_a_named_domain_error(self, tmp_path: Path) -> None:
        with pytest.raises(SpecFormatError):
            load_spec(tmp_path / "absent.yaml")

    def test_non_mapping_document_is_a_named_domain_error(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(SpecFormatError):
            load_spec(path)

    def test_duplicate_yaml_key_is_a_named_domain_error(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.yaml"
        path.write_text("operation_id: a\noperation_id: b\n", encoding="utf-8")
        with pytest.raises(SpecFormatError):
            load_spec(path)

    def test_empty_document_is_a_named_domain_error(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(SpecFormatError):
            load_spec(path)


class TestPrivacy:
    def test_the_fixture_declares_synthetic_provenance(self) -> None:
        spec = load_spec(FIXTURE)
        assert spec.privacy_status == "SYNTHETIC_ONLY"
        assert "not dwelly policy" in spec.disclaimer.lower()
        assert "no production records" in spec.data_provenance.lower()

    def test_a_non_synthetic_privacy_status_is_refused(self) -> None:
        mutated = copy.deepcopy(raw_spec())
        mutated["privacy_status"] = "PRODUCTION_DERIVED"
        with pytest.raises(SpecSchemaError):
            build(mutated)
