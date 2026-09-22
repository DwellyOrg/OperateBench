"""Artefact contract v2: what it adds, and what it must not break.

Three obligations are asserted here.

**v1 is still honoured, and the limit of that is stated.** A byte-frozen v1
artefact, written by the contract as it stood before v2 existed, still
validates: this build reads a v1 document under the v1 field set, exactly, and
adding fields to an artefact is not a licence to stop reading the records
written before they existed. Its bytes are pinned by digest, so a change that
would have silently rewritten the fixture fails instead.

What that fixture cannot do under this build is *replay*. It records a run of an
operation whose authored semantics have since moved — spec digest ``905d0445…``
against this build's ``c2c5ab09…`` — so re-executing it would compare two
different operations. That is a named incompatibility, asserted as one below,
and it is the only honest answer available: the alternatives are rewriting the
frozen evidence or narrowing the comparison until it agrees. The v1 read,
replay and tamper-detection path is therefore exercised against a v1-shaped
artefact derived from a run this build can still execute — see
:mod:`tests.artifact_shapes` — so "the v1 contract still works" stays an
executable claim rather than a historical one.

**v2 fails closed.** A v1 document that carries a v2 field is refused, a v2
document missing one is refused, and neither version tolerates a field this
build does not know. There is no "read what you recognise and shrug at the
rest": that is how two records that mean different things end up compared.

**Every field is compared by exactly one section.** Replay reports section by
section, and the claim "the replay compared everything" is only checkable if
every top-level field belongs to exactly one of them. The partition is
recomputed here from the tables themselves rather than restated, so a field
added to the artefact and forgotten in the section table fails this suite.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.transport import (
    FAULTS,
    OUTCOME_SOURCES,
    RETRY_COUNT,
)
from operatebench.artifact import (
    _FIELDS_BY_VERSION,
    _ROW_SHAPES_BY_VERSION,
    _SECTIONS_BY_VERSION,
    ARTIFACT_VERSION,
    ARTIFACT_VERSION_V1,
    ARTIFACT_VERSION_V2,
    ARTIFACT_VERSION_V3,
    ARTIFACT_VERSION_V4,
    ARTIFACT_VERSION_V5,
    ARTIFACT_VERSION_V6,
    ARTIFACT_VERSION_V7,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    REPRODUCTION_PLAYBACK,
    REPRODUCTION_RERUN,
    SUPPORTED_ARTIFACT_VERSIONS,
    build_artifact,
    content_digest,
    project_trajectory,
    read_artifact,
    replay_artifact,
    row_shape_coverage_problem,
    section_coverage_problem,
    validate_artifact,
    write_artifact,
)
from operatebench.core.errors import ArtifactError
from operatebench.core.ledger import RETRIEVAL_RECORD_TYPES
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import run_episode
from tests.artifact_shapes import as_version_shape
from tests.model_transport import TEST_MODEL
from tests.provider_evidence_runs import bound_model_artifact

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "examples/operatebench/maintenance_v0_1.yaml"

#: A v1 artefact, written by the contract as it stood before v2, and frozen.
FROZEN_V1 = Path(__file__).resolve().parent / "fixtures/artifact_v1_reference_V1.json"

#: The bytes of that fixture. Pinned so "the frozen v1 artefact still validates"
#: cannot be made true by quietly rewriting the artefact it is frozen against.
FROZEN_V1_DIGEST = "fb79884fb499043ff13d4cbd0ed2022f38a154565e156cbe0c874cfd7531af36"

#: The operation the frozen artefact was produced from, by value. This build
#: ships a later authored semantics for the same operation, and the difference
#: between these two digests is exactly why replaying that artefact here is a
#: refusal rather than a comparison.
FROZEN_V1_SPEC_DIGEST = "905d04454df05bd678cd2acc8025eb7eb785d227020c7baeba809b379d71c292"

MODEL_AGENT_ID = "model_reference"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def current_artifact(spec: OperationSpec) -> dict[str, Any]:
    return build_artifact(run_episode(spec, "V1", "reference"))


@pytest.fixture(scope="module")
def model_artifact(spec: OperationSpec, tmp_path_factory: Any) -> dict[str, Any]:
    """A model artefact bound to the journal that witnessed it.

    Contract 7 admits no other kind: a model run with no complete scored ledger
    is refused at ``build_artifact``. So this is a real recorded run over the
    real SDK and an in-process mock transport, and every edit below is made to a
    document that was actually produced.
    """
    return bound_model_artifact(tmp_path_factory, spec, agent_id=MODEL_AGENT_ID)


def tampered(payload: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    body = json.loads(json.dumps(payload))
    for name, value in changes.items():
        if value is ...:
            body.pop(name, None)
        else:
            body[name] = value
    return body


# ----------------------------------------------------------- v1 compatibility


class TestTheFrozenV1Artefact:
    def test_the_fixture_is_byte_frozen(self) -> None:
        digest = hashlib.sha256(FROZEN_V1.read_bytes()).hexdigest()
        assert digest == FROZEN_V1_DIGEST, (
            "the frozen v1 artefact is the evidence that v2 did not break v1. "
            "Rewriting it to make a test pass removes the evidence; if this run "
            "genuinely changed, that is a compatibility break to state, not a "
            "digest to update"
        )

    def test_it_still_validates_under_the_v1_contract(self) -> None:
        body = read_artifact(FROZEN_V1)
        assert body["artifact_version"] == ARTIFACT_VERSION_V1
        assert set(body) == set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V1])
        for name in ("decisions", "decisions_digest_sha256", "agent_execution"):
            assert name not in body, (
                "a v1 record does not acquire v2 fields by being read by a v2 build"
            )

    def test_replaying_it_here_is_a_named_incompatibility(
        self, spec: OperationSpec
    ) -> None:
        """The one honest outcome, and it is named rather than approximated.

        Two independent reasons this record is not replayable here, and the
        wider one is the one that answers first. It was written under artefact
        contract 1, and this build reproduces contract 4 only: the observation
        projection a decision is bound to and the body of a declared wait have
        both moved, so no rerun under this build produces what this record
        carries — and that is true whatever spec it is offered. It *also*
        records the operation as authored at ``operation_version 0.1.0``, which
        the shipped fixture has moved past.

        Both are asserted: the refusal names the contract, and the record's own
        operation digest is still the older one. A future change that made this
        pass by widening the reproducer would fail here.
        """
        recorded = read_artifact(FROZEN_V1)
        assert recorded["operation"]["spec_digest_sha256"] == FROZEN_V1_SPEC_DIGEST
        assert spec.spec_digest_sha256 != FROZEN_V1_SPEC_DIGEST

        with pytest.raises(ArtifactError) as raised:
            replay_artifact(spec, recorded)

        message = str(raised.value)
        assert "artefact contract 1" in message
        assert "reproduces contract(s) [8]" in message
        assert "compare two different contracts" in message

    def test_the_incompatibility_is_reached_before_anything_is_executed(
        self, spec: OperationSpec
    ) -> None:
        # Tampering with a field a replay would compare changes nothing: the
        # contract binding is checked first, so this artefact never reaches a
        # comparison that could report a difference — or agree about one.
        body = tampered(read_artifact(FROZEN_V1), simulated_minutes=1)
        with pytest.raises(ArtifactError, match="artefact contract 1"):
            replay_artifact(spec, body)

    def test_a_v1_row_is_still_held_to_the_v1_row_bodies(self) -> None:
        # The row-level half of the rule the field set states at the top level:
        # a v1 document is held to the bodies v1 was written under. One carrying
        # contract 3's ``bindings`` is refused rather than read as a v3 row
        # inside a v1 file, and one whose declared wait has dropped the
        # reachability flag contract 4 retracted is refused too — a v1 record
        # does not become a v4 record by being read by a v4 build.
        for row_type, edit in (
            ("effect_accepted", lambda row: row.update({"bindings": {"a": "b"}})),
            ("wait_declared", lambda row: row.pop("reachable", None)),
        ):
            body = tampered(read_artifact(FROZEN_V1))
            row = next(
                candidate
                for candidate in body["trajectory"]
                if (candidate["record_type"] == row_type and "reason" in candidate)
                or candidate["record_type"] == row_type == "effect_accepted"
            )
            edit(row)
            body["trajectory_digest_sha256"] = content_digest(body["trajectory"])
            with pytest.raises(ArtifactError, match="none of the shapes contract 1"):
                validate_artifact(body)

    def test_a_v1_artefact_carrying_a_v2_field_is_refused(self) -> None:
        body = read_artifact(FROZEN_V1)
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(tampered(body, decisions=[]))


class TestOlderContractsAreReadAndNotReproduced:
    """Where compatibility with contracts 1 to 3 now ends, and why exactly there.

    Until contract 4 there were two halves to this: the frozen fixture proved
    the reader still read a real v1 record, and a v1-shaped projection of a
    current run proved the v1 contract still *ran*. The second half is gone, and
    it is gone for a reason worth stating rather than deleting quietly.

    A current run cannot be dressed as an older document. Contract 4 removed
    ``reachable`` from every declared wait, because it asserted whether anything
    queued could deliver the wake condition — authored-future truth this build
    no longer computes. Projecting a current run onto contract 3 or earlier would
    mean writing that flag back, and there is nothing to write it back *from*:
    the value would be manufactured by the projection. The projection therefore
    only ever drops, the resulting document is not a valid older document, and
    the older reader says so by name.

    So the honest position is the one asserted here: those contracts are read,
    exactly as they were written, and they are not reproduced.
    """

    def test_this_build_reads_more_contracts_than_it_reproduces(self) -> None:
        assert REPRODUCIBLE_ARTIFACT_VERSIONS == (ARTIFACT_VERSION,)
        assert set(REPRODUCIBLE_ARTIFACT_VERSIONS) < set(SUPPORTED_ARTIFACT_VERSIONS)
        assert set(SUPPORTED_ARTIFACT_VERSIONS) == {1, 2, 3, 4, 5, 6, 7, ARTIFACT_VERSION}

    @pytest.mark.parametrize(
        "version",
        [
            ARTIFACT_VERSION_V1,
            ARTIFACT_VERSION_V2,
            ARTIFACT_VERSION_V3,
            ARTIFACT_VERSION_V4,
        ],
    )
    def test_a_current_run_cannot_be_dressed_as_an_older_document(
        self, current_artifact: Mapping[str, Any], version: int
    ) -> None:
        older = as_version_shape(current_artifact, version)
        assert older["artifact_version"] == version
        assert set(older) == set(_FIELDS_BY_VERSION[version])
        # It is refused on a row every older contract's runtime could not have
        # written. Reading became an explicit, recorded act at contract 5, so a
        # current run carries provenance rows no earlier runtime produced, and
        # relabelling one is refused by name rather than half-read.
        with pytest.raises(ArtifactError) as raised:
            validate_artifact(older)
        message = str(raised.value)
        # Contract 6 added two fields to the invocation row, so an older label
        # is refused there first — before the reader ever reaches the provenance
        # rows contract 5 added. Either refusal is the same statement: this
        # document was not written by the contract it claims.
        assert "agent_invoked" in message or "derived_context_invalidated" in message

    @pytest.mark.parametrize(
        "version", [ARTIFACT_VERSION_V1, ARTIFACT_VERSION_V2, ARTIFACT_VERSION_V3]
    )
    def test_replaying_an_older_contract_is_refused_before_anything_runs(
        self, spec: OperationSpec, current_artifact: Mapping[str, Any], version: int
    ) -> None:
        # Changing only the version does not turn a v4 document into an older
        # document: its v4-only fields remain. The strict reader refuses that
        # incoherent shape before replay can execute anything.
        with pytest.raises(ArtifactError) as raised:
            replay_artifact(spec, tampered(current_artifact, artifact_version=version))
        assert "unknown field(s)" in str(raised.value)
        assert "operation_instance_id" in str(raised.value)


# ------------------------------------------------------ what v3 requires


class TestTheV3RowContract:
    """v3's change is one row body, and it is required rather than tolerated."""

    def test_every_accepted_effect_carries_its_bindings(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        accepted = [
            row
            for row in current_artifact["trajectory"]
            if row["record_type"] == "effect_accepted"
        ]
        assert accepted
        for row in accepted:
            assert isinstance(row["bindings"], dict)
            for name, value in row["bindings"].items():
                assert isinstance(name, str) and isinstance(value, str)

    def test_a_v3_record_whose_effect_drops_its_bindings_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact)
        effect = next(
            row for row in body["trajectory"] if row["record_type"] == "effect_accepted"
        )
        effect.pop("bindings")
        body["trajectory_digest_sha256"] = content_digest(body["trajectory"])
        with pytest.raises(
            ArtifactError, match=f"none of the shapes contract {ARTIFACT_VERSION}"
        ):
            validate_artifact(body)

    def test_a_binding_that_is_not_a_string_identity_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        for forged in (["invoice_1"], 7, None, {"nested": "identity"}):
            body = tampered(current_artifact)
            effect = next(
                row
                for row in body["trajectory"]
                if row["record_type"] == "effect_accepted"
            )
            effect["bindings"] = {"invoice_id": forged}
            body["trajectory_digest_sha256"] = content_digest(body["trajectory"])
            with pytest.raises(ArtifactError):
                validate_artifact(body)

    def test_a_bindings_field_that_is_not_a_mapping_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact)
        effect = next(
            row for row in body["trajectory"] if row["record_type"] == "effect_accepted"
        )
        effect["bindings"] = ["invoice_id", "invoice_1"]
        body["trajectory_digest_sha256"] = content_digest(body["trajectory"])
        with pytest.raises(ArtifactError, match="must be a JSON object"):
            validate_artifact(body)

    def test_every_supported_version_describes_every_record_type(self) -> None:
        assert row_shape_coverage_problem() is None
        assert set(_ROW_SHAPES_BY_VERSION) == set(SUPPORTED_ARTIFACT_VERSIONS)

    def test_each_contract_moved_exactly_one_row_body(self) -> None:
        current = _ROW_SHAPES_BY_VERSION[ARTIFACT_VERSION]
        fourth = _ROW_SHAPES_BY_VERSION[ARTIFACT_VERSION_V4]
        third = _ROW_SHAPES_BY_VERSION[ARTIFACT_VERSION_V3]
        second = _ROW_SHAPES_BY_VERSION[ARTIFACT_VERSION_V2]

        fifth = _ROW_SHAPES_BY_VERSION[ARTIFACT_VERSION_V5]

        # v5 → v6 moved exactly one existing body: the invocation row, which now
        # states the authority the wake carried and the strings it carried, by
        # *adding* fields.
        assert set(current) == set(fifth)
        assert {name for name in fifth if fifth[name] != current[name]} == {
            "agent_invoked"
        }
        assert set(current["agent_invoked"][0]) - set(fifth["agent_invoked"][0]) == {
            "trigger_event_authority",
            "trigger_is_authoritative",
            "trigger_claim_values",
        }

        # v4 → v5 moved no existing body. It *added* three, and that is the
        # difference between a contract that changes what a row means and one
        # that records something the runtime did not do before.
        assert set(fifth) - set(fourth) == set(RETRIEVAL_RECORD_TYPES)
        assert all(fifth[name] == fourth[name] for name in fourth)

        # v3 → v4 moved the declared wait, and moved it by *removing* a field.
        assert {name for name in fourth if fourth[name] != third[name]} == {
            "wait_declared"
        }
        assert set(third["wait_declared"][1]) - set(fourth["wait_declared"][1]) == {
            "reachable"
        }
        assert not set(fourth["wait_declared"][1]) - set(third["wait_declared"][1])

        # v2 → v3 moved the accepted effect, by adding one.
        assert {name for name in third if third[name] != second[name]} == {
            "effect_accepted"
        }
        assert set(second["effect_accepted"][0]) | {"bindings"} == set(
            third["effect_accepted"][0]
        )

    def test_the_projection_only_drops_and_never_invents(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        """v3 strips no evidence, and a projection cannot add any.

        Two directions, both checked from the same run. Forward: the projection
        onto an older contract loses exactly the fields a *named* later contract
        added to a *named* row — ``bindings`` from ``effect_accepted`` at v3, the
        three trigger fields from ``agent_invoked`` at v6 — and changes nothing
        else about any row. It is the older contracts that carry less, not v3
        that carries less than they did.
        """
        added_at_v6 = {
            "trigger_event_authority",
            "trigger_is_authoritative",
            "trigger_claim_values",
        }
        current = current_artifact["trajectory"]
        for version in (ARTIFACT_VERSION_V1, ARTIFACT_VERSION_V2, ARTIFACT_VERSION_V3):
            projected = project_trajectory(current, version)
            assert len(projected) == len(current)
            for before, after in zip(current, projected, strict=True):
                lost = set(before) - set(after)
                assert lost <= {"bindings"} | added_at_v6
                assert not set(after) - set(before), "a projection invented a field"
                assert all(after[name] == before[name] for name in after)
                if before["record_type"] == "effect_accepted":
                    if version < ARTIFACT_VERSION_V3:
                        assert lost == {"bindings"}
                    else:
                        assert lost == set()
                elif before["record_type"] == "agent_invoked":
                    assert lost == added_at_v6
        # And projecting at the current contract is the identity.
        assert project_trajectory(current, ARTIFACT_VERSION) == list(current)


# -------------------------------------------------------------- what v2 adds


class TestTheV2Contract:
    def test_this_build_writes_v7_and_reads_all_seven(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        assert ARTIFACT_VERSION == 8
        assert SUPPORTED_ARTIFACT_VERSIONS == (
            ARTIFACT_VERSION_V1,
            ARTIFACT_VERSION_V2,
            ARTIFACT_VERSION_V3,
            ARTIFACT_VERSION_V4,
            ARTIFACT_VERSION_V5,
            ARTIFACT_VERSION_V6,
            ARTIFACT_VERSION_V7,
            ARTIFACT_VERSION,
        )
        assert current_artifact["artifact_version"] == ARTIFACT_VERSION

    def test_v2_is_v1_plus_exactly_three_fields(self) -> None:
        added = set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V2]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V1]
        )
        assert added == {"decisions", "decisions_digest_sha256", "agent_execution"}
        assert not set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V1]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V2]
        ), "v2 removes nothing v1 carried"

    def test_v3_adds_no_top_level_field(self) -> None:
        # The contract moved inside the trajectory. Asserted rather than left to
        # be inferred, because "the field sets are equal" is exactly what a
        # missed bump would also look like.
        assert set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V3]) == set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V2]
        )

    def test_v4_is_v3_plus_exactly_the_run_identity(self) -> None:
        added = set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V4]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V3]
        )
        assert added == {"operation_instance_id"}
        assert not set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V3]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V4]
        ), "v4 removes no top-level field v3 carried"

    def test_v7_is_v6_plus_exactly_the_provider_execution_binding(self) -> None:
        added = set(_FIELDS_BY_VERSION[ARTIFACT_VERSION]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION_V6]
        )
        assert added == {"provider_execution"}
        assert not set(_FIELDS_BY_VERSION[ARTIFACT_VERSION_V6]) - set(
            _FIELDS_BY_VERSION[ARTIFACT_VERSION]
        ), "v7 removes no top-level field v6 carried"

    def test_a_v2_artefact_missing_a_v2_field_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        for name in ("decisions", "decisions_digest_sha256", "agent_execution"):
            with pytest.raises(ArtifactError, match="missing required field"):
                validate_artifact(tampered(current_artifact, **{name: ...}))

    def test_an_unknown_top_level_field_is_refused_at_either_version(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(tampered(current_artifact, provenance_note="trust me"))

    def test_an_unsupported_version_is_refused_by_name(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        for version in (0, 9, "8", True, None):
            with pytest.raises(ArtifactError, match="not a contract version"):
                validate_artifact(tampered(current_artifact, artifact_version=version))

    def test_the_decisions_digest_describes_the_decisions_beside_it(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        broken = tampered(current_artifact)
        broken["decisions"][0]["observation_digest_sha256"] = "0" * 64
        with pytest.raises(ArtifactError, match="decisions_digest_sha256"):
            validate_artifact(broken)

    def test_the_artefact_round_trips_through_a_written_file(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V1", "reference")
        path = write_artifact(run, tmp_path / "run.json")
        assert read_artifact(path) == build_artifact(run)


# ------------------------------------------------------- section completeness


class TestSectionsPartitionTheArtefact:
    @pytest.mark.parametrize("version", SUPPORTED_ARTIFACT_VERSIONS)
    def test_every_field_belongs_to_exactly_one_section(self, version: int) -> None:
        assert section_coverage_problem(version) is None

    @pytest.mark.parametrize("version", SUPPORTED_ARTIFACT_VERSIONS)
    def test_the_partition_recomputed_independently_agrees(self, version: int) -> None:
        # Recomputed here rather than trusting the module's own checker, so a
        # checker that stopped checking would not take this test with it.
        placements: dict[str, int] = {}
        for names in _SECTIONS_BY_VERSION[version].values():
            for name in names:
                placements[name] = placements.get(name, 0) + 1
        assert placements == dict.fromkeys(_FIELDS_BY_VERSION[version], 1)

    def test_the_checker_notices_a_field_that_belongs_to_no_section(self) -> None:
        import operatebench.artifact as artifact

        original = artifact._FIELDS_BY_VERSION
        try:
            artifact._FIELDS_BY_VERSION = {
                **original,
                ARTIFACT_VERSION: (*original[ARTIFACT_VERSION], "an_orphan_field"),
            }
            problem = artifact.section_coverage_problem(ARTIFACT_VERSION)
        finally:
            artifact._FIELDS_BY_VERSION = original
        assert problem is not None and "an_orphan_field" in problem

    def test_the_checker_notices_a_field_placed_in_two_sections(self) -> None:
        import operatebench.artifact as artifact

        original = artifact._SECTIONS_BY_VERSION
        doubled = {
            **original[ARTIFACT_VERSION],
            "execution": ("agent_execution", "evaluation"),
        }
        try:
            artifact._SECTIONS_BY_VERSION = {**original, ARTIFACT_VERSION: doubled}
            problem = artifact.section_coverage_problem(ARTIFACT_VERSION)
        finally:
            artifact._SECTIONS_BY_VERSION = original
        assert problem is not None and "more than one" in problem

    def test_the_new_fields_sit_in_the_new_sections(self) -> None:
        sections = _SECTIONS_BY_VERSION[ARTIFACT_VERSION]
        assert sections["decisions"] == ("decisions", "decisions_digest_sha256")
        assert sections["execution"] == ("agent_execution",)

    def test_a_replay_reports_a_verdict_for_every_field_of_every_section(
        self, spec: OperationSpec, current_artifact: Mapping[str, Any]
    ) -> None:
        report = replay_artifact(spec, current_artifact)
        assert report.ok
        assert report.validated_episode_outcome is not None
        reported = {name for fields in report.sections.values() for name in fields}
        assert reported == set(_FIELDS_BY_VERSION[ARTIFACT_VERSION])
        body = report.as_dict()
        assert set(body["sections"]) == set(_SECTIONS_BY_VERSION[ARTIFACT_VERSION])
        serialized = json.dumps(current_artifact)
        assert "_provenance" not in serialized
        assert "authentication" not in serialized
        assert "validated_episode_outcome" not in json.dumps(body)

    def test_a_shortened_tape_fails_the_decisions_section(
        self, spec: OperationSpec, current_artifact: Mapping[str, Any]
    ) -> None:
        from operatebench.artifact import _content_digest

        body = tampered(current_artifact)
        body["decisions"] = body["decisions"][:-1]
        # Recompute the digest the validator checks, so what fails is the
        # replay comparison rather than the reader.
        body["decisions_digest_sha256"] = _content_digest(body["decisions"])
        report = replay_artifact(spec, body)
        assert not report.ok
        assert report.validated_episode_outcome is None
        assert not report.sections["decisions"]["decisions"]
        for verdicts in report.sections.values():
            assert isinstance(verdicts, dict)
            for field_name in verdicts:
                verdicts[field_name] = True
        assert report.ok
        assert report.validated_episode_outcome is None

    def test_a_digest_that_does_not_describe_its_tape_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact, decisions_digest_sha256="f" * 64)
        with pytest.raises(ArtifactError, match="decisions_digest_sha256"):
            validate_artifact(body)

    def test_an_execution_record_that_names_a_model_a_deterministic_run_never_called(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        """A model on an in-process run is a refusal, not a reported difference.

        Reporting it as a difference would require the reader to admit a record
        that describes an execution the run beside it could not have had. It
        does not admit one, so the execution section of a deterministic replay
        has no reachable difference to report: every field of that record is
        pinned by what produced the decisions.
        """
        body = tampered(current_artifact)
        body["agent_execution"]["model"] = TEST_MODEL
        body["agent_execution"]["protocol_version"] = "invented.protocol.v9"
        with pytest.raises(ArtifactError, match="never called"):
            validate_artifact(body)


# ---------------------------------------------------------- execution record


class TestTheExecutionRecord:
    def test_a_deterministic_run_records_no_provider_and_no_retries(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        execution = current_artifact["agent_execution"]
        assert execution["outcome_source"] == "deterministic"
        assert execution["model"] is None
        assert execution["transport_calls"] == 0
        assert execution["retry_count"] == RETRY_COUNT == 0
        assert execution["attempts"] == []
        assert execution["excluded"] is False
        assert execution["exclusion_code"] is None

    def test_a_model_run_records_its_model_calls_and_attempts(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        execution = model_artifact["agent_execution"]
        assert execution["outcome_source"] == "model"
        assert execution["model"] == GPT_5_6_LUNA_MODEL
        assert execution["protocol_version"]
        assert execution["transport_calls"] == len(model_artifact["decisions"])
        assert len(execution["attempts"]) == execution["transport_calls"]
        assert execution["excluded"] is False

    def test_exclusion_and_its_code_move_together_in_both_directions(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        for excluded, code in ((True, None), (False, FAULTS[0])):
            body = tampered(current_artifact)
            body["agent_execution"]["excluded"] = excluded
            body["agent_execution"]["exclusion_code"] = code
            with pytest.raises(ArtifactError, match="no third state"):
                validate_artifact(body)

    def test_a_record_claiming_retries_this_build_never_makes_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact)
        body["agent_execution"]["retry_count"] = 3
        with pytest.raises(ArtifactError, match="requests that were never sent"):
            validate_artifact(body)

    def test_an_in_process_run_cannot_claim_provider_calls(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact)
        body["agent_execution"]["transport_calls"] = 4
        with pytest.raises(ArtifactError, match="has a provider to call"):
            validate_artifact(body)

    def test_more_attempts_than_calls_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["agent_execution"]["transport_calls"] = 1
        with pytest.raises(ArtifactError, match="more attempts than calls"):
            validate_artifact(body)

    def test_an_unknown_source_or_fault_is_refused(
        self, current_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(current_artifact)
        body["agent_execution"]["outcome_source"] = "vibes"
        with pytest.raises(ArtifactError, match="not an outcome source"):
            validate_artifact(body)
        assert set(OUTCOME_SOURCES) == {"deterministic", "model", "recorded"}

        body = tampered(current_artifact)
        body["agent_execution"]["excluded"] = True
        body["agent_execution"]["exclusion_code"] = "the_weather"
        with pytest.raises(ArtifactError, match="not a fault this build classifies"):
            validate_artifact(body)

    def test_an_attempt_names_a_fault_exactly_when_it_failed(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["agent_execution"]["attempts"][0]["fault"] = FAULTS[0]
        with pytest.raises(ArtifactError, match="names the fault that ended it"):
            validate_artifact(body)

        body = tampered(model_artifact)
        body["agent_execution"]["attempts"][0]["outcome"] = "improvised"
        with pytest.raises(ArtifactError, match="not how an attempt ends"):
            validate_artifact(body)


# ------------------------------------------------------------ decision shapes


class TestRecordedDecisionShapes:
    def test_a_decision_carries_exactly_its_four_fields(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        for decision in model_artifact["decisions"]:
            assert set(decision) == {
                "invocation_index",
                "turn_index",
                "observation_digest_sha256",
                "outcome",
            }

    def test_a_decision_with_an_unknown_field_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0]["confidence"] = 0.9
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(body)

    def test_decisions_out_of_order_are_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0], body["decisions"][1] = (
            body["decisions"][1],
            body["decisions"][0],
        )
        with pytest.raises(ArtifactError, match="order moved"):
            validate_artifact(body)

    def test_a_decision_before_the_first_invocation_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0]["invocation_index"] = 0
        with pytest.raises(ArtifactError, match="counted from one"):
            validate_artifact(body)

    def test_a_decision_kind_this_build_does_not_write_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0]["outcome"] = {"kind": "IMPROVISE"}
        with pytest.raises(ArtifactError, match="not a decision kind"):
            validate_artifact(body)

    def test_a_malformed_decision_carries_a_known_code_and_nothing_else(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0]["outcome"] = {
            "kind": "MALFORMED",
            "code": "MODEL_INVENTED",
        }
        with pytest.raises(ArtifactError, match="not a model output classification"):
            validate_artifact(body)

        body = tampered(model_artifact)
        body["decisions"][0]["outcome"] = {
            "kind": "MALFORMED",
            "code": "MODEL_NO_TOOL_CALL",
            "provider_text": "prose",
        }
        with pytest.raises(ArtifactError, match="unknown field"):
            validate_artifact(body)

    def test_a_non_digest_observation_binding_is_refused(
        self, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["decisions"][0]["observation_digest_sha256"] = "not-a-digest"
        with pytest.raises(ArtifactError, match="SHA-256 digest"):
            validate_artifact(body)


# --------------------------------------------------------------- replay paths


class TestReplayChoosesItsReproduction:
    def test_a_deterministic_v2_artefact_re_executes_its_agent(
        self, spec: OperationSpec, current_artifact: Mapping[str, Any]
    ) -> None:
        report = replay_artifact(spec, current_artifact)
        assert report.ok
        assert report.reproduction == REPRODUCTION_RERUN
        assert report.carried_forward == ()
        assert report.artifact_version == ARTIFACT_VERSION

    def test_a_model_artefact_replays_from_its_tape_and_says_what_it_carried(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        report = replay_artifact(spec, model_artifact)
        assert report.ok
        assert report.reproduction == REPRODUCTION_PLAYBACK
        assert report.carried_forward == (
            "agent_execution",
            "agent_id",
            "agent_kind",
            "decisions",
            "decisions_digest_sha256",
            "provider_execution",
        ), (
            "a playback takes the decisions off the tape, cannot re-derive the "
            "execution record or the ledger binding of a provider session it is "
            "required not to repeat, and has no registry to check a model agent's "
            "id or kind against. All six are named rather than implying the "
            "comparison was total"
        )
        body = report.as_dict()
        assert body["carried_forward"] == list(report.carried_forward)
        assert body["total_comparison"] is False
        assert report.total_comparison is False

    def test_a_deterministic_replay_says_its_comparison_was_total(
        self, spec: OperationSpec, current_artifact: Mapping[str, Any]
    ) -> None:
        report = replay_artifact(spec, current_artifact)
        assert report.total_comparison is True
        assert report.as_dict()["total_comparison"] is True

    def test_a_model_artefact_whose_final_state_was_edited_still_fails(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        body = tampered(model_artifact)
        body["final_state"]["reopen_count"] = 9
        with pytest.raises(ArtifactError, match="final_state_digest_sha256"):
            replay_artifact(spec, body)

    def test_a_model_artefact_whose_tape_was_edited_is_refused(
        self, spec: OperationSpec, model_artifact: Mapping[str, Any]
    ) -> None:
        """Removing a decision fails the reader, without executing anything.

        A model run's attempts and its decisions are the same count, so a tape
        one decision short contradicts the execution record recorded beside it.
        Playback would refuse it too — as a tape the episode does not consume —
        but the reader does not need an episode to see it.
        """
        from operatebench.artifact import _content_digest

        body = tampered(model_artifact)
        body["decisions"] = body["decisions"][:-1]
        body["decisions_digest_sha256"] = _content_digest(body["decisions"])
        with pytest.raises(ArtifactError, match="describe different runs"):
            replay_artifact(spec, body)
