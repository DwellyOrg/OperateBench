"""The run artefact and exact replay.

An artefact is only worth writing if a later reader can reconstruct the run from
it and compare, so the properties asserted here are: the bytes are canonical and
deterministic, the artefact carries everything replay needs, replay reproduces
the final canonical state, the trajectory and the evaluation exactly, and an
artefact that does not match the spec it is replayed against is refused by name.

What is deliberately *not* claimed anywhere is immutability. A digest over the
canonical bytes says two artefacts have the same content; it does not say
anybody was prevented from editing one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boundarybench.jsonsafe import MAX_JSON_DEPTH
from operatebench.artifact import (
    ARTIFACT_VERSION,
    ReplayReport,
    build_artifact,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.core.errors import ArtifactError
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.runner import run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture()
def artifact_path(spec: OperationSpec, tmp_path: Path) -> Path:
    run = run_episode(spec, "V1", "reference")
    return write_artifact(run, tmp_path / "v1_reference.json")


class TestArtifactContent:
    def test_the_artefact_records_operation_and_run_identity(
        self, spec: OperationSpec, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        assert payload["artifact_version"] == ARTIFACT_VERSION
        assert payload["operation"]["operation_id"] == spec.operation_id
        assert payload["operation"]["spec_digest_sha256"] == spec.spec_digest_sha256
        assert payload["scenario_id"] == "V1"
        assert payload["agent_id"] == "reference"
        assert payload["semantic_scenario_id"] == spec.semantic_scenario_id

    def test_the_artefact_records_semantic_input_and_dispositions(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        dispositions = {event["disposition"] for event in payload["events"]}
        assert {"accepted", "audit", "post_terminal"} <= dispositions
        # Every delivered event carries its semantic payload and its cause.
        for event in payload["events"]:
            assert "payload" in event
            assert "caused_by" in event
            assert "at" in event

    def test_the_artefact_records_simulated_time_not_wall_clock(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        assert payload["started_at"] == "2031-03-03T09:00:00Z"
        assert payload["ended_at"] > payload["started_at"]
        assert payload["simulated_minutes"] > 0

    def test_the_artefact_records_accepted_and_rejected_effects(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V1", "trust_actor_claim")
        payload = read_artifact(write_artifact(run, tmp_path / "claim.json"))
        kinds = {row["record_type"] for row in payload["trajectory"]}
        assert {"action_proposed", "action_rejected", "effect_accepted"} <= kinds

    def test_the_artefact_carries_the_whole_result_vector(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        assert payload["evaluation"]["reliable"] is True
        assert len(payload["evaluation"]["dimensions"]) >= 9

    def test_the_artefact_states_what_its_digests_do_not_claim(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        assert "immutab" not in json.dumps(payload["note"]).lower()
        assert "content digest" in payload["note"].lower()


class TestDeterminism:
    def test_writing_one_run_twice_produces_identical_bytes(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V1", "reference")
        first = write_artifact(run, tmp_path / "a.json")
        second = write_artifact(run, tmp_path / "b.json")
        assert first.read_bytes() == second.read_bytes()

    def test_two_runs_differ_only_where_the_run_identity_reaches(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        """Determinism is about the operation, not about the run's own name.

        Two executions of one scenario are two *operation instances*, and each
        gets its own opaque identity. That identity is inside the observation the
        agent answers, so the decision tape's observation digests move with it —
        and nothing else does. Asserting byte equality across two runs would be
        asserting that a run has no identity of its own; asserting this says
        exactly which fields a second run is entitled to change.
        """
        first = json.loads(
            write_artifact(
                run_episode(spec, "V1", "reference"), tmp_path / "a.json"
            ).read_text(encoding="utf-8")
        )
        second = json.loads(
            write_artifact(
                run_episode(spec, "V1", "reference"), tmp_path / "b.json"
            ).read_text(encoding="utf-8")
        )
        moved = {name for name in first if first[name] != second[name]}
        assert moved == {
            "operation_instance_id",
            "decisions",
            "decisions_digest_sha256",
        }
        assert first["operation_instance_id"] != second["operation_instance_id"]
        # The operation itself ran identically: same world, same trajectory, same
        # canonical state, same result vector.
        assert first["trajectory_digest_sha256"] == second["trajectory_digest_sha256"]
        assert first["final_state_digest_sha256"] == second["final_state_digest_sha256"]
        assert first["evaluation"] == second["evaluation"]
        # And only the binding digest inside each decision moved: the decisions
        # themselves are the same decisions.
        assert [entry["outcome"] for entry in first["decisions"]] == [
            entry["outcome"] for entry in second["decisions"]
        ]

    def test_the_artefact_is_canonical_json(self, artifact_path: Path) -> None:
        text = artifact_path.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert list(payload) == sorted(payload)


class TestReplay:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_replay_reconstructs_the_run_exactly(
        self, spec: OperationSpec, tmp_path: Path, scenario_id: str
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        path = write_artifact(run, tmp_path / f"{scenario_id}.json")
        report = replay_artifact(spec, read_artifact(path))
        assert isinstance(report, ReplayReport)
        assert report.ok is True, report.differences
        assert report.final_state_matches is True
        assert report.trajectory_matches is True
        assert report.evaluation_matches is True

    def test_replay_reproduces_a_negative_run_too(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V1", "always_escalate")
        path = write_artifact(run, tmp_path / "neg.json")
        assert replay_artifact(spec, read_artifact(path)).ok is True

    def test_replay_refuses_a_tampered_final_state(
        self, spec: OperationSpec, artifact_path: Path
    ) -> None:
        # A rewritten final state never reaches the comparison: it is refused
        # while the artefact is read, both as a status this build never writes
        # and as a payload its own digest no longer describes.
        payload = read_artifact(artifact_path)
        payload["final_state"]["payment"]["status"] = "FAILED"
        with pytest.raises(ArtifactError) as excinfo:
            replay_artifact(spec, payload)
        assert "payment" in str(excinfo.value)

    def test_replay_refuses_a_tampered_trajectory(
        self, spec: OperationSpec, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        payload["trajectory"] = payload["trajectory"][:-1]
        with pytest.raises(ArtifactError) as excinfo:
            replay_artifact(spec, payload)
        assert "trajectory_digest_sha256" in str(excinfo.value)

    def test_replaying_against_a_different_spec_is_refused_by_name(
        self, spec: OperationSpec, artifact_path: Path, tmp_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        payload["operation"]["spec_digest_sha256"] = "0" * 64
        with pytest.raises(ArtifactError) as excinfo:
            replay_artifact(spec, payload)
        assert "spec" in str(excinfo.value).lower()


class TestMalformedArtefacts:
    def test_a_missing_file_is_a_named_domain_error(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactError):
            read_artifact(tmp_path / "absent.json")

    def test_a_non_json_file_is_a_named_domain_error(self, tmp_path: Path) -> None:
        path = tmp_path / "run.json"
        path.write_text("this is not json", encoding="utf-8")
        with pytest.raises(ArtifactError):
            read_artifact(path)

    def test_a_json_document_that_is_not_an_object_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ArtifactError):
            read_artifact(path)

    def test_an_artefact_over_the_public_json_depth_limit_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.json"
        depth = MAX_JSON_DEPTH + 1
        path.write_text("[" * depth + "0" + "]" * depth, encoding="utf-8")

        with pytest.raises(ArtifactError) as excinfo:
            read_artifact(path)

        assert f"maximum JSON nesting depth is {MAX_JSON_DEPTH}" in str(excinfo.value)

    def test_a_missing_required_field_is_refused_by_name(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        del payload["trajectory"]
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError) as excinfo:
            read_artifact(artifact_path)
        assert "trajectory" in str(excinfo.value)

    def test_an_unsupported_artefact_version_is_refused(
        self, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        payload["artifact_version"] = ARTIFACT_VERSION + 41
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError):
            read_artifact(artifact_path)

    def test_an_artefact_naming_an_unknown_agent_is_refused_at_replay(
        self, spec: OperationSpec, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        payload["agent_id"] = "a_model_we_do_not_ship"
        with pytest.raises(ArtifactError):
            replay_artifact(spec, payload)

    def test_an_artefact_naming_an_unknown_scenario_is_refused_at_replay(
        self, spec: OperationSpec, artifact_path: Path
    ) -> None:
        payload = read_artifact(artifact_path)
        payload["scenario_id"] = "V9"
        with pytest.raises(ArtifactError):
            replay_artifact(spec, payload)


class TestBuildArtifact:
    def test_the_payload_is_the_same_whether_built_or_read_back(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V2", "reference")
        built = build_artifact(run)
        path = write_artifact(run, tmp_path / "v2.json")
        assert read_artifact(path) == built

    def test_writing_refuses_to_overwrite_an_existing_artefact(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V2", "reference")
        path = write_artifact(run, tmp_path / "v2.json")
        with pytest.raises(ArtifactError):
            write_artifact(run, path)

    def test_writing_creates_parent_directories(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V2", "reference")
        path = write_artifact(run, tmp_path / "nested" / "deep" / "v2.json")
        assert path.exists()
