"""Phase 2, part 4: contract 6, and what v5 keeps and loses.

Contract 6 exists because the *meaning* of a record moved, not because a field
was added. The model observation no longer carries the normalized projection;
the result vector carries a dimension it did not carry; a business proposal is
now bound to the reads that preceded it, so the runtime writes an invocation row
with the waking event's authority and the claim values it carried. A v5 document
describes a run under semantics this build no longer executes.

So v5 stays *readable* — refusing to read it would destroy evidence about how the
benchmark stood — and stops being *reproducible*. Playback still re-serves every
read from the live operation, compares versions exactly and reaches no provider.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.artifact import (
    ARTIFACT_VERSION,
    ARTIFACT_VERSION_V4,
    ARTIFACT_VERSION_V5,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    SUPPORTED_ARTIFACT_VERSIONS,
    ArtifactError,
    RetrievalProvenanceMismatchError,
    evaluation_dimensions_for,
    project_top_level,
    project_trajectory,
    read_artifact,
    replay_artifact,
    row_shape_coverage_problem,
    section_coverage_problem,
    write_artifact,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

#: The three fields the invocation row gained at contract 6.
INVOCATION_FIELDS_NEW_IN_V6 = (
    "trigger_claim_values",
    "trigger_event_authority",
    "trigger_is_authoritative",
)


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


def _as_v5(payload: dict) -> dict:
    """The same record as contract 5 would have written it.

    The row projection is :func:`operatebench.artifact.project_trajectory` — the
    shipped function, not a restatement of it beside the test. A bespoke copy
    here was masking the bug it was written to exercise: the shipped projection
    left the three contract-6 invocation fields in place, and only the test's
    private reimplementation removed them, so the suite proved a property of
    itself. The digest is recomputed *here*, by the caller, because a projection
    changes content and the function deliberately does not hide that.
    """
    body = project_top_level(payload, ARTIFACT_VERSION_V5)
    body["artifact_version"] = ARTIFACT_VERSION_V5
    body["trajectory"] = project_trajectory(payload["trajectory"], ARTIFACT_VERSION_V5)
    body["trajectory_digest_sha256"] = hashlib.sha256(
        canonical_json_bytes(body["trajectory"], "operatebench trajectory")
    ).hexdigest()
    body["evaluation"] = _vector_as_v5(payload["evaluation"])
    return body


def _vector_as_v5(evaluation: dict) -> dict:
    """The result vector as contract 5 wrote one: without the sixth-contract
    dimension, and with every field derived from it recomputed rather than
    carried over."""
    names = evaluation_dimensions_for(ARTIFACT_VERSION_V5)
    kept = [entry for entry in evaluation["dimensions"] if entry["name"] in names]
    vector = dict(evaluation)
    vector["dimensions"] = kept
    vector["failed_dimensions"] = [entry["name"] for entry in kept if not entry["ok"]]
    vector["finding_codes"] = [
        finding["code"] for entry in kept for finding in entry["findings"]
    ]
    vector["reliable"] = bool(evaluation["legitimate_completion"]) and all(
        entry["ok"] for entry in kept
    )
    return vector


@pytest.fixture(scope="module")
def run_v1(spec):
    return run_episode(spec, "V1", "reference")


class TestTheContractMovedToSix:
    def test_this_build_writes_seven(self) -> None:
        assert ARTIFACT_VERSION == 8

    def test_it_still_reads_one_through_six(self) -> None:
        assert SUPPORTED_ARTIFACT_VERSIONS == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_only_six_is_reproducible(self) -> None:
        assert REPRODUCIBLE_ARTIFACT_VERSIONS == (ARTIFACT_VERSION,)
        assert ARTIFACT_VERSION_V5 not in REPRODUCIBLE_ARTIFACT_VERSIONS
        assert ARTIFACT_VERSION_V4 not in REPRODUCIBLE_ARTIFACT_VERSIONS

    def test_every_row_type_is_described_at_every_version(self) -> None:
        assert row_shape_coverage_problem() is None

    def test_every_field_is_compared_at_six(self) -> None:
        assert section_coverage_problem(ARTIFACT_VERSION) is None

    def test_a_written_record_states_six(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        assert payload["artifact_version"] == ARTIFACT_VERSION
        graded = {dimension["name"] for dimension in payload["evaluation"]["dimensions"]}
        assert "retrieval_discipline" in graded


class TestTheRetrievalEvidenceIsInTheRecord:
    def test_served_rows_carry_their_records(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        served = [
            row
            for row in payload["trajectory"]
            if row["record_type"] == "retrieval_served"
        ]
        assert served
        for row in served:
            assert row["records"]
            assert row["record_version"]
            assert row["initiated_by"] == "agent"

    def test_the_invocation_row_states_the_waking_authority(
        self, run_v1, tmp_path
    ) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        invoked = [
            row for row in payload["trajectory"] if row["record_type"] == "agent_invoked"
        ]
        assert invoked
        for row in invoked:
            assert "trigger_event_authority" in row
            assert "trigger_claim_values" in row

    def test_no_provider_execution_field_is_present(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        text = json.dumps(payload)
        for absent in ("prompt_tokens", "completion_tokens", "usage", "cost_usd"):
            assert absent not in text


class TestPlaybackReservesEveryRead:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_a_record_replays_with_zero_provider_calls(
        self, spec, scenario_id, tmp_path
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        payload = read_artifact(write_artifact(run, tmp_path / f"{scenario_id}.json"))
        report = replay_artifact(spec, payload)
        assert report.ok is True
        assert payload["agent_execution"]["transport_calls"] == 0

    def test_editing_one_recorded_version_fails_by_name(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served":
                row["record_version"] = "0" * len(row["record_version"])
                break
        else:
            raise AssertionError("no served row to edit")
        with pytest.raises((RetrievalProvenanceMismatchError, ArtifactError)):
            replay_artifact(load_spec(SPEC), payload)

    def test_editing_one_recorded_read_set_fails(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        trajectory = [
            row
            for row in payload["trajectory"]
            if row["record_type"] != "retrieval_served"
        ]
        payload["trajectory"] = trajectory
        with pytest.raises(ArtifactError):
            replay_artifact(load_spec(SPEC), payload)


class TestTheShippedProjectionWritesTheOlderRow:
    def test_it_drops_exactly_the_fields_contract_six_added(
        self, run_v1, tmp_path
    ) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        rows = payload["trajectory"]
        projected = project_trajectory(rows, ARTIFACT_VERSION_V5)
        assert len(projected) == len(rows)
        invoked = [row for row in rows if row["record_type"] == "agent_invoked"]
        assert invoked
        for before, after in zip(rows, projected, strict=True):
            if before["record_type"] != "agent_invoked":
                assert after == before
                continue
            assert tuple(sorted(set(before) - set(after))) == INVOCATION_FIELDS_NEW_IN_V6
            assert all(after[name] == before[name] for name in after)

    def test_it_leaves_the_current_contract_alone(self, run_v1, tmp_path) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        assert (
            project_trajectory(payload["trajectory"], ARTIFACT_VERSION)
            == (payload["trajectory"])
        )

    def test_it_still_drops_the_bindings_contract_three_added(
        self, run_v1, tmp_path
    ) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        accepted = [
            row
            for row in payload["trajectory"]
            if row["record_type"] == "effect_accepted"
        ]
        assert accepted
        assert all("bindings" in row for row in accepted)
        projected = project_trajectory(payload["trajectory"], 2)
        assert not any(
            "bindings" in row
            for row in projected
            if row["record_type"] == "effect_accepted"
        )

    def test_it_does_not_sanitise_a_field_it_does_not_own(self, run_v1, tmp_path) -> None:
        # A projection that quietly removed anything it did not recognise would
        # launder a forged row into a document the older reader accepts. It owns
        # three field names and removes those three; a sentinel it was never
        # told about survives, and the reader is what refuses it.
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        rows = json.loads(json.dumps(payload["trajectory"]))
        for row in rows:
            if row["record_type"] == "agent_invoked":
                row["smuggled_field"] = "sentinel"
                break
        else:
            raise AssertionError("no invocation row to mark")
        projected = project_trajectory(rows, ARTIFACT_VERSION_V5)
        assert any(row.get("smuggled_field") == "sentinel" for row in projected)

    def test_the_historical_reader_refuses_the_sentinel_it_kept(
        self, run_v1, tmp_path
    ) -> None:
        payload = _as_v5(read_artifact(write_artifact(run_v1, tmp_path / "v6.json")))
        for row in payload["trajectory"]:
            if row["record_type"] == "agent_invoked":
                row["smuggled_field"] = "sentinel"
                break
        else:
            raise AssertionError("no invocation row to mark")
        payload["trajectory_digest_sha256"] = hashlib.sha256(
            canonical_json_bytes(payload["trajectory"], "operatebench trajectory")
        ).hexdigest()
        path = tmp_path / "smuggled.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError):
            read_artifact(path)


class TestVersionFiveIsReadableAndNotReproducible:
    def test_a_five_record_reads(self, run_v1, tmp_path) -> None:
        # A v5-shaped document: the contract-5 invocation row, without the two
        # fields contract 6 added. It reads, because refusing to read it would
        # destroy evidence about how the benchmark stood at contract 5.
        payload = _as_v5(read_artifact(write_artifact(run_v1, tmp_path / "v6.json")))
        path = tmp_path / "v5.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        again = read_artifact(path)
        assert again["artifact_version"] == ARTIFACT_VERSION_V5
        invoked = [
            row for row in again["trajectory"] if row["record_type"] == "agent_invoked"
        ]
        assert invoked
        for row in invoked:
            assert not set(row) & set(INVOCATION_FIELDS_NEW_IN_V6)
        names = [entry["name"] for entry in again["evaluation"]["dimensions"]]
        assert tuple(names) == evaluation_dimensions_for(ARTIFACT_VERSION_V5)
        assert "retrieval_discipline" not in names

    def test_a_six_shaped_row_is_refused_under_contract_five(
        self, run_v1, tmp_path
    ) -> None:
        payload = read_artifact(write_artifact(run_v1, tmp_path / "v6.json"))
        payload["artifact_version"] = ARTIFACT_VERSION_V5
        path = tmp_path / "mislabelled.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ArtifactError):
            read_artifact(path)

    def test_a_five_record_does_not_replay(self, run_v1, tmp_path) -> None:
        payload = _as_v5(read_artifact(write_artifact(run_v1, tmp_path / "v6.json")))
        with pytest.raises(ArtifactError):
            replay_artifact(load_spec(SPEC), payload)
