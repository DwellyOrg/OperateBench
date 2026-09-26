"""Phase 1A: what a retrieval leaves behind, and what a replay does with it.

A read the record does not carry is a read nobody can check, so this file holds
the three halves of the evidence: the decision tape learns a ``RETRIEVE`` kind,
the artefact contract moves to 5 and describes the rows the runtime now writes,
and a replay re-serves every recorded batch **from the live domain** and requires
the provenance to come back identical. Editing one recorded ``record_version`` is
refused by name rather than reported as a difference among others.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.agents.playback import (
    MALFORMED_KIND,
    decision_record,
    outcome_from_record,
)
from operatebench.artifact import (
    ARTIFACT_VERSION,
    ARTIFACT_VERSION_V4,
    REPRODUCIBLE_ARTIFACT_VERSIONS,
    SUPPORTED_ARTIFACT_VERSIONS,
    RetrievalProvenanceMismatchError,
    build_artifact,
    outcome_shape_coverage_problem,
    read_artifact,
    replay_artifact,
    row_shape_coverage_problem,
    section_coverage_problem,
    validate_artifact,
)
from operatebench.core.errors import ArtifactError
from operatebench.core.retrieval import (
    RETRIEVE_KIND,
    RetrievalRequest,
    RetrieveBatch,
)
from operatebench.domains.lettings.maintenance.evaluator import RETRIEVAL_GRAMMAR_CODES
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

#: A model run is attributed to an agent id no in-process registry claims: a
#: registered deterministic control has no provider to have called.
MODEL_AGENT_ID = "model_reference"

#: The byte-frozen contract-1 record the suite keeps so that "this build still
#: reads what it wrote before" stays executable.
FROZEN_V1 = Path(__file__).resolve().parent / "fixtures/artifact_v1_reference_V1.json"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


@pytest.fixture(scope="module")
def reference_run(spec):
    return run_episode(spec, "V1", "reference")


@pytest.fixture(scope="module")
def reference_artifact(reference_run):
    return build_artifact(reference_run)


def batch(*tools: str) -> RetrieveBatch:
    return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in tools))


# -- the tape ---------------------------------------------------------------


class TestTheTapeRecordsARetrieval:
    def test_a_batch_is_recorded_as_a_retrieve_decision(self) -> None:
        record = decision_record(batch("list_quotes", "get_case_record"))
        assert record["kind"] == RETRIEVE_KIND
        assert record["requests"] == [
            {"tool": "list_quotes", "arguments": {}},
            {"tool": "get_case_record", "arguments": {}},
        ]
        assert record["kind"] != MALFORMED_KIND

    def test_the_recorded_row_distinguishes_permutation_and_multiplicity(self) -> None:
        rows = {
            canonical_json_bytes(decision_record(candidate), "retrieve row")
            for candidate in (
                batch("list_quotes", "get_case_record", "list_billing"),
                batch("list_billing", "list_quotes", "get_case_record"),
                batch("get_case_record", "list_billing", "list_quotes", "list_quotes"),
            )
        }
        assert len(rows) == 3

    def test_a_recorded_retrieval_rebuilds_into_the_same_batch(self) -> None:
        original = batch("get_case_record", "list_quotes")
        rebuilt = outcome_from_record(decision_record(original))
        assert isinstance(rebuilt, RetrieveBatch)
        assert [request.tool for request in rebuilt.requests] == [
            "get_case_record",
            "list_quotes",
        ]

    def test_an_empty_batch_round_trips_so_its_refusal_can_be_replayed(self) -> None:
        record = decision_record(RetrieveBatch(()))
        assert record == {"kind": RETRIEVE_KIND, "requests": []}
        assert outcome_from_record(record) == RetrieveBatch(())


# -- the artefact contract --------------------------------------------------


class TestTheArtefactContractMoves:
    def test_the_build_writes_seven_reads_seven_and_reproduces_only_seven(
        self,
    ) -> None:
        assert ARTIFACT_VERSION == 8
        assert ARTIFACT_VERSION_V4 == 4
        assert SUPPORTED_ARTIFACT_VERSIONS == (1, 2, 3, 4, 5, 6, 7, 8)
        assert REPRODUCIBLE_ARTIFACT_VERSIONS == (8,)

    def test_the_runtime_version_moves_with_the_rows_it_writes(self) -> None:
        assert OPERATEBENCH_VERSION == "0.13.0"

    def test_every_row_and_outcome_shape_is_covered(self) -> None:
        assert row_shape_coverage_problem() is None
        assert outcome_shape_coverage_problem() is None

    def test_every_field_of_contract_five_is_compared_by_replay(self) -> None:
        assert section_coverage_problem(5) is None
        for version in SUPPORTED_ARTIFACT_VERSIONS:
            assert section_coverage_problem(version) is None, version

    def test_a_recorded_run_carries_its_retrieval_rows(self, reference_artifact) -> None:
        served = [
            row
            for row in reference_artifact["trajectory"]
            if row["record_type"] == "retrieval_served"
        ]
        refused = [
            row
            for row in reference_artifact["trajectory"]
            if row["record_type"] == "retrieval_refused"
        ]
        invalidated = [
            row
            for row in reference_artifact["trajectory"]
            if row["record_type"] == "derived_context_invalidated"
        ]
        assert served
        assert refused == []
        assert invalidated
        assert validate_artifact(reference_artifact)["artifact_version"] == 8

    def test_a_retrieval_row_that_states_an_unknown_authority_is_refused(
        self, reference_artifact
    ) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served":
                row["authority"] = "vibes"
                break
        with pytest.raises(ArtifactError):
            validate_artifact(payload)

    def test_an_earlier_contract_is_still_read_and_no_longer_replayed(self, spec) -> None:
        # Read: it is what it is, and refusing to read it would destroy evidence
        # about how the benchmark stood. Not replayed: contract 5 moved the
        # observation a decision is bound to, so there is nothing honest to
        # compare it against.
        frozen = read_artifact(FROZEN_V1)
        assert frozen["artifact_version"] == 1
        with pytest.raises(ArtifactError) as caught:
            replay_artifact(spec, frozen)
        assert "reproduces contract(s) [8]" in str(caught.value)

    def test_a_contract_four_label_over_contract_five_rows_is_refused(
        self, reference_artifact
    ) -> None:
        # The other direction, and the one a forgery would take: relabelling a
        # run that read records as a contract whose runtime could not read any.
        # Refused at the reader, because contract 4 has no shape for a retrieval
        # row — not merely reported as a difference later.
        payload = json.loads(json.dumps(reference_artifact))
        payload["artifact_version"] = ARTIFACT_VERSION_V4
        with pytest.raises(ArtifactError) as caught:
            validate_artifact(payload)
        message = str(caught.value)
        # Contract 6 added fields to the invocation row, so a contract-4 label
        # is refused there before the reader reaches the rows contract 5 added.
        # Contract 7 added a top-level field, so a contract-4 label is now
        # refused on the field set before the reader reaches any row at all.
        # Every one of these is the same statement: this document was not
        # written by the contract it claims.
        assert (
            "retrieval" in message
            or "derived_context" in message
            or "agent_invoked" in message
            or "provider_execution" in message
        )


# -- the records the row carries --------------------------------------------


class TestTheServedRowCarriesItsRecords:
    """Contract 5 draft: the row states what came back, not only that it did.

    Without the body, the only thing that could recompute a ``record_version``
    was a replay against the live domain — which is the artefact reader's own
    machinery, not an independent check. A row that carries its records is a row
    an evaluator can hold to its own digest.
    """

    def test_a_served_row_carries_the_published_roots_for_its_tool(
        self, reference_artifact
    ) -> None:
        from operatebench.domains.lettings.maintenance.retrieval import (
            MAINTENANCE_RETRIEVAL_TOOLS,
        )

        served = [
            row
            for row in reference_artifact["trajectory"]
            if row["record_type"] == "retrieval_served"
        ]
        assert served
        for row in served:
            spec = MAINTENANCE_RETRIEVAL_TOOLS[row["tool"]]
            assert set(row["records"]) == set(spec.roots)

    def test_the_row_body_is_detached_from_the_live_state(self, spec) -> None:
        run = run_episode(spec, "V1", "reference")
        payload = build_artifact(run)
        row = next(
            row
            for row in payload["trajectory"]
            if row["record_type"] == "retrieval_served"
        )
        row["records"]["injected"] = True
        second = build_artifact(run_episode(spec, "V1", "reference"))
        clean = next(
            row
            for row in second["trajectory"]
            if row["record_type"] == "retrieval_served"
        )
        assert "injected" not in clean["records"]

    @pytest.mark.parametrize(
        "records",
        [
            "not a mapping",
            ["not", "a", "mapping"],
            {"quotes": {1: "an integer key"}},
            {"quotes": {"nested": {"value": float("inf")}}},
        ],
    )
    def test_a_records_body_json_cannot_carry_is_refused_by_the_reader(
        self, reference_artifact, records
    ) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served":
                row["records"] = records
                break
        with pytest.raises(ArtifactError):
            validate_artifact(payload)

    def test_a_contract_five_row_without_records_is_refused(
        self, reference_artifact
    ) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served":
                row.pop("records")
                break
        with pytest.raises(ArtifactError):
            validate_artifact(payload)

    def test_editing_the_recorded_records_is_refused_by_name_on_replay(
        self, spec, reference_artifact
    ) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served" and row["tool"] == "list_quotes":
                assert row["records"]["quotes"] != {"forged-quote#1": {}}
                row["records"]["quotes"] = {"forged-quote#1": {}}
                break
        payload["trajectory_digest_sha256"] = _trajectory_digest(payload["trajectory"])
        with pytest.raises(RetrievalProvenanceMismatchError):
            replay_artifact(spec, payload)


# -- replay -----------------------------------------------------------------


class TestReplayReServesFromTheLiveDomain:
    def test_a_deterministic_run_replays_whole(self, spec, reference_artifact) -> None:
        report = replay_artifact(spec, reference_artifact)
        assert report.ok
        assert report.differences == ()

    def test_a_model_run_replays_without_touching_a_provider(
        self, spec, tmp_path_factory
    ) -> None:
        """A recorded model run, replayed. Contract 7 needs the ledger with it.

        The run goes through the real SDK over an in-process mock transport with
        an evidence recorder attached, because a model run with no complete
        scored journal has no episode artefact to replay at all. What is being
        asserted is unchanged: the served retrieval rows are in the record, and
        replaying it dispatches nothing.
        """
        from tests.provider_evidence_runs import model_run_with_ledger

        directory = tmp_path_factory.mktemp("model_replay")
        directory.chmod(0o700)
        evidence = model_run_with_ledger(directory, spec, agent_id=MODEL_AGENT_ID)
        payload = build_artifact(evidence.run)
        served = [
            row
            for row in payload["trajectory"]
            if row["record_type"] == "retrieval_served"
        ]
        assert served, "a model run of the reference retrieves before it acts"
        calls_before = len(evidence.wire.bodies)
        report = replay_artifact(spec, payload)
        assert report.ok
        assert len(evidence.wire.bodies) == calls_before, "replay reaches no provider"

    def test_editing_one_recorded_record_version_is_refused_by_name(
        self, spec, reference_artifact
    ) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        edited = False
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served":
                row["record_version"] = "f" * 16
                edited = True
                break
        assert edited
        payload["trajectory_digest_sha256"] = _trajectory_digest(payload["trajectory"])
        with pytest.raises(RetrievalProvenanceMismatchError):
            replay_artifact(spec, payload)


def _trajectory_digest(rows: list[dict[str, Any]]) -> str:
    import hashlib

    return hashlib.sha256(
        canonical_json_bytes(rows, "operatebench trajectory")
    ).hexdigest()


# -- the evaluator reads the retrieval log ----------------------------------


class TestTheEvaluatorHoldsTheRetrievalGrammar:
    """Not a grading dimension — a readability check on the evidence itself.

    Whether an agent read *well* is a separate claim that needs its own
    dimension and its own negatives, and this phase deliberately does not make
    it. What is checked here is narrower and prior to it: a retrieval log whose
    ordering, binding or vocabulary does not hold is evidence of nothing.
    """

    def _evaluate(self, payload: dict[str, Any]):
        from operatebench.domains.lettings.maintenance.evaluator import evaluate

        return evaluate(
            status=payload["status"],
            replay_final=payload["replay_final"],
            simulated_minutes=payload["simulated_minutes"],
            invocations=payload["agent_invocations"],
            final_state=payload["final_state"],
            trajectory=payload["trajectory"],
            events=payload["events"],
            scenario=load_spec(SPEC).scenario("V1"),
            replay_ok=True,
        )

    def test_a_clean_run_states_no_grammar_finding(self, reference_artifact) -> None:
        evaluation = self._evaluate(json.loads(json.dumps(reference_artifact)))
        assert evaluation.reliable
        assert not set(evaluation.finding_codes) & set(RETRIEVAL_GRAMMAR_CODES)

    @pytest.mark.parametrize(
        ("edit", "code"),
        [
            ({"as_of": "2099-01-01T00:00:00Z"}, "RETRIEVAL_AS_OF_NOT_ONE_INSTANT"),
            ({"initiated_by": "environment_refresh"}, "RETRIEVAL_INITIATOR_UNKNOWN"),
            ({"authority": "vibes"}, "RETRIEVAL_RESULT_VOCABULARY_UNKNOWN"),
            (
                {"schema_id": "maintenance.list_quotes.v9"},
                "RETRIEVAL_RESULT_VOCABULARY_UNKNOWN",
            ),
            ({"batch_index": 7}, "RETRIEVAL_TURN_ORDER_BROKEN"),
        ],
    )
    def test_a_broken_row_is_named(self, reference_artifact, edit, code) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        served = [
            row
            for row in payload["trajectory"]
            if row["record_type"] == "retrieval_served"
        ]
        served[-1].update(edit)
        assert code in set(self._evaluate(payload).finding_codes), code

    def test_a_reordered_batch_is_named(self, reference_artifact) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        rows = payload["trajectory"]
        served = [
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "retrieval_served" and row["batch_index"] == 0
        ][:2]
        assert len(served) == 2
        first, second = served
        rows[first]["tool"], rows[second]["tool"] = (
            rows[second]["tool"],
            rows[first]["tool"],
        )
        codes = set(self._evaluate(payload).finding_codes)
        assert "RETRIEVAL_BATCH_NOT_CANONICAL" in codes

    def test_a_wrong_but_published_authority_is_named(self, reference_artifact) -> None:
        # Membership in the closed vocabulary is not the question. The question
        # is whether *this* read carries the authority the catalogue publishes
        # for it, and a quote relabelled as a communication passes the first
        # test while failing the second.
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "retrieval_served" and row["tool"] == "list_quotes":
                row["authority"] = "communication"
                break
        codes = set(self._evaluate(payload).finding_codes)
        assert "RETRIEVAL_RESULT_VOCABULARY_UNKNOWN" in codes

    def test_an_unknown_invalidation_cause_is_named(self, reference_artifact) -> None:
        payload = json.loads(json.dumps(reference_artifact))
        for row in payload["trajectory"]:
            if row["record_type"] == "derived_context_invalidated":
                row["cause"] = "because_i_said_so"
                break
        codes = set(self._evaluate(payload).finding_codes)
        assert "RETRIEVAL_INVALIDATION_ORDER_BROKEN" in codes
