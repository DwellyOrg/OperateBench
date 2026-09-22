"""Phase 2, part 6: the result vector is a contract, and it is read as one.

The reader held every *field* of the evaluation to its type and held none of its
*claims* to anything. So a document could name eight dimensions, or twelve, or
the same one twice, or the eleven in an order no build ever wrote; it could say
``failed_dimensions: []`` beside a dimension that did not hold; it could carry a
finding the flattened code list never mentions; it could call itself ``reliable``
while contradicting the conjunction that word is defined as. Every one of those
is a record that describes no run, and each was accepted.

They are refused here, version by version. The dimension vocabulary moved twice —
``environment_integrity`` at contract 2, ``retrieval_discipline`` at contract 6 —
so "exactly these names, in this order" is a different sentence per contract, and
a reader that enforced one of them at every version would either retroactively
invalidate v1 or let a v5 document claim a dimension its build never graded.

What is deliberately *not* asserted: that a finding forces its dimension to fail.
:class:`~operatebench.core.evaluation.OperationEvaluation` states no such rule —
``ok`` and ``findings`` are independent fields of :class:`Dimension` — and a
reader that invented it would be enforcing a contract the writer does not keep.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import pytest

from operatebench.artifact import (
    ARTIFACT_VERSION,
    ARTIFACT_VERSION_V1,
    ARTIFACT_VERSION_V5,
    SUPPORTED_ARTIFACT_VERSIONS,
    ArtifactError,
    evaluation_dimension_coverage_problem,
    evaluation_dimensions_for,
    read_artifact,
    validate_artifact,
    write_artifact,
)
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode
from tests.artifact_shapes import as_version_shape

SPEC = "examples/operatebench/maintenance_v0_1.yaml"
FROZEN_V1 = "tests/fixtures/artifact_v1_reference_V1.json"

#: The nine dimensions contract 1 was written under, restated rather than
#: derived. The frozen fixture is the evidence for this tuple, and deriving it
#: from the current vector would make the assertion circular.
DIMENSIONS_V1 = (
    "terminal_outcome",
    "critical_invariants",
    "temporal_correctness",
    "authority_boundaries",
    "action_validity",
    "human_checkpoints",
    "recovery",
    "obligations",
    "deterministic_replay",
)

DIMENSIONS_V2_TO_V5 = (*DIMENSIONS_V1, "environment_integrity")


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


@pytest.fixture(scope="module")
def reference(spec, tmp_path_factory) -> dict[str, Any]:
    run = run_episode(spec, "V1", "reference")
    path = tmp_path_factory.mktemp("evaluation") / "reference.json"
    return read_artifact(write_artifact(run, path))


@pytest.fixture(scope="module")
def negative(spec, tmp_path_factory) -> dict[str, Any]:
    """A record whose vector actually fails, with more than one finding code."""
    run = run_episode(spec, "V1", "trust_actor_claim")
    path = tmp_path_factory.mktemp("evaluation") / "trust_actor_claim.json"
    return read_artifact(write_artifact(run, path))


def _copy(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _refusal(payload: dict[str, Any]) -> str:
    with pytest.raises(ArtifactError) as caught:
        validate_artifact(payload)
    return str(caught.value)


def _as_version(payload: dict[str, Any], version: int) -> dict[str, Any]:
    """A record projected onto ``version``: field set, rows, vector and digest.

    The suite's own projection, not a restatement of it here: a record whose
    fields are contract 6's and whose ``artifact_version`` merely says 5 is
    refused for the fields, and would never reach the question this class asks.
    """
    return as_version_shape(_copy(payload), version)


class TestTheDimensionContractIsDescribedAtEveryVersion:
    def test_the_current_contract_names_the_evaluator_vector(self) -> None:
        assert evaluation_dimensions_for(ARTIFACT_VERSION) == DIMENSIONS

    def test_contract_one_names_the_nine_the_frozen_fixture_carries(self) -> None:
        assert evaluation_dimensions_for(ARTIFACT_VERSION_V1) == DIMENSIONS_V1

    @pytest.mark.parametrize("version", [2, 3, 4, 5])
    def test_contracts_two_to_five_name_ten(self, version: int) -> None:
        assert evaluation_dimensions_for(version) == DIMENSIONS_V2_TO_V5

    def test_every_readable_version_is_described(self) -> None:
        assert evaluation_dimension_coverage_problem() is None
        for version in SUPPORTED_ARTIFACT_VERSIONS:
            assert evaluation_dimensions_for(version)

    def test_each_contract_only_ever_appended(self) -> None:
        ordered = sorted(SUPPORTED_ARTIFACT_VERSIONS)
        for earlier, later in pairwise(ordered):
            before = evaluation_dimensions_for(earlier)
            after = evaluation_dimensions_for(later)
            assert after[: len(before)] == before

    def test_an_unknown_version_is_named_rather_than_guessed(self) -> None:
        with pytest.raises(ArtifactError):
            evaluation_dimensions_for(99)


class TestTheDimensionSetIsExact:
    def test_a_missing_dimension_is_refused(self, reference) -> None:
        body = _copy(reference)
        body["evaluation"]["dimensions"] = [
            entry
            for entry in body["evaluation"]["dimensions"]
            if entry["name"] != "retrieval_discipline"
        ]
        message = _refusal(body)
        assert "evaluation.dimensions" in message
        assert "retrieval_discipline" in message

    def test_an_unknown_dimension_is_refused(self, reference) -> None:
        body = _copy(reference)
        body["evaluation"]["dimensions"].append(
            {"name": "invented", "ok": True, "findings": [], "counts": {}, "note": ""}
        )
        message = _refusal(body)
        assert "evaluation.dimensions" in message
        assert "invented" in message

    def test_a_duplicated_dimension_is_refused(self, reference) -> None:
        body = _copy(reference)
        entries = body["evaluation"]["dimensions"]
        entries.insert(1, _copy(entries[0]))
        assert "evaluation.dimensions" in _refusal(body)

    def test_a_reordered_vector_is_refused(self, reference) -> None:
        body = _copy(reference)
        entries = body["evaluation"]["dimensions"]
        entries[0], entries[1] = entries[1], entries[0]
        message = _refusal(body)
        assert "evaluation.dimensions" in message

    def test_the_reference_vector_itself_still_validates(self, reference) -> None:
        assert validate_artifact(_copy(reference))["evaluation"]["dimensions"]


class TestTheDerivedFieldsAreDerived:
    def test_a_failing_dimension_must_appear_in_failed_dimensions(
        self, reference
    ) -> None:
        body = _copy(reference)
        vector = body["evaluation"]
        vector["dimensions"][2]["ok"] = False
        vector["reliable"] = False
        message = _refusal(body)
        assert "evaluation.failed_dimensions" in message

    def test_failed_dimensions_may_not_name_a_passing_dimension(self, reference) -> None:
        body = _copy(reference)
        body["evaluation"]["failed_dimensions"] = ["recovery"]
        assert "evaluation.failed_dimensions" in _refusal(body)

    def test_failed_dimensions_is_held_to_dimension_order(self, reference) -> None:
        body = _copy(reference)
        vector = body["evaluation"]
        for entry in vector["dimensions"]:
            if entry["name"] in ("action_validity", "recovery"):
                entry["ok"] = False
        vector["reliable"] = False
        vector["failed_dimensions"] = ["recovery", "action_validity"]
        assert "evaluation.failed_dimensions" in _refusal(body)

    def test_the_negative_control_vector_is_self_consistent(self, negative) -> None:
        vector = negative["evaluation"]
        assert vector["failed_dimensions"]
        assert len(vector["finding_codes"]) >= 2
        assert validate_artifact(_copy(negative))["evaluation"] == vector

    def test_finding_codes_may_not_gain_a_code(self, negative) -> None:
        body = _copy(negative)
        body["evaluation"]["finding_codes"].append("INVENTED_CODE")
        assert "evaluation.finding_codes" in _refusal(body)

    def test_finding_codes_may_not_drop_a_code(self, negative) -> None:
        body = _copy(negative)
        body["evaluation"]["finding_codes"].pop()
        assert "evaluation.finding_codes" in _refusal(body)

    def test_finding_codes_is_held_to_dimension_and_finding_order(self, negative) -> None:
        body = _copy(negative)
        codes = body["evaluation"]["finding_codes"]
        assert len(set(codes)) >= 2
        body["evaluation"]["finding_codes"] = list(reversed(codes))
        assert "evaluation.finding_codes" in _refusal(body)

    def test_reliable_may_not_be_claimed_over_a_failing_dimension(self, negative) -> None:
        body = _copy(negative)
        body["evaluation"]["reliable"] = True
        assert "evaluation.reliable" in _refusal(body)

    def test_reliable_may_not_be_claimed_over_an_illegitimate_completion(
        self, reference
    ) -> None:
        body = _copy(reference)
        body["evaluation"]["legitimate_completion"] = False
        assert "evaluation.reliable" in _refusal(body)

    def test_reliable_may_not_be_withheld_from_a_vector_that_holds(
        self, reference
    ) -> None:
        body = _copy(reference)
        body["evaluation"]["reliable"] = False
        assert "evaluation.reliable" in _refusal(body)

    def test_a_finding_beside_a_passing_dimension_is_not_invented_into_a_failure(
        self, reference
    ) -> None:
        # The writer's contract does not say a finding forces ``ok`` false, so
        # neither does the reader. If ``OperationEvaluation`` ever states it,
        # this test is the one to change — deliberately, not by accident.
        body = _copy(reference)
        vector = body["evaluation"]
        vector["dimensions"][0]["findings"] = [
            {
                "code": "NOTED",
                "detail": "recorded beside a dimension that held",
                "at": None,
            }
        ]
        vector["finding_codes"] = ["NOTED"]
        assert validate_artifact(body)["evaluation"]["finding_codes"] == ["NOTED"]


class TestTheVectorIsVersionScoped:
    def test_contract_five_refuses_the_dimension_contract_six_added(
        self, reference
    ) -> None:
        body = _as_version(reference, ARTIFACT_VERSION_V5)
        vector = body["evaluation"]
        vector["dimensions"] = _copy(reference)["evaluation"]["dimensions"]
        message = _refusal(body)
        assert "evaluation.dimensions" in message
        assert "retrieval_discipline" in message

    def test_a_five_shaped_vector_reads_at_five(self, reference) -> None:
        body = _as_version(reference, ARTIFACT_VERSION_V5)
        names = [entry["name"] for entry in body["evaluation"]["dimensions"]]
        assert tuple(names) == DIMENSIONS_V2_TO_V5
        assert validate_artifact(body)["artifact_version"] == ARTIFACT_VERSION_V5

    def test_contract_six_requires_the_dimension_a_five_vector_lacks(
        self, reference
    ) -> None:
        body = _copy(reference)
        body["evaluation"] = _as_version(reference, ARTIFACT_VERSION_V5)["evaluation"]
        message = _refusal(body)
        assert "evaluation.dimensions" in message
        assert "retrieval_discipline" in message

    def test_a_five_record_may_not_duplicate_a_dimension_it_did_grade(
        self, reference
    ) -> None:
        body = _as_version(reference, ARTIFACT_VERSION_V5)
        entries = body["evaluation"]["dimensions"]
        entries.append(_copy(entries[-1]))
        message = _refusal(body)
        assert "evaluation.dimensions" in message
        assert "environment_integrity" in message

    def test_the_frozen_v1_record_still_reads(self) -> None:
        payload = read_artifact(FROZEN_V1)
        names = [entry["name"] for entry in payload["evaluation"]["dimensions"]]
        assert tuple(names) == DIMENSIONS_V1
        assert payload["evaluation"]["reliable"] is True

    def test_the_frozen_v1_record_may_not_gain_a_later_dimension(self) -> None:
        body = _copy(read_artifact(FROZEN_V1))
        body["evaluation"]["dimensions"].append(
            {
                "name": "environment_integrity",
                "ok": True,
                "findings": [],
                "counts": {},
                "note": "",
            }
        )
        assert "evaluation.dimensions" in _refusal(body)
