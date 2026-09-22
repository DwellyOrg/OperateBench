"""Forged retrieval evidence, caught by the evaluator and by nothing else.

Every test here edits a trajectory and hands it **straight to**
:func:`~operatebench.domains.lettings.maintenance.evaluator.evaluate`. Nothing
runs the artefact reader, nothing replays, and ``replay_ok=True`` is passed in —
so a finding that appears here was derived by the evaluator from the record
alone. That is the whole point: an external trajectory arriving from somewhere
this build did not run is exactly the document the engine's own guards say
nothing about, and "the artefact validator would have refused it" is a claim
about a file this evaluator was never handed.

The forgeries are the ones a plausible fabricator would reach for: an index
moved to another *real* value rather than out of range, a provenance field
swapped for another *published* one, a bookkeeping row deleted or moved. A
range check would pass every one of them.

What is deliberately **not** claimed: that the grammar can tell whether the
recorded values were historically true. A row proves what it binds. A records
body that is the published projection in shape, with a version that is the
digest of exactly that body, is internally consistent — and
:class:`~operatebench.artifact.RetrievalProvenanceMismatchError` on replay, not
this grammar, is what compares it against the live operation. The last test in
this file states that limit rather than leaving it implied.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from operatebench.artifact import build_artifact
from operatebench.core.retrieval_evidence import public_record_version
from operatebench.domains.lettings.maintenance.evaluator import (
    RETRIEVAL_GRAMMAR_CODES,
    evaluate,
)
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    RECORD_VERSION_CONTEXT,
    RECORD_VERSION_LENGTH,
    record_identity,
    record_version,
)
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode

SPEC = "examples/operatebench/maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


@pytest.fixture(scope="module")
def reference_artifact(spec):
    return build_artifact(run_episode(spec, "V1", "reference"))


def _detached(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _reindexed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber a trajectory after rows were moved or removed.

    Contiguity is checked separately and by name. Leaving a hole here would let
    a forgery be caught as a gap in the index sequence, which proves the reader
    counts and not that it binds.
    """
    for position, row in enumerate(rows):
        row["index"] = position
    return rows


def _evaluate(payload: Mapping[str, Any], scenario_id: str = "V1"):
    return evaluate(
        status=payload["status"],
        replay_final=payload["replay_final"],
        simulated_minutes=payload["simulated_minutes"],
        invocations=payload["agent_invocations"],
        final_state=payload["final_state"],
        trajectory=payload["trajectory"],
        events=payload["events"],
        scenario=load_spec(SPEC).scenario(scenario_id),
        # The evaluator is told the replay agreed. Every finding below is
        # therefore the evaluator's own, not a replay result wearing its name.
        replay_ok=True,
    )


def _rows(payload: Mapping[str, Any], record_type: str) -> list[dict[str, Any]]:
    return [row for row in payload["trajectory"] if row["record_type"] == record_type]


def _served_for(payload: Mapping[str, Any], tool: str) -> dict[str, Any]:
    for row in _rows(payload, "retrieval_served"):
        if row["tool"] == tool:
            return row
    raise AssertionError(f"the reference run served no {tool!r} row")


def _codes(payload: Mapping[str, Any], scenario_id: str = "V1") -> set[str]:
    return set(_evaluate(payload, scenario_id).finding_codes)


def _assert_caught(payload: Mapping[str, Any], code: str) -> None:
    evaluation = _evaluate(payload)
    assert code in set(evaluation.finding_codes), (
        f"the evaluator did not name {code}; it named "
        f"{sorted(set(evaluation.finding_codes))}"
    )
    assert not evaluation.reliable, "a forged retrieval record is not a reliable run"


# -- the clean reference stays clean ----------------------------------------


class TestTheCleanReferenceIsUntouched:
    def test_a_clean_reference_run_is_reliable_and_states_no_grammar_finding(
        self, reference_artifact
    ) -> None:
        evaluation = _evaluate(_detached(reference_artifact))
        assert evaluation.reliable
        assert not set(evaluation.finding_codes) & set(RETRIEVAL_GRAMMAR_CODES)

    def test_the_grammar_vocabulary_is_closed_and_sorted(self) -> None:
        assert tuple(sorted(set(RETRIEVAL_GRAMMAR_CODES))) == RETRIEVAL_GRAMMAR_CODES
        assert all(code.startswith("RETRIEVAL_") for code in RETRIEVAL_GRAMMAR_CODES)


# -- indices moved to other real values -------------------------------------


class TestAnIndexMovedToAnotherRealValueIsCaught:
    def test_a_served_turn_index_edited_inside_its_invocation_is_named(
        self, reference_artifact
    ) -> None:
        # Turn 1 is a real call of this invocation — the business outcome that
        # followed the batch — so a range check over the invocation's turns
        # accepts this edit. It is still a batch attributed to a call that
        # served nothing.
        payload = _detached(reference_artifact)
        for row in _rows(payload, "retrieval_served"):
            if row["invocation_index"] == 1 and row["batch_index"] == 0:
                row["turn_index"] = 1
        _assert_caught(payload, "RETRIEVAL_TURN_BINDING_BROKEN")

    def test_a_refusal_turn_index_that_names_no_call_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        anchor = next(
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "retrieval_served"
            and row["invocation_index"] == 1
            and row["batch_index"] == 0
        )
        served = rows[anchor]
        rows.insert(
            anchor,
            {
                "index": 0,
                "at": served["at"],
                "record_type": "retrieval_refused",
                "invocation_index": 1,
                # Turn 4 never happened in this invocation, but it is a turn
                # index this build writes, and it is below every published bound.
                "turn_index": 4,
                "request_count": 1,
                "requested_tools": ["list_quotes"],
                "code": "RETRIEVAL_BUDGET_EXHAUSTED",
                "detail": "forged",
            },
        )
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_TURN_BINDING_BROKEN")

    def test_a_served_invocation_index_moved_to_another_real_invocation_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        for row in _rows(payload, "retrieval_served"):
            if row["invocation_index"] == 1 and row["batch_index"] == 0:
                row["invocation_index"] = 2
        _assert_caught(payload, "RETRIEVAL_INVOCATION_BINDING_BROKEN")

    def test_an_invalidation_invocation_index_moved_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        for row in _rows(payload, "derived_context_invalidated"):
            if row["invocation_index"] == 3:
                row["invocation_index"] = 2
                break
        _assert_caught(payload, "RETRIEVAL_INVOCATION_BINDING_BROKEN")


# -- provenance swapped for another published value -------------------------


class TestProvenanceSwappedForAnotherPublishedValueIsCaught:
    def test_a_known_authority_that_is_not_this_tools_authority_is_named(
        self, reference_artifact
    ) -> None:
        # ``communication`` is a class this build writes — for
        # ``list_communications``. Claiming it for the system-of-record read is
        # a claim that a quote came from a message, and a membership test in
        # the closed authority vocabulary accepts it.
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        assert row["authority"] == "system_of_record"
        row["authority"] = "communication"
        _assert_caught(payload, "RETRIEVAL_RESULT_VOCABULARY_UNKNOWN")

    def test_a_record_id_taken_from_another_published_read_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        row["record_id"] = record_identity("messaging_service")
        _assert_caught(payload, "RETRIEVAL_RECORD_IDENTITY_UNKNOWN")

    def test_the_published_record_identity_is_a_deterministic_formula(self) -> None:
        # The identity the row is held to is derived, not trusted: it is a
        # function of the source the catalogue publishes for the tool, and this
        # states the function independently of any run.
        for spec in MAINTENANCE_RETRIEVAL_TOOLS.values():
            assert spec.record_id == record_identity(spec.source)
            assert spec.record_id == f"{spec.source}:maintenance_case"
        identities = {spec.record_id for spec in MAINTENANCE_RETRIEVAL_TOOLS.values()}
        assert len(identities) == len(MAINTENANCE_RETRIEVAL_TOOLS)


# -- the version is recomputed, never trusted -------------------------------


class TestTheVersionIsRecomputedFromTheRowsOwnRecords:
    def test_a_served_row_carries_the_records_it_returned(
        self, reference_artifact
    ) -> None:
        row = _served_for(_detached(reference_artifact), "list_quotes")
        assert isinstance(row["records"], dict)
        assert set(row["records"]) == set(
            MAINTENANCE_RETRIEVAL_TOOLS["list_quotes"].roots
        )

    def test_an_edited_version_over_untouched_records_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        row["record_version"] = "f" * RECORD_VERSION_LENGTH
        _assert_caught(payload, "RETRIEVAL_RECORD_VERSION_MISMATCH")

    def test_edited_records_under_an_untouched_version_are_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        assert row["records"]["quotes"] != {"forged-quote#1": {}}
        row["records"]["quotes"] = {"forged-quote#1": {}}
        _assert_caught(payload, "RETRIEVAL_RECORD_VERSION_MISMATCH")

    def test_the_recomputation_routes_through_the_neutral_helper(self) -> None:
        # The evaluator must not reach into the domain broker for the number it
        # is checking. Both sides call one neutral implementation, and this
        # states that they agree without either being the authority.
        body = {"quotes": {"q-1#1": {"status": "OPEN"}}}
        assert record_version(body) == public_record_version(
            body, context=RECORD_VERSION_CONTEXT, length=RECORD_VERSION_LENGTH
        )
        assert len(record_version(body)) == RECORD_VERSION_LENGTH


# -- a records body forged coherently is still held to the published shape ---


class TestForgedRecordsAreHeldToThePublishedProjection:
    def _forge(
        self, payload: Mapping[str, Any], tool: str, records: Mapping[str, Any]
    ) -> None:
        """Rewrite one row's records **and** its version, coherently."""
        row = _served_for(payload, tool)
        row["records"] = dict(records)
        row["record_version"] = record_version(row["records"])

    def test_an_extra_root_is_named(self, reference_artifact) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        forged = dict(row["records"])
        forged["invoices"] = {}
        self._forge(payload, "list_quotes", forged)
        _assert_caught(payload, "RETRIEVAL_RECORDS_NOT_THE_PUBLISHED_PROJECTION")

    def test_a_missing_root_is_named(self, reference_artifact) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_billing")
        forged = dict(row["records"])
        forged.pop("payment")
        self._forge(payload, "list_billing", forged)
        _assert_caught(payload, "RETRIEVAL_RECORDS_NOT_THE_PUBLISHED_PROJECTION")

    def test_a_root_of_the_wrong_json_type_is_named(self, reference_artifact) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        forged = dict(row["records"])
        forged["quotes"] = []
        self._forge(payload, "list_quotes", forged)
        _assert_caught(payload, "RETRIEVAL_RECORDS_NOT_THE_PUBLISHED_PROJECTION")

    def test_a_records_body_that_is_not_a_mapping_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        row = _served_for(payload, "list_quotes")
        row["records"] = ["quotes"]
        _assert_caught(payload, "RETRIEVAL_RECORDS_NOT_THE_PUBLISHED_PROJECTION")

    def test_what_a_coherent_records_forgery_is_not_claimed_to_catch(
        self, reference_artifact
    ) -> None:
        """The stated limit, executable rather than promised.

        A value edited inside a root that keeps the published shape, with the
        version recomputed over it, is a self-consistent row. The grammar reads
        it as one, because that is all the row can prove about itself. Replay
        against the live operation is what disagrees, and this asserts the
        grammar does not pretend otherwise.
        """
        payload = _detached(reference_artifact)
        row = _served_for(payload, "get_case_record")
        forged = dict(row["records"])
        forged["issue_id"] = "issue-that-never-existed"
        self._forge(payload, "get_case_record", forged)
        assert not _codes(payload) & set(RETRIEVAL_GRAMMAR_CODES)


# -- bookkeeping rows removed or moved --------------------------------------


class TestInvalidationRowsAreBoundToWhatCausedThem:
    def test_removing_the_first_wake_invalidation_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        first = next(
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "wake"
        )
        rows.pop(first)
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_removing_a_later_wake_invalidation_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        wakes = [
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "wake"
        ]
        assert len(wakes) > 2
        rows.pop(wakes[2])
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_a_second_wake_invalidation_in_one_invocation_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        first = next(
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "wake"
        )
        rows.insert(first + 1, _detached(rows[first]))
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_an_effect_invalidation_moved_before_its_effect_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        moved = next(
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "effect_accepted"
        )
        assert rows[moved - 1]["record_type"] == "effect_accepted"
        rows[moved - 1], rows[moved] = rows[moved], rows[moved - 1]
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_removing_an_effect_invalidation_is_named(self, reference_artifact) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        first = next(
            position
            for position, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "effect_accepted"
        )
        rows.pop(first)
        _reindexed(rows)
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_an_effect_invalidation_naming_another_accepted_action_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        position = next(
            index
            for index, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "effect_accepted"
        )
        # ``request_approval`` is an action this operation really accepts, one
        # invocation later. The row is well-formed and every value in it is one
        # this build writes.
        assert rows[position]["action_type"] != "request_approval"
        rows[position]["action_type"] = "request_approval"
        _assert_caught(payload, "RETRIEVAL_INVALIDATION_NOT_BOUND_TO_ITS_CAUSE")

    def test_an_effect_invalidation_turn_moved_inside_its_invocation_is_named(
        self, reference_artifact
    ) -> None:
        payload = _detached(reference_artifact)
        rows = payload["trajectory"]
        position = next(
            index
            for index, row in enumerate(rows)
            if row["record_type"] == "derived_context_invalidated"
            and row["cause"] == "effect_accepted"
        )
        assert rows[position]["turn_index"] == 1
        rows[position]["turn_index"] = 0
        _assert_caught(payload, "RETRIEVAL_TURN_BINDING_BROKEN")


# -- the same evidence, on every shipped variant ----------------------------


class TestTheGrammarHoldsOnEveryShippedVariant:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_a_reference_run_of_each_variant_states_no_grammar_finding(
        self, spec, scenario_id: str
    ) -> None:
        payload = build_artifact(run_episode(spec, scenario_id, "reference"))
        evaluation = _evaluate(_detached(payload), scenario_id)
        assert not set(evaluation.finding_codes) & set(RETRIEVAL_GRAMMAR_CODES)

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_a_forged_version_is_caught_on_each_variant(
        self, spec, scenario_id: str
    ) -> None:
        payload = _detached(build_artifact(run_episode(spec, scenario_id, "reference")))
        served: Sequence[dict[str, Any]] = _rows(payload, "retrieval_served")
        assert served
        served[-1]["record_version"] = "0" * RECORD_VERSION_LENGTH
        assert "RETRIEVAL_RECORD_VERSION_MISMATCH" in _codes(payload, scenario_id)
