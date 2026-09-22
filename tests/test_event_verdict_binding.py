"""The verdict a delivery claims and the verdict its ledger row records are one
fact, and the evaluator holds them to it.

An authored event that bounces off the operation is graded by *why* it bounced:
``environment_integrity`` treats a refusal as the authored, declared one when
the code matches the scenario's declaration, and as environment desync when it
does not. That makes the refusal code load-bearing evidence — and a record in
which the delivery says ``CHECKPOINT_NOT_OPEN`` while the ledger row for the
same event says ``UNKNOWN_VISIT`` is not one run described twice. It is two
runs, and exactly one of them satisfies the declaration.

So the tests below forge the code in both directions from a genuine V3 episode:
once in the ledger, once in the delivery. Both must fail record integrity as a
binding mismatch before environment semantics get to call anything expected,
because a scenario's declaration is a statement about a record that binds. The
post-terminal case is the same property under a different mechanism: an
``event_after_terminal`` row carries no code field, so the canonical
``AFTER_REPLAY_FINAL`` is asserted against the delivery instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from operatebench.core.engine import EpisodeOutcome
from operatebench.core.evaluation import Dimension, OperationEvaluation
from operatebench.core.events import VERDICT_AFTER_REPLAY_FINAL
from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.runner import execute_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

BINDING_FINDING = "CRITICAL_EVENT_LEDGER_BINDING_MISMATCH"
UNSOUND_FINDING = "ENVIRONMENT_RECORD_UNSOUND"

#: The V3 delivery this scenario declares as a legal refusal: an approval that
#: arrives after its checkpoint expired. Its declared code is what a forgery has
#: to be able to counterfeit for the declaration to be worth anything.
DECLARED_REJECTION = ("v3_e08", "CHECKPOINT_NOT_OPEN")

#: The V1 delivery that arrives after the operation is replay-final.
POST_TERMINAL_DELIVERY = "v1_e18"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


def _episode(spec: OperationSpec, scenario_id: str) -> EpisodeOutcome:
    return execute_episode(spec, scenario_id, "reference")


def _regrade(
    outcome: EpisodeOutcome,
    spec: OperationSpec,
    scenario_id: str,
    *,
    trajectory: Sequence[Mapping[str, Any]] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> OperationEvaluation:
    """Re-evaluate a genuine run against a record that may have been edited."""
    return evaluate(
        status=outcome.status,
        replay_final=outcome.replay_final,
        simulated_minutes=outcome.simulated_minutes,
        invocations=outcome.invocations,
        final_state=outcome.final_state,
        trajectory=[dict(row) for row in (trajectory or outcome.trajectory)],
        events=[dict(event) for event in (events or outcome.events)],
        scenario=spec.scenario(scenario_id),
        replay_ok=True,
    )


def _rows(outcome: EpisodeOutcome) -> list[dict[str, Any]]:
    return [dict(row) for row in outcome.trajectory]


def _events(outcome: EpisodeOutcome) -> list[dict[str, Any]]:
    return [dict(event) for event in outcome.events]


def _dimension(evaluation: OperationEvaluation, name: str) -> Dimension:
    """The named dimension, which every evaluation is required to report.

    A missing dimension is a failure of the result vector itself, not of the
    behaviour under test, so it fails here rather than surfacing as an
    attribute error three assertions later.
    """
    found = evaluation.dimension(name)
    assert found is not None, f"the evaluation reported no {name!r} dimension"
    return found


def _finding_codes(evaluation: OperationEvaluation, dimension: str) -> list[str]:
    return [finding.code for finding in _dimension(evaluation, dimension).findings]


class TestTheGenuineRecordBinds:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_an_unedited_reference_episode_is_reliable(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        outcome = _episode(spec, scenario_id)
        evaluation = _regrade(outcome, spec, scenario_id)
        assert evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )
        assert evaluation.reliable is (scenario_id != "V2")

    def test_every_delivery_carries_the_code_its_ledger_row_records(
        self, spec: OperationSpec
    ) -> None:
        # The property the forgeries below violate, stated positively over a
        # real run: for every recorded delivery there is one ledger row, and it
        # names the same verdict.
        outcome = _episode(spec, "V3")
        coded = {
            str(row["event_id"]): str(row["code"])
            for row in _rows(outcome)
            if row["record_type"]
            in {"event_observed", "event_audit_only", "event_rejected"}
        }
        for event in _events(outcome):
            if event["disposition"] == "post_terminal":
                assert event["verdict_code"] == VERDICT_AFTER_REPLAY_FINAL
                continue
            assert coded[str(event["event_id"])] == event["verdict_code"]

    def test_the_declared_refusal_is_the_one_the_ledger_recorded(
        self, spec: OperationSpec
    ) -> None:
        event_id, code = DECLARED_REJECTION
        outcome = _episode(spec, "V3")
        row = next(row for row in _rows(outcome) if row.get("event_id") == event_id)
        delivery = next(
            event for event in _events(outcome) if event["event_id"] == event_id
        )
        assert row["record_type"] == "event_rejected"
        assert row["code"] == code
        assert delivery["verdict_code"] == code
        assert spec.scenario("V3").expected_rejection_codes()[event_id] == code


class TestAForgedVerdictCodeIsAnUnsoundRecord:
    def test_a_forged_ledger_code_fails_reliability(self, spec: OperationSpec) -> None:
        # The ledger says the world was refused for a reason the delivery does
        # not report. Before this binding existed, the delivery's code alone
        # satisfied the scenario's declaration and the episode stayed reliable.
        event_id, code = DECLARED_REJECTION
        outcome = _episode(spec, "V3")
        trajectory = _rows(outcome)
        forged = next(
            row
            for row in trajectory
            if row.get("record_type") == "event_rejected"
            and row.get("event_id") == event_id
        )
        assert forged["code"] == code
        forged["code"] = "UNKNOWN_VISIT"

        evaluation = _regrade(outcome, spec, "V3", trajectory=trajectory)

        assert evaluation.reliable is False
        assert BINDING_FINDING in _finding_codes(evaluation, "critical_invariants")
        detail = next(
            finding.detail
            for finding in _dimension(evaluation, "critical_invariants").findings
            if finding.code == BINDING_FINDING
        )
        assert "verdict_code" in detail
        assert "UNKNOWN_VISIT" in detail and code in detail

    def test_a_forged_delivery_code_fails_reliability(self, spec: OperationSpec) -> None:
        # The other direction: the ledger is genuine and the delivery is
        # rewritten to claim the declared refusal. Binding is symmetric, so
        # neither half can be trusted to describe the other.
        event_id, code = DECLARED_REJECTION
        outcome = _episode(spec, "V3")
        events = _events(outcome)
        forged = next(event for event in events if event["event_id"] == event_id)
        assert forged["verdict_code"] == code
        forged["verdict_code"] = "UNKNOWN_VISIT"

        evaluation = _regrade(outcome, spec, "V3", events=events)

        assert evaluation.reliable is False
        assert BINDING_FINDING in _finding_codes(evaluation, "critical_invariants")

    def test_a_forged_delivery_code_cannot_manufacture_a_declared_refusal(
        self, spec: OperationSpec
    ) -> None:
        # The forgery with a motive: V1 declares no refusal for this event, so
        # rewriting a delivery to carry a code some *other* scenario declares
        # must not buy an expected rejection. It buys a binding mismatch.
        outcome = _episode(spec, "V1")
        events = _events(outcome)
        genuine = next(
            event for event in events if event["disposition"] == "post_terminal"
        )
        genuine["verdict_code"] = DECLARED_REJECTION[1]

        evaluation = _regrade(outcome, spec, "V1", events=events)

        assert evaluation.reliable is False
        assert BINDING_FINDING in _finding_codes(evaluation, "critical_invariants")
        environment = _dimension(evaluation, "environment_integrity")
        assert environment.ok is False
        assert environment.counts["expected_rejections"] == 0

    def test_an_unsound_record_is_not_graded_as_an_expected_rejection(
        self, spec: OperationSpec
    ) -> None:
        # Reliability has to fail *because the record does not bind*, not
        # incidentally: environment_integrity refuses to read refusal codes out
        # of a record whose two halves describe different runs.
        event_id, _ = DECLARED_REJECTION
        outcome = _episode(spec, "V3")
        trajectory = _rows(outcome)
        forged = next(
            row
            for row in trajectory
            if row.get("record_type") == "event_rejected"
            and row.get("event_id") == event_id
        )
        forged["code"] = "UNKNOWN_VISIT"

        evaluation = _regrade(outcome, spec, "V3", trajectory=trajectory)

        environment = _dimension(evaluation, "environment_integrity")
        assert environment.ok is False
        assert _finding_codes(evaluation, "environment_integrity") == [UNSOUND_FINDING]
        assert environment.counts["expected_rejections"] == 0
        assert environment.counts["declared_rejections"] == 1
        assert set(evaluation.failed_dimensions) == {
            "critical_invariants",
            "environment_integrity",
        }


class TestThePostTerminalVerdictIsCanonical:
    def test_the_engine_and_the_evaluator_name_it_the_same_thing(
        self, spec: OperationSpec
    ) -> None:
        outcome = _episode(spec, "V1")
        delivery = next(
            event
            for event in _events(outcome)
            if event["event_id"] == POST_TERMINAL_DELIVERY
        )
        assert delivery["disposition"] == "post_terminal"
        assert delivery["verdict_code"] == VERDICT_AFTER_REPLAY_FINAL

    def test_the_ledger_row_carries_no_code_to_bind_against(
        self, spec: OperationSpec
    ) -> None:
        # Why the canonical constant exists: this row's *record type* is the
        # verdict, so there is no code field for the delivery to agree with.
        outcome = _episode(spec, "V1")
        row = next(
            row for row in _rows(outcome) if row.get("event_id") == POST_TERMINAL_DELIVERY
        )
        assert row["record_type"] == "event_after_terminal"
        assert "code" not in row

    def test_a_forged_post_terminal_verdict_fails_reliability(
        self, spec: OperationSpec
    ) -> None:
        outcome = _episode(spec, "V1")
        events = _events(outcome)
        forged = next(
            event for event in events if event["event_id"] == POST_TERMINAL_DELIVERY
        )
        forged["verdict_code"] = "OPERATION_IS_REPLAY_FINAL"

        evaluation = _regrade(outcome, spec, "V1", events=events)

        assert evaluation.reliable is False
        assert BINDING_FINDING in _finding_codes(evaluation, "critical_invariants")


class TestACodeNeitherHalfCarriesIsNotAgreement:
    """Equality is not the property; carrying the verdict is.

    ``None == None`` and ``"" == ""``, so a record that dropped the verdict from
    the ledger row *and* from the delivery bound cleanly and graded on. The
    binding is only worth anything because ``environment_integrity`` reads that
    code to decide whether a refusal is the authored, declared one — and it
    cannot read a code that is not there. The artefact reader rejects such a
    record at the file boundary, but the evaluator is also handed records built
    in memory that never passed through it, so the standard is enforced here as
    well.
    """

    @pytest.mark.parametrize("mode", ["removed", "empty", "null", "not-a-string"])
    def test_a_verdict_absent_from_both_halves_is_a_binding_mismatch(
        self, spec: OperationSpec, mode: str
    ) -> None:
        event_id, _ = DECLARED_REJECTION
        outcome = _episode(spec, "V3")
        trajectory = _rows(outcome)
        events = _events(outcome)
        row = next(
            row
            for row in trajectory
            if row.get("record_type") == "event_rejected"
            and row.get("event_id") == event_id
        )
        delivery = next(event for event in events if event["event_id"] == event_id)
        if mode == "removed":
            del row["code"]
            del delivery["verdict_code"]
        else:
            value: Any = {"empty": "", "null": None, "not-a-string": 0}[mode]
            row["code"] = value
            delivery["verdict_code"] = value

        evaluation = _regrade(outcome, spec, "V3", trajectory=trajectory, events=events)

        assert evaluation.reliable is False
        assert BINDING_FINDING in _finding_codes(evaluation, "critical_invariants")
        # And the scenario's declaration is not satisfied by the silence: an
        # unsound record is not graded for environment semantics at all.
        assert UNSOUND_FINDING in _finding_codes(evaluation, "environment_integrity")

    def test_a_post_terminal_delivery_still_binds_to_the_canonical_code(
        self, spec: OperationSpec
    ) -> None:
        # The one record type whose row carries no code is unaffected: its
        # verdict is asserted against the canonical constant, which is never
        # absent, so the stricter rule does not turn a legal record unsound.
        outcome = _episode(spec, "V1")
        evaluation = _regrade(outcome, spec, "V1")
        assert evaluation.reliable is True
        delivery = next(
            event
            for event in _events(outcome)
            if event["event_id"] == POST_TERMINAL_DELIVERY
        )
        assert delivery["verdict_code"] == VERDICT_AFTER_REPLAY_FINAL


class TestADeclaredRejectionIsEvidenceNotAnObligation:
    def test_a_declaration_that_does_not_fire_is_not_a_failure(
        self, spec: OperationSpec
    ) -> None:
        """The documented reading, asserted rather than assumed.

        ``expected_event_rejections`` says which refusals are *authored* — legal,
        and not desync. It does not say they must occur. Grading a declaration
        that never fired as a failure would make the fixture's account of what
        may bounce into a demand that it bounce, and an agent that legitimately
        kept an event out of that state would fail for succeeding. The counts
        report both numbers so the difference stays visible.
        """
        outcome = _episode(spec, "V3")
        events = [
            event
            for event in _events(outcome)
            if event["event_id"] != DECLARED_REJECTION[0]
        ]
        trajectory = [
            row for row in _rows(outcome) if row.get("event_id") != DECLARED_REJECTION[0]
        ]
        for index, row in enumerate(trajectory):
            row["index"] = index

        evaluation = _regrade(outcome, spec, "V3", trajectory=trajectory, events=events)

        environment = _dimension(evaluation, "environment_integrity")
        assert environment.ok is True
        assert environment.counts["declared_rejections"] == 1
        assert environment.counts["expected_rejections"] == 0
        assert environment.counts["unexpected_rejections"] == 0
