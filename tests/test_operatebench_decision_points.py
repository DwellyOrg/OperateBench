"""Reaching a decision point through the real engine, and grading one outcome.

The harness has to be boring in three specific ways, and each one is asserted
here rather than described:

* it reaches the state through the **real** engine and the **real** domain, with
  no provider, no socket and no fixture of a state that was written by hand;
* what it freezes is the public observation projection, unchanged, detached and
  immutable — the same object an evaluated agent would have been handed at that
  instant, and nothing about what happens next;
* it grades exactly one returned outcome against a closed admissible set and
  says, by name, why it was or was not admissible.

Two **recorded gaps** live at the bottom of this file. The shipped admissible
sets are a project-authored draft, and at two of the three decision points the
draft disagrees with the operation's own deterministic reference behaviour. The
disagreements are asserted as they stand rather than repaired by widening the
draft: a control that is edited until it agrees with the thing it grades has
stopped being a control, and the gaps are exactly the questions an independent
author has to answer.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from operatebench.controls import build_match_manifest, match_manifest_path, require_match
from operatebench.core.outcomes import (
    Act,
    Ask,
    Complete,
    Escalate,
    Wait,
    outcome_as_dict,
)
from operatebench.decision_points import (
    ACTION_TYPE_NOT_ADMISSIBLE,
    ADMISSIBLE,
    GRADED_OUTCOMES_PER_DECISION_POINT,
    GRADED_UNIT,
    MALFORMED_OUTCOME,
    NOT_AN_OUTCOME,
    OUTCOME_FIELDS_NOT_ADMISSIBLE,
    OUTCOME_KIND_NOT_ADMISSIBLE,
    RECORD_COLLECTIONS,
    SHARED_ESTIMAND,
    Constraint,
    DecisionPoint,
    DecisionPointUnreachedError,
    QuantifiedClause,
    RecordCondition,
    StatePredicate,
    decide_at,
    grade_outcome,
    observation_facts,
    observation_projection_fields,
    reach_decision_point,
)
from operatebench.domains.lettings.maintenance.agents import agent_ids, build_agent
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.domains.lettings.maintenance.state import (
    Approval,
    Cycle,
    ExceptionCheckpoint,
    Invoice,
    Obligation,
    Quote,
)
from tests.matched_control_fixtures import (
    test_only_current_validator_payload as _test_only_current_validator_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

QUOTE_POINT = "over_threshold_quote_with_no_approval_on_record"
INVOICE_POINT = "bill_arrived_before_any_verified_work"
LAPSED_POINT = "approval_window_closed_without_a_decision"

SCOPE = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"


@pytest.fixture(scope="module")
def spec():
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def manifest(spec):
    historical = yaml.safe_load(match_manifest_path().read_text(encoding="utf-8"))
    payload = _test_only_current_validator_payload(historical, spec)
    return build_match_manifest(payload, "test-only-current-decision-point-grammar")


@pytest.fixture(scope="module")
def encounters(spec, manifest):
    require_match(manifest, spec)
    return {
        point.decision_point_id: (point, reach_decision_point(spec, point))
        for point in manifest.decision_points
    }


def _admissible_request_approval(bindings) -> Act:
    quote = bindings["target_quote"]
    return Act(
        action_type="request_approval",
        payload={
            "quote_id": quote["quote_id"],
            "quote_version": quote["version"],
            "amount_minor": quote["amount_minor"],
            "currency": quote["currency"],
            "scope_digest": quote["scope_digest"],
            "cycle_id": quote["cycle_id"],
            "deadline_after_minutes": 4320,
        },
        rationale="the amount is above the authored threshold",
    )


class TestReachingTheStateThroughTheRealEngine:
    def test_every_decision_point_is_reached(self, encounters) -> None:
        assert set(encounters) == {QUOTE_POINT, INVOICE_POINT, LAPSED_POINT}

    def test_the_predicate_holds_at_the_frozen_observation(self, encounters) -> None:
        for point, encounter in encounters.values():
            match = point.state_predicate.evaluate(encounter.observation_payload())
            assert match.matched, (point.decision_point_id, match.reason)

    def test_the_state_is_reached_not_authored(self, encounters) -> None:
        _, encounter = encounters[QUOTE_POINT]
        quote = encounter.bindings["target_quote"]
        assert quote["quote_id"] == "quote_1"
        assert quote["amount_minor"] == 64000
        assert observation_facts(encounter.observation_payload())["phase"] in {
            "ACTIVE",
            "WAITING_VISIT",
            "WAITING_APPROVAL",
        }

    def test_reaching_it_twice_reaches_the_same_instant(self, spec, manifest) -> None:
        point = manifest.decision_point(LAPSED_POINT)
        first = reach_decision_point(spec, point)
        second = reach_decision_point(spec, point)
        assert first.observation_digest_sha256 == second.observation_digest_sha256
        assert (first.now, first.invocation_index, first.turn_index) == (
            second.now,
            second.invocation_index,
            second.turn_index,
        )

    def test_a_predicate_nothing_reaches_is_an_error_not_a_silent_pass(
        self, spec, manifest
    ) -> None:
        unreachable = DecisionPoint(
            decision_point_id="a_state_this_scenario_never_enters",
            scenario_id="V1",
            label="two priced proposals are open at once",
            state_predicate=StatePredicate(
                clauses=(
                    QuantifiedClause(
                        collection="quotes",
                        quantifier="exists",
                        where=(
                            RecordCondition(field="status", op="eq", value="REJECTED"),
                        ),
                    ),
                )
            ),
            observable_facts=(),
            evidence_provenance=(),
            admissible_outcomes=(),
        )
        with pytest.raises(DecisionPointUnreachedError, match="never held"):
            reach_decision_point(spec, unreachable)

    def test_the_declared_evidence_handles_resolve_in_the_state_reached(
        self, encounters
    ) -> None:
        _, encounter = encounters[LAPSED_POINT]
        assert encounter.resolved_evidence["approvals"] == ("checkpoint_1",)
        assert encounter.resolved_evidence["assertions"] == ("assertion_1",)


class TestTheObservationIsThePublicProjection:
    def test_it_carries_exactly_cores_projection_fields(self, encounters) -> None:
        for _, encounter in encounters.values():
            assert tuple(sorted(encounter.observation_payload())) == tuple(
                sorted(observation_projection_fields())
            )

    def test_it_rebuilds_as_a_real_agent_observation(self, encounters) -> None:
        _, encounter = encounters[QUOTE_POINT]
        observation = encounter.observation()
        assert observation.as_dict() == encounter.observation_payload()

    def test_it_is_detached_so_one_grader_cannot_edit_the_next_ones(
        self, encounters
    ) -> None:
        _, encounter = encounters[QUOTE_POINT]
        first = encounter.observation()
        first.retrieval["served"]["list_quotes"]["records"]["quotes"].clear()
        first.policy["approval_threshold_minor"] = 0
        second = encounter.observation()
        assert second.retrieval["served"]["list_quotes"]["records"]["quotes"], (
            "the frozen observation was mutated through a copy"
        )
        assert second.policy["approval_threshold_minor"] == 25000

    def test_hidden_environment_truth_is_not_in_it(self, encounters) -> None:
        for _, encounter in encounters.values():
            text = json.dumps(encounter.observation_payload())
            assert "hidden" not in observation_facts(encounter.observation_payload())
            assert "hidden_fixture_perished_seal" not in text
            assert "hidden_fixture_reliability_band_2" not in text

    def test_no_expectation_no_finding_and_no_trajectory_is_in_it(
        self, encounters
    ) -> None:
        for _, encounter in encounters.values():
            text = json.dumps(encounter.as_dict())
            for leaked in (
                "expected_terminal",
                "completed_successfully",
                "transferred_to_human_ownership",
                "trajectory",
                "failed_dimensions",
                "finding_codes",
                "reliable",
            ):
                assert leaked not in text, f"{leaked!r} leaked into the encounter"

    def test_nothing_from_after_the_decision_point_is_in_it(self, encounters) -> None:
        _, encounter = encounters[QUOTE_POINT]
        text = json.dumps(encounter.observation_payload())
        # Every one of these is created later in this scenario. At the instant a
        # priced proposal first lands there is no checkpoint, no bill, no claim,
        # no second visit, no verification, no settlement and no warranty cycle.
        for future in (
            "checkpoint_1",
            "invoice_1",
            "assertion_1",
            "visit_2",
            "visit_3",
            "evidence_1",
            "validation_1",
            "payment_1",
            "settlement_1",
            "work_cycle_3",
        ):
            assert future not in text, f"{future!r} exists only later in this scenario"

    def test_the_operation_has_not_ended_at_any_decision_point(self, encounters) -> None:
        for _, encounter in encounters.values():
            state = observation_facts(encounter.observation_payload())
            assert state["terminal"] is None
            assert state["replay_final"] is False


class TestGradingOneOutcome:
    def test_an_admissible_act_is_admissible(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(
            point, encounter, _admissible_request_approval(encounter.bindings)
        )
        assert verdict.admissible
        assert verdict.reason_code == ADMISSIBLE
        assert verdict.matched_outcome_id == "request_approval_for_the_quote_on_the_table"
        assert verdict.as_dict()["estimand"] == SHARED_ESTIMAND

    def test_an_amount_that_is_not_the_states_own_is_not_admissible(
        self, encounters
    ) -> None:
        point, encounter = encounters[QUOTE_POINT]
        outcome = _admissible_request_approval(encounter.bindings)
        payload = dict(outcome.payload)
        payload["amount_minor"] = payload["amount_minor"] - 1
        verdict = grade_outcome(
            point, encounter, Act(action_type="request_approval", payload=payload)
        )
        assert not verdict.admissible
        assert verdict.reason_code == OUTCOME_FIELDS_NOT_ADMISSIBLE
        assert "amount_minor" in verdict.detail

    def test_a_boolean_where_a_version_belongs_is_not_admissible(
        self, encounters
    ) -> None:
        point, encounter = encounters[QUOTE_POINT]
        outcome = _admissible_request_approval(encounter.bindings)
        payload = dict(outcome.payload)
        # ``True == 1`` in Python, so a bound-field check that compared only by
        # value would accept a yes/no answer as a quote version.
        payload["quote_version"] = True
        verdict = grade_outcome(
            point, encounter, Act(action_type="request_approval", payload=payload)
        )
        assert not verdict.admissible
        assert "quote_version" in verdict.detail

    def test_an_extra_payload_field_is_not_admissible(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        payload = dict(_admissible_request_approval(encounter.bindings).payload)
        payload["urgency"] = "high"
        verdict = grade_outcome(
            point, encounter, Act(action_type="request_approval", payload=payload)
        )
        assert not verdict.admissible
        assert "unconstrained field" in verdict.detail

    def test_a_deadline_past_the_authored_window_is_not_admissible(
        self, encounters
    ) -> None:
        point, encounter = encounters[QUOTE_POINT]
        payload = dict(_admissible_request_approval(encounter.bindings).payload)
        payload["deadline_after_minutes"] = 4321
        verdict = grade_outcome(
            point, encounter, Act(action_type="request_approval", payload=payload)
        )
        assert not verdict.admissible
        assert "4320" in verdict.detail

    def test_the_wrong_outcome_kind_is_named_as_such(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(point, encounter, Complete(reason="looks done"))
        assert verdict.reason_code == OUTCOME_KIND_NOT_ADMISSIBLE

    def test_the_wrong_action_is_named_as_such(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(
            point,
            encounter,
            Act(
                action_type="request_payment",
                payload={
                    "invoice_id": "invoice_1",
                    "amount_minor": 64000,
                    "currency": "GBP",
                },
            ),
        )
        assert verdict.reason_code == ACTION_TYPE_NOT_ADMISSIBLE

    def test_a_malformed_outcome_is_refused_before_it_is_graded(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        # A bare string satisfies ``Sequence[str]`` and iterates into characters.
        verdict = grade_outcome(
            point,
            encounter,
            Act(action_type="request_approval", evidence_refs="quote_1:v1"),
        )
        assert verdict.reason_code == MALFORMED_OUTCOME

    def test_something_that_is_not_an_outcome_at_all(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(point, encounter, {"kind": "ACT"})
        assert verdict.reason_code == NOT_AN_OUTCOME

    def test_an_encounter_from_another_decision_point_is_refused(
        self, encounters
    ) -> None:
        point, _ = encounters[QUOTE_POINT]
        _, elsewhere = encounters[LAPSED_POINT]
        with pytest.raises(Exception, match="does not belong"):
            grade_outcome(point, elsewhere, Complete())


class TestAlternativeValidOutcomes:
    def test_both_members_of_the_closed_set_are_admissible(self, encounters) -> None:
        point, encounter = encounters[LAPSED_POINT]
        escalated = grade_outcome(
            point,
            encounter,
            Escalate(
                checkpoint_id="checkpoint_probe",
                exception_type="APPROVAL_EXPIRED",
                deadline_after_minutes=1440,
            ),
        )
        acted = grade_outcome(
            point,
            encounter,
            Act(
                action_type="request_exception_resolution",
                payload={
                    "exception_type": "APPROVAL_EXPIRED",
                    "approval_checkpoint_id": encounter.bindings["lapsed_approval"][
                        "checkpoint_id"
                    ],
                    "deadline_after_minutes": 1440,
                },
            ),
        )
        assert escalated.admissible and acted.admissible
        assert escalated.matched_outcome_id != acted.matched_outcome_id

    def test_the_wrong_exception_type_is_not_admissible_either_way(
        self, encounters
    ) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point,
            encounter,
            Escalate(
                checkpoint_id="checkpoint_probe",
                exception_type="APPROVAL_REJECTED",
                deadline_after_minutes=1440,
            ),
        )
        assert not verdict.admissible
        assert "APPROVAL_EXPIRED" in verdict.detail


class TestEvidenceIsNotWhatAnActorSaysItIs:
    def test_citing_a_supplier_claim_is_not_admissible(self, encounters) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point,
            encounter,
            Escalate(
                checkpoint_id="checkpoint_probe",
                exception_type="APPROVAL_EXPIRED",
                deadline_after_minutes=1440,
                evidence_refs=("assertion_1",),
            ),
        )
        assert not verdict.admissible
        assert "assertion_1" in verdict.detail

    def test_citing_an_authoritative_record_is_admissible(self, encounters) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point,
            encounter,
            Escalate(
                checkpoint_id="checkpoint_probe",
                exception_type="APPROVAL_EXPIRED",
                deadline_after_minutes=1440,
                evidence_refs=("checkpoint_1", "quote_1:v1"),
            ),
        )
        assert verdict.admissible, verdict.detail

    def test_citing_a_reference_that_resolves_to_nothing_is_not_admissible(
        self, encounters
    ) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point,
            encounter,
            Escalate(
                checkpoint_id="checkpoint_probe",
                exception_type="APPROVAL_EXPIRED",
                deadline_after_minutes=1440,
                evidence_refs=("evidence_that_does_not_exist",),
            ),
        )
        assert not verdict.admissible


def _served(records: dict) -> dict:
    """One observation's retrieval affordance, carrying exactly these records."""
    return {
        "retrieval": {"served": {"get_case_record": {"ok": True, "records": records}}}
    }


class TestThePredicateIsSemanticNotPositional:
    def test_it_binds_by_state_not_by_identifier(self, manifest) -> None:
        """The same predicate, over a state whose identifiers are all different."""
        point = manifest.decision_point(QUOTE_POINT)
        relabelled = {
            "now": "2032-01-01T09:00:00Z",
            "operation_id": "lettings_maintenance_synthetic_v1",
            "scenario_id": "V1",
            "invocation_index": 99,
            "turn_index": 0,
            "policy": {"approval_threshold_minor": 25000},
            "actors": {},
            "message_fixture_ids": [],
            "last_rejection": None,
            # Since Phase 2 the fact surface a predicate reads is what was
            # *served*, not a projection handed over: the same records, arriving
            # the way an agent would have had to ask for them.
            **_served(
                {
                    "current_cycle_id": "some_other_cycle",
                    "quotes": {
                        "estimate_77:v4": {
                            "quote_id": "estimate_77",
                            "version": 4,
                            "cycle_id": "some_other_cycle",
                            "amount_minor": 25001,
                            "currency": "GBP",
                            "scope_digest": "f" * 64,
                            "status": "RECEIVED",
                            "valid_until": None,
                        }
                    },
                    "approvals": {},
                }
            ),
        }
        match = point.state_predicate.evaluate(relabelled)
        assert match.matched, match.reason
        assert match.bindings["target_quote"]["quote_id"] == "estimate_77"

    def test_it_declines_a_state_that_is_one_fact_different(self, manifest) -> None:
        point = manifest.decision_point(QUOTE_POINT)
        payload = {
            "policy": {"approval_threshold_minor": 25000},
            **_served(
                {
                    "current_cycle_id": "some_other_cycle",
                    "quotes": {
                        "estimate_77:v4": {
                            "quote_id": "estimate_77",
                            "version": 4,
                            "cycle_id": "some_other_cycle",
                            # One minor unit below the authored threshold: no
                            # human decision is required, so this is not that
                            # decision point.
                            "amount_minor": 25000,
                            "currency": "GBP",
                            "scope_digest": "f" * 64,
                            "status": "RECEIVED",
                            "valid_until": None,
                        }
                    },
                    "approvals": {},
                }
            ),
        }
        assert not point.state_predicate.evaluate(payload).matched


class TestTargetedNegativesAtEveryApplicablePoint:
    """Each shipped negative, offered the frozen observation at each point."""

    def test_the_clearly_wrong_ones_are_refused_everywhere(self, encounters) -> None:
        for agent_id in ("complete_early", "always_act", "always_wait"):
            for point, encounter in encounters.values():
                verdict = grade_outcome(
                    point, encounter, decide_at(encounter, build_agent(agent_id))
                )
                assert not verdict.admissible, (agent_id, point.decision_point_id)

    def test_an_untargeted_escalation_is_refused_by_its_own_type(
        self, encounters
    ) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point, encounter, decide_at(encounter, build_agent("always_escalate"))
        )
        assert verdict.reason_code == OUTCOME_FIELDS_NOT_ADMISSIBLE
        assert "APPROVAL_EXPIRED" in verdict.detail

    def test_a_claim_trusting_agent_is_refused_where_its_intervention_bites(
        self, encounters
    ) -> None:
        point, encounter = encounters[LAPSED_POINT]
        verdict = grade_outcome(
            point, encounter, decide_at(encounter, build_agent("trust_actor_claim"))
        )
        assert not verdict.admissible

    def test_a_negative_is_admissible_where_its_intervention_does_not_bite(
        self, encounters
    ) -> None:
        """Decision-point admissibility is local, and is not a proxy for the episode.

        ``trust_actor_claim`` fails its Lifecycle episode. At a decision point
        where no claim is on the table it decides correctly, and this control says
        so. A control that condemned it everywhere would be measuring the agent's
        reputation rather than its decision.
        """
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(
            point, encounter, decide_at(encounter, build_agent("trust_actor_claim"))
        )
        assert verdict.admissible

    def test_every_shipped_agent_is_graded_without_an_exception(self, encounters) -> None:
        for agent_id in agent_ids():
            for point, encounter in encounters.values():
                verdict = grade_outcome(
                    point, encounter, decide_at(encounter, build_agent(agent_id))
                )
                assert isinstance(verdict.admissible, bool)


class TestRecordedGapsInTheProjectAuthoredDraft:
    """Where the draft admissible sets and the reference path disagree, as found.

    These are asserted rather than repaired. Both are questions for the
    independent author who completes the review slot, and both would have been
    hidden by the obvious "fix": widening the draft until the reference passes.
    """

    def test_the_reference_is_admissible_where_the_draft_and_the_operation_agree(
        self, encounters
    ) -> None:
        point, encounter = encounters[QUOTE_POINT]
        verdict = grade_outcome(
            point, encounter, decide_at(encounter, build_agent("reference"))
        )
        assert verdict.admissible, verdict.detail

    def test_gap_one_the_draft_forbids_wake_events_the_wait_contract_invites(
        self, encounters
    ) -> None:
        """GAP 1 — ``bill_arrived_before_any_verified_work``.

        The draft allows a wait to be woken only by the three events that can
        resolve the outstanding decision. The reference additionally declares the
        supplier events that are genuinely pending, which Core's own wait contract
        rewards: an event the standing wait did not declare is recorded as an
        undeclared wake. The draft is therefore narrower than the operation's own
        semantics, and an independent author must decide whether "wake only on
        what resolves this" or "declare everything that can arrive" is the rule.
        """
        point, encounter = encounters[INVOICE_POINT]
        verdict = grade_outcome(
            point, encounter, decide_at(encounter, build_agent("reference"))
        )
        assert not verdict.admissible
        assert verdict.reason_code == OUTCOME_FIELDS_NOT_ADMISSIBLE
        assert "supplier_invoice_received" in verdict.detail

        # The decision the draft *does* admit is reachable and is a real WAIT, so
        # the gap is a disagreement about scope and not an empty admissible set.
        admitted = grade_outcome(
            point,
            encounter,
            Wait(
                reason="the approver holds an open decision",
                wake_on=("approver_decision_received", "approval_deadline_due"),
                fallback_after_minutes=4320,
            ),
        )
        assert admitted.admissible, admitted.detail

    def test_gap_two_one_graded_outcome_meets_a_multi_turn_invocation(
        self, encounters
    ) -> None:
        """GAP 2 — ``approval_window_closed_without_a_decision``.

        This control grades exactly one outcome: the first the agent returns at the
        matched state. The operation, however, allows several turns per invocation,
        and the reference spends its first turn notifying the affected party before
        opening the typed checkpoint on its second. Under a one-outcome control
        that is not admissible, even though the escalation the draft asks for does
        follow. This is a limitation of the Stage-1 construct, not of the agent:
        either the graded unit becomes "the invocation" or the admissible set has
        to say which preparatory moves may precede the decision. It is recorded
        here rather than resolved by quietly admitting notifications.
        """
        point, encounter = encounters[LAPSED_POINT]
        outcome = decide_at(encounter, build_agent("reference"))
        assert isinstance(outcome, Act)
        assert outcome.action_type == "send_message"
        verdict = grade_outcome(point, encounter, outcome)
        assert not verdict.admissible
        assert verdict.reason_code == ACTION_TYPE_NOT_ADMISSIBLE


class TestABoundThatCannotResolveIsRefusedRatherThanDropped:
    """The second half of the non-scalar comparand fix, at grading time.

    The match validator refuses a ceiling that cannot resolve to a number before
    anything runs. This asserts what happens if one reaches the grader anyway —
    by a hand-built constraint, or a manifest loaded by an older build: the
    outcome is refused with the handle named, rather than the maximum quietly
    ceasing to exist and every value passing.
    """

    def test_a_ceiling_that_names_a_registry_refuses_the_outcome(
        self, encounters
    ) -> None:
        _, encounter = encounters[QUOTE_POINT]
        constraint = Constraint(
            kind="integer_range", minimum=1, maximum=None, maximum_ref="state.quotes"
        )
        problem = constraint.problem(
            10**9, encounter.bindings, encounter.observation_payload(), "payload.deadline"
        )
        assert problem is not None
        assert "state.quotes" in problem

    def test_a_ceiling_that_resolves_stays_enforced(self, encounters) -> None:
        _, encounter = encounters[QUOTE_POINT]
        constraint = Constraint(
            kind="integer_range",
            minimum=1,
            maximum=None,
            maximum_ref="policy.approval_deadline_after_minutes",
        )
        payload = encounter.observation_payload()
        assert constraint.problem(4320, encounter.bindings, payload, "where") is None
        assert constraint.problem(4321, encounter.bindings, payload, "where") is not None


class TestAYesNoAnswerIsNotAQuantity:
    """``True == 1`` in Python, and every comparison here has to disagree."""

    def test_an_exact_constraint_does_not_accept_true_for_one(self) -> None:
        constraint = Constraint(kind="exact", value=1)
        assert constraint.problem(True, {}, {}, "where") is not None
        assert constraint.problem(1, {}, {}, "where") is None

    def test_a_one_of_constraint_does_not_accept_true_for_one(self) -> None:
        constraint = Constraint(kind="one_of", values=(1, 2))
        assert constraint.problem(True, {}, {}, "where") is not None
        assert constraint.problem(2, {}, {}, "where") is None

    def test_a_predicate_equality_does_not_accept_true_for_one(self) -> None:
        condition = RecordCondition(field="version", op="eq", value=1)
        assert condition.holds({"version": True}, {}) is False
        assert condition.holds({"version": 1}, {}) is True

    def test_a_predicate_membership_does_not_accept_true_for_one(self) -> None:
        condition = RecordCondition(field="version", op="in", value=[1, 2])
        assert condition.holds({"version": True}, {}) is False
        assert condition.holds({"version": 2}, {}) is True


class TestTheDerivedSchemaCannotDriftFromThePublicProjection:
    """The control names record fields; the observation carries record projections.

    ``RECORD_COLLECTIONS`` is read off the record *dataclasses* while the state an
    agent sees is built from each record's ``as_dict``. If those two vocabularies
    ever diverge, a control could name a field the projection does not carry — the
    predicate would never bind and the agent would be marked wrong for it. So the
    two are asserted equal rather than assumed equal.
    """

    RECORD_TYPES: ClassVar[dict[str, type]] = {
        "cycles": Cycle,
        "quotes": Quote,
        "approvals": Approval,
        "exceptions": ExceptionCheckpoint,
        "invoices": Invoice,
        "obligations": Obligation,
    }

    @staticmethod
    def _sample(record_type: type) -> Any:
        placeholders = {"str": "x", "int": 0, "bool": False, "tuple": ()}
        kwargs: dict[str, Any] = {}
        for entry in fields(record_type):
            if entry.default is not MISSING or entry.default_factory is not MISSING:
                continue
            token = str(entry.type).split("|")[0].strip().split("[")[0]
            kwargs[entry.name] = placeholders[token]
        return record_type(**kwargs)

    def test_every_registry_the_schema_names_is_a_live_record_type(self) -> None:
        assert set(RECORD_COLLECTIONS) == set(self.RECORD_TYPES)

    def test_every_record_projects_exactly_its_own_field_vocabulary(self) -> None:
        for collection, record_type in self.RECORD_TYPES.items():
            declared = RECORD_COLLECTIONS[collection]
            projected = tuple(self._sample(record_type).as_dict())
            assert declared == projected, (
                f"a {collection} record declares {declared} and projects {projected}; "
                "a control naming the difference would be refused a field the "
                "observation does carry, or admitted one it does not"
            )

    def test_every_outcome_projects_exactly_its_own_field_vocabulary(self) -> None:
        samples = (
            Act(action_type="request_approval"),
            Wait(reason="because", wake_on=("x",)),
            Ask(
                recipient_actor_id="supplier_1",
                message_fixture_id="message_1",
                wait=Wait(reason="because", wake_on=("x",)),
            ),
            Escalate(checkpoint_id="c", exception_type="E"),
            Complete(),
        )
        for outcome in samples:
            declared = {entry.name for entry in fields(outcome)}
            assert declared == set(outcome_as_dict(outcome)), type(outcome).__name__


class TestTheGradedUnitIsOnTheVerdict:
    def test_one_outcome_per_decision_point_is_machine_readable(self, encounters) -> None:
        point, encounter = encounters[QUOTE_POINT]
        payload = grade_outcome(
            point, encounter, _admissible_request_approval(encounter.bindings)
        ).as_dict()
        assert payload["graded_unit"] == GRADED_UNIT == "decision_point"
        assert payload["graded_outcomes"] == GRADED_OUTCOMES_PER_DECISION_POINT == 1
        assert payload["estimand"] == SHARED_ESTIMAND


class TestTheHarnessCannotReachAProviderOrAnEpisodeVerdict:
    def test_neither_module_imports_a_provider_or_a_network_client(self) -> None:
        for name in ("decision_points", "controls"):
            source = (REPO_ROOT / "src" / "operatebench" / f"{name}.py").read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)
            imported = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            } | {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            for module in sorted(imported):
                assert "anthropic" not in module
                assert "providers" not in module
                assert not module.startswith(("http", "socket", "urllib", "requests"))

    def test_neither_module_can_reach_an_episode_level_verdict(self) -> None:
        """The estimand is decision-point admissibility, and only that.

        Importing the runner or the evaluator would put ``reliable`` one attribute
        away, and a report that printed it beside an admissibility rate would be
        comparing arms whose temporal and recovery dimensions are not the same.
        """
        for name in ("decision_points", "controls"):
            source = (REPO_ROOT / "src" / "operatebench" / f"{name}.py").read_text(
                encoding="utf-8"
            )
            tree = ast.parse(source)
            imported = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            assert "operatebench.runner" not in imported
            assert not any(
                module.endswith("maintenance.evaluator") for module in imported
            )

    def test_importing_the_harness_pulls_in_no_transport(self) -> None:
        """The static check above misses transitive imports; this one does not."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, operatebench.controls, operatebench.decision_points; "
                "print(sorted(m for m in sys.modules "
                "if m.split('.')[0] in "
                "{'anthropic', 'httpx', 'requests', 'urllib3', 'socket', 'ssl'}))",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        assert result.stdout.strip() == "[]", result.stdout


class TestTheDriversOwnDecisionIsNotPartOfTheEncounter:
    def test_the_encounter_carries_no_outcome_shaped_field(self, encounters) -> None:
        """The reference path moves the world; it does not answer the question.

        Stated as an exact field set rather than a search for suspicious strings:
        the day somebody adds ``driver_outcome`` "for debugging", this fails.
        """
        _, encounter = encounters[QUOTE_POINT]
        assert {field.name for field in fields(encounter)} == {
            "decision_point_id",
            "scenario_id",
            "now",
            "invocation_index",
            "turn_index",
            "bindings",
            "resolved_evidence",
            "observation_digest_sha256",
            "_observation_json",
        }

    def test_the_only_decision_in_the_frozen_observation_is_the_worlds(
        self, encounters
    ) -> None:
        for _, encounter in encounters.values():
            # ``last_rejection`` is the one place a prior proposal could surface.
            # Nothing was refused on the way to any of these states.
            assert encounter.observation_payload()["last_rejection"] is None
