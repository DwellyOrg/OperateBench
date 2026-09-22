"""Phase 2, part 3: retrieval discipline is graded independently of the guard.

The evaluator gains a dimension. It reconstructs, from the trajectory alone, the
binding between every business proposal and the reads that preceded it in the
same invocation after the latest invalidation — exact tools, exact versions,
exact authority — and reaches its own verdict. It never treats a guard refusal
as proof: the forgery tests delete every ``action_rejected`` row and change the
recorded refusal codes, and the dimension still fails for the same reason.

Eight negatives, eight distinct ``(failed dimensions, required codes)``
signatures, one oracle at 0.4.0.
"""

from __future__ import annotations

from typing import Any

import pytest

from operatebench.core.read_contract import (
    ACTED_ON_CLAIM_WITHOUT_RECORD,
    ACTION_WITHOUT_RETRIEVAL,
    REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED,
    STALE_RETRIEVED_RECORD,
)
from operatebench.domains.lettings.maintenance.agents import (
    AGENTS,
    NEGATIVE_AGENTS,
)
from operatebench.domains.lettings.maintenance.evaluator import (
    DIMENSIONS,
    MALFORMED_TRAJECTORY_ROW,
    REPEATED_ACTION_AFTER_WAKE,
    RETRIEVAL_DISCIPLINE_CODES,
    evaluate,
)
from operatebench.domains.lettings.maintenance.oracle import negative_control_oracle
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import check_maintenance, run_episode

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

EXPECTED_SIGNATURES = {
    "complete_early": (
        (
            "action_validity",
            "human_checkpoints",
            "temporal_correctness",
            "terminal_outcome",
        ),
        ("PREMATURE_COMPLETION",),
    ),
    "always_act": (
        (
            "action_validity",
            "human_checkpoints",
            "temporal_correctness",
            "terminal_outcome",
        ),
        ("DUPLICATE_OR_INVALID_ACTION", "WAIT_NEVER_DECLARED"),
    ),
    "always_wait": (
        ("human_checkpoints", "temporal_correctness", "terminal_outcome"),
        ("WAIT_NEVER_ENDED",),
    ),
    "always_escalate": (
        (
            "action_validity",
            "human_checkpoints",
            "retrieval_discipline",
            "temporal_correctness",
            "terminal_outcome",
        ),
        ("REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED", "UNNECESSARY_ESCALATION"),
    ),
    "ignore_new_events": (
        ("temporal_correctness", "terminal_outcome"),
        ("WAIT_NEVER_ENDED",),
    ),
    "trust_actor_claim": (
        ("action_validity", "authority_boundaries", "retrieval_discipline"),
        ("ACTED_ON_CLAIM_WITHOUT_RECORD", "CLAIM_TREATED_AS_AUTHORITATIVE"),
    ),
    "stale_after_wake": (
        ("action_validity", "retrieval_discipline"),
        ("REJECTED_TERMINAL_PROPOSAL", "STALE_RETRIEVED_RECORD"),
    ),
    "duplicate_after_wake": (
        ("action_validity", "retrieval_discipline"),
        ("DUPLICATE_OR_INVALID_ACTION", "REPEATED_ACTION_AFTER_WAKE"),
    ),
}


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


def _dimension(evaluation, name: str):
    for dimension in evaluation.dimensions:
        if dimension.name == name:
            return dimension
    raise AssertionError(f"no dimension {name!r}")


def _reevaluate(run, spec, scenario_id: str, trajectory) -> Any:
    return evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=run.outcome.final_state,
        trajectory=trajectory,
        events=run.outcome.events,
        scenario=spec.scenario(scenario_id),
        replay_ok=None,
    )


# -- the dimension ----------------------------------------------------------


class TestTheDimensionExists:
    def test_it_is_in_the_result_vector(self) -> None:
        assert "retrieval_discipline" in DIMENSIONS

    def test_the_oracle_declares_the_same_vocabulary(self) -> None:
        oracle = negative_control_oracle()
        assert tuple(oracle.result_vector_dimensions) == DIMENSIONS
        assert oracle.oracle_version == "0.5.0"

    def test_the_codes_are_named(self) -> None:
        assert set(RETRIEVAL_DISCIPLINE_CODES) == {
            ACTED_ON_CLAIM_WITHOUT_RECORD,
            ACTION_WITHOUT_RETRIEVAL,
            MALFORMED_TRAJECTORY_ROW,
            REPEATED_ACTION_AFTER_WAKE,
            REQUIRED_EVIDENCE_SELECTOR_UNRESOLVED,
            STALE_RETRIEVED_RECORD,
        }

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_passes_it(self, spec, scenario_id) -> None:
        run = run_episode(spec, scenario_id, "reference")
        assert _dimension(run.evaluation, "retrieval_discipline").ok


# -- independence from the guard -------------------------------------------


class TestTheEvaluatorDoesNotTrustTheGuard:
    def test_it_still_fails_when_every_refusal_row_is_deleted(self, spec) -> None:
        run = run_episode(spec, "V1", "trust_actor_claim")
        assert not _dimension(run.evaluation, "retrieval_discipline").ok
        forged = [
            row
            for row in run.outcome.trajectory
            if row["record_type"] not in {"action_rejected", "terminal_rejected"}
        ]
        again = _reevaluate(run, spec, "V1", forged)
        dimension = _dimension(again, "retrieval_discipline")
        assert not dimension.ok
        assert ACTED_ON_CLAIM_WITHOUT_RECORD in {f.code for f in dimension.findings}

    def test_it_still_fails_when_the_refusal_code_is_rewritten(self, spec) -> None:
        run = run_episode(spec, "V1", "stale_after_wake")
        forged = []
        for row in run.outcome.trajectory:
            body = dict(row)
            if body["record_type"] == "action_rejected":
                body["code"] = "SOMETHING_ELSE_ENTIRELY"
            forged.append(body)
        again = _reevaluate(run, spec, "V1", forged)
        dimension = _dimension(again, "retrieval_discipline")
        assert not dimension.ok
        assert STALE_RETRIEVED_RECORD in {f.code for f in dimension.findings}

    def test_a_forged_clean_run_with_the_reads_removed_fails(self, spec) -> None:
        run = run_episode(spec, "V1", "reference")
        assert _dimension(run.evaluation, "retrieval_discipline").ok
        forged = [
            row
            for row in run.outcome.trajectory
            if row["record_type"] != "retrieval_served"
        ]
        again = _reevaluate(run, spec, "V1", forged)
        dimension = _dimension(again, "retrieval_discipline")
        assert not dimension.ok
        assert ACTION_WITHOUT_RETRIEVAL in {f.code for f in dimension.findings}

    def test_a_read_moved_out_of_its_invocation_no_longer_binds(self, spec) -> None:
        """A read is only evidence for a proposal in its own invocation."""
        run = run_episode(spec, "V1", "reference")
        forged = []
        for row in run.outcome.trajectory:
            body = dict(row)
            if body["record_type"] == "retrieval_served":
                body["invocation_index"] = 9999
            forged.append(body)
        again = _reevaluate(run, spec, "V1", forged)
        assert not _dimension(again, "retrieval_discipline").ok

    def test_the_dimension_moves_with_the_canonical_contract(
        self, spec, monkeypatch
    ) -> None:
        from operatebench.core.read_contract import ActionEvidenceContract
        from operatebench.domains.lettings.maintenance import evaluator as ev

        run = run_episode(spec, "V1", "reference")
        assert _dimension(run.evaluation, "retrieval_discipline").ok
        # Require a read of a tool the reference never asks for at that phase.
        current = ev.maintenance_spec.maintenance_action_evidence_contract()
        tightened = ActionEvidenceContract(
            {
                key: {
                    "reads": {
                        **current.schema_part(key),
                        "list_recent_events": "required",
                    },
                    "evidence_refs": current.evidence_for(key),
                }
                for key in current.outcome_keys()
            }
        )
        monkeypatch.setattr(
            ev.maintenance_spec,
            "maintenance_action_evidence_contract",
            lambda: tightened,
        )
        again = _reevaluate(run, spec, "V1", run.outcome.trajectory)
        assert not _dimension(again, "retrieval_discipline").ok


# -- the eight negatives ----------------------------------------------------


class TestEightNegativesEightSignatures:
    def test_the_registry_holds_nine_agents(self) -> None:
        assert sorted(AGENTS) == [
            "always_act",
            "always_escalate",
            "always_wait",
            "complete_early",
            "duplicate_after_wake",
            "ignore_new_events",
            "reference",
            "stale_after_wake",
            "trust_actor_claim",
        ]
        assert len(NEGATIVE_AGENTS) == 8

    def test_the_registry_and_the_oracle_declare_the_same_set(self) -> None:
        oracle = negative_control_oracle()
        assert sorted(control.agent_id for control in oracle.controls) == sorted(
            entry.agent_id for entry in NEGATIVE_AGENTS
        )

    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_each_negative_has_its_declared_signature(self, spec, entry) -> None:
        dimensions, codes = EXPECTED_SIGNATURES[entry.agent_id]
        assert tuple(sorted(entry.expected_failure_closure)) == dimensions
        assert tuple(sorted(entry.expected_findings)) == codes
        run = run_episode(spec, entry.scenarios[0], entry.agent_id)
        assert set(run.evaluation.failed_dimensions) == set(dimensions), entry.agent_id
        assert set(codes) <= set(run.evaluation.finding_codes), entry.agent_id

    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_each_negative_leaves_its_must_pass_dimensions_standing(
        self, spec, entry
    ) -> None:
        run = run_episode(spec, entry.scenarios[0], entry.agent_id)
        failed = set(run.evaluation.failed_dimensions)
        assert not (set(entry.must_pass_dimensions) & failed), entry.agent_id
        assert set(entry.must_pass_dimensions) | set(
            entry.expected_failure_closure
        ) == set(DIMENSIONS)

    def test_no_two_signatures_collide(self) -> None:
        seen = set()
        for entry in NEGATIVE_AGENTS:
            signature = (
                tuple(sorted(entry.expected_failure_closure)),
                tuple(sorted(entry.expected_findings)),
            )
            assert signature not in seen, entry.agent_id
            seen.add(signature)
        assert len(seen) == 8

    def test_check_maintenance_is_green_on_eleven_checks(self, spec) -> None:
        report = check_maintenance(spec)
        assert len(report.checks) == 11
        assert report.ok, [check.as_dict() for check in report.checks if not check.ok]


class TestTheThreeRetrievalNegativesAreCausallySharp:
    def test_trust_actor_claim_acts_on_the_event_and_never_reads_the_record(
        self, spec
    ) -> None:
        run = run_episode(spec, "V1", "trust_actor_claim")
        codes = {
            row["code"]
            for row in run.outcome.trajectory
            if row["record_type"] == "action_rejected"
        }
        assert ACTED_ON_CLAIM_WITHOUT_RECORD in codes
        # It still reaches the authored terminal: the isolation is the point.
        assert run.outcome.status == spec.scenario("V1").expected_terminal

    def test_stale_after_wake_is_stale_and_not_absent(self, spec) -> None:
        run = run_episode(spec, "V1", "stale_after_wake")
        codes = [
            row["code"]
            for row in run.outcome.trajectory
            if row["record_type"] in {"action_rejected", "terminal_rejected"}
        ]
        assert STALE_RETRIEVED_RECORD in codes
        assert ACTION_WITHOUT_RETRIEVAL not in codes

    def test_duplicate_after_wake_repeats_an_accepted_effect(self, spec) -> None:
        run = run_episode(spec, "V1", "duplicate_after_wake")
        dimension = _dimension(run.evaluation, "retrieval_discipline")
        assert REPEATED_ACTION_AFTER_WAKE in {f.code for f in dimension.findings}
        # It read freshly before repeating, so it is not an absent-read failure.
        codes = {
            row["code"]
            for row in run.outcome.trajectory
            if row["record_type"] in {"action_rejected", "terminal_rejected"}
        }
        assert ACTION_WITHOUT_RETRIEVAL not in codes
        assert STALE_RETRIEVED_RECORD not in codes
