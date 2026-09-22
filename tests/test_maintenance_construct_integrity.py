"""Bounded construct-integrity invariants for the Maintenance fixture.

**A proposal is not interchangeable with any other proposal.** Rewriting the
proposal that committed to name an unrelated checkpoint or quote version must
invalidate the trajectory. Every proposal carries a Core-assigned identity;
every refusal and accepted effect carries the identity of the proposal it
answers, and the evaluator re-derives the authorisation from *that* proposal.

**A malformed agent is a refusal, not a traceback.** An outcome whose payload
field holds a list is refused at the public engine boundary.
Outcome shape is checked in Core before anything indexes it, and payload shape is
checked at the Maintenance boundary the same way event payloads are.

**An accepted spec option must be executable.** A policy field nothing consumes
and a terminal class nothing can produce are both promises the build does not
keep.

**A final state is a shape, not an object.** ``final_state`` is checked
recursively, and digests recorded beside the payload are recomputed at read time.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from boundarybench.jsonsafe import canonical_json_text
from operatebench.artifact import (
    ArtifactError,
    build_artifact,
    read_artifact,
    replay_artifact,
    validate_artifact,
    write_artifact,
)
from operatebench.core.engine import STATUS_DEADLOCK, Engine
from operatebench.core.errors import SpecSchemaError, UnknownFieldError, UnknownTypeError
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import AGENTS, build_agent
from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.oracle import negative_control_oracle
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_TYPES,
    TERMINAL_OUTCOMES,
    OperationSpec,
    load_spec,
)
from operatebench.runner import check_agent, run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    loaded = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _trajectory(spec: OperationSpec, scenario_id: str = "V1") -> list[dict[str, Any]]:
    run = run_episode(spec, scenario_id, "reference")
    return [copy.deepcopy(dict(row)) for row in run.outcome.trajectory]


def _evaluate_with(
    spec: OperationSpec, trajectory: list[dict[str, Any]], scenario_id: str = "V1"
) -> Any:
    run = run_episode(spec, scenario_id, "reference")
    return evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=copy.deepcopy(dict(run.outcome.final_state)),
        trajectory=trajectory,
        events=copy.deepcopy([dict(event) for event in run.outcome.events]),
        scenario=spec.scenario(scenario_id),
        replay_ok=True,
    )


def _row(trajectory: list[dict[str, Any]], record_type: str, action_type: str) -> Any:
    return next(
        row
        for row in trajectory
        if row.get("record_type") == record_type and row.get("action_type") == action_type
    )


def _reindex(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, row in enumerate(trajectory):
        row["index"] = index
    return trajectory


# ------------------------------------------------------------- proposal binding


class TestProposalIdentity:
    """Core names every proposal, and every answer to it names the proposal."""

    def test_every_proposed_action_carries_a_unique_proposal_id(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        proposals = [row for row in trajectory if row["record_type"] == "action_proposed"]
        assert proposals
        ids = [row["proposal_id"] for row in proposals]
        assert all(isinstance(item, str) and item for item in ids)
        assert len(set(ids)) == len(ids)

    def test_every_outcome_row_names_the_proposal_it_answers(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec, "V2")
        proposals = {
            row["proposal_id"]: row
            for row in trajectory
            if row["record_type"] == "action_proposed"
        }
        outcomes = [
            row
            for row in trajectory
            if row["record_type"] in {"action_rejected", "effect_accepted"}
        ]
        assert len(outcomes) == len(proposals)
        for row in outcomes:
            proposal = proposals[row["proposal_id"]]
            assert proposal["action_type"] == row["action_type"]
            assert proposal["index"] < row["index"]

    def test_proposal_ids_are_deterministic_across_runs(
        self, spec: OperationSpec
    ) -> None:
        first = _trajectory(spec)
        second = _trajectory(spec)
        assert [row.get("proposal_id") for row in first] == [
            row.get("proposal_id") for row in second
        ]


class TestExactProposalToEffectBinding:
    """The accepted effect is re-derived from the proposal that committed it."""

    def test_a_forged_work_authorisation_payload_is_not_reliable(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        proposal = _row(trajectory, "action_proposed", "authorise_supplier_work")
        proposal["payload"]["approval_checkpoint_id"] = "checkpoint_NONEXISTENT"
        proposal["payload"]["quote_version"] = 999
        result = _evaluate_with(spec, trajectory)
        assert result.reliable is False
        assert "CRITICAL_WORK_AUTHORISED_WITHOUT_PRIOR_APPROVAL" in set(
            result.finding_codes
        )

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("approval_checkpoint_id", "checkpoint_NONEXISTENT"),
            ("quote_id", "quote_other"),
            ("quote_version", 999),
            ("cycle_id", "work_cycle_1"),
        ],
    )
    def test_each_forged_authorisation_field_breaks_the_binding(
        self, spec: OperationSpec, field_name: str, value: Any
    ) -> None:
        trajectory = _trajectory(spec)
        proposal = _row(trajectory, "action_proposed", "authorise_supplier_work")
        proposal["payload"][field_name] = value
        assert _evaluate_with(spec, trajectory).reliable is False

    def test_a_forged_payment_proposal_breaks_the_binding(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        proposal = _row(trajectory, "action_proposed", "request_payment")
        proposal["payload"]["invoice_id"] = "invoice_SOMEBODY_ELSE"
        result = _evaluate_with(spec, trajectory)
        assert result.reliable is False
        assert "CRITICAL_PAYMENT_WITHOUT_PRIOR_EVIDENCE" in set(result.finding_codes)

    def test_an_authorisation_that_dropped_its_evidence_ref_is_refused(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        proposal = _row(trajectory, "action_proposed", "authorise_supplier_work")
        proposal["evidence_refs"] = []
        assert _evaluate_with(spec, trajectory).reliable is False

    def test_a_payment_is_not_justified_by_a_second_unrelated_proposal(
        self, spec: OperationSpec
    ) -> None:
        # A later, well-formed proposal must not stand in for the committed one.
        trajectory = _trajectory(spec)
        proposal = _row(trajectory, "action_proposed", "request_payment")
        forged = copy.deepcopy(proposal)
        proposal["payload"]["invoice_id"] = "invoice_SOMEBODY_ELSE"
        forged["proposal_id"] = "proposal_forged"
        trajectory.insert(trajectory.index(proposal) + 1, forged)
        assert _evaluate_with(spec, _reindex(trajectory)).reliable is False


class TestProposalCardinalityAndOrdering:
    """Orphan, duplicate, cross-linked and reordered links are all refused."""

    def test_an_orphan_effect_is_refused(self, spec: OperationSpec) -> None:
        trajectory = _trajectory(spec)
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        effect["proposal_id"] = "proposal_nobody_made"
        result = _evaluate_with(spec, trajectory)
        assert result.reliable is False
        assert "CRITICAL_PROPOSAL_BINDING_MISMATCH" in set(result.finding_codes)

    def test_a_missing_proposal_id_is_refused(self, spec: OperationSpec) -> None:
        trajectory = _trajectory(spec)
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        del effect["proposal_id"]
        assert _evaluate_with(spec, trajectory).reliable is False

    def test_a_duplicated_effect_for_one_proposal_is_refused(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        trajectory.insert(trajectory.index(effect) + 1, copy.deepcopy(effect))
        result = _evaluate_with(spec, _reindex(trajectory))
        assert result.reliable is False
        assert "CRITICAL_PROPOSAL_BINDING_MISMATCH" in set(result.finding_codes)

    def test_a_cross_linked_effect_is_refused(self, spec: OperationSpec) -> None:
        trajectory = _trajectory(spec)
        other = _row(trajectory, "action_proposed", "request_approval")
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        effect["proposal_id"] = other["proposal_id"]
        result = _evaluate_with(spec, trajectory)
        assert result.reliable is False
        assert "CRITICAL_PROPOSAL_BINDING_MISMATCH" in set(result.finding_codes)

    def test_an_effect_recorded_before_its_proposal_is_refused(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        proposal = _row(trajectory, "action_proposed", "authorise_supplier_work")
        trajectory.remove(effect)
        effect["at"] = proposal["at"]
        trajectory.insert(trajectory.index(proposal), effect)
        result = _evaluate_with(spec, _reindex(trajectory))
        assert result.reliable is False
        assert "CRITICAL_PROPOSAL_BINDING_MISMATCH" in set(result.finding_codes)

    def test_a_proposal_with_no_outcome_at_all_is_refused(
        self, spec: OperationSpec
    ) -> None:
        trajectory = _trajectory(spec)
        effect = _row(trajectory, "effect_accepted", "authorise_supplier_work")
        trajectory.remove(effect)
        result = _evaluate_with(spec, _reindex(trajectory))
        assert result.reliable is False
        assert "CRITICAL_PROPOSAL_BINDING_MISMATCH" in set(result.finding_codes)


class TestReferencePathStaysDeterministic:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_is_still_reliable(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        assert run.reliable is (scenario_id != "V2"), run.evaluation.as_dict()
        assert run.evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )
        assert set(run.evaluation.finding_codes) == (
            {"REQUIRED_NOTIFICATION_UNDELIVERED", "OBLIGATION_MESSAGE_NOT_ESTABLISHED"}
            if scenario_id == "V2"
            else set()
        )

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_artefact_still_replays(
        self, spec: OperationSpec, tmp_path: Path, scenario_id: str
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        path = write_artifact(run, tmp_path / f"{scenario_id}.json")
        assert replay_artifact(spec, read_artifact(path)).ok is True


# --------------------------------------------------------- malformed outcomes


def _catalogue_batch(observation: AgentObservation) -> Any:
    """Every published read this invocation has not been served yet, or ``None``.

    Since Phase 2 a business proposal is refused unless the reads it publishes
    were served in this invocation, so a scripted agent that means to test
    *payload* validation has to establish the record first — otherwise the read
    guard answers first and the payload is never looked at. Reading everything
    keeps the script's own subject the only thing under test.
    """
    catalogue = (observation.retrieval or {}).get("catalogue") or {}
    served = (observation.retrieval or {}).get("served") or {}
    outstanding = sorted(set(catalogue) - set(served))
    budget = int((observation.retrieval or {}).get("batch_budget_remaining") or 0)
    if not outstanding or budget <= 0:
        return None
    return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in outstanding))


class _ScriptedAgent:
    """An agent that returns exactly what a test hands it, once, then waits."""

    agent_id = "scripted"

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome
        self._done = False

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: AgentObservation) -> Any:
        batch = _catalogue_batch(observation)
        if batch is not None:
            return batch
        if self._done:
            return Complete(reason="nothing more to try")
        self._done = True
        return self._outcome


def _run_with(spec: OperationSpec, agent: Any, scenario_id: str = "V1") -> Any:
    scenario = spec.scenario(scenario_id)
    return Engine(
        MaintenanceOperation(spec, scenario_id),
        agent,
        identity={
            "operation_id": spec.operation_id,
            "scenario_id": scenario_id,
            "agent_id": getattr(agent, "agent_id", "scripted"),
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()


class _WithAllowance:
    """The Maintenance operation, with a smaller declared invocation allowance.

    Delegates everything else, so the operation under test is the shipped one
    and the only thing the test changed is the number the plan declares.
    """

    def __init__(self, inner: Any, allowance: int) -> None:
        self._inner = inner
        self._allowance = allowance

    def build_plan(self) -> Any:
        return dataclasses.replace(
            self._inner.build_plan(), max_invocations=self._allowance
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _run_with_allowance(
    spec: OperationSpec, allowance: int, scenario_id: str = "V1"
) -> Any:
    scenario = spec.scenario(scenario_id)
    return Engine(
        _WithAllowance(MaintenanceOperation(spec, scenario_id), allowance),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "scenario_id": scenario_id,
            "agent_id": "reference",
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=scenario.dispatch_failures,
    ).run()


def _invocation_indices(outcome: Any) -> list[int]:
    return [
        int(row["invocation_index"])
        for row in outcome.trajectory
        if row["record_type"] == "agent_invoked"
    ]


def _violation_codes(outcome: Any) -> set[str]:
    return {
        str(row["code"])
        for row in outcome.trajectory
        if row["record_type"] == "critical_violation"
    }


def _refusal_codes(outcome: Any) -> set[str]:
    return {
        str(row["code"])
        for row in outcome.trajectory
        if row["record_type"] in {"action_rejected", "outcome_rejected"}
    }


class _AlwaysBad:
    """Returns the same malformed outcome on every turn, for the whole episode."""

    agent_id = "always_bad"

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: AgentObservation) -> Any:
        batch = _catalogue_batch(observation)
        if batch is not None:
            return batch
        return self._outcome


class TestMalformedActionPayloadsAreNamedRefusals:
    def test_the_parent_reproduction_is_a_refusal_not_a_typeerror(
        self, spec: OperationSpec
    ) -> None:
        bad = Act(
            action_type="request_approval",
            payload={
                "checkpoint_id": "checkpoint_bad",
                "quote_id": "quote_bad",
                "quote_version": [],
                "cycle_id": "work_cycle_bad",
            },
            evidence_refs=(),
        )
        outcome = _run_with(spec, _AlwaysBad(bad))
        assert "MALFORMED_ACTION_PAYLOAD" in _refusal_codes(outcome)
        assert outcome.status in {"operation_deadlock", "operational_horizon_exhausted"}

    @pytest.mark.parametrize(
        "action_type", [name for name in MAINTENANCE_ACTION_TYPES if name != "complete"]
    )
    def test_every_action_type_refuses_an_empty_payload_by_name(
        self, spec: OperationSpec, action_type: str
    ) -> None:
        outcome = _run_with(
            spec, _ScriptedAgent(Act(action_type=action_type, payload={}))
        )
        assert "MALFORMED_ACTION_PAYLOAD" in _refusal_codes(outcome)

    @pytest.mark.parametrize(
        "payload",
        [
            {"invoice_id": ["invoice_1"]},
            {"invoice_id": {"invoice_1": True}},
            {"invoice_id": 1},
            {"invoice_id": None},
            {"invoice_id": True},
            {"invoice_id": ""},
            {"invoice_id": "invoice_1", "surprise": "extra"},
        ],
        ids=[
            "list",
            "mapping",
            "integer",
            "null",
            "boolean",
            "empty",
            "unknown-field",
        ],
    )
    def test_container_and_type_confusion_is_refused(
        self, spec: OperationSpec, payload: Mapping[str, Any]
    ) -> None:
        outcome = _run_with(
            spec,
            _ScriptedAgent(
                Act(action_type="request_invoice_validation", payload=dict(payload))
            ),
        )
        assert "MALFORMED_ACTION_PAYLOAD" in _refusal_codes(outcome)

    def test_a_boolean_is_not_an_integer(self, spec: OperationSpec) -> None:
        outcome = _run_with(
            spec,
            _ScriptedAgent(
                Act(
                    action_type="authorise_supplier_work",
                    payload={
                        "visit_id": "visit_2",
                        "quote_id": "quote_1",
                        "quote_version": True,
                        "approval_checkpoint_id": "checkpoint_1",
                        "cycle_id": "work_cycle_2",
                    },
                )
            ),
        )
        assert "MALFORMED_ACTION_PAYLOAD" in _refusal_codes(outcome)

    def test_a_refused_payload_mutates_nothing(self, spec: OperationSpec) -> None:
        outcome = _run_with(
            spec,
            _ScriptedAgent(
                Act(action_type="request_supplier_visit", payload={"visit_id": 7})
            ),
        )
        cycles = outcome.final_state["cycles"]
        assert all(cycle["visit_id"] is None for cycle in cycles.values())


class TestMalformedOutcomesAtTheEngineBoundary:
    @pytest.mark.parametrize(
        "outcome",
        [
            Act(action_type="request_invoice_validation", payload=["invoice_1"]),
            Act(action_type="request_invoice_validation", payload="invoice_1"),
            Act(
                action_type="request_invoice_validation",
                payload={"invoice_id": "invoice_1"},
                evidence_refs="evidence_1",
            ),
            Act(
                action_type="request_invoice_validation",
                payload={"invoice_id": "invoice_1"},
                evidence_refs=(1, 2),
            ),
            Act(
                action_type="request_invoice_validation",
                payload={"invoice_id": "invoice_1"},
                evidence_refs=("",),
            ),
            Act(
                action_type="request_invoice_validation",
                payload={"invoice_id": "invoice_1"},
                evidence_refs=(True,),
            ),
        ],
        ids=[
            "payload-list",
            "payload-string",
            "evidence-string",
            "evidence-integers",
            "evidence-empty",
            "evidence-boolean",
        ],
    )
    def test_a_malformed_outcome_is_a_named_refusal(
        self, spec: OperationSpec, outcome: Any
    ) -> None:
        result = _run_with(spec, _ScriptedAgent(outcome))
        assert _refusal_codes(result) & {
            "MALFORMED_AGENT_OUTCOME",
            "MALFORMED_ACTION_PAYLOAD",
        }

    def test_an_object_that_is_not_an_outcome_at_all_is_refused(
        self, spec: OperationSpec
    ) -> None:
        result = _run_with(spec, _ScriptedAgent("just do it"))
        assert "MALFORMED_AGENT_OUTCOME" in _refusal_codes(result)

    def test_an_unknown_action_type_is_refused_without_indexing_the_payload(
        self, spec: OperationSpec
    ) -> None:
        result = _run_with(
            spec, _ScriptedAgent(Act(action_type="wire_the_money", payload={}))
        )
        assert "ACTION_OUTSIDE_AGENT_AUTHORITY" in _refusal_codes(result)

    def test_the_episode_continues_under_normal_turn_limits(
        self, spec: OperationSpec
    ) -> None:
        bad = Act(action_type="request_invoice_validation", payload=["nope"])
        result = _run_with(spec, _AlwaysBad(bad))
        violations = [
            row for row in result.trajectory if row["record_type"] == "critical_violation"
        ]
        assert violations
        assert violations[0]["code"] == "INVOCATION_TURN_LIMIT_EXCEEDED"

    def test_the_declared_invocation_allowance_is_what_the_agent_is_given(
        self, spec: OperationSpec
    ) -> None:
        # The reference agent reaches its terminal on invocation 14 of V1, so 14
        # and 13 are the two sides of this boundary. Asserted with the shipped
        # agent under a lowered allowance rather than at the default 200: the
        # boundary is the same arithmetic either way, and 200 invocations is a
        # slow way to look at it.
        allowed = _run_with_allowance(spec, 14)
        assert allowed.status == "completed_successfully"
        assert _invocation_indices(allowed) == list(range(1, 15))
        assert _violation_codes(allowed) == set()

        refused = _run_with_allowance(spec, 13)
        # The allowance is a count of calls the agent may be given, not a verdict
        # passed on calls it has already been given, so under an allowance of 13
        # the fourteenth call does not happen at all.
        assert _invocation_indices(refused) == list(range(1, 14))
        assert "INVOCATION_LIMIT_EXCEEDED" in _violation_codes(refused)
        assert refused.status == STATUS_DEADLOCK
        assert not [
            row for row in refused.trajectory if row["record_type"] == "terminal_accepted"
        ]

    def test_a_malformed_wait_is_refused_rather_than_slept_on(
        self, spec: OperationSpec
    ) -> None:
        class BadWait:
            agent_id = "bad_wait"

            def begin_episode(self, identity: Mapping[str, Any]) -> None:
                return None

            def decide(self, observation: AgentObservation) -> Any:
                return Wait(reason="soon", wake_on=(7,), fallback_after_minutes=60)

        result = _run_with(spec, BadWait())
        assert "MALFORMED_AGENT_OUTCOME" in _refusal_codes(result)


# ------------------------------------------------ every option is executable


def _perturbed(raw: Mapping[str, Any], **policy: Any) -> OperationSpec:
    body = copy.deepcopy(dict(raw))
    body["policy"].update(policy)
    return OperationSpec.from_mapping(body, source="<perturbed>")


#: One semantic perturbation per accepted policy field, and the run it must
#: change. A field nothing consumes cannot be perturbed into a different episode,
#: which is exactly the property this table asserts.
POLICY_PERTURBATIONS: Mapping[str, Any] = {
    "currency": "USD",
    "issue_classification": "SOMETHING_ELSE",
    "approval_threshold_minor": 1_000_000,
    "approval_reminder_after_minutes": 60,
    "approval_deadline_after_minutes": 5760,
    "approval_validity_minutes": 1,
    "provisional_close_minutes": 60,
    "visit_followup_after_minutes": 60,
    "completion_notice_within_minutes": 60,
    "max_turns_per_invocation": 1,
    "horizon_minutes": 60,
    # One batch per invocation. The reference reads once, acts, and — because an
    # accepted own effect stales what it read — reads again before its next
    # outcome, so a budget of one is spent inside any invocation that decides
    # twice and the refusals reach the trajectory.
    "max_retrieval_batches_per_invocation": 1,
}


class TestEveryPolicyFieldIsExecutable:
    def test_the_table_covers_every_accepted_policy_field(
        self, raw: Mapping[str, Any]
    ) -> None:
        assert set(POLICY_PERTURBATIONS) == set(raw["policy"])

    @pytest.mark.parametrize("field_name", sorted(POLICY_PERTURBATIONS))
    def test_perturbing_a_policy_field_changes_what_executes(
        self, spec: OperationSpec, raw: Mapping[str, Any], field_name: str
    ) -> None:
        baseline = run_episode(spec, "V1", "reference")
        changed = run_episode(
            _perturbed(raw, **{field_name: POLICY_PERTURBATIONS[field_name]}),
            "V1",
            "reference",
        )
        assert (
            changed.outcome.final_state != baseline.outcome.final_state
            or changed.outcome.trajectory != baseline.outcome.trajectory
        ), f"{field_name} changes the spec identity but not the episode"

    @pytest.mark.parametrize("field_name", sorted(POLICY_PERTURBATIONS))
    def test_perturbing_a_policy_field_changes_the_spec_identity(
        self, spec: OperationSpec, raw: Mapping[str, Any], field_name: str
    ) -> None:
        """And the converse: a field that moves the episode must move the digest.

        The test above asserts every policy field reaches the run, and its own
        failure message assumes the other direction already holds — "changes the
        spec identity but not the episode". That assumption was untrue for
        ``max_retrieval_batches_per_invocation``, which the hand-written
        semantic projection omitted: the field moved the episode while the
        digest stood still, so one published identity covered two different
        executable policies and neither this table nor any other test noticed.
        Asserting both directions over the same table is what closes it, for
        every field the table is pinned to cover.
        """
        perturbed = _perturbed(raw, **{field_name: POLICY_PERTURBATIONS[field_name]})
        assert perturbed.spec_digest_sha256 != spec.spec_digest_sha256, (
            f"{field_name} changes the episode but not the spec identity"
        )

    def test_approval_validity_is_the_policy_owned_quote_duration(
        self, spec: OperationSpec, raw: Mapping[str, Any]
    ) -> None:
        baseline = run_episode(spec, "V1", "reference")
        quote = baseline.outcome.final_state["quotes"]["quote_1:v1"]
        assert quote["valid_until"] is not None
        short = run_episode(
            _perturbed(raw, approval_validity_minutes=1), "V1", "reference"
        )
        assert (
            short.outcome.final_state["quotes"]["quote_1:v1"]["valid_until"]
            != (quote["valid_until"])
        )
        # A quote that expires before it can be authorised invalidates the
        # reference path rather than quietly authorising expired work.
        assert short.reliable is False

    def test_the_authored_event_no_longer_carries_a_duplicate_duration(
        self, raw: Mapping[str, Any]
    ) -> None:
        for scenario in raw["scenarios"].values():
            for event in scenario["events"]:
                assert "validity_minutes" not in event.get("payload", {})

    def test_a_quote_payload_that_states_its_own_validity_is_refused(
        self, raw: Mapping[str, Any]
    ) -> None:
        body = copy.deepcopy(dict(raw))
        quote = next(
            event
            for event in body["scenarios"]["V1"]["events"]
            if event["type"] == "supplier_quote_received"
        )
        quote["payload"]["validity_minutes"] = 10080
        with pytest.raises(UnknownFieldError):
            OperationSpec.from_mapping(body, source="<duplicate-duration>")


class TestNoGhostTerminals:
    def test_only_the_terminals_this_build_can_produce_are_accepted(self) -> None:
        assert set(TERMINAL_OUTCOMES) == {
            "completed_successfully",
            "transferred_to_human_ownership",
        }

    @pytest.mark.parametrize(
        "terminal", ["cancelled_by_authorised_actor", "failed_irrecoverably"]
    )
    def test_the_loader_refuses_a_terminal_no_scenario_can_reach(
        self, raw: Mapping[str, Any], terminal: str
    ) -> None:
        body = copy.deepcopy(dict(raw))
        body["scenarios"]["V1"]["expected_terminal"] = terminal
        with pytest.raises(UnknownTypeError) as excinfo:
            OperationSpec.from_mapping(body, source="<ghost-terminal>")
        assert terminal in str(excinfo.value)

    def test_every_accepted_terminal_is_produced_by_a_shipped_scenario(
        self, spec: OperationSpec
    ) -> None:
        produced = {
            run_episode(spec, scenario_id, "reference").outcome.status
            for scenario_id in spec.scenario_ids
        }
        assert set(TERMINAL_OUTCOMES) <= produced


# ------------------------------------------ recursive final-state validation


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        canonical_json_text(payload, "operatebench artefact section").encode("utf-8")
    ).hexdigest()


def _resigned(artifact: dict[str, Any]) -> dict[str, Any]:
    """The artefact an attacker would produce: payload changed, digests redone."""
    artifact["final_state_digest_sha256"] = _digest(artifact["final_state"])
    artifact["trajectory_digest_sha256"] = _digest(artifact["trajectory"])
    return artifact


@pytest.fixture()
def artifact(spec: OperationSpec) -> dict[str, Any]:
    return build_artifact(run_episode(spec, "V1", "reference"))


class TestRecursiveFinalStateValidation:
    def test_the_parent_reproduction_is_refused(self, artifact: dict[str, Any]) -> None:
        artifact["final_state"]["cycles"] = []
        with pytest.raises(ArtifactError):
            validate_artifact(artifact)

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("phase",), "MOSTLY_DONE"),
            (("phase",), 7),
            (("replay_final",), "yes"),
            (("reopen_count",), -1),
            (("reopen_count",), True),
            (("issue_id",), 1),
            (("customer_resolution",), "MAYBE"),
            (("cycles",), []),
            (("cycles", "work_cycle_1"), "a cycle"),
            (("cycles", "work_cycle_1", "visit_status"), "TELEPORTED"),
            (("cycles", "work_cycle_1", "requires_payment"), "no"),
            (("cycles", "work_cycle_1", "kind"), "IMAGINARY"),
            (("quotes", "quote_1:v1", "amount_minor"), "64000"),
            (("quotes", "quote_1:v1", "status"), "PENDING"),
            (("quotes", "quote_1:v1", "valid_until"), "whenever"),
            (("approvals", "checkpoint_1", "status"), "FINE"),
            (("approvals", "checkpoint_1", "quote_version"), "1"),
            (("approvals", "checkpoint_1", "opened_at"), "2031-03-03"),
            (("invoices", "invoice_1", "status"), "PAID"),
            (("invoices", "invoice_1", "amount_minor"), 6.4),
            (("obligations",), {"x": {"obligation_id": "x"}}),
            (("payment", "status"), "IN_FLIGHT"),
            (("payment",), []),
            (("assertions",), {}),
            (("communications",), [{"recipient_actor_id": "customer_1"}]),
            (("authoritative_records", "evidence_1", "kind"), "vibes"),
            (("authoritative_records", "evidence_1"), []),
            (("provisional_close", "reopen_until"), 5),
            (("terminal", "outcome"), "vanished_mysteriously"),
            (("exceptions",), []),
            (("hidden",), []),
        ],
    )
    def test_a_malformed_nested_value_is_a_named_artifact_error(
        self, artifact: dict[str, Any], path: tuple[str, ...], value: Any
    ) -> None:
        target: Any = artifact["final_state"]
        for key in path[:-1]:
            target = target[key]
        assert path[-1] in target, f"{path} is not in the canonical state"
        target[path[-1]] = value
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_an_unknown_nested_field_is_refused(self, artifact: dict[str, Any]) -> None:
        artifact["final_state"]["cycles"]["work_cycle_1"]["urgency"] = "high"
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_a_missing_nested_field_is_refused(self, artifact: dict[str, Any]) -> None:
        del artifact["final_state"]["payment"]["settlement_id"]
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_a_record_keyed_against_a_different_identity_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        artifact["final_state"]["cycles"]["work_cycle_1"]["cycle_id"] = "work_cycle_9"
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_an_untouched_artefact_is_still_accepted(
        self, artifact: dict[str, Any]
    ) -> None:
        assert validate_artifact(artifact)["final_state"] == artifact["final_state"]


class TestRecordedDigestsAreRecomputed:
    def test_a_final_state_that_does_not_match_its_digest_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        artifact["final_state"]["reopen_count"] = 99
        with pytest.raises(ArtifactError) as excinfo:
            validate_artifact(artifact)
        assert "final_state_digest_sha256" in str(excinfo.value)

    def test_a_trajectory_that_does_not_match_its_digest_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        artifact["trajectory"] = artifact["trajectory"][:-1]
        with pytest.raises(ArtifactError) as excinfo:
            validate_artifact(artifact)
        assert "trajectory_digest_sha256" in str(excinfo.value)

    def test_a_resigned_digest_does_not_rescue_a_malformed_state(
        self, artifact: dict[str, Any]
    ) -> None:
        artifact["final_state"]["payment"]["status"] = "PAID_SOMEHOW"
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_the_digest_check_happens_before_a_replay_reruns_anything(
        self, spec: OperationSpec, artifact: dict[str, Any]
    ) -> None:
        artifact["final_state_digest_sha256"] = "0" * 64
        with pytest.raises(ArtifactError):
            replay_artifact(spec, artifact)


class TestTrajectoryRecordShapes:
    def test_an_action_proposed_row_without_its_proposal_id_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        row = next(
            row
            for row in artifact["trajectory"]
            if row["record_type"] == "action_proposed"
        )
        del row["proposal_id"]
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_an_unknown_field_on_a_trajectory_row_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        artifact["trajectory"][0]["smuggled"] = "value"
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_a_wrong_typed_trajectory_field_is_refused(
        self, artifact: dict[str, Any]
    ) -> None:
        row = next(
            row
            for row in artifact["trajectory"]
            if row["record_type"] == "effect_accepted"
        )
        row["action_type"] = ["authorise_supplier_work"]
        with pytest.raises(ArtifactError):
            validate_artifact(_resigned(artifact))

    def test_a_file_on_disk_round_trips(
        self, spec: OperationSpec, tmp_path: Path
    ) -> None:
        run = run_episode(spec, "V1", "reference")
        path = write_artifact(run, tmp_path / "v1.json")
        assert read_artifact(path) == build_artifact(run)
        assert json.loads(path.read_text(encoding="utf-8"))["trajectory"]


# --------------------------------------- negative-control terminology and gate


class TestExpectedCausalFailureClosure:
    def test_the_closure_is_readable_under_its_semantic_name(self) -> None:
        for entry in AGENTS.values():
            assert entry.expected_failure_closure == entry.targets

    def test_the_closure_comes_from_the_oracle_not_the_registry(self) -> None:
        # `targets` is a view, not a field. The registry has no way to state one.
        assert isinstance(type(AGENTS["always_act"]).targets, property)
        assert "targets" not in {
            field.name for field in dataclasses.fields(AGENTS["always_act"])
        }

    def test_an_unregistered_extra_failed_dimension_breaks_the_gate(
        self, spec: OperationSpec
    ) -> None:
        oracle = negative_control_oracle()
        real = oracle.control("always_act")
        narrowed = oracle.replacing(
            dataclasses.replace(
                real,
                expected_failed_dimensions=tuple(
                    name
                    for name in real.expected_failed_dimensions
                    if name != "action_validity"
                ),
                must_pass_dimensions=(*real.must_pass_dimensions, "action_validity"),
            )
        )
        check = check_agent(spec, AGENTS["always_act"], "V1", oracle=narrowed)
        assert check.ok is False
        assert "action_validity" in " ".join(check.problems)

    def test_the_sharp_isolation_control_still_isolates(
        self, spec: OperationSpec
    ) -> None:
        run = run_episode(spec, "V1", "trust_actor_claim")
        # Three at oracle 0.4.0, not two: acting on what the event asserted is
        # both a crossing of the authority ladder and a failure to establish the
        # fact, and the two are reported separately on purpose.
        assert set(run.evaluation.failed_dimensions) == {
            "authority_boundaries",
            "action_validity",
            "retrieval_discipline",
        }
        assert len(run.evaluation.passed_dimensions) >= 3


class TestActionPayloadSchemasAreComplete:
    def test_every_agent_action_has_an_executable_payload_schema(self) -> None:
        from operatebench.domains.lettings.maintenance.spec import (
            MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
        )

        assert set(MAINTENANCE_ACTION_PAYLOAD_SCHEMAS) == {
            name for name in MAINTENANCE_ACTION_TYPES if name != "complete"
        }

    def test_an_action_type_with_no_schema_is_a_named_refusal(self) -> None:
        from operatebench.domains.lettings.maintenance.spec import (
            validate_action_payload,
        )

        with pytest.raises(SpecSchemaError):
            validate_action_payload("teleport_the_supplier", {}, "action")
