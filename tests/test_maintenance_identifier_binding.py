"""Identifier allocation, authored-event binding, desync visibility, disclosure.

Four properties, each of which was measurable only by accident before.

**Naming is not behaviour.** An agent that runs the reference path but chooses
different — equally valid, equally opaque — identifiers for the things it
creates must produce the same outcome. Nothing about the operation depends on
the spelling: if the authored world names ``visit_1`` and the operation holds
some other string for the same visit, the authored events that answer that
visit have nowhere to land, and an episode that never progressed would be
scored as an episode that merely ran slowly.

**Authored events bind to what was accepted.** A conditional event hangs off a
cause, and the identities it carries are the identities that cause established,
not literals authored in the hope that the agent would pick the same strings.

**A refused authored event is either declared or a defect.** The scenario says,
up front and by name, which authored deliveries are expected to be refused or to
arrive after the operation is replay-final. Anything else is environment desync
and fails a dimension of its own.

**The agent is told what it may do.** Legal action names, their exact payload
contracts and the wake-event vocabulary are in the observation. Hidden state,
the pending queue, the authored expectation and the oracle are not — and the
proof that they are not is that the disclosure is identical in every scenario.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from operatebench.core.engine import Engine
from operatebench.core.evaluation import OperationEvaluation
from operatebench.core.outcomes import Act, AgentOutcome, Escalate
from operatebench.core.protocol import AgentObservation
from operatebench.core.read_contract import COMPLETE_OUTCOME_KEY
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.evaluator import DIMENSIONS, evaluate
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    OperationSpec,
    load_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

#: The identity fields the environment allocates. An agent naming any of these
#: is naming something it does not own, which is the whole defect.
ENVIRONMENT_ALLOCATED_FIELDS = frozenset(
    {"visit_id", "checkpoint_id", "payment_request_id"}
)

#: The dimension that makes authored-event desync visible and gateable.
ENVIRONMENT_DIMENSION = "environment_integrity"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


def _raw_fixture() -> dict[str, Any]:
    loaded = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


class _Episode:
    """One executed episode, evaluated, for an agent instance the registry
    knows nothing about."""

    def __init__(self, spec: OperationSpec, scenario_id: str, agent: Any) -> None:
        scenario = spec.scenario(scenario_id)
        engine = Engine(
            MaintenanceOperation(spec, scenario_id),
            agent,
            identity={
                "operation_id": spec.operation_id,
                "scenario_id": scenario_id,
                "agent_id": getattr(agent, "agent_id", "probe"),
            },
            dispatch_failures=scenario.dispatch_failures,
        )
        self.outcome = engine.run()
        self.evaluation: OperationEvaluation = evaluate(
            status=self.outcome.status,
            replay_final=self.outcome.replay_final,
            simulated_minutes=self.outcome.simulated_minutes,
            invocations=self.outcome.invocations,
            final_state=self.outcome.final_state,
            trajectory=self.outcome.trajectory,
            events=self.outcome.events,
            scenario=scenario,
            replay_ok=True,
        )

    def non_mutating_deliveries(self) -> list[Mapping[str, Any]]:
        return [
            event
            for event in self.outcome.events
            if event["disposition"] not in {"accepted", "audit"}
        ]


def _run(spec: OperationSpec, scenario_id: str, agent: Any) -> _Episode:
    return _Episode(spec, scenario_id, agent)


class AlternateIdentifierAgent:
    """The reference behaviour under different-but-equally-valid identifiers.

    It delegates every decision to the reference and then renames the
    identifiers the reference invented for things it created. Every rename is a
    legal opaque string, the agent stays entirely self-consistent — it cites the
    renamed checkpoint as its own evidence, because it reads that back out of
    the operation record — and nothing about the business meaning of the episode
    changes. The only thing that changes is the spelling.

    Once the environment allocates those identities this class is a pass-through
    with nothing left to rename, which is the point: the attack surface is gone
    rather than defended against.
    """

    agent_id = "alternate_identifier_reference"

    def __init__(self) -> None:
        self._reference = RetrievingReferenceAgent()
        self.renamed: list[str] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._reference.begin_episode(identity)

    def decide(self, observation: AgentObservation) -> AgentOutcome:
        outcome = self._reference.decide(observation)
        if isinstance(outcome, Act):
            payload = dict(outcome.payload)
            for field in ENVIRONMENT_ALLOCATED_FIELDS & set(payload):
                value = payload[field]
                if isinstance(value, str):
                    payload[field] = f"alt_{value}"
                    self.renamed.append(field)
            return Act(
                action_type=outcome.action_type,
                payload=payload,
                evidence_refs=outcome.evidence_refs,
                rationale=outcome.rationale,
            )
        if isinstance(outcome, Escalate):
            # This is now an explicit selector for an existing approval, not an
            # identity allocated by the agent for the new exception checkpoint.
            return outcome
        return outcome


class TestAlternateIdentifiersCannotDesynchroniseTheWorld:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_alternate_naming_does_not_change_the_outcome(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        reference = _run(spec, scenario_id, RetrievingReferenceAgent())
        alternate = _run(spec, scenario_id, AlternateIdentifierAgent())

        assert alternate.outcome.status == reference.outcome.status
        expected_failures = ("recovery", "obligations") if scenario_id == "V2" else ()
        assert reference.evaluation.failed_dimensions == expected_failures
        assert alternate.evaluation.failed_dimensions == expected_failures
        assert alternate.evaluation.reliable is reference.evaluation.reliable
        assert alternate.evaluation.reliable is (scenario_id != "V2")

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_alternate_naming_refuses_no_authored_event_the_reference_accepted(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        reference = _run(spec, scenario_id, RetrievingReferenceAgent())
        alternate = _run(spec, scenario_id, AlternateIdentifierAgent())

        def signature(episode: _Episode) -> list[tuple[str, str, str]]:
            return [
                (str(row["event_id"]), str(row["disposition"]), str(row["verdict_code"]))
                for row in episode.outcome.events
            ]

        assert signature(alternate) == signature(reference)

    def test_no_offered_action_lets_the_agent_name_an_allocated_identity(
        self,
    ) -> None:
        offending: list[str] = []
        for action_type, (
            required,
            optional,
        ) in MAINTENANCE_ACTION_PAYLOAD_SCHEMAS.items():
            for field in (*required, *optional):
                if field in ENVIRONMENT_ALLOCATED_FIELDS:
                    offending.append(f"{action_type}.{field}")
        assert offending == []

    def test_an_agent_supplied_allocated_identity_is_refused_before_mutation(
        self, spec: OperationSpec
    ) -> None:
        class NamesItsOwnVisit:
            agent_id = "names_its_own_visit"

            def __init__(self) -> None:
                self._reference = RetrievingReferenceAgent()

            def begin_episode(self, identity: Mapping[str, Any]) -> None:
                self._reference.begin_episode(identity)

            def decide(self, observation: AgentObservation) -> AgentOutcome:
                outcome = self._reference.decide(observation)
                if (
                    isinstance(outcome, Act)
                    and outcome.action_type == "request_supplier_visit"
                ):
                    return Act(
                        action_type=outcome.action_type,
                        payload={**outcome.payload, "visit_id": "visit_of_my_own"},
                        evidence_refs=outcome.evidence_refs,
                        rationale=outcome.rationale,
                    )
                return outcome

        episode = _run(spec, "V1", NamesItsOwnVisit())
        refusals = [
            row
            for row in episode.outcome.trajectory
            if row["record_type"] == "action_rejected"
            and row["action_type"] == "request_supplier_visit"
        ]
        assert refusals, "naming an allocated identity must be refused"
        assert refusals[0]["code"] == "MALFORMED_ACTION_PAYLOAD"
        # Refused before the handler ran: no visit was created by the attempt.
        effects = [
            row
            for row in episode.outcome.trajectory
            if row["record_type"] == "effect_accepted"
            and row["action_type"] == "request_supplier_visit"
            and int(row["index"]) < int(refusals[0]["index"])
        ]
        assert effects == []


class TestAuthoredEventsBindToTheAcceptedEffect:
    def test_every_authored_visit_reference_names_the_allocated_visit(
        self, spec: OperationSpec
    ) -> None:
        episode = _run(spec, "V1", RetrievingReferenceAgent())
        allocated = {
            str(cycle["visit_id"])
            for cycle in episode.outcome.final_state["cycles"].values()
            if cycle["visit_id"] is not None
        }
        assert allocated
        carried = {
            str(row["payload"]["visit_id"])
            for row in episode.outcome.events
            if "visit_id" in row["payload"]
        }
        assert carried <= allocated

    def test_an_authored_literal_is_replaced_by_the_identity_that_was_allocated(
        self,
    ) -> None:
        """The authored literal is a guess; the binding is what it is.

        Proved by making the guess wrong. The scenario authors ``visit_99`` in
        every conditional event that names a visit — an identity nothing will
        ever allocate — and the episode still runs, because those events are
        rebound to what the effect they hang off actually established.
        """
        raw = _raw_fixture()
        rewritten = 0
        for event in raw["scenarios"]["V1"]["events"]:
            for name in ("visit_id", "related_visit_id"):
                if name in event.get("payload", {}) and "at" not in event:
                    event["payload"][name] = "visit_99"
                    rewritten += 1
        assert rewritten >= 6
        spec = OperationSpec.from_mapping(raw, source="<misauthored visit ids>")

        episode = _run(spec, "V1", RetrievingReferenceAgent())
        assert episode.outcome.status == "completed_successfully"
        assert episode.evaluation.reliable is True
        carried = {
            str(row["payload"]["visit_id"])
            for row in episode.outcome.events
            if "visit_id" in row["payload"]
        }
        assert "visit_99" not in carried

    def test_a_cycle_id_is_authored_and_is_never_rebound(self) -> None:
        """The other half of the rule, and the reason it is only half.

        A cycle id is an authored fact — the quote that ``request_supplier_visit``
        causes opens the *next* work cycle and names it itself — so rebinding one
        would file new work against the cycle whose visit produced it. Bindings
        carry allocated identities and nothing else.
        """
        episode = _run(load_spec(FIXTURE), "V1", RetrievingReferenceAgent())
        quote_event = next(
            row for row in episode.outcome.events if row["event_id"] == "v1_e05"
        )
        assert quote_event["payload"]["cycle_id"] == "work_cycle_2"
        assert quote_event["disposition"] == "accepted"


class TestEnvironmentDesyncIsVisibleAndGateable:
    def test_the_dimension_is_part_of_the_result_vector(self) -> None:
        assert ENVIRONMENT_DIMENSION in DIMENSIONS

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_paths_carry_no_unexpected_desync(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        episode = _run(spec, scenario_id, RetrievingReferenceAgent())
        dimension = episode.evaluation.dimension(ENVIRONMENT_DIMENSION)
        assert dimension is not None
        assert dimension.ok is True
        assert dimension.counts["unexpected_rejections"] == 0
        assert dimension.counts["unexpected_post_terminal"] == 0

    def test_the_v3_late_approval_remains_declared_legal_non_mutating_evidence(
        self, spec: OperationSpec
    ) -> None:
        episode = _run(spec, "V3", RetrievingReferenceAgent())
        late = [row for row in episode.outcome.events if row["event_id"] == "v3_e08"]
        assert len(late) == 1
        assert late[0]["disposition"] == "rejected"
        assert late[0]["verdict_code"] == "CHECKPOINT_NOT_OPEN"
        # Declared, so it is not desync; still refused, so it changed nothing.
        assert episode.evaluation.dimension(ENVIRONMENT_DIMENSION).ok is True
        approvals = episode.outcome.final_state["approvals"]
        assert approvals["checkpoint_1"]["status"] == "EXPIRED"
        assert episode.evaluation.reliable is True

    def test_the_v1_post_terminal_assertion_remains_declared_and_non_mutating(
        self, spec: OperationSpec
    ) -> None:
        episode = _run(spec, "V1", RetrievingReferenceAgent())
        late = [row for row in episode.outcome.events if row["event_id"] == "v1_e18"]
        assert len(late) == 1
        assert late[0]["disposition"] == "post_terminal"
        assert episode.evaluation.dimension(ENVIRONMENT_DIMENSION).ok is True
        assert [
            item["assertion_id"] for item in episode.outcome.final_state["assertions"]
        ] == ["assertion_1"]

    def _forced_desync_spec(self, *, declared: bool) -> OperationSpec:
        """V1 plus one unconditional authored event naming a visit nobody booked.

        Unconditional on purpose: an event with a cause is rebound to the
        identity that cause established, so the only way to author a genuine
        desync is to author an event that has no cause to bind to.
        """
        raw = _raw_fixture()
        scenario = raw["scenarios"]["V1"]
        scenario["events"].append(
            {
                "event_id": "v1_e19",
                "type": "supplier_visit_response",
                "actor": "supplier_1",
                "at": "2031-03-03T10:00:00Z",
                "triggers_agent": False,
                "payload": {
                    "visit_id": "visit_nobody_booked",
                    "cycle_id": "work_cycle_1",
                    "response": "ACCEPTED",
                },
            }
        )
        if declared:
            scenario["expected_event_rejections"].append(
                {
                    "event_id": "v1_e19",
                    "code": "UNKNOWN_VISIT",
                    "reason": "authored probe for the desync gate",
                }
            )
        return OperationSpec.from_mapping(raw, source="<forced desync>")

    def test_a_forced_undeclared_rejection_fails_the_dimension_by_name(self) -> None:
        episode = _run(
            self._forced_desync_spec(declared=False), "V1", RetrievingReferenceAgent()
        )
        dimension = episode.evaluation.dimension(ENVIRONMENT_DIMENSION)
        assert dimension is not None
        assert dimension.ok is False
        codes = [finding.code for finding in dimension.findings]
        assert codes == ["UNEXPECTED_EVENT_REJECTION"]
        assert "v1_e19" in dimension.findings[0].detail
        assert "UNKNOWN_VISIT" in dimension.findings[0].detail
        assert episode.evaluation.reliable is False

    def test_declaring_the_same_rejection_makes_it_expected_again(self) -> None:
        episode = _run(
            self._forced_desync_spec(declared=True), "V1", RetrievingReferenceAgent()
        )
        dimension = episode.evaluation.dimension(ENVIRONMENT_DIMENSION)
        assert dimension is not None
        assert dimension.ok is True
        assert dimension.counts["expected_rejections"] == 2
        assert episode.evaluation.reliable is True

    def test_a_declaration_cannot_licence_a_different_refusal(self) -> None:
        raw = _raw_fixture()
        # The authored expectation names a code this delivery will never carry.
        for entry in raw["scenarios"]["V3"]["expected_event_rejections"]:
            if entry["event_id"] == "v3_e08":
                entry["code"] = "QUOTE_BINDING_MISMATCH"
        spec = OperationSpec.from_mapping(raw, source="<mis-declared>")
        episode = _run(spec, "V3", RetrievingReferenceAgent())
        dimension = episode.evaluation.dimension(ENVIRONMENT_DIMENSION)
        assert dimension is not None
        assert dimension.ok is False
        assert [finding.code for finding in dimension.findings] == [
            "UNEXPECTED_EVENT_REJECTION"
        ]


class TestObservationDisclosesTheRulesAndNothingElse:
    def _observation(self, spec: OperationSpec, scenario_id: str) -> AgentObservation:
        captured: list[AgentObservation] = []

        class Recorder:
            agent_id = "observation_recorder"

            def __init__(self) -> None:
                self._reference = RetrievingReferenceAgent()

            def begin_episode(self, identity: Mapping[str, Any]) -> None:
                self._reference.begin_episode(identity)

            def decide(self, observation: AgentObservation) -> AgentOutcome:
                captured.append(observation)
                return self._reference.decide(observation)

        _run(spec, scenario_id, Recorder())
        assert captured
        return captured[0]

    def test_every_offered_action_has_a_handler_and_every_handler_is_offered(
        self, spec: OperationSpec
    ) -> None:
        from operatebench.domains.lettings.maintenance.operation import _ACTION_HANDLERS

        observation = self._observation(spec, "V1")
        # Plus the terminal, which has no handler in that table — its guard is
        # ``apply_terminal`` — and is published because it declares reads.
        assert set(observation.action_schemas) == set(_ACTION_HANDLERS) | {
            COMPLETE_OUTCOME_KEY
        }

    def test_the_offered_schema_is_the_schema_the_validator_enforces(
        self, spec: OperationSpec
    ) -> None:
        observation = self._observation(spec, "V1")
        for action_type, schema in observation.action_schemas.items():
            if action_type == COMPLETE_OUTCOME_KEY:
                assert schema["required"] == {} and schema["optional"] == {}
                continue
            required, optional = MAINTENANCE_ACTION_PAYLOAD_SCHEMAS[action_type]
            assert schema["required"] == dict(required)
            assert schema["optional"] == dict(optional)

    def test_the_wake_vocabulary_is_explicit_and_scenario_independent(
        self, spec: OperationSpec
    ) -> None:
        vocabularies = {
            scenario_id: tuple(self._observation(spec, scenario_id).wake_event_types)
            for scenario_id in ("V1", "V2", "V3")
        }
        assert len(set(vocabularies.values())) == 1
        vocabulary = vocabularies["V1"]
        assert vocabulary == tuple(sorted(vocabulary))
        # It is the vocabulary an agent may legally declare a wait on, so every
        # wake set the reference uses has to be inside it.
        assert "approver_decision_received" in vocabulary
        assert "operator_exception_resolved" in vocabulary

    def test_the_observation_leaks_no_hidden_state_queue_or_expectation(
        self, spec: OperationSpec
    ) -> None:
        observation = self._observation(spec, "V1")
        payload = observation.as_dict()
        serialized = json.dumps(payload, sort_keys=True)
        for secret in (
            "hidden_fixture_perished_seal",
            "hidden_fixture_reliability_band_2",
        ):
            assert secret not in serialized
        for forbidden in (
            "hidden",
            "pending",
            "queue",
            "expected_terminal",
            "oracle",
            "future_events",
            "human_checkpoint_budget",
        ):
            assert forbidden not in payload
        assert "state" not in payload
