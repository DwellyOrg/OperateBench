"""Phase 1A: the engine's retrieval turn, its budgets and its invalidation.

Retrieval is served by the *environment*. An agent returns a batch; the engine
alone calls :meth:`OperationDomain.serve_retrieval`, writes one provenance row
per result, and puts the results in front of the next observation of the same
invocation. Nothing an agent holds survives a wake, and nothing survives its own
accepted effect either — the record it was looking at has moved underneath it,
and in this build the agent is the thing that has to ask again.

These tests drive the shipped engine with scripted agents, so what they assert
is the loop, the ledger grammar and the observation, not a domain's rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from operatebench.agents.evidence import decision_projection
from operatebench.core.engine import Engine
from operatebench.core.ledger import RECORD_TYPES
from operatebench.core.outcomes import Act, AgentOutcome, Complete, Wait
from operatebench.core.protocol import (
    MODEL_VISIBLE_FIELDS,
    AgentObservation,
    model_projection,
)
from operatebench.core.retrieval import (
    INVALIDATION_EFFECT_ACCEPTED,
    INVALIDATION_WAKE,
    MAX_REQUESTS_PER_BATCH,
    RETRIEVAL_AUTHORITIES,
    RETRIEVAL_BUDGET_EXHAUSTED,
    RETRIEVAL_EMPTY_BATCH,
    RETRIEVAL_UNKNOWN_TOOL,
    RetrievalRequest,
    RetrieveBatch,
)
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOL_NAMES,
)
from operatebench.domains.lettings.maintenance.spec import action_schema_view, load_spec

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

DEFAULT_BATCH_BUDGET = 6


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


class ScriptedAgent:
    """Returns a scripted decision per call, then waits forever after."""

    agent_id = "scripted"

    def __init__(
        self, script: Sequence[Any], *, tail: AgentOutcome | None = None
    ) -> None:
        self._script = list(script)
        self._tail = tail or Wait(
            reason="script exhausted", fallback_after_minutes=100000
        )
        self.observations: list[AgentObservation] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: AgentObservation) -> Any:
        self.observations.append(observation)
        if self._script:
            return self._script.pop(0)
        return self._tail


def run(spec, agent, scenario_id: str = "V1"):
    domain = MaintenanceOperation(spec, scenario_id)
    engine = Engine(
        domain,
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opinst_test_0000000000000000",
            "scenario_id": scenario_id,
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=spec.scenario(scenario_id).dispatch_failures,
    )
    return engine.run(), engine


def rows(outcome, record_type: str) -> list[Mapping[str, Any]]:
    return [row for row in outcome.trajectory if row["record_type"] == record_type]


def batch(*tools: str) -> RetrieveBatch:
    return RetrieveBatch(tuple(RetrievalRequest(tool=tool) for tool in tools))


# -- the observation --------------------------------------------------------


class TestTheObservationCarriesRetrievalAffordances:
    def test_the_allowlist_carries_the_read_affordances_and_no_record(self, spec) -> None:
        # Phase 2 completed the migration: the normalized projection and the bare
        # trigger envelope are gone, and what is left is the coarse phase, the
        # waking event as a claim, and the reads.
        for name in ("phase", "event", "retrieval"):
            assert name in MODEL_VISIBLE_FIELDS
        assert "state" not in MODEL_VISIBLE_FIELDS
        assert "trigger" not in MODEL_VISIBLE_FIELDS

    def test_the_first_turn_of_every_invocation_is_served_nothing(self, spec) -> None:
        agent = ScriptedAgent([])
        run(spec, agent)
        first_turns = [obs for obs in agent.observations if obs.turn_index == 0]
        assert first_turns
        assert all(obs.retrieval["served"] == {} for obs in first_turns)

    def test_the_retrieval_affordance_states_its_own_bounds(self, spec) -> None:
        agent = ScriptedAgent([])
        run(spec, agent)
        first = agent.observations[0]
        affordance = first.retrieval
        assert sorted(affordance) == [
            "batch_budget_remaining",
            "catalogue",
            "max_requests_per_batch",
            "record_contract",
            "served",
        ]
        assert affordance["max_requests_per_batch"] == MAX_REQUESTS_PER_BATCH
        assert affordance["batch_budget_remaining"] == DEFAULT_BATCH_BUDGET
        assert affordance["record_contract"] == [
            "maintenance_case",
            "maintenance retrieval record",
            "canonical-json-sha256-prefix-v1",
            16,
        ]
        assert sorted(affordance["catalogue"]) == sorted(MAINTENANCE_RETRIEVAL_TOOL_NAMES)
        for entry in affordance["catalogue"].values():
            assert sorted(entry) == [
                "arguments",
                "authority",
                "schema_id",
                "source",
            ]
            assert entry["authority"] in RETRIEVAL_AUTHORITIES

    def test_each_observation_and_projection_gets_a_detached_json_contract(
        self, spec
    ) -> None:
        agent = ScriptedAgent([batch("get_case_record")])
        run(spec, agent)
        contracts = [
            observation.retrieval["record_contract"] for observation in agent.observations
        ]
        assert len(contracts) > 1
        assert all(type(contract) is list for contract in contracts)
        assert len({id(contract) for contract in contracts}) == len(contracts)

        projected = model_projection(agent.observations[0])
        projected_contract = projected["retrieval"]["record_contract"]
        assert projected_contract is not contracts[0]
        projected_contract.append("mutated")
        assert len(contracts[0]) == 4

    def test_the_event_envelope_states_authority_and_the_projection_carries_it(
        self, spec
    ) -> None:
        agent = ScriptedAgent([])
        run(spec, agent)
        first = agent.observations[0]
        assert first.event is not None
        assert sorted(first.event) == [
            "actor_id",
            "authority",
            "event_id",
            "event_type",
            "is_authoritative",
            "payload",
        ]
        assert first.event["authority"] in RETRIEVAL_AUTHORITIES
        assert first.event["is_authoritative"] is False
        assert first.phase
        projection = model_projection(first)
        assert tuple(sorted(projection)) == tuple(sorted(MODEL_VISIBLE_FIELDS))
        assert projection["retrieval"]["served"] == {}

    def test_every_action_publishes_the_reads_it_rests_on(self, spec) -> None:
        agent = ScriptedAgent([])
        run(spec, agent)
        schemas = agent.observations[0].action_schemas
        assert schemas
        canonical = action_schema_view()
        assert schemas == canonical
        for name, schema in schemas.items():
            assert set(schema) == set(canonical[name]), name
            assert schema["evidence_refs"] == canonical[name]["evidence_refs"], name
            assert isinstance(schema["reads"], Mapping), name
            assert set(schema["reads"]) <= set(MAINTENANCE_RETRIEVAL_TOOL_NAMES), name
            assert set(schema["reads"].values()) <= {"required"}, name
            assert list(schema["reads"]) == sorted(schema["reads"]), name


# -- serving ----------------------------------------------------------------


class TestTheEnvironmentServesTheBatch:
    def test_the_reconstructed_three_read_decision_reaches_turn_one(self, spec) -> None:
        decision = batch("get_case_record", "list_checkpoints", "list_quotes")
        assert (
            decision_projection(decision).decision_digest_sha256
            == "d5a438e024d151a4fbe6354698eb564133b781bf49f16820af07cebae62da6e9"
        )
        agent = ScriptedAgent([decision])

        run(spec, agent)

        assert agent.observations[0].retrieval["served"] == {}
        assert sorted(agent.observations[1].retrieval["served"]) == [
            "get_case_record",
            "list_checkpoints",
            "list_quotes",
        ]

    def test_results_come_back_in_canonical_request_order_at_one_instant(
        self, spec
    ) -> None:
        agent = ScriptedAgent([batch("list_quotes", "get_case_record")])
        outcome, _ = run(spec, agent)
        served_rows = rows(outcome, "retrieval_served")
        assert [row["tool"] for row in served_rows[:2]] == [
            "get_case_record",
            "list_quotes",
        ]
        assert len({row["as_of"] for row in served_rows[:2]}) == 1
        assert [row["request_index"] for row in served_rows[:2]] == [0, 1]
        assert {row["batch_index"] for row in served_rows[:2]} == {0}
        assert {row["initiated_by"] for row in served_rows[:2]} == {"agent"}

    def test_a_served_result_reaches_the_next_observation_of_the_invocation(
        self, spec
    ) -> None:
        agent = ScriptedAgent([batch("get_case_record")])
        run(spec, agent)
        assert agent.observations[0].retrieval["served"] == {}
        served = agent.observations[1].retrieval["served"]
        assert sorted(served) == ["get_case_record"]
        body = served["get_case_record"]
        assert sorted(body) == [
            "as_of",
            "authority",
            "error_code",
            "ok",
            "record_id",
            "record_version",
            "records",
            "schema_id",
            "source",
            "tool",
        ]
        assert (
            "reads"
            not in model_projection(agent.observations[1])["action_schemas"][
                "request_supplier_visit"
            ]
        )
        assert body["ok"] is True
        assert body["records"]

    def test_the_budget_counts_batches_and_is_published_as_it_is_spent(
        self, spec
    ) -> None:
        agent = ScriptedAgent([batch("list_quotes"), batch("list_billing")])
        run(spec, agent)
        assert [
            obs.retrieval["batch_budget_remaining"] for obs in agent.observations[:3]
        ] == [DEFAULT_BATCH_BUDGET, DEFAULT_BATCH_BUDGET - 1, DEFAULT_BATCH_BUDGET - 2]

    def test_a_retrieval_turn_is_not_a_business_outcome(self, spec) -> None:
        agent = ScriptedAgent([batch("list_quotes"), batch("list_billing")])
        outcome, _ = run(spec, agent)
        turns = [row["turn_index"] for row in rows(outcome, "retrieval_served")]
        # turn_index advances on every agent call, including retrieval turns.
        assert min(turns) == 0
        first_outcome = rows(outcome, "agent_outcome")[0]
        assert first_outcome["turn_index"] == 2


# -- refusals ---------------------------------------------------------------


class TestRetrievalRefusalsAreNamedAndNonMutating:
    @pytest.mark.parametrize(
        ("bad", "code"),
        [
            (RetrieveBatch(()), RETRIEVAL_EMPTY_BATCH),
            (
                RetrieveBatch((RetrievalRequest(tool="no_such_tool"),)),
                RETRIEVAL_UNKNOWN_TOOL,
            ),
        ],
    )
    def test_a_refused_batch_serves_nothing_and_is_recorded(
        self, spec, bad: RetrieveBatch, code: str
    ) -> None:
        agent = ScriptedAgent([bad])
        outcome, _ = run(spec, agent)
        refusals = rows(outcome, "retrieval_refused")
        assert refusals
        assert refusals[0]["code"] == code
        assert refusals[0]["invocation_index"] == 1
        assert refusals[0]["turn_index"] == 0
        assert agent.observations[1].retrieval["served"] == {}
        assert agent.observations[1].last_rejection is not None
        assert agent.observations[1].last_rejection["code"] == code

    def test_the_budget_is_exhausted_by_name_rather_than_silently_truncated(
        self, spec
    ) -> None:
        agent = ScriptedAgent([batch("list_quotes")] * (DEFAULT_BATCH_BUDGET + 1))
        outcome, _ = run(spec, agent)
        served_batches = {row["batch_index"] for row in rows(outcome, "retrieval_served")}
        assert served_batches == set(range(DEFAULT_BATCH_BUDGET))
        refusals = rows(outcome, "retrieval_refused")
        assert refusals[0]["code"] == RETRIEVAL_BUDGET_EXHAUSTED

    def test_a_run_away_retriever_stops_with_a_named_violation(self, spec) -> None:
        agent = ScriptedAgent([], tail=batch("list_quotes"))
        outcome, _ = run(spec, agent)
        violations = rows(outcome, "critical_violation")
        assert violations
        assert violations[0]["code"] == "INVOCATION_CALL_LIMIT_EXCEEDED"
        assert outcome.status == "operation_deadlock"


# -- invalidation -----------------------------------------------------------


class TestDerivedContextIsInvalidated:
    def test_an_accepted_effect_clears_a_stale_business_rejection(self, spec) -> None:
        rejected = Act(action_type="not_a_real_action", payload={})
        accepted = Act(
            action_type="request_supplier_visit",
            payload={"visit_type": "DIAGNOSTIC", "cycle_id": "work_cycle_1"},
        )
        agent = ScriptedAgent(
            [batch(*MAINTENANCE_RETRIEVAL_TOOL_NAMES), rejected, accepted]
        )

        run(spec, agent)

        assert agent.observations[2].last_rejection == {
            "action_type": "not_a_real_action",
            "code": "ACTION_OUTSIDE_AGENT_AUTHORITY",
            "detail": "the agent does not hold 'not_a_real_action'",
        }
        assert agent.observations[3].last_rejection is None

    def test_a_rejected_business_action_remains_visible_on_the_next_turn(
        self, spec
    ) -> None:
        rejected = Act(action_type="not_a_real_action", payload={})
        agent = ScriptedAgent([rejected])

        run(spec, agent)

        assert agent.observations[1].last_rejection == {
            "action_type": "not_a_real_action",
            "code": "ACTION_OUTSIDE_AGENT_AUTHORITY",
            "detail": "the agent does not hold 'not_a_real_action'",
        }

    def test_a_wake_clears_the_served_set_and_records_it(self, spec) -> None:
        agent = ScriptedAgent([])
        outcome, _ = run(spec, agent)
        wakes = [
            row
            for row in rows(outcome, "derived_context_invalidated")
            if row["cause"] == INVALIDATION_WAKE
        ]
        invoked = rows(outcome, "agent_invoked")
        assert len(wakes) == len(invoked) >= 1
        assert all(row["turn_index"] == 0 for row in wakes)

    def test_an_accepted_own_effect_clears_the_served_set(self, spec) -> None:
        act = Act(
            action_type="request_supplier_visit",
            payload={"visit_type": "DIAGNOSTIC", "cycle_id": "work_cycle_1"},
            rationale="book the visit this cycle is entitled to",
        )
        agent = ScriptedAgent([batch(*MAINTENANCE_RETRIEVAL_TOOL_NAMES), act])
        outcome, _ = run(spec, agent)
        accepted = rows(outcome, "effect_accepted")
        assert accepted
        cleared = [
            row
            for row in rows(outcome, "derived_context_invalidated")
            if row["cause"] == INVALIDATION_EFFECT_ACCEPTED
        ]
        assert cleared
        assert cleared[0]["action_type"] == "request_supplier_visit"
        assert cleared[0]["cleared_tools"] == sorted(MAINTENANCE_RETRIEVAL_TOOL_NAMES)
        # The next observation of the same invocation is served nothing, and the
        # environment has not refreshed anything on the agent's behalf.
        assert agent.observations[2].retrieval["served"] == {}

    def test_the_invalidation_row_follows_the_effect_it_names(self, spec) -> None:
        act = Act(
            action_type="request_supplier_visit",
            payload={"visit_type": "DIAGNOSTIC", "cycle_id": "work_cycle_1"},
        )
        agent = ScriptedAgent([batch("get_case_record"), act])
        outcome, _ = run(spec, agent)
        order = [row["record_type"] for row in outcome.trajectory]
        effect = order.index("effect_accepted")
        assert "derived_context_invalidated" in order[effect:]
        assert order[effect + 1] == "derived_context_invalidated"

    def test_what_an_observation_carries_is_exactly_what_survived(self, spec) -> None:
        # The general statement, walked turn by turn: an observation's served set
        # is exactly the tools served since the last invalidation, so no result
        # outlives the invalidation that dropped it and none appears from nowhere.
        act = Act(
            action_type="request_supplier_visit",
            payload={"visit_type": "DIAGNOSTIC", "cycle_id": "work_cycle_1"},
        )
        agent = ScriptedAgent([batch("get_case_record"), act, batch("list_billing")])
        outcome, _ = run(spec, agent)
        # Replay the ledger to derive what each agent call should have been
        # shown, and compare it with what the agent was actually handed.
        expected: dict[tuple[int, int], list[str]] = {}
        invocation = 0
        alive: set[str] = set()
        for row in outcome.trajectory:
            kind = row["record_type"]
            if kind == "agent_invoked":
                invocation = row["invocation_index"]
            elif kind == "derived_context_invalidated":
                alive = set()
            elif kind == "retrieval_served":
                expected.setdefault((invocation, row["turn_index"]), sorted(alive))
                alive.add(row["tool"])
            elif kind in {"agent_outcome", "outcome_rejected"}:
                expected.setdefault((invocation, row["turn_index"]), sorted(alive))
        assert expected, "the run made at least one recorded agent call"
        for observation in agent.observations:
            key = (observation.invocation_index, observation.turn_index)
            if key in expected:
                assert sorted(observation.retrieval["served"]) == expected[key], key


# -- the ledger vocabulary --------------------------------------------------


class TestTheLedgerNamesTheNewRows:
    def test_the_three_record_types_are_in_the_vocabulary(self) -> None:
        for name in (
            "retrieval_served",
            "retrieval_refused",
            "derived_context_invalidated",
        ):
            assert name in RECORD_TYPES

    def test_a_served_row_carries_its_whole_provenance(self, spec) -> None:
        agent = ScriptedAgent([batch("list_authoritative_records")])
        outcome, _ = run(spec, agent)
        row = rows(outcome, "retrieval_served")[0]
        assert sorted(set(row) - {"index", "at", "record_type"}) == [
            "as_of",
            "authority",
            "batch_index",
            "initiated_by",
            "invocation_index",
            "ok",
            "record_id",
            "record_version",
            "records",
            "request_index",
            "schema_id",
            "source",
            "tool",
            "turn_index",
        ]
        assert row["authority"] == "authoritative_verification"
        assert row["schema_id"] == "maintenance.list_authoritative_records.v1"
        assert len(row["record_version"]) == 16


# -- the agent never executes a read ---------------------------------------


class TestOnlyTheEnvironmentServes:
    def test_the_domain_is_the_only_thing_that_can_serve(self, spec) -> None:
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        results = domain.serve_retrieval(
            state,
            (RetrievalRequest(tool="get_case_record"),),
            "2025-01-01T00:00:00Z",
        )
        assert len(results) == 1
        assert results[0].as_of == "2025-01-01T00:00:00Z"
        assert results[0].tool == "get_case_record"
        assert "phase" in results[0].records

    def test_a_second_serve_of_an_unmoved_record_reports_the_same_version(
        self, spec
    ) -> None:
        domain = MaintenanceOperation(spec, "V1")
        state = domain.initial_state()
        request = (RetrievalRequest(tool="list_quotes"),)
        first = domain.serve_retrieval(state, request, "2025-01-01T00:00:00Z")
        second = domain.serve_retrieval(state, request, "2025-01-02T00:00:00Z")
        assert first[0].record_version == second[0].record_version
        assert first[0].as_of != second[0].as_of

    def test_completing_is_still_a_business_outcome(self, spec) -> None:
        agent = ScriptedAgent([batch("get_case_record"), Complete(reason="no")])
        outcome, _ = run(spec, agent)
        assert rows(outcome, "terminal_proposed")
