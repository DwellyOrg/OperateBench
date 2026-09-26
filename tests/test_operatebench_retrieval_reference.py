"""The reference runs the operation through batched reads, and only through them.

The migration claim is narrow and testable. The reference proposes a batch from
the coarse phase, the event type and the reads each action publishes — never
from a scenario identity, a scenario label or an authored future — uses what
comes back to make the same decisions it always made, and asks again after every
accepted own effect. The six shipped negatives keep their exact oracle closures.

The budget assertions are the ones the owner adjudicated: V1 costs no more agent
calls than the shipped baseline of 46, spends nothing on top-up batches, and
keeps the initial projection under the historical 1.25x of 295,404 B contract.
Engine 0.13.0 has a separate, fixed allowance for disclosed guidance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Ask, Complete, Escalate, Wait
from operatebench.core.protocol import AgentObservation, model_projection
from operatebench.core.read_contract import (
    READ_OPTIONAL,
    READ_REQUIRED,
    ActionEvidenceContract,
)
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance import agents as agents_module
from operatebench.domains.lettings.maintenance.agents import (
    AGENTS,
    NEGATIVE_AGENTS,
    REFERENCE_AGENT,
    AlwaysActAgent,
    MaintenanceRetrievalPlanner,
    ReferenceAgent,
    RetrievingReferenceAgent,
    _outcome_read_key,
    build_agent,
)
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import check_maintenance, run_episode
from operatebench.version import OPERATEBENCH_VERSION
from tests.test_projection_budget_contract import assert_projection_budget

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

#: Probe 1, the shipped reference at V1 on the pre-retrieval observation.
BASELINE_CALLS_V1 = 46

#: Probe 4, variant A (agent re-request), the variant the owner adjudicated.
TARGET_CALLS = {"V1": 46, "V2": 24, "V3": 28}


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


class Census:
    """Wraps an agent and counts what the episode cost it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.agent_id = inner.agent_id
        self.calls = 0
        self.batches = 0
        self.business = 0
        self.projections: list[Mapping[str, Any]] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self._inner.begin_episode(identity)

    def decide(self, observation: Any) -> Any:
        self.calls += 1
        self.projections.append(model_projection(observation))
        outcome = self._inner.decide(observation)
        if isinstance(outcome, RetrieveBatch):
            self.batches += 1
        else:
            self.business += 1
        return outcome

    @property
    def projection_field_bytes(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for projection in self.projections:
            for name, value in projection.items():
                totals[name] = totals.get(name, 0) + len(
                    json.dumps(value, sort_keys=True, default=str)
                )
        return totals

    @property
    def projection_bytes(self) -> int:
        return sum(self.projection_field_bytes.values())

    @property
    def projection_bytes_without_state(self) -> int:
        """The projection as Phase 2 publishes it. There is no ``state`` in it.

        Kept under its Phase 1A name so the number this file has always reported
        is the number it still reports; the subtraction is now a no-op, which is
        the point, and the assertion below states that rather than assuming it.
        """
        totals = self.projection_field_bytes
        assert "state" not in totals
        return sum(totals.values())


def census_run(spec, scenario_id: str):
    agent = Census(build_agent("reference"))
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
    return engine.run(), agent


def _read_gate_observation(
    reads: Mapping[str, str], served: Mapping[str, Any]
) -> AgentObservation:
    return AgentObservation(
        now="2025-01-01T09:00:00Z",
        operation_id="maintenance",
        operation_instance_id="opinst_test_0000000000000000",
        invocation_index=1,
        turn_index=0,
        policy={},
        actors={},
        message_fixture_ids=(),
        action_schemas={"request_supplier_visit": {"reads": dict(reads)}},
        phase="",
        retrieval={"served": dict(served)},
    )


def _visit() -> Act:
    return Act(
        action_type="request_supplier_visit",
        payload={
            "visit_type": "DIAGNOSTIC",
            "cycle_id": "work_cycle_1",
            "scope_digest": None,
        },
        rationale="test read gate",
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_visit(), "request_supplier_visit"),
        (
            Ask(
                recipient_actor_id="supplier_1",
                message_fixture_id="message_1",
                correlation_id="correlation_1",
                wait=Wait(reason="wait", fallback_after_minutes=1),
            ),
            "send_message",
        ),
        (
            Escalate(
                checkpoint_id="checkpoint_1",
                exception_type="APPROVAL_REJECTED",
                evidence_refs=(),
                deadline_after_minutes=1,
                rationale="escalate",
            ),
            "request_exception_resolution",
        ),
        (Complete(reason="done"), "complete"),
        (Wait(reason="wait", fallback_after_minutes=1), None),
    ],
    ids=("act", "ask", "escalate", "complete", "wait"),
)
def test_outcome_keys_name_the_exact_engine_rejection_subject(outcome, expected) -> None:
    assert _outcome_read_key(outcome) == expected


@pytest.mark.parametrize("agent_kind", ["reference", "negative"])
@pytest.mark.parametrize(
    ("reads", "served"),
    [
        ({"list_quotes": READ_OPTIONAL}, {}),
        (
            {
                "get_case_record": READ_REQUIRED,
                "list_quotes": READ_OPTIONAL,
            },
            {"get_case_record": {"ok": True}},
        ),
    ],
    ids=("optional-only-absent", "mixed-required-served"),
)
def test_retrieving_agents_do_not_gate_on_absent_optional_reads(
    agent_kind: str,
    reads: Mapping[str, str],
    served: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _read_gate_observation(reads, served)
    if agent_kind == "reference":
        agent = RetrievingReferenceAgent()
        monkeypatch.setattr(agent, "plan_retrieval", lambda _observation: None)
        monkeypatch.setattr(agent._inner, "decide_from", lambda _state, _policy: _visit())
    else:
        agent = AlwaysActAgent()
        monkeypatch.setattr(agent, "_planned", lambda _observation: None)

    outcome = agent.decide(observation)

    assert isinstance(outcome, Act)


@pytest.mark.parametrize("agent_kind", ["reference", "negative"])
def test_retrieving_agents_still_gate_on_absent_required_reads(
    agent_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _read_gate_observation(
        {
            "get_case_record": READ_REQUIRED,
            "list_quotes": READ_OPTIONAL,
        },
        {},
    )
    if agent_kind == "reference":
        agent = RetrievingReferenceAgent()
        monkeypatch.setattr(agent, "plan_retrieval", lambda _observation: None)
        monkeypatch.setattr(agent._inner, "decide_from", lambda _state, _policy: _visit())
    else:
        agent = AlwaysActAgent()
        monkeypatch.setattr(agent, "_planned", lambda _observation: None)

    outcome = agent.decide(observation)

    assert isinstance(outcome, RetrieveBatch)
    assert tuple(request.tool for request in outcome.requests) == ("get_case_record",)


@pytest.mark.parametrize(
    "agent",
    [RetrievingReferenceAgent(), AlwaysActAgent()],
    ids=("reference", "negative"),
)
def test_a_failed_required_read_waits_then_may_retry_next_invocation(agent) -> None:
    observation = replace(
        _read_gate_observation({"get_case_record": READ_REQUIRED}, {}),
        phase="FINALIZED",
    )
    agent.begin_episode({})

    first = agent.decide(observation)
    after_failed_result = agent.decide(replace(observation, turn_index=1))
    next_invocation = agent.decide(replace(observation, invocation_index=2, turn_index=0))

    assert isinstance(first, RetrieveBatch)
    assert not isinstance(after_failed_result, RetrieveBatch)
    assert isinstance(next_invocation, RetrieveBatch)


def test_an_accepted_effect_allows_the_planner_to_refresh_attempted_tools() -> None:
    planner = MaintenanceRetrievalPlanner()
    observation = replace(
        _read_gate_observation({}, {}),
        phase="FINALIZED",
    )

    first = planner.plan_retrieval(observation)
    served = planner.plan_retrieval(
        replace(
            observation,
            turn_index=1,
            retrieval={"served": {"get_case_record": {"ok": True}}},
        )
    )
    planner.decision_boundary("request_supplier_visit")
    after_effect = planner.plan_retrieval(replace(observation, turn_index=2))

    assert isinstance(first, RetrieveBatch)
    assert served is None
    assert isinstance(after_effect, RetrieveBatch)


def test_a_matching_rejection_does_not_refresh_attempted_tools() -> None:
    planner = MaintenanceRetrievalPlanner()
    observation = replace(_read_gate_observation({}, {}), phase="FINALIZED")

    assert isinstance(planner.plan_retrieval(observation), RetrieveBatch)
    planner.decision_boundary("request_supplier_visit")

    after_rejection = planner.plan_retrieval(
        replace(
            observation,
            turn_index=1,
            last_rejection={
                "action_type": "request_supplier_visit",
                "code": "DUPLICATE_VISIT",
                "detail": "already requested",
            },
        )
    )

    assert after_rejection is None


def test_an_unrelated_rejection_does_not_mask_an_accepted_effect() -> None:
    planner = MaintenanceRetrievalPlanner()
    observation = replace(_read_gate_observation({}, {}), phase="FINALIZED")

    assert isinstance(planner.plan_retrieval(observation), RetrieveBatch)
    planner.decision_boundary("request_supplier_visit")

    after_effect = planner.plan_retrieval(
        replace(
            observation,
            turn_index=1,
            last_rejection={
                "action_type": "wait",
                "code": "UNKNOWN_WAKE_TYPE",
                "detail": "earlier refusal",
            },
        )
    )

    assert isinstance(after_effect, RetrieveBatch)


def test_begin_episode_clears_attempted_tools_even_for_the_same_identity() -> None:
    planner = MaintenanceRetrievalPlanner()
    observation = replace(_read_gate_observation({}, {}), phase="FINALIZED")

    assert isinstance(planner.plan_retrieval(observation), RetrieveBatch)
    assert planner.plan_retrieval(replace(observation, turn_index=1)) is None

    planner.begin_episode()

    assert isinstance(planner.plan_retrieval(observation), RetrieveBatch)


def _all_failed_engine_run(spec, agent, monkeypatch, *, required: bool = False):
    tool = "get_case_record" if required else "list_quotes"
    strength = READ_REQUIRED if required else READ_OPTIONAL
    monkeypatch.setattr(
        agents_module,
        "PHASE_READS",
        dict.fromkeys(agents_module.PHASE_READS, (tool,)),
    )
    domain = MaintenanceOperation(spec, "V1")
    original_serve = domain.serve_retrieval
    original_schemas = domain.action_schemas

    def fail_every_result(state, requests, as_of):
        return tuple(
            replace(result, records={}, ok=False, error_code="READ_UNAVAILABLE")
            for result in original_serve(state, requests, as_of)
        )

    def visit_schema_with_only_test_read():
        schemas = dict(original_schemas())
        visit = dict(schemas["request_supplier_visit"])
        visit["reads"] = {tool: strength}
        schemas["request_supplier_visit"] = visit
        return schemas

    monkeypatch.setattr(domain, "serve_retrieval", fail_every_result)
    monkeypatch.setattr(domain, "action_schemas", visit_schema_with_only_test_read)
    monkeypatch.setattr(
        domain,
        "read_requirements",
        lambda: ActionEvidenceContract(
            {"request_supplier_visit": {"reads": {tool: strength}}}
        ),
    )
    return Engine(
        domain,
        agent,
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opinst_test_0000000000000000",
            "scenario_id": "V1",
            "agent_id": agent.agent_id,
            "spec_digest_sha256": spec.spec_digest_sha256,
        },
        dispatch_failures=spec.scenario("V1").dispatch_failures,
    ).run()


def _next_substantive_row(trajectory, start):
    substantive = {
        "retrieval_served",
        "agent_outcome",
        "action_rejected",
        "derived_context_invalidated",
    }
    return next(
        row for row in trajectory[start + 1 :] if row["record_type"] in substantive
    )


def test_empty_to_empty_accepted_effect_starts_a_fresh_decision_context(
    spec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real failed optional result leaves Engine._served empty on both sides."""
    outcome = _all_failed_engine_run(spec, AlwaysActAgent(), monkeypatch)
    rows = list(outcome.trajectory)
    invalidation_index = next(
        index
        for index, row in enumerate(rows)
        if row["record_type"] == "derived_context_invalidated"
        and row["cause"] == "effect_accepted"
        and row["cleared_tools"] == []
    )

    next_business = next(
        index
        for index, row in enumerate(
            rows[invalidation_index + 1 :], invalidation_index + 1
        )
        if row["record_type"] == "agent_outcome"
    )
    refreshed_optional = [
        row
        for row in rows[invalidation_index + 1 : next_business]
        if row["record_type"] == "retrieval_served" and row["tool"] == "list_quotes"
    ]

    assert len(refreshed_optional) == 1
    assert refreshed_optional[0]["ok"] is False


def test_empty_served_rejected_action_does_not_retry_the_failed_optional_read(
    spec, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _all_failed_engine_run(spec, AlwaysActAgent(), monkeypatch)
    rows = list(outcome.trajectory)
    rejection_index = next(
        index for index, row in enumerate(rows) if row["record_type"] == "action_rejected"
    )

    next_row = _next_substantive_row(rows, rejection_index)
    failed_after_rejection = [
        row
        for row in rows[rejection_index + 1 :]
        if row["record_type"] == "retrieval_served"
        and row["tool"] == "list_quotes"
        and row["ok"] is False
    ]

    assert next_row["record_type"] == "agent_outcome"
    assert failed_after_rejection == []
    assert any(
        row["record_type"] == "agent_outcome" for row in rows[rejection_index + 1 :]
    )
    assert not any(row.get("code") == "RETRIEVAL_BUDGET_EXHAUSTED" for row in rows)


def test_a_real_failed_required_read_waits_boundedly_and_retries_next_invocation(
    spec, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _all_failed_engine_run(spec, AlwaysActAgent(), monkeypatch, required=True)
    failed = [
        row
        for row in outcome.trajectory
        if row["record_type"] == "retrieval_served"
        and row["tool"] == "get_case_record"
        and row["ok"] is False
    ]
    attempts: dict[int, int] = {}
    for row in failed:
        invocation = int(row["invocation_index"])
        attempts[invocation] = attempts.get(invocation, 0) + 1

    assert len(attempts) >= 2
    assert set(attempts.values()) == {1}
    assert any(
        row["record_type"] == "agent_outcome" and row["outcome"]["kind"] == "WAIT"
        for row in outcome.trajectory
    )
    assert not any(
        row.get("code") == "RETRIEVAL_BUDGET_EXHAUSTED" for row in outcome.trajectory
    )


# -- the reference ----------------------------------------------------------


class TestTheReferenceRunsThroughReads:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_authored_terminal_is_reached_reliably(self, spec, scenario_id) -> None:
        run = run_episode(spec, scenario_id, "reference")
        assert run.outcome.status == spec.scenario(scenario_id).expected_terminal
        assert run.reliable is (scenario_id != "V2")
        assert run.evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_it_costs_no_more_calls_than_the_shipped_baseline(
        self, spec, scenario_id
    ) -> None:
        _outcome, agent = census_run(spec, scenario_id)
        assert agent.calls <= TARGET_CALLS[scenario_id], (
            scenario_id,
            agent.calls,
            agent.batches,
            agent.business,
        )

    def test_v1_without_the_retained_state_is_under_the_adjudicated_ceiling(
        self, spec
    ) -> None:
        _outcome, agent = census_run(spec, "V1")
        assert agent.calls <= BASELINE_CALLS_V1
        assert agent.projection_bytes_without_state == agent.projection_bytes
        assert_projection_budget(
            agent.projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )

    def test_the_duplication_phase_1a_measured_is_gone(self, spec) -> None:
        """The number that decided whether Phase 2 was worth shipping.

        Phase 1A published ``state`` *and* served the same collections back
        through the catalogue, so the two overlapped by construction and the
        total sat over the adjudicated ceiling by exactly the retained
        projection. Phase 2 publishes the collections once, through reads only,
        and engine 0.13.0 separately budgets the incremental guidance.
        """
        _outcome, agent = census_run(spec, "V1")
        totals = agent.projection_field_bytes
        assert "state" not in totals
        assert "trigger" not in totals
        assert agent.projection_bytes == agent.projection_bytes_without_state
        assert_projection_budget(
            agent.projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )

    def test_the_call_cost_is_two_per_business_decision_and_says_so(self, spec) -> None:
        """What 46 calls actually count, stated rather than inherited.

        The pre-retrieval reference ran one V1 episode in 23 agent calls; the
        adjudicated baseline of 46 is that number doubled, because
        :func:`run_episode` executes the episode twice — once to run it and once
        to check it is deterministic. This build's 46 calls are *one* episode.

        So the honest statement is 2.0x, not 1.000x: one batch for every business
        decision, because every accepted own effect empties the served set and
        the agent — not the environment — chooses to read again. There is no
        parity claim here. The legacy agent is gone (its decision rules survive
        as :class:`ReferenceAgent`, which is no longer an agent at all), so 23 is
        quoted from Phase 1A rather than re-measured against a path this build no
        longer has.
        """
        assert not hasattr(ReferenceAgent, "decide")
        legacy_business_turns = BASELINE_CALLS_V1 // 2
        _outcome, retrieving = census_run(spec, "V1")
        assert retrieving.business == legacy_business_turns
        assert retrieving.batches == legacy_business_turns
        assert retrieving.calls == 2 * legacy_business_turns == BASELINE_CALLS_V1

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_planner_never_needs_a_top_up(self, spec, scenario_id) -> None:
        agent = build_agent("reference")
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
        engine.run()
        assert agent.topup_batches == 0, agent.topup_causes

    def test_every_accepted_effect_is_followed_by_an_agent_requested_batch(
        self, spec
    ) -> None:
        outcome, _agent = census_run(spec, "V1")
        pending = False
        for row in outcome.trajectory:
            kind = row["record_type"]
            if kind == "derived_context_invalidated":
                pending = True
            elif kind == "retrieval_served":
                assert row["initiated_by"] == "agent"
                pending = False
            elif kind == "agent_outcome" and pending:
                raise AssertionError(
                    "a business outcome followed an invalidation with no fresh "
                    f"agent-requested batch: {row}"
                )

    def test_the_planner_reads_nothing_case_specific(self, spec) -> None:
        agent = build_agent("reference")
        planner_table = json.dumps(agent.phase_read_table(), sort_keys=True)
        # Nothing the planner is given names a case: it plans from the coarse
        # phase, the event type and the published reads alone.
        assert hasattr(agent, "plan_retrieval")
        for scenario_id in spec.scenario_ids:
            assert scenario_id not in planner_table
            assert spec.scenario(scenario_id).label not in planner_table

    def test_the_shipped_reference_entry_is_the_retrieving_one(self) -> None:
        assert REFERENCE_AGENT.agent_id == "reference"
        built = REFERENCE_AGENT.factory()
        assert built.agent_id == "reference"
        assert not isinstance(built, ReferenceAgent)
        assert hasattr(built, "plan_retrieval")


# -- the negatives ----------------------------------------------------------


class TestTheNegativesKeepTheirOracleClosures:
    def test_the_registry_holds_exactly_nine_agents(self) -> None:
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

    def test_check_maintenance_is_green_on_eleven_checks(self, spec) -> None:
        report = check_maintenance(spec)
        assert len(report.checks) == 11
        assert report.ok, [check.as_dict() for check in report.checks if not check.ok]

    @pytest.mark.parametrize("entry", NEGATIVE_AGENTS, ids=lambda e: e.agent_id)
    def test_each_negative_keeps_its_exact_oracle_closure(self, spec, entry) -> None:
        scenario_id = entry.scenarios[0]
        run = run_episode(spec, scenario_id, entry.agent_id)
        assert set(run.evaluation.failed_dimensions) == set(
            entry.expected_failure_closure
        ), entry.agent_id
        assert set(entry.expected_findings) <= set(run.evaluation.finding_codes)

    def test_no_two_shipped_agents_share_a_signature(self, spec) -> None:
        signatures = set()
        for entry in NEGATIVE_AGENTS:
            signature = (
                tuple(sorted(entry.expected_failure_closure)),
                tuple(sorted(entry.expected_findings)),
            )
            assert signature not in signatures, entry.agent_id
            signatures.add(signature)
