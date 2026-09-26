"""Phase 2, part 1: the normalized projection leaves the agent path entirely.

Phase 1A added retrieval beside ``state``. This phase removes ``state`` — from
:class:`AgentObservation`, from the model-visible allowlist and from
``model_projection`` — and proves the removal recursively against the *actual*
requests V1, V2 and V3 produce, rather than against a description of them.

Two poisons carry the load. ``MaintenanceState.observable`` is the normalized
projection and ``EventQueue.pending_event_types`` is the authored future; both
are replaced with raisers for the whole of a reference run, and the run still
reaches its authored terminal. Nothing on the observation or planner path may
call either.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from operatebench.core.engine import Engine
from operatebench.core.events import EventQueue
from operatebench.core.protocol import (
    MODEL_VISIBLE_FIELDS,
    AgentObservation,
    model_projection,
)
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.domains.lettings.maintenance.state import MaintenanceState
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION
from tests.test_projection_budget_contract import assert_projection_budget

SPEC = "examples/operatebench/maintenance_v0_1.yaml"

#: The identity a provider-backed run carries. Not a registered agent: a model
#: run has no registry entry, and its kind comes from its execution source.
MODEL_AGENT_ID = "model_under_test"

#: Probe 1, the shipped reference at V1 on the pre-retrieval observation.
BASELINE_CALLS_V1 = 46

#: Probe 4 variant A, the variant the owner adjudicated. One agent batch per
#: business decision: 23 -> 46 at V1, honestly 2x the legacy 23-call episode.
TARGET_CALLS = {"V1": 46, "V2": 24, "V3": 28}

#: Collection names that only a normalized projection of the record carries. If
#: one of these is a key anywhere in a model request, the removal did not happen.
NORMALIZED_COLLECTIONS: tuple[str, ...] = (
    "approvals",
    "assertions",
    "authoritative_records",
    "communications",
    "cycles",
    "exceptions",
    "invoices",
    "obligations",
    "payment",
    "provisional_close",
    "quotes",
)


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


class RequestCensus:
    """Wraps an agent and keeps every model request it was handed."""

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
    def projection_bytes(self) -> int:
        return sum(
            len(json.dumps(projection, sort_keys=True, default=str))
            for projection in self.projections
        )


def _engine(spec, scenario_id: str, agent: Any) -> Engine:
    return Engine(
        MaintenanceOperation(spec, scenario_id),
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


def census_run(spec, scenario_id: str):
    from operatebench.domains.lettings.maintenance.agents import build_agent

    agent = RequestCensus(build_agent("reference"))
    return _engine(spec, scenario_id, agent).run(), agent


def _keys_and_values(value: Any) -> tuple[list[str], list[str]]:
    """Every mapping key and every string leaf anywhere inside a projection."""
    keys: list[str] = []
    values: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for name, item in node.items():
                keys.append(str(name))
                walk(item)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in node:
                walk(item)
            return
        if isinstance(node, str):
            values.append(node)

    walk(value)
    return keys, values


# -- the allowlist ----------------------------------------------------------


class TestTheAllowlistNoLongerCarriesState:
    def test_the_observation_has_no_state_field_at_all(self) -> None:
        assert "state" not in AgentObservation.__dataclass_fields__
        # Not renamed, not emptied, not summarised: no field on the observation
        # is the normalized record under another name.
        for name in AgentObservation.__dataclass_fields__:
            assert "state" not in name

    def test_the_model_visible_allowlist_has_no_state(self) -> None:
        assert "state" not in MODEL_VISIBLE_FIELDS
        # ``trigger`` went with it: the bare envelope read as a statement about
        # the world, and ``event`` says who produced it and whether their word
        # settles anything.
        assert "trigger" not in MODEL_VISIBLE_FIELDS
        assert "phase" in MODEL_VISIBLE_FIELDS
        assert "event" in MODEL_VISIBLE_FIELDS
        assert "retrieval" in MODEL_VISIBLE_FIELDS

    def test_a_projection_states_exactly_the_allowlist(self) -> None:
        observation = AgentObservation(
            now="2025-01-01T00:00:00Z",
            operation_id="op",
            operation_instance_id="opinst_x",
            invocation_index=1,
            turn_index=0,
            policy={},
            actors={},
            message_fixture_ids=[],
        )
        assert tuple(sorted(model_projection(observation))) == tuple(
            sorted(MODEL_VISIBLE_FIELDS)
        )
        assert "state" not in observation.as_dict()


# -- the actual requests ----------------------------------------------------


class TestNoNormalizedTruthReachesAnyAgent:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_no_request_carries_a_state_key_or_a_normalized_collection(
        self, spec, scenario_id
    ) -> None:
        _outcome, agent = census_run(spec, scenario_id)
        assert agent.projections
        for projection in agent.projections:
            keys, _values = _keys_and_values(
                {
                    name: value
                    for name, value in projection.items()
                    # ``retrieval.served`` is the *answer to a read the agent
                    # asked for*, which is the whole point; it is excluded here
                    # and held to its own contract elsewhere.
                    if name != "retrieval"
                }
            )
            assert "state" not in keys
            for collection in NORMALIZED_COLLECTIONS:
                assert collection not in keys, (scenario_id, collection)

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_no_request_names_the_scenario_or_its_label(self, spec, scenario_id) -> None:
        _outcome, agent = census_run(spec, scenario_id)
        forbidden = set(spec.scenario_ids)
        forbidden |= {spec.scenario(sid).label for sid in spec.scenario_ids}
        forbidden |= {spec.scenario(sid).expected_terminal for sid in spec.scenario_ids}
        for projection in agent.projections:
            keys, values = _keys_and_values(projection)
            assert not (set(values) & forbidden), scenario_id
            assert not (set(keys) & forbidden), scenario_id


class TestThePoisonedProjectionAndTheAuthoredFuture:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_runs_with_observable_and_the_queue_poisoned(
        self, spec, scenario_id, monkeypatch
    ) -> None:
        from operatebench.domains.lettings.maintenance.agents import build_agent

        def poisoned_observable(self: MaintenanceState) -> dict[str, Any]:
            raise AssertionError("the normalized projection was read on the agent path")

        def poisoned_pending(self: EventQueue) -> frozenset[str]:
            raise AssertionError("the authored future was read on the agent path")

        monkeypatch.setattr(MaintenanceState, "observable", poisoned_observable)
        monkeypatch.setattr(EventQueue, "pending_event_types", property(poisoned_pending))
        agent = build_agent("reference")
        outcome = _engine(spec, scenario_id, agent).run()
        assert outcome.status == spec.scenario(scenario_id).expected_terminal


# -- the budgets ------------------------------------------------------------


class TestTheAdjudicatedBudgets:
    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_reference_is_reliable_through_reads_alone(
        self, spec, scenario_id
    ) -> None:
        run = run_episode(spec, scenario_id, "reference")
        assert run.outcome.status == spec.scenario(scenario_id).expected_terminal
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
    def test_the_call_budget_holds(self, spec, scenario_id) -> None:
        _outcome, agent = census_run(spec, scenario_id)
        assert agent.calls <= TARGET_CALLS[scenario_id], (
            agent.calls,
            agent.batches,
            agent.business,
        )

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_planner_never_needs_a_top_up(self, spec, scenario_id) -> None:
        from operatebench.domains.lettings.maintenance.agents import build_agent

        agent = build_agent("reference")
        _engine(spec, scenario_id, agent).run()
        assert agent.topup_batches == 0, agent.topup_causes

    def test_the_reduced_v1_projection_is_under_the_ceiling(self, spec) -> None:
        _outcome, agent = census_run(spec, "V1")
        assert_projection_budget(
            agent.projections, engine_version=OPERATEBENCH_VERSION, scenario_id="V1"
        )

    def test_every_accepted_effect_is_followed_by_an_agent_batch(self, spec) -> None:
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
                raise AssertionError(f"business outcome on a stale context: {row}")


# -- the actual model requests ----------------------------------------------


class TestNoModelRequestCarriesTheRecord:
    """The strongest form of the claim: the bytes a provider would be sent.

    ``model_projection`` is what the request is *built from*, and the tests above
    hold it. This holds the request itself — the object a transport is handed,
    with its prompt, its observation and the digests both are bound under — for
    every required scenario, walked recursively. A field that reached a provider
    without passing the allowlist would show up here and nowhere else.
    """

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_no_request_built_for_a_provider_carries_normalized_truth(
        self, spec, scenario_id
    ) -> None:
        from tests.model_transport import ReferenceDrivenTransport

        transport = ReferenceDrivenTransport()
        run_episode(
            spec,
            scenario_id,
            MODEL_AGENT_ID,
            self_check=False,
            agent_factory=lambda: _model_agent(transport),
            agent_kind="model",
        )
        assert transport.requests
        for request in transport.requests:
            observation = request.prompt["observation"]
            assert tuple(sorted(observation)) == tuple(sorted(MODEL_VISIBLE_FIELDS))
            keys, _values = _keys_and_values(
                {
                    name: value
                    for name, value in observation.items()
                    if name != "retrieval"
                }
            )
            assert "state" not in keys
            assert "trigger" not in keys
            for collection in NORMALIZED_COLLECTIONS:
                assert collection not in keys, (scenario_id, collection)

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_a_model_run_completes_with_both_poisons_in_place(
        self, spec, scenario_id, monkeypatch
    ) -> None:
        """The same two poisons, on the path that actually reaches a provider."""
        from tests.model_transport import ReferenceDrivenTransport

        def poisoned_observable(self: MaintenanceState) -> dict[str, Any]:
            raise AssertionError("the normalized projection was read on the model path")

        def poisoned_pending(self: EventQueue) -> frozenset[str]:
            raise AssertionError("the authored future was read on the model path")

        monkeypatch.setattr(MaintenanceState, "observable", poisoned_observable)
        monkeypatch.setattr(EventQueue, "pending_event_types", property(poisoned_pending))
        transport = ReferenceDrivenTransport()
        run = run_episode(
            spec,
            scenario_id,
            MODEL_AGENT_ID,
            self_check=False,
            agent_factory=lambda: _model_agent(transport),
            agent_kind="model",
        )
        assert run.outcome.status == spec.scenario(scenario_id).expected_terminal

    @pytest.mark.parametrize("scenario_id", ["V1", "V2", "V3"])
    def test_the_served_records_are_the_only_route_to_a_collection(
        self, spec, scenario_id
    ) -> None:
        from tests.model_transport import ReferenceDrivenTransport

        transport = ReferenceDrivenTransport()
        run_episode(
            spec,
            scenario_id,
            MODEL_AGENT_ID,
            self_check=False,
            agent_factory=lambda: _model_agent(transport),
            agent_kind="model",
        )
        seen: set[str] = set()
        for request in transport.requests:
            served = request.prompt["observation"]["retrieval"]["served"]
            for body in served.values():
                seen.update(body["records"])
        # Every normalized collection an agent ever sees, it saw because it asked.
        assert seen, "no request carried a served record at all"
        assert seen & set(NORMALIZED_COLLECTIONS)


def _model_agent(transport):
    from operatebench.agents.model import ModelAgent
    from tests.model_transport import TEST_MODEL

    return ModelAgent(transport, model=TEST_MODEL, agent_id=MODEL_AGENT_ID)
