"""Phase 2, part 2: one canonical read contract, and a guard that executes it.

Phase 1A stated each action's required reads in a hand-maintained table beside
the payload schemas, and published a *copy* of it on the observation. Two copies
drift; three — runtime, observation, evaluator — drift faster. This phase moves
the requirement into one validated contract object and makes every consumer read
that object.

The tests that matter are the perturbation ones: mutate one requirement in the
canonical contract and both the runtime refusal *and* the evaluator's verdict
move with it. That is what makes the contract policy authority rather than a
comment, and it is what stops the evaluator from being a restatement of the
guard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act, Complete, Wait
from operatebench.core.read_contract import (
    ACTED_ON_CLAIM_WITHOUT_RECORD,
    ACTION_WITHOUT_RETRIEVAL,
    COMPLETE_OUTCOME_KEY,
    READ_GUARD_CODES,
    READ_OPTIONAL,
    READ_REQUIRED,
    STALE_RETRIEVED_RECORD,
    ReadRequirementContract,
    read_contract_problem,
)
from operatebench.core.retrieval import RetrievalRequest, RetrieveBatch
from operatebench.domains.lettings.maintenance import spec as spec_module
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.retrieval import (
    maintenance_retrieval_catalogue,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    MAINTENANCE_READ_CONTRACT,
    action_schema_view,
    load_spec,
)

SPEC = "examples/operatebench/maintenance_v0_1.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


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


class ScriptedAgent:
    """Returns a scripted sequence of objects, then waits forever."""

    agent_id = "scripted"

    def __init__(self, script) -> None:
        self._script = list(script)
        self.observations: list[Any] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: Any) -> Any:
        self.observations.append(observation)
        if self._script:
            step = self._script.pop(0)
            return step(observation) if callable(step) else step
        return Wait(reason="nothing left in the script", fallback_after_minutes=1440)


def _rows(trajectory, *types):
    return [row for row in trajectory if row["record_type"] in types]


def _mixed_strength_contract() -> ReadRequirementContract:
    return ReadRequirementContract(
        {
            "request_supplier_visit": {
                "reads": {
                    "get_case_record": READ_REQUIRED,
                    "list_quotes": READ_OPTIONAL,
                }
            }
        }
    )


# -- the contract itself ----------------------------------------------------


class TestTheCanonicalContract:
    def test_it_is_one_validated_object(self) -> None:
        assert isinstance(MAINTENANCE_READ_CONTRACT, ReadRequirementContract)
        assert (
            read_contract_problem(
                MAINTENANCE_READ_CONTRACT,
                catalogue=maintenance_retrieval_catalogue(),
                outcome_keys=(
                    *MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
                    COMPLETE_OUTCOME_KEY,
                ),
            )
            is None
        )

    def test_the_old_hand_maintained_table_is_gone(self) -> None:
        assert not hasattr(spec_module, "MAINTENANCE_ACTION_READ_REQUIREMENTS")

    def test_the_action_schemas_publish_the_contract_itself(self) -> None:
        view = action_schema_view()
        for action_type, schema in view.items():
            assert schema["reads"] == MAINTENANCE_READ_CONTRACT.schema_part(action_type)

    def test_the_terminal_declares_its_reads(self) -> None:
        # COMPLETE is a business outcome whose correctness depends on the record:
        # open checkpoints, open obligations, billing and the case itself.
        tools = MAINTENANCE_READ_CONTRACT.tools_for(COMPLETE_OUTCOME_KEY)
        assert set(tools) == {
            "get_case_record",
            "list_billing",
            "list_checkpoints",
            "list_obligations",
        }

    def test_wait_requires_nothing(self) -> None:
        assert MAINTENANCE_READ_CONTRACT.tools_for("wait") == ()

    def test_required_tools_exclude_optional_read_metadata(self) -> None:
        contract = _mixed_strength_contract()

        assert contract.schema_part("request_supplier_visit") == {
            "get_case_record": READ_REQUIRED,
            "list_quotes": READ_OPTIONAL,
        }
        assert contract.required_tools_for("request_supplier_visit") == (
            "get_case_record",
        )
        assert contract.tools_for("request_supplier_visit") == (
            "get_case_record",
            "list_quotes",
        )

    def test_an_unknown_read_strength_is_refused_by_canonical_validation(self) -> None:
        broken = ReadRequirementContract(
            {"request_supplier_visit": {"get_case_record": "best_effort"}}
        )

        problem = read_contract_problem(
            broken,
            catalogue=maintenance_retrieval_catalogue(),
            outcome_keys=("request_supplier_visit",),
        )

        assert problem is not None
        assert "strength" in problem

    def test_every_requirement_names_a_published_tool(self) -> None:
        catalogue = maintenance_retrieval_catalogue()
        for key in MAINTENANCE_READ_CONTRACT.outcome_keys():
            for tool in MAINTENANCE_READ_CONTRACT.tools_for(key):
                assert tool in catalogue, (key, tool)

    def test_a_requirement_outside_the_catalogue_is_refused(self) -> None:
        broken = ReadRequirementContract({"request_payment": {"list_moons": "required"}})
        problem = read_contract_problem(
            broken,
            catalogue=maintenance_retrieval_catalogue(),
            outcome_keys=("request_payment",),
        )
        assert problem is not None and "list_moons" in problem

    def test_a_requirement_for_an_unknown_outcome_is_refused(self) -> None:
        broken = ReadRequirementContract(
            {"do_a_barrel_roll": {"list_quotes": "required"}}
        )
        problem = read_contract_problem(
            broken,
            catalogue=maintenance_retrieval_catalogue(),
            outcome_keys=("request_payment",),
        )
        assert problem is not None and "do_a_barrel_roll" in problem

    def test_the_domain_serves_the_same_object_to_the_runtime(self, spec) -> None:
        domain = MaintenanceOperation(spec, "V1")
        assert domain.read_requirements() is MAINTENANCE_READ_CONTRACT


# -- the runtime guard ------------------------------------------------------


class TestTheGuardRefusesByName:
    def test_an_act_with_no_read_at_all_is_refused_and_mutates_nothing(
        self, spec
    ) -> None:
        agent = ScriptedAgent(
            [
                Act(
                    action_type="request_supplier_visit",
                    payload={
                        "visit_type": "DIAGNOSTIC",
                        "cycle_id": "work_cycle_1",
                        "scope_digest": None,
                    },
                    rationale="no read first",
                )
            ]
        )
        outcome = _engine(spec, "V1", agent).run()
        rejected = _rows(outcome.trajectory, "action_rejected")
        assert rejected, "the guard refused nothing"
        assert rejected[0]["code"] == ACTION_WITHOUT_RETRIEVAL
        # Non-mutating: the refusal produced no effect of any kind.
        assert not _rows(outcome.trajectory, "effect_accepted")

    def test_a_complete_with_no_read_is_refused_by_the_same_name(self, spec) -> None:
        agent = ScriptedAgent([Complete(reason="done, surely")])
        outcome = _engine(spec, "V1", agent).run()
        rejected = _rows(outcome.trajectory, "terminal_rejected")
        assert rejected and rejected[0]["code"] == ACTION_WITHOUT_RETRIEVAL

    def test_a_read_then_an_act_is_accepted(self, spec) -> None:
        tools = MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")
        agent = ScriptedAgent(
            [
                RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in tools)),
                Act(
                    action_type="request_supplier_visit",
                    payload={
                        "visit_type": "DIAGNOSTIC",
                        "cycle_id": "work_cycle_1",
                        "scope_digest": None,
                    },
                    rationale="read first",
                ),
            ]
        )
        outcome = _engine(spec, "V1", agent).run()
        assert _rows(outcome.trajectory, "effect_accepted")

    def test_an_optional_read_does_not_block_a_runtime_proposal(
        self, spec, monkeypatch
    ) -> None:
        contract = _mixed_strength_contract()
        monkeypatch.setattr(
            MaintenanceOperation,
            "read_requirements",
            lambda self: contract,
        )
        agent = ScriptedAgent(
            [
                RetrieveBatch((RetrievalRequest(tool="get_case_record"),)),
                Act(
                    action_type="request_supplier_visit",
                    payload={
                        "visit_type": "DIAGNOSTIC",
                        "cycle_id": "work_cycle_1",
                        "scope_digest": None,
                    },
                    rationale="the required read was served",
                ),
            ]
        )

        outcome = _engine(spec, "V1", agent).run()

        assert _rows(outcome.trajectory, "effect_accepted")
        assert not _rows(outcome.trajectory, "action_rejected")

    def test_a_required_read_still_blocks_beside_optional_metadata(
        self, spec, monkeypatch
    ) -> None:
        contract = _mixed_strength_contract()
        monkeypatch.setattr(
            MaintenanceOperation,
            "read_requirements",
            lambda self: contract,
        )
        agent = ScriptedAgent(
            [
                Act(
                    action_type="request_supplier_visit",
                    payload={
                        "visit_type": "DIAGNOSTIC",
                        "cycle_id": "work_cycle_1",
                        "scope_digest": None,
                    },
                    rationale="the required read is absent",
                )
            ]
        )

        outcome = _engine(spec, "V1", agent).run()

        rejected = _rows(outcome.trajectory, "action_rejected")
        assert [row["code"] for row in rejected] == [ACTION_WITHOUT_RETRIEVAL]
        assert not _rows(outcome.trajectory, "effect_accepted")

    def test_one_omitted_published_read_is_refused_and_the_exact_batch_is_not(
        self, spec
    ) -> None:
        tools = list(
            MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")
        )
        assert tools
        partial = tools[:-1]
        act = Act(
            action_type="request_supplier_visit",
            payload={
                "visit_type": "DIAGNOSTIC",
                "cycle_id": "work_cycle_1",
                "scope_digest": None,
            },
            rationale="one read short",
        )
        script = []
        if partial:
            script.append(RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in partial)))
        script.append(act)
        outcome = _engine(spec, "V1", ScriptedAgent(script)).run()
        assert [row["code"] for row in _rows(outcome.trajectory, "action_rejected")] == [
            ACTION_WITHOUT_RETRIEVAL
        ]

    def test_a_read_invalidated_by_the_agents_own_effect_is_stale_not_absent(
        self, spec
    ) -> None:
        tools = MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")
        act = Act(
            action_type="request_supplier_visit",
            payload={
                "visit_type": "DIAGNOSTIC",
                "cycle_id": "work_cycle_1",
                "scope_digest": None,
            },
            rationale="twice, without re-reading",
        )
        agent = ScriptedAgent(
            [
                RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in tools)),
                act,
                act,
            ]
        )
        outcome = _engine(spec, "V1", agent).run()
        codes = [row["code"] for row in _rows(outcome.trajectory, "action_rejected")]
        # The first act is accepted and clears the served set; the second one is
        # *stale*, not absent — the guard has to tell "read and moved" from
        # "never read".
        assert codes == [STALE_RETRIEVED_RECORD]
        assert len(_rows(outcome.trajectory, "effect_accepted")) == 1

    def test_a_version_that_moved_between_the_read_and_the_proposal_is_stale(
        self, spec
    ) -> None:
        """The guard compares the exact recorded version against the live one."""
        domain = MaintenanceOperation(spec, "V1")
        tools = MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")

        def mutate_then_act(observation: Any) -> Any:
            # Move the record underneath the served read without an invalidation.
            engine.state.reopen_count += 1
            return Act(
                action_type="request_supplier_visit",
                payload={
                    "visit_type": "DIAGNOSTIC",
                    "cycle_id": "work_cycle_1",
                    "scope_digest": None,
                },
                rationale="the record moved under me",
            )

        agent = ScriptedAgent(
            [
                RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in tools)),
                mutate_then_act,
            ]
        )
        engine = Engine(
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
        )
        outcome = engine.run()
        codes = [row["code"] for row in _rows(outcome.trajectory, "action_rejected")]
        assert STALE_RETRIEVED_RECORD in codes

    def test_the_guard_never_inspects_the_queue(self, spec, monkeypatch) -> None:
        from operatebench.core.events import EventQueue

        def poisoned(self: EventQueue) -> frozenset[str]:
            raise AssertionError("the guard read the authored future")

        monkeypatch.setattr(EventQueue, "pending_event_types", property(poisoned))
        agent = ScriptedAgent(
            [
                Act(
                    action_type="request_supplier_visit",
                    payload={"visit_type": "DIAGNOSTIC", "cycle_id": "work_cycle_1"},
                    rationale="no read first",
                )
            ]
        )
        _engine(spec, "V1", agent).run()

    def test_every_guard_code_is_named_once(self) -> None:
        assert set(READ_GUARD_CODES) == {
            ACTED_ON_CLAIM_WITHOUT_RECORD,
            ACTION_WITHOUT_RETRIEVAL,
            STALE_RETRIEVED_RECORD,
        }


# -- perturbation: the contract is the authority ----------------------------


class TestPerturbingOneRequirementMovesBothConsumers:
    def test_removing_a_requirement_makes_the_runtime_accept_what_it_refused(
        self, spec, monkeypatch
    ) -> None:
        tools = list(
            MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")
        )
        partial = tools[:-1]
        act = Act(
            action_type="request_supplier_visit",
            payload={
                "visit_type": "DIAGNOSTIC",
                "cycle_id": "work_cycle_1",
                "scope_digest": None,
            },
            rationale="one read short",
        )
        script = []
        if partial:
            script.append(RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in partial)))
        script.append(act)

        # Refused under the shipped contract...
        refused = _engine(spec, "V1", ScriptedAgent(list(script))).run()
        assert _rows(refused.trajectory, "action_rejected")

        # ...and accepted once the contract stops requiring the omitted read.
        loosened = ReadRequirementContract(
            {
                key: (
                    dict.fromkeys(partial, "required")
                    if key == "request_supplier_visit"
                    else dict(MAINTENANCE_READ_CONTRACT.requirements[key])
                )
                for key in MAINTENANCE_READ_CONTRACT.outcome_keys()
            }
        )
        monkeypatch.setattr(
            MaintenanceOperation, "read_requirements", lambda self: loosened
        )
        allowed = _engine(spec, "V1", ScriptedAgent(list(script))).run()
        assert _rows(allowed.trajectory, "effect_accepted")

    def test_adding_a_requirement_makes_the_runtime_refuse_what_it_accepted(
        self, spec, monkeypatch
    ) -> None:
        tools = MAINTENANCE_READ_CONTRACT.required_tools_for("request_supplier_visit")
        script = [
            RetrieveBatch(tuple(RetrievalRequest(tool=t) for t in tools)),
            Act(
                action_type="request_supplier_visit",
                payload={
                    "visit_type": "DIAGNOSTIC",
                    "cycle_id": "work_cycle_1",
                    "scope_digest": None,
                },
                rationale="read exactly what the contract published",
            ),
        ]
        accepted = _engine(spec, "V1", ScriptedAgent(list(script))).run()
        assert _rows(accepted.trajectory, "effect_accepted")

        tightened = ReadRequirementContract(
            {
                key: (
                    {
                        **MAINTENANCE_READ_CONTRACT.requirements[key],
                        "list_quotes": "required",
                    }
                    if key == "request_supplier_visit"
                    else dict(MAINTENANCE_READ_CONTRACT.requirements[key])
                )
                for key in MAINTENANCE_READ_CONTRACT.outcome_keys()
            }
        )
        monkeypatch.setattr(
            MaintenanceOperation, "read_requirements", lambda self: tightened
        )
        refused = _engine(spec, "V1", ScriptedAgent(list(script))).run()
        codes = [row["code"] for row in _rows(refused.trajectory, "action_rejected")]
        assert ACTION_WITHOUT_RETRIEVAL in codes
