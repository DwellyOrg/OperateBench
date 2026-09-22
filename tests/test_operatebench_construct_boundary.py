"""Two construct blockers, asserted vertically: answer identity, and causal WAIT.

Both are properties of the *seam* rather than of any one module, so they are
asserted here end to end — through the real engine, the real shipped scenarios
and the real model request path — rather than against a hand-built projection
that could be correct while the path a run actually takes is not.

**A benchmark that names the answer at the model boundary is measuring
recognition.** ``scenario_id`` is the index into the authored expectation: the
scenario's semantic id, its expected terminal, its oracle control and its
authored event list are all one lookup away from it, and an agent that sees it
can be tuned per case without ever running the operation. So the observation an
agent is handed carries an :data:`operation instance identity` instead — opaque,
random, generated once per run — and every internal record that has to know
which case ran keeps saying so, in the artefact, the evaluator and the run
identity, where a grader reads them and an agent does not.

**A WAIT that is refused because nothing pending can deliver it is graded on
authored-future truth.** The pending queue is the environment's private
knowledge of what the scenario author decided would happen; consulting it at
declaration time makes the benchmark answer "is this event coming?" — a question
no operator can ask in a real operation — and turns a correct, well-formed wait
into a violation because of a fact the agent could not have known. What is
public is the operation's wake vocabulary, and that is all this build checks.
Time then decides: a scheduled event, the declared fallback, the operational
horizon, or deadlock.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import ModelAgent
from operatebench.agents.transport import observation_digest
from operatebench.artifact import build_artifact
from operatebench.core.engine import STATUS_HORIZON, Engine
from operatebench.core.events import EventQueue
from operatebench.core.outcomes import Act, AgentOutcome, Wait
from operatebench.core.protocol import (
    AgentObservation,
    EpisodePlan,
    PlannedEvent,
    Verdict,
)
from operatebench.core.read_contract import ReadRequirementContract
from operatebench.core.retrieval import (
    AUTHORITY_ACTOR_CLAIM,
    RetrievalRequest,
    ToolResult,
)
from operatebench.domains.lettings.maintenance.agents import AGENTS, build_agent
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.runner import check_agent, run_episode
from tests.model_transport import TEST_MODEL, ReferenceDrivenTransport

FIXTURE = (
    Path(__file__).resolve().parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
)

#: Every scenario this build ships. The leakage properties below are asserted on
#: all of them, because a projection that is clean on one case and names the
#: answer on another is not a boundary.
SCENARIOS: tuple[str, ...] = ("V1", "V2", "V3")

#: Field names that are benchmark answer identity. A model-visible projection
#: carrying any of them tells the agent which authored case it is running.
ANSWER_KEYS: frozenset[str] = frozenset(
    {
        "scenario_id",
        "semantic_scenario_id",
        "expected_terminal",
        "scenario_label",
        "expected_event_rejections",
        "oracle_id",
        "oracle_version",
        "oracle_digest_sha256",
        "expected_failed_dimensions",
        "required_finding_codes",
        "must_pass_dimensions",
    }
)


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every key and every scalar in a JSON-shaped structure, with its path."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            here = f"{path}.{key}"
            yield (here, str(key))
            yield from _leaves(nested, here)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
        return
    yield (path, value)


def answer_identity_problems(
    payload: Any, spec: OperationSpec, scenario_id: str, where: str
) -> list[str]:
    """Every place ``payload`` names the authored answer, recursively."""
    scenario = spec.scenario(scenario_id)
    forbidden_values = {
        scenario_id,
        scenario.semantic_scenario_id,
        spec.semantic_scenario_id,
        scenario.expected_terminal,
        scenario.label,
        *SCENARIOS,
    }
    problems: list[str] = []
    if isinstance(payload, Mapping):
        for key in payload:
            if str(key) in ANSWER_KEYS:
                problems.append(f"{where}: carries the answer-identity key {key!r}")
    for path, leaf in _leaves(payload, where):
        if isinstance(leaf, str) and leaf in ANSWER_KEYS:
            problems.append(f"{path}: names the answer-identity key {leaf!r}")
        if isinstance(leaf, str) and leaf in forbidden_values:
            problems.append(f"{path}: carries the answer identity {leaf!r}")
    return problems


class CapturingAgent:
    """A transparent wrapper that keeps every observation it was handed."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.agent_id = str(getattr(inner, "agent_id", "agent"))
        self.observations: list[AgentObservation] = []
        self.identities: list[Mapping[str, Any]] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.identities.append(dict(identity))
        self._inner.begin_episode(identity)

    def decide(self, observation: AgentObservation) -> Any:
        self.observations.append(observation)
        return self._inner.decide(observation)


# ------------------------------------------------ P0-A: no answer at the boundary


class TestTheObservationNamesNoAnswer:
    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_no_observation_in_a_shipped_scenario_carries_answer_identity(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        captured = CapturingAgent(build_agent("reference"))
        run_episode(
            spec,
            scenario_id,
            "reference",
            self_check=False,
            agent_factory=lambda: captured,
        )
        assert captured.observations, "the episode never invoked the agent"
        problems: list[str] = []
        for index, observation in enumerate(captured.observations):
            problems.extend(
                answer_identity_problems(
                    observation.as_dict(), spec, scenario_id, f"observation[{index}]"
                )
            )
        assert problems == []

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_the_episode_identity_an_agent_is_handed_carries_no_answer(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        captured = CapturingAgent(build_agent("reference"))
        run_episode(
            spec,
            scenario_id,
            "reference",
            self_check=False,
            agent_factory=lambda: captured,
        )
        assert captured.identities, "the episode never opened an episode"
        problems: list[str] = []
        for index, identity in enumerate(captured.identities):
            problems.extend(
                answer_identity_problems(
                    identity, spec, scenario_id, f"identity[{index}]"
                )
            )
        assert problems == []

    def test_the_projection_is_an_allowlist_and_states_it(self) -> None:
        from operatebench.core.protocol import MODEL_VISIBLE_FIELDS, model_projection

        observation = AgentObservation(
            now="2025-01-01T09:00:00Z",
            operation_id="op",
            operation_instance_id="opinst_0123456789abcdef0123456789abcdef",
            invocation_index=1,
            turn_index=0,
            policy={},
            actors={},
            message_fixture_ids=(),
        )
        assert set(model_projection(observation)) == set(MODEL_VISIBLE_FIELDS)
        assert not ANSWER_KEYS & set(MODEL_VISIBLE_FIELDS)
        # The serialised observation and the model projection are the same field
        # set: an internal field added to one and not the other is the drift this
        # asserts against.
        assert set(observation.as_dict()) == set(MODEL_VISIBLE_FIELDS)


class TestTheModelRequestNamesNoAnswer:
    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_no_request_the_model_agent_builds_carries_answer_identity(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        transport = ReferenceDrivenTransport()
        run_episode(
            spec,
            scenario_id,
            "model_reference",
            self_check=False,
            agent_factory=lambda: ModelAgent(
                transport, model=TEST_MODEL, agent_id="model_reference"
            ),
            agent_kind="model",
        )
        assert transport.requests, "the episode built no provider request"
        problems: list[str] = []
        for index, request in enumerate(transport.requests):
            # Both the material a provider would be sent and the identity the
            # prompt digest is taken over. A prompt that is clean while the
            # digest source is not would still bind the run to the answer.
            for name, payload in (
                ("prompt", json.loads(json.dumps(request.prompt))),
                ("identity", request.identity()),
            ):
                problems.extend(
                    answer_identity_problems(
                        payload, spec, scenario_id, f"request[{index}].{name}"
                    )
                )
        assert problems == []

    def test_the_request_is_built_from_the_allowlist_not_from_as_dict(self) -> None:
        """A future internal field must not reach the provider through ``as_dict``."""

        class LeakyObservation:
            """An observation whose serialisation carries more than the projection."""

            now = "2025-01-01T09:00:00Z"
            operation_id = "op"
            operation_instance_id = "opinst_0123456789abcdef0123456789abcdef"
            invocation_index = 1
            turn_index = 0
            trigger = None
            state: Mapping[str, Any] = {}
            policy: Mapping[str, Any] = {}
            actors: Mapping[str, str] = {}
            message_fixture_ids: Sequence[str] = ()
            action_schemas: Mapping[str, Mapping[str, Mapping[str, str]]] = {}
            wake_event_types: Sequence[str] = ()
            last_rejection = None
            phase = ""
            event = None
            retrieval: Mapping[str, Any] = {}

            def as_dict(self) -> dict[str, Any]:
                return {
                    "scenario_id": "V2",
                    "expected_terminal": "completed_successfully",
                }

        agent = ModelAgent(ReferenceDrivenTransport(), model=TEST_MODEL)
        request = agent.build_request(LeakyObservation())  # type: ignore[arg-type]
        rendered = json.dumps(request.prompt)
        assert "scenario_id" not in rendered
        assert "expected_terminal" not in rendered
        assert "V2" not in rendered


class TestTheOperationInstanceIdentity:
    def test_it_is_opaque_and_derived_from_nothing_the_benchmark_authored(
        self, spec: OperationSpec
    ) -> None:
        from operatebench.core.instance import (
            OPERATION_INSTANCE_ID_PREFIX,
            new_operation_instance_id,
            operation_instance_id_problem,
        )

        minted = {new_operation_instance_id() for _ in range(16)}
        assert len(minted) == 16, "an instance identity is fresh per run"
        for value in minted:
            assert operation_instance_id_problem(value) is None
            assert value.startswith(OPERATION_INSTANCE_ID_PREFIX)
            body = value[len(OPERATION_INSTANCE_ID_PREFIX) :]
            for authored in (
                *SCENARIOS,
                spec.semantic_scenario_id,
                spec.operation_id,
                spec.spec_digest_sha256,
            ):
                assert authored.lower() not in body.lower()

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_two_runs_of_one_scenario_are_two_operation_instances(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        first = run_episode(spec, scenario_id, "reference", self_check=False)
        second = run_episode(spec, scenario_id, "reference", self_check=False)
        assert first.operation_instance_id != second.operation_instance_id

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_the_instance_identity_holds_across_the_self_check(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        # The self-check re-executes the episode. A per-execution identity would
        # move every observation digest and fail deterministic_replay; a per-run
        # one reproduces exactly.
        run = run_episode(spec, scenario_id, "reference")
        assert run.evaluation.dimension("deterministic_replay").ok is True
        assert run.reliable is (scenario_id != "V2")
        assert run.evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )

    def test_a_model_run_replays_from_its_tape_under_one_instance_identity(
        self, spec: OperationSpec
    ) -> None:
        transport = ReferenceDrivenTransport()
        run = run_episode(
            spec,
            "V1",
            "model_reference",
            agent_factory=lambda: ModelAgent(
                transport, model=TEST_MODEL, agent_id="model_reference"
            ),
            agent_kind="model",
        )
        assert run.evaluation.dimension("deterministic_replay").ok is True
        # One pass over the provider: playback rebuilt every request digest from
        # the recorded instance identity and reached no provider at all.
        assert transport.calls == len(run.tape)

    def test_the_artefact_records_the_instance_identity_and_the_provenance(
        self, spec: OperationSpec
    ) -> None:
        run = run_episode(spec, "V1", "reference", self_check=False)
        artifact = build_artifact(run)
        assert artifact["operation_instance_id"] == run.operation_instance_id
        # Internal provenance is untouched: the record still says which authored
        # case this was, because a grader reads the artefact and an agent does not.
        assert artifact["scenario_id"] == "V1"
        assert artifact["semantic_scenario_id"] == spec.semantic_scenario_id
        assert artifact["expected_terminal"] == spec.scenario("V1").expected_terminal
        identity = run.identity()
        assert identity["scenario_id"] == "V1"
        assert identity["operation_instance_id"] == run.operation_instance_id


# ------------------------------------------------------- P0-B: WAIT is causal


class _StubDomain:
    """A Core-only operation with a public vocabulary and a private future.

    Exists so "the same public observation with a different hidden queue" is a
    thing a test can construct exactly. The domain accepts everything and the
    state it publishes does not depend on what is still pending, which is the
    whole point: any difference in the wait declaration would then be a
    difference the agent could not have seen.
    """

    def __init__(self, events: Sequence[PlannedEvent]) -> None:
        self._events = tuple(events)

    def build_plan(self) -> EpisodePlan:
        return EpisodePlan(
            starts_at="2025-01-01T09:00:00Z",
            events=self._events,
            horizon_minutes=10080,
            max_turns_per_invocation=4,
            max_invocations=20,
        )

    def initial_state(self) -> dict[str, Any]:
        return {"events_seen": 0}

    def canonical_state(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}

    def coarse_phase(self, state: Mapping[str, Any]) -> str:
        return "STUB"

    def event_authority(self, actor_id: str, event_type: str) -> str:
        return AUTHORITY_ACTOR_CLAIM

    def retrieval_catalogue(self) -> Mapping[str, Mapping[str, Any]]:
        return {}

    def retrieval_record_contract(self) -> tuple[str, str, str, int]:
        return ("construct", "construct record", "canonical-json-sha256-prefix-v1", 16)

    def read_requirements(self) -> ReadRequirementContract:
        return ReadRequirementContract({})

    def serve_retrieval(
        self,
        state: Any,
        requests: Sequence[RetrievalRequest],
        as_of: str,
    ) -> Sequence[ToolResult]:
        return ()

    def policy_view(self) -> Mapping[str, Any]:
        return {}

    def actor_roles(self) -> Mapping[str, str]:
        return {}

    def message_fixture_ids(self) -> Sequence[str]:
        return ()

    def action_schemas(self) -> Mapping[str, Mapping[str, Mapping[str, str]]]:
        return {}

    def wake_event_vocabulary(self) -> Sequence[str]:
        return ("alpha", "beta")

    def reduce_event(self, state: Any, event: Any, context: Any) -> Verdict:
        return Verdict.ok()

    def apply_action(
        self,
        state: Any,
        action_type: str,
        payload: Mapping[str, Any],
        evidence_refs: Sequence[str],
        context: Any,
    ) -> Verdict:
        return Verdict.ok()

    def apply_terminal(
        self, state: Any, evidence_refs: Sequence[str], context: Any
    ) -> Verdict:
        return Verdict.ok()

    def is_replay_final(self, state: Any) -> bool:
        return False

    def terminal_outcome(self, state: Any) -> str | None:
        return None

    def current_cycle_id(self, state: Any) -> str | None:
        return None


class _ScriptedAgent:
    """Returns one outcome per invocation, from a script."""

    agent_id = "scripted"

    def __init__(self, outcome: AgentOutcome) -> None:
        self._outcome = outcome
        self.observations: list[AgentObservation] = []

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        return None

    def decide(self, observation: AgentObservation) -> AgentOutcome:
        self.observations.append(observation)
        return self._outcome


def _event(event_id: str, event_type: str, at: str, sequence: int) -> PlannedEvent:
    return PlannedEvent(
        event_id=event_id,
        event_type=event_type,
        actor_id="world",
        sequence=sequence,
        at=at,
    )


def _rows(trajectory: Sequence[Mapping[str, Any]], record_type: str) -> list[dict]:
    return [dict(row) for row in trajectory if row["record_type"] == record_type]


class TestWaitIsDeclaredAgainstThePublicContract:
    def test_declaring_a_wait_never_asks_what_is_pending(
        self, spec: OperationSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(self: EventQueue) -> frozenset[str]:
            raise AssertionError(
                "the engine consulted the pending event queue while declaring a "
                "wait; what is scheduled is authored-future truth the agent cannot "
                "see, so a wait graded against it is graded on the answer"
            )

        monkeypatch.setattr(EventQueue, "pending_event_types", forbidden)
        for agent_id in sorted(AGENTS):
            for scenario_id in AGENTS[agent_id].scenarios or SCENARIOS:
                run_episode(spec, scenario_id, agent_id, self_check=False)

    def test_a_declared_wait_records_no_reachability_claim(
        self, spec: OperationSpec
    ) -> None:
        run = run_episode(spec, "V1", "reference", self_check=False)
        declared = [
            row
            for row in _rows(run.outcome.trajectory, "wait_declared")
            if "wake_on" in row
        ]
        assert declared, "the reference never waited"
        for row in declared:
            assert "reachable" not in row

    def test_a_known_wake_event_absent_from_the_hidden_queue_is_a_legal_wait(
        self,
    ) -> None:
        domain = _StubDomain((_event("e0", "alpha", "2025-01-01T09:00:00Z", 1),))
        agent = _ScriptedAgent(Wait(reason="waiting for beta", wake_on=("beta",)))
        engine = Engine(domain, agent, identity={"operation_id": "stub"})
        outcome = engine.run()
        declared = _rows(outcome.trajectory, "wait_declared")
        assert declared, "a wait for a known event type was not accepted"
        assert _rows(outcome.trajectory, "wait_rejected") == []
        # Time decided, not the queue: nothing ever delivered beta, so the
        # operation ran out of horizon.
        assert outcome.status == STATUS_HORIZON

    def test_an_unknown_wake_type_is_refused_against_the_public_vocabulary(
        self,
    ) -> None:
        domain = _StubDomain((_event("e0", "alpha", "2025-01-01T09:00:00Z", 1),))
        agent = _ScriptedAgent(
            Wait(reason="waiting for a type nobody declares", wake_on=("gamma",))
        )
        engine = Engine(domain, agent, identity={"operation_id": "stub"})
        outcome = engine.run()
        rejected = _rows(outcome.trajectory, "wait_rejected")
        assert rejected, "an unknown wake type was accepted"
        assert rejected[0]["code"] == "UNKNOWN_WAKE_EVENT_TYPE"
        assert "gamma" in rejected[0]["detail"]
        assert "alpha" in rejected[0]["detail"] and "beta" in rejected[0]["detail"]

    def test_two_different_hidden_futures_declare_the_same_wait(self) -> None:
        """Matched public observation, different private queue, identical evidence."""
        first = _event("e0", "alpha", "2025-01-01T09:00:00Z", 1)
        # The only difference is what the author decided would happen later.
        with_beta = _StubDomain((first, _event("e1", "beta", "2025-01-03T09:00:00Z", 2)))
        without_beta = _StubDomain((first,))

        rows: list[dict[str, Any]] = []
        digests: list[str] = []
        for domain in (with_beta, without_beta):
            agent = _ScriptedAgent(Wait(reason="waiting for beta", wake_on=("beta",)))
            engine = Engine(domain, agent, identity={"operation_id": "stub"})
            outcome = engine.run()
            declared = _rows(outcome.trajectory, "wait_declared")
            assert declared, "a legal wait was refused under one of the two futures"
            rows.append(declared[0])
            digests.append(observation_digest(agent.observations[0]))

        assert digests[0] == digests[1]
        assert json.dumps(rows[0], sort_keys=True) == json.dumps(rows[1], sort_keys=True)

    def test_a_standing_wait_that_outlives_the_queue_ends_at_the_horizon(self) -> None:
        domain = _StubDomain((_event("e0", "alpha", "2025-01-01T09:00:00Z", 1),))
        agent = _ScriptedAgent(Wait(reason="waiting for beta", wake_on=("beta",)))
        engine = Engine(domain, agent, identity={"operation_id": "stub"})
        outcome = engine.run()
        assert outcome.status == STATUS_HORIZON
        stranded = _rows(outcome.trajectory, "wait_unresolved_at_horizon")
        assert stranded, (
            "the queue emptied under a standing wait and the record says nothing "
            "about it; the horizon is what ended this operation and has to be "
            "recorded as such"
        )
        assert stranded[0]["wake_on"] == ["beta"]
        # The operation is carried to its horizon rather than stopping at the
        # instant the author's last event happened to sit at.
        assert outcome.simulated_minutes == domain.build_plan().horizon_minutes

    def test_a_wait_after_the_operation_is_replay_final_is_refused(self) -> None:
        class _FinalDomain(_StubDomain):
            def __init__(self) -> None:
                super().__init__((_event("e0", "alpha", "2025-01-01T09:00:00Z", 1),))
                self._final = False

            def apply_action(
                self,
                state: Any,
                action_type: str,
                payload: Mapping[str, Any],
                evidence_refs: Sequence[str],
                context: Any,
            ) -> Verdict:
                # An accepted effect that ends the operation. An ACT does not end
                # the invocation, so the agent gets another turn — which is the
                # only way a wait can be declared against a finished operation.
                self._final = True
                return Verdict.ok()

            def is_replay_final(self, state: Any) -> bool:
                return self._final

        class _ActThenWait:
            agent_id = "act_then_wait"

            def __init__(self) -> None:
                self._turns = 0

            def begin_episode(self, identity: Mapping[str, Any]) -> None:
                return None

            def decide(self, observation: AgentObservation) -> AgentOutcome:
                self._turns += 1
                if self._turns == 1:
                    return Act(action_type="finish")
                return Wait(reason="still waiting", wake_on=("alpha",))

        engine = Engine(_FinalDomain(), _ActThenWait(), identity={"operation_id": "s"})
        outcome = engine.run()
        rejected = _rows(outcome.trajectory, "wait_rejected")
        assert [row["code"] for row in rejected] == ["WAIT_AFTER_REPLAY_FINAL"]


class TestAlwaysWaitFailsThroughTime:
    def test_it_fails_the_oracle_declared_dimensions_without_a_reachability_claim(
        self, spec: OperationSpec
    ) -> None:
        check = check_agent(spec, AGENTS["always_wait"], "V1")
        assert check.problems == ()
        assert check.ok is True
        assert "UNREACHABLE_WAKE_CONDITION" not in check.finding_codes

    def test_its_failure_is_temporal_non_completion(self, spec: OperationSpec) -> None:
        run = run_episode(spec, "V1", "always_wait", self_check=False)
        assert run.outcome.status == STATUS_HORIZON
        codes = set(run.evaluation.finding_codes)
        assert "UNREACHABLE_WAKE_CONDITION" not in codes
        assert "WAIT_NEVER_ENDED" in codes
        temporal = run.evaluation.dimension("temporal_correctness")
        assert temporal.ok is False


class TestTheRetractedVocabularyIsGone:
    def test_no_shipped_module_still_names_unreachable_wake_conditions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in sorted((root / "src").rglob("*.py")):
            if "UNREACHABLE_WAKE_CONDITION" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))
        for path in sorted((root / "src").rglob("*.yaml")):
            if "UNREACHABLE_WAKE_CONDITION" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))
        assert offenders == []
