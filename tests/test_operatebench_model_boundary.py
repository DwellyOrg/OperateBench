"""The model boundary: parsing, provider faults, and replay without a provider.

What these tests are for, in one sentence each.

*The blocker.* ``run_episode`` used to satisfy ``deterministic_replay`` by
executing the episode twice. For a model agent that is a second provider run,
and turning the check off graded the run ``REPLAY_NOT_CHECKED``. Both arms are
asserted here, and so is the way out.

*Zero provider calls, counted.* The transport doubles count every call and
refuse any call past the ones a run is allowed. "Replay reached no provider" is
therefore a number a test reads, not a property a comment claims.

*Per-invocation binding.* Every recorded decision carries the digest of the
observation that produced it, so a tape cannot be replayed by index against a
run it never described.

*No provider text in durable evidence.* The fake responses carry a marker
string. No artefact, trajectory or classification detail may contain it.

*Nobody chooses the instance identity.* The provider-capable public run API
mints it, once per run, and has no parameter that would let a caller hand one
in — so an identity cannot be carried from one run into another, however
well-formed it looks. Reproduction of a persisted run is a separate, private
surface that validates what it is given before the agent begins.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from inspect import signature
from pathlib import Path
from typing import Any

import pytest

from operatebench.agents.model import (
    AGENT_TOOL_NAMES,
    CLASSIFICATIONS,
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    MalformedModelOutcome,
    ModelAgent,
    ModelBoundaryError,
    ModelExceededOutputLimit,
    ModelNamedAnUnknownTool,
    ModelOutputTruncated,
    ModelReturnedMultipleToolCalls,
    ModelReturnedNoToolCall,
    ModelToolArgumentsMalformed,
    classification,
    parse_tool_call,
    tool_schema,
)
from operatebench.agents.playback import (
    DecisionOrderError,
    DecisionTape,
    ObservationMismatchError,
    RecordedDecision,
    RecordedOutcomeAgent,
    TapeExhaustedError,
    TapeUnconsumedError,
    decision_record,
    outcome_from_record,
)
from operatebench.agents.transport import (
    FAULT_BUDGET,
    FAULT_PROTOCOL,
    FAULT_TRANSPORT,
    MAX_ATTEMPTS,
    RETRY_COUNT,
    SOURCE_MODEL,
    ForbiddenTransport,
    ModelResponse,
    ProviderCallForbiddenError,
    ProviderFailure,
    RecordingTransport,
    ToolCall,
    observation_digest,
)
from operatebench.artifact import (
    ARTIFACT_VERSION,
    artifact_text,
    build_artifact,
    replay_artifact,
)
from operatebench.core.instance import OperationInstanceError
from operatebench.core.outcomes import (
    MALFORMED_OUTCOME_CODE,
    Act,
    Ask,
    Complete,
    Escalate,
    Wait,
)
from operatebench.core.protocol import AgentObservation
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.providers.config import ProviderConfigurationError
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import (
    SELF_CHECK_PLAYBACK,
    SELF_CHECK_RERUN,
    EpisodeRun,
    execute_episode,
    playback_episode,
    run_episode,
)
from tests.model_transport import (
    PROVIDER_PROSE,
    TEST_MODEL,
    CountingTransport,
    ReferenceDrivenTransport,
    response_for,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
)

MODEL_AGENT_ID = "model_reference"


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


def model_run(
    spec: OperationSpec,
    *,
    scenario_id: str = "V1",
    transport: Any = None,
    self_check: bool = True,
    self_check_mode: str | None = None,
) -> tuple[EpisodeRun, Any]:
    """One episode whose decisions travel the model boundary."""
    wire = ReferenceDrivenTransport() if transport is None else transport
    run = run_episode(
        spec,
        scenario_id,
        MODEL_AGENT_ID,
        self_check=self_check,
        self_check_mode=self_check_mode,
        agent_factory=lambda: ModelAgent(wire, model=TEST_MODEL, agent_id=MODEL_AGENT_ID),
        agent_kind="model",
    )
    return run, wire


def _recorded_run(spec: OperationSpec, tmp_path_factory: Any, **kwargs: Any) -> Any:
    """One model run through the real SDK, with the journal that witnessed it.

    Contract 7 refuses an episode artefact for a model run with no complete
    scored ledger, so every claim below that builds one has to be made about a
    run that was actually recorded. The boundary transports above stay exactly
    where they were for the claims that do not need an artefact.
    """
    from tests.provider_evidence_runs import model_run_with_ledger

    directory = tmp_path_factory.mktemp("model_boundary_evidence")
    directory.chmod(0o700)
    return model_run_with_ledger(directory, spec, agent_id=MODEL_AGENT_ID, **kwargs)


def _prose_only() -> Any:
    """A 200 whose body is a message and no tool call at all."""
    import httpx

    from tests.openai_transport import message_item, responses_body
    from tests.provider_evidence_runs import INPUT_TOKENS, OUTPUT_TOKENS

    return httpx.Response(
        200,
        json=responses_body(
            [message_item(PROVIDER_PROSE)],
            model=GPT_5_6_LUNA_MODEL,
            input_tokens=INPUT_TOKENS,
            output_tokens=OUTPUT_TOKENS,
        ),
    )


def _invented_tool_call() -> Any:
    """A 200 naming a tool this build never offered, with prose beside it."""
    import httpx

    from tests.openai_transport import function_call_item, message_item, responses_body
    from tests.provider_evidence_runs import INPUT_TOKENS, OUTPUT_TOKENS

    return httpx.Response(
        200,
        json=responses_body(
            [
                function_call_item("invented", json.dumps({"prose": PROVIDER_PROSE})),
                message_item(PROVIDER_PROSE),
            ],
            model=GPT_5_6_LUNA_MODEL,
            input_tokens=INPUT_TOKENS,
            output_tokens=OUTPUT_TOKENS,
        ),
    )


def observation(**overrides: Any) -> AgentObservation:
    base: dict[str, Any] = {
        "now": "2025-01-01T09:00:00Z",
        "operation_id": "op",
        "operation_instance_id": "opinst_0123456789abcdef0123456789abcdef",
        "invocation_index": 1,
        "turn_index": 0,
        "policy": {},
        "actors": {},
        "message_fixture_ids": [],
        "last_rejection": None,
    }
    base.update(overrides)
    return AgentObservation(**base)


# --------------------------------------------------------------- the blocker


class TestTheVerifiedBlocker:
    """The two arms of the blocker, and the path that removes both."""

    def test_a_rerun_self_check_is_refused_for_a_model_agent_by_name(
        self, spec: OperationSpec
    ) -> None:
        transport = ReferenceDrivenTransport()
        with pytest.raises(Exception) as caught:
            model_run(spec, transport=transport, self_check_mode=SELF_CHECK_RERUN)
        message = str(caught.value)
        assert "single-use" in message
        assert SELF_CHECK_PLAYBACK in message, (
            "a refusal has to name the thing to do instead, or the caller's only "
            "option is to turn the check off"
        )

    def test_a_model_episode_satisfies_deterministic_replay_without_a_second_run(
        self, spec: OperationSpec
    ) -> None:
        run, transport = model_run(spec)
        replay = run.evaluation.dimension("deterministic_replay")
        assert replay.ok is True
        assert [finding.code for finding in replay.findings] == []
        # The whole point: the check ran, and it cost exactly one pass over the
        # provider — one call per decision the agent was asked for.
        assert transport.calls == len(run.tape)
        assert run.reliable is True

    def test_turning_the_self_check_off_still_leaves_the_dimension_unchecked(
        self, spec: OperationSpec
    ) -> None:
        # The other arm of the blocker, kept as a regression: skipping the check
        # is not an alternative to it, and must keep saying so.
        run, _ = model_run(spec, self_check=False)
        replay = run.evaluation.dimension("deterministic_replay")
        assert replay.ok is False
        assert [finding.code for finding in replay.findings] == ["REPLAY_NOT_CHECKED"]

    def test_a_deterministic_agent_still_gets_the_rerun_check(
        self, spec: OperationSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operatebench.runner as runner

        built: list[str] = []
        real = runner.build_agent
        monkeypatch.setattr(
            runner,
            "build_agent",
            lambda agent_id: (built.append(agent_id), real(agent_id))[1],
        )
        run = run_episode(spec, "V1", "reference")
        assert built == ["reference", "reference"], (
            "a deterministic agent is free to execute twice, and re-executing it "
            "checks the agent as well as the environment"
        )
        assert run.evaluation.dimension("deterministic_replay").ok is True

    def test_an_unknown_self_check_mode_is_refused(self, spec: OperationSpec) -> None:
        with pytest.raises(Exception, match="not a determinism self-check"):
            run_episode(spec, "V1", "reference", self_check_mode="trust_me")


# ------------------------------------------------------ zero provider calls


class TestProviderCallsAreCounted:
    def test_local_configuration_refusal_is_not_a_transport_attempt(self) -> None:
        response = ModelResponse(
            TEST_MODEL, "tool_use", (ToolCall("complete", {"reason": "done"}),)
        )

        class RefusingSecondDispatch:
            calls = 0

            def send(self, _request: Any) -> ModelResponse:
                self.calls += 1
                if self.calls == 1:
                    return response
                raise ProviderConfigurationError("local mutable state refused")

        transport = RefusingSecondDispatch()
        agent = ModelAgent(transport, model=TEST_MODEL)
        agent.begin_episode({})
        agent.decide(observation())

        with pytest.raises(ProviderConfigurationError):
            agent.decide(observation(turn_index=1))

        record = agent.execution_record()
        assert transport.calls == 2
        assert record.transport_calls == 1
        assert len(record.attempts) == 1
        assert record.attempts[0].outcome == "decided"
        assert all(attempt.fault != FAULT_TRANSPORT for attempt in record.attempts)

    def test_the_forbidden_transport_refuses_and_counts_every_call(self) -> None:
        forbidden = ForbiddenTransport()
        agent = ModelAgent(forbidden, model=TEST_MODEL)
        agent.begin_episode({})
        with pytest.raises(ProviderCallForbiddenError):
            agent.decide(observation())
        assert forbidden.calls == 1, (
            "counting matters as much as raising: a path that swallowed the "
            "refusal would otherwise look like a path that never called"
        )

    def test_playback_of_a_recorded_run_makes_zero_provider_calls(
        self, spec: OperationSpec, tmp_path_factory: Any
    ) -> None:
        # Through the real SDK over a mock transport, with an evidence recorder
        # attached: contract 7 writes no episode artefact for a model run that
        # has no complete scored journal behind it, so a recorded run that can
        # be replayed at all is one that was recorded.
        evidence = _recorded_run(spec, tmp_path_factory)
        spent = len(evidence.wire.bodies)
        artifact = build_artifact(evidence.run)

        first = replay_artifact(spec, artifact)
        second = replay_artifact(spec, artifact)

        assert first.ok and second.ok
        assert len(evidence.wire.bodies) == spent, (
            "a replay that reached the provider would move this number; it is the "
            "direct count, not a proxy for one"
        )
        assert first.reproduction == "playback"

    def test_a_call_past_the_recorded_ones_is_refused_by_the_scripted_transport(
        self, spec: OperationSpec, tmp_path_factory: Any
    ) -> None:
        # An injected transport that raises on any call beyond the recorded run.
        # If any part of the self-check, the artefact build or the replay asked
        # the provider again, this is where it would land.
        evidence = _recorded_run(spec, tmp_path_factory)
        spent = len(evidence.wire.bodies)
        artifact = build_artifact(evidence.run)
        assert replay_artifact(spec, artifact).ok
        assert len(evidence.wire.bodies) == spent, "replay spent no further calls"

        # And the scripted transport is still what proves a call past the end of
        # a recorded run is refused rather than answered.
        exact = RecordingTransport(
            [
                response_for(outcome_from_record(d.outcome))  # type: ignore[arg-type]
                for d in evidence.run.tape.decisions
            ]
        )
        assert exact.calls == 0
        with pytest.raises(ProviderCallForbiddenError, match="past the end"):
            for _ in range(exact.scripted + 1):
                ModelAgent(exact, model=TEST_MODEL).decide(observation())

    def test_playback_never_builds_an_agent_that_holds_a_transport(
        self, spec: OperationSpec
    ) -> None:
        run, _ = model_run(spec)
        played = playback_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            run.tape,
            operation_instance_id=run.operation_instance_id,
        )
        assert played.execution.transport_calls == 0
        assert played.execution.outcome_source == "recorded"
        assert played.outcome.trajectory_digest_sha256 == (
            run.outcome.trajectory_digest_sha256
        )


# ------------------------------------------------ the minted-only identity


class _BeginCountingAgent:
    """An agent that counts being begun and refuses to decide anything.

    It exists so "the refusal happened before the agent began" is a number a
    test reads rather than a property a comment claims: a boundary that
    validated after :meth:`begin_episode` would leave ``begun`` at one, and one
    that validated after the first decision would raise from :meth:`decide`.
    """

    agent_id = MODEL_AGENT_ID

    def __init__(self) -> None:
        self.begun = 0

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.begun += 1

    def decide(self, observation: AgentObservation) -> Any:
        raise AssertionError(
            "an episode carrying a refused instance identity reached a decision"
        )


class _CountingModelAgent(ModelAgent):
    """A real model agent that counts being begun and being asked to decide.

    Real, and not a stand-in, because the guard under test is about what an
    agent *is*: a double that merely looked provider-backed would prove the
    refusal reads a name. This one holds a transport and would spend it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.begun = 0
        self.decided = 0

    def begin_episode(self, identity: Mapping[str, Any]) -> None:
        self.begun += 1
        super().begin_episode(identity)

    def decide(self, observation: AgentObservation) -> Any:
        self.decided += 1
        return super().decide(observation)


#: Every way a value is not an instance identity this build writes, including
#: the two that matter adversarially: a scenario identity, and a string built
#: out of one. Each is a named refusal, not a value read back as an identity.
MALFORMED_INSTANCE_IDS: tuple[Any, ...] = (
    "V1",
    "scenario_id=V2",
    "",
    "opinst_",
    "opinst_" + "0" * 31,
    "opinst_" + "0" * 33,
    "opinst_" + "z" * 32,
    "opinst_" + "0123456789ABCDEF" * 2,
    "0123456789abcdef0123456789abcdef",
    None,
    7,
    b"opinst_0123456789abcdef0123456789abcdef",
)

#: Well-formed, and therefore exactly the case the shape check cannot catch: a
#: caller who could hand this in could hand the same one to a V1 run and a V2
#: run. The public API's answer is that there is nowhere to hand it in.
WELL_FORMED_INSTANCE_ID = "opinst_0123456789abcdef0123456789abcdef"


class TestOnlyAMintedInstanceIdentityReachesAProvider:
    def test_the_provider_capable_run_api_has_no_identity_parameter(self) -> None:
        for public in (run_episode, execute_episode):
            assert "operation_instance_id" not in signature(public).parameters, (
                f"{public.__name__} can be handed a model agent, so a caller-chosen "
                "instance identity is one that can be reused across runs and "
                "scenarios and still reach a provider"
            )

    @pytest.mark.parametrize("scenario_id", ["V1", "V2"])
    def test_run_episode_refuses_a_chosen_identity_before_any_provider_call(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        transport = ReferenceDrivenTransport(max_calls=0)
        with pytest.raises(TypeError, match="operation_instance_id"):
            run_episode(
                spec,
                scenario_id,
                MODEL_AGENT_ID,
                agent_factory=lambda: ModelAgent(
                    transport, model=TEST_MODEL, agent_id=MODEL_AGENT_ID
                ),
                agent_kind="model",
                operation_instance_id=WELL_FORMED_INSTANCE_ID,
            )
        assert transport.calls == 0, (
            "the refusal has to land before the transport, not after a run that "
            "already spent the provider"
        )

    def test_execute_episode_refuses_a_chosen_identity_before_any_provider_call(
        self, spec: OperationSpec
    ) -> None:
        transport = ReferenceDrivenTransport(max_calls=0)
        with pytest.raises(TypeError, match="operation_instance_id"):
            execute_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                agent_factory=lambda: ModelAgent(
                    transport, model=TEST_MODEL, agent_id=MODEL_AGENT_ID
                ),
                operation_instance_id=WELL_FORMED_INSTANCE_ID,
            )
        assert transport.calls == 0

    def test_exactly_one_identity_is_minted_per_run_and_the_self_check_reuses_it(
        self, spec: OperationSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operatebench.runner as runner

        real = runner.new_operation_instance_id
        minted: list[str] = []

        def counted() -> str:
            minted.append(real())
            return minted[-1]

        monkeypatch.setattr(runner, "new_operation_instance_id", counted)
        run = run_episode(spec, "V1", "reference")
        assert minted == [run.operation_instance_id], (
            "one run is one operation instance: the determinism self-check "
            "re-executes under the identity the run already has"
        )

    def test_a_well_formed_scenario_derived_identity_cannot_enter_a_fresh_run(
        self, spec: OperationSpec
    ) -> None:
        # No override surface, so the only thing left to assert is what the
        # runner does on its own: a fresh identity per run, shared by nothing.
        first, _ = model_run(spec, scenario_id="V1")
        second, _ = model_run(spec, scenario_id="V2")
        assert first.operation_instance_id != second.operation_instance_id
        for run in (first, second):
            assert run.operation_instance_id != WELL_FORMED_INSTANCE_ID
            body = run.operation_instance_id.removeprefix("opinst_")
            assert "v1" not in body and "v2" not in body

    def test_the_engine_boundary_names_a_malformed_identity_before_the_agent_begins(
        self, spec: OperationSpec
    ) -> None:
        # The private reproduction helper is the only surface that takes a
        # persisted identity at all, and it is reached here deliberately: this
        # is the boundary under test.
        from operatebench.runner import _reproduce_episode

        for value in MALFORMED_INSTANCE_IDS:
            agent = _BeginCountingAgent()
            built: list[str] = []
            with pytest.raises(OperationInstanceError) as caught:
                _reproduce_episode(
                    spec,
                    "V1",
                    MODEL_AGENT_ID,
                    agent_factory=lambda bound=agent, sink=built: (
                        sink.append("built"),
                        bound,
                    )[1],
                    agent_kind="model",
                    operation_instance_id=value,
                )
            assert "operation_instance_id" in str(caught.value)
            assert agent.begun == 0 and built == [], (
                f"{value!r} was refused only after the episode had already built "
                "and begun an agent"
            )

    def test_the_reproduction_helper_accepts_the_identity_the_record_carries(
        self, spec: OperationSpec
    ) -> None:
        from operatebench.runner import _reproduce_episode

        run, transport = model_run(spec)
        spent = transport.calls
        again = _reproduce_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            agent_factory=lambda: RecordedOutcomeAgent(run.tape, agent_id=MODEL_AGENT_ID),
            agent_kind="model",
            operation_instance_id=run.operation_instance_id,
        )
        assert again.operation_instance_id == run.operation_instance_id
        assert again.outcome.trajectory_digest_sha256 == (
            run.outcome.trajectory_digest_sha256
        )
        assert transport.calls == spent, "reproduction reached no provider"

    def test_the_reproduction_helper_refuses_a_provider_backed_agent_by_name(
        self, spec: OperationSpec
    ) -> None:
        # The identity here is a real one, carried by a run that actually
        # happened, so nothing else about the call can be what the refusal is
        # about: the one thing wrong is the agent the factory builds. A
        # reproduction re-executes a record, and a record is reproduced from its
        # own decisions — an agent that would ask a provider for new ones is not
        # reproducing anything, whatever identity it is handed.
        from operatebench.runner import ReproductionAgentError, _reproduce_episode

        recorded, _ = model_run(spec)
        wire = CountingTransport(
            AssertionError("a reproduction reached the provider boundary")
        )
        agent = _CountingModelAgent(wire, model=TEST_MODEL, agent_id=MODEL_AGENT_ID)
        built: list[str] = []

        with pytest.raises(ReproductionAgentError) as caught:
            _reproduce_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                operation_instance_id=recorded.operation_instance_id,
                agent_factory=lambda: (built.append("built"), agent)[1],
                agent_kind="model",
            )

        message = str(caught.value)
        assert "RecordedOutcomeAgent" in message and "RecordedModelAgent" in message, (
            "a refusal has to name what reproduction does accept, or the caller "
            "cannot tell a closed door from a broken one"
        )
        assert agent.begun == 0 and agent.decided == 0, (
            "the refusal landed after the episode had already begun the agent"
        )
        assert wire.calls == 0, (
            "zero provider calls is the number this guard exists for; it is read "
            "off the transport, not asserted about the code path"
        )
        assert built == ["built"], (
            "the factory is called exactly once, to inspect what it builds; a "
            "second call would be a second agent, and the guard would have "
            "checked something other than what runs"
        )

    def test_a_reproduction_builds_the_agent_it_was_given_exactly_once(
        self, spec: OperationSpec
    ) -> None:
        # The rerun self-check executes the episode twice, and the guard has to
        # inspect the agent before either execution. Both are satisfied by
        # building once and re-using that instance, which is also what a tape
        # needs: a RecordedOutcomeAgent resets its cursor per episode.
        from operatebench.runner import _reproduce_episode

        recorded, _ = model_run(spec)
        built: list[str] = []

        def factory() -> Any:
            built.append("built")
            return RecordedOutcomeAgent(recorded.tape, agent_id=MODEL_AGENT_ID)

        again = _reproduce_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            operation_instance_id=recorded.operation_instance_id,
            agent_factory=factory,
            agent_kind="model",
        )
        assert built == ["built"]
        assert again.evaluation.dimension("deterministic_replay").ok is True

    def test_playback_is_structurally_incapable_of_holding_an_agent(self) -> None:
        parameters = signature(playback_episode).parameters
        assert "agent_factory" not in parameters, (
            "playback may take a persisted identity only because there is no way "
            "to hand it something that could call a provider"
        )
        assert "operation_instance_id" in parameters

    def test_playback_names_a_malformed_identity_before_the_engine(
        self, spec: OperationSpec
    ) -> None:
        run, transport = model_run(spec)
        spent = transport.calls
        for value in MALFORMED_INSTANCE_IDS:
            with pytest.raises(OperationInstanceError, match="operation_instance_id"):
                playback_episode(
                    spec, "V1", MODEL_AGENT_ID, run.tape, operation_instance_id=value
                )
        assert transport.calls == spent


# --------------------------------------------------- per-invocation binding


class TestObservationBinding:
    """A tape is bound to what produced it, not to a position in a list."""

    def test_a_tape_replayed_against_another_scenario_is_refused(
        self, spec: OperationSpec
    ) -> None:
        run, _ = model_run(spec, scenario_id="V1")
        with pytest.raises(ObservationMismatchError, match="index-only replay"):
            playback_episode(
                spec,
                "V2",
                MODEL_AGENT_ID,
                run.tape,
                operation_instance_id=run.operation_instance_id,
            )

    def test_an_edited_observation_digest_is_refused(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        decisions = list(run.tape.decisions)
        decisions[3] = replace(decisions[3], observation_digest_sha256="0" * 64)
        with pytest.raises(ObservationMismatchError):
            playback_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                DecisionTape(tuple(decisions)),
                operation_instance_id=run.operation_instance_id,
            )

    def test_reordered_decisions_are_refused(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        decisions = list(run.tape.decisions)
        decisions[2], decisions[3] = decisions[3], decisions[2]
        with pytest.raises((DecisionOrderError, ObservationMismatchError)):
            playback_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                DecisionTape(tuple(decisions)),
                operation_instance_id=run.operation_instance_id,
            )

    def test_a_missing_decision_is_refused(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        short = DecisionTape(run.tape.decisions[:-1])
        with pytest.raises(TapeExhaustedError, match="refused rather than improvised"):
            playback_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                short,
                operation_instance_id=run.operation_instance_id,
            )

    def test_an_extra_decision_is_refused(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        last = run.tape.decisions[-1]
        extra = DecisionTape(
            (
                *run.tape.decisions,
                replace(last, invocation_index=last.invocation_index + 1),
            )
        )
        with pytest.raises(TapeUnconsumedError, match="not the same run"):
            playback_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                extra,
                operation_instance_id=run.operation_instance_id,
            )

    def test_an_altered_decision_identity_is_refused(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        decisions = list(run.tape.decisions)
        decisions[1] = replace(decisions[1], turn_index=decisions[1].turn_index + 7)
        with pytest.raises(DecisionOrderError, match="order moved"):
            playback_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                DecisionTape(tuple(decisions)),
                operation_instance_id=run.operation_instance_id,
            )

    def test_the_digest_is_over_the_public_observation_and_nothing_else(self) -> None:
        one = observation()
        same = observation()
        other = observation(phase="A_DIFFERENT_PHASE")
        assert observation_digest(one) == observation_digest(same)
        assert observation_digest(one) != observation_digest(other)


class TestPlaybackDeterminism:
    def test_playing_the_same_tape_twice_resets_the_cursor_and_is_identical(
        self, spec: OperationSpec
    ) -> None:
        # Internal reproduction, through the private surface that exists for it.
        # Under the recorded run's own instance identity, both times: a tape is
        # bound to the observations that produced it and those name the run, so
        # replaying it under a fresh identity is replaying it against a different
        # run — which playback refuses, correctly. The public run API has no way
        # to say "this identity", which is why this reaches past it.
        from operatebench.runner import _reproduce_episode

        run, _ = model_run(spec)
        agent = RecordedOutcomeAgent(run.tape, agent_id=MODEL_AGENT_ID)

        first = _reproduce_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            operation_instance_id=run.operation_instance_id,
            agent_factory=lambda: agent,
            agent_kind="model",
        )
        assert agent.cursor == len(run.tape)
        second = _reproduce_episode(
            spec,
            "V1",
            MODEL_AGENT_ID,
            operation_instance_id=run.operation_instance_id,
            agent_factory=lambda: agent,
            agent_kind="model",
        )

        assert first.outcome.trajectory_digest_sha256 == (
            second.outcome.trajectory_digest_sha256
        )
        assert first.outcome.final_state_digest_sha256 == (
            second.outcome.final_state_digest_sha256
        )
        assert build_artifact(first) == build_artifact(second)


# ------------------------------------------------------------ the parser


class TestOutcomeParsing:
    @pytest.mark.parametrize(
        "outcome",
        [
            Act("book_visit", {"slot": 1}, ("evidence_1",), "because"),
            Wait("waiting", ("supplier_visit_response",), 60),
            Ask("customer_1", "msg_cannot_proceed", Wait("w", (), 30), "corr"),
            Escalate("cp_1", "authority_gap", ("evidence_1",), 90, "why"),
            Complete("done", ("evidence_1",)),
        ],
    )
    def test_every_offered_outcome_round_trips(self, outcome: Any) -> None:
        from tests.model_transport import tool_call_for

        assert parse_tool_call(tool_call_for(outcome)) == outcome

    def test_the_schema_offered_and_the_schema_enforced_are_one_table(self) -> None:
        schema = tool_schema()
        assert tuple(schema) == AGENT_TOOL_NAMES
        call = ToolCall("act", {"action_type": "book_visit", "not_a_field": 1})
        parsed = parse_tool_call(call)
        assert isinstance(parsed, ModelToolArgumentsMalformed)
        for name in schema["act"]["required"] + schema["act"]["optional"]:
            assert name in parsed.detail or name == "action_type"

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            (
                ModelResponse(TEST_MODEL, "end_turn", (), PROVIDER_PROSE),
                ModelReturnedNoToolCall,
            ),
            (
                ModelResponse(
                    TEST_MODEL,
                    "tool_use",
                    (
                        ToolCall("wait", {"reason": "a", "fallback_after_minutes": 10}),
                        ToolCall("complete", {}),
                    ),
                ),
                ModelReturnedMultipleToolCalls,
            ),
            (
                ModelResponse(TEST_MODEL, "max_tokens", (ToolCall("complete", {}),)),
                ModelOutputTruncated,
            ),
            (
                ModelResponse(
                    TEST_MODEL,
                    "tool_use",
                    (ToolCall("complete", {}),),
                    output_tokens=MAX_OUTPUT_TOKENS + 1,
                ),
                ModelExceededOutputLimit,
            ),
            (
                ModelResponse(
                    TEST_MODEL, "tool_use", (ToolCall("hallucinated_tool", {}),)
                ),
                ModelNamedAnUnknownTool,
            ),
            (
                ModelResponse(TEST_MODEL, "tool_use", (ToolCall("act", {}),)),
                ModelToolArgumentsMalformed,
            ),
            (
                ModelResponse(
                    TEST_MODEL,
                    "tool_use",
                    (ToolCall("act", {"action_type": "x", "evidence_refs": "one_ref"}),),
                ),
                ModelToolArgumentsMalformed,
            ),
            (
                ModelResponse(
                    TEST_MODEL, "tool_use", (ToolCall("act", {"action_type": ""}),)
                ),
                ModelToolArgumentsMalformed,
            ),
            (
                ModelResponse(
                    TEST_MODEL, "tool_use", (ToolCall("wait", {"reason": "forever"}),)
                ),
                ModelToolArgumentsMalformed,
            ),
            (
                ModelResponse(
                    TEST_MODEL, "tool_use", (ToolCall("act", ["not", "a", "map"]),)
                ),
                ModelToolArgumentsMalformed,
            ),
            (
                ModelResponse(
                    TEST_MODEL,
                    "tool_use",
                    (
                        ToolCall(
                            "ask",
                            {
                                "recipient_actor_id": "customer_1",
                                "message_fixture_id": "msg_cannot_proceed",
                                "wait": "not a mapping",
                            },
                        ),
                    ),
                ),
                ModelToolArgumentsMalformed,
            ),
        ],
    )
    def test_each_way_a_response_is_not_a_decision_has_its_own_name(
        self, response: ModelResponse, expected: type[MalformedModelOutcome]
    ) -> None:
        agent = ModelAgent(RecordingTransport([response]), model=TEST_MODEL)
        agent.begin_episode({})
        decision = agent.decide(observation())
        assert isinstance(decision, expected)
        assert decision.code in CLASSIFICATIONS

    def test_every_classification_rebuilds_from_its_code_alone(self) -> None:
        for code, cls in CLASSIFICATIONS.items():
            assert isinstance(classification(code), cls)
        with pytest.raises(ModelBoundaryError, match="not a model output"):
            classification("MODEL_INVENTED_CODE")

    def test_no_classification_ever_quotes_what_the_provider_said(self) -> None:
        secret = "SECRET-TOOL-NAME-AND-ARGUMENT"
        calls = (
            ToolCall(secret, {"anything": secret}),
            ToolCall("act", {"action_type": "x", secret: secret}),
            ToolCall("complete", {secret: 1}),
        )
        for call in calls:
            parsed = parse_tool_call(call)
            assert isinstance(parsed, MalformedModelOutcome)
            assert secret not in parsed.detail, (
                "a classification detail is built from this build's vocabulary "
                "plus counts and lengths; quoting the model puts unreviewed "
                "output into a record the benchmark publishes"
            )


class TestRequestIdentity:
    def test_the_request_pins_model_protocol_limits_and_the_observation(self) -> None:
        agent = ModelAgent(RecordingTransport([]), model=TEST_MODEL)
        agent.begin_episode({})
        request = agent.build_request(observation())
        identity = request.identity()
        assert identity["model"] == TEST_MODEL
        assert identity["protocol_version"] == MODEL_PROTOCOL_VERSION
        assert identity["max_output_tokens"] == MAX_OUTPUT_TOKENS
        assert identity["tool_names"] == list(AGENT_TOOL_NAMES)
        assert identity["observation_digest_sha256"] == observation_digest(observation())
        assert set(identity) == {
            "model",
            "protocol_version",
            "max_output_tokens",
            "tool_names",
            "observation_digest_sha256",
            "prompt_digest_sha256",
            "invocation_index",
            "turn_index",
        }

    def test_a_different_observation_is_a_different_request(self) -> None:
        agent = ModelAgent(RecordingTransport([]), model=TEST_MODEL)
        agent.begin_episode({})
        one = agent.build_request(observation())
        other = agent.build_request(observation(phase="A_DIFFERENT_PHASE"))
        assert one.request_digest_sha256 != other.request_digest_sha256

    def test_a_different_model_is_a_different_request(self) -> None:
        first = ModelAgent(RecordingTransport([]), model=TEST_MODEL)
        second = ModelAgent(RecordingTransport([]), model="other-test-model")
        first.begin_episode({})
        second.begin_episode({})
        assert (
            first.build_request(observation()).request_digest_sha256
            != second.build_request(observation()).request_digest_sha256
        )

    def test_an_unnamed_model_or_an_impossible_limit_is_refused(self) -> None:
        wire = RecordingTransport([])
        with pytest.raises(ModelBoundaryError, match="must name the model"):
            ModelAgent(wire, model="")
        with pytest.raises(ModelBoundaryError, match="positive number of tokens"):
            ModelAgent(wire, model=TEST_MODEL, max_output_tokens=0)
        with pytest.raises(ModelBoundaryError, match="at least one call"):
            ModelAgent(wire, model=TEST_MODEL, max_transport_calls=0)


# ------------------------------------ malformed output through the runner


class TestMalformedOutputThroughThePublicRunner:
    """A model that misbehaves is a recorded refusal, not a crash and not an effect."""

    def test_a_malformed_output_is_a_named_non_mutating_refusal(
        self, spec: OperationSpec
    ) -> None:
        clean, _ = model_run(spec)
        broken, _ = model_run(
            spec,
            transport=ReferenceDrivenTransport(
                interpose={2: ModelResponse(TEST_MODEL, "end_turn", (), PROVIDER_PROSE)}
            ),
        )
        refusals = [
            row
            for row in broken.outcome.trajectory
            if row["record_type"] == "outcome_rejected"
        ]
        assert len(refusals) == 1
        assert refusals[0]["code"] == MALFORMED_OUTCOME_CODE
        assert "ModelReturnedNoToolCall" in refusals[0]["detail"], (
            "the classification travels into the trajectory as the class name, so "
            "the record says which way the model failed"
        )
        # Non-mutating: the refused turn committed nothing, so the operation
        # still reaches the same terminal, in the same final state, on the very
        # next turn. The episode survives the malformed output rather than
        # ending in a traceback.
        assert broken.outcome.terminal_outcome == clean.outcome.terminal_outcome
        assert broken.outcome.final_state_digest_sha256 == (
            clean.outcome.final_state_digest_sha256
        )
        assert broken.outcome.status == clean.outcome.status
        # Surviving it is not the same as it costing nothing. The evaluator
        # scores the refusal, and scores exactly it: one dimension, one code, and
        # the run's determinism check still passes.
        assert broken.evaluation.failed_dimensions == ("action_validity",)
        assert sorted(set(broken.evaluation.finding_codes)) == [MALFORMED_OUTCOME_CODE]
        assert broken.evaluation.dimension("deterministic_replay").ok is True
        assert clean.reliable is True and broken.reliable is False

    def test_a_run_with_a_malformed_output_still_replays_from_its_tape(
        self, spec: OperationSpec, tmp_path_factory: Any
    ) -> None:
        # The answer with no tool call is injected on the wire this time, so the
        # run is one the evidence recorder witnessed and the artefact contract 7
        # requires exists. What is asserted is unchanged: the classification is
        # on the tape, and the record replays without asking again.
        evidence = _recorded_run(spec, tmp_path_factory, faults={2: _prose_only()})
        broken = evidence.run
        spent = len(evidence.wire.bodies)
        recorded = [d.outcome for d in broken.tape.decisions]
        assert {"kind": "MALFORMED", "code": "MODEL_NO_TOOL_CALL"} in recorded

        report = replay_artifact(spec, build_artifact(broken))
        assert report.ok
        assert len(evidence.wire.bodies) == spent

    def test_the_tape_records_a_classification_by_code_and_nothing_more(self) -> None:
        record = decision_record(ModelReturnedNoToolCall("a detail with prose in it"))
        assert record == {"kind": "MALFORMED", "code": "MODEL_NO_TOOL_CALL"}

    def test_no_durable_record_carries_the_provider_prose(
        self, spec: OperationSpec, tmp_path_factory: Any
    ) -> None:
        evidence = _recorded_run(
            spec, tmp_path_factory, faults={2: _invented_tool_call()}
        )
        text = artifact_text(evidence.run)
        assert PROVIDER_PROSE not in text
        assert "invented" not in text
        assert json.loads(text)["artifact_version"] == ARTIFACT_VERSION
        # And the ledger the binding names carries none of it either: the row
        # holds digests of the answer, never the answer.
        assert PROVIDER_PROSE not in evidence.ledger_path.read_text(encoding="utf-8")
        assert "invented" not in evidence.ledger_path.read_text(encoding="utf-8")


# ----------------------------------------------------- provider failures


class TestProviderFailureIsNotABusinessOutcome:
    def test_a_transport_fault_excludes_the_run_rather_than_completing_it(
        self, spec: OperationSpec
    ) -> None:
        with pytest.raises(ProviderFailure) as caught:
            model_run(
                spec,
                transport=ReferenceDrivenTransport(
                    interpose={3: ConnectionResetError("the socket closed")}
                ),
            )
        record = caught.value.record
        assert record.excluded is True
        assert record.exclusion_code == FAULT_TRANSPORT
        assert caught.value.fault == FAULT_TRANSPORT
        assert record.outcome_source == SOURCE_MODEL
        assert record.attempts, "the attempts already made are retained"
        assert record.attempts[-1].outcome == "failed"
        assert record.attempts[-1].fault == FAULT_TRANSPORT

    def test_a_response_from_another_model_is_a_protocol_failure(
        self, spec: OperationSpec
    ) -> None:
        with pytest.raises(ProviderFailure) as caught:
            model_run(
                spec,
                transport=ReferenceDrivenTransport(
                    interpose={
                        1: ModelResponse(
                            "some-other-model", "tool_use", (ToolCall("complete", {}),)
                        )
                    }
                ),
            )
        assert caught.value.record.exclusion_code == FAULT_PROTOCOL
        assert "some-other-model" not in str(caught.value), (
            "the model a response claims is provider output; its length is enough "
            "to say the identity did not match"
        )

    def test_a_bounded_budget_stop_retains_every_prior_attempt(
        self, spec: OperationSpec
    ) -> None:
        transport = ReferenceDrivenTransport()
        with pytest.raises(ProviderFailure) as caught:
            run_episode(
                spec,
                "V1",
                MODEL_AGENT_ID,
                agent_factory=lambda: ModelAgent(
                    transport,
                    model=TEST_MODEL,
                    agent_id=MODEL_AGENT_ID,
                    max_transport_calls=4,
                ),
                agent_kind="model",
            )
        record = caught.value.record
        assert record.exclusion_code == FAULT_BUDGET
        assert record.excluded is True
        assert transport.calls == 4
        assert len(record.attempts) == 4, (
            "a bounded stop is not a lost run: what was already done stays in the record"
        )
        assert all(attempt.outcome == "decided" for attempt in record.attempts)

    def test_this_build_makes_one_attempt_and_no_retries(
        self, spec: OperationSpec
    ) -> None:
        assert RETRY_COUNT == 0
        assert MAX_ATTEMPTS == 1
        wire = CountingTransport(TimeoutError("gateway timeout"))
        agent = ModelAgent(wire, model=TEST_MODEL)
        agent.begin_episode({})
        with pytest.raises(ProviderFailure):
            agent.decide(observation())
        assert wire.calls == 1, "a retry would be a second provider run"

        run, transport = model_run(spec)
        assert run.execution.retry_count == 0
        assert run.execution.transport_calls == transport.calls
        assert len(run.execution.attempts) == transport.calls
        assert {attempt.outcome for attempt in run.execution.attempts} == {"decided"}

    def test_a_forbidden_call_is_never_swallowed_into_an_exclusion(self) -> None:
        agent = ModelAgent(ForbiddenTransport(), model=TEST_MODEL)
        agent.begin_episode({})
        with pytest.raises(ProviderCallForbiddenError):
            # Not a ProviderFailure: a violated zero-call guarantee must not
            # degrade into a soft finding on an excluded run.
            agent.decide(observation())


# ------------------------------------------------------------- tape shapes


class TestTapeShapes:
    def test_a_recorded_decision_kind_this_build_does_not_write_is_refused(
        self,
    ) -> None:
        with pytest.raises(Exception, match="not a decision kind"):
            outcome_from_record({"kind": "IMPROVISE"})

    def test_a_recorded_decision_missing_its_body_is_refused(self) -> None:
        with pytest.raises(Exception, match="not shaped like one"):
            outcome_from_record({"kind": "ACT"})

    def test_the_tape_digest_follows_its_content(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        assert run.tape.digest == DecisionTape(run.tape.decisions).digest
        moved = DecisionTape(
            (
                replace(run.tape.decisions[0], observation_digest_sha256="1" * 64),
                *run.tape.decisions[1:],
            )
        )
        assert moved.digest != run.tape.digest

    def test_a_decision_records_exactly_four_things(self, spec: OperationSpec) -> None:
        run, _ = model_run(spec)
        for decision in run.tape.decisions:
            assert set(decision.as_dict()) == {
                "invocation_index",
                "turn_index",
                "observation_digest_sha256",
                "outcome",
            }

    def test_a_recorded_decision_is_a_plain_detached_mapping(self) -> None:
        decision = RecordedDecision(1, 0, "a" * 64, {"kind": "COMPLETE"})
        body: Mapping[str, Any] = decision.as_dict()
        assert body["outcome"] is not decision.outcome
