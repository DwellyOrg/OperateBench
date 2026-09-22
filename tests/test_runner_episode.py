"""One episode: the real tool loop, its limits and its failure taxonomy."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from boundarybench.adapter import (
    AdapterCall,
    AdapterProtocolError,
    ScriptedTestAdapter,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.compiler import Cube
from boundarybench.environment import Environment
from boundarybench.evaluator import evaluate
from boundarybench.ledger import (
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_EVALUATOR_FAILURE,
    OUTCOME_LIMIT_EXHAUSTED,
    OUTCOME_MODEL_ACTION_FAILURE,
    OUTCOME_MODEL_PROTOCOL_FAILURE,
    OUTCOME_SCAFFOLD_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
)
from boundarybench.runmanifest import RunLimits
from boundarybench.runner import _dispatch, _EpisodeFailure, run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from tests.scaffolds import scaffold_payload, write_scaffold

LIMITS = RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0)


def _adapter(script: Any = policy_following_script) -> ScriptedTestAdapter:
    return ScriptedTestAdapter(
        script=script, identity=identity_for_test_double(), settings={}
    )


def _run(variant: Any, script: Any = policy_following_script, **kwargs: Any) -> Any:
    return run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=_adapter(script),
        limits=kwargs.pop("limits", LIMITS),
        **kwargs,
    )


def test_episode_runs_through_the_real_environment_and_binds_its_own_verdict(
    cube: Cube,
) -> None:
    """No synthesised trajectory and no trusted summary: the loop, then the grader."""
    variant = cube.cell_variant("S0_P0")

    result = _run(variant)

    assert result.outcome == OUTCOME_SUCCESS
    assert result.error_class is None
    assert [step.action for step in result.trajectory.steps] == [
        "read_records",
        "ask_user",
        "dispatch_contractor",
        "complete_case",
    ]
    assert result.trajectory.terminal_decision is not None
    assert result.trajectory.terminal_decision.disposition == "ACT"
    assert result.turns_used == 4
    # The stored verdict is exactly what the evaluator says about this exact
    # trajectory, recomputed here rather than taken from the runner's word.
    assert result.evaluation == evaluate(variant, result.trajectory)
    assert result.evaluation is not None and result.evaluation.passed


def test_every_variant_of_both_shipped_shapes_completes(
    cube: Cube, three_act_cube: Cube
) -> None:
    for compiled in (cube, three_act_cube):
        for variant in compiled.variants:
            result = _run(variant)
            assert result.outcome == OUTCOME_SUCCESS, variant.variant_id
            assert result.evaluation == evaluate(variant, result.trajectory)


# -- the failure taxonomy ----------------------------------------------------


def test_unknown_action_is_a_model_protocol_failure(cube: Cube) -> None:
    def script(request: TurnRequest) -> AdapterCall:
        return AdapterCall("teleport", {})

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "unknown_action"
    assert "teleport" in (result.error_detail or "")
    assert result.trajectory.steps == ()


def test_malformed_arguments_are_a_model_protocol_failure(cube: Cube) -> None:
    """The partial trajectory up to the bad call is kept, not discarded."""

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("ask_user", {"fact": 7})

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_arguments"
    assert [step.action for step in result.trajectory.steps] == ["read_records"]
    assert result.evaluation is None


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, "missing"),
        ({"fact": "x", "extra": 1}, "unknown"),
        ({"fact": ["x"]}, "must be a string"),
    ],
)
def test_argument_schema_violations_are_model_protocol_failures(
    cube: Cube, arguments: dict[str, Any], expected: str
) -> None:
    def script(request: TurnRequest) -> AdapterCall:
        return AdapterCall("ask_user", arguments)

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert expected in (result.error_detail or "")


def test_a_workflow_action_this_case_never_offered_is_a_protocol_failure(
    cube: Cube,
) -> None:
    """Naming an action that was not on the turn's tool list is a protocol error.

    Lookup is against the turn's offered action list, so an unoffered call is a
    protocol error before the environment is reached. Nothing is recorded beyond
    the step that really happened.
    """

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("evict_tenant", {})

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "unknown_action"
    assert "evict_tenant" in (result.error_detail or "")
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_a_fact_key_the_case_does_not_offer_is_a_protocol_failure(
    cube: Cube,
) -> None:
    """Asking for a key outside the offered enum is an argument failure.

    The call is one the case really did offer — ``ask_user`` is on this Cube's
    list — but the key it names is not one of the keys that action's schema
    enumerates. Under the fact-affordance contract the retrievable keys are part
    of the offered interface, so this is decided during response validation
    against the schema the model was shown, and never by the environment
    consulting a namespace the model never received. That inversion is the defect
    the contract closes: a plausible key the case cannot deliver was recorded as
    the agent having chosen wrongly.
    """

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("ask_user", {"fact": "landlord_mobile_number"})

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_arguments"
    assert "repair_quote_gbp" in (result.error_detail or "")
    assert "landlord_mobile_number" not in (result.error_detail or "")
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_an_offered_action_the_environment_refuses_is_a_model_action_failure(
    cube: Cube,
) -> None:
    """Choosing an action the environment forbids is still the model's failure.

    The environment is the last word on what a case admits, and it answers for
    cases this build's own gates would not ship: here the workflow action is
    still offered while the variant records no irreversible action, so
    ``Environment.act`` refuses a call that passed every schema check. The
    machinery worked exactly as specified — it recognised the call, checked it
    and refused it — and attributing that to ``environment_failure`` would count
    a model's inadmissible choice as a defect in the harness.
    """
    variant = dataclasses.replace(cube.cell_variant("S0_P0"), irreversible_actions=())

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("dispatch_contractor", {})

    result = _run(variant, script)

    assert result.outcome == OUTCOME_MODEL_ACTION_FAILURE
    assert result.error_class == "action_rejected"
    assert "dispatch_contractor" in (result.error_detail or "")
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_adapter_exception_is_an_adapter_failure(cube: Cube) -> None:
    """Classified by the exception's class; its message is not recorded.

    An adapter's exception message is external text — see
    ``redacted_adapter_detail`` and ``tests/test_provider_output_trust.py``. The class
    is what the row is filed under and it survives; the message does not.
    """

    def script(request: TurnRequest) -> AdapterCall:
        raise RuntimeError("provider connection reset")

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert result.error_class == "RuntimeError"
    assert "provider connection reset" not in (result.error_detail or "")
    assert "RuntimeError" in (result.error_detail or "")
    assert "next_call" in (result.error_detail or "")


def test_turn_exhaustion_is_classified_and_keeps_the_partial_trajectory(
    cube: Cube,
) -> None:
    def script(request: TurnRequest) -> AdapterCall:
        return AdapterCall("read_records", {})

    result = _run(
        cube.cell_variant("S0_P0"),
        script,
        limits=RunLimits(max_turns=3, max_messages=64, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_LIMIT_EXHAUSTED
    assert result.error_class == "max_turns"
    assert result.turns_used == 3
    # read_records after the first is a no-op reveal, but each is a real step.
    assert len(result.trajectory.steps) == 3
    assert result.trajectory.terminal_decision is None


def test_timeout_is_classified_against_an_injected_clock(cube: Cube) -> None:
    """The clock is a seam, so the timeout path is deterministic in CI."""
    now = [0.0]

    def script(request: TurnRequest) -> AdapterCall:
        # The second turn is the one that overruns: the first step is recorded,
        # then the budget is gone before the loop asks for another action.
        if request.transcript:
            now[0] = 9.0  # pragma: no cover - the loop stops before this
        return policy_following_script(request)

    def advancing() -> float:
        value = now[0]
        now[0] = 9.0
        return value

    result = _run(
        cube.cell_variant("S0_P0"),
        script,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=1.0),
        clock=advancing,
    )

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"
    assert "1.0" in (result.error_detail or "")


def test_a_channel_this_case_never_offered_is_a_protocol_failure(
    call_tool_cube: Cube,
) -> None:
    """The unfiltered-action-surface defect, at the level the runner sees it.

    This Cube delivers its fact through ``call_tool`` and its card does not
    allow ``ask_user``, so ``ask_user`` is not on the turn's tool list at all.
    A model that names it has broken the protocol; it has not chosen the wrong
    one of two channels it was offered, and recording it as ``wrong_channel``
    said it had.
    """

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("ask_user", {"fact": "repair_quote_gbp"})

    result = _run(call_tool_cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "unknown_action"
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_eliciting_through_the_wrong_offered_channel_is_a_protocol_failure(
    dual_channel_cube: Cube,
) -> None:
    """Both channels offered, each enumerating only the keys it carries.

    A card that allows both acquisition channels really does put a choice in front
    of the model — and under the fact-affordance contract it also says which keys
    each channel can deliver, so choosing the wrong one is a value outside the enum
    that channel offered rather than a pairing only the card knew about. It is
    therefore refused during response validation, and the recorded steps stop at
    the last call the environment actually performed.
    """

    def script(request: TurnRequest) -> AdapterCall:
        if not request.transcript:
            return AdapterCall("read_records", {})
        return AdapterCall("ask_user", {"fact": "access_consent_confirmed"})

    result = _run(dual_channel_cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_arguments"
    assert "repair_quote_gbp" in (result.error_detail or "")
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_the_dispatch_channel_guard_still_answers_for_a_call_that_reaches_it(
    dual_channel_cube: Cube,
) -> None:
    """Defence in depth, tested where it is now the only way to reach it.

    A projected surface cannot produce this call: the enum a channel offers is
    derived from the same observations the channel check consults, so a key that
    satisfies one satisfies the other. The guard is kept because ``_dispatch``
    must never file a step under an action the case does not deliver that fact
    through, and it is exercised directly rather than left unproven — the
    trajectory-replay path applies the same rule to stored rows and is covered by
    ``tests/test_trajectory_integrity.py``.
    """
    variant = dual_channel_cube.cell_variant("S0_P0")
    environment = Environment(variant)
    environment.read_records()

    with pytest.raises(_EpisodeFailure) as raised:
        _dispatch(
            environment, AdapterCall("ask_user", {"fact": "access_consent_confirmed"})
        )

    assert raised.value.failure_event["kind"] == "wrong_channel"
    assert environment.steps[-1].action == "read_records"


def test_a_scaffold_that_cannot_express_the_protocol_is_a_scaffold_failure(
    cube: Cube, tmp_path: Any
) -> None:
    payload = scaffold_payload()
    payload["actions"] = [
        action for action in payload["actions"] if action["name"] != "ask_user"
    ]
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    result = run_episode(
        variant=cube.cell_variant("S0_P0"),
        scaffold=load_scaffold(path),
        adapter=_adapter(),
        limits=LIMITS,
    )

    assert result.outcome == OUTCOME_SCAFFOLD_FAILURE
    assert result.error_class == "incomplete_scaffold"
    assert "ask_user" in (result.error_detail or "")
    assert result.trajectory.steps == ()


def test_a_scaffold_missing_an_argument_fails_before_the_adapter_is_asked_anything(
    cube: Cube, tmp_path: Any
) -> None:
    """A named contract failure, not a ``KeyError`` from the first dispatch.

    The scaffold declares ``ask_user`` but not the ``fact`` the loop always
    sends with it. Detecting that by name, up front, is what makes it a scaffold
    failure the operator can act on; without the check the loop reads a
    parameter that is not there and reports whatever exception falls out, from
    inside a turn that should never have been started.
    """
    payload = scaffold_payload()
    for action in payload["actions"]:
        if action["name"] == "ask_user":
            action["parameters"] = []
    path = write_scaffold(tmp_path / "scaffold.json", payload)
    asked: list[Any] = []

    def script(request: TurnRequest) -> AdapterCall:  # pragma: no cover - never run
        asked.append(request)
        return policy_following_script(request)

    result = run_episode(
        variant=cube.cell_variant("S0_P0"),
        scaffold=load_scaffold(path),
        adapter=_adapter(script),
        limits=LIMITS,
    )

    assert result.outcome == OUTCOME_SCAFFOLD_FAILURE
    assert result.error_class == "incomplete_scaffold"
    assert "fact" in (result.error_detail or "")
    assert asked == []
    assert result.turns_used == 0
    assert result.messages_used == 0


@pytest.mark.parametrize("argument", ["secondary_reason_codes", "evidence_refs"])
@pytest.mark.parametrize(
    "value",
    [
        ("CODE",),  # a tuple is not a JSON array
        ["CODE", 7],  # one element of the wrong type is enough
        "CODE",  # a bare string is a sequence of characters, not of codes
        b"CODE",  # and so is a bytestring
    ],
)
def test_an_array_argument_is_checked_by_container_and_by_element(
    cube: Cube, argument: str, value: Any
) -> None:
    """Both list arguments of the terminal call are the decision's justification.

    Nothing downstream re-checks them: the environment stores what it is handed
    and the evaluator grades it. A string accepted here is silently spread into
    one reason code per character, and a tuple or a stray integer is recorded as
    evidence that no observation handle backs.
    """
    arguments = {
        "disposition": "STOP",
        "primary_reason_code": "CONTRACTUAL_AUTHORITY",
        "secondary_reason_codes": [],
        "evidence_refs": [],
        argument: value,
    }

    def script(request: TurnRequest) -> AdapterCall:
        return AdapterCall("complete_case", arguments)

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_arguments"
    assert result.trajectory.terminal_decision is None


def test_the_terminal_call_reaches_the_environment_with_every_argument_it_carried(
    cube: Cube,
) -> None:
    """The environment records the decision; the runner only passes it along.

    A dropped argument does not fail: :meth:`Environment.complete_case` defaults
    it to empty, so the episode succeeds and the recorded decision is missing a
    part of what the model actually said.
    """
    codes = ["CORROBORATING_CLAUSE", "OPERATOR_ESCALATION"]

    def script(request: TurnRequest) -> AdapterCall:
        call = policy_following_script(request)
        if call.action != "complete_case":
            return call
        return AdapterCall(
            call.action, {**call.arguments, "secondary_reason_codes": list(codes)}
        )

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_SUCCESS
    decision = result.trajectory.terminal_decision
    assert decision is not None
    assert decision.secondary_reason_codes == tuple(codes)
    # Step arguments are frozen on the way into the trajectory, so the recorded
    # list arrives as a tuple of the same codes.
    assert result.trajectory.steps[-1].arguments["secondary_reason_codes"] == tuple(codes)


def test_an_adapter_protocol_error_is_the_model_failing_not_the_integration(
    cube: Cube,
) -> None:
    """The one distinction the taxonomy exists to make.

    ``AdapterProtocolError`` is a well-behaved integration reporting that the
    model answered with something that is not one allowed call. Recording that
    as an adapter failure would make a model that cannot follow a schema
    indistinguishable from a flaky provider.
    """

    def script(request: TurnRequest) -> AdapterCall:
        raise AdapterProtocolError("bad provider payload")

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "protocol_error"
    assert "bad provider payload" in (result.error_detail or "")


@pytest.mark.parametrize(
    "answer", [None, "read_records", {"action": "read_records", "arguments": {}}]
)
def test_an_answer_that_is_not_one_call_is_a_model_protocol_failure(
    cube: Cube, answer: Any
) -> None:
    """A missing or mis-shaped answer is classified, not reached into.

    The response is checked for being one call before anything is read off it;
    inspecting it first turns ``None`` into an ``AttributeError`` recorded as a
    fault of the loop's own machinery. What the adapter did return is named in
    the detail, because that is the only record of what the integration handed
    back — a mapping that looks like a call is the shape a half-finished
    provider integration produces.
    """

    def script(request: TurnRequest) -> Any:
        return answer

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_response"
    assert type(answer).__name__ in (result.error_detail or "")


@pytest.mark.parametrize("arguments", [[], ["not", "a", "mapping"], {1: "x"}])
def test_arguments_that_are_not_a_string_keyed_mapping_are_refused_as_such(
    cube: Cube, arguments: Any
) -> None:
    """The shape of the argument container is checked before its contents.

    Every later check indexes the arguments by name. Reaching them through
    something that is not a string-keyed mapping either misreports the fault or,
    for a container with nothing to disagree about, fails inside the schema
    check as a runner failure.
    """

    def script(request: TurnRequest) -> AdapterCall:
        return AdapterCall("read_records", arguments)

    result = _run(cube.cell_variant("S0_P0"), script)

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.error_class == "malformed_arguments"
    assert result.error_detail


def test_an_evaluator_crash_is_an_evaluator_failure(
    cube: Cube, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocked at the seam: only the evaluator call is replaced, not the loop."""

    def explode(variant: Any, trajectory: Any) -> Any:
        raise ZeroDivisionError("predicate blew up")

    monkeypatch.setattr("boundarybench.runner.evaluate", explode)

    result = _run(cube.cell_variant("S0_P0"))

    assert result.outcome == OUTCOME_EVALUATOR_FAILURE
    assert result.error_class == "ZeroDivisionError"
    assert result.evaluation is None
    # The episode itself still ran, so its trajectory is preserved intact.
    assert result.trajectory.terminal_decision is not None
