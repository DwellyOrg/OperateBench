"""Limits, deadlines, usage accounting and the failure taxonomy's remaining holes.

An episode that does not finish inside its budget, or whose cost is not measured,
must not be recorded as a clean success. The tests exercise adversarial boundary,
accounting and failure-classification cases for that invariant.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from boundarybench.adapter import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCall,
    AdapterUsage,
    ScriptedTestAdapter,
    TurnDeadline,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.compiler import Cube
from boundarybench.environment import Environment
from boundarybench.ledger import (
    OUTCOME_ADAPTER_FAILURE,
    OUTCOME_LIMIT_EXHAUSTED,
    OUTCOME_RUNNER_FAILURE,
    OUTCOME_SCAFFOLD_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
)
from boundarybench.runmanifest import RunLimits
from boundarybench.runner import run_episode
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from tests.scaffolds import scaffold_payload, write_scaffold

LIMITS = RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0)


def _run(variant: Any, adapter: Any, **kwargs: Any) -> Any:
    return run_episode(
        variant=variant,
        scaffold=kwargs.pop("scaffold", load_scaffold(STANDARD_SCAFFOLD)),
        adapter=adapter,
        limits=kwargs.pop("limits", LIMITS),
        **kwargs,
    )


def _scripted(script: Any) -> ScriptedTestAdapter:
    return ScriptedTestAdapter(
        script=script, identity=identity_for_test_double(), settings={}
    )


class ManualClock:
    """A clock that only moves when a test says so.

    Counting ticks would couple every timeout test to how many times the loop
    happens to read the clock, which is an implementation detail. Advancing it
    explicitly — from inside the adapter, where a real call would burn the time —
    tests the behaviour instead.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- the message limit -------------------------------------------------------


def test_message_exhaustion_is_distinct_from_turn_exhaustion(cube: Cube) -> None:
    """A turn is a request/response pair, so a turn costs two messages.

    Counting messages separately matters because a provider bills and rate-limits
    per message, and a scaffold that emits several messages per turn would run
    past a turn budget that still looked generous.
    """
    adapter = _scripted(lambda request: AdapterCall("read_records", {}))

    result = _run(
        cube.cell_variant("S0_P0"),
        adapter,
        limits=RunLimits(max_turns=99, max_messages=5, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_LIMIT_EXHAUSTED
    assert result.error_class == "max_messages"
    # Five messages buys two complete turns; the third would need a sixth.
    assert result.turns_used == 2
    assert result.messages_used == 4
    assert len(result.trajectory.steps) == 2


def test_turn_exhaustion_is_still_reported_as_turn_exhaustion(cube: Cube) -> None:
    adapter = _scripted(lambda request: AdapterCall("read_records", {}))

    result = _run(
        cube.cell_variant("S0_P0"),
        adapter,
        limits=RunLimits(max_turns=3, max_messages=999, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_LIMIT_EXHAUSTED
    assert result.error_class == "max_turns"
    assert result.turns_used == 3


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "8"])
def test_a_message_limit_must_be_a_positive_integer(value: Any) -> None:
    from boundarybench.runmanifest import RunManifestFormatError, build_run_manifest
    from boundarybench.scaffold import load_scaffold as _load
    from boundarybench.suite import validate_suite
    from tests.conftest import SUITE_MANIFEST

    with pytest.raises(RunManifestFormatError) as excinfo:
        build_run_manifest(
            suite=validate_suite(SUITE_MANIFEST),
            scaffold=_load(STANDARD_SCAFFOLD),
            provider="test-double",
            model="scripted-test-double",
            implementation="scripted_test_double",
            adapter_version=ADAPTER_CONTRACT_VERSION,
            adapter_settings={},
            trials=1,
            limits=RunLimits(
                max_turns=12, max_messages=value, episode_timeout_seconds=30.0
            ),
        )

    assert "max_messages must be a positive integer" in str(excinfo.value)


def test_a_two_message_budget_pays_for_exactly_one_turn(cube: Cube) -> None:
    """Two messages is a request and a response: one whole turn, and it is spent.

    A turn that cannot be paid for in full is never started, which is why the
    check is on affordability rather than on having budget left over. Refusing
    at exactly two would reject the smallest episode that can complete — one
    terminal decision — and record it as a limit failure it never hit.
    """
    adapter = _scripted(
        lambda request: AdapterCall(
            "complete_case",
            {
                "disposition": "STOP",
                "primary_reason_code": "CONTRACTUAL_AUTHORITY",
                "secondary_reason_codes": [],
                "evidence_refs": [],
            },
        )
    )

    result = _run(
        cube.cell_variant("S0_P0"),
        adapter,
        limits=RunLimits(max_turns=12, max_messages=2, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_SUCCESS
    assert result.turns_used == 1
    assert result.messages_used == 2
    assert result.trajectory.terminal_decision is not None


def test_the_agent_is_told_how_many_turns_are_left_at_every_turn(cube: Cube) -> None:
    """A budget the agent is shown wrongly is worse than one it is not shown.

    ``turns_remaining`` is the only signal an agent has that it should stop
    gathering and decide, so it counts the turns actually left: it starts at the
    whole budget and falls by exactly one per turn.
    """
    seen: list[Any] = []

    def script(request: TurnRequest) -> AdapterCall:
        seen.append(request.turns_remaining)
        return policy_following_script(request)

    result = _run(
        cube.cell_variant("S0_P0"),
        _scripted(script),
        limits=RunLimits(max_turns=6, max_messages=64, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_SUCCESS
    assert result.turns_used == 4
    assert seen == [6, 5, 4, 3]


# -- the deadline seam -------------------------------------------------------


def test_the_adapter_is_told_how_long_it_has_left(cube: Cube) -> None:
    """A provider adapter cannot honour a budget it is never shown.

    Polling the clock between turns is not a timeout: it cannot stop a network
    call that has already blocked. The contract therefore hands each call the
    remaining budget so a real integration can set its own socket timeout, and a
    cancellation probe it can check between streamed chunks.
    """
    seen: list[TurnDeadline] = []
    clock = ManualClock()

    class Recording:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            seen.append(deadline)
            clock.advance(1.0)  # every call costs a second
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            return AdapterUsage()

    result = _run(
        cube.cell_variant("S0_P0"),
        Recording(),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=10.0),
        clock=clock,
    )

    assert result.outcome == OUTCOME_SUCCESS
    assert [deadline.remaining_seconds for deadline in seen] == [10.0, 9.0, 8.0, 7.0]
    assert all(deadline.cancelled() is False for deadline in seen)


def test_an_adapter_that_overruns_the_deadline_is_a_timeout_not_a_success(
    cube: Cube,
) -> None:
    """A terminal call that crosses the budget must not succeed.

    The clock is checked *after* the adapter returns, so a call that blocked past
    the deadline is recorded as a timeout even though it produced a terminal
    decision. The decision arrived too late to count, and pretending otherwise
    would report a passing episode that a real deployment would have cut off.
    """
    clock = ManualClock()

    def slow_on_the_second_turn(request: TurnRequest) -> AdapterCall:
        if request.transcript:
            clock.advance(100.0)  # the call blocks far past the budget
        return policy_following_script(request)

    result = _run(
        cube.cell_variant("S0_P0"),
        _scripted(slow_on_the_second_turn),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=1.0),
        clock=clock,
    )

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"
    assert "adapter" in (result.error_detail or "")
    assert result.evaluation is None


def test_the_cancellation_probe_flips_exactly_when_the_budget_runs_out(
    cube: Cube,
) -> None:
    """The probe is polled *during* a call, so where it flips is the contract.

    An integration checks it between streamed chunks: flipping early cuts a call
    short while the episode still has budget, and flipping late leaves the
    adapter believing it may continue past the point at which the loop has
    already stopped counting its answer.
    """
    clock = ManualClock()
    seen: list[TurnDeadline] = []

    class Recording:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            seen.append(deadline)
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            return AdapterUsage()

    result = _run(
        cube.cell_variant("S0_P0"),
        Recording(),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=2.0),
        clock=clock,
    )
    assert result.outcome == OUTCOME_SUCCESS
    probe = seen[0]

    clock.now = 1.5  # half a second of budget left
    assert probe.cancelled() is False
    clock.now = 2.0  # exactly none
    assert probe.cancelled() is True
    clock.now = 2.5
    assert probe.cancelled() is True


def test_a_terminal_action_that_lands_exactly_on_the_deadline_is_a_timeout(
    cube: Cube, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatching the terminal call costs time too, and it is the last thing counted.

    The clock is checked once more after the environment has completed the case,
    so an episode whose terminal action finishes exactly as the budget runs out
    is a timeout — the same boundary ``cancelled()`` reports, rather than one
    that is a fraction more generous.
    """
    clock = ManualClock()
    complete_case = Environment.complete_case

    def slow_complete(self: Environment, **kwargs: Any) -> None:
        complete_case(self, **kwargs)
        clock.advance(1.0)  # the terminal action spends the last of the budget

    monkeypatch.setattr(Environment, "complete_case", slow_complete)

    result = _run(
        cube.cell_variant("S0_P0"),
        _scripted(policy_following_script),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=1.0),
        clock=clock,
    )

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"
    assert "terminal action" in (result.error_detail or "")
    assert result.evaluation is None
    # The case really was completed; what is refused is calling it done in time.
    assert result.trajectory.terminal_decision is not None


def test_an_episode_exactly_out_of_budget_does_not_start_another_turn(
    cube: Cube,
) -> None:
    """The budget is checked before a turn is started, at the same boundary.

    Time passes between one turn's answer and the next turn's request — reading
    usage, dispatching the action. An episode that arrives at the top of the
    loop with exactly nothing left has no budget to buy another round trip, so
    it stops there rather than spending a request it cannot pay for.
    """
    clock = ManualClock()

    class SlowMeter:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return AdapterCall("read_records", {})

        def last_usage(self) -> AdapterUsage:
            clock.advance(1.0)  # collecting the measurement is a round trip too
            return AdapterUsage()

    result = _run(
        cube.cell_variant("S0_P0"),
        SlowMeter(),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=1.0),
        clock=clock,
    )

    assert result.outcome == OUTCOME_TIMEOUT
    assert result.error_class == "episode_timeout"
    # The first turn is complete and counted; no second one is begun.
    assert result.turns_used == 1
    assert result.messages_used == 2
    assert len(result.trajectory.steps) == 1


# -- usage -------------------------------------------------------------------


def test_usage_is_accumulated_across_every_turn(cube: Cube) -> None:
    """One turn's numbers are not an episode's cost."""

    class Metered:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def __init__(self) -> None:
            self.turns = 0

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            self.turns += 1
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            return AdapterUsage(
                input_tokens=10,
                output_tokens=1,
                cost_usd=0.25,
                latency_seconds=0.5,
            )

    adapter = Metered()
    result = _run(cube.cell_variant("S0_P0"), adapter)

    assert result.outcome == OUTCOME_SUCCESS
    assert adapter.turns == 4
    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 4
    assert result.usage.cost_usd == pytest.approx(1.0)
    assert result.usage.latency_seconds == pytest.approx(2.0)


def test_an_unmeasured_adapter_reports_null_usage_rather_than_zero(
    cube: Cube,
) -> None:
    """Zero is a measurement. A fake has none, so every field stays null."""
    result = _run(cube.cell_variant("S0_P0"), _scripted(policy_following_script))

    assert result.usage.as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }


@pytest.mark.parametrize(
    "usage",
    [
        AdapterUsage(input_tokens=-1),
        AdapterUsage(output_tokens=True),  # bool is not a token count
        AdapterUsage(cost_usd=float("inf")),
        AdapterUsage(latency_seconds=float("nan")),
        AdapterUsage(input_tokens="many"),  # type: ignore[arg-type]
        # bool is an int subclass here too: a flag is not an amount of money,
        # and `True` would otherwise be summed into the episode as one dollar.
        AdapterUsage(cost_usd=True),  # type: ignore[arg-type]
        AdapterUsage(latency_seconds=True),  # type: ignore[arg-type]
    ],
)
def test_malformed_usage_is_an_adapter_failure_not_a_recorded_number(
    cube: Cube, usage: AdapterUsage
) -> None:
    class Bad:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            return usage

    result = _run(cube.cell_variant("S0_P0"), Bad())

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert result.error_class == "malformed_usage"


def test_a_later_unmeasured_turn_does_not_erase_an_earlier_measurement(
    cube: Cube,
) -> None:
    """``None`` is *not measured*, so it adds nothing — and removes nothing.

    A provider that reports tokens on one turn and nothing on the next has still
    spent what it reported. Letting the later silence overwrite the earlier
    number would report an episode as free because its last turn was unmetered.
    """

    class Intermittent:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def __init__(self) -> None:
            self.turns = 0

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            self.turns += 1
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            if self.turns > 1:
                return AdapterUsage()
            return AdapterUsage(
                input_tokens=10, output_tokens=2, cost_usd=0.5, latency_seconds=0.25
            )

    result = _run(cube.cell_variant("S0_P0"), Intermittent())

    assert result.outcome == OUTCOME_SUCCESS
    assert result.usage.as_dict() == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": 0.5,
        "latency_seconds": 0.25,
    }


def test_an_all_zero_measurement_is_a_measurement(cube: Cube) -> None:
    """Zero is what a cached or free turn costs, and it is not the same as null.

    Refusing it would make a provider that genuinely bills nothing look like a
    malformed integration, and the episode would be recorded as an adapter
    failure that never happened.
    """

    class Free:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            return AdapterUsage(
                input_tokens=0, output_tokens=0, cost_usd=0.0, latency_seconds=0.0
            )

    result = _run(cube.cell_variant("S0_P0"), Free())

    assert result.outcome == OUTCOME_SUCCESS
    assert result.usage.as_dict() == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_seconds": 0.0,
    }


def test_usage_that_is_not_an_adapter_usage_is_an_adapter_failure(cube: Cube) -> None:
    """A mapping that looks like usage is not usage, and is refused as such.

    Every field would then be read off something the contract never validated,
    and whatever it happened to contain would be persisted as this episode's
    measured cost.
    """

    class Loose:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return policy_following_script(request)

        def last_usage(self) -> Any:
            return {"input_tokens": 10, "output_tokens": 2}

    result = _run(cube.cell_variant("S0_P0"), Loose())

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert result.error_class == "malformed_usage"
    assert "dict" in (result.error_detail or "")


def test_a_usage_call_that_raises_is_classified(cube: Cube) -> None:
    """A raised ``RuntimeError`` must not escape unclassified."""

    class Exploding:
        identity = identity_for_test_double()
        settings: ClassVar[dict[str, Any]] = {}

        def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
            return policy_following_script(request)

        def last_usage(self) -> AdapterUsage:
            raise RuntimeError("usage endpoint unavailable")

    result = _run(cube.cell_variant("S0_P0"), Exploding())

    assert result.outcome == OUTCOME_ADAPTER_FAILURE
    assert result.error_class == "RuntimeError"
    # Reading usage is the same trust boundary as making the call, so the
    # exception's own message is redacted there too; the class and the boundary
    # it came out of are what the row records.
    assert "usage endpoint unavailable" not in (result.error_detail or "")
    assert "RuntimeError" in (result.error_detail or "")
    assert "last_usage" in (result.error_detail or "")


# -- scaffold contract -------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda action: action["parameters"][0].update(required=False),
            "must be required",
        ),
        (
            lambda action: action["parameters"][0].update(type="array_of_string"),
            "must be declared as",
        ),
        (
            lambda action: action["parameters"].append(
                {
                    "name": "temperature",
                    "type": "string",
                    "required": True,
                    "description": "An argument this loop never supplies.",
                }
            ),
            "requires argument(s)",
        ),
    ],
)
def test_a_scaffold_whose_parameters_do_not_match_the_protocol_is_refused(
    cube: Cube, tmp_path: Any, mutate: Any, expected: str
) -> None:
    """Names alone are not the contract: type and requiredness decide dispatch."""
    payload = scaffold_payload()
    payload["actions"] = _with_full_protocol(payload["actions"])
    for action in payload["actions"]:
        if action["name"] == "ask_user":
            mutate(action)
    path = write_scaffold(tmp_path / "scaffold.json", payload)

    result = _run(
        cube.cell_variant("S0_P0"),
        _scripted(policy_following_script),
        scaffold=load_scaffold(path),
    )

    assert result.outcome == OUTCOME_SCAFFOLD_FAILURE
    assert result.error_class == "incomplete_scaffold"
    assert expected in (result.error_detail or "")


def _with_full_protocol(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fixture scaffold omits two actions this loop needs; add them back."""
    return [
        *actions[:2],
        {
            "name": "call_tool",
            "description": "Call a tool for one named fact.",
            "terminal": False,
            "parameters": [
                {
                    "name": "fact",
                    "type": "string",
                    "required": True,
                    "description": "The fact key to request.",
                }
            ],
        },
        {
            "name": "perform_workflow_action",
            "description": "Perform one irreversible workflow action.",
            "terminal": False,
            "parameters": [
                {
                    "name": "action",
                    "type": "string",
                    "required": True,
                    "description": "The workflow action to perform.",
                }
            ],
        },
        *actions[2:],
    ]


# -- runner/infrastructure failures ------------------------------------------


def test_an_unexpected_internal_failure_is_a_runner_failure(
    cube: Cube, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OUTCOME_RUNNER_FAILURE`` existed but nothing ever emitted it."""

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise MemoryError("the loop's own machinery broke")

    monkeypatch.setattr("boundarybench.runner.build_turn_request", explode)

    result = _run(cube.cell_variant("S0_P0"), _scripted(policy_following_script))

    assert result.outcome == OUTCOME_RUNNER_FAILURE
    assert result.error_class == "MemoryError"
    assert "the loop's own machinery broke" in (result.error_detail or "")
