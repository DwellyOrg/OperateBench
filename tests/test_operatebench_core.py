"""Core primitives: simulated time, deterministic event order, outcome contracts.

Core is domain-neutral by construction, so everything here is asserted with
throwaway events and timestamps rather than with the Maintenance fixture. If a
test in this file needs to know what a quote is, Core has grown a dependency it
is not allowed to have.
"""

from __future__ import annotations

import pytest

from operatebench.core.clock import (
    ClockRewindError,
    MalformedTimestampError,
    SimulatedClock,
    format_timestamp,
    parse_timestamp,
    shift_minutes,
)
from operatebench.core.errors import OperateBenchError
from operatebench.core.evaluation import Dimension, Finding, OperationEvaluation
from operatebench.core.events import DuplicateEventError, Event, EventQueue
from operatebench.core.ledger import TrajectoryLedger
from operatebench.core.outcomes import Act, Complete, Wait, WaitContractError


def _event(event_id: str, at: str, sequence: int, event_type: str = "probe") -> Event:
    return Event(
        event_id=event_id,
        event_type=event_type,
        actor_id="actor_probe",
        at=at,
        sequence=sequence,
        payload={},
        triggers_agent=True,
    )


class TestTimestamps:
    def test_round_trip_is_exact(self) -> None:
        assert format_timestamp(parse_timestamp("2031-03-03T09:00:00Z", "t")) == (
            "2031-03-03T09:00:00Z"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "2031-03-03 09:00:00",
            "2031-03-03T09:00:00",
            "2031-03-03T09:00:00+00:00",
            "2031-13-03T09:00:00Z",
            "yesterday",
            "",
        ],
    )
    def test_malformed_timestamps_are_named_domain_errors(self, text: str) -> None:
        with pytest.raises(MalformedTimestampError) as excinfo:
            parse_timestamp(text, "scenario V9 event e01")
        assert "scenario V9 event e01" in str(excinfo.value)
        assert isinstance(excinfo.value, OperateBenchError)

    def test_shift_minutes_moves_forward_in_simulated_time(self) -> None:
        assert shift_minutes("2031-03-03T09:00:00Z", 90) == "2031-03-03T10:30:00Z"

    def test_shift_minutes_refuses_a_negative_offset(self) -> None:
        with pytest.raises(OperateBenchError):
            shift_minutes("2031-03-03T09:00:00Z", -1)


class TestSimulatedClock:
    def test_clock_starts_at_the_declared_instant(self) -> None:
        assert SimulatedClock("2031-03-03T09:00:00Z").now == "2031-03-03T09:00:00Z"

    def test_clock_advances_without_sleeping(self) -> None:
        clock = SimulatedClock("2031-03-03T09:00:00Z")
        clock.advance_to("2031-03-05T09:00:00Z")
        assert clock.now == "2031-03-05T09:00:00Z"
        assert clock.elapsed_minutes == 2880

    def test_clock_refuses_to_run_backwards(self) -> None:
        clock = SimulatedClock("2031-03-03T09:00:00Z")
        clock.advance_to("2031-03-05T09:00:00Z")
        with pytest.raises(ClockRewindError):
            clock.advance_to("2031-03-04T09:00:00Z")

    def test_advancing_to_the_same_instant_is_allowed(self) -> None:
        clock = SimulatedClock("2031-03-03T09:00:00Z")
        clock.advance_to("2031-03-03T09:00:00Z")
        assert clock.now == "2031-03-03T09:00:00Z"


class TestEventQueue:
    def test_same_time_events_are_ordered_by_authored_sequence(self) -> None:
        queue = EventQueue()
        queue.schedule(_event("e_b", "2031-03-03T09:00:00Z", 7))
        queue.schedule(_event("e_a", "2031-03-03T09:00:00Z", 2))
        assert [event.event_id for event in queue.drain()] == ["e_a", "e_b"]

    def test_same_time_same_sequence_falls_back_to_event_id(self) -> None:
        queue = EventQueue()
        queue.schedule(_event("e_z", "2031-03-03T09:00:00Z", 4))
        queue.schedule(_event("e_m", "2031-03-03T09:00:00Z", 4))
        assert [event.event_id for event in queue.drain()] == ["e_m", "e_z"]

    def test_insertion_order_does_not_change_delivery_order(self) -> None:
        forward = EventQueue()
        backward = EventQueue()
        authored = [
            _event("e1", "2031-03-04T09:00:00Z", 1),
            _event("e2", "2031-03-03T09:00:00Z", 2),
            _event("e3", "2031-03-03T09:00:00Z", 1),
        ]
        for event in authored:
            forward.schedule(event)
        for event in reversed(authored):
            backward.schedule(event)
        assert [event.event_id for event in forward.drain()] == [
            event.event_id for event in backward.drain()
        ]

    def test_duplicate_event_identity_is_refused_by_name(self) -> None:
        queue = EventQueue()
        queue.schedule(_event("e1", "2031-03-03T09:00:00Z", 1))
        with pytest.raises(DuplicateEventError) as excinfo:
            queue.schedule(_event("e1", "2031-03-04T09:00:00Z", 2))
        assert "e1" in str(excinfo.value)

    def test_cancelling_a_scheduled_event_removes_it(self) -> None:
        queue = EventQueue()
        queue.schedule(_event("e1", "2031-03-03T09:00:00Z", 1))
        queue.schedule(_event("e2", "2031-03-04T09:00:00Z", 1))
        assert queue.cancel("e1") is True
        assert queue.cancel("e_absent") is False
        assert [event.event_id for event in queue.drain()] == ["e2"]

    def test_pending_event_types_report_what_can_still_happen(self) -> None:
        queue = EventQueue()
        queue.schedule(_event("e1", "2031-03-03T09:00:00Z", 1, "supplier_reply"))
        assert queue.pending_event_types() == frozenset({"supplier_reply"})


class TestOutcomeContracts:
    def test_wait_without_wake_condition_or_fallback_is_refused(self) -> None:
        with pytest.raises(WaitContractError):
            Wait(reason="just waiting")

    def test_wait_with_only_a_fallback_is_a_declared_wake_up(self) -> None:
        wait = Wait(reason="deadline", fallback_after_minutes=60)
        assert wait.kind == "WAIT"

    def test_act_requires_an_action_type(self) -> None:
        with pytest.raises(OperateBenchError):
            Act(action_type="", payload={})

    def test_outcome_kinds_are_the_five_core_contracts(self) -> None:
        assert Act(action_type="a", payload={}).kind == "ACT"
        assert Wait(reason="r", wake_on=("x",)).kind == "WAIT"
        assert Complete(reason="done").kind == "COMPLETE"


class TestLedger:
    def test_records_are_indexed_in_append_order(self) -> None:
        ledger = TrajectoryLedger()
        ledger.append("2031-03-03T09:00:00Z", "event_observed", {"event_id": "e1"})
        ledger.append("2031-03-03T09:00:00Z", "agent_outcome", {"kind": "WAIT"})
        assert [record["index"] for record in ledger.as_list()] == [0, 1]
        assert [record["record_type"] for record in ledger.as_list()] == [
            "event_observed",
            "agent_outcome",
        ]

    def test_digest_depends_on_order(self) -> None:
        first = TrajectoryLedger()
        first.append("2031-03-03T09:00:00Z", "a", {})
        first.append("2031-03-03T09:00:00Z", "b", {})
        second = TrajectoryLedger()
        second.append("2031-03-03T09:00:00Z", "b", {})
        second.append("2031-03-03T09:00:00Z", "a", {})
        assert first.digest() != second.digest()

    def test_digest_is_stable_across_equal_ledgers(self) -> None:
        def build() -> TrajectoryLedger:
            ledger = TrajectoryLedger()
            ledger.append("2031-03-03T09:00:00Z", "a", {"z": 1, "y": 2})
            return ledger

        assert build().digest() == build().digest()


class TestEvaluationResult:
    def test_reliable_is_a_strict_conjunction(self) -> None:
        passing = Dimension(name="d1", ok=True)
        failing = Dimension(name="d2", ok=False, findings=(Finding("BAD", "no"),))
        good = OperationEvaluation(
            terminal_outcome="completed_successfully",
            legitimate_completion=True,
            dimensions=(passing,),
        )
        bad = OperationEvaluation(
            terminal_outcome="completed_successfully",
            legitimate_completion=True,
            dimensions=(passing, failing),
        )
        assert good.reliable is True
        assert bad.reliable is False
        assert bad.failed_dimensions == ("d2",)

    def test_illegitimate_completion_alone_defeats_reliability(self) -> None:
        result = OperationEvaluation(
            terminal_outcome="operation_deadlock",
            legitimate_completion=False,
            dimensions=(Dimension(name="d1", ok=True),),
        )
        assert result.reliable is False

    def test_finding_codes_are_reported(self) -> None:
        result = OperationEvaluation(
            terminal_outcome="operation_deadlock",
            legitimate_completion=False,
            dimensions=(
                Dimension(name="d1", ok=False, findings=(Finding("WAIT_NEVER", "x"),)),
            ),
        )
        assert result.finding_codes == ("WAIT_NEVER",)
        assert result.as_dict()["reliable"] is False
