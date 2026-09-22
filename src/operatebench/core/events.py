"""Events, their provenance, and the deterministic order they are delivered in.

Two properties matter more than anything else here.

**Identity is causal and total.** An event carries the actor that emitted it,
the instant it happens, the authored sequence number that breaks same-instant
ties and, where it exists, the record that caused it. Two events at the same
simulated instant are therefore ordered the same way no matter which order the
scenario, the reducers or a replay happened to schedule them in — ordering falls
back to the authored sequence and then to the event id, both of which are frozen
in the artefact.

**A trigger is not an audit record.** ``triggers_agent`` separates events that
wake the agent from events that only change the world and are recorded. An
attendance verification that the agent never sees still moves the operation;
conflating the two would make "did the agent react to this?" unanswerable.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from operatebench.core.clock import parse_timestamp
from operatebench.core.errors import DuplicateEventError

#: What an event's authority claim resolved to when it was delivered.
DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_REJECTED = "rejected"
DISPOSITION_AUDIT = "audit"
DISPOSITION_POST_TERMINAL = "post_terminal"

#: The verdict code every post-terminal delivery carries. A replay-final
#: operation asks no reducer for a verdict, so there is no domain code to
#: record: the arrival itself is the whole finding. The name is canonical
#: rather than incidental because the evaluator binds each delivery's verdict
#: code to the ledger row that records it, and the row for a post-terminal
#: delivery carries no code field to bind against.
VERDICT_AFTER_REPLAY_FINAL = "AFTER_REPLAY_FINAL"


@dataclass(frozen=True)
class Event:
    """One thing that happens in the world, with who made it happen.

    ``payload`` is the *semantic* event — frozen ground truth. Surface language
    is a fixture id inside it at most; nothing in this build derives business
    truth from prose.
    """

    event_id: str
    event_type: str
    actor_id: str
    at: str
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    triggers_agent: bool = True
    caused_by: str | None = None

    @property
    def order_key(self) -> tuple[int, int, str]:
        """The total order: simulated instant, then authored sequence, then id."""
        return (
            parse_timestamp(self.at, f"event {self.event_id}"),
            self.sequence,
            self.event_id,
        )

    def as_dict(self) -> dict[str, Any]:
        """A detached JSON projection for the artefact."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "at": self.at,
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "triggers_agent": self.triggers_agent,
            "caused_by": self.caused_by,
        }


class EventQueue:
    """The pending world. Deterministic, cancellable, and identity-checked.

    Kept as a list re-sorted on read rather than a heap: episodes here hold tens
    of events, cancellation by identity is a first-class operation (an approval
    that arrives early must cancel its own reminder), and a sort over the full
    total order is easier to prove correct than a heap with lazy deletion.
    """

    def __init__(self) -> None:
        self._pending: list[Event] = []
        self._seen: set[str] = set()

    def schedule(self, event: Event) -> None:
        """Queue an event, refusing an identity this episode has already used."""
        if event.event_id in self._seen:
            raise DuplicateEventError(
                f"event {event.event_id!r} was already scheduled in this episode; "
                "an event identity is used once so a duplicate cannot mutate twice"
            )
        self._seen.add(event.event_id)
        self._pending.append(event)

    def cancel(self, event_id: str) -> bool:
        """Drop a pending event. Returns whether anything was actually pending.

        The identity stays consumed: a cancelled reminder must not be able to
        come back under the same id later in the episode.
        """
        remaining = [event for event in self._pending if event.event_id != event_id]
        cancelled = len(remaining) != len(self._pending)
        self._pending = remaining
        return cancelled

    def peek_next(self) -> Event | None:
        """The next event without consuming it, so the engine can compare it to
        a declared wait deadline before deciding what happens first."""
        if not self._pending:
            return None
        self._pending.sort(key=lambda event: event.order_key)
        return self._pending[0]

    def pop_next(self) -> Event | None:
        """Take the next event in the total order, or ``None`` when the world is quiet."""
        if not self._pending:
            return None
        self._pending.sort(key=lambda event: event.order_key)
        return self._pending.pop(0)

    def drain(self) -> Iterator[Event]:
        """Every remaining event, in delivery order."""
        while True:
            event = self.pop_next()
            if event is None:
                return
            yield event

    def pending_event_types(self) -> frozenset[str]:
        """Which event types are still queued. Environment-internal, never a grade.

        This is authored-future truth: what the scenario's author decided would
        happen, which no agent can see and no operator running a real operation
        can ask. It is here for the environment's own bookkeeping and for tests
        that need to state what a queue holds. Nothing that judges an agent may
        read it — a WAIT is declared against the operation's published wake
        vocabulary, and time, not the queue, decides what ends it.
        """
        return frozenset(event.event_type for event in self._pending)

    def __len__(self) -> int:
        return len(self._pending)


__all__ = [
    "DISPOSITION_ACCEPTED",
    "DISPOSITION_AUDIT",
    "DISPOSITION_POST_TERMINAL",
    "DISPOSITION_REJECTED",
    "VERDICT_AFTER_REPLAY_FINAL",
    "DuplicateEventError",
    "Event",
    "EventQueue",
]
