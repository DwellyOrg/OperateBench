"""The episode engine: simulated time, event delivery, invocation, terminal drain.

The engine is the part that has to be boring and exactly right. It knows nothing
about maintenance; it knows about *shape*:

* the world advances by delivering the next event in the total order, never by
  sleeping — an eleven-day episode runs in milliseconds and runs the same way
  every time;
* the agent is invoked only when a *triggering* event arrives (or when its own
  declared fallback deadline comes due first), so an event that changes the
  world without waking anyone is still recorded and still counted;
* an outcome is a proposal — the domain decides, and proposal, refusal,
  accepted effect and post-commit side effect are four different records;
* a WAIT is checked against the *public contract* and never against the queue.
  Whether an event is actually coming is authored-future truth: it is the
  scenario author's private decision, the agent cannot see it, and no operator
  running a real operation can ask it. An engine that consulted it would refuse
  a correct, well-formed wait for a fact nobody could have known, and would
  grade the agent on the answer. So a wait is legal when its wake types are
  members of the operation's published vocabulary and the operation is still
  live — and *time* decides what happens next: a scheduled event, the declared
  fallback, the operational horizon, or deadlock;
* once the operation is replay-final, nothing resurrects it. Remaining events
  are still delivered — to the *record*, not to the state — because "a late
  message arrived and changed nothing" is evidence worth keeping.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from boundarybench.jsonsafe import canonical_json_bytes
from operatebench.core.clock import SimulatedClock, parse_timestamp, shift_minutes
from operatebench.core.events import (
    DISPOSITION_ACCEPTED,
    DISPOSITION_AUDIT,
    DISPOSITION_POST_TERMINAL,
    DISPOSITION_REJECTED,
    VERDICT_AFTER_REPLAY_FINAL,
    Event,
    EventQueue,
)
from operatebench.core.ledger import TrajectoryLedger
from operatebench.core.outcomes import (
    MALFORMED_OUTCOME_CODE,
    AgentOutcome,
    Ask,
    Complete,
    Escalate,
    Wait,
    outcome_as_dict,
    outcome_contract_problem,
)
from operatebench.core.protocol import (
    TRIGGER_ON_ACTION,
    TRIGGER_ON_EVENT,
    AgentObservation,
    EpisodePlan,
    OperationAgent,
    OperationDomain,
    PlannedEvent,
    Verdict,
    agent_identity,
)
from operatebench.core.read_contract import (
    ACTED_ON_CLAIM_WITHOUT_RECORD,
    ACTION_WITHOUT_RETRIEVAL,
    COMPLETE_OUTCOME_KEY,
    STALE_RETRIEVED_RECORD,
)
from operatebench.core.retrieval import (
    AUTHORITY_AUTHORITATIVE_VERIFICATION,
    INITIATED_BY_AGENT,
    INVALIDATION_EFFECT_ACCEPTED,
    INVALIDATION_WAKE,
    MAX_REQUESTS_PER_BATCH,
    RetrievalRequest,
    RetrieveBatch,
    ToolResult,
    canonical_requests,
    retrieval_batch_problem,
)
from operatebench.core.retrieval_evidence import detached_records


class ExecutionObserver(Protocol):
    """Core-owned observation seam; persistence belongs to the caller."""

    def receipt(self, row: Mapping[str, Any]) -> None: ...

    def state(self, state: Mapping[str, Any], *, at: str) -> None: ...


#: Execution statuses. These are not business terminals: they say the agent or
#: the episode ran out, not that the operation ended.
STATUS_DEADLOCK = "operation_deadlock"
STATUS_HORIZON = "operational_horizon_exhausted"

#: Why a wait was refused. Both are statements about the *public* contract — the
#: vocabulary the operation publishes, and whether the operation is still live —
#: so an agent could have avoided either from what it was told.
WAIT_UNKNOWN_WAKE_TYPE = "UNKNOWN_WAKE_EVENT_TYPE"
WAIT_AFTER_REPLAY_FINAL = "WAIT_AFTER_REPLAY_FINAL"

#: What ``last_rejection`` names when a *read* was refused. Not an action type:
#: nothing was proposed, so calling it one would put a refusal in the same slot
#: an agent reads to learn why its business proposal was turned down.
RETRIEVAL_REJECTION_SUBJECT = "retrieve"

#: Where engine-scheduled timers sort at an instant an authored event also
#: occupies. Above every authored sequence, so a timer never displaces the
#: authored event it was armed by.
TIMER_SEQUENCE_BASE = 1_000_000

#: A shared empty binding set. A module-level default that cannot be written to,
#: so "this effect established no identities" is one object rather than a fresh
#: mutable dict per call that something could quietly fill in.
MAPPING_PROXY_EMPTY: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class _EpisodeOutcomeProvenance:
    """Process-local authentication over one complete Engine result.

    This is an integrity capability, not a Python sandbox.  It distinguishes an
    outcome minted by the current process from serialized or caller-constructed
    evidence.  Code already executing inside this evaluator process can inspect
    private module state or use ``object.__setattr__``; underscore names are not
    access control, and that attacker is deliberately outside this boundary.
    """

    digest_sha256: str
    authentication: bytes


_EPISODE_OUTCOME_PROVENANCE_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class EpisodeOutcome:
    """Everything one episode produced, before evaluation."""

    status: str
    terminal_outcome: str | None
    replay_final: bool
    started_at: str
    ended_at: str
    simulated_minutes: int
    invocations: int
    final_state: Mapping[str, Any]
    final_state_digest_sha256: str
    trajectory: Sequence[Mapping[str, Any]]
    trajectory_digest_sha256: str
    events: Sequence[Mapping[str, Any]]
    _provenance: _EpisodeOutcomeProvenance | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "terminal_outcome": self.terminal_outcome,
            "replay_final": self.replay_final,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "simulated_minutes": self.simulated_minutes,
            "invocations": self.invocations,
            "final_state": dict(self.final_state),
            "final_state_digest_sha256": self.final_state_digest_sha256,
            "trajectory": [dict(record) for record in self.trajectory],
            "trajectory_digest_sha256": self.trajectory_digest_sha256,
            "events": [dict(event) for event in self.events],
        }


def _attest_episode_outcome(outcome: EpisodeOutcome) -> EpisodeOutcome:
    encoded = canonical_json_bytes(outcome.as_dict(), "operatebench episode outcome")
    object.__setattr__(
        outcome,
        "_provenance",
        _EpisodeOutcomeProvenance(
            hashlib.sha256(encoded).hexdigest(),
            hmac.digest(_EPISODE_OUTCOME_PROVENANCE_KEY, encoded, "sha256"),
        ),
    )
    return outcome


def has_canonical_episode_outcome_provenance(outcome: EpisodeOutcome) -> bool:
    """Whether this exact complete snapshot was minted by this Engine process.

    Dataclass type and caller-computed content digests are self-assertions, not
    provenance. This authentication binds terminal fields, time, invocation
    count, state, trajectory and delivered events together. A replaced outcome
    loses the non-init provenance field; mutation of nested content changes the
    authenticated bytes.  The capability protects this evaluator from
    serialized and caller-built evidence; it does not protect against arbitrary
    code already executing inside the evaluator process, where Python private
    names provide no access control.
    """
    provenance = outcome._provenance
    if provenance is None:
        return False
    try:
        encoded = canonical_json_bytes(outcome.as_dict(), "operatebench episode outcome")
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(
        provenance.digest_sha256, hashlib.sha256(encoded).hexdigest()
    ) and hmac.compare_digest(
        provenance.authentication,
        hmac.digest(_EPISODE_OUTCOME_PROVENANCE_KEY, encoded, "sha256"),
    )


@dataclass
class _Context:
    """The environment handle a domain gets while it reduces something.

    A concrete class rather than the protocol it satisfies: the protocol is what
    the *domain* is typed against, and keeping the implementation private stops
    a domain reaching into engine internals it should not know about.
    """

    engine: Engine

    @property
    def now(self) -> str:
        return self.engine.clock.now

    def record(self, record_type: str, payload: Mapping[str, Any]) -> None:
        self.engine.ledger.append(self.now, record_type, payload)

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None:
        self.engine.schedule_timer(
            event_id,
            event_type,
            actor_id,
            delay_minutes,
            payload,
            triggers_agent=triggers_agent,
        )

    def cancel_timer(self, event_id: str) -> bool:
        return self.engine.cancel_timer(event_id)

    def dispatch_fails(self, message_fixture_id: str) -> bool:
        return message_fixture_id in self.engine.dispatch_failures


class Engine:
    """Runs one episode of one operation with one agent."""

    def __init__(
        self,
        domain: OperationDomain,
        agent: OperationAgent,
        *,
        identity: Mapping[str, Any],
        dispatch_failures: frozenset[str] = frozenset(),
        partial_evidence: ExecutionObserver | None = None,
    ) -> None:
        self.domain = domain
        self.agent = agent
        self.identity = dict(identity)
        self.dispatch_failures = dispatch_failures

        self.plan: EpisodePlan = domain.build_plan()
        self.clock = SimulatedClock(self.plan.starts_at, context="episode start")
        self.queue = EventQueue()
        self.partial_evidence = partial_evidence
        self.ledger = TrajectoryLedger(
            None if partial_evidence is None else partial_evidence.receipt
        )
        self.state = domain.initial_state()
        self.context = _Context(self)

        self._start_seconds = parse_timestamp(self.plan.starts_at, "episode start")
        self._conditionals: list[PlannedEvent] = [
            event for event in self.plan.events if event.trigger is not None
        ]
        self._delivered: list[dict[str, Any]] = []
        self._timer_counter = 0
        self._proposal_counter = 0
        self._invocations = 0
        self._active_wait: Wait | None = None
        self._fallback_at: str | None = None
        self._last_rejection: dict[str, Any] | None = None
        self._halt: str | None = None
        #: What this invocation has been served. Never a run-lifetime cache: a
        #: wake empties it, and so does the agent's own accepted effect.
        self._served: dict[str, ToolResult] = {}
        self._retrieval_batches = 0
        #: Every tool this episode has ever had served, and the version it was
        #: served at. Deliberately *not* cleared by an invalidation: clearing it
        #: would make "read once and never re-read" indistinguishable from "never
        #: read", and those are two different failures with two different names.
        self._read_history: dict[str, str] = {}
        #: What the current invocation was woken by, as a claim. The strings the
        #: waking event carried are what an agent citing the event would cite, and
        #: whether that event is authoritative decides whether citing it is
        #: believing a record or believing a participant.
        self._claim_values: frozenset[str] = frozenset()
        self._trigger_is_authoritative = False
        #: Which agent call is in flight, so a refusal or an invalidation raised
        #: below the loop can say which turn it belongs to.
        self._turn_index = 0

        for event in self.plan.events:
            if event.at is not None:
                self.queue.schedule(_to_event(event, event.at))

    # ---------------------------------------------------------------- world

    def schedule_timer(
        self,
        event_id: str,
        event_type: str,
        actor_id: str,
        delay_minutes: int,
        payload: Mapping[str, Any],
        *,
        triggers_agent: bool = True,
    ) -> None:
        """Arm a timer. Its identity is the domain's, so it can disarm it later."""
        self._timer_counter += 1
        at = shift_minutes(self.clock.now, delay_minutes, context=f"timer {event_id}")
        self.queue.schedule(
            Event(
                event_id=event_id,
                event_type=event_type,
                actor_id=actor_id,
                at=at,
                sequence=TIMER_SEQUENCE_BASE + self._timer_counter,
                payload=dict(payload),
                triggers_agent=triggers_agent,
                caused_by="timer",
            )
        )
        self.ledger.append(
            self.clock.now,
            "timer_scheduled",
            {"event_id": event_id, "event_type": event_type, "due_at": at},
        )

    def cancel_timer(self, event_id: str) -> bool:
        cancelled = self.queue.cancel(event_id)
        if cancelled:
            self.ledger.append(self.clock.now, "timer_cancelled", {"event_id": event_id})
        return cancelled

    def _fire_conditionals(
        self,
        kind: str,
        type_name: str,
        cycle_id: str | None,
        bindings: Mapping[str, str] = MAPPING_PROXY_EMPTY,
    ) -> None:
        """Schedule every authored event whose cause has just happened.

        The identities the cause established are written into the event it
        causes, over whatever the scenario authored. An authored literal is a
        guess about what the operation will end up calling something; the
        binding is what it *is*. Only fields the authored payload already
        declares are rebound, so this cannot smuggle a field past the payload
        schema — it can only correct one the author had to name in advance.
        """
        remaining: list[PlannedEvent] = []
        for planned in self._conditionals:
            trigger = planned.trigger
            assert trigger is not None  # conditionals only, by construction
            if not trigger.matches(kind, type_name, cycle_id):
                remaining.append(planned)
                continue
            at = shift_minutes(
                self.clock.now,
                planned.delay_minutes,
                context=f"authored event {planned.event_id}",
            )
            self.queue.schedule(
                _to_event(
                    planned,
                    at,
                    caused_by=f"{kind}:{type_name}",
                    bindings=bindings,
                )
            )
        self._conditionals = remaining

    # ------------------------------------------------------------- episode

    def run(self) -> EpisodeOutcome:
        """Deliver the world until the operation ends or the episode runs out."""
        # The run's own identity names the scenario, the spec digest and the
        # authored expectation; the agent is handed the projection of it that
        # names the operation and this instance of it. Same separation as the
        # observation, one level up, and for the same reason: an identity the
        # agent could index the answer with is the answer.
        self.agent.begin_episode(agent_identity(self.identity))
        self.ledger.append(self.clock.now, "episode_ended", {"phase": "started"})
        # The record above opens the trajectory with the episode's start instant;
        # the closing record below states how it ended.
        while self._halt is None:
            upcoming = self.queue.peek_next()
            if self._fallback_due_before(upcoming):
                if self._beyond_horizon(self._fallback_at):
                    self._halt = STATUS_HORIZON
                    break
                fallback_at = self._fallback_at
                assert fallback_at is not None
                self.clock.advance_to(fallback_at, context="wait fallback")
                self.ledger.append(
                    self.clock.now, "wait_declared", {"fallback_fired": True}
                )
                self._clear_wait()
                if self._out_of_invocations():
                    break
                self._invoke(None)
                continue

            if upcoming is None:
                if self._active_wait is not None:
                    self._exhaust_horizon_under_wait()
                break
            if self._beyond_horizon(upcoming.at):
                self._halt = STATUS_HORIZON
                break

            event = self.queue.pop_next()
            assert event is not None
            self.clock.advance_to(event.at, context=f"event {event.event_id}")
            self._deliver(event)

        return self._finish()

    def _exhaust_horizon_under_wait(self) -> None:
        """The world went quiet under a standing wait. Run out the horizon.

        The alternative is what this replaces: stopping at the instant the
        author's last event happened to sit at and calling the wait unreachable,
        which is the benchmark announcing at declaration time that the event was
        never coming. Nobody knew that then, and the operation did not end then.

        So the operation is carried to its horizon and the record says exactly
        that: a declared wait was still standing when the operational horizon ran
        out. It is a fact about how the episode ended, written when it ended, and
        it is what an evaluator grades a wait that never resolved on.
        """
        wait = self._active_wait
        assert wait is not None
        horizon_at = shift_minutes(
            self.plan.starts_at, self.plan.horizon_minutes, context="operational horizon"
        )
        if parse_timestamp(horizon_at, "operational horizon") > parse_timestamp(
            self.clock.now, "operational horizon"
        ):
            self.clock.advance_to(horizon_at, context="operational horizon")
        self.ledger.append(
            self.clock.now,
            "wait_unresolved_at_horizon",
            {
                "reason": wait.reason,
                "wake_on": list(wait.wake_on),
                "horizon_at": horizon_at,
            },
        )
        self._halt = STATUS_HORIZON

    def _fallback_due_before(self, upcoming: Event | None) -> bool:
        if self._fallback_at is None:
            return False
        if upcoming is None:
            return True
        return parse_timestamp(self._fallback_at, "wait fallback") < parse_timestamp(
            upcoming.at, f"event {upcoming.event_id}"
        )

    def _beyond_horizon(self, moment: str | None) -> bool:
        if moment is None:
            return False
        limit = self._start_seconds + self.plan.horizon_minutes * 60
        return parse_timestamp(moment, "horizon check") > limit

    def _deliver(self, event: Event) -> None:
        if self.domain.is_replay_final(self.state):
            # A replay-final operation cannot be resurrected. The event is still
            # recorded: "this arrived and changed nothing" is the evidence.
            self.ledger.append(
                self.clock.now,
                "event_after_terminal",
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "actor_id": event.actor_id,
                    "mutating": False,
                },
            )
            self._note_delivery(
                event, DISPOSITION_POST_TERMINAL, VERDICT_AFTER_REPLAY_FINAL
            )
            return

        verdict = self.domain.reduce_event(self.state, event, self.context)
        if not verdict.accepted:
            self.ledger.append(
                self.clock.now,
                "event_rejected",
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "actor_id": event.actor_id,
                    "code": verdict.code,
                    "detail": verdict.detail,
                },
            )
            self._note_delivery(event, DISPOSITION_REJECTED, verdict.code)
            return

        record_type = "event_observed" if event.triggers_agent else "event_audit_only"
        self.ledger.append(
            self.clock.now,
            record_type,
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "code": verdict.code,
                "observational": verdict.observational,
            },
        )
        self._note_delivery(
            event,
            DISPOSITION_ACCEPTED if event.triggers_agent else DISPOSITION_AUDIT,
            verdict.code,
        )
        self._fire_conditionals(
            TRIGGER_ON_EVENT,
            event.event_type,
            verdict.cycle_id or _cycle_of(event),
            verdict.bindings,
        )

        if not event.triggers_agent:
            return
        if self.domain.is_replay_final(self.state):
            return
        self._check_declared_wake(event)
        if self._out_of_invocations():
            return
        self._invoke(event)

    def _check_declared_wake(self, event: Event) -> None:
        """Did the agent expect to be woken by this? A fallback-only wait expects
        nothing in particular, so only a declared expectation can be missed."""
        wait = self._active_wait
        if wait is None or not wait.wake_on:
            return
        if event.event_type not in wait.wake_on:
            self.ledger.append(
                self.clock.now,
                "wait_rejected",
                {
                    "code": "UNDECLARED_WAKE_EVENT",
                    "detail": (
                        f"woken by {event.event_type!r}, which the standing wait did "
                        f"not declare (declared: {list(wait.wake_on)})"
                    ),
                    "event_id": event.event_id,
                },
            )

    def _note_delivery(self, event: Event, disposition: str, code: str) -> None:
        row = event.as_dict()
        row["disposition"] = disposition
        row["verdict_code"] = code
        self._delivered.append(row)

    # ----------------------------------------------------------- invocation

    def _clear_wait(self) -> None:
        self._active_wait = None
        self._fallback_at = None

    def _out_of_invocations(self) -> bool:
        """Has the declared allowance already been spent? Asked before invoking.

        The counter moves inside :meth:`_invoke`, so a check at the top of the
        delivery loop would read the count as it stood *before* the previous
        invocation: at exactly the allowance such a check passes, the next event
        produces one more call, and a terminal on that call ends the episode one
        invocation past what the operation declared.

        Asked here, the last permitted call is the one that brings the count up to
        the allowance and the call after it is refused before the agent is reached
        at all, so what the record says about the allowance and what the agent was
        given are the same number.
        """
        if self._invocations < self.plan.max_invocations:
            return False
        self._refuse_invocation(
            "INVOCATION_LIMIT_EXCEEDED",
            f"the agent was invoked {self._invocations} times, the whole declared "
            f"allowance of {self.plan.max_invocations}, and the operation still "
            "requires another invocation to reach a terminal state",
        )
        return True

    def _invoke(self, trigger: Event | None) -> None:
        self._invocations += 1
        self._clear_wait()
        self._last_rejection = None
        envelope = self._event_envelope(trigger)
        authority = None if envelope is None else str(envelope["authority"])
        self._trigger_is_authoritative = bool(
            envelope is not None and envelope["is_authoritative"]
        )
        self._claim_values = _claim_values(trigger)
        # The authority the wake carried and the strings it carried are written
        # here, once, where the invocation opens. An evaluator handed only this
        # record has to be able to ask "did the agent cite what somebody told it,
        # rather than what the record holds?" without asking the engine.
        self.ledger.append(
            self.clock.now,
            "agent_invoked",
            {
                "invocation_index": self._invocations,
                "trigger_event_id": trigger.event_id if trigger else None,
                "trigger_event_type": trigger.event_type if trigger else None,
                "trigger_event_authority": authority,
                "trigger_is_authoritative": self._trigger_is_authoritative,
                "trigger_claim_values": sorted(self._claim_values),
            },
        )
        # A wake stales every derived fact. Whatever the agent read last time was
        # read from a document the world has since moved, and the record says so
        # before the agent is asked anything.
        self._invalidate_derived_context(INVALIDATION_WAKE, turn_index=0)
        # Two budgets, deliberately not one. ``max_turns_per_invocation`` bounds
        # *business* outcomes, exactly as it always did — a retrieval is not one,
        # so reading carefully never costs an agent the turns it needs to work.
        # The call bound is what stops an agent that only ever reads: it allows
        # every business turn, every batch the budget permits, and one call more
        # so the budget refusal itself can be seen and answered.
        call_limit = (
            self.plan.max_turns_per_invocation
            + self.plan.max_retrieval_batches_per_invocation
            + 1
        )
        business = 0
        turn = 0
        while business < self.plan.max_turns_per_invocation:
            if turn >= call_limit:
                self._refuse_invocation(
                    "INVOCATION_CALL_LIMIT_EXCEEDED",
                    f"the agent was called {turn} time(s) in one invocation — every "
                    f"business turn ({self.plan.max_turns_per_invocation}) plus every "
                    f"retrieval batch ({self.plan.max_retrieval_batches_per_invocation}) "
                    "plus one — without waiting, escalating or completing",
                )
                return
            self._turn_index = turn
            observation = self._observation(trigger, turn)
            if self.partial_evidence is not None:
                self.partial_evidence.state(
                    self.domain.canonical_state(self.state), at=self.clock.now
                )
            outcome = self.agent.decide(observation)
            turn += 1
            # The retrieval branch sits *before* the outcome contract, and that
            # is the point of ``RetrieveBatch`` not being in the union: the
            # boundary below keeps refusing everything it does not recognise,
            # and a read is recognised by type one frame earlier rather than by
            # widening the contract five outcomes are held to.
            if isinstance(outcome, RetrieveBatch):
                self._serve_batch(outcome, turn - 1)
                continue
            business += 1
            # The public boundary. Everything below indexes, serialises or
            # dispatches on what an agent returned, so what it returned is held
            # to the outcome contract first. A malformed outcome is a named,
            # non-mutating refusal the episode continues past under the ordinary
            # turn limit — never a programming exception escaping the run.
            problem = outcome_contract_problem(outcome)
            if problem is not None:
                self._refuse_outcome(turn - 1, problem)
                continue
            self.ledger.append(
                self.clock.now,
                "agent_outcome",
                {
                    "invocation_index": self._invocations,
                    "turn_index": turn - 1,
                    "outcome": outcome_as_dict(outcome),
                },
            )
            if self._handle(outcome):
                return
        self._refuse_invocation(
            "INVOCATION_TURN_LIMIT_EXCEEDED",
            f"the agent produced {self.plan.max_turns_per_invocation} outcomes "
            "in one invocation without waiting, escalating or completing",
        )

    def _refuse_invocation(self, code: str, detail: str) -> None:
        self.ledger.append(
            self.clock.now, "critical_violation", {"code": code, "detail": detail}
        )
        self._halt = STATUS_DEADLOCK

    # ------------------------------------------------------------- retrieval

    def _serve_batch(self, batch: RetrieveBatch, turn_index: int) -> None:
        """Serve one batch atomically, or refuse it by name having served nothing.

        The environment executes the read. The agent never holds a state handle
        and never calls the domain, which is what makes "this decision rested on
        these records" a fact the ledger states rather than a claim the agent
        makes about itself.
        """
        catalogue = self.domain.retrieval_catalogue()
        remaining = (
            self.plan.max_retrieval_batches_per_invocation - self._retrieval_batches
        )
        problem = retrieval_batch_problem(
            batch, catalogue=catalogue, batch_budget_remaining=remaining
        )
        if problem is not None:
            code, detail = problem
            self.ledger.append(
                self.clock.now,
                "retrieval_refused",
                {
                    "invocation_index": self._invocations,
                    "turn_index": turn_index,
                    "request_count": _request_count(batch),
                    "requested_tools": _known_tools(batch, catalogue),
                    "code": code,
                    "detail": detail,
                },
            )
            self._last_rejection = {
                "action_type": RETRIEVAL_REJECTION_SUBJECT,
                "code": code,
                "detail": detail,
            }
            return
        requests = canonical_requests(batch)
        batch_index = self._retrieval_batches
        self._retrieval_batches += 1
        # One instant for the whole batch. Results inside it cannot disagree
        # about when they were true, which is what "served atomically" means
        # when the clock is simulated and nothing takes any time.
        as_of = self.clock.now
        for position, result in enumerate(
            self.domain.serve_retrieval(self.state, requests, as_of)
        ):
            if result.ok:
                self._served[result.tool] = result
                self._read_history[result.tool] = result.record_version
            self.ledger.append(
                self.clock.now,
                "retrieval_served",
                {
                    "invocation_index": self._invocations,
                    "turn_index": turn_index,
                    "batch_index": batch_index,
                    "request_index": position,
                    "initiated_by": INITIATED_BY_AGENT,
                    "tool": result.tool,
                    "source": result.source,
                    "authority": result.authority,
                    "record_id": result.record_id,
                    "record_version": result.record_version,
                    "as_of": result.as_of,
                    "schema_id": result.schema_id,
                    "records": detached_records(
                        result.records,
                        f"retrieval_served row for {result.tool!r}",
                    ),
                    "ok": result.ok,
                },
            )

    def _invalidate_derived_context(
        self, cause: str, *, turn_index: int, action_type: str | None = None
    ) -> None:
        """Drop every served result and say why. The environment refreshes nothing.

        Both causes are the same fact seen from two sides: the document the agent
        was looking at has moved underneath it. What this build deliberately does
        *not* do is re-serve the same set at the new instant on the agent's
        behalf. That would be cheaper by one call per accepted effect and it
        would make the re-read the environment's decision rather than the
        agent's — and a re-read nobody chose is a re-read nothing can grade.
        """
        cleared = sorted(self._served)
        self._served = {}
        if cause == INVALIDATION_WAKE:
            self._retrieval_batches = 0
        self.ledger.append(
            self.clock.now,
            "derived_context_invalidated",
            {
                "invocation_index": self._invocations,
                "turn_index": turn_index,
                "cause": cause,
                "action_type": action_type,
                "cleared_tools": cleared,
            },
        )

    def _refuse_outcome(self, turn: int, detail: str) -> None:
        """Record an unreadable outcome and let the invocation carry on."""
        self.ledger.append(
            self.clock.now,
            "outcome_rejected",
            {
                "invocation_index": self._invocations,
                "turn_index": turn,
                "code": MALFORMED_OUTCOME_CODE,
                "detail": detail,
            },
        )
        self._last_rejection = {
            "action_type": "<malformed outcome>",
            "code": MALFORMED_OUTCOME_CODE,
            "detail": detail,
        }

    def _retrieval_affordance(self) -> dict[str, Any]:
        """What the agent may read, what it has read, and what it has left.

        ``served`` exists only inside the current invocation. It is rebuilt from
        the engine's own map on every observation rather than accumulated in the
        payload, so there is no path by which a result outlives the invalidation
        that dropped it.
        """
        return {
            "catalogue": dict(self.domain.retrieval_catalogue()),
            "record_contract": list(self.domain.retrieval_record_contract()),
            "served": {
                tool: result.as_dict() for tool, result in sorted(self._served.items())
            },
            "batch_budget_remaining": (
                self.plan.max_retrieval_batches_per_invocation - self._retrieval_batches
            ),
            "max_requests_per_batch": MAX_REQUESTS_PER_BATCH,
        }

    def _event_envelope(self, trigger: Event | None) -> dict[str, Any] | None:
        """The waking event as a claim, not as a fact.

        The authority class comes from the domain — Core does not know which of
        its actors are authoritative — and ``is_authoritative`` is derived from
        it rather than stated a second time.
        """
        if trigger is None:
            return None
        authority = self.domain.event_authority(trigger.actor_id, trigger.event_type)
        return {
            "event_id": trigger.event_id,
            "event_type": trigger.event_type,
            "actor_id": trigger.actor_id,
            "payload": dict(trigger.payload),
            "authority": authority,
            "is_authoritative": authority == AUTHORITY_AUTHORITATIVE_VERIFICATION,
        }

    def _observation(self, trigger: Event | None, turn: int) -> AgentObservation:
        return AgentObservation(
            now=self.clock.now,
            operation_id=str(self.identity.get("operation_id", "")),
            operation_instance_id=str(self.identity.get("operation_instance_id", "")),
            invocation_index=self._invocations,
            turn_index=turn,
            policy=self.domain.policy_view(),
            actors=self.domain.actor_roles(),
            message_fixture_ids=list(self.domain.message_fixture_ids()),
            action_schemas=self.domain.action_schemas(),
            wake_event_types=list(self.domain.wake_event_vocabulary()),
            last_rejection=self._last_rejection,
            phase=self.domain.coarse_phase(self.state),
            event=self._event_envelope(trigger),
            retrieval=self._retrieval_affordance(),
        )

    def _handle(self, outcome: AgentOutcome) -> bool:
        """Apply one outcome. Returns whether the invocation is over."""
        if isinstance(outcome, Wait):
            return self._declare_wait(outcome)
        if isinstance(outcome, Complete):
            return self._propose_terminal(outcome)
        if isinstance(outcome, Ask):
            accepted = self._propose_action(
                "send_message",
                {
                    "recipient_actor_id": outcome.recipient_actor_id,
                    "message_fixture_id": outcome.message_fixture_id,
                    "correlation_id": outcome.correlation_id,
                },
                (),
            )
            if not accepted:
                return False
            return self._declare_wait(outcome.wait)
        if isinstance(outcome, Escalate):
            # The escalation label selects the existing approval checkpoint the
            # exception is about.  The new exception checkpoint identity remains
            # environment-allocated; the selected authority is never reused as it.
            accepted = self._propose_action(
                "request_exception_resolution",
                {
                    "exception_type": outcome.exception_type,
                    "approval_checkpoint_id": outcome.checkpoint_id,
                    "deadline_after_minutes": outcome.deadline_after_minutes,
                },
                outcome.evidence_refs,
            )
            if not accepted:
                return False
            # An escalation is a wait by construction: the checkpoint it opened
            # is what wakes it, and the deadline it declared is the fallback.
            return self._declare_wait(
                Wait(
                    reason="awaiting the exception checkpoint it just opened",
                    wake_on=("operator_exception_resolved",),
                    fallback_after_minutes=outcome.deadline_after_minutes,
                )
            )
        self._propose_action(outcome.action_type, outcome.payload, outcome.evidence_refs)
        return False

    def _declare_wait(self, wait: Wait) -> bool:
        """Accept a wait on the public contract alone. Time decides the rest.

        The two checks here are the two an agent could have made itself from what
        it was told: the wake types are members of the operation's published
        vocabulary, which the observation carries in full, and the operation is
        still live. Neither reads the queue. A known event type with no scheduled
        instance is a legal wait — the agent has no way to know the difference,
        and a benchmark that refused it would be scoring the authored future
        rather than the decision.

        What ends it is time: a delivery, the declared fallback, the operational
        horizon or deadlock. That is a fact about the episode as it ran, recorded
        when it happens, rather than a prediction made at declaration.
        """
        vocabulary = tuple(self.domain.wake_event_vocabulary())
        unknown = sorted(set(wait.wake_on) - set(vocabulary))
        if unknown:
            # A public refusal: this operation has never heard of these types, so
            # nothing it does could ever deliver one and the observation said so
            # before the wait was declared. Non-fatal — the agent is told what it
            # named, what exists, and gets its next turn to say something legal.
            return self._refuse_wait(
                WAIT_UNKNOWN_WAKE_TYPE,
                f"WAIT named wake event type(s) {unknown}, which are not in this "
                f"operation's published wake vocabulary {list(vocabulary)}; a wait "
                "is declared against the vocabulary the observation discloses, "
                "never against what this episode happens to have queued",
            )
        if self.domain.is_replay_final(self.state):
            # Nothing can wake a finished operation, and this is not a guess
            # about the future: the operation is already over.
            self._refuse_wait(
                WAIT_AFTER_REPLAY_FINAL,
                "WAIT was declared after the operation became replay-final; a "
                "finished operation has nothing left to wake for",
            )
            return True
        self._active_wait = wait
        self._fallback_at = (
            shift_minutes(
                self.clock.now, wait.fallback_after_minutes, context="wait fallback"
            )
            if wait.fallback_after_minutes is not None
            else None
        )
        self.ledger.append(
            self.clock.now,
            "wait_declared",
            {
                "reason": wait.reason,
                "wake_on": list(wait.wake_on),
                "fallback_at": self._fallback_at,
            },
        )
        return True

    def _refuse_wait(self, code: str, detail: str) -> bool:
        """Record a refused wait and hand the reason back to the agent.

        Returns ``False``: the invocation is not over. A refusal the agent can
        act on — it named a type outside a vocabulary it was shown — belongs in
        ``last_rejection`` beside every other refusal, and the ordinary turn
        limit is what stops an agent that keeps repeating it.
        """
        self.ledger.append(
            self.clock.now, "wait_rejected", {"code": code, "detail": detail}
        )
        self._last_rejection = {"action_type": "wait", "code": code, "detail": detail}
        return False

    # ------------------------------------------------------------ read guard

    def _read_guard_problem(
        self, outcome_key: str, evidence_refs: Sequence[str]
    ) -> tuple[str, str] | None:
        """Why this outcome may not be proposed on what has been read, or ``None``.

        Non-mutating in every branch. Nothing here writes to the operation: the
        version check re-serves the required reads through the domain's own
        broker at the current instant and compares, which is a read of the record
        and not a change to it. The domain's own handler is never reached when
        this refuses, so a proposal that never should have been made cannot be
        half-executed on its way to being refused.

        Nothing here consults the queue. What a required read *would* say, and
        whether an event that would answer it is coming, are both authored-future
        truth; this asks only what the agent has actually been served and whether
        the record it was served still says what it said.

        The three names are ordered by what they mean, not by convenience. Citing
        a claim the current non-authoritative event carried, while the read that
        would settle it is missing, is the sharpest thing that can be said about
        a proposal — so it is said first. Then a required read that was performed
        earlier and is no longer fresh, which is a different failure from one that
        was never performed at all. Then absence.
        """
        contract = self.domain.read_requirements()
        required = contract.required_tools_for(outcome_key)
        if not required:
            return None
        catalogue = self.domain.retrieval_catalogue()
        absent = [tool for tool in required if tool not in self._served]
        if absent:
            cited = sorted({str(ref) for ref in evidence_refs} & set(self._claim_values))
            unverified = [
                tool
                for tool in absent
                if str((catalogue.get(tool) or {}).get("authority"))
                == AUTHORITY_AUTHORITATIVE_VERIFICATION
            ]
            if cited and unverified and not self._trigger_is_authoritative:
                return (
                    ACTED_ON_CLAIM_WITHOUT_RECORD,
                    f"{outcome_key} cites {cited}, which is what the current "
                    "non-authoritative event asserted, and the authoritative "
                    f"read(s) {unverified} that would settle it were not served in "
                    "this invocation; an event says someone asserted this and only "
                    "a retrieved authoritative record says the record holds it",
                )
            prior = [tool for tool in absent if tool in self._read_history]
            if prior:
                return (
                    STALE_RETRIEVED_RECORD,
                    f"{outcome_key} rests on read(s) {prior} that were served earlier "
                    "and have since been invalidated by a wake or by this agent's own "
                    "accepted effect; the record has moved and was not read again",
                )
            return (
                ACTION_WITHOUT_RETRIEVAL,
                f"{outcome_key} rests on read(s) {absent}, which this invocation "
                f"never asked for; the action schema publishes {list(required)} and "
                "an agent is told its read set rather than discovering it by refusal",
            )
        moved = [
            result.tool
            for result in self.domain.serve_retrieval(
                self.state,
                [RetrievalRequest(tool=tool) for tool in required],
                self.clock.now,
            )
            if not result.ok
            or self._served[result.tool].record_version != result.record_version
        ]
        if moved:
            return (
                STALE_RETRIEVED_RECORD,
                f"{outcome_key} rests on read(s) {sorted(moved)} whose record version "
                "moved between the read and the proposal; the exact version the agent "
                "was served is what it is held to, not the tool name alone",
            )
        return None

    def _propose_action(
        self,
        action_type: str,
        payload: Mapping[str, Any],
        evidence_refs: Sequence[str],
    ) -> bool:
        # Every proposal is named, and its refusal or its accepted effect carries
        # that name. Without it the record says only that *a* proposal and *an*
        # effect of the same type both happened, which is exactly enough to let a
        # rewritten proposal keep a real effect: the evaluator would look for any
        # earlier authority instead of the authority this effect actually rests
        # on. Deterministic and monotonic, so ordering is a property of the id.
        self._proposal_counter += 1
        proposal_id = f"proposal_{self._proposal_counter}"
        self.ledger.append(
            self.clock.now,
            "action_proposed",
            {
                "proposal_id": proposal_id,
                "action_type": action_type,
                "payload": dict(payload),
                "evidence_refs": list(evidence_refs),
            },
        )
        # The read guard sits between the proposal and the handler, so a
        # proposal the record does not support never reaches the code that would
        # execute it. Its refusal is an ordinary ``action_rejected`` row: a
        # refused proposal is a graded event, and it is graded in the same place
        # every other refused proposal is.
        guard = self._read_guard_problem(action_type, evidence_refs)
        verdict = (
            Verdict.refused(*guard)
            if guard is not None
            else self.domain.apply_action(
                self.state, action_type, payload, evidence_refs, self.context
            )
        )
        if not verdict.accepted:
            self.ledger.append(
                self.clock.now,
                "action_rejected",
                {
                    "proposal_id": proposal_id,
                    "action_type": action_type,
                    "code": verdict.code,
                    "detail": verdict.detail,
                },
            )
            self._last_rejection = {
                "action_type": action_type,
                "code": verdict.code,
                "detail": verdict.detail,
            }
            return False
        # ``last_rejection`` describes only the immediately preceding refused
        # business proposal. Once this proposal commits, carrying an older
        # refusal into the next observation would falsely describe this accepted
        # decision as rejected.
        self._last_rejection = None
        # ``bindings`` is written here, not left to be re-inferred. What an
        # effect established is a fact about the moment it committed, and a
        # reader that recovers it from the final state instead is reading a
        # document assembled after every later event had its say — which is how
        # a proposal rewritten to name an unrelated identity kept a real effect
        # and a correct-looking ending. Detached, so the row the ledger holds is
        # not the mapping the domain still has a reference to.
        self.ledger.append(
            self.clock.now,
            "effect_accepted",
            {
                "proposal_id": proposal_id,
                "action_type": action_type,
                "code": verdict.code,
                "cycle_id": verdict.cycle_id,
                "bindings": dict(verdict.bindings),
            },
        )
        # The agent's own accepted effect stales its reads for the same reason a
        # wake does: it is now holding a view of a document that changed because
        # of what it just did. Recorded immediately after the effect, so the
        # ordering in the ledger is the ordering in the world.
        self._invalidate_derived_context(
            INVALIDATION_EFFECT_ACCEPTED,
            turn_index=self._turn_index,
            action_type=action_type,
        )
        self._fire_conditionals(
            TRIGGER_ON_ACTION, action_type, verdict.cycle_id, verdict.bindings
        )
        return True

    def _propose_terminal(self, outcome: Complete) -> bool:
        self.ledger.append(
            self.clock.now,
            "terminal_proposed",
            {"reason": outcome.reason, "evidence_refs": list(outcome.evidence_refs)},
        )
        guard = self._read_guard_problem(COMPLETE_OUTCOME_KEY, outcome.evidence_refs)
        verdict = (
            Verdict.refused(*guard)
            if guard is not None
            else self.domain.apply_terminal(
                self.state, outcome.evidence_refs, self.context
            )
        )
        if not verdict.accepted:
            self.ledger.append(
                self.clock.now,
                "terminal_rejected",
                {"code": verdict.code, "detail": verdict.detail},
            )
            self._last_rejection = {
                "action_type": "complete",
                "code": verdict.code,
                "detail": verdict.detail,
            }
            return False
        self._last_rejection = None
        self._fire_conditionals(
            TRIGGER_ON_ACTION, "complete", verdict.cycle_id, verdict.bindings
        )
        return True

    # -------------------------------------------------------------- result

    def _finish(self) -> EpisodeOutcome:
        terminal = self.domain.terminal_outcome(self.state)
        if self._halt is not None:
            status = self._halt
        elif terminal is not None and self.domain.is_replay_final(self.state):
            status = terminal
        else:
            status = STATUS_HORIZON if len(self.queue) == 0 else STATUS_DEADLOCK
        self.ledger.append(
            self.clock.now,
            "episode_ended",
            {"phase": "ended", "status": status, "invocations": self._invocations},
        )
        final_state = dict(self.domain.canonical_state(self.state))
        return _attest_episode_outcome(
            EpisodeOutcome(
                status=status,
                terminal_outcome=terminal,
                replay_final=self.domain.is_replay_final(self.state),
                started_at=self.plan.starts_at,
                ended_at=self.clock.now,
                simulated_minutes=self.clock.elapsed_minutes,
                invocations=self._invocations,
                final_state=final_state,
                final_state_digest_sha256=hashlib.sha256(
                    canonical_json_bytes(final_state, "operatebench final state")
                ).hexdigest(),
                trajectory=self.ledger.as_list(),
                trajectory_digest_sha256=self.ledger.digest(),
                events=list(self._delivered),
            )
        )


def _to_event(
    planned: PlannedEvent,
    at: str,
    *,
    caused_by: str | None = None,
    bindings: Mapping[str, str] = MAPPING_PROXY_EMPTY,
) -> Event:
    payload = dict(planned.payload)
    for name, value in bindings.items():
        if name in payload:
            payload[name] = value
    return Event(
        event_id=planned.event_id,
        event_type=planned.event_type,
        actor_id=planned.actor_id,
        at=at,
        sequence=planned.sequence,
        payload=payload,
        triggers_agent=planned.triggers_agent,
        caused_by=caused_by,
    )


def _request_count(batch: RetrieveBatch) -> int:
    """How many requests a batch carried, whatever shape they were in."""
    requests = batch.requests
    if isinstance(requests, (str, bytes, Mapping)) or not isinstance(requests, Sequence):
        return 0
    return len(requests)


def _known_tools(
    batch: RetrieveBatch, catalogue: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """The catalogue members a refused batch named, canonically ordered.

    Only the members. A tool name an agent invented is provider-controlled text,
    and the record says how many of them there were rather than quoting one into
    durable evidence — the same rule the model boundary already applies to an
    unknown tool call.
    """
    requests = batch.requests
    if isinstance(requests, (str, bytes, Mapping)) or not isinstance(requests, Sequence):
        return []
    return sorted(
        {
            request.tool
            for request in requests
            if isinstance(request, RetrievalRequest)
            and isinstance(request.tool, str)
            and request.tool in catalogue
        }
    )


def _claim_values(event: Event | None) -> frozenset[str]:
    """Every string the waking event carried, as the set an agent could cite.

    The identity of the event and the string leaves of its payload. An agent that
    puts one of these in an ``evidence_refs`` is citing *what it was told*, and
    whether that is enough depends on who told it — which is what
    ``is_authoritative`` on the envelope answers.
    """
    if event is None:
        return frozenset()
    values = {event.event_id}
    for value in event.payload.values():
        if isinstance(value, str) and value:
            values.add(value)
    return frozenset(values)


def _cycle_of(event: Event) -> str | None:
    value = event.payload.get("cycle_id")
    return value if isinstance(value, str) else None


__all__ = [
    "STATUS_DEADLOCK",
    "STATUS_HORIZON",
    "TIMER_SEQUENCE_BASE",
    "WAIT_AFTER_REPLAY_FINAL",
    "WAIT_UNKNOWN_WAKE_TYPE",
    "Engine",
    "EpisodeOutcome",
    "has_canonical_episode_outcome_provenance",
]
