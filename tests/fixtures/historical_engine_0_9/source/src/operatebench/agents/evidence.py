"""Binding what a Lifecycle run sent, what came back, and what it decided.

Three seams, and each of them exists because a fact this build already measured
had nowhere to go.

**The bytes that left.** :class:`WireCaptureTransport` is an ``httpx`` transport
wrapper. It observes the method, the endpoint and the exact body bytes, and then
hands the *same request object* to the transport underneath it, unchanged. It
never reads a header and never records a query string, so a credential cannot
reach a capture: the only thing it keeps about a request besides its size is a
digest of its body. That digest is the missing half of the pair described in
:class:`~operatebench.execution_ledger.LedgerCall` — the identity digest says
what this build pinned, and this one says what the SDK actually serialised.

**What came back, exactly enough to bind it.**
:func:`normalized_response_digest` is over the provider's function-call *names
and full argument values*, canonically ordered, beside the response model and
this build's own stop classification. Not a shape summary: a shape digest is
equal for two answers that asked for different actions, which makes it useless
as a binding. It carries no prose — the message text beside a tool call is never
read here — and it is what makes "this row's answer is the answer this decision
came from" checkable rather than asserted.

**What the run took from it.** :class:`EvidenceRecordingModelAgent` closes each
call row with the decision digest, taken over the same projection the decision
tape stores, and bound to the same invocation and turn. A call that faulted has
no decision and its row says so; a decision with no answering call is refused by
the ledger schema.

The recorder is **optional everywhere**. A transport without one behaves exactly
as it did before this module existed, which is the only acceptable answer to
"does capturing evidence change the experiment".

Nothing here reads a credential or opens a socket of its own.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import httpx

from operatebench.agents.model import MalformedModelOutcome, ModelAgent
from operatebench.agents.playback import decision_record
from operatebench.agents.pricing import LifecycleCostGuard
from operatebench.agents.transport import (
    FAULT_BUDGET,
    ExecutionRecord,
    ModelResponse,
    ProviderFailure,
    ProviderTurnEvidence,
    ToolCall,
    content_digest,
)
from operatebench.core.clock import TIMESTAMP_FORMAT
from operatebench.core.errors import OperateBenchError
from operatebench.core.protocol import AgentObservation
from operatebench.execution_ledger import (
    ABORTED_CODE,
    EXCLUSION_BUDGET,
    EXCLUSION_DEADLINE,
    STOP_CLASSIFICATIONS,
    TERMINAL_ABORTED,
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    ExecutionLedgerJournal,
    LedgerAttempt,
    LedgerCall,
    LedgerControls,
    LedgerDecision,
    LedgerHeader,
    LedgerProviderIdentity,
    LedgerStateError,
    ledger_digest,
    new_execution_run_id,
)
from operatebench.jsonsafe import JsonSafetyError, canonical_json_text
from operatebench.providers.cost import request_input_token_bound, usd_text
from operatebench.providers.faults import (
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_PRE_DISPATCH_REFUSED,
)

#: What an argument value that this build could not read as JSON is digested as.
#: A marker rather than the value, because the value is provider output that did
#: not decode and quoting it is exactly what durable evidence must not do.
UNDECODABLE_ARGUMENTS = "arguments_undecodable"


class EvidenceError(OperateBenchError):
    """Provider evidence could not be recorded, or does not bind what it claims."""


class EvidenceBindingError(EvidenceError):
    """A recorded row does not bind the answer or the decision it is offered."""


def utc_instant() -> str:
    """The current instant, in the one format this build writes."""
    return datetime.now(UTC).strftime(TIMESTAMP_FORMAT)


class InstantSource(Protocol):
    def __call__(self) -> str: ...


# --------------------------------------------------------------- the wire seam


@dataclass(frozen=True)
class WireCapture:
    """One request as it left: method, endpoint, size and body digest.

    No header, no query string and no body. A credential travels in a header on
    this API, and the surest way to keep one out of durable evidence is to have
    no field it could be written to.
    """

    method: str
    url: str
    byte_count: int
    digest_sha256: str
    started_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "byte_count": self.byte_count,
            "digest_sha256": self.digest_sha256,
            "started_at": self.started_at,
        }


class WireCaptureTransport(httpx.BaseTransport):
    """An ``httpx`` transport that observes a request and forwards it unchanged.

    The forwarding is the contract. ``handle_request`` reads the request's body
    — which ``httpx`` caches, so the read is idempotent — digests it, and hands
    the *same object* to the transport underneath. There is no re-serialisation,
    no header added and no field touched, so the bytes the provider receives are
    the bytes the SDK produced whether or not this wrapper is installed.
    """

    def __init__(self, *, now: InstantSource = utc_instant) -> None:
        self._inner: httpx.BaseTransport | None = None
        self._now = now
        self.captures: list[WireCapture] = []
        self._cursor = 0

    def attach(self, inner: httpx.BaseTransport) -> None:
        """Name the transport this one wraps. Once."""
        if self._inner is not None:
            raise EvidenceError(
                "this wire capture already wraps a transport; re-pointing it mid-run "
                "would leave its records describing two different destinations"
            )
        self._inner = inner

    @property
    def calls(self) -> int:
        """How many requests passed through. An independent count of the budget."""
        return len(self.captures)

    def take_pending(self) -> tuple[WireCapture, ...]:
        """Every capture since the last take, in the order the requests were made.

        All of them rather than the next one. A turn dispatches once under this
        build's pinned retry policy, but a policy that allowed a second attempt
        would dispatch twice — and a recorder that took one capture per *turn*
        would then bind a row to the first request while recording the answer to
        the second. Draining is what keeps that from being an assumption.
        """
        pending = tuple(self.captures[self._cursor :])
        self._cursor = len(self.captures)
        return pending

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._inner is None:
            raise EvidenceError(
                "this wire capture was never attached to a transport, so there is "
                "nothing to forward the request to"
            )
        body = request.read()
        url = request.url
        self.captures.append(
            WireCapture(
                method=request.method,
                url=f"{url.scheme}://{url.host}{url.path}",
                byte_count=len(body),
                digest_sha256=hashlib.sha256(body).hexdigest(),
                started_at=self._now(),
            )
        )
        return self._inner.handle_request(request)


# ------------------------------------------------------------- response binding


def _argument_projection(arguments: Any) -> Any:
    """One tool call's arguments, canonically, or a marker if they never decoded."""
    try:
        canonical_json_text(arguments, "provider tool call arguments")
    except JsonSafetyError:
        return UNDECODABLE_ARGUMENTS
    return arguments


def normalized_response_digest(
    *,
    model: str,
    stop_classification: str,
    tool_calls: Sequence[ToolCall],
) -> str:
    """The digest that binds one provider answer to the decision it produced.

    Over the exact normalized function-call names and their **full argument
    values**, in the order the answer stated them, beside the response's own
    model and this build's stop classification. Object keys are canonically
    ordered by the encoding, so two answers that differ only in how they were
    typed hash the same and two answers that ask for different things never do.

    Deliberately not a shape summary. A digest over key *names* is equal for an
    action on one case and the same action on another, which would let a row
    claim to bind an answer it does not describe.
    """
    if stop_classification not in STOP_CLASSIFICATIONS:
        raise EvidenceBindingError(
            f"{stop_classification!r} is not a stop classification this build "
            f"writes; allowed: {list(STOP_CLASSIFICATIONS)}"
        )
    return ledger_digest(
        {
            "model": model,
            "stop_classification": stop_classification,
            "tool_calls": [
                {
                    "name": call.name if isinstance(call.name, str) else None,
                    "arguments": _argument_projection(call.arguments),
                }
                for call in tool_calls
            ],
        }
    )


def decision_projection(outcome: Any) -> LedgerDecision:
    """The decision this call produced, as the tape records it, digested.

    Taken over :func:`~operatebench.agents.playback.decision_record` — the same
    projection the artefact's decision tape stores — so the ledger's digest and
    the tape's contents are two readings of one statement rather than two
    statements that have to be kept in step.
    """
    record = decision_record(outcome)
    kind = str(record.get("kind", "UNKNOWN"))
    code = record.get("code")
    return LedgerDecision(
        kind=kind,
        decision_digest_sha256=content_digest(record),
        classification_code=None if code is None else str(code),
    )


def check_response_binding(
    call: LedgerCall,
    *,
    model: str,
    stop_classification: str,
    tool_calls: Sequence[ToolCall],
    decision_outcome: Mapping[str, Any],
) -> None:
    """Prove one recorded row binds this answer to this decision, or refuse.

    The reader's half of the binding. It recomputes both digests from values it
    was handed and compares them with the row's, so editing an argument, a
    digest or the decision beside it is a named refusal rather than a row that
    still looks well-formed.
    """
    answered = [attempt for attempt in call.attempts if attempt.response_received]
    if not answered:
        raise EvidenceBindingError(
            f"call {call.call_index} received no response, so there is no provider "
            "answer on it to bind a decision to"
        )
    expected = normalized_response_digest(
        model=model, stop_classification=stop_classification, tool_calls=tool_calls
    )
    if answered[-1].response_normalized_digest_sha256 != expected:
        raise EvidenceBindingError(
            f"call {call.call_index} states a normalized response digest that is not "
            "the digest of the answer offered here. The names and the argument "
            "values are both inside it, so an edited argument cannot pass as the "
            "answer that was recorded"
        )
    if call.decision is None:
        raise EvidenceBindingError(
            f"call {call.call_index} records no decision, so it binds none"
        )
    if call.decision.decision_digest_sha256 != content_digest(dict(decision_outcome)):
        raise EvidenceBindingError(
            f"call {call.call_index} states a decision digest that is not the digest "
            "of the decision offered here; the row and the tape describe two "
            "different answers"
        )


# ------------------------------------------------------------------ the recorder


@dataclass(frozen=True)
class EvidenceRunIdentity:
    """Which run this evidence is about, without naming what it decided."""

    operation_id: str
    spec_digest_sha256: str
    scenario_id: str
    agent_id: str
    agent_kind: str = "model"


class LifecycleEvidenceTransport(Protocol):
    """Provider identity exposed by any Lifecycle transport."""

    @property
    def provider(self) -> str: ...

    @property
    def api(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def request_mapping(self) -> str: ...

    @property
    def settings(self) -> Mapping[str, Any]: ...


def provider_identity_for(
    transport: LifecycleEvidenceTransport,
) -> LedgerProviderIdentity:
    """The provider identity of a live transport, read off the transport itself.

    Read rather than restated. A header that named the provider, model, mapping
    or SDK from a second list would describe the run somebody meant to configure.
    """
    settings = dict(transport.settings)
    configured_implementation = settings.get("implementation")
    if configured_implementation is not None and (
        type(configured_implementation) is not str or not configured_implementation
    ):
        raise EvidenceError(
            "a provider implementation stated in transport settings must be one "
            "non-empty exact string; it is part of the ledger identity and is not "
            "coerced from another type"
        )
    implementation = (
        type(transport).__name__
        if configured_implementation is None
        else configured_implementation
    )
    return LedgerProviderIdentity(
        provider=transport.provider,
        api=transport.api,
        model=transport.model,
        base_url=str(settings["base_url"]),
        implementation=implementation,
        request_mapping=transport.request_mapping,
        protocol_version=str(settings["protocol_version"]),
        sdk=str(settings["sdk"]),
        sdk_version=str(settings["sdk_version"]),
        settings=settings,
    )


class ProviderEvidenceRecorder:
    """Opens the journal, records every turn, and finalises exactly once.

    The order is the whole design. :meth:`begin` writes the header *before* the
    first request leaves. :meth:`observe_provider_turn` builds a call row the
    moment a turn ends — fault or answer — and :meth:`bind_decision` closes it
    with what the run took from it; a turn that raised is flushed by
    :meth:`abandon_open_call` with no decision, so the row survives even though
    the decision never existed. :meth:`finalize_scored`, :meth:`finalize_excluded`
    and :meth:`finalize_aborted` write the one terminal row.

    Every one of those appends, flushes and fsyncs before returning, so a
    process that dies at any point leaves a verifiable prefix and never a
    journal that reads as a cheap successful run.
    """

    def __init__(
        self,
        *,
        path: Path | str,
        identity: EvidenceRunIdentity,
        provider: LedgerProviderIdentity,
        controls: LedgerControls,
        wire: WireCaptureTransport | None = None,
        guard: LifecycleCostGuard | None = None,
        now: InstantSource = utc_instant,
        dir_fd: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._identity = identity
        self._provider = provider
        self._controls = controls
        self._wire = wire
        self._guard = guard
        self._now = now
        self._dir_fd = None if dir_fd is None else os.dup(dir_fd)
        self._journal: ExecutionLedgerJournal | None = None
        self._pending: dict[str, Any] | None = None
        self._control_stop: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def controls(self) -> LedgerControls:
        return self._controls

    @property
    def calls_recorded(self) -> int:
        return 0 if self._journal is None else self._journal.calls_recorded

    @property
    def remaining_calls(self) -> int:
        return max(0, self._controls.max_provider_calls - self.calls_recorded)

    @property
    def journal(self) -> ExecutionLedgerJournal:
        if self._journal is None:
            raise LedgerStateError(
                "this evidence recorder has not begun, so there is no journal. The "
                "header is written before the first request leaves, and a row "
                "recorded before it would describe a run nothing identifies"
            )
        return self._journal

    def begin(self, *, operation_instance_id: str) -> LedgerHeader:
        """Mint this execution's identity and write the header row."""
        if self._journal is not None:
            raise LedgerStateError(
                "this evidence recorder has already begun; one recorder records one "
                "run, so a second beginning would append one run's calls to another"
            )
        header = LedgerHeader(
            execution_run_id=new_execution_run_id(),
            operation_instance_id=operation_instance_id,
            operation_id=self._identity.operation_id,
            spec_digest_sha256=self._identity.spec_digest_sha256,
            scenario_id=self._identity.scenario_id,
            agent_id=self._identity.agent_id,
            agent_kind=self._identity.agent_kind,
            started_at=self._now(),
            provider=self._provider,
            controls=self._controls,
        )
        owned_dir_fd = self._dir_fd
        self._dir_fd = None
        try:
            self._journal = (
                ExecutionLedgerJournal.open(self._path, header)
                if owned_dir_fd is None
                else ExecutionLedgerJournal.open_at(
                    self._path, header, dir_fd=owned_dir_fd
                )
            )
        finally:
            if owned_dir_fd is not None:
                with suppress(OSError):
                    os.close(owned_dir_fd)
        return header

    def close(self) -> None:
        """Release every descriptor this recorder owns, exactly once."""
        owned_dir_fd = self._dir_fd
        self._dir_fd = None
        if owned_dir_fd is not None:
            os.close(owned_dir_fd)
        if self._journal is not None:
            self._journal.close()

    # -- per turn -----------------------------------------------------------

    def check_call_budget(self) -> None:
        """Refuse a call this run is no longer authorised to make.

        Enforced here as well as by :class:`~operatebench.agents.model.ModelAgent`
        and by whatever counts requests at the transport, because a budget with
        one enforcement point is a budget that stops being enforced the moment
        that point is bypassed.
        """
        if self.calls_recorded >= self._controls.max_provider_calls:
            raise EvidenceError(
                f"this run is authorised to make {self._controls.max_provider_calls} "
                f"provider call(s) and has made {self.calls_recorded}"
            )

    def observe_provider_turn(self, evidence: ProviderTurnEvidence) -> None:
        """Hold what one turn measured, pending the decision it produced."""
        journal = self.journal
        if self._pending is not None:
            self._flush(decision=None)
        dispatched = () if self._wire is None else self._wire.take_pending()
        attempted = len(evidence.telemetry.attempts)
        if len(dispatched) > max(attempted, 1):
            raise EvidenceError(
                f"this turn put {len(dispatched)} request(s) on the wire against "
                f"{attempted} recorded attempt(s). A call row states one request, so "
                "evidence that cannot say which request the recorded answer came "
                "from is refused rather than written"
            )
        # The last dispatch is the one whose answer this row records; the first
        # is where the turn began.
        capture = dispatched[-1] if dispatched else None
        ended_at = self._now()
        started_at = dispatched[0].started_at if dispatched else ended_at
        response = evidence.response
        attempts = self._attempts(evidence, response)
        self._control_stop = (
            evidence.telemetry.terminal_reason
            if evidence.telemetry.terminal_reason
            in (
                TURN_END_COST_CAP_EXHAUSTED,
                TURN_END_DEADLINE_EXCEEDED,
                TURN_END_PRE_DISPATCH_REFUSED,
            )
            else None
        )
        if not attempts:
            # A turn that was refused before it dispatched: the cost cap, token
            # ceiling, clock or final client admission stopped the request, so
            # nothing was sent and there is no provider call to record. Inventing a call
            # row here would put a request in the evidence that never happened;
            # what is kept instead is *which control* refused it, so the terminal
            # row can name it rather than reporting the outage it was not.
            return
        self._pending = {
            "call_index": journal.calls_recorded,
            "invocation_index": evidence.request.invocation_index,
            "turn_index": evidence.request.turn_index,
            "request_digest_sha256": evidence.request.request_digest_sha256,
            "prompt_digest_sha256": evidence.request.prompt_digest_sha256,
            "observation_digest_sha256": evidence.request.observation_digest_sha256,
            "payload_digest_sha256": content_digest(dict(evidence.payload)),
            "wire_request_digest_sha256": (
                None if capture is None else capture.digest_sha256
            ),
            "wire_request_bytes": None if capture is None else capture.byte_count,
            "wire_request_method": None if capture is None else capture.method,
            "wire_request_url": None if capture is None else capture.url,
            "input_token_upper_bound": request_input_token_bound(
                dict(evidence.payload), "Lifecycle evidence request"
            ),
            "started_at": started_at,
            "ended_at": ended_at,
            "turn_latency_seconds": evidence.telemetry.turn_latency_seconds,
            "terminal_reason": evidence.telemetry.terminal_reason,
            "attempts": attempts,
        }

    def _attempts(
        self, evidence: ProviderTurnEvidence, response: ModelResponse | None
    ) -> tuple[LedgerAttempt, ...]:
        """The kernel's own attempts, with this run's measured amounts on them.

        The reservation and its settlement are the kernel's; the *amounts* come
        from the cost guard, because the guard holds them as exact decimals and
        the usage object carries a float. The two never disagree about whether
        an attempt was measured or forfeited — that is the kernel's word — only
        about how precisely the amount can be written down.
        """
        usage = evidence.usage
        events = list(() if self._guard is None else self._guard.take_events())
        rows: list[LedgerAttempt] = []
        for attempt in evidence.telemetry.attempts:
            event = events.pop(0) if events else None
            measured = attempt.cost_settlement == "measured"
            forfeited = attempt.cost_settlement == "forfeited"
            answered = attempt.response_received and attempt.outcome == "response"
            rows.append(
                LedgerAttempt(
                    attempt_index=attempt.index,
                    outcome=attempt.outcome,
                    fault=attempt.fault,
                    http_status=attempt.http_status,
                    latency_seconds=attempt.latency_seconds,
                    response_received=attempt.response_received,
                    usage_reported=attempt.usage_reported,
                    cost_reservation_usd=attempt.cost_reservation_usd,
                    cost_settlement=attempt.cost_settlement,
                    cost_measured_usd=(
                        _amount(event, "measured_usd", attempt) if measured else None
                    ),
                    cost_forfeited_usd=(
                        _amount(event, "forfeited_usd", attempt) if forfeited else None
                    ),
                    input_tokens=usage.input_tokens if attempt.usage_reported else None,
                    output_tokens=(
                        usage.output_tokens if attempt.usage_reported else None
                    ),
                    response_model=(
                        response.model if answered and response is not None else None
                    ),
                    response_id_digest_sha256=(
                        evidence.response_id_digest_sha256 if answered else None
                    ),
                    response_stop_classification=(
                        response.stop_reason
                        if answered and response is not None
                        else None
                    ),
                    response_normalized_digest_sha256=(
                        normalized_response_digest(
                            model=response.model,
                            stop_classification=response.stop_reason,
                            tool_calls=response.tool_calls,
                        )
                        if answered and response is not None
                        else None
                    ),
                )
            )
        return tuple(rows)

    def bind_decision(
        self, outcome: Any, *, invocation_index: int, turn_index: int
    ) -> None:
        """Close the open call row with the decision the run took from it."""
        if self._pending is None:
            return
        if (self._pending["invocation_index"], self._pending["turn_index"]) != (
            invocation_index,
            turn_index,
        ):
            raise EvidenceBindingError(
                "a decision was offered for invocation "
                f"{invocation_index} turn {turn_index} against an open call for "
                f"invocation {self._pending['invocation_index']} turn "
                f"{self._pending['turn_index']}. A decision belongs to the call that "
                "produced it or to none"
            )
        self._flush(decision=decision_projection(outcome))

    def abandon_open_call(self) -> None:
        """Write the open call row with no decision, because there was none."""
        if self._pending is not None:
            self._flush(decision=None)

    def _flush(self, *, decision: LedgerDecision | None) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        journal = self.journal
        if journal.poisoned:
            # The writer failed to make an earlier row durable and is finished.
            # There is nowhere to put this row: appending past a rolled-back
            # offset is exactly what poisoning prevents, and the journal already
            # reads as the incomplete run it is.
            return
        journal.record_call(LedgerCall(**pending, decision=decision))

    # -- finalising ---------------------------------------------------------

    def finalize_scored(self) -> None:
        self._finalize(TERMINAL_SCORED, None)

    def finalize_excluded(self, exclusion_code: str) -> None:
        if self._control_stop == TURN_END_PRE_DISPATCH_REFUSED:
            self._finalize(TERMINAL_ABORTED, ABORTED_CODE)
            return
        self._finalize(TERMINAL_EXCLUDED, self._exclusion_for(exclusion_code))

    def _exclusion_for(self, exclusion_code: str) -> str:
        """The code this journal states, which is not always the one it was told.

        A run stopped by one of this build's *own* controls before a request was
        dispatched reaches the runner as an adapter failure, and the Lifecycle
        model boundary reports every adapter failure it cannot place as
        ``provider_transport``. Writing that here would say the provider failed
        on a turn no request was sent to — the same collapse this ledger exists
        to undo. Budget and deadline stops therefore name those controls. A local
        final client-admission refusal is handled by :meth:`finalize_excluded`
        as an abort rather than represented as a provider-style exclusion.
        """
        if self._control_stop is None:
            return exclusion_code
        if self._control_stop == TURN_END_COST_CAP_EXHAUSTED:
            return EXCLUSION_BUDGET
        if self._control_stop == TURN_END_DEADLINE_EXCEEDED:
            return EXCLUSION_DEADLINE
        return exclusion_code

    def finalize_aborted(self) -> None:
        self._finalize(TERMINAL_ABORTED, ABORTED_CODE)

    def provider_execution_binding(self) -> dict[str, Any] | None:
        """The artefact-7 binding to this journal, or ``None`` if there is none.

        ``None`` in every case where a binding would be a claim rather than a
        reference: no journal was opened, the writer was poisoned, the terminal
        row was never written, or the ending it states is not ``scored``. An
        excluded or aborted run has evidence — the journal — and no episode
        artefact, so it has nothing to bind and says so by returning nothing.

        Every value comes off the journal rather than off the run: the digest is
        the chain tip the writer made durable, the totals are the ones derived
        from the rows for the terminal row, and the call indexes are the rows'
        own. A binding assembled from what the *agent* believed would agree with
        the agent by construction, which is the one thing it must not do.
        """
        journal = self._journal
        if journal is None or journal.poisoned:
            return None
        terminal = journal.terminal
        if terminal is None or terminal.kind != TERMINAL_SCORED:
            return None
        digest = journal.ledger_digest_sha256
        if digest is None:  # pragma: no cover - a terminal row implies a tip
            return None
        header = journal.header
        totals = terminal.totals
        return {
            "execution_run_id": header.execution_run_id,
            "execution_ledger_version": journal.ledger_version,
            "execution_ledger_digest_sha256": digest,
            "provider": header.provider.provider,
            "api": header.provider.api,
            "model": header.provider.model,
            "settings_digest_sha256": header.provider.settings_digest_sha256,
            "pricing_digest_sha256": header.controls.pricing_digest_sha256,
            "provider_calls": totals.provider_calls,
            "attempts": totals.attempts,
            "measured_cost_usd": totals.measured_cost_usd,
            "forfeited_reservation_usd": totals.forfeited_reservation_usd,
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "decision_call_index": [
                call.call_index for call in journal.calls if call.decision is not None
            ],
        }

    def _finalize(self, kind: str, exclusion_code: str | None) -> None:
        journal = self._journal
        if journal is None or journal.finalized:
            return
        if journal.poisoned:
            # A finaliser runs on the way out of a run that is already failing,
            # often while an exception unwinds. A poisoned writer cannot state an
            # ending — the last thing it did was fail to make a row durable — so
            # this appends nothing and raises nothing, and the journal keeps the
            # verified prefix it has. A ledger with no terminal row reads as
            # incomplete, which is exactly what this run is.
            journal.close()
            return
        self.abandon_open_call()
        journal.finalize(kind, ended_at=self._now(), exclusion_code=exclusion_code)
        journal.close()


def _amount(event: Any, name: str, attempt: Any) -> str:
    """The exact amount the guard resolved, falling back to the reservation.

    The fallback is not a guess: it is the reservation the kernel already
    recorded on the attempt, which is what a forfeited attempt consumed. A run
    with no cost guard has no amounts at all, and states none.
    """
    value = None if event is None else getattr(event, name, None)
    if isinstance(value, Decimal):
        return usd_text(value)
    reservation = attempt.cost_reservation_usd
    if isinstance(reservation, str):
        return reservation
    raise EvidenceError(
        "an attempt states a settlement with no amount behind it; a settled "
        "reservation this build cannot render is refused rather than rounded"
    )


# ------------------------------------------------------------------- the agent


class EvidenceRecordingModelAgent(ModelAgent):
    """A model agent that closes each recorded call with its own decision.

    The seam is :meth:`decide` and nothing else: the request, the transport, the
    parse and the execution record are the base class's, unchanged. What this
    adds is the two ends of a call row — the budget check before it and the
    decision digest after it — bound to the same invocation and turn the request
    carried.

    A turn that raises is not silently dropped. The open row is flushed without
    a decision before the exception continues, so a faulted call is durable at
    the moment it failed rather than at whatever point a finaliser happens to
    run.
    """

    def __init__(
        self,
        transport: Any,
        *,
        recorder: ProviderEvidenceRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(transport, **kwargs)
        self._recorder = recorder

    @property
    def recorder(self) -> ProviderEvidenceRecorder:
        return self._recorder

    def decide(self, observation: AgentObservation) -> Any:
        try:
            self._recorder.check_call_budget()
        except EvidenceError as exc:
            raise ProviderFailure(
                self._budget_excluded(),
                f"the recorded call budget is spent; the "
                f"{self._recorder.calls_recorded} call(s) already made are retained "
                "and the run is excluded rather than completed",
            ) from exc
        try:
            decision = super().decide(observation)
        except BaseException:
            self._recorder.abandon_open_call()
            raise
        self._recorder.bind_decision(
            decision,
            invocation_index=observation.invocation_index,
            turn_index=observation.turn_index,
        )
        return decision

    def _budget_excluded(self) -> ExecutionRecord:
        base = self.execution_record()
        return ExecutionRecord(
            outcome_source=base.outcome_source,
            model=base.model,
            protocol_version=base.protocol_version,
            max_output_tokens=base.max_output_tokens,
            transport_calls=base.transport_calls,
            retry_count=base.retry_count,
            attempts=base.attempts,
            excluded=True,
            exclusion_code=FAULT_BUDGET,
            exclusion_detail=(
                f"execution stopped at the provider boundary ({FAULT_BUDGET})"
            ),
        )


def malformed_code(outcome: Any) -> str | None:
    """The classification code of a malformed decision, or ``None``."""
    return outcome.code if isinstance(outcome, MalformedModelOutcome) else None


__all__ = [
    "UNDECODABLE_ARGUMENTS",
    "EvidenceBindingError",
    "EvidenceError",
    "EvidenceRecordingModelAgent",
    "EvidenceRunIdentity",
    "InstantSource",
    "ProviderEvidenceRecorder",
    "WireCapture",
    "WireCaptureTransport",
    "check_response_binding",
    "decision_projection",
    "malformed_code",
    "normalized_response_digest",
    "provider_identity_for",
    "utc_instant",
]
