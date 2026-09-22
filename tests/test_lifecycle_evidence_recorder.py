"""The evidence seam, driven at its edges rather than down its happy path.

:mod:`tests.test_provider_evidence_capture` runs whole episodes through the real
SDK and asserts the ledger a good run leaves. This module asks the three parts
of the seam what they do when the run is *not* good: a capture wired to two
transports or to none, an answer whose arguments never decoded, a decision
offered for the wrong turn, a device that stops accepting bytes half way
through, and a finaliser running on the way out of a run that already failed.

The rule every test here holds the recorder to is the same one: **evidence is
never invented and never silently dropped**. A turn that dispatched nothing
records no call; a turn that dispatched more requests than it recorded attempts
is refused rather than written; a row that cannot be made durable poisons the
journal and every later append becomes a no-op rather than a row sitting on an
offset nothing verified.

The amounts in these fixtures are the real guard's arithmetic over the real
request bound, so a row that would not survive the ledger's own accounting check
fails here rather than passing on invented numbers.

No socket is opened and no credential is read.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from operatebench.agents.evidence import (
    UNDECODABLE_ARGUMENTS,
    EvidenceBindingError,
    EvidenceError,
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCapture,
    WireCaptureTransport,
    check_response_binding,
    decision_projection,
    malformed_code,
    normalized_response_digest,
    utc_instant,
)
from operatebench.agents.model import MalformedModelOutcome
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ProviderTurnEvidence,
    ToolCall,
    content_digest,
)
from operatebench.core.outcomes import Wait, outcome_as_dict
from operatebench.execution_ledger import (
    ABORTED_CODE,
    EXCLUSION_BUDGET,
    EXCLUSION_DEADLINE,
    STOP_COMPLETED,
    TERMINAL_ABORTED,
    TERMINAL_EXCLUDED,
    LedgerStateError,
    LedgerWriteError,
    read_execution_ledger,
)
from operatebench.providers.cost import usd_text
from operatebench.providers.faults import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_PRE_DISPATCH_REFUSED,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
)
from operatebench.providers.openai_responses import request_token_bound
from operatebench.providers.telemetry import (
    AdapterUsage,
    ProviderAttempt,
    ProviderTelemetry,
)
from tests.execution_ledger_fixtures import (
    DIGEST_A,
    DIGEST_B,
    DIGEST_C,
    INSTANT,
    LATER,
    call,
    controls,
    faulted_call,
    provider_identity,
    rows_of,
)

MAX_OUTPUT_TOKENS = 4096
REPORTED_INPUT_TOKENS = 4211
REPORTED_OUTPUT_TOKENS = 118


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


def policy() -> LifecyclePricingPolicy:
    """The same pinned table the shared ledger fixtures price their rows at."""
    return LifecyclePricingPolicy(
        policy_id="operator_pinned_v1",
        input_usd_per_mtok=Decimal("1.25"),
        output_usd_per_mtok=Decimal("10.00"),
        rate_source=RATE_SOURCE_OPERATOR,
    )


def guard() -> LifecycleCostGuard:
    return LifecycleCostGuard(
        policy=policy(), cap_usd=Decimal("5.00"), max_output_tokens=MAX_OUTPUT_TOKENS
    )


def waited() -> Wait:
    """One well-formed decision, used wherever a turn has to produce something."""
    return Wait(reason="await_quote", fallback_after_minutes=60)


def instants() -> Any:
    """A clock that states the fixtures' two instants and then holds the later one."""
    remaining = [INSTANT, LATER]

    def now() -> str:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return now


def payload(turn: int = 0) -> dict[str, Any]:
    """A body of the shape the OpenAI adapter serialises, sized like a real one."""
    return {
        "model": "gpt-5.6-luna",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [{"role": "user", "content": "observation " * 40 + str(turn)}],
    }


def request_for(*, invocation: int = 1, turn: int = 0) -> ModelRequest:
    return ModelRequest(
        model="gpt-5.6-luna",
        protocol_version="operatebench.model.v3",
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tool_names=("act", "wait"),
        observation_digest_sha256=DIGEST_C,
        prompt_digest_sha256=DIGEST_B,
        invocation_index=invocation,
        turn_index=turn,
    )


def answer(tool_calls: tuple[ToolCall, ...] = ()) -> ModelResponse:
    return ModelResponse(
        model="gpt-5.6-luna",
        stop_reason=STOP_COMPLETED,
        tool_calls=tool_calls or (ToolCall("wait", {"reason": "await_quote"}),),
    )


def answered_turn(
    *,
    cost_guard: LifecycleCostGuard | None,
    invocation: int = 1,
    turn: int = 0,
    reservation: str | Decimal | None = None,
    settlement: str = ATTEMPT_SETTLEMENT_MEASURED,
) -> ProviderTurnEvidence:
    """One turn that answered, priced by the guard that authorised it.

    The reservation is the guard's own amount over this exact body's bound, so
    the row the recorder builds satisfies the ledger's independent accounting
    check rather than carrying a number chosen to make a test pass.
    """
    body = payload(turn)
    bound = request_token_bound(body)
    if reservation is None and cost_guard is not None:
        amount: Any = cost_guard.authorize(input_tokens_upper_bound=bound)
        if settlement == ATTEMPT_SETTLEMENT_MEASURED:
            cost_guard.settle(
                amount,
                input_tokens=REPORTED_INPUT_TOKENS,
                output_tokens=REPORTED_OUTPUT_TOKENS,
            )
        else:
            cost_guard.settle(amount, input_tokens=None, output_tokens=None)
        reservation = usd_text(amount)
    measured = settlement == ATTEMPT_SETTLEMENT_MEASURED
    return ProviderTurnEvidence(
        request=request_for(invocation=invocation, turn=turn),
        payload=body,
        settings={},
        telemetry=ProviderTelemetry(
            attempts=(
                ProviderAttempt(
                    index=1,
                    outcome=ATTEMPT_OUTCOME_RESPONSE,
                    latency_seconds=0.5,
                    response_received=True,
                    usage_reported=measured,
                    cost_reservation_usd=reservation,
                    cost_settlement=None if reservation is None else settlement,
                ),
            ),
            turn_latency_seconds=0.75,
            terminal_reason=TURN_END_RESPONSE,
        ),
        usage=AdapterUsage(
            input_tokens=REPORTED_INPUT_TOKENS if measured else None,
            output_tokens=REPORTED_OUTPUT_TOKENS if measured else None,
        ),
        response=answer(),
        response_id_digest_sha256=DIGEST_A,
    )


def faulted_turn(
    *, cost_guard: LifecycleCostGuard, turn: int = 0
) -> ProviderTurnEvidence:
    """One turn that reached the provider and met an outage it cannot retry past."""
    body = payload(turn)
    amount = cost_guard.authorize(input_tokens_upper_bound=request_token_bound(body))
    cost_guard.settle(amount, input_tokens=None, output_tokens=None)
    return ProviderTurnEvidence(
        request=request_for(turn=turn),
        payload=body,
        settings={},
        telemetry=ProviderTelemetry(
            attempts=(
                ProviderAttempt(
                    index=1,
                    outcome=ATTEMPT_OUTCOME_FAULT,
                    fault="provider_server_error",
                    http_status=500,
                    latency_seconds=0.5,
                    response_received=True,
                    cost_reservation_usd=usd_text(amount),
                    cost_settlement=ATTEMPT_SETTLEMENT_FORFEITED,
                ),
            ),
            turn_latency_seconds=0.75,
            terminal_reason=TURN_END_RETRIES_EXHAUSTED,
        ),
        usage=AdapterUsage(),
        fault="provider_server_error",
    )


def refused_turn(reason: str) -> ProviderTurnEvidence:
    """A turn a control refused before it could dispatch: no attempts at all."""
    return ProviderTurnEvidence(
        request=request_for(),
        payload=payload(),
        settings={},
        telemetry=ProviderTelemetry(attempts=(), terminal_reason=reason),
        usage=AdapterUsage(),
    )


def recorder_for(
    path: Path,
    *,
    wire: WireCaptureTransport | None = None,
    cost_guard: LifecycleCostGuard | None = None,
    ledger_controls: Any = None,
) -> ProviderEvidenceRecorder:
    return ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id="lettings.maintenance.v0_1",
            spec_digest_sha256=DIGEST_A,
            scenario_id="V1",
            agent_id="openai-gpt-5-6-luna",
        ),
        provider=provider_identity(),
        controls=ledger_controls or controls(),
        wire=wire,
        guard=cost_guard,
        now=instants(),
    )


def begun(path: Path, **kwargs: Any) -> ProviderEvidenceRecorder:
    recorder = recorder_for(path, **kwargs)
    recorder.begin(operation_instance_id="opinst_" + "1" * 32)
    return recorder


# -- the wire capture wraps one transport, once --------------------------------


def test_a_capture_already_wrapping_a_transport_refuses_a_second_one() -> None:
    """Re-pointing it mid-run would leave its records describing two destinations."""
    first = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    second = httpx.MockTransport(lambda request: httpx.Response(500, json={}))
    capture = WireCaptureTransport()
    capture.attach(first)
    with pytest.raises(EvidenceError):
        capture.attach(second)
    response = capture.handle_request(httpx.Request("POST", "https://api.example/v1/x"))
    assert response.status_code == 200
    assert capture.calls == 1


def test_a_capture_that_was_never_attached_forwards_nothing_and_records_nothing() -> None:
    capture = WireCaptureTransport()
    with pytest.raises(EvidenceError):
        capture.handle_request(httpx.Request("POST", "https://api.example/v1/x"))
    assert capture.captures == []
    assert capture.calls == 0


def test_taking_pending_captures_drains_them_in_order_and_leaves_the_count() -> None:
    """A policy that retried would dispatch twice; the row must see both."""
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    capture.handle_request(httpx.Request("POST", "https://api.example/v1/a"))
    capture.handle_request(httpx.Request("POST", "https://api.example/v1/b"))
    first = capture.take_pending()
    assert [item.url for item in first] == [
        "https://api.example/v1/a",
        "https://api.example/v1/b",
    ]
    assert capture.take_pending() == ()
    capture.handle_request(httpx.Request("POST", "https://api.example/v1/c"))
    assert [item.url for item in capture.take_pending()] == ["https://api.example/v1/c"]
    assert capture.calls == 3


def test_a_capture_states_the_body_digest_and_size_and_no_query_or_header() -> None:
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    capture.handle_request(
        httpx.Request(
            "POST",
            "https://api.example/v1/responses?key=leak",
            headers={"authorization": "Bearer sk-not-a-real-key"},
            content=b'{"a":1}',
        )
    )
    (recorded,) = capture.take_pending()
    assert isinstance(recorded, WireCapture)
    assert recorded.as_dict()["url"] == "https://api.example/v1/responses"
    assert recorded.as_dict()["byte_count"] == 7
    assert "leak" not in json.dumps(recorded.as_dict())
    assert "Bearer" not in json.dumps(recorded.as_dict())


# -- binding an answer ---------------------------------------------------------


def test_a_stop_classification_outside_the_closed_set_cannot_be_digested() -> None:
    with pytest.raises(EvidenceBindingError):
        normalized_response_digest(
            model="gpt-5.6-luna", stop_classification="stop", tool_calls=()
        )


def test_arguments_this_build_could_not_read_are_digested_as_a_marker() -> None:
    """The value is provider output that did not decode; quoting it is the failure."""
    unreadable = normalized_response_digest(
        model="gpt-5.6-luna",
        stop_classification=STOP_COMPLETED,
        tool_calls=(ToolCall("act", {"amount": float("nan")}),),
    )
    marker = normalized_response_digest(
        model="gpt-5.6-luna",
        stop_classification=STOP_COMPLETED,
        tool_calls=(ToolCall("act", UNDECODABLE_ARGUMENTS),),
    )
    readable = normalized_response_digest(
        model="gpt-5.6-luna",
        stop_classification=STOP_COMPLETED,
        tool_calls=(ToolCall("act", {"amount": 1}),),
    )
    assert unreadable == marker
    assert unreadable != readable


def test_two_answers_whose_arguments_both_failed_to_decode_are_not_told_apart() -> None:
    """Deliberate: neither value was read, so neither is what the row binds."""

    def digest(arguments: Any) -> str:
        return normalized_response_digest(
            model="gpt-5.6-luna",
            stop_classification=STOP_COMPLETED,
            tool_calls=(ToolCall("act", arguments),),
        )

    assert digest({"amount": float("nan")}) == digest({"cases": {object()}})


def test_a_call_that_never_received_an_answer_binds_no_decision() -> None:
    with pytest.raises(EvidenceBindingError):
        check_response_binding(
            faulted_call(0),
            model="gpt-5.6-luna",
            stop_classification=STOP_COMPLETED,
            tool_calls=(),
            decision_outcome={"kind": "WAIT"},
        )


def test_a_call_that_answered_but_records_no_decision_binds_none() -> None:
    """The answer verifies; there is simply no decision on the row to check."""
    outcome = {"kind": "WAIT", "reason": "await_quote"}
    tool_calls = (ToolCall("wait", {"reason": "await_quote"}),)
    digest = normalized_response_digest(
        model="gpt-5.6-luna",
        stop_classification=STOP_COMPLETED,
        tool_calls=tool_calls,
    )
    answered = call(
        0,
        attempts=(
            call(0)
            .attempts[0]
            .__class__(
                **{
                    **call(0).attempts[0].as_dict(),
                    "response_normalized_digest_sha256": digest,
                }
            ),
        ),
        decision=None,
    )
    with pytest.raises(EvidenceBindingError):
        check_response_binding(
            answered,
            model="gpt-5.6-luna",
            stop_classification=STOP_COMPLETED,
            tool_calls=tool_calls,
            decision_outcome=outcome,
        )


def test_a_decision_projection_is_the_digest_of_the_tape_s_own_projection() -> None:
    """The ledger's digest and the tape's contents are two readings of one thing."""
    outcome = waited()
    projected = decision_projection(outcome)
    assert projected.kind == "WAIT"
    assert projected.classification_code is None
    assert projected.decision_digest_sha256 == content_digest(outcome_as_dict(outcome))


def test_two_decisions_that_differ_are_never_bound_by_the_same_digest() -> None:
    other = Wait(reason="await_quote", fallback_after_minutes=61)
    assert (
        decision_projection(waited()).decision_digest_sha256
        != decision_projection(other).decision_digest_sha256
    )


def test_a_malformed_decision_carries_its_stable_code_and_no_provider_prose() -> None:
    outcome = MalformedModelOutcome("a tool nobody defined")
    assert malformed_code(outcome) == MalformedModelOutcome.code
    assert malformed_code(waited()) is None
    projected = decision_projection(outcome)
    assert projected.classification_code == MalformedModelOutcome.code
    assert "a tool nobody defined" not in json.dumps(projected.as_dict())


def test_an_object_that_is_no_decision_at_all_projects_as_a_refusal_code() -> None:
    """Recorded as the classification the engine will refuse it under."""
    projected = decision_projection(object())
    assert projected.classification_code == "MODEL_OUTPUT_UNCLASSIFIED"


def test_the_instant_source_states_the_one_format_this_build_writes() -> None:
    stamped = utc_instant()
    assert stamped.endswith("Z")
    assert len(stamped) == len(INSTANT)


# -- the recorder before it has begun ------------------------------------------


def test_a_recorder_that_has_not_begun_has_no_journal_to_offer(ledger_dir: Path) -> None:
    recorder = recorder_for(ledger_dir / "run.ndjson")
    with pytest.raises(LedgerStateError):
        _ = recorder.journal
    assert recorder.calls_recorded == 0
    assert recorder.remaining_calls == recorder.controls.max_provider_calls
    assert recorder.path == ledger_dir / "run.ndjson"
    assert not recorder.path.exists()


def test_provider_binding_takes_ledger_version_from_the_journal_it_binds(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = guard()
    recorder = begun(ledger_dir / "binding.ndjson", cost_guard=shared)
    recorder.observe_provider_turn(answered_turn(cost_guard=shared))
    recorder.bind_decision(waited(), invocation_index=1, turn_index=0)
    recorder.finalize_scored()
    monkeypatch.setattr(recorder.journal, "_ledger_version", 2)

    binding = recorder.provider_execution_binding()

    assert binding is not None
    assert binding["execution_ledger_version"] == recorder.journal.ledger_version == 2


def test_a_recorder_records_one_run_and_refuses_a_second_beginning(
    ledger_dir: Path,
) -> None:
    """A second beginning would append one run's calls to another run's header."""
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    first = recorder.journal.header.execution_run_id
    with pytest.raises(LedgerStateError):
        recorder.begin(operation_instance_id="opinst_" + "2" * 32)
    assert recorder.journal.header.execution_run_id == first
    assert len(rows_of(path)) == 1


def test_remaining_calls_falls_as_rows_are_recorded(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    before = recorder.remaining_calls
    recorder.observe_provider_turn(answered_turn(cost_guard=shared))
    recorder.bind_decision(waited(), invocation_index=1, turn_index=0)
    assert recorder.calls_recorded == 1
    assert recorder.remaining_calls == before - 1


# -- what a turn is allowed to have put on the wire ----------------------------


def test_more_requests_on_the_wire_than_recorded_attempts_is_refused_not_written(
    ledger_dir: Path,
) -> None:
    """A row states one request; evidence that cannot say which one is refused."""
    path = ledger_dir / "run.ndjson"
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    capture.handle_request(httpx.Request("POST", "https://api.example/v1/responses"))
    capture.handle_request(httpx.Request("POST", "https://api.example/v1/responses"))
    recorder = begun(path, wire=capture, cost_guard=guard())
    with pytest.raises(EvidenceError):
        recorder.observe_provider_turn(answered_turn(cost_guard=guard()))
    recorder.finalize_excluded("provider_transport")
    audit = read_execution_ledger(path)
    assert audit.calls == ()
    assert audit.status == TERMINAL_EXCLUDED


def test_a_second_turn_flushes_the_first_row_with_no_decision(
    ledger_dir: Path,
) -> None:
    """A turn the run never took a decision from is durable as the call it was."""
    path = ledger_dir / "run.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    recorder.observe_provider_turn(answered_turn(cost_guard=shared, turn=0))
    recorder.observe_provider_turn(answered_turn(cost_guard=shared, turn=1))
    recorder.bind_decision(waited(), invocation_index=1, turn_index=1)
    audit = read_execution_ledger(path)
    assert [c.decision is None for c in audit.calls] == [True, False]


# -- a turn no request left on -------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (TURN_END_COST_CAP_EXHAUSTED, EXCLUSION_BUDGET),
        (TURN_END_DEADLINE_EXCEEDED, EXCLUSION_DEADLINE),
    ],
)
def test_a_run_stopped_by_a_control_names_the_control_and_not_the_provider(
    ledger_dir: Path, reason: str, expected: str
) -> None:
    """No request was sent, so reporting a transport fault would be a fabrication."""
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    recorder.observe_provider_turn(refused_turn(reason))
    recorder.finalize_excluded("provider_transport")
    audit = read_execution_ledger(path)
    assert audit.calls == ()
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == expected


def test_a_stop_this_build_cannot_place_leaves_the_given_exclusion_code_alone(
    ledger_dir: Path,
) -> None:
    """Only the two control stops are rewritten; anything else is reported as given."""
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    recorder.observe_provider_turn(refused_turn(TURN_END_RETRIES_EXHAUSTED))
    recorder.finalize_excluded("provider_protocol")
    audit = read_execution_ledger(path)
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == "provider_protocol"


def test_a_turn_that_dispatched_clears_an_earlier_control_stop(
    ledger_dir: Path,
) -> None:
    """The control that refused turn one did not refuse the turn that then ran.

    The last turn reached the provider and met an outage, so the run reports the
    outage rather than the budget that stopped an earlier turn.
    """
    path = ledger_dir / "run.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    recorder.observe_provider_turn(refused_turn(TURN_END_COST_CAP_EXHAUSTED))
    recorder.observe_provider_turn(faulted_turn(cost_guard=shared))
    recorder.finalize_excluded("provider_transport")
    audit = read_execution_ledger(path)
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == "provider_transport"
    assert audit.calls[0].attempts[0].fault == "provider_server_error"


def test_a_later_pre_dispatch_refusal_aborts_but_preserves_the_prior_attempt(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "retry-then-local-refusal.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    attempted = faulted_turn(cost_guard=shared)
    recorder.observe_provider_turn(
        replace(
            attempted,
            telemetry=replace(
                attempted.telemetry,
                terminal_reason=TURN_END_PRE_DISPATCH_REFUSED,
            ),
        )
    )

    recorder.finalize_excluded("provider_transport")

    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_ABORTED
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == ABORTED_CODE
    assert len(audit.calls) == 1
    assert audit.calls[0].terminal_reason == TURN_END_PRE_DISPATCH_REFUSED
    assert len(audit.calls[0].attempts) == 1
    assert audit.calls[0].attempts[0].fault == "provider_server_error"
    assert audit.totals.provider_calls == 1
    assert audit.totals.attempts == 1
    assert audit.totals.fault_counts == {"provider_server_error": 1}


# -- a decision belongs to the call that produced it ---------------------------


def test_a_decision_offered_for_another_turn_is_refused_and_leaves_the_row_open(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    recorder.observe_provider_turn(answered_turn(cost_guard=shared, turn=0))
    with pytest.raises(EvidenceBindingError):
        recorder.bind_decision(waited(), invocation_index=1, turn_index=7)
    assert recorder.calls_recorded == 0
    recorder.bind_decision(waited(), invocation_index=1, turn_index=0)
    assert recorder.calls_recorded == 1


def test_a_decision_offered_with_no_open_call_is_not_an_error(
    ledger_dir: Path,
) -> None:
    """Nothing to close: a run that decided without dispatching writes no row."""
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    recorder.bind_decision(waited(), invocation_index=1, turn_index=0)
    assert recorder.calls_recorded == 0


def test_abandoning_with_no_open_call_writes_nothing(ledger_dir: Path) -> None:
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    recorder.abandon_open_call()
    assert len(rows_of(path)) == 1


# -- amounts on a row ----------------------------------------------------------


def test_a_forfeited_reservation_the_recorder_has_no_guard_for_is_read_off_the_attempt(
    ledger_dir: Path,
) -> None:
    """The fallback is the reservation the kernel already recorded, not a guess."""
    path = ledger_dir / "run.ndjson"
    pricing_guard = guard()
    evidence = answered_turn(
        cost_guard=pricing_guard, settlement=ATTEMPT_SETTLEMENT_FORFEITED
    )
    recorder = begun(path)  # no guard: the recorder has no amounts of its own
    recorder.observe_provider_turn(evidence)
    recorder.abandon_open_call()
    audit = read_execution_ledger(path)
    attempt = audit.calls[0].attempts[0]
    assert attempt.cost_settlement == ATTEMPT_SETTLEMENT_FORFEITED
    assert attempt.cost_forfeited_usd == attempt.cost_reservation_usd
    assert attempt.cost_measured_usd is None


def test_a_reservation_that_is_not_an_exact_decimal_string_is_refused_not_rounded(
    ledger_dir: Path,
) -> None:
    """A settled reservation this build cannot render is a refusal, not a float."""
    path = ledger_dir / "run.ndjson"
    evidence = ProviderTurnEvidence(
        request=request_for(),
        payload=payload(),
        settings={},
        telemetry=ProviderTelemetry(
            attempts=(
                ProviderAttempt(
                    index=1,
                    outcome=ATTEMPT_OUTCOME_FAULT,
                    fault="provider_timeout",
                    latency_seconds=0.5,
                    cost_reservation_usd=Decimal("0.0593425"),
                    cost_settlement=ATTEMPT_SETTLEMENT_FORFEITED,
                ),
            ),
            terminal_reason=TURN_END_RETRIES_EXHAUSTED,
        ),
        usage=AdapterUsage(),
    )
    recorder = begun(path)
    with pytest.raises(EvidenceError):
        recorder.observe_provider_turn(evidence)


# -- a device that stops accepting bytes ---------------------------------------


def _fail_writes_after(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    """Let the first ``rows`` rows through, then refuse every later write."""
    real_write = os.write
    state = {"written": 0}

    def failing_write(fd: int, data: Any) -> int:
        state["written"] += 1
        if state["written"] > rows:
            raise OSError(28, "no space left on device")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", failing_write)


def test_a_row_that_cannot_be_made_durable_stops_every_later_append(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is appended, finalised or abandoned through a poisoned writer."""
    path = ledger_dir / "run.ndjson"
    shared = guard()
    recorder = begun(path, cost_guard=shared)
    _fail_writes_after(monkeypatch, rows=0)
    recorder.observe_provider_turn(answered_turn(cost_guard=shared, turn=0))
    with pytest.raises(LedgerWriteError):
        recorder.bind_decision(waited(), invocation_index=1, turn_index=0)
    assert recorder.journal.poisoned
    # A later turn is dropped rather than written past an offset nothing verified.
    recorder.observe_provider_turn(answered_turn(cost_guard=shared, turn=1))
    recorder.bind_decision(waited(), invocation_index=1, turn_index=1)
    recorder.finalize_excluded("provider_transport")
    monkeypatch.undo()
    rows = rows_of(path)
    assert [row["row_kind"] for row in rows] == ["header"]


def test_a_finaliser_on_a_recorder_that_never_began_writes_nothing(
    ledger_dir: Path,
) -> None:
    """Finalisers run while an exception unwinds; one that raised would mask it."""
    path = ledger_dir / "run.ndjson"
    recorder = recorder_for(path)
    recorder.finalize_aborted()
    recorder.finalize_scored()
    recorder.finalize_excluded("provider_transport")
    assert not path.exists()


def test_a_second_finalisation_is_a_no_op_rather_than_a_second_ending(
    ledger_dir: Path,
) -> None:
    path = ledger_dir / "run.ndjson"
    recorder = begun(path)
    recorder.finalize_excluded("provider_budget")
    recorder.finalize_scored()
    recorder.finalize_aborted()
    rows = rows_of(path)
    assert [row["row_kind"] for row in rows] == ["header", "terminal"]
    assert rows[-1]["terminal"]["exclusion_code"] == EXCLUSION_BUDGET


# -- the agent seam ------------------------------------------------------------


def test_the_recording_agent_exposes_the_recorder_it_closes_rows_through(
    ledger_dir: Path,
) -> None:
    class Transport:
        def send(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError("no request is dispatched by this test")

    recorder = recorder_for(ledger_dir / "run.ndjson")
    agent = EvidenceRecordingModelAgent(
        Transport(),
        recorder=recorder,
        model="gpt-5.6-luna",
        agent_id="openai-gpt-5-6-luna",
    )
    assert agent.recorder is recorder
