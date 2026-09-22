"""Provider execution evidence: what was sent, what came back, what it cost.

Every provider interaction in this module runs through the *real* OpenAI SDK
over an in-process ``httpx.MockTransport``. The request is serialised by the SDK
and the answer is parsed into real ``openai.types.responses`` models, so a claim
here about the bytes on the wire is a claim about the bytes the SDK produced.

No credential is read, no socket is opened, and no test in this module reaches a
provider. The API key handed to the client is a placeholder that never leaves
the process.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

import operatebench.agents.evidence as evidence_module
from operatebench.agents.evidence import (
    EvidenceBindingError,
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    check_response_binding,
    normalized_response_digest,
    provider_identity_for,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, ModelAgent
from operatebench.agents.openai_responses import (
    OpenAIResponsesTransport,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import (
    FAULT_BUDGET,
    ProviderFailure,
    ToolCall,
    content_digest,
)
from operatebench.artifact import (
    build_artifact,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.core.instance import new_operation_instance_id
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_ledger import (
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    LedgerChainError,
    LedgerControls,
    read_execution_ledger,
)
from operatebench.providers.cost import (
    CostCapExceededError,
    request_input_token_bound,
    usd_text,
)
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import run_episode
from tests.model_transport import PROVIDER_PROSE, observation_from_dict, tool_call_for
from tests.openai_transport import (
    FAKE_API_KEY,
    function_call_item,
    message_item,
    responses_body,
)

FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")
DEADLINE_SECONDS = 30.0
INPUT_TOKENS = 4211
OUTPUT_TOKENS = 118


def test_evidence_recorder_uses_the_provider_neutral_request_bound() -> None:
    assert evidence_module.request_input_token_bound is request_input_token_bound


#: Operator-supplied, pinned for offline evidence. Not a claim about any
#: vendor's current price list.
INPUT_RATE = Decimal("1.25")
OUTPUT_RATE = Decimal("10.00")


def policy() -> LifecyclePricingPolicy:
    return LifecyclePricingPolicy(
        policy_id="offline_operator_pinned_v1",
        input_usd_per_mtok=INPUT_RATE,
        output_usd_per_mtok=OUTPUT_RATE,
        rate_source=RATE_SOURCE_OPERATOR,
    )


def guard(cap: str = "5.00") -> LifecycleCostGuard:
    return LifecycleCostGuard(
        policy=policy(), cap_usd=Decimal(cap), max_output_tokens=MAX_OUTPUT_TOKENS
    )


def controls(*, max_calls: int = 60, cap: str = "5.00") -> LedgerControls:
    return LedgerControls(
        max_provider_calls=max_calls,
        max_attempts_per_call=1,
        expected_provider_calls=min(46, max_calls),
        token_hard_cap=1_650_300,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cost_cap_usd=cap,
        cost_cap_policy="refuse_before_dispatch",
        turn_deadline_seconds=DEADLINE_SECONDS,
        wall_clock_deadline_seconds=1800.0,
        pricing=policy().as_dict(),
    )


def fixed_clock(step: float = 0.25) -> Any:
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += step
        return ticks[0]

    return clock


class Wire:
    """The exact bytes the SDK put on the wire, captured by the mock itself."""

    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self.methods: list[str] = []
        self.tool_calls: list[ToolCall] = []

    def digests(self) -> list[str]:
        return [hashlib.sha256(body).hexdigest() for body in self.bodies]


def reference_handler(wire: Wire, *, faults: dict[int, Any] | None = None) -> Any:
    """Answers every turn with the shipped reference agent's own decision."""
    reference = build_agent("reference")
    faults = faults or {}
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        wire.bodies.append(request.content)
        wire.methods.append(request.method)
        planned = faults.get(state["calls"])
        if planned is not None:
            return planned
        prompt = json.loads(request.content)["input"][0]["content"]
        observation = observation_from_dict(json.loads(prompt)["observation"])
        call = tool_call_for(reference.decide(observation))
        wire.tool_calls.append(ToolCall(call.name, call.arguments))
        return httpx.Response(
            200,
            json=responses_body(
                [
                    function_call_item(call.name, json.dumps(call.arguments)),
                    message_item(PROVIDER_PROSE),
                ],
                model=GPT_5_6_LUNA_MODEL,
                input_tokens=INPUT_TOKENS,
                output_tokens=OUTPUT_TOKENS,
            ),
        )

    return handler


def client_for(
    handler: Any, *, capture: WireCaptureTransport | None = None
) -> openai.OpenAI:
    transport: httpx.BaseTransport = httpx.MockTransport(handler)
    if capture is not None:
        capture.attach(transport)
        transport = capture
    return openai.OpenAI(
        api_key=FAKE_API_KEY,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )


def transport_for(
    handler: Any,
    *,
    capture: WireCaptureTransport | None = None,
    cost_guard: LifecycleCostGuard | None = None,
    recorder: ProviderEvidenceRecorder | None = None,
) -> OpenAIResponsesTransport:
    return OpenAIResponsesTransport(
        model=GPT_5_6_LUNA_MODEL,
        client=client_for(handler, capture=capture),
        deadline_seconds=DEADLINE_SECONDS,
        clock=fixed_clock(),
        cost_guard=cost_guard,
        recorder=recorder,
    )


def recorder_for(
    path: Path,
    transport: OpenAIResponsesTransport,
    *,
    spec: Any,
    wire: WireCaptureTransport,
    cost_guard: LifecycleCostGuard,
    ledger_controls: LedgerControls | None = None,
    agent_id: str = "openai-gpt-5-6-luna",
) -> ProviderEvidenceRecorder:
    return ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id="V1",
            agent_id=agent_id,
        ),
        provider=provider_identity_for(transport),
        controls=ledger_controls or controls(),
        wire=wire,
        guard=cost_guard,
    )


@pytest.fixture()
def ledger_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    return directory


@pytest.fixture()
def spec() -> Any:
    return load_spec(FIXTURE)


# -- the wire is untouched ---------------------------------------------------


def _observations(spec: Any, count: int = 3) -> list[Any]:
    """The first real V1 observations, read back off the bytes that carried them.

    A run mints a fresh operation instance identity, and that identity travels
    inside the observation — so two *runs* cannot be compared byte for byte and
    nothing is proven by trying. Comparing two transports on the *same*
    observations is the question that matters: does installing the capture
    change what leaves.
    """
    wire = Wire()
    transport = transport_for(reference_handler(wire))
    agent = ModelAgent(transport, model=GPT_5_6_LUNA_MODEL, agent_id="a")
    run_episode(spec, "V1", "a", agent_factory=lambda: agent, agent_kind="model")
    prompts = [json.loads(body)["input"][0]["content"] for body in wire.bodies[:count]]
    return [
        observation_from_dict(json.loads(prompt)["observation"]) for prompt in prompts
    ]


def test_the_capture_forwards_the_identical_request_and_records_its_bytes(
    spec: Any,
) -> None:
    observations = _observations(spec)
    plain, captured = Wire(), Wire()
    without = transport_for(reference_handler(plain))
    bare = ModelAgent(without, model=GPT_5_6_LUNA_MODEL, agent_id="a")
    bare.begin_episode({"probe": True})

    capture = WireCaptureTransport()
    watched = transport_for(reference_handler(captured), capture=capture)
    observed = ModelAgent(watched, model=GPT_5_6_LUNA_MODEL, agent_id="a")
    observed.begin_episode({"probe": True})

    for observation in observations:
        bare.decide(observation)
        observed.decide(observation)

    assert len(plain.bodies) == len(observations)
    assert plain.bodies == captured.bodies
    assert [c.digest_sha256 for c in capture.captures] == captured.digests()
    assert [c.byte_count for c in capture.captures] == [
        len(body) for body in captured.bodies
    ]


def test_the_capture_records_no_header_and_no_query(spec: Any) -> None:
    wire = Wire()
    capture = WireCaptureTransport()
    transport = transport_for(reference_handler(wire), capture=capture)
    agent = ModelAgent(transport, model=GPT_5_6_LUNA_MODEL, agent_id="a")
    agent.begin_episode({"probe": True})
    agent.decide(_observations(spec, 1)[0])
    record = capture.captures[0]
    rendered = json.dumps(record.as_dict())
    assert "authorization" not in rendered.lower()
    assert FAKE_API_KEY not in rendered
    assert record.url == "https://api.openai.com/v1/responses"
    assert record.method == "POST"


# -- a whole successful run --------------------------------------------------


def _full_run(ledger_dir: Path, spec: Any) -> tuple[Path, Wire, Any]:
    path = ledger_dir / "execution.ndjson"
    wire = Wire()
    capture = WireCaptureTransport()
    cost_guard = guard()
    transport = transport_for(
        reference_handler(wire), capture=capture, cost_guard=cost_guard
    )
    recorder = recorder_for(
        path, transport, spec=spec, wire=capture, cost_guard=cost_guard
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id="openai-gpt-5-6-luna",
        recorder=recorder,
        max_transport_calls=60,
    )
    run = run_episode(
        spec,
        "V1",
        "openai-gpt-5-6-luna",
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    return path, wire, run


def test_a_full_v1_run_writes_a_scored_ledger_with_exact_totals(
    ledger_dir: Path, spec: Any
) -> None:
    path, wire, run = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_SCORED
    assert audit.scored is True
    assert audit.totals.provider_calls == 46
    assert audit.totals.attempts == 46
    assert audit.totals.input_tokens == 46 * INPUT_TOKENS
    assert audit.totals.output_tokens == 46 * OUTPUT_TOKENS
    assert audit.totals.measured_cost_usd == "0.2964125"
    assert audit.totals.forfeited_reservation_usd == "0"
    assert audit.totals.fault_counts == {}
    assert len(wire.bodies) == 46
    assert run.outcome.status == "completed_successfully"
    assert [c.call_index for c in audit.calls] == list(range(46))


def test_the_wire_digest_on_each_row_is_the_exact_body_the_sdk_sent(
    ledger_dir: Path, spec: Any
) -> None:
    path, wire, _ = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path)
    assert [c.wire_request_digest_sha256 for c in audit.calls] == wire.digests()
    assert [c.wire_request_bytes for c in audit.calls] == [
        len(body) for body in wire.bodies
    ]


def test_the_request_identity_and_the_wire_digest_coexist_and_differ(
    ledger_dir: Path, spec: Any
) -> None:
    path, _, run = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path)
    row = audit.calls[0]
    assert row.request_digest_sha256 != row.wire_request_digest_sha256
    assert row.request_digest_sha256 == run.execution.attempts[0].request_digest_sha256
    assert row.invocation_index == run.execution.attempts[0].invocation_index
    assert row.turn_index == run.execution.attempts[0].turn_index


def test_the_ledger_states_the_digest_of_the_exact_lifecycle_settings(
    ledger_dir: Path, spec: Any
) -> None:
    path, _, _ = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path)
    transport = transport_for(reference_handler(Wire()))
    assert audit.header.provider.settings_digest_sha256 == content_digest(
        dict(transport.settings)
    )
    assert audit.header.provider.settings["request_mapping"] == (
        "lifecycle_openai_responses_model_request_v9"
    )
    assert audit.header.intended_artifact_version == 8


def test_the_ledger_verifies_after_every_row_as_it_is_written(
    ledger_dir: Path, spec: Any
) -> None:
    path = ledger_dir / "execution.ndjson"
    wire = Wire()
    capture = WireCaptureTransport()
    cost_guard = guard()
    transport = transport_for(
        reference_handler(wire), capture=capture, cost_guard=cost_guard
    )
    recorder = recorder_for(
        path, transport, spec=spec, wire=capture, cost_guard=cost_guard
    )
    transport.attach_recorder(recorder)
    seen: list[int] = []

    class Watched(EvidenceRecordingModelAgent):
        def decide(self, observation: Any) -> Any:
            decision = super().decide(observation)
            audit = read_execution_ledger(path)
            seen.append(len(audit.calls))
            assert audit.status == "incomplete"
            return decision

    agent = Watched(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id="openai-gpt-5-6-luna",
        recorder=recorder,
        max_transport_calls=60,
    )
    run_episode(
        spec,
        "V1",
        "openai-gpt-5-6-luna",
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    assert seen == list(range(1, 47))


def test_the_artifact_binds_this_ledger_and_replays_with_no_provider_call(
    ledger_dir: Path, spec: Any
) -> None:
    """Contract 7: the artefact names the journal beside it, and replays clean.

    Phase 3A wrote the ledger and bound nothing into the artefact. Phase 3B
    binds it, and the two claims this test has always made are unchanged: the
    record reads back, and replaying it dispatches nothing. What is added is
    that the binding names *this* journal — and that neither the ledger's path
    nor any provider prose reaches the document.
    """
    path, _, run = _full_run(ledger_dir, spec)
    artifact = build_artifact(run)
    assert artifact["artifact_version"] == 8
    binding = artifact["provider_execution"]
    assert binding is not None
    audit = read_execution_ledger(path, require_complete=True)
    assert binding["execution_run_id"] == audit.header.execution_run_id
    assert binding["execution_ledger_digest_sha256"] == audit.ledger_digest_sha256
    assert binding["provider_calls"] == audit.totals.provider_calls == 46
    target = ledger_dir / "artifact.json"
    write_artifact(run, target)
    record = read_artifact(target)
    report = replay_artifact(spec, record)
    assert report.ok is True
    assert report.reproduction == "playback"
    text = target.read_text()
    assert PROVIDER_PROSE not in text
    assert FAKE_API_KEY not in text
    assert path.name not in text


# -- faults ------------------------------------------------------------------


def _faulted_run(
    ledger_dir: Path, spec: Any, *, response: httpx.Response, at: int = 4
) -> tuple[Path, Any]:
    path = ledger_dir / "execution.ndjson"
    wire = Wire()
    capture = WireCaptureTransport()
    cost_guard = guard()
    transport = transport_for(
        reference_handler(wire, faults={at: response}),
        capture=capture,
        cost_guard=cost_guard,
    )
    recorder = recorder_for(
        path, transport, spec=spec, wire=capture, cost_guard=cost_guard
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id="openai-gpt-5-6-luna",
        recorder=recorder,
        max_transport_calls=60,
    )
    with pytest.raises(ProviderFailure) as caught:
        run_episode(
            spec,
            "V1",
            "openai-gpt-5-6-luna",
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    return path, caught.value


@pytest.mark.parametrize(
    ("status", "kernel_fault"),
    [(500, "provider_server_error"), (429, "provider_rate_limited")],
)
def test_a_faulted_run_records_the_true_kernel_fault_and_status(
    ledger_dir: Path, spec: Any, status: int, kernel_fault: str
) -> None:
    path, failure = _faulted_run(
        ledger_dir,
        spec,
        response=httpx.Response(status, json={"error": {"message": "mock"}}),
    )
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.status == TERMINAL_EXCLUDED
    assert audit.scored is False
    last = audit.calls[-1]
    assert last.decision is None
    assert last.attempts[-1].fault == kernel_fault
    assert last.attempts[-1].http_status == status
    assert last.attempts[-1].response_received is True
    assert last.attempts[-1].usage_reported is False
    assert audit.totals.fault_counts == {kernel_fault: 1}
    # The Lifecycle exclusion code is its own vocabulary and does not overwrite
    # what the kernel measured.
    assert audit.terminal is not None
    assert audit.terminal.exclusion_code == failure.fault == "provider_transport"
    assert audit.terminal.exclusion_code not in audit.totals.fault_counts


def test_an_invalid_response_body_is_recorded_as_a_response_that_arrived(
    ledger_dir: Path, spec: Any
) -> None:
    body = responses_body(
        [
            function_call_item(
                "complete", json.dumps({"reason": "x", "evidence_refs": []})
            )
        ],
        model="gpt-not-the-pinned-model",
    )
    path, _ = _faulted_run(ledger_dir, spec, response=httpx.Response(200, json=body))
    audit = read_execution_ledger(path, require_complete=True)
    last = audit.calls[-1].attempts[-1]
    assert last.fault == "provider_response_invalid"
    assert last.response_received is True
    assert last.usage_reported is False
    assert audit.status == TERMINAL_EXCLUDED


def test_measured_spend_and_forfeited_reservation_are_never_conflated(
    ledger_dir: Path, spec: Any
) -> None:
    path, _ = _faulted_run(
        ledger_dir, spec, response=httpx.Response(500, json={"error": {}})
    )
    audit = read_execution_ledger(path)
    measured = Decimal(audit.totals.measured_cost_usd)
    forfeited = Decimal(audit.totals.forfeited_reservation_usd)
    reserved = Decimal(audit.totals.reserved_usd)
    assert measured > 0
    assert forfeited > 0
    assert measured != forfeited
    # A reservation is a conservative upper bound, so it dominates both, and
    # the forfeited one is budget consumed rather than measured provider spend.
    assert reserved > measured
    assert reserved > forfeited
    settled = [a.cost_settlement for call in audit.calls for a in call.attempts]
    assert settled.count("forfeited") == 1
    assert settled.count("measured") == len(settled) - 1


def test_a_faulted_run_writes_no_episode_artifact(ledger_dir: Path, spec: Any) -> None:
    path, _ = _faulted_run(
        ledger_dir, spec, response=httpx.Response(500, json={"error": {}})
    )
    assert sorted(p.name for p in ledger_dir.iterdir()) == sorted(
        [path.name, path.with_suffix(".partial.ndjson").name]
    )
    partial_rows = [
        json.loads(line)
        for line in path.with_suffix(".partial.ndjson").read_text().splitlines()
    ]
    assert partial_rows[0]["scoring"] == "NON-SCORED"
    assert partial_rows[-1]["classification"] == "excluded"


def test_a_provider_call_without_a_decision_is_recorded_before_the_terminal(
    ledger_dir: Path, spec: Any
) -> None:
    path, _ = _faulted_run(
        ledger_dir, spec, response=httpx.Response(500, json={"error": {}}), at=4
    )
    audit = read_execution_ledger(path)
    assert len(audit.calls) == 4
    assert [c.decision is None for c in audit.calls] == [False, False, False, True]
    assert audit.totals.provider_calls == 4


# -- binding provider output to the decision ---------------------------------


def test_the_normalized_digest_binds_the_exact_names_and_argument_values() -> None:
    first = normalized_response_digest(
        model=GPT_5_6_LUNA_MODEL,
        stop_classification="completed",
        tool_calls=(ToolCall("act", {"action_type": "x", "payload": {"a": 1}}),),
    )
    same_shape = normalized_response_digest(
        model=GPT_5_6_LUNA_MODEL,
        stop_classification="completed",
        tool_calls=(ToolCall("act", {"action_type": "x", "payload": {"a": 2}}),),
    )
    reordered = normalized_response_digest(
        model=GPT_5_6_LUNA_MODEL,
        stop_classification="completed",
        tool_calls=(ToolCall("act", {"payload": {"a": 1}, "action_type": "x"}),),
    )
    assert first != same_shape
    assert first == reordered


def test_the_recorded_row_binds_the_provider_answer_to_the_taped_decision(
    ledger_dir: Path, spec: Any
) -> None:
    path, wire, run = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path)
    row = audit.calls[0]
    sent = _as_parsed(wire.tool_calls[0])
    check_response_binding(
        row,
        model=GPT_5_6_LUNA_MODEL,
        stop_classification="completed",
        tool_calls=(sent,),
        decision_outcome=run.tape.decisions[0].outcome,
    )
    with pytest.raises(EvidenceBindingError):
        check_response_binding(
            row,
            model=GPT_5_6_LUNA_MODEL,
            stop_classification="completed",
            tool_calls=(ToolCall(sent.name, {**sent.arguments, "extra": 1}),),
            decision_outcome=run.tape.decisions[0].outcome,
        )
    with pytest.raises(EvidenceBindingError):
        check_response_binding(
            row,
            model=GPT_5_6_LUNA_MODEL,
            stop_classification="completed",
            tool_calls=(sent,),
            decision_outcome=run.tape.decisions[1].outcome,
        )


def test_editing_the_stored_response_digest_breaks_the_row(
    ledger_dir: Path, spec: Any
) -> None:
    path, _, _ = _full_run(ledger_dir, spec)
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["call"]["attempts"][0]["response_normalized_digest_sha256"] = "f" * 64
    lines[1] = json.dumps(row, sort_keys=True)
    target = path.with_name("tampered.ndjson")
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerChainError):
        read_execution_ledger(target)


def test_editing_a_call_index_breaks_the_row(ledger_dir: Path, spec: Any) -> None:
    path, _, _ = _full_run(ledger_dir, spec)
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["call"]["call_index"] = 41
    lines[1] = json.dumps(row, sort_keys=True)
    target = path.with_name("tampered.ndjson")
    target.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerChainError):
        read_execution_ledger(target)


# -- budgets and cost --------------------------------------------------------


def test_the_recorder_refuses_to_record_past_the_call_budget(
    ledger_dir: Path, spec: Any
) -> None:
    path = ledger_dir / "execution.ndjson"
    wire = Wire()
    capture = WireCaptureTransport()
    cost_guard = guard()
    transport = transport_for(
        reference_handler(wire), capture=capture, cost_guard=cost_guard
    )
    recorder = recorder_for(
        path,
        transport,
        spec=spec,
        wire=capture,
        cost_guard=cost_guard,
        ledger_controls=controls(max_calls=3),
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id="openai-gpt-5-6-luna",
        recorder=recorder,
        max_transport_calls=60,
    )
    with pytest.raises(ProviderFailure) as caught:
        run_episode(
            spec,
            "V1",
            "openai-gpt-5-6-luna",
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    assert caught.value.fault == FAULT_BUDGET
    audit = read_execution_ledger(path, require_complete=True)
    assert audit.totals.provider_calls == 3
    assert audit.scored is False
    assert len(wire.bodies) == 3


def test_the_cost_guard_refuses_before_a_request_is_dispatched(
    ledger_dir: Path, spec: Any
) -> None:
    wire = Wire()
    cost_guard = guard(cap="0.000001")
    transport = transport_for(reference_handler(wire), cost_guard=cost_guard)
    agent = ModelAgent(transport, model=GPT_5_6_LUNA_MODEL, agent_id="a")
    agent.begin_episode({"probe": True})
    with pytest.raises(ProviderFailure):
        run_episode(spec, "V1", "a", agent_factory=lambda: agent, agent_kind="model")
    assert wire.bodies == []


def test_the_ledger_can_express_every_fault_the_model_boundary_raises() -> None:
    from operatebench.agents.transport import FAULTS
    from operatebench.execution_ledger import ABORTED_CODE, EXCLUSION_CODES

    assert set(EXCLUSION_CODES) == {*FAULTS, ABORTED_CODE}


def test_a_terminal_code_outside_that_vocabulary_is_refused() -> None:
    from operatebench.execution_ledger import (
        LedgerSchemaError,
        LedgerTerminal,
        recompute_totals,
    )

    with pytest.raises(LedgerSchemaError):
        LedgerTerminal(
            kind=TERMINAL_EXCLUDED,
            ended_at="2031-03-03T09:00:00Z",
            exclusion_code="the provider was unhappy",
            totals=recompute_totals(()),
        )


def test_the_pricing_policy_is_digested_and_states_its_rate_source() -> None:
    stated = policy()
    assert stated.as_dict()["rate_source"] == RATE_SOURCE_OPERATOR
    assert stated.as_dict()["input_usd_per_mtok"] == "1.25"
    assert stated.digest_sha256 == content_digest(stated.as_dict())
    other = LifecyclePricingPolicy(
        policy_id="offline_operator_pinned_v1",
        input_usd_per_mtok=Decimal("2.50"),
        output_usd_per_mtok=OUTPUT_RATE,
        rate_source=RATE_SOURCE_OPERATOR,
    )
    assert other.digest_sha256 != stated.digest_sha256


def test_the_pessimistic_worst_case_is_computed_from_the_stated_rates() -> None:
    worst = policy().worst_case_usd(
        calls=60, input_token_bound=23_409, max_output_tokens=4096
    )
    expected = (
        Decimal(60) * (Decimal(23_409) * INPUT_RATE + Decimal(4096) * OUTPUT_RATE)
    ) / Decimal(1_000_000)
    assert usd_text(worst) == usd_text(expected)


def test_a_cap_that_cannot_cover_the_worst_case_is_refused_by_the_guard() -> None:
    tight = LifecycleCostGuard(
        policy=policy(), cap_usd=Decimal("0.00001"), max_output_tokens=MAX_OUTPUT_TOKENS
    )
    with pytest.raises(CostCapExceededError):
        tight.authorize(input_tokens_upper_bound=14_706)


# -- the recorder is optional ------------------------------------------------


def test_a_deterministic_run_writes_no_provider_ledger(
    ledger_dir: Path, spec: Any
) -> None:
    run = run_episode(spec, "V1", "reference")
    assert run.execution.outcome_source == "deterministic"
    assert list(ledger_dir.iterdir()) == []


def test_run_episode_without_a_recorder_is_unchanged(spec: Any) -> None:
    first = run_episode(spec, "V1", "reference")
    second = run_episode(spec, "V1", "reference", evidence_recorder=None)
    assert (
        first.outcome.trajectory_digest_sha256 == second.outcome.trajectory_digest_sha256
    )
    assert first.outcome.final_state_digest_sha256 == (
        second.outcome.final_state_digest_sha256
    )


def test_an_operation_instance_id_is_minted_before_the_recorder_begins(
    ledger_dir: Path, spec: Any
) -> None:
    path, _, run = _full_run(ledger_dir, spec)
    audit = read_execution_ledger(path)
    assert audit.header.operation_instance_id == run.operation_instance_id
    assert audit.header.operation_instance_id != new_operation_instance_id()
    assert audit.header.execution_run_id.startswith("exec_")


# -- helpers -----------------------------------------------------------------


def _as_parsed(call: ToolCall) -> ToolCall:
    """The call as the response projection produces it, null optionals elided.

    The mock answers with the reference agent's own arguments, and the Lifecycle
    response projection elides null semantic optionals on the way back in. The
    digest binds what was *parsed*, so a test that recomputes it has to make the
    same journey rather than a shorter one.
    """
    from operatebench.agents.openai_responses import elide_null_optionals

    return ToolCall(call.name, elide_null_optionals(call.name, call.arguments))
