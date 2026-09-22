"""One complete Lifecycle model run, over the real SDK and a mock transport.

Shared by every suite that needs a *scored* provider execution to bind an
artefact to: the artefact-7 binding suite, the bundle audit suite, and the
existing artefact suites whose model runs must now carry evidence rather than
a null binding.

Nothing here opens a socket or reads a credential. Every request is answered
in process by :class:`httpx.MockTransport` after being serialised by the real
``openai`` SDK, and the answers are the shipped reference agent's own decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import openai

from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    provider_identity_for,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS
from operatebench.agents.openai_responses import OpenAIResponsesTransport
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import ProviderFailure, ToolCall
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.execution_ledger import LedgerControls
from operatebench.providers.openai_responses import GPT_5_6_LUNA_MODEL
from operatebench.runner import run_episode
from tests.model_transport import PROVIDER_PROSE, observation_from_dict, tool_call_for
from tests.openai_transport import (
    FAKE_API_KEY,
    function_call_item,
    message_item,
    responses_body,
)

MODEL_AGENT_ID = "openai-gpt-5-6-luna"
SCENARIO_ID = "V1"
DEADLINE_SECONDS = 30.0
EXPECTED_PROVIDER_CALLS = 46
INPUT_TOKENS = 4211
OUTPUT_TOKENS = 118

#: Operator-supplied and pinned for offline evidence. Not a claim about any
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


def controls(*, max_calls: int = 60, cap: str = "5.00") -> LedgerControls:
    return LedgerControls(
        max_provider_calls=max_calls,
        max_attempts_per_call=1,
        expected_provider_calls=min(EXPECTED_PROVIDER_CALLS, max_calls),
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
        self.tool_calls: list[ToolCall] = []

    def digests(self) -> list[str]:
        return [hashlib.sha256(body).hexdigest() for body in self.bodies]


def reference_handler(wire: Wire, *, faults: dict[int, Any] | None = None) -> Any:
    """Answers every turn with the shipped reference agent's own decision."""
    reference = build_agent("reference")
    planned = faults or {}
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        wire.bodies.append(request.content)
        fault = planned.get(state["calls"])
        if fault is not None:
            return fault
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


@dataclass(frozen=True)
class ModelRunEvidence:
    """One model run, the journal it wrote and the bytes it put on the wire."""

    ledger_path: Path
    wire: Wire
    run: Any
    recorder: ProviderEvidenceRecorder
    failure: ProviderFailure | None = None


def model_run_with_ledger(
    directory: Path,
    spec: Any,
    *,
    ledger_name: str = "execution_ledger.ndjson",
    faults: dict[int, Any] | None = None,
    max_calls: int = 60,
    cap: str = "5.00",
    agent_id: str = MODEL_AGENT_ID,
) -> ModelRunEvidence:
    """Run V1 with a model agent whose every call is recorded to a journal.

    The order is the one the runner contracts for: the recorder is handed to
    :func:`~operatebench.runner.run_episode`, which begins it before the first
    request leaves and finalises it on every exit.
    """
    path = Path(directory) / ledger_name
    wire = Wire()
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(reference_handler(wire, faults=faults)))
    guard = LifecycleCostGuard(
        policy=policy(), cap_usd=Decimal(cap), max_output_tokens=MAX_OUTPUT_TOKENS
    )
    transport = OpenAIResponsesTransport(
        model=GPT_5_6_LUNA_MODEL,
        client=openai.OpenAI(
            api_key=FAKE_API_KEY,
            base_url="https://api.openai.com/v1",
            max_retries=0,
            http_client=httpx.Client(transport=capture),
        ),
        deadline_seconds=DEADLINE_SECONDS,
        clock=fixed_clock(),
        cost_guard=guard,
    )
    recorder = ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=SCENARIO_ID,
            agent_id=agent_id,
        ),
        provider=provider_identity_for(transport),
        controls=controls(max_calls=max_calls, cap=cap),
        wire=capture,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id=agent_id,
        recorder=recorder,
        max_transport_calls=max_calls,
    )
    failure: ProviderFailure | None = None
    run = None
    try:
        run = run_episode(
            spec,
            SCENARIO_ID,
            agent_id,
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    except ProviderFailure as exc:
        failure = exc
    return ModelRunEvidence(
        ledger_path=path, wire=wire, run=run, recorder=recorder, failure=failure
    )


def bound_model_artifact(
    tmp_path_factory: Any, spec: Any, *, agent_id: str = MODEL_AGENT_ID
) -> dict[str, Any]:
    """One artefact-7 record of a model run, bound to the journal that witnessed it.

    For the suites whose subject is the *reader* rather than the run: they need
    a genuine bound document to edit adversarially, and contract 7 gives them no
    way to assemble one without an execution ledger behind it. That is the point
    of the contract, so the fixture pays for a real recorded run rather than
    routing around it.
    """
    from operatebench.artifact import build_artifact

    directory = Path(tmp_path_factory.mktemp("bound_evidence"))
    directory.chmod(0o700)
    return build_artifact(model_run_with_ledger(directory, spec, agent_id=agent_id).run)


__all__ = [
    "DEADLINE_SECONDS",
    "EXPECTED_PROVIDER_CALLS",
    "INPUT_TOKENS",
    "MODEL_AGENT_ID",
    "OUTPUT_TOKENS",
    "SCENARIO_ID",
    "ModelRunEvidence",
    "Wire",
    "bound_model_artifact",
    "controls",
    "fixed_clock",
    "model_run_with_ledger",
    "policy",
    "reference_handler",
]
