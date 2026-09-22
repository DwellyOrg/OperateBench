"""Offline preflight for Lifecycle provider execution evidence.

Run as a command::

    uv run python -m tools.preflight_lifecycle_provider_evidence --output-dir DIR

**This tool never reaches a provider.** It builds a real ``openai.OpenAI``
client over an in-process ``httpx.MockTransport``, so the request is serialised
by the real SDK and the answer is parsed into real ``openai.types.responses``
models — and the answers are the shipped reference agent's own decisions, scripted
per turn. It reads no credential: the key handed to the client is a fixed
placeholder that never leaves the process, and the tool *refuses to start* if any
of the live provider environment variables is set, so it cannot be pointed at a
service by accident.

There is deliberately no flag here that would make a live call. Whether this
build may make one is not a command-line decision.

What it proves, in order, and it exits non-zero on the first thing that fails:

1. the operation spec re-derives its own identity;
2. the pinned model's request profile, the payload built under it and the
   endpoint the client is pinned to are the ones this build states;
3. the SDK's own retry loop is off and this build's retry policy allows one
   attempt per call;
4. the pessimistic worst case — every one of the maximum calls at the largest
   request this scaffold produces, plus the whole output ceiling, at the
   operator's stated rates — fits inside the stated cost cap, **before** any
   request is dispatched;
5. a whole scenario runs over the mock, writing an execution ledger row by row;
   the ledger reads back, verifies its own chain and states totals that follow
   from its rows;
6. the episode artefact this build writes is version 7, binds the ledger that
   witnessed the run, passes a bundle audit against that sidecar, reads back and
   replays clean at zero provider calls;
7. three fault cells — HTTP 500, HTTP 429 and a body naming a model this run did
   not pin — each write an excluded ledger carrying the true kernel fault and
   status, with measured spend and forfeited reservation stated apart, and no
   episode artefact.

The output directory is named by the operator and is not created here: evidence
is written where somebody said, or nowhere. It must already exist and be mode
0700.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import openai

from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    check_response_binding,
    provider_identity_for,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, ModelAgent
from operatebench.agents.openai_responses import (
    LIFECYCLE_RETRY_POLICY,
    OpenAIResponsesTransport,
    build_model_payload,
    check_model_request,
    elide_null_optionals,
    lifecycle_openai_settings,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import ProviderFailure, ToolCall
from operatebench.artifact import (
    ARTIFACT_VERSION,
    build_artifact,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.core.outcomes import Act, Ask, Complete, Escalate, Wait
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    COST_CAP_POLICY_REFUSE,
    TERMINAL_EXCLUDED,
    TERMINAL_SCORED,
    LedgerControls,
    read_execution_ledger,
)
from operatebench.providers.config import check_payload_fields
from operatebench.providers.cost import usd_text
from operatebench.providers.openai_responses import (
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    OPENAI_BASE_URL,
    REJECTED_VARIABLES,
    check_client_endpoint,
    check_request_profile,
    request_token_bound,
)
from operatebench.runner import run_episode

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The fixture this preflight runs. The shipped example, by relative path, so
#: this tool names no filesystem outside the tree it lives in.
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"

#: The scenario the reference preflight runs, and the number of provider calls a
#: successful one makes. The count is the *expectation*, not the bound: the
#: bound is ``--max-provider-calls`` and is deliberately larger.
SCENARIO_ID = "V1"
EXPECTED_PROVIDER_CALLS = 46

#: The safety bound on provider calls. Head-room over the expected count for
#: classification retries, enforced by the model agent, by the recorder and by
#: this tool's own request counter — three places, because a budget with one
#: enforcement point stops being one the moment that point is bypassed.
#:
#: **Absolute.** It is this build's maximum and not a default an operator can
#: raise: the token ceiling below is computed from it, and a command line that
#: could raise one without the other would authorise exposure nothing checked.
#: Asking for *fewer* calls is always allowed; asking for more is refused while
#: the arguments are still being read, before any client exists.
MAX_PROVIDER_CALLS = 60
DEFAULT_MAX_PROVIDER_CALLS = MAX_PROVIDER_CALLS

#: The absolute worst-case token exposure of a run under those bounds: every
#: call at the largest request this scaffold has been measured to produce, plus
#: the whole output ceiling. Fixed by this build at 1,650,300 tokens; there is
#: no flag that moves it, and moving it is a code change.
LARGEST_REQUEST_TOKEN_BOUND = 23_409


def token_worst_case(max_calls: int) -> int:
    """Every authorised call at the largest request, plus the whole ceiling."""
    return int(max_calls * (LARGEST_REQUEST_TOKEN_BOUND + MAX_OUTPUT_TOKENS))


TOKEN_HARD_CAP = token_worst_case(MAX_PROVIDER_CALLS)

#: Per-turn counts the scripted answers report, so the arithmetic in the summary
#: is arithmetic over stated numbers rather than over invented ones.
SCRIPTED_INPUT_TOKENS = 4211
SCRIPTED_OUTPUT_TOKENS = 118

#: Wall-clock bounds. Both are infrastructure decisions, stated rather than
#: defaulted somewhere inside a transport.
TURN_DEADLINE_SECONDS = 30.0
WALL_CLOCK_DEADLINE_SECONDS = 1800.0

#: The placeholder handed to the SDK. Not a credential, not read from anywhere,
#: and it never leaves this process: every request is answered by an in-process
#: mock transport.
PLACEHOLDER_KEY = "preflight-mock-transport-not-a-credential"

#: The variable this build reads a real credential from, refused here alongside
#: the ones the shared adapter already rejects. A preflight that ran with a live
#: credential in the environment would be one edit away from being a live call.
CREDENTIAL_VARIABLE = "OPENAI_API_KEY"

AGENT_ID = "openai-gpt-5-6-luna"


class PreflightFailure(Exception):
    """A preflight check did not pass. The message is what failed and why."""


# ------------------------------------------------------------ scripted answers


def _wait_arguments(wait: Wait) -> dict[str, Any]:
    return {
        "reason": wait.reason,
        "wake_on": list(wait.wake_on),
        "fallback_after_minutes": wait.fallback_after_minutes,
    }


def tool_call_for(outcome: Any) -> ToolCall:
    """Encode one decision as the structured call a model would have emitted."""
    if isinstance(outcome, RetrieveBatch):
        return ToolCall(
            "retrieve",
            {"requests": [request.as_dict() for request in outcome.requests]},
        )
    if isinstance(outcome, Act):
        return ToolCall(
            "act",
            {
                "action_type": outcome.action_type,
                "payload": dict(outcome.payload),
                "evidence_refs": list(outcome.evidence_refs),
                "rationale": outcome.rationale,
            },
        )
    if isinstance(outcome, Wait):
        return ToolCall("wait", _wait_arguments(outcome))
    if isinstance(outcome, Ask):
        return ToolCall(
            "ask",
            {
                "recipient_actor_id": outcome.recipient_actor_id,
                "message_fixture_id": outcome.message_fixture_id,
                "correlation_id": outcome.correlation_id,
                "wait": _wait_arguments(outcome.wait),
            },
        )
    if isinstance(outcome, Escalate):
        return ToolCall(
            "escalate",
            {
                "checkpoint_id": outcome.checkpoint_id,
                "exception_type": outcome.exception_type,
                "evidence_refs": list(outcome.evidence_refs),
                "deadline_after_minutes": outcome.deadline_after_minutes,
                "rationale": outcome.rationale,
            },
        )
    if isinstance(outcome, Complete):
        return ToolCall(
            "complete",
            {"reason": outcome.reason, "evidence_refs": list(outcome.evidence_refs)},
        )
    raise PreflightFailure(
        f"the reference agent produced a {type(outcome).__name__}, which this "
        "preflight has no scripted answer for"
    )


def observation_from(payload: Mapping[str, Any]) -> AgentObservation:
    """Rebuild the observation a request carried, from the request alone."""
    return AgentObservation(
        now=payload["now"],
        operation_id=payload["operation_id"],
        operation_instance_id=payload["operation_instance_id"],
        invocation_index=payload["invocation_index"],
        turn_index=payload["turn_index"],
        policy=payload["policy"],
        actors=payload["actors"],
        message_fixture_ids=payload["message_fixture_ids"],
        action_schemas=payload["action_schemas"],
        wake_event_types=payload["wake_event_types"],
        last_rejection=payload["last_rejection"],
        phase=payload["phase"],
        event=payload["event"],
        retrieval=payload["retrieval"],
    )


def responses_body(
    output: Sequence[Any], *, input_tokens: int, output_tokens: int, model: str
) -> dict[str, Any]:
    """One Responses API body, as the wire carries it."""
    return {
        "id": "resp_preflight",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": "completed",
        "output": list(output),
        "parallel_tool_calls": False,
        "tool_choice": "required",
        "tools": [],
        "incomplete_details": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def function_call_item(name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_preflight",
        "call_id": "call_preflight",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


class ScriptedProvider:
    """Answers each turn with the shipped reference agent's own decision.

    Also this preflight's independent request counter: it counts every request
    that reaches it, whatever the model agent and the recorder believe, and
    refuses one past the run's stated bound.
    """

    def __init__(
        self, *, max_calls: int, fault: httpx.Response | None = None, fault_at: int = 0
    ) -> None:
        self._reference = build_agent("reference")
        self._max_calls = max_calls
        self._fault = fault
        self._fault_at = fault_at
        self.calls = 0
        self.tool_calls: list[ToolCall] = []
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls > self._max_calls:
            raise PreflightFailure(
                f"a request was made past the run's bound of {self._max_calls} "
                f"provider call(s); this is call {self.calls}"
            )
        self.bodies.append(request.content)
        if self._fault is not None and self.calls == self._fault_at:
            return self._fault
        prompt = json.loads(request.content)["input"][0]["content"]
        observation = observation_from(json.loads(prompt)["observation"])
        call = tool_call_for(self._reference.decide(observation))
        self.tool_calls.append(
            ToolCall(call.name, elide_null_optionals(call.name, call.arguments))
        )
        return httpx.Response(
            200,
            json=responses_body(
                [function_call_item(call.name, json.dumps(call.arguments))],
                input_tokens=SCRIPTED_INPUT_TOKENS,
                output_tokens=SCRIPTED_OUTPUT_TOKENS,
                model=GPT_5_6_LUNA_MODEL,
            ),
        )


# ------------------------------------------------------------------- the checks


def refuse_live_environment() -> list[str]:
    """Refuse to start if anything in the environment points at a service."""
    named = [CREDENTIAL_VARIABLE, *REJECTED_VARIABLES]
    present = sorted(name for name in named if os.environ.get(name))
    if present:
        raise PreflightFailure(
            f"the environment states {present}. This preflight makes no live call "
            "and reads no credential, and it refuses to run beside variables that "
            "would point a client at a service — a run that started here would be "
            "one edit away from being a live one. Unset them and run again"
        )
    return named


def build_client(handler: Callable[[httpx.Request], httpx.Response]) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=PLACEHOLDER_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def check_static_settings(client: openai.OpenAI) -> dict[str, Any]:
    """Everything about the configuration that can be proven before a request."""
    profile = check_request_profile(GPT_5_6_LUNA_PROFILE, model=GPT_5_6_LUNA_MODEL)
    check_client_endpoint(client)
    if client.max_retries != 0:
        raise PreflightFailure(
            f"the SDK client was built with max_retries={client.max_retries}; this "
            "build owns the retry policy and a second, hidden one underneath it "
            "would make a turn's evidence describe an unknown number of requests"
        )
    if LIFECYCLE_RETRY_POLICY.max_attempts != 1:
        raise PreflightFailure(
            "this build's retry policy allows "
            f"{LIFECYCLE_RETRY_POLICY.max_attempts} attempts per call; the ledger's "
            "control block states one"
        )
    settings = lifecycle_openai_settings(
        model=GPT_5_6_LUNA_MODEL,
        deadline_seconds=TURN_DEADLINE_SECONDS,
        profile=profile,
        retry=LIFECYCLE_RETRY_POLICY,
    )
    return {
        "request_profile": profile.profile_id,
        "request_mapping": settings["request_mapping"],
        "api": settings["api"],
        "base_url": settings["base_url"],
        "sdk": settings["sdk"],
        "sdk_version": settings["sdk_version"],
        "sdk_max_retries": client.max_retries,
        "max_attempts_per_call": LIFECYCLE_RETRY_POLICY.max_attempts,
        "store": settings["store"],
        "tool_choice": settings["tool_choice"],
    }


def check_one_request(spec: Any) -> dict[str, Any]:
    """Prove a request this build builds is one it will accept and send.

    Runs entirely in process: the observation comes from a deterministic
    reference run through a transport that answers without a socket, and the
    payload is checked against the profile the way the shared exchange checks
    it, before anything is dispatched.
    """
    provider = ScriptedProvider(max_calls=DEFAULT_MAX_PROVIDER_CALLS)
    transport = OpenAIResponsesTransport(
        model=GPT_5_6_LUNA_MODEL,
        client=build_client(provider),
        deadline_seconds=TURN_DEADLINE_SECONDS,
    )
    agent = ModelAgent(transport, model=GPT_5_6_LUNA_MODEL, agent_id=AGENT_ID)
    agent.begin_episode({"preflight": True})
    run_episode(
        spec, SCENARIO_ID, AGENT_ID, agent_factory=lambda: agent, agent_kind="model"
    )
    prompt = json.loads(provider.bodies[0])["input"][0]["content"]
    observation = observation_from(json.loads(prompt)["observation"])
    request = agent.build_request(observation)
    check_model_request(request, model=GPT_5_6_LUNA_MODEL)
    payload = build_model_payload(
        request, model=GPT_5_6_LUNA_MODEL, profile=GPT_5_6_LUNA_PROFILE
    )
    check_payload_fields(
        payload,
        profile_id=GPT_5_6_LUNA_PROFILE.profile_id,
        sent=GPT_5_6_LUNA_PROFILE.sent_fields(),
        omitted=GPT_5_6_LUNA_PROFILE.omitted_fields(),
    )
    return {
        "checked_request_digest_sha256": request.request_digest_sha256,
        "largest_measured_request_bytes": max(len(body) for body in provider.bodies),
        "input_token_upper_bound": request_token_bound(payload),
        "payload_fields": sorted(payload),
    }


def check_max_provider_calls(max_calls: int) -> int:
    """Refuse a call count outside this build's absolute bound, and say so.

    Called before anything is built. A run of sixty-one is not a run with a
    larger cap: it is a run whose worst-case token exposure passes a ceiling
    this build fixed, and refusing it here means no SDK client, no transport
    and no ledger directory is touched on the way to finding that out.
    """
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_PROVIDER_CALLS:
        raise PreflightFailure(
            f"this build authorises 1 to {MAX_PROVIDER_CALLS} provider call(s) and "
            f"was asked for {max_calls}. The bound is absolute: the token ceiling of "
            f"{TOKEN_HARD_CAP} is computed from it, so raising one without the other "
            "would authorise exposure nothing checked. Fewer calls are always allowed"
        )
    return max_calls


def check_token_exposure(*, max_calls: int, token_hard_cap: int) -> dict[str, Any]:
    """Refuse the run if its worst-case token exposure passes the ceiling.

    The cost cap and this are two different bounds and both are checked before
    dispatch. A cap in dollars moves with the rates an operator states; the
    token ceiling is fixed by this build, so it is the one that still holds if
    the rates are wrong.
    """
    worst = token_worst_case(max_calls)
    if worst > token_hard_cap:
        raise PreflightFailure(
            f"the worst-case token exposure of {worst} token(s) — {max_calls} call(s) "
            f"at {LARGEST_REQUEST_TOKEN_BOUND} input plus {MAX_OUTPUT_TOKENS} output — "
            f"exceeds this build's ceiling of {token_hard_cap}. The run is refused "
            "before anything is dispatched"
        )
    return {
        "token_worst_case": worst,
        "token_hard_cap": token_hard_cap,
        "largest_request_token_bound": LARGEST_REQUEST_TOKEN_BOUND,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def check_worst_case(
    policy: LifecyclePricingPolicy, *, cap: Decimal, max_calls: int
) -> dict[str, Any]:
    """Refuse the run if its pessimistic worst case does not fit the cap."""
    worst = policy.worst_case_usd(
        calls=max_calls,
        input_token_bound=LARGEST_REQUEST_TOKEN_BOUND,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if worst > cap:
        raise PreflightFailure(
            f"the pessimistic worst case of {usd_text(worst)} USD — {max_calls} "
            f"call(s), each the largest request measured plus the whole output "
            f"ceiling, at the stated rates — exceeds the cap of {usd_text(cap)} USD. "
            "The run is refused before anything is dispatched"
        )
    return {
        "worst_case_usd": usd_text(worst),
        "cost_cap_usd": usd_text(cap),
        "token_hard_cap": TOKEN_HARD_CAP,
        "pricing_digest_sha256": policy.digest_sha256,
        "rate_source": policy.rate_source,
    }


def controls_for(
    policy: LifecyclePricingPolicy, *, cap: Decimal, max_calls: int
) -> LedgerControls:
    return LedgerControls(
        max_provider_calls=max_calls,
        max_attempts_per_call=LIFECYCLE_RETRY_POLICY.max_attempts,
        expected_provider_calls=min(EXPECTED_PROVIDER_CALLS, max_calls),
        token_hard_cap=TOKEN_HARD_CAP,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cost_cap_usd=usd_text(cap),
        cost_cap_policy=COST_CAP_POLICY_REFUSE,
        turn_deadline_seconds=TURN_DEADLINE_SECONDS,
        wall_clock_deadline_seconds=WALL_CLOCK_DEADLINE_SECONDS,
        pricing=policy.as_dict(),
    )


def run_cell(
    spec: Any,
    *,
    ledger_path: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
    max_calls: int,
    fault: httpx.Response | None = None,
    fault_at: int = 0,
) -> dict[str, Any]:
    """One whole scenario over the mock, with its evidence written as it runs."""
    provider = ScriptedProvider(max_calls=max_calls, fault=fault, fault_at=fault_at)
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(provider))
    client = openai.OpenAI(
        api_key=PLACEHOLDER_KEY,
        base_url=OPENAI_BASE_URL,
        max_retries=0,
        http_client=httpx.Client(transport=capture),
    )
    guard = LifecycleCostGuard(
        policy=policy,
        cap_usd=cap,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        token_hard_cap=TOKEN_HARD_CAP,
    )
    transport = OpenAIResponsesTransport(
        model=GPT_5_6_LUNA_MODEL,
        client=client,
        deadline_seconds=TURN_DEADLINE_SECONDS,
        cost_guard=guard,
    )
    recorder = ProviderEvidenceRecorder(
        path=ledger_path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=SCENARIO_ID,
            agent_id=AGENT_ID,
        ),
        provider=provider_identity_for(transport),
        controls=controls_for(policy, cap=cap, max_calls=max_calls),
        wire=capture,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=GPT_5_6_LUNA_MODEL,
        agent_id=AGENT_ID,
        recorder=recorder,
        max_transport_calls=max_calls,
    )
    failure: ProviderFailure | None = None
    run = None
    try:
        run = run_episode(
            spec,
            SCENARIO_ID,
            AGENT_ID,
            agent_factory=lambda: agent,
            agent_kind="model",
            evidence_recorder=recorder,
        )
    except ProviderFailure as exc:
        failure = exc
    return {
        "run": run,
        "failure": failure,
        "provider": provider,
        "capture": capture,
        "guard": guard,
    }


def reference_cell(
    spec: Any,
    *,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
    max_calls: int,
) -> dict[str, Any]:
    """The successful cell: a whole scenario, a ledger, an artefact, a replay."""
    ledger_path = directory / "execution_ledger.ndjson"
    result = run_cell(
        spec, ledger_path=ledger_path, policy=policy, cap=cap, max_calls=max_calls
    )
    if result["failure"] is not None:
        raise PreflightFailure(
            "the reference cell did not complete: it stopped at the provider "
            f"boundary ({result['failure'].fault})"
        )
    run = result["run"]
    provider = result["provider"]
    audit = read_execution_ledger(ledger_path, require_complete=True)
    if audit.status != TERMINAL_SCORED:
        raise PreflightFailure(f"the reference ledger reads as {audit.status!r}")
    if audit.totals.provider_calls != EXPECTED_PROVIDER_CALLS:
        raise PreflightFailure(
            f"the reference cell made {audit.totals.provider_calls} provider "
            f"call(s) against an expectation of {EXPECTED_PROVIDER_CALLS}"
        )
    if audit.totals.fault_counts:
        raise PreflightFailure(
            f"the reference cell recorded fault(s) {audit.totals.fault_counts}; a "
            "successful reference run records none"
        )
    if audit.totals.forfeited_reservation_usd != "0":
        raise PreflightFailure(
            "the reference cell forfeited "
            f"{audit.totals.forfeited_reservation_usd} USD of reservation; a run "
            "with no faults forfeits nothing"
        )
    wire_digests = [call.wire_request_digest_sha256 for call in audit.calls]
    captured = [record.digest_sha256 for record in result["capture"].captures]
    if wire_digests != captured:
        raise PreflightFailure(
            "the ledger's wire digests are not the digests of the bodies the SDK "
            "serialised"
        )
    check_response_binding(
        audit.calls[0],
        model=GPT_5_6_LUNA_MODEL,
        stop_classification="completed",
        tool_calls=(provider.tool_calls[0],),
        decision_outcome=run.tape.decisions[0].outcome,
    )

    artifact = build_artifact(run)
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise PreflightFailure(
            f"the episode artefact is version {artifact['artifact_version']} and this "
            f"build writes {ARTIFACT_VERSION}"
        )
    binding = artifact["provider_execution"]
    if binding is None:
        raise PreflightFailure(
            "the episode artefact carries no provider execution binding; contract "
            f"{ARTIFACT_VERSION} requires one for a run whose decisions came off a "
            "provider"
        )
    if binding["execution_ledger_digest_sha256"] != audit.ledger_digest_sha256:
        raise PreflightFailure(
            "the artefact's binding does not state the digest of the ledger beside it"
        )
    artifact_path = directory / "episode_artifact.json"
    write_artifact(run, artifact_path)
    record = read_artifact(artifact_path)
    # The binding read out of the artefact attributes nothing on its own. This is
    # the check that opens the sidecar and holds the two documents to each other.
    bundle = audit_execution_bundle_files(artifact_path, ledger_path)
    report = replay_artifact(spec, record)
    if not report.ok:
        raise PreflightFailure(f"the artefact did not replay: {list(report.differences)}")
    calls_before_replay = provider.calls
    if calls_before_replay != EXPECTED_PROVIDER_CALLS:
        raise PreflightFailure(
            f"the mock transport answered {calls_before_replay} request(s) and the "
            f"run recorded {EXPECTED_PROVIDER_CALLS}"
        )
    return {
        "ledger": str(ledger_path.name),
        "artifact": str(artifact_path.name),
        "status": run.outcome.status,
        "reliable": run.reliable,
        "artifact_version": artifact["artifact_version"],
        "provider_execution": dict(binding),
        "bundle_audit": bundle.summary(),
        "replay_ok": report.ok,
        "replay_reproduction": report.reproduction,
        "replay_carried_forward": list(report.carried_forward),
        "provider_calls_during_replay": provider.calls - calls_before_replay,
        "audit": audit.summary(),
    }


def fault_cell(
    spec: Any,
    *,
    directory: Path,
    policy: LifecyclePricingPolicy,
    cap: Decimal,
    max_calls: int,
    label: str,
    response: httpx.Response,
    at: int = 4,
) -> dict[str, Any]:
    """One faulted cell: an excluded ledger, and no episode artefact."""
    ledger_path = directory / f"execution_ledger_{label}.ndjson"
    before = sorted(p.name for p in directory.iterdir())
    result = run_cell(
        spec,
        ledger_path=ledger_path,
        policy=policy,
        cap=cap,
        max_calls=max_calls,
        fault=response,
        fault_at=at,
    )
    if result["failure"] is None:
        raise PreflightFailure(f"the {label} cell completed; it was expected to fault")
    audit = read_execution_ledger(ledger_path, require_complete=True)
    if audit.status != TERMINAL_EXCLUDED or audit.scored:
        raise PreflightFailure(
            f"the {label} ledger reads as {audit.status!r} and must read as "
            f"{TERMINAL_EXCLUDED!r}"
        )
    if not audit.totals.fault_counts:
        raise PreflightFailure(f"the {label} ledger names no kernel fault")
    last = audit.calls[-1].attempts[-1]
    if audit.calls[-1].decision is not None:
        raise PreflightFailure(
            f"the {label} ledger records a decision on the call that faulted"
        )
    added = sorted({p.name for p in directory.iterdir()} - set(before))
    if added != [ledger_path.name]:
        raise PreflightFailure(
            f"the {label} cell wrote {added}; an excluded run writes its ledger and "
            "no episode artefact"
        )
    return {
        "ledger": ledger_path.name,
        "lifecycle_fault": result["failure"].fault,
        "kernel_fault": last.fault,
        "http_status": last.http_status,
        "response_received": last.response_received,
        "usage_reported": last.usage_reported,
        "terminal_reason": audit.calls[-1].terminal_reason,
        "provider_calls": audit.totals.provider_calls,
        "measured_cost_usd": audit.totals.measured_cost_usd,
        "forfeited_reservation_usd": audit.totals.forfeited_reservation_usd,
        "reserved_usd": audit.totals.reserved_usd,
        "episode_artifact_written": False,
        "audit": audit.summary(),
    }


def wrong_model_body() -> dict[str, Any]:
    return responses_body(
        [
            function_call_item(
                "complete", json.dumps({"reason": "x", "evidence_refs": []})
            )
        ],
        input_tokens=SCRIPTED_INPUT_TOKENS,
        output_tokens=SCRIPTED_OUTPUT_TOKENS,
        model="gpt-not-the-pinned-model",
    )


def preflight(
    *,
    directory: Path,
    cap: Decimal,
    input_rate: Decimal,
    output_rate: Decimal,
    policy_id: str,
    max_calls: int,
) -> dict[str, Any]:
    refused = refuse_live_environment()
    check_max_provider_calls(max_calls)
    exposure = check_token_exposure(max_calls=max_calls, token_hard_cap=TOKEN_HARD_CAP)
    spec = load_spec(FIXTURE)
    spec.verify_identity()
    policy = LifecyclePricingPolicy(
        policy_id=policy_id,
        input_usd_per_mtok=input_rate,
        output_usd_per_mtok=output_rate,
        rate_source=RATE_SOURCE_OPERATOR,
    )
    probe = ScriptedProvider(max_calls=max_calls)
    settings = check_static_settings(build_client(probe))
    budget = check_worst_case(policy, cap=cap, max_calls=max_calls)
    request = check_one_request(spec)
    reference = reference_cell(
        spec, directory=directory, policy=policy, cap=cap, max_calls=max_calls
    )
    faults = [
        fault_cell(
            spec,
            directory=directory,
            policy=policy,
            cap=cap,
            max_calls=max_calls,
            label="http_500",
            response=httpx.Response(500, json={"error": {"message": "mock"}}),
        ),
        fault_cell(
            spec,
            directory=directory,
            policy=policy,
            cap=cap,
            max_calls=max_calls,
            label="http_429",
            response=httpx.Response(429, json={"error": {"message": "mock"}}),
        ),
        fault_cell(
            spec,
            directory=directory,
            policy=policy,
            cap=cap,
            max_calls=max_calls,
            label="response_invalid",
            response=httpx.Response(200, json=wrong_model_body()),
        ),
    ]
    return {
        "preflight": "lifecycle_provider_evidence",
        "live_provider_call": False,
        "credential_read": False,
        "network_access": False,
        "transport": "httpx.MockTransport through the real OpenAI SDK",
        "refused_environment_variables": refused,
        "scenario_id": SCENARIO_ID,
        "model": GPT_5_6_LUNA_MODEL,
        "settings": settings,
        "budget": budget,
        "token_exposure": exposure,
        "request": request,
        "controls": controls_for(policy, cap=cap, max_calls=max_calls).as_dict(),
        "reference_cell": reference,
        "fault_cells": faults,
        "ok": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline preflight for Lifecycle provider execution evidence. Makes no "
            "live provider call and reads no credential."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "an existing directory, mode 0700, that the ledgers and the episode "
            "artefact are written to. It is not created here"
        ),
    )
    parser.add_argument(
        "--cost-cap-usd",
        default="10.00",
        help="the run's cost cap, as an exact decimal string",
    )
    parser.add_argument(
        "--input-usd-per-mtok",
        default="1.25",
        help=(
            "the operator's stated input rate per million tokens. Pinned for this "
            "run; this build makes no claim it is any vendor's current price"
        ),
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        default="10.00",
        help="the operator's stated output rate per million tokens, on the same terms",
    )
    parser.add_argument(
        "--pricing-policy-id",
        default="offline_operator_pinned_v1",
        help="the identity this run's rate table is recorded under",
    )
    parser.add_argument(
        "--max-provider-calls",
        type=int,
        default=DEFAULT_MAX_PROVIDER_CALLS,
        help=(
            f"the safety bound on provider calls, 1 to {MAX_PROVIDER_CALLS} "
            f"(default {DEFAULT_MAX_PROVIDER_CALLS}). The reference scenario is "
            f"measured to need {EXPECTED_PROVIDER_CALLS}. Fewer calls are allowed; "
            "more are refused, because this build's fixed token ceiling of "
            f"{TOKEN_HARD_CAP} is computed from the bound"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    # Before anything is constructed: a bound this build does not authorise is
    # an argument error, not a preflight result, and nothing about a provider —
    # not a client, not a transport, not a directory — is touched to report it.
    try:
        check_max_provider_calls(arguments.max_provider_calls)
        check_token_exposure(
            max_calls=arguments.max_provider_calls, token_hard_cap=TOKEN_HARD_CAP
        )
    except PreflightFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        amounts = [
            Decimal(arguments.cost_cap_usd),
            Decimal(arguments.input_usd_per_mtok),
            Decimal(arguments.output_usd_per_mtok),
        ]
    except InvalidOperation:
        print("every rate and cap is an exact decimal string", file=sys.stderr)
        return 2
    try:
        summary = preflight(
            directory=arguments.output_dir,
            cap=amounts[0],
            input_rate=amounts[1],
            output_rate=amounts[2],
            policy_id=arguments.pricing_policy_id,
            max_calls=arguments.max_provider_calls,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "preflight": "lifecycle_provider_evidence",
                    "ok": False,
                    "live_provider_call": False,
                    "credential_read": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
