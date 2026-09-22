"""The Lifecycle Track's projection onto the shared OpenAI Responses exchange.

What this file is for, in one sentence each.

*The prompt that was hashed is the prompt that was sent.* ``ModelAgent`` hashes
``request.prompt`` into ``prompt_digest_sha256`` and every recorded attempt
carries the request digest that covers it. If the bytes on the wire were a
re-rendering of that prompt rather than the prompt itself, every one of those
digests would be a claim about something the provider never saw. So the digest
is recomputed here from the *actual* ``httpx`` request body, over real
``ModelAgent`` requests from real maintenance episodes, on three scenarios and
across every invocation each of them makes.

*Non-strict tools, and the nulls they still allow.* OpenAI's strict
function-calling mode cannot state this build's outcome contract — it requires
every object closed and prohibits a union at the root of a tool's schema, and
Core needs an open ``ACT`` payload and a root liveness rule on ``WAIT``. So the
tools are sent ``strict=false`` (the adjudication is proved in
``tests/test_lifecycle_bridge_blockers.py``), ``required`` names only Core's own
required fields, and an optional field may simply be left out. The optionals
stay declared as nullable unions, because a model may still answer ``null`` for
a field it did not use and ``Act(payload=None)`` is not an ``ACT``: those nulls
are elided before the outcome parser sees them. Both encodings are pinned here,
for all five outcomes — omitted and null — and so is the control: without the
elision, every null answer traps as ``MODEL_MALFORMED_ARGUMENTS``.

*Every other way an answer is not a decision keeps its existing name.* Unknown,
zero and multiple tool calls, undecodable and duplicate-keyed and non-object
arguments, a required field left null, truncation and an over-limit output all
map into classifications this build already has, rather than into a guess.

*No socket, no credential, no provider.* Every client is a real
``openai.OpenAI`` over ``httpx.MockTransport``. Nothing here reads an
environment variable, builds a live client or opens a connection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from openai.types.responses import Response

from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    provider_identity_for,
)
from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    TOOL_NAMES,
    ModelAgent,
    ModelBoundaryError,
    ModelExceededOutputLimit,
    ModelNamedAnUnknownTool,
    ModelOutputTruncated,
    ModelReturnedMultipleToolCalls,
    ModelReturnedNoToolCall,
    ModelToolArgumentsMalformed,
    parse_tool_call,
    tool_schema,
)
from operatebench.agents.openai_responses import (
    COMPLETED_STOP_REASON,
    LIFECYCLE_NULL_ELISION,
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    LIFECYCLE_RETRY_POLICY,
    MAX_TRANSIENT_TEXT_CHARACTERS,
    ModelRequestRefused,
    OpenAIResponsesTransport,
    PromptDigestMismatch,
    RequestModelMismatch,
    RequestPromptStructureError,
    RequestProtocolMismatch,
    RequestToolNamesMismatch,
    ToolArgumentsUndecodable,
    build_model_payload,
    decode_tool_arguments,
    elide_null_optionals,
    lifecycle_openai_settings,
    model_response_from,
    outcome_tools,
    request_instructions,
)
from operatebench.agents.outcome_contract import (
    AGENT_TOOL_NAMES,
    AGENT_TOOLS,
    EVIDENCE_REFS_DESCRIPTION,
    FIELD_KINDS,
    FIELD_REFERENCE_LIST,
    OUTCOME_TOOLS,
    RETRIEVE_TOOL,
    OutcomeField,
    field_kinds,
    optional_field_names,
    outcome_fields,
    required_field_names,
)
from operatebench.agents.transport import (
    ModelRequest,
    ProviderFailure,
    ToolCall,
    content_digest,
)
from operatebench.core.outcomes import Act, Ask, Complete, Escalate, Wait
from operatebench.core.protocol import MODEL_VISIBLE_FIELDS, AgentObservation
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.domains.lettings.maintenance.spec import OperationSpec, load_spec
from operatebench.execution_ledger import read_execution_ledger
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.openai_responses import (
    BASELINE_PROFILE,
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    OpenAIResponsesExchange,
)
from operatebench.providers.wire import MAX_PROVIDER_RESPONSE_ID_CHARACTERS, WireResponse
from operatebench.runner import run_episode
from tests.model_transport import PROVIDER_PROSE, observation_from_dict
from tests.openai_transport import (
    FAKE_API_KEY,
    RecordingTransport,
    function_call_item,
    message_item,
    responses_body,
    scripted_client,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "examples/operatebench/maintenance_v0_1.yaml"
)

MODEL = GPT_5_6_LUNA_MODEL
MODEL_AGENT_ID = "model_reference"

#: The scenarios whose real requests are put on the wire.
SCENARIOS: tuple[str, ...] = ("V1", "V2", "V3")

#: A wall-clock deadline for a turn. Explicit because the transport requires it.
DEADLINE_SECONDS = 30.0


@pytest.fixture(scope="module")
def spec() -> OperationSpec:
    return load_spec(FIXTURE)


def fixed_clock(step: float = 0.25) -> Callable[[], float]:
    """A deterministic monotonic clock: no wall time enters a test."""
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += step
        return ticks[0]

    return clock


def no_sleep(_seconds: float) -> None:
    raise AssertionError("this build makes one attempt, so nothing may back off")


def transport_for(
    client: openai.OpenAI,
    *,
    model: str = MODEL,
    before_dispatch: Callable[[], None] | None = None,
) -> OpenAIResponsesTransport:
    return OpenAIResponsesTransport(
        model=model,
        client=client,
        deadline_seconds=DEADLINE_SECONDS,
        sleep=no_sleep,
        clock=fixed_clock(),
        before_dispatch=before_dispatch,
    )


def test_dispatch_observer_runs_immediately_before_real_sdk_request() -> None:
    order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        order.append("sdk_transport")
        return httpx.Response(
            200,
            json=responses_body(
                [function_call_item("complete", '{"summary":"done"}')], model=MODEL
            ),
            request=request,
        )

    wire = RecordingTransport([])
    client = openai.OpenAI(
        api_key=FAKE_API_KEY,
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    agent = ModelAgent(
        transport_for(client, before_dispatch=lambda: order.append("observer")),
        model=MODEL,
        agent_id=MODEL_AGENT_ID,
    )
    agent.begin_episode({"operation_id": "op", "agent_id": MODEL_AGENT_ID})

    agent.decide(observation())

    assert order == ["observer", "sdk_transport"]
    assert wire.calls == 0


def test_dispatch_observer_failure_records_zero_attempts_and_sends_nothing() -> None:
    wire, client = scripted_client()
    marker = RuntimeError("synthetic observer refusal")
    lane = transport_for(client, before_dispatch=lambda: (_ for _ in ()).throw(marker))
    request = ModelAgent(
        _NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID
    ).build_request(observation())

    with pytest.raises(RuntimeError) as caught:
        lane.send(request)

    assert caught.value is marker
    assert wire.calls == 0
    assert lane.last_telemetry().attempts == ()


def observation(**overrides: Any) -> AgentObservation:
    base: dict[str, Any] = {
        "now": "2025-01-01T09:00:00Z",
        "operation_id": "op",
        "operation_instance_id": "opinst_0123456789abcdef0123456789abcdef",
        "invocation_index": 1,
        "turn_index": 0,
        "policy": {},
        "actors": {},
        "message_fixture_ids": [],
        "last_rejection": None,
    }
    base.update(overrides)
    return AgentObservation(**base)


def agent_over(client: openai.OpenAI, *, model: str = MODEL) -> ModelAgent:
    agent = ModelAgent(
        transport_for(client, model=model), model=model, agent_id=MODEL_AGENT_ID
    )
    agent.begin_episode({"operation_id": "op", "agent_id": MODEL_AGENT_ID})
    return agent


def decide_over(body: Mapping[str, Any]) -> tuple[Any, RecordingTransport]:
    """One decision, taken over one scripted Responses body."""
    wire, client = scripted_client(dict(body))
    decision = agent_over(client).decide(observation())
    return decision, wire


@pytest.mark.parametrize("ceiling", [2048, 4096.0, True])
def test_openai_noncanonical_output_ceiling_refuses_before_dispatch(ceiling: Any) -> None:
    wire, client = scripted_client()
    lane = transport_for(client)
    request = ModelAgent(
        _NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID
    ).build_request(observation())
    request = replace(request, max_output_tokens=ceiling)

    with pytest.raises(ModelRequestRefused, match="output-token ceiling"):
        lane.send(request)

    assert wire.calls == 0


def user_text(body: Mapping[str, Any]) -> str:
    """The one user message text a Lifecycle request puts on the wire."""
    entries = body["input"]
    assert len(entries) == 1, "one request is one user message"
    assert entries[0]["role"] == "user"
    content = entries[0]["content"]
    assert isinstance(content, str)
    return content


# ------------------------------------------------- a reference-driven provider


class CapturingTransport:
    """The Lifecycle transport, with the requests it was handed kept beside it.

    A wrapper and nothing else: it appends and delegates, so the payload that
    reaches the exchange is the one the transport built. It exists because the
    claim under test is about *the request ``ModelAgent`` hashed*, and that
    object is otherwise never visible from outside the agent.
    """

    def __init__(self, transport: OpenAIResponsesTransport) -> None:
        self._transport = transport
        self.requests: list[ModelRequest] = []

    def send(self, request: ModelRequest) -> Any:
        self.requests.append(request)
        return self._transport.send(request)


class ReferenceDrivenOpenAI:
    """A ``MockTransport`` handler that answers as the reference agent would.

    The only honest way to get *real* ``ModelAgent`` requests for a whole
    eleven-day maintenance episode onto a wire: each request body is read back,
    the public observation it carries is handed to the shipped reference agent,
    and what that agent decides is encoded as an OpenAI ``function_call`` with
    every declared property present and semantic optionals sent as ``null``.
    The actual wire tools are non-strict; this deliberately verbose answer shape
    exercises the projection and null elision while its decisions remain the
    reference's.

    Prose is attached to every answer so a durable-evidence test has a marker to
    look for.
    """

    def __init__(self, *, agent_id: str = "reference") -> None:
        self._agent = build_agent(agent_id)
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        #: Every answer this handler produced, as ``(tool name, arguments)``.
        #: Kept so a test can ask what the provider *said* rather than only what
        #: the run made of it.
        self.answers: list[tuple[str, dict[str, Any]]] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        self.bodies.append(body)
        prompt = json.loads(user_text(body))
        outcome = self._agent.decide(observation_from_dict(prompt["observation"]))
        name, arguments = strict_arguments(outcome)
        self.answers.append((name, arguments))
        return httpx.Response(
            200,
            json=responses_body(
                [
                    message_item(PROVIDER_PROSE),
                    function_call_item(name, json.dumps(arguments)),
                ],
                model=MODEL,
            ),
            request=request,
        )

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="test-key-not-a-credential",
            base_url="https://api.openai.com/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


def strict_wait(wait: Wait) -> dict[str, Any]:
    return {
        "reason": wait.reason,
        "wake_on": list(wait.wake_on) or None,
        "fallback_after_minutes": wait.fallback_after_minutes,
    }


def strict_arguments(outcome: Any) -> tuple[str, dict[str, Any]]:
    """One Core object as strict-mode arguments: every property, nulls and all."""
    if isinstance(outcome, RetrieveBatch):
        return "retrieve", {
            "requests": [
                {"tool": request.tool, "arguments": dict(request.arguments) or None}
                for request in outcome.requests
            ]
        }
    if isinstance(outcome, Act):
        return "act", {
            "action_type": outcome.action_type,
            # The reference agent's ``ACT``s carry real payloads, and under this
            # mapping version the tool surface can state one. Sending it is what
            # makes the episode below a test of the *bridge* rather than of the
            # narrowing the bridge used to impose.
            "payload": dict(outcome.payload) or None,
            "evidence_refs": list(outcome.evidence_refs) or None,
            "rationale": outcome.rationale or None,
        }
    if isinstance(outcome, Wait):
        return "wait", strict_wait(outcome)
    if isinstance(outcome, Ask):
        return "ask", {
            "recipient_actor_id": outcome.recipient_actor_id,
            "message_fixture_id": outcome.message_fixture_id,
            "correlation_id": outcome.correlation_id,
            "wait": strict_wait(outcome.wait),
        }
    if isinstance(outcome, Escalate):
        return "escalate", {
            "checkpoint_id": outcome.checkpoint_id,
            "exception_type": outcome.exception_type,
            "evidence_refs": list(outcome.evidence_refs) or None,
            "deadline_after_minutes": outcome.deadline_after_minutes,
            "rationale": outcome.rationale or None,
        }
    return "complete", {
        "reason": outcome.reason or None,
        "evidence_refs": list(outcome.evidence_refs) or None,
    }


def episode_over_the_wire(
    spec: OperationSpec, scenario_id: str, *, ledger_path: Path | None = None
) -> tuple[ReferenceDrivenOpenAI, CapturingTransport, Any]:
    """One real maintenance episode whose every decision crossed an SDK client.

    ``ledger_path`` attaches a provider evidence recorder to the transport
    underneath the capture. It is needed by exactly the tests that build an
    episode artefact: contract 7 writes none for a model run with no complete
    scored journal behind it. Every other test here is about the request and the
    response rather than the record, and gets the plain agent it always had.
    """
    provider = ReferenceDrivenOpenAI()
    inner = transport_for(provider.client())
    capture = CapturingTransport(inner)
    if ledger_path is None:
        return (
            provider,
            capture,
            run_episode(
                spec,
                scenario_id,
                MODEL_AGENT_ID,
                agent_factory=lambda: ModelAgent(
                    capture, model=MODEL, agent_id=MODEL_AGENT_ID
                ),
                agent_kind="model",
            ),
        )
    recorder = _recorder_for(inner, spec, scenario_id, ledger_path)
    inner.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        capture, model=MODEL, agent_id=MODEL_AGENT_ID, recorder=recorder
    )
    run = run_episode(
        spec,
        scenario_id,
        MODEL_AGENT_ID,
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    return provider, capture, run


def _recorder_for(
    transport: OpenAIResponsesTransport,
    spec: OperationSpec,
    scenario_id: str,
    path: Path,
) -> ProviderEvidenceRecorder:
    from tests.provider_evidence_runs import controls

    return ProviderEvidenceRecorder(
        path=path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=scenario_id,
            agent_id=MODEL_AGENT_ID,
        ),
        provider=provider_identity_for(transport),
        controls=controls(max_calls=200),
    )


@pytest.mark.parametrize(
    ("case", "valid", "expected_digest"),
    [
        ("lone-surrogate", False, None),
        (
            "maximum",
            True,
            hashlib.sha256(b"a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS).hexdigest(),
        ),
        ("over-maximum", False, None),
        ("empty", True, None),
        ("absent", True, None),
        ("invalid-type", False, None),
    ],
)
def test_openai_response_id_is_admitted_before_settlement_and_recording(
    tmp_path: Path,
    spec: OperationSpec,
    case: str,
    valid: bool,
    expected_digest: str | None,
) -> None:
    body = responses_body(
        [
            function_call_item(
                "complete", json.dumps({"reason": "done", "evidence_refs": []})
            )
        ],
        model=MODEL,
    )
    if case == "lone-surrogate":
        body["id"] = "\ud800"
    elif case == "maximum":
        body["id"] = "a" * MAX_PROVIDER_RESPONSE_ID_CHARACTERS
    elif case == "over-maximum":
        body["id"] = "a" * (MAX_PROVIDER_RESPONSE_ID_CHARACTERS + 1)
    elif case == "empty":
        body["id"] = ""
    elif case == "absent":
        body.pop("id")
    else:
        body["id"] = 7
    raw = json.dumps(body).encode("ascii")
    wire, client = scripted_client(raw)
    lane = transport_for(client)
    path = tmp_path / f"openai-response-id-{case}.ndjson"
    recorder = _recorder_for(lane, spec, "V1", path)
    recorder.begin(operation_instance_id="opinst_" + "5" * 32)
    lane.attach_recorder(recorder)
    request = ModelAgent(
        _NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID
    ).build_request(observation())

    if valid:
        lane.send(request)
    else:
        with pytest.raises(AdapterProviderError) as raised:
            lane.send(request)
        assert raised.value.fault == "provider_response_invalid"
        assert lane.last_usage().input_tokens is None
        assert lane.last_telemetry().attempts[0].usage_reported is False
    recorder.abandon_open_call()

    assert wire.calls == 1
    attempt = read_execution_ledger(path).calls[0].attempts[0]
    assert attempt.response_id_digest_sha256 == expected_digest
    if not valid:
        assert attempt.response_model is None
        assert attempt.response_stop_classification is None
        assert attempt.response_normalized_digest_sha256 is None
        persisted = path.read_text(encoding="utf-8")
        assert "ud800" not in persisted
        assert "UnicodeEncodeError" not in persisted


# ------------------------------------------------------ the exact request bind


class TestTheExactRequestBinding:
    """What ``ModelAgent`` hashed is what the provider was sent, byte for byte."""

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_the_wire_user_text_digest_is_the_prompt_digest_of_every_request(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        provider, capture, _run = episode_over_the_wire(spec, scenario_id)

        assert provider.calls > 1, "an episode makes more than one invocation"
        assert len(capture.requests) == provider.calls
        for request, body in zip(capture.requests, provider.bodies, strict=True):
            text = user_text(body)
            # The claim, stated over the bytes that actually left: the exact
            # prompt this request was hashed under is the user content.
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == (
                request.prompt_digest_sha256
            )
            assert json.loads(text) == dict(request.prompt)

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_the_wire_observation_digest_is_the_one_every_request_is_recorded_under(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        """The other digest, over the bytes that actually left.

        ``observation_digest_sha256`` is what playback replays a recorded
        decision against, and the transport now refuses a request whose
        observation does not hash to it. This is the same claim measured from
        outside: for every invocation of three real episodes, the observation
        embedded in the request body hashes to the digest the run recorded — so
        the binding holds on the traffic as well as on the refusal path.
        """
        provider, capture, _run = episode_over_the_wire(spec, scenario_id)

        assert provider.calls > 1
        for request, body in zip(capture.requests, provider.bodies, strict=True):
            observed = json.loads(user_text(body))["observation"]
            assert content_digest(observed) == request.observation_digest_sha256
            assert tuple(sorted(observed)) == tuple(sorted(MODEL_VISIBLE_FIELDS))

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_the_request_digest_of_every_request_is_bound_into_the_instructions(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        provider, capture, _run = episode_over_the_wire(spec, scenario_id)

        for request, body in zip(capture.requests, provider.bodies, strict=True):
            assert request.request_digest_sha256 in body["instructions"]
            assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION in body["instructions"]

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_no_run_internal_label_reaches_the_wire(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        provider, _capture, _run = episode_over_the_wire(spec, scenario_id)

        for body in provider.bodies:
            rendered = json.dumps(body).lower()
            for label in (
                "scenario",
                "expected_terminal",
                "oracle",
                "spec_digest",
                "authored",
                "semantic_scenario",
            ):
                assert label not in rendered, label

    def test_the_user_text_is_the_prompt_and_not_a_re_rendering_of_it(self) -> None:
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        agent = agent_over(client)
        request = agent.build_request(observation())

        agent.decide(observation())

        text = user_text(wire.bodies[0])
        assert json.loads(text) == dict(request.prompt)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == (
            request.prompt_digest_sha256
        )

    def test_the_payload_the_projection_builds_is_the_payload_that_is_sent(self) -> None:
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        agent = agent_over(client)
        request = agent.build_request(observation())

        agent.decide(observation())

        expected = build_model_payload(request, model=MODEL)
        assert json.loads(json.dumps(expected)) == wire.bodies[0]

    def test_the_instructions_are_fixed_text_plus_this_build_s_own_digest(self) -> None:
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        rendered = request_instructions(request)

        assert request.request_digest_sha256 in rendered
        assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION in rendered
        assert MODEL_PROTOCOL_VERSION in rendered


class _NeverCalled:
    """A transport for the paths that build a request and send nothing."""

    def send(self, request: ModelRequest) -> Any:  # pragma: no cover - never called
        raise AssertionError("this path must not reach a provider")


# ------------------------------------------------------------ the request shape


class TestTheRequestProfileOnTheWire:
    """The body is exactly what the pinned model's frozen profile states."""

    def test_the_body_carries_exactly_the_fields_the_luna_profile_sends(self) -> None:
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        agent_over(client).decide(observation())

        body = wire.bodies[0]
        assert tuple(sorted(body)) == GPT_5_6_LUNA_PROFILE.sent_fields()
        assert "temperature" not in body
        assert "top_p" not in body
        assert body["reasoning"] == {"effort": "none"}
        assert body["store"] is False
        assert body["parallel_tool_calls"] is False
        assert body["tool_choice"] == "required"
        assert body["model"] == MODEL
        assert body["max_output_tokens"] == MAX_OUTPUT_TOKENS

    def test_a_turn_makes_exactly_one_request_and_never_backs_off(self) -> None:
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        agent_over(client).decide(observation())

        assert wire.calls == 1
        assert LIFECYCLE_RETRY_POLICY.max_attempts == 1

    def test_the_mapping_version_and_settings_are_inspectable(self) -> None:
        _wire, client = scripted_client()
        transport = transport_for(client)

        settings = transport.settings

        assert settings == lifecycle_openai_settings(
            model=MODEL, deadline_seconds=DEADLINE_SECONDS
        )
        assert settings["request_mapping"] == LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION
        assert settings["request_profile"] == GPT_5_6_LUNA_PROFILE.profile_id
        assert settings["retry"] == LIFECYCLE_RETRY_POLICY.as_dict()
        assert settings["null_elision"] == LIFECYCLE_NULL_ELISION
        assert settings["turn_deadline_seconds"] == DEADLINE_SECONDS
        assert transport.model == MODEL
        assert transport.request_profile == GPT_5_6_LUNA_PROFILE

    def test_a_deadline_that_is_not_a_positive_wall_clock_budget_is_refused(self) -> None:
        _wire, client = scripted_client()
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ModelBoundaryError):
                OpenAIResponsesTransport(model=MODEL, client=client, deadline_seconds=bad)


# ---------------------------------------------------- the non-strict tool surface


class TestTheNonStrictToolSurface:
    """Six tools, one contract, and the exact schemas Core's semantics need."""

    def test_the_provider_request_mapping_moves_with_evidence_reference_guidance(
        self,
    ) -> None:
        assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION == (
            "lifecycle_openai_responses_model_request_v9"
        )

    def test_every_evidence_reference_field_states_the_same_canonical_contract(
        self,
    ) -> None:
        exposed = {
            name: next(field for field in fields if field.name == "evidence_refs")
            for name, fields in OUTCOME_TOOLS.items()
            if any(field.name == "evidence_refs" for field in fields)
        }

        assert tuple(exposed) == ("act", "escalate", "complete")
        assert {field.description for field in exposed.values()} == {
            EVIDENCE_REFS_DESCRIPTION
        }
        assert all(field.required is False for field in exposed.values())
        assert all(field.kind == FIELD_REFERENCE_LIST for field in exposed.values())

    def test_prompt_summary_guides_only_tools_with_evidence_references(
        self,
    ) -> None:
        summary = tool_schema()
        guided = {"act", "escalate", "complete"}

        for name in AGENT_TOOL_NAMES:
            expected = {
                "required": list(required_field_names(name)),
                "optional": list(optional_field_names(name)),
            }
            if name in guided:
                expected["field_guidance"] = {"evidence_refs": EVIDENCE_REFS_DESCRIPTION}
            assert summary[name] == expected

    def test_provider_schemas_still_project_every_canonical_field_description(
        self,
    ) -> None:
        provider = {tool["name"]: tool for tool in outcome_tools()}

        for name in AGENT_TOOL_NAMES:
            properties = provider[name]["parameters"]["properties"]
            assert {
                field_name: schema["description"]
                for field_name, schema in properties.items()
            } == {field.name: field.description for field in outcome_fields(name)}

    def test_prompt_guidance_and_provider_descriptions_are_read_from_the_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import operatebench.agents.outcome_contract as contract

        evidence_probe = "canonical evidence projection probe"
        unrelated_probe = "canonical unrelated projection probe"
        fields = tuple(
            replace(
                field,
                description=(
                    evidence_probe if field.name == "evidence_refs" else unrelated_probe
                ),
            )
            for field in outcome_fields("act")
        )
        monkeypatch.setattr(contract, "AGENT_TOOLS", {**AGENT_TOOLS, "act": fields})

        summary = tool_schema()["act"]
        projected = next(tool for tool in outcome_tools() if tool["name"] == "act")

        assert summary["field_guidance"] == {"evidence_refs": evidence_probe}
        assert unrelated_probe not in json.dumps(summary)
        assert (
            projected["parameters"]["properties"]["evidence_refs"]["description"]
            == evidence_probe
        )
        assert (
            projected["parameters"]["properties"]["action_type"]["description"]
            == unrelated_probe
        )

    def test_internal_case_labels_cannot_enter_contract_prompt_or_provider_schema(
        self,
    ) -> None:
        poisoned = {
            "scenario_id": "POISON_SCENARIO",
            "arm_id": "POISON_ARM",
            "expected_outcome": "POISON_EXPECTED",
        }
        observed = observation()
        for name, value in poisoned.items():
            object.__setattr__(observed, name, value)
        request = ModelAgent(
            RecordingTransport([]), model=MODEL, agent_id=MODEL_AGENT_ID
        ).build_request(observed)
        surfaces = (
            json.dumps(
                {
                    name: [field.description for field in fields]
                    for name, fields in OUTCOME_TOOLS.items()
                }
            ),
            json.dumps(request.prompt),
            json.dumps(outcome_tools()),
        )

        assert tuple(sorted(request.prompt["observation"])) == tuple(
            sorted(MODEL_VISIBLE_FIELDS)
        )
        for surface in surfaces:
            assert all(value not in surface for value in poisoned.values())

    def test_every_offered_tool_is_named_by_the_contract(self) -> None:
        tools = outcome_tools()

        # The five outcomes, in contract order, then the read. ``retrieve`` is
        # offered beside them and is deliberately not one of them: it commits
        # nothing, and Core's outcome union is unchanged.
        assert [tool["name"] for tool in tools] == list(AGENT_TOOL_NAMES)
        assert [tool["name"] for tool in tools[: len(TOOL_NAMES)]] == list(TOOL_NAMES)
        assert tools[-1]["name"] == RETRIEVE_TOOL
        for tool in tools:
            assert tool["type"] == "function"
            assert tool["strict"] is False

    def test_every_object_that_declares_a_shape_is_closed(self) -> None:
        for tool in outcome_tools():
            _assert_closed(tool["parameters"])

    def test_required_is_the_contract_s_required_fields_and_no_others(self) -> None:
        """``required`` states Core's rule, not a provider mode's rule.

        Under strict mode every property had to be in ``required`` whatever the
        contract said, so the list on the wire was a statement about OpenAI
        rather than about the outcome. Non-strict, the two agree: what the
        parser requires is what the schema requires.
        """
        for tool in outcome_tools():
            name = tool["name"]
            parameters = tool["parameters"]
            assert parameters["required"] == list(required_field_names(name)), name
            for optional in optional_field_names(name):
                assert optional in parameters["properties"], (name, optional)
                assert optional not in parameters["required"], (name, optional)

    def test_a_tool_with_no_required_fields_states_an_empty_required(self) -> None:
        """``COMPLETE`` requires nothing, and says so rather than saying nothing.

        An absent ``required`` and an empty one mean the same thing to a reader
        of JSON Schema. They do not mean the same thing to a reader of a
        *recorded request*, which is what this build is auditable through: the
        empty list is the difference between "this build decided nothing is
        required" and "this build forgot".
        """
        complete = next(
            tool["parameters"] for tool in outcome_tools() if tool["name"] == "complete"
        )

        assert required_field_names("complete") == ()
        assert complete["required"] == []

    def test_the_declared_types_are_the_types_the_outcome_contract_enforces(self) -> None:
        by_name = {
            tool["name"]: tool["parameters"]["properties"] for tool in outcome_tools()
        }

        act = by_name["act"]
        assert act["action_type"]["type"] == "string"
        assert act["payload"]["type"] == ["object", "null"]
        assert act["evidence_refs"]["type"] == ["array", "null"]
        assert act["evidence_refs"]["items"]["type"] == "string"

        wait = by_name["wait"]
        assert wait["fallback_after_minutes"]["type"] == ["integer", "null"]
        assert wait["fallback_after_minutes"]["minimum"] == 1

        ask = by_name["ask"]
        # The nested wait is a union of its liveness branches rather than one
        # permissive object, so it has no ``type`` of its own — see
        # ``TestTheWaitLivenessIsInTheSchema`` in the blockers file. Each branch
        # is still a whole, closed ``WAIT``.
        assert "type" not in ask["wait"]
        assert ask["correlation_id"]["type"] == ["string", "null"]
        _assert_closed(ask["wait"])
        for branch in ask["wait"]["anyOf"]:
            assert branch["type"] == "object"

        escalate = by_name["escalate"]
        assert escalate["deadline_after_minutes"]["type"] == ["integer", "null"]
        assert escalate["deadline_after_minutes"]["minimum"] == 1

    def test_the_schema_and_the_parser_read_one_field_contract(self) -> None:
        summary = tool_schema()

        assert set(summary) == set(AGENT_TOOLS)
        for name in AGENT_TOOL_NAMES:
            assert summary[name]["required"] == list(required_field_names(name))
            assert summary[name]["optional"] == list(optional_field_names(name))
            properties = next(
                tool["parameters"]["properties"]
                for tool in outcome_tools()
                if tool["name"] == name
            )
            assert set(properties) == set(summary[name]["required"]) | set(
                summary[name]["optional"]
            )

    def test_the_contract_states_a_kind_this_projection_knows_for_every_field(
        self,
    ) -> None:
        """Every field is projectable: no kind reaches the schema unhandled."""
        for name, fields in OUTCOME_TOOLS.items():
            for declared in fields:
                assert declared.kind in FIELD_KINDS, (name, declared.name)
            assert set(FIELD_KINDS) >= {field.kind for field in fields}


def _assert_closed(schema: Mapping[str, Any]) -> None:
    """One JSON-Schema node, closed and coherent wherever it declares a shape.

    Two claims, and neither of them is strict mode's. An object that states
    ``properties`` is a shape this build claims to know, so it is closed against
    keys it does not name — an unknown argument is a malformed call and the
    schema should say so. And every name in ``required`` is a property the
    schema declares, which is the internal coherence check; *which* names those
    are is Core's question and is asserted against the contract in
    ``test_required_is_the_contract_s_required_fields_and_no_others``.

    An object that states no properties is the free-form ``ACT`` payload Core
    accepts, and closing it would be this projection inventing a contract — so
    it is walked past rather than asserted about, and
    ``TestTheActPayloadIsAnOpenObject`` in
    ``tests/test_lifecycle_bridge_blockers.py`` pins that exactly one node in
    the whole surface is that node.
    """
    for branch in schema.get("anyOf", ()):
        _assert_closed(branch)
    properties = schema.get("properties")
    if properties is None:
        return
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) <= set(properties)
    for child in properties.values():
        if "anyOf" in child:
            _assert_closed(child)
            continue
        declared = child.get("type")
        if declared == "object" or (isinstance(declared, list) and "object" in declared):
            _assert_closed(child)
        if declared == "array" or (isinstance(declared, list) and "array" in declared):
            item = child["items"]
            item_type = item.get("type")
            if item_type == "object" or (
                isinstance(item_type, list) and "object" in item_type
            ):
                _assert_closed(item)


# ------------------------------------------------------------- the null elision


#: One null-optional answer per outcome: every property present, semantic
#: optionals sent as ``null``, and each one chosen so that the *un-elided*
#: arguments trap on the outcome contract rather than parsing anyway.
NULL_OPTIONAL_ANSWERS: tuple[tuple[str, dict[str, Any], Any], ...] = (
    (
        "act",
        {
            "action_type": "open_cycle",
            "payload": None,
            "evidence_refs": None,
            "rationale": None,
        },
        Act(action_type="open_cycle"),
    ),
    (
        "wait",
        {
            "reason": "awaiting the contractor",
            "wake_on": None,
            "fallback_after_minutes": 90,
        },
        Wait(reason="awaiting the contractor", fallback_after_minutes=90),
    ),
    (
        "ask",
        {
            "recipient_actor_id": "tenant",
            "message_fixture_id": "fixture_1",
            "correlation_id": None,
            "wait": {
                "reason": "awaiting the answer",
                "wake_on": None,
                "fallback_after_minutes": 45,
            },
        },
        Ask(
            recipient_actor_id="tenant",
            message_fixture_id="fixture_1",
            wait=Wait(reason="awaiting the answer", fallback_after_minutes=45),
        ),
    ),
    (
        "escalate",
        {
            "checkpoint_id": "cp_1",
            "exception_type": "AUTHORITY_LIMIT",
            "evidence_refs": None,
            "deadline_after_minutes": None,
            "rationale": None,
        },
        Escalate(checkpoint_id="cp_1", exception_type="AUTHORITY_LIMIT"),
    ),
    ("complete", {"reason": None, "evidence_refs": None}, Complete()),
)


class TestTheNullElision:
    """Non-strict answers may omit optionals or state null; both are pinned."""

    @pytest.mark.parametrize(
        ("name", "arguments", "expected"),
        NULL_OPTIONAL_ANSWERS,
        ids=[entry[0] for entry in NULL_OPTIONAL_ANSWERS],
    )
    def test_each_null_optional_answer_reaches_the_outcome_it_names(
        self, name: str, arguments: dict[str, Any], expected: Any
    ) -> None:
        decision, wire = decide_over(
            responses_body([function_call_item(name, json.dumps(arguments))], model=MODEL)
        )

        assert decision == expected
        assert wire.calls == 1

    @pytest.mark.parametrize(
        ("name", "arguments", "expected"),
        NULL_OPTIONAL_ANSWERS,
        ids=[entry[0] for entry in NULL_OPTIONAL_ANSWERS],
    )
    def test_each_omitted_optional_answer_reaches_the_outcome_it_names(
        self, name: str, arguments: dict[str, Any], expected: Any
    ) -> None:
        omitted = elide_null_optionals(name, arguments)
        decision, wire = decide_over(
            responses_body([function_call_item(name, json.dumps(omitted))], model=MODEL)
        )

        assert decision == expected
        assert wire.calls == 1

    @pytest.mark.parametrize(
        ("name", "arguments", "expected"),
        NULL_OPTIONAL_ANSWERS,
        ids=[entry[0] for entry in NULL_OPTIONAL_ANSWERS],
    )
    def test_removing_the_elision_reproduces_the_malformed_arguments_trap(
        self, name: str, arguments: dict[str, Any], expected: Any
    ) -> None:
        """The control. Without elision every one of these is a classification."""
        trapped = parse_tool_call(ToolCall(name, arguments))

        assert isinstance(trapped, ModelToolArgumentsMalformed), name
        assert parse_tool_call(ToolCall(name, elide_null_optionals(name, arguments))) == (
            expected
        )

    def test_a_required_field_left_null_stays_and_is_classified(self) -> None:
        decision, wire = decide_over(
            responses_body(
                [
                    function_call_item(
                        "act",
                        json.dumps(
                            {
                                "action_type": None,
                                "payload": None,
                                "evidence_refs": None,
                                "rationale": None,
                            }
                        ),
                    )
                ],
                model=MODEL,
            )
        )

        assert isinstance(decision, ModelToolArgumentsMalformed)
        assert wire.calls == 1

    def test_the_elision_reaches_the_wait_nested_inside_an_ask(self) -> None:
        arguments = {
            "recipient_actor_id": "tenant",
            "message_fixture_id": "fixture_1",
            "correlation_id": None,
            "wait": {
                "reason": "awaiting",
                "wake_on": ["contractor_replied"],
                "fallback_after_minutes": None,
            },
        }

        elided = elide_null_optionals("ask", arguments)

        assert elided == {
            "recipient_actor_id": "tenant",
            "message_fixture_id": "fixture_1",
            "wait": {"reason": "awaiting", "wake_on": ["contractor_replied"]},
        }

    def test_an_unknown_field_survives_the_elision_and_is_still_refused(self) -> None:
        """Elision drops nulls this build declared, never fields it did not."""
        elided = elide_null_optionals(
            "complete", {"reason": "done", "evidence_refs": None, "invented": None}
        )

        assert "invented" in elided
        assert isinstance(
            parse_tool_call(ToolCall("complete", elided)), ModelToolArgumentsMalformed
        )


# ------------------------------------------------- every other way it is not one


class TestTheNamedClassifications:
    """An answer that is not a decision keeps the name this build already has."""

    def test_no_tool_call_at_all(self) -> None:
        decision, wire = decide_over(
            responses_body([message_item(PROVIDER_PROSE)], model=MODEL)
        )

        assert isinstance(decision, ModelReturnedNoToolCall)
        assert wire.calls == 1

    def test_more_than_one_tool_call(self) -> None:
        decision, wire = decide_over(
            responses_body(
                [
                    function_call_item("complete", json.dumps({"reason": "a"})),
                    function_call_item("complete", json.dumps({"reason": "b"})),
                ],
                model=MODEL,
            )
        )

        assert isinstance(decision, ModelReturnedMultipleToolCalls)
        assert wire.calls == 1

    def test_a_tool_this_build_does_not_offer(self) -> None:
        decision, wire = decide_over(
            responses_body([function_call_item("teleport", "{}")], model=MODEL)
        )

        assert isinstance(decision, ModelNamedAnUnknownTool)
        assert wire.calls == 1

    @pytest.mark.parametrize(
        "arguments",
        ['{"reason":', '["reason"]', '"reason"', "17", "null"],
        ids=["undecodable", "array", "string", "number", "null"],
    )
    def test_arguments_that_are_not_a_json_object(self, arguments: str) -> None:
        decision, wire = decide_over(
            responses_body([function_call_item("complete", arguments)], model=MODEL)
        )

        assert isinstance(decision, ModelToolArgumentsMalformed)
        assert wire.calls == 1

    def test_arguments_that_name_the_same_key_twice(self) -> None:
        decision, wire = decide_over(
            responses_body(
                [function_call_item("complete", '{"reason": "a", "reason": "b"}')],
                model=MODEL,
            )
        )

        assert isinstance(decision, ModelToolArgumentsMalformed)
        assert wire.calls == 1

    def test_an_answer_truncated_at_the_output_ceiling(self) -> None:
        decision, wire = decide_over(
            responses_body(
                [],
                status="incomplete",
                incomplete_reason="max_output_tokens",
                model=MODEL,
            )
        )

        assert isinstance(decision, ModelOutputTruncated)
        assert wire.calls == 1

    def test_an_answer_reporting_more_output_than_the_pinned_ceiling(self) -> None:
        decision, wire = decide_over(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                output_tokens=MAX_OUTPUT_TOKENS + 1,
                model=MODEL,
            )
        )

        assert isinstance(decision, ModelExceededOutputLimit)
        assert wire.calls == 1

    def test_an_answer_from_another_model_is_refused_before_it_is_read(self) -> None:
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model="some-other-model",
            )
        )
        agent = agent_over(client)

        with pytest.raises(ProviderFailure) as caught:
            agent.decide(observation())

        assert wire.calls == 1
        assert caught.value.record.excluded is True


class TestTheRefusalsBeforeDispatch:
    """A request this transport cannot honestly send never becomes a request."""

    def test_a_request_naming_another_model_than_the_exchange_is_refused(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model="another-model", agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        with pytest.raises(RequestModelMismatch):
            transport.send(request)

        assert wire.calls == 0

    def test_a_tampered_prompt_digest_is_refused(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        with pytest.raises(PromptDigestMismatch):
            transport.send(replace(request, prompt_digest_sha256="0" * 64))

        assert wire.calls == 0

    def test_a_tampered_prompt_is_refused_by_its_own_digest(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())
        edited = dict(request.prompt)
        edited["observation"] = {**edited["observation"], "now": "1999-01-01T00:00:00Z"}

        with pytest.raises(PromptDigestMismatch):
            transport.send(replace(request, prompt=edited))

        assert wire.calls == 0

    def test_another_protocol_version_is_refused(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        with pytest.raises(RequestProtocolMismatch):
            transport.send(replace(request, protocol_version="operatebench.model.v99"))

        assert wire.calls == 0

    def test_another_tool_vocabulary_is_refused(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        with pytest.raises(RequestToolNamesMismatch):
            transport.send(replace(request, tool_names=("act", "wait")))

        assert wire.calls == 0

    def test_a_prompt_that_is_not_the_pinned_structure_is_refused(self) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        request = agent.build_request(observation())

        with pytest.raises(RequestPromptStructureError):
            transport.send(replace(request, prompt={"observation": {}}))

        assert wire.calls == 0


# ------------------------------------------------------------- the composition


class TestTheComposition:
    """One send path, and it is the shared exchange's."""

    def test_the_transport_keeps_no_executor_or_client_of_its_own(self) -> None:
        from operatebench.providers.executor import TurnExecutor

        _wire, client = scripted_client()
        transport = transport_for(client)

        held = list(vars(transport).values())
        assert [value for value in held if isinstance(value, TurnExecutor)] == []
        exchanges = [
            value for value in held if isinstance(value, OpenAIResponsesExchange)
        ]
        assert len(exchanges) == 1
        assert any(
            isinstance(value, TurnExecutor) for value in vars(exchanges[0]).values()
        )

    def test_a_real_turn_reaches_the_provider_through_the_shared_exchange(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Mapping[str, Any]] = []
        original = OpenAIResponsesExchange.exchange

        def recording(
            self: OpenAIResponsesExchange, payload: Mapping[str, Any], deadline: Any
        ) -> Any:
            seen.append(dict(payload))
            return original(self, payload, deadline)

        monkeypatch.setattr(OpenAIResponsesExchange, "exchange", recording)

        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        agent = agent_over(client)
        request = agent.build_request(observation())
        agent.decide(observation())

        assert len(seen) == 1
        assert seen[0] == build_model_payload(request, model=MODEL)
        assert wire.calls == 1

    def test_the_exchange_states_the_model_it_is_pinned_to(self) -> None:
        _wire, client = scripted_client()
        exchange = OpenAIResponsesExchange(
            model=MODEL, client=client, single_flight_label="test caller"
        )

        assert exchange.model == MODEL
        with pytest.raises(AttributeError):
            exchange.model = "another-model"  # type: ignore[misc]

    def test_the_measurements_the_exchange_took_are_readable_on_the_transport(
        self,
    ) -> None:
        _wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=MODEL,
            )
        )
        transport = transport_for(client)
        agent = ModelAgent(transport, model=MODEL, agent_id=MODEL_AGENT_ID)
        agent.begin_episode({})
        agent.decide(observation())

        assert transport.last_usage().output_tokens == 29
        assert transport.last_telemetry().attempt_count == 1
        assert transport.last_telemetry().attempts[0].response_received is True

    def test_the_json_tool_argument_decoder_has_one_implementation(self) -> None:
        import boundarybench.providers.common as boundary
        import operatebench.agents.openai_responses as lifecycle
        from operatebench.providers import toolcalls as kernel

        assert boundary.decode_tool_arguments is kernel.decode_tool_arguments
        assert lifecycle.decode_tool_arguments is kernel.decode_tool_arguments


# ------------------------------------------------- no provider prose, anywhere


class TestNoProviderTextInDurableEvidence:
    """Prose may cross the seam. It may not survive it."""

    def test_the_transport_bounds_the_prose_it_carries(self) -> None:
        long_prose = PROVIDER_PROSE + "x" * (MAX_TRANSIENT_TEXT_CHARACTERS * 3)
        _wire, client = scripted_client(
            responses_body(
                [
                    message_item(long_prose),
                    function_call_item("complete", json.dumps({"reason": "done"})),
                ],
                model=MODEL,
            )
        )
        transport = transport_for(client)
        agent = ModelAgent(transport, model=MODEL, agent_id=MODEL_AGENT_ID)
        agent.begin_episode({})
        request = agent.build_request(observation())

        response = transport.send(request)

        assert len(response.text) <= MAX_TRANSIENT_TEXT_CHARACTERS

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_no_durable_record_of_a_wire_episode_quotes_the_prose(
        self, spec: OperationSpec, scenario_id: str, tmp_path: Path
    ) -> None:
        from operatebench.artifact import artifact_text

        directory = tmp_path / "evidence"
        directory.mkdir(mode=0o700)
        ledger_path = directory / "execution.ndjson"
        _provider, _capture, run = episode_over_the_wire(
            spec, scenario_id, ledger_path=ledger_path
        )

        rendered = artifact_text(run)

        # The artefact carries the tape, the trajectory and the execution
        # record, so one search over it covers all three.
        assert PROVIDER_PROSE not in rendered
        assert run.execution.transport_calls > 1
        assert run.execution.outcome_source == "model"
        # Nor the sidecar the artefact's binding names: a call row holds digests
        # of the answer it received, never the answer.
        assert PROVIDER_PROSE not in ledger_path.read_text(encoding="utf-8")


class TestTheActPayloadThisMappingCanState:
    """The narrowing that used to be pinned here, measured as lifted.

    ``ACT``'s ``payload`` is a free-form object in the Core outcome contract, and
    this mapping used to send a *closed* object with no properties — which reads
    like a schema and accepts only ``{}``. Under it a model could state which
    action it proposed but not the arguments it proposed it with, and a
    maintenance operation whose actions need arguments could not be driven to
    its terminal through this bridge at all.

    The payload is open now, so the same episode that used to be graded
    unreliable runs to its terminal. The measurement is kept rather than
    deleted, pointed the other way: it fails the day somebody closes the payload
    again, which is the day this mapping stops being able to carry a decision.
    """

    def test_the_act_payload_schema_states_an_object_and_closes_nothing(self) -> None:
        payload = next(
            tool["parameters"]["properties"]["payload"]
            for tool in outcome_tools()
            if tool["name"] == "act"
        )

        assert payload["type"] == ["object", "null"]
        assert "properties" not in payload
        assert "required" not in payload
        assert "additionalProperties" not in payload

    @pytest.mark.parametrize("scenario_id", SCENARIOS)
    def test_a_reference_driven_episode_over_this_bridge_reaches_its_terminal(
        self, spec: OperationSpec, scenario_id: str
    ) -> None:
        """The consequence, measured rather than asserted from the schema.

        The reference agent's own ``ACT`` decisions carry payloads. Sent through
        a surface that can state one, they arrive as the proposals the reference
        agent made — so the domain validator accepts them, the episode reaches a
        terminal, and only V2 has the authored notification failures in recovery
        and obligations. Every decision still crossed a
        real SDK client on an ``httpx.MockTransport``, and the tape still
        replays without a provider.
        """
        _provider, _capture, run = episode_over_the_wire(spec, scenario_id)

        assert run.reliable is (scenario_id != "V2")
        assert run.evaluation.failed_dimensions == (
            ("recovery", "obligations") if scenario_id == "V2" else ()
        )
        assert run.evaluation.dimension("action_validity").findings == ()
        assert run.evaluation.dimension("deterministic_replay").ok is True
        assert run.execution.excluded is False

    def test_the_payloads_the_reference_agent_proposed_crossed_the_wire(
        self, spec: OperationSpec
    ) -> None:
        """Not just "it worked": the arguments themselves made the trip.

        An episode whose actions happened to need no arguments would also be
        reliable, so the answers the provider actually produced are read back
        and at least one of them is required to state a non-empty payload.
        """
        provider, _capture, _run = episode_over_the_wire(spec, "V1")

        stated = [
            arguments["payload"]
            for name, arguments in provider.answers
            if name == "act" and arguments["payload"]
        ]

        assert stated, "the reference agent proposed at least one ACT with arguments"
        for payload in stated:
            assert isinstance(payload, dict)
            assert all(isinstance(key, str) and key for key in payload)


# ------------------------------------------------- the rest of the refusal set


class TestThePromptStructureRefusals:
    """The three ways a prompt can be the wrong shape, each named separately."""

    def _request(self) -> ModelRequest:
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)
        return agent.build_request(observation())

    def _refused(self, prompt: dict[str, Any]) -> None:
        wire, client = scripted_client()
        transport = transport_for(client)
        request = replace(self._request(), prompt=prompt)

        with pytest.raises(RequestPromptStructureError):
            transport.send(request)

        assert wire.calls == 0

    def test_a_prompt_stating_another_protocol_version_than_its_request(self) -> None:
        prompt = dict(self._request().prompt)
        prompt["protocol_version"] = "operatebench.model.v99"

        self._refused(prompt)

    def test_a_prompt_stating_a_tool_summary_this_build_does_not_derive(self) -> None:
        prompt = dict(self._request().prompt)
        prompt["tools"] = {"act": {"required": [], "optional": []}}

        self._refused(prompt)

    def test_a_prompt_whose_observation_is_not_a_mapping(self) -> None:
        prompt = dict(self._request().prompt)
        prompt["observation"] = ["not", "a", "mapping"]

        self._refused(prompt)


class TestTheResponsesThisBoundaryWillNotRead:
    """A body that is not an answer is a provider fault, never a decision.

    Each of these could be dressed up as a classification — "no tool call",
    "truncated" — and each of them would then blame the model for something the
    service did. They are refused as
    ``provider_response_invalid`` instead, which excludes the run with its
    attempts intact rather than completing it with a decision nobody made.
    """

    def _refused(self, body: Mapping[str, Any]) -> None:
        from operatebench.providers.faults import AdapterProviderError

        wire, client = scripted_client(dict(body))
        transport = transport_for(client)
        agent = ModelAgent(_NeverCalled(), model=MODEL, agent_id=MODEL_AGENT_ID)

        with pytest.raises(AdapterProviderError):
            transport.send(agent.build_request(observation()))

        assert wire.calls == 1

    def test_an_answer_that_stopped_short_for_a_reason_this_build_does_not_classify(
        self,
    ) -> None:
        self._refused(
            responses_body(
                [], status="incomplete", incomplete_reason="content_filter", model=MODEL
            )
        )

    def test_an_answer_whose_status_is_neither_completed_nor_incomplete(self) -> None:
        self._refused(responses_body([], status="in_progress", model=MODEL))

    def test_an_answer_carrying_an_output_item_this_boundary_cannot_read(self) -> None:
        self._refused(
            responses_body(
                [{"type": "reasoning", "id": "rs_test", "summary": []}], model=MODEL
            )
        )


class TestWhatTheResponseObjectStates:
    """The response's own words, carried rather than echoed back."""

    def test_the_model_reported_is_the_one_the_body_states(self) -> None:
        """So ``ModelAgent``'s identity check is a check and not a tautology.

        Read through :func:`model_response_from` directly, because the exchange
        refuses a foreign model before this projection ever sees one — which is
        the right place for that refusal and also the reason this claim has to
        be proven one layer down.
        """
        body = responses_body(
            [function_call_item("complete", json.dumps({"reason": "done"}))],
            model="a-model-nobody-asked-for",
        )
        # Constructed the way this SDK constructs a response model — leniently
        # — rather than validated, because that is what the client does and what
        # ``WireResponse`` exists to hold beside the bytes. Every check the
        # exchange would have run has been run by the time this projection is
        # reached; what is under test here is only where the model string is
        # read from.
        wire: WireResponse[Response] = WireResponse(
            json.dumps(body),
            lambda: Response.construct(**body),
            kind="openai response",
        )

        response = model_response_from(wire)

        assert response.model == "a-model-nobody-asked-for"
        assert response.stop_reason == COMPLETED_STOP_REASON
        assert response.output_tokens == 29

    def test_the_transport_names_the_run_and_never_the_credential(self) -> None:
        _wire, client = scripted_client()
        transport = transport_for(client)

        rendered = repr(transport)

        assert MODEL in rendered
        assert LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION in rendered
        assert FAKE_API_KEY not in rendered
        assert "api_key" not in rendered

    def test_the_turn_deadline_is_the_one_the_transport_was_configured_with(
        self,
    ) -> None:
        _wire, client = scripted_client()

        assert transport_for(client).turn_deadline_seconds == DEADLINE_SECONDS


class TestAModelThisBuildShipsNoProfileFor:
    """The baseline shape, so the other arm of the profile branch is exercised."""

    def test_it_is_sent_the_baseline_body_rather_than_the_pinned_model_s(self) -> None:
        other = "gpt-not-a-pinned-model"
        wire, client = scripted_client(
            responses_body(
                [function_call_item("complete", json.dumps({"reason": "done"}))],
                model=other,
            )
        )
        agent = ModelAgent(
            transport_for(client, model=other), model=other, agent_id=MODEL_AGENT_ID
        )
        agent.begin_episode({})

        agent.decide(observation())

        body = wire.bodies[0]
        assert body["temperature"] == 0.0
        assert "reasoning" not in body
        assert tuple(sorted(body)) == BASELINE_PROFILE.sent_fields()


class TestTheFieldContractRefusesWhatItCannotState:
    """The contract's own guards, which nothing else in this suite reaches."""

    def test_a_field_of_a_kind_the_contract_does_not_define_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outcome field kind"):
            OutcomeField(
                name="invented", kind="invented_kind", required=True, description="x"
            )

    def test_a_tool_this_build_does_not_offer_has_no_fields_to_read(self) -> None:
        for reader in (outcome_fields, required_field_names, field_kinds):
            with pytest.raises(KeyError):
                reader("teleport")

    def test_the_kinds_read_back_are_the_kinds_the_contract_declared(self) -> None:
        assert field_kinds("wait") == {
            field.name: field.kind for field in OUTCOME_TOOLS["wait"]
        }


class TestTheSharedDecoderTakesEitherShape:
    """One SDK in this build types a call's arguments as text *or* a mapping."""

    def test_an_already_decoded_mapping_is_accepted_as_it_stands(self) -> None:
        decoded = decode_tool_arguments(
            {"reason": "done"},
            label="lifecycle tool call arguments",
            error=ToolArgumentsUndecodable,
        )

        assert decoded == {"reason": "done"}

    def test_a_value_this_build_could_not_record_is_refused(self) -> None:
        with pytest.raises(ToolArgumentsUndecodable):
            decode_tool_arguments(
                {"reason": float("nan")},
                label="lifecycle tool call arguments",
                error=ToolArgumentsUndecodable,
            )

    def test_a_message_item_stating_no_content_contributes_no_prose(self) -> None:
        """Read defensively, because the wire contract does not require content.

        ``MESSAGE_ITEM_WIRE_SHAPE`` requires only ``type``, and this SDK
        constructs its output items leniently, so an assistant message with no
        content at all reaches the projection as an object with no ``content``
        attribute. It contributes nothing and refuses nothing: prose is not a
        decision either way.
        """
        decision, wire = decide_over(
            responses_body(
                [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                    },
                    function_call_item("complete", json.dumps({"reason": "done"})),
                ],
                model=MODEL,
            )
        )

        assert decision == Complete(reason="done")
        assert wire.calls == 1
