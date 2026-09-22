"""The Lifecycle Track's projection of one ``ModelRequest`` onto xAI Responses.

What this module owns is the *question*, not the call. Two projections —
:func:`build_model_payload`, which turns a
:class:`~operatebench.agents.transport.ModelRequest` into a request body, and
:func:`model_response_from`, which reads one answer back as a
:class:`~operatebench.agents.transport.ModelResponse` — are the whole of what is
Lifecycle about talking to this service. Everything between them is
:class:`~operatebench.providers.xai_responses.XAIResponsesExchange`: the
endpoint check, the profile check, single-flight, the bounded loop with its cost
guard and attempt evidence, the raw-wire capture and the response contract. The
Boundary Track composes the same object with its own two projections, and there
is deliberately no second send path here — a duplicated loop would produce
evidence that looked exactly like the shared one's while drifting from it.

Nothing passes through a ``TurnRequest``. A Lifecycle request reaches the wire as
itself.

**The prompt that was hashed is the prompt that is sent.** ``ModelAgent`` hashes
``request.prompt`` into ``prompt_digest_sha256``, and every attempt it records
carries a request digest computed over that. So the user content this module
puts on the wire is ``canonical_json_text(request.prompt)`` — the exact bytes
that digest is over, not a re-rendering of the same information — and the digest
is *recomputed and compared* before dispatch. A mismatch is
:class:`PromptDigestMismatch` and no request is made: a run whose recorded
digests describe something the provider never saw is false in the one way an
audit cannot detect.

**A digest cannot check itself.** Recomputing the prompt digest catches an
edited prompt and nothing else, because the edit worth making recomputes the
digest too. So two more things are checked before dispatch and neither of them
is a hash of the thing it is checking: the observation states exactly
:data:`OBSERVATION_FIELDS` — Core's own model-visible allowlist, so a
``scenario_id`` cannot be inserted whatever it hashes to — and
``observation_digest_sha256``, the field playback replays a recorded decision
against, is recomputed from the observation actually being sent.

**Non-strict tools, because Core does not fit the strict subset.** The five
outcome tools are sent ``strict=False``, stated rather than omitted. Two of
Core's own semantics leave OpenAI's documented strict subset and there is no
third form that keeps them inside it: strict mode requires *every* object closed
with ``additionalProperties: false``, which an arbitrary ``ACT`` payload is not,
and it prohibits a union at the root of a tool's schema, which is the only place
a standalone ``WAIT``'s liveness rule can be stated for the whole outcome. Each
narrowing that would satisfy strict mode disagrees with Core about a concrete
value — a closed payload refuses every action that states arguments, and a root
without its branches admits the wait that can never end — so the choice is to
drop strict rather than to weaken Core.

**What that makes these schemas.** Guidance to the model, and nothing a server
is claimed to enforce. Under ``strict=False`` the honest assumption is that
nothing on the provider side applies them at all, so this build's own
fail-closed parser — :func:`~operatebench.agents.model.parse_tool_call` and
Core's :func:`~operatebench.core.outcomes.outcome_contract_problem` — remains
the authority on whether an answer is a decision. Every shape the schema refuses
is refused again there, with no schema in the path.

**What the schema states, therefore.** Exactly Core's contract. ``required``
names the canonical required fields of each outcome and no others, so a
semantically optional field may simply be left out. ``ACT``'s ``payload`` is an
open JSON object, because Core's contract for it is "an object whose keys are
non-empty strings" and the *domain* owns what those keys are. And a ``WAIT`` —
standalone, and the one an ``ASK`` carries — states its liveness as ``anyOf``
branches over the optional fields of the outcome contract, each branch requiring
Core's required fields plus the one liveness field it exists to require stated.

**The nulls, still accepted and still elided.** The property schemas stay
nullable for the field the contract calls optional. A model — or a provider
still shaping answers as strict mode taught it to — may send ``null`` for a
field it did not use, and ``Act(payload=None)`` is not an ``ACT``, so a null in
a field the *contract* calls optional is elided before the parser sees it,
recursively, including inside the ``WAIT`` an ``ASK`` carries. A null in a
**required** field is left exactly where it is and is classified. That rule is
:data:`LIFECYCLE_NULL_ELISION`, it is named in the mapping version, and it is in
the settings this transport reports. An omitted optional and a null optional
therefore parse to the same decision.

**No provider text becomes durable.** Prose beside a tool call is bounded to
:data:`MAX_TRANSIENT_TEXT_CHARACTERS` and carried on
:attr:`~operatebench.agents.transport.ModelResponse.text`, which nothing in this
package writes to a tape, a trajectory or an artefact. No refusal raised here
quotes a tool name, an argument or a status the provider stated.

Nothing in this module pre-authorises a configuration, reads a credential, or
opens a socket on import.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, cast

import openai
from openai.types.responses import Response

from operatebench.agents.lifecycle_contract import (
    AGENT_TOOL_NAMES,
    INSTRUCTIONS_CONTRACT,
    LIFECYCLE_NULL_ELISION,
    MAX_TRANSIENT_TEXT_CHARACTERS,
    MINIMUM_ITEMS,
    MINIMUM_MINUTES,
    NON_EMPTY_PATTERN,
    OBSERVATION_FIELDS,
    PROMPT_FIELDS,
    LifecycleRequestContract,
    ModelRequestRefused,
    ObservationDigestMismatch,
    PromptDigestMismatch,
    RequestModelMismatch,
    RequestObservationStructureError,
    RequestPromptStructureError,
    RequestProtocolMismatch,
    RequestToolNamesMismatch,
    ToolArgumentsUndecodable,
    UndecodableArguments,
    check_model_request,
    check_output_contract,
    elide_null_optionals,
    lifecycle_output_tokens,
    outcome_tool,
    outcome_tools,
    wait_liveness_branches,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS as MAX_OUTPUT_TOKENS
from operatebench.agents.model import (
    MODEL_PROTOCOL_VERSION,
    TRUNCATED_STOP_REASON,
    ModelBoundaryError,
)
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ProviderTurnEvidence,
    ProviderTurnObserver,
    ToolCall,
)
from operatebench.jsonsafe import canonical_json_bytes, canonical_json_text
from operatebench.providers.cost import CostGuard
from operatebench.providers.executor import Clock, ProviderRetryPolicy, Sleep
from operatebench.providers.faults import (
    PROVIDER_FAULT_RESPONSE_INVALID,
    AdapterProviderError,
)
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.toolcalls import decode_tool_arguments
from operatebench.providers.wire import WireResponse
from operatebench.providers.xai_responses import (
    ACCEPTED_ITEM_TYPES,
    COMPLETED_STATUS,
    FUNCTION_CALL_ITEM,
    INCOMPLETE_STATUS,
    MESSAGE_ITEM,
    OUTPUT_LIMIT_REASON,
    PARALLEL_TOOL_CALLS,
    SERVER_EXTENSION_DIGEST,
    STORE,
    TOOL_CHOICE,
    XAI_BASE_URL,
    XAI_PROVIDER,
    XAI_RESPONSE_CAPTURE,
    XAI_RESPONSE_SERVER_EXTENSIONS,
    XAI_RESPONSES_API,
    ProviderDispatchObserver,
    RequestProfile,
    XAIResponsesExchange,
    check_request_profile,
    response_usage,
)

#: The first xAI Responses mapping: canonical prompt JSON, fixed instructions,
#: five Core outcomes plus ``retrieve`` as non-strict client function tools, and
#: local fail-closed argument parsing with optional-null elision. This identity is
#: separate from both OpenAI Responses and xAI Chat Completions.
LIFECYCLE_XAI_REQUEST_MAPPING_VERSION_V1 = "lifecycle_xai_responses_model_request_v1"
LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION_V2 = (
    "lifecycle_xai_responses_model_request_v2"
)

# Compact action-schema projection successor; predecessor stays literal.
LIFECYCLE_XAI_REQUEST_MAPPING_VERSION_V4 = "lifecycle_xai_responses_model_request_v4"

LIFECYCLE_XAI_REQUEST_MAPPING_VERSION = "lifecycle_xai_responses_model_request_v6"
# Wire spelling successors, model-qualified: Grok 4.5 v3→v4 and 4.6 v4→v5.
# The existing output-policy enum stays frozen; Core semantics do not change.
# Compact action-schema projection successor; predecessor stays literal.
LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION_V5 = (
    "lifecycle_xai_responses_model_request_v5"
)

LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION = "lifecycle_xai_responses_model_request_v7"


def _request_contract(model: str) -> LifecycleRequestContract:
    return (
        LifecycleRequestContract.XAI_GROK46
        if model == "grok-4.6"
        else LifecycleRequestContract.LEGACY
    )


def lifecycle_xai_request_mapping(model: str) -> str:
    """Request-profile identity, independent of the unchanged Core engine."""
    if _request_contract(model) is LifecycleRequestContract.XAI_GROK46:
        return LIFECYCLE_XAI_GROK46_REQUEST_MAPPING_VERSION
    return LIFECYCLE_XAI_REQUEST_MAPPING_VERSION


#: Raw response work is bounded before either JSON parser sees function-call
#: arguments. Core's largest accepted tool schema supplies the structural
#: allowance; the fixed 64-KiB payload allowance covers domain-owned ACT data.
#: This is a private resource bound, not a statement about provider token billing.
TOOL_ARGUMENT_PAYLOAD_CAP_BYTES = 64 * 1024
CORE_TOOL_SCHEMA_MAX_BYTES = max(
    len(canonical_json_bytes(tool["parameters"], "Lifecycle tool schema"))
    for tool in outcome_tools()
)
MAX_TOOL_ARGUMENT_WIRE_BYTES = (
    CORE_TOOL_SCHEMA_MAX_BYTES + TOOL_ARGUMENT_PAYLOAD_CAP_BYTES
)


#: One attempt. Constructed rather than defaulted: the shared executor's own
#: default is three, and a Lifecycle turn that retried would be a second
#: provider run under one recorded request digest — which
#: :data:`~operatebench.agents.transport.RETRY_COUNT` already pins at zero for
#: this boundary. The backoff values are the identity of a schedule that is
#: never reached; a multiplier of one is the smallest the shared policy check
#: accepts.
LIFECYCLE_RETRY_POLICY = ProviderRetryPolicy(
    max_attempts=1, initial_backoff_seconds=0.0, backoff_multiplier=1.0
)

#: The stop reason a completed answer is reported under. Not the provider's own
#: word passed through: this is a value of the Lifecycle boundary's vocabulary,
#: and what
#: :meth:`~operatebench.agents.model.ModelAgent._parse` asks of it is only
#: whether it is :data:`~operatebench.agents.model.TRUNCATED_STOP_REASON`.
COMPLETED_STOP_REASON = "completed"


def request_instructions(request: ModelRequest) -> str:
    """The ``instructions`` field one request carries.

    Fixed text plus three values this build itself computed: the mapping version
    under which the body was built, the protocol the answer will be parsed
    under, and the digest of the request's own identity. The digest is bound
    here rather than only recorded locally so that what the provider was told
    and what the run records about the same turn are one statement — the value
    on the wire is the value in every attempt this turn produces.

    Nothing case-specific is here. The case is the user message, which is
    exactly the bytes ``prompt_digest_sha256`` covers.
    """
    return (
        f"{INSTRUCTIONS_CONTRACT}\n"
        f"request_mapping: {lifecycle_xai_request_mapping(request.model)}\n"
        f"protocol_version: {request.protocol_version}\n"
        f"request_digest_sha256: {request.request_digest_sha256}"
    )


# ------------------------------------------------------------------- the request


def _xai_outcome_tools() -> list[dict[str, Any]]:
    """Equivalent nonempty spelling for xAI; never mutate the Core schema."""
    tools = outcome_tools()

    def project(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("pattern") == r"[\s\S]":
                value["pattern"] = r"^[\s\S]+$"
            for child in value.values():
                project(child)
        elif isinstance(value, list):
            for child in value:
                project(child)

    project(tools)
    return tools


def build_model_payload(
    request: ModelRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    """The exact request body one Lifecycle turn sends, under this model's profile.

    Built as a plain mapping and handed to the SDK unchanged, so a test can
    compare what the transport saw against what this function produces and know
    the two are the same object rather than two hopefully-equivalent ones.

    The user content is ``canonical_json_text(request.prompt)`` — the whole
    prompt, in the exact encoding its digest is taken over. Not a summary of it,
    not a re-rendering of the observation, and not the observation alone: those
    would all put a *different* string on the wire from the one every recorded
    attempt is hashed against.

    ``max_output_tokens`` is the request's own ceiling rather than a constant
    here, because the ceiling is part of what
    :meth:`~operatebench.agents.transport.ModelRequest.identity` hashes: a body
    sent under a different one would not be the request that was recorded.
    """
    shape = check_request_profile(profile, model=model)
    check_output_contract(request, model=model, contract=_request_contract(model))
    if request.model != model:
        raise RequestModelMismatch(
            "payload model differs from request; nothing is dispatched"
        )
    payload: dict[str, Any] = {
        "model": model,
        "max_output_tokens": request.max_output_tokens,
        "instructions": request_instructions(request),
        "tools": _xai_outcome_tools(),
        "tool_choice": TOOL_CHOICE,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "store": STORE,
        "input": [
            {
                "role": "user",
                "content": canonical_json_text(request.prompt, "lifecycle model request"),
            }
        ],
    }
    # Added rather than defaulted to a sentinel: an omitted field is *absent*
    # from the body, and a null one is a value the API would read.
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    if shape.reasoning is not None:
        payload["reasoning"] = dict(shape.reasoning)
    return payload


def lifecycle_xai_settings(
    *,
    model: str,
    deadline_seconds: float,
    profile: RequestProfile | None = None,
    retry: ProviderRetryPolicy | None = None,
) -> dict[str, Any]:
    """Every result-affecting setting this projection asks under, and no credential.

    Exposed and tested because "which mapping, which profile, which tool shape,
    what happens to a null" are the questions a reader of a Lifecycle run has to
    be able to answer without reading this file. The SDK version is in it
    deliberately: the SDK decides serialisation, retry and error mapping, so an
    upgrade is a change to the experiment.

    Not yet hashed into anything. Artefact v4 does not carry provider settings
    and this phase does not change it; the mapping is here so that the phase
    which does has one place to read them from rather than a second, hand-copied
    list.
    """
    shape = check_request_profile(profile, model=model)
    return {
        "api": XAI_RESPONSES_API,
        "base_url": XAI_BASE_URL,
        "max_output_tokens": lifecycle_output_tokens(
            model=model, contract=_request_contract(model)
        ),
        "max_tool_argument_wire_bytes": MAX_TOOL_ARGUMENT_WIRE_BYTES,
        "null_elision": LIFECYCLE_NULL_ELISION,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "protocol_version": MODEL_PROTOCOL_VERSION,
        "request_mapping": lifecycle_xai_request_mapping(model),
        **shape.as_settings(),
        "response_capture": XAI_RESPONSE_CAPTURE,
        "response_contract": "exactly_one_tool_call",
        "response_server_extensions": XAI_RESPONSE_SERVER_EXTENSIONS,
        "response_server_extensions_digest": SERVER_EXTENSION_DIGEST,
        "retry": (retry or LIFECYCLE_RETRY_POLICY).as_dict(),
        "sdk": "openai",
        "sdk_version": openai.__version__,
        "store": STORE,
        "tool_choice": TOOL_CHOICE,
        "tool_names": list(AGENT_TOOL_NAMES),
        # False, and recorded rather than left implicit. It is the setting that
        # says what the tool schemas on the wire *are*: guidance to the model,
        # not a constraint any server is claimed to apply. A reader of a run who
        # wants to know why an answer was refused reads the parser, not this.
        "tool_strict_mode": False,
        "turn_deadline_seconds": deadline_seconds,
    }


# ------------------------------------------------------------------ the response


def _response_invalid(detail: str) -> AdapterProviderError:
    """A body this build cannot read as an answer at all, as a provider fault.

    Not a classification. Every :class:`~operatebench.agents.model.
    MalformedModelOutcome` is a statement about how the *model* behaved, and a
    response that stopped for a reason this build does not classify or carries
    an output item this scaffold never asked for is a statement about the
    service — so filing it under the model would put a service-side fault in the
    bucket that is supposed to mean "the model answered badly". It reaches
    ``ModelAgent`` as an execution fault, which excludes the run with its
    attempts intact rather than completing it with a decision nobody made.
    """
    return AdapterProviderError(PROVIDER_FAULT_RESPONSE_INVALID, detail)


def _response_id_digest(wire: WireResponse[Response] | None) -> str | None:
    """The digest of the provider's own response identifier, or ``None``.

    A digest rather than the identifier, and for the reason every other
    provider-controlled string in this build is treated the same way: it is a
    value nobody here reviewed. The digest is enough to say "these two rows are
    the same answer" and "this row is not that answer", which is what durable
    evidence needs it for.

    The body has already been parsed by the time this runs, so reading it costs
    nothing; a turn that never got one, or whose answer this build refused,
    states ``None`` rather than an invented value.
    """
    return None if wire is None else wire.response_id_digest_sha256


def _bounded_text(parts: list[str]) -> str:
    """The prose beside a tool call, bounded and never read for a decision."""
    return "\n".join(parts)[:MAX_TRANSIENT_TEXT_CHARACTERS]


def _message_text(item: Any) -> list[str]:
    """Whatever text one assistant message states, and nothing else.

    Read defensively and typed as ``Any`` on purpose. This SDK's output item is
    a union of some thirty models and only a few of them carry ``content`` at
    all, so a reader that named one of them would be asserting which member the
    SDK resolved rather than reading what arrived. Nothing here is a decision:
    the value only ever reaches
    :attr:`~operatebench.agents.transport.ModelResponse.text`.
    """
    content = getattr(item, "content", None)
    if not isinstance(content, list):
        return []
    return [part.text for part in content if isinstance(getattr(part, "text", None), str)]


def _tool_call_from(item: Any) -> ToolCall:
    """One ``function_call`` output item as one structured outcome.

    The arguments arrive as a JSON *string* on this API, so they have more ways
    to be wrong than a decoded object does — not JSON at all, JSON that is not an
    object, an object naming a key twice, a value canonical JSON cannot
    round-trip. Every one of them is the model failing the protocol rather than
    the transport failing, so the shared decoder's refusal becomes
    :class:`UndecodableArguments` and the existing parser classifies the call as
    ``MODEL_MALFORMED_ARGUMENTS``. Nothing the model wrote is quoted.
    """
    name = item.name
    try:
        decoded: Any = decode_tool_arguments(
            item.arguments,
            label="lifecycle tool call arguments",
            error=ToolArgumentsUndecodable,
        )
    except ToolArgumentsUndecodable:
        return ToolCall(name, UndecodableArguments())
    return ToolCall(name, elide_null_optionals(name, decoded))


def model_response_from(response: WireResponse[Response]) -> ModelResponse:
    """One checked provider answer, as the Lifecycle boundary's response object.

    The body has already been proven admissible by the shared exchange — the
    wire contract, the model's stated identity, the SDK's agreement with the
    bytes, and countable token usage — so what is left here is the projection.

    Three things it decides, and each of them maps onto something
    :class:`~operatebench.agents.model.ModelAgent` already names:

    * **Truncation.** An answer the service stopped at *this run's own output
      ceiling* becomes :data:`~operatebench.agents.model.TRUNCATED_STOP_REASON`,
      which the agent classifies as ``MODEL_OUTPUT_TRUNCATED``. An answer that
      stopped for any other reason is not classified at all — it is a provider
      fault, because guessing would file a service behaviour under the model.
    * **The action.** Exactly the ``function_call`` items, in order, decoded and
      elided. Zero, one or many: the agent decides what each count means, so a
      response carrying none and one carrying three both arrive intact rather
      than as a guess made here.
    * **The count.** ``output_tokens`` from the usage block the exchange
      validated on the wire, so the ceiling check the agent applies is against a
      number the provider actually stated.
    """
    parsed = response.parsed
    # ``response_usage`` refuses the whole body rather than returning a count it
    # could not verify — absent, null, boolean, fractional and negative are all
    # ``provider_response_invalid`` — so both counts are stated by the time it
    # returns. The narrowing is of a fact already proven, not a claim of its own.
    output_tokens = cast("int", response_usage(response).output_tokens)
    if parsed.status == INCOMPLETE_STATUS:
        details = parsed.incomplete_details
        if details is not None and details.reason == OUTPUT_LIMIT_REASON:
            return ModelResponse(
                model=parsed.model,
                stop_reason=TRUNCATED_STOP_REASON,
                output_tokens=output_tokens,
            )
        raise _response_invalid(
            "the answer stopped before it was complete, for a reason other than "
            "this run's output ceiling, so what it carries was cut short rather "
            "than decided. The reason the provider stated is not recorded here: it "
            "is provider-controlled text, and a durable failure row records only "
            "this build's own fixed detail and closed-set values"
        )
    if parsed.status != COMPLETED_STATUS:
        raise _response_invalid(
            "the answer did not complete, so what it carries is not the decision "
            "the model was making. The status the provider stated is not recorded "
            "here: it is provider-controlled text, and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    calls: list[ToolCall] = []
    prose: list[str] = []
    for item in parsed.output:
        declared = getattr(item, "type", None)
        if declared not in ACCEPTED_ITEM_TYPES:
            raise _response_invalid(
                "the answer carries an output item that is neither an assistant "
                "message nor a function call; this boundary accepts one structured "
                "outcome beside optional prose, and cannot tell which part of an "
                "answer containing anything else is the decision. The type the item "
                "declared is not recorded here: it is provider-controlled text, and "
                "a durable failure row records only this build's own fixed detail "
                "and closed-set values"
            )
        if declared == FUNCTION_CALL_ITEM:
            calls.append(_tool_call_from(item))
        elif declared == MESSAGE_ITEM:
            prose.extend(_message_text(item))
    return ModelResponse(
        model=parsed.model,
        stop_reason=COMPLETED_STOP_REASON,
        tool_calls=tuple(calls),
        text=_bounded_text(prose),
        output_tokens=output_tokens,
    )


# ----------------------------------------------------------------- the transport


class XAIResponsesTransport:
    """One pinned model, one shared exchange, one ``ModelRequest`` per turn.

    A :class:`~operatebench.agents.transport.ModelTransport`, and a projection
    and a composition and nothing else. The two Lifecycle-specific steps are
    :func:`build_model_payload` and :func:`model_response_from`; everything
    between them — the endpoint check, the profile check, single-flight, the
    bounded loop, the cost guard, the attempt evidence, the raw wire capture and
    the response contract — is
    :class:`~operatebench.providers.xai_responses.XAIResponsesExchange`,
    which this transport holds exactly one of and which the Boundary Track holds
    its own of. There is no SDK client call, no retry loop, no cost arithmetic
    and no telemetry construction in this class, and deliberately not.

    ``deadline_seconds`` is required and has no default. A wall-clock budget is
    an infrastructure decision an operator makes, and a defaulted one would be
    this module quietly deciding how long a turn may hang.
    """

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        deadline_seconds: float,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        recorder: ProviderTurnObserver | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        if (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ModelBoundaryError(
                "a model transport's turn deadline must be a finite, positive "
                "number of seconds. A deadline of zero or fewer dispatches nothing, "
                "and NaN compares false against every bound the dispatch loop "
                "checks a deadline against — so a turn carrying one would have no "
                f"wall-clock budget at all. Got {deadline_seconds!r}"
            )
        # Every refusal the constructor makes — the profile that is not this
        # model's, the client pointed elsewhere or carrying an organization, the
        # SDK retry loop left on — is the exchange's, made in that order, and
        # made before this object exists.
        self._exchange = XAIResponsesExchange(
            model=model,
            client=client,
            single_flight_label="XAIResponsesTransport",
            retry=LIFECYCLE_RETRY_POLICY,
            sleep=sleep,
            clock=clock,
            cost_guard=cost_guard,
            profile=profile,
            before_dispatch=before_dispatch,
            max_tool_argument_wire_bytes=MAX_TOOL_ARGUMENT_WIRE_BYTES,
        )
        self._model = self._exchange.model
        self._profile = self._exchange.request_profile
        self._deadline_seconds = float(deadline_seconds)
        self._clock = clock
        self._recorder = recorder

    def __repr__(self) -> str:
        """Names the run, never the credential.

        Written out rather than left to a dataclass: a generated ``repr`` would
        render whatever the exchange holds, and the exchange holds a client that
        holds an API key.
        """
        return (
            f"<XAIResponsesTransport model={self._model} api={XAI_RESPONSES_API} "
            f"mapping={self.request_mapping}>"
        )

    @property
    def provider(self) -> str:
        """The service this transport exchanges with."""
        return XAI_PROVIDER

    @property
    def api(self) -> str:
        """The provider API surface this transport uses."""
        return XAI_RESPONSES_API

    @property
    def request_mapping(self) -> str:
        """The Lifecycle projection version sent by this transport."""
        return lifecycle_xai_request_mapping(self._model)

    @property
    def model(self) -> str:
        """The one model this transport sends to. The exchange's, not a copy."""
        return self._model

    @property
    def request_profile(self) -> RequestProfile:
        """The frozen request shape this transport's model is sent."""
        return self._profile

    @property
    def turn_deadline_seconds(self) -> float:
        """The wall-clock budget one turn is dispatched under."""
        return self._deadline_seconds

    @property
    def settings(self) -> Mapping[str, Any]:
        """Every result-affecting setting this transport asks under."""
        return lifecycle_xai_settings(
            model=self._model,
            deadline_seconds=self._deadline_seconds,
            profile=self._profile,
            retry=self._exchange.retry,
        )

    def last_usage(self) -> AdapterUsage:
        """What the last turn cost, as the shared executor measured it.

        Read-only and unwritten-to. Artefact v4 carries no provider usage, so
        nothing in this build stores this yet; it is exposed because the phase
        that does needs the measurement to come from the object that took it
        rather than from a second count.
        """
        return self._exchange.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        """Every attempt the last turn made, as the shared executor recorded it."""
        return self._exchange.last_telemetry()

    @property
    def recorder(self) -> ProviderTurnObserver | None:
        """The optional evidence recorder this transport reports turns to."""
        return self._recorder

    def attach_recorder(self, recorder: ProviderTurnObserver) -> None:
        """Attach a recorder to a transport that was built without one.

        Exists because a recorder's header states *this transport's* own
        settings and their digest, so the transport has to exist before the
        recorder can be built honestly. Attaching is write-once: a second
        recorder mid-run would leave one journal describing part of a run and
        another describing the rest, with neither saying so.
        """
        if self._recorder is not None:
            raise ModelBoundaryError(
                "this transport already reports to an evidence recorder. A second "
                "one would split one run's provider evidence across two journals, "
                "neither of which could say it was partial"
            )
        self._recorder = recorder

    def send(self, request: ModelRequest) -> ModelResponse:
        """One turn: mark the origin, refuse, project, exchange, read.

        The order is the contract. The clock is read *first*, before anything
        else this method does, because everything after it is inside the turn's
        wall-clock budget: checking the request, building the body, and — inside
        the exchange — checking the endpoint, checking the body against the
        profile and bounding the tokens. A deadline built after that work would
        hand it out for free, and the shared loop would then dispatch with the
        whole budget as its per-attempt timeout on a turn that had already spent
        part of it. The reading costs one call on a clock this transport already
        owns, and it is taken even on the path that refuses, because a request
        that is refused is a turn that happened.

        :func:`check_model_request` runs before a payload exists, so a request
        this transport cannot honestly send costs an exception rather than a
        dispatched body and a spent reservation. Clearing the previous turn's
        measurement happens next and inside the single-flight guard, because
        everything after it can fail and a measurement that outlives the call it
        measured is stale evidence. Reading the answer comes last and after the
        measurement: a response that arrives is paid for whether or not this
        boundary can read a decision out of it, so :meth:`last_usage` answers
        for a classified turn's cost too.

        An attached recorder is told what this turn measured, on the way out and
        on the way through a fault alike, in a ``finally`` after the exchange so
        the telemetry it reads is this turn's. It is told *after* the request has
        been built and dispatched and it is handed the objects that already
        exist, so there is no path on which recording changes a byte of what
        left. A refusal from :func:`check_model_request` is deliberately outside
        it: nothing was dispatched, so there is no provider attempt to record.
        """
        started = self._clock()
        check_model_request(
            request, model=self._model, contract=_request_contract(self._model)
        )
        payload = build_model_payload(request, model=self._model, profile=self._profile)
        answer: ModelResponse | None = None
        wire: WireResponse[Response] | None = None
        fault: str | None = None
        try:
            with self._exchange.turn():
                self._exchange.clear()
                wire = self._exchange.exchange(payload, self._deadline(started))
            answer = model_response_from(wire)
            return answer
        except AdapterProviderError as exc:
            fault = exc.fault
            raise
        finally:
            if self._recorder is not None:
                self._recorder.observe_provider_turn(
                    ProviderTurnEvidence(
                        request=request,
                        payload=payload,
                        settings=self.settings,
                        telemetry=self._exchange.last_telemetry(),
                        usage=self._exchange.last_usage(),
                        response=answer,
                        response_id_digest_sha256=_response_id_digest(wire),
                        fault=fault,
                    )
                )

    def _deadline(self, started: float) -> TurnDeadline:
        """This turn's wall-clock budget, counted from where the turn began.

        The origin is passed in rather than read here: this method is called
        after the request checks and the payload build, and a deadline that
        started its own clock would be counting from the wrong end of them. It
        travels on the deadline itself as ``started_at``, which is what lets the
        shared executor compute each attempt's timeout as what is left *of the
        turn* rather than what is left of the loop — one origin, subtracted
        once, for the budget, the cancellation test and the retry affordability
        arithmetic alike.

        The clock is the one the exchange was built with, so a test that drives
        the loop deterministically drives the deadline deterministically too, and
        neither half can be measuring a different time from the other.
        """
        budget = self._deadline_seconds
        clock = self._clock
        return TurnDeadline(
            remaining_seconds=budget,
            cancelled=lambda: (clock() - started) >= budget,
            started_at=started,
        )


__all__ = [
    "COMPLETED_STOP_REASON",
    "CORE_TOOL_SCHEMA_MAX_BYTES",
    "INSTRUCTIONS_CONTRACT",
    "LIFECYCLE_NULL_ELISION",
    "LIFECYCLE_RETRY_POLICY",
    "LIFECYCLE_XAI_REQUEST_MAPPING_VERSION",
    "MAX_TOOL_ARGUMENT_WIRE_BYTES",
    "MAX_TRANSIENT_TEXT_CHARACTERS",
    "MINIMUM_ITEMS",
    "MINIMUM_MINUTES",
    "NON_EMPTY_PATTERN",
    "OBSERVATION_FIELDS",
    "PROMPT_FIELDS",
    "TOOL_ARGUMENT_PAYLOAD_CAP_BYTES",
    "ModelRequestRefused",
    "ObservationDigestMismatch",
    "PromptDigestMismatch",
    "RequestModelMismatch",
    "RequestObservationStructureError",
    "RequestPromptStructureError",
    "RequestProtocolMismatch",
    "RequestToolNamesMismatch",
    "ToolArgumentsUndecodable",
    "UndecodableArguments",
    "XAIResponsesTransport",
    "build_model_payload",
    "check_model_request",
    "decode_tool_arguments",
    "elide_null_optionals",
    "lifecycle_xai_settings",
    "model_response_from",
    "outcome_tool",
    "outcome_tools",
    "request_instructions",
    "wait_liveness_branches",
]
