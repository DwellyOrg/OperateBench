"""The OpenAI Responses adapter: the Boundary Track's projection of one turn.

What it is *not* is a benchmark result: it is the integration a credentialed
operator needs before any model run can happen. This build ships no provider
result, and it pre-authorises no run — a live run is refused until an operator
authorises the exact configuration they audited.

It speaks the **Responses** API rather than Chat Completions, and that is a
pinned decision rather than a default. The two are different request mappings
with different tool shapes and different response objects; a run under one is
not a run under the other, so which one was used is in
:data:`OPENAI_API` and therefore in run identity.

**What this module owns is the question, not the call.** Two functions —
:func:`build_responses_request`, which turns a
:class:`~boundarybench.adapter.TurnRequest` into a body, and
:func:`parse_response`, which reads one answer back as an
:class:`~boundarybench.adapter.AdapterCall` — are the whole of what is Boundary
about talking to this service. Everything between them is
:class:`~operatebench.providers.openai_responses.OpenAIResponsesExchange`: the
endpoint check, the profile check, single-flight, the bounded retry loop with
its cost guard and attempt evidence, the raw-wire capture and the response
contract. The exchange is track-neutral and the Lifecycle Track will hold its
own of the same class, which is the point — a second retry loop would produce
evidence that looked exactly like this one's while drifting from it.

Nothing observable moved when it was extracted. The names this module used to
define — the request profiles, the wire shapes, the server-extension contract
and its digest, the endpoint and its variables — are imported from the exchange
and re-exported here, as the same objects; the request bytes, the settings
mapping hashed into ``configuration_id`` and :data:`OPENAI_ADAPTER_VERSION` are
byte-for-byte what they were.

The properties this module holds to are the same three the Anthropic
integration holds to, for the same reasons:

* **The provider boundary remains narrow.** The adapter is handed a
  :class:`~boundarybench.adapter.TurnRequest` and nothing else, so the system
  prompt, the tool schemas and the model-visible history are all projections of
  what the environment actually revealed. The compiled ``Variant`` — expected
  disposition, expected reason, required evidence, content digest — is not
  reachable from here, so it cannot be serialised into a request by accident.
* **Every result-affecting request setting is pinned, per model.** Model, SDK,
  SDK version, output ceiling, the request profile — which sampling and
  reasoning fields are sent, which are omitted, and with what values —
  tool-choice behaviour, retention and the retry policy are all reported
  through :attr:`OpenAIResponsesAdapter.settings`, which the run manifest hashes
  into run identity. A body that does not match the settings is not dispatched.
* **The credential never becomes data.** It is read from ``OPENAI_API_KEY``
  once, handed straight to the SDK client, and never stored on this object,
  written to settings, put in an exception message or rendered by ``repr``.

One thing this adapter accepts deserves naming beside those three. A real
response from this service carries a block of top-level fields the pinned SDK's
own model does not declare, so it names an exact contract of its own for that
block — :data:`OPENAI_RESPONSE_SERVER_EXTENSIONS` — and accepts precisely it:
closed recursively, typed scalar by scalar, complete wherever it is present, and
read for nothing. Each of the five top-level names may be absent independently,
because a block this build never reads is not one a run depends on; a name that
*is* there is a statement, and it has to be the whole statement the contract
transcribes. The contract is provider-*observed* rather than vendor-declared,
which is why it carries a digest in settings and why both accepting it and
narrowing what it accepts moved this integration's version.

One setting deserves naming here because it is easy to miss and hard to undo:
``store`` is pinned to ``False``. The Responses API retains request and response
content server-side by default, and a benchmark that silently left its prompts
and answers in a vendor's dashboard would be making a durability decision on an
operator's behalf.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any, cast

import openai
from openai.types.responses import Response

from boundarybench.adapter import (
    AdapterCall,
    AdapterIdentity,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterUsage,
    ProviderTelemetry,
    TurnDeadline,
    TurnRequest,
)
from boundarybench.budget import RunCostGuard
from boundarybench.providers.common import (
    SDK_MAX_RETRIES,
    Clock,
    ProviderRetryPolicy,
    Sleep,
    check_configuration_audited,
    check_environment,
    check_model,
    check_retry_policy,
    decode_tool_arguments,
    flat_tool_schema,
    observed_case,
    resolve_api_key,
)
from boundarybench.providers.wire import WireResponse
from boundarybench.queries import QUERY_RESOLUTION_CONTRACT
from boundarybench.scaffold import ACTION_SURFACE_CONTRACT, FACT_AFFORDANCE_CONTRACT
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.openai_responses import (
    ACCEPTED_ITEM_TYPES,
    API_KEY_VARIABLE,
    BASELINE_PROFILE,
    COMPLETED_STATUS,
    EXTENSION_COUNT,
    EXTENSION_FLAG,
    EXTENSION_PENALTY,
    EXTENSION_TEXT,
    FUNCTION_CALL_ITEM,
    FUNCTION_CALL_WIRE_SHAPE,
    GPT_5_6_LUNA_CONTRACT_SOURCE,
    GPT_5_6_LUNA_MODEL,
    GPT_5_6_LUNA_PROFILE,
    INCOMPLETE_STATUS,
    MAX_EXTENSION_TEXT_CHARACTERS,
    MAX_OUTPUT_TOKENS,
    MESSAGE_ITEM_WIRE_SHAPE,
    MODEL_REQUEST_PROFILES,
    NO_REASONING,
    OPENAI_API,
    OPENAI_BASE_URL,
    OPENAI_PROVIDER,
    OPENAI_RESPONSE_CAPTURE,
    OPENAI_RESPONSE_SERVER_EXTENSIONS,
    OUTPUT_LIMIT_REASON,
    PARALLEL_TOOL_CALLS,
    PENALTY_MAXIMUM,
    PENALTY_MINIMUM,
    PROHIBITED_REQUEST_FIELDS,
    REASONING_EFFORT_DEFAULT,
    REASONING_EFFORT_NONE,
    REASONING_EFFORT_VALUES,
    REJECTED_VARIABLES,
    RESPONSE_WIRE_SHAPE,
    SERVER_EXTENSION_CONTRACT,
    SERVER_EXTENSION_DIGEST,
    SERVER_EXTENSION_NAMES,
    SERVER_EXTENSION_SCHEMA,
    STORE,
    TEMPERATURE,
    TOOL_CHOICE,
    USAGE_FIELDS,
    USAGE_WIRE_SHAPE,
    OpenAIConfigurationError,
    OpenAIResponsesExchange,
    RequestProfile,
    build_client,
    check_request_profile,
    check_response_admissible,
    check_response_model,
    check_server_extensions,
    check_server_extensions_agree,
    check_wire_response,
    classify_exception,
    request_profile_for,
    request_token_bound,
    response_usage,
)

#: The code that talks to it. Distinct from the provider: two implementations of
#: the same API are two different runs.
OPENAI_IMPLEMENTATION = "openai_responses"
#: This integration's own version. Moves when anything it sends or accepts moves.
#:
#: ``0.2.0`` is where it started reading the wire. What it accepts changed — a
#: response is now checked as the exact JSON the provider sent before the SDK's
#: typed reading of it is trusted — and so did what it sends, by one header: the
#: SDK's raw-response surface marks the request with one.
#:
#: ``0.3.0`` moves what it *sends* to the pinned model, on the strength of a
#: published contract this build can now cite: ``gpt-5.6-luna`` receives an
#: explicit ``reasoning`` object and no sampling parameter at all, where it
#: previously received the reverse. That is a different question asked of the
#: model rather than a different way of asking the same one, so it is a version
#: move and not a settings edit. Both numbers are inside run identity, and a run
#: made under either earlier one is not comparable with one made here.
#:
#: ``0.4.0`` moves what it *accepts*. The service returns a block of top-level
#: fields on a real response that the pinned SDK does not declare, and this
#: build refused every body carrying one — correctly, under the only contract it
#: had. It now names an exact, closed, versioned contract for that block and
#: accepts precisely that, validating it recursively on both readings of the
#: body and reading nothing out of it. What it sends is byte-identical to
#: ``0.3.0``; what it will read as an answer is not, so the number moves.
#:
#: ``0.5.0`` moves what it accepts again, and in the other direction. ``0.4.0``
#: named that block's shape and then made every member of it optional under its
#: own parent, so it accepted bodies nothing observed and nothing justified: an
#: attribution object stating no payer, a tool-usage object stating neither tool,
#: an empty object anywhere the contract fixes one. A contract that calls itself
#: exact cannot also mean "any subset of the observed shape". Under this version
#: the five top-level names stay independently optional and every child under a
#: *present* object is required. What it sends is still byte-identical to
#: ``0.3.0``; a body ``0.4.0`` would have read as an answer is now refused, so
#: the number moves and a run made under either is not comparable with the other.
OPENAI_ADAPTER_VERSION = "0.5.0"

#: How a :class:`~boundarybench.adapter.TurnRequest` becomes a request body.
#: Pinned in settings because a different mapping is a different experiment even
#: with an identical scaffold.
#:
#: Named for this integration rather than shared with the Anthropic one. The
#: *content* of the projection is deliberately identical — see
#: :func:`~boundarybench.providers.common.observed_case` — but the body that
#: carries it is not, so one version string covering both would claim a byte
#: equality that does not hold.
REQUEST_MAPPING_VERSION = "openai_responses_turn_request_json_v1"


def build_openai_adapter(
    *,
    model: str,
    configuration_id: str | None = None,
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    retry: ProviderRetryPolicy | None = None,
    cost_guard: RunCostGuard | None = None,
) -> OpenAIResponsesAdapter:
    """Everything a credentialed run needs, or a clean refusal.

    Called before a run directory, a manifest or a lock exists, so an
    unconfigured environment costs the operator an error message rather than a
    half-created run whose evidence claims a provider that was never contacted.

    The order is load-bearing: the configuration is refused before
    ``OPENAI_API_KEY`` is read, so an unauthorised invocation never handles the
    credential at all.
    """
    checked = check_model(model, provider=OPENAI_PROVIDER, error=OpenAIConfigurationError)
    # Before the client exists, and so before a credential has been read: a
    # retry policy the dispatch loop cannot honour is an operator mistake, and
    # refusing it here costs an error message rather than a spent reservation.
    check_retry_policy(
        retry or ProviderRetryPolicy(),
        provider=OPENAI_PROVIDER,
        error=OpenAIConfigurationError,
    )
    check_configuration_audited(
        configuration_id,
        error=OpenAIConfigurationError,
        audited=audited,
        environ=environ,
    )
    check_environment(
        REJECTED_VARIABLES,
        provider=OPENAI_PROVIDER,
        base_url=OPENAI_BASE_URL,
        error=OpenAIConfigurationError,
        environ=environ,
    )
    api_key = resolve_api_key(
        API_KEY_VARIABLE,
        provider=OPENAI_PROVIDER,
        error=OpenAIConfigurationError,
        environ=environ,
    )
    return OpenAIResponsesAdapter(
        model=checked,
        client=build_client(api_key=api_key),
        retry=retry,
        cost_guard=cost_guard,
    )


def openai_identity(model: str) -> AdapterIdentity:
    """Who answered: the service, the pinned model, and this integration."""
    return AdapterIdentity(
        provider=OPENAI_PROVIDER,
        model=model,
        implementation=OPENAI_IMPLEMENTATION,
        version=OPENAI_ADAPTER_VERSION,
    )


def profile_settings(
    profile: RequestProfile, retry: ProviderRetryPolicy
) -> dict[str, Any]:
    """Every result-affecting request setting, and no credential.

    This mapping is hashed into run identity, so it is the answer to "what
    exactly was asked of the provider". The SDK version is in it deliberately:
    the SDK decides serialisation, retry and error mapping, so an upgrade is a
    change to the experiment and a resume must refuse it.

    It is also an exact description of the request body's field set, and
    :func:`~boundarybench.providers.common.check_payload_fields` refuses to
    dispatch a body that does not match it.
    """
    return {
        "action_surface": ACTION_SURFACE_CONTRACT,
        "api": OPENAI_API,
        "base_url": OPENAI_BASE_URL,
        "fact_affordance": FACT_AFFORDANCE_CONTRACT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "query_resolution": QUERY_RESOLUTION_CONTRACT,
        "request_mapping": REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_capture": OPENAI_RESPONSE_CAPTURE,
        "response_contract": "exactly_one_tool_call",
        # Named *and* hashed. The name says which contract this run accepted a
        # server-extension block under; the digest is what makes that claim
        # re-derivable, because a provider-observed contract can be widened
        # without its name changing and a reader would have no way to tell.
        "response_server_extensions": OPENAI_RESPONSE_SERVER_EXTENSIONS,
        "response_server_extensions_digest": SERVER_EXTENSION_DIGEST,
        "retry": retry.as_dict(),
        "sdk": "openai",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_version": openai.__version__,
        "store": STORE,
        "tool_choice": TOOL_CHOICE,
    }


def openai_settings(
    retry: ProviderRetryPolicy | None = None, *, model: str
) -> dict[str, Any]:
    """The settings a run of this model records, under this model's profile.

    The model is required rather than defaulted. Settings that could be stated
    without naming a model would be settings that describe a request shape
    instead of a request.
    """
    return profile_settings(request_profile_for(model), retry or ProviderRetryPolicy())


# -- the request -------------------------------------------------------------


def build_responses_request(
    request: TurnRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    """The exact request body one turn sends, under this model's profile.

    Built as a plain mapping and handed to the SDK unchanged, so a test can
    compare what the transport saw against what this function produces and know
    the two are the same object rather than two hopefully-equivalent ones.

    History is one deterministic user message rather than a replayed
    call/output exchange. The environment's transcript is the only record of
    what happened, and it carries no provider call ids to replay; inventing them
    would make the request depend on state outside the
    :class:`~boundarybench.adapter.TurnRequest`. The convention is pinned as
    :data:`REQUEST_MAPPING_VERSION` so it is part of run identity.
    """
    shape = check_request_profile(profile, model=model)
    payload: dict[str, Any] = {
        "model": model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "instructions": request.system_prompt,
        "tools": [flat_tool_schema(action) for action in request.actions],
        "tool_choice": TOOL_CHOICE,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "store": STORE,
        "input": [
            {
                "role": "user",
                "content": canonical_json_text(
                    observed_case(request), "openai turn request"
                ),
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


# -- the response ------------------------------------------------------------


def _function_calls(response: Response) -> list[Any]:
    """Every function call in the answer, refusing any item that is neither.

    Text beside an action is ordinary commentary and is ignored. Anything else —
    a reasoning item, a server-side tool result, an item type this build has
    never heard of — is content this scaffold did not ask for and cannot
    interpret, and reading past it would mean guessing which part of the answer
    was the action.
    """
    calls: list[Any] = []
    for item in response.output:
        declared = getattr(item, "type", None)
        if declared not in ACCEPTED_ITEM_TYPES:
            raise AdapterProtocolError(
                "the answer carries an output item that is neither an assistant "
                "message nor a function call; this scaffold accepts one function "
                "call beside optional text, and cannot tell which part of an "
                "answer containing anything else is the action. The type the item "
                "declared is not quoted here: it is provider-controlled text, and "
                "a durable failure row records only this build's own fixed detail "
                "and closed-set values"
            )
        if declared == FUNCTION_CALL_ITEM:
            calls.append(item)
    return calls


def parse_response(response: Response, request: TurnRequest) -> AdapterCall:
    """The provider's answer, as exactly one action call, or a refusal.

    Every message this raises is a fixed sentence plus values this build itself
    chose — a count of the calls that arrived, the ceiling the run pinned, the
    action names the scaffold declares. Nothing that arrived in the response is
    quoted: not the status it stated, not an item's declared type, not a tool
    name, not an argument, not a header or a request id.
    """
    # Truncation is decided first, and before the call count, because it
    # *explains* the call count. A turn cut off at the ceiling may carry no call,
    # half of one, or a complete-looking one the model had not finished
    # committing to; reporting "the answer carries 0 function call(s)" would
    # describe the symptom and hide the cause.
    if response.status == INCOMPLETE_STATUS:
        details = response.incomplete_details
        if details is not None and details.reason == OUTPUT_LIMIT_REASON:
            raise AdapterOutputLimitError(
                "the answer reached this run's pinned output-token ceiling of "
                f"{MAX_OUTPUT_TOKENS} and was cut off there, so what it carries is "
                "a fragment rather than the action the model was making. This is "
                "an output-budget limit and not a malformed answer: the ceiling is "
                "this run's own setting, it is recorded in run identity, and "
                "raising it is the change that would let the same model finish"
            )
        raise AdapterProtocolError(
            "the answer stopped before it was complete, for a reason other than "
            "this run's output ceiling, so the call it carries was cut short "
            "rather than completed and is not the action the model was making. "
            "The reason the provider stated is not quoted here: it is "
            "provider-controlled text, and a durable failure row records only "
            "this build's own fixed detail and closed-set values"
        )
    if response.status != COMPLETED_STATUS:
        raise AdapterProtocolError(
            "the answer did not complete, so the call it carries is not the "
            "action the model was making. The status the provider stated is not "
            "quoted here: it is provider-controlled text, and a durable failure "
            "row records only this build's own fixed detail and closed-set values"
        )
    calls = _function_calls(response)
    if len(calls) != 1:
        raise AdapterProtocolError(
            f"the answer carries {len(calls)} function call(s); this scaffold "
            "requires exactly one action per turn, and neither prose alone nor "
            "two actions at once names the single thing the agent did"
        )
    call = calls[0]
    if call.name not in {action.name for action in request.actions}:
        raise AdapterProtocolError(
            "the answer calls a tool the scaffold does not declare; the scaffold "
            f"offers {sorted(action.name for action in request.actions)}. The name "
            "the model produced is not quoted here: it is model-authored text and "
            "a durable failure row records only stable, closed-set detail"
        )
    # ``arguments`` arrives as a JSON *string* on this API, not as a decoded
    # object the way the Messages API delivers it, so it has more ways to be
    # wrong — not JSON at all, JSON that is not an object, an object that names
    # a key twice — and every one of them is the model failing the protocol
    # rather than the transport failing. The decoding rules are this benchmark's
    # rather than this vendor's, so they are stated once, in
    # ``providers.common``, and the other two integrations apply the same ones.
    return AdapterCall(
        action=call.name,
        arguments=decode_tool_arguments(
            call.arguments,
            label="openai tool call arguments",
            error=AdapterProtocolError,
        ),
    )


# -- the adapter -------------------------------------------------------------


class OpenAIResponsesAdapter:
    """One provider client, one pinned model, one action per turn.

    A projection and a composition, and nothing else. The two Boundary-specific
    steps are :func:`build_responses_request` and :func:`parse_response`;
    everything between them — the endpoint check, the profile check,
    single-flight, the retry loop, the cost guard, the attempt evidence, the raw
    wire capture and the response contract — is
    :class:`~operatebench.providers.openai_responses.OpenAIResponsesExchange`,
    which this adapter holds one of and which the Lifecycle Track will hold its
    own of. There is no second send path here, and deliberately not: a duplicated
    retry loop would produce evidence that looked exactly like this one's while
    drifting from it.
    """

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        retry: ProviderRetryPolicy | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: RunCostGuard | None = None,
        profile: RequestProfile | None = None,
    ) -> None:
        # Every refusal the constructor makes — the profile that is not this
        # model's, the client pointed elsewhere or carrying an organization, the
        # SDK retry loop left on, the retry policy the loop cannot honour — is
        # the exchange's, made in that order, and made before this object exists.
        self._exchange = OpenAIResponsesExchange(
            model=model,
            client=client,
            single_flight_label="OpenAIResponsesAdapter",
            retry=retry,
            sleep=sleep,
            clock=clock,
            cost_guard=cost_guard,
            profile=profile,
        )
        self._model = model
        # The client this adapter was handed, kept beside the exchange that
        # sends through it. It is the same object, so a caller that points it
        # somewhere else between two turns points the exchange there too — which
        # is exactly the case the endpoint is re-proven for on every send.
        self._client = client
        self._profile = self._exchange.request_profile
        self._retry = self._exchange.retry

    def __repr__(self) -> str:
        """Names the run, never the credential.

        Written out rather than left to a dataclass: a generated ``repr`` would
        render whatever the client holds, and the client holds an API key.
        """
        return (
            f"<OpenAIResponsesAdapter provider={OPENAI_PROVIDER} "
            f"model={self._model} implementation={OPENAI_IMPLEMENTATION}>"
        )

    @property
    def identity(self) -> AdapterIdentity:
        return openai_identity(self._model)

    @property
    def request_profile(self) -> RequestProfile:
        """The frozen request shape this adapter's model is sent."""
        return self._profile

    @property
    def settings(self) -> Mapping[str, Any]:
        return profile_settings(self._profile, self._retry)

    @property
    def cost_guard(self) -> RunCostGuard | None:
        """The guard this adapter asks before every request, if the run has one.

        Exposed so the runner can prove that the guard bound to the manifest is
        the guard the requests will actually be authorised by.
        """
        # Narrowed back to the concrete guard a Boundary run builds, and sound
        # by construction: the only thing that ever reaches the executor is this
        # constructor's ``cost_guard``. The kernel asks a three-method protocol
        # because it must not know what a token costs; this track does know, and
        # the runner compares this object against the guard it bound to the
        # manifest.
        return cast("RunCostGuard | None", self._exchange.cost_guard)

    def last_usage(self) -> AdapterUsage:
        return self._exchange.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._exchange.last_telemetry()

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        """One turn: clear the measurement, ask, measure, then read the answer.

        The order is the contract. Clearing happens *first* — before the request
        is even serialised — because everything after it can fail, and a
        measurement that outlives the call it measured is stale evidence.

        Parsing comes last and deliberately after the measurement. A response
        that arrives is paid for whether or not this scaffold can read it as one
        action, so :meth:`last_usage` answers for a protocol failure's cost too.

        The whole of it happens under the exchange's single-flight guard, taken
        *before* the clearing: a second overlapping turn would reset the
        measurement of the one in flight, so it is refused immediately rather
        than allowed to interleave. The guard is taken here rather than inside
        the exchange because the state it protects is wider than the call —
        the payload this track builds and the answer it parses are both part of
        the turn that owns the measurement.
        """
        with self._exchange.turn():
            self._exchange.clear()
            payload = build_responses_request(
                request, model=self._model, profile=self._profile
            )
            response = self._exchange.exchange(payload, deadline)
            return parse_response(response.parsed, request)

    def _send_payload(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[Response]:
        """The shared exchange, under the name this module has always had.

        Kept because it is the seam callers and tests already enter the adapter
        at with a body they built themselves — which is exactly what the
        exchange is for. It delegates; it checks nothing of its own.
        """
        return self._exchange.exchange(payload, deadline)


__all__ = [
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "EXTENSION_COUNT",
    "EXTENSION_FLAG",
    "EXTENSION_PENALTY",
    "EXTENSION_TEXT",
    "FUNCTION_CALL_WIRE_SHAPE",
    "GPT_5_6_LUNA_CONTRACT_SOURCE",
    "GPT_5_6_LUNA_MODEL",
    "GPT_5_6_LUNA_PROFILE",
    "MAX_EXTENSION_TEXT_CHARACTERS",
    "MAX_OUTPUT_TOKENS",
    "MESSAGE_ITEM_WIRE_SHAPE",
    "MODEL_REQUEST_PROFILES",
    "NO_REASONING",
    "OPENAI_ADAPTER_VERSION",
    "OPENAI_API",
    "OPENAI_BASE_URL",
    "OPENAI_IMPLEMENTATION",
    "OPENAI_PROVIDER",
    "OPENAI_RESPONSE_CAPTURE",
    "OPENAI_RESPONSE_SERVER_EXTENSIONS",
    "PARALLEL_TOOL_CALLS",
    "PENALTY_MAXIMUM",
    "PENALTY_MINIMUM",
    "PROHIBITED_REQUEST_FIELDS",
    "REASONING_EFFORT_DEFAULT",
    "REASONING_EFFORT_NONE",
    "REASONING_EFFORT_VALUES",
    "REJECTED_VARIABLES",
    "REQUEST_MAPPING_VERSION",
    "RESPONSE_WIRE_SHAPE",
    "SERVER_EXTENSION_CONTRACT",
    "SERVER_EXTENSION_DIGEST",
    "SERVER_EXTENSION_NAMES",
    "SERVER_EXTENSION_SCHEMA",
    "STORE",
    "TEMPERATURE",
    "TOOL_CHOICE",
    "USAGE_FIELDS",
    "USAGE_WIRE_SHAPE",
    "OpenAIConfigurationError",
    "OpenAIResponsesAdapter",
    "RequestProfile",
    "build_client",
    "build_openai_adapter",
    "build_responses_request",
    "check_request_profile",
    "check_response_admissible",
    "check_response_model",
    "check_server_extensions",
    "check_server_extensions_agree",
    "check_wire_response",
    "classify_exception",
    "openai_identity",
    "openai_settings",
    "parse_response",
    "profile_settings",
    "request_profile_for",
    "request_token_bound",
    "response_usage",
]
