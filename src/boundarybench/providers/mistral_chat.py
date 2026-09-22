"""The Mistral chat-completions adapter.

Built on the official ``mistralai`` SDK. In version 2.x that distribution is a
namespace and the client class lives at ``mistralai.client.Mistral``, which is
why the import here does not look like the package name.

The same three properties every adapter in this build holds to apply: the
provider boundary stays narrow, every result-affecting setting is pinned per
model and hashed into run identity, and the credential never becomes data.

One thing is genuinely different here and is worth naming at the top, because it
is the only place in this build where what leaves the process is not exactly
what the adapter built: **this SDK injects ``stream: false`` into the request
body.** It is not a field this integration authors, and it is not optional. So
it is declared as :data:`SDK_INJECTED_FIELDS`, recorded in settings, and
asserted separately from the adapter's own fields — the alternative would be
either a settings mapping that does not describe the request the provider
received, or a body check that quietly tolerates one unexplained key and would
therefore tolerate the next one too.

One thing about the *response* is different in the same way. This service
returns, on an ordinary HTTP 200 from the pinned model, a ``prompt_tokens_details``
object on the usage block that the pinned SDK's ``UsageInfo`` does not declare —
so this lane names an exact contract of its own for it,
:data:`MISTRAL_RESPONSE_EXTENSIONS`, and accepts precisely it: one name, one
required count inside it, bounded, and read for nothing. The contract is
provider-*observed* rather than vendor-declared, which is why it carries a digest
in settings and why accepting it moved this integration's version. What this
build *sends* is untouched.

Not declared is not the same as dropped, and the difference decides the contract's
shape. ``UsageInfo`` is configured ``extra="allow"``, so this SDK parks the
undeclared object verbatim in ``model_extra`` rather than discarding it: the
object is stated *twice*, once in the exact JSON the provider sent and once on
the typed reading of the same bytes. So it is checked on both readings and the
two must state the same thing — see :func:`check_response_extensions_agree` — and
``model_extra`` is never cleared to force agreement. A raw-only reading would
also not work here: :func:`~boundarybench.providers.common.check_response_contract`
counts the typed extra and refuses the body on its own, so accepting this body
requires naming the typed allowance beside the wire one.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version as _distribution_version
from types import MappingProxyType
from typing import Any, cast

import httpx
from mistralai.client import Mistral
from mistralai.client.errors import (
    MistralError,
    NoResponseError,
    ResponseValidationError,
)
from mistralai.client.httpclient import HttpClient
from mistralai.client.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from mistralai.client.utils.retries import BackoffStrategy, RetryConfig

from boundarybench.adapter import (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    AdapterCall,
    AdapterIdentity,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterProviderError,
    AdapterUsage,
    ProviderTelemetry,
    SingleFlight,
    TurnDeadline,
    TurnRequest,
)
from boundarybench.budget import RunCostGuard
from boundarybench.jsonsafe import canonical_json_bytes, canonical_json_text
from boundarybench.pricing import MAX_EXACT_TOKEN_COUNT
from boundarybench.providers.common import (
    SDK_MAX_RETRIES,
    Clock,
    Fault,
    ProviderConfigurationError,
    ProviderRetryPolicy,
    Sleep,
    TokenUsage,
    TurnExecutor,
    check_configuration_audited,
    check_endpoint,
    check_environment,
    check_model,
    check_payload_fields,
    check_response_collection,
    check_response_contract,
    check_retry_policy,
    checked_token_usage,
    classify_http_status,
    nested_tool_schema,
    observed_case,
    request_input_token_bound,
    resolve_api_key,
    resolve_single_tool_call,
)
from boundarybench.providers.wire import (
    WireResponse,
    WireShape,
    checked_wire_list,
    checked_wire_mapping,
    checked_wire_object,
    checked_wire_string,
    declared_wire_fields,
    is_wire_token_count,
    wire_invalid,
    wire_token_usage,
)
from boundarybench.queries import QUERY_RESOLUTION_CONTRACT
from boundarybench.scaffold import ACTION_SURFACE_CONTRACT, FACT_AFFORDANCE_CONTRACT

#: The service under test, as recorded in run identity.
MISTRAL_PROVIDER = "mistral"
#: The code that talks to it.
MISTRAL_IMPLEMENTATION = "mistral_chat_completions"
#: This integration's own version. Moves when anything it sends or accepts moves.
#:
#: ``0.2.0`` is where it started reading the wire. ``0.3.0`` is where it began
#: accepting this service's own response extension — see
#: :data:`MISTRAL_RESPONSE_EXTENSIONS`. A body ``0.2.0`` refused whole is one
#: ``0.3.0`` reads an action and two token counts out of, so the two are not the
#: same integration whatever else they share. What it *sends* is unchanged — the
#: capture wrapper only observes — but what it accepts is not, and the two are
#: equally inside run identity.
MISTRAL_ADAPTER_VERSION = "0.3.0"

#: How this adapter gets at the bytes the provider sent.
#:
#: Not this SDK's doing: it exposes no raw-response surface, and it discards
#: undeclared fields before any object this build can inspect exists. So the
#: configured HTTP client is wrapped in an audited object that implements the
#: SDK's own ``HttpClient`` protocol and keeps the ``httpx.Response`` of the
#: call in flight — see :class:`CapturedHttpClient`. Named in settings because
#: it decides what this build is able to refuse.
MISTRAL_RESPONSE_CAPTURE = "mistral_sdk_captured_http_response_v1"

#: The API surface this adapter speaks.
MISTRAL_API = "chat_completions"

#: How a :class:`~boundarybench.adapter.TurnRequest` becomes a request body.
REQUEST_MAPPING_VERSION = "mistral_chat_turn_request_json_v1"

#: The output ceiling, pinned rather than exposed. The same number every adapter
#: in this build pins: two models in one matrix that were cut off at different
#: lengths are not answering the same question.
MAX_OUTPUT_TOKENS = 1024

#: Pinned, and pinned to zero.
TEMPERATURE = 0.0

#: The response contract, enforced on both sides.
TOOL_CHOICE = "required"

#: One turn is one action.
PARALLEL_TOOL_CALLS = False

#: What the SDK puts in the body that this integration did not author.
#:
#: ``stream`` is not a field this build sets, omits or has an opinion about at
#: the mapping level: the SDK adds it to every non-streaming request. Declared
#: as data rather than tolerated as an exception, so that a body check can
#: assert an exact set — the adapter's fields plus exactly these — and a
#: *second* injected field appearing in a future SDK version fails a test
#: instead of passing unnoticed.
SDK_INJECTED_FIELDS: Mapping[str, Any] = MappingProxyType({"stream": False})

#: The fields whose presence or absence is a *profile's* decision.
PROHIBITED_REQUEST_FIELDS: tuple[str, ...] = (
    "presence_penalty",
    "random_seed",
    "temperature",
    "top_p",
)

#: The fields the mapping sends for every model, whatever its profile.
_COMMON_REQUEST_FIELDS: tuple[str, ...] = (
    "max_tokens",
    "messages",
    "model",
    "parallel_tool_calls",
    "tool_choice",
    "tools",
)


@dataclass(frozen=True)
class RequestProfile:
    """One model's frozen request shape: what is sent, and what is left out."""

    profile_id: str
    temperature: float | None
    #: Whether this build transcribed the shape from the model's own published
    #: request contract, or pinned it as its own choice.
    contract_verified: bool = True

    def sent_fields(self) -> tuple[str, ...]:
        fields = list(_COMMON_REQUEST_FIELDS)
        if self.temperature is not None:
            fields.append("temperature")
        return tuple(sorted(fields))

    def omitted_fields(self) -> tuple[str, ...]:
        sent = set(self.sent_fields())
        return tuple(
            sorted(field for field in PROHIBITED_REQUEST_FIELDS if field not in sent)
        )

    def as_settings(self) -> dict[str, Any]:
        return {
            "request_profile": self.profile_id,
            "request_fields_sent": list(self.sent_fields()),
            "request_fields_omitted": list(self.omitted_fields()),
            "temperature": self.temperature,
            "model_contract_verified": self.contract_verified,
        }


#: The one Mistral model this lane pins. The API model identifier from the
#: provider's own model page, not a display name.
MISTRAL_SMALL_MODEL = "mistral-small-2603"

BASELINE_PROFILE = RequestProfile(
    profile_id="mistral_baseline_temperature_zero_v1",
    temperature=TEMPERATURE,
    contract_verified=False,
)

#: ``mistral-small-2603``: sampling pinned to zero.
#:
#: ``contract_verified`` is ``False`` for the same reason it is on the other two
#: new profiles: this repository verified the model's price and its API
#: identifier against Mistral's published pages, and did not find a published
#: statement about this model's sampling contract. The shape is this build's
#: pinned choice, the settings say so, and a model that rejects the parameter
#: fails closed on an HTTP 4xx recorded as a request rejection rather than
#: running under a shape nobody checked.
MISTRAL_SMALL_PROFILE = RequestProfile(
    profile_id="mistralsmall2603_temperature_zero_v1",
    temperature=TEMPERATURE,
    contract_verified=False,
)

MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {MISTRAL_SMALL_MODEL: MISTRAL_SMALL_PROFILE}
)


def request_profile_for(model: str) -> RequestProfile:
    """The one request shape this build sends to this model."""
    return MODEL_REQUEST_PROFILES.get(model, BASELINE_PROFILE)


def check_request_profile(
    profile: RequestProfile | None, *, model: str
) -> RequestProfile:
    """Prove a profile is this model's, before it can shape a request."""
    expected = request_profile_for(model)
    if profile is None or profile == expected:
        return expected
    raise MistralConfigurationError(
        f"request profile {profile.profile_id!r} is not the profile this build "
        f"sends to model {model!r}, which is {expected.profile_id!r}"
    )


#: The only ``finish_reason`` a turn may end on.
REQUIRED_FINISH_REASON = "tool_calls"

#: The only kind of tool call this scaffold can read as an action. Compared
#: against the value the response *declares* rather than against a Python class:
#: this SDK resolves the field to an unrecognised-string arm rather than
#: failing, so an ``isinstance`` would answer "the SDK coped", not "the provider
#: sent a function call".
FUNCTION_TOOL_CALL = "function"

#: The one other finish reason this build classifies rather than redacts,
#: because it names a cause *this run controls*.
OUTPUT_LIMIT_FINISH_REASON = "length"

#: The two counts this surface's usage block is read for. The Chat Completions
#: names, not the Responses API's: a reader that proved the wrong pair present
#: would have proven nothing about the fields it then read.
USAGE_FIELDS: tuple[str, ...] = ("prompt_tokens", "completion_tokens")


class MistralConfigurationError(ProviderConfigurationError):
    """The environment cannot produce a usable, unambiguous Mistral client."""


# -- this service's response extension ----------------------------------------
#
# This API returns, on an otherwise ordinary HTTP 200 from the pinned model, one
# object on the usage block that the pinned SDK's ``UsageInfo`` does not declare:
# ``prompt_tokens_details``, carrying a single ``cached_tokens`` count. Under the
# response contract that object is an undeclared field, so the whole body was
# refused: the action went unparsed, the usage went unmeasured, and the turn's
# reservation was held as exposure over a field nothing here reads. That is the
# correct default and the wrong answer for a field that is stable, named and
# inert.
#
# So this lane names **one exact contract** for it and accepts precisely it. The
# same four properties the other two lanes' blocks are accepted under:
#
# * It is *closed*, and at both levels. One name on the usage block, one name
#   inside the object it carries, and nothing else at either. A name this
#   contract fixes for the usage block buys nothing at the root or on a message,
#   so the same object arriving elsewhere is still ``provider_response_invalid``.
# * It is *optional as a whole, complete when it is present*. The object may be
#   absent — this build reads nothing out of it, so a service that stops sending
#   it has not broken anything a run depends on — but an object that does arrive
#   states the one member fixed for it. A vacuous ``{}`` was never observed, and
#   accepting one would widen an exact contract into a guess.
# * It is *typed*, on the wire rather than after coercion. The count is an exact
#   non-negative integer inside a stated bound, and never a boolean, a string, a
#   fraction, a null or a negative.
# * It is *inert*. Nothing in it reaches the action, the model's stated identity,
#   or the counts a turn is priced and capped on. ``cached_tokens`` says the
#   provider charged less for part of the prompt; this build does not model that
#   discount, and :data:`~boundarybench.pricing.MISTRAL_PRICING_POLICY` charges
#   every provider-reported prompt token at the uncached input rate. That is
#   conservative in the only direction that matters — a run never under-reports
#   what it spent — and it is why moving this count changes no measurement.
#
# It is checked on **both readings** of the same bytes, which is a property of
# this SDK rather than a preference. ``UsageInfo`` is configured ``extra="allow"``,
# so a field it does not declare is parked verbatim in ``model_extra`` rather
# than discarded: the object is stated once in the exact JSON and once on the
# typed object, and a body whose two statements differ is not one body this build
# can describe. See :func:`check_response_extensions_agree`.
#
# What this is **not** is a vendor declaration. It transcribes the name and type
# shape two bounded schema-only diagnostics established against this exact model;
# the raw bodies were not kept, and no parameter that would turn the object off
# is invented here, so the request profile is unchanged. A provider-observed
# contract can move without notice, so it is named *and* hashed in adapter
# settings, and both are inside ``configuration_id``.

#: The name of that contract, as adapter settings record it.
MISTRAL_RESPONSE_EXTENSIONS = "mistral_response_extensions_v1"

#: The scalar vocabulary the contract fixes. One kind, and it is a closed-set
#: label this build chose, so it may appear in a durable failure detail where a
#: field name or a value may not.
EXTENSION_COUNT = "non_negative_bounded_integer"

#: The largest integer that count may state.
#:
#: :data:`~boundarybench.pricing.MAX_EXACT_TOKEN_COUNT`, and deliberately the
#: same constant the token counts are bounded by rather than a second number that
#: means the same thing: it is the largest integer that survives a JSON number in
#: every reader this build's artefacts cross, so a value past it is one this build
#: could not carry unchanged even if it wanted to.
MAX_EXTENSION_COUNT = MAX_EXACT_TOKEN_COUNT


def _frozen(node: Any) -> Any:
    """One schema node, deeply immutable. A shared mutable contract is not one."""
    if isinstance(node, Mapping):
        return MappingProxyType({key: _frozen(value) for key, value in node.items()})
    return node


def _plain(node: Any) -> Any:
    """The same node as plain JSON-encodable data, for the digest below."""
    if isinstance(node, Mapping):
        return {key: _plain(value) for key, value in node.items()}
    return node


#: The contract itself: the one name the usage block may carry beyond its
#: declared fields, and the exact shape of the object it names.
#:
#: A mapping value is an object whose keys are closed to exactly these children.
#: A string value is one of the scalar kinds above.
RESPONSE_EXTENSION_SCHEMA: Mapping[str, Any] = _frozen(
    {"prompt_tokens_details": {"cached_tokens": EXTENSION_COUNT}}
)

#: The names the usage block may carry beyond the ones its SDK model declares.
USAGE_EXTENSION_NAMES: frozenset[str] = frozenset(RESPONSE_EXTENSION_SCHEMA)

#: The whole contract, as one document: the name, the shape, the bound and the
#: two rules that decide what a present object has to state.
#:
#: It exists because the shape alone is not the contract. A build that
#: transcribed this same name and kind and then accepted a vacuous object would
#: read a body this one refuses, and a digest over the transcription alone would
#: be equal across the two — so two manifests would claim one configuration for
#: two readings. Hashing the rules and the bound beside the shape is what makes
#: the recorded claim answer "which bodies will this run read as answers".
RESPONSE_EXTENSION_CONTRACT: Mapping[str, Any] = _frozen(
    {
        "contract": MISTRAL_RESPONSE_EXTENSIONS,
        # Stated in the document rather than left to prose: this build reads no
        # value in it, which is why the object as a whole may be absent.
        "extension_values_read": False,
        "max_count": MAX_EXTENSION_COUNT,
        "members": RESPONSE_EXTENSION_SCHEMA,
        # Present object, whole object...
        "object_members_required": True,
        # ...and the object itself may be absent.
        "top_level_members_required": False,
    }
)

#: A digest over that document, recorded beside its name.
#:
#: The name alone would be a label: two builds could ship one
#: ``mistral_response_extensions`` version and accept different bodies, and the
#: run manifests would agree. The digest moves the moment the accepted contract
#: does, and it is inside ``configuration_id``.
RESPONSE_EXTENSION_DIGEST = hashlib.sha256(
    canonical_json_bytes(
        _plain(RESPONSE_EXTENSION_CONTRACT), "mistral response extensions"
    )
).hexdigest()

#: The same permission, expressed for the *typed* reading: which pinned SDK
#: response model is entitled to carry which of these names in ``model_extra``.
#:
#: Keyed by model rather than by name so the allowance is exactly as narrow as
#: the wire's: the name is proven for the usage block and buys nothing on a
#: message or at the root, and no object outside this one accepts an undeclared
#: field at all.
TYPED_EXTENSION_EXTRAS: Mapping[type, frozenset[str]] = MappingProxyType(
    {UsageInfo: USAGE_EXTENSION_NAMES}
)

#: How a refusal names what it was checking. One fixed label for either level:
#: which field carried the fault is not something a durable row needs, and the
#: value beside it is provider-controlled.
_EXTENSION_KIND = "named response extension"


def _extension_scalar_valid(value: Any, kind: str) -> bool:
    """Whether one value is the exact wire scalar the contract fixes for it.

    Exact, and by ``type`` rather than by ``isinstance``. Python makes ``True``
    an ``int``, so an ``isinstance`` count check would accept a boolean as one —
    and a lenient reader downstream would render it as a measurement nobody took.

    The vocabulary is closed and is this module's own, so :data:`EXTENSION_COUNT`
    is the only kind there is and there is no second one to fall through to.
    """
    del kind
    # The same rule the token counts are read under, and deliberately the same
    # function: two independently written definitions of "an exact non-negative
    # integer on the wire" would drift.
    return is_wire_token_count(value) and value <= MAX_EXTENSION_COUNT


def _check_extension_node(value: Any, schema: Any) -> None:
    """One node of the contract, against the node that arrived, recursively."""
    if not isinstance(schema, Mapping):
        if not _extension_scalar_valid(value, schema):
            raise wire_invalid(
                f"the response states a {_EXTENSION_KIND} whose value is not the "
                f"exact one of kind {schema!r} that {MISTRAL_RESPONSE_EXTENSIONS} "
                "fixes for it, so what arrived is not the contract this build "
                "accepts and the body is refused rather than read past. Nothing "
                "about the value is recorded here — neither which field carried it "
                "nor what it was: the field set is this contract's own and the "
                "value is provider-controlled, and a durable failure row records "
                "only this build's own fixed detail and closed-set values"
            )
        return
    body = checked_wire_mapping(value, kind=_EXTENSION_KIND)
    undeclared = sum(1 for key in body if str(key) not in schema)
    if undeclared:
        raise wire_invalid(
            f"the response's {_EXTENSION_KIND}s carry {undeclared} key(s) at one "
            f"level that {MISTRAL_RESPONSE_EXTENSIONS} does not fix. The contract "
            "is closed at every depth: this build accepts this object only because "
            "it can describe the whole of it, so a part it cannot describe is "
            "refused exactly as an undeclared field anywhere else in the body is. "
            "The names and values are not recorded here: they are "
            "provider-controlled, and a durable failure row records only this "
            "build's own fixed detail and closed-set values"
        )
    if any(name not in body for name in schema):
        raise wire_invalid(
            f"the response states a {_EXTENSION_KIND} object that omits at least "
            f"one member {MISTRAL_RESPONSE_EXTENSIONS} fixes for it. The whole "
            "object may be absent — this build reads none of it — but an object "
            "that is present is a statement, and this contract describes one shape "
            "for it rather than every subset of one: a partly stated object was "
            "never observed, so accepting it would widen an exact contract into a "
            "guess. How many members were missing, which ones, and what the object "
            "did state are not recorded here: the absences are provider-controlled, "
            "and a durable failure row records only this build's own fixed detail "
            "and closed-set values"
        )
    for name, child in schema.items():
        _check_extension_node(body[name], child)


def check_response_extensions(node: Mapping[str, Any]) -> None:
    """Prove every extension the usage block states is the exact fixed shape.

    Given the whole usage block rather than a projection of it, so a caller
    cannot accidentally hand over a filtered copy and prove the contract about
    something the provider did not send. Names this contract does not fix are not
    this function's business: the object's own allowed field set refuses an
    undeclared field, and
    :func:`~boundarybench.providers.common.check_response_contract` refuses an
    undeclared typed extra.

    A name that is absent is not checked, and that is this level's rule rather
    than the contract's rule everywhere: the object as a whole is optional, and
    everything below it is required once it is here.
    """
    for name, schema in RESPONSE_EXTENSION_SCHEMA.items():
        if name in node:
            _check_extension_node(node[name], schema)


def _stated_extensions(node: Any) -> dict[str, Any]:
    """Exactly the extension members one usage block states, for the comparison below."""
    if not isinstance(node, Mapping):
        return {}
    return {name: node[name] for name in RESPONSE_EXTENSION_SCHEMA if name in node}


def _typed_extras(value: Any) -> Mapping[str, Any]:
    """One parsed object's undeclared fields, as this SDK parked them."""
    extra = getattr(value, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}


def check_response_extensions_agree(
    wire: Mapping[str, Any], parsed: ChatCompletionResponse
) -> None:
    """Check the SDK's own reading of the extension, and that the two readings agree.

    This is possible here for a reason worth stating precisely, because the
    opposite is the more natural assumption about this SDK: it does not *declare*
    ``prompt_tokens_details``, but it does not *drop* it either. ``UsageInfo`` is
    configured ``extra="allow"``, so the object survives the SDK's own unmarshal
    path verbatim and lands in ``model_extra``. The object is therefore stated
    twice — in the exact JSON the provider sent, and on the typed reading of the
    same bytes — and both are validated, because a contract proven about one
    reading is not a contract about the other.

    Neither reading is preferred when they differ, because there is no principled
    way to pick: a body whose typed extras are not the extras on the wire is not
    one body this build can describe, and it is refused whole.

    What this deliberately does **not** do is clear or rewrite ``model_extra``.
    Emptying it would make the two readings agree by force, hide the very field
    this contract exists to describe, and leave the SDK's typed object in a state
    the provider never sent. The typed parse stays exactly as the SDK produced it.
    """
    typed = _typed_extras(getattr(parsed, "usage", None))
    check_response_extensions(typed)
    if _stated_extensions(wire.get("usage")) != _stated_extensions(typed):
        raise wire_invalid(
            "the exact JSON the provider sent and this pinned SDK's typed reading "
            "of the same bytes do not state the same response extensions, so there "
            "is no single body here for this build to check against "
            f"{MISTRAL_RESPONSE_EXTENSIONS}. Neither reading is preferred over the "
            "other and neither is edited to agree with the other: the response is "
            "refused whole. What each reading stated is not recorded here: it is "
            "provider-controlled, and a durable failure row records only this "
            "build's own fixed detail and closed-set values"
        )


# -- the wire contract --------------------------------------------------------
#
# Field sets taken from the pinned SDK's own models, for the reason the other
# two integrations take theirs the same way — and for one more that is specific
# to this SDK. Its response models resolve leniently and coerce, so a count the
# wire stated as a string or a boolean reaches the typed object as a number that
# no longer says what arrived. On this surface, reading the exact JSON is the
# only way to see what the parse would have flattened.

RESPONSE_WIRE_SHAPE = WireShape(
    kind="completion",
    allowed=declared_wire_fields(ChatCompletionResponse),
    required=("model", "usage"),
)
CHOICE_WIRE_SHAPE = WireShape(
    kind="completion choice",
    allowed=declared_wire_fields(ChatCompletionChoice),
    required=("finish_reason", "message"),
)
MESSAGE_WIRE_SHAPE = WireShape(
    kind="completion message",
    allowed=declared_wire_fields(AssistantMessage),
    required=(),
)
TOOL_CALL_WIRE_SHAPE = WireShape(
    kind="tool call",
    allowed=declared_wire_fields(ToolCall),
    required=("function",),
)
FUNCTION_WIRE_SHAPE = WireShape(
    kind="tool call function object",
    allowed=declared_wire_fields(FunctionCall),
    required=("name", "arguments"),
)
#: The usage block. Its declared fields are the vendor's schema; the one extra
#: name is this lane's own contract, and what an object under it may carry is
#: proven by :func:`check_response_extensions`.
USAGE_WIRE_SHAPE = WireShape(
    kind="usage block",
    allowed=declared_wire_fields(UsageInfo) | USAGE_EXTENSION_NAMES,
    required=USAGE_FIELDS,
)


class CapturedHttpClient:
    """The SDK's configured HTTP client, wrapped so the response can be read.

    An audited object that implements this SDK's own ``HttpClient`` protocol —
    ``send``, ``build_request`` and ``close`` — and delegates every one of them.
    It changes nothing about the request: the SDK builds it, the inner client
    sends it, and the bytes that leave this process are the SDK's. What it adds
    is a single reference to the ``httpx.Response`` of the call in flight, which
    is the only way this build can see the body before the SDK parses it and
    drops whatever its schema does not declare.

    "The call in flight" is enforced rather than assumed. The adapter clears the
    capture immediately before every request, takes exactly one response after
    it, and clears it again on every way out of the dispatch — including the one
    where the SDK read a whole body and then raised rather than returning it. A
    capture holding none, or more than one, is refused rather than guessed at. A
    body left over from a previous turn would be the same staleness the executor
    clears its measurement to avoid, one layer lower.

    One capture belongs to one adapter instance, and :meth:`claim` is where that
    is decided — see its own docstring for why it is a refusal rather than a
    shared lock.

    The response is held for the duration of one turn and is never written
    anywhere. Nothing durable in this build carries a provider body.
    """

    def __init__(self, inner: HttpClient) -> None:
        self._inner = inner
        self._captured: list[httpx.Response] = []
        # The adapter instance this capture belongs to, once one has claimed it.
        # A strong reference on purpose: the claim is decided with ``is``, and an
        # owner that could be collected would free its ``id`` for a later object
        # to be handed and mistaken for the same adapter. The reference cycle it
        # creates — adapter to capture to adapter — is one the collector already
        # handles, and the client that holds the capture outlives the adapter in
        # every arrangement this build makes anyway.
        self._owner: object | None = None
        self._claim_lock = threading.Lock()

    def __repr__(self) -> str:
        """Names what this is, never what it is holding."""
        return f"<CapturedHttpClient captured={len(self._captured)}>"

    @property
    def captured_count(self) -> int:
        """How many responses this wrapper is currently holding."""
        return len(self._captured)

    def claim(self, owner: object) -> None:
        """Bind this capture to one adapter instance, for that instance's life.

        An adapter instance is single-flight, and the lock that enforces it is
        *per instance*. So two adapters built over one SDK client would hold two
        locks over one capture: neither turn would be visible to the other's
        guard, both would clear and fill the same list, and each would find
        either the other's body or two bodies where its own contract says one.
        Every one of those is a measurement taken from a body that cannot be
        identified, which is the single question this capture exists to answer.

        The second adapter is refused rather than made to wait. A lock shared
        across instances would turn one adapter's turn into wall-clock the other
        adapter's turn spends and does not record, and a benchmark whose latency
        depends on unrecorded contention is measuring something else. Refusing
        costs a configuration error at construction, before any transport; the
        remedy is one SDK client per adapter, and the message says so.

        Idempotent for the same owner, because the adapter re-establishes its
        wrapper before every dispatch — this SDK's configuration is mutable, so
        the client it will send through is checked at the moment it is used
        rather than trusted from construction.

        Identity, not equality: two adapters are two turns whatever their
        settings happen to compare as.
        """
        with self._claim_lock:
            current = self._owner
            if current is None or current is owner:
                self._owner = owner
                return
        raise MistralConfigurationError(
            "this SDK client's response capture is already claimed by another "
            "adapter instance, and a capture belongs to exactly one adapter for "
            "its lifetime. An adapter instance is single-flight and holds its own "
            "lock, so two adapters over one client would clear and fill one "
            "capture from turns neither lock can see, and neither turn could then "
            "say which response was its own. This build refuses the second "
            "adapter rather than making one adapter's turn wait on another's, "
            "which would spend wall-clock no run records: separate adapters "
            "require separate SDK clients, one per adapter, built with "
            "`build_client`"
        )

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Delegate the send, and remember what came back."""
        response = self._inner.send(request, **kwargs)
        self._captured.append(response)
        return response

    def build_request(self, method: str, url: Any, **kwargs: Any) -> httpx.Request:
        """Delegate. The request is the SDK's; this object does not shape it."""
        return self._inner.build_request(method, url, **kwargs)

    def close(self) -> None:
        """Delegate, so the SDK's own teardown still closes what it opened."""
        self._inner.close()

    def clear(self) -> None:
        """Forget any response held, before the request whose one this will be."""
        self._captured.clear()

    def take(self) -> httpx.Response:
        """The one response this call produced, or a refusal.

        Refused rather than guessed at in both directions. None captured means
        the SDK returned an object this build cannot tie to a body, and more
        than one means the call was not the single request this adapter's retry
        policy says it is — in which case "the response for this call" has no
        answer and the turn must not be measured from either.
        """
        if len(self._captured) != 1:
            raise wire_invalid(
                f"this turn's request captured {len(self._captured)} HTTP "
                "response(s), and exactly one is what a single provider call "
                "produces. This build owns its retry policy and disables the "
                "SDK's, so a count other than one means the body a measurement "
                "would be taken from cannot be identified — and a measurement "
                "taken from an unidentified body is not a measurement"
            )
        return self._captured[0]


#: The one place a capture wrapper is decided on, across every client in this
#: process.
#:
#: :func:`ensure_capturing_client` reads a client's HTTP client, wraps what it
#: read and installs the wrapper. Per wrapper, the claim already decides which
#: adapter owns a capture; but two adapters constructed at once over one
#: *unwrapped* client never meet on one wrapper to begin with. Each reads the
#: same raw client, builds its own wrapper and claims the one it built, so both
#: claims succeed on two different objects and both constructors return —
#: while only the wrapper installed last is the one the SDK client will send
#: through. The other adapter is left holding a capture attached to nothing,
#: told it was built, and refused at its first turn instead: a configuration
#: error charged as a spent turn.
#:
#: So the read, the wrap, the claim and the install are made one step. The lock
#: is module-level because the state being decided is *the client's*, and two
#: adapters racing over one client have no earlier object in common to lock on.
#:
#: What this serialises is setup, and only setup: everything under it is
#: attribute reads on an SDK configuration object. No provider call, no request
#: and no wait for one is made while it is held — dispatch happens well outside
#: it — so this does not put a global lock in front of two adapters' concurrent
#: turns, which would make a benchmark's measured wall-clock depend on
#: contention no run records.
#:
#: Process-local, and this build does not pretend otherwise: two processes
#: sharing one SDK client is not an arrangement Python's object model permits,
#: so there is nothing wider to guard against.
_CAPTURE_INSTALL_LOCK = threading.Lock()


def ensure_capturing_client(
    client: Mistral, *, owner: object | None = None
) -> CapturedHttpClient:
    """Wrap this client's HTTP client so the wire can be read, exactly once.

    Idempotent, and that matters: a client wrapped twice would hold two copies
    of a body and make "the response for this call" ambiguous — the one question
    the capture exists to answer.

    Idempotent *under concurrency*, too, which is what
    :data:`_CAPTURE_INSTALL_LOCK` is for: checking for an existing wrapper,
    building one, claiming it and installing it are one step, so two callers
    over one client meet on one wrapper and exactly one of them owns it. See the
    lock's own comment for why it is module-level and why serialising it costs
    no measured wall-clock.

    ``owner`` is the adapter instance the capture is being established *for*.
    Given one, the wrapper is claimed by it — one capture, one adapter, for that
    adapter's lifetime — and a client whose capture another adapter already owns
    is refused here rather than at dispatch. Omitted, the client is wrapped and
    left unclaimed, which is what a caller establishing the wrapper outside an
    adapter is doing.

    A client with no configured HTTP client is refused here rather than at
    dispatch. There would be nothing to wrap and therefore no way to see what
    the provider sent, and a run that could not check its responses must not
    make a request in the first place.
    """
    with _CAPTURE_INSTALL_LOCK:
        configuration = getattr(client, "sdk_configuration", None)
        inner = getattr(configuration, "client", None)
        if configuration is None or inner is None:
            raise MistralConfigurationError(
                "this client exposes no configured HTTP client for this build to read "
                "the provider's response through. This SDK discards fields its own "
                "schema does not declare before any object this build can inspect "
                "exists, so without the raw response there is no way to check that a "
                "body is the response contract this run asked for — and an unchecked "
                "response must not be paid for. Build the client with `build_client`"
            )
        if isinstance(inner, CapturedHttpClient):
            if owner is not None:
                inner.claim(owner)
            return inner
        wrapper = CapturedHttpClient(inner)
        # Claimed before it is installed, so that a refusal — which this branch's
        # fresh wrapper cannot raise, but the branch above can — never leaves a
        # client half-rewired.
        if owner is not None:
            wrapper.claim(owner)
        configuration.client = wrapper
        return wrapper


#: The one endpoint this build talks to. The SDK appends its own ``/v1`` path.
MISTRAL_BASE_URL = "https://api.mistral.ai"

#: The credential, and the only place it may come from.
API_KEY_VARIABLE = "MISTRAL_API_KEY"

#: Environment variables that would silently redirect this request.
REJECTED_VARIABLES: tuple[str, ...] = ("MISTRAL_BASE_URL", "MISTRAL_SERVER_URL")

#: The SDK's own retry loop, disabled.
#:
#: The strategy is ``"none"`` and the backoff values are inert placeholders the
#: constructor requires; nothing reads them under that strategy. Passed
#: explicitly rather than left to the SDK's default because the default is a
#: property of the SDK version, and this build's claim that it owns its retries
#: must not depend on one.
_NO_SDK_RETRIES = RetryConfig("none", BackoffStrategy(1, 1, 1, 1), False)


def build_client(*, api_key: str, http_client: httpx.Client | None = None) -> Mistral:
    """The SDK client this build talks through, with its retry loop disabled.

    ``http_client`` exists so a test can bind a real client to an in-process
    transport. It is not a way to reach a different service: the server URL is
    pinned here and is not a parameter.
    """
    return Mistral(
        api_key=api_key,
        server_url=MISTRAL_BASE_URL,
        client=http_client,
        retry_config=_NO_SDK_RETRIES,
    )


def check_client_retries_disabled(client: Mistral) -> None:
    """Prove this client will not retry underneath the adapter's own loop.

    The other two integrations read ``max_retries`` off their client; this SDK
    carries a whole retry *strategy* object instead, so the check is that the
    strategy is the one that does nothing. A second, hidden retry loop would
    spend the episode's wall-clock budget on calls the runner never saw, make a
    turn's measured latency describe an unknown number of requests, and hide a
    rate limit the run should have recorded.
    """
    configured = getattr(client.sdk_configuration, "retry_config", None)
    strategy = getattr(configured, "strategy", None)
    if strategy != _NO_SDK_RETRIES.strategy:
        raise MistralConfigurationError(
            "this client was built with an SDK-level retry strategy of "
            f"{strategy!r}. The adapter owns the retry policy, so a second, "
            "hidden one underneath it would spend the episode's wall-clock "
            "budget on calls the runner never saw, make a turn's measured "
            "latency describe an unknown number of requests, and hide a rate "
            "limit the run should have recorded. Build the client with "
            "`build_client`, which disables it"
        )


def check_client_endpoint(client: Mistral) -> None:
    """Prove this client sends Mistral traffic to the endpoint settings record.

    This SDK does not expose a base URL attribute the way the other two do: the
    server is part of its configuration object, and a client built with no
    ``server_url`` falls back to a default. So the value checked is the one the
    configuration itself resolves — ``get_server_details`` — which is what the
    request will actually be built against, rather than the constructor argument
    that may or may not have been supplied.

    Checked in the adapter's constructor and again immediately before dispatch,
    because :func:`build_client` pins the server and a caller handing in their
    own client does not.
    """
    configuration = getattr(client, "sdk_configuration", None)
    details = getattr(configuration, "get_server_details", None)
    if details is None:
        raise MistralConfigurationError(
            "this client exposes no SDK configuration this build can read a server "
            "URL from, so there is no way to prove it sends the credential to "
            f"{MISTRAL_BASE_URL}, which is the endpoint adapter settings record. "
            "Build the client with `build_client`"
        )
    check_endpoint(
        details()[0],
        expected=MISTRAL_BASE_URL,
        provider=MISTRAL_PROVIDER,
        setting="base_url",
        error=MistralConfigurationError,
    )


def build_mistral_adapter(
    *,
    model: str,
    configuration_id: str | None = None,
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    retry: ProviderRetryPolicy | None = None,
    cost_guard: RunCostGuard | None = None,
) -> MistralChatAdapter:
    """Everything a credentialed run needs, or a clean refusal.

    The order is load-bearing: the configuration is refused before
    ``MISTRAL_API_KEY`` is read, so an unauthorised invocation never handles the
    credential at all.
    """
    checked = check_model(
        model, provider=MISTRAL_PROVIDER, error=MistralConfigurationError
    )
    # Before the client exists, and so before a credential has been read.
    check_retry_policy(
        retry or ProviderRetryPolicy(),
        provider=MISTRAL_PROVIDER,
        error=MistralConfigurationError,
    )
    check_configuration_audited(
        configuration_id,
        error=MistralConfigurationError,
        audited=audited,
        environ=environ,
    )
    check_environment(
        REJECTED_VARIABLES,
        provider=MISTRAL_PROVIDER,
        base_url=MISTRAL_BASE_URL,
        error=MistralConfigurationError,
        environ=environ,
    )
    api_key = resolve_api_key(
        API_KEY_VARIABLE,
        provider=MISTRAL_PROVIDER,
        error=MistralConfigurationError,
        environ=environ,
    )
    return MistralChatAdapter(
        model=checked,
        client=build_client(api_key=api_key),
        retry=retry,
        cost_guard=cost_guard,
    )


def mistral_identity(model: str) -> AdapterIdentity:
    """Who answered: the service, the pinned model, and this integration."""
    return AdapterIdentity(
        provider=MISTRAL_PROVIDER,
        model=model,
        implementation=MISTRAL_IMPLEMENTATION,
        version=MISTRAL_ADAPTER_VERSION,
    )


def sdk_version() -> str:
    """The installed SDK's version, read from distribution metadata.

    Read from the distribution rather than from a module attribute: this SDK is
    a namespace package and exposes no single ``__version__`` that names the
    distribution a run was executed under.
    """
    return _distribution_version("mistralai")


def profile_settings(
    profile: RequestProfile, retry: ProviderRetryPolicy
) -> dict[str, Any]:
    """Every result-affecting request setting, and no credential."""
    return {
        "action_surface": ACTION_SURFACE_CONTRACT,
        "api": MISTRAL_API,
        "base_url": MISTRAL_BASE_URL,
        "fact_affordance": FACT_AFFORDANCE_CONTRACT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "query_resolution": QUERY_RESOLUTION_CONTRACT,
        "request_mapping": REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_capture": MISTRAL_RESPONSE_CAPTURE,
        "response_contract": "exactly_one_tool_call",
        # The name records which reading of this service's own response extension
        # a run accepted bodies under; the digest is what makes that claim
        # re-derivable rather than a label. Both are inside ``configuration_id``.
        "response_extensions": MISTRAL_RESPONSE_EXTENSIONS,
        "response_extensions_digest": RESPONSE_EXTENSION_DIGEST,
        "retry": retry.as_dict(),
        "sdk": "mistralai",
        "sdk_injected_fields": dict(SDK_INJECTED_FIELDS),
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_version": sdk_version(),
        "tool_choice": TOOL_CHOICE,
    }


def mistral_settings(
    retry: ProviderRetryPolicy | None = None, *, model: str
) -> dict[str, Any]:
    """The settings a run of this model records, under this model's profile."""
    return profile_settings(request_profile_for(model), retry or ProviderRetryPolicy())


# -- the request -------------------------------------------------------------


def request_token_bound(payload: Mapping[str, Any]) -> int:
    """An upper bound on the input tokens this exact request can be charged for.

    One definition with two callers, which is the whole point of it existing at
    all: the adapter computes this before it asks the cost guard to authorise a
    request, and the ledger reader computes it again when it rebuilds what a
    recorded turn reserved. A second, separately written computation of the same
    bound would drift, and a drifted bound makes every stored reservation of an
    otherwise correct run unre-derivable — a ledger that was written honestly
    and can no longer be read.

    The bound itself is
    :func:`~boundarybench.providers.common.request_input_token_bound` over this
    integration's own body; the label is this integration's, so a failure while
    encoding names the request that failed.
    """
    return request_input_token_bound(payload, "mistral request bound")


def build_mistral_request(
    request: TurnRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    """The exact request body one turn sends, under this model's profile.

    What this returns is what *this build* authors. The SDK adds
    :data:`SDK_INJECTED_FIELDS` on top before the bytes leave the process, so
    the wire body is this mapping merged with those — see the module docstring.
    """
    shape = check_request_profile(profile, model=model)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "tools": [nested_tool_schema(action) for action in request.actions],
        "tool_choice": TOOL_CHOICE,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": canonical_json_text(
                    observed_case(request), "mistral turn request"
                ),
            },
        ],
    }
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    return payload


# -- the response ------------------------------------------------------------


def check_response_model(response: ChatCompletionResponse, *, model: str) -> None:
    """Prove the answer came from the model the run pinned and asked for.

    Neither model string is quoted: the response's is provider-controlled text,
    and the run's is already on every row.
    """
    if response.model != model:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the response states that a model other than the one this run pinned "
            "and requested produced it, so it is not evidence about the model "
            "under test and neither its action nor its token counts are accepted. "
            "Neither model identifier is recorded here: the response's is "
            "provider-controlled text, and the run's is already named by the "
            "adapter identity on every row",
        )


def check_wire_completion(raw: Mapping[str, Any]) -> None:
    """Prove the exact JSON the provider sent is the contract this build asked for.

    The check that has no equivalent on the parsed object for this SDK. A field
    the vendor's schema does not declare is gone by the time
    :class:`~mistralai.client.models.ChatCompletionResponse` exists, so a body
    carrying one — a surface that moved under a pinned SDK, or content that
    reached the response from somewhere other than the model — was previously
    accepted in full and silently.

    ``choices`` is optional on this surface, so its absence is left to
    :func:`parse_completion`, which reports it as the zero-choice answer it is.
    ``usage`` is not optional here: a turn this build cannot price is a turn it
    must not settle.

    Nothing that arrived is quoted; see :mod:`boundarybench.providers.wire`.
    """
    body = checked_wire_object(raw, RESPONSE_WIRE_SHAPE)
    for entry in checked_wire_list(body.get("choices") or [], kind="completion choices"):
        choice = checked_wire_object(entry, CHOICE_WIRE_SHAPE)
        message = choice["message"]
        if message is None:
            continue
        parsed_message = checked_wire_object(message, MESSAGE_WIRE_SHAPE)
        calls = parsed_message.get("tool_calls")
        if calls is None:
            continue
        for raw_call in checked_wire_list(calls, kind="tool calls"):
            call = checked_wire_mapping(raw_call, kind="tool call")
            # This SDK types a call's ``type`` as ``"function"`` or an
            # unrecognised string, and :func:`parse_completion` refuses anything
            # but the first. A call of another kind is a statement about the
            # answer rather than the transport, so it is left to that refusal.
            if call.get("type") not in (None, FUNCTION_TOOL_CALL):
                continue
            function = checked_wire_object(call, TOOL_CALL_WIRE_SHAPE)["function"]
            named = checked_wire_object(function, FUNCTION_WIRE_SHAPE)
            checked_wire_string(named["name"], kind="tool call name")
            # ``arguments`` is proven *present* and nothing more. Unlike the
            # other two surfaces, this SDK types the field as a JSON string or an
            # already-decoded object and refuses anything else during its own
            # parse — which, on this SDK, has already happened by the time this
            # check runs, because it exposes no raw-response surface and the body
            # is captured from the transport rather than deferred. Re-checking
            # the shape here would be a branch no response can reach, and what is
            # *inside* the field belongs to
            # :func:`~boundarybench.providers.common.decode_tool_arguments`,
            # which has to record it.
    usage = checked_wire_object(body["usage"], USAGE_WIRE_SHAPE)
    check_response_extensions(usage)
    wire_token_usage(usage, fields=USAGE_FIELDS)


def check_response_admissible(
    response: WireResponse[ChatCompletionResponse], *, model: str
) -> None:
    """Everything that must be true of a body before anything is read out of it.

    The same four questions the other two integrations ask, in the same order:
    the pinned model produced it, it carries no field the SDK's models do not
    declare, its ``choices`` are a list where the body sent one, and its token
    counts are countable. None of the four is a vendor's decision, which is why
    the rules live in ``providers.common`` and only the field names are here.

    ``choices`` is optional on this surface, so ``None`` is the SDK's own
    absence rather than a malformed list and is left to
    :func:`parse_completion`, which reports it as the zero-choice answer it is.
    Anything else in that field is a shape this API does not document.

    All four are :data:`~boundarybench.adapter.PROVIDER_FAULT_RESPONSE_INVALID`
    and all four are decided before the action or the usage is accepted.

    All of them now happen *after* the same body has been checked as the exact
    JSON the provider sent — see :func:`check_wire_completion`. That check is
    first because it is the only one that can see what the parse flattened: a
    coerced token count, or a union arm this SDK resolved to something this build
    would then read.

    The typed contract check names one narrow allowance, and the agreement check
    beside it is what pays for it: the one undeclared name this lane accepts is
    permitted on the usage block's SDK model only, and only after the typed
    reading of it has been proven to state exactly what the wire did — see
    :func:`check_response_extensions_agree`. Every *other* typed extra, at every
    other object and depth, still counts exactly as it did before.
    """
    check_wire_completion(response.wire)
    parsed = response.parsed
    check_response_model(parsed, model=model)
    check_response_contract(parsed, allowed_extras_by_model=TYPED_EXTENSION_EXTRAS)
    check_response_extensions_agree(response.wire, parsed)
    if parsed.choices is not None:
        check_response_collection(parsed.choices, kind="completion choices")
    usage = parsed.usage
    if usage is not None:
        checked_token_usage(usage.prompt_tokens, usage.completion_tokens)


def parse_completion(
    response: ChatCompletionResponse, request: TurnRequest
) -> AdapterCall:
    """The provider's answer, as exactly one action call, or a refusal."""
    choices = response.choices or []
    if len(choices) != 1:
        raise AdapterProtocolError(
            f"the answer carries {len(choices)} choice(s); this build asks for one "
            "completion per turn and cannot tell which of several is the action "
            "the agent took"
        )
    choice = choices[0]
    # Truncation is decided first, and before the call count, because it
    # *explains* the call count.
    if choice.finish_reason == OUTPUT_LIMIT_FINISH_REASON:
        raise AdapterOutputLimitError(
            "the answer reached this run's pinned output-token ceiling of "
            f"{MAX_OUTPUT_TOKENS} and was cut off there, so what it carries is a "
            "fragment rather than the action the model was making. This is an "
            "output-budget limit and not a malformed answer: the ceiling is this "
            "run's own setting, it is recorded in run identity, and raising it is "
            "the change that would let the same model finish"
        )
    if choice.message is None:
        # A choice with no message is not an answer this build can read. Reported
        # as the protocol failure it is rather than allowed to become an
        # attribute error four frames down, where it would be recorded as this
        # integration crashing instead of as the provider answering oddly.
        raise AdapterProtocolError(
            "the answer carries a choice with no message, so it states no action "
            "at all; this scaffold requires exactly one action per turn"
        )
    calls = choice.message.tool_calls or []
    if not isinstance(calls, list):
        raise AdapterProtocolError(
            "the answer's tool calls are not a list, so the answer does not state "
            "how many actions it carries. What arrived in that field is not quoted "
            "here: it is provider-controlled, and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        )
    for call in calls:
        # This SDK types a call's ``type`` as ``"function"`` or an unrecognised
        # string, so a surface that grew a second kind of call parses cleanly and
        # arrives here carrying a ``function`` field this adapter would otherwise
        # read. A call whose kind this build does not model is not evidence of
        # the action the model took, and reading its name and arguments as one
        # would record an action nobody can show was requested. The xAI adapter
        # refuses the same shape on the same grounds.
        if call.type != FUNCTION_TOOL_CALL:
            raise AdapterProtocolError(
                "the answer carries a tool call that does not declare itself a "
                "function call; this scaffold requires one named action with "
                "structured arguments, and a call of another kind is not one. The "
                "type the call declared is not quoted here: it is "
                "provider-controlled text, and a durable failure row records only "
                "this build's own fixed detail and closed-set values"
            )
        if getattr(call, "function", None) is None:
            # A call that declares itself a function call and carries no function
            # object names nothing. Refused here rather than reached for below,
            # where it would be an attribute error dressed as an adapter bug.
            raise AdapterProtocolError(
                "the answer carries a function tool call with no function object, "
                "so it names no action and no arguments; this scaffold requires "
                "one named action with structured arguments"
            )
    resolved = resolve_single_tool_call(
        [(call.function.name, call.function.arguments) for call in calls],
        request,
        label="mistral tool call arguments",
    )
    if choice.finish_reason != REQUIRED_FINISH_REASON:
        raise AdapterProtocolError(
            f"the answer stopped for something other than {REQUIRED_FINISH_REASON!r}, "
            "so the call it carries was cut short rather than completed and is not "
            "the action the model was making. The reason the provider stated is not "
            "quoted here: it is provider-controlled text, and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    return resolved


def response_usage(response: WireResponse[ChatCompletionResponse]) -> TokenUsage:
    """The two counts this build prices a turn on, taken from the wire.

    From the wire and not from the SDK's object, because this SDK's usage model
    both coerces and defaults: ``true`` reaches it as one token, ``"12"`` as
    twelve, and a usage block that states no counts at all reaches it as a
    *measured* zero. That last one is the worst of the three — a turn that
    really consumed tokens, settled at a cost of nothing, with its reservation
    released rather than held.

    Validated again here rather than trusted from
    :func:`check_response_admissible`, because this is the function whose return
    value becomes an :class:`~boundarybench.adapter.AdapterUsage` and a settled
    cost.
    """
    usage = checked_wire_object(response.wire.get("usage"), USAGE_WIRE_SHAPE)
    return wire_token_usage(usage, fields=USAGE_FIELDS)


def classify_exception(exception: Exception) -> Fault | None:
    """Map one SDK error onto the contract's fault set, or decline.

    This SDK talks through ``httpx`` and lets transport exceptions out
    unwrapped, so the two ``httpx`` cases are checked here rather than being
    reachable only through an SDK type. ``ResponseValidationError`` is checked
    before its ``MistralError`` base: it carries the status of a response that
    *arrived* — often a 200 — so classifying it by status would file a body this
    build could not parse as a rejected request.
    """
    if isinstance(exception, httpx.TimeoutException):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, httpx.TransportError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, NoResponseError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, ResponseValidationError):
        # A whole body arrived — the exception carries it as ``raw_response``,
        # and it is usually an HTTP 200 — and the SDK refused to read it. So the
        # attempt records that a response *was* received. Saying otherwise would
        # report a transport failure that did not happen, and would hide the one
        # case where the provider answered with something unparseable.
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, MistralError):
        return classify_http_status(exception.status_code)
    return None


# -- the adapter -------------------------------------------------------------


class MistralChatAdapter:
    """One provider client, one pinned model, one action per turn."""

    def __init__(
        self,
        *,
        model: str,
        client: Mistral,
        retry: ProviderRetryPolicy | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: RunCostGuard | None = None,
        profile: RequestProfile | None = None,
    ) -> None:
        self._profile = check_request_profile(profile, model=model)
        # Before anything else this client is trusted for: where it sends. The
        # factory builds its own client against the pinned server URL, so the
        # factory was never the way in — a caller handing in a client is, and
        # this constructor is public.
        check_client_endpoint(client)
        check_client_retries_disabled(client)
        self._retry = check_retry_policy(
            retry or ProviderRetryPolicy(),
            provider=MISTRAL_PROVIDER,
            error=MistralConfigurationError,
        )
        self._model = model
        self._client = client
        # Last, and after every check that can refuse this configuration: the
        # HTTP client this SDK will send through is wrapped so the response can
        # be read before the SDK parses it, and the wrapper is claimed by this
        # instance. Done in the constructor rather than at dispatch so a client
        # this build cannot read the wire through costs an error message rather
        # than a spent reservation and an unchecked body — and done here rather
        # than earlier so that a constructor which goes on to refuse the client
        # for some other reason leaves no claim behind on a capture no adapter
        # exists to use.
        self._capture = ensure_capturing_client(client, owner=self)
        self._executor: TurnExecutor[WireResponse[ChatCompletionResponse]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        # One turn at a time on this instance, and here it is load-bearing twice
        # over: besides the executor's usage and attempt evidence, the capture
        # above holds *the* response of the call in flight, and two overlapping
        # turns would leave it holding two — so neither turn could identify the
        # body it is measured from. See
        # :class:`~boundarybench.adapter.SingleFlight`.
        self._single_flight = SingleFlight(adapter="MistralChatAdapter")

    def __repr__(self) -> str:
        """Names the run, never the credential."""
        return (
            f"<MistralChatAdapter provider={MISTRAL_PROVIDER} "
            f"model={self._model} implementation={MISTRAL_IMPLEMENTATION}>"
        )

    @property
    def identity(self) -> AdapterIdentity:
        return mistral_identity(self._model)

    @property
    def request_profile(self) -> RequestProfile:
        return self._profile

    @property
    def settings(self) -> Mapping[str, Any]:
        return profile_settings(self._profile, self._retry)

    @property
    def cost_guard(self) -> RunCostGuard | None:
        # Narrowed back to the concrete guard a Boundary run builds, and sound
        # by construction: the only thing that ever reaches the executor is this
        # constructor's ``cost_guard``. The shared executor asks a three-method
        # protocol because it must not know what a token costs.
        return cast("RunCostGuard | None", self._executor.cost_guard)

    def last_usage(self) -> AdapterUsage:
        return self._executor.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._executor.last_telemetry()

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        """One turn: clear the measurement, ask, measure, then read the answer.

        Under this instance's single-flight guard, taken *before* the clearing.
        Two overlapping turns here would each clear the shared capture and each
        find two responses in it, so both would refuse to measure a body that
        did arrive — and the second caller would also have erased the first's
        telemetry. The second call is refused immediately instead: it dispatches
        nothing and changes nothing.
        """
        with self._single_flight:
            self._executor.clear()
            payload = build_mistral_request(
                request, model=self._model, profile=self._profile
            )
            response = self._send_payload(payload, deadline)
            return parse_completion(response.parsed, request)

    def _send_payload(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[ChatCompletionResponse]:
        """Check the body against the settings, then run the turn.

        The check covers the fields this build authors. The SDK's injected ones
        are declared in :data:`SDK_INJECTED_FIELDS` and recorded in settings, so
        the run's description of the request the provider received is complete
        without this check having to pretend it authored them.

        The endpoint is proven again here and not only in the constructor: this
        SDK's configuration object is mutable, so a client that was pinned when
        the adapter was built can be pointed somewhere else before the turn that
        uses it, and the check that matters is the one immediately before the
        credential is sent.

        The capture wrapper is re-established here for the same reason: the
        configuration object is mutable, so a client that was wrapped when the
        adapter was built can have its HTTP client replaced before the turn that
        uses it, and a request whose response this build could not read must not
        be made.
        """
        check_client_endpoint(self._client)
        self._capture = ensure_capturing_client(self._client, owner=self)
        check_payload_fields(
            payload,
            profile_id=self._profile.profile_id,
            sent=self._profile.sent_fields(),
            omitted=self._profile.omitted_fields(),
        )
        return self._executor.run(
            payload=payload,
            deadline=deadline,
            token_bound=request_token_bound(payload),
            dispatch=lambda timeout: self._dispatch(payload, timeout),
            check_before_dispatch=self._check_before_dispatch,
            classify=classify_exception,
            check_identity=lambda response: check_response_admissible(
                response, model=self._model
            ),
            usage_of=response_usage,
        )

    def _check_before_dispatch(self) -> None:
        check_client_endpoint(self._client)
        check_client_retries_disabled(self._client)
        self._capture = ensure_capturing_client(self._client, owner=self)

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[ChatCompletionResponse]:
        """One request, with this attempt's remaining budget as its timeout.

        The SDK takes milliseconds. The conversion rounds *up* to at least one,
        because a sub-millisecond remainder that truncated to zero would be read
        as "no timeout" and let a request outlive the episode that authorised
        it — the opposite of what a spent budget means.

        The capture is cleared immediately before the call and exactly one
        response is taken immediately after it, so what this turn is measured
        from is this turn's own body and nothing else. The text is read here and
        the response is dropped; the bytes live for the length of the turn and
        are never written anywhere.

        Cleared in a ``finally``, so that is true of every way out of this call
        and not only the one that returns. The case that made it necessary is
        the malformed HTTP 200: the SDK reads a whole body, refuses to parse it
        and raises ``ResponseValidationError`` from inside ``complete``, so the
        line below that consumes the response never runs and a provider body
        would be held on this client past the end of the turn — and would be
        waiting there as a second capture for the next one. Nothing about the
        attempt evidence depends on this: the fault, and the fact that a
        response *arrived*, are classified from the exception's own
        ``raw_response``, which this does not touch.
        """
        self._capture.clear()
        try:
            response = self._client.chat.complete(
                **payload, timeout_ms=max(1, int(timeout * 1000))
            )
            if response is None:
                # A ``None`` return is not a response this build can read. Raised
                # as the SDK's own no-response error so it lands in the taxonomy
                # through the same path every other transport failure does.
                raise NoResponseError("the SDK returned no response object")
            text = self._capture.take().text
        finally:
            self._capture.clear()
        return WireResponse(text, lambda: response, kind="mistral completion")


__all__ = [
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "CHOICE_WIRE_SHAPE",
    "EXTENSION_COUNT",
    "FUNCTION_TOOL_CALL",
    "FUNCTION_WIRE_SHAPE",
    "MAX_EXTENSION_COUNT",
    "MAX_OUTPUT_TOKENS",
    "MESSAGE_WIRE_SHAPE",
    "MISTRAL_ADAPTER_VERSION",
    "MISTRAL_API",
    "MISTRAL_BASE_URL",
    "MISTRAL_IMPLEMENTATION",
    "MISTRAL_PROVIDER",
    "MISTRAL_RESPONSE_CAPTURE",
    "MISTRAL_RESPONSE_EXTENSIONS",
    "MISTRAL_SMALL_MODEL",
    "MISTRAL_SMALL_PROFILE",
    "MODEL_REQUEST_PROFILES",
    "OUTPUT_LIMIT_FINISH_REASON",
    "PARALLEL_TOOL_CALLS",
    "PROHIBITED_REQUEST_FIELDS",
    "REJECTED_VARIABLES",
    "REQUEST_MAPPING_VERSION",
    "REQUIRED_FINISH_REASON",
    "RESPONSE_EXTENSION_CONTRACT",
    "RESPONSE_EXTENSION_DIGEST",
    "RESPONSE_EXTENSION_SCHEMA",
    "RESPONSE_WIRE_SHAPE",
    "SDK_INJECTED_FIELDS",
    "TEMPERATURE",
    "TOOL_CALL_WIRE_SHAPE",
    "TOOL_CHOICE",
    "TYPED_EXTENSION_EXTRAS",
    "USAGE_EXTENSION_NAMES",
    "USAGE_FIELDS",
    "USAGE_WIRE_SHAPE",
    "CapturedHttpClient",
    "MistralChatAdapter",
    "MistralConfigurationError",
    "RequestProfile",
    "build_client",
    "build_mistral_adapter",
    "build_mistral_request",
    "check_client_endpoint",
    "check_client_retries_disabled",
    "check_request_profile",
    "check_response_admissible",
    "check_response_extensions",
    "check_response_extensions_agree",
    "check_response_model",
    "check_wire_completion",
    "classify_exception",
    "ensure_capturing_client",
    "mistral_identity",
    "mistral_settings",
    "parse_completion",
    "profile_settings",
    "request_profile_for",
    "request_token_bound",
    "response_usage",
    "sdk_version",
]
