"""The xAI adapter, over xAI's OpenAI-compatible Chat Completions surface.

**This integration does not use ``xai-sdk``, and its name says so.** The
implementation string it records is ``xai_openai_compat_chat_completions``, the
SDK it names in settings is ``openai``, and the API it names is a compatibility
surface. All three go into run identity, so a run executed through this module
can never be read as a run executed through the vendor's own client.

Why the vendor's client was not used, recorded here because the decision is a
limitation and not a preference:

* ``xai-sdk`` is a gRPC/protobuf client. Its ``Client`` exposes no channel,
  interceptor or transport injection point — only a host name and channel
  options — so there is no way to drive it in-process. Every other adapter in
  this build proves what it puts on the wire by running the *real* SDK over an
  in-process HTTP transport and asserting the serialised body; against
  ``xai-sdk`` the equivalent test would have to monkeypatch the generated stub,
  which proves only that the adapter agrees with the stub. That is precisely the
  test this repository refuses to rely on elsewhere.
* It would also add ``grpcio`` and ``protobuf`` to a published distribution's
  dependency closure for one of five models.

What that costs is real and is not hidden: this build talks to xAI through an
interface xAI documents as OpenAI-compatible rather than through its first-party
one, so a behaviour that differs between the two — a field the compatibility
layer drops, a usage counter it rounds, a tool-call shape it normalises — would
be invisible from here. An operator who needs the first-party client needs a
different adapter, and it would be a different configuration identity.

One thing this adapter accepts deserves naming beside that limitation, because
it is the limitation made visible. A real response from this compatibility
endpoint carries fields the pinned OpenAI SDK does not declare — one on the
assistant message, two on the usage block, and a pair inside a detail object the
SDK *does* declare — so this lane names an exact contract of its own for them,
:data:`XAI_COMPAT_RESPONSE_EXTENSIONS`, and accepts precisely it: closed per
site, typed scalar by scalar, bounded, and read for nothing. It is checked on
both readings of the same bytes and over the union of what each of them states,
so an object or a field only one reading carries is refused rather than compared
with nothing — see :func:`check_response_extensions_agree`. The contract is
provider-*observed* rather than vendor-declared, which is why it carries a
digest in settings and why accepting it moved this integration's version. What
this build *sends* is untouched: the model's published page states that it
reasons and that it calls functions, and publishes no control that would turn
the first of those off, so no such parameter is invented here.

Everything else follows the same three properties every adapter in this build
holds to: the provider boundary stays narrow, every result-affecting setting is
pinned and hashed into run identity, and the credential never becomes data.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

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
from boundarybench.jsonsafe import (
    canonical_json_bytes,
    canonical_json_text,
    surrogate_index,
)
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
    check_environment,
    check_model,
    check_payload_fields,
    check_response_collection,
    check_response_contract,
    check_response_object,
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
from operatebench.providers.openai_compat_client import (
    check_xai_openai_compat_client_state,
)

#: The service under test, as recorded in run identity.
XAI_PROVIDER = "xai"

#: The code that talks to it, named for what it actually is.
#:
#: Load-bearing, and the reason this constant has a docstring at all: the
#: implementation string is hashed into run identity and printed in reports, so
#: it is this build's durable claim about *what talked to the service*. Naming
#: the vendor's SDK here while using a compatibility endpoint would make every
#: row this adapter writes false in the one place an audit checks.
XAI_IMPLEMENTATION = "xai_openai_compat_chat_completions"

#: This integration's own version. Moves when anything it sends or accepts moves.
#:
#: ``0.2.0`` was where it started reading the wire. See the OpenAI integration's
#: constant of the same name: the same SDK, the same raw-response surface, the
#: same header on the request and the same change to what a response has to
#: state before it is believed.
#:
#: ``0.3.0`` is where it began accepting this endpoint's own response extensions
#: — see :data:`XAI_COMPAT_RESPONSE_EXTENSIONS`. A body ``0.2.0`` refused whole
#: is read here as an action and a measured turn, so what a run will read as an
#: answer changed and the identity a row carries has to say which reading it was.
#: What this integration *sends* did not change at all: the request body is
#: byte-identical to ``0.2.0``'s, temperature pinned to zero and no reasoning
#: field, and the tests assert that against the body the transport received.
XAI_ADAPTER_VERSION = "0.3.0"

#: How this adapter gets at the bytes the provider sent. The OpenAI SDK's own
#: raw-response surface, because the OpenAI SDK is what talks to this endpoint.
XAI_RESPONSE_CAPTURE = "openai_sdk_with_raw_response_v1"

#: The API surface this adapter speaks.
XAI_API = "chat_completions"

#: That the surface is a compatibility layer rather than the vendor's own API,
#: stated as its own setting so a reader does not have to infer it from the
#: implementation name.
XAI_API_COMPATIBILITY = "openai_compatible"

#: How a :class:`~boundarybench.adapter.TurnRequest` becomes a request body.
REQUEST_MAPPING_VERSION = "xai_openai_compat_turn_request_json_v1"

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

#: The fields whose presence or absence is a *profile's* decision.
#:
#: ``reasoning_effort`` is in the set although no profile this build ships sends
#: it. Grok models expose a reasoning control, so "this build does not send one"
#: is a claim worth recording in run identity rather than a silence.
PROHIBITED_REQUEST_FIELDS: tuple[str, ...] = (
    "reasoning_effort",
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
    reasoning_effort: str | None
    #: Whether this build transcribed the shape from the model's own published
    #: request contract, or pinned it as its own choice. See
    #: :data:`GROK_4_5_PROFILE`.
    contract_verified: bool = True

    def sent_fields(self) -> tuple[str, ...]:
        fields = list(_COMMON_REQUEST_FIELDS)
        if self.temperature is not None:
            fields.append("temperature")
        if self.reasoning_effort is not None:
            fields.append("reasoning_effort")
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
            "reasoning_effort": self.reasoning_effort,
            "model_contract_verified": self.contract_verified,
        }


#: The one xAI model this lane pins.
GROK_4_5_MODEL = "grok-4.5"

BASELINE_PROFILE = RequestProfile(
    profile_id="xai_baseline_temperature_zero_reasoning_omitted_v1",
    temperature=TEMPERATURE,
    reasoning_effort=None,
    contract_verified=False,
)

#: ``grok-4.5``: sampling pinned to zero, no reasoning control sent.
#:
#: ``contract_verified`` is ``False``, for the same reason it is on the OpenAI
#: profile and one more besides. This repository read the model's price and its
#: identifier off xAI's published model page; it found no published statement
#: about how this model treats a sampling parameter or an absent reasoning
#: control. And because the request travels through a compatibility layer rather
#: than the vendor's own client, even a documented first-party contract would
#: not settle what this endpoint does with the field. So the shape is this
#: build's pinned choice, the settings say so, and an operator authorising a
#: live configuration authorises that claim with the rest.
GROK_4_5_PROFILE = RequestProfile(
    profile_id="grok45_temperature_zero_reasoning_omitted_v1",
    temperature=TEMPERATURE,
    reasoning_effort=None,
    contract_verified=False,
)

MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {GROK_4_5_MODEL: GROK_4_5_PROFILE}
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
    raise XAIConfigurationError(
        f"request profile {profile.profile_id!r} is not the profile this build "
        f"sends to model {model!r}, which is {expected.profile_id!r}"
    )


#: The only ``finish_reason`` a turn may end on.
REQUIRED_FINISH_REASON = "tool_calls"

#: The one other finish reason this build classifies rather than redacts,
#: because it names a cause *this run controls*: the answer ran into
#: :data:`MAX_OUTPUT_TOKENS`.
OUTPUT_LIMIT_FINISH_REASON = "length"

#: The two counts this surface's usage block is read for. Not the Responses
#: API's pair: the same two numbers are called ``prompt_tokens`` and
#: ``completion_tokens`` here, and a reader that proved the wrong pair present
#: would have proven nothing about the fields it then read.
USAGE_FIELDS: tuple[str, ...] = ("prompt_tokens", "completion_tokens")

#: What :func:`parse_completion` reads off one choice, and therefore what a
#: choice has to state before it is read as one.
CHOICE_FIELDS: tuple[str, ...] = ("finish_reason", "message")

#: What it reads off that choice's message.
MESSAGE_FIELDS: tuple[str, ...] = ("tool_calls",)

#: What it reads off one tool call in that message.
TOOL_CALL_FIELDS: tuple[str, ...] = ("type",)

# -- the compatibility layer's response extensions ----------------------------
#
# This endpoint returns, on an otherwise ordinary HTTP 200 from the pinned model,
# a small set of fields the pinned OpenAI SDK does not declare. Under the
# response contract every one of them is an undeclared field, so the whole body
# was refused: the action went unparsed, the usage went unmeasured, and the
# turn's reservation was held as exposure over fields nothing here reads. That is
# the correct default and the wrong answer for a field set that is stable, named
# and inert.
#
# So this lane names **one exact contract** for them and accepts precisely it.
# Four properties make that safe to do, and they are the same four the OpenAI
# lane's block is accepted under:
#
# * It is *closed*, and per site. A fixed set of names on the assistant message,
#   a fixed pair on the usage block and a fixed pair inside the SDK's own
#   ``prompt_tokens_details`` — and nothing else, anywhere. A name this contract
#   fixes for one object buys nothing on another, so an extension name that
#   arrives at the root, or a usage name on a message, is still
#   ``provider_response_invalid``.
# * It is *optional where absence was observed, complete where a group was*.
#   Each message and usage member may be absent on its own: this build reads none
#   of them, so a service that stops sending one has not broken anything a run
#   depends on. The detail pair is the one group that is all-or-nothing, because
#   half of it was never observed and accepting half would widen an exact
#   contract into a guess.
# * It is *typed*, on the wire rather than after coercion. The counts are exact
#   non-negative integers inside a stated bound and never booleans, strings,
#   fractions or nulls; the reasoning string is an exact JSON string of valid
#   Unicode scalars inside a stated length.
# * It is *inert*. Nothing here reaches the action, the model's stated identity,
#   or the counts a turn is priced and capped on. In particular
#   ``cost_in_usd_ticks`` is **not** a cost: this build's measured cost is the
#   run's pinned price policy times the two token counts, and a provider's own
#   number for what it charged is not a measurement this benchmark can re-derive.
#
# What this is **not** is a vendor declaration. It transcribes the names and type
# shape two bounded schema-only diagnostics established against this exact
# endpoint; the raw bodies were not kept, and the published model page states
# that this model reasons and function-calls without publishing a schema for any
# of these fields — or a control that would turn the first of them off. So the
# request profile is unchanged and no disable parameter is invented here. A
# provider-observed contract can move without notice, so it is named *and*
# hashed in adapter settings, and both are inside ``configuration_id``.

#: The name of that contract, as adapter settings record it.
XAI_COMPAT_RESPONSE_EXTENSIONS = "xai_compat_response_extensions_v1"

#: The scalar vocabulary the contract fixes. Two kinds, and each one is a
#: closed-set label this build chose, so it may appear in a durable failure
#: detail where a field name or a value may not.
EXTENSION_COUNT = "non_negative_bounded_integer"
EXTENSION_REASONING_TEXT = "bounded_reasoning_text"

#: The largest integer any of these counts may state.
#:
#: :data:`~boundarybench.pricing.MAX_EXACT_TOKEN_COUNT`, and deliberately the
#: same constant the token counts are bounded by rather than a second number
#: that means the same thing: it is the largest integer that survives a JSON
#: number in every reader this build's artefacts cross, so a value past it is one
#: this build could not carry unchanged even if it wanted to.
MAX_EXTENSION_COUNT = MAX_EXACT_TOKEN_COUNT

#: The longest reasoning string this contract accepts.
#:
#: Tied to this run's own pinned output ceiling rather than guessed at, because
#: the ceiling is the one thing about the length of a generated answer this build
#: controls: sixteen characters per token is far above what any tokenizer this
#: surface could be using emits, so :data:`MAX_OUTPUT_TOKENS` times sixteen
#: bounds any reasoning text a turn under this ceiling could return. It is also
#: two orders of magnitude above the longest string the diagnostics observed,
#: which is what makes it conservative in both directions: nothing real is
#: refused, and a body held in memory for the length of a turn stays bounded.
MAX_REASONING_CONTENT_CHARACTERS = MAX_OUTPUT_TOKENS * 16


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


#: Where the assistant message's extension is fixed.
MESSAGE_EXTENSION_SITE = "completion_message"

#: Where the usage block's two are.
USAGE_EXTENSION_SITE = "usage_block"

#: Where the pair inside the SDK's own detail object is.
PROMPT_TOKENS_DETAILS_EXTENSION_SITE = "prompt_token_details"

#: The contract's shape: which names may appear on which documented object, and
#: the exact wire kind each of them must be when it does.
RESPONSE_EXTENSION_SCHEMA: Mapping[str, Mapping[str, str]] = _frozen(
    {
        MESSAGE_EXTENSION_SITE: {"reasoning_content": EXTENSION_REASONING_TEXT},
        PROMPT_TOKENS_DETAILS_EXTENSION_SITE: {
            "image_tokens": EXTENSION_COUNT,
            "text_tokens": EXTENSION_COUNT,
        },
        USAGE_EXTENSION_SITE: {
            "cost_in_usd_ticks": EXTENSION_COUNT,
            "num_sources_used": EXTENSION_COUNT,
        },
    }
)

#: The sites whose members arrive together or not at all.
#:
#: One of the three, and the difference is what was observed rather than a
#: preference. The message's single member and the usage block's two were each
#: seen to come and go on their own; the detail pair was only ever seen whole, so
#: a body stating one half of it is not the shape this contract transcribed.
RESPONSE_EXTENSION_SITES_REQUIRED_WHOLE: frozenset[str] = frozenset(
    {PROMPT_TOKENS_DETAILS_EXTENSION_SITE}
)

#: The whole contract, as one document: the name, the shape, the two bounds and
#: the per-site rule that decides what a partly stated group means.
#:
#: It exists because the shape alone is not the contract. A build that
#: transcribed these same names and kinds and then accepted a lone
#: ``image_tokens`` would read a body this one refuses, and a digest taken over
#: the transcription alone would be equal across the two — so two manifests would
#: claim one configuration for two readings. Hashing the rules and the bounds
#: beside the shape is what makes the recorded claim answer "which bodies will
#: this run read as answers".
RESPONSE_EXTENSION_CONTRACT: Mapping[str, Any] = _frozen(
    {
        "contract": XAI_COMPAT_RESPONSE_EXTENSIONS,
        # Stated in the document rather than left to prose: this build reads no
        # value in it, which is why every member of it is allowed to be absent.
        "extension_values_read": False,
        "max_count": MAX_EXTENSION_COUNT,
        "max_reasoning_characters": MAX_REASONING_CONTENT_CHARACTERS,
        "members": RESPONSE_EXTENSION_SCHEMA,
        "members_required_together": {
            site: site in RESPONSE_EXTENSION_SITES_REQUIRED_WHOLE
            for site in RESPONSE_EXTENSION_SCHEMA
        },
    }
)

#: A digest over that document, recorded beside its name.
#:
#: The name alone would be a label: two builds could ship one
#: ``xai_compat_response_extensions`` version and accept different bodies, and
#: the run manifests would agree. The digest moves the moment the accepted
#: contract does, and it is inside ``configuration_id``.
RESPONSE_EXTENSION_DIGEST = hashlib.sha256(
    canonical_json_bytes(
        _plain(RESPONSE_EXTENSION_CONTRACT), "xai compat response extensions"
    )
).hexdigest()

#: The extension names each site may carry, as a set, for the two checks that
#: need one: the wire's allowed field set and the typed extras allowance.
MESSAGE_EXTENSION_NAMES: frozenset[str] = frozenset(
    RESPONSE_EXTENSION_SCHEMA[MESSAGE_EXTENSION_SITE]
)
USAGE_EXTENSION_NAMES: frozenset[str] = frozenset(
    RESPONSE_EXTENSION_SCHEMA[USAGE_EXTENSION_SITE]
)
PROMPT_TOKENS_DETAILS_EXTENSION_NAMES: frozenset[str] = frozenset(
    RESPONSE_EXTENSION_SCHEMA[PROMPT_TOKENS_DETAILS_EXTENSION_SITE]
)

#: The same permission, expressed for the *typed* reading: which pinned SDK
#: response model is entitled to carry which of these names in ``model_extra``.
#:
#: Keyed by model rather than by name so the allowance is exactly as narrow as
#: the wire's: a name proven for the usage block buys nothing on a message, and
#: no object outside these three accepts an undeclared field at all.
TYPED_EXTENSION_EXTRAS: Mapping[type, frozenset[str]] = MappingProxyType(
    {
        ChatCompletionMessage: MESSAGE_EXTENSION_NAMES,
        CompletionUsage: USAGE_EXTENSION_NAMES,
        PromptTokensDetails: PROMPT_TOKENS_DETAILS_EXTENSION_NAMES,
    }
)

#: How a refusal names what it was checking. One fixed label for every site:
#: which field carried the fault is not something a durable row needs, and the
#: value beside it is provider-controlled.
_EXTENSION_KIND = "named response extension"


def _extension_scalar_valid(value: Any, kind: str) -> bool:
    """Whether one value is the exact wire scalar the contract fixes for it.

    Exact, and by ``type`` rather than by ``isinstance``, everywhere it matters.
    Python makes ``True`` an ``int``, so an ``isinstance`` count check would
    accept a boolean as one — and a lenient reader downstream would render it as
    a measurement nobody took.
    """
    if kind == EXTENSION_COUNT:
        # The same rule the token counts are read under, and deliberately the
        # same function: two independently written definitions of "an exact
        # non-negative integer on the wire" would drift.
        return is_wire_token_count(value) and value <= MAX_EXTENSION_COUNT
    # :data:`EXTENSION_REASONING_TEXT`, the other of the two. The vocabulary is
    # closed and is this module's own, so there is no third kind to fall through
    # to.
    #
    # Empty is accepted. A model that stated no reasoning for a turn has made a
    # statement this build reads nothing out of either way, and refusing it would
    # refuse a body that can be read perfectly well. What is not accepted is text
    # UTF-8 cannot encode: a lone surrogate is not a Unicode scalar, nothing in
    # this build could hash or write it, and a value that could not survive being
    # recorded is not one to accept in the first place — even here, where it
    # never will be.
    return (
        type(value) is str
        and len(value) <= MAX_REASONING_CONTENT_CHARACTERS
        and surrogate_index(value) is None
    )


def check_response_extensions(node: Mapping[str, Any], *, site: str) -> None:
    """Prove every extension one documented object states is the exact fixed shape.

    Given the whole object rather than a projection of it, so a caller cannot
    accidentally hand over a filtered copy and prove the contract about something
    the provider did not send. Names this contract does not fix for this site are
    not this function's business: the object's own allowed field set refuses an
    undeclared field, and
    :func:`~boundarybench.providers.common.check_response_contract` refuses an
    undeclared typed extra.

    Two rules, and which one applies is the site's own property. Everywhere, a
    member that is present is checked against its exact wire kind. At a site in
    :data:`RESPONSE_EXTENSION_SITES_REQUIRED_WHOLE`, a member that is present
    also requires its siblings, because a partly stated group was never observed
    and this contract describes one shape rather than every subset of one.
    """
    schema = RESPONSE_EXTENSION_SCHEMA[site]
    present = [name for name in schema if name in node]
    if (
        site in RESPONSE_EXTENSION_SITES_REQUIRED_WHOLE
        and present
        and len(present) != len(schema)
    ):
        raise wire_invalid(
            f"the response states part of one group of {_EXTENSION_KIND}s that "
            f"{XAI_COMPAT_RESPONSE_EXTENSIONS} fixes as a whole. A group this "
            "build reads nothing out of may be absent entirely, but a group that "
            "is present is a statement, and this contract describes one shape for "
            "it rather than every subset of one: a partly stated group was never "
            "observed, so accepting it would widen an exact contract into a guess. "
            "Which members arrived and which did not is not recorded here: the "
            "absences are provider-controlled, and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        )
    for name in present:
        if not _extension_scalar_valid(node[name], schema[name]):
            raise wire_invalid(
                f"the response states a {_EXTENSION_KIND} whose value is not the "
                f"exact one of kind {schema[name]!r} that "
                f"{XAI_COMPAT_RESPONSE_EXTENSIONS} fixes for it, so what arrived "
                "is not the contract this build accepts and the body is refused "
                "rather than read past. Nothing about the value is recorded here — "
                "neither which field carried it nor what it was: the field set is "
                "this contract's own and the value is provider-controlled, and a "
                "durable failure row records only this build's own fixed detail "
                "and closed-set values"
            )


def _stated_extensions(node: Any, *, site: str) -> dict[str, Any]:
    """Exactly the extension members one object states, for the comparison below."""
    if not isinstance(node, Mapping):
        return {}
    return {name: node[name] for name in RESPONSE_EXTENSION_SCHEMA[site] if name in node}


def _typed_extras(value: Any) -> Mapping[str, Any]:
    """One parsed object's undeclared fields, as this SDK parked them."""
    extra = getattr(value, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}


def _object_stated_by_both(raw: Any, typed: Any) -> bool:
    """Whether both readings state the object one site lives on, or neither does.

    Two different questions, because the two readings answer in two different
    ways: the exact JSON states an object by carrying a JSON object there, and
    the typed reading states one by resolving it to something rather than to
    ``None``. A value that is neither — a count where this API documents an
    object, say — is not an object either reading stated, and the shape checks
    that surround this refuse it under the field its own reader is about to use.
    """
    return isinstance(raw, Mapping) == (typed is not None)


def _extension_readings(
    wire: Mapping[str, Any], parsed: ChatCompletion
) -> list[tuple[str, Any, Any]]:
    """Every object this contract fixes names on, as both readings of it.

    One entry per site the *union* of the two readings reaches, each carrying the
    object as the exact JSON states it beside the object as the typed reading
    states it — either of which may be nothing, which is precisely what the
    caller has to be able to see. Driving this from the typed side alone would
    make a site the SDK resolved to nothing invisible, and an extension stated
    under such an object would be compared with nothing and accepted although
    only one of the two readings ever stated it.

    A choice list is walked to the length of the longer reading for the same
    reason, and the typed one is guarded rather than trusted to be a list: this
    SDK resolves a union leniently, and a ``choices`` that is not a list is
    refused a step later by
    :func:`~boundarybench.providers.common.check_response_collection` rather than
    here as a ``TypeError``.
    """
    raw_choices = wire.get("choices")
    choices = raw_choices if isinstance(raw_choices, list) else []
    typed_choices = parsed.choices if isinstance(parsed.choices, list) else []
    readings: list[tuple[str, Any, Any]] = []
    for index in range(max(len(choices), len(typed_choices))):
        raw = choices[index] if index < len(choices) else None
        choice = typed_choices[index] if index < len(typed_choices) else None
        readings.append(
            (
                MESSAGE_EXTENSION_SITE,
                raw.get("message") if isinstance(raw, Mapping) else None,
                getattr(choice, "message", None),
            )
        )
    raw_usage = wire.get("usage")
    usage = parsed.usage
    readings.append((USAGE_EXTENSION_SITE, raw_usage, usage))
    readings.append(
        (
            PROMPT_TOKENS_DETAILS_EXTENSION_SITE,
            raw_usage.get("prompt_tokens_details")
            if isinstance(raw_usage, Mapping)
            else None,
            getattr(usage, "prompt_tokens_details", None),
        )
    )
    return readings


def check_response_extensions_agree(
    wire: Mapping[str, Any], parsed: ChatCompletion
) -> None:
    """Check the SDK's own reading of the extensions, and that the two agree.

    Each of these fields is stated twice by the time it gets here: once in the
    exact JSON the provider sent, and once on the typed response, where this SDK
    parks a field its models do not declare in ``model_extra`` rather than
    dropping it — on the message, on the usage block and inside the detail object
    alike. Both readings are validated, because a contract proven about one is
    not a contract about the other, and neither is preferred when they differ:
    there is no principled way to pick, so a body whose typed extras are not the
    extras on the wire is refused whole.

    Which sites are compared is the union of the two readings, and the objects
    those sites live on are the first thing compared. An object one reading
    states and the other does not is a disagreement in itself: the reading that
    has it could have stated extensions under it that the other reading had
    nowhere to state, so comparing only the sites both readings happen to carry
    would prove this contract about a body neither of them is. Presence is
    therefore proven at every site — the message on each choice, the usage block,
    and the detail object inside it — before a single value is read.

    What this deliberately does **not** do is clear or rewrite ``model_extra``.
    Emptying it would make the two readings agree by force, hide the very fields
    this contract exists to describe, and leave the SDK's typed object in a state
    the provider never sent. The typed parse stays exactly as the SDK produced
    it.
    """
    readings = _extension_readings(wire, parsed)
    if not all(_object_stated_by_both(raw, typed) for _, raw, typed in readings):
        raise wire_invalid(
            "the exact JSON the provider sent and this pinned SDK's typed reading "
            "of the same bytes do not state the same objects at the sites "
            f"{XAI_COMPAT_RESPONSE_EXTENSIONS} fixes names on, so there is no "
            "single body here for this build to check that contract against. An "
            "object only one reading states is a disagreement before any value "
            "is: the reading that has it could have stated extensions under it "
            "that the other reading had nowhere to state. Neither reading is "
            "preferred over the other and neither is edited to agree with the "
            "other: the response is refused whole. Which objects each reading "
            "stated is not recorded here: it is provider-controlled, and a "
            "durable failure row records only this build's own fixed detail and "
            "closed-set values"
        )
    stated = [
        (
            site,
            _stated_extensions(raw, site=site),
            _stated_extensions(_typed_extras(typed), site=site),
        )
        for site, raw, typed in readings
    ]
    for site, _, typed in stated:
        check_response_extensions(typed, site=site)
    if any(raw != typed for _, raw, typed in stated):
        raise wire_invalid(
            "the exact JSON the provider sent and this pinned SDK's typed reading "
            "of the same bytes do not state the same response extensions, so there "
            "is no single body here for this build to check against "
            f"{XAI_COMPAT_RESPONSE_EXTENSIONS}. Neither reading is preferred over "
            "the other and neither is edited to agree with the other: the response "
            "is refused whole. What each reading stated is not recorded here: it is "
            "provider-controlled, and a durable failure row records only this "
            "build's own fixed detail and closed-set values"
        )


# -- the wire contract --------------------------------------------------------
#
# One shape per documented object this build reads, with its allowed field set
# taken from the pinned SDK's own model for that object rather than hand-listed.
# The nesting is what matters here: this surface carries the action four levels
# down — response, choice, message, tool call, function — and a check that
# looked only at the top level would pass a body whose action was carried by a
# tool call nobody can describe.

RESPONSE_WIRE_SHAPE = WireShape(
    kind="completion",
    allowed=declared_wire_fields(ChatCompletion),
    required=("model", "choices", "usage"),
)
CHOICE_WIRE_SHAPE = WireShape(
    kind="completion choice",
    allowed=declared_wire_fields(Choice),
    required=CHOICE_FIELDS,
)
#: The assistant message. Its allowed set is the SDK's declared fields *plus* the
#: one name this lane's extension contract fixes for a message, and the two
#: halves are kept visibly separate because they are two different kinds of
#: claim: the first is the vendor's own schema as the pinned SDK models it, and
#: the second is what this repository observed the compatibility layer return and
#: then fixed a shape for. Membership here only buys a name the right to appear
#: on this object; the shape of whatever it carries is proven by
#: :func:`check_response_extensions`.
MESSAGE_WIRE_SHAPE = WireShape(
    kind="completion message",
    allowed=declared_wire_fields(ChatCompletionMessage) | MESSAGE_EXTENSION_NAMES,
    required=(),
)
#: One tool call. ``function`` is required because this build reads it, and
#: requiring it is what turns "``function`` is a list" from an ``AttributeError``
#: several frames down into a named refusal before anything is read.
TOOL_CALL_WIRE_SHAPE = WireShape(
    kind="tool call",
    allowed=declared_wire_fields(ChatCompletionMessageFunctionToolCall),
    required=("type", "function"),
)
FUNCTION_WIRE_SHAPE = WireShape(
    kind="tool call function object",
    allowed=declared_wire_fields(Function),
    required=("name", "arguments"),
)
#: The usage block, and the detail object inside it this lane's extension
#: contract also names. The same two-halves rule as the message above: the
#: declared fields are the vendor's schema, and the extension names are this
#: repository's own observation. ``prompt_tokens_details`` is an object the SDK
#: *does* declare — what it does not declare is the pair inside it.
USAGE_WIRE_SHAPE = WireShape(
    kind="usage block",
    allowed=declared_wire_fields(CompletionUsage) | USAGE_EXTENSION_NAMES,
    required=USAGE_FIELDS,
)
PROMPT_TOKENS_DETAILS_WIRE_SHAPE = WireShape(
    kind="prompt token details",
    allowed=(
        declared_wire_fields(PromptTokensDetails) | PROMPT_TOKENS_DETAILS_EXTENSION_NAMES
    ),
    required=(),
)
COMPLETION_TOKENS_DETAILS_WIRE_SHAPE = WireShape(
    kind="completion token details",
    allowed=declared_wire_fields(CompletionTokensDetails),
    required=(),
)


class XAIConfigurationError(ProviderConfigurationError):
    """The environment cannot produce a usable, unambiguous xAI client."""


#: The one endpoint this build talks to: xAI's documented OpenAI-compatible base.
XAI_BASE_URL = "https://api.x.ai/v1"

#: The credential, and the only place it may come from.
API_KEY_VARIABLE = "XAI_API_KEY"

#: Environment variables that would silently redirect this request or change
#: which account it bills.
#:
#: ``OPENAI_BASE_URL`` is in the list and belongs there. It is the *OpenAI SDK*
#: doing the talking, so OpenAI's own redirect variable would move an xAI run
#: somewhere else just as effectively as an xAI-named one would — and a reader
#: scanning for xAI-prefixed variables would never think to look for it.
REJECTED_VARIABLES: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "XAI_BASE_URL",
)


def build_client(*, api_key: str) -> openai.OpenAI:
    """The SDK client this build talks through, pinned to xAI and unretried.

    Both arguments are load-bearing. ``base_url`` is what makes this an xAI
    client rather than an OpenAI one — the SDK's default is OpenAI's endpoint,
    so omitting it would send an xAI credential to OpenAI. ``max_retries=0``
    keeps the retry policy in this build, where it is bounded, deterministic and
    inside run identity.
    """
    return openai.OpenAI(
        api_key=api_key, base_url=XAI_BASE_URL, max_retries=SDK_MAX_RETRIES
    )


def _check_client_state(client: openai.OpenAI) -> None:
    """Prove this client's effective wire state matches recorded settings.

    Load-bearing twice over for this integration. The SDK doing the talking is
    OpenAI's, and its *default* base URL is OpenAI's endpoint — so a client that
    simply forgot to pin one does not fail, it sends an xAI credential to a
    different vendor. And a client pinned to somewhere else entirely leaves
    settings recording ``https://api.x.ai/v1`` for a request that never went
    there.

    Endpoint and account scope are checked beside every SDK and httpx channel
    that can add unrecorded headers, query, authentication, cookies, or retries.
    The shared provider-kernel helper derives pristine httpx headers from the
    installed version and never includes a caller-controlled value in an error.
    """
    check_xai_openai_compat_client_state(
        client,
        expected_base_url=XAI_BASE_URL,
        provider=XAI_PROVIDER,
        max_retries=SDK_MAX_RETRIES,
        error=XAIConfigurationError,
    )


def build_xai_adapter(
    *,
    model: str,
    configuration_id: str | None = None,
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    retry: ProviderRetryPolicy | None = None,
    cost_guard: RunCostGuard | None = None,
) -> XAIOpenAICompatAdapter:
    """Everything a credentialed run needs, or a clean refusal.

    The credential comes from ``XAI_API_KEY`` and from nowhere else. That is
    worth stating twice for this integration specifically: the OpenAI SDK will
    read ``OPENAI_API_KEY`` out of the environment if it is not given a key, so
    a build that let the SDK resolve its own credential could authenticate an
    xAI run with an OpenAI key — or send one to xAI's endpoint.
    """
    checked = check_model(model, provider=XAI_PROVIDER, error=XAIConfigurationError)
    # Before the client exists, and so before a credential has been read.
    check_retry_policy(
        retry or ProviderRetryPolicy(),
        provider=XAI_PROVIDER,
        error=XAIConfigurationError,
    )
    check_configuration_audited(
        configuration_id, error=XAIConfigurationError, audited=audited, environ=environ
    )
    check_environment(
        REJECTED_VARIABLES,
        provider=XAI_PROVIDER,
        base_url=XAI_BASE_URL,
        error=XAIConfigurationError,
        environ=environ,
    )
    api_key = resolve_api_key(
        API_KEY_VARIABLE,
        provider=XAI_PROVIDER,
        error=XAIConfigurationError,
        environ=environ,
    )
    return XAIOpenAICompatAdapter(
        model=checked,
        client=build_client(api_key=api_key),
        retry=retry,
        cost_guard=cost_guard,
    )


def xai_identity(model: str) -> AdapterIdentity:
    """Who answered: the service, the pinned model, and this integration."""
    return AdapterIdentity(
        provider=XAI_PROVIDER,
        model=model,
        implementation=XAI_IMPLEMENTATION,
        version=XAI_ADAPTER_VERSION,
    )


def profile_settings(
    profile: RequestProfile, retry: ProviderRetryPolicy
) -> dict[str, Any]:
    """Every result-affecting request setting, and no credential."""
    return {
        "action_surface": ACTION_SURFACE_CONTRACT,
        "api": XAI_API,
        "api_compatibility": XAI_API_COMPATIBILITY,
        "base_url": XAI_BASE_URL,
        "fact_affordance": FACT_AFFORDANCE_CONTRACT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "query_resolution": QUERY_RESOLUTION_CONTRACT,
        "request_mapping": REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_capture": XAI_RESPONSE_CAPTURE,
        "response_contract": "exactly_one_tool_call",
        # Named *and* hashed. The name says which contract this run accepted this
        # compatibility layer's response extensions under; the digest is what
        # makes that claim re-derivable, because a provider-observed contract can
        # be widened without its name changing and a reader would have no way to
        # tell. Both are inside ``configuration_id``.
        "response_extensions": XAI_COMPAT_RESPONSE_EXTENSIONS,
        "response_extensions_digest": RESPONSE_EXTENSION_DIGEST,
        "retry": retry.as_dict(),
        "sdk": "openai",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_version": openai.__version__,
        "tool_choice": TOOL_CHOICE,
    }


def xai_settings(
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
    return request_input_token_bound(payload, "xai request bound")


def build_chat_request(
    request: TurnRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    """The exact request body one turn sends, under this model's profile.

    The system prompt travels as a ``system`` message because this API has no
    separate instructions parameter. That is a real difference from the
    Responses mapping rather than a cosmetic one — it puts the prompt in the
    same list as the case — and it is why this mapping has its own version
    string.
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
                    observed_case(request), "xai turn request"
                ),
            },
        ],
    }
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    if shape.reasoning_effort is not None:
        payload["reasoning_effort"] = shape.reasoning_effort
    return payload


# -- the response ------------------------------------------------------------


def check_response_model(response: ChatCompletion, *, model: str) -> None:
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

    Walked to the depth the action is carried at, because that is where this
    surface's failures live. The blocker this closes is one of them: a
    ``function`` that arrives as a list, a string, a number or a boolean
    survives the SDK's union resolution, passes every check on the parsed
    object, and reaches ``.name`` several frames later as an ``AttributeError``
    — after the attempt has been published as a successful response with usage.

    A tool call whose declared type is not ``function`` is deliberately not
    checked against the function shape. That is a statement about the answer
    rather than the transport, and :func:`parse_completion` reports it as the
    protocol failure it is.

    Nothing that arrived is quoted; see :mod:`boundarybench.providers.wire`.
    """
    body = checked_wire_object(raw, RESPONSE_WIRE_SHAPE)
    for entry in checked_wire_list(body["choices"], kind="completion choices"):
        choice = checked_wire_object(entry, CHOICE_WIRE_SHAPE)
        message = choice["message"]
        if message is None:
            # This API's own "no message", reported by :func:`parse_completion`
            # as the answer that states no action. Anything else in that field
            # is a container rather than an absence.
            continue
        parsed_message = checked_wire_object(message, MESSAGE_WIRE_SHAPE)
        check_response_extensions(parsed_message, site=MESSAGE_EXTENSION_SITE)
        calls = parsed_message.get("tool_calls")
        if calls is None:
            continue
        for raw_call in checked_wire_list(calls, kind="tool calls"):
            call = checked_wire_mapping(raw_call, kind="tool call")
            if call.get("type") != "function":
                continue
            function = checked_wire_object(call, TOOL_CALL_WIRE_SHAPE)["function"]
            named = checked_wire_object(function, FUNCTION_WIRE_SHAPE)
            checked_wire_string(named["name"], kind="tool call name")
            checked_wire_string(named["arguments"], kind="tool call arguments")
    usage = checked_wire_object(body["usage"], USAGE_WIRE_SHAPE)
    check_response_extensions(usage, site=USAGE_EXTENSION_SITE)
    for field, shape, site in (
        (
            "prompt_tokens_details",
            PROMPT_TOKENS_DETAILS_WIRE_SHAPE,
            PROMPT_TOKENS_DETAILS_EXTENSION_SITE,
        ),
        ("completion_tokens_details", COMPLETION_TOKENS_DETAILS_WIRE_SHAPE, None),
    ):
        if usage.get(field) is not None:
            details = checked_wire_object(usage[field], shape)
            if site is not None:
                check_response_extensions(details, site=site)
    wire_token_usage(usage, fields=USAGE_FIELDS)


def check_response_admissible(
    response: WireResponse[ChatCompletion], *, model: str
) -> None:
    """Everything that must be true of a body before anything is read out of it.

    The same four questions the OpenAI integration asks, in the same order and
    for the same reasons — the pinned model produced it, it carries no field the
    SDK's models do not declare, its ``choices`` really are a list, and its
    token counts are countable — because none of the four is a property of a
    vendor. What differs is only which field holds the answers and what the
    counts are called, which is exactly the part that stays in each integration.

    All four are :data:`~boundarybench.adapter.PROVIDER_FAULT_RESPONSE_INVALID`
    and all four are decided before the action or the usage is accepted. A
    ``null`` in ``choices`` matters here specifically: this SDK parses it into
    ``None``, and the first thing that would notice is ``len()`` in
    :func:`parse_completion` — as a ``TypeError`` carrying the interpreter's
    words, raised after the turn was already measured and settled.

    The same leniency runs one level deeper, and so does the check. A ``[]``, a
    string or a number where this API documents a choice, a message or a usage
    block is carried through as that value, and every one of them would surface
    as an ``AttributeError`` from inside the parse — outside this contract's
    fault set, and, for the usage block, before the attempt that received the
    body had been published at all. Each container is therefore proven by the
    fields its reader is about to use; see
    :func:`~boundarybench.providers.common.check_response_object`.

    All of it now happens *after* the same body has been checked as the exact
    JSON the provider sent — see :func:`check_wire_completion`. That check is
    first because it is the only one that can see what the parse destroyed: a
    coerced token count, a dropped field, a union arm the SDK resolved to
    something this build would then read.
    """
    check_wire_completion(response.wire)
    parsed = response.parsed
    check_response_model(parsed, model=model)
    check_response_contract(parsed, allowed_extras_by_model=TYPED_EXTENSION_EXTRAS)
    check_response_extensions_agree(response.wire, parsed)
    check_response_collection(parsed.choices, kind="completion choices")
    for choice in parsed.choices:
        check_response_object(choice, fields=CHOICE_FIELDS, kind="completion choice")
        # ``None`` is this API's own "no message", and :func:`parse_completion`
        # reports it as the answer that states no action. Anything else in that
        # field is a container rather than an absence.
        if choice.message is not None:
            check_response_object(
                choice.message, fields=MESSAGE_FIELDS, kind="completion message"
            )
    usage = parsed.usage
    if usage is not None:
        check_response_object(usage, fields=USAGE_FIELDS, kind="usage block")
        checked_token_usage(usage.prompt_tokens, usage.completion_tokens)


def parse_completion(response: ChatCompletion, request: TurnRequest) -> AdapterCall:
    """The provider's answer, as exactly one action call, or a refusal."""
    if len(response.choices) != 1:
        raise AdapterProtocolError(
            f"the answer carries {len(response.choices)} choice(s); this build asks "
            "for one completion per turn and cannot tell which of several is the "
            "action the agent took"
        )
    choice = response.choices[0]
    # Truncation is decided first, and before the call count, because it
    # *explains* the call count: an answer cut off at the ceiling may carry no
    # call, or half of one, and reporting the count would hide the cause.
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
        # attribute error further down, where it would be recorded as this
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
        # Before the union is read: an entry in that list that is not a tool
        # call object at all states nothing this build can classify, and
        # reaching for ``type`` on it would be an attribute error dressed as an
        # adapter bug rather than the malformed body it is.
        check_response_object(call, fields=TOOL_CALL_FIELDS, kind="tool call")
        # This SDK's tool-call union has two arms, and only one of them is a
        # function call. A "custom" tool call carries free-form text instead of
        # named arguments, so it is not something this scaffold can read as an
        # action — and reading past it, or trusting a ``function`` attribute
        # that is not there, would turn a protocol failure into a crash.
        if call.type != "function":
            raise AdapterProtocolError(
                "the answer carries a tool call that is not a function call; this "
                "scaffold requires one named action with structured arguments, and "
                "a free-form call is not one. The type the call declared is not "
                "quoted here: it is provider-controlled text, and a durable failure "
                "row records only this build's own fixed detail and closed-set values"
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
        [
            (call.function.name, call.function.arguments)
            for call in calls
            if call.type == "function"
        ],
        request,
        label="xai tool call arguments",
    )
    # Checked *after* the call has been resolved, so a well-formed single call
    # that arrived under an unexpected finish reason is still reported as the
    # stop-reason failure it is rather than as whatever the count happened to be.
    if choice.finish_reason != REQUIRED_FINISH_REASON:
        raise AdapterProtocolError(
            f"the answer stopped for something other than {REQUIRED_FINISH_REASON!r}, "
            "so the call it carries was cut short rather than completed and is not "
            "the action the model was making. The reason the provider stated is not "
            "quoted here: it is provider-controlled text, and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    return resolved


def response_usage(response: WireResponse[ChatCompletion]) -> TokenUsage:
    """The two counts this build prices a turn on, taken from the wire.

    This surface names them ``prompt_tokens`` and ``completion_tokens`` — not
    the Responses API's ``input_tokens``/``output_tokens``. Reading the wrong
    pair would price every turn at null and silently forfeit every reservation,
    which is a failure that looks like a quiet run rather than like a bug.

    Read off the exact JSON rather than off the parsed model, because this SDK
    coerces: ``true`` becomes one token and ``"12"`` becomes twelve, and both
    then look exactly like a measurement. Validated again here rather than
    trusted from :func:`check_response_admissible`, because this is the function
    whose return value becomes an
    :class:`~boundarybench.adapter.AdapterUsage` and a settled cost.
    """
    usage = checked_wire_object(response.wire.get("usage"), USAGE_WIRE_SHAPE)
    return wire_token_usage(usage, fields=USAGE_FIELDS)


def classify_exception(exception: Exception) -> Fault | None:
    """Map one SDK error onto the contract's fault set, or decline."""
    if isinstance(exception, openai.APITimeoutError):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, openai.APIConnectionError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, openai.APIResponseValidationError):
        # A whole body arrived and the SDK refused to read it, so the attempt
        # records that a response was received. See the OpenAI integration.
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, openai.APIStatusError):
        return classify_http_status(exception.status_code)
    return None


# -- the adapter -------------------------------------------------------------


class XAIOpenAICompatAdapter:
    """One provider client, one pinned model, one action per turn."""

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
        self._profile = check_request_profile(profile, model=model)
        # Before anything else this client is trusted for: where it sends. The
        # SDK's own default endpoint is another vendor's, so this is not only a
        # redirection check — it is what makes an injected client an xAI client.
        _check_client_state(client)
        self._model = model
        self._client = client
        self._retry = check_retry_policy(
            retry or ProviderRetryPolicy(),
            provider=XAI_PROVIDER,
            error=XAIConfigurationError,
        )
        self._executor: TurnExecutor[WireResponse[ChatCompletion]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        self._clock = clock
        # One turn at a time on this instance: the executor's usage and attempt
        # evidence are per-adapter state that a turn owns while it runs. See
        # :class:`~boundarybench.adapter.SingleFlight`.
        self._single_flight = SingleFlight(adapter="XAIOpenAICompatAdapter")

    def __repr__(self) -> str:
        """Names the run, never the credential."""
        return (
            f"<XAIOpenAICompatAdapter provider={XAI_PROVIDER} "
            f"model={self._model} implementation={XAI_IMPLEMENTATION}>"
        )

    @property
    def identity(self) -> AdapterIdentity:
        return xai_identity(self._model)

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

        Under this instance's single-flight guard, taken *before* the clearing:
        a second overlapping turn would reset the measurement of the one in
        flight, so it is refused immediately rather than allowed to interleave.
        """
        with self._single_flight:
            self._executor.clear()
            payload = build_chat_request(
                request, model=self._model, profile=self._profile
            )
            response = self._send_payload(payload, deadline)
            return parse_completion(response.parsed, request)

    def _send_payload(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[ChatCompletion]:
        """Check the body against the settings, then run the turn.

        The endpoint is proven again here and not only in the constructor. This
        SDK's ``base_url`` is a settable attribute, so a client that was pinned
        when the adapter was built can be pointed somewhere else before the turn
        that uses it — and the check that matters is the one immediately before
        the credential is sent.
        """
        admission_started = self._clock()
        try:
            _check_client_state(self._client)
        except XAIConfigurationError:
            self._executor.record_pre_dispatch_refusal(started_at=admission_started)
            raise
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
            check_before_dispatch=lambda: _check_client_state(self._client),
            classify=classify_exception,
            check_identity=lambda response: check_response_admissible(
                response, model=self._model
            ),
            usage_of=response_usage,
        )

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[ChatCompletion]:
        """One request, holding the bytes that came back beside the SDK's reading.

        The SDK's own ``with_raw_response`` surface, for the reasons the OpenAI
        integration states — and with one extra weight here. This lane already
        talks to xAI through a compatibility layer rather than the vendor's own
        client, so "a field the layer drops, a usage counter it rounds, a
        tool-call shape it normalises" is a stated limitation of this adapter.
        Reading the exact JSON is the only thing that makes any of that visible
        from here.
        """
        raw = self._client.chat.completions.with_raw_response.create(
            **payload, timeout=timeout
        )
        return WireResponse(raw.text, raw.parse, kind="xai completion")


__all__ = [
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "CHOICE_FIELDS",
    "CHOICE_WIRE_SHAPE",
    "EXTENSION_COUNT",
    "EXTENSION_REASONING_TEXT",
    "FUNCTION_WIRE_SHAPE",
    "GROK_4_5_MODEL",
    "GROK_4_5_PROFILE",
    "MAX_EXTENSION_COUNT",
    "MAX_OUTPUT_TOKENS",
    "MAX_REASONING_CONTENT_CHARACTERS",
    "MESSAGE_EXTENSION_NAMES",
    "MESSAGE_EXTENSION_SITE",
    "MESSAGE_FIELDS",
    "MESSAGE_WIRE_SHAPE",
    "MODEL_REQUEST_PROFILES",
    "OUTPUT_LIMIT_FINISH_REASON",
    "PARALLEL_TOOL_CALLS",
    "PROHIBITED_REQUEST_FIELDS",
    "PROMPT_TOKENS_DETAILS_EXTENSION_NAMES",
    "PROMPT_TOKENS_DETAILS_EXTENSION_SITE",
    "PROMPT_TOKENS_DETAILS_WIRE_SHAPE",
    "REJECTED_VARIABLES",
    "REQUEST_MAPPING_VERSION",
    "REQUIRED_FINISH_REASON",
    "RESPONSE_EXTENSION_CONTRACT",
    "RESPONSE_EXTENSION_DIGEST",
    "RESPONSE_EXTENSION_SCHEMA",
    "RESPONSE_EXTENSION_SITES_REQUIRED_WHOLE",
    "RESPONSE_WIRE_SHAPE",
    "TEMPERATURE",
    "TOOL_CALL_FIELDS",
    "TOOL_CALL_WIRE_SHAPE",
    "TOOL_CHOICE",
    "TYPED_EXTENSION_EXTRAS",
    "USAGE_EXTENSION_NAMES",
    "USAGE_EXTENSION_SITE",
    "USAGE_FIELDS",
    "USAGE_WIRE_SHAPE",
    "XAI_ADAPTER_VERSION",
    "XAI_API",
    "XAI_API_COMPATIBILITY",
    "XAI_BASE_URL",
    "XAI_COMPAT_RESPONSE_EXTENSIONS",
    "XAI_IMPLEMENTATION",
    "XAI_PROVIDER",
    "XAI_RESPONSE_CAPTURE",
    "RequestProfile",
    "XAIConfigurationError",
    "XAIOpenAICompatAdapter",
    "build_chat_request",
    "build_client",
    "build_xai_adapter",
    "check_request_profile",
    "check_response_admissible",
    "check_response_extensions",
    "check_response_extensions_agree",
    "check_response_model",
    "check_wire_completion",
    "classify_exception",
    "parse_completion",
    "profile_settings",
    "request_profile_for",
    "request_token_bound",
    "response_usage",
    "xai_identity",
    "xai_settings",
]
