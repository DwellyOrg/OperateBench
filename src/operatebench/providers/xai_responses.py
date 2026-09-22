"""The xAI Responses exchange: a payload in, a checked wire answer out.

This is everything about talking to the xAI Responses API that is *not* a
question about what was being asked. Which endpoint the credential goes to, which
request shape a pinned model is entitled to, that the body matches the settings a
run recorded, one turn at a time per client, the bounded retry loop with its cost
guard and attempt evidence, the raw bytes held beside the SDK's typed reading of
them, and whether what came back is the response contract this build asked for.

What it deliberately does not know is the *question*. A caller hands
:meth:`XAIResponsesExchange.exchange` a finished request body and a deadline
and gets back a :class:`~operatebench.providers.wire.WireResponse`. It has never
heard of a ``TurnRequest``, a scaffold, an observed case, an ``AdapterCall``, a
``ModelRequest`` or an outcome — those live in whichever track built the payload,
and two tracks can therefore compose this without either one inheriting the
other's vocabulary.

The Boundary Track's :mod:`boundarybench.providers.openai_responses` is the first
such composition: it owns ``TurnRequest -> body`` and ``response ->
AdapterCall``, and re-exports the names below so nothing that imported them
before has to move.

Two settings deserve naming here because they are easy to miss and hard to undo.
``store`` is pinned to ``False``: the Responses API retains request and response
content server-side by default, and a benchmark that silently left its prompts
and answers in a vendor's dashboard would be making a durability decision on an
operator's behalf. And a real response from this service carries a block of
top-level fields the pinned SDK's own model does not declare, so this module
names an exact contract of its own for that block —
:data:`XAI_RESPONSE_SERVER_EXTENSIONS` — and accepts precisely it: closed
recursively, typed scalar by scalar, complete wherever it is present, and read
for nothing.

Nothing here pre-authorises a configuration, reads a credential into a durable
place, or opens a socket on import.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

from operatebench.jsonsafe import canonical_json_bytes
from operatebench.providers.config import (
    ProviderConfigurationError,
    check_endpoint,
    check_payload_fields,
)
from operatebench.providers.cost import CostGuard, request_input_token_bound
from operatebench.providers.executor import (
    SDK_MAX_RETRIES,
    Clock,
    ProviderRetryPolicy,
    Sleep,
    TurnExecutor,
    check_retry_policy,
)
from operatebench.providers.faults import (
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_TIMEOUT,
    AdapterProviderError,
    Fault,
    SingleFlight,
    classify_http_status,
)
from operatebench.providers.response import (
    check_response_collection,
    check_response_contract,
    check_response_object,
)
from operatebench.providers.telemetry import (
    AdapterUsage,
    ProviderTelemetry,
    TurnDeadline,
)
from operatebench.providers.usage import (
    MAX_EXACT_TOKEN_COUNT,
    TokenUsage,
    checked_token_usage,
)
from operatebench.providers.wire import (
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

#: The service under test, as recorded in run identity.
XAI_PROVIDER = "xai"

#: How this exchange gets at the bytes the provider sent.
#:
#: The SDK's own ``with_raw_response`` surface, rather than a hand-rolled
#: transport wrapper: it is the vendor's supported way to hold the HTTP response
#: and the typed model at once, it keeps the request serialisation the SDK's,
#: and it is exercised in these tests over a real client on an in-process
#: transport. Named in settings because it is result-affecting — it adds a
#: header to every request, and it decides what this build is able to refuse.
XAI_RESPONSE_CAPTURE = "openai_sdk_with_raw_response_v1"

#: The API surface this exchange speaks. Pinned, and named in settings: the same
#: model reached through Chat Completions is a different request mapping and a
#: different response contract, so it would be a different experiment.
XAI_RESPONSES_API = "responses"

#: The output ceiling, pinned rather than exposed. A run whose answers were
#: truncated at a different limit is not comparable with one that was not.
#:
#: The same number the Anthropic integration pins, on purpose: two models in one
#: matrix that were cut off at different lengths are not answering the same
#: question.
MAX_OUTPUT_TOKENS = 1024

#: Pinned, and pinned to zero, wherever a profile sends it at all: this
#: benchmark measures whether a decision is made for the right reason, and
#: sampling temperature is a confound the standardized scaffold has no reason to
#: vary. Which profiles send it is :class:`RequestProfile`'s decision, not this
#: constant's — the pinned model's does not.
TEMPERATURE = 0.0

#: The response contract, enforced on both sides. ``required`` makes the model
#: answer with a tool call rather than prose.
TOOL_CHOICE = "required"

#: One turn is one action, so parallel tool calls are disabled.
PARALLEL_TOOL_CALLS = False

#: Whether the provider may retain this request and its answer. Pinned off.
STORE = False


class ProviderDispatchObserver(Protocol):
    """Observe one SDK request attempt immediately before it begins."""

    def __call__(self) -> None: ...


#: The fields whose presence or absence is a *profile's* decision, and which
#: therefore have to be accounted for by every profile: each one is either in
#: the body this build sends or named as omitted from it.
#:
#: ``top_p`` is in the set although no profile this build ships sends it. It is
#: the other knob this API accepts that would change an answer, so "this build
#: does not send it" is a claim worth recording in run identity rather than a
#: silence a reader has to infer.
PROHIBITED_REQUEST_FIELDS: tuple[str, ...] = ("reasoning", "temperature", "top_p")

#: The fields the mapping sends for every model, whatever its profile.
_COMMON_REQUEST_FIELDS: tuple[str, ...] = (
    "input",
    "instructions",
    "max_output_tokens",
    "model",
    "parallel_tool_calls",
    "store",
    "tool_choice",
    "tools",
)


@dataclass(frozen=True)
class RequestProfile:
    """One model's frozen request shape: what is sent, and what is left out.

    A profile exists because the request that is correct for one model is
    refused by another: a reasoning model may reject a sampling parameter
    outright, and a model whose reasoning is adaptive by default reads an absent
    ``reasoning`` field as "on". Each pinned model therefore receives its
    shipped profile rather than a shared shape.

    Immutable, and identified by :attr:`profile_id`, which goes into adapter
    settings and so into ``configuration_id``. Two runs that asked for different
    things are two configurations, and a resume across them is refused.

    ``temperature`` is ``None`` for "not sent", not for "sent as null" — the
    field is absent from the body entirely. ``reasoning`` is the same.
    """

    profile_id: str
    temperature: float | None
    reasoning: Mapping[str, Any] | None
    #: What this build could not establish about the model's own documented
    #: request contract, or ``None`` when there was nothing to state. Recorded
    #: in settings rather than in a comment: an operator pinning a model whose
    #: contract this build did not verify is the one who has to check it, and
    #: they can only do that if the artefact says so.
    contract_verified: bool = True

    def sent_fields(self) -> tuple[str, ...]:
        """Exactly the top-level keys of the body this profile produces."""
        fields = list(_COMMON_REQUEST_FIELDS)
        if self.temperature is not None:
            fields.append("temperature")
        if self.reasoning is not None:
            fields.append("reasoning")
        return tuple(sorted(fields))

    def omitted_fields(self) -> tuple[str, ...]:
        """The profile-controlled fields this profile deliberately leaves out."""
        sent = set(self.sent_fields())
        return tuple(
            sorted(field for field in PROHIBITED_REQUEST_FIELDS if field not in sent)
        )

    def as_settings(self) -> dict[str, Any]:
        """The part of adapter settings this profile is answerable for."""
        return {
            "request_profile": self.profile_id,
            "request_fields_sent": list(self.sent_fields()),
            "request_fields_omitted": list(self.omitted_fields()),
            "temperature": self.temperature,
            "reasoning": None if self.reasoning is None else dict(self.reasoning),
            "model_contract_verified": self.contract_verified,
        }


#: The exact currently available xAI slug this private lane pins. The slug is
#: moving, not immutable; this is not matched or public reproducibility evidence.
GROK_4_5_MODEL = "grok-4.5"

#: The published model page this build read :data:`GROK_4_5_PROFILE`'s
#: request contract from.
#:
#: Named as a constant beside the profile that relies on it, because
#: ``model_contract_verified: true`` is a claim in run identity and a claim is
#: worth what a reader's ability to check it is worth. A boolean on its own says
#: only that somebody said they looked.
GROK_4_5_CONTRACT_SOURCE = "https://docs.x.ai/docs/models/grok-4-5"

#: The reasoning efforts that page publishes for this model, in its own order.
#:
#: Transcribed from the model page rather than imported from ``openai.types``.
#: The SDK's ``ReasoningEffort`` alias is the union over every model that SDK
#: serves and carries at least one value this model's page does not publish, so
#: a profile that validated itself against the alias would be claiming a
#: contract it never read.
REASONING_EFFORT_VALUES: tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

#: The effort this build asks for, and the effort it would get by saying nothing.
#:
#: The second one is why the first is sent explicitly. An absent ``reasoning``
#: field selects the documented default, so a standardized no-reasoning track
#: that omitted the field would run under ``medium`` while its manifest recorded
#: no reasoning setting at all — the request and the artefact describing it would
#: be about two different experiments.
REASONING_EFFORT_LOW = "low"
REASONING_EFFORT_DEFAULT = "medium"

#: The exact reasoning object the no-reasoning track sends. Immutable: a profile
#: is frozen, and a mapping shared by reference is not.
LOW_REASONING: Mapping[str, Any] = MappingProxyType({"effort": REASONING_EFFORT_LOW})

#: What a model this build ships no model-specific profile for is sent.
#:
#: :func:`~operatebench.providers.config.check_model` is a shape check rather
#: than an allowlist — a build that refused every model released after it is not
#: what a benchmark needs — so an unencoded model still has to be sent
#: something, and this is it.
BASELINE_PROFILE = RequestProfile(
    profile_id="xai_responses_baseline_sampling_omitted_v1",
    temperature=None,
    reasoning=None,
    contract_verified=False,
)

#: ``grok-4.5``: reasoning explicitly off, no sampling parameter at all.
#:
#: Both halves are read off the model page named in
#: :data:`GROK_4_5_CONTRACT_SOURCE`, and they are read in opposite
#: directions, which is why ``contract_verified`` is ``True`` for this profile
#: without being a claim about everything the model does:
#:
#: * The page states that this model supports the Responses API and function
#:   calling, and that ``reasoning.effort`` accepts
#:   :data:`REASONING_EFFORT_VALUES` with ``medium`` as the default. So the
#:   standardized no-reasoning track can *state* what it asks for rather than
#:   infer it, and it sends :data:`LOW_REASONING` explicitly. Omitting the field
#:   would select the default instead, silently.
#: * The same page does not establish that this model accepts a sampling
#:   parameter. An undocumented knob sent anyway is a request shape nobody can
#:   check, so ``temperature`` is ``None`` — absent from the body entirely
#:   rather than sent as null.
#:
#: ``contract_verified`` therefore means what it says on this profile: every
#: field this profile *claims* — the reasoning object it sends and the sampling
#: parameter it omits — follows the model's own published request contract.
GROK_4_5_PROFILE = RequestProfile(
    profile_id="grok45_reasoning_low_sampling_omitted_v1",
    temperature=None,
    reasoning=LOW_REASONING,
    contract_verified=False,
)

# Grok 4.6 is a moving model identifier, not a dated snapshot. Its high-effort
# profile is not comparable to the historical Grok 4.5 low-effort profile.
GROK_4_6_MODEL = "grok-4.6"
GROK_4_6_CONTRACT_SOURCE = "https://docs.x.ai/developers/models/grok-4.6"
GROK_4_6_REASONING_EFFORT_VALUES = ("low", "medium", "high", "xhigh")
GROK_4_6_REASONING_EFFORT_DEFAULT = "high"
GROK_4_6_PROFILE = RequestProfile(
    profile_id="grok46_reasoning_high_sampling_omitted_v1",
    temperature=None,
    reasoning=MappingProxyType({"effort": "high"}),
    contract_verified=True,
)

#: Model identifier to frozen request shape. Matched whole, like a price: an
#: alias is not the model it currently points at.
MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {GROK_4_5_MODEL: GROK_4_5_PROFILE, GROK_4_6_MODEL: GROK_4_6_PROFILE}
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
    raise XAIResponsesConfigurationError(
        f"request profile {profile.profile_id!r} is not the profile this build "
        f"sends to model {model!r}, which is {expected.profile_id!r}. A model's "
        "request shape is frozen per model — sampling and reasoning fields are "
        "accepted by one model and refused by another — so a profile cannot be "
        "carried across models"
    )


#: The only response status a turn may end on.
COMPLETED_STATUS = "completed"
#: A response that stopped short. Which *reason* it names decides how it is
#: classified — that classification is a track's own parser's business.
INCOMPLETE_STATUS = "incomplete"
#: The one incomplete reason this build classifies rather than redacts, because
#: it names a cause *this run controls*: the answer ran into
#: :data:`MAX_OUTPUT_TOKENS`, which this build pins.
OUTPUT_LIMIT_REASON = "max_output_tokens"

#: The two counts this API's usage block is read for, and therefore the two
#: fields a body has to state before this build calls what it sent a usage
#: block at all. Named here rather than spelled at each reader: the response
#: surface decides them — the Chat Completions surface calls the same two
#: numbers something else — and a reader that proved one pair and then read the
#: other would prove nothing.
USAGE_FIELDS: tuple[str, ...] = ("input_tokens", "output_tokens")

#: The only two output item types this scaffold reads. Checked by declared value
#: rather than by Python class: an SDK resolves a union leniently, so an
#: ``isinstance`` answers "the SDK fell back", not "the provider sent this".
FUNCTION_CALL_ITEM = "function_call"
MESSAGE_ITEM = "message"
REASONING_ITEM = "reasoning"
ACCEPTED_ITEM_TYPES: frozenset[str] = frozenset(
    {FUNCTION_CALL_ITEM, MESSAGE_ITEM, REASONING_ITEM}
)

# -- the xAI Responses server-extension contract ------------------------------

XAI_RESPONSE_SERVER_EXTENSIONS_V1 = "xai_responses_server_extensions_v1"
XAI_RESPONSE_SERVER_EXTENSIONS = "xai_responses_server_extensions_v2"
EXTENSION_COUNT = "exact_count"
EXTENSION_ZERO = "exact_zero"
EXTENSION_PENALTY = "finite_penalty_number"
EXTENSION_FLAG = "boolean"
EXTENSION_UNION = "one_of"
PENALTY_MINIMUM = -2.0
PENALTY_MAXIMUM = 2.0
MAX_EXTENSION_COUNT = MAX_EXACT_TOKEN_COUNT
ROOT_EXTENSION_SITE = "response_root"
USAGE_EXTENSION_SITE = "usage_block"


def _frozen(node: Any) -> Any:
    if isinstance(node, Mapping):
        return MappingProxyType({key: _frozen(value) for key, value in node.items()})
    if isinstance(node, (list, tuple)):
        return tuple(_frozen(value) for value in node)
    return node


def _plain(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {key: _plain(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_plain(value) for value in node]
    return node


RESPONSE_EXTENSION_SCHEMA_V1: Mapping[str, Any] = _frozen(
    {
        ROOT_EXTENSION_SITE: {
            "frequency_penalty": EXTENSION_PENALTY,
            "presence_penalty": EXTENSION_PENALTY,
            "store": EXTENSION_FLAG,
        },
        USAGE_EXTENSION_SITE: {
            "context_details": {
                "image_tokens": EXTENSION_COUNT,
                "text_tokens": EXTENSION_COUNT,
            },
            "cost_in_usd_ticks": EXTENSION_COUNT,
            "num_server_side_tools_used": EXTENSION_ZERO,
            "num_sources_used": EXTENSION_ZERO,
        },
    }
)
RESPONSE_EXTENSION_CONTRACT_V1: Mapping[str, Any] = _frozen(
    {
        "contract": XAI_RESPONSE_SERVER_EXTENSIONS_V1,
        "members": RESPONSE_EXTENSION_SCHEMA_V1,
        "sites_optional": True,
        "structured_members_required": True,
    }
)
RESPONSE_EXTENSION_DIGEST_V1 = hashlib.sha256(
    canonical_json_bytes(
        _plain(RESPONSE_EXTENSION_CONTRACT_V1), "xAI Responses extensions"
    )
).hexdigest()
RESPONSE_EXTENSION_SCHEMA: Mapping[str, Any] = _frozen(
    {
        ROOT_EXTENSION_SITE: {
            "frequency_penalty": EXTENSION_PENALTY,
            "presence_penalty": EXTENSION_PENALTY,
            "store": EXTENSION_FLAG,
        },
        USAGE_EXTENSION_SITE: {
            "context_details": {
                EXTENSION_UNION: [
                    {
                        "image_tokens": EXTENSION_COUNT,
                        "text_tokens": EXTENSION_COUNT,
                    },
                    {
                        "input_tokens": EXTENSION_COUNT,
                        "output_tokens": EXTENSION_COUNT,
                    },
                ]
            },
            "cost_in_usd_ticks": EXTENSION_COUNT,
            "num_server_side_tools_used": EXTENSION_ZERO,
            "num_sources_used": EXTENSION_ZERO,
        },
    }
)
RESPONSE_EXTENSION_CONTRACT: Mapping[str, Any] = _frozen(
    {
        "contract": XAI_RESPONSE_SERVER_EXTENSIONS,
        "members": RESPONSE_EXTENSION_SCHEMA,
        "sites_optional": True,
        "structured_members_required": True,
    }
)
RESPONSE_EXTENSION_DIGEST = hashlib.sha256(
    canonical_json_bytes(_plain(RESPONSE_EXTENSION_CONTRACT), "xAI Responses extensions")
).hexdigest()
# Compatibility spelling retained for callers that read the old generic name.
SERVER_EXTENSION_CONTRACT = RESPONSE_EXTENSION_CONTRACT
SERVER_EXTENSION_DIGEST = RESPONSE_EXTENSION_DIGEST
SERVER_EXTENSION_SCHEMA = RESPONSE_EXTENSION_SCHEMA
ROOT_EXTENSION_NAMES = frozenset(RESPONSE_EXTENSION_SCHEMA[ROOT_EXTENSION_SITE])
USAGE_EXTENSION_NAMES = frozenset(RESPONSE_EXTENSION_SCHEMA[USAGE_EXTENSION_SITE])
SERVER_EXTENSION_NAMES = ROOT_EXTENSION_NAMES

TYPED_EXTENSION_EXTRAS: Mapping[type, frozenset[str]] = MappingProxyType(
    {Response: ROOT_EXTENSION_NAMES, ResponseUsage: USAGE_EXTENSION_NAMES}
)


def _extension_scalar_valid(value: Any, kind: str) -> bool:
    if kind == EXTENSION_COUNT:
        return is_wire_token_count(value) and value <= MAX_EXTENSION_COUNT
    if kind == EXTENSION_ZERO:
        return type(value) is int and value == 0
    if kind == EXTENSION_PENALTY:
        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and PENALTY_MINIMUM <= value <= PENALTY_MAXIMUM
        )
    return type(value) is bool


def _check_extension_node(value: Any, schema: Any) -> None:
    if not isinstance(schema, Mapping):
        if not _extension_scalar_valid(value, schema):
            raise wire_invalid(
                "the response states a named xAI Responses extension with a value "
                "outside its exact fixed wire kind; no provider value is recorded"
            )
        return
    body = checked_wire_mapping(value, kind="xAI Responses extension object")
    if set(schema) == {EXTENSION_UNION}:
        variants = schema[EXTENSION_UNION]
        variant = next(
            (
                candidate
                for candidate in variants
                if isinstance(candidate, Mapping) and set(body) == set(candidate)
            ),
            None,
        )
        if variant is None:
            raise wire_invalid(
                "the response states an xAI Responses extension union outside its "
                "closed complete variants; no names or values are recorded"
            )
        for name, child in variant.items():
            _check_extension_node(body[name], child)
        return
    if not body or set(body) != set(schema):
        raise wire_invalid(
            "the response states an xAI Responses extension object that is vacuous, "
            "incomplete, or carries an unknown member; no names or values are recorded"
        )
    for name, child in schema.items():
        _check_extension_node(body[name], child)


def _check_response_extensions_against(
    node: Mapping[str, Any], *, site: str, schema_by_site: Mapping[str, Any]
) -> None:
    schema = schema_by_site[site]
    for name, child in schema.items():
        if name in node:
            _check_extension_node(node[name], child)


def check_response_extensions_v1(node: Mapping[str, Any], *, site: str) -> None:
    """Validate against the immutable historical v1 extension contract."""
    _check_response_extensions_against(
        node, site=site, schema_by_site=RESPONSE_EXTENSION_SCHEMA_V1
    )


def check_response_extensions(node: Mapping[str, Any], *, site: str) -> None:
    _check_response_extensions_against(
        node, site=site, schema_by_site=RESPONSE_EXTENSION_SCHEMA
    )


def _typed_extras(value: Any) -> Mapping[str, Any]:
    extra = getattr(value, "model_extra", None)
    return extra if isinstance(extra, Mapping) else {}


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _stated(node: Any, *, site: str) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        return {}
    return {name: node[name] for name in RESPONSE_EXTENSION_SCHEMA[site] if name in node}


def check_response_extensions_agree(wire: Mapping[str, Any], parsed: Response) -> None:
    raw_usage = wire.get("usage")
    typed_usage = parsed.usage
    readings = (
        (ROOT_EXTENSION_SITE, wire, parsed),
        (USAGE_EXTENSION_SITE, raw_usage, typed_usage),
    )
    if isinstance(raw_usage, Mapping) != (typed_usage is not None):
        raise wire_invalid(
            "the exact JSON and SDK reading do not state the same xAI Responses "
            "extension sites; neither reading is preferred or persisted"
        )
    for site, raw, typed in readings:
        typed_values = _stated(_typed_extras(typed), site=site)
        check_response_extensions(typed_values, site=site)
        raw_values = _stated(raw, site=site)
        if not _exact_json_equal(raw_values, typed_values):
            raise wire_invalid(
                "the exact JSON and SDK reading disagree about xAI Responses "
                "extensions; neither reading is preferred or persisted"
            )


def check_server_extensions(fields: Mapping[str, Any]) -> None:
    """Validate root-level fields using the xAI Responses extension contract."""
    check_response_extensions(fields, site=ROOT_EXTENSION_SITE)


def check_server_extensions_agree(wire: Mapping[str, Any], parsed: Response) -> None:
    """Validate that raw and SDK extension readings agree exactly."""
    check_response_extensions_agree(wire, parsed)


# -- the wire contract --------------------------------------------------------
#
# One :class:`~operatebench.providers.wire.WireShape` per documented object this
# build reads, with its allowed field set taken from the pinned SDK's own model
# for that object. Hand-listing them would drift from the schema they claim to
# transcribe, and the silent direction of that drift is a field the vendor
# documents and this build forgot — which would refuse every real response.

#: The response body itself. ``status``, ``output`` and ``usage`` are required
#: because all three are read: the first decides truncation, the second carries
#: the action, and the third is what the turn is priced and capped on. A body
#: that omits the usage block is refused rather than treated as unmeasured — see
#: :func:`check_wire_response`.
#:
#: The allowed set is the SDK's declared fields *plus* the names of the
#: server-extension contract above, and the two halves are kept visibly separate
#: because they are two different kinds of claim: the first is the vendor's own
#: schema as the pinned SDK models it, and the second is what this repository
#: observed the service return and then fixed a shape for. Membership here only
#: buys a name the right to appear at the top level; the shape of whatever it
#: carries is proven by :func:`check_server_extensions`.
RESPONSE_WIRE_SHAPE = WireShape(
    kind="response",
    allowed=declared_wire_fields(Response) | ROOT_EXTENSION_NAMES,
    required=("model", "status", "output", "usage"),
)
#: The usage block, and the two nested detail objects it may carry. The details
#: are not read — this build enables no caching feature and counts no reasoning
#: tokens separately — but they are part of the usage object, so an undeclared
#: field inside one is an undeclared field in the block this turn is priced on.
USAGE_WIRE_SHAPE = WireShape(
    kind="usage block",
    allowed=declared_wire_fields(ResponseUsage) | USAGE_EXTENSION_NAMES,
    required=USAGE_FIELDS,
)
INPUT_TOKENS_DETAILS_WIRE_SHAPE = WireShape(
    kind="input token details",
    allowed=declared_wire_fields(InputTokensDetails),
    required=(),
)
OUTPUT_TOKENS_DETAILS_WIRE_SHAPE = WireShape(
    kind="output token details",
    allowed=declared_wire_fields(OutputTokensDetails),
    required=(),
)
#: The two output item types this scaffold reads. An item of any other declared
#: type is left to the track's own parser, which refuses it as the protocol
#: failure it is: an item this scaffold did not ask for says something about the
#: answer, not about the transport.
FUNCTION_CALL_WIRE_SHAPE = WireShape(
    kind="function call output item",
    allowed=declared_wire_fields(ResponseFunctionToolCall),
    required=("type", "name", "arguments"),
)
MESSAGE_ITEM_WIRE_SHAPE = WireShape(
    kind="assistant message output item",
    allowed=declared_wire_fields(ResponseOutputMessage),
    required=("type",),
)
REASONING_ITEM_WIRE_SHAPE = WireShape(
    kind="reasoning output item",
    allowed=declared_wire_fields(ResponseReasoningItem),
    required=("type",),
)


class XAIResponsesConfigurationError(ProviderConfigurationError):
    """The environment cannot produce a usable, unambiguous xAI Responses client."""


class _XAIDispatchObserverRefusal(XAIResponsesConfigurationError):
    """Carry an observer exception through the executor's zero-attempt path."""

    def __init__(self, original: Exception) -> None:
        super().__init__("the provider dispatch observer refused the SDK request")
        self.original = original


#: The one endpoint this build talks to. Custom endpoints are not supported: a
#: base URL is where the credential is sent, so making it configurable would add
#: a redirection this build cannot audit for no measurement benefit.
XAI_BASE_URL = "https://api.x.ai/v1"

#: The credential, and the only place it may come from.
API_KEY_VARIABLE = "XAI_RESPONSES_API_KEY"

#: Environment variables the SDK honours that would silently change where a
#: request goes or what it carries. Refused rather than ignored.
#:
#: ``OPENAI_ORG_ID`` and ``OPENAI_PROJECT_ID`` are in the set for a reason worth
#: stating: they do not redirect the request, but they change *which account* it
#: is billed to and which policy applies to it, and a run whose manifest cannot
#: say who paid for it is not fully described.
REJECTED_VARIABLES: tuple[str, ...] = (
    "XAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
)


def build_client(*, api_key: str) -> openai.OpenAI:
    """The SDK client this build talks through, with its retry loop disabled.

    ``max_retries=0`` is the load-bearing argument. The SDK retries some
    failures itself by default, which would spend an episode's wall-clock budget
    on calls the runner never saw, make the recorded latency of a turn describe
    an unknown number of requests, and hide a rate limit the run should have
    recorded.
    """
    return openai.OpenAI(
        api_key=api_key, base_url=XAI_BASE_URL, max_retries=SDK_MAX_RETRIES
    )


def check_client_endpoint(client: openai.OpenAI) -> None:
    """Prove this client sends xAI traffic to the endpoint settings record.

    ``client.base_url`` is an ``httpx.URL``, and the SDK stores it with a
    trailing slash whatever form it was given, so the comparison normalises that
    one difference and nothing else — see
    :func:`~operatebench.providers.config.normalized_endpoint`.

    ``organization`` and ``project`` are checked beside it because they are the
    same hole through a different field: they do not move the request, they move
    *who is billed for it and whose policy applies*, and
    :data:`REJECTED_VARIABLES` already refuses the environment variables that
    set them. An injected client can carry them directly, and a run whose
    manifest cannot say which account paid for it is not fully described.
    """
    check_endpoint(
        client.base_url,
        expected=XAI_BASE_URL,
        provider=XAI_PROVIDER,
        setting="base_url",
        error=XAIResponsesConfigurationError,
    )
    for field, value in (
        ("organization", client.organization),
        ("project", client.project),
    ):
        if value is not None:
            raise XAIResponsesConfigurationError(
                f"this client carries an explicit {field}. That does not change "
                "where the request goes, but it changes which account is billed "
                "for it and which policy applies to it, and adapter settings — "
                "which are hashed into run identity — record neither. This build "
                f"refuses the {field} environment variable for the same reason. "
                "Build the client with `build_client`"
            )


# -- the request -------------------------------------------------------------


def request_token_bound(payload: Mapping[str, Any]) -> int:
    """An upper bound on the input tokens this exact request can be charged for.

    One definition with two callers, which is the whole point of it existing at
    all: the exchange computes this before it asks the cost guard to authorise a
    request, and a track's ledger reader computes it again when it rebuilds what
    a recorded turn reserved. A second, separately written computation of the
    same bound would drift, and a drifted bound makes every stored reservation of
    an otherwise correct run unre-derivable — a ledger that was written honestly
    and can no longer be read.

    The bound itself is
    :func:`~operatebench.providers.cost.request_input_token_bound` over this
    integration's own body; the label is this integration's, so a failure while
    encoding names the request that failed.
    """
    return request_input_token_bound(payload, "xAI Responses request bound")


_JSON_SIMPLE_ESCAPES: Mapping[str, str] = MappingProxyType(
    {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
)
_ARGUMENTS_KEY = "arguments"


def _scan_json_string(
    payload: str,
    start: int,
    *,
    max_content_bytes: int | None = None,
    collect_key_candidate: bool = True,
) -> tuple[int, str]:
    """Scan a JSON string without invoking either response JSON parser."""
    pieces: list[str] = []
    content_bytes = 0
    index = start + 1
    while index < len(payload):
        character = payload[index]
        if character == '"':
            return index + 1, "".join(pieces)
        if character == "\\":
            if index + 1 >= len(payload):
                return len(payload), ""
            escaped = payload[index + 1]
            raw_width = 2
            if escaped == "u" and index + 5 < len(payload):
                digits = payload[index + 2 : index + 6]
                try:
                    piece = chr(int(digits, 16))
                except ValueError:
                    piece = ""
                raw_width = 6
            else:
                piece = _JSON_SIMPLE_ESCAPES.get(escaped, "")
            if collect_key_candidate and len(pieces) <= len(_ARGUMENTS_KEY):
                pieces.append(piece)
            content_bytes += raw_width
            index += raw_width
        else:
            try:
                content_bytes += len(character.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise wire_invalid(
                    "the xAI response is not Unicode scalar text, so its function "
                    "arguments cannot be bounded before JSON decoding"
                ) from exc
            if collect_key_candidate and len(pieces) <= len(_ARGUMENTS_KEY):
                pieces.append(character)
            index += 1
        if max_content_bytes is not None and content_bytes > max_content_bytes:
            raise wire_invalid(
                "the xAI response carries function arguments beyond this build's "
                f"fixed {max_content_bytes}-byte limit. The arguments are refused "
                "before JSON decoding or SDK coercion, and their content is not recorded"
            )
    return len(payload), ""


def check_tool_argument_wire_bytes(payload: str, *, max_bytes: int) -> None:
    """Bound each raw ``arguments`` string before JSON or SDK coercion."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise XAIResponsesConfigurationError(
            "the raw function-argument byte limit must be a positive exact integer"
        )
    index = 0
    while index < len(payload):
        if payload[index] != '"':
            index += 1
            continue
        end, candidate = _scan_json_string(payload, index)
        cursor = end
        while cursor < len(payload) and payload[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(payload) and payload[cursor] == ":":
            cursor += 1
            while cursor < len(payload) and payload[cursor] in " \t\r\n":
                cursor += 1
            if (
                candidate == _ARGUMENTS_KEY
                and cursor < len(payload)
                and payload[cursor] == '"'
            ):
                index, _ = _scan_json_string(
                    payload,
                    cursor,
                    max_content_bytes=max_bytes,
                    collect_key_candidate=False,
                )
                continue
        index = end


# -- the response ------------------------------------------------------------


def check_response_model(response: Response, *, model: str) -> None:
    """Prove the answer came from the model the run pinned and asked for.

    ``response.model`` is the provider's own statement of which model produced
    the body, and it is the only place a run can learn that its request was not
    served by what it requested — an alias silently resolved, a request routed
    elsewhere, a response substituted in front of the client. A benchmark whose
    rows say ``model=X`` while ``X`` never answered is false in exactly the way
    an audit cannot detect, so the mismatch is refused *before* the answer's
    action is read or its token counts are banked.

    Classified as
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`, not
    as a protocol failure. Nothing about the model's *behaviour* is wrong here —
    the answer may be a flawless tool call — so filing it under the model would
    put a service-side provenance fault in the bucket that is supposed to mean
    "the model answered badly".

    Neither model string is quoted. The one the response carries is
    provider-controlled text of unbounded shape, and the one the run pinned is
    already on every row, in the adapter identity and in run identity.
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


def check_wire_response(raw: Mapping[str, Any]) -> None:
    """Prove the exact JSON the provider sent is the contract this build asked for.

    Run before the SDK's typed reading of the same bytes is trusted for
    anything, because the typed reading is not a transcript of the wire: this
    SDK constructs its response models leniently, so a ``"not-a-list"`` where
    the schema declares ``output`` arrives as that string and a usage block that
    states no counts at all arrives as a turn that simply reported none.

    Every documented object this build reads is checked against its own shape —
    the response, each output item it can interpret, the usage block and the two
    detail objects inside it — so an undeclared field carried by a *tool call*
    fails as surely as one carried by the response. An item whose declared type
    this scaffold does not read is deliberately left alone: that is a statement
    about the answer rather than about the transport, and the track's own parser
    reports it as the protocol failure it is.

    The one thing that is accepted rather than refused is the named
    server-extension block, and it is accepted only after being proven against
    the exact shape :data:`SERVER_EXTENSION_SCHEMA` fixes for it — top-level
    names, nested keys and scalar wire types alike. A field of that block whose
    shape this contract does not fix fails exactly as an undeclared field does.

    Nothing that arrived is quoted. Every message raised names the object by a
    label this build chose and counts fields; see
    :mod:`operatebench.providers.wire`.
    """
    body = checked_wire_object(raw, RESPONSE_WIRE_SHAPE)
    check_response_extensions(body, site=ROOT_EXTENSION_SITE)
    checked_wire_string(body["status"], kind="response status")
    for item in checked_wire_list(body["output"], kind="output items"):
        entry = checked_wire_mapping(item, kind="output item")
        declared = entry.get("type")
        if declared == FUNCTION_CALL_ITEM:
            call = checked_wire_object(entry, FUNCTION_CALL_WIRE_SHAPE)
            checked_wire_string(call["name"], kind="tool call name")
            checked_wire_string(call["arguments"], kind="tool call arguments")
        elif declared == MESSAGE_ITEM:
            checked_wire_object(entry, MESSAGE_ITEM_WIRE_SHAPE)
        elif declared == REASONING_ITEM:
            checked_wire_object(entry, REASONING_ITEM_WIRE_SHAPE)
    usage = checked_wire_object(body["usage"], USAGE_WIRE_SHAPE)
    check_response_extensions(usage, site=USAGE_EXTENSION_SITE)
    for field, shape in (
        ("input_tokens_details", INPUT_TOKENS_DETAILS_WIRE_SHAPE),
        ("output_tokens_details", OUTPUT_TOKENS_DETAILS_WIRE_SHAPE),
    ):
        if usage.get(field) is not None:
            checked_wire_object(usage[field], shape)
    wire_token_usage(usage, fields=USAGE_FIELDS)


def check_response_admissible(response: WireResponse[Response], *, model: str) -> None:
    """Everything that must be true of a body before anything is read out of it.

    Four questions, in this order, and all of them answered before the action is
    parsed or the token counts are banked:

    #. **Is the body on the wire the response contract this build asked for?**
       Field presence, field types, nested shape and undeclared extras, checked
       on the exact JSON the provider sent — see :func:`check_wire_response`.
       Asked first because it is the question the SDK's object cannot answer at
       all: a coerced count and a measured one are the same integer by the time
       that object exists, and a body that states no model at all would reach
       the identity check below as a mismatch rather than as the malformed
       response it is.
    #. **Did the model this run pinned produce it?** See
       :func:`check_response_model`.
    #. **Does the SDK's own reading agree?** The typed model is produced from
       the same bytes and checked for the fields this build reads, so a
       disagreement between the two readings is refused rather than resolved.
       That includes the named server-extension block, which this SDK parks in
       ``model_extra`` rather than dropping: it is validated on the typed object
       as well as on the wire, and the two readings of it must state the same
       thing — see :func:`check_server_extensions_agree`. Every *other* typed
       extra, at the root or at any depth, is still refused.
    #. **Are its token counts countable?** Asked twice on purpose: once against
       the wire, where a boolean is a boolean, and once against the object that
       becomes an :class:`~operatebench.providers.telemetry.AdapterUsage`.

    All four are
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`,
    which is what makes them different from the protocol failures a track's own
    parser raises. Nothing here is a claim about how the *model* behaved — the
    answer may name a perfect action — so the turn is refused with the attempt
    recorded as one that received a response and its reservation kept as
    exposure, never settled at a cost this build did not measure.
    """
    check_wire_response(response.wire)
    parsed = response.parsed
    check_response_model(parsed, model=model)
    check_response_contract(parsed, allowed_extras_by_model=TYPED_EXTENSION_EXTRAS)
    check_response_extensions_agree(response.wire, parsed)
    check_response_collection(parsed.output, kind="output items")
    usage = parsed.usage
    if usage is not None:
        check_response_object(usage, fields=USAGE_FIELDS, kind="usage block")
        checked_token_usage(usage.input_tokens, usage.output_tokens)


def response_usage(response: WireResponse[Response]) -> TokenUsage:
    """The two counts this build prices a turn on, taken from the wire.

    From the wire and not from the SDK's object, and that is the whole change:
    the object cannot say whether ``137`` arrived as a JSON number, as ``"137"``
    or as ``true``, because all three reach it as integers. A count this build
    is going to multiply by a rate and commit against a cap has to be one the
    provider actually stated.

    Not null-tolerant. A body with no usage block, or one whose counts are
    absent, never reaches here: :func:`check_wire_response` has already refused
    it as
    :data:`~operatebench.providers.faults.PROVIDER_FAULT_RESPONSE_INVALID`,
    which keeps the reservation as exposure instead of quietly settling a turn
    that really did consume tokens at a cost of nothing. Validated again here
    rather than trusted, because this is the function whose return value becomes
    an :class:`~operatebench.providers.telemetry.AdapterUsage` and a settled
    cost.
    """
    usage = checked_wire_object(response.wire.get("usage"), USAGE_WIRE_SHAPE)
    return wire_token_usage(usage, fields=USAGE_FIELDS)


def classify_exception(exception: Exception) -> Fault | None:
    """Map one SDK error onto the contract's fault set, or decline.

    ``None`` means "not a provider fault": something inside this integration
    broke, and the runner should record it as the adapter failure it is rather
    than have it dressed up as an outage.
    """
    # Checked before its base class: an SDK timeout *is* an
    # ``APIConnectionError``, and the two are worth telling apart.
    if isinstance(exception, openai.APITimeoutError):
        return Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, openai.APIConnectionError):
        return Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, openai.APIResponseValidationError):
        # A whole body arrived and the SDK refused to read it. Recorded as an
        # attempt that *received a response*, because it did: saying otherwise
        # would report a transport failure that did not happen and would hide
        # the one case where the provider answered with something unreadable.
        return Fault(PROVIDER_FAULT_RESPONSE_INVALID, False, None, response_received=True)
    if isinstance(exception, openai.APIStatusError):
        return classify_http_status(exception.status_code)
    return None


# -- the exchange -------------------------------------------------------------


class XAIResponsesExchange:
    """One client, one pinned model, one payload exchanged for one wire answer.

    Constructed by whichever track is going to build the bodies. It holds the
    per-instance state a turn owns while it runs — the retry loop's usage and
    attempt evidence, and the single-flight guard that stops a second turn
    erasing them — and it exposes exactly three things a caller needs around a
    turn: :meth:`turn` to take the guard, :meth:`clear` to drop the previous
    turn's measurement, and :meth:`exchange` to make the call.

    ``single_flight_label`` is the name the busy refusal uses for the caller.
    It is required rather than defaulted: the message an operator reads should
    name the object they hold, and this class does not know what that is.
    """

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        single_flight_label: str,
        retry: ProviderRetryPolicy | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        profile: RequestProfile | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
        max_tool_argument_wire_bytes: int,
    ) -> None:
        # First, and before anything is read off the client: a profile that is
        # not this model's is refused where it costs an error message rather
        # than a rejected request, a spent reservation and a durable row.
        self._profile = check_request_profile(profile, model=model)
        # Before anything else this client is trusted for: where it sends. A
        # factory builds its own client against the pinned endpoint, so the
        # factory was never the way in — a caller handing in a client is, and
        # this constructor is public.
        check_client_endpoint(client)
        if client.max_retries != SDK_MAX_RETRIES:
            raise XAIResponsesConfigurationError(
                f"this client was built with max_retries={client.max_retries}. The "
                "adapter owns the retry policy, so a second, hidden one underneath "
                "it would spend the episode's wall-clock budget on calls the runner "
                "never saw, make a turn's measured latency describe an unknown "
                "number of requests, and hide a rate limit the run should have "
                f"recorded. Build the client with max_retries={SDK_MAX_RETRIES}"
            )
        self._model = model
        self._client = client
        self._retry = check_retry_policy(
            retry or ProviderRetryPolicy(),
            provider=XAI_PROVIDER,
            error=XAIResponsesConfigurationError,
        )
        self._executor: TurnExecutor[WireResponse[Response]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        # One turn at a time on this instance: the executor's usage and attempt
        # evidence are per-instance state that a turn owns while it runs. See
        # :class:`~operatebench.providers.faults.SingleFlight`.
        self._single_flight = SingleFlight(adapter=single_flight_label)
        if before_dispatch is not None and not callable(before_dispatch):
            raise XAIResponsesConfigurationError(
                "the provider dispatch observer must be a zero-argument callable"
            )
        self._before_dispatch = before_dispatch
        if (
            type(max_tool_argument_wire_bytes) is not int
            or max_tool_argument_wire_bytes <= 0
        ):
            raise XAIResponsesConfigurationError(
                "the raw function-argument byte limit must be a positive exact integer"
            )
        self._max_tool_argument_wire_bytes = max_tool_argument_wire_bytes

    def __repr__(self) -> str:
        """Names the run, never the credential.

        Written out rather than left to a dataclass: a generated ``repr`` would
        render whatever the client holds, and the client holds an API key.
        """
        return (
            f"<XAIResponsesExchange provider={XAI_PROVIDER} "
            f"model={self._model} api={XAI_RESPONSES_API}>"
        )

    @property
    def model(self) -> str:
        """The one model this exchange sends to, and reads answers from.

        Read-only, and exposed because a composing track has to be able to prove
        that the request it is about to build names the same model this exchange
        would send it to — *before* the request leaves. The check inside
        :func:`check_response_model` catches the mismatch on the way back, which
        is one wasted request and one spent reservation too late.
        """
        return self._model

    @property
    def retry(self) -> ProviderRetryPolicy:
        return self._retry

    @property
    def request_profile(self) -> RequestProfile:
        """The frozen request shape this exchange's model is sent."""
        return self._profile

    @property
    def cost_guard(self) -> CostGuard | None:
        """The guard this exchange asks before every request, if the run has one.

        Exposed so a runner can prove that the guard bound to the manifest is
        the guard the requests will actually be authorised by.
        """
        return self._executor.cost_guard

    def turn(self) -> SingleFlight:
        """Take this instance's single-flight guard for the whole of one turn.

        Entered by the caller rather than inside :meth:`exchange`, because the
        state the guard protects is wider than the call: a track clears the
        measurement, builds a body, exchanges it and parses the answer, and a
        second turn that interleaved with any of that would leave neither able to
        say which response was its own.
        """
        return self._single_flight

    def clear(self) -> None:
        """Drop the previous turn's usage and attempt evidence."""
        self._executor.clear()

    def last_usage(self) -> AdapterUsage:
        return self._executor.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._executor.last_telemetry()

    def exchange(
        self, payload: Mapping[str, Any], deadline: TurnDeadline
    ) -> WireResponse[Response]:
        """Check the body against the settings, then run the turn.

        The check happens before the clock starts, before an attempt is counted
        and before the cost guard is asked for a reservation. Nothing below can
        put it right — a request that has left is evidence, and the manifest it
        would be read against would be describing a different one.

        The endpoint is proven again here and not only in the constructor. This
        SDK's ``base_url`` is a settable attribute, so a client that was pinned
        when the exchange was built can be pointed somewhere else before the turn
        that uses it — and the check that matters is the one immediately before
        the credential is sent.
        """
        check_client_endpoint(self._client)
        check_payload_fields(
            payload,
            profile_id=self._profile.profile_id,
            sent=self._profile.sent_fields(),
            omitted=self._profile.omitted_fields(),
        )

        def check_before_settlement(response: WireResponse[Response]) -> None:
            check_response_admissible(response, model=self._model)
            response.admit_response_id()

        try:
            return self._executor.run(
                payload=payload,
                deadline=deadline,
                token_bound=request_token_bound(payload),
                dispatch=lambda timeout: self._dispatch(payload, timeout),
                check_before_dispatch=self._check_before_dispatch,
                classify=classify_exception,
                check_identity=check_before_settlement,
                usage_of=response_usage,
            )
        except _XAIDispatchObserverRefusal as refusal:
            raise refusal.original from None

    def _check_before_dispatch(self) -> None:
        check_client_endpoint(self._client)
        if self._before_dispatch is not None:
            try:
                self._before_dispatch()
            except Exception as exc:
                raise _XAIDispatchObserverRefusal(exc) from exc

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[Response]:
        """One request, holding the bytes that came back beside the SDK's reading.

        ``with_raw_response`` is the SDK's own supported way to keep both, and
        using it rather than a hand-rolled transport wrapper is what keeps the
        request serialisation, the error mapping and the response model the
        SDK's. It is not free: it marks the request with a header, which is a
        change to what leaves this process, so the mode is named in adapter
        settings as :data:`XAI_RESPONSE_CAPTURE` and the adapter version
        moved with it.

        Both halves are deferred rather than computed here. A body that will not
        decode has to be refused *inside* the identity check, where the attempt
        is recorded as one that received a response; computing it here would
        make it an exception out of dispatch, which is where this build records
        the attempts that received nothing.
        """
        raw = self._client.responses.with_raw_response.create(**payload, timeout=timeout)
        return WireResponse(
            raw.text,
            raw.parse,
            kind="xAI response",
            pre_json_check=lambda text: check_tool_argument_wire_bytes(
                text, max_bytes=self._max_tool_argument_wire_bytes
            ),
        )


__all__ = [
    "ACCEPTED_ITEM_TYPES",
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "COMPLETED_STATUS",
    "EXTENSION_COUNT",
    "EXTENSION_FLAG",
    "EXTENSION_PENALTY",
    "EXTENSION_ZERO",
    "FUNCTION_CALL_ITEM",
    "FUNCTION_CALL_WIRE_SHAPE",
    "GROK_4_5_CONTRACT_SOURCE",
    "GROK_4_5_MODEL",
    "GROK_4_5_PROFILE",
    "INCOMPLETE_STATUS",
    "INPUT_TOKENS_DETAILS_WIRE_SHAPE",
    "LOW_REASONING",
    "MAX_EXTENSION_COUNT",
    "MAX_OUTPUT_TOKENS",
    "MESSAGE_ITEM",
    "MESSAGE_ITEM_WIRE_SHAPE",
    "MODEL_REQUEST_PROFILES",
    "OUTPUT_LIMIT_REASON",
    "OUTPUT_TOKENS_DETAILS_WIRE_SHAPE",
    "PARALLEL_TOOL_CALLS",
    "PENALTY_MAXIMUM",
    "PENALTY_MINIMUM",
    "PROHIBITED_REQUEST_FIELDS",
    "REASONING_EFFORT_DEFAULT",
    "REASONING_EFFORT_LOW",
    "REASONING_EFFORT_VALUES",
    "REASONING_ITEM",
    "REASONING_ITEM_WIRE_SHAPE",
    "REJECTED_VARIABLES",
    "RESPONSE_EXTENSION_CONTRACT",
    "RESPONSE_EXTENSION_DIGEST",
    "RESPONSE_EXTENSION_SCHEMA",
    "RESPONSE_WIRE_SHAPE",
    "ROOT_EXTENSION_NAMES",
    "ROOT_EXTENSION_SITE",
    "SERVER_EXTENSION_CONTRACT",
    "SERVER_EXTENSION_DIGEST",
    "SERVER_EXTENSION_NAMES",
    "SERVER_EXTENSION_SCHEMA",
    "STORE",
    "TEMPERATURE",
    "TOOL_CHOICE",
    "USAGE_EXTENSION_NAMES",
    "USAGE_EXTENSION_SITE",
    "USAGE_FIELDS",
    "USAGE_WIRE_SHAPE",
    "XAI_BASE_URL",
    "XAI_PROVIDER",
    "XAI_RESPONSES_API",
    "XAI_RESPONSE_CAPTURE",
    "XAI_RESPONSE_SERVER_EXTENSIONS",
    "RequestProfile",
    "XAIResponsesConfigurationError",
    "XAIResponsesExchange",
    "build_client",
    "check_client_endpoint",
    "check_request_profile",
    "check_response_admissible",
    "check_response_extensions",
    "check_response_extensions_agree",
    "check_response_model",
    "check_server_extensions",
    "check_server_extensions_agree",
    "check_tool_argument_wire_bytes",
    "check_wire_response",
    "classify_exception",
    "request_profile_for",
    "request_token_bound",
    "response_usage",
]
