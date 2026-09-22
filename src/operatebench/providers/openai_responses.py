"""The OpenAI Responses exchange: a payload in, a checked wire answer out.

This is everything about talking to the OpenAI Responses API that is *not* a
question about what was being asked. Which endpoint the credential goes to, which
request shape a pinned model is entitled to, that the body matches the settings a
run recorded, one turn at a time per client, the bounded retry loop with its cost
guard and attempt evidence, the raw bytes held beside the SDK's typed reading of
them, and whether what came back is the response contract this build asked for.

What it deliberately does not know is the *question*. A caller hands
:meth:`OpenAIResponsesExchange.exchange` a finished request body and a deadline
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
:data:`OPENAI_RESPONSE_SERVER_EXTENSIONS` — and accepts precisely it: closed
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
from operatebench.providers.usage import TokenUsage, checked_token_usage
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
OPENAI_PROVIDER = "openai"

#: How this exchange gets at the bytes the provider sent.
#:
#: The SDK's own ``with_raw_response`` surface, rather than a hand-rolled
#: transport wrapper: it is the vendor's supported way to hold the HTTP response
#: and the typed model at once, it keeps the request serialisation the SDK's,
#: and it is exercised in these tests over a real client on an in-process
#: transport. Named in settings because it is result-affecting — it adds a
#: header to every request, and it decides what this build is able to refuse.
OPENAI_RESPONSE_CAPTURE = "openai_sdk_with_raw_response_v1"

#: The API surface this exchange speaks. Pinned, and named in settings: the same
#: model reached through Chat Completions is a different request mapping and a
#: different response contract, so it would be a different experiment.
OPENAI_API = "responses"

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


#: The one OpenAI model this lane pins.
GPT_5_6_LUNA_MODEL = "gpt-5.6-luna"

#: The published model page this build read :data:`GPT_5_6_LUNA_PROFILE`'s
#: request contract from.
#:
#: Named as a constant beside the profile that relies on it, because
#: ``model_contract_verified: true`` is a claim in run identity and a claim is
#: worth what a reader's ability to check it is worth. A boolean on its own says
#: only that somebody said they looked.
GPT_5_6_LUNA_CONTRACT_SOURCE = (
    "https://developers.openai.com/api/docs/models/gpt-5.6-luna.md"
)

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
REASONING_EFFORT_NONE = "none"
REASONING_EFFORT_DEFAULT = "medium"

#: The exact reasoning object the no-reasoning track sends. Immutable: a profile
#: is frozen, and a mapping shared by reference is not.
NO_REASONING: Mapping[str, Any] = MappingProxyType({"effort": REASONING_EFFORT_NONE})

#: What a model this build ships no model-specific profile for is sent.
#:
#: :func:`~operatebench.providers.config.check_model` is a shape check rather
#: than an allowlist — a build that refused every model released after it is not
#: what a benchmark needs — so an unencoded model still has to be sent
#: something, and this is it.
BASELINE_PROFILE = RequestProfile(
    profile_id="openai_baseline_temperature_zero_reasoning_omitted_v1",
    temperature=TEMPERATURE,
    reasoning=None,
    contract_verified=False,
)

#: ``gpt-5.6-luna``: reasoning explicitly off, no sampling parameter at all.
#:
#: Both halves are read off the model page named in
#: :data:`GPT_5_6_LUNA_CONTRACT_SOURCE`, and they are read in opposite
#: directions, which is why ``contract_verified`` is ``True`` for this profile
#: without being a claim about everything the model does:
#:
#: * The page states that this model supports the Responses API and function
#:   calling, and that ``reasoning.effort`` accepts
#:   :data:`REASONING_EFFORT_VALUES` with ``medium`` as the default. So the
#:   standardized no-reasoning track can *state* what it asks for rather than
#:   infer it, and it sends :data:`NO_REASONING` explicitly. Omitting the field
#:   would select the default instead, silently.
#: * The same page does not establish that this model accepts a sampling
#:   parameter. An undocumented knob sent anyway is a request shape nobody can
#:   check, so ``temperature`` is ``None`` — absent from the body entirely
#:   rather than sent as null.
#:
#: ``contract_verified`` therefore means what it says on this profile: every
#: field this profile *claims* — the reasoning object it sends and the sampling
#: parameter it omits — follows the model's own published request contract.
GPT_5_6_LUNA_PROFILE = RequestProfile(
    profile_id="gpt56luna_reasoning_none_sampling_omitted_v2",
    temperature=None,
    reasoning=NO_REASONING,
    contract_verified=True,
)

#: Model identifier to frozen request shape. Matched whole, like a price: an
#: alias is not the model it currently points at.
MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {GPT_5_6_LUNA_MODEL: GPT_5_6_LUNA_PROFILE}
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
    raise OpenAIConfigurationError(
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
ACCEPTED_ITEM_TYPES: frozenset[str] = frozenset({FUNCTION_CALL_ITEM, MESSAGE_ITEM})

# -- the response server-extension contract -----------------------------------
#
# The service returns a small block of top-level fields on a Responses body that
# neither the pinned SDK's ``Response`` model nor the next major SDK's declares.
# Refusing a body whole because it carries one is the right default and it was
# this build's only answer, so a real HTTP 200 from the pinned model could not be
# read at all: the action went unparsed and the turn's reservation was held as
# exposure over fields nothing here reads.
#
# So this lane names **one exact contract** for that block and accepts precisely
# it. Four properties make that safe to do:
#
# * It is *closed*, recursively. Five top-level names, fixed children under each,
#   and an unknown key at any depth is still ``provider_response_invalid``.
# * It is *complete where it is present*. The five top-level names are
#   independently optional — a service that stops sending a whole block has
#   stopped making a statement nothing here reads — but an object that does
#   arrive states every child fixed for it. A half-stated object is not the shape
#   this contract transcribed, and accepting one would quietly turn an exact
#   contract into "any subset of the observed shape".
# * It is *typed*. Every scalar is checked against the wire type the observed
#   shape stated — not against what a lenient reader would coerce it to.
# * It is *inert*. Nothing in the block is read for an action, for the model's
#   stated identity or for the token counts a turn is priced and capped on. It is
#   validated so that a body carrying it can be accepted, and then it is dropped:
#   no value from it is measured, returned or written anywhere.
#
# What this contract is **not** is a vendor declaration. It transcribes the field
# names and type shape this repository observed the service return; the pinned
# SDK declares none of them, which is exactly why the block needed a contract of
# its own rather than an entry in :func:`declared_wire_fields`. A
# provider-observed contract can move without notice, so it is named, hashed and
# recorded in adapter settings, and it is inside ``configuration_id``: a run made
# under it is not a run made under any other reading of the same body.

#: The name of that contract, as adapter settings record it.
#:
#: ``_v2``. The ``_v1`` reading transcribed these same names and types and then
#: accepted a present object that stated any subset of them, including none; this
#: one requires a present object to be whole. Two readings of one body are two
#: contracts, so the name moves with the reading rather than only with the shape.
OPENAI_RESPONSE_SERVER_EXTENSIONS = "openai_response_server_extensions_v2"

#: The scalar vocabulary the contract fixes. Four kinds, and each one is a
#: closed-set label this build chose, so it may appear in a durable failure
#: detail where a field name or a value may not.
EXTENSION_COUNT = "non_negative_integer"
EXTENSION_PENALTY = "penalty_number"
EXTENSION_FLAG = "boolean"
EXTENSION_TEXT = "bounded_text"

#: The inclusive range the vendor publishes for these two parameters on its
#: request surface, and the range this contract holds their echoes to. A number
#: outside it is not the parameter this contract names, whatever else it may be,
#: and is refused rather than ignored: the alternative is a build that says
#: "these two fields have a fixed shape" while accepting any float.
PENALTY_MINIMUM = -2.0
PENALTY_MAXIMUM = 2.0

#: The longest string this contract accepts where it fixes text. Bounded because
#: the value is provider-controlled text of otherwise unbounded size, and this
#: build holds a response body in memory for the length of a turn; non-empty
#: because a stated value of nothing states nothing.
MAX_EXTENSION_TEXT_CHARACTERS = 128


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


#: The contract itself: every name it accepts, at every level, and the exact
#: shape each one must have when it is present.
#:
#: A mapping value is an object whose keys are closed to exactly these children.
#: A string value is one of the four scalar kinds above.
#:
#: Optionality is decided by *level*, and the two levels are not the same
#: question. Each of the five top-level names may be absent, independently of the
#: others: this build reads none of them, so a service that stops sending a whole
#: block has not broken anything a run depends on, and demanding it would refuse
#: a body this build can read perfectly well. Inside a block that *is* present,
#: every child listed here is required — see
#: :data:`SERVER_EXTENSION_CONTRACT`, which is the pair of rules this shape is
#: read under and is hashed together with it.
SERVER_EXTENSION_SCHEMA: Mapping[str, Any] = _frozen(
    {
        "billing": {"payer": EXTENSION_TEXT},
        "frequency_penalty": EXTENSION_PENALTY,
        "presence_penalty": EXTENSION_PENALTY,
        "store": EXTENSION_FLAG,
        "tool_usage": {
            "image_gen": {
                "input_tokens": EXTENSION_COUNT,
                "input_tokens_details": {
                    "image_tokens": EXTENSION_COUNT,
                    "text_tokens": EXTENSION_COUNT,
                },
                "output_tokens": EXTENSION_COUNT,
                "output_tokens_details": {
                    "image_tokens": EXTENSION_COUNT,
                    "text_tokens": EXTENSION_COUNT,
                },
                "total_tokens": EXTENSION_COUNT,
            },
            "web_search": {"num_requests": EXTENSION_COUNT},
        },
    }
)

#: The top-level names the contract covers, and the only extras this lane will
#: accept on a response object.
SERVER_EXTENSION_NAMES: frozenset[str] = frozenset(SERVER_EXTENSION_SCHEMA)

#: The whole contract, as one document: the name, the two rules that decide what
#: a present object has to state, and the shape above.
#:
#: It exists because the shape alone is not the contract. This version and the
#: superseded one transcribe identical names and identical scalar kinds and still
#: read the same body differently — one accepts an object stating half of itself
#: and one refuses it — so a digest taken over
#: :data:`SERVER_EXTENSION_SCHEMA` alone would be equal across the two, and two
#: manifests would claim one configuration for two readings. Hashing the rules
#: beside the shape is what makes the recorded claim answer "which bodies will
#: this run read as answers" rather than "which field names did it know about".
SERVER_EXTENSION_CONTRACT: Mapping[str, Any] = _frozen(
    {
        "contract": OPENAI_RESPONSE_SERVER_EXTENSIONS,
        "members": SERVER_EXTENSION_SCHEMA,
        # Present object, whole object.
        "object_members_required": True,
        # ...and each of the five top-level names may be absent on its own.
        "top_level_members_required": False,
    }
)

#: A digest over that document, recorded beside its name.
#:
#: The name alone would be a label: two builds could ship one
#: ``openai_response_server_extensions`` version and accept different bodies, and
#: the run manifests would agree. The digest is what makes the recorded claim
#: re-derivable — it moves the moment the accepted contract does, and it is
#: inside ``configuration_id``.
SERVER_EXTENSION_DIGEST = hashlib.sha256(
    canonical_json_bytes(
        _plain(SERVER_EXTENSION_CONTRACT), "openai response server extensions"
    )
).hexdigest()

#: How a refusal names the object it was checking. One fixed label for every
#: level of the block: which field carried the fault is not something a durable
#: row needs, and the value beside it is provider-controlled.
_EXTENSION_OBJECT_KIND = "named server-extension object"


def _extension_scalar_valid(value: Any, kind: str) -> bool:
    """Whether one value is the exact wire scalar the contract fixes for it.

    Exact, and by ``type`` rather than by ``isinstance``, everywhere it matters.
    Python makes ``True`` an ``int``, so an ``isinstance`` count check would
    accept a boolean as one and an ``isinstance`` number check would accept it
    as a penalty of one — and a lenient reader downstream would render either as
    a measurement nobody took.
    """
    if kind == EXTENSION_COUNT:
        # The same rule the token counts are read under, and deliberately the
        # same function: two independently written definitions of "an exact
        # non-negative integer on the wire" would drift.
        return is_wire_token_count(value)
    if kind == EXTENSION_PENALTY:
        return (
            type(value) in (int, float)
            # NaN compares false against every bound, so a range check alone
            # would let it through the comparison that exists to exclude it.
            and math.isfinite(value)
            and PENALTY_MINIMUM <= value <= PENALTY_MAXIMUM
        )
    if kind == EXTENSION_FLAG:
        return type(value) is bool
    # :data:`EXTENSION_TEXT`, the last of the four. The vocabulary is closed and
    # is this module's own, so there is no fifth kind to fall through to.
    return (
        type(value) is str
        and bool(value.strip())
        and len(value) <= MAX_EXTENSION_TEXT_CHARACTERS
    )


def _check_extension_node(value: Any, schema: Any) -> None:
    """One node of the contract, against the node that arrived, recursively."""
    if not isinstance(schema, Mapping):
        if not _extension_scalar_valid(value, schema):
            raise wire_invalid(
                "the response's named server-extension block states a value where "
                f"{OPENAI_RESPONSE_SERVER_EXTENSIONS} fixes an exact one of kind "
                f"{schema!r}, so the block is not the contract this build accepts "
                "and the body is refused rather than read past. Nothing about the "
                "value is recorded here — neither which field carried it nor what "
                "it was: the field set is this contract's own and the value is "
                "provider-controlled, and a durable failure row records only this "
                "build's own fixed detail and closed-set values"
            )
        return
    body = checked_wire_mapping(value, kind=_EXTENSION_OBJECT_KIND)
    undeclared = sum(1 for key in body if str(key) not in schema)
    if undeclared:
        raise wire_invalid(
            f"the response's named server-extension block carries {undeclared} "
            f"key(s) at one level that {OPENAI_RESPONSE_SERVER_EXTENSIONS} does "
            "not fix. The contract is closed at every depth: this build accepts "
            "this block only because it can describe the whole of it, so a part it "
            "cannot describe is refused exactly as an undeclared field anywhere "
            "else in the body is. The names and values are not recorded here: they "
            "are provider-controlled, and a durable failure row records only this "
            "build's own fixed detail and closed-set values"
        )
    if any(name not in body for name in schema):
        raise wire_invalid(
            "the response's named server-extension block states an object that "
            f"omits at least one member {OPENAI_RESPONSE_SERVER_EXTENSIONS} fixes "
            "for it. A whole block of this contract may be absent — this build "
            "reads none of it — but an object that is present is a statement, and "
            "this contract describes one shape for it rather than every subset of "
            "one: a partly stated object was never observed, so accepting it would "
            "widen an exact contract into a guess. How many members were missing, "
            "which ones, and what the object did state are not recorded here: the "
            "absences are provider-controlled, and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        )
    for name, child in schema.items():
        _check_extension_node(body[name], child)


def check_server_extensions(fields: Mapping[str, Any]) -> None:
    """Prove every named extension this mapping states is the exact fixed shape.

    A name that is absent is not checked, and that is this level's rule rather
    than the contract's rule everywhere: the five top-level names are
    independently optional, and everything below one of them is required once
    its parent is here.

    Given the whole object rather than a projection of it, so the caller cannot
    accidentally hand over a filtered copy and prove the contract about
    something the provider did not send. Names the contract does not cover are
    not this function's business: the response's own allowed field set refuses
    an undeclared top-level field, and
    :func:`~operatebench.providers.response.check_response_contract` refuses an
    undeclared typed extra.
    """
    for name, schema in SERVER_EXTENSION_SCHEMA.items():
        if name in fields:
            _check_extension_node(fields[name], schema)


def check_server_extensions_agree(wire: Mapping[str, Any], parsed: Response) -> None:
    """Check the SDK's own reading of the block, and that the two readings agree.

    The block is stated twice by the time it gets here: once in the exact JSON
    the provider sent, and once on the typed response, where this SDK parks a
    field its model does not declare in ``model_extra`` rather than dropping it.
    Both are validated, because a contract proven about one reading is not a
    contract about the other — and neither reading is preferred over the other
    when they differ, because there is no principled way to pick: a body whose
    typed extras are not the extras on the wire is not one body this build can
    describe, and it is refused.

    What this deliberately does **not** do is clear or rewrite ``model_extra``.
    Emptying it would make the two readings agree by force, hide the very fields
    this contract exists to describe, and leave the SDK's typed object in a state
    the provider never sent. The typed parse stays exactly as the SDK produced
    it.
    """
    extra = getattr(parsed, "model_extra", None)
    typed: Mapping[str, Any] = extra if isinstance(extra, Mapping) else {}
    check_server_extensions(typed)
    if {name: wire[name] for name in SERVER_EXTENSION_NAMES if name in wire} != {
        name: typed[name] for name in SERVER_EXTENSION_NAMES if name in typed
    }:
        raise wire_invalid(
            "the exact JSON the provider sent and this pinned SDK's typed reading "
            "of the same bytes do not state the same named server-extension block, "
            "so there is no single body here for this build to check against "
            f"{OPENAI_RESPONSE_SERVER_EXTENSIONS}. Neither reading is preferred "
            "over the other and neither is edited to agree with the other: the "
            "response is refused whole. What each reading stated is not recorded "
            "here: it is provider-controlled, and a durable failure row records "
            "only this build's own fixed detail and closed-set values"
        )


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
    allowed=declared_wire_fields(Response) | SERVER_EXTENSION_NAMES,
    required=("model", "status", "output", "usage"),
)
#: The usage block, and the two nested detail objects it may carry. The details
#: are not read — this build enables no caching feature and counts no reasoning
#: tokens separately — but they are part of the usage object, so an undeclared
#: field inside one is an undeclared field in the block this turn is priced on.
USAGE_WIRE_SHAPE = WireShape(
    kind="usage block",
    allowed=declared_wire_fields(ResponseUsage),
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


class OpenAIConfigurationError(ProviderConfigurationError):
    """The environment cannot produce a usable, unambiguous OpenAI client."""


class _DispatchObserverRefusal(OpenAIConfigurationError):
    """Carry an observer exception through the executor's zero-attempt path."""

    def __init__(self, original: Exception) -> None:
        super().__init__("the provider dispatch observer refused the SDK request")
        self.original = original


#: The one endpoint this build talks to. Custom endpoints are not supported: a
#: base URL is where the credential is sent, so making it configurable would add
#: a redirection this build cannot audit for no measurement benefit.
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: The credential, and the only place it may come from.
API_KEY_VARIABLE = "OPENAI_API_KEY"

#: Environment variables the SDK honours that would silently change where a
#: request goes or what it carries. Refused rather than ignored.
#:
#: ``OPENAI_ORG_ID`` and ``OPENAI_PROJECT_ID`` are in the set for a reason worth
#: stating: they do not redirect the request, but they change *which account* it
#: is billed to and which policy applies to it, and a run whose manifest cannot
#: say who paid for it is not fully described.
REJECTED_VARIABLES: tuple[str, ...] = (
    "OPENAI_BASE_URL",
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
        api_key=api_key, base_url=OPENAI_BASE_URL, max_retries=SDK_MAX_RETRIES
    )


def check_client_endpoint(client: openai.OpenAI) -> None:
    """Prove this client sends OpenAI traffic to the endpoint settings record.

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
        expected=OPENAI_BASE_URL,
        provider=OPENAI_PROVIDER,
        setting="base_url",
        error=OpenAIConfigurationError,
    )
    for field, value in (
        ("organization", client.organization),
        ("project", client.project),
    ):
        if value is not None:
            raise OpenAIConfigurationError(
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
    return request_input_token_bound(payload, "openai request bound")


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
    check_server_extensions(body)
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
    usage = checked_wire_object(body["usage"], USAGE_WIRE_SHAPE)
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
    check_response_contract(parsed, allowed_root_extras=SERVER_EXTENSION_NAMES)
    check_server_extensions_agree(response.wire, parsed)
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


class OpenAIResponsesExchange:
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
            raise OpenAIConfigurationError(
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
            provider=OPENAI_PROVIDER,
            error=OpenAIConfigurationError,
        )
        self._executor: TurnExecutor[WireResponse[Response]] = TurnExecutor(
            retry=self._retry, sleep=sleep, clock=clock, cost_guard=cost_guard
        )
        # One turn at a time on this instance: the executor's usage and attempt
        # evidence are per-instance state that a turn owns while it runs. See
        # :class:`~operatebench.providers.faults.SingleFlight`.
        self._single_flight = SingleFlight(adapter=single_flight_label)
        if before_dispatch is not None and not callable(before_dispatch):
            raise OpenAIConfigurationError(
                "the provider dispatch observer must be a zero-argument callable"
            )
        self._before_dispatch = before_dispatch

    def __repr__(self) -> str:
        """Names the run, never the credential.

        Written out rather than left to a dataclass: a generated ``repr`` would
        render whatever the client holds, and the client holds an API key.
        """
        return (
            f"<OpenAIResponsesExchange provider={OPENAI_PROVIDER} "
            f"model={self._model} api={OPENAI_API}>"
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
        except _DispatchObserverRefusal as refusal:
            raise refusal.original from None

    def _check_before_dispatch(self) -> None:
        check_client_endpoint(self._client)
        if self._before_dispatch is not None:
            try:
                self._before_dispatch()
            except Exception as exc:
                raise _DispatchObserverRefusal(exc) from exc

    def _dispatch(
        self, payload: Mapping[str, Any], timeout: float
    ) -> WireResponse[Response]:
        """One request, holding the bytes that came back beside the SDK's reading.

        ``with_raw_response`` is the SDK's own supported way to keep both, and
        using it rather than a hand-rolled transport wrapper is what keeps the
        request serialisation, the error mapping and the response model the
        SDK's. It is not free: it marks the request with a header, which is a
        change to what leaves this process, so the mode is named in adapter
        settings as :data:`OPENAI_RESPONSE_CAPTURE` and the adapter version
        moved with it.

        Both halves are deferred rather than computed here. A body that will not
        decode has to be refused *inside* the identity check, where the attempt
        is recorded as one that received a response; computing it here would
        make it an exception out of dispatch, which is where this build records
        the attempts that received nothing.
        """
        raw = self._client.responses.with_raw_response.create(**payload, timeout=timeout)
        return WireResponse(raw.text, raw.parse, kind="openai response")


__all__ = [
    "ACCEPTED_ITEM_TYPES",
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "COMPLETED_STATUS",
    "EXTENSION_COUNT",
    "EXTENSION_FLAG",
    "EXTENSION_PENALTY",
    "EXTENSION_TEXT",
    "FUNCTION_CALL_ITEM",
    "FUNCTION_CALL_WIRE_SHAPE",
    "GPT_5_6_LUNA_CONTRACT_SOURCE",
    "GPT_5_6_LUNA_MODEL",
    "GPT_5_6_LUNA_PROFILE",
    "INCOMPLETE_STATUS",
    "INPUT_TOKENS_DETAILS_WIRE_SHAPE",
    "MAX_EXTENSION_TEXT_CHARACTERS",
    "MAX_OUTPUT_TOKENS",
    "MESSAGE_ITEM",
    "MESSAGE_ITEM_WIRE_SHAPE",
    "MODEL_REQUEST_PROFILES",
    "NO_REASONING",
    "OPENAI_API",
    "OPENAI_BASE_URL",
    "OPENAI_PROVIDER",
    "OPENAI_RESPONSE_CAPTURE",
    "OPENAI_RESPONSE_SERVER_EXTENSIONS",
    "OUTPUT_LIMIT_REASON",
    "OUTPUT_TOKENS_DETAILS_WIRE_SHAPE",
    "PARALLEL_TOOL_CALLS",
    "PENALTY_MAXIMUM",
    "PENALTY_MINIMUM",
    "PROHIBITED_REQUEST_FIELDS",
    "REASONING_EFFORT_DEFAULT",
    "REASONING_EFFORT_NONE",
    "REASONING_EFFORT_VALUES",
    "REJECTED_VARIABLES",
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
    "OpenAIResponsesExchange",
    "RequestProfile",
    "build_client",
    "check_client_endpoint",
    "check_request_profile",
    "check_response_admissible",
    "check_response_model",
    "check_server_extensions",
    "check_server_extensions_agree",
    "check_wire_response",
    "classify_exception",
    "request_profile_for",
    "request_token_bound",
    "response_usage",
]
