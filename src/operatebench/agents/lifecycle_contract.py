"""Provider-neutral Lifecycle request validation and tool-schema authority.

This module owns the model-visible Lifecycle instruction, request integrity
checks, tool schemas, and optional-null normalization. Provider projections
consume this one authority and only wrap it for their wire APIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    ModelBoundaryError,
    tool_schema,
)
from operatebench.agents.outcome_contract import (
    AGENT_TOOL_NAMES,
    FIELD_NON_EMPTY_TEXT,
    FIELD_PAYLOAD_OBJECT,
    FIELD_POSITIVE_MINUTES,
    FIELD_REFERENCE_LIST,
    FIELD_RETRIEVAL_BATCH,
    FIELD_TEXT,
    FIELD_WAIT,
    MAX_RETRIEVAL_REQUESTS,
    MIN_RETRIEVAL_REQUESTS,
    TOOL_DESCRIPTIONS,
    WAIT_FIELDS,
    OutcomeField,
    outcome_fields,
)
from operatebench.agents.transport import ModelRequest, content_digest
from operatebench.core.protocol import MODEL_VISIBLE_FIELDS

#: What happens to a null in a field this contract calls optional, named so a run
#: can record it.
#:
#: Under ``_v3`` the schema no longer *forces* those nulls — an optional field
#: may simply be omitted — but it still declares them acceptable, and a model is
#: free to send one. So the rule stays, and it is the reason an omitted optional
#: and a null optional reach the parser as the same call.
#:
#: Elide a ``null`` in a field this build's own outcome contract calls
#: *optional*, recursively, including inside the ``WAIT`` an ``ASK`` carries.
#: Leave a ``null`` in a required field exactly where it is, so the existing
#: parser classifies it as ``MODEL_MALFORMED_ARGUMENTS`` rather than this module
#: deciding what the model meant. Leave a field this build never declared
#: untouched, so it is still refused as the unknown argument it is.
LIFECYCLE_NULL_ELISION = "elide_null_semantic_optionals_v1"
#: How much of the prose beside a tool call is carried across the seam.
#:
#: Bounded because it is provider-controlled text of otherwise unbounded size
#: held in memory for the length of a turn, and because
#: :attr:`~operatebench.agents.transport.ModelResponse.text` exists so a
#: transport can be *honest* about what it received, not so a decision can be
#: read out of it. Nothing in this package writes it anywhere durable.
MAX_TRANSIENT_TEXT_CHARACTERS = 512

#: The JSON Schema pattern that means "at least one character".
#:
#: A pattern rather than ``minLength`` on purpose. ``pattern`` is a keyword
#: OpenAI documents for strings and the string-length keywords are not, so this
#: is the spelling a model reading the schema is most likely to have been trained
#: to honour. Neither is a constraint anything but the parser enforces — these
#: tools are non-strict and nothing on the provider side is claimed to apply
#: them — so the choice is about which words state the rule most legibly, not
#: about which ones a service would check.
NON_EMPTY_PATTERN = "[\\s\\S]"

#: The smallest deadline any of these fields may name.
MINIMUM_MINUTES = 1

#: The smallest a list a branch requires *stated* may be.
#:
#: One, and expressed with ``minItems`` rather than with a comment, so that a
#: branch requiring ``wake_on`` stated says what "stated" means in the schema's
#: own vocabulary. An empty list and an absent one are the same statement in
#: Core's own reading of a wait — ``if not self.wake_on`` is true of both — so a
#: branch that accepted ``[]`` would be a liveness rule that admits the wait
#: Core refuses. What actually refuses it on the wire is the parser: these tools
#: are non-strict and no provider enforcement is claimed for any keyword here.
MINIMUM_ITEMS = 1
# ------------------------------------------------------ pre-dispatch refusals


class ModelRequestRefused(ModelBoundaryError):
    """A request this transport cannot honestly send, refused before it is sent.

    Every subclass is raised *before* a payload is built, so nothing has left the
    process and nothing has been reserved. They are named rather than collapsed
    into one because they are different mistakes: a request built by another
    build, a request for another model, and a request whose prompt no longer
    matches the digest it is recorded under are three different things to go
    looking for.
    """


class RequestModelMismatch(ModelRequestRefused):
    """The request names a model this exchange does not send to."""


class RequestProtocolMismatch(ModelRequestRefused):
    """The request was built under a model protocol this build does not speak."""


class RequestToolNamesMismatch(ModelRequestRefused):
    """The request offers a tool vocabulary that is not the five Core outcomes."""


class RequestPromptStructureError(ModelRequestRefused):
    """The prompt is not the pinned three-part structure this mapping sends."""


class RequestObservationStructureError(RequestPromptStructureError):
    """The observation is not exactly the model-visible projection.

    A subclass rather than a sibling because an observation this boundary
    cannot send *is* a prompt this boundary cannot send, and the callers that
    already refuse on :class:`RequestPromptStructureError` are refusing the same
    thing for the same reason. It is named separately because the mistake is a
    different one to go looking for: not "the prompt has the wrong keys" but
    "the thing the model would be shown is not the allowlisted projection", and
    the field that shows up here is by definition one nobody meant to disclose.
    """


class PromptDigestMismatch(ModelRequestRefused):
    """The prompt does not hash to the digest the request is recorded under."""


class ObservationDigestMismatch(ModelRequestRefused):
    """The observation does not hash to the digest the request is recorded under.

    Distinct from :class:`PromptDigestMismatch`, and the distinction is the
    whole point. The prompt digest catches an edited prompt; it cannot catch an
    edited prompt whose digest was *recomputed*, which is the only edit worth
    making. ``observation_digest_sha256`` is what playback replays against and
    what binds a recorded decision to the state that provoked it, so a request
    whose observation and observation digest disagree would put a decision on
    the record under a state the model never saw.
    """


class ToolArgumentsUndecodable(ModelBoundaryError):
    """A tool call's argument text could not be read as one unambiguous object.

    Raised by the shared decoder and caught here: at the Lifecycle boundary a
    model that wrote unreadable arguments is a *classification*, not an
    execution fault, so it never leaves this module as an exception.
    """


class UndecodableArguments:
    """What a tool call carries when its arguments could not be decoded.

    Deliberately not a mapping, and deliberately not an empty one. The existing
    parser refuses arguments that are not a ``Mapping`` as
    ``MODEL_MALFORMED_ARGUMENTS`` and names the type it got, so handing it this
    object reuses that classification exactly and puts this build's own class
    name — never the text the model wrote — into the detail. An empty mapping
    would instead be read as a call that stated nothing, which is a different
    thing that a model can legitimately do.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "UndecodableArguments()"


#: The three keys a Lifecycle prompt states, and the only three.
PROMPT_FIELDS: tuple[str, ...] = ("protocol_version", "tools", "observation")

#: Exactly the fields a prompt's observation may state, which is Core's own
#: model-visible allowlist and not a copy of it.
#:
#: Re-exported rather than re-typed on purpose. The allowlist is the disclosure
#: contract — :func:`~operatebench.core.protocol.model_projection` builds from
#: it and this transport refuses anything that is not it — and two hand-written
#: copies of a disclosure contract is one copy too many: the day a field is
#: added to the projection and not to the list here, this boundary starts
#: refusing every request for a reason nobody can find, and the day it is added
#: here and not there it starts sending a field nobody allowed.
OBSERVATION_FIELDS: tuple[str, ...] = MODEL_VISIBLE_FIELDS


class LifecycleRequestContract(Enum):
    """Closed output policies; the historical default remains byte-compatible."""

    LEGACY = "lifecycle_model_request_4096_v1"
    XAI_GROK46 = "lifecycle_xai_responses_model_request_v4"


def lifecycle_output_tokens(
    *, model: str, contract: LifecycleRequestContract = LifecycleRequestContract.LEGACY
) -> int:
    """Resolve an exact contract, never an operator-supplied positive limit."""
    if contract is LifecycleRequestContract.LEGACY:
        return MAX_OUTPUT_TOKENS
    if contract is LifecycleRequestContract.XAI_GROK46 and model == "grok-4.6":
        return 8192
    raise ModelRequestRefused(
        "unsupported Lifecycle output contract; nothing is dispatched"
    )


def check_output_contract(
    request: ModelRequest,
    *,
    model: str,
    contract: LifecycleRequestContract = LifecycleRequestContract.LEGACY,
) -> None:
    """Keep direct payload projection under the same closed output policy."""
    ceiling = lifecycle_output_tokens(model=model, contract=contract)
    if type(request.max_output_tokens) is not int or (
        request.max_output_tokens != ceiling
    ):
        raise ModelRequestRefused(
            "the Lifecycle request's output-token ceiling must be the canonical "
            f"exact integer {ceiling}; settings, controls, request identity "
            "and wire must describe one run. Nothing is dispatched"
        )


def check_model_request(
    request: ModelRequest,
    *,
    model: str,
    contract: LifecycleRequestContract = LifecycleRequestContract.LEGACY,
) -> None:
    """Everything that must be true of a request before it can become a body.

    In this order, and all of it before the payload exists:

    #. **Does it use the canonical output ceiling?** The exact non-boolean
       integer is shared by request identity, provider settings, ledger controls
       and Artifact 8; accepting an equal float would make those disagree about
       the request put on the wire.
    #. **Is it for the model this exchange sends to?** A request for another
       model would be answered by this one, and the answer would be filed under
       the model the request named.
    #. **Was it built under this protocol?** A response parsed under one
       protocol version is not evidence about another.
    #. **Does it offer the five Core outcomes?** The tools on the wire come from
       this build's own contract, so a request claiming a different vocabulary
       would be recorded as having offered something it did not.
    #. **Is the prompt the pinned structure?** Three keys, the protocol version
       agreeing with the request's, and the tool summary this build shows.
    #. **Is the observation exactly the model-visible projection?** The same
       allowlist Core projects from, checked in both directions: a field it does
       not name cannot be sent, and a field it names cannot be dropped. This is
       the one check a digest cannot stand in for, because a digest recomputed
       over a tampered observation agrees with itself perfectly — so an inserted
       ``scenario_id`` would be certified rather than caught, and the whole
       reason the observation exists as a projection is that the scenario's
       identity is the answer's name.
    #. **Does the prompt still hash to its own digest?** Every recorded attempt
       carries a request digest computed over this prompt, so a prompt that does
       not hash to it would make every one of those digests a claim about
       something else.
    #. **Does the observation still hash to its own digest?** Asked last, and
       the reason the previous check is not sufficient: an edit to the
       observation that *also* recomputes the prompt digest is coherent, and
       nothing above it would notice.
       ``observation_digest_sha256`` is the field playback replays against, so a
       request whose observation and observation digest disagree records a
       decision under a state that did not produce it. The digest is taken with
       :func:`~operatebench.agents.transport.content_digest` over
       ``prompt['observation']``, which is the same primitive over the same
       projection that
       :func:`~operatebench.agents.transport.observation_digest` uses — one
       definition, so the two cannot drift.

    Nothing here quotes the prompt. The digests are this build's own hex.
    """
    check_output_contract(request, model=model, contract=contract)
    if request.model != model:
        raise RequestModelMismatch(
            f"this transport sends to {model!r} and the request names a model "
            f"{len(request.model)} character(s) long that is not it. A request is "
            "not carried across models: the answer would be produced by one model "
            "and recorded under another's identity"
        )
    if request.protocol_version != MODEL_PROTOCOL_VERSION:
        raise RequestProtocolMismatch(
            f"this build speaks {MODEL_PROTOCOL_VERSION!r} and the request states a "
            f"protocol version {len(request.protocol_version)} character(s) long "
            "that is not it. A response parsed under one protocol version is not "
            "evidence about another"
        )
    if tuple(request.tool_names) != AGENT_TOOL_NAMES:
        raise RequestToolNamesMismatch(
            f"this transport offers {list(AGENT_TOOL_NAMES)} and the request states a "
            f"vocabulary of {len(request.tool_names)} name(s) that is not it. The "
            "tools on the wire are built from this build's outcome contract, so a "
            "request claiming another vocabulary would be recorded as having "
            "offered something it did not"
        )
    prompt = request.prompt
    if not isinstance(prompt, Mapping) or tuple(sorted(prompt)) != tuple(
        sorted(PROMPT_FIELDS)
    ):
        raise RequestPromptStructureError(
            f"a Lifecycle prompt states exactly {sorted(PROMPT_FIELDS)}; this one "
            f"states {len(prompt) if isinstance(prompt, Mapping) else 0} field(s). "
            "The structure is pinned because the whole of it is what goes on the "
            "wire as one user message"
        )
    if prompt["protocol_version"] != request.protocol_version:
        raise RequestPromptStructureError(
            "the prompt states a different protocol version from the request that "
            "carries it, so there is no single version this turn was asked under"
        )
    if prompt["tools"] != tool_schema():
        raise RequestPromptStructureError(
            "the prompt states a tool summary that is not the one this build "
            "derives from its outcome contract, so what the model would be shown "
            "and what its answer would be checked against are two different things"
        )
    observation = prompt["observation"]
    if not isinstance(observation, Mapping):
        raise RequestObservationStructureError(
            "the prompt's observation is not a mapping, so it is not the "
            "model-visible projection this boundary sends"
        )
    if tuple(sorted(observation)) != tuple(sorted(OBSERVATION_FIELDS)):
        raise RequestObservationStructureError(
            f"a Lifecycle observation states exactly the {len(OBSERVATION_FIELDS)} "
            f"model-visible field(s) this build allows; this one states "
            f"{len(observation)}. The allowlist is checked rather than the digest "
            "because a digest recomputed over an edited observation agrees with "
            "itself: a field outside it is a disclosure whatever it hashes to, and "
            "a field missing from it is not the projection a decision can be "
            "replayed against. No field name is quoted here"
        )
    if content_digest(prompt) != request.prompt_digest_sha256:
        raise PromptDigestMismatch(
            "the prompt does not hash to the digest this request is recorded "
            f"under: it hashes to {content_digest(prompt)} and the request states "
            f"{request.prompt_digest_sha256}. Every attempt this run records "
            "carries a request digest computed over that prompt, so sending this "
            "one would make each of them a claim about something the provider "
            "never saw. Nothing is dispatched"
        )
    if content_digest(observation) != request.observation_digest_sha256:
        raise ObservationDigestMismatch(
            "the observation does not hash to the digest this request is recorded "
            f"under: it hashes to {content_digest(observation)} and the request "
            f"states {request.observation_digest_sha256}. That digest is what a "
            "recorded decision is replayed against, so a request carrying one that "
            "is not its own observation's would bind this turn's answer to a state "
            "the model was never shown. Nothing is dispatched"
        )


# ------------------------------------------------------------- the instructions


#: The fixed part of the instructions, in this build's own words. It states the
#: contract and nothing about the case: everything case-specific is in the one
#: user message, which is what the prompt digest covers.
INSTRUCTIONS_CONTRACT = (
    "You are operating one business operation through a fixed decision protocol.\n"
    "The user message is the whole of what you may rely on: a canonical JSON "
    "object stating the protocol version, the decision tools you may call, and "
    "the current observation.\n"
    f"Answer with exactly one call to one of the {len(AGENT_TOOL_NAMES)} decision tools: "
    f"{', '.join(AGENT_TOOL_NAMES)}. Prose "
    "is not a decision and is not read.\n"
    "Every property the tool lists as required must be present. A property that "
    "is not required may be omitted when it does not apply, or stated as null; "
    "an optional null is read as 'not stated'. A required property stated as "
    "null is refused.\n"
    "A wait must declare what would end it: event types in wake_on, a fallback "
    "deadline in minutes, or both. A wait with neither can never end and is "
    "refused."
)
# ----------------------------------------------------- the non-strict tool surface


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """The same schema, widened to accept ``null`` as well.

    Not how this mapping says "optional" — ``required`` says that now, by naming
    only the fields Core requires — but what it says *in addition*: a model may
    omit an optional field or state it as ``null``, and both are the same
    decision by the time :func:`elide_null_optionals` is done. The nullability is
    kept rather than dropped with strict mode because a model shaped by strict
    answers, or a provider that fills declared properties in, will send those
    nulls whatever ``required`` says, and a schema that called them malformed
    would be picking a fight the parser does not need.

    The keyword constraints beside ``type`` — ``pattern``, ``minimum``,
    ``minItems``, ``items`` — apply per JSON Schema only to the instance types
    they belong to, so a ``null`` satisfies them vacuously and a stated value is
    still held to them.

    A schema stated as a union of branches has no ``type`` of its own, so it is
    widened by *adding a branch* rather than by widening one. Nothing this build
    declares reaches that arm today — the only union it states is the ``WAIT``
    an ``ASK`` carries, which is required — and it is written because the
    alternative is a ``KeyError`` on the day the contract makes one optional.
    """
    declared = schema.get("type")
    if declared is None:
        return {**schema, "anyOf": [*schema["anyOf"], {"type": "null"}]}
    return {**schema, "type": [declared, "null"]}


def _base_schema(field: OutcomeField) -> dict[str, Any]:
    """One field of the outcome contract, as the JSON Schema this mapping sends."""
    if field.kind == FIELD_TEXT:
        return {"type": "string", "description": field.description}
    if field.kind == FIELD_NON_EMPTY_TEXT:
        return {
            "type": "string",
            "pattern": NON_EMPTY_PATTERN,
            "description": field.description,
        }
    if field.kind == FIELD_REFERENCE_LIST:
        return {
            "type": "array",
            "items": {"type": "string", "pattern": NON_EMPTY_PATTERN},
            "description": field.description,
        }
    if field.kind == FIELD_POSITIVE_MINUTES:
        return {
            "type": "integer",
            "minimum": MINIMUM_MINUTES,
            "description": field.description,
        }
    if field.kind == FIELD_PAYLOAD_OBJECT:
        # Open, and stated as an object with no property set at all. Core's
        # contract for an ACT payload is "a JSON object whose keys are non-empty
        # strings" and nothing more — the *domain* decides what an action's
        # arguments are, and it is free to add one — so any property list here
        # would be this projection inventing a contract Core does not have. A
        # closed empty object is the worst of them: it reads as a schema and
        # accepts only ``{}``, which is how a model ends up able to name an
        # action but not state it.
        #
        # ``additionalProperties`` is deliberately absent rather than ``false``.
        # ``false`` closes the payload; there is no third value that means "the
        # domain's own keys, whatever they are".
        return {"type": "object", "description": field.description}
    if field.kind == FIELD_RETRIEVAL_BATCH:
        # A bounded array of named reads. ``tool`` is a non-empty string and not
        # an enumeration, because the finite vocabulary is published *on the
        # observation* — the model reads the catalogue and each action's read
        # requirements in the same payload — and restating it here would put a
        # second copy of one domain's tool names inside a vendor projection that
        # knows no domain. A name outside the catalogue is refused by the
        # environment, by name, and the refusal reaches the next observation.
        #
        # ``arguments`` is open for the reason an ACT payload is: the *domain*
        # decides what a read takes, and a property list here would be this
        # projection inventing a contract the catalogue does not have.
        return {
            "type": "array",
            "minItems": MIN_RETRIEVAL_REQUESTS,
            "maxItems": MAX_RETRIEVAL_REQUESTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tool": {"type": "string", "pattern": NON_EMPTY_PATTERN},
                    "arguments": {"type": ["object", "null"]},
                },
                "required": ["tool"],
            },
            "description": field.description,
        }
    # :data:`FIELD_WAIT`, the last kind. The vocabulary is closed, so there is
    # no seventh to fall through to. A wait is stated as its liveness branches
    # rather than as one permissive object, for the reason
    # :func:`wait_liveness_branches` gives.
    return {"anyOf": wait_liveness_branches(), "description": field.description}


def _object_schema(fields: tuple[OutcomeField, ...]) -> dict[str, Any]:
    """One closed object over a set of contract fields.

    ``required`` is exactly the fields the contract calls required, in the order
    the contract states them — which is
    :func:`~operatebench.agents.outcome_contract.required_field_names`, derived
    here rather than read from it only so that the names and the property set
    come out of one traversal. Under the strict mapping this was *every*
    property, which was a statement about OpenAI rather than about the outcome;
    non-strict, what the schema requires and what the parser requires are one
    thing. A tool that requires nothing states an empty list rather than no list,
    because a recorded request has to distinguish "this build required nothing"
    from "this build forgot".

    ``additionalProperties`` stays ``False``: a shape this build claims to know
    is closed against keys it does not name, and an unknown argument is a
    malformed call the schema should say is one. The optionals keep their
    nullable unions for the reason :func:`_nullable` gives.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field.name: (
                _base_schema(field) if field.required else _nullable(_base_schema(field))
            )
            for field in fields
        },
        "required": [field.name for field in fields if field.required],
    }


def _stated(field: OutcomeField) -> dict[str, Any]:
    """One contract field as the schema of a branch that *requires* it stated.

    "Stated" is the outcome contract's own word for it, read off the field's
    kind rather than off its name: a value that is not null, and — for the kinds
    where emptiness is indistinguishable from absence — not empty either. A
    reference list satisfies that with at least one item, which is exactly what
    Core means by ``if not self.wake_on``; a positive-minutes field satisfies it
    by being present at all, because its ``minimum`` already refuses the values
    Core refuses.
    """
    schema = _base_schema(field)
    if field.kind == FIELD_REFERENCE_LIST:
        return {**schema, "minItems": MINIMUM_ITEMS}
    return schema


def wait_liveness_branches() -> list[dict[str, Any]]:
    """The ways a ``WAIT`` can end, one closed object each.

    Core refuses a wait that declares neither an event to wake on nor a fallback
    deadline: it can never end, and an agent that returns one has not made a
    decision the environment can honour. A ``WAIT`` whose two liveness fields are
    merely optional — omittable, or nullable, or both — states the exact
    opposite, that a wait with neither is well-formed. The parser caught it and
    the schema invited it, which is the worst of both: the model is told the
    unbounded wait is legal and then refused for making it.

    So the rule is stated where the model reads it. One branch per *optional*
    ``WAIT`` field, derived from :data:`~operatebench.agents.outcome_contract.
    WAIT_FIELDS` rather than from a list of names here, because "the wait is
    alive if at least one of its optional fields is stated" is exactly Core's
    rule and stays exactly Core's rule if the contract ever grows a third way to
    end a wait.

    Each branch is a whole ``WAIT``: a closed object over the *same* properties
    as the root, so a stray key is refused by every branch as it is by the root,
    which matters most for the ``WAIT`` an ``ASK`` carries — there the branches
    are the only shape stated. What each branch narrows is its ``required``:
    Core's own required fields plus the one liveness field that branch exists to
    require *stated*, which is non-nullable and non-empty there. The other
    optionals are not required by any branch, so omitting them is legal — and a
    wait that omits or nulls both liveness fields satisfies no branch and is
    refused by the schema, exactly as Core refuses it.
    """
    optional = tuple(field for field in WAIT_FIELDS if not field.required)
    required = [field.name for field in WAIT_FIELDS if field.required]
    return [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                field.name: (
                    _stated(field)
                    if field is alive
                    else (
                        _base_schema(field)
                        if field.required
                        else _nullable(_base_schema(field))
                    )
                )
                for field in WAIT_FIELDS
            },
            "required": [*required, alive.name],
        }
        for alive in optional
    ]


def outcome_tool(name: str) -> dict[str, Any]:
    """One offered tool, as a **non-strict** OpenAI function tool.

    Flat — name, description and parameters beside ``type`` — because that is
    the shape the Responses API takes; the Chat Completions surface takes a
    nested one, and the two reject each other's. Which one this mapping uses is
    part of what :data:`LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION` means.

    ``strict`` is stated as ``False`` rather than left out. The two are the same
    request to the service and are not the same thing to a reader of a recorded
    body: omission reads as an oversight, and the whole reason this mapping is
    non-strict is a decision — Core's open ``ACT`` payload and the root liveness
    union on ``WAIT`` cannot be expressed in OpenAI's documented strict subset,
    and every narrowing that would fit the subset disagrees with Core about a
    concrete value. So what these schemas are is *guidance to the model*: nothing
    here claims a provider enforces them, and under ``strict=False`` the honest
    assumption is that nothing does. The authority on whether an answer is a
    decision is local and fail-closed —
    :func:`~operatebench.agents.model.parse_tool_call` and Core's
    :func:`~operatebench.core.outcomes.outcome_contract_problem` — and it refuses
    everything the schema refuses, with no schema in the path.

    The standalone ``WAIT`` carries its liveness branches as a *sibling* of its
    own closed property set rather than in place of it, so the two are read as a
    conjunction: the root says what a ``WAIT`` is and the branches say which of
    them can end. The form is also purely additive: drop the branches and what is
    left is a whole ``WAIT`` schema, never a weaker statement of one.
    """
    fields = outcome_fields(name)
    parameters = _object_schema(fields)
    if fields is WAIT_FIELDS:
        parameters["anyOf"] = wait_liveness_branches()
    return {
        "type": "function",
        "name": name,
        "description": TOOL_DESCRIPTIONS[name],
        "strict": False,
        "parameters": parameters,
    }


def outcome_tools() -> list[dict[str, Any]]:
    """Every tool this build offers, in the order the contract states them.

    The five outcomes first, then ``retrieve``. Ordered rather than sorted so a
    recorded body reads in contract order and the five business tools sit exactly
    where ``_v3`` put them.
    """
    return [outcome_tool(name) for name in AGENT_TOOL_NAMES]


# ---------------------------------------------------------------- the null elision


def elide_null_optionals(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the nulls a stated-but-unused optional arrives as, and only those.

    Under ``_v2`` strict mode forced every one of them; under ``_v3`` the schema
    lets a model omit an optional instead, and this still runs — because the
    optionals are still declared nullable, a model may still state one as
    ``null``, and the two spellings have to reach the parser as the same call.

    A field this build's outcome contract calls **optional** and that arrived as
    ``null`` is removed, so the parser sees the call a model that used no
    optional field actually made. Everything else is passed through untouched:

    * a **required** field that arrived as ``null`` stays, and the existing
      parser classifies the call as ``MODEL_MALFORMED_ARGUMENTS`` — deciding what
      a missing ``action_type`` meant is not this module's job;
    * a field this build never declared stays, so the parser still refuses it as
      the unknown argument it is;
    * a nested ``WAIT`` is elided under ``WAIT``'s own fields, because the ``ASK``
      that carries one has the same problem one level down.

    Given the tool name rather than inferring it, so a call naming a tool this
    build does not offer is never rewritten on the way to being classified.
    """
    if name not in AGENT_TOOL_NAMES:
        return dict(arguments)
    fields = {field.name: field for field in outcome_fields(name)}
    elided: dict[str, Any] = {}
    for key, value in arguments.items():
        field = fields.get(key)
        if field is None:
            elided[key] = value
            continue
        if value is None and not field.required:
            continue
        if field.kind == FIELD_WAIT and isinstance(value, Mapping):
            elided[key] = elide_null_optionals("wait", value)
            continue
        elided[key] = value
    return elided


__all__ = [
    "AGENT_TOOL_NAMES",
    "INSTRUCTIONS_CONTRACT",
    "LIFECYCLE_NULL_ELISION",
    "MAX_TRANSIENT_TEXT_CHARACTERS",
    "MINIMUM_ITEMS",
    "MINIMUM_MINUTES",
    "NON_EMPTY_PATTERN",
    "OBSERVATION_FIELDS",
    "PROMPT_FIELDS",
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
    "check_model_request",
    "elide_null_optionals",
    "outcome_tool",
    "outcome_tools",
    "wait_liveness_branches",
]
