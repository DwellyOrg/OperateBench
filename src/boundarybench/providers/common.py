"""What every tool-calling provider adapter in this build does the same way.

Three integrations arrived at once — OpenAI, xAI and Mistral — and almost all of
what they do is not provider-specific. They all resolve exactly one credential
from exactly one variable, refuse an environment that would redirect the
request, gate live traffic on an audited configuration identity, project the
same :class:`~boundarybench.adapter.TurnRequest` into the same JSON-Schema tool
surface, run the same bounded retry loop against the same cost guard, and
publish the same attempt evidence. Writing that four times would mean four
places for one rule to live, and the direction that matters is the silent one: a
retry policy that drifted in one integration would produce real ledger rows the
reader refuses.

What is *not* here is anything a provider decides. The request body, the
response contract, the exception taxonomy and the model's stated identity are
each integration's own, because each of those is exactly where the providers
genuinely differ — and pretending otherwise is how an adapter ends up sending a
body that is correct for a different service.

What is no longer *defined* here is the part of that list which is not a
**Boundary** question either. The endpoint, environment, model and payload
checks, the retry policy and its executor, the fault taxonomy, the token counts,
the response-shape checks and the strict decoding of a tool call's JSON
arguments now live under :mod:`operatebench.providers`, the
provider kernel both tracks compose, and this module imports and re-exports
them under the names they have always had — the same objects, so ``isinstance``
and every existing ``except`` clause behave exactly as they did. A retry loop
that lived in one track could not be the one the other track uses, and two of
them is precisely the drift the paragraph above is about.

What remains defined here is what is genuinely the Boundary Track's: the
authorisation gate an operator drives with two ``BOUNDARYBENCH_*`` variables,
the projection of a :class:`~boundarybench.adapter.TurnRequest` into JSON-Schema
tools and a model-visible case, and the reading of one Chat Completions answer
back into an :class:`~boundarybench.adapter.AdapterCall`.

:mod:`boundarybench.providers.anthropic_messages` deliberately does **not** use
this module. It is the integration that priced and executed the runs already on
disk, and every behaviour in it is pinned by a version string that a resume
refuses to cross; refactoring it to share code here would risk changing what it
sends or accepts in exchange for removing duplication that costs nothing. The
duplication is the cheaper of the two, and it is recorded here rather than left
to be discovered.

Nothing in this module opens a socket, reads a credential into a durable place,
or imports a provider SDK.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from boundarybench.adapter import (
    AdapterCall,
    AdapterProtocolError,
    TurnRequest,
    offered_queries,
)
from boundarybench.scaffold import ActionParameter, ActionSchema
from operatebench.providers.config import (
    MODEL_PATTERN,
    ProviderConfigurationError,
    check_endpoint,
    check_environment,
    check_model,
    check_payload_fields,
    normalized_endpoint,
    resolve_api_key,
)
from operatebench.providers.cost import request_input_token_bound
from operatebench.providers.executor import (
    MINIMUM_BACKOFF_MULTIPLIER,
    SDK_MAX_RETRIES,
    Clock,
    ProviderRetryPolicy,
    Sleep,
    TurnExecutor,
    check_retry_policy,
)
from operatebench.providers.faults import (
    Fault,
    classify_http_status,
    fault_detail,
)
from operatebench.providers.response import (
    MAX_RESPONSE_DEPTH,
    check_response_collection,
    check_response_contract,
    check_response_object,
)
from operatebench.providers.toolcalls import decode_tool_arguments
from operatebench.providers.usage import (
    MAX_EXACT_TOKEN_COUNT,
    TokenUsage,
    checked_token_usage,
    is_token_count,
)

# -- what this module still owns ----------------------------------------------
#
# Everything above is imported from :mod:`operatebench.providers` and re-exported
# here under the names it has always had, because none of it is a Boundary
# question: an endpoint check, a credential resolution, a retry policy, a fault
# taxonomy, a token count, a response-shape check and the strict reading of a
# tool call's JSON arguments mean the same thing whichever track built the body.
# They are the same objects, so ``isinstance``, the exception hierarchy and every
# existing ``except`` clause are unchanged, and the three integrations below
# still import them from here.
#
# :func:`decode_tool_arguments` is the one of them that takes the refusal type
# as an argument rather than owning it: the rules for reading model-authored
# argument text are shared, and the exception a refusal is raised as is each
# track's own. Every call site in this track passes
# :class:`~boundarybench.adapter.AdapterProtocolError`, so the refusal an
# operator reads and the class an ``except`` clause catches are the ones this
# track has always raised for a call it could not read.
#
# What is left in this module is what is genuinely this track's: the
# authorisation gate an operator drives with two ``BOUNDARYBENCH_*`` variables,
# the projection of a :class:`~boundarybench.adapter.TurnRequest` into JSON-Schema
# tools and a model-visible case, and the reading of one Chat Completions answer
# back into an :class:`~boundarybench.adapter.AdapterCall`.


#: Where an operator states the configuration identities their own audit
#: approved. Whitespace- or comma-separated; unset means nothing is approved.
#:
#: The same two variables the Anthropic integration uses, deliberately. A
#: configuration identity already covers the provider, the model and every
#: setting, so it cannot be confused across providers; giving each provider its
#: own variable would instead mean an operator authorising a matrix had to
#: export four of them and could silently forget one.
AUDITED_CONFIGURATIONS_VARIABLE = "BOUNDARYBENCH_AUTHORISED_CONFIGURATIONS"

#: Where an operator states which of those identities *this* invocation is for.
#: Required as well as the allowlist: an allowlist alone would authorise a run
#: whose configuration nobody checked against it before the client was built.
LIVE_CONFIGURATION_VARIABLE = "BOUNDARYBENCH_LIVE_CONFIGURATION"

#: The configuration identities this build is pre-authorised to send live
#: provider traffic under. **Empty, and empty by construction** — see the
#: Anthropic integration's constant of the same name for the full argument.
AUDITED_CONFIGURATIONS: frozenset[str] = frozenset()

#: What a configuration identity looks like: the hex digest ``configuration_id``
#: is. A shape check, so a typo is refused as a typo rather than as an
#: unauthorised run.
_CONFIGURATION_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def audited_configurations(
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Every configuration identity live traffic is authorised for, here and now."""
    values = os.environ if environ is None else environ
    declared = values.get(AUDITED_CONFIGURATIONS_VARIABLE, "")
    parsed = {entry for entry in re.split(r"[,\s]+", declared) if entry}
    return frozenset(AUDITED_CONFIGURATIONS | parsed)


def check_configuration_audited(
    configuration_id: str | None,
    *,
    error: type[ProviderConfigurationError],
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Prove this configuration is authorised, before any transport exists.

    Raised before the credential is read, before an SDK client is constructed
    and long before a request is serialised, because an unauthorised live run is
    an operator mistake rather than a run outcome — there must be nothing left
    behind to mistake for evidence.

    Three separate refusals, because they are three different mistakes: the
    build authorises nothing at all (the published default); the invocation did
    not say which configuration it is; and the configuration it named is not one
    the operator approved. The message never quotes the approved set, which
    would turn an error into a menu.
    """
    approved = audited_configurations(environ) if audited is None else frozenset(audited)
    if not approved:
        raise error(
            "this build authorises no configuration for live provider traffic. A "
            "published build pre-authorises nothing: a configuration identity is "
            "a hash of an exact experiment, so shipping one would both publish a "
            "run and let an unaudited tree spend a credential on a previous "
            "audit's authority. Run `preflight` to compute this configuration's "
            f"identity offline, then export {AUDITED_CONFIGURATIONS_VARIABLE} "
            "with the identities your own audit approved"
        )
    identity = (configuration_id or "").strip()
    if not identity:
        raise error(
            "no configuration identity was named for this live run, so there is "
            "nothing to check against the audited set. Export "
            f"{LIVE_CONFIGURATION_VARIABLE} with the identity `preflight` reports "
            "for this exact suite, scaffold, model, settings and limits"
        )
    if not _CONFIGURATION_ID_PATTERN.fullmatch(identity):
        raise error(
            "the configuration identity named for this live run is not a "
            "configuration identity: it is the 64-character hex digest "
            "`preflight` reports as `configuration_id`. A value of another shape "
            "is a typo, and a typo must not be answered by sending a request"
        )
    if identity not in approved:
        raise error(
            f"configuration {identity} is not one this environment has authorised "
            "for live provider traffic. A configuration identity covers the suite, "
            "the scaffold, the provider, the model, every request setting and "
            "every limit, so changing any of them produces an identity your audit "
            "has not seen. Re-run `preflight`, audit what it reports, then add it "
            f"to {AUDITED_CONFIGURATIONS_VARIABLE}"
        )
    return identity


# -- the request projection ---------------------------------------------------


def parameter_schema(parameter: ActionParameter) -> dict[str, Any]:
    """One declared argument, as the JSON Schema every one of these APIs carries.

    ``enum`` is emitted exactly when the projected parameter has one, which for
    this protocol means a fact-acquisition argument whose Cube states the keys
    it can retrieve. It is the wire half of
    :data:`~boundarybench.scaffold.FACT_AFFORDANCE_CONTRACT`: the model is told
    the closed set of values the environment will accept, rather than being
    offered a string and graded against a set it never saw.
    """
    if parameter.type == "array_of_string":
        schema: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string"},
            "description": parameter.description,
        }
    else:
        schema = {"type": "string", "description": parameter.description}
    if parameter.enum is not None:
        schema["enum"] = list(parameter.enum)
    return schema


def input_schema(action: ActionSchema) -> dict[str, Any]:
    """The JSON Schema object one action's arguments must satisfy.

    ``additionalProperties: false`` is not decoration. The runner refuses a call
    carrying an argument the scaffold never declared, so a schema that permitted
    one would turn a stateable constraint into a recorded protocol failure.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            parameter.name: parameter_schema(parameter) for parameter in action.parameters
        },
        "required": [
            parameter.name for parameter in action.parameters if parameter.required
        ],
    }


def flat_tool_schema(action: ActionSchema) -> dict[str, Any]:
    """One projected action, in the shape the OpenAI Responses API takes.

    Flat: the name, description and parameters sit beside ``type``. This is not
    a stylistic difference from :func:`nested_tool_schema` — the two APIs reject
    each other's shape — so which one an integration uses is part of what its
    request mapping version means.
    """
    return {
        "type": "function",
        "name": action.name,
        "description": action.description,
        "parameters": input_schema(action),
    }


def nested_tool_schema(action: ActionSchema) -> dict[str, Any]:
    """One projected action, in the shape a Chat Completions API takes."""
    return {
        "type": "function",
        "function": {
            "name": action.name,
            "description": action.description,
            "parameters": input_schema(action),
        },
    }


def observed_case(request: TurnRequest) -> dict[str, Any]:
    """The case as the agent has actually seen it, and nothing else.

    Every key is a field of :class:`~boundarybench.adapter.TurnRequest`, or —
    like ``available_actions`` and ``resolvable_queries`` — a projection of one.
    The two it omits travel elsewhere in the request — the system prompt is the
    instructions or system message, and the action schemas are the tools — so
    this projection plus those two is the whole of what the model is shown.

    Byte-identical in content to the Anthropic integration's projection of the
    same name, and deliberately so: the case a model is shown is a property of
    the benchmark, not of the vendor it is sent to. If this and that one
    disagreed, two models in the same matrix would be answering different
    questions and the comparison between them would be meaningless.

    ``resolvable_queries`` states, in the message the model reads, the same
    query keys the tool schemas enumerate. Both come from
    :func:`~boundarybench.adapter.offered_queries` over the same action schemas,
    so the two cannot disagree; it is here because an affordance a model has to
    infer from a schema it may or may not have been shown is not an affordance,
    and because the query *names* are operational, unlike their answers. What
    each key resolves to is never published here: that is the answer, and it
    arrives only on the observation the query produces.
    """
    return {
        "available_actions": list(request.available_actions),
        "resolvable_queries": offered_queries(request),
        "observations": [entry.as_dict() for entry in request.observations],
        "transcript": [entry.as_dict() for entry in request.transcript],
        "turns_remaining": request.turns_remaining,
        "scaffold_id": request.scaffold_id,
        "scaffold_version": request.scaffold_version,
    }


def resolve_single_tool_call(
    calls: Sequence[tuple[str, Any]], request: TurnRequest, *, label: str
) -> AdapterCall:
    """The one action a Chat Completions answer carries, or a refusal.

    Shared by the two integrations that speak a Chat Completions surface,
    because every rule it applies is this benchmark's rather than a vendor's:
    one action per turn, a tool the scaffold actually declared, arguments that
    are a JSON object with string keys, and nothing in them this build could not
    write into a ledger row.

    ``calls`` is a sequence of ``(name, raw_arguments)`` pairs the caller has
    already pulled off its own SDK's objects. ``raw_arguments`` may be the JSON
    *string* these APIs normally send or an already-decoded mapping — one SDK in
    this build types that field as either — and both are handled by
    :func:`decode_tool_arguments`, which is where the strictness about the text
    itself lives.

    Every message raised is a fixed sentence plus values this build itself
    chose. Nothing that arrived in the response is quoted: not a tool name, not
    an argument, not a request id.
    """
    if len(calls) != 1:
        raise AdapterProtocolError(
            f"the answer carries {len(calls)} tool call(s); this scaffold requires "
            "exactly one action per turn, and neither prose alone nor two actions "
            "at once names the single thing the agent did"
        )
    name, raw = calls[0]
    if name not in {action.name for action in request.actions}:
        raise AdapterProtocolError(
            "the answer calls a tool the scaffold does not declare; the scaffold "
            f"offers {sorted(action.name for action in request.actions)}. The name "
            "the model produced is not quoted here: it is model-authored text and "
            "a durable failure row records only stable, closed-set detail"
        )
    return AdapterCall(
        action=name,
        arguments=decode_tool_arguments(raw, label=label, error=AdapterProtocolError),
    )


__all__ = [
    "AUDITED_CONFIGURATIONS",
    "AUDITED_CONFIGURATIONS_VARIABLE",
    "LIVE_CONFIGURATION_VARIABLE",
    "MAX_EXACT_TOKEN_COUNT",
    "MAX_RESPONSE_DEPTH",
    "MINIMUM_BACKOFF_MULTIPLIER",
    "MODEL_PATTERN",
    "SDK_MAX_RETRIES",
    "Clock",
    "Fault",
    "ProviderConfigurationError",
    "ProviderRetryPolicy",
    "Sleep",
    "TokenUsage",
    "TurnExecutor",
    "audited_configurations",
    "check_configuration_audited",
    "check_endpoint",
    "check_environment",
    "check_model",
    "check_payload_fields",
    "check_response_collection",
    "check_response_contract",
    "check_response_object",
    "check_retry_policy",
    "checked_token_usage",
    "classify_http_status",
    "decode_tool_arguments",
    "fault_detail",
    "flat_tool_schema",
    "input_schema",
    "is_token_count",
    "nested_tool_schema",
    "normalized_endpoint",
    "observed_case",
    "parameter_schema",
    "request_input_token_bound",
    "resolve_api_key",
    "resolve_single_tool_call",
]
