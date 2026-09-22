"""The Anthropic Messages adapter.

This is the first adapter in the build that talks to a real service. What it is
*not* is a benchmark result: it is the integration a credentialed operator needs
before any model run can happen. This build ships no provider result, and it
pre-authorises no run: :data:`AUDITED_CONFIGURATIONS` is empty, so a live run is
refused until an operator authorises the exact configuration they audited.

Three properties are the reason this lives in its own module rather than in
:mod:`boundarybench.adapter`:

* **The provider boundary remains narrow.** The adapter is handed a
  :class:`~boundarybench.adapter.TurnRequest` and nothing else, so the system
  prompt, the tool schemas and the model-visible history are all projections of
  what the environment actually revealed. The compiled ``Variant`` — expected
  disposition, expected reason, required evidence, content digest — is not
  reachable from here, so it cannot be serialised into a request by accident.
* **Every result-affecting request setting is pinned, per model.** Model, SDK,
  SDK version, output ceiling, the request profile — which sampling and thinking
  fields are sent, which are omitted, and with what values — tool-choice
  behaviour and the retry policy are all reported through
  :attr:`AnthropicMessagesAdapter.settings`, which the run manifest hashes into
  run identity. Changing any of them is a different run and a resume refuses it.
  The request shape is a property of the pinned model (:class:`RequestProfile`),
  because a body one model accepts is refused by another, and a body that does
  not match the settings is not dispatched at all.
* **The credential never becomes data.** It is read from ``ANTHROPIC_API_KEY``
  once, handed straight to the SDK client, and never stored on this object,
  written to settings, put in an exception message or rendered by ``repr``.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import anthropic
from anthropic.types import Message, ToolUseBlock

from boundarybench.adapter import (
    ATTEMPT_OUTCOME_FAULT,
    ATTEMPT_OUTCOME_RESPONSE,
    ATTEMPT_OUTCOME_UNCLASSIFIED,
    ATTEMPT_SETTLEMENT_FORFEITED,
    ATTEMPT_SETTLEMENT_MEASURED,
    PROVIDER_FAULT_AUTHENTICATION,
    PROVIDER_FAULT_CONFIGURATION,
    PROVIDER_FAULT_NETWORK_ERROR,
    PROVIDER_FAULT_RATE_LIMITED,
    PROVIDER_FAULT_REQUEST_REJECTED,
    PROVIDER_FAULT_RESPONSE_INVALID,
    PROVIDER_FAULT_SERVER_ERROR,
    PROVIDER_FAULT_TIMEOUT,
    RETRYABLE_PROVIDER_FAULTS,
    TURN_END_BACKOFF_UNAFFORDABLE,
    TURN_END_COST_CAP_EXHAUSTED,
    TURN_END_DEADLINE_EXCEEDED,
    TURN_END_FAULT_NOT_RETRYABLE,
    TURN_END_RESPONSE,
    TURN_END_RETRIES_EXHAUSTED,
    TURN_END_UNCLASSIFIED,
    AdapterCall,
    AdapterError,
    AdapterIdentity,
    AdapterOutputLimitError,
    AdapterProtocolError,
    AdapterProviderError,
    AdapterUsage,
    ProviderAttempt,
    ProviderTelemetry,
    SingleFlight,
    TurnDeadline,
    TurnRequest,
    offered_queries,
)
from boundarybench.budget import (
    CostCapExceededError,
    CostReservationBreachedError,
    RunCostGuard,
    conservative_input_token_bound,
    usd_text,
)
from boundarybench.jsonsafe import JsonSafetyError, canonical_json_text, ensure_json_safe
from boundarybench.queries import QUERY_RESOLUTION_CONTRACT
from boundarybench.scaffold import (
    ACTION_SURFACE_CONTRACT,
    FACT_AFFORDANCE_CONTRACT,
    ActionParameter,
    ActionSchema,
)

#: The service under test, as recorded in run identity.
ANTHROPIC_PROVIDER = "anthropic"
#: The code that talks to it. Distinct from the provider: two implementations of
#: the same API are two different runs.
ANTHROPIC_IMPLEMENTATION = "anthropic_messages"
#: This integration's own version. Moves when anything it sends or accepts moves.
#:
#: ``dev.2`` moved what it *accepts*: a response whose stated model is not the
#: pinned one is refused (:func:`check_response_model`), and a content block
#: whose declared type this build does not know is refused rather than read as
#: commentary. Both are answers ``dev.1`` accepted, so a run under each is a
#: different experiment and their identities have to say so.
#:
#: ``dev.3`` moved what it *measures*: the measurement is cleared at the top of
#: every ``next_call`` rather than inside the send loop, and a turn that ends in
#: a protocol failure still reports what the response that caused it consumed.
#: A run under each records different usage for the same provider traffic.
#:
#: ``0.2.0`` moved both again. It reports structured provider-attempt evidence
#: for every turn, and it classifies an answer truncated at the output ceiling
#: as its own output-limit failure rather than as a malformed protocol answer.
#: An episode that was rate limited twice and then answered, and an episode cut
#: off at the ceiling, are both recorded differently under it.
#:
#: ``0.3.0`` moved what it *classifies*: HTTP 404 — the provider saying the
#: model or resource this run pins does not exist for this account — is reported
#: as the configuration fault rather than as a request rejection. Under it a
#: run stops on a missing model and continues past a refused payload; under
#: ``0.2.0`` both stopped it. The request the adapter sends is unchanged.
#:
#: ``0.4.0`` moved what it *refuses to do*: under a cost guard it asks for
#: authorisation before every attempt and does not send a request the run's
#: remaining budget cannot cover, and it prices the usage a response reported at
#: the run's pinned pricing policy instead of leaving ``cost_usd`` null. A run
#: under it can stop with episodes unattempted that a ``0.3.0`` run would have
#: sent, and its rows carry a measured cost where the older ones carry null.
#:
#: ``0.5.0`` moved what it *records*: every turn now names the exit its retry
#: loop took, from the contract's closed
#: :data:`~boundarybench.adapter.TURN_TERMINAL_REASONS` set, so a turn that
#: stopped early because the clock, a cancellation or the run's remaining budget
#: said so is distinguishable from one whose later attempts are simply missing.
#: A ``0.4.0`` row cannot make that distinction, and the requests this version
#: sends are otherwise unchanged.
#:
#: ``0.6.0`` moved what it *sends*, and it is the first version to send
#: different things to different models. Up to ``0.5.0`` one request shape went
#: to every model: ``temperature: 0.0`` and no ``thinking`` field. Anthropic's
#: Sonnet 5 contract rejects a non-default sampling parameter outright and reads
#: an absent ``thinking`` field as *adaptive thinking on*, so that one shape was
#: not a neutral default — it was a request one pinned model refuses and another
#: accepts. Under ``0.6.0`` the shape is a property of the model
#: (:class:`RequestProfile`), the settings state exactly which fields are sent
#: and which are omitted, and a payload that does not match the settings is not
#: dispatched. Every run under ``0.5.0`` asked for something different from what
#: a ``0.6.0`` run asks the same model for.
#: ``0.7.0`` changes what the request *offers*. Every version before it sent the
#: scaffold's whole action list as the tool schemas whatever the case allowed,
#: so a Cube excluding ``call_tool`` still declared that tool and the runtime
#: then refused any use of it — an action surface the model was shown and the
#: environment would not honour. Under ``0.7.0`` the tools are the Cube's
#: ``allowed_actions``, exactly.
#:
#: ``0.8.0`` changes what the request *permits inside* those tools. Every
#: version before it declared a fact-acquisition action's ``fact`` argument as an
#: unconstrained string, while the environment scored the answer against the
#: Cube's own retrievable keys — so a plausible key the case never had was
#: refused from a namespace the model had never received. Under ``0.8.0`` that
#: argument carries the exact enum of keys the Cube can deliver, both in the tool
#: schema and in the model-visible ``retrievable_facts`` of the observed case.
ANTHROPIC_ADAPTER_VERSION = "0.8.0"

#: The API surface this adapter speaks.
ANTHROPIC_API = "messages"

#: How a :class:`~boundarybench.adapter.TurnRequest` becomes a request body.
#: Pinned in settings because a different mapping is a different experiment even
#: with an identical scaffold.
#:
#: ``v2`` was the model-specific one: the body's field *set* is decided by the
#: pinned model's :class:`RequestProfile` rather than being the same for every
#: model.
#:
#: ``v3`` changes the projection itself. ``tools`` and the ``available_actions``
#: inside the user message are now the same list — the case's own
#: ``allowed_actions`` — where ``v1`` and ``v2`` sent the scaffold's whole action
#: list as ``tools`` and the case's list as ``available_actions``. Two bodies
#: that offer different actions are two different experiments, so the mapping
#: version moves and every run under it has a new identity.
#:
#: ``v4`` changes the projection again, in the same direction. A tool's
#: fact-acquisition argument carries the Cube's deterministic enum of retrievable
#: fact keys, and the user message carries the same keys as
#: ``retrievable_facts``. Two bodies that permit different arguments are two
#: different experiments for the same reason two that offer different actions
#: are, and the enum also changes the body's size, so what a turn reserves under
#: this mapping is not what it reserved under ``v3``.
REQUEST_MAPPING_VERSION = "turn_request_json_v5"

#: The output ceiling, pinned rather than exposed. A run whose answers were
#: truncated at a different limit is not comparable with one that was not.
MAX_OUTPUT_TOKENS = 1024

#: Pinned, and pinned to zero: this benchmark measures whether a decision is
#: made for the right reason, and sampling temperature is a confound the
#: standardized scaffold has no reason to vary.
#:
#: Pinned is not the same as *sent*. A model whose documented contract refuses a
#: non-default sampling parameter is asked without one — see
#: :class:`RequestProfile` — and this constant is then the value that profile
#: records as deliberately omitted rather than the value it puts on the wire.
TEMPERATURE = 0.0

#: The exact thinking block a profile sends when the documented way to run a
#: model *without* thinking is to say so rather than to say nothing.
#:
#: Verbatim from Anthropic's Sonnet 5 documentation. Omitting the field is not
#: the same request: for a model whose adaptive thinking is on by default,
#: silence selects a feature a run may have been configured to exclude.
DISABLED_THINKING: Mapping[str, Any] = {"type": "disabled"}

#: The response contract, enforced on both sides. ``any`` requires the model to
#: answer with a tool call rather than prose, and parallel tool use is disabled
#: because one turn is one action.
TOOL_CHOICE: Mapping[str, Any] = {"type": "any", "disable_parallel_tool_use": True}

#: The fields whose presence or absence is a *profile's* decision, and which
#: therefore have to be accounted for by every profile: each one is either in
#: the body this build sends or named as omitted from it.
#:
#: ``top_p`` and ``top_k`` are in the set although no profile this build ships
#: sends either. They are the other two sampling parameters the Messages API
#: accepts, so "this build does not send them" is a claim worth recording in run
#: identity rather than a silence a reader has to infer.
PROHIBITED_REQUEST_FIELDS: tuple[str, ...] = ("temperature", "thinking", "top_k", "top_p")

#: The fields the mapping sends for every model, whatever its profile.
_COMMON_REQUEST_FIELDS: tuple[str, ...] = (
    "max_tokens",
    "messages",
    "model",
    "system",
    "tool_choice",
    "tools",
)


@dataclass(frozen=True)
class RequestProfile:
    """One model's frozen request shape: what is sent, and what is left out.

    A profile exists because the request that is correct for one model is
    refused by another. Sonnet 5 rejects a non-default ``temperature`` with HTTP
    400 and treats an absent ``thinking`` field as adaptive thinking *enabled*;
    Haiku 4.5 accepts ``temperature: 0.0`` and has no thinking mode to disable.
    Each pinned model therefore receives its shipped, tested model-specific
    profile rather than a shared shape.

    Immutable, and identified by :attr:`profile_id`, which goes into adapter
    settings and so into ``configuration_id``. Two runs that asked for different
    things are two configurations, and a resume across them is refused.

    ``temperature`` is ``None`` for "not sent", not for "sent as null" — the
    field is absent from the body entirely. ``thinking`` is the same.
    """

    profile_id: str
    temperature: float | None
    thinking: Mapping[str, Any] | None

    def sent_fields(self) -> tuple[str, ...]:
        """Exactly the top-level keys of the body this profile produces."""
        fields = list(_COMMON_REQUEST_FIELDS)
        if self.temperature is not None:
            fields.append("temperature")
        if self.thinking is not None:
            fields.append("thinking")
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
            "thinking": None if self.thinking is None else dict(self.thinking),
        }


#: The two models this build holds a model-specific request shape for, named once.
SONNET_5_MODEL = "claude-sonnet-5"
HAIKU_4_5_MODEL = "claude-haiku-4-5-20251001"

#: Sonnet 5: no sampling parameters at all, and thinking explicitly disabled.
#:
#: Both halves are Anthropic's documented contract for this model, and both are
#: load-bearing: the sampling omission is what stops the request being rejected,
#: and the explicit disable is what stops the model reasoning adaptively in a
#: run that was configured to exclude it.
SONNET_5_PROFILE = RequestProfile(
    profile_id="sonnet5_sampling_omitted_thinking_disabled_v1",
    temperature=None,
    thinking=DISABLED_THINKING,
)

#: Haiku 4.5: the sampling-pinned profile, and no thinking field.
#:
#: Identical in content to :data:`BASELINE_PROFILE` and deliberately not the
#: same object: this one is a claim that the shape was checked against *this
#: model's* published contract, and that claim is what a run identity records.
HAIKU_4_5_PROFILE = RequestProfile(
    profile_id="haiku45_temperature_zero_thinking_omitted_v1",
    temperature=TEMPERATURE,
    thinking=None,
)

#: What a model this build ships no model-specific profile for is sent.
#:
#: :func:`check_model` is a shape check rather than an allowlist — a build that
#: refused every model released after it is not what a benchmark needs — so an
#: unencoded model still has to be sent something, and this is it. Its id says
#: which it is, so a run under it is never mistaken for a run under a profile
#: that was read off a model's own documentation. An operator pinning a model
#: this build does not name is the one who has to check its contract.
BASELINE_PROFILE = RequestProfile(
    profile_id="baseline_temperature_zero_thinking_omitted_v1",
    temperature=TEMPERATURE,
    thinking=None,
)

#: Model identifier to frozen request shape. Matched whole, like a price: an
#: alias is not the model it currently points at.
MODEL_REQUEST_PROFILES: Mapping[str, RequestProfile] = MappingProxyType(
    {
        SONNET_5_MODEL: SONNET_5_PROFILE,
        HAIKU_4_5_MODEL: HAIKU_4_5_PROFILE,
    }
)


def request_profile_for(model: str) -> RequestProfile:
    """The one request shape this build sends to this model."""
    return MODEL_REQUEST_PROFILES.get(model, BASELINE_PROFILE)


def check_request_profile(
    profile: RequestProfile | None, *, model: str
) -> RequestProfile:
    """Prove a profile is this model's, before it can shape a request.

    Refused rather than accommodated, and refused here — before a client is
    built, a request is serialised or a run directory exists. A profile that is
    not the pinned model's is a request shaped for one model sent to another,
    under settings that describe the shape rather than the model it is wrong for.
    """
    expected = request_profile_for(model)
    if profile is None or profile == expected:
        return expected
    raise AnthropicConfigurationError(
        f"request profile {profile.profile_id!r} is not the profile this build "
        f"sends to model {model!r}, which is {expected.profile_id!r}. A model's "
        "request shape is frozen per model — sampling parameters and the thinking "
        "field are accepted by one model and refused by another — so a profile "
        "cannot be carried across models"
    )


#: The only ``stop_reason`` a turn may end on. Anything else means the answer
#: was cut short or ended for a reason this scaffold does not model, and the
#: call it carries cannot be trusted to be the one the model meant to make.
REQUIRED_STOP_REASON = "tool_use"

#: The one other stop reason this build classifies rather than redacts.
#:
#: It is singled out because it names a *cause this run controls*: the answer
#: ran into :data:`MAX_OUTPUT_TOKENS`, which this build pins. Everything else —
#: a refusal, a stop sequence, a reason introduced after this build shipped — is
#: provider-authored and stays redacted, because guessing at what an unknown
#: reason meant is how a taxonomy starts lying.
OUTPUT_LIMIT_STOP_REASON = "max_tokens"


class AnthropicConfigurationError(AdapterError):
    """The environment cannot produce a usable, unambiguous provider client.

    Raised before a run directory, a manifest or a ledger row exists, because a
    missing credential is an operator mistake rather than a recorded run
    outcome. Its message never carries the credential, or any part of one.
    """


@dataclass(frozen=True)
class AnthropicRetryPolicy:
    """The bounded, deterministic retry policy this adapter owns.

    Owned here rather than delegated to the SDK. The SDK's own retry loop is
    disabled (:data:`SDK_MAX_RETRIES`) because a hidden retry spends the
    episode's wall-clock budget without the runner or the deadline knowing, and
    a benchmark that cannot say how many provider calls an episode made cannot
    report latency honestly.

    Not to be confused with the run manifest's ``retry_policy``, which is about
    *episodes*: a failed episode is still never re-run. This is one turn's
    transport-level policy, and it is inside adapter settings, so it is inside
    run identity all the same.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "backoff_multiplier": self.backoff_multiplier,
        }


#: Zero. The adapter owns its retries; see :class:`AnthropicRetryPolicy`.
SDK_MAX_RETRIES = 0

#: The one endpoint this build talks to. Custom endpoints are not supported:
#: a base URL is where the credential is sent, so making it configurable would
#: add a redirection this build cannot audit for no measurement benefit.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

#: The credential, and the only place it may come from.
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"

#: The configuration identities this build is pre-authorised to send live
#: provider traffic under. **Empty, and empty by construction.**
#:
#: Default-deny rather than default-allow, for two reasons that both matter. A
#: published build that named an identity would be publishing a run: a
#: ``configuration_id`` is a hash of an exact experiment, so the value is a claim
#: that this suite, scaffold, model, settings and limits were executed against a
#: paid endpoint. A seeded allowlist would also grant authority not established
#: for the current tree, which is precisely the authorisation this constant
#: exists to withhold.
#:
#: An operator authorises their own configuration, per environment, by naming it
#: in :data:`AUDITED_CONFIGURATIONS_VARIABLE`. ``preflight`` computes and prints
#: the identity without opening a socket, so the value is discoverable offline.
AUDITED_CONFIGURATIONS: frozenset[str] = frozenset()

#: Where an operator states the configuration identities their own audit
#: approved. Whitespace- or comma-separated; unset means nothing is approved.
AUDITED_CONFIGURATIONS_VARIABLE = "BOUNDARYBENCH_AUTHORISED_CONFIGURATIONS"

#: Where an operator states which of those identities *this* invocation is for.
#: Required as well as the allowlist: an allowlist alone would authorise a run
#: whose configuration nobody checked against it before the client was built.
LIVE_CONFIGURATION_VARIABLE = "BOUNDARYBENCH_LIVE_CONFIGURATION"

#: What a configuration identity looks like: the hex digest ``configuration_id``
#: is. A shape check, so a typo is refused as a typo rather than as an
#: unauthorised run.
_CONFIGURATION_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")

#: Environment variables the SDK honours that would silently change where a
#: request goes or what it carries. Refused rather than ignored: an operator who
#: set one meant it to take effect, and a run that quietly disregarded it would
#: record a provenance nobody agreed to.
REJECTED_VARIABLES: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
)


#: What a model identifier may look like. Deliberately a shape check and not a
#: list: a build that shipped an allowlist of model names would refuse every
#: model released after it, which is the opposite of what a benchmark needs.
#:
#: What this cannot check is whether the identifier is *pinned*. ``--model`` is
#: required precisely so nothing here silently picks an alias, but an operator
#: who passes a moving alias gets a run whose recorded model name does not name
#: one model. Pinning a dated snapshot id is the operator's decision, and the
#: documentation says so rather than this pattern pretending to enforce it.
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def check_model(model: str) -> str:
    """Prove the model identifier is one, before it becomes run identity."""
    if not isinstance(model, str) or not _MODEL_PATTERN.fullmatch(model):
        raise AnthropicConfigurationError(
            "an Anthropic run needs an explicit model identifier of letters, "
            "digits, dots, dashes and underscores. Pin the dated snapshot id of "
            "the model under test: an alias moves, and a run whose recorded model "
            "names more than one model is not comparable with anything"
        )
    return model


def resolve_api_key(environ: Mapping[str, str] | None = None) -> str:
    """The credential, from the one variable it may come from.

    Read here and passed explicitly to the client, rather than left to the SDK's
    own resolution order. The SDK will otherwise fall back to an auth token, an
    on-disk profile or a workload-identity configuration, any of which would let
    a run authenticate as something the operator did not choose and the manifest
    cannot record. One variable, or nothing.

    The value is returned and never stored, logged or interpolated into a
    message — including this module's own errors.
    """
    values = os.environ if environ is None else environ
    key = values.get(API_KEY_VARIABLE, "")
    if not key.strip():
        raise AnthropicConfigurationError(
            f"{API_KEY_VARIABLE} is not set, so there is no credential to run "
            "against Anthropic with. Export it in the environment; it is never "
            "read from a command-line setting, and never written to a run "
            "manifest, a ledger row or this build's output"
        )
    return key


def check_environment(environ: Mapping[str, str] | None = None) -> None:
    """Refuse an environment that would silently redirect or rewrite requests.

    The SDK honours these variables. Ignoring them would be worse than either
    obeying or refusing: the operator who set one would get a run that looks
    like it went where they asked and did not, and the credential would be sent
    somewhere the manifest does not record.
    """
    values = os.environ if environ is None else environ
    for variable in REJECTED_VARIABLES:
        if values.get(variable, "").strip():
            raise AnthropicConfigurationError(
                f"{variable} is set. This build talks to {ANTHROPIC_BASE_URL} and "
                "only there, because the endpoint is where the credential is sent "
                "and a run manifest that recorded one endpoint while the request "
                "went to another would be false. Unset it, or use a build that "
                "pins the endpoint you want"
            )


def audited_configurations(
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Every configuration identity live traffic is authorised for, here and now.

    The build's own set — :data:`AUDITED_CONFIGURATIONS`, which is empty —
    unioned with whatever the operator declared in the environment. Returned as
    a value rather than consulted in place so a caller can be handed a synthetic
    set for a test without the module's default ever being mutated.
    """
    values = os.environ if environ is None else environ
    declared = values.get(AUDITED_CONFIGURATIONS_VARIABLE, "")
    parsed = {entry for entry in re.split(r"[,\s]+", declared) if entry}
    return frozenset(AUDITED_CONFIGURATIONS | parsed)


def check_configuration_audited(
    configuration_id: str | None,
    *,
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
        raise AnthropicConfigurationError(
            "this build authorises no configuration for live provider traffic. A "
            "published build pre-authorises nothing: a configuration identity is "
            "a hash of an exact experiment, so shipping one would both publish a "
            "run and let an unaudited tree spend a credential on a previous "
            f"audit's authority. Run `preflight` to compute this configuration's "
            f"identity offline, then export {AUDITED_CONFIGURATIONS_VARIABLE} "
            "with the identities your own audit approved"
        )
    identity = (configuration_id or "").strip()
    if not identity:
        raise AnthropicConfigurationError(
            "no configuration identity was named for this live run, so there is "
            "nothing to check against the audited set. Export "
            f"{LIVE_CONFIGURATION_VARIABLE} with the identity `preflight` reports "
            "for this exact suite, scaffold, model, settings and limits"
        )
    if not _CONFIGURATION_ID_PATTERN.fullmatch(identity):
        raise AnthropicConfigurationError(
            "the configuration identity named for this live run is not a "
            "configuration identity: it is the 64-character hex digest "
            "`preflight` reports as `configuration_id`. A value of another shape "
            "is a typo, and a typo must not be answered by sending a request"
        )
    if identity not in approved:
        raise AnthropicConfigurationError(
            f"configuration {identity} is not one this environment has authorised "
            "for live provider traffic. A configuration identity covers the suite, "
            "the scaffold, the model, every request setting and every limit, so "
            "changing any of them produces an identity your audit has not seen. "
            f"Re-run `preflight`, audit what it reports, then add it to "
            f"{AUDITED_CONFIGURATIONS_VARIABLE}"
        )
    return identity


def build_client(*, api_key: str) -> anthropic.Anthropic:
    """The SDK client this build talks through, with its retry loop disabled.

    ``max_retries=0`` is the load-bearing argument. The SDK retries some
    failures itself by default, which would spend an episode's wall-clock budget
    on calls the runner never saw, make the recorded latency of a turn describe
    an unknown number of requests, and hide a rate limit the run should have
    recorded. The policy lives in :class:`AnthropicRetryPolicy` instead, where
    it is bounded, deterministic and inside run identity.
    """
    return anthropic.Anthropic(
        api_key=api_key,
        base_url=ANTHROPIC_BASE_URL,
        max_retries=SDK_MAX_RETRIES,
    )


def build_anthropic_adapter(
    *,
    model: str,
    configuration_id: str | None = None,
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    retry: AnthropicRetryPolicy | None = None,
    cost_guard: RunCostGuard | None = None,
) -> AnthropicMessagesAdapter:
    """Everything a credentialed run needs, or a clean refusal.

    Called before a run directory, a manifest or a lock exists, so an
    unconfigured environment costs the operator an error message rather than a
    half-created run whose evidence claims a provider that was never contacted.

    This is the only function in the build that turns a credential into a live
    client, so it is where the authorised-configuration gate belongs. The order is
    load-bearing: the configuration is refused before ``ANTHROPIC_API_KEY`` is
    read, so an unauthorised invocation never handles the credential at all.
    """
    checked = check_model(model)
    check_configuration_audited(configuration_id, audited=audited, environ=environ)
    check_environment(environ)
    return AnthropicMessagesAdapter(
        model=checked,
        client=build_client(api_key=resolve_api_key(environ)),
        retry=retry,
        cost_guard=cost_guard,
    )


def anthropic_identity(model: str) -> AdapterIdentity:
    """Who answered: the service, the pinned model, and this integration."""
    return AdapterIdentity(
        provider=ANTHROPIC_PROVIDER,
        model=model,
        implementation=ANTHROPIC_IMPLEMENTATION,
        version=ANTHROPIC_ADAPTER_VERSION,
    )


def profile_settings(
    profile: RequestProfile, retry: AnthropicRetryPolicy
) -> dict[str, Any]:
    """Every result-affecting request setting, and no credential.

    This mapping is hashed into run identity, so it is the answer to "what
    exactly was asked of the provider". The SDK version is in it deliberately:
    the SDK decides serialisation, retry and error mapping, so an upgrade is a
    change to the experiment and a resume must refuse it.

    It is also an exact description of the request body's field set — which
    fields are sent, which are omitted, and with what values — and
    :func:`check_payload_profile` refuses to dispatch a body that does not match
    it. The settings must describe the request the provider receives.

    ``query_resolution`` says how a request for information was *answered*: which
    registry contract decided the channel, the outcome and whether any value came
    back. Two runs whose Cubes resolve the same query key to different outcomes
    are two experiments, so the contract is in the hash.

    ``action_surface`` says what the body *offered*, which is the other half of
    the same question. Two runs that put
    different tools in front of the same model are two configurations, so the
    contract is in the hash rather than in a comment. ``fact_affordance`` says
    what those tools *permitted as arguments*, which is the same question one
    level down and is also recorded in identity.
    """
    return {
        "action_surface": ACTION_SURFACE_CONTRACT,
        "api": ANTHROPIC_API,
        "base_url": ANTHROPIC_BASE_URL,
        "fact_affordance": FACT_AFFORDANCE_CONTRACT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "query_resolution": QUERY_RESOLUTION_CONTRACT,
        "request_mapping": REQUEST_MAPPING_VERSION,
        **profile.as_settings(),
        "response_contract": "exactly_one_tool_call",
        "retry": retry.as_dict(),
        "sdk": "anthropic",
        "sdk_max_retries": SDK_MAX_RETRIES,
        "sdk_version": anthropic.__version__,
        "tool_choice": dict(TOOL_CHOICE),
    }


def anthropic_settings(retry: AnthropicRetryPolicy, *, model: str) -> dict[str, Any]:
    """The settings a run of this model records, under this model's profile.

    The model is required rather than defaulted. Settings that could be stated
    without naming a model would once again be settings that describe a request
    shape instead of a request, which is the defect this contract closes.
    """
    return profile_settings(request_profile_for(model), retry)


# -- the request -------------------------------------------------------------


def _parameter_schema(parameter: ActionParameter) -> dict[str, Any]:
    """One declared argument, as JSON Schema the Messages API will carry.

    ``enum`` is emitted exactly when the projected parameter has one, which for
    this protocol means a fact-acquisition argument whose Cube states the keys it
    can retrieve. It is the wire half of
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


def tool_schema(action: ActionSchema) -> dict[str, Any]:
    """One projected action, as a Messages API tool.

    ``additionalProperties: false`` is not decoration. The runner refuses a call
    carrying an argument the scaffold never declared, so a schema that permitted
    one would turn a stateable constraint into a recorded protocol failure.
    """
    return {
        "name": action.name,
        "description": action.description,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                parameter.name: _parameter_schema(parameter)
                for parameter in action.parameters
            },
            "required": [
                parameter.name for parameter in action.parameters if parameter.required
            ],
        },
    }


def observed_case(request: TurnRequest) -> dict[str, Any]:
    """The case as the agent has actually seen it, and nothing else.

    Every key is a field of :class:`~boundarybench.adapter.TurnRequest`, or —
    like ``available_actions`` and ``retrievable_facts`` — a projection of one.
    The two it omits travel elsewhere in the request — the system prompt is the
    system parameter and the action schemas are the tools — so this projection
    plus those two is the whole of what the model is shown.

    ``resolvable_queries`` states, in the message the model reads, the same query
    keys the tool schemas enumerate. Both come from
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


def build_messages_request(
    request: TurnRequest, *, model: str, profile: RequestProfile | None = None
) -> dict[str, Any]:
    """The exact request body one turn sends, under this model's profile.

    Built as a plain mapping and handed to the SDK unchanged, so a test can
    compare what the transport saw against what this function produces and know
    the two are the same object rather than two hopefully-equivalent ones.

    History is one deterministic user message rather than a replayed
    ``tool_use``/``tool_result`` exchange. The environment's transcript is the
    only record of what happened, and it carries no provider block ids to replay;
    inventing them would make the request depend on state outside the
    :class:`~boundarybench.adapter.TurnRequest`. The convention is pinned as
    :data:`REQUEST_MAPPING_VERSION` so it is part of run identity.

    ``profile`` is accepted only so a caller that already resolved one can prove
    it is the model's; it is checked, never trusted. Passing the wrong model's
    profile is refused rather than obeyed, because obeying it is what sends a
    request no run identity describes.
    """
    shape = check_request_profile(profile, model=model)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": request.system_prompt,
        "tools": [tool_schema(action) for action in request.actions],
        "tool_choice": dict(TOOL_CHOICE),
        "messages": [
            {
                "role": "user",
                "content": canonical_json_text(
                    observed_case(request), "anthropic turn request"
                ),
            }
        ],
    }
    # Added rather than defaulted to a sentinel: an omitted field is *absent*
    # from the body, and a null one is a value the API would read.
    if shape.temperature is not None:
        payload["temperature"] = shape.temperature
    if shape.thinking is not None:
        payload["thinking"] = dict(shape.thinking)
    return payload


def check_payload_profile(payload: Mapping[str, Any], profile: RequestProfile) -> None:
    """Prove the body about to be sent is the one the settings describe.

    The settings are hashed into ``configuration_id`` and written into the run
    manifest, so they are this run's durable claim about what was asked of the
    provider. A body that carries a field they say is omitted, or omits one they
    say is sent, makes that claim false — and the evidence would look identical
    to a run where it was true. So the request is not made.

    Checked as an exact field set, not as a search for known-bad keys: a field
    nobody thought to look for is exactly the one a settings mapping would fail
    to describe.
    """
    sent = tuple(sorted(payload))
    if sent != profile.sent_fields():
        raise AdapterError(
            f"this run records request profile {profile.profile_id!r}, which sends "
            f"the fields {list(profile.sent_fields())} and omits "
            f"{list(profile.omitted_fields())}; the request built for this turn "
            f"carries {list(sent)}. Adapter settings are hashed into run identity "
            "as this run's statement of what was asked of the provider, so a "
            "request they do not describe is not sent"
        )
    for field, expected in (
        ("temperature", profile.temperature),
        ("thinking", None if profile.thinking is None else dict(profile.thinking)),
    ):
        if expected is not None and payload.get(field) != expected:
            raise AdapterError(
                f"this run records request profile {profile.profile_id!r}, which "
                f"sends {field}={expected!r}; the request built for this turn "
                f"carries {payload.get(field)!r}. The value a run's identity "
                "records and the value the provider is sent are the same value"
            )


def request_input_token_bound(payload: Mapping[str, Any]) -> int:
    """An upper bound on the input tokens this exact request can be charged for.

    Derived from the body the SDK will send, canonically encoded, because that
    is the only thing about the request this build can measure. Every token a
    byte-pair encoder emits consumes at least one byte of its UTF-8 source, so
    the encoded length bounds the count from above; the allowance
    :func:`~boundarybench.budget.conservative_input_token_bound` adds on top
    covers the provider-side framing of tools and system content that the body
    does not show. The result is used only to *refuse* requests, so being loose
    costs head-room and being tight would cost the guarantee.
    """
    return conservative_input_token_bound(
        len(canonical_json_text(payload, "anthropic request bound").encode("utf-8"))
    )


# -- the response ------------------------------------------------------------


#: The only two content block types this scaffold reads.
TEXT_BLOCK_TYPE = "text"
TOOL_USE_BLOCK_TYPE = "tool_use"

#: Checked by *value*, not by Python class. The SDK resolves a content block
#: against a union leniently: a block declaring a type the installed version
#: does not know is constructed as the union's first member — ``TextBlock`` —
#: carrying its unknown ``type`` and a null ``text``. So ``isinstance(block,
#: TextBlock)`` answers "the SDK fell back", not "the provider sent text", and
#: trusting it let content this build cannot interpret pass as commentary and be
#: ignored. The declared type is compared against this closed set instead.
ACCEPTED_BLOCK_TYPES: frozenset[str] = frozenset({TEXT_BLOCK_TYPE, TOOL_USE_BLOCK_TYPE})


def _tool_uses(message: Message) -> list[ToolUseBlock]:
    """Every tool call in the answer, refusing any block that is neither.

    Text beside an action is ordinary commentary and is ignored. Anything else —
    a thinking block, a server-side tool result, a block type this build has
    never heard of — is content this scaffold did not ask for and cannot
    interpret, and reading past it would mean guessing which part of the answer
    was the action.
    """
    calls: list[ToolUseBlock] = []
    for block in message.content:
        if block.type not in ACCEPTED_BLOCK_TYPES:
            raise AdapterProtocolError(
                "the answer carries a content block that is neither text nor a tool "
                "call; this scaffold accepts one tool call beside optional text, and "
                "cannot tell which part of an answer containing anything else is the "
                "action. The type the block declared is not quoted here: it is "
                "provider-controlled text, and a durable failure row records only "
                "this build's own fixed detail and closed-set values"
            )
        if isinstance(block, ToolUseBlock):
            calls.append(block)
    return calls


def check_response_model(message: Message, *, model: str) -> None:
    """Prove the answer came from the model the run pinned and asked for.

    ``message.model`` is the provider's own statement of which model produced
    the body, and it is the only place a run can learn that its request was not
    served by what it requested — an alias silently resolved, a request routed
    elsewhere, a response substituted in front of the client. A benchmark whose
    rows say ``model=X`` while ``X`` never answered is false in exactly the way
    an audit cannot detect, so the mismatch is refused *before* the answer's
    action is read or its token counts are banked.

    Classified as :data:`PROVIDER_FAULT_RESPONSE_INVALID`, not as a protocol
    failure. Nothing about the model's *behaviour* is wrong here — the answer
    may be a flawless tool call — so filing it under the model would put a
    service-side provenance fault in the bucket that is supposed to mean "the
    model answered badly", and a report's model-protocol counts would then
    include episodes no model got wrong.

    Neither model string is quoted. The one the response carries is
    provider-controlled text of unbounded shape, and a durable failure row
    records only fixed detail and closed-set values; the one the run pinned is
    already on every row, in the adapter identity and in run identity, so
    repeating it here would add nothing and would invite reading the sentence as
    a trustworthy diff of two names when only one of them is trustworthy.
    """
    if message.model != model:
        raise AdapterProviderError(
            PROVIDER_FAULT_RESPONSE_INVALID,
            "the response states that a model other than the one this run pinned "
            "and requested produced it, so it is not evidence about the model "
            "under test and neither its action nor its token counts are accepted. "
            "Neither model identifier is recorded here: the response's is "
            "provider-controlled text, and the run's is already named by the "
            "adapter identity on every row",
        )


def parse_message(message: Message, request: TurnRequest) -> AdapterCall:
    """The provider's answer, as exactly one action call, or a refusal.

    Every message this raises is a fixed sentence plus values this build itself
    chose — a count of the calls that arrived, the stop reason the contract
    *requires*, the action names the scaffold declares. Nothing that arrived in
    the response is quoted: not the stop reason it stated, not a content block's
    declared type, not a tool name, not an argument, not a header or a request
    id. A protocol failure is recorded in a durable ledger row, and a row is not
    a place to put text that arrived from outside — a ``stop_reason`` and a
    block ``type`` are as provider-controlled as a payload is, however
    enumerated they look in the API documentation.
    """
    # Truncation is decided first, and before the call count, because it
    # *explains* the call count. A turn cut off at the ceiling may carry no tool
    # call, half of one, or a complete-looking one the model had not finished
    # committing to; reporting "the answer carries 0 tool call(s)" would describe
    # the symptom and hide the cause.
    if message.stop_reason == OUTPUT_LIMIT_STOP_REASON:
        raise AdapterOutputLimitError(
            "the answer reached this run's pinned output-token ceiling of "
            f"{MAX_OUTPUT_TOKENS} and was cut off there, so what it carries is a "
            "fragment rather than the action the model was making. This is an "
            "output-budget limit and not a malformed answer: the ceiling is this "
            "run's own setting, it is recorded in run identity, and raising it is "
            "the change that would let the same model finish. The provider's "
            "stated stop reason is not quoted here: it is provider-controlled "
            "text, and a durable failure row records only this build's own fixed "
            "detail and closed-set values"
        )
    calls = _tool_uses(message)
    if len(calls) != 1:
        raise AdapterProtocolError(
            f"the answer carries {len(calls)} tool call(s); this scaffold requires "
            "exactly one action per turn, and neither prose alone nor two actions "
            "at once names the single thing the agent did"
        )
    if message.stop_reason != REQUIRED_STOP_REASON:
        raise AdapterProtocolError(
            f"the answer stopped for something other than {REQUIRED_STOP_REASON!r}, "
            "so the call it carries was cut short rather than completed and is not "
            "the action the model was making. The reason the provider stated is not "
            "quoted here: it is provider-controlled text, and a durable failure row "
            "records only this build's own fixed detail and closed-set values"
        )
    block = calls[0]
    if block.name not in {action.name for action in request.actions}:
        raise AdapterProtocolError(
            "the answer calls a tool the scaffold does not declare; the scaffold "
            f"offers {sorted(action.name for action in request.actions)}. The name "
            "the model produced is not quoted here: it is model-authored text and "
            "a durable failure row records only stable, closed-set detail"
        )
    arguments = block.input
    if not isinstance(arguments, Mapping) or not all(
        isinstance(key, str) for key in arguments
    ):
        raise AdapterProtocolError(
            f"the call to a {len(request.actions)}-tool scaffold did not carry a "
            "JSON object with string keys as its arguments"
        )
    try:
        ensure_json_safe(dict(arguments), "anthropic tool call arguments")
    except JsonSafetyError as exc:
        raise AdapterProtocolError(
            "the call's arguments carry a value this build cannot record in a "
            "ledger row, so the answer cannot be preserved as evidence of what "
            "the model said. The offending value is not quoted here"
        ) from exc
    return AdapterCall(action=block.name, arguments=dict(arguments))


# -- provider faults ---------------------------------------------------------


@dataclass(frozen=True)
class _Fault:
    """What one SDK exception means, and whether repeating helps."""

    fault: str
    retryable: bool
    status: int | None
    response_received: bool = False

    def __post_init__(self) -> None:
        """Whether a fault is worth repeating is the contract's answer, not this
        module's second opinion.

        The ledger decides whether a recorded attempt was allowed to be followed
        by another one from :data:`~boundarybench.adapter.RETRYABLE_PROVIDER_FAULTS`,
        and it never imports this module. Two independent tables of the same
        fact would drift, and the direction that matters is the silent one: a
        fault classified as retryable here and terminal there produces real rows
        the reader refuses. So this is bound to that set rather than checked
        against it by hand.
        """
        if self.retryable != (self.fault in RETRYABLE_PROVIDER_FAULTS):
            raise AdapterError(
                f"this integration classifies {self.fault!r} as "
                f"{'retryable' if self.retryable else 'terminal'}, and the adapter "
                f"contract does not. Retryability is a property of the fault and is "
                f"named once, in RETRYABLE_PROVIDER_FAULTS: "
                f"{list(RETRYABLE_PROVIDER_FAULTS)}"
            )


def classify_exception(exception: Exception) -> _Fault | None:
    """Map one SDK error onto the contract's fault set, or decline.

    ``None`` means "not a provider fault": something inside this integration
    broke, and the runner should record it as the adapter failure it is rather
    than have it dressed up as an outage.

    Retryability is a property of the fault, not of the caller's patience. A
    refused credential, an unknown model and an oversized payload all produce
    exactly the same answer the second time, so repeating them spends the
    episode's budget to learn nothing; a rate limit, a 5xx and a dropped
    connection are the three that can differ on the next attempt.
    """
    # Checked before its base class: an SDK timeout *is* an
    # ``APIConnectionError``, and the two are worth telling apart.
    if isinstance(exception, anthropic.APITimeoutError):
        return _Fault(PROVIDER_FAULT_TIMEOUT, True, None)
    if isinstance(exception, anthropic.APIConnectionError):
        return _Fault(PROVIDER_FAULT_NETWORK_ERROR, True, None)
    if isinstance(exception, anthropic.APIResponseValidationError):
        return _Fault(
            PROVIDER_FAULT_RESPONSE_INVALID,
            False,
            None,
            response_received=True,
        )
    if isinstance(exception, anthropic.APIStatusError):
        status = exception.status_code
        if status in (401, 403):
            return _Fault(
                PROVIDER_FAULT_AUTHENTICATION, False, status, response_received=True
            )
        # The one 4xx that is proven to be about the *run* rather than the
        # request: the Messages API answers 404 when the model or resource named
        # in the URL/body does not exist for this account. The model id is
        # pinned in configuration identity and is sent identically on every turn
        # of every episode, so this cannot come out differently later, which is
        # what earns it the right to stop the whole plan. Every other 4xx is
        # about the payload this turn happened to send and stays one episode's
        # failure.
        if status == 404:
            return _Fault(
                PROVIDER_FAULT_CONFIGURATION, False, status, response_received=True
            )
        if status == 408:
            return _Fault(PROVIDER_FAULT_TIMEOUT, True, status, response_received=True)
        if status == 429:
            return _Fault(
                PROVIDER_FAULT_RATE_LIMITED, True, status, response_received=True
            )
        if status >= 500:
            return _Fault(
                PROVIDER_FAULT_SERVER_ERROR, True, status, response_received=True
            )
        return _Fault(
            PROVIDER_FAULT_REQUEST_REJECTED, False, status, response_received=True
        )
    return None


def _fault_detail(fault: _Fault, *, attempts: int, reason: str) -> str:
    """The durable sentence one provider fault is recorded as.

    Assembled from a fixed template, the fault's own name, an HTTP status and a
    count — and nothing else. The provider's message, body, headers and request
    id are all in hand at the call site and all deliberately dropped: a ledger
    row is durable evidence, and vendor- or attacker-controlled text of
    unbounded shape does not belong in one.
    """
    answered = (
        f"the provider answered HTTP {fault.status}"
        if fault.status is not None
        else "the request to the provider did not complete"
    )
    return (
        f"{answered} ({fault.fault}) after {attempts} attempt(s); {reason}. The "
        "provider's own message, response body, headers and request id are not "
        "recorded here by design"
    )


#: What the adapter waits on, and what it reads the clock with. Injected so the
#: retry and deadline paths are exercised deterministically and instantly.
Sleep = Callable[[float], None]
Clock = Callable[[], float]


# -- the adapter -------------------------------------------------------------


class AnthropicMessagesAdapter:
    """One provider client, one pinned model, one action per turn."""

    def __init__(
        self,
        *,
        model: str,
        client: anthropic.Anthropic,
        retry: AnthropicRetryPolicy | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: RunCostGuard | None = None,
        profile: RequestProfile | None = None,
    ) -> None:
        # First, and before anything is read off the client: a profile that is
        # not this model's is refused where it costs an error message rather
        # than a rejected request, a spent reservation and a durable row.
        self._profile = check_request_profile(profile, model=model)
        if client.max_retries != SDK_MAX_RETRIES:
            raise AnthropicConfigurationError(
                f"this client was built with max_retries={client.max_retries}. The "
                "adapter owns the retry policy, so a second, hidden one underneath "
                "it would spend the episode's wall-clock budget on calls the runner "
                "never saw, make a turn's measured latency describe an unknown "
                f"number of requests, and hide a rate limit the run should have "
                f"recorded. Build the client with max_retries={SDK_MAX_RETRIES}"
            )
        self._model = model
        self._client = client
        self._retry = retry or AnthropicRetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._usage = AdapterUsage()
        self._telemetry = ProviderTelemetry()
        # One turn at a time on this instance: the two measurements above are
        # per-adapter state that a turn owns while it runs. See
        # :class:`~boundarybench.adapter.SingleFlight`.
        self._single_flight = SingleFlight(adapter="AnthropicMessagesAdapter")
        # ``None`` on a run with no cost cap, and then nothing here prices
        # anything: a cost computed from a table this run never pinned would be
        # a measurement no manifest records the provenance of.
        self._cost_guard = cost_guard

    def __repr__(self) -> str:
        """Names the run, never the credential.

        Written out rather than left to a dataclass: a generated ``repr`` would
        render whatever the client holds, and the client holds an API key.
        """
        return (
            f"<AnthropicMessagesAdapter provider={ANTHROPIC_PROVIDER} "
            f"model={self._model} implementation={ANTHROPIC_IMPLEMENTATION}>"
        )

    @property
    def identity(self) -> AdapterIdentity:
        return anthropic_identity(self._model)

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
        the guard the requests will actually be authorised by. Checking the
        runner's own reference alone would not: two guards can hold the same
        controls, and the one that matters is the one in here.
        """
        return self._cost_guard

    def last_usage(self) -> AdapterUsage:
        return self._usage

    def last_telemetry(self) -> ProviderTelemetry:
        """What this turn asked of the provider, and what each attempt met.

        Never ``None``: this adapter does call a provider, so every turn has an
        answer to give — including a turn whose budget was spent before a
        request could be dispatched, which honestly reports zero attempts.
        """
        return self._telemetry

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        """One turn: clear the measurement, ask, measure, then read the answer.

        The order is the contract. Clearing happens *first* — before the request
        is even serialised — because everything after it can fail, and a
        measurement that outlives the call it measured is stale evidence. If
        clearing happened inside :meth:`_send`, request encoding could fail before
        the reset and leave another turn's tokens readable through
        :meth:`last_usage`.

        Both measurements are cleared, for the same reason and at the same
        point. Attempt evidence that survived into the next turn would be worse
        than stale usage: it would attribute one turn's provider faults to
        another turn, which is the exact question the evidence exists to answer.

        Parsing comes last and deliberately after the measurement. A response
        that arrives is paid for whether or not this scaffold can read it as one
        action, so :meth:`last_usage` answers for a protocol failure's cost too.

        The whole of it happens under this instance's single-flight guard, taken
        *before* the clearing above: a second overlapping turn would reset the
        measurement of the one in flight, so it is refused immediately rather
        than allowed to interleave. Sequential turns are unaffected, which is
        what every episode of every run actually does.
        """
        with self._single_flight:
            self._usage = AdapterUsage()
            self._telemetry = ProviderTelemetry()
            payload = build_messages_request(
                request, model=self._model, profile=self._profile
            )
            message = self._send(payload, deadline)
            return parse_message(message, request)

    def _send(self, payload: dict[str, Any], deadline: TurnDeadline) -> Message:
        """One turn's provider call, retried within the turn's own budget.

        The deadline is the ceiling on the whole loop, not on one request: the
        remaining budget is recomputed before every attempt and handed to the
        SDK as that attempt's timeout, so retrying cannot extend an episode past
        the wall-clock it was given. A wait that would not leave time for the
        request it precedes is not taken at all — sleeping through the rest of
        the budget and then failing wastes the turn to learn nothing.

        Only :class:`Exception` is caught, and only the subset
        :func:`classify_exception` recognises. A ``KeyboardInterrupt`` is not a
        provider fault and is not retried, delayed or reclassified.
        """
        # Before the clock starts, before an attempt is counted and before the
        # cost guard is asked for a reservation: the body has to be the one this
        # run's settings describe. Nothing below can put that right — a request
        # that has left is evidence, and the manifest it would be read against
        # would be describing a different one.
        check_payload_profile(payload, self._profile)
        # Both measurements were cleared by ``next_call``, before the request
        # this loop is about to send was even built. Nothing here re-clears
        # them: two clears would be two places for the rule to live, and the one
        # that runs first is the one that matters.
        started = self._clock()
        wait = self._retry.initial_backoff_seconds
        attempts = 0
        record: list[ProviderAttempt] = []

        def left() -> float:
            return deadline.remaining_seconds - (self._clock() - started)

        def commit(now: float, reason: str) -> None:
            """Publish what the turn has done so far, before anything raises.

            Called on every exit from this loop rather than only on the ones
            that succeed. The whole point of this evidence is that it survives a
            failure: a turn that made three requests and got nothing is exactly
            the turn whose attempts an operator needs to see, and it is also the
            turn from which nothing else can be learned.

            The instant is passed in rather than read here. Every exit already
            has one in hand, and taking a second reading would make the turn's
            recorded latency depend on how many times this loop happened to
            consult the clock rather than on how long the turn took.

            ``reason`` is which exit this is, named at the exit itself. It is
            the only part of a turn's evidence that cannot be re-derived from
            the attempts — a turn that stopped early because the clock, the
            cancellation or the budget said so looks exactly like one whose
            later attempts were deleted — so it is stated where the fact is
            known rather than reconstructed later from a message.
            """
            self._telemetry = ProviderTelemetry(
                attempts=tuple(record),
                turn_latency_seconds=now - started,
                terminal_reason=reason,
            )

        while True:
            # One reading, used as both the budget's origin and this attempt's
            # start. Two would drift apart under a clock that moves between them.
            attempt_started = self._clock()
            budget = deadline.remaining_seconds - (attempt_started - started)
            if budget <= 0.0 or deadline.cancelled():
                commit(attempt_started, TURN_END_DEADLINE_EXCEEDED)
                raise AdapterProviderError(
                    PROVIDER_FAULT_TIMEOUT,
                    _fault_detail(
                        _Fault(PROVIDER_FAULT_TIMEOUT, True, None),
                        attempts=attempts,
                        reason=(
                            "the turn's remaining wall-clock budget was spent or "
                            "cancelled before a request could be made"
                        ),
                    ),
                )
            # Before the attempt is counted, and before a socket is opened: a
            # request the run's remaining authorised budget cannot cover is not
            # made at all. The refusal raises out of this loop without touching
            # the attempt record, because an attempt that never happened is not
            # evidence of one that did. ``commit`` still publishes what the turn
            # did before it, so a turn refused on its third retry still reports
            # the two attempts it made.
            reservation: Decimal | None = None
            if self._cost_guard is not None:
                try:
                    reservation = self._cost_guard.authorize(
                        input_tokens_upper_bound=request_input_token_bound(payload)
                    )
                except CostCapExceededError:
                    commit(self._clock(), TURN_END_COST_CAP_EXHAUSTED)
                    raise
            # Rendered once, here, and carried onto every attempt record this
            # iteration can produce: the amount a row states as reserved is the
            # amount the guard actually authorised for that same request.
            reserved = None if reservation is None else usd_text(reservation)
            attempts += 1
            try:
                # Annotated rather than inferred: ``**payload`` is a plain
                # mapping, so the SDK's overloads resolve to ``Any`` and the
                # loop's return type would silently become untyped.
                message: Message = self._client.messages.create(**payload, timeout=budget)
            except Exception as exception:
                failed_at = self._clock()
                # An attempt that returned nothing this build could measure keeps
                # its whole conservative bound as exposure. Whether the provider
                # billed for it is not something a client can determine, and the
                # only error that can breach a cap is under-stating it.
                self._forfeit(reservation)
                fault = classify_exception(exception)
                if fault is None:
                    # Not the provider's doing. Left to the runner, which records
                    # it as this integration's own failure rather than an outage.
                    # The attempt is still recorded — the request was made, the
                    # time it took is the turn's whatever broke afterwards, and
                    # the reservation it forfeited is money the run has to
                    # account for. Append before committing so the attempt and
                    # its exposure cannot vanish from the evidence.
                    record.append(
                        ProviderAttempt(
                            index=attempts,
                            outcome=ATTEMPT_OUTCOME_UNCLASSIFIED,
                            latency_seconds=failed_at - attempt_started,
                            cost_reservation_usd=reserved,
                            cost_settlement=(
                                None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                            ),
                        )
                    )
                    commit(failed_at, TURN_END_UNCLASSIFIED)
                    raise
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_FAULT,
                        fault=fault.fault,
                        http_status=fault.status,
                        latency_seconds=failed_at - attempt_started,
                        response_received=fault.response_received,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                if not fault.retryable:
                    commit(failed_at, TURN_END_FAULT_NOT_RETRYABLE)
                    raise AdapterProviderError(
                        fault.fault,
                        _fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "this fault is not retried, because repeating the "
                                "same request cannot change the answer"
                            ),
                        ),
                    ) from exception
                if attempts >= self._retry.max_attempts:
                    commit(failed_at, TURN_END_RETRIES_EXHAUSTED)
                    raise AdapterProviderError(
                        fault.fault,
                        _fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "the adapter's bounded retry policy is exhausted "
                                "and no answer was received"
                            ),
                        ),
                    ) from exception
                pause, wait = wait, wait * self._retry.backoff_multiplier
                # Two different facts, kept apart: a cancelled episode ran out
                # of wall clock, and a backoff that will not fit is this
                # policy's own arithmetic over what is left of it.
                cancelled = deadline.cancelled()
                if cancelled or pause >= left():
                    commit(
                        failed_at,
                        TURN_END_DEADLINE_EXCEEDED
                        if cancelled
                        else TURN_END_BACKOFF_UNAFFORDABLE,
                    )
                    raise AdapterProviderError(
                        fault.fault,
                        _fault_detail(
                            fault,
                            attempts=attempts,
                            reason=(
                                "the turn's remaining wall-clock budget cannot pay "
                                "for the next backoff and the request after it, so "
                                "the retry was abandoned rather than started"
                            ),
                        ),
                    ) from exception
                self._sleep(pause)
                continue
            # Before the measurement, and so before anything is parsed out of
            # the body: a response whose stated model is not the pinned one is
            # refused whole. This is the one narrow exception to "a response
            # that arrived was paid for, so it is measured". Every usage figure
            # this build records is attributed to the adapter identity on the
            # row that carries it, so banking these counts would attribute spend
            # to a model that did not produce it — the same falsehood as
            # accepting the action would be, and the reason the provenance check
            # was added. A protocol-invalid answer *from the pinned model* is
            # the opposite case and is measured, immediately below.
            answered_at = self._clock()
            try:
                check_response_model(message, model=self._model)
            except AdapterProviderError:
                # A body arrived, so the attempt is recorded as one that got a
                # response — and as one that reported no usage, because this
                # build refuses to attribute those counts to a model that did
                # not produce them. Recording it as a transport fault instead
                # would say no answer came back, which is not what happened.
                #
                # The reservation is forfeited for the same reason the counts
                # are refused: the request was made and something was produced,
                # and this run cannot attribute the spend to the model it pinned.
                # It is exposure, not measured cost.
                self._forfeit(reservation)
                record.append(
                    ProviderAttempt(
                        index=attempts,
                        outcome=ATTEMPT_OUTCOME_FAULT,
                        fault=PROVIDER_FAULT_RESPONSE_INVALID,
                        latency_seconds=answered_at - attempt_started,
                        response_received=True,
                        cost_reservation_usd=reserved,
                        cost_settlement=(
                            None if reserved is None else ATTEMPT_SETTLEMENT_FORFEITED
                        ),
                    )
                )
                commit(answered_at, TURN_END_FAULT_NOT_RETRYABLE)
                raise

            breach: CostReservationBreachedError | None = None
            try:
                settlement = self._record_usage(
                    message, seconds=answered_at - started, reservation=reservation
                )
            except CostReservationBreachedError as exc:
                # The response arrived, was measured and was banked at its
                # measured value; what failed is the authorisation, not the
                # observation. So the attempt is recorded exactly as the
                # successful one it was — including the settlement, which really
                # did measure — and only then does the turn raise. Committing the
                # evidence first is the whole point: the row that stops the run
                # is the one that has to show what the run had spent when it did.
                breach, settlement = exc, ATTEMPT_SETTLEMENT_MEASURED
            record.append(
                ProviderAttempt(
                    index=attempts,
                    outcome=ATTEMPT_OUTCOME_RESPONSE,
                    latency_seconds=answered_at - attempt_started,
                    response_received=True,
                    usage_reported=True,
                    cost_reservation_usd=reserved,
                    cost_settlement=settlement,
                )
            )
            commit(answered_at, TURN_END_RESPONSE)
            if breach is not None:
                raise breach
            return message

    def _forfeit(self, reservation: Decimal | None) -> None:
        """Resolve an outstanding reservation as exposure, if there is one."""
        if self._cost_guard is not None and reservation is not None:
            self._cost_guard.forfeit(reservation)

    def _record_usage(
        self, message: Message, *, seconds: float, reservation: Decimal | None = None
    ) -> str | None:
        """One turn's measurement, under one stated convention.

        * ``input_tokens``/``output_tokens`` are the provider's own counts on
          the response that *returned*, accepted or not. Recorded before the
          answer is parsed, so a turn that ends in a protocol failure still
          reports what the provider charged for producing it: those tokens were
          consumed, and a benchmark that omitted them would under-report real
          spend on exactly the turns where a model misbehaved. Attempts that
          failed in transport returned no body to measure, so they contribute
          nothing rather than zero — null is "not measured", and this build
          never renders it as a number. The counts are passed through unaltered:
          if a provider ever reported something that is not a token count, the
          runner refuses the measurement rather than this adapter quietly
          repairing it.
        * A turn's measurement covers every response the turn received. At most
          one of them can be measurable in this build — a response either ends
          the turn or ends the run, so the retry loop never gets a second body —
          which is why this assigns rather than sums; the wall-clock below is
          what carries the cost of the attempts that produced no body.
        * Cache-related counters are deliberately not folded in. This build sets
          no ``cache_control``, so there is nothing cached to count, and summing
          fields whose meaning depends on a feature that is off would make the
          number mean different things in different runs.
        * ``latency_seconds`` is locally measured wall-clock for the whole turn:
          from immediately before the first provider request to immediately
          after the measured response returned, *including* retried attempts and
          the backoff waits between them. So a turn's latency is the total the
          turn actually consumed across all of its attempts, not one request's.
          It is not the provider's server-side processing time, and nothing here
          can measure that.
        * ``cost_usd`` is computed **only** under a cost guard, from the counts
          above at the run's pinned pricing policy
          (:mod:`boundarybench.pricing`), in decimal arithmetic. Without a guard
          this run has pinned no price, and a number derived from a table its
          manifest does not record would be a fabricated measurement dressed as
          a recorded one — so it stays ``None``. Under a guard, a response whose
          counts the price policy cannot use is settled as *unmeasured*: its
          reservation becomes exposure and the cost stays null, never zero.

        Returns how the reservation was resolved, for the attempt record that
        has to state it, or ``None`` when there was no reservation. A measured
        cost above the reservation that authorised it raises
        :class:`~boundarybench.budget.CostReservationBreachedError` out of the
        guard — and the measurement is published here first, so the row that
        records the breach records what it cost rather than losing it to the
        exception.
        """
        cost: Decimal | None = None
        settlement: str | None = None
        breach: CostReservationBreachedError | None = None
        if self._cost_guard is not None and reservation is not None:
            try:
                cost = self._cost_guard.settle(
                    reservation,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                )
            except CostReservationBreachedError as exc:
                cost, breach = exc.measured_usd, exc
            settlement = (
                ATTEMPT_SETTLEMENT_FORFEITED
                if cost is None
                else ATTEMPT_SETTLEMENT_MEASURED
            )
        self._usage = AdapterUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            # Converted once, here, at the boundary where a JSON row is written.
            # Every sum before this point is exact decimal arithmetic; see
            # ``boundarybench.budget``.
            cost_usd=None if cost is None else float(cost),
            latency_seconds=seconds,
        )
        if breach is not None:
            raise breach
        return settlement


__all__ = [
    "ANTHROPIC_ADAPTER_VERSION",
    "ANTHROPIC_API",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_IMPLEMENTATION",
    "ANTHROPIC_PROVIDER",
    "API_KEY_VARIABLE",
    "BASELINE_PROFILE",
    "DISABLED_THINKING",
    "HAIKU_4_5_MODEL",
    "HAIKU_4_5_PROFILE",
    "MAX_OUTPUT_TOKENS",
    "MODEL_REQUEST_PROFILES",
    "OUTPUT_LIMIT_STOP_REASON",
    "PROHIBITED_REQUEST_FIELDS",
    "REJECTED_VARIABLES",
    "REQUEST_MAPPING_VERSION",
    "REQUIRED_STOP_REASON",
    "SDK_MAX_RETRIES",
    "SONNET_5_MODEL",
    "SONNET_5_PROFILE",
    "TEMPERATURE",
    "TOOL_CHOICE",
    "AnthropicConfigurationError",
    "AnthropicMessagesAdapter",
    "AnthropicRetryPolicy",
    "RequestProfile",
    "anthropic_identity",
    "anthropic_settings",
    "build_anthropic_adapter",
    "build_client",
    "build_messages_request",
    "check_environment",
    "check_model",
    "check_payload_profile",
    "check_request_profile",
    "check_response_model",
    "classify_exception",
    "observed_case",
    "parse_message",
    "profile_settings",
    "request_input_token_bound",
    "request_profile_for",
    "resolve_api_key",
    "tool_schema",
]
