"""The one place that says what each credentialed provider *is*.

Before this module there was one provider, and the CLI could name its
credential variable, its adapter factory, its identity function, its settings
function and its retry policy inline without that being a design decision. With
four, inlining them would mean four copies of the same five-way branch in the
run path, the preflight path and the matrix orchestrator — and the failure mode
of that is silent: a provider added to one branch and missed in another produces
a CLI that accepts an adapter it cannot preflight, or preflights one it cannot
price.

So each provider is one :class:`ProviderChoice`, and everything that needs to
act per provider reads it from here. What a caller may *not* do is hold a second
opinion: the pricing key, the credential variable and the provider string an
adapter identity records are all read off the same object, so a run priced under
one provider and executed against another cannot be constructed.

Nothing in this module opens a socket or reads a credential. Building an adapter
does both, which is why that is a callable stored here rather than something
this module does at import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from boundarybench.adapter import AdapterIdentity
from boundarybench.budget import RunCostGuard
from boundarybench.providers import anthropic_messages, mistral_chat, openai_responses
from boundarybench.providers import xai_openai_compat as xai
from boundarybench.providers.common import (
    ProviderRetryPolicy,
    check_environment,
    check_model,
)


class AdapterFactory(Protocol):
    """How a run turns a pinned model into a live, authorised adapter."""

    def __call__(
        self,
        *,
        model: str,
        configuration_id: str | None = None,
        audited: Iterable[str] | None = None,
        environ: Mapping[str, str] | None = None,
        cost_guard: RunCostGuard | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ProviderChoice:
    """Everything the build needs to know about one credentialed provider.

    ``name`` is what an operator types after ``--adapter``; ``provider`` is what
    the adapter identity records and what the price table is keyed on. They are
    the same string for every provider this build ships, and they are two fields
    anyway — a CLI name is a user-interface decision and a provider key is a
    durable identifier, and collapsing them would mean a future rename of one
    silently rewriting the other in every manifest ever written.
    """

    name: str
    provider: str
    api_key_variable: str
    base_url: str
    #: The environment variables that would redirect this provider's requests or
    #: change which account they bill, and are therefore refused.
    rejected_variables: tuple[str, ...]
    #: The output ceiling this provider's adapter pins. Read from the adapter
    #: rather than restated, so the registry cannot disagree with what is sent.
    max_output_tokens: int
    _build: AdapterFactory
    _identity: Callable[[str], AdapterIdentity]
    _settings: Callable[..., dict[str, Any]]
    _check_model: Callable[[str], str]
    _check_environment: Callable[..., None]

    def identity(self, model: str) -> AdapterIdentity:
        """Who a run of this model would record as having answered."""
        return self._identity(model)

    def settings(self, model: str) -> dict[str, Any]:
        """Every result-affecting request setting, computed offline.

        No credential is read and no client is built, which is what makes a
        configuration identity discoverable before it is authorised.
        """
        return self._settings(model=model)

    def check_model(self, model: str) -> str:
        """Prove the model identifier is one, before it becomes run identity."""
        return self._check_model(model)

    def check_environment(self, environ: Mapping[str, str] | None = None) -> None:
        """Refuse an environment that would redirect or rebill this provider."""
        self._check_environment(environ)

    def build(
        self,
        *,
        model: str,
        configuration_id: str | None = None,
        audited: Iterable[str] | None = None,
        environ: Mapping[str, str] | None = None,
        cost_guard: RunCostGuard | None = None,
    ) -> Any:
        """The live adapter, or a clean refusal.

        This is the only call in the registry that reads a credential or opens a
        client, and it refuses an unauthorised configuration before it does
        either.
        """
        return self._build(
            model=model,
            configuration_id=configuration_id,
            audited=audited,
            environ=environ,
            cost_guard=cost_guard,
        )

    def retry_attempts(self) -> int:
        """How many attempts one turn may make, for a preflight's worst case."""
        return ProviderRetryPolicy().max_attempts


def _anthropic_settings(*, model: str) -> dict[str, Any]:
    """Adapt the Anthropic settings signature to the registry's.

    The Anthropic integration takes its retry policy positionally and predates
    :class:`~boundarybench.providers.common.ProviderRetryPolicy`. Wrapped rather
    than changed: that module is what priced and executed the runs already on
    disk, and its behaviour is pinned by a version string a resume refuses to
    cross.
    """
    return anthropic_messages.anthropic_settings(
        anthropic_messages.AnthropicRetryPolicy(), model=model
    )


def _anthropic_build(
    *,
    model: str,
    configuration_id: str | None = None,
    audited: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    cost_guard: RunCostGuard | None = None,
) -> Any:
    return anthropic_messages.build_anthropic_adapter(
        model=model,
        configuration_id=configuration_id,
        audited=audited,
        environ=environ,
        cost_guard=cost_guard,
    )


ANTHROPIC_CHOICE = ProviderChoice(
    name="anthropic",
    provider=anthropic_messages.ANTHROPIC_PROVIDER,
    api_key_variable=anthropic_messages.API_KEY_VARIABLE,
    base_url=anthropic_messages.ANTHROPIC_BASE_URL,
    rejected_variables=anthropic_messages.REJECTED_VARIABLES,
    max_output_tokens=anthropic_messages.MAX_OUTPUT_TOKENS,
    _build=_anthropic_build,
    _identity=anthropic_messages.anthropic_identity,
    _settings=_anthropic_settings,
    _check_model=anthropic_messages.check_model,
    _check_environment=anthropic_messages.check_environment,
)

OPENAI_CHOICE = ProviderChoice(
    name="openai",
    provider=openai_responses.OPENAI_PROVIDER,
    api_key_variable=openai_responses.API_KEY_VARIABLE,
    base_url=openai_responses.OPENAI_BASE_URL,
    rejected_variables=openai_responses.REJECTED_VARIABLES,
    max_output_tokens=openai_responses.MAX_OUTPUT_TOKENS,
    _build=openai_responses.build_openai_adapter,
    _identity=openai_responses.openai_identity,
    _settings=openai_responses.openai_settings,
    _check_model=lambda model: check_model(
        model,
        provider=openai_responses.OPENAI_PROVIDER,
        error=openai_responses.OpenAIConfigurationError,
    ),
    _check_environment=lambda environ=None: check_environment(
        openai_responses.REJECTED_VARIABLES,
        provider=openai_responses.OPENAI_PROVIDER,
        base_url=openai_responses.OPENAI_BASE_URL,
        error=openai_responses.OpenAIConfigurationError,
        environ=environ,
    ),
)

XAI_CHOICE = ProviderChoice(
    name="xai",
    provider=xai.XAI_PROVIDER,
    api_key_variable=xai.API_KEY_VARIABLE,
    base_url=xai.XAI_BASE_URL,
    rejected_variables=xai.REJECTED_VARIABLES,
    max_output_tokens=xai.MAX_OUTPUT_TOKENS,
    _build=xai.build_xai_adapter,
    _identity=xai.xai_identity,
    _settings=xai.xai_settings,
    _check_model=lambda model: check_model(
        model, provider=xai.XAI_PROVIDER, error=xai.XAIConfigurationError
    ),
    _check_environment=lambda environ=None: check_environment(
        xai.REJECTED_VARIABLES,
        provider=xai.XAI_PROVIDER,
        base_url=xai.XAI_BASE_URL,
        error=xai.XAIConfigurationError,
        environ=environ,
    ),
)

MISTRAL_CHOICE = ProviderChoice(
    name="mistral",
    provider=mistral_chat.MISTRAL_PROVIDER,
    api_key_variable=mistral_chat.API_KEY_VARIABLE,
    base_url=mistral_chat.MISTRAL_BASE_URL,
    rejected_variables=mistral_chat.REJECTED_VARIABLES,
    max_output_tokens=mistral_chat.MAX_OUTPUT_TOKENS,
    _build=mistral_chat.build_mistral_adapter,
    _identity=mistral_chat.mistral_identity,
    _settings=mistral_chat.mistral_settings,
    _check_model=lambda model: check_model(
        model,
        provider=mistral_chat.MISTRAL_PROVIDER,
        error=mistral_chat.MistralConfigurationError,
    ),
    _check_environment=lambda environ=None: check_environment(
        mistral_chat.REJECTED_VARIABLES,
        provider=mistral_chat.MISTRAL_PROVIDER,
        base_url=mistral_chat.MISTRAL_BASE_URL,
        error=mistral_chat.MistralConfigurationError,
        environ=environ,
    ),
)

#: Every credentialed provider this build can run, keyed by its CLI name.
#:
#: A read-only proxy rather than a dict: a run's provider set is part of what
#: this build *is*, and a single assignment must not be able to add one.
PROVIDER_ADAPTERS: Mapping[str, ProviderChoice] = MappingProxyType(
    {
        ANTHROPIC_CHOICE.name: ANTHROPIC_CHOICE,
        MISTRAL_CHOICE.name: MISTRAL_CHOICE,
        OPENAI_CHOICE.name: OPENAI_CHOICE,
        XAI_CHOICE.name: XAI_CHOICE,
    }
)


def provider_choice(name: str) -> ProviderChoice:
    """The registered provider by CLI name, or a :class:`KeyError`.

    Matched whole and case-sensitively, like every other identifier in this
    build: a provider name that folded case would accept a string no adapter
    identity records.
    """
    try:
        return PROVIDER_ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a provider this build ships an adapter for; it "
            f"covers exactly {sorted(PROVIDER_ADAPTERS)}"
        ) from None


__all__ = [
    "ANTHROPIC_CHOICE",
    "MISTRAL_CHOICE",
    "OPENAI_CHOICE",
    "PROVIDER_ADAPTERS",
    "XAI_CHOICE",
    "AdapterFactory",
    "ProviderChoice",
    "provider_choice",
]
