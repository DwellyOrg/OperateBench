"""Proving a client will do what a run's settings say it will.

Four questions, asked before a request exists: is the model identifier one, is
there exactly one credential and did it come from the one place it may come
from, will the environment silently redirect or re-bill the request, and does the
body about to be sent match the field set the settings describe. Each of them is
a statement a run's durable record makes about itself, and each of them is
checkable before a socket exists.

None of it is a track's decision, and none of it is a provider's. What *is*
per-provider — which variable, which endpoint, which error class — arrives as an
argument, so an integration's traceback still names the provider that refused.

Nothing in this module opens a socket, reads a credential into a durable place,
or imports a provider SDK.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from operatebench.providers.faults import AdapterError


class ProviderConfigurationError(AdapterError):
    """The environment cannot produce a usable, unambiguous provider client.

    Raised before a run directory, a manifest or a durable row exists, because a
    missing credential is an operator mistake rather than a recorded run
    outcome. Its message never carries the credential, or any part of one.

    Each integration subclasses this so an operator's traceback names the
    provider that refused, while a caller that wants to catch "any provider
    refused to configure" has one type to catch.
    """


#: What a model identifier may look like. Deliberately a shape check and not a
#: list: a build that shipped an allowlist of model names would refuse every
#: model released after it, which is the opposite of what a benchmark needs.
#:
#: What this cannot check is whether the identifier is *pinned*. ``--model`` is
#: required precisely so nothing here silently picks an alias, but an operator
#: who passes a moving alias gets a run whose recorded model name does not name
#: one model. Pinning a dated snapshot id is the operator's decision, and the
#: documentation says so rather than this pattern pretending to enforce it.
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def check_model(
    model: str, *, provider: str, error: type[ProviderConfigurationError]
) -> str:
    """Prove the model identifier is one, before it becomes run identity."""
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
        raise error(
            f"a {provider} run needs an explicit model identifier of letters, "
            "digits, dots, dashes and underscores. Pin the exact model "
            "identifier under test: an alias moves, and a run whose recorded "
            "model names more than one model is not comparable with anything"
        )
    return model


def resolve_api_key(
    variable: str,
    *,
    provider: str,
    error: type[ProviderConfigurationError],
    environ: Mapping[str, str] | None = None,
) -> str:
    """The credential, from the one variable it may come from.

    Read here and passed explicitly to the client, rather than left to an SDK's
    own resolution order. An SDK will otherwise fall back to a second variable,
    an on-disk profile or a workload-identity configuration, any of which would
    let a run authenticate as something the operator did not choose and the
    manifest cannot record. One variable, or nothing.

    The value is returned and never stored, logged or interpolated into a
    message — including this module's own errors.
    """
    values = os.environ if environ is None else environ
    key = values.get(variable, "")
    if not key.strip():
        raise error(
            f"{variable} is not set, so there is no credential to run against "
            f"{provider} with. Export it in the environment; it is never read "
            "from a command-line setting, and never written to a run manifest, "
            "a ledger row or this build's output"
        )
    return key


def check_environment(
    variables: Iterable[str],
    *,
    provider: str,
    base_url: str,
    error: type[ProviderConfigurationError],
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse an environment that would silently redirect or rewrite requests.

    An SDK honours these variables. Ignoring them would be worse than either
    obeying or refusing: the operator who set one would get a run that looks
    like it went where they asked and did not, and the credential would be sent
    somewhere the manifest does not record.
    """
    values = os.environ if environ is None else environ
    for variable in variables:
        if values.get(variable, "").strip():
            raise error(
                f"{variable} is set. This build talks to {base_url} for "
                f"{provider} and only there, because the endpoint is where the "
                "credential is sent and a run manifest that recorded one "
                "endpoint while the request went to another would be false. "
                "Unset it, or use a build that pins the endpoint you want"
            )


def normalized_endpoint(value: str) -> str:
    """One endpoint URL with at most one trailing slash removed.

    The *only* normalisation this build performs on an endpoint. An SDK that
    stores ``https://api.openai.com/v1/`` for the base URL this repository pins
    as ``https://api.openai.com/v1`` has not been redirected, and refusing that
    would refuse the officially-built client. Everything else a URL can differ
    by — a second path segment, an explicit default port, a different host, a
    different scheme, embedded userinfo, a query or a fragment — changes where
    the credential goes or what the request means, so none of it is normalised
    away and all of it is refused by the exact comparison in
    :func:`check_endpoint`.
    """
    return value[:-1] if value.endswith("/") else value


def check_endpoint(
    endpoint: object,
    *,
    expected: str,
    provider: str,
    setting: str,
    error: type[ProviderConfigurationError],
) -> str:
    """Prove an SDK client will send this run's credential where settings say.

    Adapter settings record ``base_url``, that mapping is hashed into
    ``configuration_id``, and the run manifest publishes it as this run's claim
    about which service answered. A client constructed elsewhere and handed in
    carries its own endpoint, so without this check the claim is a constant in
    the source rather than a statement about the run: the request goes to the
    injected host and the manifest names the official one. That is the one
    falsehood an audit cannot detect from the artefact, and it is also how a
    credential reaches a third party.

    Checked in the *constructor*, not only in the factory. The factory builds
    its own pinned client, so it was never the way in; a public constructor that
    accepts a client is.

    The endpoint that arrived is never quoted. It is caller-controlled text of
    unbounded shape, and a URL is one of the few places a credential can hide —
    ``https://user:secret@host/`` is a valid base URL. The message names the one
    endpoint this build talks to, which is already public in these settings.
    """
    text = endpoint if isinstance(endpoint, str) else str(endpoint)
    if normalized_endpoint(text) != normalized_endpoint(expected):
        raise error(
            f"this client is configured to send {provider} requests to an endpoint "
            f"other than {expected}, which is the only one this build talks to and "
            f"the one it records in adapter settings as {setting!r}. A base URL is "
            "where the credential is sent, and a run whose manifest named one "
            "endpoint while the request went to another would be false in the one "
            "place an audit checks. The configured endpoint is not quoted here: it "
            "is caller-controlled text and a URL can carry credentials in its "
            "userinfo. Build the client with this module's `build_client`"
        )
    return expected


def check_payload_fields(
    payload: Mapping[str, Any],
    *,
    profile_id: str,
    sent: tuple[str, ...],
    omitted: tuple[str, ...],
) -> None:
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
    actual = tuple(sorted(payload))
    if actual != sent:
        raise AdapterError(
            f"this run records request profile {profile_id!r}, which sends the "
            f"fields {list(sent)} and omits {list(omitted)}; the request built "
            f"for this turn carries {list(actual)}. Adapter settings are hashed "
            "into run identity as this run's statement of what was asked of the "
            "provider, so a request they do not describe is not sent"
        )


__all__ = [
    "MODEL_PATTERN",
    "ProviderConfigurationError",
    "check_endpoint",
    "check_environment",
    "check_model",
    "check_payload_fields",
    "normalized_endpoint",
    "resolve_api_key",
]
