"""Closed admission for OpenAI SDK clients used with xAI compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import httpx
import openai

from operatebench.providers.config import ProviderConfigurationError, check_endpoint


def check_xai_openai_compat_client_state(
    client: openai.OpenAI,
    *,
    expected_base_url: str,
    provider: str,
    max_retries: int,
    error: type[ProviderConfigurationError],
) -> None:
    """Refuse effective wire state not represented by xAI adapter settings.

    The check derives the installed httpx version's pristine headers and event
    hook shape at runtime and reports only which state channel is non-pristine.
    It never reads a credential, cookie, header, query, hook callable, or account
    value into an error message.

    An injected ``MockTransport`` or other custom transport is trusted offline
    test plumbing: transport implementations are not introspected. The supported
    live builder owns its transport. This admission instead closes mutable httpx
    request/response hooks that run outside request construction and response
    parsing without broad arbitrary-object inspection.
    """
    check_endpoint(
        client.base_url,
        expected=expected_base_url,
        provider=provider,
        setting="base_url",
        error=error,
    )
    for field, value in (
        ("organization", client.organization),
        ("project", client.project),
    ):
        if value is not None:
            raise error(
                f"this client carries an explicit {field}. That changes account "
                "routing or policy without appearing in adapter settings. Inject a "
                f"client without {field} scoping"
            )

    custom_headers = getattr(client, "_custom_headers", None)
    custom_query = getattr(client, "_custom_query", None)
    if not isinstance(custom_headers, Mapping) or custom_headers:
        raise error(
            "this OpenAI SDK client carries custom default headers, which may "
            "override credentials or account routing without appearing in adapter "
            "settings. Inject a client with no custom default headers"
        )
    if not isinstance(custom_query, Mapping) or custom_query:
        raise error(
            "this OpenAI SDK client carries custom default query parameters, which "
            "would change the request without appearing in adapter settings. Inject "
            "a client with no custom default query parameters"
        )

    http_client = getattr(client, "_client", None)
    if not isinstance(http_client, httpx.Client):
        raise error(
            "this OpenAI SDK client does not expose the expected synchronous httpx "
            "client, so its wire defaults cannot be validated"
        )
    with httpx.Client(trust_env=False) as pristine_http_client:
        if http_client.headers != pristine_http_client.headers:
            raise error(
                "this OpenAI SDK client's underlying httpx client carries non-default "
                "headers, which may override credentials or account routing without "
                "appearing in adapter settings. Inject a client with default httpx "
                "headers"
            )
        event_hooks = getattr(http_client, "event_hooks", None)
        pristine_event_hooks = pristine_http_client.event_hooks
        malformed_event_hooks = type(event_hooks) is not type(pristine_event_hooks)
        hooks_by_type: dict[str, list[object]] = {}
        if not malformed_event_hooks:
            hooks_by_type = cast("dict[str, list[object]]", event_hooks)
            malformed_event_hooks = hooks_by_type.keys() != pristine_event_hooks.keys()
        if not malformed_event_hooks:
            for hook_type, pristine_hooks in pristine_event_hooks.items():
                hooks = hooks_by_type[hook_type]
                if type(hooks) is not type(pristine_hooks) or hooks:
                    malformed_event_hooks = True
                    break
        if malformed_event_hooks:
            raise error(
                "this OpenAI SDK client's underlying httpx client carries "
                "non-default or malformed event hooks, which could change a request "
                "after admission or a response before parsing. Inject a client with "
                "pristine empty httpx event hooks"
            )
    if http_client.params != httpx.QueryParams():
        raise error(
            "this OpenAI SDK client's underlying httpx client carries default query "
            "parameters, which would change the request without appearing in adapter "
            "settings. Inject a client with no httpx default query parameters"
        )
    if http_client.auth is not None:
        raise error(
            "this OpenAI SDK client's underlying httpx client carries authentication, "
            "which may add an unrecorded Authorization header. Inject a client with "
            "no httpx-level authentication"
        )
    if http_client.cookies:
        raise error(
            "this OpenAI SDK client's underlying httpx client carries cookies, which "
            "may add unrecorded routing or authentication state. Inject a client with "
            "no httpx cookies"
        )
    if type(client.max_retries) is not int or client.max_retries != max_retries:
        raise error(
            "the OpenAI SDK retry loop must be disabled with the exact integer "
            f"max_retries={max_retries}"
        )


__all__ = ["check_xai_openai_compat_client_state"]
