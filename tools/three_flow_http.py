"""Explicit-capability HTTP SDK bridge; no credentials, approvals or default wire.

Candidate settings are not price/endpoint admission. One SDK attempt per turn,
no artificial wall deadline. Successful raw bodies are owner-only captures;
non-success HTTP responses publish only a labelled, body-free status projection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import replace
from importlib.metadata import version
from types import FunctionType, SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import openai
from anthropic.types import Message
from mistralai.client import Mistral
from mistralai.client.httpclient import HttpClient

from operatebench.agents import anthropic_messages as aa
from operatebench.agents import mistral_chat as ma
from operatebench.agents import openai_responses as oa
from operatebench.agents.lifecycle_contract import check_model_request, outcome_tools
from operatebench.agents.model import MAX_OUTPUT_TOKENS, MODEL_PROTOCOL_VERSION
from operatebench.agents.transport import ModelRequest, ModelResponse
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers import anthropic_messages as ap
from operatebench.providers import meta_responses as metap
from operatebench.providers import mistral_chat as mp
from operatebench.providers import openai_responses as op
from operatebench.providers.cost import request_input_token_bound
from operatebench.providers.faults import AdapterProviderError, SingleFlight
from operatebench.providers.wire import WireResponse, wire_invalid
from tools.aggregate_budget import SharedGuard
from tools.three_flow_errors import ProviderPacer, http_error
from tools.three_flow_meta import validate_meta_profile
from tools.three_flow_profiles import (
    ENDPOINTS as ENDPOINTS,
)
from tools.three_flow_profiles import (
    LATEST_HTTP_MODELS as LATEST_HTTP_MODELS,
)
from tools.three_flow_profiles import (
    MODE_PROFILES,
    CampaignSelection,
    http_mode_fields,
    http_reasoning_mode,
    selection_for,
)
from tools.three_flow_profiles import (
    MODELS as MODELS,
)
from tools.three_flow_profiles import (
    minimum_effort as minimum_effort,
)

MISTRAL_PACER = ProviderPacer()


def explicit_sdk_client(cls: Any, **kwargs: Any) -> Any:
    """Run the pinned SDK constructor with an empty constructor environment.

    These SDKs unconditionally consult custom-header environment variables even
    with explicit credentials. Clone only the constructor's globals; do not
    mutate process environment or SDK module globals shared by other threads.
    SDK code, defaults, closure, request serializer and parser remain unchanged.
    No credential discovery is used: api_key and base_url are explicit.
    """
    original = cls.__init__
    constructor = FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "os": SimpleNamespace(environ={}),
            "_warn_env_shadow": lambda **_kwargs: None,
        },
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    constructor.__kwdefaults__ = original.__kwdefaults__
    instance = cls.__new__(cls)
    constructor(instance, **kwargs)
    return instance


class BudgetWire(httpx.BaseTransport):
    def __init__(
        self, inner: httpx.BaseTransport, guard: SharedGuard, endpoint: str
    ) -> None:
        self.inner, self.guard, self.endpoint = inner, guard, endpoint
        self.captures: list[dict[str, Any]] = []
        self.expected_payload: dict[str, Any] | None = None
        self.local_failure: str | None = None
        self.pacer: ProviderPacer | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.pacer is not None:
            return self.pacer.call(lambda: self._handle_request(request))
        return self._handle_request(request)

    def _handle_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) != self.endpoint or request.method != "POST":
            raise ValueError("noncanonical provider endpoint")
        if self.guard._allows_no_money_cap:
            # Mistral 2.x substitutes 300000ms for timeout_ms=None. Override at
            # the real HTTP transport boundary, not with a large deadline.
            # Only the explicit successor policy reaches this branch; old capped
            # requests keep their SDK timeout interpretation byte-for-byte.
            request.extensions["timeout"] = dict.fromkeys(
                ("connect", "read", "write", "pool"), None
            )
        raw = request.read()
        body = json.loads(raw)
        expected = dict(self.expected_payload or {})
        if self.endpoint == ENDPOINTS["mistral"]:
            expected["stream"] = False
        if body != expected:
            raise ValueError("SDK wire differs from exact prepared request")
        if getattr(self.guard, "diagnostic", False):
            cast(Any, self.guard).check_context(
                request_input_token_bound(body, "actual SDK context"),
                body.get("max_output_tokens", body.get("max_tokens")),
            )
        try:
            self.guard.budget.dispatch(self.guard.cell, body)
        except Exception:
            # This trusted local seam executes before the inner wire. SDKs may
            # wrap its exception as a connection error; retain its actual cause.
            if self.guard.budget.broken:
                self.local_failure = "dispatch_journal_error"
            raise
        row: dict[str, Any] = {
            "request_utf8": raw.decode(),
            "request_sha256": hashlib.sha256(raw).hexdigest(),
        }
        self.captures.append(row)
        # Persisted by the trial observer before crossing the actual transport.
        if self.observer is not None:
            self.observer({"kind": "wire_request", **row})
        if hasattr(self.guard.budget, "raw_rpc"):
            self.guard.budget.raw_rpc(
                self.guard.cell, self.endpoint, row["request_sha256"]
            )
        response = self.inner.handle_request(request)
        data = response.read()
        row["status_code"] = response.status_code
        if response.is_success:
            # Successful bodies retain their exact binding and SDK/replay path.
            row.update(
                response_utf8=data.decode(),
                response_sha256=hashlib.sha256(data).hexdigest(),
            )
        else:
            # Provider text (even nested/escaped error fields or headers) can
            # reflect credentials. Publish only this bounded typed projection,
            # BEFORE any observer sees the response. No original-body digest or
            # reversible copy is retained. This is not a provider fault code.
            operation = {
                ENDPOINTS[k]: v
                for k, v in (
                    ("openai", "responses"),
                    ("meta", "responses"),
                    ("anthropic", "messages"),
                    ("mistral", "chat_completions"),
                )
            }[self.endpoint]
            row["response_projection"] = http_error(
                response.status_code, operation, response.headers.get("Retry-After")
            )
        # The untouched response remains available to the actual SDK in memory.
        if self.observer is not None:
            self.observer({"kind": "wire_response", **row})
        return response

    observer: Any = None

    def close(self) -> None:
        self.inner.close()


class HTTPCampaignTransport:
    @property
    def local_failure(self) -> str | None:
        return self.wire.local_failure or (
            "budget_journal_error" if self.guard.budget.broken else None
        )

    def __init__(
        self,
        *,
        provider: str,
        model: str | None = None,
        api_key: str,
        inner: httpx.BaseTransport,
        guard: SharedGuard,
        max_output_tokens: int,
        network_timeout: float | None = None,
        context_tokens: int = 256000,
        admitted_cache_envelope: bool = False,
        mode_profile: str = "legacy-v1",
        transport_policy: str = "legacy-v1",
    ) -> None:
        if transport_policy not in ("legacy-v1", "paced-safe-errors-v1"):
            raise ValueError("unknown transport policy")
        if transport_policy != "legacy-v1" and provider != "mistral":
            raise ValueError("pacing policy is Mistral-only")
        if mode_profile not in MODE_PROFILES:
            raise ValueError("unknown reasoning mode profile")
        meta = provider == "meta"
        if meta:
            metap.check_model(cast(str, model))
            metap.check_key(api_key)
            validate_meta_profile(cast(Any, guard).profile, provider, cast(str, model))
            if max_output_tokens != context_tokens:
                raise ValueError("Meta initial envelope must equal admitted context")
            if (
                mode_profile != "off-or-minimum-v1"
                or network_timeout is not None
                or not getattr(guard, "diagnostic", False)
                or context_tokens != 1048576
            ):
                raise ValueError(
                    "Meta requires diagnostic context and unbounded generation"
                )
        model = MODELS.get(provider) if model is None else model
        if (
            not isinstance(model, str)
            or (provider not in MODELS and not meta)
            or (not meta and selection_for(provider, model, "off-or-minimum-v1") is None)
        ):
            raise ValueError("unsupported HTTP provider/model pair")
        selection = (
            CampaignSelection(
                "meta", model, "responses", mode_profile, "tools/three_flow_http.py"
            )
            if meta
            else selection_for(provider, model, mode_profile)
        )
        if selection is None:
            raise ValueError("new models require explicit off-or-minimum profile")
        self.mode_profile = mode_profile
        self.mode_fields = (
            {"reasoning": {"effort": "minimal"}}
            if meta
            else http_mode_fields(provider, model, mode_profile)
        )
        if provider not in MODELS and not meta:
            raise ValueError("unsupported HTTP provider")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("explicit output profile required")
        if provider not in ("mistral", "meta") and max_output_tokens != 128000:
            raise ValueError("successor flagship output profile is 128000")
        if guard._max_output_tokens != max_output_tokens:
            raise ValueError("guard and request output differ")
        self.admitted_cache_envelope = admitted_cache_envelope
        self.network_timeout = network_timeout
        self.context_tokens = context_tokens
        self.provider, self.model = provider, model
        self.selection = selection
        self.api = selection.api
        self.settings_source = selection.settings_source
        self.request_mapping = "three-flow-" + provider + "-v1"
        if mode_profile != "legacy-v1":
            self.request_mapping += "-" + mode_profile
        if meta:
            self.request_mapping = "three-flow-meta-snapshot-minimal-auto-v3"
        self.guard, self.max_output_tokens = guard, max_output_tokens
        self.wire = BudgetWire(inner, guard, ENDPOINTS[provider])
        if transport_policy == "paced-safe-errors-v1":
            self.wire.pacer = MISTRAL_PACER
            self.request_mapping += "-" + transport_policy
        self.http = (
            ap.anthropic_http_client(transport=self.wire, timeout=network_timeout)
            if provider == "anthropic"
            else httpx.Client(
                transport=self.wire,
                trust_env=False,
                timeout=httpx.Timeout(None, connect=10, pool=10)
                if meta
                else network_timeout,
                follow_redirects=False,
            )
        )
        self.client: Any
        if provider in ("openai", "meta"):
            self.client = explicit_sdk_client(
                openai.OpenAI,
                api_key=api_key,
                base_url=metap.BASE_URL if meta else "https://api.openai.com/v1",
                organization=None,
                project=None,
                max_retries=0,
                timeout=self.http.timeout if meta else network_timeout,
                http_client=self.http,
            )
        elif provider == "anthropic":
            self.client = explicit_sdk_client(
                anthropic.Anthropic,
                api_key=api_key,
                base_url="https://api.anthropic.com",
                max_retries=0,
                timeout=network_timeout,
                http_client=self.http,
            )
        else:
            assert isinstance(self.http, httpx.Client)
            quiet = logging.Logger("three-flow-sdk-disabled")
            quiet.disabled = True
            self.client = Mistral(
                api_key=api_key,
                server_url=mp.MISTRAL_BASE_URL,
                client=cast("HttpClient", self.http),
                retry_config=mp._no_sdk_retries(),
                debug_logger=quiet,
            )
        if provider == "mistral":
            self.client.sdk_configuration.__dict__["telemetry"] = False
        if meta:
            metap.check_client(self.client)
        self.settings = {
            "sdk": provider,
            "sdk_version": version(
                "openai" if meta else "mistralai" if provider == "mistral" else provider
            ),
            "endpoint": ENDPOINTS[provider],
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "max_output_tokens": max_output_tokens,
            "telemetry": False,
            "max_retries": 0,
            "credential_source": "explicit capability; constructor environment disabled",
            "timeout": network_timeout,
            "tool_choice": {"type": "auto"}
            if provider == "anthropic"
            else "auto"
            if meta
            else "required",
            "model_contract_verified": False,
            "admission": False,
            "evidence_origin": "SDK_MOCK_NOT_LLM"
            if isinstance(inner, httpx.MockTransport)
            else "PROVIDER_CANDIDATE",
        }
        if provider == "anthropic":
            self.settings.update(
                thinking={"type": "adaptive"}, output_config={"effort": "high"}
            )
        elif provider in ("openai", "meta"):
            self.settings.update(
                reasoning_items_contract=op.OPENAI_REASONING_ITEMS_CONTRACT,
                store=False,
                parallel_tool_calls=False,
                omitted=["temperature", "reasoning", "top_p", "seed"],
            )
        else:
            self.settings.update(
                temperature=0.0,
                parallel_tool_calls=False,
                context_candidate=context_tokens,
                output_policy=(
                    "min(explicit maximum, context minus conservative input bound)"
                ),
            )
        if mode_profile != "legacy-v1":
            self.settings.update(self.mode_fields)
            self.settings.update(
                mode_profile=mode_profile,
                reasoning_mode="MINIMUM" if meta else http_reasoning_mode(model),
            )
            if provider in ("openai", "meta"):
                self.settings["omitted"] = ["temperature", "top_p", "seed"]
        if transport_policy != "legacy-v1":
            self.settings.update(
                transport_policy=transport_policy,
                pacing_scope="process-wide-mistral",
                minimum_spacing_seconds=1,
                retry_after_max_seconds=86400,
                retry_eligibility="none-proven",
                automatic_resubmission=False,
            )
        if meta:
            self.settings.pop("reasoning_items_contract", None)
            self.settings.update(
                sdk="openai",
                provider="meta",
                comparison="equal-scaffold independent snapshots",
                native_reasoning_continuation=False,
                tool_strict_mode=False,
                bounds_policy=cast(Any, guard).profile["bounds_policy"],
                context_request_bound=context_tokens,
                dedicated_output_max=None,
                connection_timeout_seconds=10,
                pool_timeout_seconds=10,
                billing_assurance="estimated usage exposure; not invoice guarantee",
                omitted=[
                    "temperature",
                    "top_p",
                    "seed",
                    "include",
                    "previous_response_id",
                ],
            )
        self.last_turn: dict[str, Any] | None = None
        self.flight = SingleFlight(adapter="three-flow HTTP")

    def prepare_request(self, request: ModelRequest) -> ModelRequest:
        if not getattr(self.guard, "diagnostic", False):
            return request
        # Count complete SDK-emitted defaults before selecting output. The
        # envelope's integer has at least as many digits as the chosen value.
        payload = self.payload(replace(request, max_output_tokens=self.max_output_tokens))
        bound = request_input_token_bound(
            self.wire_payload(payload), "complete SDK projection"
        )
        output = cast(Any, self.guard).output_bound(bound)
        return replace(request, max_output_tokens=min(output, self.max_output_tokens))

    def wire_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(payload, stream=False) if self.provider == "mistral" else payload

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        check_model_request(
            replace(request, max_output_tokens=MAX_OUTPUT_TOKENS), model=self.model
        )
        diagnostic = getattr(self.guard, "diagnostic", False)
        if (
            not diagnostic and request.max_output_tokens != self.max_output_tokens
        ) or not (
            type(request.max_output_tokens) is int
            and 0 < request.max_output_tokens <= self.max_output_tokens
        ):
            raise ValueError("request output differs from bound settings")
        text = canonical_json_text(request.prompt, "campaign prompt")
        if self.provider in ("openai", "meta"):
            return {
                "model": self.model,
                "max_output_tokens": request.max_output_tokens,
                "instructions": (
                    oa.request_instructions(request).replace(
                        oa.LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION, self.request_mapping
                    )
                    if self.provider == "meta"
                    else oa.request_instructions(request)
                ),
                "tools": outcome_tools(),
                "tool_choice": "auto" if self.provider == "meta" else "required",
                "parallel_tool_calls": False,
                "store": False,
                "input": text,
                **self.mode_fields,
            }
        if self.provider == "anthropic":
            return {
                "model": self.model,
                "max_tokens": request.max_output_tokens,
                "system": aa.request_instructions(request),
                "tools": aa.anthropic_outcome_tools(),
                "tool_choice": {"type": "auto"},
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": text}],
                **self.mode_fields,
            }
        payload = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "tools": [
                {"type": "function", "function": t} for t in ma.mistral_outcome_tools()
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": ma.request_instructions(request)},
                {"role": "user", "content": text},
            ],
        }
        payload.update(self.mode_fields)
        if diagnostic:
            return payload
        bound = request_input_token_bound(payload, "Mistral context")
        output = min(self.max_output_tokens, self.context_tokens - bound)
        if output <= 0:
            raise ValueError("context exhausted before dispatch")
        payload["max_tokens"] = output
        return payload

    def _dispatch(self, payload: dict[str, Any]) -> tuple[Any, ModelResponse]:
        if self.provider in ("openai", "meta"):
            if self.provider == "meta":
                metap.check_client(self.client)
            raw = self.client.responses.with_raw_response.create(
                **payload,
                timeout=self.http.timeout
                if self.provider == "meta"
                else self.network_timeout,
            )
            wire = WireResponse(raw.text, raw.parse, kind=self.provider + " campaign")
            if self.provider == "meta":
                meta_projection = metap.check_response(wire, model=self.model)
            else:
                op.check_response_admissible(wire, model=self.model)
                meta_projection = wire
            return op.response_usage(wire), oa.model_response_from(meta_projection)
        if self.provider == "anthropic":
            raw = self.client.messages.with_raw_response.create(
                **ap.anthropic_sdk_kwargs(payload), timeout=self.network_timeout
            )
            wire = WireResponse(raw.text(), raw.parse, kind="Fable campaign")
            # Validate hidden reasoning blocks, then use the unchanged legacy
            # validator on the semantic text/tool projection. Full raw retained.
            body = dict(wire.wire)
            content = body.get("content")
            if not isinstance(content, list):
                raise wire_invalid("Fable content is not a list")
            semantic = []
            for block in content:
                if not isinstance(block, dict):
                    raise wire_invalid("invalid Fable content block")
                if block.get("type") == "thinking":
                    if set(block) != {"type", "thinking", "signature"} or not all(
                        isinstance(block[k], str) for k in ("thinking", "signature")
                    ):
                        raise wire_invalid("invalid thinking block")
                elif block.get("type") == "redacted_thinking":
                    if set(block) != {"type", "data"} or not isinstance(
                        block["data"], str
                    ):
                        raise wire_invalid("invalid hidden thinking block")
                else:
                    semantic.append(block)
            body["content"] = semantic
            if self.admitted_cache_envelope:
                # Preserve raw capture. Validate each category, then project total
                # billable input under the externally admitted worst-tier rate.
                raw_usage = body.get("usage")
                if not isinstance(raw_usage, dict):
                    raise wire_invalid("invalid billable usage object")
                u = dict(raw_usage)
                counts = [
                    u.get("input_tokens"),
                    *(u.get(k, 0) for k in ap.CACHE_USAGE_FIELDS),
                ]
                if any(type(v) is not int or v < 0 for v in counts):
                    raise wire_invalid("invalid billable input categories")
                u["input_tokens"] = sum(counts)
                for k in ap.CACHE_USAGE_FIELDS:
                    if k in u:
                        u[k] = 0
                body["usage"] = u
            projected = WireResponse(
                json.dumps(body),
                lambda: Message.model_validate(body),
                kind="Fable semantic projection",
            )
            ap.check_response_admissible(projected, model=self.model)
            usage = ap.response_usage(projected)
            if body["stop_reason"] in ("end_turn", "refusal"):
                # Valid refusal/no action is model behavior, not a provider outage.
                return usage, ModelResponse(
                    model=self.model,
                    stop_reason="completed",
                    output_tokens=usage.output_tokens or 0,
                )
            return usage, aa.model_response_from(projected)
        response = self.client.chat.complete(
            **payload,
            timeout_ms=None
            if self.network_timeout is None
            else int(self.network_timeout * 1000),
        )
        wire = WireResponse(
            self.wire.captures[-1]["response_utf8"],
            lambda: response,
            kind="Mistral campaign",
        )
        mp.check_response_admissible(wire, model=self.model)
        usage = mp.response_usage(wire)
        choices = wire.parsed.choices or []
        if (
            len(choices) == 1
            and choices[0].finish_reason == "stop"
            and not choices[0].message.tool_calls
        ):
            return usage, ModelResponse(
                model=self.model,
                stop_reason="completed",
                output_tokens=usage.output_tokens or 0,
            )
        return usage, ma.model_response_from(wire)

    def send(self, request: ModelRequest) -> ModelResponse:
        with self.flight:
            self.last_turn = None
            payload = self.payload(request)
            self.wire.expected_payload = payload
            self.guard._max_output_tokens = int(
                payload["max_output_tokens"]
                if self.provider in ("openai", "meta")
                else payload["max_tokens"]
            )
            reservation = self.guard.authorize(
                input_tokens_upper_bound=request_input_token_bound(
                    self.wire_payload(payload), "campaign request"
                )
            )
            started = time.monotonic()
            try:
                usage, response = self._dispatch(payload)
                measured = self.guard.settle(
                    reservation,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                self.last_turn = {
                    "requested_output_bound": request.max_output_tokens,
                    "input_bound": request_input_token_bound(
                        self.wire_payload(payload), "SDK input bound"
                    ),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "measured_usd": None if measured is None else str(measured),
                    "latency_seconds": time.monotonic() - started,
                    "wire": dict(self.wire.captures[-1]),
                }
            except BaseException as exc:
                if self.guard.ticket is not None and not self.guard.budget.broken:
                    self.guard.forfeit(reservation)
                fault = None
                if isinstance(exc, AdapterProviderError):
                    fault = exc.fault
                elif isinstance(exc, Exception):
                    module = {
                        "openai": op,
                        "meta": metap,
                        "anthropic": ap,
                        "mistral": mp,
                    }[self.provider]
                    classified = module.classify_exception(exc)
                    fault = None if classified is None else classified.fault
                self.last_turn = {
                    "classification": "internal_error"
                    if self.local_failure is not None
                    else fault or "internal_error",
                    "measured_usd": None,
                    "latency_seconds": time.monotonic() - started,
                }
                if getattr(self.guard, "diagnostic", False) and self.wire.captures:
                    status = self.wire.captures[-1].get("status_code")
                    if status in (400, 422):
                        self.last_turn.update(
                            diagnostic_category="provider_request_refusal",
                            safe_status=status,
                            automatic_resubmission=False,
                        )
                if fault is not None and self.local_failure is None:
                    raise AdapterProviderError(
                        fault, "provider operation failed"
                    ) from None
                raise
            finally:
                self.http.cookies.clear()
            if self.provider == "meta" and not response.tool_calls:
                raise wire_invalid("Meta auto response has no Core function call")
            return response

    def close(self) -> None:
        self.http.close()
