"""Native Gemini Developer API mapping; live wire requires explicit admission.

REST preserves parametersJsonSchema and exact response JSON without SDK
coercion. The existing BudgetWire owns dispatch, durable captures and liability.
The default constructor is offline-only. No environment credential discovery.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import httpx

from operatebench.agents.lifecycle_contract import (
    INSTRUCTIONS_CONTRACT,
    check_model_request,
    elide_null_optionals,
    outcome_tools,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, MODEL_PROTOCOL_VERSION
from operatebench.agents.playback import decision_record
from operatebench.agents.transport import ModelRequest, ModelResponse, ToolCall
from operatebench.core.protocol import AgentObservation
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.cost import CostCapExceededError, request_input_token_bound
from operatebench.providers.faults import AdapterProviderError, SingleFlight
from operatebench.providers.usage import MAX_EXACT_TOKEN_COUNT
from operatebench.providers.wire import wire_invalid, wire_json_object
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.aggregate_budget import SharedGuard
from tools.three_flow_http import BudgetWire
from tools.three_flow_profiles import (
    GEMINI_ROSTER_PROFILE,
)
from tools.three_flow_profiles import (
    GEMINI_UNCAPPED_PROFILE as LIMIT_PROFILE,
)

ROSTER_PROFILE = GEMINI_ROSTER_PROFILE
MAPPING = "three-flow-gemini-developer-native-low-v2"
OUTPUT_POLICY = "gemini-complete-input-context-intersection-v1"


@dataclass(frozen=True)
class GeminiProfile:
    model: str
    input_rate: str
    output_rate: str
    price_valid_through: str
    context: int = 1048576
    output: int = 65536
    thinking: str = "low"

    @property
    def endpoint(self) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def settings(self, maximum: int) -> dict[str, Any]:
        if type(maximum) is not int or maximum != self.output:
            raise ValueError("nominal Gemini physical output maximum required")
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "max_output_tokens": maximum,
            "context_request_bound": self.context,
            "context_bound_basis": "conservative operational envelope",
            "output_reservation_policy": OUTPUT_POLICY,
            "bounds_policy": OUTPUT_POLICY,
            "physical_output_maximum": self.output,
            "model_source": f"https://ai.google.dev/gemini-api/docs/models/{self.model}",
            "native_schema_source": "https://ai.google.dev/api/generate-content#FunctionDeclaration",
            "signature_source": "https://ai.google.dev/gemini-api/docs/function-calling",
            "quota_admission": "absent; live transport forbidden",
            "thinking": self.thinking,
            "generationConfig": {
                "maxOutputTokens": maximum,
                "thinkingConfig": {"thinkingLevel": self.thinking},
            },
            "mapping": MAPPING,
            "settings_source": "tools/three_flow_gemini.py",
            "max_retries": 0,
            "automatic_resubmission": False,
            "admission": False,
            "evidence_origin": "SDK_MOCK_NOT_LLM",
            "credential_source": "none; offline mock only",
            "pricing_basis": {
                "service": "Standard",
                "verified_on": "2026-10-14",
                "input_usd_per_mtok": self.input_rate,
                "output_usd_per_mtok": self.output_rate,
                "valid_through": self.price_valid_through,
                "thinking_included": True,
                "implicit_cache": "reported; conservatively priced at full input rate",
                "tool_use_prompt_tokens": (
                    "reported separately; estimated at full input rate"
                ),
                "unknown_additional_fees": "unknown; not zero",
                "accounting_policy": "estimated-usage-not-invoice",
                "source": "https://ai.google.dev/gemini-api/docs/pricing",
            },
        }

    def live_settings(self, maximum: int) -> dict[str, Any]:
        """Direct-wire provenance; this is not launch authority."""
        value = self.settings(maximum)
        value.update(
            quota_admission="external launcher required; transport is not authority",
            evidence_origin="PROVIDER_CANDIDATE",
            credential_source="explicit header only",
        )
        return value


MODELS = MappingProxyType(
    {
        "gemini-4-argon": GeminiProfile("gemini-4-argon", "2", "10", "2026-11-30"),
        "gemini-3.8-flash": GeminiProfile(
            "gemini-3.8-flash", "0.75", "3.75", "2026-12-31"
        ),
    }
)


def declarations() -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parametersJsonSchema": t["parameters"],
        }
        for t in outcome_tools()
    ]


def native_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    part = payload["contents"][-1]["parts"][0]
    if "functionResponse" in part:
        return dict(part["functionResponse"]["response"])
    return dict(wire_json_object(part["text"], kind="Gemini prompt"))


def parse_response(
    raw: str, model: str
) -> tuple[ModelResponse, dict[str, int], dict[str, Any]]:
    body = wire_json_object(raw, kind="Gemini response")
    if body.get("modelVersion") != model:
        raise wire_invalid("Gemini exact model identity missing or different")
    usage = body.get("usageMetadata")
    if not isinstance(usage, dict):
        raise wire_invalid("Gemini usage missing")
    counts = {}
    optional_usage = (
        "cachedContentTokenCount",
        "thoughtsTokenCount",
        "toolUsePromptTokenCount",
    )
    for field in (
        "promptTokenCount",
        "candidatesTokenCount",
        "thoughtsTokenCount",
        "toolUsePromptTokenCount",
        "totalTokenCount",
        "cachedContentTokenCount",
    ):
        # Absent optional usage categories mean zero; cache is a prompt subset.
        default = 0 if field in optional_usage else None
        value = usage.get(field, default)
        if type(value) is not int or not 0 <= value <= MAX_EXACT_TOKEN_COUNT:
            raise wire_invalid("Gemini usage count invalid or missing")
        counts[field] = value
    if (
        counts["cachedContentTokenCount"] > counts["promptTokenCount"]
        or counts["totalTokenCount"]
        != counts["promptTokenCount"]
        + counts["toolUsePromptTokenCount"]
        + counts["candidatesTokenCount"]
        + counts["thoughtsTokenCount"]
    ):
        raise wire_invalid("Gemini usage categories inconsistent")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise wire_invalid("Gemini requires one candidate")
    candidate = candidates[0]
    if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
        raise wire_invalid("Gemini incomplete or refused response")
    content = candidate.get("content")
    if not isinstance(content, dict) or content.get("role") != "model":
        raise wire_invalid("Gemini model content missing")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise wire_invalid("Gemini parts missing")
    calls = []
    for part in parts:
        if not isinstance(part, dict) or set(part) - {
            "text",
            "thought",
            "thoughtSignature",
            "functionCall",
        }:
            raise wire_invalid("Gemini unsupported content part")
        if "thoughtSignature" in part and (
            not isinstance(part["thoughtSignature"], str) or not part["thoughtSignature"]
        ):
            raise wire_invalid("Gemini invalid thought signature")
        if "thought" in part and type(part["thought"]) is not bool:
            raise wire_invalid("Gemini invalid thought")
        if "text" in part and not isinstance(part["text"], str):
            raise wire_invalid("Gemini invalid text")
        if "functionCall" in part:
            call = part["functionCall"]
            if not isinstance(call, dict) or set(call) - {"name", "args", "id"}:
                raise wire_invalid("Gemini invalid function call")
            if "id" in call and not isinstance(call["id"], str):
                raise wire_invalid("Gemini invalid function call id")
            if not isinstance(call.get("name"), str) or not isinstance(
                call.get("args"), dict
            ):
                raise wire_invalid("Gemini invalid native arguments")
            if not isinstance(part.get("thoughtSignature"), str):
                raise wire_invalid(
                    "Gemini function call requires preserved thought signature"
                )
            calls.append(
                ToolCall(call["name"], elide_null_optionals(call["name"], call["args"]))
            )
    if len(calls) != 1:
        raise wire_invalid("Gemini requires exactly one Core function call")
    output = counts["candidatesTokenCount"] + counts["thoughtsTokenCount"]
    return (
        ModelResponse(
            model=model,
            stop_reason="completed",
            tool_calls=tuple(calls),
            output_tokens=output,
        ),
        counts,
        content,
    )


class GeminiCampaignTransport:
    provider = "gemini"
    api = "generate_content"
    request_mapping = MAPPING

    @classmethod
    def live(
        cls,
        *,
        model: str,
        api_key: str,
        guard: SharedGuard,
        max_output_tokens: int,
        admitted: bool = False,
        inner: httpx.BaseTransport | None = None,
    ) -> GeminiCampaignTransport:
        """Launcher-owned admission is required; this is not launch authority."""
        if admitted is not True:
            raise ValueError("explicit Gemini live admission required")
        if (
            model not in MODELS
            or getattr(guard.budget, "controller_profile", None) != LIMIT_PROFILE
            or guard.budget.cap is not None
        ):
            raise ValueError("Gemini live guard/profile admission differs")
        if (
            type(api_key) is not str
            or not api_key
            or not api_key.isascii()
            or not api_key.isprintable()
            or api_key != api_key.strip()
        ):
            raise ValueError("nonempty explicit Gemini key required")
        if inner is not None and not isinstance(inner, httpx.MockTransport):
            raise ValueError("only mock injection or default direct HTTP transport")
        direct = inner is None
        instance = cls.__new__(cls)
        instance._initialize(
            model=model,
            inner=inner
            if inner is not None
            else httpx.HTTPTransport(trust_env=False, retries=0),
            guard=guard,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            direct=direct,
        )
        return instance

    def __init__(
        self,
        *,
        model: str,
        inner: httpx.BaseTransport,
        guard: SharedGuard,
        max_output_tokens: int,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError("exact stable Gemini model required")
        self.profile = MODELS[model]
        if api_key is not None or not isinstance(inner, httpx.MockTransport):
            raise ValueError("offline Gemini accepts no credential or live transport")
        if endpoint is not None and endpoint != self.profile.endpoint:
            raise ValueError("unauthorized Gemini endpoint")
        self._initialize(
            model=model,
            inner=inner,
            guard=guard,
            max_output_tokens=max_output_tokens,
        )

    def _initialize(
        self,
        *,
        model: str,
        inner: httpx.BaseTransport,
        guard: SharedGuard,
        max_output_tokens: int,
        api_key: str | None = None,
        direct: bool = False,
    ) -> None:
        self.profile = MODELS[model]
        if guard._max_output_tokens != max_output_tokens:
            raise ValueError("Gemini guard/output binding differs")
        if guard.policy.input_usd_per_mtok != Decimal(
            self.profile.input_rate
        ) or guard.policy.output_usd_per_mtok != Decimal(self.profile.output_rate):
            raise ValueError("Gemini Standard pricing differs")
        self.model, self.guard, self.max_output_tokens = model, guard, max_output_tokens
        self.settings = (
            self.profile.live_settings(max_output_tokens)
            if direct
            else self.profile.settings(max_output_tokens)
        )
        self.settings_source = self.settings["settings_source"]
        self.wire = BudgetWire(
            inner, guard, self.profile.endpoint, operation="generate_content"
        )
        self.wire.sensitive_response = api_key.encode() if api_key is not None else None
        self.http = httpx.Client(
            transport=self.wire,
            trust_env=False,
            timeout=(
                httpx.Timeout(None, connect=10, write=10, pool=10)
                if api_key is not None
                else None
            ),
            follow_redirects=False,
            headers=({"x-goog-api-key": api_key} if api_key is not None else None),
        )
        self.history: list[dict[str, Any]] = []
        self.last_turn: dict[str, Any] | None = None
        self.flight = SingleFlight(adapter="Gemini")
        self.stopped = False

    @property
    def local_failure(self) -> str | None:
        return self.wire.local_failure or (
            "budget_journal_error" if self.guard.budget.broken else None
        )

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        check_model_request(
            replace(request, max_output_tokens=MAX_OUTPUT_TOKENS), model=self.model
        )
        if not 0 < request.max_output_tokens <= self.max_output_tokens:
            raise ValueError("Gemini request/output binding differs")
        history = copy.deepcopy(self.history)
        if history:
            history[-1]["parts"][0]["functionResponse"]["response"] = dict(request.prompt)
        else:
            history = [
                {
                    "role": "user",
                    "parts": [
                        {"text": canonical_json_text(request.prompt, "Gemini prompt")}
                    ],
                }
            ]
        return {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            f"{INSTRUCTIONS_CONTRACT}\nrequest_mapping: {MAPPING}\n"
                            f"protocol_version: {request.protocol_version}\n"
                            f"request_digest_sha256: {request.request_digest_sha256}"
                        )
                    }
                ]
            },
            "contents": history,
            "tools": [{"functionDeclarations": declarations()}],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": list(request.tool_names),
                }
            },
            "generationConfig": {
                **copy.deepcopy(self.settings["generationConfig"]),
                "maxOutputTokens": request.max_output_tokens,
            },
        }

    def bounded_payload(self, request: ModelRequest) -> tuple[dict[str, Any], int, int]:
        payload = self.payload(request)
        payload["generationConfig"]["maxOutputTokens"] = self.max_output_tokens
        # Count the complete nominal envelope before reducing its numeric ceiling.
        # This remains conservative for the actual bytes (the cap can only shrink).
        bound = request_input_token_bound(payload, "complete Gemini native request")
        maximum = min(self.profile.output, self.profile.context - bound)
        if maximum <= 0:
            raise CostCapExceededError("Gemini context exhausted before dispatch")
        payload["generationConfig"]["maxOutputTokens"] = maximum
        return payload, bound, maximum

    def prepare_request(self, request: ModelRequest) -> ModelRequest:
        _, _, maximum = self.bounded_payload(request)
        return replace(request, max_output_tokens=maximum)

    def send(self, request: ModelRequest) -> ModelResponse:
        with self.flight:
            self.last_turn = None
            if self.stopped:
                raise AdapterProviderError(
                    "provider_request_rejected", "Gemini dispatch stopped; no retry"
                )
            payload, bound, maximum = self.bounded_payload(request)
            if request.max_output_tokens != maximum:
                raise ValueError("Gemini actual request/output binding differs")
            self.guard._max_output_tokens = maximum
            self.wire.expected_payload = payload
            self.wire.current_turn_dispatched = False
            reservation = self.guard.authorize(input_tokens_upper_bound=bound)
            started = time.monotonic()
            try:
                raw = canonical_json_text(payload, "Gemini wire")
                reply = self.http.post(
                    self.profile.endpoint,
                    content=raw.encode(),
                    headers={"Content-Type": "application/json"},
                )
                if not reply.is_success:
                    self.stopped = True
                    fault = (
                        "provider_rate_limited"
                        if reply.status_code == 429
                        else "provider_authentication"
                        if reply.status_code in (401, 403)
                        else "provider_server_error"
                        if reply.status_code >= 500
                        else "provider_request_rejected"
                    )
                    raise AdapterProviderError(fault, "Gemini HTTP operation failed")
                response, counts, content = parse_response(reply.text, self.model)
                # Tool-use prompt is outside promptTokenCount but inside the
                # reported total. Conservatively estimate both at input rate.
                input_tokens = (
                    counts["promptTokenCount"] + counts["toolUsePromptTokenCount"]
                )
                measured = self.guard.settle(
                    reservation,
                    input_tokens=input_tokens,
                    output_tokens=response.output_tokens,
                )
                self.last_turn = {
                    "requested_output_bound": maximum,
                    "output_reservation_policy": OUTPUT_POLICY,
                    "input_bound": bound,
                    "input_tokens": input_tokens,
                    "output_tokens": response.output_tokens,
                    "thinking_tokens": counts["thoughtsTokenCount"],
                    "cached_input_tokens": counts["cachedContentTokenCount"],
                    "usage_metadata": counts,
                    "measured_usd": str(measured),
                    "latency_seconds": time.monotonic() - started,
                    "wire": dict(self.wire.captures[-1]),
                    "network_rpcs": 1,
                }
                # Filled with the next actual Core observation before dispatch.
                call = next(
                    p["functionCall"] for p in content["parts"] if "functionCall" in p
                )
                self.history = [
                    *payload["contents"],
                    copy.deepcopy(content),
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": call["name"],
                                    **({"id": call["id"]} if "id" in call else {}),
                                    "response": {},
                                }
                            }
                        ],
                    },
                ]
                return response
            except BaseException as exc:
                self.stopped = True
                if self.guard.ticket is not None and not self.guard.budget.broken:
                    self.guard.forfeit(reservation)
                if isinstance(exc, httpx.TransportError):
                    raise AdapterProviderError(
                        "provider_network_error", "Gemini operation failed"
                    ) from None
                raise
            finally:
                self.http.cookies.clear()

    def close(self) -> None:
        self.http.close()


class GeminiMockTransport(httpx.MockTransport):
    def __init__(self, model: str, flow: str) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self.reference = (
            build_agent("reference")
            if flow == "maintenance"
            else (
                COMMERCE_COMMANDS if flow == "commerce" else COMPLIANCE_COMMANDS
            ).factories.build_agent("reference")
        )
        self.reference.begin_episode({})
        super().__init__(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        public = native_prompt(body)["observation"]
        decision = decision_record(self.reference.decide(AgentObservation(**public)))
        name = decision.pop("kind").lower()
        if name == "ask":
            decision["wait"].pop("kind")
        return httpx.Response(
            200,
            json={
                "modelVersion": self.model,
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {"name": name, "args": decision},
                                    "thoughtSignature": "offline-signature",
                                }
                            ],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 4,
                    "cachedContentTokenCount": 0,
                },
            },
        )


def offline_limit_policy() -> dict[str, Any]:
    """Synthetic evidence for offline tests only; never owner launch authority."""
    from operatebench.agents.transport import content_digest
    from tools.three_flow_campaign import source_commit

    return {
        "mode": LIMIT_PROFILE,
        "owner_accepted": True,
        "owner_evidence_sha256": content_digest(
            "OFFLINE_NULL_CAP_TEST_NOT_LAUNCH_AUTHORITY"
        ),
        "source_sha": source_commit(),
        "historical_liability_manifest_sha256": content_digest(
            "FRESH_OFFLINE_NO_HISTORICAL_JOURNAL"
        ),
    }


def offline_factory(slot: dict[str, Any], budget: Any) -> GeminiCampaignTransport:
    from operatebench.agents.pricing import LifecyclePricingPolicy

    profile = MODELS[slot["model"]]
    if budget.controller_profile != LIMIT_PROFILE or budget.cap is not None:
        raise ValueError("Gemini offline campaign requires distinct null-cap journal")
    guard = SharedGuard(
        budget=budget,
        cell=slot["trial_id"],
        policy=LifecyclePricingPolicy(
            "gemini-standard-estimated-v1",
            Decimal(profile.input_rate),
            Decimal(profile.output_rate),
        ),
        max_output_tokens=profile.output,
    )
    return GeminiCampaignTransport(
        model=profile.model,
        inner=GeminiMockTransport(profile.model, slot["flow"]),
        guard=guard,
        max_output_tokens=profile.output,
    )


def validate_replay(record: dict[str, Any]) -> None:
    """Rebuild native requests/history and rederive usage and decisions, offline."""
    import hashlib

    from operatebench.agents.model import AGENT_TOOL_NAMES, parse_tool_call
    from operatebench.agents.transport import content_digest

    try:
        model = record["trial"]["model"]
        maximum = record["trial"]["max_output_tokens"]
        profile = MODELS[model]
        provider = record["provider"]
        settings = provider["settings"]
        if settings not in (
            profile.settings(maximum),
            profile.live_settings(maximum),
        ) or provider != {
            "provider": "gemini",
            "api": "generate_content",
            "model": model,
            "request_mapping": MAPPING,
            "settings": settings,
        }:
            raise ValueError("Gemini profile provenance differs")
        history: list[dict[str, Any]] = []
        for turn, digest, decision in zip(
            record["provider_turns"],
            record["request_digests"],
            record["decisions"],
            strict=True,
        ):
            wire = turn["wire"]
            if (
                wire["request_url"] != profile.endpoint
                or wire["request_method"] != "POST"
            ):
                raise ValueError("Gemini native endpoint/model provenance differs")
            for kind in ("request", "response"):
                if (
                    hashlib.sha256(wire[kind + "_utf8"].encode()).hexdigest()
                    != wire[kind + "_sha256"]
                ):
                    raise ValueError("Gemini wire checksum differs")
            if wire["status_code"] != 200:
                raise ValueError("Gemini scored HTTP status differs")
            payload = dict(
                wire_json_object(wire["request_utf8"], kind="Gemini replay request")
            )
            prompt = native_prompt(payload)
            observation = prompt["observation"]
            request = ModelRequest(
                model=model,
                protocol_version=MODEL_PROTOCOL_VERSION,
                max_output_tokens=turn["requested_output_bound"],
                tool_names=AGENT_TOOL_NAMES,
                observation_digest_sha256=content_digest(observation),
                prompt_digest_sha256=content_digest(prompt),
                invocation_index=observation["invocation_index"],
                turn_index=observation["turn_index"],
                prompt=prompt,
            )
            check_model_request(
                replace(request, max_output_tokens=MAX_OUTPUT_TOKENS), model=model
            )
            if request.request_digest_sha256 != digest:
                raise ValueError("Gemini request identity differs")
            # Use the same native mapping without constructing a client or guard.
            mapping = object.__new__(GeminiCampaignTransport)
            mapping.model, mapping.max_output_tokens = model, maximum
            mapping.profile = profile
            mapping.history, mapping.settings = history, profile.settings(maximum)
            expected, bound, actual_maximum = mapping.bounded_payload(request)
            if payload != expected:
                raise ValueError("Gemini native request/history differs")
            response, counts, content = parse_response(wire["response_utf8"], model)
            call = response.tool_calls[0]
            if decision_record(parse_tool_call(call)) != decision["outcome"]:
                raise ValueError("Gemini decision differs from native response")
            input_tokens = counts["promptTokenCount"] + counts["toolUsePromptTokenCount"]
            cost = (
                Decimal(input_tokens) * Decimal(profile.input_rate)
                + Decimal(response.output_tokens) * Decimal(profile.output_rate)
            ) / Decimal(1000000)
            if (
                bound + actual_maximum > profile.context
                or input_tokens > bound
                or response.output_tokens > actual_maximum
                or turn["input_bound"] != bound
                or turn["requested_output_bound"] != actual_maximum
                or turn["output_reservation_policy"] != OUTPUT_POLICY
                or turn["input_tokens"] != input_tokens
                or turn["output_tokens"] != response.output_tokens
                or turn["thinking_tokens"] != counts["thoughtsTokenCount"]
                or turn["cached_input_tokens"] != counts["cachedContentTokenCount"]
                or turn["usage_metadata"] != counts
                or Decimal(turn["measured_usd"]) != cost
            ):
                raise ValueError("Gemini usage/context/cost provenance differs")
            native_call = next(
                p["functionCall"] for p in content["parts"] if "functionCall" in p
            )
            history = [
                *payload["contents"],
                content,
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": native_call["name"],
                                **(
                                    {"id": native_call["id"]}
                                    if "id" in native_call
                                    else {}
                                ),
                                "response": {},
                            }
                        }
                    ],
                },
            ]
    except (KeyError, TypeError, AdapterProviderError) as exc:
        raise ValueError("Gemini native provenance invalid") from exc
