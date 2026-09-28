"""Finite campaign selections, not admission or a second API capability registry.

Candidate mode overrides belong here. Reviewed API capabilities remain the
provider RequestProfile objects; a missing reference is not a new capability.
Settings provenance literals remain historical recorded identities.
No clients, credentials, pricing, budgets, or native parsers are owned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from operatebench.providers import anthropic_messages, mistral_chat, openai_responses

MODELS = {
    "openai": "gpt-6-astra",
    "anthropic": "claude-fable-5-1",
    "mistral": "mistral-medium-3-5",
}
# Explicit successor candidates; legacy MODELS remains frozen for old readers.
# This explicit roster does not select Terra; model identities are not substituted.
LATEST_HTTP_MODELS = (
    ("openai", "gpt-6-luna"),
    ("openai", "gpt-6-sol"),
    ("openai", "gpt-6-astra"),
    ("anthropic", "claude-fable-5-1"),
    ("anthropic", "claude-opus-5-5"),
    ("mistral", "mistral-medium-3-5"),
)


def minimum_effort(model: str) -> str:
    return "none" if model in ("gpt-6-luna", "gpt-6-sol") else "low"


ENDPOINTS = {
    "openai": "https://api.openai.com/v1/responses",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}

RIVER_MODELS = (
    "Qwen/Qwen3.8-27B-FP8",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.5-9B",
    "nvidia/Kimi-K2.6-NVFP4",
    "nvidia/GLM-5.2-NVFP4-262K",
    "zai-org/GLM-5.3-Flash",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
)
ROSTER = tuple(("river", m) for m in RIVER_MODELS) + tuple(MODELS.items())
LATEST_ROSTER_PROFILE = "latest-http-v1"
EXCLUDED_FUTURE_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"


def selected_roster(roster_profile: str = "legacy-v1") -> tuple[tuple[str, str], ...]:
    if roster_profile == "legacy-v1":
        return ROSTER
    if roster_profile == LATEST_ROSTER_PROFILE:
        return (
            tuple(("river", m) for m in RIVER_MODELS if m != EXCLUDED_FUTURE_MODEL)
            + LATEST_HTTP_MODELS
        )
    raise ValueError("unknown roster profile")


HTTP_APIS = MappingProxyType(
    {"openai": "responses", "anthropic": "messages", "mistral": "chat_completions"}
)


@dataclass(frozen=True)
class CampaignSelection:
    provider: str
    model: str
    api: str
    mode_profile: str
    settings_source: str
    request_profile: (
        openai_responses.RequestProfile
        | anthropic_messages.RequestProfile
        | mistral_chat.RequestProfile
        | None
    )
    native_family: str | None


def native_family(model: str) -> str:
    if "DeepSeek" in model:
        return "deepseek"
    if "Kimi" in model:
        return "kimi"
    if "GLM" in model:
        return "glm"
    if "Nemotron" in model:
        return "nemotron"
    return "qwen"


# References, never copied capability declarations. Candidate-only models may
# have no reviewed profile, exactly as before this selection owner existed.
_PROFILE_OWNERS = {
    "openai": openai_responses,
    "anthropic": anthropic_messages,
    "mistral": mistral_chat,
}
SELECTIONS = tuple(
    CampaignSelection(
        p,
        m,
        HTTP_APIS.get(p, "grpc_inference_generate"),
        mode,
        "tools/three_flow_" + ("river" if p == "river" else "http") + ".py",
        _PROFILE_OWNERS[p].MODEL_REQUEST_PROFILES.get(m)
        if p in _PROFILE_OWNERS
        else None,
        native_family(m) if p == "river" else None,
    )
    for p, m in tuple(("river", m) for m in RIVER_MODELS) + LATEST_HTTP_MODELS
    for mode in ("legacy-v1", "off-or-minimum-v1")
    if mode != "legacy-v1" or p == "river" or m in MODELS.values()
)


def http_mode_fields(provider: str, model: str, mode_profile: str) -> dict[str, Any]:
    controls: dict[str, dict[str, Any]] = {
        "openai": {"reasoning": {"effort": minimum_effort(model)}},
        "anthropic": {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        },
        "mistral": {"reasoning_effort": "none"},
    }
    return controls.get(provider, {}) if mode_profile == "off-or-minimum-v1" else {}


def river_reasoning_prefilled(model: str, mode_profile: str) -> bool:
    return mode_profile == "legacy-v1" or "GLM-5.3" in model


def river_legacy_fields(model: str) -> dict[str, Any]:
    return {
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "thinking": True,
        "stop": ["<|im_end|>"] if "Kimi" in model else [],
        "deepseek_reasoning_effort": 75 if "V4.1" in model else None,
    }


def river_mode_fields(model: str, mode_profile: str) -> dict[str, Any]:
    if mode_profile == "legacy-v1":
        return {}
    minimum = river_reasoning_prefilled(model, mode_profile)
    fields: dict[str, Any] = {
        "mode_profile": mode_profile,
        "reasoning_mode": "MINIMUM" if minimum else "OFF",
        "reasoning_prefilled": minimum,
        "thinking": minimum,
        "deepseek_reasoning_effort": None,
    }
    if "DeepSeek" in model:
        fields["encoder"] = {"thinking_mode": "chat", "reasoning_effort": None}
    else:
        fields["template_kwargs"] = (
            {"reasoning_effort": "low"}
            if minimum
            else {"thinking": False}
            if "Kimi" in model
            else {"enable_thinking": False}
        )
    return fields


def registry(
    *,
    mode_profile: str = "legacy-v1",
    transport_policy: str = "legacy-v1",
    roster_profile: str = "legacy-v1",
) -> list[dict[str, Any]]:
    if transport_policy not in ("legacy-v1", "paced-safe-errors-v1"):
        raise ValueError("unknown transport policy")
    if mode_profile not in ("legacy-v1", "off-or-minimum-v1"):
        raise ValueError("unknown reasoning mode profile")
    rows: list[dict[str, Any]] = [
        {
            "provider": p,
            "model": m,
            "endpoint": ENDPOINTS.get(p, "api.river.ai:443"),
            "mapping": "three-flow-river-native-v1"
            if p == "river"
            else "three-flow-" + p + "-v1",
            "request_settings": (
                river_legacy_fields(m)
                if p == "river"
                else {
                    "tool_choice": {"type": "auto"},
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                }
                if p == "anthropic"
                else {
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                    "store": False,
                    "omitted": ["temperature", "reasoning", "top_p", "seed"],
                }
                if p == "openai"
                else {
                    "tool_choice": "required",
                    "temperature": 0.0,
                    "parallel_tool_calls": False,
                    "output_policy": (
                        "min(explicit maximum, context minus conservative input bound)"
                    ),
                }
            ),
            "settings_source": "tools/three_flow_"
            + ("river" if p == "river" else "http")
            + ".py",
            "retries": 0,
            "output_limit": (
                "externally admitted verified maximum or explicit published-context "
                "diagnostic bound; no automatic smaller-output resubmission"
            ),
            "admitted": False,
        }
        for p, m in selected_roster(roster_profile)
    ]
    for row in rows:
        if mode_profile == "legacy-v1":
            break
        p, m = row["provider"], row["model"]
        settings = row["request_settings"]
        minimum = (p != "river" or "GLM-5.3" in m) and minimum_effort(m) != "none"
        row["mapping"] += "-" + mode_profile
        settings.update(
            mode_profile=mode_profile, reasoning_mode="MINIMUM" if minimum else "OFF"
        )
        if p == "openai":
            settings.update(
                reasoning={"effort": minimum_effort(m)},
                omitted=["temperature", "top_p", "seed"],
            )
        elif p == "anthropic":
            settings["output_config"] = {"effort": "low"}
        elif p == "mistral":
            settings["reasoning_effort"] = "none"
        else:
            settings.update(river_mode_fields(m, mode_profile))
    if transport_policy != "legacy-v1":
        for row in rows:
            if row["provider"] == "mistral":
                row["mapping"] += "-" + transport_policy
                row["request_settings"].update(
                    transport_policy=transport_policy,
                    pacing_scope="process-wide-mistral",
                    minimum_spacing_seconds=1,
                    retry_after_max_seconds=86400,
                    retry_eligibility="none-proven",
                    automatic_resubmission=False,
                )
    return rows
