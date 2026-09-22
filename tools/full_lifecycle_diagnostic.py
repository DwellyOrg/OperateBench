"""Private five-model Full-lifecycle diagnostic, Slice 2B (offline only).

All five implemented-offline cells execute over in-process transports through
their real pinned SDKs. ``--live`` is an immediate, closed refusal. No code in
this module reads a credential or constructs a live client.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anthropic
import httpx
import openai

from operatebench.agents.anthropic_messages import (
    LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
    AnthropicMessagesTransport,
)
from operatebench.agents.evidence import (
    EvidenceRecordingModelAgent,
    EvidenceRunIdentity,
    ProviderEvidenceRecorder,
    WireCaptureTransport,
    provider_identity_for,
)
from operatebench.agents.mistral_chat import (
    LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION,
    MistralChatCompletionsTransport,
)
from operatebench.agents.model import MAX_OUTPUT_TOKENS, MODEL_PROTOCOL_VERSION
from operatebench.agents.openai_responses import (
    LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    OpenAIResponsesTransport,
)
from operatebench.agents.pricing import (
    RATE_SOURCE_OPERATOR,
    LifecycleCostGuard,
    LifecyclePricingPolicy,
)
from operatebench.agents.transport import ToolCall
from operatebench.agents.xai_chat_completions import (
    LIFECYCLE_XAI_REQUEST_MAPPING_VERSION,
    XAIOpenAICompatChatCompletionsTransport,
)
from operatebench.artifact import (
    ARTIFACT_VERSION,
    read_artifact,
    replay_artifact,
    write_artifact,
)
from operatebench.core.outcomes import Act, AgentOutcome, Ask, Escalate, Wait
from operatebench.core.protocol import AgentObservation
from operatebench.core.retrieval import RetrieveBatch
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.execution_bundle import audit_execution_bundle_files
from operatebench.execution_ledger import (
    EXECUTION_LEDGER_VERSION,
    LedgerControls,
    read_execution_ledger,
)
from operatebench.providers.anthropic_messages import ANTHROPIC_BASE_URL
from operatebench.providers.mistral_chat import build_client as build_mistral_client
from operatebench.providers.openai_responses import OPENAI_BASE_URL
from operatebench.providers.xai_openai_compat import XAI_BASE_URL
from operatebench.runner import run_episode
from operatebench.version import OPERATEBENCH_VERSION

# SHA-256 of ``git archive --format=tar <diagnostic-base>``. This pins the exact
# baseline source bytes without publishing an internal Git object identifier.
SOURCE_BASE_ARCHIVE_DIGEST = (
    "26ee847e0304e1a1caba671bcdb1a76078b99f207d24e85719e38018d71d582d"
)
SCENARIO_ID = "V1"
EPISODE_ID = "maintenance-v1-synthetic"
SCAFFOLD_ID = "lifecycle_full_v1_outcome_tools_v1"
SCAFFOLD_SCHEMA_DIGEST = (
    "f5165dd0915e72e2b4a574b509c7723d64071738d34a048702ba63ec41b35efa"
)
SPEC_DIGEST = "9e99b409b153e03aeb240c6402e14ff21c8bf0df3b9a1df2016c8fa64138e44d"
FIXTURE = Path("examples/operatebench/maintenance_v0_1.yaml")
_PLACEHOLDER_KEY = "offline-in-process-placeholder-not-a-credential"
_INPUT_TOKENS = 137
_OUTPUT_TOKENS = 29
_ANTHROPIC_PROSE_SENTINEL = "offline-provider-prose-must-not-persist"
_XAI_PROSE_SENTINEL = "offline-xai-provider-prose-must-not-persist"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


_CELLS = (
    (
        "anthropic-haiku-4-5",
        "anthropic",
        "messages",
        "claude-haiku-4-5-20251001",
        "implemented_offline",
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
    ),
    (
        "anthropic-sonnet-5",
        "anthropic",
        "messages",
        "claude-sonnet-5",
        "implemented_offline",
        LIFECYCLE_ANTHROPIC_REQUEST_MAPPING_VERSION,
    ),
    (
        "openai-gpt-5-6-luna",
        "openai",
        "responses",
        "gpt-5.6-luna",
        "implemented_offline",
        LIFECYCLE_OPENAI_REQUEST_MAPPING_VERSION,
    ),
    (
        "xai-grok-4-5",
        "xai",
        "chat_completions",
        "grok-4.5",
        "implemented_offline",
        LIFECYCLE_XAI_REQUEST_MAPPING_VERSION,
    ),
    (
        "mistral-small-2603",
        "mistral",
        "chat_completions",
        "mistral-small-2603",
        "implemented_offline",
        LIFECYCLE_MISTRAL_REQUEST_MAPPING_VERSION,
    ),
)

_SONNET_FIXED_OPERATOR = {
    "status": "implemented_offline_verified_not_live_executed",
    "rates_verified_on": "2026-09-07",
    "rate_sources": [
        "https://platform.claude.com/docs/en/models/overview",
        "https://platform.claude.com/docs/en/about-claude/pricing",
    ],
    "input_usd_per_mtok": "2",
    "output_usd_per_mtok": "10",
    "expected_provider_calls": 46,
    "max_provider_calls": 50,
    "max_native_input_token_bound": 25_652,
    "max_output_tokens_per_call": 4_096,
    "expected_call_token_ceiling": 1_368_408,
    "hard_call_token_cap": 1_487_400,
    "exact_worst_case_usd": "4.6132",
    "minimal_cent_cap_usd": "4.62",
}

_OPENAI_LUNA_FIXED_OPERATOR = {
    "status": "implemented_offline_verified_not_live_executed",
    "repeat_ready": True,
    "rates_verified_on": "2026-09-08",
    "rate_sources": [
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna.md",
        "https://developers.openai.com/api/docs/pricing",
    ],
    "model_identifier": "gpt-5.6-luna",
    "request_profile": "gpt56luna_reasoning_none_sampling_omitted_v2",
    "reasoning_effort": "none",
    "sampling_fields_omitted": ["temperature", "top_p"],
    "input_usd_per_mtok": "0.20",
    "output_usd_per_mtok": "1.20",
    "expected_provider_calls": 46,
    "max_provider_calls": 50,
    "max_native_input_token_bound": 27_009,
    "max_output_tokens_per_call": 4_096,
    "expected_call_token_ceiling": 1_430_830,
    "hard_call_token_cap": 1_555_250,
    "exact_worst_case_usd": "0.515850",
    "minimal_cent_cap_usd": "0.52",
}

_MISTRAL_FIXED_OPERATOR = {
    "status": "implemented_offline_verified_not_live_executed",
    "rates_verified_on": "2026-09-07",
    "rate_sources": [
        "https://docs.mistral.ai/models/mistral-small-4-0-26-03",
        "https://docs.mistral.ai/api",
        "https://docs.mistral.ai/openapi.yaml",
        "https://docs.mistral.ai/inference/pricing",
    ],
    "model_identifier": "mistral-small-2603",
    "moving_alias_not_used": "mistral-small-latest",
    "context_window_provider_text": "256k text",
    "benchmark_output_cap_not_official_ceiling": 4_096,
    "omitted_unresolved_reasoning_field": "reasoning_effort",
    "omitted_service_tier": "Global Standard",
    "input_usd_per_mtok": "0.15",
    "output_usd_per_mtok": "0.60",
    "expected_provider_calls": 46,
    "max_provider_calls": 50,
    "max_native_input_token_bound": 27_099,
    "max_output_tokens_per_call": 4_096,
    "expected_call_token_ceiling": 1_434_970,
    "hard_call_token_cap": 1_559_750,
    "exact_worst_case_usd": "0.32612250",
    "minimal_cent_cap_usd": "0.33",
}

DIAGNOSTIC_PLAN: Mapping[str, Any] = _freeze(
    {
        "plan": "full_lifecycle_diagnostic_slice_2b",
        "status": "private_engineering_diagnostic",
        "live_authorizable": False,
        "claim_boundary": {
            "integration_and_behavior_evidence_only": True,
            "ranking": False,
            "matched_arm_contrast": False,
            "public_claim": False,
            "model_capability_claim": False,
            "production_inference": False,
            "alpha_replacement": False,
        },
        "identities": {
            "source_base_archive_digest_sha256": SOURCE_BASE_ARCHIVE_DIGEST,
            "engine": OPERATEBENCH_VERSION,
            "operation_id": "lettings_maintenance_synthetic_v1",
            "operation_version": "0.6.0",
            "operation_spec_digest_sha256": SPEC_DIGEST,
            "scenario": SCENARIO_ID,
            "scaffold": SCAFFOLD_ID,
            "scaffold_schema_digest_sha256": SCAFFOLD_SCHEMA_DIGEST,
            "protocol": MODEL_PROTOCOL_VERSION,
            "artifact": ARTIFACT_VERSION,
            "execution_ledger": EXECUTION_LEDGER_VERSION,
        },
        "cells": [
            {
                "cell_id": cell_id,
                "provider": provider,
                "api": api,
                "model": model,
                "adapter_status": status,
                "request_mapping": mapping,
                "episode": EPISODE_ID,
                "scenario": SCENARIO_ID,
                "lifecycle": "full",
                "output_path": f"cells/{cell_id}",
                **(
                    {"fixed_operator": _SONNET_FIXED_OPERATOR}
                    if model == "claude-sonnet-5"
                    else (
                        {"fixed_operator": _OPENAI_LUNA_FIXED_OPERATOR}
                        if model == "gpt-5.6-luna"
                        else (
                            {"fixed_operator": _MISTRAL_FIXED_OPERATOR}
                            if model == "mistral-small-2603"
                            else {}
                        )
                    )
                ),
            }
            for cell_id, provider, api, model, status, mapping in _CELLS
        ],
        "session_policy": "fresh_stateless_across_operation_invocations",
        "controls": {
            "attempts_per_cell": 1,
            "sdk_retries": 0,
            "max_provider_calls_per_cell": 50,
            "expected_provider_calls_per_cell": 46,
            "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
            "token_hard_cap_per_cell": 1_221_650,
            "turn_deadline_seconds": 30.0,
            "wall_clock_deadline_seconds_per_cell": 2700.0,
            "cost_cap_usd_per_cell": None,
            "input_usd_per_mtok": None,
            "output_usd_per_mtok": None,
            "rates_verified": False,
            "rate_status": "pending_operator_freeze_against_current_official_docs",
        },
        "staged_order": (
            "offline_all_cells",
            "one_cell",
            "manual_audit",
            "next_cell",
        ),
        "prohibitions": (
            "hidden_retries",
            "replacement_cells",
            "omitted_failed_assigned_cells",
            "ranking",
            "aggregation",
            "public_result",
        ),
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _observation(payload: Mapping[str, Any]) -> AgentObservation:
    """Rebuild exactly the public observation carried across the SDK wire."""
    return AgentObservation(
        now=payload["now"],
        operation_id=payload["operation_id"],
        operation_instance_id=payload["operation_instance_id"],
        invocation_index=payload["invocation_index"],
        turn_index=payload["turn_index"],
        policy=payload["policy"],
        actors=payload["actors"],
        message_fixture_ids=payload["message_fixture_ids"],
        action_schemas=payload["action_schemas"],
        wake_event_types=payload["wake_event_types"],
        last_rejection=payload["last_rejection"],
        phase=payload["phase"],
        event=payload["event"],
        retrieval=payload["retrieval"],
    )


def _wait_arguments(wait: Wait) -> dict[str, Any]:
    return {
        "reason": wait.reason,
        "wake_on": list(wait.wake_on),
        "fallback_after_minutes": wait.fallback_after_minutes,
    }


def _tool_call(outcome: AgentOutcome | RetrieveBatch) -> ToolCall:
    """Project one authored reference decision through the public tool contract."""
    if isinstance(outcome, RetrieveBatch):
        return ToolCall(
            "retrieve", {"requests": [item.as_dict() for item in outcome.requests]}
        )
    if isinstance(outcome, Act):
        return ToolCall(
            "act",
            {
                "action_type": outcome.action_type,
                "payload": dict(outcome.payload),
                "evidence_refs": list(outcome.evidence_refs),
                "rationale": outcome.rationale,
            },
        )
    if isinstance(outcome, Wait):
        return ToolCall("wait", _wait_arguments(outcome))
    if isinstance(outcome, Ask):
        return ToolCall(
            "ask",
            {
                "recipient_actor_id": outcome.recipient_actor_id,
                "message_fixture_id": outcome.message_fixture_id,
                "correlation_id": outcome.correlation_id,
                "wait": _wait_arguments(outcome.wait),
            },
        )
    if isinstance(outcome, Escalate):
        return ToolCall(
            "escalate",
            {
                "checkpoint_id": outcome.checkpoint_id,
                "exception_type": outcome.exception_type,
                "evidence_refs": list(outcome.evidence_refs),
                "deadline_after_minutes": outcome.deadline_after_minutes,
                "rationale": outcome.rationale,
            },
        )
    return ToolCall(
        "complete",
        {"reason": outcome.reason, "evidence_refs": list(outcome.evidence_refs)},
    )


class _ScriptedReference:
    """Provider-free HTTP handler driven only by the shipped reference agent."""

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.calls = 0
        self._reference = build_agent("reference")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        if self.provider == "openai":
            prompt = json.loads(body["input"][0]["content"])
        elif self.provider in ("xai", "mistral"):
            prompt = json.loads(body["messages"][1]["content"])
        else:
            prompt = json.loads(body["messages"][0]["content"])
        call = _tool_call(self._reference.decide(_observation(prompt["observation"])))
        if self.provider == "openai":
            response = {
                "id": f"resp_offline_{self.calls}",
                "object": "response",
                "created_at": 1,
                "model": self.model,
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": f"fc_offline_{self.calls}",
                        "call_id": f"call_offline_{self.calls}",
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                        "status": "completed",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "required",
                "tools": [],
                "usage": {
                    "input_tokens": _INPUT_TOKENS,
                    "output_tokens": _OUTPUT_TOKENS,
                    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                },
                "incomplete_details": None,
            }
        elif self.provider == "anthropic":
            response = {
                "id": f"msg_offline_{self.calls}",
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [
                    {"type": "text", "text": _ANTHROPIC_PROSE_SENTINEL},
                    {
                        "type": "tool_use",
                        "id": f"toolu_offline_{self.calls}",
                        "name": call.name,
                        "input": call.arguments,
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": _INPUT_TOKENS,
                    "output_tokens": _OUTPUT_TOKENS,
                },
            }
        elif self.provider == "xai":
            response = {
                "id": f"chatcmpl_offline_{self.calls}",
                "object": "chat.completion",
                "created": 1,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": _XAI_PROSE_SENTINEL,
                            "tool_calls": [
                                {
                                    "id": f"call_offline_{self.calls}",
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": json.dumps(call.arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": _INPUT_TOKENS,
                    "completion_tokens": _OUTPUT_TOKENS,
                    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                },
            }
        else:
            response = {
                "id": f"cmpl{self.calls:05d}",
                "object": "chat.completion",
                "created": 1,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"m{self.calls:08d}",
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": json.dumps(call.arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": _INPUT_TOKENS,
                    "completion_tokens": _OUTPUT_TOKENS,
                    "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        return httpx.Response(200, json=response, request=request)


def _controls(policy: LifecyclePricingPolicy) -> LedgerControls:
    configured = DIAGNOSTIC_PLAN["controls"]
    return LedgerControls(
        max_provider_calls=configured["max_provider_calls_per_cell"],
        max_attempts_per_call=configured["attempts_per_cell"],
        expected_provider_calls=configured["expected_provider_calls_per_cell"],
        token_hard_cap=configured["token_hard_cap_per_cell"],
        max_output_tokens=configured["max_output_tokens_per_call"],
        cost_cap_usd="1",
        cost_cap_policy="refuse_before_dispatch",
        turn_deadline_seconds=configured["turn_deadline_seconds"],
        wall_clock_deadline_seconds=configured["wall_clock_deadline_seconds_per_cell"],
        pricing=policy.as_dict(),
    )


def _run_offline_cell(
    spec: Any, cell: Mapping[str, Any], directory: Path
) -> dict[str, Any]:
    directory.mkdir(mode=0o700, parents=True)
    handler = _ScriptedReference(provider=cell["provider"], model=cell["model"])
    capture = WireCaptureTransport()
    capture.attach(httpx.MockTransport(handler))
    http_client = httpx.Client(transport=capture)
    configured = DIAGNOSTIC_PLAN["controls"]
    deadline = configured["turn_deadline_seconds"]
    max_output_tokens = configured["max_output_tokens_per_call"]
    if cell["provider"] in ("openai", "xai"):
        client: Any = openai.OpenAI(
            api_key=_PLACEHOLDER_KEY,
            base_url=(OPENAI_BASE_URL if cell["provider"] == "openai" else XAI_BASE_URL),
            max_retries=0,
            http_client=http_client,
        )
    elif cell["provider"] == "anthropic":
        client = anthropic.Anthropic(
            api_key=_PLACEHOLDER_KEY,
            base_url=ANTHROPIC_BASE_URL,
            max_retries=0,
            http_client=http_client,
        )
    else:
        client = build_mistral_client(
            api_key=_PLACEHOLDER_KEY,
            http_client=http_client,
        )

    # Zero rates describe this synthetic in-process transport, not provider pricing.
    policy = LifecyclePricingPolicy(
        policy_id="offline_in_process_zero_cost_v1",
        input_usd_per_mtok=Decimal(0),
        output_usd_per_mtok=Decimal(0),
        rate_source=RATE_SOURCE_OPERATOR,
    )
    guard = LifecycleCostGuard(
        policy=policy,
        cap_usd=Decimal(1),
        max_output_tokens=max_output_tokens,
        token_hard_cap=configured["token_hard_cap_per_cell"],
    )
    # The provider exchange owns the guard, so construct it once with the final guard.
    transport: Any
    if cell["provider"] == "openai":
        transport = OpenAIResponsesTransport(
            model=cell["model"],
            client=client,
            deadline_seconds=deadline,
            cost_guard=guard,
        )
    elif cell["provider"] == "anthropic":
        transport = AnthropicMessagesTransport(
            model=cell["model"],
            client=client,
            deadline_seconds=deadline,
            cost_guard=guard,
        )
    elif cell["provider"] == "xai":
        transport = XAIOpenAICompatChatCompletionsTransport(
            model=cell["model"],
            client=client,
            deadline_seconds=deadline,
            cost_guard=guard,
        )
    else:
        transport = MistralChatCompletionsTransport(
            model=cell["model"],
            client=client,
            deadline_seconds=deadline,
            cost_guard=guard,
        )
    ledger_path = directory / "execution_ledger.ndjson"
    recorder = ProviderEvidenceRecorder(
        path=ledger_path,
        identity=EvidenceRunIdentity(
            operation_id=spec.operation_id,
            spec_digest_sha256=spec.spec_digest_sha256,
            scenario_id=SCENARIO_ID,
            agent_id=cell["cell_id"],
        ),
        provider=provider_identity_for(transport),
        controls=_controls(policy),
        wire=capture,
        guard=guard,
    )
    transport.attach_recorder(recorder)
    agent = EvidenceRecordingModelAgent(
        transport,
        model=cell["model"],
        agent_id=cell["cell_id"],
        recorder=recorder,
        max_output_tokens=max_output_tokens,
        max_transport_calls=DIAGNOSTIC_PLAN["controls"]["max_provider_calls_per_cell"],
    )
    run = run_episode(
        spec,
        SCENARIO_ID,
        cell["cell_id"],
        agent_factory=lambda: agent,
        agent_kind="model",
        evidence_recorder=recorder,
    )
    artifact_path = write_artifact(run, directory / "episode_artifact.json")
    artifact = read_artifact(artifact_path)
    ledger = read_execution_ledger(ledger_path, require_complete=True)
    bundle = audit_execution_bundle_files(artifact_path, ledger_path)
    calls_before_replay = handler.calls
    replay = replay_artifact(spec, artifact)
    calls_during_replay = handler.calls - calls_before_replay
    if not bundle.ok or not replay.ok or calls_during_replay:
        raise RuntimeError(
            "offline evidence did not pass bundle and provider-free replay"
        )
    return {
        "cell_id": cell["cell_id"],
        "provider_calls": ledger.totals.provider_calls,
        "artifact_version": artifact["artifact_version"],
        "execution_ledger_version": bundle.execution_ledger_version,
        "replay_provider_calls": calls_during_replay,
        "transport": (
            "httpx.MockTransport through the real OpenAI SDK"
            if cell["provider"] in ("openai", "xai")
            else (
                "httpx.MockTransport through the real Anthropic SDK"
                if cell["provider"] == "anthropic"
                else "httpx.MockTransport through the real Mistral SDK"
            )
        ),
    }


def run_offline_matrix(directory: Path) -> Mapping[str, Any]:
    """Execute all five Slice-2B cells through real SDKs in process."""
    spec = load_spec(FIXTURE)
    results = [
        _run_offline_cell(spec, cell, directory / cell["output_path"])
        for cell in DIAGNOSTIC_PLAN["cells"]
        if cell["adapter_status"] == "implemented_offline"
    ]
    return {
        "ok": True,
        "executed_cells": len(results),
        "specified_only_cells": sum(
            cell["adapter_status"] == "specified_only"
            for cell in DIAGNOSTIC_PLAN["cells"]
        ),
        "cells": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--offline-verify", action="store_true")
    mode.add_argument(
        "--live",
        action="store_true",
        help="always refused in Slice 2B; five implemented-offline cells only",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.live:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "live_execution_not_implemented",
                    "live_authorizable": False,
                },
                sort_keys=True,
            )
        )
        return 2
    if arguments.offline_verify:
        if arguments.output_dir is None:
            print("--offline-verify requires --output-dir")
            return 2
        print(
            json.dumps(_plain(run_offline_matrix(arguments.output_dir)), sort_keys=True)
        )
        return 0
    print(json.dumps(_plain(DIAGNOSTIC_PLAN), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
