"""Meta Standard text models under the fixed Lifecycle snapshot scaffold.

Shares Core tools and the checked Responses projection, never OpenAI identity.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import openai
from openai.types.responses import Response

from operatebench.agents.lifecycle_contract import (
    AGENT_TOOL_NAMES,
    INSTRUCTIONS_CONTRACT,
    LIFECYCLE_NULL_ELISION,
    check_model_request,
    outcome_tools,
)
from operatebench.agents.model import (
    MAX_OUTPUT_TOKENS,
    MODEL_PROTOCOL_VERSION,
    ModelBoundaryError,
)
from operatebench.agents.openai_responses import model_response_from
from operatebench.agents.pricing import LifecyclePricingPolicy
from operatebench.agents.responses_projection import _response_id_digest
from operatebench.agents.transport import (
    ModelRequest,
    ModelResponse,
    ProviderTurnEvidence,
    ProviderTurnObserver,
)
from operatebench.jsonsafe import canonical_json_text
from operatebench.providers.cost import CostGuard
from operatebench.providers.executor import Clock, Sleep
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.meta_responses import (
    BASE_URL,
    PROVIDER,
    MetaResponsesExchange,
    check_model,
)
from operatebench.providers.openai_responses import ProviderDispatchObserver
from operatebench.providers.telemetry import AdapterUsage, ProviderTelemetry, TurnDeadline
from operatebench.providers.wire import WireResponse, wire_invalid

MAPPING_VERSION = "lifecycle_meta_responses_snapshot_auto_v2"
PRICING_SOURCE = "https://dev.meta.ai/docs/pricing-rate-limits"
# Cached rates are disclosed; the fixed two-count guard conservatively prices
# all input at the uncached rate. It cannot claim an exact discounted invoice.
STANDARD_RATES = {
    "muse-wonder-1.0": ("4.00", "0.20", "20.00"),
    "muse-spark-1.3": ("1.25", "0.15", "4.25"),
}


def pricing_profile(model: str) -> LifecyclePricingPolicy:
    check_model(model)
    input_rate, _, output_rate = STANDARD_RATES[model]
    return LifecyclePricingPolicy(
        policy_id=f"meta_standard_{model}_2026_10_13_uncached_upper_bound_v1",
        input_usd_per_mtok=Decimal(input_rate),
        output_usd_per_mtok=Decimal(output_rate),
    )


def build_model_payload(request: ModelRequest, *, model: str) -> dict[str, Any]:
    check_model(model)
    check_model_request(request, model=model)
    return {
        "model": model,
        "max_output_tokens": request.max_output_tokens,
        "instructions": (
            f"{INSTRUCTIONS_CONTRACT}\nrequest_mapping: {MAPPING_VERSION}\n"
            f"protocol_version: {request.protocol_version}\n"
            f"request_digest_sha256: {request.request_digest_sha256}"
        ),
        "tools": outcome_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "input": [
            {
                "role": "user",
                "content": canonical_json_text(request.prompt, "lifecycle model request"),
            }
        ],
    }


class MetaResponsesTransport:
    """ModelTransport with common executor evidence and independent snapshots."""

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI,
        deadline_seconds: float,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        cost_guard: CostGuard | None = None,
        recorder: ProviderTurnObserver | None = None,
        before_dispatch: ProviderDispatchObserver | None = None,
    ) -> None:
        if (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, (int, float))
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ModelBoundaryError("Meta turn deadline must be finite and positive")
        self._exchange = MetaResponsesExchange(
            model=model,
            client=client,
            sleep=sleep,
            clock=clock,
            cost_guard=cost_guard,
            before_dispatch=before_dispatch,
        )
        self._clock = clock
        self._deadline_seconds = float(deadline_seconds)
        self._recorder = recorder

    def __repr__(self) -> str:
        return (
            f"<MetaResponsesTransport provider=meta model={self.model} "
            f"mapping={MAPPING_VERSION}>"
        )

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def api(self) -> str:
        return "responses"

    @property
    def model(self) -> str:
        return self._exchange.model

    @property
    def request_mapping(self) -> str:
        return MAPPING_VERSION

    @property
    def turn_deadline_seconds(self) -> float:
        return self._deadline_seconds

    @property
    def settings(self) -> Mapping[str, Any]:
        return {
            "api": self.api,
            "base_url": BASE_URL,
            "request_mapping": MAPPING_VERSION,
            "request_profile": "meta_standard_snapshot_auto_v2",
            "comparison": "equal_scaffold_stateless",
            "reasoning_continuity": False,
            "request_fields_sent": [
                "input",
                "instructions",
                "max_output_tokens",
                "model",
                "parallel_tool_calls",
                "store",
                "tool_choice",
                "tools",
            ],
            "request_fields_omitted": [
                "reasoning",
                "temperature",
                "top_p",
                "include",
                "previous_response_id",
            ],
            "reasoning": None,
            "reasoning_policy": "provider_default",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_output_tokens_policy": "scaffold_ceiling",
            "protocol_version": MODEL_PROTOCOL_VERSION,
            "null_elision": LIFECYCLE_NULL_ELISION,
            "tool_names": list(AGENT_TOOL_NAMES),
            "tool_strict_mode": False,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "response_capture": "openai_sdk_with_raw_response_v1",
            "response_contract": "meta_sdk_declared_responses_usage_v1",
            "sdk": "openai",
            "sdk_version": openai.__version__,
            "retry": self._exchange.retry.as_dict(),
            "turn_deadline_seconds": self._deadline_seconds,
            "pricing_profile": pricing_profile(self.model).as_dict(),
            "pricing_source": PRICING_SOURCE,
            "cached_input_usd_per_mtok": STANDARD_RATES[self.model][1],
            "pricing_policy": "uncached_input_upper_bound_reasoning_included_in_output",
        }

    def last_usage(self) -> AdapterUsage:
        return self._exchange.last_usage()

    def last_telemetry(self) -> ProviderTelemetry:
        return self._exchange.last_telemetry()

    @property
    def recorder(self) -> ProviderTurnObserver | None:
        return self._recorder

    def attach_recorder(self, recorder: ProviderTurnObserver) -> None:
        if self._recorder is not None:
            raise ModelBoundaryError("Meta transport already has an evidence recorder")
        self._recorder = recorder

    def send(self, request: ModelRequest) -> ModelResponse:
        started = self._clock()
        payload = build_model_payload(request, model=self.model)
        answer: ModelResponse | None = None
        wire: WireResponse[Response] | None = None
        fault: str | None = None
        try:
            with self._exchange.turn():
                self._exchange.clear()
                budget = self._deadline_seconds
                wire = self._exchange.exchange(
                    payload,
                    TurnDeadline(
                        remaining_seconds=budget,
                        started_at=started,
                        cancelled=lambda: (self._clock() - started) >= budget,
                    ),
                )
                projected = model_response_from(wire)
                if not projected.tool_calls:
                    raise wire_invalid("Meta auto response has no Core function call")
                answer = projected
            return answer
        except AdapterProviderError as exc:
            fault = exc.fault
            raise
        finally:
            if self._recorder is not None:
                self._recorder.observe_provider_turn(
                    ProviderTurnEvidence(
                        request=request,
                        payload=payload,
                        settings=self.settings,
                        telemetry=self.last_telemetry(),
                        usage=self.last_usage(),
                        response=answer,
                        response_id_digest_sha256=_response_id_digest(wire),
                        fault=fault,
                    )
                )
