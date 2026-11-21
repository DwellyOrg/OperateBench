"""Non-authorizing synthetic transport factory for the finite late-November roster."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from operatebench.agents.pricing import LifecyclePricingPolicy
from tools.three_flow_gemini import offline_factory as gemini_factory
from tools.three_flow_http import HTTPCampaignTransport
from tools.three_flow_live_factory import AdmittedGuard
from tools.three_flow_mock import SDKMockTransport
from tools.three_flow_profiles import LATE_NOVEMBER_MODELS, LATE_NOVEMBER_PROFILE
from tools.three_flow_ripple import MODEL as RIPPLE_MODEL
from tools.three_flow_river import GRPC_OPTIONS, RiverCampaignTransport
from tools.three_flow_river_mock import RiverMockChannel


def factory(assets: Path) -> Any:
    """Explicit mock capabilities only; caller owns the shared offline journal."""

    def make(slot: dict[str, Any], budget: Any) -> Any:
        if (
            slot.get("provider"),
            slot.get("model"),
        ) not in LATE_NOVEMBER_MODELS or budget.roster_profile != LATE_NOVEMBER_PROFILE:
            raise ValueError("outside synthetic late November cohort")
        if slot["provider"] == "gemini":
            return gemini_factory(slot, budget)
        if slot["provider"] == "meta":
            p = {
                "endpoint": "https://api.meta.ai/v1/responses",
                "context_max": 1048576,
                "input_max": None,
                "output_max": 524288,
                "bounds_policy": {
                    "mode": "published-context-diagnostic-v1",
                    "source": "https://dev.meta.ai/docs",
                    "scope": "endpoint",
                    "interpretation": "published-integer",
                },
                "network_timeout_seconds": None,
                "input_usd_per_million": "0.30",
                "output_usd_per_million": "1.50",
            }
            guard = AdmittedGuard(
                profile=p,
                budget=budget,
                cell=slot["trial_id"],
                policy=LifecyclePricingPolicy(
                    "synthetic-ember-envelope", Decimal("0.30"), Decimal("1.50")
                ),
                max_output_tokens=524288,
            )
            mock = SDKMockTransport("openai", slot["flow"], model=slot["model"])

            def handle(request: httpx.Request) -> httpx.Response:
                result = mock.handle(request)
                body = json.loads(result.content)
                body.update(
                    store=False, parallel_tool_calls=False, tool_choice="auto", tools=[]
                )
                body["usage"].update(
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                )
                return httpx.Response(200, json=body)

            transport = HTTPCampaignTransport(
                provider="meta",
                model=slot["model"],
                api_key="offline-synthetic-key",
                inner=httpx.MockTransport(handle),
                guard=guard,
                max_output_tokens=524288,
                context_tokens=1048576,
                mode_profile="off-or-minimum-v1",
                roster_profile=LATE_NOVEMBER_PROFILE,
            )
            transport.settings.update(
                bounds_policy=p["bounds_policy"], context_request_bound=1048576
            )
            return transport
        if slot["model"] != RIPPLE_MODEL:
            raise ValueError("unsupported native model")
        p = {
            "endpoint": "api.river.ai:443",
            "context_max": 262144,
            "input_max": None,
            "output_max": 30000,
            "owner_output_cap_tokens": 30000,
            "bounds_policy": {"mode": "published-context-diagnostic-v1"},
        }
        guard = AdmittedGuard(
            profile=p,
            budget=budget,
            cell=slot["trial_id"],
            policy=LifecyclePricingPolicy(
                "synthetic-ripple-envelope", Decimal("1"), Decimal("3")
            ),
            max_output_tokens=30000,
        )
        channel = RiverMockChannel(RIPPLE_MODEL, slot["flow"])
        river_transport = RiverCampaignTransport(
            model=RIPPLE_MODEL,
            assets=assets,
            channel=channel,
            guard=guard,
            max_output_tokens=30000,
            channel_options=GRPC_OPTIONS,
            network_timeout=None,
            mode_profile="ripple-default-v1",
        )
        channel.tokenizer = river_transport.tokenizer
        river_transport.settings.update(
            bounds_policy=p["bounds_policy"], context_request_bound=262144
        )
        return river_transport

    return make
