"""Meta Standard campaign selection checks.

No credential access, provider requests or admission generation lives here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from tools.three_flow_budget import META_UNCAPPED_PROFILE
from tools.three_flow_profiles import ENDPOINTS, META_ENVELOPES


def validate_meta_profile(profile: dict[str, Any], provider: str, model: str) -> None:
    """Bind exact Standard identity, published context and all-tier price envelope."""
    if provider != "meta" or model not in META_ENVELOPES:
        raise ValueError("only exact Meta Standard text models are eligible")
    context, input_rate, output_rate = META_ENVELOPES[model]
    if (
        profile["endpoint"] != ENDPOINTS["meta"]
        or "network_timeout_seconds" not in profile
        or profile["network_timeout_seconds"] is not None
        or profile["context_max"] != context
        or profile["input_max"] is not None
        or profile["output_max"] is not None
        or not isinstance(profile.get("bounds_policy"), dict)
        or profile["bounds_policy"].get("mode") != "published-context-diagnostic-v1"
        or profile["bounds_policy"].get("source") != "https://dev.meta.ai/docs"
        or profile["bounds_policy"].get("scope") != "endpoint"
        or profile["bounds_policy"].get("interpretation") != "published-integer"
        or (
            "execution_policy" in profile
            and profile["execution_policy"] != META_UNCAPPED_PROFILE
        )
        or (
            "owner_output_cap_tokens" in profile
            and (
                profile.get("execution_policy") != META_UNCAPPED_PROFILE
                or type(profile["owner_output_cap_tokens"]) is not int
                or profile["owner_output_cap_tokens"] != 16384
            )
        )
        or Decimal(profile["input_usd_per_million"]) != Decimal(input_rate)
        or Decimal(profile["output_usd_per_million"]) != Decimal(output_rate)
    ):
        raise ValueError(
            "Meta requires exact tariff/context and unknown native output bound"
        )
