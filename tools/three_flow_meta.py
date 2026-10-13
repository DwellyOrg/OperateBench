"""Finite Meta Standard selection checks; launch remains fail-closed pending integration.

No credential access, provider requests or admission generation lives here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, NoReturn

from tools.three_flow_profiles import ENDPOINTS, META_ENVELOPES


def validate_meta_profile(profile: dict[str, Any], provider: str, model: str) -> None:
    """Bind exact Standard identity, published context and all-tier price envelope."""
    if provider != "meta" or model not in META_ENVELOPES:
        raise ValueError("only exact Meta Standard text models are eligible")
    context, input_rate, output_rate = META_ENVELOPES[model]
    if (
        profile["endpoint"] != ENDPOINTS["meta"]
        or profile["context_max"] != context
        or profile["input_max"] is not None
        or profile["output_max"] is not None
        or profile.get("bounds_policy") is None
        or "execution_policy" in profile
        or "owner_output_cap_tokens" in profile
        or Decimal(profile["input_usd_per_million"]) != Decimal(input_rate)
        or Decimal(profile["output_usd_per_million"]) != Decimal(output_rate)
    ):
        raise ValueError(
            "Meta requires exact tariff/context and unknown native output bound"
        )


def campaign_transport() -> NoReturn:
    """Never substitute OpenAI or claim the separate snapshot adapter is ready.

    Integration TODO: operatebench.agents.meta_responses.MetaResponsesTransport
    and operatebench.providers.meta_responses.build_client. A reviewed bridge must
    bind the existing SharedGuard, pre-wire dispatch journal, raw wire evidence,
    provider observers and offline replay, with no artificial turn deadline.
    """
    raise ValueError("Meta campaign wire/evidence/replay integration is not ready")
