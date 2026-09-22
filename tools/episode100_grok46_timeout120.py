# SPDX-License-Identifier: Apache-2.0
"""One separately identified Grok46 diagnostic, not a default timeout change."""

from dataclasses import replace
from decimal import Decimal
from types import ModuleType
from typing import Any

from operatebench.agents.pricing import LifecyclePricingPolicy
from tools import episode100

PROFILE = "synthetic-demo-episode100-grok46-timeout120-v1"
AGENT_ID = "model-synthetic-demo-episode100-grok46-timeout120-v1"
TURN_DEADLINE_SECONDS = 120.0

# Entirely synthetic accounting example; hashes describe invented strings.
# This profile carries no paid authority or actual execution history.
SERIES: dict[str, Any] = {
    "schema": "episode100-series-v1",
    "series_id": "synthetic-demo-timeout120-series-v1",
    "prior_evidence_sha256": [
        "b5a50e1ea59395ac4ca9eeb5a6c8eba740d95598faf3ace1ddad7b7e741b30a5",
        "836f97152caf42560c602db3c495387271679c781948234c250b9bd9e98dbcc9",
        "b07ce466783c73e94c19a4fd276d4d9167fde7809179bc125b5c68c768e4e8c6",
    ],
    "authorized_total_usd": "20",
    "committed_exposure_usd": "2",
    "new_cap_usd": "18",
}


def load_profile_spec(module: ModuleType) -> Any:
    # Business policy has no transport timeout: retain its 43200-minute horizon.
    return episode100.load_profile_spec(module)


def controls_for(module: ModuleType, policy: LifecyclePricingPolicy, cap: Decimal) -> Any:
    return replace(
        episode100.controls_for(module, policy, cap),
        turn_deadline_seconds=TURN_DEADLINE_SECONDS,
        wall_clock_deadline_seconds=100 * TURN_DEADLINE_SECONDS,
    )
