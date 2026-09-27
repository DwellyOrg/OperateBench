"""Compatibility name for the successor OpenAI campaign SDK composition."""

from typing import Any

from tools.three_flow_http import HTTPCampaignTransport


class OpenAICampaignTransport(HTTPCampaignTransport):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider="openai", **kwargs)
