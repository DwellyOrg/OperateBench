"""Reference-driven actual SDK MockTransport fixtures, explicitly NOT LLM results."""

from __future__ import annotations

import json
from typing import Any

import httpx

from operatebench.agents.playback import decision_record
from operatebench.core.protocol import AgentObservation
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.three_flow_http import MODELS


class SDKMockTransport(httpx.MockTransport):
    def __init__(
        self,
        provider: str,
        flow: str,
        *,
        agent: str = "reference",
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = MODELS[provider] if model is None else model
        self.calls: list[dict[str, Any]] = []
        if flow == "maintenance":
            self.reference = build_agent(agent)
        else:
            commands = COMMERCE_COMMANDS if flow == "commerce" else COMPLIANCE_COMMANDS
            self.reference = commands.factories.build_agent(agent)
        self.reference.begin_episode({})
        super().__init__(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        prompt = (
            body["input"]
            if self.provider == "openai"
            else body["messages"][-1]["content"]
        )
        public = json.loads(prompt)["observation"]
        decision = decision_record(self.reference.decide(AgentObservation(**public)))
        name = decision.pop("kind").lower()
        if name == "ask":
            decision["wait"].pop("kind")
        model = self.model
        if self.provider == "openai":
            result = {
                "id": "resp_offline",
                "object": "response",
                "created_at": 1,
                "model": model,
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_offline",
                        "call_id": "call_offline",
                        "name": name,
                        "arguments": json.dumps(decision),
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        elif self.provider == "anthropic":
            result = {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": model,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "content": [
                    {"type": "redacted_thinking", "data": "offline-hidden-reasoning"},
                    {
                        "type": "tool_use",
                        "id": "tool_offline",
                        "name": name,
                        "input": decision,
                    },
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        else:
            result = {
                "id": "chat_offline",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool_offline",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(decision),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=result)
