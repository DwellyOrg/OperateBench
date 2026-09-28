"""Reference-driven SDK protobuf channel; only the decoded public observation.

Synthetic mechanics, never LLM performance or a live capability. No future
state, historical tape, fixture-private data, credentials or network is read.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from operatebench.agents.playback import decision_record
from operatebench.agents.transport import ToolCall
from operatebench.core.protocol import AgentObservation
from operatebench.domains.lettings.maintenance.agents import build_agent
from operatebench.sdk.profile_packs.pack import COMMERCE_COMMANDS, COMPLIANCE_COMMANDS
from tools.three_flow_river import sdk
from tools.three_flow_river_assets import frame, synthetic_frame


class RiverMockChannel:
    offline_mock = True

    def __init__(
        self,
        model: str,
        flow: str,
        tokenizer: Any = None,
        *,
        agent: str = "reference",
        pending_polls: int = 0,
    ) -> None:
        self.pb, _ = sdk()
        self.model, self.tokenizer = model, tokenizer
        if flow == "maintenance":
            self.reference = build_agent(agent)
        else:
            commands = COMMERCE_COMMANDS if flow == "commerce" else COMPLIANCE_COMMANDS
            self.reference = commands.factories.build_agent(agent)
        self.reference.begin_episode({})
        self.submissions: list[Any] = []
        self.polls: list[str] = []
        self.pending: dict[str, Any] = {}
        self.pending_polls = pending_polls
        self.remaining: dict[str, int] = {}

    def generate(self, request: Any) -> Any:
        if request.base_model != self.model or len(request.prompts) != 1:
            raise ValueError("mock model/sample identity differs")
        self.submissions.append(request)
        prompt = request.prompts[0]
        text = self.tokenizer.decode(list(prompt.input_ids))
        public = None
        for i, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = json.JSONDecoder().raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(candidate, dict) and "observation" in candidate:
                public = candidate["observation"]
                break
        if public is None:
            raise ValueError("missing rendered public observation")
        decision = decision_record(self.reference.decide(AgentObservation(**public)))
        name = decision.pop("kind").lower()
        if name == "ask":
            decision["wait"].pop("kind")
        call = ToolCall(name, decision)
        text = frame(call) if "Kimi" in self.model else synthetic_frame(self.model, call)
        tokens = self.tokenizer.encode(text)
        if len(tokens) >= prompt.max_tokens:
            raise ValueError("mock fixture exceeds explicit output setting")
        result = self.pb.InferenceResponse(
            results=[self.pb.InferenceResult(text=text, token_ids=tokens)],
            usage=self.pb.Usage(
                prompt_tokens=len(prompt.input_ids),
                completion_tokens=len(tokens),
                total_tokens=len(prompt.input_ids) + len(tokens),
            ),
        )
        ident = "offline-" + str(uuid4())
        self.pending[ident] = result
        self.remaining[ident] = self.pending_polls
        return self.pb.AsyncResponse(request_id=ident)

    def retrieve(self, request: Any) -> Any:
        ident = request.request_id
        self.polls.append(ident)
        if self.remaining[ident]:
            self.remaining[ident] -= 1
            return self.pb.RetrieveFutureResponse(
                try_again=self.pb.TryAgainResponse(request_id=ident)
            )
        return self.pb.RetrieveFutureResponse(inference=self.pending.pop(ident))

    def unary_unary(
        self,
        path: str,
        *,
        request_serializer: Any,
        response_deserializer: Any,
        **_kwargs: Any,
    ) -> Any:
        def invoke(request: Any, *, timeout: float) -> Any:
            if timeout != 120.0:
                raise ValueError("unexpected per-RPC timeout")
            raw = request_serializer(request)
            if path.endswith("/InferenceGenerate"):
                result = self.generate(self.pb.InferenceGenerateRequest.FromString(raw))
            elif path.endswith("/RetrieveFuture"):
                result = self.retrieve(self.pb.RetrieveFutureRequest.FromString(raw))
            else:
                raise AssertionError("unexpected SDK network method")
            return response_deserializer(result.SerializeToString(deterministic=True))

        return invoke

    def close(self) -> None:
        pass
