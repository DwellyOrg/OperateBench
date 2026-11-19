"""Pinned Ripple checkpoint and official SDK renderer; no SDK Client."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

from operatebench.agents.lifecycle_contract import INSTRUCTIONS_CONTRACT, outcome_tools
from operatebench.agents.transport import ToolCall

MODEL = "ricursiveAI/Ripple-65B-Preview-NVFP4"
PROVENANCE_SHA256 = "784091f92ab08b07ef3ef527cb5cf2c6a8779c7829ef3d38ada4061349e307e4"
ASSETS = {
    "chat_template.jinja": (
        "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3"
    ),
    "special_tokens_map.json": (
        "3f7cd7d0cfb20f9faa47c75fd4df91b1c9a78aef0be27ded12cb705b23951341"
    ),
    "tokenizer.json": "18e309ad7f9c60037eaf26aca401a96128187cad2bdfa01a508dca7526bf6300",
    "tokenizer_config.json": (
        "eef580161a5d941607f7ec7ca95c50bfcd24812d2921c110ac049c620345ed85"
    ),
}


class RippleTokenizer:
    def __init__(self, root: Path) -> None:
        from tokenizers import Tokenizer

        provenance = (root / "provenance.json").read_bytes()
        if hashlib.sha256(provenance).hexdigest() != PROVENANCE_SHA256:
            raise ValueError("Ripple public provenance digest mismatch")
        metadata = json.loads(provenance)
        self.revision = metadata["revision"]
        if metadata["repo"] != MODEL:
            raise ValueError("Ripple public provenance identity mismatch")
        data = {}
        for name, expected in ASSETS.items():
            raw = (root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Ripple asset digest mismatch")
            data[name] = raw
        tokenizer_class: Any = importlib.import_module(
            "transformers"
        ).PreTrainedTokenizerFast
        renderer_class: Any = importlib.import_module(
            "river_client.renderers.ripple"
        ).RippleRenderer
        native = tokenizer_class(
            tokenizer_object=Tokenizer.from_str(data["tokenizer.json"].decode("utf-8")),
            chat_template=data["chat_template.jinja"].decode("utf-8"),
        )
        self.native = native
        self.renderer = renderer_class(native)

    def encode(self, text: str) -> list[int]:
        return list(self.native.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return str(self.native.decode(ids, skip_special_tokens=False))

    def render(self, prompt: dict[str, Any]) -> str:
        return cast(
            str,
            self.renderer.build_prompt_str(
                [
                    {"role": "system", "content": INSTRUCTIONS_CONTRACT},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                tools=outcome_tools(),
            ),
        )

    def parse(self, text: str) -> tuple[ToolCall, ...]:
        message = self.renderer.parse_response(text).message
        if message.get("unparsed_tool_calls"):
            return ()
        result = []
        for call in message.get("tool_calls", []):
            args = json.loads(cast(str, call["function"]["arguments"]))
            if not isinstance(args, dict):
                return ()
            result.append(ToolCall(call["function"]["name"], args))
        return tuple(result)


def validate_replay(record: dict[str, Any], assets: Path) -> None:
    """Reparse saved generated protobufs with the pinned official renderer."""
    import base64
    from dataclasses import replace
    from decimal import Decimal

    from operatebench.agents.model import (
        AGENT_TOOL_NAMES,
        MAX_OUTPUT_TOKENS,
        MODEL_PROTOCOL_VERSION,
        ModelAgent,
    )
    from operatebench.agents.playback import decision_record
    from operatebench.agents.transport import (
        ForbiddenTransport,
        ModelRequest,
        content_digest,
    )
    from tools.three_flow_river import RiverCampaignTransport, sdk

    tokenizer = RippleTokenizer(assets)
    pb, _ = sdk(ripple=True)
    parser = object.__new__(RiverCampaignTransport)
    parser.pb, parser.tokenizer, parser.model = pb, tokenizer, MODEL
    parser.max_output_tokens, parser.ripple = 30000, True
    parser.reasoning_prefilled, parser.mode_profile = False, "ripple-default-v1"
    settings = record["provider"]["settings"]
    if (
        settings.get("assets") != ASSETS
        or settings.get("tokenizer_revision") != tokenizer.revision
        or settings.get("context_request_bound") != 262144
        or settings.get("owner_output_cap_tokens") != 30000
    ):
        raise ValueError("Ripple replay profile differs")

    def decode(row: dict[str, Any], side: str, cls: Any) -> Any:
        raw = base64.b64decode(row[side + "_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != row[side + "_sha256"]:
            raise ValueError("Ripple protobuf checksum differs")
        message = cls.FromString(raw)
        known = cls()
        known.CopyFrom(message)
        known.DiscardUnknownFields()
        if known.SerializeToString(deterministic=True) != message.SerializeToString(
            deterministic=True
        ):
            raise ValueError("Ripple protobuf contains unknown fields")
        return message

    for turn, expected_digest, decision in zip(
        record["provider_turns"],
        record["request_digests"],
        record["decisions"],
        strict=True,
    ):
        wires = turn["wire"]
        if (
            not isinstance(wires, list)
            or len(wires) < 2
            or turn["network_rpcs"] != len(wires)
        ):
            raise ValueError("Ripple RPC trace differs")
        first = wires[0]
        if (
            first["rpc"] != "/river.api.v1.RiverService/InferenceGenerate"
            or first["model_submission"] is not True
        ):
            raise ValueError("Ripple generation trace differs")
        request = decode(first, "request", pb.InferenceGenerateRequest)
        ack = decode(first, "response", pb.AsyncResponse)
        if request.base_model != MODEL or len(request.prompts) != 1 or not ack.request_id:
            raise ValueError("Ripple request identity differs")
        prompt = request.prompts[0]
        ids = list(prompt.input_ids)
        rendered = tokenizer.decode(ids)
        prefix = "<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        if prefix not in rendered or not rendered.endswith(suffix):
            raise ValueError("Ripple native prompt framing differs")
        prompt_data = json.loads(rendered.rsplit(prefix, 1)[1][: -len(suffix)])
        if tokenizer.render(prompt_data) != rendered:
            raise ValueError("Ripple native instruction or tool rendering differs")
        observation = prompt_data["observation"]
        model_request = ModelRequest(
            model=MODEL,
            protocol_version=MODEL_PROTOCOL_VERSION,
            max_output_tokens=prompt.max_tokens,
            tool_names=AGENT_TOOL_NAMES,
            observation_digest_sha256=content_digest(observation),
            prompt_digest_sha256=content_digest(prompt_data),
            invocation_index=observation["invocation_index"],
            turn_index=observation["turn_index"],
            prompt=prompt_data,
        )
        from operatebench.agents.lifecycle_contract import check_model_request

        check_model_request(
            replace(model_request, max_output_tokens=MAX_OUTPUT_TOKENS), model=MODEL
        )
        if (
            model_request.request_digest_sha256 != expected_digest
            or not 0 < prompt.max_tokens <= 30000
        ):
            raise ValueError("Ripple request identity or output differs")
        if len(ids) + prompt.max_tokens > 262144 or turn["input_bound"] != len(ids):
            raise ValueError("Ripple context bound differs")
        final = None
        for row in wires[1:]:
            if (
                row["rpc"] != "/river.api.v1.RiverService/RetrieveFuture"
                or row["model_submission"] is not False
            ):
                raise ValueError("Ripple retrieval trace differs")
            poll = decode(row, "request", pb.RetrieveFutureRequest)
            answer = decode(row, "response", pb.RetrieveFutureResponse)
            if poll.request_id != ack.request_id:
                raise ValueError("Ripple retrieval identity differs")
            variant = answer.WhichOneof("response")
            if (
                variant == "try_again"
                and answer.try_again.request_id == ack.request_id
                and final is None
            ):
                continue
            if variant != "inference" or final is not None:
                raise ValueError("Ripple retrieval result differs")
            final = answer.inference
        if final is None:
            raise ValueError("Ripple inference missing")
        response = parser.response(final, ids, max_output_tokens=prompt.max_tokens)
        if (
            turn["input_tokens"] != final.usage.prompt_tokens
            or turn["output_tokens"] != final.usage.completion_tokens
            or turn["requested_output_bound"] != prompt.max_tokens
            or Decimal(turn["measured_usd"])
            != (Decimal(turn["input_tokens"]) + Decimal(turn["output_tokens"]) * 3)
            / Decimal(1000000)
        ):
            raise ValueError("Ripple recorded usage differs")
        outcome = ModelAgent(
            ForbiddenTransport(), model=MODEL, max_output_tokens=prompt.max_tokens
        )._parse(response)
        if decision_record(outcome) != decision["outcome"]:
            raise ValueError("Ripple raw decision differs")
