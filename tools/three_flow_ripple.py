"""Pinned Ripple checkpoint and official SDK renderer; no SDK Client."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, cast

from operatebench.agents.lifecycle_contract import INSTRUCTIONS_CONTRACT, outcome_tools
from operatebench.agents.transport import ToolCall

MODEL = "ricursiveAI/Ripple-65B-Preview-NVFP4"
PROVENANCE_SHA256 = "784091f92ab08b07ef3ef527cb5cf2c6a8779c7829ef3d38ada4061349e307e4"
CATALOG_SHA256 = "df2c6a6836b8ce572769d6ccc214dac74a60b3a87836946e461ce64491d1d72f"
ASSET_NAMES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _catalog() -> dict[str, str]:
    raw = Path(__file__).with_name("three_flow_river_ripple_assets.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
        raise ValueError("Ripple catalog digest mismatch")
    catalog = json.loads(raw)
    if not isinstance(catalog, dict) or set(catalog) != {"assets"}:
        raise ValueError("invalid Ripple catalog")
    records = catalog["assets"]
    if not isinstance(records, list) or len(records) != len(ASSET_NAMES):
        raise ValueError("invalid Ripple catalog records")
    assets: dict[str, str] = {}
    for record, name in zip(records, ASSET_NAMES, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"filename", "sha256"}
            or record["filename"] != name
            or name in assets
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is None
        ):
            raise ValueError("invalid Ripple catalog member")
        assets[name] = record["sha256"]
    return assets


ASSETS = _catalog()


def read_verified_assets(root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read pinned files from one owner-provided directory, without following links."""
    expected = _catalog()
    if expected != ASSETS:
        raise ValueError("Ripple asset catalog changed")
    if root.anchor != "/" or ".." in root.parts:
        raise ValueError("canonical absolute Ripple asset directory required")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            next_directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory

        def read_member(name: str, digest: str) -> bytes:
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
                    raise ValueError("invalid Ripple asset file")
                chunks = []
                remaining = before.st_size
                while remaining:
                    chunk = os.read(fd, min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("short Ripple asset file")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                after = os.fstat(fd)
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("Ripple asset changed during read")
                raw = b"".join(chunks)
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError("Ripple asset digest mismatch")
                return raw
            finally:
                os.close(fd)

        provenance = read_member("provenance.json", PROVENANCE_SHA256)
        metadata = json.loads(provenance)
        if not isinstance(metadata, dict) or metadata.get("repo") != MODEL:
            raise ValueError("Ripple public provenance identity mismatch")
        return {name: read_member(name, expected[name]) for name in ASSET_NAMES}, metadata
    finally:
        os.close(directory)


class RippleTokenizer:
    def __init__(self, root: Path) -> None:
        data, metadata = read_verified_assets(root)
        self.revision = metadata["revision"]
        backend: Any = importlib.import_module("tokenizers").Tokenizer
        tokenizer_class: Any = importlib.import_module(
            "transformers"
        ).PreTrainedTokenizerFast
        renderer_class: Any = importlib.import_module(
            "river_client.renderers.ripple"
        ).RippleRenderer
        native = tokenizer_class(
            tokenizer_object=backend.from_str(data["tokenizer.json"].decode("utf-8")),
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
