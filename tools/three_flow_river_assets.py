"""Pinned native rendering and tokenizer data; no remote Python execution."""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import base64
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from operatebench.agents.lifecycle_contract import INSTRUCTIONS_CONTRACT, outcome_tools
from operatebench.agents.transport import ToolCall
from tools.three_flow_river_kimi import ARG, CALL, CLOSE, END, SECTION


def catalog() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(Path(__file__).with_suffix(".json").read_text())
    return result


class KimiTokenizer:
    def __init__(self, cache: Path, row: dict[str, Any]):
        from tools.three_flow_river_native import _row

        pinned = _row(row["model"])
        if any(
            row.get(field) != pinned.get(field)
            for field in ("assets", "revision", "tokenizer_model")
        ):
            raise ValueError("River tokenizer model/asset binding mismatch")
        source = row.get("tokenizer_model", row["model"])
        directory = cache / source
        raw = {}
        for name in ("tokenizer_config.json", "tiktoken.model"):
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["assets"][name]["sha256"]:
                raise ValueError("River asset digest mismatch")
            raw[name] = data
        config = json.loads(raw["tokenizer_config.json"])
        ranks = {
            base64.b64decode(k): int(v)
            for k, v in (line.split() for line in raw["tiktoken.model"].splitlines())
        }
        known = {v["content"]: int(k) for k, v in config["added_tokens_decoder"].items()}
        specials = {
            next((k for k, v in known.items() if v == i), f"<|reserved_token_{i}|>"): i
            for i in range(len(ranks), len(ranks) + 256)
        }
        tiktoken = importlib.import_module("tiktoken")
        self.native = tiktoken.Encoding(
            name="river-pinned-kimi",
            pat_str="|".join(
                json.loads(
                    Path(__file__)
                    .with_name("three_flow_river_kimi_pattern.json")
                    .read_text()
                )
            ),
            mergeable_ranks=ranks,
            special_tokens=specials,
        )
        self.model = row["model"]

    def encode(self, text: str) -> list[int]:
        # Native pure tokenizer chunks at 400k characters or 25k consecutive
        # whitespace/nonwhitespace. Reject that domain rather than silently drift.
        import re

        if len(text) > 400_000 or re.search(r"\s{25001}|\S{25001}", text):
            raise ValueError("native tokenizer chunking boundary exceeded")
        return list(self.native.encode(text, allowed_special="all"))

    def decode(self, ids: list[int]) -> str:
        return str(self.native.decode(ids))


def render(prompt: dict[str, Any], *, thinking: bool = True) -> str:
    # Text-only, fresh observation on every request: no lossy assistant-history
    # parser. Matches public KimiRenderer tool_declare + system/user headers.
    tools = [
        {k: v for k, v in tool.items() if k not in ("type", "strict")}
        for tool in outcome_tools()
    ]
    return (
        "<|im_system|>tool_declare<|im_middle|>"
        + json.dumps(tools, ensure_ascii=False)
        + "<|im_end|><|im_system|>system<|im_middle|>"
        + INSTRUCTIONS_CONTRACT
        + "<|im_end|><|im_user|>user<|im_middle|>"
        + json.dumps(prompt, ensure_ascii=False)
        + "<|im_end|>"
        + "<|im_assistant|>assistant<|im_middle|><think>"
        + ("" if thinking else "</think>")
    )


def frame(call: ToolCall) -> str:
    return (
        "</think>"
        + SECTION
        + CALL
        + "functions."
        + str(call.name)
        + ":0"
        + ARG
        + json.dumps(call.arguments, ensure_ascii=False)
        + CLOSE
        + END
        + "<|im_end|>"
    )


def family_tokenizer(cache: Path, row: dict[str, Any]) -> Any:
    if "Kimi" in row["model"]:
        return KimiTokenizer(cache, row)
    from tools.three_flow_river_native import NativeTokenizer

    return NativeTokenizer(row["model"], cache)


def family_render(
    model: str, prompt: dict[str, Any], cache: Path, *, mode_profile: str = "legacy-v1"
) -> str:
    if mode_profile not in ("legacy-v1", "off-or-minimum-v1"):
        raise ValueError("unknown reasoning mode profile")
    if "Kimi" in model:
        return render(prompt, thinking=mode_profile == "legacy-v1")
    from tools.three_flow_river_native import (
        native_profile,
        render_messages,
        reserved_sequences,
    )

    # Encode the domain table so announcing reserved tokens is not itself an
    # out-of-domain history string. This is disclosure, not repair of user data.
    table = (
        json.dumps(reserved_sequences(model), ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("[gMASK]", "\\u005bgMASK]")
    )
    announcement = (
        "Interface profile: " + native_profile(model) + ". Conditional string domain: "
        "verbatim top-level tool strings and history text must not contain the "
        "reserved sequences in this JSON-escaped table: " + table + ". "
        "Nested object/array JSON strings are unrestricted. Observable domain "
        "violations are non-scored interface exclusions, not model failures. "
        "Only the in-domain interpretation is scored; hidden collisions cannot "
        "be detected. Use the declared native tool grammar, not JSON-only output."
    )
    return render_messages(
        model,
        [
            {"role": "system", "content": INSTRUCTIONS_CONTRACT + "\n\n" + announcement},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        outcome_tools(),
        cache,
        mode_profile=mode_profile,
    )


def synthetic_frame(model: str, call: ToolCall) -> str:
    """Synthetic mock output only; never used to repair a provider response."""
    name, args = call.name, call.arguments
    if "DeepSeek" in model:
        space = " " if "V4.1" in model else ""
        root = " calls" if space else "tool_calls"
        params = "\n".join(
            f'<｜DSML｜{space}parameter name="{k}" string="{str(isinstance(v, str)).lower()}">{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</｜DSML｜{space}parameter>'
            for k, v in args.items()
        )
        return (
            "</think>"
            + f'<｜DSML｜{root}>\n<｜DSML｜{space}invoke name="{name}">\n{params}\n</｜DSML｜{space}invoke>\n</｜DSML｜{root}>'
        )
    if "GLM" in model:
        params = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</arg_value>"
            for k, v in args.items()
        )
        return "</think>" + f"<tool_call>{name}{params}</tool_call>"
    params = "".join(
        f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n"
        for k, v in args.items()
    )
    return (
        "</think>" + f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
    )
