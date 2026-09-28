"""Strict River Kimi text framing. No vendor parser or model-output repair."""

from __future__ import annotations

import json
import re

from operatebench.agents.transport import ToolCall
from operatebench.providers.toolcalls import decode_tool_arguments

SECTION = "<|tool_calls_section_begin|>"
END = "<|tool_calls_section_end|>"
CALL = "<|tool_call_begin|>"
ARG = "<|tool_call_argument_begin|>"
CLOSE = "<|tool_call_end|>"


def parse_kimi(raw: str, *, reasoning_prefilled: bool = True) -> tuple[ToolCall, ...]:
    """Read exact K2 JSON argument slices before any vendor deserialization.

    raw_decode finds the end of a JSON value respecting quotes/escapes. Its
    value is discarded; the untouched slice goes to the shared strict decoder.
    Marker-looking string values are data. No inferred names or trailing repair.
    """
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError("invalid River output")
    # The generation prompt prefills <think>. The first closing reasoning
    # marker ends that region; tool-looking text inside reasoning is not a call.
    if reasoning_prefilled:
        if "</think>" not in raw:
            raise ValueError("incomplete reasoning frame")
        rest = raw.split("</think>", 1)[1].strip()
    else:
        rest = raw.strip()
        if rest.startswith("<think>"):
            reasoning, delimiter, rest = rest[len("<think>") :].partition("</think>")
            if not delimiter or "<think>" in reasoning:
                raise ValueError("incomplete or nested reasoning frame")
            rest = rest.strip()
        if not any(
            m in rest
            for m in (SECTION, END, CALL, ARG, CLOSE, "<think>", "</think>", "<|im_end|>")
        ):
            return ()
    prefix, marker, rest = rest.partition(SECTION)
    if not marker or any(
        m in prefix for m in (END, CALL, ARG, CLOSE, "<think>", "</think>", "<|im_end|>")
    ):
        raise ValueError("missing or ambiguous tool section")
    calls = []
    ids = set()
    while rest.startswith(CALL):
        rest = rest[len(CALL) :]
        header, sep, rest = rest.partition(ARG)
        match = re.fullmatch(r"functions\.([A-Za-z_][A-Za-z_0-9]*):([0-9]+)", header)
        if not sep or match is None or header in ids:
            raise ValueError("invalid or repeated tool identity")
        ids.add(header)
        rest = rest.lstrip()
        try:
            _, end = json.JSONDecoder().raw_decode(rest)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid argument framing") from exc
        args = decode_tool_arguments(
            rest[:end], label="River arguments", error=ValueError
        )
        rest = rest[end:].lstrip()
        if not rest.startswith(CLOSE):
            raise ValueError("missing call end")
        rest = rest[len(CLOSE) :].lstrip()
        calls.append(ToolCall(match[1], args))
    if not calls or rest not in (END, END + "<|im_end|>"):
        raise ValueError("incomplete or trailing tool frame")
    return tuple(calls)
