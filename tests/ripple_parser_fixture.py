# SPDX-License-Identifier: Apache-2.0
"""Test-only parser excerpt from river-client 0.22.0 (Apache-2.0).

Extracted from renderers/ripple.py, tool_arguments.py and base.py. TypedDict
constructors are represented by dict; no SDK client or renderer is emulated.
The SDK differential control verifies this excerpt against the installed parser.
Source SHA256: ripple b3a56c22ce21f88def102b99b9df6d1435d9557021ebde4b5581e48a12f36ace
               tool_arguments:
               451a30135bbe86b18ee7d0d874740827852acd2d519fb980464089940c0fc874
               base be213f9dfbaba0f961fb22d51b9d964ede44ff3f052581cb8cdf5493da72e34e
"""

from __future__ import annotations

import json
from typing import Any

ContentPart = ToolCall = UnparsedToolCall = ToolCallFunction = ThinkingPart = TextPart = (
    dict
)
_MAX_JSON_DEPTH = 127
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    members = dict(pairs)
    if len(members) != len(pairs):
        raise ValueError("JSON object has a duplicate key")
    return members


def _nesting_depth(raw: str) -> int:
    """The deepest array or object nesting in ``raw``, outside strings."""
    depth = deepest = 0
    in_string = escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}":
            depth -= 1
    return deepest


def standard_json(raw: str) -> Any:
    """Decode ``raw`` as standard JSON, raising ValueError for anything else.

    Python's json also accepts NaN, Infinity and numbers that overflow to
    them, and keeps only the last of duplicate keys, at any depth. No value
    must reach a tool in a form other readers would not return, or with part
    of what the model wrote silently dropped. Nesting deeper than River
    Serve's JSON reader allows is refused before decoding, which would
    otherwise exhaust the interpreter's recursion limit.
    """
    if _nesting_depth(raw) > _MAX_JSON_DEPTH:
        raise ValueError(f"JSON nests deeper than {_MAX_JSON_DEPTH} levels")
    value = json.loads(raw, object_pairs_hook=_unique_members)
    json.dumps(value, allow_nan=False)
    return value


def find_outside_json_strings(
    text: str, markers: tuple[str, ...], start: int = 0
) -> tuple[int, str | None]:
    """The first of ``markers`` in ``text`` from ``start`` outside a JSON string.

    Where a format carries JSON, marker text inside a string value is content,
    so a block runs to the first end marker outside a string. Returns
    ``(len(text), None)`` when there is none, including while a string is
    still open.
    """
    firsts = {marker[0] for marker in markers}
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in firsts:
            for marker in markers:
                if text.startswith(marker, index):
                    return index, marker
    return len(text), None


def append_visible_text(parts: list[ContentPart], text: str) -> None:
    """Append visible text that parsing lifted from around removed markup.

    Renderers write a message's text parts together, apart from the tool
    calls and reasoning that parsing removed between them, so text that
    markup separated stays separated by a newline. No marker contains one,
    so text can never join with its new neighbours into markup.
    """
    if not text:
        return
    if any(part["type"] == "text" for part in parts):
        text = "\n" + text
    parts.append(TextPart(type="text", text=text))


def _strip_one(text: str, *, leading: bool, trailing: bool) -> str:
    """Drop one framing newline the template writes around a block."""
    if leading and text.startswith("\n"):
        text = text[1:]
    if trailing and text.endswith("\n"):
        text = text[:-1]
    return text


def _parse_ripple_call(body: str, raw_text: str) -> ToolCall | UnparsedToolCall:
    """One closed block's JSON: exactly a string ``name`` and object ``arguments``.

    Any other member would be dropped from the call, so it rejects the block
    instead (R1); so does anything that is not one standard JSON object (R5).
    """
    try:
        call = standard_json(body)
    except ValueError as exc:
        return UnparsedToolCall(
            raw_text=raw_text, error=f"Ripple tool call is not valid JSON: {exc}"
        )
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return UnparsedToolCall(
            raw_text=raw_text,
            error='Ripple tool call must be {"name": ..., "arguments": {...}}',
        )
    name, arguments = call["name"], call["arguments"]
    if not isinstance(name, str) or not name:
        return UnparsedToolCall(raw_text=raw_text, error="Missing function name")
    if not isinstance(arguments, dict):
        return UnparsedToolCall(
            raw_text=raw_text, error="Ripple tool call arguments must be a JSON object"
        )
    return ToolCall(
        type="function",
        id=None,
        function=ToolCallFunction(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        ),
    )


def parse_ripple_content_blocks(
    content: str,
) -> tuple[list[ContentPart], list[ToolCall | UnparsedToolCall]] | None:
    """Split Ripple output into reasoning, visible text and tool-call blocks.

    Returns None when the output holds no block. A closed ``<think>`` block is
    reasoning whatever it holds; an unclosed one (a truncated generation) ends
    where the next call opens. A call's JSON runs to the first ``</tool_call>``
    outside a JSON string; a block the output ends inside, or that another
    ``<tool_call>`` interrupts, is rejected whole. The newlines the template
    writes around reasoning and before each call are framing, not text.
    """
    if _THINK_OPEN not in content and _TOOL_CALL_OPEN not in content:
        return None

    parts: list[ContentPart] = []
    calls: list[ToolCall | UnparsedToolCall] = []
    pos = 0
    while True:
        think_at = content.find(_THINK_OPEN, pos)
        call_at = content.find(_TOOL_CALL_OPEN, pos)
        if think_at < 0 and call_at < 0:
            break
        if call_at < 0 or 0 <= think_at < call_at:
            append_visible_text(parts, content[pos:think_at])
            body = think_at + len(_THINK_OPEN)
            close = content.find(_THINK_CLOSE, body)
            if close >= 0:
                thinking = content[body:close]
                pos = close + len(_THINK_CLOSE)
                if content.startswith("\n", pos):
                    pos += 1
            else:
                next_call = content.find(_TOOL_CALL_OPEN, body)
                pos = len(content) if next_call < 0 else next_call
                thinking = content[body:pos]
            thinking = _strip_one(thinking, leading=True, trailing=True)
            if thinking:
                parts.append(ThinkingPart(type="thinking", thinking=thinking))
            continue
        append_visible_text(
            parts, _strip_one(content[pos:call_at], leading=False, trailing=True)
        )
        body = call_at + len(_TOOL_CALL_OPEN)
        end, marker = find_outside_json_strings(
            content, (_TOOL_CALL_CLOSE, _TOOL_CALL_OPEN), body
        )
        if marker != _TOOL_CALL_CLOSE:
            calls.append(
                UnparsedToolCall(
                    raw_text=content[call_at:end], error="Unclosed Ripple tool call"
                )
            )
            pos = end
            continue
        pos = end + len(_TOOL_CALL_CLOSE)
        calls.append(_parse_ripple_call(content[body:end], content[call_at:pos]))
    append_visible_text(parts, content[pos:])
    return parts, calls
