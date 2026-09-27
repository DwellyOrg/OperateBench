"""Opt-in, separately reported River limited native profiles (not Core changes).

Only standard-library imports at import time. See docs/THREE_FLOW_RIVER.md
for the announced string domain, exclusion policy and unavoidable hidden intent
limitation. No SDK parser, network, credentials, weights or output repair.
"""

# Fullwidth DSML delimiters are intentional protocol bytes.
# ruff: noqa: RUF001, E501
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NativeAssetError(RuntimeError):
    """Trusted local pinned asset unavailable or corrupt; never a provider fault."""


class InterfaceAmbiguityError(Exception):
    """Non-scored interface exclusion, NEVER a malformed model decision."""

    def __init__(self, message: str, *, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class NativeSyntaxError(ValueError):
    """Genuinely malformed native syntax; caller retains raw and grades normally."""

    def __init__(self, message: str, *, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class DecodedResponse:
    profile: str
    raw: str
    tool_calls: tuple[Any, ...]


def _row(model: str) -> dict[str, Any]:
    from tools.three_flow_river_assets import catalog

    for row in catalog()["models"]:
        if model in (row["model"], row["model_slug"]):
            return dict(row)
    raise InterfaceAmbiguityError("Unknown catalog identity")


def _family(model: str) -> str:
    name = _row(model)["model"]
    for needle, family in (
        ("Qwen/", "qwen"),
        ("GLM-", "glm"),
        ("DeepSeek-", "deepseek"),
        ("Nemotron-", "nemotron"),
    ):
        if needle in name:
            return family
    raise InterfaceAmbiguityError("Kimi remains on its existing adapter")


def native_profile(model: str) -> str:
    family = _family(model)
    if family == "deepseek":
        family += "-v41" if "V4.1" in _row(model)["model"] else "-v4"
    return f"river-{family}-limited-native-v1"


def _ds(model: str) -> tuple[str, str, str]:
    return (
        (" calls", " invoke", " parameter")
        if "V4.1" in _row(model)["model"]
        else ("tool_calls", "invoke", "parameter")
    )


def reserved_sequences(model: str) -> tuple[str, ...]:
    """Frame tokens/prefixes forbidden in verbatim top-level string values."""
    family = _family(model)
    common = ("<think>", "</think>", "<tool_response>", "</tool_response>")
    if family == "deepseek":
        return (
            *common,
            "<｜DSML｜",
            "</｜DSML｜",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜System｜>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<tool_result>",
            "</tool_result>",
        )
    return (
        *common,
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<|im_start|>",
        "<|im_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "[gMASK]",
        "<sop>",
        "<|endoftext|>",
    )


def _string_domain(model: str, value: str) -> None:
    if any(marker in value for marker in reserved_sequences(model)):
        raise InterfaceAmbiguityError("Verbatim string outside announced native domain")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeSyntaxError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise NativeSyntaxError("Non-finite JSON value")


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise NativeSyntaxError("Non-finite JSON number")
    return result


_JSON = json.JSONDecoder(
    object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
)
_NAME = r"[A-Za-z_][A-Za-z_0-9]*"


def _schemas(tools: Any) -> dict[str, Any]:
    if isinstance(tools, dict):
        return tools
    result = {}
    for tool in tools:
        function = tool.get("function", tool)
        result[function["name"]] = function["parameters"]
    return result


def _types(schema: dict[str, Any]) -> set[str]:
    declared = schema.get("type", [])
    result = {declared} if isinstance(declared, str) else set(declared)
    for branch in schema.get("anyOf", []):
        result |= _types(branch)
    if not result:
        raise InterfaceAmbiguityError("Schema does not determine native value grammar")
    return result


class _Cursor:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def take(self, token: str) -> None:
        if not self.text.startswith(token, self.pos):
            raise NativeSyntaxError("Missing native frame delimiter")
        self.pos += len(token)

    def match(self, pattern: str) -> re.Match[str]:
        match = re.compile(pattern).match(self.text, self.pos)
        if match is None:
            raise NativeSyntaxError("Malformed native name or header")
        self.pos = match.end()
        return match

    def at(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def value(
        self, model: str, close: str, string: bool, *, newline: bool = False
    ) -> Any:
        if newline:
            self.take("\n")
        if string:
            end = self.text.find(close, self.pos)
            if end < 0:
                raise NativeSyntaxError("Truncated string parameter")
            value = self.text[self.pos : end]
            _string_domain(model, value)
            self.pos = end
        else:
            self.ws()
            try:
                value, self.pos = _JSON.raw_decode(self.text, self.pos)
            except (ValueError, RecursionError) as exc:
                raise NativeSyntaxError("Invalid JSON parameter") from exc
            if newline:
                # JSON allows insignificant whitespace, but close includes the
                # formatter's one structural newline. Do not strip strings.
                while (
                    not self.at(close)
                    and self.pos < len(self.text)
                    and self.text[self.pos].isspace()
                ):
                    self.pos += 1
            else:
                self.ws()
        self.take(close)
        return value


def decode_response(
    model: str,
    raw: str,
    schemas: Any,
    *,
    reasoning_prefilled: bool = False,
    mode_profile: str = "legacy-v1",
) -> DecodedResponse:
    """Schema-directed exact values; no normalization or null elision.

    Supports completed thinking-prefill responses and bare native frames. A
    reasoning close, if present before the first call, ends the reasoning region.
    Ordinary answer text may precede calls; trailing prose/partial frames fail.
    Multi-calls survive unchanged for Core's invocation policy to classify.
    """
    from operatebench.agents.transport import ToolCall

    if mode_profile not in ("legacy-v1", "off-or-minimum-v1"):
        raise ValueError("unknown reasoning mode profile")
    profile = native_profile(model)
    family = _family(model)
    definitions = _schemas(schemas)
    try:
        if not isinstance(raw, str) or len(raw) > 2_000_000:
            raise NativeSyntaxError("Invalid native output size/type")
        text = raw
        invoke_tag = parameter_tag = ""
        if family == "deepseek":
            calls_tag, invoke_tag, parameter_tag = _ds(model)
            opening = f"<｜DSML｜{calls_tag}>"
            closing = f"</｜DSML｜{calls_tag}>"
            call_close = f"</｜DSML｜{invoke_tag}>"
        else:
            opening = "<tool_call>"
            closing = ""
            call_close = "</tool_call>"
        # Never search for reasoning delimiters inside an already opened call:
        # JSON strings can legally contain every marker. Explicit <think> starts
        # reasoning; a prefill close before the first frame does the same.
        if mode_profile != "legacy-v1":
            if reasoning_prefilled or text.lstrip().startswith("<think>"):
                region = text if reasoning_prefilled else text.lstrip()[len("<think>") :]
                thought, close, text = region.partition("</think>")
                if not close or "<think>" in thought:
                    raise NativeSyntaxError("Incomplete or nested reasoning")
            first = text.find(opening)
            prefix = text if first < 0 else text[:first]
            if "<think>" in prefix or "</think>" in prefix:
                raise NativeSyntaxError("Unexpected reasoning delimiter")
            if first < 0:
                _string_domain(model, text)
                return DecodedResponse(profile, raw, ())
            # The selected profile has already validated and removed reasoning.
            # Do not let the legacy delimiter inference reinterpret tool data.
            reasoning_prefilled = False
        first = text.find(opening)
        think_end = text.find("</think>")
        if reasoning_prefilled and think_end < 0:
            raise NativeSyntaxError("Incomplete prefilled reasoning")
        if (
            reasoning_prefilled
            or text.lstrip().startswith("<think>")
            or (think_end >= 0 and (first < 0 or think_end < first))
        ):
            if think_end < 0:
                raise NativeSyntaxError("Incomplete reasoning")
            text = text[think_end + len("</think>") :]
        start = text.find(opening)
        if start < 0:
            raise NativeSyntaxError("No native call")
        _string_domain(model, text[:start])
        c = _Cursor(text[start:])
        if family == "deepseek":
            c.take(opening)
            c.ws()
        calls = []
        while True:
            if family == "deepseek":
                name = c.match(rf'<｜DSML｜{invoke_tag} name="({_NAME})">')[1]
            else:
                c.take(opening)
                c.ws()
                if family == "glm":
                    name = c.match(_NAME)[0]
                else:
                    name = c.match(rf"<function=({_NAME})>")[1]
            if name not in definitions:
                raise NativeSyntaxError("Unknown tool name")
            schema = definitions[name]
            props = schema.get("properties")
            if props is None and schema.get("anyOf"):
                props = schema["anyOf"][0].get("properties")
            if props is None:
                raise InterfaceAmbiguityError("Tool schema has no parameter grammar")
            args = {}
            c.ws()
            end_params = call_close if family in ("glm", "deepseek") else "</function>"
            while not c.at(end_params):
                explicit_string = None
                if family == "deepseek":
                    match = c.match(
                        rf'<｜DSML｜{parameter_tag} name="({_NAME})" string="(true|false)">'
                    )
                    key, explicit_string = match[1], match[2] == "true"
                    param_close = f"</｜DSML｜{parameter_tag}>"
                elif family == "glm":
                    key = c.match(rf"<arg_key>({_NAME})</arg_key><arg_value>")[1]
                    param_close = "</arg_value>"
                else:
                    key = c.match(rf"<parameter=({_NAME})>")[1]
                    param_close = "\n</parameter>"
                if key in args:
                    raise NativeSyntaxError("Duplicate native parameter")
                if key not in props:
                    raise NativeSyntaxError("Unknown parameter name")
                types = _types(props[key])
                if explicit_string is None:
                    if "string" in types and types - {"string", "null"}:
                        raise InterfaceAmbiguityError(
                            "String/JSON union has multiple native interpretations"
                        )
                    string = "string" in types
                else:
                    string = explicit_string
                args[key] = c.value(
                    model, param_close, string, newline=family in ("qwen", "nemotron")
                )
                if explicit_string is False and isinstance(args[key], str):
                    raise NativeSyntaxError("DSML non-string flag contains a JSON string")
                c.ws()
            c.take(end_params)
            if family in ("qwen", "nemotron"):
                c.ws()
                c.take(call_close)
            calls.append(ToolCall(name, args))
            c.ws()
            if family == "deepseek":
                if c.at(closing):
                    c.take(closing)
                    break
            elif not c.at(opening):
                break
        c.ws()
        suffix = c.text[c.pos :]
        stops = {
            "deepseek": ("", "<｜end▁of▁sentence｜>"),
            "glm": ("", "<|observation|>", "<|user|>", "<|endoftext|>"),
            "qwen": ("", "<|im_end|>"),
            "nemotron": ("", "<|im_end|>"),
        }
        if suffix.strip() not in stops[family]:
            raise NativeSyntaxError("Trailing or incomplete native frame")
        return DecodedResponse(profile, raw, tuple(calls))
    except (NativeSyntaxError, InterfaceAmbiguityError) as exc:
        exc.raw = raw
        raise


def _asset(model: str, assets: Path, name: str) -> bytes:
    row = _row(model)
    source = row.get("tokenizer_model", row["model"])
    try:
        data = (Path(assets) / source / name).read_bytes()
        expected = row["assets"][name]["sha256"]
    except (OSError, KeyError) as exc:
        raise NativeAssetError("Pinned native asset unavailable") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise NativeAssetError("Pinned native asset digest mismatch")
    return bytes(data)


class NativeTokenizer:
    """Exact checkpoint tokenizer data; optional Rust tokenizers, no transformers.

    Full-string text encoding, not training/multimodal chunk packing. No silent
    truncation; this API deliberately rejects a requested chunking boundary.
    """

    def __init__(self, model: str, assets: Path):
        Tokenizer = importlib.import_module("tokenizers").Tokenizer

        native_profile(model)
        self.source = _row(model).get("tokenizer_model", _row(model)["model"])
        self.native = Tokenizer.from_str(_asset(model, assets, "tokenizer.json").decode())
        self.native.no_truncation()
        self.native.no_padding()
        self.model = model

    def encode(self, text: str, *, chunk_size: int | None = None) -> list[int]:
        if chunk_size is not None:
            raise InterfaceAmbiguityError(
                "Chunked/training encoding is not this text inference profile"
            )
        return list(self.native.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self.native.decode(ids, skip_special_tokens=False))


# Verbatim official DeepSeek tools instruction, also reproduced in River SDK
# renderers/deepseek.py _TOOLS_TEMPLATE. No downloaded Python is executed.
_DS_TOOLS = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜{calls}>" block like the following:

<｜DSML｜{calls}>
<｜DSML｜{invoke} name="$TOOL_NAME">
<｜DSML｜{param} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜{param}>
...
</｜DSML｜{invoke}>
<｜DSML｜{invoke} name="$TOOL_NAME2">
...
</｜DSML｜{invoke}>
</｜DSML｜{calls}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _functions(tools: Any) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in t.get("function", t).items() if k not in ("type", "strict")}
        for t in tools
    ]


def _messages(
    model: str, messages: list[dict[str, Any]], tools: Any
) -> list[dict[str, Any]]:
    """Validate history before official verbatim formatting; never silently escape."""
    definitions = _schemas(tools)
    result = []
    for original in messages:
        message = dict(original)
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise InterfaceAmbiguityError("Unsupported message role")
        if not isinstance(message.get("content", ""), str):
            raise InterfaceAmbiguityError("Only text messages supported")
        _string_domain(model, message.get("content", ""))
        if message.get("reasoning_content") is not None:
            _string_domain(model, message["reasoning_content"])
        converted = []
        for tc in message.get("tool_calls", []):
            function = dict(tc.get("function", tc))
            name = function.get("name")
            if name not in definitions:
                raise InterfaceAmbiguityError("History tool absent from schema")
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _JSON.decode(args)
                except ValueError as exc:
                    raise InterfaceAmbiguityError("Invalid historical arguments") from exc
            if not isinstance(args, dict):
                raise InterfaceAmbiguityError("History arguments must be an object")
            schema = definitions[name]
            properties = schema.get(
                "properties", schema.get("anyOf", [{}])[0].get("properties", {})
            )
            for key, value in args.items():
                if key not in properties:
                    raise InterfaceAmbiguityError("History parameter absent from schema")
                types = _types(properties[key])
                if (
                    _family(model) != "deepseek"
                    and "string" in types
                    and (not isinstance(value, str) or types - {"string", "null"})
                ):
                    raise InterfaceAmbiguityError(
                        "History string/JSON encoding is not injective"
                    )
                if not re.fullmatch(_NAME, key):
                    raise InterfaceAmbiguityError("Unrepresentable parameter name")
                if isinstance(value, str):
                    _string_domain(model, value)
                elif _family(model) in ("qwen", "nemotron") and (
                    value is None or isinstance(value, bool)
                ):
                    raise InterfaceAmbiguityError(
                        "Checkpoint scalar string filter cannot faithfully encode null/bool history"
                    )
            # JSON validity is checked recursively, without excluding markers in
            # nested JSON string data. Duplicate dict keys cannot exist here.
            json.dumps(args, ensure_ascii=False, allow_nan=False)
            function["arguments"] = args
            converted.append({"type": "function", "function": function})
        if "tool_calls" in message:
            message["tool_calls"] = converted
        result.append(message)
    return result


def _deepseek_render(
    model: str,
    messages: list[dict[str, Any]],
    tools: Any,
    generation: bool,
    *,
    thinking: bool = True,
) -> str:
    calls, invoke, param = _ds(model)
    v41 = "V4.1" in _row(model)["model"]
    specs = _functions(tools)
    block = (
        _DS_TOOLS.format(
            calls=calls,
            invoke=invoke,
            param=param,
            schemas="\n".join(json.dumps(t, ensure_ascii=False) for t in specs),
        )
        if specs
        else ""
    )
    prompt = "<｜begin▁of▁sentence｜>"
    if v41 and thinking:
        prompt += "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    if block and (not messages or messages[0]["role"] != "system"):
        prompt += ("<｜System｜>" if v41 and not thinking else "") + "\n\n" + block
    previous = None
    for index, message in enumerate(messages):
        role, content = message["role"], message.get("content", "")
        if role == "system":
            if v41 and (index > 0 or not thinking):
                prompt += "<｜System｜>"
            prompt += content + ("\n\n" + block if block and index == 0 else "")
        elif role in ("user", "tool"):
            prompt += "\n\n" if previous in ("user", "tool") else "<｜User｜>"
            prompt += (
                "<tool_result>" + content + "</tool_result>"
                if role == "tool"
                else content
            )
        else:
            # Fixed thinking=True, strip_thinking_from_history=True; tools
            # preserve history reasoning in the official encoder.
            prompt += "<｜Assistant｜>"
            prompt += (
                "<think>" + message.get("reasoning_content", "") + "</think>"
                if specs and thinking
                else "</think>"
            )
            prompt += content
            if message.get("tool_calls"):
                prompt += f"\n\n<｜DSML｜{calls}>\n"
                for tc in message["tool_calls"]:
                    function = tc["function"]
                    parameters = []
                    for key, value in function["arguments"].items():
                        string = isinstance(value, str)
                        value_text = (
                            value if string else json.dumps(value, ensure_ascii=False)
                        )
                        parameters.append(
                            f'<｜DSML｜{param} name="{key}" string="{str(string).lower()}">{value_text}</｜DSML｜{param}>'
                        )
                    prompt += (
                        f'<｜DSML｜{invoke} name="{function["name"]}">\n'
                        + "\n".join(parameters)
                        + f"\n</｜DSML｜{invoke}>\n"
                    )
                prompt += f"</｜DSML｜{calls}>"
            prompt += "<｜end▁of▁sentence｜>"
        previous = role
    return prompt + (
        "<｜Assistant｜>" + ("<think>" if thinking else "</think>") if generation else ""
    )


def _ordered_deepseek_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Official chat encoder sorts tool slots within merged user/tool groups."""
    result = list(messages)
    order: dict[str, int] = {}
    index = 0
    while index < len(result):
        message = result[index]
        if message["role"] == "assistant" and message.get("tool_calls"):
            order = {
                tc.get("id") or tc.get("function", {}).get("id", ""): i
                for i, tc in enumerate(message["tool_calls"])
                if tc.get("id") or tc.get("function", {}).get("id", "")
            }
        if message["role"] not in ("user", "tool"):
            index += 1
            continue
        end = index
        while end < len(result) and result[end]["role"] in ("user", "tool"):
            end += 1
        slots = [i for i in range(index, end) if result[i]["role"] == "tool"]
        ordered = sorted(
            (result[i] for i in slots),
            key=lambda m: order.get(m.get("tool_call_id", ""), 0),
        )
        for i, item in zip(slots, ordered, strict=True):
            result[i] = item
        index = end
    return result


def render_messages(
    model: str,
    messages: list[dict[str, Any]],
    tools: Any,
    assets: Path,
    *,
    add_generation_prompt: bool = True,
    mode_profile: str = "legacy-v1",
) -> str:
    """Native text prompt with explicit versioned thinking controls.

    tools accepts outcome_tools() or OpenAI wrapped schemas. The caller announces
    this profile out of band; this function does not secretly alter tool schemas
    or replace the native prompt with a JSON-output prompt.
    """
    if mode_profile not in ("legacy-v1", "off-or-minimum-v1"):
        raise ValueError("unknown reasoning mode profile")
    native_profile(model)
    config = json.loads(_asset(model, assets, "tokenizer_config.json"))
    if mode_profile != "legacy-v1" and _family(model) == "deepseek":
        messages = _ordered_deepseek_results(messages)
    validated = _messages(model, messages, tools)
    if _family(model) == "deepseek":
        return _deepseek_render(
            model,
            validated,
            tools,
            add_generation_prompt,
            thinking=mode_profile == "legacy-v1",
        )
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
        value, **({"ensure_ascii": False} | kwargs)
    )

    def fail(message: str) -> Any:
        raise InterfaceAmbiguityError("Checkpoint template rejected history: " + message)

    environment.globals["raise_exception"] = fail
    template = _asset(model, assets, "chat_template.jinja").decode()
    return environment.from_string(template).render(
        messages=validated,
        tools=[{"type": "function", "function": t} for t in _functions(tools)],
        bos_token=config.get("bos_token", ""),
        eos_token=config.get("eos_token", ""),
        add_generation_prompt=add_generation_prompt,
        **(
            {"enable_thinking": True}
            if mode_profile == "legacy-v1"
            else {"reasoning_effort": "low"}
            if "GLM-5.3" in model
            else {"enable_thinking": False}
        ),
    )
