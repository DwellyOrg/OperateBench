"""Independent pure-projection expectations exercised on immutable 420b984 first.

These deliberately do not derive the expected result from either API wrapper.
They also cover every function call in the complete model-response projection.
"""

import importlib
from types import SimpleNamespace as NS

import pytest

from operatebench.agents.lifecycle_contract import UndecodableArguments
from operatebench.agents.transport import ToolCall

MODULES = (
    "operatebench.agents.openai_responses",
    "operatebench.agents.xai_responses",
)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "arguments",
    [
        "not JSON",
        "[]",
        "null",
        "true",
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1e400}',
        '{"a":"\\ud800"}',
        '{"\\udfff":1}',
        '{"a":"\ud800"}',
        None,
    ],
)
def test_malformed_arguments_are_protocol_outcomes(module, arguments):
    call = importlib.import_module(module)._tool_call_from(
        NS(name="act", arguments=arguments)
    )
    assert call.name == "act"
    assert isinstance(call.arguments, UndecodableArguments)


@pytest.mark.parametrize("module", MODULES)
def test_decoded_mapping_arguments_remain_accepted(module):
    project = importlib.import_module(module)._tool_call_from
    assert project(NS(name="act", arguments={})) == ToolCall("act", {})
    assert project(NS(name="unknown", arguments={"雪": 1})) == ToolCall(
        "unknown", {"雪": 1}
    )


@pytest.mark.parametrize("module", MODULES)
def test_unicode_arguments_are_fresh_independent_results(module):
    projection = importlib.import_module(module)._tool_call_from
    item = NS(name="unknown", arguments='{"雪":{"é":["😀",null]},"z":null}')
    expected = ToolCall("unknown", {"雪": {"é": ["😀", None]}, "z": None})
    first, second = projection(item), projection(item)
    assert first == second == expected
    first.arguments["雪"]["é"].append("mutation")
    assert second == expected
    assert projection(item) == expected


@pytest.mark.parametrize("module", MODULES)
def test_text_projection_and_digest(module):
    mod = importlib.import_module(module)
    assert mod._message_text(NS(content=(NS(text="ignored"),))) == []
    assert mod._message_text(NS(content=None)) == []
    item = NS(content=[NS(text="雪"), NS(text=1), NS(), NS(text="\ud800"), NS(text="")])
    first = mod._message_text(item)
    assert first == ["雪", "\ud800", ""]
    first.append("mutation")
    assert mod._message_text(item) == ["雪", "\ud800", ""]
    assert mod._bounded_text(["é", "雪"]) == "é\n雪"
    bound = mod.MAX_TRANSIENT_TEXT_CHARACTERS
    assert mod._bounded_text(["😀" * (bound + 1)]) == "😀" * bound
    assert mod._response_id_digest(None) is None
    assert mod._response_id_digest(NS(response_id_digest_sha256="digest")) == "digest"
    error = mod._response_invalid("fixed detail")
    assert str(error) == "fixed detail"
    assert error.fault == "provider_response_invalid"


@pytest.mark.parametrize("module", MODULES)
def test_complete_projection_preserves_all_calls_and_independence(module):
    mod = importlib.import_module(module)
    parsed = NS(
        status="completed",
        model="synthetic",
        output=[
            NS(type="function_call", name="unknown", arguments='{"雪":[1]}'),
            NS(type="message", content=[NS(text="é😀")]),
            NS(type="function_call", name="act", arguments='{"a":1,"a":2}'),
            NS(type="function_call", name="unknown", arguments='{"last":true}'),
        ],
    )
    wire = NS(parsed=parsed, wire={"usage": {"input_tokens": 7, "output_tokens": 3}})
    first, second = mod.model_response_from(wire), mod.model_response_from(wire)
    assert first.model == "synthetic" and first.output_tokens == 3
    assert first.text == "é😀"
    assert len(first.tool_calls) == 3
    assert first.tool_calls[0] == ToolCall("unknown", {"雪": [1]})
    assert first.tool_calls[1].name == "act"
    assert isinstance(first.tool_calls[1].arguments, UndecodableArguments)
    assert first.tool_calls[2] == ToolCall("unknown", {"last": True})
    first.tool_calls[0].arguments["雪"].append(2)
    assert second.tool_calls[0] == ToolCall("unknown", {"雪": [1]})
    assert mod.model_response_from(wire).tool_calls[0] == second.tool_calls[0]


@pytest.mark.parametrize(
    "module,function",
    [
        ("operatebench.agents.mistral_chat", "mistral_outcome_tools"),
        ("operatebench.agents.xai_chat_completions", "xai_outcome_tools"),
    ],
)
def test_function_definitions_do_not_share_mutable_results(module, function):
    project = getattr(importlib.import_module(module), function)
    first, second = project(), project()
    assert first == second and first is not second
    assert [tool["name"] for tool in first] == [
        "act",
        "wait",
        "ask",
        "escalate",
        "complete",
        "retrieve",
    ]
    assert all(
        list(tool) == ["name", "description", "parameters", "strict"] for tool in first
    )
    first[0]["parameters"]["properties"].clear()
    assert second[0]["parameters"]["properties"]
    assert project() == second
