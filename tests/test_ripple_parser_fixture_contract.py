# SPDX-License-Identifier: Apache-2.0
"""Contract examples for the frozen, test-only Ripple parser excerpt."""

from tests.ripple_parser_fixture import parse_ripple_content_blocks


def parser_cases():
    return (
        "plain text",
        '<think>private</think>\n<tool_call>{"name":"act","arguments":{"x":1}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{"x":NaN}}</tool_call>',
        '<tool_call>{"name":"act","name":"act","arguments":{}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{"x":{"a":1,"a":2}}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{},"extra":1}</tool_call>',
        '<tool_call>{"name":"act","arguments":{"x":"</tool_call>"}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{}}</tool_call>'
        '<tool_call>{"name":"wait","arguments":{}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{}}',
        '<tool_call>{"name":"act","arguments":{}<tool_call>{"name":"wait","arguments":{}}</tool_call>',
        '<tool_call>{"name":"act","arguments":{"x":'
        + "[" * 128
        + "0"
        + "]" * 128
        + "}}</tool_call>",
    )


def test_frozen_parser_contract():
    results = [parse_ripple_content_blocks(case) for case in parser_cases()]
    assert results[0] is None
    assert results[1][0][0]["type"] == "thinking"
    for index in (2, 3, 4, 5, 8, 9, 10):
        assert "error" in results[index][1][0]
    assert results[6][1][0]["function"]["name"] == "act"
    assert len(results[7][1]) == 2
