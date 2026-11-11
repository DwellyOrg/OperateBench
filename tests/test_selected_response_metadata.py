"""Invented structural responses for narrowly observed SDK compatibility."""

import json

import pytest
from anthropic.types import Message
from openai.types.responses import Response

from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.openai_responses import check_server_extensions
from operatebench.providers.wire import WireResponse
from tests.openai_transport import responses_body
from tools.three_flow_selected import check_anthropic_reading, check_openai_usage_reading


@pytest.mark.parametrize("value", [None, {"cyber": "standard"}])
def test_closed_access_metadata(value):
    check_server_extensions({"access_programs": value})


@pytest.mark.parametrize(
    "value",
    [{}, {"cyber": False}, {"cyber": "other"}, {"cyber": "standard", "unknown": 0}, []],
)
def test_unknown_access_metadata_refused(value):
    with pytest.raises(AdapterProviderError):
        check_server_extensions({"access_programs": value})


@pytest.mark.parametrize("mutation", [None, "bool", "subset", "unknown", "disagree"])
def test_openai_usage_and_timestamp_reading(mutation):
    body = responses_body([])
    body["usage"]["input_tokens_details"] = {"cache_write_tokens": 4, "cached_tokens": 2}
    body["usage"]["output_tokens_details"] = {"reasoning_tokens": 1}
    typed = Response.model_validate(body)
    if mutation == "bool":
        body["usage"]["input_tokens_details"]["cache_write_tokens"] = True
    elif mutation == "subset":
        body["usage"]["input_tokens_details"]["cache_write_tokens"] = 1000
    elif mutation == "unknown":
        body["usage"]["input_tokens_details"]["unknown_billable"] = 1
    elif mutation == "disagree":
        typed.usage.input_tokens += 1
    else:
        check_openai_usage_reading(
            WireResponse(json.dumps(body), lambda: typed, kind="test")
        )
        return
    with pytest.raises(AdapterProviderError):
        check_openai_usage_reading(
            WireResponse(json.dumps(body), lambda: typed, kind="test")
        )


@pytest.mark.parametrize("mutation", [None, "speed", "thinking", "cache", "disagree"])
def test_anthropic_original_usage_reading(mutation):
    body = {
        "id": "msg_test",
        "model": "test",
        "type": "message",
        "role": "assistant",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 8,
            "output_tokens": 5,
            "speed": "standard",
            "output_tokens_details": {"thinking_tokens": 2},
            "cache_creation_input_tokens": 3,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 3,
                "ephemeral_1h_input_tokens": 0,
            },
        },
    }
    typed = Message.model_validate(body)
    if mutation == "speed":
        body["usage"]["speed"] = False
    elif mutation == "thinking":
        body["usage"]["output_tokens_details"]["thinking_tokens"] = 6
    elif mutation == "cache":
        body["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] = -1
    elif mutation == "disagree":
        typed.usage.input_tokens = 9
    else:
        check_anthropic_reading(
            WireResponse(json.dumps(body), lambda: typed, kind="test")
        )
        return
    with pytest.raises(AdapterProviderError):
        check_anthropic_reading(
            WireResponse(json.dumps(body), lambda: typed, kind="test")
        )


@pytest.mark.parametrize("typed", [{"value": 1, "extra": None}, {"value": True}])
def test_agreement_refuses_unstated_fields_and_bool_coercion(typed):
    from tools.three_flow_selected import _agree

    with pytest.raises(AdapterProviderError):
        _agree({"value": 1}, typed)
