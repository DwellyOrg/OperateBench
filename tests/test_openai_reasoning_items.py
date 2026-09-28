"""Strict reasoning-item compatibility; all content here is synthetic."""

from __future__ import annotations

import copy
import json
import socket

import httpx
import pytest
from openai.types.responses import Response

from operatebench.providers import openai_responses as op
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tests.openai_transport import function_call_item, scripted_client
from tests.test_http_null_metadata import response_body
from tools.three_flow_http import HTTPCampaignTransport

MODEL = "gpt-test-20990101"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)


def reasoning_item():
    return {
        "type": "reasoning",
        "id": "synthetic-reasoning",
        "summary": [{"type": "summary_text", "text": "synthetic summary"}],
        "content": [{"type": "reasoning_text", "text": "synthetic content"}],
        "encrypted_content": "synthetic-opaque",
        "status": "completed",
    }


def body_with(items):
    body = response_body("openai")
    body.update(model=MODEL, output=items)
    body["usage"].update(input_tokens=137, output_tokens=29, total_tokens=166)
    body["usage"]["output_tokens_details"] = {"reasoning_tokens": 17}
    return body


def dispatch(body):
    recording, client = scripted_client(body)
    transport = object.__new__(HTTPCampaignTransport)
    transport.provider = "openai"
    transport.model = MODEL
    transport.client = client
    transport.network_timeout = None
    try:
        result = transport._dispatch({"model": MODEL, "input": "synthetic"})
        assert recording.calls == 1
        return result
    finally:
        client.close()


def test_complete_sdk_reasoning_then_action_returns_usage_and_decodes():
    body = body_with(
        [reasoning_item(), function_call_item("complete", '{"reason":"done"}')]
    )
    Response.model_validate(body)  # Whole benign response, not a root-only stub.
    usage, response = dispatch(body)
    assert usage.input_tokens == 137
    assert usage.output_tokens == response.output_tokens == 29
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "complete"
    assert response.tool_calls[0].arguments == {"reason": "done"}
    assert response.text == ""


@pytest.mark.parametrize(
    "mode", ["minimal", "nulls", "empty", "incomplete", "in_progress"]
)
def test_documented_optional_shapes(mode):
    item = reasoning_item()
    if mode == "minimal":
        for key in ("content", "encrypted_content", "status"):
            del item[key]
    elif mode == "nulls":
        item.update(content=None, encrypted_content=None, status=None)
    elif mode == "empty":
        item.update(summary=[], content=[], encrypted_content="")
    else:
        item["status"] = mode
    usage, response = dispatch(body_with([item]))
    assert usage.output_tokens == 29
    assert response.tool_calls == ()
    assert response.text == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", None),
        ("id", 1),
        ("id", True),
        ("summary", None),
        ("summary", {}),
        ("summary", ""),
        ("content", {}),
        ("content", ""),
        ("encrypted_content", 1),
        ("encrypted_content", False),
        ("encrypted_content", {}),
        ("status", "other"),
        ("status", True),
        ("status", []),
        ("extra", None),
    ],
)
def test_malformed_reasoning_scalar_or_collection_refused(field, value):
    item = reasoning_item()
    item[field] = value
    with pytest.raises(AdapterProviderError):
        dispatch(body_with([item]))


@pytest.mark.parametrize("field", ["id", "summary", "type"])
def test_required_reasoning_fields(field):
    item = reasoning_item()
    del item[field]
    with pytest.raises(AdapterProviderError):
        dispatch(body_with([item]))


@pytest.mark.parametrize("parent", ["summary", "content"])
@pytest.mark.parametrize(
    "mutation",
    ["unknown", "missing_text", "missing_type", "wrong_type", "number", "null", "nested"],
)
def test_recursive_closed_parts(parent, mutation):
    item = reasoning_item()
    part = item[parent][0]
    if mutation == "unknown":
        part["extra"] = None
    elif mutation.startswith("missing_"):
        del part[mutation.removeprefix("missing_")]
    elif mutation == "wrong_type":
        part["type"] = "output_text"
    else:
        part["text"] = {"number": 1, "null": None, "nested": {"text": "synthetic"}}[
            mutation
        ]
    with pytest.raises(AdapterProviderError):
        dispatch(body_with([item]))


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "text",
        "opaque",
        "status",
        "drop",
        "add_null",
        "swap",
        "extra",
        "typed_number",
        "typed_tuple",
    ],
)
def test_raw_typed_disagreement_refused(mutation):
    body = body_with([reasoning_item(), function_call_item("complete", "{}")])
    if mutation == "add_null":
        del body["output"][0]["status"]
    parsed = Response.model_validate(copy.deepcopy(body))
    item = parsed.output[0]
    if mutation == "id":
        item.id = "different-synthetic"
    elif mutation == "text":
        item.summary[0].text = "different-synthetic"
    elif mutation == "opaque":
        item.encrypted_content = "different-synthetic"
    elif mutation == "status":
        item.status = "incomplete"
    elif mutation == "drop":
        item.model_fields_set.remove("status")
    elif mutation == "add_null":
        item.status = None
    elif mutation == "swap":
        parsed.output.reverse()
    elif mutation == "extra":
        item.summary[0].extra = None
    elif mutation == "typed_number":
        item.summary[0].text = 1
    else:
        item.summary = tuple(item.summary)
    wire = WireResponse(json.dumps(body), lambda: parsed, kind="synthetic reasoning")
    with pytest.raises(AdapterProviderError):
        op.check_response_admissible(wire, model=MODEL)


@pytest.mark.parametrize("case", ["none", "malformed", "multiple", "over_limit"])
def test_non_action_preserves_existing_model_outcomes(case):
    from operatebench.agents.model import (
        MAX_OUTPUT_TOKENS,
        ModelExceededOutputLimit,
        ModelReturnedMultipleToolCalls,
        ModelReturnedNoToolCall,
        ModelToolArgumentsMalformed,
    )
    from tests.test_lifecycle_openai_bridge import MODEL as LIFECYCLE_MODEL
    from tests.test_lifecycle_openai_bridge import decide_over

    items = [reasoning_item()]
    if case == "malformed":
        items.append(function_call_item("complete", '{"reason":'))
    elif case == "multiple":
        items += [function_call_item("complete", '{"reason":"done"}')] * 2
    body = body_with(items)
    body["model"] = LIFECYCLE_MODEL
    if case == "over_limit":
        body["usage"].update(
            output_tokens=MAX_OUTPUT_TOKENS + 1, total_tokens=137 + MAX_OUTPUT_TOKENS + 1
        )
    decision, recording = decide_over(body)
    assert isinstance(
        decision,
        {
            "none": ModelReturnedNoToolCall,
            "malformed": ModelToolArgumentsMalformed,
            "multiple": ModelReturnedMultipleToolCalls,
            "over_limit": ModelExceededOutputLimit,
        }[case],
    )
    assert recording.calls == 1


def test_reasoning_does_not_mask_malformed_following_wire_action():
    call = function_call_item("complete", "{}")
    call["arguments"] = 17
    with pytest.raises(AdapterProviderError):
        dispatch(body_with([reasoning_item(), call]))


def test_reasoning_contract_golden_settings_and_sdk_shape():
    from openai.types.responses.response_reasoning_item import (
        Content,
        ResponseReasoningItem,
        Summary,
    )

    from boundarybench.providers.openai_responses import openai_settings
    from operatebench.agents.openai_responses import lifecycle_openai_settings

    expected = "openai_reasoning_items_strict_non_action_v1"
    assert expected == op.OPENAI_REASONING_ITEMS_CONTRACT
    assert op.REASONING_ITEM_WIRE_SHAPE.allowed == frozenset(
        {"type", "id", "summary", "content", "encrypted_content", "status"}
    )
    assert op.REASONING_ITEM_WIRE_SHAPE.required == ("type", "id", "summary")
    assert set(ResponseReasoningItem.model_fields) == set(
        op.REASONING_ITEM_WIRE_SHAPE.allowed
    )
    assert set(Summary.model_fields) == set(Content.model_fields) == {"text", "type"}
    assert openai_settings(model=MODEL)["reasoning_items_contract"] == expected
    assert (
        lifecycle_openai_settings(model=MODEL, deadline_seconds=10)[
            "reasoning_items_contract"
        ]
        == expected
    )


def test_reasoning_combined_consuming_cli_and_zero_call_replay(tmp_path, monkeypatch):
    from tests.test_three_flow_context import (
        test_context_consuming_cli_sdk_dynamic_identity_replay,
    )
    from tools.three_flow_mock import SDKMockTransport

    original = SDKMockTransport.handle

    def handle(mock, request):
        payload = json.loads(request.content)
        # Every real campaign constructor call is a fresh single snapshot, not
        # a Responses conversation or a function-call-output continuation.
        assert type(payload["input"]) is str
        assert "previous_response_id" not in payload
        assert "conversation" not in payload
        assert payload["store"] is False
        body = json.loads(original(mock, request).content)
        body["output"].insert(0, reasoning_item())
        body["access_programs"] = None
        return httpx.Response(200, json=body)

    monkeypatch.setattr(SDKMockTransport, "handle", handle)
    test_context_consuming_cli_sdk_dynamic_identity_replay(tmp_path, "openai")
    records = list(tmp_path.rglob("record.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["provider"]["settings"]["reasoning_items_contract"] == (
        "openai_reasoning_items_strict_non_action_v1"
    )
