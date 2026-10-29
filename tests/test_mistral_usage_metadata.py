"""Invented Mistral metadata; no provider response fixtures."""

import copy
import json

import pytest
from mistralai.client.models import ChatCompletionResponse

from operatebench.providers import mistral_chat as mp
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tests.test_lifecycle_mistral_chat import response_body


def metadata_body():
    body = response_body()
    body["usage"].update(
        prompt_tokens_details={"cached_tokens": 3},
        completion_tokens_details={"reasoning_tokens": 2},
        service_tier="standard",
    )
    body["choices"][0].update(logprobs=None, prompt_logprobs=None)
    return body


def admit(body, typed_body=None):
    parsed = ChatCompletionResponse.model_validate(
        body if typed_body is None else typed_body
    )
    wire = WireResponse(json.dumps(body), lambda: parsed, kind="invented Mistral")
    mp.check_response_admissible(wire, model=body["model"])
    return wire


def test_metadata_preserves_complete_wire_and_usage():
    body = metadata_body()
    wire = admit(body)
    assert wire.wire == body
    assert (
        wire.parsed.choices[0].message.tool_calls[0].function.arguments
        == (body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    )
    usage = mp.response_usage(wire)
    assert (usage.input_tokens, usage.output_tokens) == (137, 29)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"reasoning_tokens": True},
        {"reasoning_tokens": "2"},
        {"reasoning_tokens": -1},
        {"reasoning_tokens": 30},
        {"reasoning_tokens": 2, "unknown_billable": 1},
        {"reasoning_tokens": mp.MAX_EXACT_TOKEN_COUNT + 1},
    ],
)
def test_completion_details_remain_closed(value):
    body = metadata_body()
    body["usage"]["completion_tokens_details"] = value
    with pytest.raises(AdapterProviderError):
        admit(body)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_usage",
        "unknown_root",
        "cache_subset",
        "typed_value",
        "typed_missing",
        "typed_extra",
        "misplaced",
    ],
)
def test_unknown_fields_subsets_and_disagreement_refuse(mutation):
    body = metadata_body()
    typed = copy.deepcopy(body)
    if mutation == "unknown_usage":
        body["usage"]["unknown_billable"] = 1
    elif mutation == "unknown_root":
        body["unknown_metadata"] = None
    elif mutation == "cache_subset":
        body["usage"]["prompt_tokens_details"]["cached_tokens"] = 138
    elif mutation == "typed_value":
        typed["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
    elif mutation == "typed_missing":
        del typed["usage"]["completion_tokens_details"]
    elif mutation == "typed_extra":
        typed["usage"]["unknown_billable"] = 1
    else:
        body["choices"][0]["completion_tokens_details"] = {"reasoning_tokens": 2}
    with pytest.raises(AdapterProviderError):
        admit(body, typed)
