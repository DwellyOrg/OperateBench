"""Synthetic observed cache TTL accounting metadata."""

import json

from openai.types.responses import Response

from operatebench.providers import openai_responses as op
from operatebench.providers.wire import WireResponse
from tests.openai_transport import responses_body
from tools.three_flow_selected import check_openai_usage_reading


def body_with_ttl():
    body = responses_body([])
    body["usage"]["output_tokens_details"] = {"reasoning_tokens": 1}
    body["image"] = {"context_compression": "disabled"}
    body["input_transformations"] = []
    body["usage"]["input_tokens_details"] = {
        "cached_tokens": 2,
        "cache_write_tokens": 4,
        "cache_write_tokens_details": {"ttl_30m_tokens": 1, "ttl_12h_tokens": 3},
    }
    return body


def test_observed_ttl_positive():
    body = body_with_ttl()
    wire = WireResponse(
        json.dumps(body), lambda: Response.model_validate(body), kind="test"
    )
    op.check_response_admissible(wire, model=body["model"])
    check_openai_usage_reading(wire)


def _validate(body, typed_body=None):
    from copy import deepcopy

    typed_source = deepcopy(body if typed_body is None else typed_body)
    wire = WireResponse(
        json.dumps(body), lambda: Response.model_validate(typed_source), kind="test"
    )
    op.check_response_admissible(wire, model=body["model"])
    check_openai_usage_reading(wire)
    return op.response_usage(wire)


def test_legacy_and_unchanged_pricing():
    from copy import deepcopy

    body = body_with_ttl()
    legacy = deepcopy(body)
    del legacy["usage"]["input_tokens_details"]["cache_write_tokens_details"]
    assert _validate(body) == _validate(legacy)


def test_ttl_invalid_shapes_and_partitions():
    from copy import deepcopy

    import pytest

    from operatebench.providers.faults import AdapterProviderError

    original = body_with_ttl()
    for value in [
        None,
        [],
        0,
        {},
        {"ttl_30m_tokens": 4},
        {"ttl_30m_tokens": 1, "ttl_12h_tokens": 3, "other": 0},
        {"ttl_30m_tokens": True, "ttl_12h_tokens": 3},
        {"ttl_30m_tokens": 1.0, "ttl_12h_tokens": 3},
        {"ttl_30m_tokens": -1, "ttl_12h_tokens": 5},
        {"ttl_30m_tokens": 1, "ttl_12h_tokens": 2},
    ]:
        body = deepcopy(original)
        body["usage"]["input_tokens_details"]["cache_write_tokens_details"] = value
        with pytest.raises(AdapterProviderError):
            _validate(body)
    for mutation in ("missing_aggregate", "boolean_aggregate", "overlap", "unknown"):
        body = deepcopy(original)
        details = body["usage"]["input_tokens_details"]
        if mutation == "missing_aggregate":
            del details["cache_write_tokens"]
        elif mutation == "boolean_aggregate":
            details["cache_write_tokens"] = True
        elif mutation == "overlap":
            details["cached_tokens"] = body["usage"]["input_tokens"]
        else:
            details["unknown"] = 0
        with pytest.raises(AdapterProviderError):
            _validate(body)


def test_ttl_raw_typed_disagreement():
    from copy import deepcopy

    import pytest

    from operatebench.providers.faults import AdapterProviderError

    raw = body_with_ttl()
    for mutation in ("missing", "different", "bool", "extra"):
        typed = deepcopy(raw)
        details = typed["usage"]["input_tokens_details"]
        if mutation == "missing":
            del details["cache_write_tokens_details"]
        elif mutation == "different":
            details["cache_write_tokens_details"] = {
                "ttl_30m_tokens": 2,
                "ttl_12h_tokens": 2,
            }
        elif mutation == "bool":
            details["cache_write_tokens_details"]["ttl_30m_tokens"] = True
        else:
            details["cache_write_tokens_details"]["extra"] = 0
        with pytest.raises(AdapterProviderError):
            _validate(raw, typed)
