"""Synthetic controls for the Standard-only served tier."""

import json

import pytest
from openai.types.responses import Response

from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tests.openai_transport import responses_body
from tools.three_flow_profiles import OPENAI62_MODELS
from tools.three_flow_selected import check_openai62_service_tier


@pytest.mark.parametrize("_,model", OPENAI62_MODELS)
@pytest.mark.parametrize(
    "raw,typed",
    [
        ("default", "default"),
        (None, None),
        ("absent", "absent"),
        ("auto", "auto"),
        ("flex", "flex"),
        ("priority", "priority"),
        ("unknown", "unknown"),
        (True, True),
        (0, 0),
        ("default", "priority"),
        ("priority", "default"),
        ("default", "absent"),
        ("absent", "default"),
    ],
)
def test_served_tier(model, _, raw, typed):
    body = responses_body([])
    body["model"] = model
    body["usage"]["input_tokens_details"] = {"cached_tokens": 0, "cache_write_tokens": 0}
    body["usage"]["output_tokens_details"] = {"reasoning_tokens": 0}
    if raw != "absent":
        body["service_tier"] = raw
    else:
        body.pop("service_tier", None)
    typed_body = dict(body)
    if typed == "absent":
        typed_body.pop("service_tier", None)
    else:
        typed_body["service_tier"] = typed
    parsed = Response.model_construct(**typed_body)
    wire = WireResponse(json.dumps(body), lambda: parsed, kind="synthetic")
    if raw == typed == "default":
        check_openai62_service_tier(wire, model=model)
    else:
        with pytest.raises(AdapterProviderError):
            check_openai62_service_tier(wire, model=model)


def test_other_models_keep_existing_behavior():
    body = responses_body([])
    body["service_tier"] = "priority"
    wire = WireResponse(
        json.dumps(body), lambda: Response.model_construct(**body), kind="test"
    )
    check_openai62_service_tier(wire, model="gpt-5.6-luna")
