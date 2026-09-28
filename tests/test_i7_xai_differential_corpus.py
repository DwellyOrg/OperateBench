"""Shared offline fixtures run unchanged on db7ecc5 and the I7 candidate.

The external differential runner records actual baseline outcomes, including the
exact exception class, fault and detail. No expected refusal text is synthesized.
"""

import copy
import json

import pytest

from boundarybench.providers import xai_openai_compat as boundary
from operatebench.providers import xai_openai_compat as kernel
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tests.openai_transport import chat_body, tool_call, xai_scripted_client

MODEL = "grok-4.5"


def body():
    return chat_body([tool_call("read_records", "{}")], model=MODEL)


def site_node(value, site):
    if site == "message":
        return value["choices"][0]["message"]
    if site == "usage":
        return value["usage"]
    return value["usage"].setdefault("prompt_tokens_details", {})


def parsed(value):
    # Explicit JSON bytes retain lone surrogate escapes for the real SDK reader;
    # httpx's convenience JSON encoder would fail before that reader is exercised.
    _, client = xai_scripted_client(json.dumps(value).encode("utf-8"))
    try:
        return client.chat.completions.with_raw_response.create(
            model=MODEL, messages=[]
        ).parse()
    finally:
        client.close()


def exercise(module, path, raw, typed):
    if path == "wire":
        module.check_wire_completion(raw)
    elif path == "agreement":
        module.check_response_extensions_agree(raw, parsed(typed))
    else:
        response = WireResponse(
            json.dumps(raw), lambda: parsed(typed), kind="xai completion"
        )
        module.check_response_admissible(response, model=MODEL)


def corpus():
    cases = []

    def add(name, raw, typed=None):
        cases.append((name, raw, copy.deepcopy(raw) if typed is None else typed))

    add("absent", body())
    for site, field in (
        ("usage", "cost_in_usd_ticks"),
        ("usage", "num_sources_used"),
        ("details", "image_tokens"),
        ("details", "text_tokens"),
    ):
        for label, value in (
            ("zero", 0),
            ("max", 9007199254740991),
            ("max-plus-one", 9007199254740992),
            ("negative", -1),
            ("true", True),
            ("false", False),
            ("float", 1.0),
            ("string", "1"),
            ("null", None),
        ):
            raw = body()
            node = site_node(raw, site)
            if site == "details":
                node.update(image_tokens=0, text_tokens=1)
            node[field] = value
            add(f"count-{field}-{label}", raw)
    for label, value in (
        ("empty", ""),
        ("ascii", "reasoning"),
        ("non-bmp", "\U0001f680"),
        ("surrogate", "\ud800"),
        ("integer", 1),
        ("boolean", True),
        ("float", 1.0),
        ("null", None),
        ("list", []),
        ("object", {}),
        ("boundary-at", "x" * 16384),
        ("boundary-over", "x" * 16385),
        ("kernel-at", "x" * 65536),
        ("kernel-over", "x" * 65537),
        ("non-bmp-boundary-at", "\U0001f680" * 16384),
        ("non-bmp-boundary-over", "\U0001f680" * 16385),
        ("non-bmp-kernel-at", "\U0001f680" * 65536),
        ("non-bmp-kernel-over", "\U0001f680" * 65537),
    ):
        raw = body()
        site_node(raw, "message")["reasoning_content"] = value
        add(f"reasoning-{label}", raw)
    for label, values in (
        ("image-only", {"image_tokens": 0}),
        ("text-only", {"text_tokens": 1}),
        ("pair", {"image_tokens": 0, "text_tokens": 1}),
        ("absent", {}),
    ):
        raw = body()
        site_node(raw, "details").update(values)
        add(f"group-{label}", raw)
    for site, first, second in (
        ("message", {"reasoning_content": "a"}, {"reasoning_content": "b"}),
        ("usage", {"num_sources_used": 1}, {"num_sources_used": 2}),
        (
            "details",
            {"image_tokens": 1, "text_tokens": 2},
            {"image_tokens": 2, "text_tokens": 2},
        ),
    ):
        raw = body()
        site_node(raw, site).update(first)
        other = copy.deepcopy(raw)
        site_node(other, site).update(second)
        add(f"values-disagree-{site}", raw, other)
        empty_extensions = copy.deepcopy(raw)
        for key in first:
            del site_node(empty_extensions, site)[key]
        add(f"extensions-raw-only-{site}", raw, empty_extensions)
        add(f"extensions-typed-only-{site}", empty_extensions, raw)
        missing = copy.deepcopy(raw)
        if site == "message":
            missing["choices"][0]["message"] = None
        elif site == "usage":
            missing["usage"] = None
        else:
            missing["usage"]["prompt_tokens_details"] = None
        add(f"site-raw-only-{site}", raw, missing)
        add(f"site-typed-only-{site}", missing, raw)
        add(f"site-neither-{site}", missing)
    raw = body()
    raw["choices"].append(copy.deepcopy(raw["choices"][0]))
    raw["choices"][1]["index"] = 1
    raw["choices"][1]["message"]["reasoning_content"] = "second"
    add("multiple-choices", raw)
    other = copy.deepcopy(raw)
    other["choices"][1]["message"]["reasoning_content"] = "different"
    add("multiple-choices-disagree-second", raw, other)
    shorter = copy.deepcopy(raw)
    shorter["choices"].pop()
    add("choices-raw-longer", raw, shorter)
    add("choices-typed-longer", shorter, raw)
    return cases


@pytest.mark.parametrize("module,limit", [(kernel, 65536), (boundary, 16384)])
@pytest.mark.parametrize("path", ["wire", "agreement", "admissible"])
@pytest.mark.parametrize("length", [16384, 16385, 65536, 65537])
@pytest.mark.parametrize("character", ["x", "\U0001f680"])
def test_literal_track_reasoning_bounds(module, limit, path, length, character):
    raw = body()
    site_node(raw, "message")["reasoning_content"] = character * length
    if length <= limit:
        exercise(module, path, raw, raw)
    else:
        with pytest.raises(AdapterProviderError) as caught:
            exercise(module, path, raw, raw)
        assert caught.value.fault == "provider_response_invalid"


@pytest.mark.parametrize("module", [kernel, boundary])
@pytest.mark.parametrize("path", ["wire", "agreement", "admissible"])
def test_lone_surrogate_reaches_the_parser_and_is_refused(module, path):
    raw = body()
    site_node(raw, "message")["reasoning_content"] = "\ud800"
    extra = parsed(raw).choices[0].message.model_extra
    assert extra is not None
    assert extra["reasoning_content"] == "\ud800"
    with pytest.raises(AdapterProviderError) as caught:
        exercise(module, path, raw, raw)
    assert caught.value.fault == "provider_response_invalid"
