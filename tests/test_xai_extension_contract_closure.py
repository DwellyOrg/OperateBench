"""Offline I7 closure and fixed per-track parser-policy regressions."""

import inspect

import pytest

from boundarybench.providers import xai_openai_compat as boundary
from operatebench.providers import xai_openai_compat as kernel


@pytest.mark.parametrize("module", [kernel, boundary])
@pytest.mark.parametrize("agree", [False, True])
def test_wrappers_bind_their_own_contract_object(monkeypatch, module, agree):
    name = (
        "check_extensions_agree_under_contract"
        if agree
        else "check_extensions_under_contract"
    )
    assert hasattr(kernel, name), "shared contract-aware closure is required"
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, name, spy)
    node, parsed = {}, object()
    if agree:
        module.check_response_extensions_agree(node, parsed)
        assert calls == [
            ((node, parsed), {"contract": module.RESPONSE_EXTENSION_CONTRACT})
        ]
    else:
        module.check_response_extensions(node, site="completion_message")
        assert calls == [
            (
                (node,),
                {
                    "site": "completion_message",
                    "contract": module.RESPONSE_EXTENSION_CONTRACT,
                },
            )
        ]
    assert calls[0][1]["contract"] is module.RESPONSE_EXTENSION_CONTRACT


@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("check_extensions_under_contract", ({},), {"site": "completion_message"}),
        ("check_extensions_agree_under_contract", ({}, object()), {}),
    ],
)
def test_shared_entrypoints_require_keyword_only_contract(name, args, kwargs):
    assert hasattr(kernel, name), "shared contract-aware closure is required"
    function = getattr(kernel, name)
    parameter = inspect.signature(function).parameters["contract"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="contract"):
        function(*args, **kwargs)


# Literal db7ecc5 snapshots, captured before extracting the implementation.
KERNEL_CONTRACT = {
    "contract": "xai_compat_response_extensions_v1",
    "extension_values_read": False,
    "max_count": 9007199254740991,
    "max_reasoning_characters": 65536,
    "members": {
        "completion_message": {"reasoning_content": "bounded_reasoning_text"},
        "prompt_token_details": {
            "image_tokens": "non_negative_bounded_integer",
            "text_tokens": "non_negative_bounded_integer",
        },
        "usage_block": {
            "cost_in_usd_ticks": "non_negative_bounded_integer",
            "num_sources_used": "non_negative_bounded_integer",
        },
    },
    "members_required_together": {
        "completion_message": False,
        "prompt_token_details": True,
        "usage_block": False,
    },
}
BOUNDARY_CONTRACT = {**KERNEL_CONTRACT, "max_reasoning_characters": 16384}
KERNEL_JSON = (
    b'{"contract":"xai_compat_response_extensions_v1","extension_values_read'
    b'":false,"max_count":9007199254740991,"max_reasoning_characters":65536,'
    b'"members":{"completion_message":{"reasoning_content":"bounded_reasonin'
    b'g_text"},"prompt_token_details":{"image_tokens":"non_negative_bounded_'
    b'integer","text_tokens":"non_negative_bounded_integer"},"usage_block":{'
    b'"cost_in_usd_ticks":"non_negative_bounded_integer","num_sources_used":'
    b'"non_negative_bounded_integer"}},"members_required_together":{"complet'
    b'ion_message":false,"prompt_token_details":true,"usage_block":false}}'
)
BOUNDARY_JSON = (
    b'{"contract":"xai_compat_response_extensions_v1","extension_values_read'
    b'":false,"max_count":9007199254740991,"max_reasoning_characters":16384,'
    b'"members":{"completion_message":{"reasoning_content":"bounded_reasonin'
    b'g_text"},"prompt_token_details":{"image_tokens":"non_negative_bounded_'
    b'integer","text_tokens":"non_negative_bounded_integer"},"usage_block":{'
    b'"cost_in_usd_ticks":"non_negative_bounded_integer","num_sources_used":'
    b'"non_negative_bounded_integer"}},"members_required_together":{"complet'
    b'ion_message":false,"prompt_token_details":true,"usage_block":false}}'
)


@pytest.mark.parametrize(
    "module,expected,encoded,digest",
    [
        (
            kernel,
            KERNEL_CONTRACT,
            KERNEL_JSON,
            "58027d0272a1017dd5a0c456ebd864dec1a5f732a450a0cf86e428535c10dffb",
        ),
        (
            boundary,
            BOUNDARY_CONTRACT,
            BOUNDARY_JSON,
            "9249fb6a8362e7b7ae9ea85e7fd3262a83b64bf55011e3b8eb48fc9e6dd2b152",
        ),
    ],
)
def test_frozen_contract_json_and_digest(module, expected, encoded, digest):
    actual = module._plain(module.RESPONSE_EXTENSION_CONTRACT)
    assert actual == expected
    assert module.canonical_json_bytes(actual, "snapshot") == encoded
    assert digest == module.RESPONSE_EXTENSION_DIGEST
