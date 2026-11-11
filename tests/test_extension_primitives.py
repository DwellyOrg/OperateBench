"""Mapping-only primitives and independently stated response contract goldens."""

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

MAPPING_MODULES = (
    "operatebench.providers.openai_responses",
    "operatebench.providers.xai_openai_compat",
    "boundarybench.providers.xai_openai_compat",
    "boundarybench.providers.mistral_chat",
)
EXTRA_MODULES = (
    "operatebench.providers.xai_openai_compat",
    "operatebench.providers.xai_responses",
    "operatebench.providers.mistral_chat",
    "boundarybench.providers.mistral_chat",
)


def test_shared_ownership():
    modules = [importlib.import_module(name) for name in MAPPING_MODULES]
    assert len({m._frozen for m in modules}) == 1
    assert len({m._plain for m in modules}) == 1
    assert len({importlib.import_module(n)._typed_extras for n in EXTRA_MODULES}) == 1


@pytest.mark.parametrize("name", MAPPING_MODULES)
def test_mapping_only_semantics(name):
    mod = importlib.import_module(name)
    sequence = [{"snow": "雪", "surrogate": "\ud800"}]
    pair = ({"a": 1},)
    original = {"z": {"é": 1}, "a": sequence, "tuple": pair}
    frozen = mod._frozen(original)
    assert list(frozen) == ["z", "a", "tuple"]
    with pytest.raises(TypeError):
        frozen["z"]["é"] = 2
    original["z"]["é"] = 3
    assert frozen["z"]["é"] == 1
    assert frozen["a"] is sequence
    assert frozen["tuple"] is pair
    plain = mod._plain(frozen)
    assert type(plain) is dict and type(plain["z"]) is dict
    assert plain["a"] is sequence and plain["tuple"] is pair
    assert mod._plain(frozen) is not plain


def test_responses_sequence_policy_stays_separate():
    mod = importlib.import_module("operatebench.providers.xai_responses")
    frozen = mod._frozen({"seq": [{"key": "雪"}]})
    assert isinstance(frozen["seq"], tuple)
    with pytest.raises(TypeError):
        frozen["seq"][0]["key"] = "changed"
    assert mod._plain(frozen) == {"seq": [{"key": "雪"}]}
    assert type(mod._plain(frozen)["seq"]) is list


@pytest.mark.parametrize("name", EXTRA_MODULES)
def test_extras_mapping_identity_and_fresh_empty(name):
    function = importlib.import_module(name)._typed_extras
    extra = {"雪": {"nested": 1}}
    assert function(SimpleNamespace(model_extra=extra)) is extra
    for value in (None, [], (), "text", 1):
        obj = SimpleNamespace(model_extra=value)
        assert function(obj) == {}
        assert function(obj) is not function(obj)
    assert function(object()) == {}


# Keep each entry independent: a mismatch must not hide later providers.
CONTRACT_GOLDENS = json.loads(
    (Path(__file__).parent / "fixtures/extension_contract_goldens.json").read_text()
)


@pytest.mark.parametrize(
    "name,key,value",
    [
        (name, key, value)
        for name, constants in CONTRACT_GOLDENS.items()
        for key, value in constants.items()
    ],
)
def test_literal_contracts_and_digests(name, key, value):

    def plain(value):
        if isinstance(value, Mapping):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    mod = importlib.import_module(name)
    assert plain(getattr(mod, key)) == value, (name, key)


def test_lifecycle_mistral_settings_snapshot():
    from operatebench.agents.mistral_chat import lifecycle_mistral_settings
    from operatebench.providers.mistral_chat import MISTRAL_SMALL_PROFILE

    # Literal request controls, identity and interpretation, independently reviewed
    # against the request builder and profile. Boundary Mistral stays separate.
    assert lifecycle_mistral_settings(
        model="mistral-small-2603", deadline_seconds=30.0, profile=MISTRAL_SMALL_PROFILE
    ) == {
        "provider": "mistral",
        "implementation": "mistral_chat_completions",
        "api": "chat_completions",
        "api_compatibility": "native_mistral",
        "base_url": "https://api.mistral.ai",
        "max_output_tokens": 4096,
        "null_elision": "elide_null_semantic_optionals_v1",
        "parallel_tool_calls": False,
        "protocol_version": "operatebench.model.v4",
        "request_mapping": "lifecycle_mistral_chat_completions_model_request_v3",
        "request_profile": "mistralsmall2603_temperature_zero_v1",
        "request_fields_sent": [
            "max_tokens",
            "messages",
            "model",
            "parallel_tool_calls",
            "temperature",
            "tool_choice",
            "tools",
        ],
        "request_fields_omitted": ["presence_penalty", "random_seed", "top_p"],
        "temperature": 0.0,
        "model_contract_verified": False,
        "response_capture": "mistral_sdk_captured_http_response_v1",
        "response_contract": "closed_exactly_one_tool_call",
        "response_extensions": "mistral_response_extensions_v1",
        "response_extensions_digest": (
            "c7af4de43ae98c475c8b04e250fa127e090b25338afc032125e0bda7ee3ce66d"
        ),
        "response_state_policy": "clear_owned_httpx_response_cookies_v1",
        "retry": {
            "max_attempts": 1,
            "initial_backoff_seconds": 0.0,
            "backoff_multiplier": 1.0,
        },
        "sdk": "mistralai",
        "sdk_injected_fields": {"stream": False},
        "sdk_max_retries": 0,
        "sdk_version": "3.2.0",
        "tool_choice": "required",
        "tool_names": ["act", "wait", "ask", "escalate", "complete", "retrieve"],
        "turn_deadline_seconds": 30.0,
    }
