"""Mapping-only contract primitives; literal goldens from unmodified 420b984."""

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


def test_literal_contracts_and_digests():
    expected = json.loads(
        (Path(__file__).parent / "fixtures/extension_contract_goldens.json").read_text()
    )

    def plain(value):
        if isinstance(value, Mapping):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    for name, constants in expected.items():
        mod = importlib.import_module(name)
        for key, value in constants.items():
            assert plain(getattr(mod, key)) == value, (name, key)
