"""Detached plain-JSON projection regressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, SupportsIndex

import pytest

import operatebench.jsonsafe as operatebench_jsonsafe
from boundarybench.freezing import to_json, to_json_preserving_tuples
from boundarybench.jsonsafe import (
    MAX_JSON_DEPTH,
    CyclicStructureError,
    NestingDepthError,
    NonJsonValueError,
    canonical_json_bytes,
    ensure_json_safe,
)


class _HostileStr(str):
    __hash__ = str.__hash__

    def __str__(self) -> str:
        raise AssertionError("hostile __str__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile __eq__ executed")

    def __deepcopy__(self, memo: object) -> str:
        raise AssertionError("hostile __deepcopy__ executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile __reduce__ executed")

    def __reduce_ex__(self, protocol: int) -> object:
        raise AssertionError("hostile __reduce_ex__ executed")


class _HostileInt(int):
    def __int__(self) -> int:
        raise AssertionError("hostile __int__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile __eq__ executed")

    def __deepcopy__(self, memo: object) -> int:
        raise AssertionError("hostile __deepcopy__ executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile __reduce__ executed")

    def __reduce_ex__(self, protocol: int) -> object:
        raise AssertionError("hostile __reduce_ex__ executed")


class _HostileFloat(float):
    def __float__(self) -> float:
        raise AssertionError("hostile __float__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile __eq__ executed")

    def __deepcopy__(self, memo: object) -> float:
        raise AssertionError("hostile __deepcopy__ executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile __reduce__ executed")

    def __reduce_ex__(self, protocol: int) -> object:
        raise AssertionError("hostile __reduce_ex__ executed")


class _HostileMappingKey:
    def __init__(self, hooks: list[str]) -> None:
        self.hooks = hooks

    def _fail(self, hook: str) -> Any:
        self.hooks.append(hook)
        raise AssertionError(f"hostile {hook} executed")

    def __hash__(self) -> int:
        return self._fail("__hash__")

    def __eq__(self, other: object) -> bool:
        return self._fail("__eq__")

    def __repr__(self) -> str:
        return self._fail("__repr__")

    def __str__(self) -> str:
        return self._fail("__str__")

    def __int__(self) -> int:
        return self._fail("__int__")

    def __deepcopy__(self, memo: object) -> object:
        return self._fail("__deepcopy__")

    def __reduce__(self) -> tuple[Any, ...]:
        return self._fail("__reduce__")

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return self._fail("__reduce_ex__")


class _NonStringKeyMapping(Mapping[object, object]):
    def __init__(self, key: object) -> None:
        self.key = key
        self.reads = 0

    def __getitem__(self, key: object) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        if self.reads != 1:
            raise AssertionError("mapping items read more than once")
        yield self.key, "value"


class _DuplicateKeyMapping(Mapping[object, object]):
    def __init__(self, keys: tuple[object, object]) -> None:
        self.item_keys = keys
        self.reads = 0

    def __getitem__(self, key: object) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return 2

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        if self.reads != 1:
            raise AssertionError("mapping items read more than once")
        yield self.item_keys[0], {"source": "first"}
        yield self.item_keys[1], _UnreadDuplicateValue()


class _UnreadDuplicateValue(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("duplicate value traversed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("duplicate value traversed")

    def __len__(self) -> int:
        raise AssertionError("duplicate value traversed")

    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("duplicate value traversed")


class _HostileDuplicateStr(str):
    def __new__(cls, value: str, hooks: list[str]) -> _HostileDuplicateStr:
        return super().__new__(cls, value)

    def __init__(self, value: str, hooks: list[str]) -> None:
        self.hooks = hooks

    def _fail(self, hook: str) -> Any:
        self.hooks.append(hook)
        raise AssertionError(f"hostile {hook} executed")

    def __hash__(self) -> int:
        return self._fail("__hash__")

    def __eq__(self, other: object) -> bool:
        return self._fail("__eq__")

    def __repr__(self) -> str:
        return self._fail("__repr__")

    def __str__(self) -> str:
        return self._fail("__str__")

    def __deepcopy__(self, memo: object) -> str:
        return self._fail("__deepcopy__")

    def __reduce__(self) -> tuple[Any, ...]:
        return self._fail("__reduce__")

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return self._fail("__reduce_ex__")


def _nested_lists(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = [value]
    return value


class _ChangingMapping(Mapping[str, object]):
    def __init__(self, projected: object) -> None:
        self.projected = projected
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        if key != "value":
            raise KeyError(key)
        return "safe on pre-validation"

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield "value"

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        yield "value", "safe on pre-validation" if self.reads == 1 else self.projected


class _ChangingSequence(Sequence[object]):
    def __init__(self, projected: object) -> None:
        self.projected = projected
        self.reads = 0

    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return "safe on pre-validation"

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        self.reads += 1
        yield "safe on pre-validation" if self.reads == 1 else self.projected


class _SelfSequence(Sequence[object]):
    def __getitem__(self, index: int) -> object:
        if index != 0:
            raise IndexError(index)
        return self

    def __len__(self) -> int:
        return 1


def _assert_exact_json_builtins(value: Any) -> None:
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_exact_json_builtins(item)
    elif type(value) is list:
        for item in value:
            _assert_exact_json_builtins(item)
    else:
        assert type(value) in {type(None), bool, str, int, float}


@pytest.mark.parametrize(
    ("value", "expected", "exact_type"),
    (
        (_HostileStr("value"), "value", str),
        (_HostileInt(7), 7, int),
        (_HostileFloat(2.5), 2.5, float),
    ),
    ids=("str-subclass", "int-subclass", "float-subclass"),
)
def test_to_json_normalizes_each_scalar_subclass(
    value: object, expected: object, exact_type: type[object]
) -> None:
    projected = to_json([value])

    assert type(projected[0]) is exact_type
    assert projected[0] == expected


@pytest.mark.parametrize("projection", (to_json, to_json_preserving_tuples))
def test_to_json_normalizes_scalar_and_key_subclasses_without_user_hooks(
    projection: Any,
) -> None:
    source = {
        _HostileStr("outer"): (
            _HostileStr("value"),
            {_HostileStr("numbers"): [_HostileInt(7), _HostileFloat(2.5)]},
        )
    }

    first = projection(source)
    second = projection(source)

    expected: Any = {"outer": ["value", {"numbers": [7, 2.5]}]}
    if projection is to_json_preserving_tuples:
        expected = {"outer": ("value", {"numbers": [7, 2.5]})}
    else:
        _assert_exact_json_builtins(first)
        _assert_exact_json_builtins(second)
    assert first == expected
    assert second == expected
    assert first is not second
    assert first["outer"] is not second["outer"]
    assert first["outer"][1] is not second["outer"][1]


@pytest.mark.parametrize(
    "projection",
    (operatebench_jsonsafe.to_json, operatebench_jsonsafe.to_json_preserving_tuples),
)
@pytest.mark.parametrize("key_kind", ("hostile", "unhashable-list"))
def test_projection_rejects_non_string_mapping_keys_before_hooks_or_insertion(
    projection: Any, key_kind: str
) -> None:
    hooks: list[str] = []
    key: object = _HostileMappingKey(hooks) if key_kind == "hostile" else []
    source = _NonStringKeyMapping(key)

    with pytest.raises(NonJsonValueError) as raised:
        projection(source)

    assert str(raised.value) == (
        "JSON projection: mapping key at $ must be an exact string"
    )
    assert hooks == []
    assert source.reads == 1


@pytest.mark.parametrize(
    "projection",
    (operatebench_jsonsafe.to_json, operatebench_jsonsafe.to_json_preserving_tuples),
)
@pytest.mark.parametrize("key_kind", ("exact", "hostile-subclasses"))
def test_projection_rejects_normalized_duplicate_keys_before_duplicate_value(
    projection: Any, key_kind: str
) -> None:
    hooks: list[str] = []
    keys: tuple[object, object]
    if key_kind == "exact":
        keys = ("same", "same")
    else:
        keys = (
            _HostileDuplicateStr("same", hooks),
            _HostileDuplicateStr("same", hooks),
        )
    source = _DuplicateKeyMapping(keys)

    with pytest.raises(NonJsonValueError) as raised:
        projection(source)

    assert str(raised.value) == "JSON projection: duplicate mapping key 'same' at $"
    assert source.reads == 1
    assert hooks == []


def test_tuple_preserving_projection_keeps_exact_builtin_container_types() -> None:
    source = {
        "tuple": (1, [2, {"nested": (3,)}]),
        "list": [4, (5,)],
    }

    projected = to_json_preserving_tuples(source)

    assert projected == source
    assert type(projected) is dict
    assert type(projected["tuple"]) is tuple
    assert type(projected["tuple"][1]) is list
    assert type(projected["tuple"][1][1]) is dict
    assert type(projected["tuple"][1][1]["nested"]) is tuple
    assert type(projected["list"]) is list
    assert type(projected["list"][1]) is tuple
    assert projected is not source
    assert projected["tuple"] is not source["tuple"]
    assert projected["tuple"][1] is not source["tuple"][1]


def test_tuple_preserving_projection_normalizes_custom_sequences_to_lists() -> None:
    projected = to_json_preserving_tuples(_ChangingSequence((1, 2)))

    assert projected == ["safe on pre-validation"]
    assert type(projected) is list


def test_default_projection_still_normalizes_tuples_to_lists() -> None:
    projected = to_json({"tuple": (1, [2])})

    assert projected == {"tuple": [1, [2]]}
    assert type(projected["tuple"]) is list


def test_to_json_preserves_exact_scalars_and_canonical_bytes() -> None:
    for scalar in (None, False, True, "plain", 17, -2.5):
        assert to_json(scalar) is scalar

    source = {"nested": [None, False, True, "plain", 17, -2.5]}
    projected = to_json(source)
    source_bytes = canonical_json_bytes(source, "source")
    projected_bytes = canonical_json_bytes(projected, "projection")

    assert projected_bytes == source_bytes
    assert sha256(projected_bytes).digest() == sha256(source_bytes).digest()


def test_to_json_projects_shared_acyclic_aliases_independently() -> None:
    shared = {"rows": [1, 2]}

    projected = to_json({"left": shared, "right": shared})

    assert projected == {"left": {"rows": [1, 2]}, "right": {"rows": [1, 2]}}
    assert projected["left"] is not projected["right"]
    assert projected["left"]["rows"] is not projected["right"]["rows"]


@pytest.mark.parametrize("kind", ("mapping", "list", "tuple", "custom-sequence"))
def test_to_json_names_ordinary_cycles(kind: str) -> None:
    if kind == "mapping":
        cyclic: object = {}
        cyclic["self"] = cyclic  # type: ignore[index]
    elif kind == "list":
        cyclic = []
        cyclic.append(cyclic)  # type: ignore[attr-defined]
    elif kind == "tuple":
        bridge: list[object] = []
        cyclic = (bridge,)
        bridge.append(cyclic)
    else:
        cyclic = _SelfSequence()

    with pytest.raises(CyclicStructureError, match=r"JSON projection: cyclic reference"):
        to_json(cyclic)


def test_to_json_names_excessive_depth_before_python_recursion_exhaustion() -> None:
    with pytest.raises(NestingDepthError, match=rf"deeper than {MAX_JSON_DEPTH} levels"):
        to_json(_nested_lists(max(MAX_JSON_DEPTH + 1, 1_500)))


@pytest.mark.parametrize("kind", ("mapping", "sequence"))
def test_to_json_names_cycle_that_appears_during_projection(kind: str) -> None:
    changing: _ChangingMapping | _ChangingSequence = (
        _ChangingMapping(None) if kind == "mapping" else _ChangingSequence(None)
    )
    changing.projected = changing
    ensure_json_safe(changing, "pre-validation")

    with pytest.raises(CyclicStructureError, match=r"JSON projection.*cyclic reference"):
        to_json(changing)

    assert changing.reads == 2


@pytest.mark.parametrize("kind", ("mapping", "sequence"))
def test_to_json_names_depth_that_appears_during_projection(kind: str) -> None:
    projected = _nested_lists(max(MAX_JSON_DEPTH + 1, 1_500))
    changing: _ChangingMapping | _ChangingSequence = (
        _ChangingMapping(projected) if kind == "mapping" else _ChangingSequence(projected)
    )
    ensure_json_safe(changing, "pre-validation")

    with pytest.raises(NestingDepthError, match=rf"deeper than {MAX_JSON_DEPTH} levels"):
        to_json(changing)

    assert changing.reads == 2


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_projection_does_not_make_nonfinite_numbers_json_safe(value: float) -> None:
    with pytest.raises(NonJsonValueError):
        ensure_json_safe(to_json(value), "projection")


@pytest.mark.parametrize("projection", (to_json, to_json_preserving_tuples))
@pytest.mark.parametrize("value", ({1: "value"}, {"value": object()}))
def test_projection_does_not_broaden_the_accepted_json_domain(
    projection: Any, value: object
) -> None:
    with pytest.raises(NonJsonValueError):
        ensure_json_safe(projection(value), "projection")
