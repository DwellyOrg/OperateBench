# SPDX-FileCopyrightText: 2026 OperateBench contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from abc import ABCMeta
from collections.abc import Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, SupportsIndex

import pytest

import operatebench.core.errors as errors_module
import operatebench.core.ledger as ledger_module
from boundarybench.jsonsafe import (
    MAX_JSON_DEPTH,
    CyclicStructureError,
    NestingDepthError,
    NonJsonValueError,
    canonical_json_bytes,
)
from operatebench.core.errors import (
    LedgerFieldError,
    LedgerPayloadError,
    OperateBenchError,
)
from operatebench.core.ledger import TrajectoryLedger

_AT = "2031-03-03T09:00:00Z"


def test_append_admission_canonicalizes_only_each_new_singleton_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted: list[tuple[object, str]] = []
    canonicalize = ledger_module.canonical_json_bytes

    def observe_admission(value: object, context: str) -> bytes:
        admitted.append((value, context))
        return canonicalize(value, context)

    monkeypatch.setattr(ledger_module, "canonical_json_bytes", observe_admission)
    ledger = TrajectoryLedger()

    for index in range(8):
        ledger.append(_AT, "event_observed", {"value": index})

    assert len(admitted) == 8
    for index, (value, context) in enumerate(admitted):
        assert type(value) is list
        assert len(value) == 1
        row = value[0]
        assert type(row) is dict
        assert row["index"] == index
        assert row["value"] == index
        assert context == "operatebench trajectory"


def _nested_lists(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = [value]
    return value


def test_append_rejects_row_that_only_exceeds_depth_in_trajectory_atomically() -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()

    with pytest.raises(NestingDepthError):
        ledger.append(
            _AT,
            "event_observed",
            {"nested": _nested_lists(MAX_JSON_DEPTH - 1)},
        )

    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_append_accepts_row_at_exact_trajectory_depth_limit() -> None:
    ledger = TrajectoryLedger()

    appended = ledger.append(
        _AT,
        "event_observed",
        {"nested": _nested_lists(MAX_JSON_DEPTH - 2)},
    )

    assert appended["index"] == 0
    assert ledger.as_list() == [appended]
    assert len(ledger.digest()) == 64


def test_retained_source_nested_alias_cannot_mutate_stored_row_or_digest() -> None:
    ledger = TrajectoryLedger()
    nested = {"steps": [{"status": "original"}]}
    ledger.append(_AT, "event_observed", {"nested": nested})
    observed_digest = ledger.digest()

    nested["steps"][0]["status"] = "mutated"
    nested["steps"].append({"status": "injected"})

    assert ledger.as_list()[0]["nested"] == {"steps": [{"status": "original"}]}
    assert ledger.digest() == observed_digest


def test_nested_alias_in_append_return_cannot_mutate_storage() -> None:
    ledger = TrajectoryLedger()
    appended = ledger.append(_AT, "event_observed", {"nested": {"steps": ["original"]}})

    appended["nested"]["steps"].append("injected")

    assert ledger.as_list()[0]["nested"] == {"steps": ["original"]}


def test_nested_alias_in_as_list_cannot_mutate_storage() -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"nested": {"steps": ["original"]}})
    projection = ledger.as_list()

    projection[0]["nested"]["steps"].append("injected")

    assert ledger.as_list()[0]["nested"] == {"steps": ["original"]}


def test_repeated_public_projections_are_fresh_at_every_container() -> None:
    ledger = TrajectoryLedger()
    first_append = ledger.append(
        _AT, "event_observed", {"nested": {"steps": ["original"]}}
    )
    first = ledger.as_list()
    second = ledger.as_list()

    assert first is not second
    assert first[0] is not second[0]
    assert first[0]["nested"] is not second[0]["nested"]
    assert first[0]["nested"]["steps"] is not second[0]["nested"]["steps"]
    assert first_append["nested"] is not first[0]["nested"]


def test_shared_acyclic_aliases_are_legal_and_independently_detached() -> None:
    ledger = TrajectoryLedger()
    shared = {"steps": ["original"]}

    appended = ledger.append(_AT, "event_observed", {"left": shared, "right": shared})
    projected = ledger.as_list()[0]

    assert appended["left"] == appended["right"] == shared
    assert appended["left"] is not appended["right"]
    assert appended["left"]["steps"] is not appended["right"]["steps"]
    assert projected["left"] is not projected["right"]
    assert projected["left"]["steps"] is not projected["right"]["steps"]


class _HostileUnsupported:
    def __deepcopy__(self, memo: object) -> object:
        raise AssertionError("__deepcopy__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("__eq__ executed")

    def __reduce__(self) -> tuple[Any, ...]:
        raise AssertionError("__reduce__ executed")

    def __repr__(self) -> str:
        raise AssertionError("__repr__ executed")


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


class _StatefulClassMappingKey(_HostileMappingKey):
    def __init__(self, hooks: list[str], *, spoof_str: bool) -> None:
        super().__init__(hooks)
        self.spoof_str = spoof_str
        self.class_reads = 0

    @property
    def __class__(self) -> type[object]:
        self.class_reads += 1
        self.hooks.append("__class__")
        if self.spoof_str:
            return str
        raise AssertionError("hostile __class__ executed")

    def __copy__(self) -> object:
        return self._fail("__copy__")


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
        yield self.item_keys[1], _UnreadMapping()


class _HostileStrMeta(type):
    def __getattribute__(cls, name: str) -> Any:
        if name == "__mro__":
            hooks = type.__getattribute__(cls, "_metaclass_hooks")
            hooks.append("metaclass __mro__")
            raise AssertionError("hostile metaclass __getattribute__ executed")
        return type.__getattribute__(cls, name)


class _HostileDuplicateStr(str, metaclass=_HostileStrMeta):
    _metaclass_hooks: ClassVar[list[str]] = []

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


class _OneReadMapping(Mapping[str, object]):
    def __init__(self, value: object) -> None:
        self.value = value
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        if self.reads != 1:
            yield "value", _HostileUnsupported()
        else:
            yield "value", self.value


class _OneReadSequence(Sequence[object]):
    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, index: int) -> object:  # type: ignore[override]
        raise AssertionError("sequence __getitem__ executed")

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        self.reads += 1
        if self.reads != 1:
            yield _HostileUnsupported()
        else:
            yield (1, 2)


class _ReentrantMapping(Mapping[str, object]):
    def __init__(
        self, ledger: TrajectoryLedger, *, catch_refusal: bool = False, fail: bool = False
    ) -> None:
        self.ledger = ledger
        self.catch_refusal = catch_refusal
        self.fail = fail

    def __getitem__(self, key: str) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        if self.catch_refusal:
            with pytest.raises(
                RuntimeError, match="trajectory append already in progress"
            ):
                self.ledger.append("inner", "event_observed", _UnreadMapping())
        else:
            self.ledger.append("inner", "event_observed", _UnreadMapping())
        yield "value", _HostileUnsupported() if self.fail else "safe"


class _ReentrantSequence(Sequence[object]):
    def __init__(self, ledger: TrajectoryLedger) -> None:
        self.ledger = ledger

    def __getitem__(self, index: int) -> object:  # type: ignore[override]
        raise AssertionError("sequence __getitem__ executed")

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        self.ledger.append("inner", "event_observed", _UnreadMapping())
        yield "safe"


class _UnreadMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("nested append traversed its payload")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("nested append traversed its payload")

    def __len__(self) -> int:
        raise AssertionError("nested append traversed its payload")

    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("nested append traversed its payload")


class _HostileReservedStr(str):
    def __new__(cls, value: str, hooks: list[str]) -> _HostileReservedStr:
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

    def __copy__(self) -> str:
        return self._fail("__copy__")

    def __deepcopy__(self, memo: object) -> str:
        return self._fail("__deepcopy__")

    def __reduce__(self) -> tuple[Any, ...]:
        return self._fail("__reduce__")

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return self._fail("__reduce_ex__")


class _HostileReservedStrA(_HostileReservedStr):
    pass


class _HostileReservedStrB(_HostileReservedStr):
    pass


class _HostileReservedStrC(_HostileReservedStr):
    pass


class _ReservedKeyMapping(Mapping[object, object]):
    def __init__(self, item_keys: tuple[object, ...]) -> None:
        self.item_keys = item_keys
        self.reads = 0

    def __getitem__(self, key: object) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return len(self.item_keys)

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        if self.reads != 1:
            raise AssertionError("mapping items read more than once")
        for key in self.item_keys:
            yield key, _UnreadMapping()


class _HostileNonMapping:
    def __init__(self) -> None:
        self.hooks: list[str] = []

    def _fail(self, hook: str) -> Any:
        self.hooks.append(hook)
        raise AssertionError(f"hostile {hook} executed")

    def __repr__(self) -> str:
        return self._fail("__repr__")

    def __str__(self) -> str:
        return self._fail("__str__")

    def __hash__(self) -> int:
        return self._fail("__hash__")

    def __eq__(self, other: object) -> bool:
        return self._fail("__eq__")

    def __copy__(self) -> object:
        return self._fail("__copy__")

    def __deepcopy__(self, memo: object) -> object:
        return self._fail("__deepcopy__")

    def __reduce__(self) -> tuple[Any, ...]:
        return self._fail("__reduce__")

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return self._fail("__reduce_ex__")


class _HostilePairList(_HostileNonMapping, list[tuple[str, object]]):
    def __init__(self) -> None:
        _HostileNonMapping.__init__(self)
        list.__init__(
            self,
            [("index", 999), ("at", "forged"), ("record_type", "forged")],
        )

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self._fail("__iter__")


class _HostilePairTuple(_HostileNonMapping, tuple[tuple[str, object], ...]):
    def __new__(cls) -> _HostilePairTuple:
        return tuple.__new__(
            cls,
            (("index", 999), ("at", "forged"), ("record_type", "forged")),
        )

    def __init__(self) -> None:
        _HostileNonMapping.__init__(self)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self._fail("__iter__")


class _HostileDuckMapping(_HostileNonMapping):
    def keys(self) -> object:
        return self._fail("keys")

    def __getitem__(self, key: object) -> object:
        return self._fail("__getitem__")


class _StatefulClassDuckPayload:
    def __init__(self) -> None:
        self.hooks: list[str] = []
        self.class_reads = 0

    @property
    def __class__(self) -> type[object]:
        self.class_reads += 1
        self.hooks.append("__class__")
        return dict if self.class_reads == 1 else type(self)

    def keys(self) -> tuple[str, str, str]:
        self.hooks.append("keys")
        return ("index", "at", "record_type")

    def __getitem__(self, key: str) -> object:
        self.hooks.append("__getitem__")
        return {
            "index": -1,
            "at": "1999-01-01T00:00:00Z",
            "record_type": "effect_accepted",
        }[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile __iter__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __hash__(self) -> int:
        raise AssertionError("hostile __hash__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile __eq__ executed")

    def __copy__(self) -> object:
        raise AssertionError("hostile __copy__ executed")

    def __reduce__(self) -> tuple[Any, ...]:
        raise AssertionError("hostile __reduce__ executed")


class _VirtualInt(int):
    pass


class _VirtualStr(str):
    pass


class _VirtualList(list[object]):
    pass


class _VirtualDuck:
    def __init__(self, hooks: list[str]) -> None:
        self._hooks = hooks

    def _fail(self, hook: str) -> Any:
        self._hooks.append(hook)
        raise AssertionError(f"hostile {hook} executed")

    def __getitem__(self, key: object) -> object:
        return self._fail("__getitem__")

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self._fail("__iter__")

    def __len__(self) -> int:
        return self._fail("__len__")

    def items(self):  # type: ignore[no-untyped-def]
        return self._fail("items")


for _virtual_mapping_type in (_VirtualInt, _VirtualStr, _VirtualList, _VirtualDuck):
    Mapping.register(_virtual_mapping_type)


class _DictSubclass(dict[str, object]):
    pass


class _ConcreteMutableMapping(MutableMapping[str, object]):
    def __init__(self) -> None:
        self._value: dict[str, object] = {"concrete": True}
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        return self._value[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._value[key] = value

    def __delitem__(self, key: str) -> None:
        del self._value[key]

    def __iter__(self):
        self.reads += 1
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


class _HostileProjectedMapping:
    def __init__(self) -> None:
        self.hooks: list[str] = []

    def keys(self) -> object:
        self.hooks.append("keys")
        raise AssertionError("hostile keys executed")

    def __getitem__(self, key: object) -> object:
        self.hooks.append("__getitem__")
        raise AssertionError("hostile __getitem__ executed")


_METACLASS_HOOKS: list[str] = []
_METACLASS_ARMED = False


class _HostilePayloadMeta(type):
    def _fail(cls, hook: str) -> None:
        if _METACLASS_ARMED:
            _METACLASS_HOOKS.append(hook)
            raise AssertionError(f"hostile metaclass {hook} executed")

    def __getattribute__(cls, name: str) -> Any:
        if _METACLASS_ARMED:
            _METACLASS_HOOKS.append("__getattribute__")
            raise AssertionError("hostile metaclass __getattribute__ executed")
        return type.__getattribute__(cls, name)

    def __hash__(cls) -> int:
        cls._fail("__hash__")
        return type.__hash__(cls)

    def __eq__(cls, other: object) -> bool:
        cls._fail("__eq__")
        return type.__eq__(cls, other)

    def __subclasscheck__(cls, subclass: type[object]) -> bool:
        cls._fail("__subclasscheck__")
        return type.__subclasscheck__(cls, subclass)

    def __instancecheck__(cls, instance: object) -> bool:
        cls._fail("__instancecheck__")
        return type.__instancecheck__(cls, instance)


class _HostileMetaclassPayload(metaclass=_HostilePayloadMeta):
    pass


_CONCRETE_MAPPING_META_HOOKS: list[str] = []
_CONCRETE_MAPPING_META_ARMED = False


def _record_concrete_mapping_meta_hook(hook: str) -> None:
    if _CONCRETE_MAPPING_META_ARMED:
        _CONCRETE_MAPPING_META_HOOKS.append(hook)
        raise AssertionError(f"hostile concrete mapping metaclass {hook} executed")


class _HostileConcreteMappingMeta(ABCMeta):
    def __getattribute__(cls, name: str) -> Any:
        _record_concrete_mapping_meta_hook("__getattribute__")
        return type.__getattribute__(cls, name)

    def __hash__(cls) -> int:
        _record_concrete_mapping_meta_hook("__hash__")
        return type.__hash__(cls)

    def __eq__(cls, other: object) -> bool:
        _record_concrete_mapping_meta_hook("__eq__")
        return type.__eq__(cls, other)

    def __subclasscheck__(cls, subclass: type[object]) -> bool:
        _record_concrete_mapping_meta_hook("__subclasscheck__")
        return ABCMeta.__subclasscheck__(cls, subclass)

    def __instancecheck__(cls, instance: object) -> bool:
        _record_concrete_mapping_meta_hook("__instancecheck__")
        return ABCMeta.__instancecheck__(cls, instance)


class _HostileConcreteMapping(
    Mapping[str, object], metaclass=_HostileConcreteMappingMeta
):
    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        raise AssertionError("mapping __getitem__ executed")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping __iter__ executed")

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        self.reads += 1
        if self.reads != 1:
            raise AssertionError("mapping items read more than once")
        yield "safe", {"nested": True}


class _MroPropertyMeta(type):
    mode: str = "spoof"
    hooks: ClassVar[list[str]] = []

    @property
    def __mro__(cls) -> object:
        cls.hooks.append("mro")
        if cls.mode == "raise":
            raise ValueError("attacker-controlled mro failure")
        if cls.mode == "int":
            return 7
        return (dict, object)


class _MroPropertyPayload(metaclass=_MroPropertyMeta):
    def items(self):  # type: ignore[no-untyped-def]
        type(self).hooks.append("items")
        yield "smuggled", "yes"


class _MroPropertyPayloadWithoutItems(metaclass=_MroPropertyMeta):
    pass


class _MroKeyPropertyMeta(type):
    mode: str = "spoof"
    hooks: ClassVar[list[str]] = []

    @property
    def __mro__(cls) -> object:
        cls.hooks.append("mro")
        if cls.mode == "raise":
            raise ValueError("attacker-controlled key mro failure")
        return (str, object)


class _MroPropertyKey(metaclass=_MroKeyPropertyMeta):
    pass


class _ProxyDuck:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.hooks: list[str] = []

    def keys(self):  # type: ignore[no-untyped-def]
        self.hooks.append("keys")
        return ("smuggled",)

    def __getitem__(self, key: object) -> object:
        self.hooks.append("__getitem__")
        if self.mode == "getitem-raise":
            raise ValueError("attacker-controlled proxy value failure")
        return "yes"

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.hooks.append("__iter__")
        raise AssertionError("proxy duck __iter__ executed")

    def items(self):  # type: ignore[no-untyped-def]
        self.hooks.append("items")
        if self.mode == "items-raise":
            raise ValueError("attacker-controlled proxy items failure")
        if self.mode == "malformed-items":
            return (("malformed",),)
        return (("smuggled", "yes"),)


class _ProxyDuckWithoutItems:
    def __init__(self) -> None:
        self.hooks: list[str] = []

    def keys(self):  # type: ignore[no-untyped-def]
        self.hooks.append("keys")
        return ("smuggled",)

    def __getitem__(self, key: object) -> object:
        self.hooks.append("__getitem__")
        return "yes"

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.hooks.append("__iter__")
        raise AssertionError("proxy duck __iter__ executed")


@pytest.mark.parametrize(
    "payload_factory", (_HostilePairList, _HostilePairTuple, _HostileDuckMapping)
)
def test_append_rejects_non_mapping_payload_before_caller_access_atomically(
    payload_factory: type[_HostileNonMapping],
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    payload = payload_factory()

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert "LedgerPayloadError" in errors_module.__all__
    assert issubclass(LedgerFieldError, LedgerPayloadError)
    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert payload.hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_append_rejects_stateful_class_duck_before_any_caller_hook_atomically() -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    payload = _StatefulClassDuckPayload()

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert payload.class_reads == 0
    assert payload.hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_append_shape_admission_bypasses_hostile_metaclass_hooks() -> None:
    global _METACLASS_ARMED

    ledger = TrajectoryLedger()
    payload = _HostileMetaclassPayload()
    _METACLASS_HOOKS.clear()
    _METACLASS_ARMED = True
    try:
        with pytest.raises(LedgerPayloadError) as raised:
            ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]
    finally:
        _METACLASS_ARMED = False

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert _METACLASS_HOOKS == []
    assert len(ledger) == 0


@pytest.mark.parametrize(
    ("payload_type", "mode"),
    (
        (_MroPropertyPayload, "spoof"),
        (_MroPropertyPayload, "int"),
        (_MroPropertyPayload, "raise"),
        (_MroPropertyPayloadWithoutItems, "spoof"),
    ),
    ids=("spoof-mapping", "non-mro", "raising-property", "missing-items"),
)
def test_append_rejects_metaclass_mro_property_without_hooks_atomically(
    payload_type: type[object], mode: str
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    _MroPropertyMeta.mode = mode
    _MroPropertyMeta.hooks.clear()

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", payload_type())  # type: ignore[arg-type]

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert _MroPropertyMeta.hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_append_projects_concrete_mapping_without_second_root_abc_check() -> None:
    global _CONCRETE_MAPPING_META_ARMED

    ledger = TrajectoryLedger()
    payload = _HostileConcreteMapping()
    _CONCRETE_MAPPING_META_HOOKS.clear()
    _CONCRETE_MAPPING_META_ARMED = True
    try:
        appended = ledger.append(_AT, "event_observed", payload)
    finally:
        _CONCRETE_MAPPING_META_ARMED = False

    assert _CONCRETE_MAPPING_META_HOOKS == []
    assert payload.reads == 1
    assert appended == {
        "index": 0,
        "at": _AT,
        "record_type": "event_observed",
        "safe": {"nested": True},
    }
    assert ledger.as_list() == [appended]


@pytest.mark.parametrize("mode", ("spoof", "raise"))
def test_mapping_key_metaclass_mro_property_is_canonical_refusal_without_hooks(
    mode: str,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    _MroKeyPropertyMeta.mode = mode
    _MroKeyPropertyMeta.hooks.clear()
    payload = _NonStringKeyMapping(_MroPropertyKey())

    with pytest.raises(NonJsonValueError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert type(raised.value) is NonJsonValueError
    assert raised.value.args == (
        "JSON projection: mapping key at $ must be an exact string",
    )
    assert _MroKeyPropertyMeta.hooks == []
    assert payload.reads == 1
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize(
    "payload_factory",
    (
        lambda hooks: _VirtualInt(7),
        lambda hooks: _VirtualStr("not-a-mapping"),
        lambda hooks: _VirtualList(),
        _VirtualDuck,
    ),
)
def test_append_rejects_virtual_mapping_before_hooks_or_interpreter_errors(
    payload_factory: Any,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    payload = payload_factory(hooks)

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", payload)

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"exact": True}, {"exact": True}),
        (_DictSubclass(subclass=True), {"subclass": True}),
        (_ConcreteMutableMapping(), {"concrete": True}),
    ),
)
def test_append_accepts_concrete_mapping_shapes_once(
    payload: Mapping[str, object], expected: dict[str, object]
) -> None:
    ledger = TrajectoryLedger()

    appended = ledger.append(_AT, "event_observed", payload)

    assert {key: appended[key] for key in expected} == expected
    if isinstance(payload, _ConcreteMutableMapping):
        assert payload.reads == 1


def test_append_accepts_exact_mapping_proxy_once_with_detached_wire_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = TrajectoryLedger()
    nested = {"steps": [{"status": "original"}]}
    source = {"nested": nested}
    payload = MappingProxyType(source)
    project = ledger_module._to_json_preserving_tuples_with_root_key_admission
    projected: list[object] = []

    def observe_projection(value: object, admission: Any) -> object:
        projected.append(value)
        return project(value, admission)

    monkeypatch.setattr(
        ledger_module,
        "_to_json_preserving_tuples_with_root_key_admission",
        observe_projection,
    )

    appended = ledger.append(_AT, "event_observed", payload)
    expected_wire = (
        b'[{"at":"2031-03-03T09:00:00Z","index":0,'
        b'"nested":{"steps":[{"status":"original"}]},'
        b'"record_type":"event_observed"}]'
    )
    observed_digest = ledger.digest()

    assert len(projected) == 1
    assert projected[0] is payload
    assert type(appended) is dict
    assert appended["index"] == 0
    assert appended["at"] == _AT
    assert appended["record_type"] == "event_observed"
    assert (
        canonical_json_bytes(ledger.as_list(), "operatebench trajectory") == expected_wire
    )
    assert observed_digest == hashlib.sha256(expected_wire).hexdigest()

    nested["steps"][0]["status"] = "source-mutated"
    appended["nested"]["steps"].append({"status": "return-mutated"})

    assert (
        canonical_json_bytes(ledger.as_list(), "operatebench trajectory") == expected_wire
    )
    assert ledger.digest() == observed_digest


def test_append_accepts_mapping_proxy_over_concrete_mapping_once() -> None:
    ledger = TrajectoryLedger()
    source = _HostileConcreteMapping()
    payload = MappingProxyType(source)

    appended = ledger.append(_AT, "event_observed", payload)

    assert source.reads == 1
    assert appended == {
        "index": 0,
        "at": _AT,
        "record_type": "event_observed",
        "safe": {"nested": True},
    }
    assert ledger.as_list() == [appended]


@pytest.mark.parametrize(
    "source",
    (
        _ProxyDuck("safe-items"),
        _ProxyDuckWithoutItems(),
        _ProxyDuck("malformed-items"),
        _ProxyDuck("items-raise"),
    ),
    ids=("safe-items", "missing-items", "malformed-items", "raising-items"),
)
def test_append_rejects_mapping_proxy_over_duck_before_referent_hooks_atomically(
    source: object,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    payload = MappingProxyType(source)  # type: ignore[arg-type]

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", payload)

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert source.hooks == []  # type: ignore[attr-defined]
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_mapping_proxy_reserved_root_refuses_before_child_traversal_atomically() -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    payload = MappingProxyType({"index": _UnreadMapping(), "child": _UnreadMapping()})

    with pytest.raises(LedgerFieldError) as raised:
        ledger.append(_AT, "event_observed", payload)

    assert type(raised.value) is LedgerFieldError
    assert raised.value.args == ("trajectory payload contains a ledger-owned field",)
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1


def test_append_rejects_non_dict_projection_before_update_hooks_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    projected = _HostileProjectedMapping()
    monkeypatch.setattr(
        ledger_module,
        "_to_json_preserving_tuples_with_root_key_admission",
        lambda payload, admission: projected,
    )

    with pytest.raises(LedgerPayloadError) as raised:
        ledger.append(_AT, "event_observed", {"safe": True})

    assert type(raised.value) is LedgerPayloadError
    assert raised.value.args == ("trajectory payload must be a mapping",)
    assert projected.hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1


class _OrderedReservedKeyMapping(Mapping[object, object]):
    def __init__(self, reserved_key: object) -> None:
        self.reserved_key = reserved_key
        self.reads = 0
        self.yields = 0

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
        self.yields += 1
        yield "legitimate", {"value": "projected-first"}
        self.yields += 1
        yield self.reserved_key, _UnreadMapping()


def test_ordered_legitimate_prefix_does_not_leak_before_reserved_key_refusal() -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    payload = _OrderedReservedKeyMapping(_HostileReservedStr("index", hooks))

    with pytest.raises(LedgerFieldError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert type(raised.value) is LedgerFieldError
    assert raised.value.args == ("trajectory payload contains a ledger-owned field",)
    assert payload.reads == 1
    assert payload.yields == 2
    assert hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize(
    "reserved_keys",
    [("index",), ("at",), ("record_type",), ("index", "at", "record_type")],
)
@pytest.mark.parametrize("key_kind", ("exact", "hostile-subclasses"))
def test_append_refuses_ledger_owned_payload_fields_before_values_atomically(
    reserved_keys: tuple[str, ...], key_kind: str
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    keys: tuple[object, ...]
    if key_kind == "exact":
        keys = reserved_keys
    else:
        classes = (_HostileReservedStrA, _HostileReservedStrB, _HostileReservedStrC)
        keys = tuple(
            classes[index](key, hooks) for index, key in enumerate(reserved_keys)
        )
    payload = _ReservedKeyMapping(keys)

    with pytest.raises(LedgerFieldError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert type(raised.value) is LedgerFieldError
    assert isinstance(raised.value, OperateBenchError)
    assert raised.value.args == ("trajectory payload contains a ledger-owned field",)
    assert hooks == []
    assert payload.reads == 1
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


def test_ledger_owned_field_names_remain_legal_below_payload_root() -> None:
    ledger = TrajectoryLedger()
    nested = {"index": 99, "at": "caller-data", "record_type": "caller-data"}

    appended = ledger.append(_AT, "event_observed", {"nested": nested})

    assert appended["nested"] == nested
    assert appended["index"] == 0
    assert appended["at"] == _AT
    assert appended["record_type"] == "event_observed"


@pytest.mark.parametrize(
    "payload",
    [
        lambda ledger: _ReentrantMapping(ledger),
        lambda ledger: _ReentrantMapping(ledger, fail=True),
        lambda ledger: {"sequence": _ReentrantSequence(ledger)},
    ],
)
def test_reentrant_append_is_refused_atomically_before_nested_argument_traversal(
    payload: Any,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()

    with pytest.raises(RuntimeError, match="trajectory append already in progress"):
        ledger.append(_AT, "event_observed", payload(ledger))

    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert [row["index"] for row in ledger.as_list()] == [0]
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1
    assert [row["index"] for row in ledger.as_list()] == [0, 1]


def test_outer_append_may_continue_after_catching_reentrant_refusal() -> None:
    ledger = TrajectoryLedger()

    appended = ledger.append(
        _AT, "event_observed", _ReentrantMapping(ledger, catch_refusal=True)
    )

    assert appended["value"] == "safe"
    assert ledger.as_list() == [appended]
    assert appended["index"] == 0


def test_stateful_custom_containers_are_read_once_before_atomic_admission() -> None:
    ledger = TrajectoryLedger()
    sequence = _OneReadSequence()
    payload = _OneReadMapping(sequence)

    appended = ledger.append(_AT, "event_observed", payload)

    assert payload.reads == 1
    assert sequence.reads == 1
    assert appended["value"] == [(1, 2)]
    assert type(appended["value"]) is list
    assert type(appended["value"][0]) is tuple
    assert ledger.as_list()[0]["value"] == [(1, 2)]


@pytest.mark.parametrize("key_kind", ("hostile", "unhashable-list"))
def test_append_rejects_non_string_mapping_keys_without_hooks_and_recovers(
    key_kind: str,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    key: object = _HostileMappingKey(hooks) if key_kind == "hostile" else []
    payload = _NonStringKeyMapping(key)

    with pytest.raises(NonJsonValueError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert str(raised.value) == (
        "JSON projection: mapping key at $ must be an exact string"
    )
    assert hooks == []
    assert payload.reads == 1
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize("spoof_str", (False, True), ids=("raises", "spoofs-str"))
def test_append_rejects_stateful_class_spoof_mapping_key_without_hooks(
    spoof_str: bool,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    key = _StatefulClassMappingKey(hooks, spoof_str=spoof_str)
    payload = _NonStringKeyMapping(key)

    with pytest.raises(NonJsonValueError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert str(raised.value) == (
        "JSON projection: mapping key at $ must be an exact string"
    )
    assert hooks == []
    assert key.class_reads == 0
    assert payload.reads == 1
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize("key_kind", ("exact", "hostile-subclasses"))
def test_append_rejects_normalized_duplicate_keys_atomically_and_recovers(
    key_kind: str,
) -> None:
    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()
    hooks: list[str] = []
    keys: tuple[object, object]
    if key_kind == "exact":
        keys = ("same", "same")
    else:
        type.__setattr__(_HostileDuplicateStr, "_metaclass_hooks", hooks)
        keys = (
            _HostileDuplicateStr("same", hooks),
            _HostileDuplicateStr("same", hooks),
        )
    payload = _DuplicateKeyMapping(keys)

    with pytest.raises(NonJsonValueError) as raised:
        ledger.append(_AT, "event_observed", payload)  # type: ignore[arg-type]

    assert str(raised.value) == "JSON projection: duplicate mapping key 'same' at $"
    assert payload.reads == 1
    assert hooks == []
    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1
    appended = ledger.append(_AT, "event_observed", {"after": True})
    assert appended["index"] == 1


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"nested": {"value": _HostileUnsupported()}}, NonJsonValueError),
        ({"nested": None}, CyclicStructureError),
        ({"nested": None}, NestingDepthError),
    ],
)
def test_rejected_nested_payload_is_atomic_and_uses_json_safety_errors(
    payload: dict[str, Any], error_type: type[ValueError]
) -> None:
    if error_type is CyclicStructureError:
        cycle: list[Any] = []
        cycle.append(cycle)
        payload["nested"] = cycle
    elif error_type is NestingDepthError:
        nested: list[Any] = []
        for _ in range(MAX_JSON_DEPTH + 1):
            nested = [nested]
        payload["nested"] = nested

    ledger = TrajectoryLedger()
    ledger.append(_AT, "event_observed", {"stable": True})
    before = ledger.as_list()
    digest = ledger.digest()

    with pytest.raises(error_type):
        ledger.append(_AT, "event_observed", payload)

    assert ledger.as_list() == before
    assert ledger.digest() == digest
    assert len(ledger) == 1


def test_existing_tuple_payload_api_remains_accepted() -> None:
    ledger = TrajectoryLedger()

    payload = {
        "tuple": (1, [2, {"three": (3, 4)}]),
        "list": [5, (6, [7])],
        "dict": {"tuple": (8,), "list": [9]},
    }
    appended = ledger.append(_AT, "event_observed", payload)
    projected = ledger.as_list()[0]

    expected = {
        "tuple": (1, [2, {"three": (3, 4)}]),
        "list": [5, (6, [7])],
        "dict": {"tuple": (8,), "list": [9]},
    }
    for row in (appended, projected):
        assert {key: row[key] for key in expected} == expected
        assert type(row["tuple"]) is tuple
        assert type(row["tuple"][1]) is list
        assert type(row["tuple"][1][1]) is dict
        assert type(row["tuple"][1][1]["three"]) is tuple
        assert type(row["list"]) is list
        assert type(row["list"][1]) is tuple
        assert type(row["list"][1][1]) is list
        assert type(row["dict"]) is dict
        assert type(row["dict"]["tuple"]) is tuple
        assert type(row["dict"]["list"]) is list

    assert appended["tuple"] is not projected["tuple"]
    assert appended["tuple"][1] is not projected["tuple"][1]
    assert appended["tuple"][1][1] is not projected["tuple"][1][1]
    assert appended["tuple"][1][1]["three"] is not projected["tuple"][1][1]["three"]

    payload["tuple"][1].append("source mutation")
    payload["list"][1][1].append("source mutation")
    payload["dict"]["list"].append("source mutation")
    appended["tuple"][1].append("append-return mutation")
    projected["dict"]["list"].append("as-list mutation")

    fresh = ledger.as_list()[0]
    assert {key: fresh[key] for key in expected} == expected
    assert type(fresh["tuple"]) is tuple
    assert type(fresh["list"]) is list
    assert type(fresh["dict"]) is dict

    assert canonical_json_bytes(ledger.as_list(), "compatibility probe") == (
        b'[{"at":"2031-03-03T09:00:00Z","dict":{"list":[9],"tuple":[8]},'
        b'"index":0,"list":[5,[6,[7]]],"record_type":"event_observed",'
        b'"tuple":[1,[2,{"three":[3,4]}]]}]'
    )


def test_ordinary_payload_keeps_exact_canonical_wire_bytes_and_digest() -> None:
    ledger = TrajectoryLedger()
    payload = {
        "nested": {"steps": [1, True, None, "é"]},
        "score": 2.5,
    }

    appended = ledger.append(_AT, "event_observed", payload)
    projected = ledger.as_list()
    wire = canonical_json_bytes(projected, "compatibility probe")

    assert appended == projected[0]
    assert wire == (
        b'[{"at":"2031-03-03T09:00:00Z","index":0,'
        b'"nested":{"steps":[1,true,null,"\xc3\xa9"]},'
        b'"record_type":"event_observed","score":2.5}]'
    )
    assert ledger.digest() == (
        "d6ecbe1515bcd232e78ac9709d1639d581fd08e870c87e14fcab3ecfb2159894"
    )
