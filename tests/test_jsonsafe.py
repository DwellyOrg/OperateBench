"""Direct regressions for the canonical JSON-safety boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from operatebench.jsonsafe import (
    MAX_JSON_INTEGER_DIGITS,
    NonJsonValueError,
    TextEncodingError,
    UnrenderableValueError,
    ensure_json_safe,
)


class _HostileStr(str):
    __hash__ = str.__hash__

    def __str__(self) -> str:
        raise AssertionError("hostile __str__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __eq__(self, other: object) -> bool:
        raise AssertionError("hostile comparison executed")

    def __deepcopy__(self, memo: object) -> str:
        raise AssertionError("hostile deepcopy executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile reduce executed")


class _HostileInt(int):
    def __int__(self) -> int:
        raise AssertionError("hostile __int__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __lt__(self, other: object) -> bool:
        raise AssertionError("hostile comparison executed")

    def __deepcopy__(self, memo: object) -> int:
        raise AssertionError("hostile deepcopy executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile reduce executed")


class _HostileFloat(float):
    def __float__(self) -> float:
        raise AssertionError("hostile __float__ executed")

    def __repr__(self) -> str:
        raise AssertionError("hostile __repr__ executed")

    def __lt__(self, other: object) -> bool:
        raise AssertionError("hostile comparison executed")

    def __deepcopy__(self, memo: object) -> float:
        raise AssertionError("hostile deepcopy executed")

    def __reduce__(self) -> object:
        raise AssertionError("hostile reduce executed")


class _ItemsMapping(Mapping[str, object]):
    def __init__(self, items: tuple[tuple[str, object], ...]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> object:
        raise AssertionError("mapping lookup must not run")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("mapping iteration must not run")

    def __len__(self) -> int:
        return len(self._items)

    def items(self):  # type: ignore[no-untyped-def]
        return self._items


def test_ensure_json_safe_validates_accepted_subclasses_without_user_hooks() -> None:
    value = _ItemsMapping(
        (
            (
                _HostileStr("outer"),
                [_HostileStr("value"), _HostileInt(7), _HostileFloat(2.5)],
            ),
        )
    )

    assert ensure_json_safe(value, "payload") is value


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (_HostileFloat(math.nan), NonJsonValueError),
        (_HostileFloat(math.inf), NonJsonValueError),
        (_HostileFloat(-math.inf), NonJsonValueError),
        (_HostileInt(10**MAX_JSON_INTEGER_DIGITS), UnrenderableValueError),
        (_HostileStr("\ud800"), TextEncodingError),
    ),
    ids=("nan", "positive-infinity", "negative-infinity", "wide-int", "surrogate"),
)
def test_ensure_json_safe_rejects_hostile_invalid_scalars_canonically(
    value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        ensure_json_safe({"value": value}, "payload")


def test_ensure_json_safe_preserves_ordinary_failure_diagnostics() -> None:
    with pytest.raises(
        NonJsonValueError,
        match=(
            r"^payload\.bad: value of type set is not JSON-safe; use null, bool, int, "
            r"finite float, string, list or string-keyed mapping$"
        ),
    ):
        ensure_json_safe({"bad": set()}, "payload")

    with pytest.raises(
        NonJsonValueError,
        match=(r"^payload: mappings must use string keys, got 7 of type int$"),
    ):
        ensure_json_safe({7: "value"}, "payload")
