"""Synthetic controls for the narrowly observed response metadata."""

import json

import pytest

from operatebench.providers import openai_responses as op
from operatebench.providers.faults import AdapterProviderError
from operatebench.providers.wire import WireResponse
from tests.test_openai_response_extensions import _body, _observed, _parsed


def _check(raw, typed=None):
    typed = raw if typed is None else typed
    op.check_response_admissible(
        WireResponse(json.dumps(raw), lambda: _parsed(typed), kind="response"),
        model=raw["model"],
    )


def test_observed_synthetic_and_absent():
    body = _body(_observed())
    _check(body)
    body.update(image={"context_compression": "disabled"}, input_transformations=[])
    _check(body)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", {}),
        ("image", {"context_compression": "enabled"}),
        ("image", {"context_compression": "disabled", "extra": None}),
        ("image", True),
        ("image", {"context_compression": 0}),
        ("image", {"context_compression": False}),
        ("input_transformations", [None]),
        ("input_transformations", {}),
        ("input_transformations", False),
        ("input_transformations", 0),
        ("unknown_metadata", None),
        ("store", 1),
    ],
)
def test_invalid_synthetic_metadata(field, value):
    body = _body(_observed())
    body[field] = value
    with pytest.raises(AdapterProviderError):
        _check(body)


def test_readings_must_agree():
    raw = _body(_observed())
    raw.update(image={"context_compression": "disabled"}, input_transformations=[])
    typed = dict(raw)
    del typed["image"]
    with pytest.raises(AdapterProviderError):
        _check(raw, typed)
