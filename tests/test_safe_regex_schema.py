"""Published authored-regex grammar generation and differential soundness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operatebench import cards
from tools.generate_safe_regex_schema import (
    INPUT_MAX_CHARACTERS,
    PATTERN_MAX_CHARACTERS,
    PATTERN_MAX_UTF8_BYTES,
    REPEAT_MAXIMUM,
    authored_pattern_schema,
)
from tools.prove_safe_regex_schema import (
    _bounded_cases,
    _random_cases,
    _runtime_accepts,
    _schema_accepts,
    _structured_cases,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs/schemas"
PACKAGED = REPO_ROOT / "src/operatebench/resources/cards"


def _load(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def test_generated_authored_pattern_schema_is_the_published_source_of_truth() -> None:
    definitions = _load("card-defs-v1.schema.json")["$defs"]
    assert definitions["safeRegexPattern"] == authored_pattern_schema()
    expected_reference = {"$ref": "#/$defs/safeRegexPattern"}
    assert (
        definitions["fieldShape"]["oneOf"][0]["properties"]["pattern"]["oneOf"][0]
        == expected_reference
    )
    assert (
        definitions["jsonSchemaSubset"]["oneOf"][3]["properties"]["pattern"]["oneOf"][0]
        == expected_reference
    )


def test_safe_regex_registry_limits_match_runtime_and_generator() -> None:
    registry = _load("card-registry-v1.json")["strict_json_schema_subset"]["safe_regex"]
    assert (
        registry["pattern_max_characters"]
        == PATTERN_MAX_CHARACTERS
        == cards._SAFE_PATTERN_MAX_CHARS
    )
    assert (
        registry["pattern_max_utf8_bytes"]
        == PATTERN_MAX_UTF8_BYTES
        == cards._SAFE_PATTERN_MAX_BYTES
    )
    assert (
        registry["input_max_characters"]
        == INPUT_MAX_CHARACTERS
        == cards._SAFE_PATTERN_INPUT_MAX_CHARS
    )
    assert registry["repeat_maximum"] == REPEAT_MAXIMUM == cards._SAFE_PATTERN_MAX_REPEAT


@pytest.mark.parametrize("name", ("card-defs-v1.schema.json", "card-registry-v1.json"))
def test_normative_and_packaged_safe_regex_contracts_are_byte_identical(
    name: str,
) -> None:
    assert (DOCS / name).read_bytes() == (PACKAGED / name).read_bytes()


def test_draft_2020_12_meta_schema_accepts_generated_card_definitions() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(_load("card-defs-v1.schema.json"))


def test_bounded_structured_and_seeded_differential_has_no_false_accepts() -> None:
    schema = authored_pattern_schema()
    # The standalone proof expands this to length four plus 5,000 seeded cases.
    bounded = [pattern for pattern in _bounded_cases() if len(pattern) <= 3]
    structured = sorted(_structured_cases())
    seeded = _random_cases()[:250]
    false_accepts = [
        pattern
        for pattern in bounded + structured + seeded
        if _schema_accepts(pattern, schema) and not _runtime_accepts(pattern)
    ]
    assert false_accepts == []


def test_documented_schema_conservatism_is_runtime_accepted_but_not_authorable() -> None:
    schema = authored_pattern_schema()
    compatibility_forms = (
        "a{01}",
        "a{01,02}",
        "a{01,}",
        "a{2,2}",
        "a{10,20}",
        r"[\.-/]",
        "[b-y]",
        "[a-b-c]",
    )
    assert all(_runtime_accepts(pattern) for pattern in compatibility_forms)
    assert not any(_schema_accepts(pattern, schema) for pattern in compatibility_forms)
