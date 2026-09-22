"""YAML loading contract.

A duplicate mapping key is silently last-wins in stock PyYAML. For a benchmark
card that is a correctness hazard: two `disposition_table` blocks would compile
without complaint and only one of them would be the ground truth.
"""

from __future__ import annotations

import pytest

from boundarybench.loader import load_card
from boundarybench.schema import SchemaError
from tests.conftest import EXAMPLE_CARD


def write(tmp_path, text: str):
    path = tmp_path / "card.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_the_shipped_example(tmp_path):
    assert load_card(EXAMPLE_CARD).cube_id == "lettings_maintenance_authority_v1"


def test_duplicate_top_level_key_is_rejected(tmp_path):
    path = write(tmp_path, "cube_id: first\ndomain: lettings\ncube_id: second\n")
    with pytest.raises(SchemaError, match="duplicate"):
        load_card(path)


def test_duplicate_nested_key_is_rejected(tmp_path):
    path = write(
        tmp_path,
        "state_axis:\n  S0:\n    label: a\n    label: b\n  S1:\n    label: c\n",
    )
    with pytest.raises(SchemaError, match="duplicate"):
        load_card(path)


def test_duplicate_key_inside_a_list_item_is_rejected(tmp_path):
    path = write(tmp_path, "distractors:\n  - key: a\n    value: 1\n    key: b\n")
    with pytest.raises(SchemaError, match="duplicate"):
        load_card(path)


def test_duplicate_key_message_names_the_key(tmp_path):
    path = write(tmp_path, "cube_id: first\ncube_id: second\n")
    with pytest.raises(SchemaError, match="cube_id"):
        load_card(path)


def test_invalid_yaml_is_rejected(tmp_path):
    path = write(tmp_path, "cube_id: [unclosed\n")
    with pytest.raises(SchemaError, match="not valid YAML"):
        load_card(path)


def test_empty_file_is_rejected(tmp_path):
    path = write(tmp_path, "")
    with pytest.raises(SchemaError, match="empty"):
        load_card(path)


def test_comment_only_file_is_rejected(tmp_path):
    path = write(tmp_path, "# nothing here\n")
    with pytest.raises(SchemaError, match="empty"):
        load_card(path)


def test_non_mapping_top_level_is_rejected(tmp_path):
    path = write(tmp_path, "- one\n- two\n")
    with pytest.raises(SchemaError, match="mapping"):
        load_card(path)


def test_scalar_top_level_is_rejected(tmp_path):
    path = write(tmp_path, "just a string\n")
    with pytest.raises(SchemaError, match="mapping"):
        load_card(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(SchemaError, match="cannot read"):
        load_card(tmp_path / "absent.yaml")


def test_a_binary_card_is_a_schema_error_not_a_unicode_traceback(tmp_path):
    """A card that is not UTF-8 text is malformed input, not a crash."""
    path = tmp_path / "card.yaml"
    path.write_bytes(b"cube_id: \xff\xfe not utf-8\n")
    with pytest.raises(SchemaError, match="not valid UTF-8"):
        load_card(path)


def test_the_binary_card_message_names_the_file_and_the_artefact(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_bytes(b"\x80\x81\x82")
    with pytest.raises(SchemaError) as excinfo:
        load_card(path)
    message = str(excinfo.value)
    assert "construct card" in message
    assert str(path) in message


def test_a_decoding_failure_does_not_swallow_unrelated_programming_errors(
    tmp_path, monkeypatch
):
    """The UTF-8 guard is narrow: only a decode failure becomes a SchemaError."""

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("programming error")

    monkeypatch.setattr("pathlib.Path.read_text", boom)
    with pytest.raises(RuntimeError, match="programming error"):
        load_card(tmp_path / "card.yaml")


def test_loader_does_not_construct_arbitrary_python_objects(tmp_path):
    """Safe construction must survive the duplicate-key subclassing."""
    path = write(tmp_path, "value: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(SchemaError, match="not valid YAML"):
        load_card(path)
