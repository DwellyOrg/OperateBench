"""The semantic manifest verifier is production code, not a test helper.

Everything here exercises `boundarybench.manifest` directly: the canonical card
fingerprint, strict JSON loading, and the verification that a manifest still
describes the card sitting next to it. Every rejection must be a named domain
error carrying enough context to act on — never a raw ``KeyError``,
``TypeError`` or JSON traceback.
"""

from __future__ import annotations

import copy
import json

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.loader import load_card
from boundarybench.manifest import (
    MANIFEST_VERSION,
    ManifestError,
    ManifestFormatError,
    ManifestMismatchError,
    card_fingerprint,
    load_manifest,
    verify_manifest,
)
from tests.conftest import ACCESS_CONSENT_CARD, EXAMPLE_CARD, EXAMPLE_MANIFEST


def valid_manifest() -> dict:
    """The shipped manifest, as a mutable plain dict."""
    return json.loads(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))


def written(tmp_path, payload, *, name="maintenance_authority.manifest.json"):
    path = tmp_path / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def verify(tmp_path, payload, *, card=EXAMPLE_CARD):
    return verify_manifest(card, written(tmp_path, payload))


# -- the canonical fingerprint ----------------------------------------------


def test_the_fingerprint_is_a_lowercase_sha256_hex_digest():
    digest = card_fingerprint(load_card(EXAMPLE_CARD))
    assert len(digest) == 64
    assert digest == digest.lower()
    assert int(digest, 16) >= 0


def test_the_fingerprint_is_stable_across_reloads():
    assert card_fingerprint(load_card(EXAMPLE_CARD)) == card_fingerprint(
        load_card(EXAMPLE_CARD)
    )


def test_the_fingerprint_separates_the_two_shipped_cards():
    assert card_fingerprint(load_card(EXAMPLE_CARD)) != card_fingerprint(
        load_card(ACCESS_CONSENT_CARD)
    )


def test_the_fingerprint_ignores_yaml_formatting_but_not_meaning(tmp_path):
    """Reflowing the file is free; changing a value is not."""
    import yaml

    from boundarybench.freezing import to_json

    raw = to_json(load_card(EXAMPLE_CARD).raw)
    reflowed = tmp_path / "reflowed.yaml"
    reflowed.write_text(
        yaml.safe_dump(raw, sort_keys=True, default_flow_style=True), encoding="utf-8"
    )
    assert card_fingerprint(load_card(reflowed)) == card_fingerprint(
        load_card(EXAMPLE_CARD)
    )

    raw["distractors"][0]["value"] = "SYN-PROP-9999"
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    assert card_fingerprint(load_card(changed)) != card_fingerprint(
        load_card(EXAMPLE_CARD)
    )


# -- strict loading ----------------------------------------------------------


def test_the_shipped_manifest_loads_and_reports_its_version():
    manifest = load_manifest(EXAMPLE_MANIFEST)
    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.card == "maintenance_authority.yaml"
    assert manifest.cube_id == "lettings_maintenance_authority_v1"
    assert len(manifest.variants) == 6


def test_a_missing_manifest_is_a_named_error(tmp_path):
    with pytest.raises(ManifestFormatError, match="cannot read"):
        load_manifest(tmp_path / "nope.manifest.json")


@pytest.mark.parametrize(
    "text", ["", "   ", "{", "not json", "[]", '"a string"', "null", "7"]
)
def test_manifest_json_that_is_not_an_object_is_rejected(tmp_path, text):
    with pytest.raises(ManifestFormatError):
        load_manifest(written(tmp_path, text))


def test_a_binary_manifest_is_a_named_error_not_a_unicode_traceback(tmp_path):
    """A manifest that is not UTF-8 text is malformed input, not a crash."""
    path = tmp_path / "maintenance_authority.manifest.json"
    path.write_bytes(b'{"cube_id": "\xff\xfe"}')
    with pytest.raises(ManifestFormatError, match="not valid UTF-8"):
        load_manifest(path)


def test_the_binary_manifest_message_names_the_file(tmp_path):
    path = tmp_path / "maintenance_authority.manifest.json"
    path.write_bytes(b"\x80\x81\x82")
    with pytest.raises(ManifestFormatError) as excinfo:
        load_manifest(path)
    assert str(path) in str(excinfo.value)


def test_a_manifest_decoding_failure_does_not_swallow_programming_errors(
    tmp_path, monkeypatch
):
    """The UTF-8 guard is narrow: only a decode failure becomes a domain error."""

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("programming error")

    monkeypatch.setattr("pathlib.Path.read_text", boom)
    with pytest.raises(RuntimeError, match="programming error"):
        load_manifest(tmp_path / "nope.manifest.json")


def test_duplicate_json_keys_are_rejected(tmp_path):
    """Stock json keeps the last duplicate; a versioned artefact cannot."""
    payload = valid_manifest()
    text = json.dumps(payload)
    injected = text.replace('{"', '{"cube_id": "sneaky_v1", "', 1)
    with pytest.raises(ManifestFormatError, match="duplicate"):
        load_manifest(written(tmp_path, injected))


def test_duplicate_json_keys_are_rejected_inside_a_variant(tmp_path):
    payload = valid_manifest()
    text = json.dumps(payload).replace(
        '{"variant_id"', '{"variant_id": "sneaky", "variant_id"', 1
    )
    with pytest.raises(ManifestFormatError, match="duplicate"):
        load_manifest(written(tmp_path, injected := text))
    assert "sneaky" in injected


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_are_rejected(tmp_path, constant):
    payload = valid_manifest()
    text = json.dumps(payload).replace('"manifest_version": 1', f'"x": {constant}', 1)
    with pytest.raises(ManifestFormatError):
        load_manifest(written(tmp_path, text))


@pytest.mark.parametrize(
    "field",
    ["manifest_version", "card", "cube_id", "card_fingerprint_sha256", "variants"],
)
def test_a_missing_required_field_is_rejected(tmp_path, field):
    payload = valid_manifest()
    del payload[field]
    with pytest.raises(ManifestFormatError, match=field):
        load_manifest(written(tmp_path, payload))


def test_an_unknown_field_is_rejected_rather_than_ignored(tmp_path):
    payload = valid_manifest()
    payload["reviewed_by"] = "nobody"
    with pytest.raises(ManifestFormatError, match="reviewed_by"):
        load_manifest(written(tmp_path, payload))


def test_the_optional_note_is_allowed_but_must_be_a_string(tmp_path):
    payload = valid_manifest()
    payload["note"] = "explicitly reviewed artefact revision"
    assert load_manifest(written(tmp_path, payload)).note.startswith("explicitly")

    payload["note"] = 7
    with pytest.raises(ManifestFormatError, match="note"):
        load_manifest(written(tmp_path, payload))

    payload.pop("note")
    assert load_manifest(written(tmp_path, payload)).note is None


@pytest.mark.parametrize("keep_note", [True, False])
def test_a_parsed_manifest_round_trips_through_its_own_dict(tmp_path, keep_note):
    """`as_dict` is the writer side of the format: what it emits must reload.

    A manifest is regenerated whenever a card is legitimately revised, so the
    two directions have to agree exactly — including the optional note, which
    is emitted only when it was present.
    """
    payload = valid_manifest()
    if not keep_note:
        payload.pop("note")
    original = load_manifest(written(tmp_path, payload))

    emitted = original.as_dict()
    assert emitted == payload
    assert ("note" in emitted) is keep_note
    assert load_manifest(written(tmp_path, emitted, name="round.json")) == original


def test_an_explicitly_null_note_is_rejected_rather_than_read_as_absent(tmp_path):
    """Omitting a field and writing ``null`` into it are different statements.

    Reading ``"note": null`` as "no note" would be the one place a manifest
    field is silently coerced instead of refused.
    """
    payload = valid_manifest()
    payload["note"] = None
    with pytest.raises(ManifestFormatError, match="note"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("value", [2, 0, -1, "1", 1.0, True, None])
def test_an_unsupported_manifest_version_is_rejected(tmp_path, value):
    payload = valid_manifest()
    payload["manifest_version"] = value
    with pytest.raises(ManifestFormatError, match="manifest_version"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("field", ["card", "cube_id", "card_fingerprint_sha256"])
@pytest.mark.parametrize("value", [7, None, [], {}, "", "   "])
def test_string_fields_must_be_non_empty_strings(tmp_path, field, value):
    payload = valid_manifest()
    payload[field] = value
    with pytest.raises(ManifestFormatError, match=field):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize(
    "value",
    [
        "not-a-hash",
        "ABCDEF" + "0" * 58,
        "0" * 63,
        "0" * 65,
        "g" * 64,
    ],
)
def test_a_malformed_card_fingerprint_is_rejected(tmp_path, value):
    payload = valid_manifest()
    payload["card_fingerprint_sha256"] = value
    with pytest.raises(ManifestFormatError, match="fingerprint"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize(
    "value", ["../elsewhere.yaml", "sub/card.yaml", "/abs/card.yaml"]
)
def test_the_card_field_must_be_a_bare_filename(tmp_path, value):
    payload = valid_manifest()
    payload["card"] = value
    with pytest.raises(ManifestFormatError, match="filename"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("value", [{}, "six", 7, None])
def test_variants_must_be_a_list(tmp_path, value):
    payload = valid_manifest()
    payload["variants"] = value
    with pytest.raises(ManifestFormatError, match="variants"):
        load_manifest(written(tmp_path, payload))


def test_a_manifest_must_record_exactly_six_variants(tmp_path):
    payload = valid_manifest()
    payload["variants"] = payload["variants"][:5]
    with pytest.raises(ManifestFormatError, match="exactly 6"):
        load_manifest(written(tmp_path, payload))

    payload = valid_manifest()
    payload["variants"].append(copy.deepcopy(payload["variants"][0]))
    with pytest.raises(ManifestFormatError, match="exactly 6"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("value", ["a string", 7, None, []])
def test_each_variant_entry_must_be_an_object(tmp_path, value):
    payload = valid_manifest()
    payload["variants"][2] = value
    with pytest.raises(ManifestFormatError, match=r"variants\[2\]"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("field", ["variant_id", "content_digest"])
def test_each_variant_entry_needs_both_fields(tmp_path, field):
    payload = valid_manifest()
    del payload["variants"][0][field]
    with pytest.raises(ManifestFormatError, match=field):
        load_manifest(written(tmp_path, payload))


def test_a_variant_entry_may_not_carry_unknown_fields(tmp_path):
    payload = valid_manifest()
    payload["variants"][0]["comment"] = "why not"
    with pytest.raises(ManifestFormatError, match="comment"):
        load_manifest(written(tmp_path, payload))


@pytest.mark.parametrize("value", [7, None, ["a"], "z" * 64, "abc"])
def test_a_malformed_variant_digest_is_rejected(tmp_path, value):
    payload = valid_manifest()
    payload["variants"][0]["content_digest"] = value
    with pytest.raises(ManifestFormatError, match="content_digest"):
        load_manifest(written(tmp_path, payload))


def test_a_duplicate_variant_id_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["variants"][1]["variant_id"] = payload["variants"][0]["variant_id"]
    with pytest.raises(ManifestFormatError, match="duplicate variant id"):
        load_manifest(written(tmp_path, payload))


# -- verification against the card ------------------------------------------


def test_the_shipped_pair_verifies():
    verification = verify_manifest(EXAMPLE_CARD, EXAMPLE_MANIFEST)
    assert verification.cube.cube_id == "lettings_maintenance_authority_v1"
    assert verification.manifest.manifest_version == MANIFEST_VERSION
    payload = verification.as_dict()
    assert payload["verified"] is True
    assert payload["variant_count"] == 6
    assert payload["cube_type"] == "1-ACT"
    assert payload["fact_location"] == "elicited"
    assert payload["privacy_status"] == "SYNTHETIC_ONLY"
    assert len(payload["variants"]) == 6


def test_a_manifest_naming_a_different_card_file_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["card"] = "access_consent.yaml"
    with pytest.raises(ManifestMismatchError, match="names card"):
        verify(tmp_path, payload)


def test_a_manifest_naming_a_different_cube_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["cube_id"] = "lettings_access_consent_v1"
    with pytest.raises(ManifestMismatchError, match="cube_id"):
        verify(tmp_path, payload)


def test_a_stale_card_fingerprint_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["card_fingerprint_sha256"] = "0" * 64
    with pytest.raises(ManifestMismatchError, match="fingerprint"):
        verify(tmp_path, payload)


def test_a_stale_variant_digest_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["variants"][3]["content_digest"] = "0" * 64
    with pytest.raises(ManifestMismatchError, match="content digest"):
        verify(tmp_path, payload)


def test_a_renamed_variant_is_rejected(tmp_path):
    payload = valid_manifest()
    payload["variants"][0]["variant_id"] += "_extra"
    with pytest.raises(ManifestMismatchError, match="variant ids"):
        verify(tmp_path, payload)


def test_reordered_variants_are_rejected(tmp_path):
    """Order is part of the contract: the compiler emits a canonical sequence."""
    payload = valid_manifest()
    payload["variants"].reverse()
    with pytest.raises(ManifestMismatchError, match="variant ids"):
        verify(tmp_path, payload)


def test_a_manifest_for_the_wrong_card_entirely_is_rejected(tmp_path):
    with pytest.raises(ManifestMismatchError):
        verify_manifest(ACCESS_CONSENT_CARD, EXAMPLE_MANIFEST)


def test_an_unreadable_card_surfaces_as_a_schema_error(tmp_path):
    from boundarybench.schema import SchemaError

    with pytest.raises(SchemaError):
        verify_manifest(tmp_path / "nope.yaml", EXAMPLE_MANIFEST)


# -- outcome-preserving semantic edits still break verification -------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("pressure_probes.alarmist.cues.0", "A differently alarming line."),
        ("pressure_probes.reassurance.cues.2", "A differently reassuring line."),
        ("distractors.0.value", "SYN-PROP-9999"),
        ("state_axis.S0.description", "Rewritten but equivalent description."),
        ("policy_axis.P0.rules.0.id", "R_P0_RENAMED"),
    ],
)
def test_an_edit_that_preserves_every_disposition_still_breaks_the_manifest(
    tmp_path, path, value
):
    import yaml

    from boundarybench.freezing import to_json

    original = load_card(EXAMPLE_CARD)
    raw = to_json(original.raw)
    target = raw
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    last = parts[-1]
    target[int(last) if last.isdigit() else last] = value

    edited = tmp_path / "maintenance_authority.yaml"
    edited.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    card = load_card(edited)
    assert dict(card.disposition_table) == dict(original.disposition_table)
    assert [v.expected_disposition for v in compile_cube(card).variants] == [
        v.expected_disposition for v in compile_cube(original).variants
    ]

    with pytest.raises(ManifestMismatchError):
        verify_manifest(edited, EXAMPLE_MANIFEST)


def test_every_named_error_is_a_manifest_error():
    assert issubclass(ManifestFormatError, ManifestError)
    assert issubclass(ManifestMismatchError, ManifestError)
    assert issubclass(ManifestError, ValueError)
