"""Both shipped Cubes are versioned artefacts with committed fingerprints.

`examples/<card>.manifest.json` records a canonical hash of each card's parsed
semantic content and the digest of every compiled variant. YAML formatting and
comments are ignored; only meaning is locked. Any legitimate edit to a card must
update its manifest in the same, reviewed commit.

The fingerprint and verification logic itself lives in `boundarybench.manifest`
and is exercised in `test_manifest_module.py`; this file pins the two artefacts
actually shipped in `examples/`.
"""

from __future__ import annotations

import json

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.freezing import to_json
from boundarybench.loader import load_card
from boundarybench.manifest import (
    MANIFEST_VERSION,
    ManifestMismatchError,
    card_fingerprint,
    load_manifest,
    verify_manifest,
)
from boundarybench.schema import ConstructCard
from tests.conftest import (
    ACCESS_CONSENT_CARD,
    ACCESS_CONSENT_MANIFEST,
    EXAMPLE_CARD,
    EXAMPLE_MANIFEST,
)

SHIPPED = [
    pytest.param(EXAMPLE_CARD, EXAMPLE_MANIFEST, id="maintenance_authority"),
    pytest.param(ACCESS_CONSENT_CARD, ACCESS_CONSENT_MANIFEST, id="access_consent"),
]


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_every_shipped_card_verifies_against_its_manifest(card_path, manifest_path):
    verification = verify_manifest(card_path, manifest_path)
    assert verification.cube.cube_id == load_card(card_path).cube_id
    assert len(verification.cube.variants) == 6


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_every_shipped_manifest_declares_the_supported_format_version(
    card_path, manifest_path
):
    assert load_manifest(manifest_path).manifest_version == MANIFEST_VERSION


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_every_shipped_manifest_names_its_card_and_cube(card_path, manifest_path):
    manifest = load_manifest(manifest_path)
    assert manifest.card == card_path.name
    assert manifest.cube_id == load_card(card_path).cube_id


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_every_compiled_variant_matches_its_manifest(card_path, manifest_path):
    cube = compile_cube(load_card(card_path))
    manifest = load_manifest(manifest_path)
    assert [v.variant_id for v in cube.variants] == list(manifest.variant_ids)
    assert [v.content_digest for v in cube.variants] == list(manifest.content_digests)


def test_the_two_shipped_cubes_share_no_variant_id_or_digest():
    """Cross-Cube uniqueness: a suite cannot silently ship the same task twice."""
    ids: list[str] = []
    digests: list[str] = []
    for _card_path, manifest_path in (p.values for p in SHIPPED):
        manifest = load_manifest(manifest_path)
        ids.extend(manifest.variant_ids)
        digests.extend(manifest.content_digests)
    assert len(set(ids)) == len(ids) == 12
    assert len(set(digests)) == len(digests) == 12


# -- sensitivity: semantics-preserving edits must still be caught ----------


def edited(card_path, **changes) -> ConstructCard:
    raw = json.loads(json.dumps(to_json(load_card(card_path).raw)))
    for path, value in changes.items():
        target = raw
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else target[part]
        last = parts[-1]
        target[int(last) if last.isdigit() else last] = value
    return ConstructCard.from_dict(raw)


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
@pytest.mark.parametrize("direction", ["alarmist", "reassurance"])
def test_changing_a_pressure_cue_breaks_the_fingerprint(
    card_path, manifest_path, direction
):
    card = edited(card_path, **{f"pressure_probes.{direction}.cues.0": "A new line."})
    recorded = load_manifest(manifest_path)
    assert card_fingerprint(card) != recorded.card_fingerprint_sha256
    # Dispositions are untouched, which is exactly why the hash has to notice.
    assert dict(card.disposition_table) == dict(load_card(card_path).disposition_table)


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_changing_a_distractor_value_breaks_the_fingerprint(card_path, manifest_path):
    card = edited(card_path, **{"distractors.0.value": "SYN-PROP-9999"})
    recorded = load_manifest(manifest_path)
    assert card_fingerprint(card) != recorded.card_fingerprint_sha256
    assert dict(card.disposition_table) == dict(load_card(card_path).disposition_table)


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
@pytest.mark.parametrize("direction", ["alarmist", "reassurance"])
def test_a_cue_change_also_moves_that_probe_digest(card_path, manifest_path, direction):
    card = edited(card_path, **{f"pressure_probes.{direction}.cues.0": "A new line."})
    probe = compile_cube(card).probe_variant(direction)
    recorded = {
        record.variant_id: record.content_digest
        for record in load_manifest(manifest_path).variants
    }
    assert probe.content_digest != recorded[probe.variant_id]


@pytest.mark.parametrize(("card_path", "manifest_path"), SHIPPED)
def test_a_distractor_change_moves_every_variant_digest(card_path, manifest_path):
    """Distractors are in every variant's observation set, so all six move."""
    card = edited(card_path, **{"distractors.0.value": "SYN-PROP-9999"})
    recorded = {
        record.variant_id: record.content_digest
        for record in load_manifest(manifest_path).variants
    }
    for variant in compile_cube(card).variants:
        assert variant.content_digest != recorded[variant.variant_id]


def test_a_manifest_cannot_be_swapped_between_the_two_shipped_cubes():
    with pytest.raises(ManifestMismatchError):
        verify_manifest(EXAMPLE_CARD, ACCESS_CONSENT_MANIFEST)
    with pytest.raises(ManifestMismatchError):
        verify_manifest(ACCESS_CONSENT_CARD, EXAMPLE_MANIFEST)
