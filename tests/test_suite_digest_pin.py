"""The suite identity pin: a declared digest that must match what is computed.

A suite manifest that only *reports* its content digest cannot detect drift —
canonicalisation or payload changes silently rewrite the collection's identity
while every test stays green. The digest is therefore declared in the versioned
artefact and verified on every load, so semantic drift needs an explicit,
reviewed re-pin in the same commit.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import yaml

from boundarybench.compiler import compile_cube
from boundarybench.loader import load_card
from boundarybench.manifest import MANIFEST_VERSION, card_fingerprint
from boundarybench.suite import (
    SuiteDigestError,
    SuiteFormatError,
    load_suite,
    suite_content_digest,
    validate_suite,
)
from tests.conftest import SUITE_MANIFEST
from tests.test_suite import repinned, rewritten, suite_dict

#: The digest committed in ``examples/methodology_spike_suite.yaml``. Written out
#: by hand, never derived from production code at test time — see
#: ``test_the_committed_pin_is_this_exact_digest``.
COMMITTED_SUITE_DIGEST = (
    "7200bb46dc6cacba47ca4df04102c0ce5a318bfce1a495343eb706471091c293"
)

#: A disclaimer a lettings suite would plausibly ship: it quotes a currency, so
#: it is not ASCII. Every shipped artefact today happens to be pure ASCII, which
#: leaves the canonical encoding's non-ASCII behaviour unobserved.
UNICODE_DISCLAIMER = (
    "Synthetic methodology spike for benchmark development only. Repair "
    "authorisation thresholds are quoted in £, so the canonical suite encoding "
    "is exercised on non-ASCII text. Not official Dwelly policy."
)

#: The digest of the shipped collection carrying :data:`UNICODE_DISCLAIMER`,
#: computed out of band by hand-encoding the documented canonical form — sorted
#: keys, compact separators, non-ASCII left unescaped, hashed as UTF-8 bytes —
#: never by calling ``suite_content_digest``. See
#: ``test_the_canonical_encoding_leaves_non_ascii_text_unescaped``.
UNICODE_DISCLAIMER_SUITE_DIGEST = (
    "f76b4e9df605566a28ad1c1b3c0d0ca6fda32151847d8f34facce4ee31902dfc"
)


def rewrite_member_card(suite_dir, card_name, mutate):
    """Mutate one member card and re-pin only *its* semantic manifest.

    The suite's own pin is deliberately left stale, so what the test observes is
    the collection-level identity gate rather than the per-card one.
    """
    card_path = suite_dir / card_name
    raw = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    mutate(raw)
    card_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    card = load_card(card_path)
    cube = compile_cube(card)
    (suite_dir / f"{card_path.stem}.manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "card": card_name,
                "cube_id": card.cube_id,
                "card_fingerprint_sha256": card_fingerprint(card),
                "variants": [
                    {"variant_id": v.variant_id, "content_digest": v.content_digest}
                    for v in cube.variants
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return suite_dir / "methodology_spike_suite.yaml"


def test_a_suite_without_a_declared_content_digest_is_rejected(suite_dir):
    payload = suite_dict()
    del payload["suite_content_digest"]
    with pytest.raises(SuiteFormatError, match="suite_content_digest"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "value",
    [
        None,
        7,
        True,
        1.0,
        [],
        {},
        "",
        "   ",
        "not a digest",
        "0" * 63,
        "0" * 65,
        # The real committed digest, wearing three shapes the schema refuses.
        COMMITTED_SUITE_DIGEST.upper(),
        COMMITTED_SUITE_DIGEST[:-1] + "g",
        " " + COMMITTED_SUITE_DIGEST,
    ],
    ids=[
        "null",
        "int",
        "bool",
        "float",
        "list",
        "mapping",
        "empty",
        "blank",
        "prose",
        "short",
        "long",
        "uppercase",
        "non-hex",
        "padded",
    ],
)
def test_a_malformed_declared_digest_is_a_named_format_error(suite_dir, value):
    """Never a raw exception: a hand-edited pin is untrusted input."""
    payload = suite_dict()
    payload["suite_content_digest"] = value
    with pytest.raises(SuiteFormatError, match="suite_content_digest"):
        load_suite(rewritten(suite_dir, payload))


def test_a_declared_digest_that_disagrees_with_the_artefacts_fails_closed(suite_dir):
    """A well-formed but wrong pin is a mismatch, not a warning."""
    payload = suite_dict()
    payload["suite_content_digest"] = "0" * 64
    with pytest.raises(SuiteDigestError) as excinfo:
        validate_suite(rewritten(suite_dir, payload))
    error = excinfo.value
    assert isinstance(error, SuiteFormatError)
    assert error.declared == "0" * 64
    assert error.computed == validate_suite(SUITE_MANIFEST).content_digest


# -- the pin is the digest the report carries --------------------------------


def test_the_shipped_report_digest_is_the_verified_declared_digest():
    report = validate_suite(SUITE_MANIFEST)
    assert report.content_digest == report.manifest.suite_content_digest
    assert report.as_dict()["suite_content_digest_sha256"] == report.content_digest


def test_the_committed_pin_is_this_exact_digest():
    """An independent oracle for the committed identity.

    The expected value is written out by hand in this module, not recomputed
    from ``suite_content_digest`` at test time, so a change to the payload
    fields or to the canonical encoding cannot move the shipped identity and the
    expectation together. Updating this constant is a reviewed decision.
    """
    declared = yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))
    assert declared["suite_content_digest"] == COMMITTED_SUITE_DIGEST
    assert validate_suite(SUITE_MANIFEST).content_digest == COMMITTED_SUITE_DIGEST


def test_the_canonical_encoding_leaves_non_ascii_text_unescaped():
    """``ensure_ascii=False`` is part of the identity contract, not a detail.

    Every artefact shipped today is pure ASCII, so escaping or not escaping
    non-ASCII text produces the same digest and the choice is unobservable —
    yet the first suite that quotes a currency, a name or a place would get a
    different identity under each spelling. The expectation is therefore a
    literal, computed out of band by hand-encoding the documented canonical form
    rather than by calling ``suite_content_digest``, so dropping or flipping
    ``ensure_ascii=False`` cannot move the production encoding and this
    expectation together. Nothing shipped changes: the manifest is replaced on
    the *parsed* suite, over the real validated member Cubes.
    """
    assert not UNICODE_DISCLAIMER.isascii()
    report = validate_suite(SUITE_MANIFEST)
    drifted = dataclasses.replace(report.manifest, disclaimer=UNICODE_DISCLAIMER)
    assert suite_content_digest(drifted, report.cubes) == (
        UNICODE_DISCLAIMER_SUITE_DIGEST
    )


# -- every identity-relevant component moves the digest ----------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suite_id", "lettings_methodology_spike_b"),
        ("benchmark_version", "0.2.0-dev.2"),
        ("disclaimer", "A different synthetic disclaimer. Not official Dwelly policy."),
    ],
)
def test_stale_pin_rejected_after_suite_metadata_changes(suite_dir, field, value):
    payload = suite_dict()
    payload[field] = value
    with pytest.raises(SuiteDigestError):
        validate_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "field",
    ["schema_version", "suite_id", "benchmark_version", "status", "privacy_status"],
)
def test_the_payload_covers_every_pinned_metadata_field(field):
    """Including the fields the schema currently admits only one value for.

    ``schema_version``, ``status`` and ``privacy_status`` cannot be varied
    through the artefact today, so the only way to prove the identity payload
    covers them — and that widening any of those vocabularies later cannot let
    two different suites share a digest — is to vary them on the parsed
    manifest directly.
    """
    report = validate_suite(SUITE_MANIFEST)
    original = getattr(report.manifest, field)
    altered = original + 1 if isinstance(original, int) else original + "_x"
    drifted = dataclasses.replace(report.manifest, **{field: altered})
    assert suite_content_digest(drifted, report.cubes) != report.content_digest


def test_the_payload_covers_the_normalized_disclaimer():
    report = validate_suite(SUITE_MANIFEST)
    drifted = dataclasses.replace(
        report.manifest, disclaimer="Some other synthetic disclaimer."
    )
    assert suite_content_digest(drifted, report.cubes) != report.content_digest


@pytest.mark.parametrize(
    ("section", "counts"),
    [
        ("cube_types", {"1-ACT": 2, "3-ACT": 0}),
        ("fact_locations", {"elicited": 2, "initial_state": 0}),
        ("expected_dispositions", {"ACT": 7, "STOP": 5}),
    ],
)
def test_stale_pin_rejected_after_a_declared_constraint_changes(
    suite_dir, section, counts
):
    """Each constraint family is inside the identity, not beside it."""
    payload = suite_dict()
    payload["constraints"][section] = counts
    with pytest.raises(SuiteDigestError):
        validate_suite(rewritten(suite_dir, payload))


def test_stale_pin_rejected_after_the_entry_order_changes(suite_dir):
    """Order is identity: the same two Cubes swapped is a different collection.

    Every declared tally still matches, and every path, id and digest is
    unchanged, so nothing but the ordered payload can catch this.
    """
    payload = suite_dict()
    payload["entries"] = list(reversed(payload["entries"]))
    with pytest.raises(SuiteDigestError):
        validate_suite(rewritten(suite_dir, payload))
    assert validate_suite(repinned(suite_dir, payload)).ok is True


def test_stale_pin_rejected_after_a_member_cards_content_changes(suite_dir):
    """A re-pinned member manifest is not enough; the suite pin must move too."""

    def mutate(raw):
        raw["distractors"][0]["value"] = "SYN-PROP-9999"

    path = rewrite_member_card(suite_dir, "access_consent.yaml", mutate)
    with pytest.raises(SuiteDigestError):
        validate_suite(path)


def test_stale_pin_rejected_after_a_variant_grading_field_changes(suite_dir):
    """Dropping a critical invariant weakens grading and must move the identity.

    Nothing about the collection's shape changes: same Cubes, same paths, same
    cube types, fact locations and ACT/STOP balance. Only the grading contract
    inside each variant's content digest moves.
    """

    def mutate(raw):
        raw["critical_invariants"] = [
            name
            for name in raw["critical_invariants"]
            if name != "NO_FABRICATED_EVIDENCE_REFERENCE"
        ]

    path = rewrite_member_card(suite_dir, "access_consent.yaml", mutate)
    with pytest.raises(SuiteDigestError):
        validate_suite(path)


def test_the_payload_covers_each_card_fingerprint_on_its_own():
    """Isolated from the variant digests, which move with it through a card."""
    report = validate_suite(SUITE_MANIFEST)
    cubes = list(report.cubes)
    cubes[1] = dataclasses.replace(cubes[1], card_fingerprint_sha256="0" * 64)
    assert suite_content_digest(report.manifest, cubes) != report.content_digest


def test_the_payload_covers_each_ordered_variant_digest_on_its_own():
    report = validate_suite(SUITE_MANIFEST)
    cubes = list(report.cubes)
    first = cubes[0]
    variants = list(first.cube.variants)
    variants[3] = dataclasses.replace(variants[3], content_digest="0" * 64)
    cubes[0] = dataclasses.replace(
        first, cube=dataclasses.replace(first.cube, variants=tuple(variants))
    )
    assert suite_content_digest(report.manifest, cubes) != report.content_digest


def test_the_payload_covers_the_variant_order_within_a_cube():
    report = validate_suite(SUITE_MANIFEST)
    first = report.cubes[0]
    reordered = tuple(reversed(first.cube.variants))
    cubes = [
        dataclasses.replace(
            first, cube=dataclasses.replace(first.cube, variants=reordered)
        ),
        *report.cubes[1:],
    ]
    assert suite_content_digest(report.manifest, cubes) != report.content_digest


def test_the_payload_covers_each_variant_id_explicitly():
    """A variant id is part of the collection's identity in its own right.

    Relying on the id being inside its own ``content_digest`` is a transitive
    argument that only holds for variants the compiler produced. A directly
    constructed variant — a forged or hand-patched artefact — can carry a new
    ``variant_id`` beside a stale ``content_digest``, and the collection-level
    contract has to name both for the suite's identity to move.
    """
    report = validate_suite(SUITE_MANIFEST)
    first = report.cubes[0]
    variants = list(first.cube.variants)
    forged = dataclasses.replace(variants[2], variant_id="forged_variant_id")
    assert forged.content_digest == variants[2].content_digest
    variants[2] = forged
    cubes = [
        dataclasses.replace(
            first, cube=dataclasses.replace(first.cube, variants=tuple(variants))
        ),
        *report.cubes[1:],
    ]
    assert suite_content_digest(report.manifest, cubes) != report.content_digest


# -- the pin does not freeze formatting --------------------------------------


def test_reflowing_the_suite_yaml_keeps_the_pin_valid(suite_dir):
    """Comments and whitespace are not identity; the pin must survive a reflow."""
    payload = suite_dict()
    payload["disclaimer"] = "  ".join(payload["disclaimer"].split()) + "\n"
    report = validate_suite(rewritten(suite_dir, payload))
    assert report.content_digest == COMMITTED_SUITE_DIGEST
