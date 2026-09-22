"""The suite manifest is the collection-level contract.

A single Cube proves the machinery runs. A suite proves it generalises: the
declared constraints below (one `1-ACT` and one `3-ACT` shape, one `elicited`
and one `initial_state` fact location, a 6/6 ACT/STOP split across all twelve
variants) are enforced, not reported, so the collection cannot quietly drift
into a single answer pattern.

Path handling is treated as untrusted: a suite file names sibling artefacts and
must not be able to reach anything else on the filesystem.
"""

from __future__ import annotations

import copy
import json
import shutil

import pytest
import yaml

from boundarybench.loader import load_card
from boundarybench.suite import (
    BENCHMARK_VERSION,
    SUITE_SCHEMA_VERSION,
    SuiteConstraintError,
    SuiteCubeError,
    SuiteDigestError,
    SuiteError,
    SuiteFormatError,
    SuiteManifestError,
    SuitePathError,
    card_fingerprint,
    check_suite,
    load_suite,
    validate_suite,
)
from tests.conftest import EXAMPLES, SUITE_MANIFEST


def suite_dict() -> dict:
    return yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))


def rewritten(suite_dir, payload, *, name="methodology_spike_suite.yaml"):
    path = suite_dir / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


#: A well-formed pin that is almost certainly not the real digest, used to make
#: validation report the digest it actually computes.
UNPINNED = "0" * 64


def repinned(suite_dir, payload, *, name="methodology_spike_suite.yaml"):
    """Write a mutated suite and re-pin its declared digest, as a revision must.

    A suite that changed meaning has to be re-pinned in the same reviewed commit,
    so a test that mutates one has to do the same — otherwise every mutation
    fails on the identity gate before it reaches the behaviour under test. The
    new digest is taken from the mismatch the gate itself reports, which is the
    same value a maintainer would read off a failed ``validate-suite`` run.
    """
    payload = {**payload, "suite_content_digest": UNPINNED}
    path = rewritten(suite_dir, payload, name=name)
    try:
        validate_suite(path)
    except SuiteDigestError as exc:
        payload = {**payload, "suite_content_digest": exc.computed}
        return rewritten(suite_dir, payload, name=name)
    return path


def downgrade_member_privacy(suite_dir, card_name="access_consent.yaml"):
    """Make one member card ``SYNTHETIC_POLICY``, re-pinning its manifest.

    ``privacy_status`` is not part of a variant's content digest, so only the
    card fingerprint moves. Re-pinning it isolates the privacy claim from the
    stale-manifest gate, which would otherwise fire first and hide it.
    """
    card_path = suite_dir / card_name
    original = card_path.read_text(encoding="utf-8")
    assert original.count("privacy_status: SYNTHETIC_ONLY") == 1
    card_path.write_text(
        original.replace(
            "privacy_status: SYNTHETIC_ONLY", "privacy_status: SYNTHETIC_POLICY"
        ),
        encoding="utf-8",
    )
    manifest_path = suite_dir / f"{card_path.stem}.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["card_fingerprint_sha256"] = card_fingerprint(load_card(card_path))
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return suite_dir / "methodology_spike_suite.yaml"


# -- the shipped suite -------------------------------------------------------


def test_the_shipped_suite_loads():
    suite = load_suite(SUITE_MANIFEST)
    assert suite.schema_version == SUITE_SCHEMA_VERSION
    assert suite.suite_id == "lettings_methodology_spike"
    assert suite.benchmark_version == BENCHMARK_VERSION
    assert suite.status == "METHODOLOGY_SPIKE"
    assert suite.privacy_status == "SYNTHETIC_ONLY"
    assert [entry.cube_id for entry in suite.entries] == [
        "lettings_maintenance_authority_v1",
        "lettings_access_consent_v1",
    ]


def test_the_benchmark_version_is_an_explicit_prerelease():
    """No stable version number until there is something stable to claim."""
    assert "-" in BENCHMARK_VERSION
    assert BENCHMARK_VERSION.startswith("0.")
    assert load_suite(SUITE_MANIFEST).benchmark_version == BENCHMARK_VERSION


def test_the_suite_disclaims_dwelly_policy_and_production_data():
    disclaimer = load_suite(SUITE_MANIFEST).disclaimer.lower()
    assert "synthetic" in disclaimer
    assert "not official dwelly policy" in disclaimer
    assert "methodology" in disclaimer


def test_the_shipped_suite_validates():
    report = validate_suite(SUITE_MANIFEST)
    assert report.ok is True
    assert report.cube_count == 2
    assert report.variant_count == 12


def test_the_suite_reports_complementary_cube_shapes():
    report = validate_suite(SUITE_MANIFEST)
    assert report.cube_types == {"1-ACT": 1, "3-ACT": 1}
    assert report.fact_locations == {"elicited": 1, "initial_state": 1}


def test_the_suite_dispositions_balance_six_and_six():
    report = validate_suite(SUITE_MANIFEST)
    assert report.expected_dispositions == {"ACT": 6, "STOP": 6}


def test_the_degenerate_strategies_score_half_the_suite():
    report = validate_suite(SUITE_MANIFEST)
    assert report.constant_strategy_scores == {"always_act": 6, "always_stop": 6}
    assert report.degenerate_strategies_correct is True
    payload = report.as_dict()
    # Reported as a degenerate baseline, never as a floor on autonomy.
    assert "degenerate" in payload["constant_strategy_note"].lower()
    assert "autonomy floor" not in payload["constant_strategy_note"].lower()


def test_the_per_cube_report_names_each_shape_and_fingerprint():
    report = validate_suite(SUITE_MANIFEST)
    by_id = {cube.cube_id: cube for cube in report.cubes}
    maintenance = by_id["lettings_maintenance_authority_v1"]
    access = by_id["lettings_access_consent_v1"]
    assert (maintenance.cube_type, maintenance.fact_location) == ("1-ACT", "elicited")
    assert (access.cube_type, access.fact_location) == ("3-ACT", "initial_state")
    assert maintenance.expected_dispositions == {"ACT": 2, "STOP": 4}
    assert access.expected_dispositions == {"ACT": 4, "STOP": 2}
    assert len(maintenance.card_fingerprint_sha256) == 64
    assert maintenance.card_fingerprint_sha256 != access.card_fingerprint_sha256


def test_every_variant_id_and_digest_is_unique_across_the_suite():
    report = validate_suite(SUITE_MANIFEST)
    ids = [v.variant_id for cube in report.cubes for v in cube.cube.variants]
    digests = [v.content_digest for cube in report.cubes for v in cube.cube.variants]
    assert len(set(ids)) == len(ids) == 12
    assert len(set(digests)) == len(digests) == 12


def test_the_report_states_its_synthetic_methodology_spike_scope():
    payload = validate_suite(SUITE_MANIFEST).as_dict()
    assert payload["status"] == "METHODOLOGY_SPIKE"
    assert payload["privacy_status"] == "SYNTHETIC_ONLY"
    assert "synthetic" in payload["scope"].lower()
    assert "not a model benchmark" in payload["scope"].lower()


# -- the suite content digest ------------------------------------------------


def test_the_suite_digest_is_stable_and_covers_the_cubes():
    first = validate_suite(SUITE_MANIFEST).content_digest
    second = validate_suite(SUITE_MANIFEST).content_digest
    assert first == second
    assert len(first) == 64


def test_reflowing_the_suite_yaml_does_not_change_the_digest(suite_dir):
    baseline = validate_suite(SUITE_MANIFEST).content_digest
    payload = suite_dict()
    payload["disclaimer"] = "  ".join(payload["disclaimer"].split()) + "\n"
    assert validate_suite(rewritten(suite_dir, payload)).content_digest == baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suite_id", "lettings_methodology_spike_b"),
        ("benchmark_version", "0.2.0-dev.2"),
        ("disclaimer", "A different synthetic disclaimer. Not official Dwelly policy."),
    ],
)
def test_changing_suite_metadata_changes_the_digest(suite_dir, field, value):
    baseline = validate_suite(SUITE_MANIFEST).content_digest
    payload = suite_dict()
    payload[field] = value
    assert validate_suite(repinned(suite_dir, payload)).content_digest != baseline


def test_changing_a_card_changes_the_suite_digest(suite_dir):
    baseline = validate_suite(SUITE_MANIFEST).content_digest
    card_path = suite_dir / "access_consent.yaml"
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card["distractors"][0]["value"] = "SYN-PROP-9999"
    card_path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")

    # The manifest must move with it, or verification fails before the digest.
    with pytest.raises(SuiteError):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")

    import json

    from boundarybench.compiler import compile_cube
    from boundarybench.loader import load_card
    from boundarybench.manifest import MANIFEST_VERSION, card_fingerprint

    reloaded = load_card(card_path)
    cube = compile_cube(reloaded)
    (suite_dir / "access_consent.manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "card": "access_consent.yaml",
                "cube_id": reloaded.cube_id,
                "card_fingerprint_sha256": card_fingerprint(reloaded),
                "variants": [
                    {"variant_id": v.variant_id, "content_digest": v.content_digest}
                    for v in cube.variants
                ],
            }
        ),
        encoding="utf-8",
    )
    # The card's meaning moved, so the suite's own pinned identity is now stale.
    with pytest.raises(SuiteDigestError):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")

    updated = validate_suite(repinned(suite_dir, suite_dict()))
    assert updated.ok is True
    assert updated.content_digest != baseline


# -- format strictness -------------------------------------------------------


def test_a_missing_suite_file_is_a_named_error(tmp_path):
    with pytest.raises(SuiteFormatError, match="cannot read"):
        load_suite(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", ["", "# nothing\n", "- one\n", "a string\n"])
def test_a_suite_that_is_not_a_yaml_mapping_is_rejected(suite_dir, text):
    with pytest.raises(SuiteFormatError):
        load_suite(rewritten(suite_dir, text))


def test_invalid_yaml_is_rejected(suite_dir):
    with pytest.raises(SuiteFormatError, match="not valid YAML"):
        load_suite(rewritten(suite_dir, "suite_id: [unclosed\n"))


def test_a_binary_suite_file_is_a_named_error_not_a_unicode_traceback(suite_dir):
    """A suite manifest that is not UTF-8 text is malformed input, not a crash."""
    path = suite_dir / "methodology_spike_suite.yaml"
    path.write_bytes(b"suite_id: \xff\xfe\n")
    with pytest.raises(SuiteFormatError, match="not valid UTF-8"):
        load_suite(path)


def test_a_binary_member_card_is_a_named_suite_error(suite_dir):
    """A member artefact that will not decode fails through the suite family."""
    (suite_dir / "access_consent.yaml").write_bytes(b"\x80\x81\x82")
    with pytest.raises(SuiteCubeError, match="not valid UTF-8"):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")


def test_a_binary_member_manifest_is_a_named_suite_error(suite_dir):
    (suite_dir / "access_consent.manifest.json").write_bytes(b"\x80\x81\x82")
    with pytest.raises(SuiteManifestError, match="not valid UTF-8"):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")


def test_duplicate_yaml_keys_are_rejected(suite_dir):
    text = SUITE_MANIFEST.read_text(encoding="utf-8") + "\nsuite_id: sneaky\n"
    with pytest.raises(SuiteFormatError, match="duplicate"):
        load_suite(rewritten(suite_dir, text))


def test_duplicate_yaml_keys_are_rejected_inside_an_entry(suite_dir):
    payload = suite_dict()
    text = yaml.safe_dump(payload, sort_keys=False).replace(
        "- cube_id:", "- cube_id: sneaky\n    cube_id:", 1
    )
    with pytest.raises(SuiteFormatError, match="duplicate"):
        load_suite(rewritten(suite_dir, text))


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "suite_id",
        "benchmark_version",
        "status",
        "privacy_status",
        "disclaimer",
        "suite_content_digest",
        "constraints",
        "entries",
    ],
)
def test_a_missing_required_field_is_rejected(suite_dir, field):
    payload = suite_dict()
    del payload[field]
    with pytest.raises(SuiteFormatError, match=field):
        load_suite(rewritten(suite_dir, payload))


def test_an_unknown_top_level_field_is_rejected(suite_dir):
    payload = suite_dict()
    payload["maintainer"] = "nobody"
    with pytest.raises(SuiteFormatError, match="maintainer"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("key", [1, True, 2.5, None])
def test_a_non_string_top_level_key_is_rejected(suite_dir, key):
    """YAML types its scalars, so `1:` is an integer key, not the string '1'."""
    payload = suite_dict()
    payload[key] = "whatever this was meant to say"
    with pytest.raises(SuiteFormatError, match="must use string keys"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "value", ["Lettings_Spike", "-leading-hyphen", "spike!", "spike suite", "_under"]
)
def test_a_malformed_suite_id_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["suite_id"] = value
    with pytest.raises(SuiteFormatError, match="suite_id"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", [2, 0, "1", 1.0, True, None])
def test_an_unsupported_schema_version_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["schema_version"] = value
    with pytest.raises(SuiteFormatError, match="schema_version"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", ["0.2.0", "1.0.0", "v0.2.0-dev.1", "dev", "0.2-dev.1"])
def test_a_benchmark_version_without_an_explicit_prerelease_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["benchmark_version"] = value
    with pytest.raises(SuiteFormatError, match="benchmark_version"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", ["RELEASED", "methodology_spike", "", 7])
def test_an_unsupported_status_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["status"] = value
    with pytest.raises(SuiteFormatError, match="status"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", ["PRODUCTION", "SYNTHETIC_POLICY", ""])
def test_a_non_synthetic_privacy_status_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["privacy_status"] = value
    with pytest.raises(SuiteFormatError, match="privacy_status"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", [{}, [], "two", 7, None])
def test_entries_must_be_a_non_empty_list(suite_dir, value):
    payload = suite_dict()
    payload["entries"] = value
    with pytest.raises(SuiteFormatError, match="entries"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("field", ["cube_id", "card", "semantic_manifest"])
def test_an_entry_missing_a_field_is_rejected(suite_dir, field):
    payload = suite_dict()
    del payload["entries"][1][field]
    with pytest.raises(SuiteFormatError, match=field):
        load_suite(rewritten(suite_dir, payload))


def test_an_entry_with_an_unknown_field_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][0]["notes"] = "extra"
    with pytest.raises(SuiteFormatError, match="notes"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", ["a string", 7, None, []])
def test_an_entry_that_is_not_a_mapping_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["entries"][0] = value
    with pytest.raises(SuiteFormatError, match=r"entries\[0\]"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "section", ["cube_types", "fact_locations", "expected_dispositions"]
)
def test_a_missing_constraint_section_is_rejected(suite_dir, section):
    payload = suite_dict()
    del payload["constraints"][section]
    with pytest.raises(SuiteFormatError, match=section):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", [[], "cube_types", 7, None])
def test_constraints_must_themselves_be_a_mapping(suite_dir, value):
    payload = suite_dict()
    payload["constraints"] = value
    with pytest.raises(SuiteFormatError, match="constraints"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "section", ["cube_types", "fact_locations", "expected_dispositions"]
)
@pytest.mark.parametrize("value", [{}, [], "1-ACT", 7, None])
def test_a_constraint_section_must_be_a_non_empty_mapping(suite_dir, section, value):
    """An empty section declares nothing, which is not the same as declaring zero."""
    payload = suite_dict()
    payload["constraints"][section] = value
    with pytest.raises(SuiteFormatError, match=section):
        load_suite(rewritten(suite_dir, payload))


def test_an_unknown_constraint_section_is_rejected(suite_dir):
    payload = suite_dict()
    payload["constraints"]["vibes"] = {"good": 2}
    with pytest.raises(SuiteFormatError, match="vibes"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("cube_types", "2-ACT"),
        ("fact_locations", "runtime_emergent"),
        ("expected_dispositions", "ASK"),
    ],
)
def test_a_constraint_naming_an_unsupported_value_is_rejected(suite_dir, section, key):
    payload = suite_dict()
    payload["constraints"][section][key] = 1
    with pytest.raises(SuiteFormatError, match=key):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", [-1, "six", None, 1.0, True])
def test_a_constraint_count_must_be_a_non_negative_integer(suite_dir, value):
    payload = suite_dict()
    payload["constraints"]["cube_types"]["1-ACT"] = value
    with pytest.raises(SuiteFormatError, match="1-ACT"):
        load_suite(rewritten(suite_dir, payload))


# -- path safety -------------------------------------------------------------


@pytest.mark.parametrize("field", ["card", "semantic_manifest"])
def test_an_absolute_path_is_rejected(suite_dir, field):
    payload = suite_dict()
    payload["entries"][0][field] = str(EXAMPLES / "maintenance_authority.yaml")
    with pytest.raises(SuitePathError, match="relative"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "value", ["../examples/maintenance_authority.yaml", "sub/../../escape.yaml", ".."]
)
def test_a_path_that_climbs_out_of_the_suite_directory_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["entries"][0]["card"] = value
    with pytest.raises(SuitePathError, match=r"\.\."):
        load_suite(rewritten(suite_dir, payload))


def test_a_symlinked_artefact_is_rejected(suite_dir, tmp_path):
    outside = tmp_path / "outside.yaml"
    shutil.copy(EXAMPLES / "maintenance_authority.yaml", outside)
    link = suite_dir / "linked_card.yaml"
    link.symlink_to(outside)
    payload = suite_dict()
    payload["entries"][0]["card"] = "linked_card.yaml"
    with pytest.raises(SuitePathError, match="symlink"):
        load_suite(rewritten(suite_dir, payload))


def test_a_symlinked_parent_directory_is_rejected(suite_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy(EXAMPLES / "maintenance_authority.yaml", outside / "card.yaml")
    (suite_dir / "linked_dir").symlink_to(outside, target_is_directory=True)
    payload = suite_dict()
    payload["entries"][0]["card"] = "linked_dir/card.yaml"
    with pytest.raises(SuitePathError, match="symlink"):
        load_suite(rewritten(suite_dir, payload))


def test_a_missing_artefact_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][0]["card"] = "not_here.yaml"
    with pytest.raises(SuitePathError, match="does not exist"):
        load_suite(rewritten(suite_dir, payload))


def test_a_directory_where_a_file_is_expected_is_rejected(suite_dir):
    (suite_dir / "a_directory").mkdir()
    payload = suite_dict()
    payload["entries"][0]["card"] = "a_directory"
    with pytest.raises(SuitePathError, match="not a file"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "value", [" maintenance_authority.yaml", "maintenance_authority.yaml\t", "\ncard "]
)
def test_a_path_with_surrounding_whitespace_is_rejected(suite_dir, value):
    """Padding is invisible in review and would silently name a different file."""
    payload = suite_dict()
    payload["entries"][0]["card"] = value
    with pytest.raises(SuitePathError, match="whitespace"):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize("value", ["", "   ", "./card.yaml", "."])
def test_a_malformed_relative_path_is_rejected(suite_dir, value):
    payload = suite_dict()
    payload["entries"][0]["card"] = value
    with pytest.raises((SuitePathError, SuiteFormatError)):
        load_suite(rewritten(suite_dir, payload))


@pytest.mark.parametrize(
    "value",
    [
        "./maintenance_authority.yaml",
        "maintenance_authority.yaml/",
        "nested//maintenance_authority.yaml",
        "nested/./maintenance_authority.yaml",
    ],
)
def test_a_non_canonical_spelling_of_a_real_artefact_is_rejected(suite_dir, value):
    """The path a reviewer reads must be the only spelling that resolves.

    ``./x.yaml``, ``x.yaml/`` and ``a//x.yaml`` all name the same file as
    ``x.yaml``, so admitting them would let one artefact enter a suite under
    several distinct strings — and the duplicate-card, duplicate-manifest and
    cross-Cube guards all compare the strings as authored.
    """
    nested = suite_dir / "nested"
    nested.mkdir()
    shutil.copy(suite_dir / "maintenance_authority.yaml", nested)
    payload = suite_dict()
    payload["entries"][0]["card"] = value
    with pytest.raises(SuitePathError, match="segment"):
        load_suite(rewritten(suite_dir, payload))


def test_a_subdirectory_artefact_is_accepted(suite_dir):
    """Relative subpaths are fine; only escapes are not."""
    nested = suite_dir / "nested"
    nested.mkdir()
    shutil.copy(suite_dir / "access_consent.yaml", nested / "access_consent.yaml")
    shutil.copy(
        suite_dir / "access_consent.manifest.json",
        nested / "access_consent.manifest.json",
    )
    payload = suite_dict()
    payload["entries"][1]["card"] = "nested/access_consent.yaml"
    payload["entries"][1]["semantic_manifest"] = "nested/access_consent.manifest.json"
    # The recorded artefact paths are part of the suite's identity, so moving one
    # into a subdirectory is a revision that has to be re-pinned.
    assert validate_suite(repinned(suite_dir, payload)).ok is True


# -- duplicates --------------------------------------------------------------


def test_a_duplicate_cube_id_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][1]["cube_id"] = payload["entries"][0]["cube_id"]
    with pytest.raises(SuiteFormatError, match="duplicate cube_id"):
        load_suite(rewritten(suite_dir, payload))


def test_a_duplicate_card_path_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][1]["card"] = payload["entries"][0]["card"]
    with pytest.raises(SuiteFormatError, match="duplicate card"):
        load_suite(rewritten(suite_dir, payload))


def test_a_duplicate_manifest_path_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][1]["semantic_manifest"] = payload["entries"][0][
        "semantic_manifest"
    ]
    with pytest.raises(SuiteFormatError, match="duplicate semantic_manifest"):
        load_suite(rewritten(suite_dir, payload))


def test_two_entries_pointing_at_copies_of_the_same_cube_are_rejected(suite_dir):
    """Distinct paths, same content: caught on variant ids, not on filenames."""
    shutil.copy(suite_dir / "access_consent.yaml", suite_dir / "copy.yaml")
    shutil.copy(
        suite_dir / "access_consent.manifest.json", suite_dir / "copy.manifest.json"
    )
    import json

    manifest_path = suite_dir / "copy.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["card"] = "copy.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = suite_dict()
    payload["entries"].append(
        {
            "cube_id": "lettings_access_consent_v1",
            "card": "copy.yaml",
            "semantic_manifest": "copy.manifest.json",
        }
    )
    with pytest.raises(SuiteFormatError, match="duplicate cube_id"):
        validate_suite(rewritten(suite_dir, payload))


def test_renaming_a_cube_makes_it_a_genuinely_distinct_semantic_unit(suite_dir):
    """The variant id is inside the content digest, so a clone shares neither.

    That is what makes the cross-Cube uniqueness sweep below a defence rather
    than the primary guard: a near-copy under a new id is a new semantic unit,
    and the thing that must never happen is two entries sharing an *identity*.
    """
    from boundarybench.compiler import compile_cube
    from boundarybench.loader import load_card

    card = yaml.safe_load((suite_dir / "access_consent.yaml").read_text("utf-8"))
    clone = copy.deepcopy(card)
    clone["cube_id"] = "lettings_access_consent_clone_v1"
    (suite_dir / "clone.yaml").write_text(
        yaml.safe_dump(clone, sort_keys=False), encoding="utf-8"
    )
    original = compile_cube(load_card(suite_dir / "access_consent.yaml"))
    cloned = compile_cube(load_card(suite_dir / "clone.yaml"))
    assert not {v.content_digest for v in original.variants} & {
        v.content_digest for v in cloned.variants
    }


def test_the_cross_cube_sweep_rejects_a_repeated_variant_id(suite_dir):
    from boundarybench.suite import _reject_cross_cube_duplicates

    report = validate_suite(SUITE_MANIFEST)
    first = report.cubes[0]
    with pytest.raises(SuiteFormatError, match="duplicate variant id"):
        _reject_cross_cube_duplicates([first, first])


def test_the_cross_cube_sweep_rejects_a_repeated_content_digest(suite_dir):
    """Distinct ids, identical digest: two entries describing the same task."""
    import dataclasses

    from boundarybench.suite import _reject_cross_cube_duplicates

    report = validate_suite(SUITE_MANIFEST)
    first = report.cubes[0]
    renamed = dataclasses.replace(
        first.cube,
        cube_id="lettings_renamed_v1",
        variants=tuple(
            dataclasses.replace(v, variant_id=v.variant_id.replace(v.cube_id, "renamed"))
            for v in first.cube.variants
        ),
    )
    second = dataclasses.replace(first, cube=renamed, card="renamed.yaml")
    with pytest.raises(SuiteFormatError, match="duplicate variant content digest"):
        _reject_cross_cube_duplicates([first, second])


# -- cross-artefact agreement ------------------------------------------------


def test_an_entry_cube_id_that_disagrees_with_the_card_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][0]["cube_id"] = "lettings_something_else_v1"
    with pytest.raises(SuiteConstraintError, match="cube_id"):
        validate_suite(rewritten(suite_dir, payload))


def test_a_swapped_semantic_manifest_is_rejected(suite_dir):
    payload = suite_dict()
    payload["entries"][0]["semantic_manifest"] = "access_consent.manifest.json"
    payload["entries"][1]["semantic_manifest"] = "maintenance_authority.manifest.json"
    with pytest.raises(SuiteError):
        validate_suite(rewritten(suite_dir, payload))


def test_a_stale_semantic_manifest_fails_the_suite(suite_dir):
    import json

    manifest_path = suite_dir / "access_consent.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["card_fingerprint_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SuiteError, match="fingerprint"):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")


def test_a_card_that_does_not_compile_fails_the_suite(suite_dir):
    from boundarybench.suite import SuiteCubeError

    card_path = suite_dir / "access_consent.yaml"
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card["cube_type"] = "1-ACT"
    card_path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
    with pytest.raises(SuiteCubeError, match="1 ACT cell"):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")


def test_a_card_that_does_not_even_parse_fails_the_suite(suite_dir):
    from boundarybench.suite import SuiteCubeError

    (suite_dir / "access_consent.yaml").write_text("cube_id: [\n", encoding="utf-8")
    with pytest.raises(SuiteCubeError, match="not valid YAML"):
        validate_suite(suite_dir / "methodology_spike_suite.yaml")


# -- declared constraints are enforced, not decorative -----------------------


def test_a_cube_type_count_that_does_not_match_reality_is_rejected(suite_dir):
    payload = suite_dict()
    payload["constraints"]["cube_types"] = {"1-ACT": 2, "3-ACT": 0}
    with pytest.raises(SuiteConstraintError, match="cube_types"):
        validate_suite(repinned(suite_dir, payload))


def test_a_fact_location_count_that_does_not_match_reality_is_rejected(suite_dir):
    payload = suite_dict()
    payload["constraints"]["fact_locations"] = {"elicited": 2, "initial_state": 0}
    with pytest.raises(SuiteConstraintError, match="fact_locations"):
        validate_suite(repinned(suite_dir, payload))


def test_a_disposition_balance_that_does_not_match_reality_is_rejected(suite_dir):
    payload = suite_dict()
    payload["constraints"]["expected_dispositions"] = {"ACT": 7, "STOP": 5}
    with pytest.raises(SuiteConstraintError, match="expected_dispositions"):
        validate_suite(repinned(suite_dir, payload))


def test_dropping_a_cube_breaks_the_declared_constraints(suite_dir):
    payload = suite_dict()
    payload["entries"] = payload["entries"][:1]
    with pytest.raises(SuiteConstraintError):
        validate_suite(repinned(suite_dir, payload))


# -- the privacy claim is about the members, not just the suite file ---------


def test_the_shipped_suite_only_contains_synthetic_only_cards():
    report = validate_suite(SUITE_MANIFEST)
    assert report.manifest.privacy_status == "SYNTHETIC_ONLY"
    assert {cube.cube.card.privacy_status for cube in report.cubes} == {"SYNTHETIC_ONLY"}


def test_a_synthetic_only_suite_refuses_a_synthetic_policy_member(suite_dir):
    """The suite-level claim must not be stronger than any card in it.

    ``SYNTHETIC_POLICY`` is admissible for an individual Cube but is a weaker
    guarantee than ``SYNTHETIC_ONLY``, so a suite advertising the stronger status
    while carrying such a member overstates its own members.
    """
    path = downgrade_member_privacy(suite_dir)
    with pytest.raises(SuiteConstraintError) as excinfo:
        validate_suite(path)
    message = str(excinfo.value)
    assert "SYNTHETIC_ONLY" in message
    assert "SYNTHETIC_POLICY" in message
    assert "access_consent.yaml" in message


def test_the_overstated_privacy_claim_is_a_suite_error(suite_dir):
    """One exception family for a caller validating a collection."""
    path = downgrade_member_privacy(suite_dir, card_name="maintenance_authority.yaml")
    with pytest.raises(SuiteError):
        validate_suite(path)


# -- causal CI over the whole suite ------------------------------------------


def test_check_suite_runs_the_causal_gate_for_every_cube():
    report = check_suite(validate_suite(SUITE_MANIFEST))
    assert report.ok is True
    assert [r.cube_id for r in report.results] == [
        "lettings_maintenance_authority_v1",
        "lettings_access_consent_v1",
    ]
    for result in report.results:
        assert result.report.ok
        assert result.report.outcome("reference").passed_variants == 6


def test_check_suite_json_is_machine_readable():
    payload = check_suite(validate_suite(SUITE_MANIFEST)).as_dict()
    assert payload["ok"] is True
    assert payload["suite"]["variant_count"] == 12
    assert len(payload["causal_ci"]) == 2
    for entry in payload["causal_ci"]:
        assert entry["ok"] is True
        assert len(entry["outcomes"]) == 6
        assert {o["name"] for o in entry["outcomes"]} >= {"reference", "wrong_evidence"}


def test_check_suite_reports_a_failing_cube_without_raising(suite_dir, monkeypatch):
    """A negative solver that stops biting must surface as a failed gate."""
    import boundarybench.suite as suite_module
    from boundarybench.solvers import SolverOutcome, SolverReport

    real = suite_module.check_solvers

    def broken(cube):
        report = real(cube)
        if cube.cube_id.endswith("access_consent_v1"):
            first = report.outcomes[0]
            return SolverReport(
                cube_id=report.cube_id,
                outcomes=(
                    SolverOutcome(
                        name=first.name,
                        kind=first.kind,
                        description=first.description,
                        evaluations=first.evaluations,
                        passed_variants=first.passed_variants,
                        total_variants=first.total_variants,
                        satisfied=False,
                        failures=("synthetic failure",),
                    ),
                    *report.outcomes[1:],
                ),
            )
        return report

    monkeypatch.setattr(suite_module, "check_solvers", broken)
    report = check_suite(validate_suite(SUITE_MANIFEST))
    assert report.ok is False
    assert report.as_dict()["ok"] is False
    failing = [entry for entry in report.as_dict()["causal_ci"] if not entry["ok"]]
    assert len(failing) == 1
    assert failing[0]["cube_id"] == "lettings_access_consent_v1"


def test_every_named_error_is_a_suite_error():
    from boundarybench.suite import SuiteCubeError, SuiteManifestError

    for error in (
        SuiteFormatError,
        SuitePathError,
        SuiteConstraintError,
        SuiteCubeError,
        SuiteManifestError,
    ):
        assert issubclass(error, SuiteError)
    assert issubclass(SuiteError, ValueError)
