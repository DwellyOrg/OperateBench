"""CLI gates for semantic manifests and the methodology-spike suite.

All three commands are meant to be wired straight into CI, so every failure path
must exit non-zero with a readable message on stderr and no traceback, and the
``--json`` payloads must be stable enough to assert on.
"""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from boundarybench.cli import main
from tests.conftest import (
    ACCESS_CONSENT_CARD,
    ACCESS_CONSENT_MANIFEST,
    EXAMPLE_CARD,
    EXAMPLE_MANIFEST,
    SUITE_MANIFEST,
)
from tests.test_suite import downgrade_member_privacy

SHIPPED = [
    pytest.param(EXAMPLE_CARD, EXAMPLE_MANIFEST, id="maintenance_authority"),
    pytest.param(ACCESS_CONSENT_CARD, ACCESS_CONSENT_MANIFEST, id="access_consent"),
]


def suite_path(suite_dir):
    return str(suite_dir / "methodology_spike_suite.yaml")


def rewrite_suite(suite_dir, mutate):
    path = suite_dir / "methodology_spike_suite.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)


# -- verify-manifest ---------------------------------------------------------


@pytest.mark.parametrize(("card", "manifest"), SHIPPED)
def test_verify_manifest_accepts_every_shipped_pair(capsys, card, manifest):
    assert main(["verify-manifest", str(card), str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "manifest OK" in out
    assert card.name in out


@pytest.mark.parametrize(("card", "manifest"), SHIPPED)
def test_verify_manifest_json_is_machine_readable(capsys, card, manifest):
    assert main(["verify-manifest", str(card), str(manifest), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["card"] == card.name
    assert payload["manifest"] == manifest.name
    assert payload["variant_count"] == 6
    assert len(payload["variants"]) == 6
    assert payload["manifest_version"] == 1
    assert len(payload["card_fingerprint_sha256"]) == 64


def test_verify_manifest_states_its_synthetic_scope(capsys):
    main(["verify-manifest", str(EXAMPLE_CARD), str(EXAMPLE_MANIFEST)])
    out = capsys.readouterr().out.lower()
    assert "synthetic" in out
    assert "not a model benchmark" in out


def test_verify_manifest_reports_both_cube_shapes(capsys):
    main(["verify-manifest", str(EXAMPLE_CARD), str(EXAMPLE_MANIFEST)])
    assert "1-ACT" in capsys.readouterr().out
    main(["verify-manifest", str(ACCESS_CONSENT_CARD), str(ACCESS_CONSENT_MANIFEST)])
    out = capsys.readouterr().out
    assert "3-ACT" in out
    assert "initial_state" in out


def test_verify_manifest_rejects_a_swapped_pair(capsys):
    assert main(["verify-manifest", str(EXAMPLE_CARD), str(ACCESS_CONSENT_MANIFEST)]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()
    assert captured.out == ""


def test_verify_manifest_rejects_a_stale_fingerprint(capsys, tmp_path):
    payload = json.loads(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    payload["card_fingerprint_sha256"] = "0" * 64
    stale = tmp_path / "maintenance_authority.manifest.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["verify-manifest", str(EXAMPLE_CARD), str(stale)]) == 1
    captured = capsys.readouterr()
    assert "fingerprint" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("not_json", "definitely not json"),
        ("duplicate_key", '{"cube_id": "a", "cube_id": "b"}'),
        ("wrong_type", '{"manifest_version": "one"}'),
        ("empty", ""),
    ],
)
def test_verify_manifest_rejects_a_malformed_manifest(capsys, tmp_path, name, text):
    path = tmp_path / f"{name}.manifest.json"
    path.write_text(text, encoding="utf-8")
    assert main(["verify-manifest", str(EXAMPLE_CARD), str(path)]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()
    assert captured.out == ""


def test_verify_manifest_rejects_a_missing_manifest(capsys, tmp_path):
    assert main(["verify-manifest", str(EXAMPLE_CARD), str(tmp_path / "nope.json")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_verify_manifest_rejects_a_missing_card(capsys, tmp_path):
    args = ["verify-manifest", str(tmp_path / "nope.yaml"), str(EXAMPLE_MANIFEST)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()


# -- validate-suite ----------------------------------------------------------


def test_validate_suite_accepts_the_shipped_suite(capsys):
    assert main(["validate-suite", str(SUITE_MANIFEST)]) == 0
    out = capsys.readouterr().out
    assert "lettings_methodology_spike" in out
    assert "suite OK" in out


def test_validate_suite_reports_the_collection_shape(capsys):
    main(["validate-suite", str(SUITE_MANIFEST)])
    out = capsys.readouterr().out
    assert "1-ACT" in out and "3-ACT" in out
    assert "elicited" in out and "initial_state" in out
    assert "lettings_maintenance_authority_v1" in out
    assert "lettings_access_consent_v1" in out
    assert "12" in out


def test_validate_suite_shows_the_degenerate_baselines_without_calling_them_a_floor(
    capsys,
):
    main(["validate-suite", str(SUITE_MANIFEST)])
    out = capsys.readouterr().out
    assert "always_act" in out
    assert "always_stop" in out
    assert "degenerate" in out.lower()
    assert "autonomy floor" not in out.lower()


def test_validate_suite_states_its_synthetic_methodology_spike_scope(capsys):
    main(["validate-suite", str(SUITE_MANIFEST)])
    out = capsys.readouterr().out.lower()
    assert "methodology_spike" in out or "methodology spike" in out
    assert "synthetic" in out
    assert "not a model benchmark" in out


def test_validate_suite_json_is_machine_readable(capsys):
    assert main(["validate-suite", str(SUITE_MANIFEST), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["cube_count"] == 2
    assert payload["variant_count"] == 12
    assert payload["cube_types"] == {"1-ACT": 1, "3-ACT": 1}
    assert payload["fact_locations"] == {"elicited": 1, "initial_state": 1}
    assert payload["expected_dispositions"] == {"ACT": 6, "STOP": 6}
    assert payload["constant_strategy_scores"] == {"always_act": 6, "always_stop": 6}
    assert payload["degenerate_strategies_correct"] is True
    assert len(payload["suite_content_digest_sha256"]) == 64
    assert [c["cube_id"] for c in payload["cubes"]] == [
        "lettings_maintenance_authority_v1",
        "lettings_access_consent_v1",
    ]


def test_validate_suite_json_is_stable_across_runs(capsys):
    main(["validate-suite", str(SUITE_MANIFEST), "--json"])
    first = capsys.readouterr().out
    main(["validate-suite", str(SUITE_MANIFEST), "--json"])
    assert capsys.readouterr().out == first


def test_validate_suite_rejects_a_missing_suite(capsys, tmp_path):
    assert main(["validate-suite", str(tmp_path / "nope.yaml")]) == 1
    captured = capsys.readouterr()
    assert "cannot read" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    ("label", "mutation"),
    [
        ("dropped_cube", lambda p: p.__setitem__("entries", p["entries"][:1])),
        (
            "wrong_type_counts",
            lambda p: p["constraints"].__setitem__(
                "cube_types", {"1-ACT": 2, "3-ACT": 0}
            ),
        ),
        (
            "wrong_disposition_balance",
            lambda p: p["constraints"].__setitem__(
                "expected_dispositions", {"ACT": 8, "STOP": 4}
            ),
        ),
        (
            "escaping_path",
            lambda p: p["entries"][0].__setitem__(
                "card", "../maintenance_authority.yaml"
            ),
        ),
        (
            "missing_artefact",
            lambda p: p["entries"][0].__setitem__("card", "not_here.yaml"),
        ),
        ("bad_version", lambda p: p.__setitem__("benchmark_version", "1.0.0")),
        ("bad_status", lambda p: p.__setitem__("status", "RELEASED")),
        (
            "duplicate_cube_id",
            lambda p: p["entries"][1].__setitem__("cube_id", p["entries"][0]["cube_id"]),
        ),
        ("malformed_pin", lambda p: p.__setitem__("suite_content_digest", "nope")),
        ("stale_pin", lambda p: p.__setitem__("suite_content_digest", "0" * 64)),
    ],
)
def test_validate_suite_exits_one_on_every_contract_failure(
    capsys, suite_dir, label, mutation
):
    path = rewrite_suite(suite_dir, mutation)
    assert main(["validate-suite", path]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.err.strip()
    assert captured.out == ""


def test_validate_suite_exits_one_on_a_json_run_too(capsys, suite_dir):
    path = rewrite_suite(suite_dir, lambda p: p.__setitem__("entries", p["entries"][:1]))
    assert main(["validate-suite", path, "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_validate_suite_exits_one_on_a_symlinked_artefact(capsys, suite_dir, tmp_path):
    outside = tmp_path / "outside.yaml"
    shutil.copy(EXAMPLE_CARD, outside)
    (suite_dir / "linked.yaml").symlink_to(outside)
    path = rewrite_suite(
        suite_dir, lambda p: p["entries"][0].__setitem__("card", "linked.yaml")
    )
    assert main(["validate-suite", path]) == 1
    assert "symlink" in capsys.readouterr().err


@pytest.mark.parametrize("artefact", ["methodology_spike_suite.yaml"])
def test_validate_suite_exits_one_on_a_binary_suite_file(capsys, suite_dir, artefact):
    (suite_dir / artefact).write_bytes(b"\x80\x81\x82")
    assert main(["validate-suite", suite_path(suite_dir)]) == 1
    captured = capsys.readouterr()
    assert "not valid UTF-8" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "artefact", ["access_consent.yaml", "access_consent.manifest.json"]
)
def test_validate_suite_exits_one_on_a_binary_member_artefact(
    capsys, suite_dir, artefact
):
    (suite_dir / artefact).write_bytes(b"\xff\xfe\x00binary")
    assert main(["validate-suite", suite_path(suite_dir)]) == 1
    captured = capsys.readouterr()
    assert "not valid UTF-8" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_verify_manifest_exits_one_on_a_binary_manifest(capsys, tmp_path):
    binary = tmp_path / "maintenance_authority.manifest.json"
    binary.write_bytes(b"\x80\x81\x82")
    assert main(["verify-manifest", str(EXAMPLE_CARD), str(binary)]) == 1
    captured = capsys.readouterr()
    assert "not valid UTF-8" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_validate_suite_exits_one_on_an_overstated_privacy_claim(capsys, suite_dir):
    """A SYNTHETIC_ONLY suite carrying a SYNTHETIC_POLICY card fails cleanly."""
    path = downgrade_member_privacy(suite_dir)
    assert main(["validate-suite", str(path)]) == 1
    captured = capsys.readouterr()
    assert "privacy_status" in captured.err
    assert "SYNTHETIC_POLICY" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_check_suite_exits_one_on_an_overstated_privacy_claim(capsys, suite_dir):
    path = downgrade_member_privacy(suite_dir)
    assert main(["check-suite", str(path), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_validate_suite_exits_one_on_a_stale_semantic_manifest(capsys, suite_dir):
    manifest_path = suite_dir / "access_consent.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["variants"][0]["content_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["validate-suite", suite_path(suite_dir)]) == 1
    captured = capsys.readouterr()
    assert "content digest" in captured.err
    assert "Traceback" not in captured.err


# -- check-suite -------------------------------------------------------------


def test_check_suite_runs_the_causal_gate_for_every_cube(capsys):
    assert main(["check-suite", str(SUITE_MANIFEST)]) == 0
    out = capsys.readouterr().out
    assert out.count("causal CI") >= 2
    assert "lettings_maintenance_authority_v1" in out
    assert "lettings_access_consent_v1" in out
    assert "reference" in out
    assert "suite causal CI: OK" in out


def test_check_suite_json_reports_every_cube_and_solver(capsys):
    assert main(["check-suite", str(SUITE_MANIFEST), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["suite"]["variant_count"] == 12
    assert len(payload["causal_ci"]) == 2
    for entry in payload["causal_ci"]:
        assert entry["ok"] is True
        assert len(entry["outcomes"]) == 6
        reference = next(o for o in entry["outcomes"] if o["name"] == "reference")
        assert reference["passed_variants"] == 6
        assert reference["failures"] == []


def test_check_suite_also_reports_the_suite_summary(capsys):
    main(["check-suite", str(SUITE_MANIFEST)])
    out = capsys.readouterr().out
    assert "12" in out
    assert "degenerate" in out.lower()
    assert "not a model benchmark" in out.lower()


def test_check_suite_exits_one_when_validation_fails(capsys, suite_dir):
    path = rewrite_suite(suite_dir, lambda p: p.__setitem__("entries", p["entries"][:1]))
    assert main(["check-suite", path]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert captured.out == ""


def _failing_check_suite(report):
    """A check_suite whose first solver outcome reports a failure on every Cube."""
    from boundarybench.solvers import SolverOutcome, SolverReport, check_solvers
    from boundarybench.suite import SuiteCausalResult, SuiteCheckReport

    results = []
    for cube in report.cubes:
        solver_report = check_solvers(cube.cube)
        first = solver_report.outcomes[0]
        results.append(
            SuiteCausalResult(
                cube_id=cube.cube_id,
                card=cube.card,
                report=SolverReport(
                    cube_id=solver_report.cube_id,
                    outcomes=(
                        SolverOutcome(
                            name=first.name,
                            kind=first.kind,
                            description=first.description,
                            evaluations=first.evaluations,
                            passed_variants=first.passed_variants,
                            total_variants=first.total_variants,
                            satisfied=False,
                            failures=("synthetic causal failure",),
                        ),
                        *solver_report.outcomes[1:],
                    ),
                ),
            )
        )
    return SuiteCheckReport(suite=report, results=tuple(results))


def test_check_suite_exits_one_when_a_causal_gate_fails(capsys, monkeypatch):
    import boundarybench.cli as cli_module

    monkeypatch.setattr(cli_module, "check_suite", _failing_check_suite)
    assert main(["check-suite", str(SUITE_MANIFEST)]) == 1
    out = capsys.readouterr().out
    assert "synthetic causal failure" in out
    assert "suite causal CI: FAILED" in out


def test_check_suite_json_exits_one_when_a_causal_gate_fails(capsys, monkeypatch):
    import boundarybench.cli as cli_module

    monkeypatch.setattr(cli_module, "check_suite", _failing_check_suite)
    assert main(["check-suite", str(SUITE_MANIFEST), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any(not entry["ok"] for entry in payload["causal_ci"])


# -- the suite identity pin --------------------------------------------------


def test_validate_suite_reports_the_pinned_digest_it_verified(capsys):
    """The printed digest is now a verified pin, not an after-the-fact report."""
    assert main(["validate-suite", str(SUITE_MANIFEST)]) == 0
    declared = yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))
    assert declared["suite_content_digest"] in capsys.readouterr().out


@pytest.mark.parametrize("command", ["validate-suite", "check-suite"])
@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_a_stale_suite_pin_exits_non_zero_cleanly(capsys, suite_dir, command, as_json):
    """The CI probe: a drifted collection identity fails, with no traceback."""
    path = rewrite_suite(
        suite_dir, lambda p: p.__setitem__("suite_content_digest", "0" * 64)
    )
    argv = [command, path] + (["--json"] if as_json else [])
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    # Both values, so an operator can act without rerunning anything. Asserting
    # the digests rather than the wording keeps the prose free to change.
    declared = yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))
    assert "0" * 64 in captured.err
    assert declared["suite_content_digest"] in captured.err
