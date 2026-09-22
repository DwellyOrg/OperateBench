"""Regression contracts for ownership, licensing, release controls, and DCO."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from tools import check_public_release as checks
from tools.check_dco import SIGN_OFF

ROOT = Path(__file__).resolve().parents[1]
CONTROL_FILES = {
    "PUBLICATION_MANIFEST.json",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "PUBLIC_CANDIDATE_VERIFICATION.md",
}
OWNER = "PROSPIRE TECHNOLOGIES LTD"


def test_owner_and_human_author_are_unambiguous() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert f"Copyright 2026 {OWNER}" in notice
    assert OWNER in citation
    assert "Dmitry Khanukov" in citation
    assert project["authors"] == [
        {"name": "Dmitry Khanukov", "email": "dmitry@dwelly.group"}
    ]
    assert project["maintainers"] == [
        {"name": "Dmitry Khanukov", "email": "dmitry@dwelly.group"}
    ]


def test_explicit_license_map_covers_imported_schemas_as_code() -> None:
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text(encoding="utf-8"))
    text = (ROOT / "LICENSE-DATA").read_text(encoding="utf-8")
    assert "Imported or executable schemas" in text
    assert "Apache-2.0" in text
    assert reuse["version"] == 1
    annotated = reuse["annotations"]
    assert any(
        "src/**" in item["path"] and item["SPDX-License-Identifier"] == "Apache-2.0"
        for item in annotated
    )
    assert any(
        "examples/**" in item["path"] and item["SPDX-License-Identifier"] == "CC-BY-4.0"
        for item in annotated
    )


def test_notice_is_apache_and_dco_is_not_a_project_content_license() -> None:
    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    annotations = reuse["annotations"]
    notice = next(item for item in annotations if item["path"] == "NOTICE")
    dco = next(item for item in annotations if item["path"] == "DCO")
    assert notice["SPDX-License-Identifier"] == "Apache-2.0"
    assert dco["SPDX-License-Identifier"] == "LicenseRef-DCO-1.1"
    assert "DCO" not in project["license-files"]
    assert not any("LicenseRef-DCO" in path for path in project["license-files"])


def test_wheel_metadata_contains_canonical_apache_and_cc_license_texts(
    tmp_path: Path,
) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert "LICENSES/Apache-2.0.txt" in project["license-files"]
    assert "LICENSES/CC-BY-4.0.txt" in project["license-files"]

    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True
    )
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as wheel:
        members = set(wheel.namelist())
    licenses = {
        member.split(".dist-info/licenses/", 1)[1]
        for member in members
        if ".dist-info/licenses/" in member
    }
    assert "LICENSES/Apache-2.0.txt" in licenses
    assert "LICENSES/CC-BY-4.0.txt" in licenses


def test_manifest_has_stable_posture_and_no_embedded_human_approvals() -> None:
    manifest = json.loads(
        (ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert "approvals" not in manifest
    serialized = json.dumps(manifest).lower()
    for forbidden in ("reviewer", "decided_at", "no-push-yet", "remote_configured"):
        assert forbidden not in serialized
    assert manifest["target"]["visibility"] == "private-prepublication"
    assert (
        manifest["human_gates_record_system"] == "controlled records outside public Git"
    )
    assert manifest["published_artifacts"] == []
    artifact_note = manifest["published_artifacts_note"].lower()
    assert "solely" in artifact_note
    assert "external controlled release record" in artifact_note
    assert "no post-review manifest edit" in artifact_note


def test_manifest_gate_ledger_matches_the_current_candidate_evidence() -> None:
    """Current technical evidence passes only the gates it actually completes."""
    manifest = json.loads(
        (ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    gates = manifest["gates"]
    assert gates, "a manifest with no gates claims nothing and proves nothing"
    expected_ids = {
        "G1_clean_history",
        "G2_package_build_and_install",
        "G3_cli_offline",
        "G4_causal_acceptance",
        "G5_deterministic_replay",
        "G6_offline_suite_and_quality_gates",
        "G7_secret_and_private_reference_scan",
        "G8_claims_scan",
        "G9_links",
        "G10_licensing",
        "G11_independent_human_gate",
        "G12_explicit_publication_approval",
    }
    assert {gate["id"] for gate in gates} == expected_ids
    assert len(gates) == len(expected_ids)
    expected_passed = {"G1", "G2", "G3", "G4", "G5", "G7", "G10"}
    for gate in gates:
        gate_id = gate["id"].split("_")[0]
        assert gate["status"] == ("passed" if gate_id in expected_passed else "pending")
        assert gate["evidence_candidate"] == "verified_base"
        assert (
            gate["evidence"]
            and gate["evidence"] != "Pending exact successor verification"
        )
        assert gate["blocking"] is True
    assert "successful hosted CI" in " ".join(manifest["notes"])
    acceptance = manifest["candidate_acceptance"]
    assert acceptance["status"] == "requires_exact_source_record"
    assert acceptance["hosted_ci"] == "pending"
    assert "Human requirements cannot be closed by CI" in acceptance["closure"]


def test_history_gate_separates_verified_base_from_candidate_additions() -> None:
    manifest = json.loads(
        (ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    source = manifest["source"]
    source_history = next(
        entry
        for entry in manifest["excluded_classes"]
        if entry["class"] == "source_history"
    )
    g1 = next(gate for gate in manifest["gates"] if gate["id"] == "G1_clean_history")

    assert g1["status"] == "passed"
    assert source["history_policy"] == (
        "Fresh history only after exact staged-blob privacy admission"
    )
    assert source["export_method"] == (
        "curated tracked-file archive; no predecessor Git objects"
    )
    assert "predecessor/internal source history" in source_history["description"].lower()
    assert "proposed public surface" in source_history["description"].lower()
    assert (
        "Later candidate commits and advertised refs require exact-source review"
        in source_history["reason"]
    )


def _assert_bounded_verification_claims(text: str) -> None:
    """Base acceptance, new hosted evidence and human authority are separate."""
    flowed = " ".join(text.lower().split())
    for required in (
        "**changed-candidate hosted ci: pending.**",
        "verified-base full suites completed on python 3.11 and 3.14",
        "not a claim that the changed candidate has completed its full matrix",
        "g8 full manual claims review and g9 complete external-link rechecking "
        "remain open",
        "independent ai review is not independent human review",
        "g11 independent human assessment and know-how/rights confirmation",
        "g12 explicit publication authority, remain mandatory",
        "ci cannot close human requirements",
        "it passed 11 checks",
        "reference agent on `v1`, `v2` and `v3`",
        "eight targeted negatives on `v1`",
        "each expected failure closure satisfied",
        "deterministic construct evidence, not model evidence",
        "statement is scoped to this exercised gate",
        "technical passes do not authorize a tag, release or package-index publication",
        "no rankings, paid calls, remote creation or public visibility changes "
        "are authorized by this document",
    ):
        assert required in flowed
    for forbidden in (
        "release approved",
        "release is approved",
        "publication approved",
        "publication is approved",
        "legal approval granted",
        "hosted ci: passed",
        "all gates passed",
    ):
        assert forbidden not in flowed


def test_current_verification_record_has_bounded_claims_and_no_stale_record() -> None:
    text = (ROOT / "PUBLIC_CANDIDATE_VERIFICATION.md").read_text(encoding="utf-8")
    _assert_bounded_verification_claims(text)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("hosted CI: pending", "hosted CI: passed"),
        (
            "Independent AI review is not independent\nhuman review",
            "Independent AI review is independent human review",
        ),
        ("remain mandatory", "are optional"),
        ("It passed 11 checks", "The full matrix passed 11 checks"),
        (
            "## Candidate acceptance boundary",
            "Release is approved.\n\n## Candidate acceptance boundary",
        ),
        ("package-index publication.", "package-index publication. Release approved."),
    ],
)
def test_verification_contract_rejects_overclaimed_approval(old: str, new: str) -> None:
    text = (ROOT / "PUBLIC_CANDIDATE_VERIFICATION.md").read_text(encoding="utf-8")
    assert text.count(old) == 1
    _assert_bounded_verification_claims(text)
    with pytest.raises(AssertionError):
        _assert_bounded_verification_claims(text.replace(old, new))


def test_release_checklist_summarises_the_current_gate_ledger() -> None:
    checklist = " ".join(
        (ROOT / "PUBLIC_RELEASE_CHECKLIST.md").read_text(encoding="utf-8").lower().split()
    )
    assert "scoped verified-base technical evidence" in checklist
    assert "the manifest's blocking gates are all `passed`" not in checklist
    assert "every blocking gate is closed for the exact candidate" in checklist
    assert "in the manifest or an exact-source controlled acceptance record" in checklist
    assert "ci cannot close the human requirements" in checklist
    assert "g11 and g12 remain human gates" in checklist
    assert "not legal or publication authorization" in checklist


def test_public_records_do_not_embed_the_final_commit_or_artifact_hashes() -> None:
    verification = (ROOT / "PUBLIC_CANDIDATE_VERIFICATION.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    source = manifest["source"]

    assert "git rev-parse HEAD" not in verification
    assert "final commit" not in verification.lower()
    assert "wheel sha256" not in verification.lower()
    assert "sdist sha256" not in verification.lower()
    assert source["release_commit_reference"] == (
        "the commit containing this manifest, resolved externally"
    )
    assert manifest["published_artifacts"] == []


def test_release_docs_treat_internal_storage_as_distinct_from_publication() -> None:
    checklist = (ROOT / "PUBLIC_RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    verification = (ROOT / "PUBLIC_CANDIDATE_VERIFICATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    combined = f"{checklist}\n{verification}\n{readme}".lower()
    assert "push" not in combined
    assert "internal storage is not publication" in combined
    assert "final public visibility" in combined


def test_licensing_candidate_gate_names_owner_authorization_not_brand_legal() -> None:
    checklist = (ROOT / "PUBLIC_RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    assert "PROSPIRE TECHNOLOGIES LTD legal authorization/counsel" in checklist
    assert " ".join(("Dwelly", "legal")) not in checklist


def test_sdist_excludes_release_control_files_but_checkout_keeps_them(
    tmp_path: Path,
) -> None:
    assert all((ROOT / name).is_file() for name in CONTROL_FILES)
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(dist)], cwd=ROOT, check=True
    )
    with tarfile.open(next(dist.glob("*.tar.gz")), "r:gz") as archive:
        members = {Path(name).name for name in archive.getnames()}
    assert CONTROL_FILES.isdisjoint(members)


def test_extracted_sdist_scanner_uses_package_surface_not_missing_controls(
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(dist)], cwd=ROOT, check=True
    )
    extracted = tmp_path / "src"
    with tarfile.open(next(dist.glob("*.tar.gz")), "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    package_root = next(extracted.iterdir())
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_public_release", "--surface", "sdist"],
        cwd=package_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_surface_requires_controls_even_with_forged_valid_pkg_info(
    tmp_path: Path,
) -> None:
    (tmp_path / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\nName: operatebench\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "src" / "operatebench").mkdir(parents=True)
    problems = checks.check_required_files(tmp_path, surface="source")
    assert any("PUBLICATION_MANIFEST.json" in problem for problem in problems)


def test_sdist_surface_explicitly_omits_source_controls(tmp_path: Path) -> None:
    problems = checks.check_required_files(tmp_path, surface="sdist")
    assert not any("PUBLICATION_MANIFEST.json" in problem for problem in problems)


def test_scanner_rejects_an_invalid_surface_as_usage_error() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_public_release", "--surface", "wheel"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_sdist_link_omission_is_only_for_the_exact_root_control_target(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "links.md").write_text(
        "[exact](../PUBLICATION_MANIFEST.json)\n"
        "[wrong path](missing/PUBLICATION_MANIFEST.json)\n"
        "[outside](../../PUBLICATION_MANIFEST.json)\n",
        encoding="utf-8",
    )
    problems = checks.check_relative_links(tmp_path, surface="sdist")
    assert len(problems) == 2
    assert any("missing/PUBLICATION_MANIFEST.json" in problem for problem in problems)
    assert any("../../PUBLICATION_MANIFEST.json" in problem for problem in problems)


def test_dco_policy_and_pr_only_gate() -> None:
    dco = (ROOT / "DCO").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/dco.yml").read_text(encoding="utf-8")
    assert "Developer Certificate of Origin\nVersion 1.1" in dco
    assert "Signed-off-by:" in contributing
    assert "git commit -s" in contributing
    assert "pull_request_target:" in workflow
    assert "push:" not in workflow
    assert "check_dco.py" in workflow
    assert "CLA" not in contributing


def test_dco_workflow_keeps_pr_code_outside_the_execution_boundary() -> None:
    workflow_path = ROOT / ".github/workflows/dco.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)

    assert set(workflow["on"]) == {"pull_request_target"}
    assert workflow["permissions"] == {"contents": "read"}
    assert "permissions" not in workflow["jobs"]["dco"]

    steps = workflow["jobs"]["dco"]["steps"]
    checkout_steps = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkout_steps) == 1
    checkout = checkout_steps[0]["with"]
    assert checkout["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["persist-credentials"] == "false"
    assert checkout["fetch-depth"] == "0"

    run_steps = [step for step in steps if "run" in step]
    assert len(run_steps) == 1
    check_step = run_steps[0]
    assert check_step["env"] == {
        "PR_NUMBER": "${{ github.event.pull_request.number }}",
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
        "GITHUB_TOKEN": "${{ github.token }}",
    }
    assert "secrets." not in workflow_text
    script = check_step["run"]
    assert '[[ "$PR_NUMBER" =~ ^[0-9]+$ ]]' in script
    assert script.count('[[ "$BASE_SHA" =~ ^[0-9a-fA-F]{40}$ ]]') == 1
    assert script.count('[[ "$HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]]') == 1
    fetch_position = script.index("refs/pull/${PR_NUMBER}/head")
    verification_position = script.index('[[ "$FETCHED_SHA" == "$HEAD_SHA" ]]')
    checker_position = script.index(
        'python tools/check_dco.py "${BASE_SHA}..${HEAD_SHA}"'
    )
    assert fetch_position < verification_position < checker_position
    assert "FETCHED_SHA=" in script
    assert "CHECKED_OUT_BASE=" in script
    assert '[[ "$CHECKED_OUT_BASE" == "$BASE_SHA" ]]' in script

    # Pull-request contents may be fetched as objects, but are never checked out or run.
    assert "github.event.pull_request.head.ref" not in workflow_text
    assert "github.event.pull_request.head.repo" not in workflow_text
    assert "git checkout" not in script
    assert "git switch" not in script


def test_dco_trailer_parser_rejects_malformed_or_missing_identity() -> None:
    assert SIGN_OFF.search("Subject\n\nSigned-off-by: A Person <person@example.com>\n")
    assert not SIGN_OFF.search("Signed-off-by: A Person\n")
    assert not SIGN_OFF.search("Signed-off-by: <person@example.com>\n")
    assert not SIGN_OFF.search("Subject only\n")


def test_dco_cli_rejects_malformed_range_as_usage_error() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_dco.py"), "not-a-sha-range"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "exact SHA40..SHA40" in result.stderr


def _temporary_git_range(
    tmp_path: Path, message: str, *, signoff: bool = False
) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Test Contributor")
    git("config", "user.email", "contributor@example.com")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "Base")
    base = git("rev-parse", "HEAD")
    (repository / "tracked.txt").write_text("change\n", encoding="utf-8")
    commit = ["commit", "--quiet", "-am", message]
    if signoff:
        commit.append("--signoff")
    git(*commit)
    head = git("rev-parse", "HEAD")
    return repository, f"{base}..{head}"


def _run_dco(repository: Path, revision_range: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check_dco.py"), revision_range],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dco_real_git_range_accepts_signed_commit(tmp_path: Path) -> None:
    repository, revision_range = _temporary_git_range(
        tmp_path, "Signed contribution", signoff=True
    )
    result = _run_dco(repository, revision_range)
    assert result.returncode == 0, result.stderr


def test_dco_real_git_range_rejects_unsigned_commit(tmp_path: Path) -> None:
    repository, revision_range = _temporary_git_range(tmp_path, "Unsigned contribution")
    result = _run_dco(repository, revision_range)
    assert result.returncode == 1
    assert revision_range.split("..", 1)[1] in result.stderr


def test_dco_rejects_signoff_looking_body_line(tmp_path: Path) -> None:
    repository, revision_range = _temporary_git_range(
        tmp_path,
        "Spoofed contribution\n\n"
        "Signed-off-by: Test Contributor <contributor@example.com>\n\n"
        "This paragraph proves the sign-off-looking line is body text.",
    )
    result = _run_dco(repository, revision_range)
    assert result.returncode == 1
