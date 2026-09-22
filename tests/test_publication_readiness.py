"""Release-only regressions; credential canaries are assembled, never authentic."""

from pathlib import Path

import pytest

from tools import check_public_release as checks


@pytest.mark.parametrize("family", ["proj", "svcacct"])
def test_complete_openai_prefix_is_detected(tmp_path: Path, family: str) -> None:
    token = "-".join(("sk", family, "Ab9_" * 35 + "-Z7"))
    (tmp_path / "NOTICE").write_text(token)
    findings = checks.check_no_credential_shapes(tmp_path)
    assert len(findings) == 1
    assert "OpenAI" in findings[0]


@pytest.mark.parametrize("family", ["proj", "svcacct"])
@pytest.mark.parametrize("length", [32, 48, 143, 512])
@pytest.mark.parametrize("wrapper", [("", ""), ("'", "'"), ("(", "),"), ("\n", "\n")])
def test_openai_token_envelope(
    tmp_path: Path, family: str, length: int, wrapper: tuple[str, str]
) -> None:
    payload = ("aB9_-" * 103)[:length]
    token = "-".join(("sk", family, payload))
    (tmp_path / ".config").write_text(wrapper[0] + token + wrapper[1])
    findings = checks.check_no_credential_shapes(tmp_path)
    assert len(findings) == 1
    assert repr(token) in findings[0]


@pytest.mark.parametrize("family", ["proj", "svcacct"])
@pytest.mark.parametrize(
    "payload", ["a_" * 15 + "a", "a_" * 256 + "a", "short", "a_" * 12 + "." + "a_" * 12]
)
def test_outside_documented_envelope_is_not_a_complete_key(
    tmp_path: Path, family: str, payload: str
) -> None:
    (tmp_path / "NOTICE").write_text("-".join(("sk", family, payload)))
    assert checks.check_no_credential_shapes(tmp_path) == []


@pytest.mark.parametrize("prefix", ["x", "_", "-", "9"])
def test_embedded_prefix_is_not_a_token_boundary(tmp_path: Path, prefix: str) -> None:
    token = "-".join(("sk", "proj", "a_" * 24))
    (tmp_path / "NOTICE").write_text(prefix + token)
    assert checks.check_no_credential_shapes(tmp_path) == []


def test_repeated_prefix_is_scanned_as_one_complete_token(tmp_path: Path) -> None:
    token = "-".join(("sk", "proj", "sk", "proj", "a_" * 24))
    (tmp_path / "NOTICE").write_text(token + "," + token)
    findings = checks.check_no_credential_shapes(tmp_path)
    assert len(findings) == 2
    assert all(repr(token) in finding for finding in findings)


@pytest.mark.parametrize("surface", ["source", "wheel", "sdist"])
@pytest.mark.parametrize(
    "name",
    [
        "NOTICE",
        ".config",
        "METADATA",
        "PKG-INFO",
        "CITATION.cff",
        "uv.lock",
        "data.future",
        "module.py",
    ],
)
@pytest.mark.parametrize("family", ["proj", "svcacct"])
def test_complete_prefix_on_every_text_surface(
    tmp_path: Path, surface: str, name: str, family: str
) -> None:
    import io
    import tarfile
    import zipfile

    from tests.test_public_release import _empty_distribution

    token = "-".join(("sk", family, "aB9_-" * 30))
    if surface == "source":
        (tmp_path / name).write_text(token)
        findings = checks.check_no_credential_shapes(tmp_path)
    else:
        dist = tmp_path / "dist"
        _empty_distribution(dist)
        if surface == "wheel":
            with zipfile.ZipFile(dist / "pkg-0.1.0-py3-none-any.whl", "w") as archive:
                archive.writestr("pkg/" + name, token)
        else:
            with tarfile.open(dist / "pkg-0.1.0.tar.gz", "w:gz") as archive:
                info = tarfile.TarInfo("pkg-0.1.0/" + name)
                info.size = len(token)
                archive.addfile(info, io.BytesIO(token.encode()))
        findings = checks.check_built_distributions(tmp_path)
    assert len(findings) == 1
    assert name + ":1" in findings[0]
    assert repr(token) in findings[0]


def test_readme_leads_with_runnable_bounded_preview() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text()
    assert text.splitlines().index("## Quickstart") < 50
    opening = " ".join(text.split("## A task is not an operation")[0].split())
    for phrase in (
        "one Semantic Scenario",
        "three frozen scenario variants",
        "Commerce",
        "Property Compliance",
        "No proven incremental",
        "not production-ready",
    ):
        assert phrase in opening
    assert "not yet published as a repository" not in text
    assert "executable schemas" in text


def test_manifest_records_scoped_base_acceptance_not_blanket_staleness() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    assert manifest["target"]["repository"] == "https://github.com/DwellyOrg/OperateBench"
    assert manifest["candidate_acceptance"]["status"] == "requires_exact_source_record"
    assert manifest["candidate_acceptance"]["hosted_ci"] == "pending"
    gates = {g["id"].split("_")[0]: g for g in manifest["gates"]}
    assert {k for k, v in gates.items() if v["status"] == "passed"} == {
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G7",
        "G10",
    }
    assert all(g["evidence_candidate"] == "verified_base" for g in gates.values())
    assert all(
        g["status"] == "pending"
        for k, g in gates.items()
        if k in {"G6", "G8", "G9", "G11", "G12"}
    )


def test_ci_keeps_full_matrix_and_independent_secret_scan() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    for required in (
        "workflow_dispatch:",
        "name: independent secret scan",
        "name: build distribution",
        "name: test (py${{ matrix.python-version }})",
        'python-version: ["3.11", "3.14"]',
        "--cov-branch",
        "gitleaks dir --redact",
        "sha256sum --check",
        "persist-credentials: false",
    ):
        assert required in workflow
    assert "--exit-code 0" not in workflow
    assert "--baseline-path" not in workflow
    assert "secrets." not in "\n".join(
        line for line in workflow.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.parametrize(
    "family", ["legacy", "anthropic", "github", "aws", "google", "slack", "pem", "bearer"]
)
def test_existing_credential_families_stay_detectable(
    tmp_path: Path, family: str
) -> None:
    tokens = {
        "legacy": "sk-" + "aB9" * 16,
        "anthropic": "-".join(("sk", "ant", "aB9_" * 10)),
        "github": "ghp" + "_" + "aB9" * 12,
        "aws": "AK" + "IA" + "AB09" * 4,
        "google": "AI" + "za" + "aB9_" * 8 + "XYZ",
        "slack": "xox" + "b-" + "aB9-" * 8,
        "pem": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "bearer": "Bearer " + "aB9_" * 8 + ".",
    }
    (tmp_path / "NOTICE").write_text(tokens[family])
    assert checks.check_no_credential_shapes(tmp_path)


def _assert_candidate_and_human_requirements(manifest: dict) -> None:
    assert manifest["candidate_acceptance"]["status"] == "requires_exact_source_record"
    assert manifest["candidate_acceptance"]["hosted_ci"] == "pending"
    gates = {gate["id"].split("_")[0]: gate for gate in manifest["gates"]}
    for name in ("G6", "G8", "G9", "G11", "G12"):
        assert gates[name]["status"] == "pending"
        assert gates[name]["blocking"] is True
        assert gates[name]["evidence"]
    assert (
        "Human requirements cannot be closed by CI"
        in manifest["candidate_acceptance"]["closure"]
    )


@pytest.mark.parametrize("gate_name", ["G6", "G8", "G9", "G11", "G12"])
@pytest.mark.parametrize(
    "field,value", [("status", "passed"), ("blocking", False), ("evidence", "")]
)
def test_gate_contract_rejects_unproved_closure(
    gate_name: str, field: str, value: object
) -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text())
    _assert_candidate_and_human_requirements(manifest)
    gate = next(g for g in manifest["gates"] if g["id"].startswith(gate_name + "_"))
    gate[field] = value
    with pytest.raises(AssertionError):
        _assert_candidate_and_human_requirements(manifest)
