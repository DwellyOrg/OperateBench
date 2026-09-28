"""Focused contract tests for the reproducible public source distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sdist_declares_technical_sources_and_excludes_release_controls() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist_include = set(
        pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    )

    for control in (
        "PUBLICATION_MANIFEST.json",
        "PUBLIC_RELEASE_CHECKLIST.md",
        "PUBLIC_CANDIDATE_VERIFICATION.md",
    ):
        assert f"/{control}" not in sdist_include
        assert (REPO_ROOT / control).is_file()
    for technical in (
        "uv.lock",
        ".github/workflows/ci.yml",
        ".github/workflows/dco.yml",
        "REUSE.toml",
        "DCO",
    ):
        assert f"/{technical}" in sdist_include


def test_extracted_sdist_passes_its_bundled_release_scanner_offline(
    extracted_sdist_scan: tuple[int, str, str],
) -> None:
    returncode, stdout, stderr = extracted_sdist_scan
    assert returncode == 0, stdout + stderr
