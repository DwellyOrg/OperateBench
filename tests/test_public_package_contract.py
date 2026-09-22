"""Focused contract tests for the reproducible public source distribution."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
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
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--sdist", "--offline", "--out-dir", str(dist)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    archive = next(dist.glob("*.tar.gz"))
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as sdist:
        sdist.extractall(extracted, filter="data")
    root = next(extracted.iterdir())
    environment = os.environ.copy()
    environment.update(
        {
            "HTTP_PROXY": "http://" + "127.0.0.1:9",
            "HTTPS_PROXY": "http://" + "127.0.0.1:9",
            "NO_PROXY": "",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "tools.check_public_release", "--surface", "sdist"],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
