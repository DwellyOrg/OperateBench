# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0

"""Regression contract for the platform claimed by the public package."""

from __future__ import annotations

import subprocess
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINUX_CLASSIFIER = "Operating System :: POSIX :: Linux"
OS_INDEPENDENT_CLASSIFIER = "Operating System :: OS Independent"


def test_source_metadata_and_readme_state_the_linux_capability_boundary() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    classifiers = project["classifiers"]
    assert OS_INDEPENDENT_CLASSIFIER not in classifiers
    assert LINUX_CLASSIFIER in classifiers

    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    for required in (
        "Linux kernel `openat2`",
        "`RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS|RESOLVE_NO_MAGICLINKS`",
        "exported libc wrapper",
        "audited x86_64 fallback",
        "`O_PATH`",
        "seccomp",
        "fail closed before any artifact filesystem mutation",
        "Linux x86_64 / Ubuntu",
    ):
        assert required in readme


def test_built_wheel_metadata_claims_linux_not_os_independent(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(dist)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    classifiers = metadata.get_all("Classifier", [])
    assert OS_INDEPENDENT_CLASSIFIER not in classifiers
    assert LINUX_CLASSIFIER in classifiers
