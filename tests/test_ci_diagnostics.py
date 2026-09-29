"""Bounded ordinary pytest exits and the CI diagnostic artifact contract."""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_ci_keeps_lane_semantics_and_always_uploads_only_junit():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["test"]
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.14"]
    command = next(s["run"] for s in job["steps"] if s["name"] == "pytest")
    assert "pytest -q" not in command
    assert "--durations=30" in command
    assert '--junitxml="$RUNNER_TEMP/pytest-${{ matrix.python-version }}.xml"' in command
    assert (
        "--cov=operatebench --cov=boundarybench --cov-branch --cov-report=term-missing"
        in command
    )
    upload = next(s for s in job["steps"] if s["name"] == "Upload pytest diagnostics")
    assert upload["if"] == "always()"
    distribution_upload = next(
        s
        for s in workflow["jobs"]["build"]["steps"]
        if s["name"] == "Upload the distribution as an artifact"
    )
    assert upload["uses"] == distribution_upload["uses"]
    assert re.fullmatch(r"actions/upload-artifact@[0-9a-f]{40}", upload["uses"])
    assert (
        upload["with"]["path"]
        == "${{ runner.temp }}/pytest-${{ matrix.python-version }}.xml"
    )
    assert upload["with"]["retention-days"] == 3
    assert "${{ matrix.python-version }}" in upload["with"]["name"]


@pytest.mark.parametrize("succeeds", [True, False])
def test_ordinary_pytest_exit_and_credential_free_junit(tmp_path, succeeds):
    test = tmp_path / "test_bounded.py"
    test.write_text(f"def test_example():\n    assert {succeeds!r}\n")
    xml = tmp_path / "pytest-local.xml"
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "--durations=30",
            f"--junitxml={xml}",
            "-o",
            "junit_logging=no",
            "-o",
            "junit_log_passing_tests=false",
            str(test),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (0 if succeeds else 1)
    suite = ET.parse(xml).getroot().find("testsuite")
    assert suite is not None
    assert suite.get("tests") == "1"
    assert suite.get("failures") == ("0" if succeeds else "1")
    assert suite.find(".//system-out") is None
    assert suite.find(".//system-err") is None
