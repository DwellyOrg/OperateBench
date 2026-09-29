"""Historical provenance is independent of current dependency/reporting state."""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests import test_artifact8_evidence_freeze as freeze


def test_historical_audit_does_not_read_current_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_bytes = Path.read_bytes

    def read(path: Path) -> bytes:
        if path == freeze.ROOT / "uv.lock":
            raise AssertionError("historical audit read current dependency lock")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    freeze._audit()


@pytest.mark.parametrize("missing", [False, True])
def test_historical_audit_refuses_changed_or_missing_generation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    archived = freeze.ROOT / "tests/fixtures/historical_engine_0_9/source/uv.lock"
    original = archived.read_bytes()
    changed = tmp_path / "uv.lock"
    if not missing:
        changed.write_bytes(original[:-1] + b" ")
    read_bytes = Path.read_bytes

    def read(path: Path) -> bytes:
        # Make the old, incorrectly coupled audit otherwise pass, so rejection
        # must come from reading the damaged historical snapshot itself.
        if path == freeze.ROOT / "uv.lock":
            return original
        if path == archived:
            return read_bytes(changed)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(FileNotFoundError if missing else AssertionError):
        freeze._audit()


def test_acceptance_failure_is_reported_normally_with_junit(tmp_path: Path) -> None:
    test = tmp_path / "test_controlled_failure.py"
    test.write_text(
        "from tests import test_artifact8_evidence_freeze as freeze\n"
        "def test_controlled_failure(monkeypatch):\n"
        "    def fail():\n"
        "        raise AssertionError('controlled acceptance failure')\n"
        "    monkeypatch.setattr(freeze, '_audit', fail)\n"
        "    freeze.test_acceptance_constructs_no_provider_transport_"
        "socket_or_credentials(monkeypatch)\n",
        encoding="utf-8",
    )
    report = tmp_path / "junit.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            os.devnull,
            "--confcutdir",
            str(tmp_path),
            "-q",
            str(test),
            f"--junitxml={report}",
        ],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": os.pathsep.join((str(freeze.ROOT / "src"), str(freeze.ROOT))),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 1
    failure = cases[0].find("failure")
    assert failure is not None
    assert "controlled acceptance failure" in (failure.text or "")
    assert cases[0].find("error") is None


@pytest.mark.parametrize(
    "attempt",
    [
        "os.environ.__getitem__",
        "os.environ.get",
        "os.getenv",
        "openai.OpenAI",
        "httpx.Client",
        "httpx.MockTransport",
        "socket.socket",
        "socket.create_connection",
        "openai_responses.OpenAIResponsesTransport",
        "evidence.EvidenceRecordingModelAgent",
        "evidence.WireCaptureTransport",
    ],
)
def test_acceptance_still_forbids_every_trapped_access(
    monkeypatch: pytest.MonkeyPatch, attempt: str
) -> None:
    from operatebench.agents import evidence, openai_responses

    def access() -> None:
        namespace = {
            **vars(freeze),
            "evidence": evidence,
            "openai_responses": openai_responses,
        }
        module, *attributes = attempt.split(".")
        target = namespace[module]
        for attribute in attributes:
            target = getattr(target, attribute)
        target("ANY_KEY")

    monkeypatch.setattr(freeze, "_audit", access)
    # The outer context also protects pytest when running against the old code.
    with (
        pytest.raises(AssertionError, match="acceptance touched credentials"),
        monkeypatch.context() as guarded,
    ):
        freeze.test_acceptance_constructs_no_provider_transport_socket_or_credentials(
            guarded
        )
