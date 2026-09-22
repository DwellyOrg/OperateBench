"""Historical mock regeneration is self-contained in a source export."""

from pathlib import Path

from tests.historical_runtime import run_historical_mock


def test_b3_regenerates_without_git_objects():
    root = Path(__file__).resolve().parents[1]
    run_historical_mock("b3", root / "tests/fixtures/b3")


def test_every_historical_file_has_an_independent_pin():
    import hashlib
    import json

    from tests import historical_runtime as history
    from tests.historical_source_pins import MANIFEST_SHA256, PINS

    fixture = history.ROOT / "tests/fixtures/historical_engine_0_9"
    manifest = (fixture / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == MANIFEST_SHA256
    assert json.loads(manifest)["files"] == PINS
    actual = {
        p.relative_to(fixture / "source").as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in (fixture / "source").rglob("*")
        if p.is_file()
    }
    assert actual == PINS


def test_historical_tamper_refuses_before_execution(tmp_path, monkeypatch):
    import shutil

    import pytest

    from tests import historical_runtime as history

    target = tmp_path / "tests/fixtures/historical_engine_0_9"
    shutil.copytree(history.ROOT / "tests/fixtures/historical_engine_0_9", target)
    source = target / "source/src/operatebench/core/engine.py"
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(history, "ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered source reached subprocess")

    monkeypatch.setattr(history.subprocess, "run", forbidden)
    with pytest.raises(
        history.HistoricalRuntimeUnavailable, match="source content mismatch"
    ):
        history.run_historical_mock("alpha", tmp_path / "output")
    assert not (tmp_path / "output").exists()
