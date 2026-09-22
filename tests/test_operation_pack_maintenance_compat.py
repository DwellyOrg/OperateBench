"""Golden compatibility between generic dispatch and Maintenance aliases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from operatebench.artifact import ARTIFACT_VERSION, read_artifact
from operatebench.cli import EXIT_ERROR, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "examples" / "operatebench" / "maintenance_v0_1.yaml"


def test_artifact_contract_remains_v8() -> None:
    assert ARTIFACT_VERSION == 8


def test_generic_and_implicit_validate_are_identical_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", str(FIXTURE), "--json"]) == EXIT_OK
    legacy = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "validate",
                "--pack",
                "maintenance",
                str(FIXTURE),
                "--json",
            ]
        )
        == EXIT_OK
    )
    generic = json.loads(capsys.readouterr().out)
    assert generic == legacy


def test_generic_and_legacy_run_write_identical_v8(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Observation and tape digests include this opaque id, so bind both command
    # paths to one deterministic test value before comparing complete bytes.
    monkeypatch.setattr(
        "operatebench.runner.new_operation_instance_id",
        lambda: "opinst_00000000000000000000000000000001",
    )
    legacy = tmp_path / "legacy.json"
    generic = tmp_path / "generic.json"
    common = [
        "--spec",
        str(FIXTURE),
        "--scenario",
        "V1",
        "--agent",
        "reference",
    ]
    assert main(["run-maintenance", *common, "--output", str(legacy)]) == EXIT_OK
    capsys.readouterr()
    assert (
        main(
            [
                "run",
                "--pack",
                "maintenance",
                *common,
                "--output",
                str(generic),
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()
    assert generic.read_bytes() == legacy.read_bytes()
    assert read_artifact(generic)["artifact_version"] == 8


def test_wrong_operation_type_is_refused_before_an_output_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(
            "operation_type: lettings.maintenance.synthetic",
            "operation_type: commerce.return_refund.synthetic",
            1,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"
    code = main(
        [
            "run",
            "--pack",
            "maintenance",
            "--spec",
            str(wrong),
            "--scenario",
            "V1",
            "--agent",
            "reference",
            "--output",
            str(output),
        ]
    )
    assert code == EXIT_ERROR
    assert not output.exists()
    error = capsys.readouterr().err
    assert "operation_type" in error
    assert "Traceback" not in error
