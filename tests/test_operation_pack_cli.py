"""Domain-neutral CLI dispatch and Maintenance compatibility aliases."""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import pytest

from operatebench.cli import (
    EXIT_CONTRACT_FAILED,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    main,
)
from operatebench.jsonsafe import MAX_JSON_DEPTH
from operatebench.sdk import (
    CheckRequest,
    CommandResult,
    OperationPackMetadata,
    OperationPackRegistry,
    ReplayRequest,
    RunRequest,
    ValidateRequest,
)


class RecordingPack:
    metadata = OperationPackMetadata(
        pack_id="commerce.return_refund.synthetic",
        operation_type="commerce.return_refund.synthetic",
        operation_id="commerce_return_refund_synthetic_v1",
        pack_version="0.1.0",
        display_name="Synthetic return and refund",
        owner_id="prospire",
        owner_display_name="PROSPIRE TECHNOLOGIES LTD",
        contribution_kind="maintainer",
        status="incubator",
        privacy_status="SYNTHETIC_ONLY",
        default_spec=None,
        agent_ids=("negative", "reference"),
    )

    def __init__(self, *, passes: bool = True) -> None:
        self.passes = passes
        self.requests: list[object] = []

    def _result(self, request: object) -> CommandResult:
        self.requests.append(request)
        return CommandResult(
            self.passes,
            {"request_type": type(request).__name__, "ok": self.passes},
            (f"handled {type(request).__name__}",),
        )

    def validate(self, request: ValidateRequest) -> CommandResult:
        return self._result(request)

    def run(self, request: RunRequest) -> CommandResult:
        return self._result(request)

    def replay(self, request: ReplayRequest) -> CommandResult:
        return self._result(request)

    def check(self, request: CheckRequest) -> CommandResult:
        return self._result(request)


def _main(pack: RecordingPack, *argv: str) -> int:
    return main(list(argv), registry=OperationPackRegistry((pack,)))


class _DispatchParser:
    """Bypass argparse so dispatch defenses are executable in isolation."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def parse_args(self, argv: object) -> Namespace:
        return self.args

    def print_usage(self, file: TextIO) -> None:
        print("usage: operatebench", file=file)


@pytest.mark.parametrize(
    ("argv", "request_type"),
    [
        (
            (
                "validate",
                "--pack",
                "commerce.return_refund.synthetic",
                "spec.yaml",
            ),
            ValidateRequest,
        ),
        (
            (
                "run",
                "--pack",
                "commerce.return_refund.synthetic",
                "--spec",
                "spec.yaml",
                "--scenario",
                "V1",
                "--agent",
                "reference",
                "--output",
                "run.json",
            ),
            RunRequest,
        ),
        (
            (
                "replay",
                "--pack",
                "commerce.return_refund.synthetic",
                "--spec",
                "spec.yaml",
                "--run",
                "run.json",
            ),
            ReplayRequest,
        ),
        (
            (
                "check",
                "--pack",
                "commerce.return_refund.synthetic",
                "--spec",
                "spec.yaml",
            ),
            CheckRequest,
        ),
    ],
)
def test_generic_commands_dispatch_without_maintenance_fields(
    argv: tuple[str, ...],
    request_type: type[object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = RecordingPack()
    assert _main(pack, *argv) == EXIT_OK
    assert isinstance(pack.requests[-1], request_type)
    assert "handled" in capsys.readouterr().out


def test_contract_failure_is_exit_two_and_json_is_pack_owned(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = RecordingPack(passes=False)
    code = _main(
        pack,
        "check",
        "--pack",
        pack.metadata.pack_id,
        "--spec",
        "spec.yaml",
        "--json",
    )
    assert code == EXIT_CONTRACT_FAILED
    assert json.loads(capsys.readouterr().out) == {
        "request_type": "CheckRequest",
        "ok": False,
    }


def test_computed_mapping_becoming_unsafe_is_exit_one_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ChangingMapping(Mapping[str, object]):
        def __init__(self) -> None:
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            if key != "value":
                raise KeyError(key)
            self.reads += 1
            return "safe on validation" if self.reads == 1 else {"not-json"}

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "value"

        def __len__(self) -> int:
            return 1

    changing = ChangingMapping()

    class ChangingPack(RecordingPack):
        def _result(self, request: object) -> CommandResult:
            self.requests.append(request)
            return CommandResult(True, {"nested": changing})

    pack = ChangingPack()
    code = _main(
        pack,
        "validate",
        "--pack",
        pack.metadata.pack_id,
        "spec.yaml",
    )

    assert code == EXIT_ERROR
    assert changing.reads == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot be persisted" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("kind", "failure", "message"),
    (
        ("mapping", "cycle", "cyclic reference"),
        ("sequence", "cycle", "cyclic reference"),
        ("mapping", "depth", f"deeper than {MAX_JSON_DEPTH} levels"),
        ("sequence", "depth", f"deeper than {MAX_JSON_DEPTH} levels"),
    ),
)
def test_projection_time_structural_refusal_is_canonical_without_cli_traceback(
    kind: str,
    failure: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ChangingMapping(Mapping[str, object]):
        def __init__(self, projected: object) -> None:
            self.projected = projected
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            if key != "value":
                raise KeyError(key)
            return "safe on pre-validation"

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "value"

        def __len__(self) -> int:
            return 1

        def items(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            yield "value", "safe on pre-validation" if self.reads == 1 else self.projected

    class ChangingSequence(Sequence[object]):
        def __init__(self, projected: object) -> None:
            self.projected = projected
            self.reads = 0

        def __getitem__(self, index: int) -> object:  # type: ignore[override]
            if index != 0:
                raise IndexError(index)
            return "safe on pre-validation"

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            self.reads += 1
            yield "safe on pre-validation" if self.reads == 1 else self.projected

    projected: object = "leaf"
    if failure == "depth":
        for _ in range(max(MAX_JSON_DEPTH + 1, 1_500)):
            projected = [projected]
    changing: ChangingMapping | ChangingSequence = (
        ChangingMapping(projected) if kind == "mapping" else ChangingSequence(projected)
    )
    if failure == "cycle":
        changing.projected = changing

    class ChangingPack(RecordingPack):
        def _result(self, request: object) -> CommandResult:
            self.requests.append(request)
            return CommandResult(True, {"nested": changing})

    pack = ChangingPack()
    code = _main(
        pack,
        "validate",
        "--pack",
        pack.metadata.pack_id,
        "spec.yaml",
    )

    assert code == EXIT_ERROR
    assert changing.reads == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot be persisted" in captured.err
    assert "JSON projection" in captured.err
    assert message in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("unknown-command",),
        ("run", "--pack", "commerce.return_refund.synthetic"),
    ),
)
def test_malformed_invocation_is_distinct_usage_exit_without_traceback(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(list(argv), registry=OperationPackRegistry((RecordingPack(),)))

    assert code == EXIT_USAGE
    assert code != EXIT_CONTRACT_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: operatebench")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ("run", "check"))
def test_generic_dispatch_defends_against_a_missing_pack(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = RecordingPack()
    monkeypatch.setattr(
        "operatebench.cli._build_parser",
        lambda: _DispatchParser(Namespace(command=command, pack=None)),
    )

    code = main([], registry=OperationPackRegistry((pack,)))

    assert code == EXIT_USAGE
    assert pack.requests == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: operatebench")
    assert "requires --pack" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_pack_is_exit_one_before_output_is_touched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = RecordingPack()
    output = tmp_path / "must-not-exist.json"
    code = _main(
        pack,
        "run",
        "--pack",
        "missing.synthetic",
        "--spec",
        "spec.yaml",
        "--scenario",
        "V1",
        "--agent",
        "reference",
        "--output",
        str(output),
    )
    assert code == EXIT_ERROR
    assert not output.exists()
    assert "Traceback" not in capsys.readouterr().err


def test_list_packs_exposes_development_non_admission(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = RecordingPack()
    assert _main(pack, "list-packs", "--json") == EXIT_OK
    [metadata] = json.loads(capsys.readouterr().out)
    assert metadata["pack_id"] == pack.metadata.pack_id
    assert metadata["operation_id"] == pack.metadata.operation_id
    assert metadata["status"] == "incubator"
    assert metadata["owner_id"] == "prospire"
    assert metadata["owner_display_name"] == "PROSPIRE TECHNOLOGIES LTD"
    assert metadata["contribution_kind"] == "maintainer"
    assert metadata["evidence_eligible"] is False


def test_list_packs_human_output_exposes_owner_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pack = RecordingPack()
    assert _main(pack, "list-packs") == EXIT_OK
    output = capsys.readouterr().out
    assert "owner=prospire" in output
    assert "PROSPIRE TECHNOLOGIES LTD" in output
    assert "maintainer" in output
    assert "operation_id=commerce_return_refund_synthetic_v1" in output


def test_init_refuses_a_registered_operation_type_before_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = RecordingPack()
    pack.metadata = replace(
        pack.metadata,
        pack_id="reviewed.commerce.synthetic",
        operation_type="commerce.return_refund.synthetic",
    )
    destination = tmp_path / "must_not_exist"
    code = _main(
        pack,
        "init-operation",
        "--pack-id",
        pack.metadata.operation_type,
        "--operation-id",
        "new_operation_v1",
        "--owner-id",
        "prospire",
        "--owner-display-name",
        "PROSPIRE TECHNOLOGIES LTD",
        "--contribution-kind",
        "maintainer",
        "--destination",
        str(destination),
    )
    assert code == EXIT_ERROR
    assert not destination.exists()
    assert "already registered" in capsys.readouterr().err


def test_init_refuses_registered_commerce_operation_id_before_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = RecordingPack()
    destination = tmp_path / "must-not-exist"
    code = _main(
        pack,
        "init-operation",
        "--pack-id",
        "new.commerce.synthetic",
        "--operation-id",
        pack.metadata.operation_id,
        "--owner-id",
        "prospire",
        "--owner-display-name",
        "PROSPIRE TECHNOLOGIES LTD",
        "--contribution-kind",
        "maintainer",
        "--destination",
        str(destination),
    )
    assert code == EXIT_ERROR
    assert not destination.exists()
    assert "already registered" in capsys.readouterr().err


def test_init_refuses_its_own_cross_kind_identity_collision_before_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = RecordingPack()
    destination = tmp_path / "sample"
    identity = "partner.northstar.sample"
    code = _main(
        pack,
        "init-operation",
        "--pack-id",
        identity,
        "--operation-id",
        identity,
        "--owner-id",
        "northstar",
        "--owner-display-name",
        "Northstar Fictional Company",
        "--contribution-kind",
        "partner",
        "--destination",
        str(destination),
    )
    assert code == EXIT_ERROR
    assert not destination.exists()
    assert "must be distinct" in capsys.readouterr().err


def test_init_refuses_non_nfc_owner_display_before_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "sample"
    code = _main(
        RecordingPack(),
        "init-operation",
        "--pack-id",
        "new.sample.synthetic",
        "--operation-id",
        "new_sample_v1",
        "--owner-id",
        "etoile",
        "--owner-display-name",
        "E\u0301toile",
        "--contribution-kind",
        "maintainer",
        "--destination",
        str(destination),
    )
    assert code == EXIT_ERROR
    assert not destination.exists()
    assert "owner_display_name" in capsys.readouterr().err


def test_init_preserves_nfc_international_owner_display(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "sample"
    display = "Étoile 技術株式会社"
    code = _main(
        RecordingPack(),
        "init-operation",
        "--pack-id",
        "new.sample.synthetic",
        "--operation-id",
        "new_sample_v1",
        "--owner-id",
        "etoile",
        "--owner-display-name",
        display,
        "--contribution-kind",
        "maintainer",
        "--destination",
        str(destination),
    )
    assert code == EXIT_OK
    assert display in (destination / "pack.py").read_text()
    assert "created unregistered incubator scaffold" in capsys.readouterr().out
