"""The ``run-suite`` command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from boundarybench.cli import main
from tests.conftest import SUITE_MANIFEST
from tests.scaffolds import scaffold_payload, write_scaffold


def _argv(output: Path, *extra: str) -> list[str]:
    return [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        "fake-scripted",
        "--output-dir",
        str(output),
        *extra,
    ]


def test_fake_run_reports_its_counts_and_says_what_it_is_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_argv(tmp_path / "run", "--trials", "2"))

    assert code == 0
    out = capsys.readouterr().out
    assert "planned            24" in out
    assert "completed          24" in out
    assert "succeeded          24" in out
    assert "failed             0" in out
    assert "resumed            0" in out
    assert "success 24" in out
    assert "configuration id" in out
    assert "execution id" in out
    assert "suite digest" in out
    assert "scaffold digest" in out
    assert str(tmp_path / "run" / "episodes.jsonl") in out
    # The disclaimer is not optional decoration.
    assert "not a model benchmark result" in out
    assert "no ranking" in out


def test_json_output_is_stable_and_carries_the_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv(tmp_path / "run", "--trials", "1", "--json")) == 0
    first = capsys.readouterr().out

    assert main(_argv(tmp_path / "run", "--trials", "1", "--json")) == 0
    second = capsys.readouterr().out

    payload = json.loads(first)
    assert payload["counts"]["planned"] == 12
    assert payload["counts"]["executed"] == 12
    assert payload["outcome_counts"] == {"success": 12}
    assert "not a model benchmark result" in payload["scope"]
    assert payload["retry_policy"] == {"retries": 0, "policy": "none"}

    repeat = json.loads(second)
    # Same configuration, and the *same* execution: the second invocation minted
    # a candidate execution id and discarded it in favour of the one the run
    # directory already holds, so its rows all belong to one execution.
    assert repeat["configuration_id"] == payload["configuration_id"]
    assert repeat["execution_id"] == payload["execution_id"]
    assert repeat["created_at_utc"] == payload["created_at_utc"]
    assert repeat["counts"]["executed"] == 0
    assert repeat["counts"]["resumed"] == 12


def test_rerun_is_a_no_op_over_the_ledger_bytes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    assert main(_argv(root, "--trials", "1")) == 0
    before = (root / "episodes.jsonl").read_bytes()

    assert main(_argv(root, "--trials", "1")) == 0

    assert (root / "episodes.jsonl").read_bytes() == before


def test_a_different_request_in_the_same_directory_fails_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    assert main(_argv(root, "--trials", "1")) == 0

    code = main(_argv(root, "--trials", "2"))

    captured = capsys.readouterr()
    assert code == 1
    assert "already holds configuration" in captured.err
    assert "Traceback" not in captured.err


def test_a_symlinked_output_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    code = main(_argv(link))

    captured = capsys.readouterr()
    assert code == 1
    assert "symlink" in captured.err
    assert "Traceback" not in captured.err


def test_a_stale_scaffold_pin_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = scaffold_payload()
    path = write_scaffold(tmp_path / "scaffold.json", payload, digest="0" * 64)

    code = main(_argv(tmp_path / "run", "--scaffold", str(path)))

    captured = capsys.readouterr()
    assert code == 1
    assert "does not match the computed digest" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (("--trials", "0"), "trials must be a positive integer"),
        (("--max-turns", "0"), "max_turns must be a positive integer"),
        (("--timeout", "0"), "episode_timeout_seconds must be a positive"),
        (("--setting", "api_key=x"), "looks like a credential"),
        (("--setting", "novalue"), "must be given as key=value"),
    ],
)
def test_invalid_run_parameters_exit_non_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: tuple[str, ...],
    expected: str,
) -> None:
    code = main(_argv(tmp_path / "run", *extra))

    captured = capsys.readouterr()
    assert code == 1
    assert expected in captured.err
    assert "Traceback" not in captured.err


def test_settings_are_recorded_in_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "run"
    assert main(_argv(root, "--trials", "1", "--setting", "temperature=0")) == 0

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_settings"] == {"temperature": "0"}
    assert manifest["adapter"]["provider"] == "fake"
    assert manifest["status"] == "SYNTHETIC_INFRASTRUCTURE_SMOKE"


def test_an_unwritable_output_directory_fails_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusing ``/proc`` path is reported as an operator-facing error.

    The command exits 1 without a traceback and names the path it cannot create.
    """
    code = main(_argv(Path("/proc/boundarybench-impossible")))

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "/proc/boundarybench-impossible" in captured.err


def test_a_ledger_replaced_by_a_symlink_fails_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Substituting the ledger after the run directory was validated."""
    root = tmp_path / "run"
    assert main(_argv(root, "--trials", "1")) == 0
    ledger = root / "episodes.jsonl"
    ledger.unlink()
    ledger.symlink_to(tmp_path / "elsewhere.jsonl")

    code = main(_argv(root, "--trials", "1"))

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "symlink" in captured.err


def test_a_second_concurrent_writer_is_refused_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two real processes, one run directory: only one may append."""
    root = tmp_path / "run"
    assert main(_argv(root, "--trials", "1")) == 0
    before = (root / "episodes.jsonl").read_bytes()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            "fd = open(sys.argv[1], 'a+')\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "sys.stdin.readline()\n",
            str(root / ".run.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"

        code = main(_argv(root, "--trials", "1"))

        captured = capsys.readouterr()
        assert code == 1
        assert "already running" in captured.err
        assert "Traceback" not in captured.err
        assert (root / "episodes.jsonl").read_bytes() == before
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        holder.wait(timeout=30)


def test_an_unexpected_filesystem_error_becomes_a_named_cli_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing filesystem-shaped reaches the operator as a traceback.

    The specific paths a run touches are wrapped where they happen; this is the
    backstop for the ones that are not, so a new call site cannot regress the
    no-traceback guarantee silently.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("boundarybench.cli.validate_suite", refuse)

    code = main(_argv(tmp_path / "run"))

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "No space left on device" in captured.err


def test_a_setting_given_twice_is_refused_rather_than_silently_last_wins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two values for one key is an ambiguous request, and identity depends on it.

    Adapter settings are inside configuration identity, so quietly keeping the
    last one would record a run under a configuration the operator did not ask
    for and could not see they had not asked for.
    """
    root = tmp_path / "run"

    code = main(_argv(root, "--setting", "temperature=0", "--setting", "temperature=1"))

    captured = capsys.readouterr()
    assert code == 1
    assert "given twice" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_an_unrepresentable_input_becomes_a_named_cli_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backstop for representability, pinned by injection.

    Every artefact reader restates this as its own domain error where it
    happens, so no shipped path reaches this handler today. It exists for the
    call site that has not been wrapped yet, and an unwrapped call site is
    exactly the case a test cannot discover by running the wrapped ones — so the
    escape is injected and the operator-facing contract asserted directly.
    """
    from boundarybench.jsonsafe import JsonSafetyError

    def refuse(*args: object, **kwargs: object) -> None:
        raise JsonSafetyError("a lone surrogate cannot be encoded")

    monkeypatch.setattr("boundarybench.cli.validate_suite", refuse)

    code = main(_argv(tmp_path / "run"))

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "cannot be persisted" in captured.err
    assert "a lone surrogate cannot be encoded" in captured.err


def test_an_over_nested_input_becomes_a_named_cli_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RecursionError`` is not a ``ValueError`` and escapes every domain handler.

    Injected for the same reason as the representability backstop: the readers
    that walk nested input name their own over-nesting today, and this handler
    is what stops the next one that does not from reaching the operator as a
    traceback.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("boundarybench.cli.validate_suite", refuse)

    code = main(_argv(tmp_path / "run"))

    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "nested too deeply" in captured.err
    assert "RecursionError" in captured.err
