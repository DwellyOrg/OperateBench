"""Historical fixture generation must not run under the current engine."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import regenerate_b3_evidence as b3


@pytest.mark.parametrize("allowance", [1, 2])
def test_ordinary_fallback_wakes_obey_invocation_allowance(allowance: int) -> None:
    from operatebench.core.engine import Engine
    from operatebench.core.outcomes import Wait
    from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
    from operatebench.domains.lettings.maintenance.spec import load_spec
    from tests.test_maintenance_construct_integrity import (
        _invocation_indices,
        _violation_codes,
        _WithAllowance,
    )

    class FallbackAgent:
        agent_id = "fallback-allowance-test"
        calls = 0

        def begin_episode(self, identity: Any) -> None:
            pass

        def decide(self, observation: Any) -> Wait:
            self.calls += 1
            return Wait(reason="ordinary fallback", fallback_after_minutes=1)

    spec = load_spec(b3.FIXTURE)
    agent = FallbackAgent()
    outcome = Engine(
        _WithAllowance(MaintenanceOperation(spec, "V1"), allowance),
        agent,
        identity={"agent_id": agent.agent_id},
    ).run()
    assert agent.calls == allowance
    assert _invocation_indices(outcome) == list(range(1, allowance + 1))
    assert _violation_codes(outcome) == {"INVOCATION_LIMIT_EXCEEDED"}
    assert outcome.status == "operation_deadlock"


@pytest.mark.parametrize("entry", ["check", "generate"])
def test_b3_refuses_current_engine_before_generation(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("generation reached current runtime")

    monkeypatch.setattr(b3, "_build", forbidden)
    with pytest.raises(b3.RegenerationFailure, match="historical_runtime_required"):
        getattr(b3, entry)(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_exact_history_is_explicit_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import historical_runtime as history

    monkeypatch.setattr(history, "HISTORICAL_SOURCE", "0" * 40)
    with pytest.raises(
        history.HistoricalRuntimeUnavailable, match="historical_runtime_unavailable"
    ):
        history.run_historical_mock("b3", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data_filter_initially_absent", [False, True])
@pytest.mark.parametrize("missing", ["filter_keyword", "data_filter"])
def test_missing_tar_filter_refuses_before_archive_or_extraction(
    missing: str,
    data_filter_initially_absent: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests import historical_runtime as history

    if data_filter_initially_absent:
        monkeypatch.delattr(history.tarfile, "data_filter", raising=False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("archive or extraction reached without safe tar support")

    def legacy_extractall(self: Any, path: Any = ".", members: Any = None) -> None:
        forbidden()

    if missing == "filter_keyword":
        monkeypatch.setattr(history.tarfile.TarFile, "extractall", legacy_extractall)
    else:
        monkeypatch.delattr(history.tarfile, "data_filter", raising=False)
    monkeypatch.setattr(history.subprocess, "run", forbidden)
    with pytest.raises(
        history.HistoricalRuntimeUnavailable, match="tar_data_filter_required"
    ):
        history.run_historical_mock("b3", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_ci_uses_bundled_history_without_private_source_input() -> None:
    from tests import historical_runtime as history

    text = (history.ROOT / ".github/workflows/ci.yml").read_text()
    assert "OPERATEBENCH_HISTORICAL_SOURCE" not in text
    assert "Provision historical Git object" not in text
    workflow = yaml.safe_load(text)
    for step in workflow["jobs"]["test"]["steps"]:
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False


def test_alpha_build_pin_does_not_accept_current_runtime() -> None:
    from operatebench.version import OPERATEBENCH_VERSION
    from tools import matched_arm_alpha as alpha

    assert OPERATEBENCH_VERSION == "0.13.0"
    assert alpha.ENGINE_VERSION == "0.9.0"
    with pytest.raises(alpha.AlphaRefusal, match="pinned_build_control_changed"):
        alpha.check_pinned_build()


@pytest.mark.parametrize("args", [["--check"], ["--output-dir"]])
def test_b3_cli_refuses_without_writing(
    args: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invocation = args if args == ["--check"] else [*args, str(tmp_path)]
    assert b3.main(invocation) == 1
    captured = capsys.readouterr()
    assert "historical_runtime_required" in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []
