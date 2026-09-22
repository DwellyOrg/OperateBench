"""``run-suite`` under explicit cost and episode limits.

No test here contacts Anthropic: the ones that execute replace the SDK client
with the in-process transport, exactly as ``tests/test_cli_anthropic.py`` does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.cli import main
from boundarybench.providers.anthropic_messages import (
    AUDITED_CONFIGURATIONS_VARIABLE as AUTHORISED_CONFIGURATIONS_VARIABLE,
)
from boundarybench.providers.anthropic_messages import (
    LIVE_CONFIGURATION_VARIABLE,
)
from tests.anthropic_transport import RecordingTransport, message_body, tool_use_block
from tests.conftest import SUITE_MANIFEST

SONNET = "claude-sonnet-5"

#: A clearly synthetic, purely local configuration identity: the shape a
#: ``configuration_id`` has, and a value none ever is.
SYNTHETIC_CONFIGURATION = "0" * 64

STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


def _argv(output: Path, *extra: str, model: str = SONNET) -> list[str]:
    return [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        model,
        "--output-dir",
        str(output),
        *extra,
    ]


def _install_transport(monkeypatch: pytest.MonkeyPatch) -> RecordingTransport:
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    client = transport.client()

    def factory(*, model: str, **kwargs: Any) -> AnthropicMessagesAdapter:
        return AnthropicMessagesAdapter(
            model=model,
            client=client,
            cost_guard=kwargs.get("cost_guard"),
            sleep=lambda seconds: None,
        )

    monkeypatch.setattr("boundarybench.cli.build_anthropic_adapter", factory)
    return transport


# -- refusals before anything exists -----------------------------------------


def test_an_unpriced_model_under_a_cap_is_refused_before_anything_is_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No output directory, and no provider client either."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    built: list[str] = []
    monkeypatch.setattr(
        "boundarybench.cli.build_anthropic_adapter",
        lambda **kwargs: built.append("built"),
    )
    root = tmp_path / "run"

    code = main(
        _argv(root, "--max-cost-usd", "100", model="claude-opus-5"),
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "claude-opus-5" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()
    assert built == []


def test_a_malformed_cap_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"

    assert main(_argv(root, "--max-cost-usd", "lots")) == 1
    assert "--max-cost-usd" in capsys.readouterr().err
    assert not root.exists()


def test_a_zero_cap_is_refused_rather_than_read_as_unlimited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"

    assert main(_argv(root, "--max-cost-usd", "0")) == 1
    assert not root.exists()


def test_a_plan_larger_than_the_episode_limit_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    _install_transport(monkeypatch)
    root = tmp_path / "run"

    code = main(_argv(root, "--max-episodes", "5"))

    captured = capsys.readouterr()
    assert code == 1
    assert "authorised for at most 5" in captured.err
    assert not root.exists()


def test_the_fake_adapter_takes_no_cost_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fake buys nothing, so a dollar cap on one would be theatre."""
    code = main(
        [
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            "fake-scripted",
            "--output-dir",
            str(tmp_path / "run"),
            "--max-cost-usd",
            "100",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "--max-cost-usd" in captured.err
    assert "Traceback" not in captured.err


def test_the_fake_adapter_still_accepts_an_episode_ceiling(tmp_path: Path) -> None:
    """The episode ceiling is about the plan, and every run has one of those."""
    assert (
        main(
            [
                "run-suite",
                str(SUITE_MANIFEST),
                "--adapter",
                "fake-scripted",
                "--output-dir",
                str(tmp_path / "run"),
                "--max-episodes",
                "12",
            ]
        )
        == 0
    )


# -- an executed capped run --------------------------------------------------


def test_a_capped_run_records_its_limits_and_reports_its_spend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    transport = _install_transport(monkeypatch)
    root = tmp_path / "run"

    code = main(_argv(root, "--max-cost-usd", "100", "--max-episodes", "72", "--json"))

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert transport.calls == 12
    assert payload["cost_controls"]["max_cost_usd"] == "100"
    assert payload["cost_controls"]["max_episodes"] == 72
    assert payload["cost_accounting"]["measured_usd"] == "0.006768"
    assert payload["cost_accounting"]["exposure_usd"] == "0"
    assert payload["usage_totals"]["cost_usd"] == pytest.approx(0.006768)

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cost_controls"]["max_cost_usd"] == "100"
    assert manifest["cost_controls"]["pricing"]["model"] == SONNET


def test_the_text_report_shows_the_cap_and_what_is_left(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    _install_transport(monkeypatch)

    assert main(_argv(tmp_path / "run", "--max-cost-usd", "100")) == 0

    out = capsys.readouterr().out
    assert "cost cap" in out
    assert "measured" in out
    assert "exposure" in out


def test_resuming_a_capped_run_under_a_different_cap_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    _install_transport(monkeypatch)
    assert main(_argv(root, "--max-cost-usd", "100")) == 0
    capsys.readouterr()

    _install_transport(monkeypatch)
    code = main(_argv(root, "--max-cost-usd", "500"))

    captured = capsys.readouterr()
    assert code == 1
    assert "already holds configuration" in captured.err
    assert "Traceback" not in captured.err


def test_a_run_that_spends_its_cap_stops_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    transport = _install_transport(monkeypatch)
    root = tmp_path / "run"

    code = main(_argv(root, "--max-cost-usd", "0.000001"))

    out = capsys.readouterr().out
    assert code == 1
    assert transport.calls == 0
    assert "cost_cap_reached" in out
    assert "run terminated" in out


# -- what has not changed ----------------------------------------------------


def test_an_uncapped_provider_run_still_records_no_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a pinned price there is nothing to compute a cost from."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    _install_transport(monkeypatch)

    assert main(_argv(tmp_path / "run", "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["usage_totals"]["cost_usd"] is None
    assert payload["cost_accounting"] is None
    assert payload["cost_controls"] == {
        "max_cost_usd": None,
        "max_episodes": None,
        "pricing": None,
    }


def test_a_missing_credential_still_costs_an_error_and_no_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authorised with a synthetic local identity, so the credential is reached.

    The audited-configuration gate is deliberately earlier than the credential,
    so without this the refusal under test would never be the one that fires.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(AUTHORISED_CONFIGURATIONS_VARIABLE, SYNTHETIC_CONFIGURATION)
    monkeypatch.setenv(LIVE_CONFIGURATION_VARIABLE, SYNTHETIC_CONFIGURATION)
    root = tmp_path / "run"

    code = main(_argv(root, "--max-cost-usd", "100"))

    captured = capsys.readouterr()
    assert code == 1
    assert "ANTHROPIC_API_KEY" in captured.err
    assert not root.exists()
