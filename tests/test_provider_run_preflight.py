"""The no-network preflight for an authorised run plan.

Preflight answers, without contacting anything: *is this exactly the run that
was authorised, and does its worst case fit inside the budget that was
approved?* It reads the suite, the scaffold, the pricing policy and the caps,
it checks that a credential is **present** without reading its value, and it
checks that the output directory is a safe, private place to write evidence.

It never builds a provider client, never constructs a Messages request and never
creates a run directory, a manifest, a ledger or a lock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from boundarybench.cli import main
from tests.anthropic_transport import RecordingTransport, message_body, tool_use_block
from tests.conftest import SUITE_MANIFEST

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

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
        "preflight",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        model,
        "--output-dir",
        str(output),
        "--max-cost-usd",
        "100",
        "--max-episodes",
        "72",
        "--trials",
        "3",
        *extra,
    ]


def _private(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def _run(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any], str]:
    code = main(argv)
    captured = capsys.readouterr()
    payload: dict[str, Any] = {}
    if "--json" in argv and captured.out.strip():
        payload = json.loads(captured.out)
    return code, payload, captured.err


# -- no network --------------------------------------------------------------


def test_preflight_makes_no_provider_call_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof: an armed transport that is never touched.

    The transport is wired in the same way every executing test wires it, so a
    preflight that reached the SDK would answer through it and be counted here.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    client = transport.client()
    built: list[str] = []

    def factory(*, model: str, **kwargs: Any) -> AnthropicMessagesAdapter:
        built.append(model)
        return AnthropicMessagesAdapter(model=model, client=client)

    monkeypatch.setattr("boundarybench.cli.build_anthropic_adapter", factory)

    code, _payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    assert code == 0
    assert transport.calls == 0
    # And no client was constructed either: preflight does not need one, and a
    # client is where a credential goes.
    assert built == []


def test_preflight_never_constructs_a_request_or_a_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not sent is a weaker claim than not built, so building is barred too.

    The two seams a provider request has to pass through are booby-trapped: the
    function that serialises a Messages body, and the constructor of the SDK
    client that would carry it. A preflight that touched either fails here
    rather than reporting a clean plan.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    import anthropic

    from boundarybench.providers import anthropic_messages

    def refuse_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("preflight serialised a provider Messages request")

    def refuse_client(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("preflight constructed a provider client")

    monkeypatch.setattr(anthropic_messages, "build_messages_request", refuse_request)
    monkeypatch.setattr(anthropic, "Anthropic", refuse_client)

    code, payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    assert code == 0
    assert payload["ok"] is True


def test_preflight_creates_no_run_directory_or_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"

    code, _payload, _err = _run(_argv(root, "--json"), capsys)

    assert code == 0
    assert not root.exists()


def test_a_failing_preflight_leaves_nothing_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = tmp_path / "run"

    code, _payload, err = _run(_argv(root, "--json"), capsys)

    assert code == 1
    assert not root.exists()
    assert "ANTHROPIC_API_KEY" in err


# -- what it reports ---------------------------------------------------------


def test_the_plan_report_states_the_approved_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    code, payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    assert code == 0
    assert payload["ok"] is True
    assert payload["adapter"]["provider"] == "anthropic"
    assert payload["adapter"]["model"] == SONNET
    assert payload["plan"]["planned_episodes"] == 36
    assert payload["plan"]["trials"] == 3
    assert payload["plan"]["variants"] == 12
    assert payload["suite"]["suite_id"]
    assert len(payload["suite"]["suite_content_digest"]) == 64
    assert len(payload["scaffold"]["content_digest"]) == 64
    assert payload["cost_controls"]["max_cost_usd"] == "100"
    assert payload["cost_controls"]["max_episodes"] == 72
    assert payload["cost_controls"]["pricing"]["policy_version"]
    # The configuration this plan would execute under, so the plan a reviewer
    # approves and the run that happens can be compared by identity.
    assert len(payload["configuration_id"]) == 64


def test_the_report_carries_no_credential_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence only: the variable is named, its value never appears."""
    secret = "sk-ant-not-a-real-key-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    code, payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    assert code == 0
    assert payload["credentials"] == {
        "variable": "ANTHROPIC_API_KEY",
        "present": True,
        "note": payload["credentials"]["note"],
    }
    assert secret not in json.dumps(payload)


def test_the_report_states_the_frozen_model_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run pins these; the plan has to show them, not imply them.

    Two of them are model-specific, and point in opposite directions: Sonnet 5
    is asked with *no* sampling parameter — the pinned temperature is recorded
    as omitted rather than sent — and with an explicit
    ``thinking: {"type": "disabled"}``, because for this model an absent
    thinking field selects adaptive thinking, which a run may declare out of
    scope. The pinned scope is unchanged; what changes per model is the request
    that honours it. See ``tests/test_request_profiles.py``.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    _code, payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    settings = payload["settings"]
    assert settings["temperature"] is None
    assert settings["thinking"] == {"type": "disabled"}
    assert "temperature" in settings["request_fields_omitted"]
    assert "thinking" in settings["request_fields_sent"]
    assert settings["max_output_tokens"] == 1024
    assert settings["sdk_max_retries"] == 0
    assert payload["scope"]["optional_extended_thinking"] is False
    assert payload["scope"]["external_provider_tools"] is False
    # The only tools the run can offer are the scaffold's own actions.
    assert payload["scope"]["tools"] == sorted(payload["scope"]["tools"])
    assert "complete_case" in payload["scope"]["tools"]


def test_the_cost_bound_is_computed_and_compared_against_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    code, payload, _err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    bound = payload["cost_bound"]
    assert code == 0
    assert bound["planning_input_tokens_per_request"] > 0
    assert bound["max_requests"] == 36 * 12 * 3
    assert float(bound["upper_bound_usd"]) > 0
    assert float(bound["upper_bound_usd"]) <= 100
    assert bound["fits_within_cap"] is True


def test_a_plan_whose_worst_case_exceeds_the_cap_fails_the_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the exercise: the run is not started."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = _private(tmp_path / "run")

    code, _payload, err = _run(
        _argv(root, "--planning-input-tokens", "200000", "--json"), capsys
    )

    assert code == 1
    assert "upper bound" in err.lower()
    assert not (root / "run_manifest.json").exists()


# -- refusals ----------------------------------------------------------------


def test_an_unapproved_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    code, _payload, err = _run(
        _argv(_private(tmp_path / "run"), "--json", model="claude-opus-5"), capsys
    )

    assert code == 1
    assert "claude-opus-5" in err


def test_the_second_approved_model_passes_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    code, payload, _err = _run(
        _argv(_private(tmp_path / "run"), "--json", model=HAIKU), capsys
    )

    assert code == 0
    assert payload["cost_controls"]["pricing"]["input_usd_per_million_tokens"] == "1"


def test_a_plan_larger_than_the_episode_ceiling_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    argv = _argv(_private(tmp_path / "run"), "--json")
    argv[argv.index("--max-episodes") + 1] = "12"

    code, _payload, err = _run(argv, capsys)

    assert code == 1
    assert "authorised for at most 12" in err


def test_an_environment_that_would_redirect_the_request_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.invalid")

    code, _payload, err = _run(_argv(_private(tmp_path / "run"), "--json"), capsys)

    assert code == 1
    assert "ANTHROPIC_BASE_URL" in err


def test_a_world_readable_output_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trajectories are private research evidence, not a shared directory."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    root.mkdir()
    root.chmod(0o755)

    code, _payload, err = _run(_argv(root, "--json"), capsys)

    assert code == 1
    assert "output" in err.lower()
    assert oct(os.stat(root).st_mode)[-3:] == "755"


def test_an_output_path_behind_a_symlink_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    real = _private(tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(real)

    code, _payload, err = _run(_argv(link, "--json"), capsys)

    assert code == 1
    assert "symlink" in err.lower()


def test_preflight_requires_a_model_like_a_run_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "preflight",
            str(SUITE_MANIFEST),
            "--adapter",
            "anthropic",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert code == 1
    assert "--model" in capsys.readouterr().err


def test_the_text_report_is_readable_without_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    assert main(_argv(_private(tmp_path / "run"))) == 0

    out = capsys.readouterr().out
    assert "preflight OK" in out
    assert "planned episodes" in out
    assert "cost upper bound" in out
