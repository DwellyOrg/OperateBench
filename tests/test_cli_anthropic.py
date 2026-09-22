"""``run-suite --adapter anthropic``.

No test here contacts Anthropic. The ones that execute a run replace the SDK
client with the in-process transport from ``tests.anthropic_transport``, which
is the only boundary that is faked: the CLI, the manifest, the runner, the
environment, the evaluator and the ledger are all the real ones.

What these prove is that a credentialed operator's invocation is well-formed and
that an uncredentialed one costs them an error rather than a half-written run.
They are not evidence that any model was run, because none was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.cli import main
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    LIVE_CONFIGURATION_VARIABLE,
)
from boundarybench.providers.anthropic_messages import (
    AUDITED_CONFIGURATIONS_VARIABLE as AUTHORISED_CONFIGURATIONS_VARIABLE,
)
from boundarybench.runmanifest import (
    RUN_STATUS_PROVIDER_EXECUTION,
    RUN_TRACK_PROVIDER_EXECUTION,
)
from tests.anthropic_transport import (
    RecordingTransport,
    message_body,
    tool_use_block,
)
from tests.conftest import SUITE_MANIFEST

MODEL = "claude-test-20990101"

#: Clearly synthetic, purely local configuration identities. They have the shape
#: a ``configuration_id`` has and are values none ever is: no suite, scaffold or
#: settings hash to either. They exist so a test can authorise *something*
#: without this tree carrying the identity of a run that happened.
SYNTHETIC_CONFIGURATION = "0" * 64
OTHER_SYNTHETIC_CONFIGURATION = "1" * 64


#: One answer that every case admits and the loop can always dispatch. Choosing
#: an always-valid *action* keeps these tests about the plumbing: the verdict is
#: the evaluator's business and is not what is under test here.
STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


def _argv(output: Path, *extra: str) -> list[str]:
    return [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        MODEL,
        "--output-dir",
        str(output),
        *extra,
    ]


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> RecordingTransport:
    """Point the CLI's adapter factory at an in-process provider."""
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter

    transport = RecordingTransport([], default=message_body([STOP_CALL]))
    client = transport.client()

    def factory(*, model: str, **_kwargs: Any) -> AnthropicMessagesAdapter:
        return AnthropicMessagesAdapter(model=model, client=client, **overrides)

    monkeypatch.setattr("boundarybench.cli.build_anthropic_adapter", factory)
    return transport


# -- configuration refusals --------------------------------------------------


def test_anthropic_without_a_model_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    code = main(
        [
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            "anthropic",
            "--output-dir",
            str(root),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "--model" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_an_unauthorised_configuration_is_refused_before_the_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default-deny, and denied first.

    A published build authorises no configuration for live provider traffic, so
    an invocation that names none is refused before ``ANTHROPIC_API_KEY`` is
    read — with a credential sitting in the environment, and with the refusal
    naming the authorisation rather than the credential. Nothing is written.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.delenv(AUTHORISED_CONFIGURATIONS_VARIABLE, raising=False)
    monkeypatch.delenv(LIVE_CONFIGURATION_VARIABLE, raising=False)
    root = tmp_path / "run"

    code = main(_argv(root))

    captured = capsys.readouterr()
    assert code == 1
    assert AUTHORISED_CONFIGURATIONS_VARIABLE in captured.err
    assert "ANTHROPIC_API_KEY" not in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_a_configuration_outside_the_authorised_set_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An authorisation for one configuration authorises no other."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.setenv(AUTHORISED_CONFIGURATIONS_VARIABLE, SYNTHETIC_CONFIGURATION)
    monkeypatch.setenv(LIVE_CONFIGURATION_VARIABLE, OTHER_SYNTHETIC_CONFIGURATION)
    root = tmp_path / "run"

    code = main(_argv(root))

    captured = capsys.readouterr()
    assert code == 1
    assert OTHER_SYNTHETIC_CONFIGURATION in captured.err
    assert SYNTHETIC_CONFIGURATION not in captured.err
    assert not root.exists()


def test_a_missing_credential_costs_an_error_and_not_a_half_written_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential probe: nonzero, no traceback, no secret, no evidence.

    Authorised first, with a synthetic local identity, so the run reaches the
    credential at all: the audited-configuration gate is deliberately earlier.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(AUTHORISED_CONFIGURATIONS_VARIABLE, SYNTHETIC_CONFIGURATION)
    monkeypatch.setenv(LIVE_CONFIGURATION_VARIABLE, SYNTHETIC_CONFIGURATION)
    root = tmp_path / "run"

    code = main(_argv(root))

    captured = capsys.readouterr()
    assert code == 1
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "Traceback" not in captured.err
    assert not root.exists()


def test_the_fake_adapter_still_refuses_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``fake-scripted`` names one fixed model; it is not the operator's to set."""
    code = main(
        [
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            "fake-scripted",
            "--model",
            MODEL,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "--model" in captured.err
    assert "Traceback" not in captured.err


def test_arbitrary_settings_are_refused_for_a_real_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned settings are the experiment, and a free-text setting is a leak.

    ``--setting`` values are operator-supplied strings that land in the run
    manifest. The secret screen refuses credential-*looking* keys, but nothing
    stops a credential being pasted under an innocuous name, so a provider run
    takes no settings at all.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    code = main(_argv(tmp_path / "run", "--setting", "note=hello"))

    captured = capsys.readouterr()
    assert code == 1
    assert "--setting" in captured.err
    assert "Traceback" not in captured.err


# -- an executed run ---------------------------------------------------------


def test_a_provider_run_records_its_pinned_identity_and_no_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-ant-not-a-real-key-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    transport = _install_transport(monkeypatch)
    root = tmp_path / "run"

    assert main(_argv(root, "--json")) == 0

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"] == {
        "provider": "anthropic",
        "model": MODEL,
        "implementation": "anthropic_messages",
        "version": ANTHROPIC_ADAPTER_VERSION,
    }
    # And the run says what kind of run it is, rather than borrowing the fake
    # track's "no model was executed".
    assert manifest["track"] == RUN_TRACK_PROVIDER_EXECUTION
    assert manifest["status"] == RUN_STATUS_PROVIDER_EXECUTION
    settings = manifest["adapter_settings"]
    assert settings["max_output_tokens"] == 1024
    assert settings["temperature"] == 0.0
    assert settings["sdk_max_retries"] == 0
    assert settings["retry"]["max_attempts"] == 3
    assert settings["sdk"] == "anthropic"
    assert settings["sdk_version"]

    # Twelve variants, one turn each, all dispatched through the real loop.
    assert transport.calls == 12
    written = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.iterdir()
        if path.is_file()
    )
    assert secret not in written


def test_provider_token_and_latency_telemetry_reaches_the_run_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    _install_transport(monkeypatch)

    assert main(_argv(tmp_path / "run", "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    totals = payload["usage_totals"]
    assert totals["input_tokens"] == 137 * 12
    assert totals["output_tokens"] == 29 * 12
    assert totals["latency_seconds"] is not None
    # Null, and rendered as null: there is no reviewed pricing source in this
    # repository, so a number here would be an invented measurement.
    assert totals["cost_usd"] is None
    assert payload["usage_measured_episodes"]["cost_usd"] == 0


def test_an_identical_provider_rerun_resumes_without_calling_the_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    _install_transport(monkeypatch)
    assert main(_argv(root)) == 0
    before = (root / "episodes.jsonl").read_bytes()

    second = _install_transport(monkeypatch)
    assert main(_argv(root)) == 0

    assert second.calls == 0
    assert (root / "episodes.jsonl").read_bytes() == before


def test_changing_the_model_is_a_different_run_and_the_resume_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    _install_transport(monkeypatch)
    assert main(_argv(root)) == 0
    capsys.readouterr()

    _install_transport(monkeypatch)
    argv = _argv(root)
    argv[argv.index("--model") + 1] = "claude-other-20990101"
    code = main(argv)

    captured = capsys.readouterr()
    assert code == 1
    assert "already holds configuration" in captured.err
    assert "Traceback" not in captured.err


def test_changing_the_retry_policy_is_a_different_run_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The provider retry policy is inside run identity, so drift blocks a resume."""
    from boundarybench.providers.anthropic_messages import AnthropicRetryPolicy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    _install_transport(monkeypatch)
    assert main(_argv(root)) == 0
    capsys.readouterr()

    _install_transport(monkeypatch, retry=AnthropicRetryPolicy(max_attempts=5))
    code = main(_argv(root))

    captured = capsys.readouterr()
    assert code == 1
    assert "already holds configuration" in captured.err
    assert "Traceback" not in captured.err


def test_a_pinned_output_ceiling_is_not_mistaken_for_a_credential() -> None:
    """``max_output_tokens`` is a number, and a number is not a secret."""
    from boundarybench.runmanifest import RunSecretError, normalized_adapter_settings

    assert normalized_adapter_settings({"max_output_tokens": 1024}) == {
        "max_output_tokens": 1024
    }
    # The exemption is exact, and it is paired with the value's type: anything
    # a credential could actually be is still refused under the same key.
    with pytest.raises(RunSecretError):
        normalized_adapter_settings({"max_output_tokens": "sk-ant-not-a-real-key"})
    with pytest.raises(RunSecretError):
        normalized_adapter_settings({"max_output_tokens_secret": 1024})


def test_a_terminated_run_says_so_in_the_text_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run-wide failure is on the operator's screen, not only in the JSON.

    A refused credential stops the run after one recorded episode. The plain
    report is what an operator actually reads, and a run that stopped for a
    reason that applies to every remaining episode has to say which episode it
    stopped at and why — otherwise the visible sentence is "1 of 12 completed"
    with no explanation of the other eleven.
    """
    from boundarybench.ledger import OUTCOME_PROVIDER_AUTH_FAILURE
    from boundarybench.providers.anthropic_messages import AnthropicMessagesAdapter
    from tests.anthropic_transport import error_body

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    transport = RecordingTransport([(401, error_body())] * 12)
    client = transport.client()
    monkeypatch.setattr(
        "boundarybench.cli.build_anthropic_adapter",
        lambda *, model, **_kwargs: AnthropicMessagesAdapter(
            model=model, client=client, sleep=lambda seconds: None
        ),
    )
    root = tmp_path / "run"

    code = main(_argv(root))

    out = capsys.readouterr().out
    assert code == 1
    assert transport.calls == 1
    assert "run terminated" in out
    assert OUTCOME_PROVIDER_AUTH_FAILURE in out
    assert "planned            12" in out
    assert "completed          1" in out
    # The episode it stopped at is named, so the ledger row can be found.
    assert (
        json.loads((root / "episodes.jsonl").read_text(encoding="utf-8"))["episode_id"]
        in out
    )
