"""Staged execution of one immutable plan, one invocation at a time.

An authorisation may permit an immutable plan to execute across multiple bounded
invocations. The run retains one directory, manifest, configuration identity,
execution identity and set of caps throughout.

``--stop-after`` is the whole of that control. It bounds how many *new* episodes
one invocation executes and nothing else. These tests hold it to that:

* the exact number of provider calls and durable rows an invocation may produce;
* that what it executes is the plan's strict prefix, episode for episode;
* that repeated invocations extend the same durable prefix in place;
* that it is nowhere in the manifest or in a row, because it is not a setting
  the run was configured with;
* that it can only stop an invocation earlier, never authorise an episode or a
  dollar that the run's own durable limits refuse.

Every provider run here is executed by the real runner through the real
Anthropic adapter over the in-process ``httpx.MockTransport``. No socket is
opened and no credential is read.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import PolicyFollowingFakeAdapter, fake_identity
from boundarybench.budget import CostControls, RunCostGuard
from boundarybench.cli import main
from boundarybench.ledger import OUTCOME_COST_CAP_REACHED, LedgerError
from boundarybench.pricing import price_for
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    ANTHROPIC_IMPLEMENTATION,
    ANTHROPIC_PROVIDER,
    MAX_OUTPUT_TOKENS,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
)
from boundarybench.runmanifest import (
    MANIFEST_FILENAME,
    RunLimits,
    build_run_manifest,
    open_run_session,
)
from boundarybench.runner import RunnerError, execute_run
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.anthropic_transport import RecordingTransport, message_body, tool_use_block
from tests.conftest import SUITE_MANIFEST
from tests.test_runtime_execution_integrity import _rewrite_first_row

SONNET = "claude-sonnet-5"
MAX_TURNS = 4

#: One well-formed terminal call, so one episode costs exactly one provider
#: request and the call count is the episode count.
STOP_CALL = tool_use_block(
    "complete_case",
    {
        "disposition": "STOP",
        "primary_reason_code": "NO_APPLICABLE_RULE",
        "secondary_reason_codes": [],
        "evidence_refs": [],
    },
)


# -- provider runs, over the in-process transport ----------------------------


def _controls(*, cap: str | None = None, episodes: int | None = 36) -> CostControls:
    return CostControls(
        max_cost_usd=None if cap is None else Decimal(cap),
        max_episodes=episodes,
        price=None if cap is None else price_for(provider="anthropic", model=SONNET),
    )


def _provider_manifest(*, trials: int = 3, controls: CostControls) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        provider=ANTHROPIC_PROVIDER,
        model=SONNET,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=ANTHROPIC_ADAPTER_VERSION,
        adapter_settings=AnthropicMessagesAdapter(
            model=SONNET, client=RecordingTransport([]).client()
        ).settings,
        trials=trials,
        limits=RunLimits(
            max_turns=MAX_TURNS, max_messages=16, episode_timeout_seconds=30.0
        ),
        cost_controls=controls,
    )


def _tightened(controls: CostControls, episodes: int) -> CostControls:
    return CostControls(
        max_cost_usd=controls.max_cost_usd,
        max_episodes=episodes,
        price=controls.price,
    )


def _rehashed(manifest: Any, controls: CostControls) -> Any:
    """A manifest whose episode ceiling sits below its own plan, rehashed.

    ``build_run_manifest`` refuses to produce one — a plan larger than the
    authorisation is refused before a directory exists — so reaching the
    runner's own execution ceiling, the backstop behind that check, takes this.
    Nothing but the precedence probe below uses it.
    """
    import dataclasses

    from boundarybench.runmanifest import configuration_digest, execution_digest_of

    tightened = dataclasses.replace(manifest, cost_controls=controls, configuration_id="")
    identity = configuration_digest(tightened.configuration_payload())
    return dataclasses.replace(
        tightened,
        configuration_id=identity,
        execution_digest=execution_digest_of(
            configuration_id=identity,
            execution_id=tightened.execution_id,
            created_at_utc=tightened.created_at_utc,
        ),
    )


class Staged:
    """One run directory, executed across as many invocations as a test wants.

    The manifest is built once and reused, because that is the claim under
    test: staging is several invocations of *one* run, not several runs.
    """

    def __init__(
        self,
        root: Path,
        *,
        trials: int = 3,
        controls: CostControls | None = None,
        tighten_episodes: int | None = None,
    ) -> None:
        self.root = root
        self.suite = validate_suite(SUITE_MANIFEST)
        self.scaffold = load_scaffold(STANDARD_SCAFFOLD)
        self.controls = _controls() if controls is None else controls
        self.manifest = _provider_manifest(trials=trials, controls=self.controls)
        if tighten_episodes is not None:
            self.controls = _tightened(self.controls, tighten_episodes)
            self.manifest = _rehashed(self.manifest, self.controls)
        self.transport = RecordingTransport(
            [], default=message_body([STOP_CALL], model=SONNET)
        )

    @property
    def ledger(self) -> Path:
        return self.root / "episodes.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    def invoke(self, *, stop_after: int | None = None) -> Any:
        guard = None
        if self.controls.enforces_cost:
            guard = RunCostGuard(
                controls=self.controls,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                max_attempts_per_turn=AnthropicRetryPolicy().max_attempts,
            )
        adapter = AnthropicMessagesAdapter(
            model=SONNET,
            client=self.transport.client(),
            cost_guard=guard,
            sleep=lambda seconds: None,
        )
        with open_run_session(self.root, self.manifest) as session:
            return execute_run(
                suite=self.suite,
                scaffold=self.scaffold,
                adapter=adapter,
                session=session,
                cost_guard=guard,
                stop_after=stop_after,
            )

    def rows(self) -> list[dict[str, Any]]:
        text = self.ledger.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line]


def test_stop_after_one_makes_exactly_one_provider_episode(tmp_path: Path) -> None:
    """The canary: one episode, one request, one row, and then nothing.

    Killing the process after the first response is the alternative this
    replaces, and it is not an alternative: it leaves the lock held, the report
    unwritten and the operator unable to say whether a second request had
    already left.
    """
    staged = Staged(tmp_path / "run")

    report = staged.invoke(stop_after=1)

    assert staged.transport.calls == 1
    assert len(staged.rows()) == 1
    assert report.executed_count == 1
    assert report.resumed_count == 0
    # The plan is untouched: 36 episodes were planned and 35 are still pending.
    assert report.planned_count == 36
    assert report.completed_count == 1


def test_the_executed_episodes_are_the_plans_strict_prefix(tmp_path: Path) -> None:
    """Not *an* episode: the *next* one, in plan order, every time."""
    staged = Staged(tmp_path / "run")
    planned = [entry.episode_id for entry in staged.manifest.episode_plan]

    staged.invoke(stop_after=1)
    assert [row["episode_id"] for row in staged.rows()] == planned[:1]

    staged.invoke(stop_after=11)
    assert [row["episode_id"] for row in staged.rows()] == planned[:12]


def test_one_then_eleven_then_the_rest_is_a_single_staged_run(
    tmp_path: Path,
) -> None:
    """Several bounded invocations extend one directory under one manifest."""
    staged = Staged(tmp_path / "run")

    canary = staged.invoke(stop_after=1)
    assert (staged.transport.calls, len(staged.rows())) == (1, 1)

    block = staged.invoke(stop_after=11)
    assert (staged.transport.calls, len(staged.rows())) == (12, 12)
    assert block.executed_count == 11
    assert block.resumed_count == 1

    rest = staged.invoke()
    assert (staged.transport.calls, len(staged.rows())) == (36, 36)
    assert rest.executed_count == 24
    assert rest.resumed_count == 12
    assert rest.completed_count == rest.planned_count == 36

    # One run throughout: same configuration, same execution, same creation time.
    for report in (block, rest):
        assert report.configuration_id == canary.configuration_id
        assert report.execution_id == canary.execution_id
        assert report.manifest.created_at_utc == canary.manifest.created_at_utc


def test_staging_only_ever_appends_to_the_bytes_already_written(
    tmp_path: Path,
) -> None:
    """Every persisted prefix survives later invocations byte for byte.

    A staged run only appends; it never rewrites evidence already present.
    """
    staged = Staged(tmp_path / "run")

    staged.invoke(stop_after=1)
    after_canary = staged.ledger.read_bytes()
    manifest_bytes = staged.manifest_path.read_bytes()

    staged.invoke(stop_after=11)
    after_block = staged.ledger.read_bytes()
    assert after_block.startswith(after_canary)

    staged.invoke()
    final = staged.ledger.read_bytes()
    assert final.startswith(after_block)
    # And the manifest is the same file it was before the first episode ran.
    assert staged.manifest_path.read_bytes() == manifest_bytes


def test_the_invocation_limit_is_in_no_durable_artefact(tmp_path: Path) -> None:
    """It is staging, not a setting: nothing on disk records that it was used.

    A ``stop_after`` field in the manifest would be a false immutable run
    setting — it would change configuration identity, so the resume that
    completes the run would be refused as a different run, and a reader would
    see a limit that bounded one invocation recorded as a limit the whole run
    was configured under.
    """
    staged = Staged(tmp_path / "run")
    staged.invoke(stop_after=1)

    manifest_text = staged.manifest_path.read_text(encoding="utf-8")
    ledger_text = staged.ledger.read_text(encoding="utf-8")

    for text in (manifest_text, ledger_text):
        assert "stop_after" not in text
        assert "stop-after" not in text
        assert "invocation_staging" not in text
    # The durable ceiling is still the authorised one, not the staged one.
    assert json.loads(manifest_text)["cost_controls"]["max_episodes"] == 36


def test_an_unstaged_run_and_a_staged_one_reach_the_same_ledger(
    tmp_path: Path,
) -> None:
    """Staging changes when rows are written, never what they say.

    Two run directories, one executed in three stages and one in a single
    invocation, differ only in the fields that are timestamps and identifiers of
    *this* execution. Every field that describes the episode is identical.
    """
    # Everything that measures *this* execution rather than the episode: when it
    # ran, how long it took, and which execution it belonged to. Latency is the
    # reason ``provider_telemetry`` is reduced to its two countable fields.
    volatile = {
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "execution_id",
        "execution_digest",
        "row_digest",
    }

    def stable(row: dict[str, Any]) -> dict[str, Any]:
        kept = {k: v for k, v in row.items() if k not in volatile}
        kept["usage"] = {k: v for k, v in kept["usage"].items() if k != "latency_seconds"}
        telemetry = kept.pop("provider_telemetry")
        kept["attempts"] = telemetry["attempt_count"]
        kept["measurement_status"] = telemetry["measurement_status"]
        return kept

    staged = Staged(tmp_path / "staged")
    staged.invoke(stop_after=1)
    staged.invoke(stop_after=11)
    staged.invoke()

    whole = Staged(tmp_path / "whole")
    whole.invoke()

    assert len(staged.rows()) == len(whole.rows()) == 36
    for left, right in zip(staged.rows(), whole.rows(), strict=True):
        assert stable(left) == stable(right)


def test_fewer_episodes_remaining_than_asked_for_is_not_an_error(
    tmp_path: Path,
) -> None:
    """A limit larger than the work left executes the work left and returns."""
    staged = Staged(tmp_path / "run", trials=1)

    report = staged.invoke(stop_after=1000)

    assert report.executed_count == report.planned_count == 12
    assert staged.transport.calls == 12
    # Nothing was withheld, so nothing is reported as withheld.
    assert report.as_dict()["invocation_staging"]["stopped_early"] is False


def test_a_completed_run_is_still_a_no_op_to_repeat_under_a_limit(
    tmp_path: Path,
) -> None:
    """Resume decides on rows, not on the limit: a finished run stays finished."""
    staged = Staged(tmp_path / "run", trials=1)
    staged.invoke()
    before = staged.ledger.read_bytes()

    report = staged.invoke(stop_after=5)

    assert report.executed_count == 0
    assert report.resumed_count == 12
    assert staged.transport.calls == 12
    assert staged.ledger.read_bytes() == before


# -- the limit cannot borrow anything ----------------------------------------


def test_the_authorised_episode_ceiling_stops_the_run_before_the_staging_limit(
    tmp_path: Path,
) -> None:
    """A staging limit above a hard ceiling does not raise the ceiling.

    The report names the hard limit, not the staging one, because that is the
    limit that actually stopped this run and the one that requires a new
    configuration to raise.
    """
    staged = Staged(tmp_path / "run", trials=1, tighten_episodes=2)

    report = staged.invoke(stop_after=9)

    assert report.executed_count == 2
    assert staged.transport.calls == 2
    assert report.as_dict()["limits_stop"]["reason"] == "episode_limit"


def test_the_cost_cap_stops_the_run_and_the_staging_limit_does_not_override_it(
    tmp_path: Path,
) -> None:
    """An unaffordable first request is a recorded, run-terminal stop.

    ``--stop-after 9`` asks for nine episodes; the cap refuses the first
    request, so one row is written and the run is over. Staging cannot buy the
    other eight.
    """
    staged = Staged(
        tmp_path / "run", trials=1, controls=_controls(cap="0.000001", episodes=12)
    )

    report = staged.invoke(stop_after=9)

    assert staged.transport.calls == 0
    assert len(staged.rows()) == 1
    assert report.records[0].outcome == OUTCOME_COST_CAP_REACHED


def test_a_tampered_ledger_is_still_refused_when_a_stage_resumes(
    tmp_path: Path,
) -> None:
    """Strict resume is checked before the limit is consulted, not after."""
    staged = Staged(tmp_path / "run", trials=1)
    staged.invoke(stop_after=1)
    _rewrite_first_row(
        staged.ledger, lambda row: row.update(variant_content_digest="0" * 64)
    )
    tampered = staged.ledger.read_bytes()

    with pytest.raises(LedgerError):
        staged.invoke(stop_after=1)

    assert staged.transport.calls == 1
    assert staged.ledger.read_bytes() == tampered


# -- malformed limits are refused before anything happens --------------------


@pytest.mark.parametrize("value", [0, -1, -36, True, False, 1.0, 1.5, "1", Decimal(1)])
def test_a_limit_that_is_not_a_positive_whole_number_is_refused(
    tmp_path: Path, value: Any
) -> None:
    """Refused before the lock, before the ledger and before any request.

    ``True`` is in this list because ``bool`` is a subclass of ``int``: a
    ``--stop-after`` wired up from a flag would otherwise mean "execute one
    episode" and read as "no limit".
    """
    staged = Staged(tmp_path / "run", trials=1)

    with pytest.raises(RunnerError) as excinfo:
        staged.invoke(stop_after=value)

    assert "stop" in str(excinfo.value).lower()
    assert staged.transport.calls == 0
    assert not staged.ledger.exists()


# -- the deterministic fake, through the CLI ---------------------------------


def _argv(output: Path, *extra: str) -> list[str]:
    return [
        "run-suite",
        str(SUITE_MANIFEST),
        "--adapter",
        "fake-scripted",
        "--output-dir",
        str(output),
        "--trials",
        "3",
        *extra,
    ]


def _json_run(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any]]:
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_cli_stop_after_stages_one_then_twelve_then_thirty_six(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    ledger = root / "episodes.jsonl"

    code, canary = _json_run(_argv(root, "--stop-after", "1"), capsys)
    assert code == 0
    assert canary["counts"]["executed"] == 1
    assert canary["counts"]["completed"] == 1
    assert canary["counts"]["planned"] == 36
    after_canary = ledger.read_bytes()

    code, block = _json_run(_argv(root, "--stop-after", "11"), capsys)
    assert code == 0
    assert block["counts"]["executed"] == 11
    assert block["counts"]["completed"] == 12
    assert ledger.read_bytes().startswith(after_canary)
    after_block = ledger.read_bytes()

    code, rest = _json_run(_argv(root), capsys)
    assert code == 0
    assert rest["counts"]["executed"] == 24
    assert rest["counts"]["completed"] == 36
    assert ledger.read_bytes().startswith(after_block)

    # One run: the staging never minted a second identity.
    assert {block["configuration_id"], rest["configuration_id"]} == {
        canary["configuration_id"]
    }
    assert {block["execution_id"], rest["execution_id"]} == {canary["execution_id"]}


def test_cli_reports_the_limit_as_staging_and_not_as_a_run_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _json_run(_argv(tmp_path / "run", "--stop-after", "1"), capsys)

    assert code == 0
    staging = payload["invocation_staging"]
    assert staging["stop_after"] == 1
    assert staging["stopped_early"] is True
    assert "not part of" in staging["note"]
    # Not smuggled into the durable settings block.
    assert "stop_after" not in json.dumps(payload["cost_controls"])
    assert "stop_after" not in json.dumps(payload["limits"])


def test_cli_text_report_shows_the_staging_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv(tmp_path / "run", "--stop-after", "1")) == 0

    out = capsys.readouterr().out
    assert "staging" in out
    assert "--stop-after 1" in out
    assert "planned            36" in out
    # And the line says, in the report itself, that it is not a run setting.
    assert "not part of" in out


def test_an_unstaged_cli_run_says_it_was_not_staged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _json_run(_argv(tmp_path / "run", "--trials", "1"), capsys)

    assert code == 0
    assert payload["invocation_staging"]["stop_after"] is None
    assert payload["invocation_staging"]["stopped_early"] is False


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "1.5", "1.0", "abc", "true", "", " 1", "+1", "1_0", "1e2", "０１"],  # noqa: RUF001 -- Unicode digits must be refused
)
def test_cli_refuses_a_malformed_limit_before_it_creates_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    root = tmp_path / "run"

    code = main(_argv(root, "--stop-after", value))

    captured = capsys.readouterr()
    assert code == 1
    assert "--stop-after" in captured.err
    assert captured.out == ""
    # No run directory, so no manifest, no ledger and no lock.
    assert not root.exists()


def test_cli_refuses_a_malformed_limit_before_it_builds_a_provider_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal happens before the credential is read, not after."""
    built: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        built.append("client")
        raise AssertionError("a client was built for a refused invocation")

    monkeypatch.setattr("boundarybench.cli.build_anthropic_adapter", refuse)

    code = main(
        [
            "run-suite",
            str(SUITE_MANIFEST),
            "--adapter",
            "anthropic",
            "--model",
            SONNET,
            "--output-dir",
            str(tmp_path / "run"),
            "--stop-after",
            "0",
        ]
    )

    assert code == 1
    assert built == []
    assert "--stop-after" in capsys.readouterr().err


def test_the_fake_adapter_executes_exactly_the_limit(tmp_path: Path) -> None:
    """Same control, same numbers, with no provider anywhere near it."""
    suite = validate_suite(SUITE_MANIFEST)
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    identity = fake_identity()
    manifest = build_run_manifest(
        suite=suite,
        scaffold=scaffold,
        provider=identity.provider,
        model=identity.model,
        implementation=identity.implementation,
        adapter_version=identity.version,
        adapter_settings={},
        trials=1,
        limits=RunLimits(max_turns=8, max_messages=32, episode_timeout_seconds=30.0),
    )
    with open_run_session(tmp_path / "run", manifest) as session:
        report = execute_run(
            suite=suite,
            scaffold=scaffold,
            adapter=PolicyFollowingFakeAdapter(settings={}),
            session=session,
            stop_after=3,
        )

    assert report.executed_count == 3
    assert report.planned_count == 12
    assert report.as_dict()["invocation_staging"]["stopped_early"] is True


# -- preflight stays a zero-call, zero-output check ---------------------------


def _preflight_argv(output: Path, *extra: str) -> list[str]:
    return [
        "preflight",
        str(SUITE_MANIFEST),
        "--adapter",
        "anthropic",
        "--model",
        SONNET,
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


def test_preflight_reports_the_full_plan_and_the_requested_staging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan a reviewer approves is 36 episodes whatever a stage asks for."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    root = tmp_path / "run"
    root.mkdir()
    root.chmod(0o700)
    transport = RecordingTransport([], default=message_body([STOP_CALL], model=SONNET))
    monkeypatch.setattr(
        "boundarybench.cli.build_anthropic_adapter",
        lambda **kwargs: AnthropicMessagesAdapter(
            model=kwargs["model"], client=transport.client()
        ),
    )

    code = main([*_preflight_argv(root, "--stop-after", "1"), "--json"])
    staged = json.loads(capsys.readouterr().out)

    assert code == 0
    assert transport.calls == 0
    assert staged["plan"]["planned_episodes"] == 36
    assert staged["invocation_staging"]["stop_after"] == 1

    # And nothing was written: no manifest, no ledger, no lock.
    assert list(root.iterdir()) == []

    code = main([*_preflight_argv(root), "--json"])
    plain = json.loads(capsys.readouterr().out)
    assert code == 0
    assert plain["invocation_staging"]["stop_after"] is None
    # The staging limit is not in run identity, so both plans are one plan.
    assert staged["configuration_id"] == plain["configuration_id"]


def test_preflight_refuses_a_malformed_limit_without_reading_a_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = tmp_path / "run"
    root.mkdir()
    root.chmod(0o700)

    code = main(_preflight_argv(root, "--stop-after", "0"))

    captured = capsys.readouterr()
    assert code == 1
    # The limit is refused first, so the message is about the limit and not
    # about the missing credential.
    assert "--stop-after" in captured.err
    assert "ANTHROPIC_API_KEY" not in captured.err
    assert captured.out == ""
