"""``provider-matrix``: the limits it advertises are the limits it enforces.

An option a command accepts and ignores is worse than one it does not offer.
``--max-episodes`` is documented as a hard ceiling, so a plan that reports 180
episodes under ``--max-episodes 1`` and exits 0 has told an operator their
ceiling was applied when nothing checked it. ``--trials`` is the same fault in
the other direction: the header said three trials while every configuration it
generated planned twelve episodes rather than thirty-six, and the two numbers
that disagreed were printed side by side.

The third property here is the stage arithmetic. A cumulative per-model target
and a per-invocation episode count are different numbers, and handing the first
to ``--stop-after`` executes the prefix a second time. This command must print
both, name each one for what it is, and never instruct an operator to run 13
episodes against a 12-episode block.

Nothing here contacts a provider: the command is arithmetic and hashing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boundarybench.cli import main
from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    TOTAL_EPISODES,
    TRIALS,
)
from tests.conftest import SUITE_MANIFEST

GLOBAL_CAP = "50.00"


def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for entry in MATRIX_MODELS:
        monkeypatch.setenv(entry.api_key_variable, "not-a-real-key")


def _argv(root: Path, *extra: str) -> list[str]:
    return [
        "provider-matrix",
        str(SUITE_MANIFEST),
        "--output-root",
        str(root),
        "--max-cost-usd",
        GLOBAL_CAP,
        "--json",
        *extra,
    ]


def _plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
) -> dict[str, Any]:
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix", *extra)) == 0
    return json.loads(capsys.readouterr().out)


def _refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
) -> str:
    _credentials(monkeypatch)
    root = tmp_path / "matrix"
    assert main(_argv(root, *extra)) != 0
    assert not root.exists()
    return capsys.readouterr().err


# -- the global episode ceiling -----------------------------------------------


@pytest.mark.parametrize("supplied", ["1", "36", "179", "181"])
def test_an_episode_ceiling_other_than_the_whole_matrix_is_refused(
    supplied: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ceiling this command cannot honour is refused, never ignored."""
    error = _refused(tmp_path, capsys, monkeypatch, "--max-episodes", supplied)
    assert str(TOTAL_EPISODES) in error


def test_the_exact_episode_ceiling_is_accepted_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan(tmp_path, capsys, monkeypatch, "--max-episodes", "180")
    assert payload["global_episode_ceiling"] == TOTAL_EPISODES
    assert payload["total_episodes"] == TOTAL_EPISODES
    assert payload["per_model_run_episode_cap"] == EPISODES_PER_MODEL
    assert all(cell["episode_cap"] == EPISODES_PER_MODEL for cell in payload["models"])


def test_the_default_invocation_authorises_exactly_the_whole_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan(tmp_path, capsys, monkeypatch)
    assert payload["global_episode_ceiling"] == TOTAL_EPISODES


# -- the fixed trial count ----------------------------------------------------


@pytest.mark.parametrize("supplied", ["1", "2", "4"])
def test_a_trial_count_other_than_three_is_refused(
    supplied: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old failure printed "3 trials" beside twelve-episode configurations."""
    error = _refused(tmp_path, capsys, monkeypatch, "--trials", supplied)
    assert str(TRIALS) in error


def test_the_fixed_trial_count_plans_thirty_six_episodes_for_every_cell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header's arithmetic and the generated configurations are one claim."""
    payload = _plan(tmp_path, capsys, monkeypatch, "--trials", "3")
    assert payload["trials"] == TRIALS
    assert payload["episodes_per_model"] == EPISODES_PER_MODEL
    assert [cell["planned_episodes"] for cell in payload["models"]] == [
        EPISODES_PER_MODEL
    ] * len(MATRIX_MODELS)
    assert sum(cell["planned_episodes"] for cell in payload["models"]) == TOTAL_EPISODES


def test_the_default_trial_count_is_the_fixed_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan(tmp_path, capsys, monkeypatch)
    assert payload["trials"] == TRIALS
    assert all(
        cell["planned_episodes"] == EPISODES_PER_MODEL for cell in payload["models"]
    )


# -- targets, deltas, and the 13-episode instruction that must not exist ------


def test_the_json_names_the_cumulative_target_and_the_invocation_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan(tmp_path, capsys, monkeypatch, "--stage", "trial_block")
    assert payload["stage"] == "trial_block"
    assert payload["stage_cumulative_episode_target_per_model"] == 12
    assert payload["stage_new_episodes_per_model"] == 11
    assert payload["stage_is_manual"] is False
    assert "stage_episode_cap_per_model" not in payload
    stages = {entry["stage"]: entry for entry in payload["stages"]}
    assert stages["smoke"]["cumulative_episode_target_per_model"] == 1
    assert stages["smoke"]["new_episodes_per_model"] == 1
    assert stages["audit_gate"]["new_episodes_per_model"] == 0
    assert stages["remainder"]["new_episodes_per_model"] == 24


@pytest.mark.parametrize(
    ("stage", "delta"), [("smoke", 1), ("trial_block", 11), ("remainder", 24)]
)
def test_the_execution_refusal_instructs_the_delta_and_the_immutable_cap(
    stage: str,
    delta: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sentence an operator copies must be the one that runs the plan.

    ``--stop-after`` bounds *new* episodes in one invocation, so the number
    beside it is the stage's delta. The per-model manifest cap is 36 whatever
    stage is running — it is the run's own immutable authority, not a stage's —
    and the two numbers are named separately because an operator who swaps them
    either re-runs a prefix or authorises the wrong ceiling.
    """
    error = _refused(tmp_path, capsys, monkeypatch, "--execute", "--stage", stage)
    assert f"--stop-after {delta}" in error
    assert f"--max-episodes {EPISODES_PER_MODEL}" in error
    assert "13" not in error


def test_the_execution_refusal_never_hands_a_cumulative_target_to_stop_after(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 then 12 is 13 executed episodes against a 12-episode block."""
    error = _refused(tmp_path, capsys, monkeypatch, "--execute", "--stage", "trial_block")
    assert "--stop-after 12" not in error


def test_the_offline_stage_is_refused_as_one_that_contacts_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline stage has no run to instruct, so it is given no command."""
    error = _refused(
        tmp_path, capsys, monkeypatch, "--execute", "--stage", "offline_preflight"
    )
    assert "--stop-after" not in error
    assert "no episodes" in error


def test_the_manual_gate_is_refused_as_a_stage_that_executes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    error = _refused(tmp_path, capsys, monkeypatch, "--execute", "--stage", "audit_gate")
    assert "--stop-after" not in error
    assert "manual" in error.lower()


def test_the_human_readable_output_states_both_numbers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-JSON form is what most operators read, so it carries both too."""
    _credentials(monkeypatch)
    argv = [
        arg
        for arg in _argv(tmp_path / "matrix", "--stage", "trial_block")
        if arg != "--json"
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "cumulative" in out.lower()
    assert "11 new episode" in out
