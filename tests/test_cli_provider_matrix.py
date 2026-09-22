"""``boundarybench provider-matrix``: the offline planner for the five-model spike.

The two tests that matter most here prove negatives, and they prove them the
hard way rather than by inspection. ``test_a_dry_run_makes_zero_transport_calls``
breaks ``httpx`` at the socket boundary for the duration of the command, so any
request any SDK attempted — through any client, to any endpoint — fails the test
rather than quietly succeeding against a mock. And every refusal path asserts
that no run directory exists afterwards, because a directory left behind by a
refused invocation is a thing an audit can mistake for a run.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from boundarybench.cli import main
from boundarybench.providermatrix import EPISODES_PER_MODEL, MATRIX_MODELS
from tests.conftest import SUITE_MANIFEST

GLOBAL_CAP = "50.00"


@pytest.fixture
def no_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break every HTTP send for the duration of a test.

    Installed on ``httpx.Client.send`` rather than on any SDK's own entry point,
    because that is the one chokepoint all three SDKs in this build go through.
    A guard placed higher up would only prove that the code path a test happened
    to know about was not taken.
    """

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "a command that must contact nothing attempted an HTTP request"
        )

    monkeypatch.setattr(httpx.Client, "send", forbidden)


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


# -- the dry run --------------------------------------------------------------


def test_a_dry_run_makes_zero_transport_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    no_transport: None,
) -> None:
    """Planning is arithmetic and hashing. It contacts nothing, so it can prove it."""
    _credentials(monkeypatch)
    root = tmp_path / "matrix"

    assert main(_argv(root)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["live_enabled"] is False
    assert payload["total_episodes"] == EPISODES_PER_MODEL * 5


def test_a_dry_run_creates_no_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    root = tmp_path / "matrix"
    assert main(_argv(root)) == 0
    capsys.readouterr()
    assert not root.exists(), "a plan is not a run"


def test_the_dry_run_reports_a_configuration_identity_for_every_cell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identity an operator authorises has to be computable before spending.

    Five cells, five distinct identities: a shared one would let the audit of
    one model authorise a run against another.
    """
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    identities = [entry["configuration_id"] for entry in payload["models"]]
    assert len(identities) == 5
    assert all(len(identity) == 64 for identity in identities)
    assert len(set(identities)) == 5


def test_the_dry_run_never_prints_a_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-THIS-MUST-NEVER-BE-PRINTED"
    for entry in MATRIX_MODELS:
        monkeypatch.setenv(entry.api_key_variable, secret)
    assert main(_argv(tmp_path / "matrix")) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    payload = json.loads(captured.out)
    for entry in payload["models"]:
        assert entry["credential"]["present"] is True
        assert "value" not in entry["credential"]


def test_the_dry_run_states_the_stage_order_and_its_prefix_caps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    stages = [
        (
            entry["stage"],
            entry["cumulative_episode_target_per_model"],
            entry["new_episodes_per_model"],
        )
        for entry in payload["stages"]
    ]
    # The cumulative targets are the prefix chain; the deltas are what each
    # stage's own invocation executes, and they sum to the 36-episode plan
    # rather than to the 49 the targets add up to.
    assert stages == [
        ("offline_preflight", 0, 0),
        ("smoke", 1, 1),
        ("audit_gate", 1, 0),
        ("trial_block", 12, 11),
        ("remainder", 36, 24),
    ]
    assert sum(delta for _stage, _target, delta in stages) == 36


def test_the_dry_run_states_that_this_is_not_a_ranking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["semantic_units"] == 2
    assert "rank" in payload["methodology_note"].lower()


def test_the_dry_run_names_the_five_exact_models_and_flags_the_one_limitation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five exact identifiers, and the one that is not a dated snapshot says so.

    The plan an operator authorises is the artefact that has to carry this: four
    of the identifiers are dated or versioned by their providers, and ``grok-4.5``
    is the most exact string xAI's published model page exposes. A plan that
    presented all five as equally pinned would overstate what one of them proves.
    """
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["model"] for entry in payload["models"]] == [
        entry.model for entry in MATRIX_MODELS
    ]
    limited = {
        entry["model"]: entry["identifier_limitation"]
        for entry in payload["models"]
        if entry["identifier_limitation"] is not None
    }
    assert set(limited) == {"grok-4.5"}


def test_the_allocations_sum_to_the_authorised_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    total = sum(
        (Decimal(entry["cost_allocation_usd"]) for entry in payload["models"]),
        Decimal(0),
    )
    assert total == Decimal(GLOBAL_CAP) == Decimal(payload["total_cost_usd"])


# -- refusals -----------------------------------------------------------------


def test_a_missing_credential_is_reported_without_refusing_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline plan is exactly what an operator runs *before* they have keys.

    So a missing credential is a reported boolean rather than a refusal: the
    point of the stage is to compute the identities that need authorising, and
    demanding the credential first would invert the order the gate exists to
    impose.
    """
    for entry in MATRIX_MODELS:
        monkeypatch.delenv(entry.api_key_variable, raising=False)
    assert main(_argv(tmp_path / "matrix")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(entry["credential"]["present"] is False for entry in payload["models"])
    assert payload["credentials_ready"] is False


def test_a_cap_that_is_not_a_positive_amount_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    root = tmp_path / "matrix"
    code = main(
        [
            "provider-matrix",
            str(SUITE_MANIFEST),
            "--output-root",
            str(root),
            "--max-cost-usd",
            "0",
        ]
    )
    assert code != 0
    assert not root.exists()


def test_a_refusal_leaves_no_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirected endpoint invalidates the plan, and must leave nothing behind."""
    _credentials(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid")
    root = tmp_path / "matrix"

    code = main(_argv(root))

    assert code != 0
    assert "OPENAI_BASE_URL" in capsys.readouterr().err
    assert not root.exists()


def test_live_execution_is_refused_without_an_explicit_authorisation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    no_transport: None,
) -> None:
    """Default-deny, and the refusal happens before any client could exist."""
    _credentials(monkeypatch)
    root = tmp_path / "matrix"

    code = main(_argv(root, "--execute", "--stage", "smoke"))

    assert code != 0
    assert "authoris" in capsys.readouterr().err.lower()
    assert not root.exists()


def test_an_unknown_stage_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _credentials(monkeypatch)
    root = tmp_path / "matrix"
    code = main(_argv(root, "--stage", "ship-it"))
    assert code != 0
    assert not root.exists()
