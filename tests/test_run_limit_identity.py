"""Hard run limits in run identity, the ledger and the runner.

The limits a run was authorised to spend are part of *what the run is*: they go
into configuration identity, so a resume under a different cap is a different
experiment and is refused; they are re-derived when the manifest is read, so an
edited cap does not survive; and the spend they bound is recovered from the
ledger, so a resumed run cannot forget what the run before it consumed.

Nothing here contacts a provider.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.budget import BudgetError, CostControls, RunCostGuard
from boundarybench.ledger import (
    LEDGER_SCHEMA_VERSION,
    OUTCOME_COST_CAP_REACHED,
    RUN_TERMINAL_OUTCOMES,
    LedgerError,
    read_ledger,
)
from boundarybench.pricing import price_for
from boundarybench.runmanifest import (
    RUN_MANIFEST_SCHEMA_VERSION,
    RunLimits,
    RunManifestError,
    RunManifestFormatError,
    RunManifestMismatchError,
    build_run_manifest,
    load_run_manifest,
    open_run_session,
)
from boundarybench.runner import RunTerminatedError, compiled_variants
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import validate_suite
from tests.conftest import SUITE_MANIFEST

SONNET = "claude-sonnet-5"


def _controls(cap: str | None = "100", episodes: int | None = 72) -> CostControls:
    return CostControls(
        max_cost_usd=None if cap is None else Decimal(cap),
        max_episodes=episodes,
        price=None if cap is None else price_for(provider="anthropic", model=SONNET),
    )


def _suite() -> Any:
    return validate_suite(SUITE_MANIFEST)


def _manifest(**overrides: Any) -> Any:
    suite = _suite()
    arguments: dict[str, Any] = {
        "suite": suite,
        "scaffold": load_scaffold(STANDARD_SCAFFOLD),
        "provider": "anthropic",
        "model": SONNET,
        "implementation": "anthropic_messages",
        "adapter_version": "0.4.0",
        "adapter_settings": {"max_output_tokens": 1024},
        "trials": 1,
        "limits": RunLimits(max_turns=4, max_messages=16, episode_timeout_seconds=5.0),
        "cost_controls": _controls(),
    }
    arguments.update(overrides)
    return build_run_manifest(**arguments)


# -- the manifest ------------------------------------------------------------


def test_the_manifest_schema_version_moved_with_the_block_it_added() -> None:
    """An older manifest names one thing this build reads differently."""
    assert RUN_MANIFEST_SCHEMA_VERSION == 5


def test_the_cost_controls_are_recorded_in_the_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = manifest.configuration_payload()["cost_controls"]

    assert payload["max_cost_usd"] == "100"
    assert payload["max_episodes"] == 72
    pricing = payload["pricing"]
    assert pricing["model"] == SONNET
    assert pricing["input_usd_per_million_tokens"] == "2"
    assert pricing["output_usd_per_million_tokens"] == "10"
    assert pricing["effective_through"] is None
    assert pricing["source"].startswith("https://")
    assert pricing["recorded_on"] == "2026-09-07"
    assert len(pricing["policy_digest"]) == 64


def test_a_run_without_a_cap_records_the_absence_rather_than_omitting_it() -> None:
    payload = _manifest(cost_controls=_controls(cap=None, episodes=None))
    stored = payload.configuration_payload()["cost_controls"]

    assert stored == {"max_cost_usd": None, "max_episodes": None, "pricing": None}


def test_changing_the_cap_is_a_different_configuration() -> None:
    assert (
        _manifest().configuration_id
        != _manifest(cost_controls=_controls(cap="50")).configuration_id
    )


def test_changing_the_episode_limit_is_a_different_configuration() -> None:
    assert (
        _manifest().configuration_id
        != _manifest(cost_controls=_controls(episodes=36)).configuration_id
    )


def test_a_plan_larger_than_the_episode_limit_is_refused_before_anything_exists() -> None:
    """72 planned is 72 authorised; 73 planned breaches the authorisation."""
    with pytest.raises(RunManifestError) as refused:
        _manifest(trials=1, cost_controls=_controls(episodes=11))
    assert "11" in str(refused.value)


def test_a_plan_exactly_at_the_episode_limit_is_allowed() -> None:
    """Boundary equality: the authorised number is authorised."""
    manifest = _manifest(trials=1, cost_controls=_controls(episodes=12))
    assert len(manifest.episode_plan) == 12


def test_resuming_under_a_different_cap_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    with open_run_session(root, _manifest()) as session:
        assert session.resumed is False

    with (
        pytest.raises(RunManifestMismatchError),
        open_run_session(root, _manifest(cost_controls=_controls(cap="50"))),
    ):
        pass


def test_an_edited_cap_does_not_survive_being_read_back(tmp_path: Path) -> None:
    root = tmp_path / "run"
    with open_run_session(root, _manifest()):
        pass
    path = root / "run_manifest.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["cost_controls"]["max_cost_usd"] = "100000"
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(RunManifestMismatchError):
        load_run_manifest(path)


def test_an_edited_price_is_refused_even_when_the_file_is_rehashed(
    tmp_path: Path,
) -> None:
    """Rehashing is free; agreeing with the shipped price table is not.

    The digest proves the file is self-consistent and nothing more, so the
    stored pricing policy is compared against the one this build ships rather
    than trusted. A run priced at a rate nobody reviewed cannot be continued.
    """
    from boundarybench.runmanifest import configuration_digest, execution_digest_of

    root = tmp_path / "run"
    with open_run_session(root, _manifest()):
        pass
    path = root / "run_manifest.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["cost_controls"]["pricing"]["input_usd_per_million_tokens"] = "0.0001"
    payload = {
        key: value
        for key, value in stored.items()
        if key not in ("configuration_id", "execution")
    }
    identity = configuration_digest(payload)
    stored["configuration_id"] = identity
    stored["execution"]["configuration_id"] = identity
    stored["execution"]["execution_digest"] = execution_digest_of(
        configuration_id=identity,
        execution_id=stored["execution"]["execution_id"],
        created_at_utc=stored["execution"]["created_at_utc"],
    )
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(RunManifestError) as refused:
        load_run_manifest(path)
    assert "pric" in str(refused.value).lower()


def test_a_manifest_from_the_previous_schema_is_refused_explicitly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    with open_run_session(root, _manifest()):
        pass
    path = root / "run_manifest.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["cost_controls"]
    stored["schema_version"] = 3
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(RunManifestFormatError) as refused:
        load_run_manifest(path)
    # Named by its version, before any field of it is interpreted: a manifest
    # from a build that did not record cost controls is not an unlimited run,
    # it is a run this build cannot say anything about.
    assert "schema_version" in str(refused.value)
    assert str(RUN_MANIFEST_SCHEMA_VERSION) in str(refused.value)


def test_a_stored_cap_as_a_json_number_is_refused(tmp_path: Path) -> None:
    """A JSON number is a binary float to most readers; a cap is exact."""
    with pytest.raises(BudgetError):
        CostControls.from_stored(
            {"max_cost_usd": 100.0, "max_episodes": None, "pricing": None},
            provider="anthropic",
            model=SONNET,
        )


# -- the ledger --------------------------------------------------------------


def test_the_ledger_schema_version_moved_with_the_field_it_added() -> None:
    assert LEDGER_SCHEMA_VERSION == 7


def test_a_row_from_the_previous_ledger_schema_is_refused(tmp_path: Path) -> None:
    from tests.test_runner_cost_control import run_one_capped_run

    root = tmp_path / "run"
    run_one_capped_run(root)
    path = root / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    del rows[0]["cost_exposure_usd"]
    rows[0]["schema_version"] = 3
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = load_run_manifest(root / "run_manifest.json")
    with pytest.raises(LedgerError):
        read_ledger(path, manifest, compiled_variants(_suite()))


def test_the_cost_cap_outcome_stops_the_whole_run() -> None:
    """A spent budget is true of every remaining episode, so the plan stops."""
    assert OUTCOME_COST_CAP_REACHED in RUN_TERMINAL_OUTCOMES


# -- the runner --------------------------------------------------------------


def test_a_guard_is_seeded_from_the_ledger_before_any_request(tmp_path: Path) -> None:
    from tests.test_runner_cost_control import run_one_capped_run

    root = tmp_path / "run"
    run_one_capped_run(root)
    guard = RunCostGuard(
        controls=_controls(),
        max_output_tokens=1024,
        max_attempts_per_turn=3,
    )
    manifest = load_run_manifest(root / "run_manifest.json")
    records = read_ledger(root / "episodes.jsonl", manifest, compiled_variants(_suite()))

    from boundarybench.runner import recovered_spend

    measured, exposure = recovered_spend(records)
    guard.seed(measured_usd=measured, exposure_usd=exposure)

    assert measured > 0
    assert guard.remaining_usd < Decimal("100")


def test_a_run_that_stopped_on_its_cost_cap_cannot_be_continued(tmp_path: Path) -> None:
    from tests.test_runner_cost_control import run_until_cap_exhausted

    root = tmp_path / "run"
    run_until_cap_exhausted(root)

    with pytest.raises(RunTerminatedError):
        run_until_cap_exhausted(root)
