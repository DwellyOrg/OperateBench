"""What a run refuses to execute against.

Every fact a ledger row asserts — which suite, which scaffold, which adapter,
which contracts, which plan — is pinned in the run manifest before the first
episode. The objects actually handed to the runner are a separate claim about
the same run, and the two are only the same run if every pinned field is
re-derived from the objects in hand and agrees. These tests take a real,
internally consistent manifest and hand the runner something else.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from boundarybench.adapter import (
    AdapterCall,
    AdapterUsage,
    ScriptedTestAdapter,
    TurnDeadline,
    TurnRequest,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.ledger import OUTCOME_SUCCESS
from boundarybench.runmanifest import (
    BUILD_CONTRACT_VERSIONS,
    EpisodePlanEntry,
    RunLimits,
    RunManifestFormatError,
    build_run_manifest,
    configuration_digest,
    execution_digest_of,
)
from boundarybench.runner import (
    RunProvenanceError,
    ScaffoldContractError,
    check_scaffold_contract,
    compiled_variants,
    run_episode,
    verify_execution_provenance,
)
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.suite import (
    SuiteReport,
    suite_content_digest,
    validate_suite,
)
from tests.conftest import SUITE_MANIFEST
from tests.scaffolds import write_scaffold
from tests.test_suite import repinned, suite_dict

# -- fixtures ----------------------------------------------------------------


def _suite() -> SuiteReport:
    return validate_suite(SUITE_MANIFEST)


def _scaffold() -> Any:
    return load_scaffold(STANDARD_SCAFFOLD)


def _manifest(*, suite: Any = None, scaffold: Any = None, trials: int = 1) -> Any:
    who = identity_for_test_double()
    return build_run_manifest(
        suite=suite if suite is not None else _suite(),
        scaffold=scaffold if scaffold is not None else _scaffold(),
        provider=who.provider,
        model=who.model,
        implementation=who.implementation,
        adapter_version=who.version,
        adapter_settings={},
        trials=trials,
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )


def _adapter(**overrides: Any) -> ScriptedTestAdapter:
    fields: dict[str, Any] = {
        "script": policy_following_script,
        "identity": identity_for_test_double(),
        "settings": {},
    }
    fields.update(overrides)
    return ScriptedTestAdapter(**fields)


def _verify(
    *, suite: Any = None, scaffold: Any = None, manifest: Any, adapter: Any = None
) -> None:
    return verify_execution_provenance(
        suite=suite if suite is not None else _suite(),
        scaffold=scaffold if scaffold is not None else _scaffold(),
        adapter=adapter if adapter is not None else _adapter(),
        manifest=manifest,
    )


def _reidentified(manifest: Any, **overrides: Any) -> Any:
    """A manifest edited *and* re-hashed, as a local editor could produce.

    Both digests are recomputed, not just one: the configuration digest over the
    edited configuration, and the execution digest that binds this manifest's
    execution id and creation time to the *new* configuration id. A helper that
    only refreshed the first would leave the manifest self-inconsistent, and the
    tests below would then be watching :func:`verify_manifest_identity` reject a
    broken binding rather than watching provenance reject a coherent manifest
    that describes a different run.

    That coherence is the point. Both digests are unkeyed and their inputs are
    all in the file, so a local editor can produce exactly this object; what it
    cannot do is make the manifest agree with the suite, scaffold, adapter and
    build actually in hand, which is what the provenance check compares.
    """
    edited = dataclasses.replace(manifest, **overrides)
    configuration_id = configuration_digest(edited.configuration_payload())
    return dataclasses.replace(
        edited,
        configuration_id=configuration_id,
        execution_digest=execution_digest_of(
            configuration_id=configuration_id,
            execution_id=edited.execution_id,
            created_at_utc=edited.created_at_utc,
        ),
    )


def _rebuilt_suite(tmp_path: Path, **changes: Any) -> SuiteReport:
    """A genuinely valid suite that differs from the shipped one, re-pinned."""
    import shutil

    from tests.conftest import EXAMPLES

    target = tmp_path / "examples"
    if not target.exists():
        shutil.copytree(EXAMPLES, target)
    payload = {**suite_dict(), **changes}
    return validate_suite(repinned(target, payload))


# -- the suite in hand -------------------------------------------------------


def test_a_suite_with_another_id_cannot_execute_this_run(tmp_path: Path) -> None:
    other = _rebuilt_suite(tmp_path, suite_id="another_methodology_spike")

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(suite=other, manifest=_manifest())

    assert "suite_id" in str(excinfo.value)


def test_a_suite_at_another_benchmark_version_cannot_execute_this_run(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    other = _rebuilt_suite(tmp_path, benchmark_version="0.2.0-dev.99")

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(suite=other, manifest=manifest)

    assert "benchmark_version" in str(excinfo.value)


def test_a_suite_whose_content_moved_cannot_execute_this_run(tmp_path: Path) -> None:
    """Same identity, same version, different content: the digest is the check."""
    manifest = _manifest()
    other = _rebuilt_suite(tmp_path, disclaimer="Synthetic. Reworded after pinning.")

    assert other.manifest.suite_id == manifest.suite_id
    assert other.content_digest != manifest.suite_content_digest

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(suite=other, manifest=manifest)

    assert "suite_content_digest" in str(excinfo.value)


def test_a_suite_that_no_longer_hashes_to_its_own_digest_cannot_execute() -> None:
    """The digest travels inside the suite, so it is a claim until re-derived."""
    suite = _suite()
    tampered = dataclasses.replace(suite, cubes=suite.cubes[:1])

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(suite=tampered, manifest=_manifest())

    assert "hashes to" in str(excinfo.value)


def test_a_suite_that_no_longer_satisfies_its_own_constraints_cannot_execute() -> None:
    """Re-pinning a dropped cube fixes the digest and breaks the declaration.

    The constraints are the reason the suite exists — one 1-ACT and one 3-ACT
    shape, both fact locations, a 6/6 disposition split. A collection that has
    been cut down to a single cube still hashes consistently once re-pinned, so
    the digest alone would pass it; the declaration has to be re-derived from
    the cubes actually in hand.
    """
    suite = _suite()
    cut = dataclasses.replace(suite, cubes=suite.cubes[:1])
    cut = dataclasses.replace(
        cut, content_digest=suite_content_digest(cut.manifest, cut.cubes)
    )

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(suite=cut, manifest=_manifest())

    assert "constraints it declares" in str(excinfo.value)


# -- the scaffold in hand ----------------------------------------------------


def _shipped_payload() -> dict[str, Any]:
    """The released scaffold as a plain document, minus its pin."""
    payload = json.loads(STANDARD_SCAFFOLD.read_text(encoding="utf-8"))
    payload.pop("content_digest", None)
    return payload


def _scaffold_variant(directory: Path, **changes: Any) -> Any:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**_shipped_payload(), **changes}
    return load_scaffold(write_scaffold(directory / "scaffold.json", payload))


def test_a_scaffold_with_another_id_cannot_execute_this_run(tmp_path: Path) -> None:
    shipped = _scaffold_variant(tmp_path / "a")
    manifest = _manifest(scaffold=shipped)
    other = _scaffold_variant(tmp_path / "b", scaffold_id="another_scaffold")

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(scaffold=other, manifest=manifest)

    assert "scaffold_id" in str(excinfo.value)


def test_a_scaffold_at_another_version_cannot_execute_this_run(tmp_path: Path) -> None:
    shipped = _scaffold_variant(tmp_path / "a")
    manifest = _manifest(scaffold=shipped)
    other = _scaffold_variant(tmp_path / "b", scaffold_version="0.9.0-dev.1")

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(scaffold=other, manifest=manifest)

    assert "scaffold_version" in str(excinfo.value)


def test_a_scaffold_whose_prompt_moved_cannot_execute_this_run(tmp_path: Path) -> None:
    shipped = _scaffold_variant(tmp_path / "a")
    manifest = _manifest(scaffold=shipped)
    other = _scaffold_variant(
        tmp_path / "b", system_prompt="You are a lettings service agent. Be brief."
    )

    assert other.scaffold_id == shipped.scaffold_id
    assert other.content_digest != shipped.content_digest

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(scaffold=other, manifest=manifest)

    assert "scaffold_content_digest" in str(excinfo.value)


def test_a_scaffold_that_terminates_with_another_action_is_refused(
    tmp_path: Path,
) -> None:
    """The loop terminates on one named call; a scaffold may not move the flag.

    Every call the loop makes is still declared here, so the scaffold passes
    every check above: the only thing wrong with it is which action it marks as
    the one that ends an episode.
    """
    payload = _shipped_payload()
    for action in payload["actions"]:
        action["terminal"] = action["name"] == "read_records"
    scaffold = load_scaffold(write_scaffold(tmp_path / "scaffold.json", payload))

    with pytest.raises(ScaffoldContractError) as excinfo:
        check_scaffold_contract(scaffold)

    assert "terminal action" in str(excinfo.value)


# -- the adapter in hand -----------------------------------------------------


class _MuteAdapter:
    """An adapter that cannot state what it is."""

    settings: ClassVar[dict[str, Any]] = {}

    @property
    def identity(self) -> Any:
        raise RuntimeError("provider integration is not configured")

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        raise AssertionError("never reached")  # pragma: no cover - never dispatched

    def last_usage(self) -> AdapterUsage:
        raise AssertionError("never reached")  # pragma: no cover - never dispatched


class _UnrepresentableSettingsAdapter:
    """An adapter whose settings canonical JSON cannot carry."""

    identity = identity_for_test_double()
    settings: ClassVar[dict[str, Any]] = {"note": "bad \ud800"}

    def next_call(self, request: TurnRequest, deadline: TurnDeadline) -> AdapterCall:
        raise AssertionError("never reached")  # pragma: no cover - never dispatched

    def last_usage(self) -> AdapterUsage:
        raise AssertionError("never reached")  # pragma: no cover - never dispatched


def test_an_adapter_that_cannot_state_its_identity_fails_before_execution() -> None:
    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(manifest=_manifest(), adapter=_MuteAdapter())

    assert "could not state its own identity" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)


def test_unrepresentable_adapter_settings_keep_their_own_named_error() -> None:
    """The settings validator already names this; the runner must not reword it."""
    with pytest.raises(RunManifestFormatError) as excinfo:
        _verify(manifest=_manifest(), adapter=_UnrepresentableSettingsAdapter())

    assert "surrogate" in str(excinfo.value).lower()


# -- the build and the plan --------------------------------------------------


def test_a_manifest_from_another_build_cannot_execute_here() -> None:
    """A resumed run may only continue under the contracts it was planned under.

    The manifest is re-hashed here, so this is not a stale-id forgery: it is a
    coherent manifest that names a different evaluator contract, which is what
    an upgraded build finds in an old run directory.
    """
    manifest = _manifest()
    older = _reidentified(manifest, evaluator_contract_version="0.0.1-dev.0")

    assert dict(older.contract_versions) != dict(BUILD_CONTRACT_VERSIONS)

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(manifest=older)

    assert "contract_versions" in str(excinfo.value)


def test_a_manifest_whose_plan_is_not_this_suites_plan_cannot_execute() -> None:
    """The plan is derived, so a coherent manifest can still name the wrong one."""
    manifest = _manifest()
    shortened = _reidentified(manifest, episode_plan=manifest.episode_plan[:-1])

    with pytest.raises(RunProvenanceError) as excinfo:
        _verify(manifest=shortened)

    assert "episode plan" in str(excinfo.value)


def test_a_plan_in_another_order_is_not_this_suites_plan() -> None:
    """Order is part of the plan: rows resume by position, not by set membership."""
    manifest = _manifest()
    reversed_plan = tuple(reversed(manifest.episode_plan))
    shuffled = _reidentified(manifest, episode_plan=reversed_plan)

    with pytest.raises(RunProvenanceError):
        _verify(manifest=shuffled)


def test_the_shipped_run_verifies_and_returns_the_variants_it_will_execute() -> None:
    """The positive case, so every refusal above is a difference and not the norm."""
    suite = _suite()
    variants = _verify(suite=suite, manifest=_manifest(suite=suite))

    assert variants == compiled_variants(suite)


# -- the result a verified run produces --------------------------------------


def test_a_successful_episode_reports_itself_as_succeeded() -> None:
    suite = _suite()
    variant = next(iter(compiled_variants(suite).values()))
    result = run_episode(
        variant=variant,
        scaffold=_scaffold(),
        adapter=_adapter(),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )

    assert result.outcome == OUTCOME_SUCCESS
    assert result.succeeded is True


def test_a_failed_episode_does_not_report_itself_as_succeeded() -> None:
    suite = _suite()
    variant = next(iter(compiled_variants(suite).values()))

    def refuses(request: TurnRequest) -> AdapterCall:
        return AdapterCall("no_such_action", {})

    result = run_episode(
        variant=variant,
        scaffold=_scaffold(),
        adapter=_adapter(script=refuses),
        limits=RunLimits(max_turns=12, max_messages=64, episode_timeout_seconds=30.0),
    )

    assert result.outcome != OUTCOME_SUCCESS
    assert result.succeeded is False


def test_a_plan_entry_round_trips_through_its_stored_form() -> None:
    entry = EpisodePlanEntry(variant_id="fixture_v1#S0_P0", trial_index=1)

    assert json.loads(json.dumps(entry.as_dict()))["trial_index"] == 1
    assert yaml.safe_load(yaml.safe_dump(entry.as_dict()))["variant_id"].endswith("S0_P0")
