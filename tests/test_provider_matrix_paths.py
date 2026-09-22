"""Where a matrix cell's evidence would actually land.

A run directory is one run's immutable evidence, and the plan's claim that five
cells write into five different directories is the claim that makes the ledger
attributable afterwards. That claim is about the *filesystem*, not about the
strings the plan carries: two spellings can name one directory, and a link can
make a directory that was distinct when the plan was approved stop being
distinct before the run starts.

Every test here creates real directories and real links under ``tmp_path``.
None of them runs an episode, contacts a provider or writes a run artefact.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    MatrixAuthorization,
    MatrixError,
    ModelAllocation,
    PlannedModel,
    build_matrix_plan,
    canonical_output_dir,
    check_authorizations,
    check_output_directories,
    proportional_allocations,
)

CONFIGURATION_ID = "a" * 64


def _planned(provider: str, model: str, output_dir: Path) -> PlannedModel:
    return PlannedModel(
        provider=provider,
        model=model,
        slug=model,
        episode_cap=EPISODES_PER_MODEL,
        cost_allocation_usd=Decimal("1.00"),
        output_dir=output_dir,
        api_key_variable="UNUSED_FOR_THIS_TEST",
        credential_present=False,
    )


# -- one directory, two spellings ---------------------------------------------


def test_two_cells_reaching_one_directory_through_a_parent_hop_are_refused(
    tmp_path: Path,
) -> None:
    """``alias/../evidence`` and ``evidence`` are the same directory.

    A lexical comparison sees two different strings and passes the plan. The two
    runs then interleave their ledgers under one manifest, and no row can
    afterwards be attributed to the model that produced it — which is the exact
    outcome this check exists to prevent.
    """
    (tmp_path / "alias").mkdir()
    (tmp_path / "evidence").mkdir()
    models = [
        _planned("openai", "gpt-5.6-luna", tmp_path / "evidence"),
        _planned("xai", "grok-4.5", tmp_path / "alias" / ".." / "evidence"),
    ]
    with pytest.raises(MatrixError) as raised:
        check_output_directories(models)
    assert "same output directory" in str(raised.value)


def test_two_cells_reaching_one_directory_through_a_symlink_are_refused(
    tmp_path: Path,
) -> None:
    """A link is refused before the question of duplication is even reached.

    A run directory has to be reachable through real directories only: a link
    standing in for one redirects the evidence, and the operator reading the
    plan sees a path that looks like every other path.
    """
    real = tmp_path / "evidence"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    models = [
        _planned("openai", "gpt-5.6-luna", real),
        _planned("xai", "grok-4.5", tmp_path / "alias"),
    ]
    with pytest.raises(MatrixError) as raised:
        check_output_directories(models)
    assert "symlink" in str(raised.value)


def test_a_symlinked_ancestor_of_an_output_directory_is_refused(
    tmp_path: Path,
) -> None:
    """Checking only the final component would miss the ancestor entirely."""
    real = tmp_path / "real-root"
    real.mkdir()
    (tmp_path / "linked-root").symlink_to(real, target_is_directory=True)
    with pytest.raises(MatrixError) as raised:
        canonical_output_dir(tmp_path / "linked-root" / "openai-gpt-5-6-luna")
    assert "symlink" in str(raised.value)


def test_a_relative_and_an_absolute_spelling_resolve_to_one_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan may carry either, and both name the same evidence."""
    (tmp_path / "evidence").mkdir()
    monkeypatch.chdir(tmp_path)
    assert canonical_output_dir("evidence") == canonical_output_dir(tmp_path / "evidence")


def test_the_five_shipped_cells_still_resolve_to_five_distinct_directories(
    tmp_path: Path,
) -> None:
    """The property the whole check exists to preserve, asserted directly."""
    plan = build_matrix_plan(
        output_root=tmp_path / "runs",
        total_cost_usd=Decimal("100.00"),
        allocations=proportional_allocations(Decimal("100.00")),
        environ={},
    )
    resolved = {canonical_output_dir(entry.output_dir) for entry in plan.models}
    assert len(resolved) == len(MATRIX_MODELS) == 5


def test_planning_creates_no_directory_and_no_output(tmp_path: Path) -> None:
    """A preflight that made directories would leave a trace of a run nobody ran.

    Resolving a path is a question about the filesystem, and it is asked without
    creating, opening or writing anything — including the root the plan names.
    """
    root = tmp_path / "runs"
    plan = build_matrix_plan(
        output_root=root,
        total_cost_usd=Decimal("100.00"),
        allocations=proportional_allocations(Decimal("100.00")),
        environ={},
    )
    assert not root.exists()
    assert list(tmp_path.iterdir()) == []
    assert all(not entry.output_dir.exists() for entry in plan.models)


def test_a_path_that_does_not_exist_yet_still_resolves(tmp_path: Path) -> None:
    """Every planned directory is one that has not been created yet."""
    target = tmp_path / "runs" / "openai-gpt-5-6-luna"
    assert canonical_output_dir(target) == Path(os.path.realpath(target))


# -- binding an approval to a directory ---------------------------------------


def _authorization(output_dir: Path) -> MatrixAuthorization:
    return MatrixAuthorization(
        provider="openai",
        model="gpt-5.6-luna",
        configuration_id=CONFIGURATION_ID,
        episode_cap=EPISODES_PER_MODEL,
        cost_allocation_usd=Decimal("1.00"),
        output_dir=output_dir,
        live_enabled=True,
    )


def test_an_approval_and_a_run_that_resolve_to_one_directory_agree(
    tmp_path: Path,
) -> None:
    """Two spellings of one directory are one approval, not a mismatch."""
    (tmp_path / "alias").mkdir()
    (tmp_path / "evidence").mkdir()
    planned = _planned("openai", "gpt-5.6-luna", tmp_path / "evidence")
    approval = _authorization(tmp_path / "alias" / ".." / "evidence")
    approval.check(planned, configuration_id=CONFIGURATION_ID)


def test_an_approval_for_another_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "evidence").mkdir()
    (tmp_path / "elsewhere").mkdir()
    planned = _planned("openai", "gpt-5.6-luna", tmp_path / "evidence")
    approval = _authorization(tmp_path / "elsewhere")
    with pytest.raises(MatrixError) as raised:
        approval.check(planned, configuration_id=CONFIGURATION_ID)
    assert "output directory" in str(raised.value)


def test_a_directory_swapped_for_a_link_between_plan_and_execution_is_refused(
    tmp_path: Path,
) -> None:
    """The recheck, and the reason the plan's earlier answer is not trusted.

    The approval is written against a real directory and binds cleanly. The
    directory is then replaced by a link to somewhere else — which is what an
    attacker with write access to the run root, or a well-meaning operator
    tidying up between stages, would leave behind. The same approval, unchanged,
    is refused the next time it is bound, because it is resolved again at that
    moment rather than compared as text.
    """
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    planned = _planned("openai", "gpt-5.6-luna", evidence)
    approval = _authorization(evidence)
    approval.check(planned, configuration_id=CONFIGURATION_ID)

    evidence.rmdir()
    evidence.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(MatrixError) as raised:
        approval.check(planned, configuration_id=CONFIGURATION_ID)
    assert "symlink" in str(raised.value)


def test_the_whole_authorisation_pass_rechecks_every_cells_directory(
    tmp_path: Path,
) -> None:
    """The recheck is on the path a run actually takes, not only on one object."""
    plan = build_matrix_plan(
        output_root=tmp_path / "runs",
        total_cost_usd=Decimal("100.00"),
        allocations=proportional_allocations(Decimal("100.00")),
        environ={},
    )
    identities = {
        (entry.provider, entry.model): CONFIGURATION_ID for entry in plan.models
    }
    approvals = [
        MatrixAuthorization(
            provider=entry.provider,
            model=entry.model,
            configuration_id=CONFIGURATION_ID,
            episode_cap=entry.episode_cap,
            cost_allocation_usd=entry.cost_allocation_usd,
            output_dir=entry.output_dir,
            live_enabled=True,
        )
        for entry in plan.models
    ]
    check_authorizations(plan, approvals, configuration_ids=identities)

    # One cell's directory is created as a link, after the plan and the
    # approvals were written against it.
    (tmp_path / "runs").mkdir()
    (tmp_path / "target").mkdir()
    plan.models[0].output_dir.symlink_to(tmp_path / "target", target_is_directory=True)

    with pytest.raises(MatrixError) as raised:
        check_authorizations(plan, approvals, configuration_ids=identities)
    assert "symlink" in str(raised.value)


def test_an_allocation_is_still_bound_to_its_own_cell(tmp_path: Path) -> None:
    """The other five bindings an approval carries are unchanged by any of this."""
    (tmp_path / "evidence").mkdir()
    planned = _planned("openai", "gpt-5.6-luna", tmp_path / "evidence")
    approval = MatrixAuthorization(
        provider="openai",
        model="gpt-5.6-luna",
        configuration_id=CONFIGURATION_ID,
        episode_cap=EPISODES_PER_MODEL,
        cost_allocation_usd=Decimal("2.00"),
        output_dir=tmp_path / "evidence",
        live_enabled=True,
    )
    with pytest.raises(MatrixError) as raised:
        approval.check(planned, configuration_id=CONFIGURATION_ID)
    assert "cost allocation" in str(raised.value)


def test_a_plan_whose_cells_are_distinct_is_still_accepted(tmp_path: Path) -> None:
    """The refusals above must not have made every plan unbuildable."""
    check_output_directories(
        [
            _planned("openai", "gpt-5.6-luna", tmp_path / "openai"),
            _planned("xai", "grok-4.5", tmp_path / "xai"),
            _planned("mistral", "mistral-small-2603", tmp_path / "mistral"),
        ]
    )


def test_an_allocation_object_is_unaffected_by_path_resolution() -> None:
    """A guard on the fixture: the money side of a cell is not path-derived."""
    allocation = ModelAllocation(
        provider="openai", model="gpt-5.6-luna", cost_usd=Decimal("1.00")
    )
    assert allocation.key == ("openai", "gpt-5.6-luna")
