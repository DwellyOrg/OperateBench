"""What a matrix cell may be, and what a stage actually asks a runner to do.

Three properties are under test, and each one is a way an approved plan could
authorise something other than what it describes:

* **A cell is one identity.** Two cells naming one ``(provider, model)`` pair
  collapse into a single allocation, a single configuration identity and a
  single approval, however differently they are spelled — so the plan would say
  it covers six cells while the money and the authorisations covered five.
* **A slug is one directory name.** It is a path component this build joins onto
  the operator's ``--output-root``; anything that can climb out of that root, or
  name a place that is not a child of it, redirects a run's evidence somewhere
  nobody approved.
* **A stage's cumulative target is not an invocation's episode count.** Each
  stage names a per-model *ceiling* the run resumes towards; what a single
  invocation executes is the delta between that ceiling and what is already
  durable. Handing the ceiling to ``--stop-after`` re-executes the part of the
  ceiling that is already durable, so the run exceeds the approved plan. The
  ceilings and the deltas are distinct quantities, and their exact values are
  pinned by the assertions below.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    STAGE_AUDIT_GATE,
    STAGE_OFFLINE_PREFLIGHT,
    STAGE_ORDER,
    STAGE_REMAINDER,
    STAGE_SMOKE,
    STAGE_TRIAL_BLOCK,
    TOTAL_EPISODES,
    TRIALS,
    MatrixError,
    MatrixModel,
    ModelAllocation,
    PlannedModel,
    build_matrix_plan,
    check_global_episode_ceiling,
    check_matrix_trials,
    check_output_directories,
    proportional_allocations,
    stage_cumulative_episode_target,
    stage_is_manual,
    stage_new_episodes,
)

CAP = Decimal("50.00")


def _plan(root: Path, models: Any = MATRIX_MODELS, allocations: Any = None) -> Any:
    return build_matrix_plan(
        output_root=root,
        total_cost_usd=CAP,
        allocations=(
            proportional_allocations(CAP) if allocations is None else allocations
        ),
        models=models,
        environ={},
    )


def _planned(provider: str, model: str, output_dir: Path) -> PlannedModel:
    return PlannedModel(
        provider=provider,
        model=model,
        slug=output_dir.name,
        episode_cap=EPISODES_PER_MODEL,
        cost_allocation_usd=Decimal("1.00"),
        output_dir=output_dir,
        api_key_variable="UNUSED_FOR_THIS_TEST",
        credential_present=False,
    )


# -- a slug is one portable directory name ------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "../../escaped",
        "..",
        ".",
        "",
        "nested/child",
        "..\\windows",
        "back\\slash",
        "/absolute",
        "C:\\drive",
        "C:relative",
        "control\nnewline",
        "control\x00nul",
        "control\x7fdelete",
    ],
)
def test_a_slug_that_is_not_one_safe_path_component_is_refused(slug: str) -> None:
    """The slug is joined onto an operator's output root, so it is checked there.

    Refused at construction rather than at planning time: a ``MatrixModel`` is
    the thing an operator reads, exports and approves, and one carrying a slug
    that climbs out of the root would describe a run whose evidence lands
    somewhere the approval never named. A backslash and a drive letter are
    refused on every platform for the same reason a slash is — a slug is one
    *portable* component, and a name that is inert here is a separator there.
    """
    with pytest.raises(MatrixError) as caught:
        MatrixModel("openai", "gpt-5.6-luna", slug)
    assert "slug" in str(caught.value).lower()


def test_the_five_shipped_slugs_are_still_accepted() -> None:
    """The rule must not have outlawed the matrix it exists to protect."""
    assert len(MATRIX_MODELS) == 5
    for entry in MATRIX_MODELS:
        assert MatrixModel(entry.provider, entry.model, entry.slug).slug == entry.slug


def test_a_cell_whose_directory_leaves_the_output_root_is_refused(
    tmp_path: Path,
) -> None:
    """Containment is proven against the resolved root, not against the text."""
    root = tmp_path / "matrix"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    with pytest.raises(MatrixError) as caught:
        check_output_directories(
            [_planned("openai", "gpt-5.6-luna", outside)], output_root=root
        )
    assert "output root" in str(caught.value).lower()


def test_cells_under_the_output_root_are_accepted(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    check_output_directories(
        [
            _planned("openai", "gpt-5.6-luna", root / "openai"),
            _planned("xai", "grok-4.5", root / "xai"),
        ],
        output_root=root,
    )


# -- a cell is one identity ---------------------------------------------------


def test_two_cells_naming_one_provider_and_model_are_refused(tmp_path: Path) -> None:
    """Different slugs do not make one model two cells.

    The pair is the key everything downstream is held under: the allocation, the
    price, the configuration identity and the approval. Two cells sharing it
    would plan two directories' worth of episodes against one approval and one
    share of the cap.
    """
    duplicated = (
        MatrixModel("openai", "gpt-5.6-luna", "openai-one"),
        MatrixModel("openai", "gpt-5.6-luna", "openai-two"),
    )
    allocations = (
        ModelAllocation(provider="openai", model="gpt-5.6-luna", cost_usd=CAP),
    )
    with pytest.raises(MatrixError) as caught:
        _plan(tmp_path / "matrix", models=duplicated, allocations=allocations)
    message = str(caught.value)
    assert "twice" in message
    assert "gpt-5.6-luna" in message


def test_two_cells_sharing_a_slug_are_refused(tmp_path: Path) -> None:
    """One slug is one directory, so two cells carrying it is one run's evidence."""
    clashing = (
        MatrixModel("openai", "gpt-5.6-luna", "shared"),
        MatrixModel("xai", "grok-4.5", "shared"),
    )
    allocations = (
        ModelAllocation(
            provider="openai", model="gpt-5.6-luna", cost_usd=Decimal("25.00")
        ),
        ModelAllocation(provider="xai", model="grok-4.5", cost_usd=Decimal("25.00")),
    )
    with pytest.raises(MatrixError) as caught:
        _plan(tmp_path / "matrix", models=clashing, allocations=allocations)
    assert "slug" in str(caught.value).lower()


def test_a_duplicate_cell_is_refused_before_it_is_priced(tmp_path: Path) -> None:
    """The identity check runs first, so its message is the one an operator gets.

    A duplicated cell this build holds no price for must still be refused as a
    duplicate: the pair being ambiguous is the more fundamental fault, and
    reporting the price would send the operator to fix the wrong thing.
    """
    duplicated = (
        MatrixModel("openai", "a-model-with-no-reviewed-price", "one"),
        MatrixModel("openai", "a-model-with-no-reviewed-price", "two"),
    )
    allocations = (
        ModelAllocation(
            provider="openai", model="a-model-with-no-reviewed-price", cost_usd=CAP
        ),
    )
    with pytest.raises(MatrixError) as caught:
        _plan(tmp_path / "matrix", models=duplicated, allocations=allocations)
    assert "twice" in str(caught.value)


def test_the_shipped_matrix_still_plans_cleanly(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "matrix")
    assert plan.total_episodes == TOTAL_EPISODES
    assert {entry.slug for entry in plan.models} == {
        entry.slug for entry in MATRIX_MODELS
    }


# -- the target chain and the invocation deltas -------------------------------


def test_the_cumulative_targets_are_the_approved_prefix_chain() -> None:
    assert [stage_cumulative_episode_target(stage) for stage in STAGE_ORDER] == [
        0,
        1,
        1,
        12,
        36,
    ]


def test_the_new_episode_deltas_are_what_one_invocation_executes() -> None:
    """Each delta is this stage's target minus what the last stage made durable."""
    assert [stage_new_episodes(stage) for stage in STAGE_ORDER] == [0, 1, 0, 11, 24]


def test_the_deltas_sum_to_the_per_model_plan_and_never_exceed_it() -> None:
    """The whole chain executes 36 episodes per model, not 49.

    This is the arithmetic the old artefact got wrong: 1 + 12 + 36 is what a
    reader adds up if the cumulative targets are handed to ``--stop-after``, and
    the first two alone already execute 13 episodes against a 12-episode block.
    """
    assert sum(stage_new_episodes(stage) for stage in STAGE_ORDER) == EPISODES_PER_MODEL
    assert (
        stage_new_episodes(STAGE_SMOKE) + stage_new_episodes(STAGE_TRIAL_BLOCK)
        == stage_cumulative_episode_target(STAGE_TRIAL_BLOCK)
        == 12
    )


def test_each_delta_is_the_difference_between_consecutive_targets() -> None:
    previous = 0
    for stage in STAGE_ORDER:
        target = stage_cumulative_episode_target(stage)
        assert stage_new_episodes(stage) == target - previous
        previous = target


def test_the_manual_gate_executes_nothing_at_all() -> None:
    assert stage_is_manual(STAGE_AUDIT_GATE) is True
    assert stage_new_episodes(STAGE_AUDIT_GATE) == 0
    assert stage_cumulative_episode_target(STAGE_AUDIT_GATE) == (
        stage_cumulative_episode_target(STAGE_SMOKE)
    )
    assert stage_new_episodes(STAGE_OFFLINE_PREFLIGHT) == 0


def test_an_unknown_stage_has_neither_a_target_nor_a_delta() -> None:
    for reader in (stage_cumulative_episode_target, stage_new_episodes):
        with pytest.raises(MatrixError):
            reader("ship_it")


def test_the_plan_states_both_numbers_under_names_that_cannot_be_confused(
    tmp_path: Path,
) -> None:
    """The artefact names the target and the delta, and calls neither a stop-after."""
    payload = _plan(tmp_path / "matrix").as_dict()
    stages = payload["stages"]
    assert [entry["stage"] for entry in stages] == list(STAGE_ORDER)
    assert [entry["cumulative_episode_target_per_model"] for entry in stages] == [
        0,
        1,
        1,
        12,
        36,
    ]
    assert [entry["new_episodes_per_model"] for entry in stages] == [0, 1, 0, 11, 24]
    for entry in stages:
        assert not any("stop_after" in key for key in entry)
        assert "episode_cap_per_model" not in entry
    assert payload["per_model_run_episode_cap"] == EPISODES_PER_MODEL
    assert payload["global_episode_ceiling"] == TOTAL_EPISODES


def test_the_remainder_stage_is_the_whole_per_model_plan() -> None:
    assert stage_cumulative_episode_target(STAGE_REMAINDER) == EPISODES_PER_MODEL


# -- the two exact authorisations ---------------------------------------------


@pytest.mark.parametrize("supplied", [0, 1, 36, 179, 181, 360])
def test_an_episode_ceiling_that_is_not_the_whole_matrix_is_refused(
    supplied: int,
) -> None:
    """One global ceiling, exact in both directions.

    A smaller number is authority the matrix cannot run under; a larger one is
    authority nobody needs and nothing would consume, and an approval carrying
    unused authority is an approval for a bigger experiment than the one
    described.
    """
    with pytest.raises(MatrixError) as caught:
        check_global_episode_ceiling(supplied)
    assert str(TOTAL_EPISODES) in str(caught.value)


def test_the_exact_episode_ceiling_is_accepted_and_is_the_default() -> None:
    assert check_global_episode_ceiling(TOTAL_EPISODES) == TOTAL_EPISODES
    assert check_global_episode_ceiling(None) == TOTAL_EPISODES


@pytest.mark.parametrize("supplied", [0, 1, 2, 4, 12])
def test_a_trial_count_other_than_the_fixed_three_is_refused(supplied: int) -> None:
    """The repeat structure is the methodology, not a knob.

    Two Cubes x six variants x three trials x five models is 180 episodes, and
    the per-model manifest that number is derived from is exactly 36. A run
    planned at another trial count is a different experiment reported under this
    plan's name.
    """
    with pytest.raises(MatrixError) as caught:
        check_matrix_trials(supplied)
    assert str(TRIALS) in str(caught.value)


def test_the_fixed_trial_count_is_accepted() -> None:
    assert check_matrix_trials(TRIALS) == TRIALS
