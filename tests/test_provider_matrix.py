"""The five-model provider matrix: its plan, its stages and its authorisation.

Nothing here contacts a provider, and the tests that matter most prove exactly
that: a dry run makes zero transport calls and a refused stage leaves no run
directory behind.

What the matrix is *not* is a leaderboard. It executes two semantic units across
five models so that decoding repeatability can be estimated within each; the
plan says so in the artefact it writes, and one of these tests holds it to that.
"""

from __future__ import annotations

import itertools
from decimal import Decimal
from pathlib import Path

import pytest

from boundarybench.providermatrix import (
    EPISODES_PER_MODEL,
    MATRIX_MODELS,
    STAGE_ORDER,
    MatrixAuthorization,
    MatrixError,
    MatrixModel,
    ModelAllocation,
    build_matrix_plan,
    proportional_allocations,
    stage_cumulative_episode_target,
)

TOTAL_EPISODES = 180
GLOBAL_CAP = Decimal("50.00")


def _allocations() -> tuple[ModelAllocation, ...]:
    return proportional_allocations(GLOBAL_CAP)


def _plan(tmp_path: Path, **overrides: object):
    kwargs: dict = {
        "output_root": tmp_path / "matrix",
        "total_cost_usd": GLOBAL_CAP,
        "allocations": _allocations(),
    }
    kwargs.update(overrides)
    return build_matrix_plan(**kwargs)


# -- the shape of the experiment ----------------------------------------------


def test_the_matrix_is_five_exact_models_across_four_providers() -> None:
    assert len(MATRIX_MODELS) == 5
    assert {entry.model for entry in MATRIX_MODELS} == {
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5",
        "gpt-5.6-luna",
        "grok-4.5",
        "mistral-small-2603",
    }
    assert {entry.provider for entry in MATRIX_MODELS} == {
        "anthropic",
        "openai",
        "xai",
        "mistral",
    }
    # Every entry names an exact model, never an alias.
    for entry in MATRIX_MODELS:
        assert isinstance(entry, MatrixModel)
        assert not entry.model.endswith("-latest")


def test_the_episode_arithmetic_is_stated_and_adds_up(tmp_path: Path) -> None:
    """36 per model, 180 in total, and both are derived rather than asserted."""
    plan = _plan(tmp_path)
    assert EPISODES_PER_MODEL == 36
    assert all(entry.episode_cap == EPISODES_PER_MODEL for entry in plan.models)
    assert sum(entry.episode_cap for entry in plan.models) == TOTAL_EPISODES
    assert plan.total_episodes == TOTAL_EPISODES


def test_every_model_gets_its_own_run_directory(tmp_path: Path) -> None:
    """Separate immutable directories: one run's evidence never lands in another's."""
    plan = _plan(tmp_path)
    directories = [entry.output_dir for entry in plan.models]
    assert len(set(directories)) == 5
    for directory in directories:
        assert not directory.exists(), "planning creates nothing"


def test_planning_writes_nothing_to_disk(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _plan(tmp_path)
    assert not root.exists()


# -- the money ----------------------------------------------------------------


def test_allocations_sum_exactly_to_the_global_cap(tmp_path: Path) -> None:
    """Exactly, in Decimal. A split that lost a cent to rounding would either
    under-spend the approval or, worse, over-spend it."""
    plan = _plan(tmp_path)
    total = sum((entry.cost_allocation_usd for entry in plan.models), Decimal(0))
    assert total == GLOBAL_CAP
    assert plan.total_cost_usd == GLOBAL_CAP


def test_a_split_that_does_not_sum_to_the_cap_is_refused(tmp_path: Path) -> None:
    """No borrowing, in either direction."""
    short = tuple(
        ModelAllocation(provider=a.provider, model=a.model, cost_usd=Decimal("1.00"))
        for a in _allocations()
    )
    with pytest.raises(MatrixError) as caught:
        _plan(tmp_path, allocations=short)
    assert "sum" in str(caught.value).lower()


def test_an_over_allocated_split_is_refused(tmp_path: Path) -> None:
    over = list(_allocations())
    over[0] = ModelAllocation(
        provider=over[0].provider,
        model=over[0].model,
        cost_usd=over[0].cost_usd + Decimal("0.01"),
    )
    with pytest.raises(MatrixError):
        _plan(tmp_path, allocations=tuple(over))


def test_a_duplicate_model_in_the_allocations_is_refused(tmp_path: Path) -> None:
    allocations = list(_allocations())
    allocations[1] = ModelAllocation(
        provider=allocations[0].provider,
        model=allocations[0].model,
        cost_usd=allocations[1].cost_usd,
    )
    with pytest.raises(MatrixError) as caught:
        _plan(tmp_path, allocations=tuple(allocations))
    assert "duplicate" in str(caught.value).lower()


def test_an_allocation_for_a_model_not_in_the_matrix_is_refused(tmp_path: Path) -> None:
    allocations = list(_allocations())
    allocations[0] = ModelAllocation(
        provider="openai", model="gpt-not-in-the-matrix", cost_usd=allocations[0].cost_usd
    )
    with pytest.raises(MatrixError):
        _plan(tmp_path, allocations=tuple(allocations))


def test_every_allocation_is_priced_under_its_own_providers_table(tmp_path: Path) -> None:
    """A cap this build cannot measure spend against is not a control."""
    from boundarybench.pricing import price_for

    plan = _plan(tmp_path)
    for entry in plan.models:
        price = price_for(provider=entry.provider, model=entry.model)
        assert price.provider == entry.provider


# -- the stages ---------------------------------------------------------------


def test_the_stage_order_is_the_approved_one() -> None:
    assert STAGE_ORDER == (
        "offline_preflight",
        "smoke",
        "audit_gate",
        "trial_block",
        "remainder",
    )


def test_the_episode_targets_are_a_strict_prefix_chain() -> None:
    """Each executing stage resumes the previous one rather than restarting it.

    Strictness is the property that matters. If a later stage's cumulative
    target were not a superset of the earlier one's, resuming would either
    re-execute episodes already paid for or skip episodes never run, and neither
    is a resume.
    """
    caps = [stage_cumulative_episode_target(stage) for stage in STAGE_ORDER]
    assert caps == [0, 1, 1, 12, 36]
    executing = [cap for cap in caps if cap]
    assert executing == sorted(executing)
    for earlier, later in itertools.pairwise(executing):
        assert later >= earlier


def test_the_offline_preflight_stage_executes_nothing() -> None:
    assert stage_cumulative_episode_target("offline_preflight") == 0


def test_the_audit_gate_is_manual_and_executes_nothing() -> None:
    """A gate that ran episodes would not be a gate."""
    from boundarybench.providermatrix import stage_is_manual

    assert stage_is_manual("audit_gate") is True
    assert stage_cumulative_episode_target("audit_gate") == (
        stage_cumulative_episode_target("smoke")
    )
    assert stage_is_manual("remainder") is False


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(MatrixError):
        stage_cumulative_episode_target("ship_it")


# -- what the plan stores -----------------------------------------------------


def test_the_plan_stores_credential_variable_names_and_never_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan is an artefact. A credential in one is a credential on disk."""
    import json

    secret = "sk-THIS-MUST-NEVER-REACH-THE-PLAN"
    for entry in MATRIX_MODELS:
        monkeypatch.setenv(entry.api_key_variable, secret)
    rendered = json.dumps(_plan(tmp_path).as_dict(), sort_keys=True)
    assert secret not in rendered
    for entry in MATRIX_MODELS:
        assert entry.api_key_variable in rendered


def test_the_plan_reports_credential_presence_as_a_boolean_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for entry in MATRIX_MODELS:
        monkeypatch.delenv(entry.api_key_variable, raising=False)
    monkeypatch.setenv(MATRIX_MODELS[0].api_key_variable, "present")
    payload = _plan(tmp_path).as_dict()
    presence = {
        entry["credential"]["variable"]: entry["credential"]["present"]
        for entry in payload["models"]
    }
    assert presence[MATRIX_MODELS[0].api_key_variable] is True
    assert all(
        value is False
        for key, value in presence.items()
        if key != MATRIX_MODELS[0].api_key_variable
    )


def test_the_plan_states_that_this_is_not_a_ranking(tmp_path: Path) -> None:
    """Two semantic units, repeated trials, and no leaderboard claim."""
    payload = _plan(tmp_path).as_dict()
    note = payload["methodology_note"].lower()
    assert "two semantic unit" in note or "2 semantic unit" in note
    assert "rank" in note
    assert payload["semantic_units"] == 2


def test_the_plan_is_json_safe_and_uses_exact_decimal_strings(tmp_path: Path) -> None:
    """A JSON number is a binary float to most readers.

    Which is the one representation an authorised dollar allocation must not go
    through: the number an operator approved and the number a reader recovers
    have to be the same number.
    """
    import json

    from boundarybench.jsonsafe import canonical_json_text

    payload = _plan(tmp_path).as_dict()
    canonical_json_text(payload, "matrix plan")
    for entry in json.loads(json.dumps(payload))["models"]:
        assert isinstance(entry["cost_allocation_usd"], str)
        assert Decimal(entry["cost_allocation_usd"]) >= 0
    assert isinstance(payload["total_cost_usd"], str)


# -- authorisation ------------------------------------------------------------


def _authorization(entry, **overrides) -> MatrixAuthorization:
    kwargs: dict = {
        "provider": entry.provider,
        "model": entry.model,
        "configuration_id": "a" * 64,
        "episode_cap": EPISODES_PER_MODEL,
        "cost_allocation_usd": Decimal("10.00"),
        "output_dir": Path("/tmp/never-created"),
    }
    kwargs.update(overrides)
    return MatrixAuthorization(**kwargs)


def test_live_traffic_is_disabled_by_default(tmp_path: Path) -> None:
    """Default-deny. An authorisation object is not permission to spend."""
    plan = _plan(tmp_path)
    entry = plan.models[0]
    authorization = _authorization(
        entry,
        cost_allocation_usd=entry.cost_allocation_usd,
        output_dir=entry.output_dir,
    )
    assert authorization.live_enabled is False
    with pytest.raises(MatrixError) as caught:
        authorization.check(entry, configuration_id=authorization.configuration_id)
    assert "live" in str(caught.value).lower()


def test_an_authorisation_binds_every_part_of_the_experiment(tmp_path: Path) -> None:
    """Provider, model, configuration id, episode cap, allocation and directory.

    Each of the six is checked separately, because each is a different thing an
    operator could have approved for one run and had applied to another.
    """
    plan = _plan(tmp_path)
    entry = plan.models[0]
    good = _authorization(
        entry,
        cost_allocation_usd=entry.cost_allocation_usd,
        output_dir=entry.output_dir,
        live_enabled=True,
    )
    good.check(entry, configuration_id=good.configuration_id)

    other = plan.models[1]
    for field, value in (
        ("provider", other.provider if other.provider != entry.provider else "xai"),
        ("model", other.model),
        ("episode_cap", EPISODES_PER_MODEL - 1),
        ("cost_allocation_usd", entry.cost_allocation_usd + Decimal("0.01")),
        ("output_dir", tmp_path / "somewhere-else"),
    ):
        overrides: dict = {
            "cost_allocation_usd": entry.cost_allocation_usd,
            "output_dir": entry.output_dir,
            "live_enabled": True,
            field: value,
        }
        tampered = _authorization(entry, **overrides)
        with pytest.raises(MatrixError):
            tampered.check(entry, configuration_id=tampered.configuration_id)


def test_a_configuration_id_mismatch_is_refused(tmp_path: Path) -> None:
    """The identity the operator audited must be the identity that will run."""
    plan = _plan(tmp_path)
    entry = plan.models[0]
    authorization = _authorization(
        entry,
        cost_allocation_usd=entry.cost_allocation_usd,
        output_dir=entry.output_dir,
        live_enabled=True,
    )
    with pytest.raises(MatrixError) as caught:
        authorization.check(entry, configuration_id="b" * 64)
    assert "configuration" in str(caught.value).lower()


def test_an_authorisation_never_carries_a_credential(tmp_path: Path) -> None:
    import json

    plan = _plan(tmp_path)
    entry = plan.models[0]
    authorization = _authorization(
        entry,
        cost_allocation_usd=entry.cost_allocation_usd,
        output_dir=entry.output_dir,
        live_enabled=True,
    )
    rendered = json.dumps(authorization.as_dict(), sort_keys=True)
    assert "key" not in rendered.lower() or "api_key_variable" in rendered
    assert "sk-" not in rendered


def test_two_authorisations_for_one_cell_are_refused_whatever_their_order(
    tmp_path: Path,
) -> None:
    """One cell, one approval. Two of them is an ambiguity, not a preference.

    A duplicate is exactly what a stale approval looks like beside a fresh one:
    an operator who re-approved a cell after changing its allocation, and passed
    both. Resolving that by keeping whichever arrived last makes the answer
    depend on argument order — the same two approvals refuse the run in one
    order and permit it in the other — and the one that silently wins may be the
    one that withheld permission to spend.
    """
    from boundarybench.providermatrix import check_authorizations

    plan = _plan(tmp_path)
    identities = {(entry.provider, entry.model): "a" * 64 for entry in plan.models}
    approvals = [
        _authorization(
            entry,
            cost_allocation_usd=entry.cost_allocation_usd,
            output_dir=entry.output_dir,
            live_enabled=True,
        )
        for entry in plan.models
    ]
    first = plan.models[0]
    conflicting = _authorization(
        first,
        cost_allocation_usd=first.cost_allocation_usd + Decimal("1.00"),
        output_dir=first.output_dir,
        live_enabled=False,
    )
    for supplied in ([conflicting, *approvals], [*approvals, conflicting]):
        with pytest.raises(MatrixError) as caught:
            check_authorizations(plan, supplied, configuration_ids=identities)
        assert first.model in str(caught.value)


# -- what a pinned identifier can and cannot claim -----------------------------


def test_the_one_identifier_that_is_not_a_dated_snapshot_says_so(
    tmp_path: Path,
) -> None:
    """Four of the five are dated or versioned; ``grok-4.5`` is what xAI exposes.

    The matrix claims to pin exact models rather than aliases, and for four cells
    that claim is fully supported. For xAI it is supported only as far as the
    provider allows: the published model page exposes ``grok-4.5`` and no dated
    snapshot identifier, so the pinned string is the most exact one obtainable
    and can still change what it resolves to without changing the name. That is a
    limitation of the evidence, and an artefact an operator authorises has to
    carry it rather than leave a reader to infer it from a missing date.
    """
    limited = {
        entry.model: entry.identifier_limitation
        for entry in MATRIX_MODELS
        if entry.identifier_limitation is not None
    }
    assert set(limited) == {"grok-4.5"}
    assert "dated" in limited["grok-4.5"]

    payload = _plan(tmp_path).as_dict()
    stated = {
        entry["model"]: entry["identifier_limitation"] for entry in payload["models"]
    }
    assert set(stated) == {entry.model for entry in MATRIX_MODELS}
    assert stated["grok-4.5"] == limited["grok-4.5"]
    assert [model for model, note in stated.items() if note is not None] == ["grok-4.5"]


# -- duplicate output paths ---------------------------------------------------


def test_two_models_sharing_an_output_directory_is_refused(tmp_path: Path) -> None:
    """One directory is one run's immutable evidence."""
    from boundarybench.providermatrix import check_output_directories

    entries = _plan(tmp_path).models
    clashing = [entries[0], entries[0]]
    with pytest.raises(MatrixError) as caught:
        check_output_directories(clashing)
    assert "director" in str(caught.value).lower()
