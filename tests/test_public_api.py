"""Lookup and error branches on the public surface.

These are the paths a caller hits when something is wrong, so they should fail
with a named error rather than an IndexError three frames down.
"""

from __future__ import annotations

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.evaluator import evaluate
from boundarybench.schema import Condition, ConstructCard, SchemaError
from boundarybench.solvers import SolverError, check_solvers, solver


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


# -- the exported surface ---------------------------------------------------


def test_the_package_exports_the_manifest_and_suite_entry_points():
    import boundarybench

    for name in (
        "card_fingerprint",
        "verify_manifest",
        "load_manifest",
        "ManifestError",
        "validate_suite",
        "check_suite",
        "load_suite",
        "SuiteError",
        "BENCHMARK_VERSION",
    ):
        assert name in boundarybench.__all__, name
        assert getattr(boundarybench, name) is not None


def test_every_exported_name_resolves():
    import boundarybench

    assert len(set(boundarybench.__all__)) == len(boundarybench.__all__)
    for name in boundarybench.__all__:
        assert hasattr(boundarybench, name), name


# -- lookups ----------------------------------------------------------------


def test_unknown_variant_id_raises_key_error(cube):
    with pytest.raises(KeyError):
        cube.variant("fixture_authority_v1::cell::S9_P9")


def test_unknown_cell_raises_key_error(cube):
    with pytest.raises(KeyError):
        cube.cell_variant("S9_P9")


def test_unknown_probe_direction_raises_key_error(cube):
    with pytest.raises(KeyError):
        cube.probe_variant("sarcasm")


def test_unknown_observation_id_returns_none(cube):
    assert cube.cell_variant("S0_P0").observation("obs:nope:nope:99") is None


def test_unknown_solver_name_raises_key_error():
    with pytest.raises(KeyError):
        solver("clairvoyant")


def test_unknown_outcome_name_raises_key_error(cube):
    with pytest.raises(KeyError):
        check_solvers(cube).outcome("clairvoyant")


def test_unknown_predicate_name_raises_key_error(cube):
    variant = cube.cell_variant("S0_P0")
    result = evaluate(variant, solver("reference").solve(variant, cube))
    with pytest.raises(KeyError):
        result.predicate("vibes")


def test_on_request_observations_are_reported(cube):
    variant = cube.cell_variant("S0_P0")
    assert [o.key for o in variant.on_request_observations()] == ["repair_quote_gbp"]


def test_first_index_of_returns_none_for_an_absent_action(cube):
    variant = cube.cell_variant("S0_P1")
    trajectory = solver("reference").solve(variant, cube)
    assert trajectory.first_index_of("dispatch_contractor") is None


# -- evaluator guards -------------------------------------------------------


def test_evaluating_a_trajectory_from_another_variant_is_refused(cube):
    variant = cube.cell_variant("S0_P0")
    other = cube.cell_variant("S0_P1")
    trajectory = solver("reference").solve(variant, cube)
    with pytest.raises(ValueError, match="belongs to"):
        evaluate(other, trajectory)


def test_solver_error_is_raised_when_the_policy_was_never_observed(cube):
    """A solver that decides without reading the record has nothing to cite."""
    from boundarybench.environment import Environment
    from boundarybench.solvers import _policy_handle

    env = Environment(cube.cell_variant("S0_P0"))
    with pytest.raises(SolverError, match="active policy"):
        _policy_handle(env)


def test_solver_error_when_no_presented_rule_covers_the_facts(cube):
    from boundarybench.environment import Environment
    from boundarybench.solvers import _apply

    env = Environment(cube.cell_variant("S0_P0"))
    env.read_records()
    with pytest.raises(SolverError, match="no presented rule"):
        _apply(env, ())


# -- condition operators ----------------------------------------------------


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("eq", 180, True),
        ("eq", 181, False),
        ("ne", 181, True),
        ("ne", 180, False),
        ("lt", 181, True),
        ("lt", 180, False),
        ("lte", 180, True),
        ("lte", 179, False),
        ("gt", 179, True),
        ("gt", 180, False),
        ("gte", 180, True),
        ("gte", 181, False),
    ],
)
def test_every_condition_operator_behaves(op, value, expected):
    condition = Condition(fact="repair_quote_gbp", op=op, value=value)
    assert condition.holds({"repair_quote_gbp": 180}) is expected


def test_a_condition_over_an_unknown_fact_never_holds():
    assert Condition(fact="moon_phase", op="eq", value=1).holds({}) is False


def test_an_unknown_operator_is_reported_at_evaluation_time():
    condition = Condition(fact="repair_quote_gbp", op="approximately", value=1)
    with pytest.raises(SchemaError, match="approximately"):
        condition.holds({"repair_quote_gbp": 180})
