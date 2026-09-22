from __future__ import annotations

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.loader import load_card
from boundarybench.solvers import check_solvers
from tests.conftest import EXAMPLE_CARD


@pytest.fixture(scope="module")
def cube():
    return compile_cube(load_card(EXAMPLE_CARD))


def test_example_card_exists_and_loads():
    card = load_card(EXAMPLE_CARD)
    assert card.cube_id == "lettings_maintenance_authority_v1"
    assert card.construct_family == "maintenance_authority_and_approvals"
    assert card.cube_type == "1-ACT"
    assert card.fact_location == "elicited"


def test_example_card_is_marked_synthetic_and_disclaims_dwelly_policy():
    card = load_card(EXAMPLE_CARD)
    assert card.privacy_status == "SYNTHETIC_ONLY"
    disclaimer = card.disclaimer.lower()
    assert "not" in disclaimer
    assert "dwelly" in disclaimer
    assert "synthetic" in disclaimer


def test_example_card_has_no_obvious_identifier_markers():
    """A smoke check only, not a privacy assurance.

    Absence of these markers says nothing about whether the card is safe to
    publish. Privacy status is decided by the external review recorded on the
    card, not by this test.
    """
    card = load_card(EXAMPLE_CARD)
    rendered = str(card.raw)
    for marker in ("@", "http", "+44"):
        assert marker not in rendered


def test_example_cube_compiles_to_exactly_six_variants(cube):
    assert len(cube.variants) == 6
    assert [v.variant_id for v in cube.variants] == [
        "lettings_maintenance_authority_v1::cell::S0_P0",
        "lettings_maintenance_authority_v1::cell::S0_P1",
        "lettings_maintenance_authority_v1::cell::S1_P0",
        "lettings_maintenance_authority_v1::cell::S1_P1",
        "lettings_maintenance_authority_v1::probe::alarmist",
        "lettings_maintenance_authority_v1::probe::reassurance",
    ]


def test_example_cube_is_a_one_act_cube_with_live_edges_on_both_axes(cube):
    assert [v.expected_disposition for v in cube.variants[:4]] == [
        "ACT",
        "STOP",
        "STOP",
        "STOP",
    ]
    assert cube.live_state_edges and cube.live_policy_edges
    assert cube.preserve_state_edges and cube.preserve_policy_edges


def test_example_cube_passes_the_causal_ci_gate(cube):
    report = check_solvers(cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    assert report.outcome("reference").passed_variants == 6
    for outcome in report.outcomes:
        if outcome.kind == "negative":
            assert outcome.passed_variants < 6
