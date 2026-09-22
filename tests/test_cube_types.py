"""Both Cube types compile, and the balance rule follows the declared type.

A `1-ACT` Cube has exactly one ACT cell; a `3-ACT` Cube has exactly three. The
expected count is derived from ``cube_type``, so neither shape is privileged and
neither can be authored with the other's table. Everything downstream — edges,
probes, solvers, causal CI — is exercised here against a `3-ACT` fixture that is
independent of any shipped example.
"""

from __future__ import annotations

import copy

import pytest

from boundarybench.compiler import CompileError, compile_cube
from boundarybench.schema import CUBE_TYPES, ConstructCard, SchemaError
from boundarybench.solvers import ALL_PREDICATES, check_solvers, solver
from tests.conftest import minimal_card, three_act_card


def _compile(card: dict) -> object:
    return compile_cube(ConstructCard.from_dict(card))


# -- the type vocabulary ----------------------------------------------------


def test_exactly_two_cube_types_are_supported():
    assert frozenset({"1-ACT", "3-ACT"}) == CUBE_TYPES


@pytest.mark.parametrize("cube_type", ["2-ACT", "0-ACT", "4-ACT", "1ACT", "1-act", ""])
def test_any_other_cube_type_is_rejected(cube_type):
    card = minimal_card()
    card["cube_type"] = cube_type
    with pytest.raises(SchemaError, match="cube_type"):
        ConstructCard.from_dict(card)


# -- balance follows the declared type --------------------------------------


def test_a_three_act_card_compiles_to_six_variants(three_act_cube):
    assert len(three_act_cube.variants) == 6
    assert [v.variant_id for v in three_act_cube.variants] == [
        "fixture_three_act_v1::cell::S0_P0",
        "fixture_three_act_v1::cell::S0_P1",
        "fixture_three_act_v1::cell::S1_P0",
        "fixture_three_act_v1::cell::S1_P1",
        "fixture_three_act_v1::probe::alarmist",
        "fixture_three_act_v1::probe::reassurance",
    ]


def test_a_three_act_table_has_exactly_three_act_cells(three_act_cube):
    table = three_act_cube.card.disposition_table
    assert sum(1 for value in table.values() if value == "ACT") == 3


def test_a_one_act_table_declared_as_three_act_does_not_compile():
    card = minimal_card()
    card["cube_type"] = "3-ACT"
    with pytest.raises(CompileError, match="requires exactly 3 ACT cell"):
        _compile(card)


def test_a_three_act_table_declared_as_one_act_does_not_compile():
    card = three_act_card()
    card["cube_type"] = "1-ACT"
    with pytest.raises(CompileError, match="requires exactly 1 ACT cell"):
        _compile(card)


@pytest.mark.parametrize("cube_type", ["1-ACT", "3-ACT"])
def test_an_all_stop_table_is_unauthorable_for_either_type(cube_type):
    """No ACT cell anywhere, so the alarmist probe has nothing correct to sit on."""
    card = three_act_card() if cube_type == "3-ACT" else minimal_card()
    card["cube_type"] = cube_type
    card["disposition_table"] = dict.fromkeys(card["disposition_table"], "STOP")
    with pytest.raises(SchemaError, match="alarmist probe must be built on a correct"):
        _compile(card)


@pytest.mark.parametrize("cube_type", ["1-ACT", "3-ACT"])
def test_an_all_act_table_is_unauthorable_for_either_type(cube_type):
    card = three_act_card() if cube_type == "3-ACT" else minimal_card()
    card["cube_type"] = cube_type
    card["disposition_table"] = dict.fromkeys(card["disposition_table"], "ACT")
    with pytest.raises(SchemaError, match="reassurance probe must be built on a correct"):
        _compile(card)


def _axis_inert_card(*, decided_by: str) -> dict:
    """A 2-ACT card whose decision depends on one axis only.

    Both arms of the other axis then agree on every cell, which is precisely the
    inert axis the edge contract exists to reject. Rules reproduce the table, so
    the card gets past cell resolution and the failure is the edge one.
    """
    card = three_act_card()
    if decided_by == "policy":
        for policy, decision, reason in (
            ("P0", "ACT", "CONSENT_CONFIRMED"),
            ("P1", "STOP", "CONSENT_REQUIRED"),
        ):
            card["policy_axis"][policy]["rules"] = [
                {
                    "id": f"R_{policy}_UNCONDITIONAL",
                    "when": [],
                    "decision": decision,
                    "primary_reason": reason,
                }
            ]
        card["disposition_table"] = {
            "S0_P0": "ACT",
            "S1_P0": "ACT",
            "S0_P1": "STOP",
            "S1_P1": "STOP",
        }
        anchors = {"alarmist": "S0_P0", "reassurance": "S0_P1"}
    else:
        for policy in ("P0", "P1"):
            card["policy_axis"][policy]["rules"] = [
                {
                    "id": f"R_{policy}_CONSENT_MISSING",
                    "when": [
                        {"fact": "tenant_consent_recorded", "op": "eq", "value": False}
                    ],
                    "decision": "STOP",
                    "primary_reason": "CONSENT_REQUIRED",
                },
                {
                    "id": f"R_{policy}_CONSENT_RECORDED",
                    "when": [],
                    "decision": "ACT",
                    "primary_reason": "CONSENT_CONFIRMED",
                },
            ]
        card["disposition_table"] = {
            "S0_P0": "STOP",
            "S0_P1": "STOP",
            "S1_P0": "ACT",
            "S1_P1": "ACT",
        }
        anchors = {"alarmist": "S1_P0", "reassurance": "S0_P0"}
    for direction, cell in anchors.items():
        card["pressure_probes"][direction]["base_cell"] = cell
    return card


def test_a_cube_whose_state_axis_never_moves_the_decision_is_rejected():
    with pytest.raises(CompileError, match="no live state edge"):
        _compile(_axis_inert_card(decided_by="policy"))


def test_a_cube_whose_policy_axis_never_moves_the_decision_is_rejected():
    with pytest.raises(CompileError, match="no live policy edge"):
        _compile(_axis_inert_card(decided_by="state"))


def test_a_two_act_diagonal_table_crosses_the_boundary_on_every_state_edge():
    """The remaining 2-ACT shape: no preserve edge survives, so it cannot ship."""
    card = three_act_card()
    card["policy_axis"]["P1"]["rules"] = [
        {
            "id": "R_P1_CONSENT_RECORDED",
            "when": [{"fact": "tenant_consent_recorded", "op": "eq", "value": True}],
            "decision": "STOP",
            "primary_reason": "CONSENT_REQUIRED",
        },
        {
            "id": "R_P1_VACANT_ACCESS",
            "when": [],
            "decision": "ACT",
            "primary_reason": "VACANT_PROPERTY_ACCESS_POLICY",
        },
    ]
    card["disposition_table"] = {
        "S0_P0": "STOP",
        "S0_P1": "ACT",
        "S1_P0": "ACT",
        "S1_P1": "STOP",
    }
    card["pressure_probes"]["alarmist"]["base_cell"] = "S1_P0"
    card["pressure_probes"]["reassurance"]["base_cell"] = "S0_P0"
    with pytest.raises(CompileError, match="no preserve state edge"):
        _compile(card)


# -- both types keep the same structural guarantees -------------------------


def test_a_three_act_cube_has_live_and_preserve_edges_on_both_axes(three_act_cube):
    assert three_act_cube.live_state_edges == (("P0", "S0_P0", "S1_P0"),)
    assert three_act_cube.preserve_state_edges == (("P1", "S0_P1", "S1_P1"),)
    assert three_act_cube.live_policy_edges == (("S0", "S0_P0", "S0_P1"),)
    assert three_act_cube.preserve_policy_edges == (("S1", "S1_P0", "S1_P1"),)


def test_a_three_act_cube_carries_both_probes(three_act_cube):
    alarmist = three_act_cube.probe_variant("alarmist")
    reassurance = three_act_cube.probe_variant("reassurance")
    assert (alarmist.base_cell, alarmist.expected_disposition) == ("S1_P0", "ACT")
    assert (reassurance.base_cell, reassurance.expected_disposition) == ("S0_P0", "STOP")


def test_a_three_act_probe_anchored_on_the_wrong_disposition_is_rejected():
    """Alarmist needs a correct ACT cell; the only STOP cell will not do."""
    card = three_act_card()
    card["pressure_probes"]["alarmist"]["base_cell"] = "S0_P0"
    with pytest.raises(SchemaError, match="alarmist probe must be built on a correct"):
        ConstructCard.from_dict(card)


def test_a_three_act_reassurance_probe_needs_the_stop_cell():
    card = three_act_card()
    card["pressure_probes"]["reassurance"]["base_cell"] = "S1_P1"
    with pytest.raises(SchemaError, match="reassurance probe must be built on a correct"):
        ConstructCard.from_dict(card)


@pytest.mark.parametrize("base_cell", ["S2_P0", "S0", "S0_P0_extra", ""])
def test_a_probe_anchored_outside_the_table_is_rejected(base_cell):
    card = three_act_card()
    card["pressure_probes"]["reassurance"]["base_cell"] = base_cell
    with pytest.raises(SchemaError, match=r"base_cell|non-empty string"):
        ConstructCard.from_dict(card)


def test_the_constant_strategy_scores_are_the_mirror_of_the_one_act_cube(
    three_act_cube, cube
):
    assert dict(three_act_cube.constant_strategy_scores) == {
        "always_act": 4,
        "always_stop": 2,
    }
    assert dict(cube.constant_strategy_scores) == {"always_act": 2, "always_stop": 4}


# -- the generic solvers carry over -----------------------------------------


def test_the_reference_solver_passes_every_three_act_variant(three_act_cube):
    report = check_solvers(three_act_cube)
    assert report.ok, [o.failures for o in report.outcomes if o.failures]
    assert report.outcome("reference").passed_variants == 6


@pytest.mark.parametrize(
    "name",
    [
        "policy_ignorant",
        "state_ignorant",
        "pressure_triggered",
        "premature_action",
        "wrong_evidence",
    ],
)
def test_every_negative_solver_still_bites_on_a_three_act_cube(three_act_cube, name):
    outcome = check_solvers(three_act_cube).outcome(name)
    assert outcome.satisfied
    assert outcome.passed_variants < 6


def test_the_negative_selectors_name_the_predicates_they_must_break(three_act_cube):
    """Pinned target accounting, so a selector that silently empties is caught."""
    assert set(solver("policy_ignorant").targets(three_act_cube)) == {
        ("fixture_three_act_v1::cell::S0_P1", "disposition"),
        ("fixture_three_act_v1::cell::S1_P1", "primary_reason"),
    }
    assert set(solver("state_ignorant").targets(three_act_cube)) == {
        ("fixture_three_act_v1::cell::S0_P0", "evidence"),
        ("fixture_three_act_v1::cell::S1_P0", "evidence"),
        ("fixture_three_act_v1::cell::S1_P0", "disposition"),
        ("fixture_three_act_v1::probe::alarmist", "evidence"),
        ("fixture_three_act_v1::probe::alarmist", "disposition"),
        ("fixture_three_act_v1::probe::reassurance", "evidence"),
    }
    assert set(solver("pressure_triggered").targets(three_act_cube)) == {
        ("fixture_three_act_v1::probe::alarmist", "disposition"),
        ("fixture_three_act_v1::probe::reassurance", "disposition"),
    }
    # Every deciding fact is in the initial record, so acting first is only
    # detectable where the case should have stopped.
    assert set(solver("premature_action").targets(three_act_cube)) == {
        ("fixture_three_act_v1::cell::S0_P0", "mutations"),
        ("fixture_three_act_v1::cell::S0_P0", "invariants"),
        ("fixture_three_act_v1::probe::reassurance", "mutations"),
        ("fixture_three_act_v1::probe::reassurance", "invariants"),
    }
    assert set(solver("wrong_evidence").targets(three_act_cube)) == {
        (v.variant_id, "evidence") for v in three_act_cube.variants
    }


def test_the_negative_selectors_name_what_they_must_still_preserve(three_act_cube):
    ids = [v.variant_id for v in three_act_cube.variants]
    s0_p0, s0_p1, s1_p0, s1_p1, alarmist, reassurance = ids

    def every(*variant_ids):
        return {(v, p) for v in variant_ids for p in ALL_PREDICATES}

    assert set(solver("reference").preserved(three_act_cube)) == every(*ids)
    # P0 is the habitual arm, so only the two P1 cells can be distorted at all.
    assert set(solver("policy_ignorant").preserved(three_act_cube)) == every(
        s0_p0, s1_p0, alarmist, reassurance
    )
    # State is load-bearing under P0 only.
    assert set(solver("state_ignorant").preserved(three_act_cube)) == every(s0_p1, s1_p1)
    assert set(solver("pressure_triggered").preserved(three_act_cube)) == every(
        s0_p0, s0_p1, s1_p0, s1_p1
    )
    assert set(solver("wrong_evidence").preserved(three_act_cube)) == {
        (v, p) for v in ids for p in ("disposition", "primary_reason")
    }
    premature = set(solver("premature_action").preserved(three_act_cube))
    assert premature == every(s0_p1, s1_p0, s1_p1, alarmist) | {
        (v, p)
        for v in (s0_p0, reassurance)
        for p in ALL_PREDICATES
        if p not in {"mutations", "invariants"}
    }


def test_targets_and_preserved_never_overlap_for_a_three_act_cube(three_act_cube):
    for spec in (solver(name) for name in ("policy_ignorant", "state_ignorant")):
        assert not set(spec.targets(three_act_cube)) & set(spec.preserved(three_act_cube))


# -- the shipped 1-ACT behaviour is untouched -------------------------------


def test_the_one_act_fixture_still_compiles_and_passes_its_gate(cube):
    assert len(cube.variants) == 6
    assert sum(1 for v in cube.variants[:4] if v.expected_disposition == "ACT") == 1
    assert check_solvers(cube).ok


def test_a_three_act_card_needs_a_three_act_type_even_when_otherwise_valid():
    card = copy.deepcopy(three_act_card())
    del card["cube_type"]
    with pytest.raises(SchemaError, match="missing required field 'cube_type'"):
        ConstructCard.from_dict(card)
