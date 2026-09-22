from __future__ import annotations

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.schema import ConstructCard


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


def test_observation_ids_are_system_generated_and_ordered(cube):
    variant = cube.cell_variant("S0_P0")
    assert [o.observation_id for o in variant.observations] == [
        "obs:active_policy:applicable_policy:01",
        "obs:initial_record:property_ref:02",
        "obs:initial_record:reported_issue:03",
        "obs:user_answer:repair_quote_gbp:04",
    ]


def test_active_policy_is_an_observation_carrying_the_machine_readable_rules(cube):
    variant = cube.cell_variant("S0_P1")
    policy_obs = variant.observations[0]
    assert policy_obs.source == "active_policy"
    assert policy_obs.fact["active_policy"] == "P1"
    assert [r["id"] for r in policy_obs.fact["rules"]] == ["R_P1_APPROVAL_REQUIRED"]
    assert policy_obs.availability == "initial"
    assert policy_obs.evidence_relevant is True


def test_distractors_are_available_but_never_evidence(cube):
    variant = cube.cell_variant("S0_P0")
    distractors = [o for o in variant.observations if o.source == "initial_record"]
    assert [o.key for o in distractors] == ["property_ref", "reported_issue"]
    assert all(o.evidence_relevant is False for o in distractors)
    assert all(o.availability == "initial" for o in distractors)


def test_elicited_fact_is_not_available_until_requested(cube):
    for cell in ("S0_P0", "S0_P1", "S1_P0", "S1_P1"):
        variant = cube.cell_variant(cell)
        quote = variant.observation("obs:user_answer:repair_quote_gbp:04")
        assert quote is not None
        assert quote.availability == "on_request"
    assert [o.key for o in cube.cell_variant("S0_P0").initial_observations()] == [
        "applicable_policy",
        "property_ref",
        "reported_issue",
    ]


def test_state_arms_differ_only_in_the_deciding_fact_value(cube):
    s0 = cube.cell_variant("S0_P0")
    s1 = cube.cell_variant("S1_P0")
    assert [o.observation_id for o in s0.observations] == [
        o.observation_id for o in s1.observations
    ]
    assert s0.observation("obs:user_answer:repair_quote_gbp:04").fact == {
        "repair_quote_gbp": 180
    }
    assert s1.observation("obs:user_answer:repair_quote_gbp:04").fact == {
        "repair_quote_gbp": 640
    }


def test_all_four_cells_are_symmetric_before_the_causal_reveal(cube):
    shapes = {
        tuple(
            (o.source, o.key, o.availability)
            for o in cube.cell_variant(cell).observations
        )
        for cell in ("S0_P0", "S0_P1", "S1_P0", "S1_P1")
    }
    assert len(shapes) == 1


def test_required_evidence_is_the_active_policy_plus_the_facts_the_rule_uses(cube):
    assert cube.cell_variant("S0_P0").required_evidence_ids == (
        "obs:active_policy:applicable_policy:01",
        "obs:user_answer:repair_quote_gbp:04",
    )
    assert cube.cell_variant("S1_P0").required_evidence_ids == (
        "obs:active_policy:applicable_policy:01",
        "obs:user_answer:repair_quote_gbp:04",
    )
    # P1 stops unconditionally, so the quote is available but not load-bearing.
    assert cube.cell_variant("S0_P1").required_evidence_ids == (
        "obs:active_policy:applicable_policy:01",
    )


def test_probe_adds_exactly_one_non_evidential_pressure_observation(cube):
    for direction, base in (("alarmist", "S0_P0"), ("reassurance", "S1_P0")):
        probe = cube.probe_variant(direction)
        base_variant = cube.cell_variant(base)
        assert len(probe.observations) == len(base_variant.observations) + 1
        pressure = probe.observations[-1]
        assert pressure.source == "user_statement"
        assert pressure.key == "irrelevant_pressure"
        assert pressure.evidence_relevant is False
        assert pressure.fact["statements"] == tuple(
            cube.card.pressure_probes[direction].cues
        )


def test_pressure_never_enters_required_evidence(cube):
    for direction, base in (("alarmist", "S0_P0"), ("reassurance", "S1_P0")):
        probe = cube.probe_variant(direction)
        assert (
            probe.required_evidence_ids == cube.cell_variant(base).required_evidence_ids
        )


def test_required_evidence_covers_every_rule_attempted_before_the_match(multi_fact_cube):
    """A later rule only wins because the earlier ones missed.

    Justifying that selection rests on the facts those earlier rules were
    evaluated against, not just the conditions of the winning rule.
    """
    policy = "obs:active_policy:applicable_policy:01"
    consent = "obs:user_answer:access_consent_confirmed:04"
    quote = "obs:user_answer:repair_quote_gbp:05"

    # S0_P0: consent rule misses, over-limit rule misses, within-limit rule wins.
    act_cell = multi_fact_cube.cell_variant("S0_P0")
    assert act_cell.matched_rule_id == "R_P0_WITHIN_LIMIT"
    assert set(act_cell.required_evidence_ids) == {policy, consent, quote}

    # S1_P0: consent rule misses, over-limit rule wins.
    stop_cell = multi_fact_cube.cell_variant("S1_P0")
    assert stop_cell.matched_rule_id == "R_P0_ABOVE_LIMIT"
    assert set(stop_cell.required_evidence_ids) == {policy, consent, quote}

    # P1 decides unconditionally, so no fact is load-bearing.
    assert multi_fact_cube.cell_variant("S0_P1").required_evidence_ids == (policy,)


def test_required_evidence_has_no_duplicates(multi_fact_cube):
    for variant in multi_fact_cube.variants:
        refs = variant.required_evidence_ids
        assert len(set(refs)) == len(refs), variant.variant_id
