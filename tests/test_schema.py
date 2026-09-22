from __future__ import annotations

import copy
import dataclasses

import pytest

from boundarybench.schema import ConstructCard, SchemaError
from tests.conftest import mutate


def test_parses_a_valid_card(card_dict):
    card = ConstructCard.from_dict(card_dict)

    assert card.cube_id == "fixture_authority_v1"
    assert card.cube_type == "1-ACT"
    assert card.fact_location == "elicited"
    assert sorted(card.state_axis) == ["S0", "S1"]
    assert sorted(card.policy_axis) == ["P0", "P1"]
    assert card.disposition("S0", "P0") == "ACT"
    assert card.disposition("S1", "P1") == "STOP"


def test_card_is_immutable(card_dict):
    card = ConstructCard.from_dict(card_dict)
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.cube_id = "other"  # type: ignore[misc]


def test_parsing_does_not_alias_the_input_dict(card_dict):
    original = copy.deepcopy(card_dict)
    card = ConstructCard.from_dict(card_dict)
    card_dict["state_axis"]["S0"]["facts"]["repair_quote_gbp"] = 9999
    assert card.state_axis["S0"].facts["repair_quote_gbp"] == 180
    assert original["state_axis"]["S0"]["facts"]["repair_quote_gbp"] == 180


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(SchemaError, match="schema_version"):
        ConstructCard.from_dict(mutate(schema_version=3))


@pytest.mark.parametrize(
    "field",
    ["cube_id", "domain", "workflow", "cube_type", "fact_location", "disclaimer"],
)
def test_missing_required_field_is_rejected(field):
    card = mutate()
    del card[field]
    with pytest.raises(SchemaError, match=field):
        ConstructCard.from_dict(card)


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(SchemaError, match="unknown"):
        ConstructCard.from_dict(mutate(leaderboard_rank=1))


def test_state_axis_must_be_binary():
    card = mutate()
    card["state_axis"]["S2"] = card["state_axis"]["S0"]
    with pytest.raises(SchemaError, match="state_axis"):
        ConstructCard.from_dict(card)


def test_policy_axis_must_be_binary():
    card = mutate()
    del card["policy_axis"]["P1"]
    with pytest.raises(SchemaError, match="policy_axis"):
        ConstructCard.from_dict(card)


def test_disposition_table_must_cover_all_four_cells():
    card = mutate()
    del card["disposition_table"]["S1_P1"]
    with pytest.raises(SchemaError, match="disposition_table"):
        ConstructCard.from_dict(card)


def test_disposition_values_are_restricted_to_the_binary_ontology():
    card = mutate()
    card["disposition_table"]["S0_P1"] = "ESCALATE"
    with pytest.raises(SchemaError, match="ESCALATE"):
        ConstructCard.from_dict(card)


def test_unknown_fact_location_is_rejected():
    with pytest.raises(SchemaError, match="fact_location"):
        ConstructCard.from_dict(mutate(fact_location="telepathy"))


def test_production_privacy_status_is_rejected_in_this_slice():
    with pytest.raises(SchemaError, match="privacy_status"):
        ConstructCard.from_dict(mutate(privacy_status="PUBLIC_SAFE"))


def test_empty_disclaimer_is_rejected():
    with pytest.raises(SchemaError, match="disclaimer"):
        ConstructCard.from_dict(mutate(disclaimer="   "))


def test_rule_reason_must_be_declared_in_precedence():
    card = mutate()
    card["policy_axis"]["P1"]["rules"][0]["primary_reason"] = "VIBES"
    with pytest.raises(SchemaError, match="VIBES"):
        ConstructCard.from_dict(card)


def test_rule_condition_must_reference_a_known_fact():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][0]["when"][0]["fact"] = "moon_phase"
    with pytest.raises(SchemaError, match="moon_phase"):
        ConstructCard.from_dict(card)


def test_unknown_condition_operator_is_rejected():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][0]["when"][0]["op"] = "approximately"
    with pytest.raises(SchemaError, match="approximately"):
        ConstructCard.from_dict(card)


def test_irreversible_actions_must_be_allowed_actions():
    with pytest.raises(SchemaError, match="irreversible"):
        ConstructCard.from_dict(mutate(irreversible_actions=["demolish_property"]))


def test_unknown_critical_invariant_is_rejected():
    with pytest.raises(SchemaError, match="BE_NICE"):
        ConstructCard.from_dict(mutate(critical_invariants=["BE_NICE"]))


# -- supported enums for this vertical slice -------------------------------


def test_both_shipped_cube_types_are_accepted_by_the_schema():
    """Balance against the table is the compiler's job, not the schema's."""
    for cube_type in ("1-ACT", "3-ACT"):
        assert ConstructCard.from_dict(mutate(cube_type=cube_type)).cube_type == cube_type


@pytest.mark.parametrize("cube_type", ["2-ACT", "4-ACT", "N-ACT"])
def test_unshipped_cube_types_are_rejected(cube_type):
    with pytest.raises(SchemaError, match=cube_type):
        ConstructCard.from_dict(mutate(cube_type=cube_type))


@pytest.mark.parametrize("location", ["tool_mediated", "runtime_emergent"])
def test_unshipped_fact_locations_are_rejected(location):
    with pytest.raises(SchemaError, match=location):
        ConstructCard.from_dict(mutate(fact_location=location))


def test_supported_enums_are_narrowed_to_what_is_implemented():
    from boundarybench.schema import CUBE_TYPES, EXPECTED_ACT_CELLS, FACT_LOCATIONS

    assert frozenset({"1-ACT", "3-ACT"}) == CUBE_TYPES
    assert dict(EXPECTED_ACT_CELLS) == {"1-ACT": 1, "3-ACT": 3}
    assert frozenset({"initial_state", "elicited"}) == FACT_LOCATIONS


# -- distractors -----------------------------------------------------------


def test_distractor_must_be_a_mapping():
    with pytest.raises(SchemaError, match="distractors"):
        ConstructCard.from_dict(mutate(distractors=["property_ref"]))


def test_distractor_missing_value_is_rejected():
    with pytest.raises(SchemaError, match="value"):
        ConstructCard.from_dict(mutate(distractors=[{"key": "property_ref"}]))


def test_distractor_missing_key_is_rejected():
    with pytest.raises(SchemaError, match="key"):
        ConstructCard.from_dict(mutate(distractors=[{"value": "SYN-0001"}]))


def test_distractor_with_an_unknown_field_is_rejected():
    with pytest.raises(SchemaError, match="hint"):
        ConstructCard.from_dict(
            mutate(distractors=[{"key": "property_ref", "value": "x", "hint": "act"}])
        )


@pytest.mark.parametrize("bad_key", ["", "   ", 7, None])
def test_distractor_key_must_be_a_non_empty_string(bad_key):
    with pytest.raises(SchemaError, match="key"):
        ConstructCard.from_dict(mutate(distractors=[{"key": bad_key, "value": "x"}]))


def test_duplicate_distractor_keys_are_rejected():
    with pytest.raises(SchemaError, match="property_ref"):
        ConstructCard.from_dict(
            mutate(
                distractors=[
                    {"key": "property_ref", "value": "SYN-0001"},
                    {"key": "property_ref", "value": "SYN-0002"},
                ]
            )
        )


def test_distractor_colliding_with_a_declared_fact_is_rejected():
    with pytest.raises(SchemaError, match="repair_quote_gbp"):
        ConstructCard.from_dict(
            mutate(distractors=[{"key": "repair_quote_gbp", "value": 999}])
        )


def test_duplicate_rule_ids_are_rejected():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][1]["id"] = "R_P0_ABOVE_LIMIT"
    with pytest.raises(SchemaError, match="R_P0_ABOVE_LIMIT"):
        ConstructCard.from_dict(card)


def test_allowed_actions_must_include_the_terminal_call():
    """The terminal action can never be forbidden: a case must be closable."""
    with pytest.raises(SchemaError, match="complete_case"):
        ConstructCard.from_dict(
            mutate(
                allowed_actions=["read_records", "ask_user", "dispatch_contractor"],
                irreversible_actions=["dispatch_contractor"],
            )
        )
