"""Malformed-shape rejection across the whole authored surface.

Every case here is a card an author could plausibly write by mistake. Each must
fail with a named SchemaError rather than an IndexError or a KeyError from deep
inside the compiler.
"""

from __future__ import annotations

import pytest

from boundarybench.schema import ConstructCard, SchemaError
from tests.conftest import mutate


def parse(card):
    return ConstructCard.from_dict(card)


def with_rule(**overrides):
    card = mutate()
    card["policy_axis"]["P1"]["rules"][0].update(overrides)
    return card


# -- top level --------------------------------------------------------------


@pytest.mark.parametrize("value", ["a string", ["a", "list"], 7, None])
def test_a_card_that_is_not_a_mapping_is_rejected(value):
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(value)


def test_a_field_that_should_be_a_mapping_is_checked():
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(mutate(state_axis="S0 and S1"))


def test_a_field_that_should_be_a_list_is_checked():
    with pytest.raises(SchemaError, match="must be a list"):
        parse(mutate(allowed_actions="read_records"))


def test_an_empty_string_list_is_rejected():
    with pytest.raises(SchemaError, match="must not be empty"):
        parse(mutate(allowed_actions=[]))


def test_a_card_without_distractors_is_valid():
    card = mutate()
    del card["distractors"]
    assert parse(card).distractors == ()


def test_distractors_must_be_a_list():
    with pytest.raises(SchemaError, match="must be a list"):
        parse(mutate(distractors={"key": "a", "value": "b"}))


def test_a_null_distractor_value_is_rejected():
    with pytest.raises(SchemaError, match="must not be null"):
        parse(mutate(distractors=[{"key": "property_ref", "value": None}]))


# -- rules and conditions ---------------------------------------------------


def test_a_rule_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["policy_axis"]["P1"]["rules"] = ["approve everything"]
    with pytest.raises(SchemaError, match="rule must be a mapping"):
        parse(card)


def test_unknown_rule_keys_are_rejected():
    with pytest.raises(SchemaError, match="severity"):
        parse(with_rule(severity="high"))


def test_a_rule_decision_outside_the_binary_ontology_is_rejected():
    with pytest.raises(SchemaError, match="ESCALATE"):
        parse(with_rule(decision="ESCALATE"))


def test_an_empty_rule_set_is_rejected():
    card = mutate()
    card["policy_axis"]["P1"]["rules"] = []
    with pytest.raises(SchemaError, match="must not be empty"):
        parse(card)


def test_a_condition_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][0]["when"] = ["quote over 250"]
    with pytest.raises(SchemaError, match="condition must be a mapping"):
        parse(card)


def test_a_condition_without_a_value_is_rejected():
    card = mutate()
    del card["policy_axis"]["P0"]["rules"][0]["when"][0]["value"]
    with pytest.raises(SchemaError, match="must declare 'value'"):
        parse(card)


def test_unknown_condition_keys_are_rejected():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][0]["when"][0]["tolerance"] = 5
    with pytest.raises(SchemaError, match="tolerance"):
        parse(card)


# -- axes -------------------------------------------------------------------


def test_a_state_arm_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["state_axis"]["S0"] = "quote is low"
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(card)


def test_a_state_arm_without_facts_is_rejected():
    card = mutate()
    card["state_axis"]["S0"]["facts"] = {}
    with pytest.raises(SchemaError, match="facts must not be empty"):
        parse(card)


def test_state_arms_with_different_fact_keys_are_rejected():
    card = mutate()
    card["state_axis"]["S1"]["facts"] = {"other_fact": 640}
    with pytest.raises(SchemaError, match="same fact keys"):
        parse(card)


def test_identical_state_arms_are_rejected():
    card = mutate()
    card["state_axis"]["S1"]["facts"] = dict(card["state_axis"]["S0"]["facts"])
    with pytest.raises(SchemaError, match="identical"):
        parse(card)


def test_a_policy_arm_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["policy_axis"]["P1"] = "approval required"
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(card)


# -- fact specs -------------------------------------------------------------


def test_facts_must_describe_exactly_the_state_axis_keys():
    card = mutate()
    card["facts"]["surplus_fact"] = {"availability": "initial"}
    with pytest.raises(SchemaError, match="exactly the state_axis fact keys"):
        parse(card)


def test_a_fact_spec_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["facts"]["repair_quote_gbp"] = "ask the contractor"
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(card)


def test_an_unknown_availability_is_rejected():
    card = mutate()
    card["facts"]["repair_quote_gbp"]["availability"] = "eventually"
    with pytest.raises(SchemaError, match="eventually"):
        parse(card)


def test_an_on_request_fact_needs_a_known_elicitation_action():
    card = mutate()
    card["facts"]["repair_quote_gbp"]["elicitation_action"] = "telepathy"
    with pytest.raises(SchemaError, match="elicitation_action"):
        parse(card)


def test_an_initial_fact_must_not_declare_an_elicitation_action():
    card = mutate()
    card["facts"]["repair_quote_gbp"] = {
        "availability": "initial",
        "elicitation_action": "ask_user",
    }
    with pytest.raises(SchemaError, match="must not declare an elicitation_action"):
        parse(card)


def test_a_call_tool_question_is_validated_when_present():
    card = mutate()
    card["facts"]["repair_quote_gbp"] = {
        "availability": "on_request",
        "elicitation_action": "call_tool",
        "question": "   ",
    }
    with pytest.raises(SchemaError, match="question"):
        parse(card)


def test_a_call_tool_fact_may_omit_the_question():
    card = mutate()
    card["facts"]["repair_quote_gbp"] = {
        "availability": "on_request",
        "elicitation_action": "call_tool",
    }
    card["allowed_actions"] = [
        "read_records",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    assert parse(card).facts["repair_quote_gbp"].question is None


# -- pressure probes --------------------------------------------------------


def test_pressure_probes_must_declare_exactly_both_directions():
    card = mutate()
    del card["pressure_probes"]["reassurance"]
    with pytest.raises(SchemaError, match="alarmist"):
        parse(card)


def test_a_probe_that_is_not_a_mapping_is_rejected():
    card = mutate()
    card["pressure_probes"]["alarmist"] = "solicitor is watching"
    with pytest.raises(SchemaError, match="must be a mapping"):
        parse(card)


def test_a_probe_anchored_on_a_cell_that_does_not_exist_is_rejected():
    card = mutate()
    card["pressure_probes"]["alarmist"]["base_cell"] = "S9_P9"
    with pytest.raises(SchemaError, match="S9_P9"):
        parse(card)


def test_an_alarmist_probe_on_a_stop_cell_is_rejected():
    card = mutate()
    card["pressure_probes"]["alarmist"]["base_cell"] = "S1_P1"
    with pytest.raises(SchemaError, match="must be built on a correct ACT cell"):
        parse(card)


def test_a_reassurance_probe_on_an_act_cell_is_rejected():
    card = mutate()
    card["pressure_probes"]["reassurance"]["base_cell"] = "S0_P0"
    with pytest.raises(SchemaError, match="must be built on a correct STOP cell"):
        parse(card)
