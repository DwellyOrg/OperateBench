"""Strictness of the authored surface.

A construct card is ground truth for a benchmark, so anything the author did not
mean to say should be an error rather than a silently ignored key or a value the
compiler cannot serialise.
"""

from __future__ import annotations

import datetime
import math

import pytest

from boundarybench.schema import ConstructCard, SchemaError
from tests.conftest import mutate


def parse(card):
    return ConstructCard.from_dict(card)


# -- B4: unknown keys at every authored nesting level ----------------------


def test_unknown_key_in_a_state_arm_is_rejected():
    card = mutate()
    card["state_axis"]["S0"]["hint"] = "this is the ACT arm"
    with pytest.raises(SchemaError, match="hint"):
        parse(card)


def test_unknown_key_in_a_policy_arm_is_rejected():
    card = mutate()
    card["policy_axis"]["P1"]["version"] = 2
    with pytest.raises(SchemaError, match="version"):
        parse(card)


def test_unknown_key_in_a_fact_spec_is_rejected():
    card = mutate()
    card["facts"]["repair_quote_gbp"]["default"] = 0
    with pytest.raises(SchemaError, match="default"):
        parse(card)


def test_unknown_key_in_a_pressure_probe_is_rejected():
    card = mutate()
    card["pressure_probes"]["alarmist"]["intensity"] = "high"
    with pytest.raises(SchemaError, match="intensity"):
        parse(card)


# -- B5: list and string surfaces ------------------------------------------


@pytest.mark.parametrize("bad", [["", "FINANCIAL_AUTHORITY"], [7], [None]])
def test_reason_code_precedence_entries_must_be_non_empty_strings(bad):
    with pytest.raises(SchemaError, match="reason_code_precedence"):
        parse(mutate(reason_code_precedence=bad))


@pytest.mark.parametrize(
    "field", ["allowed_actions", "irreversible_actions", "critical_invariants"]
)
def test_action_and_invariant_entries_must_be_non_empty_strings(field):
    with pytest.raises(SchemaError, match=field):
        parse(mutate(**{field: ["   "]}))


def test_duplicate_allowed_actions_are_rejected():
    with pytest.raises(SchemaError, match="allowed_actions"):
        parse(
            mutate(
                allowed_actions=[
                    "read_records",
                    "read_records",
                    "ask_user",
                    "dispatch_contractor",
                    "complete_case",
                ]
            )
        )


def test_duplicate_irreversible_actions_are_rejected():
    with pytest.raises(SchemaError, match="irreversible_actions"):
        parse(mutate(irreversible_actions=["dispatch_contractor", "dispatch_contractor"]))


def test_duplicate_critical_invariants_are_rejected():
    with pytest.raises(SchemaError, match="critical_invariants"):
        parse(
            mutate(
                critical_invariants=[
                    "NO_IRREVERSIBLE_ACTION_WHEN_STOP",
                    "NO_IRREVERSIBLE_ACTION_WHEN_STOP",
                ]
            )
        )


def test_state_fact_keys_must_be_non_empty_strings():
    card = mutate()
    card["state_axis"]["S0"]["facts"] = {"": 180}
    card["state_axis"]["S1"]["facts"] = {"": 640}
    card["facts"] = {"": {"availability": "initial"}}
    with pytest.raises(SchemaError, match="fact"):
        parse(card)


@pytest.mark.parametrize("bad_cues", [[""], ["   "], [3]])
def test_pressure_cues_must_be_non_empty_strings(bad_cues):
    card = mutate()
    card["pressure_probes"]["alarmist"]["cues"] = bad_cues
    with pytest.raises(SchemaError, match="cues"):
        parse(card)


def test_duplicate_pressure_cues_are_rejected():
    card = mutate()
    card["pressure_probes"]["alarmist"]["cues"] = ["Same cue.", "Same cue."]
    with pytest.raises(SchemaError, match="cues"):
        parse(card)


def test_ask_user_facts_require_a_non_empty_question():
    card = mutate()
    del card["facts"]["repair_quote_gbp"]["question"]
    with pytest.raises(SchemaError, match="question"):
        parse(card)


def test_ask_user_question_must_not_be_blank():
    card = mutate()
    card["facts"]["repair_quote_gbp"]["question"] = "   "
    with pytest.raises(SchemaError, match="question"):
        parse(card)


def test_question_type_is_validated_when_present():
    card = mutate()
    card["facts"]["repair_quote_gbp"]["question"] = 42
    with pytest.raises(SchemaError, match="question"):
        parse(card)


# -- B6: recursively JSON-safe authored values -----------------------------


def test_yaml_date_in_a_state_fact_is_rejected():
    card = mutate()
    for arm in ("S0", "S1"):
        card["state_axis"][arm]["facts"]["quoted_on"] = datetime.date(2026, 1, 1)
    card["facts"]["quoted_on"] = {"availability": "initial"}
    with pytest.raises(SchemaError, match="date"):
        parse(card)


def test_yaml_timestamp_in_a_distractor_is_rejected():
    card = mutate(
        distractors=[{"key": "logged_at", "value": datetime.datetime(2026, 1, 1, 9, 0)}]
    )
    with pytest.raises(SchemaError, match="datetime"):
        parse(card)


def test_non_finite_float_in_a_condition_value_is_rejected():
    card = mutate()
    card["policy_axis"]["P0"]["rules"][0]["when"][0]["value"] = math.inf
    with pytest.raises(SchemaError, match="finite"):
        parse(card)


def test_nan_in_a_state_fact_is_rejected():
    card = mutate()
    for arm in ("S0", "S1"):
        card["state_axis"][arm]["facts"]["repair_quote_gbp"] = math.nan
    with pytest.raises(SchemaError, match="finite"):
        parse(card)


def test_non_string_mapping_keys_in_an_authored_value_are_rejected():
    card = mutate()
    for arm, quote in (("S0", 180), ("S1", 640)):
        card["state_axis"][arm]["facts"]["quote_breakdown"] = {1: quote}
    card["facts"]["quote_breakdown"] = {"availability": "initial"}
    with pytest.raises(SchemaError, match="string keys"):
        parse(card)


def test_nested_json_safe_values_are_accepted():
    card = mutate()
    for arm, quote in (("S0", 180), ("S1", 640)):
        card["state_axis"][arm]["facts"]["quote_breakdown"] = {
            "labour": quote - 60,
            "parts": [10, 20.5],
            "urgent": False,
            "note": None,
        }
    card["facts"]["quote_breakdown"] = {"availability": "initial"}
    parsed = parse(card)
    assert parsed.state_axis["S0"].facts["quote_breakdown"]["parts"] == (10, 20.5)


def test_ensure_json_safe_rejects_a_self_referential_list():
    from boundarybench.schema import ensure_json_safe

    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(SchemaError, match="cyclic"):
        ensure_json_safe(cyclic, "test")


def test_ensure_json_safe_rejects_a_self_referential_mapping():
    from boundarybench.schema import ensure_json_safe

    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(SchemaError, match="cyclic"):
        ensure_json_safe(cyclic, "test")


def test_ensure_json_safe_accepts_a_repeated_non_cyclic_reference():
    from boundarybench.schema import ensure_json_safe

    shared = {"a": [1, 2]}
    value = [shared, shared]
    assert ensure_json_safe(value, "test") == value


def test_a_recursive_list_alias_distractor_is_rejected_cleanly():
    cyclic: list = []
    cyclic.append(cyclic)
    card = mutate(distractors=[{"key": "loop", "value": cyclic}])
    with pytest.raises(SchemaError, match="cyclic"):
        parse(card)


def test_a_recursive_mapping_alias_distractor_is_rejected_cleanly():
    cyclic: dict = {}
    cyclic["loop"] = cyclic
    card = mutate(distractors=[{"key": "loop", "value": cyclic}])
    with pytest.raises(SchemaError, match="cyclic"):
        parse(card)


def test_a_repeated_non_cyclic_distractor_alias_is_accepted_and_detached():
    shared = {"note": ["kitchen", "tap"]}
    card = mutate(
        distractors=[
            {"key": "property_ref", "value": shared},
            {"key": "reported_issue", "value": shared},
        ]
    )
    parsed = parse(card)
    first, second = (d["value"] for d in parsed.distractors)
    assert first == second == {"note": ("kitchen", "tap")}
    assert first is not second


# -- B7: cube_id is a safe, stable namespace -------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "Lettings_Maintenance",
        "lettings maintenance",
        "_leading_underscore",
        "cube::cell::S0_P0",
        "cube.with.dots",
        "cube/with/slash",
        "",
    ],
)
def test_unsafe_cube_ids_are_rejected(bad_id):
    with pytest.raises(SchemaError, match="cube_id"):
        parse(mutate(cube_id=bad_id))


@pytest.mark.parametrize(
    "good_id",
    [
        "lettings_maintenance_authority_v1",
        "a",
        "a1",
        "cube-with-hyphen",
        "x_1-2_z",
        "1_leading_digit",
    ],
)
def test_safe_cube_ids_are_accepted(good_id):
    assert parse(mutate(cube_id=good_id)).cube_id == good_id
