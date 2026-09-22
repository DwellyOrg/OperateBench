"""Deep immutability and serialization detachment.

Compiled artefacts are the benchmark's ground truth. If any nested structure
stays mutable after compilation, a caller can change what a variant means after
its content digest was taken, and the digest silently stops describing it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from boundarybench.compiler import compile_cube
from boundarybench.environment import Environment
from boundarybench.schema import ConstructCard
from boundarybench.suite import SuiteConstraints, validate_suite
from tests.conftest import SUITE_MANIFEST, mutate


@pytest.fixture()
def cube(card_dict):
    return compile_cube(ConstructCard.from_dict(card_dict))


def is_plain_json(value: Any) -> bool:
    """True when the tree uses only built-in dict/list/scalar containers."""
    if isinstance(value, dict):
        return all(
            type(key) is str and is_plain_json(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return all(is_plain_json(item) for item in value)
    if type(value) is tuple:
        return False
    return value is None or isinstance(value, (str, int, float, bool))


# -- A1: nested structures are recursively immutable -----------------------


def test_active_policy_rules_are_not_mutable_after_compilation(cube):
    fact = cube.cell_variant("S0_P0").observations[0].fact
    rules = fact["rules"]

    with pytest.raises(TypeError):
        fact["rules"] = ()
    with pytest.raises((TypeError, AttributeError)):
        rules.append({"id": "R_FORGED"})  # type: ignore[union-attr]
    with pytest.raises(TypeError):
        rules[0]["decision"] = "ACT"
    with pytest.raises((TypeError, AttributeError)):
        rules[0]["when"].append({"fact": "x", "op": "eq", "value": 1})


def test_nested_condition_mappings_are_not_mutable(cube):
    condition = cube.cell_variant("S0_P0").observations[0].fact["rules"][0]["when"][0]
    with pytest.raises(TypeError):
        condition["value"] = 999


def test_pressure_statements_are_not_mutable(cube):
    statements = cube.probe_variant("alarmist").observations[-1].fact["statements"]
    with pytest.raises((TypeError, AttributeError)):
        statements.append("and the press has been called")  # type: ignore[union-attr]


def test_state_arm_facts_are_not_mutable_at_any_depth():
    card = ConstructCard.from_dict(nested_value_card())
    facts = card.state_axis["S0"].facts
    with pytest.raises(TypeError):
        facts["repair_quote_gbp"] = 1
    with pytest.raises(TypeError):
        facts["quote_breakdown"]["labour"] = 1
    with pytest.raises((TypeError, AttributeError)):
        facts["quote_breakdown"]["notes"].append("x")  # type: ignore[union-attr]


def test_card_raw_is_not_mutable_at_any_depth(card_dict):
    card = ConstructCard.from_dict(card_dict)
    with pytest.raises(TypeError):
        card.raw["cube_id"] = "other"
    with pytest.raises(TypeError):
        card.raw["state_axis"]["S0"]["facts"]["repair_quote_gbp"] = 1


def test_distractor_entries_are_not_mutable(card_dict):
    card = ConstructCard.from_dict(card_dict)
    with pytest.raises(TypeError):
        card.distractors[0]["value"] = "tampered"


def test_step_arguments_are_not_mutable_at_any_depth(cube):
    env = Environment(cube.cell_variant("S0_P1"))
    env.read_records()
    env.complete_case(
        disposition="STOP",
        primary_reason_code="CONTRACTUAL_AUTHORITY",
        evidence_refs=("obs:active_policy:applicable_policy:01",),
    )
    arguments = env.steps[-1].arguments
    with pytest.raises(TypeError):
        arguments["disposition"] = "ACT"
    with pytest.raises((TypeError, AttributeError)):
        arguments["evidence_refs"].append("obs:forged:0:99")  # type: ignore[union-attr]


# -- A2: as_dict returns detached, plain, JSON-serializable structures -----


def test_variant_as_dict_is_plain_json_and_detached(cube):
    variant = cube.cell_variant("S0_P0")
    payload = variant.as_dict()
    assert is_plain_json(payload)

    payload["observations"][0]["fact"]["rules"][0]["decision"] = "TAMPERED"
    payload["required_evidence_ids"].append("obs:forged:0:99")
    payload["expected_disposition"] = "TAMPERED"

    assert variant.observations[0].fact["rules"][0]["decision"] == "STOP"
    assert "obs:forged:0:99" not in variant.required_evidence_ids
    assert variant.expected_disposition == "ACT"
    assert variant.as_dict()["expected_disposition"] == "ACT"


def test_cube_as_dict_is_plain_json_and_json_serializable(cube):
    payload = cube.as_dict()
    assert is_plain_json(payload)
    json.dumps(payload)  # must not raise


def test_observation_as_dict_is_detached(cube):
    observation = cube.cell_variant("S0_P0").observations[0]
    payload = observation.as_dict()
    assert is_plain_json(payload)
    payload["fact"]["rules"][0]["id"] = "R_TAMPERED"
    assert observation.fact["rules"][0]["id"] == "R_P0_ABOVE_LIMIT"


def test_trajectory_as_dict_is_plain_json_and_detached(cube):
    env = Environment(cube.cell_variant("S0_P1"))
    env.read_records()
    env.complete_case(
        disposition="STOP",
        primary_reason_code="CONTRACTUAL_AUTHORITY",
        evidence_refs=("obs:active_policy:applicable_policy:01",),
    )
    trajectory = env.trajectory()
    payload = trajectory.as_dict()
    assert is_plain_json(payload)

    payload["steps"][0]["revealed_observation_ids"].append("obs:forged:0:99")
    payload["terminal_decision"]["disposition"] = "ACT"
    assert trajectory.as_dict()["terminal_decision"]["disposition"] == "STOP"
    assert "obs:forged:0:99" not in trajectory.steps[0].revealed_observation_ids


# -- A3: suite constraints stay frozen once the suite digest is taken ------


@pytest.fixture()
def suite_report():
    # Deliberately function-scoped: a test that proves a mutation is refused
    # must not be able to leak that mutation into the next test if it regresses.
    return validate_suite(SUITE_MANIFEST)


def test_declared_suite_constraints_are_not_mutable_at_any_depth(suite_report):
    constraints = suite_report.manifest.constraints
    with pytest.raises(TypeError):
        constraints.cube_types["1-ACT"] = 99
    with pytest.raises(TypeError):
        constraints.fact_locations["elicited"] = 99
    with pytest.raises(TypeError):
        constraints.expected_dispositions["ACT"] = 99
    with pytest.raises(TypeError):
        del constraints.cube_types["1-ACT"]


def test_directly_constructed_suite_constraints_are_frozen_too():
    """The authored path and the constructed path carry the same semantics."""
    constraints = SuiteConstraints(
        cube_types={"1-ACT": 1},
        fact_locations={"elicited": 1},
        expected_dispositions={"ACT": 2, "STOP": 4},
    )
    with pytest.raises(TypeError):
        constraints.cube_types["3-ACT"] = 1


def test_suite_constraints_as_dict_is_plain_json_and_detached(suite_report):
    constraints = suite_report.manifest.constraints
    payload = constraints.as_dict()
    assert is_plain_json(payload)

    payload["cube_types"]["1-ACT"] = 99
    payload["expected_dispositions"]["ACT"] = 99
    assert constraints.cube_types["1-ACT"] == 1
    assert constraints.expected_dispositions["ACT"] == 6
    assert constraints.as_dict() == suite_report.as_dict()["declared_constraints"]


def test_frozen_constraints_keep_the_report_and_its_digest_in_step(suite_report):
    """A mutable constraint desynchronises a taken digest.

    ``suite_content_digest_sha256`` covers ``constraints``, so a constraint that
    could still be edited after validation would leave a report whose declared
    constraints no longer hash to the digest printed beside them.
    """
    before = suite_report.as_dict()
    with pytest.raises(TypeError):
        suite_report.manifest.constraints.expected_dispositions["ACT"] = 99
    after = suite_report.as_dict()
    assert after == before
    # The digest was taken over exactly the constraints the report still shows.
    assert (
        after["suite_content_digest_sha256"]
        == validate_suite(SUITE_MANIFEST).content_digest
    )


# -- F16: exact exception types for frozen dataclasses ---------------------


def test_frozen_dataclasses_raise_frozen_instance_error(cube):
    variant = cube.cell_variant("S0_P0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.expected_disposition = "STOP"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.observations[0].source = "user_answer"  # type: ignore[misc]


def nested_value_card() -> dict[str, Any]:
    """A card whose deciding facts carry nested JSON-safe structures."""
    card = mutate()
    for arm, quote in (("S0", 180), ("S1", 640)):
        card["state_axis"][arm]["facts"] = {
            "repair_quote_gbp": quote,
            "quote_breakdown": {"labour": quote - 60, "notes": ["synthetic"]},
        }
    card["facts"]["quote_breakdown"] = {"availability": "initial"}
    return card
