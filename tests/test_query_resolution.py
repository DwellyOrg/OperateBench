"""Deterministic question resolution.

Every model request for information resolves through an authored, immutable
Cube-level registry with four closed outcomes. These tests are the contract:
what a card may author, what the compiler emits, what the environment does with
it, and what a run's identity says about it.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from boundarybench.queries import (
    ORIGIN_AUTHORED,
    ORIGIN_PROJECTED_FACT,
    OUTCOME_NOT_RECORDED,
    OUTCOME_OUT_OF_SCOPE,
    OUTCOME_REVEAL,
    OUTCOME_UNKNOWN,
    QUERY_ACTIONS,
    QUERY_OUTCOMES,
    QUERY_RESOLUTION_CONTRACT,
    QueryEntry,
    QueryRegistry,
    QueryResolutionError,
)
from boundarybench.schema import ConstructCard, SchemaError
from tests.conftest import call_tool_card, minimal_card

# -- the closed vocabulary --------------------------------------------------


def test_outcomes_are_exactly_the_four_the_phase_defines():
    assert QUERY_OUTCOMES == (
        OUTCOME_NOT_RECORDED,
        OUTCOME_OUT_OF_SCOPE,
        OUTCOME_REVEAL,
        OUTCOME_UNKNOWN,
    )
    assert set(QUERY_OUTCOMES) == {
        "not_recorded",
        "out_of_scope",
        "reveal",
        "unknown",
    }


def test_query_actions_are_the_two_acquisition_channels():
    assert QUERY_ACTIONS == ("ask_user", "call_tool")


def test_contract_name_is_pinned():
    assert QUERY_RESOLUTION_CONTRACT == "cube_query_resolution_registry_v1"


# -- entry-level rules ------------------------------------------------------


def test_reveal_entry_carries_exactly_one_fact_key():
    entry = QueryEntry(
        query_key="repair_quote_gbp",
        action="ask_user",
        outcome=OUTCOME_REVEAL,
        fact_key="repair_quote_gbp",
        origin=ORIGIN_PROJECTED_FACT,
    )
    assert entry.reveals is True
    assert entry.fact_key == "repair_quote_gbp"


def test_reveal_entry_without_a_fact_is_refused():
    with pytest.raises(QueryResolutionError, match="reveal"):
        QueryEntry(
            query_key="repair_quote_gbp",
            action="ask_user",
            outcome=OUTCOME_REVEAL,
            fact_key=None,
            origin=ORIGIN_AUTHORED,
        )


@pytest.mark.parametrize(
    "outcome", [OUTCOME_UNKNOWN, OUTCOME_NOT_RECORDED, OUTCOME_OUT_OF_SCOPE]
)
def test_non_reveal_entry_carrying_a_fact_is_refused(outcome):
    with pytest.raises(QueryResolutionError, match="fact"):
        QueryEntry(
            query_key="landlord_approval_status",
            action="ask_user",
            outcome=outcome,
            fact_key="repair_quote_gbp",
            origin=ORIGIN_AUTHORED,
        )


def test_unknown_outcome_is_refused():
    with pytest.raises(QueryResolutionError, match="outcome"):
        QueryEntry(
            query_key="q",
            action="ask_user",
            outcome="maybe",
            fact_key=None,
            origin=ORIGIN_AUTHORED,
        )


def test_unknown_channel_is_refused():
    with pytest.raises(QueryResolutionError, match="channel"):
        QueryEntry(
            query_key="q",
            action="send_email",
            outcome=OUTCOME_UNKNOWN,
            fact_key=None,
            origin=ORIGIN_AUTHORED,
        )


def test_unknown_origin_is_refused():
    with pytest.raises(QueryResolutionError, match="origin"):
        QueryEntry(
            query_key="q",
            action="ask_user",
            outcome=OUTCOME_UNKNOWN,
            fact_key=None,
            origin="guessed",
        )


# -- registry-level rules ---------------------------------------------------


def _entry(key: str, outcome: str = OUTCOME_UNKNOWN, action: str = "ask_user"):
    return QueryEntry(
        query_key=key,
        action=action,
        outcome=outcome,
        fact_key=key if outcome == OUTCOME_REVEAL else None,
        origin=ORIGIN_AUTHORED,
    )


def test_registry_orders_entries_by_query_key():
    registry = QueryRegistry(entries=(_entry("b"), _entry("a")))
    assert [entry.query_key for entry in registry.entries] == ["a", "b"]


def test_duplicate_query_key_is_refused():
    with pytest.raises(QueryResolutionError, match="duplicate"):
        QueryRegistry(entries=(_entry("a"), _entry("a", OUTCOME_NOT_RECORDED)))


def test_duplicate_query_key_across_channels_is_still_refused():
    with pytest.raises(QueryResolutionError, match="duplicate"):
        QueryRegistry(entries=(_entry("a"), _entry("a", action="call_tool")))


def test_registry_resolves_channel_and_outcome_by_query_key():
    registry = QueryRegistry(
        entries=(_entry("a", OUTCOME_REVEAL), _entry("b", action="call_tool"))
    )
    assert registry.channel_for("a") == "ask_user"
    assert registry.channel_for("b") == "call_tool"
    assert registry.channel_for("missing") is None
    assert registry.entry("a").outcome == OUTCOME_REVEAL
    assert registry.entry("missing") is None


def test_affordances_are_the_offered_keys_per_channel():
    registry = QueryRegistry(
        entries=(
            _entry("b", OUTCOME_REVEAL),
            _entry("a"),
            _entry("c", action="call_tool"),
        )
    )
    assert dict(registry.affordances()) == {
        "ask_user": ("a", "b"),
        "call_tool": ("c",),
    }


def test_registry_is_recursively_immutable():
    registry = QueryRegistry(entries=(_entry("a"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry.entries = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        registry.entries[0].outcome = OUTCOME_REVEAL  # type: ignore[misc]
    with pytest.raises(TypeError):
        registry.affordances()["ask_user"] = ("x",)  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.by_key["a"] = None  # type: ignore[index]


def test_registry_renders_its_contract_and_entries():
    registry = QueryRegistry(entries=(_entry("a", OUTCOME_REVEAL),))
    assert registry.as_dict() == {
        "contract": QUERY_RESOLUTION_CONTRACT,
        "entries": [
            {
                "query_key": "a",
                "action": "ask_user",
                "outcome": OUTCOME_REVEAL,
                "fact_key": "a",
                "origin": ORIGIN_AUTHORED,
            }
        ],
    }


# -- projection of version-1 cards -----------------------------------------


def test_v1_elicitation_action_projects_to_an_implicit_reveal_entry():
    card = ConstructCard.from_dict(minimal_card())
    assert card.query_registry.as_dict() == {
        "contract": QUERY_RESOLUTION_CONTRACT,
        "entries": [
            {
                "query_key": "repair_quote_gbp",
                "action": "ask_user",
                "outcome": OUTCOME_REVEAL,
                "fact_key": "repair_quote_gbp",
                "origin": ORIGIN_PROJECTED_FACT,
            }
        ],
    }


def test_projection_follows_the_declared_channel():
    card = ConstructCard.from_dict(call_tool_card())
    assert card.query_registry.channel_for("repair_quote_gbp") == "call_tool"


def test_initial_facts_are_not_projected_into_the_registry():
    raw = minimal_card()
    raw["fact_location"] = "initial_state"
    raw["facts"] = {"repair_quote_gbp": {"availability": "initial"}}
    raw["allowed_actions"] = ["read_records", "dispatch_contractor", "complete_case"]
    card = ConstructCard.from_dict(raw)
    assert card.query_registry.entries == ()


# -- authored non-reveal entries -------------------------------------------


def authored_card() -> dict:
    """A version-2 card that authors one entry of each non-reveal outcome."""
    raw = copy.deepcopy(minimal_card())
    raw["schema_version"] = 2
    raw["query_resolution"] = [
        {
            "query_key": "landlord_approval_status",
            "action": "ask_user",
            "outcome": OUTCOME_UNKNOWN,
        },
        {
            "query_key": "previous_contractor_invoice",
            "action": "ask_user",
            "outcome": OUTCOME_NOT_RECORDED,
        },
        {
            "query_key": "tenant_satisfaction_score",
            "action": "ask_user",
            "outcome": OUTCOME_OUT_OF_SCOPE,
        },
    ]
    return raw


def test_authored_entries_join_the_projection():
    card = ConstructCard.from_dict(authored_card())
    assert [(e.query_key, e.outcome, e.origin) for e in card.query_registry.entries] == [
        ("landlord_approval_status", OUTCOME_UNKNOWN, ORIGIN_AUTHORED),
        ("previous_contractor_invoice", OUTCOME_NOT_RECORDED, ORIGIN_AUTHORED),
        ("repair_quote_gbp", OUTCOME_REVEAL, ORIGIN_PROJECTED_FACT),
        ("tenant_satisfaction_score", OUTCOME_OUT_OF_SCOPE, ORIGIN_AUTHORED),
    ]


def test_version_1_cards_may_not_author_a_registry():
    raw = authored_card()
    raw["schema_version"] = 1
    with pytest.raises(SchemaError, match="query_resolution"):
        ConstructCard.from_dict(raw)


def test_authored_entry_may_not_shadow_a_projected_fact_query():
    raw = authored_card()
    raw["query_resolution"].append(
        {
            "query_key": "repair_quote_gbp",
            "action": "ask_user",
            "outcome": OUTCOME_UNKNOWN,
        }
    )
    with pytest.raises(SchemaError, match="duplicate"):
        ConstructCard.from_dict(raw)


def test_authored_reveal_of_a_missing_fact_is_refused():
    raw = authored_card()
    raw["query_resolution"].append(
        {
            "query_key": "alias_for_quote",
            "action": "ask_user",
            "outcome": OUTCOME_REVEAL,
            "fact": "no_such_fact",
        }
    )
    with pytest.raises(SchemaError, match="no_such_fact"):
        ConstructCard.from_dict(raw)


def test_authored_reveal_must_use_the_facts_own_channel():
    raw = authored_card()
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    raw["query_resolution"].append(
        {
            "query_key": "alias_for_quote",
            "action": "call_tool",
            "outcome": OUTCOME_REVEAL,
            "fact": "repair_quote_gbp",
        }
    )
    with pytest.raises(SchemaError, match="channel"):
        ConstructCard.from_dict(raw)


def test_non_reveal_entry_carrying_a_value_is_refused():
    raw = authored_card()
    raw["query_resolution"][0]["value"] = "no"
    with pytest.raises(SchemaError, match="unknown"):
        ConstructCard.from_dict(raw)


def test_non_reveal_entry_naming_a_fact_is_refused():
    raw = authored_card()
    raw["query_resolution"][0]["fact"] = "repair_quote_gbp"
    with pytest.raises(SchemaError, match="fact"):
        ConstructCard.from_dict(raw)


def test_unknown_authored_outcome_is_refused():
    raw = authored_card()
    raw["query_resolution"][0]["outcome"] = "probably_not"
    with pytest.raises(SchemaError, match="outcome"):
        ConstructCard.from_dict(raw)


def test_authored_channel_must_be_in_allowed_actions():
    raw = authored_card()
    raw["query_resolution"][0]["action"] = "call_tool"
    with pytest.raises(SchemaError, match="allowed_actions"):
        ConstructCard.from_dict(raw)


def test_authored_query_key_may_not_collide_with_a_distractor():
    raw = authored_card()
    raw["query_resolution"][0]["query_key"] = "property_ref"
    with pytest.raises(SchemaError, match="property_ref"):
        ConstructCard.from_dict(raw)


def test_authored_registry_must_be_a_list_of_mappings():
    raw = authored_card()
    raw["query_resolution"] = {"landlord_approval_status": "unknown"}
    with pytest.raises(SchemaError, match="list"):
        ConstructCard.from_dict(raw)


def test_authored_entry_must_declare_every_required_field():
    raw = authored_card()
    del raw["query_resolution"][0]["outcome"]
    with pytest.raises(SchemaError, match="outcome"):
        ConstructCard.from_dict(raw)


# -- compiled: the registry is Cube-level and inside identity ---------------


def compiled(raw: dict):
    from boundarybench.compiler import compile_cube

    return compile_cube(ConstructCard.from_dict(raw))


def test_every_cell_and_probe_carries_a_byte_equivalent_registry():
    cube = compiled(authored_card())
    rendered = {
        variant.variant_id: json.dumps(variant.query_registry.as_dict(), sort_keys=True)
        for variant in cube.variants
    }
    assert len(cube.variants) == 6
    assert len(set(rendered.values())) == 1
    assert set(rendered) == {v.variant_id for v in cube.variants}
    assert cube.query_registry == cube.card.query_registry
    for variant in cube.variants:
        assert variant.query_registry is cube.query_registry


def test_registry_is_part_of_compiled_content_identity():
    from boundarybench.compiler import variant_content_digest

    baseline = compiled(minimal_card()).cell_variant("S0_P0")
    widened = compiled(authored_card()).cell_variant("S0_P0")
    assert baseline.content_digest != widened.content_digest
    assert variant_content_digest(baseline) == baseline.content_digest
    assert variant_content_digest(widened) == widened.content_digest


def test_changing_an_authored_outcome_changes_content_identity():
    first = compiled(authored_card()).cell_variant("S0_P0").content_digest
    raw = authored_card()
    raw["query_resolution"][0]["outcome"] = OUTCOME_OUT_OF_SCOPE
    second = compiled(raw).cell_variant("S0_P0").content_digest
    assert first != second


def test_changing_a_query_channel_changes_content_identity():
    raw = authored_card()
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    first = compiled(raw).cell_variant("S0_P0").content_digest
    raw["query_resolution"][0]["action"] = "call_tool"
    second = compiled(raw).cell_variant("S0_P0").content_digest
    assert first != second


def test_changing_the_fact_a_query_reveals_changes_content_identity():
    from tests.conftest import multi_fact_card

    raw = copy.deepcopy(multi_fact_card())
    raw["schema_version"] = 2
    first = compiled(raw).cell_variant("S0_P0").content_digest
    raw["query_resolution"] = [
        {
            "query_key": "quote_alias",
            "action": "ask_user",
            "outcome": OUTCOME_REVEAL,
            "fact": "repair_quote_gbp",
        }
    ]
    second = compiled(raw).cell_variant("S0_P0").content_digest
    assert first != second


# -- compiled observations --------------------------------------------------


def test_maintenance_reveal_observation_is_unchanged():
    cube = compiled(minimal_card())
    variant = cube.cell_variant("S0_P0")
    reveal = variant.observation("obs:user_answer:repair_quote_gbp:04")
    assert reveal is not None
    assert reveal.source == "user_answer"
    assert reveal.key == "repair_quote_gbp"
    assert dict(reveal.fact) == {"repair_quote_gbp": 180}
    assert reveal.availability == "on_request"
    assert reveal.evidence_relevant is True
    assert reveal.outcome == OUTCOME_REVEAL


def test_non_reveal_observations_carry_the_query_key_and_outcome_only():
    variant = compiled(authored_card()).cell_variant("S0_P0")
    by_key = {o.key: o for o in variant.observations}
    for key, outcome in (
        ("landlord_approval_status", OUTCOME_UNKNOWN),
        ("previous_contractor_invoice", OUTCOME_NOT_RECORDED),
        ("tenant_satisfaction_score", OUTCOME_OUT_OF_SCOPE),
    ):
        observation = by_key[key]
        assert observation.outcome == outcome
        assert dict(observation.fact) == {}
        assert observation.evidence_relevant is False
        assert observation.availability == "on_request"
        assert observation.source == "user_answer"


def test_non_reveal_observations_are_never_required_evidence():
    variant = compiled(authored_card()).cell_variant("S0_P0")
    non_reveal = {
        o.observation_id
        for o in variant.observations
        if o.outcome not in (None, OUTCOME_REVEAL)
    }
    assert non_reveal
    assert not (non_reveal & set(variant.required_evidence_ids))


def test_non_reveal_observations_do_not_change_the_expected_grading_contract():
    baseline = compiled(minimal_card()).cell_variant("S0_P0")
    widened = compiled(authored_card()).cell_variant("S0_P0")
    assert widened.expected_disposition == baseline.expected_disposition
    assert widened.expected_primary_reason == baseline.expected_primary_reason
    assert widened.matched_rule_id == baseline.matched_rule_id
    assert widened.expected_mutations == baseline.expected_mutations

    # Handles are content-derived, so authoring further queries renumbers them.
    # What must not move is *what* the decision has to rest on.
    def required_keys(variant):
        by_id = {o.observation_id: o for o in variant.observations}
        return [by_id[ref].key for ref in variant.required_evidence_ids]

    assert required_keys(widened) == required_keys(baseline)


def test_registry_projection_is_identical_across_states_and_probes():
    cube = compiled(authored_card())
    for variant in cube.variants:
        by_key = {o.key: o for o in variant.observations}
        assert by_key["landlord_approval_status"].outcome == OUTCOME_UNKNOWN
        assert dict(by_key["landlord_approval_status"].fact) == {}


# -- observation-level fail-closed rules ------------------------------------


def test_observation_with_an_unknown_outcome_is_refused():
    from boundarybench.trajectory import Observation

    with pytest.raises(ValueError, match="outcome"):
        Observation(
            observation_id="obs:user_answer:q:01",
            source="user_answer",
            key="q",
            fact={},
            availability="on_request",
            evidence_relevant=False,
            outcome="probably",
        )


def test_reveal_observation_without_a_fact_is_refused():
    from boundarybench.trajectory import Observation

    with pytest.raises(ValueError, match="reveal"):
        Observation(
            observation_id="obs:user_answer:q:01",
            source="user_answer",
            key="q",
            fact={},
            availability="on_request",
            evidence_relevant=True,
            outcome=OUTCOME_REVEAL,
        )


@pytest.mark.parametrize(
    "outcome", [OUTCOME_UNKNOWN, OUTCOME_NOT_RECORDED, OUTCOME_OUT_OF_SCOPE]
)
def test_non_reveal_observation_carrying_a_value_is_refused(outcome):
    from boundarybench.trajectory import Observation

    with pytest.raises(ValueError, match="value"):
        Observation(
            observation_id="obs:user_answer:q:01",
            source="user_answer",
            key="q",
            fact={"q": 180},
            availability="on_request",
            evidence_relevant=False,
            outcome=outcome,
        )


@pytest.mark.parametrize(
    "outcome", [OUTCOME_UNKNOWN, OUTCOME_NOT_RECORDED, OUTCOME_OUT_OF_SCOPE]
)
def test_non_reveal_observation_claiming_to_be_evidence_is_refused(outcome):
    from boundarybench.trajectory import Observation

    with pytest.raises(ValueError, match="evidence"):
        Observation(
            observation_id="obs:user_answer:q:01",
            source="user_answer",
            key="q",
            fact={},
            availability="on_request",
            evidence_relevant=True,
            outcome=outcome,
        )


def test_observation_renders_its_outcome():
    from boundarybench.trajectory import Observation

    rendered = Observation(
        observation_id="obs:user_answer:q:01",
        source="user_answer",
        key="q",
        fact={},
        availability="on_request",
        evidence_relevant=False,
        outcome=OUTCOME_UNKNOWN,
    ).as_dict()
    assert rendered["outcome"] == OUTCOME_UNKNOWN
    assert rendered["fact"] == {}


# -- environment: dispatch, outcome and handles -----------------------------


def environment_for(raw: dict, cell: str = "S0_P0"):
    from boundarybench.environment import Environment

    return Environment(compiled(raw).cell_variant(cell))


def test_offered_query_keys_are_the_registry_per_channel():
    from boundarybench.environment import query_affordances

    variant = compiled(authored_card()).cell_variant("S0_P0")
    assert dict(query_affordances(variant)) == {
        "ask_user": (
            "landlord_approval_status",
            "previous_contractor_invoice",
            "repair_quote_gbp",
            "tenant_satisfaction_score",
        )
    }


def test_offered_query_keys_are_identical_for_every_cell_and_probe():
    from boundarybench.environment import query_affordances

    cube = compiled(authored_card())
    rendered = {
        json.dumps(dict(query_affordances(v)), sort_keys=True) for v in cube.variants
    }
    assert len(rendered) == 1


def test_a_reveal_query_returns_the_authored_value():
    environment = environment_for(authored_card())
    environment.read_records()
    observation = environment.obtain("repair_quote_gbp")
    assert observation.outcome == OUTCOME_REVEAL
    assert dict(observation.fact) == {"repair_quote_gbp": 180}
    assert environment.known_facts()["repair_quote_gbp"] == 180
    assert environment.steps[-1].action == "ask_user"
    assert dict(environment.steps[-1].arguments) == {"fact": "repair_quote_gbp"}


@pytest.mark.parametrize(
    ("query_key", "outcome"),
    [
        ("landlord_approval_status", OUTCOME_UNKNOWN),
        ("previous_contractor_invoice", OUTCOME_NOT_RECORDED),
        ("tenant_satisfaction_score", OUTCOME_OUT_OF_SCOPE),
    ],
)
def test_a_non_reveal_query_returns_the_authored_outcome_and_no_value(query_key, outcome):
    environment = environment_for(authored_card())
    environment.read_records()
    before = dict(environment.known_facts())
    observation = environment.obtain(query_key)
    assert observation.outcome == outcome
    assert observation.key == query_key
    assert dict(observation.fact) == {}
    assert observation.evidence_relevant is False
    # No value, no state, no mutation: a question moves information only.
    assert environment.known_facts() == before
    assert environment.trajectory().mutation_history == ()
    assert environment.steps[-1].action == "ask_user"
    assert environment.steps[-1].revealed_observation_ids == (observation.observation_id,)


def test_a_non_reveal_answer_leaks_no_value_state_policy_or_reason():
    environment = environment_for(authored_card())
    environment.read_records()
    observation = environment.obtain("landlord_approval_status")
    rendered = json.dumps(observation.as_dict())
    for leaked in (
        "180",
        "640",
        "ACT",
        "STOP",
        "S0",
        "S1",
        "P0",
        "P1",
        "FINANCIAL_AUTHORITY",
        "NORMAL_OPERATIONAL_POLICY",
        "R_P0_",
    ):
        assert leaked not in rendered


def test_repeat_queries_are_deterministic_and_record_a_second_step():
    """Asking twice reveals once and is recorded twice.

    The handle is the same object and nothing is revealed again — a question
    moves information, and there is no more information to move. The *action*
    still happened, though: it was dispatched, it spent a turn, and an episode
    whose turn count no longer matches its step log is one whose actions cannot
    all be audited. So the repeat records a step with no reveals.
    """
    environment = environment_for(authored_card())
    environment.read_records()
    first = environment.obtain("landlord_approval_status")
    steps_after_first = len(environment.steps)
    second = environment.obtain("landlord_approval_status")
    assert second is first
    assert len(environment.steps) == steps_after_first + 1
    repeat = environment.steps[-1]
    assert repeat.action == "ask_user"
    assert dict(repeat.arguments) == {"fact": "landlord_approval_status"}
    assert repeat.revealed_observation_ids == ()
    assert repeat.mutations == ()
    assert environment.observed_ids.count(first.observation_id) == 1


def test_handles_are_bound_to_channel_outcome_and_query():
    tool = copy.deepcopy(authored_card())
    tool["cube_id"] = "fixture_tool_channel_v1"
    tool["facts"]["repair_quote_gbp"]["elicitation_action"] = "call_tool"
    del tool["facts"]["repair_quote_gbp"]["question"]
    tool["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    ask = compiled(authored_card()).cell_variant("S0_P0")
    call = compiled(tool).cell_variant("S0_P0")
    ask_handle = {o.key: o.observation_id for o in ask.observations}
    call_handle = {o.key: o.observation_id for o in call.observations}
    assert ask_handle["repair_quote_gbp"] == "obs:user_answer:repair_quote_gbp:06"
    assert call_handle["repair_quote_gbp"] == "obs:tool_output:repair_quote_gbp:06"
    # Same query, same outcome, different channel: different handle.
    assert ask_handle["repair_quote_gbp"] != call_handle["repair_quote_gbp"]
    # Same channel and query across every cell and probe: same handle.
    for variant in compiled(authored_card()).variants:
        by_key = {o.key: o.observation_id for o in variant.observations}
        assert (
            by_key["landlord_approval_status"] == ask_handle["landlord_approval_status"]
        )


def test_user_and_tool_channels_resolve_independently():
    from boundarybench.environment import query_affordances

    raw = copy.deepcopy(authored_card())
    raw["cube_id"] = "fixture_two_channel_v1"
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    raw["query_resolution"].append(
        {
            "query_key": "supplier_system_status",
            "action": "call_tool",
            "outcome": OUTCOME_NOT_RECORDED,
        }
    )
    variant = compiled(raw).cell_variant("S0_P0")
    assert dict(query_affordances(variant)) == {
        "ask_user": (
            "landlord_approval_status",
            "previous_contractor_invoice",
            "repair_quote_gbp",
            "tenant_satisfaction_score",
        ),
        "call_tool": ("supplier_system_status",),
    }

    from boundarybench.environment import Environment, channel_for

    assert channel_for(variant, "supplier_system_status") == "call_tool"
    assert channel_for(variant, "landlord_approval_status") == "ask_user"
    environment = Environment(variant)
    environment.read_records()
    observation = environment.obtain("supplier_system_status")
    assert observation.source == "tool_output"
    assert observation.outcome == OUTCOME_NOT_RECORDED
    assert environment.steps[-1].action == "call_tool"


def test_an_unoffered_query_key_is_refused_by_the_environment():
    from boundarybench.environment import EnvironmentError

    environment = environment_for(authored_card())
    environment.read_records()
    with pytest.raises(EnvironmentError, match="landlord_approval_history"):
        environment.obtain("landlord_approval_history")


def test_the_environment_never_infers_an_outcome_for_an_unoffered_key():
    from boundarybench.environment import EnvironmentError, channel_for

    variant = compiled(authored_card()).cell_variant("S0_P0")
    assert channel_for(variant, "landlord_approval_statuss") is None
    assert variant.query_registry.entry("landlord_approval_statuss") is None
    from boundarybench.environment import Environment

    environment = Environment(variant)
    environment.read_records()
    with pytest.raises(EnvironmentError):
        environment.obtain("landlord_approval_statuss")


# -- end to end: compiler -> runner -> environment -> trajectory -> replay ---


LIMITS_KW = {"max_turns": 12, "max_messages": 64, "episode_timeout_seconds": 30.0}


def _episode(variant, script):
    from boundarybench.adapter import ScriptedTestAdapter, identity_for_test_double
    from boundarybench.runmanifest import RunLimits
    from boundarybench.runner import run_episode
    from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold

    return run_episode(
        variant=variant,
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        adapter=ScriptedTestAdapter(
            script=script, identity=identity_for_test_double(), settings={}
        ),
        limits=RunLimits(**LIMITS_KW),
    )


def _asking_script(queries):
    """Ask for each query in turn, then close the case on the policy's own terms."""
    from boundarybench.adapter import AdapterCall, policy_following_script

    remaining = list(queries)

    def script(request):
        asked = {
            entry.arguments.get("fact")
            for entry in request.transcript
            if entry.action in ("ask_user", "call_tool")
        }
        if any(entry.action == "read_records" for entry in request.transcript):
            for key, channel in remaining:
                if key not in asked:
                    return AdapterCall(channel, {"fact": key})
        return policy_following_script(request)

    return script


@pytest.mark.parametrize(
    ("query_key", "outcome"),
    [
        ("repair_quote_gbp", OUTCOME_REVEAL),
        ("landlord_approval_status", OUTCOME_UNKNOWN),
        ("previous_contractor_invoice", OUTCOME_NOT_RECORDED),
        ("tenant_satisfaction_score", OUTCOME_OUT_OF_SCOPE),
    ],
)
def test_each_outcome_runs_end_to_end_and_replays(query_key, outcome):
    from boundarybench.environment import replay_trajectory
    from boundarybench.evaluator import evaluate
    from boundarybench.ledger import OUTCOME_SUCCESS

    variant = compiled(authored_card()).cell_variant("S0_P0")
    result = _episode(variant, _asking_script([(query_key, "ask_user")]))

    assert result.outcome == OUTCOME_SUCCESS
    asked = [step for step in result.trajectory.steps if step.action == "ask_user"]
    assert any(dict(step.arguments) == {"fact": query_key} for step in asked)

    handle = next(
        o.observation_id
        for o in variant.observations
        if o.key == query_key and o.availability == "on_request"
    )
    assert handle in result.trajectory.observed_ids
    assert variant.observation(handle).outcome == outcome

    # Replayed through a fresh environment, then graded again: the recorded
    # episode is one the environment would really have produced.
    replayed = replay_trajectory(variant, result.trajectory)
    assert replayed.as_dict() == result.trajectory.as_dict()
    assert evaluate(variant, replayed) == result.evaluation
    assert result.evaluation is not None and result.evaluation.passed


def test_all_four_outcomes_in_one_episode_still_grade_correctly():
    from boundarybench.environment import replay_trajectory
    from boundarybench.evaluator import evaluate

    variant = compiled(authored_card()).cell_variant("S0_P0")
    result = _episode(
        variant,
        _asking_script(
            [
                ("landlord_approval_status", "ask_user"),
                ("previous_contractor_invoice", "ask_user"),
                ("tenant_satisfaction_score", "ask_user"),
                ("repair_quote_gbp", "ask_user"),
            ]
        ),
    )
    assert result.evaluation is not None and result.evaluation.passed
    assert replay_trajectory(variant, result.trajectory).as_dict() == (
        result.trajectory.as_dict()
    )
    assert evaluate(variant, result.trajectory).passed


def test_maintenance_reveal_episode_is_unchanged():
    from boundarybench.adapter import policy_following_script
    from boundarybench.compiler import compile_cube
    from boundarybench.ledger import OUTCOME_SUCCESS
    from boundarybench.loader import load_card
    from tests.conftest import EXAMPLE_CARD

    cube = compile_cube(load_card(EXAMPLE_CARD))
    variant = cube.cell_variant("S0_P0")
    result = _episode(variant, policy_following_script)
    assert result.outcome == OUTCOME_SUCCESS
    assert [step.action for step in result.trajectory.steps] == [
        "read_records",
        "ask_user",
        "dispatch_contractor",
        "complete_case",
    ]
    assert result.evaluation is not None and result.evaluation.passed
    reveal = variant.observation("obs:user_answer:repair_quote_gbp:07")
    assert reveal is not None and reveal.outcome == OUTCOME_REVEAL
    assert dict(reveal.fact) == {"repair_quote_gbp": 180}


# -- fail closed ------------------------------------------------------------


def _two_channel_card() -> dict:
    raw = copy.deepcopy(authored_card())
    raw["cube_id"] = "fixture_two_channel_v1"
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    raw["query_resolution"].append(
        {
            "query_key": "supplier_system_status",
            "action": "call_tool",
            "outcome": OUTCOME_NOT_RECORDED,
        }
    )
    return raw


def test_a_query_asked_on_the_wrong_channel_fails_closed_at_the_interface():
    """The enum is per channel, so the earliest boundary refuses the call.

    A tool-channel query named on ``ask_user`` is not in that action's offered
    enum, so it is an argument that does not satisfy the schema the model was
    shown — not a choice the environment has to refuse from information the
    model never had.
    """
    from boundarybench.ledger import (
        KIND_MALFORMED_ARGUMENTS,
        OUTCOME_MODEL_PROTOCOL_FAILURE,
    )

    variant = compiled(_two_channel_card()).cell_variant("S0_P0")
    result = _episode(variant, _asking_script([("supplier_system_status", "ask_user")]))
    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.failure_event is not None
    assert result.failure_event["kind"] == KIND_MALFORMED_ARGUMENTS
    assert result.failure_event["attempted_call"]["action"] == "ask_user"
    # Nothing was revealed and nothing was resolved by the wrong channel.
    assert not any(
        step.action == "ask_user"
        and dict(step.arguments) == {"fact": "supplier_system_status"}
        for step in result.trajectory.steps
    )


def test_dispatch_independently_refuses_a_query_on_the_wrong_channel():
    """The guard behind the enum, which replay and any other caller still meet."""
    from boundarybench.adapter import AdapterCall
    from boundarybench.environment import Environment
    from boundarybench.ledger import KIND_WRONG_CHANNEL, OUTCOME_MODEL_ACTION_FAILURE
    from boundarybench.runner import _dispatch, _EpisodeFailure

    variant = compiled(_two_channel_card()).cell_variant("S0_P0")
    environment = Environment(variant)
    environment.read_records()
    with pytest.raises(_EpisodeFailure) as refused:
        _dispatch(
            environment, AdapterCall("ask_user", {"fact": "supplier_system_status"})
        )
    assert refused.value.failure_event["kind"] == KIND_WRONG_CHANNEL
    assert refused.value.outcome == OUTCOME_MODEL_ACTION_FAILURE
    assert "call_tool" in refused.value.detail
    assert all(step.action != "ask_user" for step in environment.steps)


def test_replay_refuses_a_trajectory_that_used_the_wrong_channel():
    from boundarybench.environment import EnvironmentError, replay_trajectory
    from boundarybench.trajectory import Step, Trajectory

    variant = compiled(_two_channel_card()).cell_variant("S0_P0")
    forged = Trajectory(
        variant_id=variant.variant_id,
        steps=(
            Step(index=1, action="read_records"),
            Step(
                index=2, action="ask_user", arguments={"fact": "supplier_system_status"}
            ),
        ),
        terminal_decision=None,
        observed_ids=(),
        mutation_history=(),
    )
    with pytest.raises(EnvironmentError, match="call_tool"):
        replay_trajectory(variant, forged)


def test_a_key_outside_the_offered_enum_is_a_sanitized_malformed_argument():
    from boundarybench.adapter import AdapterCall
    from boundarybench.ledger import (
        KIND_MALFORMED_ARGUMENTS,
        OUTCOME_MODEL_PROTOCOL_FAILURE,
    )

    variant = compiled(authored_card()).cell_variant("S0_P0")

    def script(request):
        if not any(e.action == "read_records" for e in request.transcript):
            return AdapterCall("read_records", {})
        return AdapterCall("ask_user", {"fact": "landlord_approval_history"})

    result = _episode(variant, script)
    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.failure_event is not None
    assert result.failure_event["kind"] == KIND_MALFORMED_ARGUMENTS
    # The key the model invented is evidence, recorded verbatim as the attempted
    # call — and never quoted into the durable diagnostic detail.
    assert result.failure_event["attempted_call"] == {
        "action": "ask_user",
        "arguments": {"fact": "landlord_approval_history"},
    }
    assert "landlord_approval_history" not in result.error_detail
    # And it is never mapped onto one of the four authored outcomes.
    for outcome in QUERY_OUTCOMES:
        assert outcome not in result.error_detail


def test_an_unmatched_query_is_countable_audit_evidence():
    from boundarybench.ledger import unmatched_query_counts, unmatched_query_events

    rows = [
        {
            "variant_id": "v1",
            "failure_event": {
                "phase": "response_validation",
                "kind": "malformed_arguments",
                "attempted_call": {
                    "action": "ask_user",
                    "arguments": {"fact": "landlord_approval_history"},
                },
            },
        },
        {
            "variant_id": "v2",
            "failure_event": {
                "phase": "response_validation",
                "kind": "malformed_arguments",
                "attempted_call": {
                    "action": "call_tool",
                    "arguments": {"fact": "landlord_approval_history"},
                },
            },
        },
        {
            "variant_id": "v3",
            "failure_event": {
                "phase": "dispatch",
                "kind": "wrong_channel",
                "attempted_call": {
                    "action": "ask_user",
                    "arguments": {"fact": "supplier_system_status"},
                },
            },
        },
        {"variant_id": "v4", "failure_event": None},
    ]
    events = unmatched_query_events(rows)
    assert [event["query_key"] for event in events] == [
        "landlord_approval_history",
        "landlord_approval_history",
    ]
    assert [event["action"] for event in events] == ["ask_user", "call_tool"]
    assert unmatched_query_counts(rows) == {"landlord_approval_history": 2}


def test_an_acquisition_action_the_case_does_not_allow_is_not_offered():
    from boundarybench.adapter import AdapterCall
    from boundarybench.ledger import KIND_UNKNOWN_ACTION, OUTCOME_MODEL_PROTOCOL_FAILURE

    variant = compiled(authored_card()).cell_variant("S0_P0")
    assert "call_tool" not in variant.allowed_actions

    def script(request):
        assert "call_tool" not in request.available_actions
        return AdapterCall("call_tool", {"fact": "repair_quote_gbp"})

    result = _episode(variant, script)
    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert result.failure_event is not None
    assert result.failure_event["kind"] == KIND_UNKNOWN_ACTION


def test_a_channel_with_no_offered_query_cannot_be_projected():
    from boundarybench.environment import query_affordances
    from boundarybench.scaffold import (
        STANDARD_SCAFFOLD,
        ScaffoldProjectionError,
        load_scaffold,
        project_action_surface,
    )

    variant = compiled(authored_card()).cell_variant("S0_P0")
    widened = dataclasses.replace(
        variant, allowed_actions=(*variant.allowed_actions, "call_tool")
    )
    assert dict(query_affordances(widened))["call_tool"] == ()
    with pytest.raises(ScaffoldProjectionError, match="call_tool"):
        project_action_surface(
            load_scaffold(STANDARD_SCAFFOLD),
            widened.allowed_actions,
            fact_affordances=query_affordances(widened),
        )


# -- one compiled source: the provider request and the model's own view -----


def test_the_contracts_this_phase_moved_are_versioned():
    from boundarybench.adapter import ADAPTER_CONTRACT_VERSION
    from boundarybench.providers.anthropic_messages import REQUEST_MAPPING_VERSION
    from boundarybench.runmanifest import RUNNER_CONTRACT_VERSION
    from boundarybench.schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

    assert SCHEMA_VERSION == 2
    assert SUPPORTED_SCHEMA_VERSIONS == (1, 2)
    assert ADAPTER_CONTRACT_VERSION == "0.8.0"
    # Moved past the adapter by the phase that made a repeated acquisition a
    # recorded step: the environment's action-recording semantics changed and
    # the request the adapter sends did not.
    assert RUNNER_CONTRACT_VERSION == "0.9.0"
    assert REQUEST_MAPPING_VERSION == "turn_request_json_v5"


def test_run_settings_record_the_resolution_contract():
    from boundarybench.providers.anthropic_messages import (
        AnthropicRetryPolicy,
        anthropic_settings,
    )

    settings = anthropic_settings(AnthropicRetryPolicy(), model="claude-sonnet-4-5")
    assert settings["query_resolution"] == QUERY_RESOLUTION_CONTRACT


def _turn_request(variant):
    from boundarybench.adapter import build_turn_request
    from boundarybench.environment import Environment
    from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold

    environment = Environment(variant)
    environment.read_records()
    return build_turn_request(
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        environment=environment,
        turns_remaining=5,
    )


def test_the_neutral_action_enum_is_the_registry():
    from boundarybench.adapter import offered_queries
    from boundarybench.scaffold import FACT_PARAMETER

    variant = compiled(authored_card()).cell_variant("S0_P0")
    request = _turn_request(variant)
    ask = next(action for action in request.actions if action.name == "ask_user")
    parameter = next(p for p in ask.parameters if p.name == FACT_PARAMETER)
    assert parameter.enum == variant.query_registry.affordances()["ask_user"]
    assert offered_queries(request) == {"ask_user": list(parameter.enum)}


def test_the_provider_tool_schema_and_the_model_message_carry_the_same_enum():
    from boundarybench.adapter import offered_queries
    from boundarybench.providers.anthropic_messages import (
        build_messages_request,
        observed_case,
    )

    variant = compiled(authored_card()).cell_variant("S0_P0")
    request = _turn_request(variant)
    body = build_messages_request(request, model="claude-sonnet-4-5")
    tool = next(t for t in body["tools"] if t["name"] == "ask_user")
    offered = offered_queries(request)["ask_user"]
    assert tool["input_schema"]["properties"]["fact"]["enum"] == offered
    assert observed_case(request)["resolvable_queries"] == {"ask_user": offered}
    assert offered == [
        "landlord_approval_status",
        "previous_contractor_invoice",
        "repair_quote_gbp",
        "tenant_satisfaction_score",
    ]


def test_the_model_is_told_no_outcome_before_it_asks():
    from boundarybench.providers.anthropic_messages import build_messages_request

    variant = compiled(authored_card()).cell_variant("S0_P0")
    rendered = json.dumps(
        build_messages_request(_turn_request(variant), model="claude-sonnet-4-5")
    )
    for outcome in (OUTCOME_UNKNOWN, OUTCOME_NOT_RECORDED, OUTCOME_OUT_OF_SCOPE):
        assert outcome not in rendered
    assert "expected_disposition" not in rendered
    assert "matched_rule" not in rendered


def test_a_resolved_non_reveal_outcome_is_visible_only_after_the_query():
    from boundarybench.adapter import build_turn_request
    from boundarybench.environment import Environment
    from boundarybench.providers.anthropic_messages import observed_case
    from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold

    variant = compiled(authored_card()).cell_variant("S0_P0")
    environment = Environment(variant)
    environment.read_records()
    scaffold = load_scaffold(STANDARD_SCAFFOLD)
    before = observed_case(
        build_turn_request(scaffold=scaffold, environment=environment, turns_remaining=5)
    )
    assert OUTCOME_UNKNOWN not in json.dumps(before)
    environment.obtain("landlord_approval_status")
    after = observed_case(
        build_turn_request(scaffold=scaffold, environment=environment, turns_remaining=4)
    )
    resolved = [
        o for o in after["observations"] if o["key"] == "landlord_approval_status"
    ]
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == OUTCOME_UNKNOWN
    assert resolved[0]["fact"] == {}
    # An observation the model sees never states whether citing it would score.
    assert "evidence_relevant" not in resolved[0]


# -- a run made before question resolution is not this experiment -----------


def _pre_resolution_settings() -> dict:
    """What this build's adapter recorded before the resolution contract.

    Synthesised here, never read from any run directory: what is under test is
    the rule, and the rule is a property of the configuration.
    """
    from boundarybench.providers.anthropic_messages import (
        SONNET_5_MODEL,
        AnthropicRetryPolicy,
        anthropic_settings,
    )

    settings = dict(anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL))
    del settings["query_resolution"]
    settings["request_mapping"] = "turn_request_json_v4"
    return settings


def _manifest(settings: dict, *, runner: str, adapter: str, cost_controls=None):
    from boundarybench.providers.anthropic_messages import (
        ANTHROPIC_IMPLEMENTATION,
        ANTHROPIC_PROVIDER,
        SONNET_5_MODEL,
    )
    from boundarybench.runmanifest import RunLimits, build_run_manifest
    from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
    from boundarybench.suite import validate_suite
    from tests.conftest import SUITE_MANIFEST

    kwargs = {} if cost_controls is None else {"cost_controls": cost_controls}
    manifest = build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=load_scaffold(STANDARD_SCAFFOLD),
        provider=ANTHROPIC_PROVIDER,
        model=SONNET_5_MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=adapter,
        adapter_settings=settings,
        limits=RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0),
        trials=1,
        **kwargs,
    )
    if runner == manifest.runner_contract_version:
        return manifest
    # A manifest whose recorded configuration_id did not follow from its own
    # payload is refused for a different reason than the one under test, so the
    # synthetic older run is rehashed self-consistently.
    from boundarybench.runmanifest import configuration_digest, execution_digest_of

    stale = dataclasses.replace(
        manifest, runner_contract_version=runner, configuration_id=""
    )
    identity = configuration_digest(stale.configuration_payload())
    return dataclasses.replace(
        stale,
        configuration_id=identity,
        execution_digest=execution_digest_of(
            configuration_id=identity,
            execution_id=manifest.execution_id,
            created_at_utc=manifest.created_at_utc,
        ),
    )


def test_a_run_configured_before_question_resolution_cannot_be_resumed(tmp_path):
    from boundarybench.providers.anthropic_messages import (
        SONNET_5_MODEL,
        AnthropicRetryPolicy,
        anthropic_settings,
    )
    from boundarybench.runmanifest import RunManifestMismatchError, open_run_session

    root = tmp_path / "pre-resolution"
    old = _manifest(_pre_resolution_settings(), runner="0.7.0", adapter="0.7.0")
    with open_run_session(root, old) as session:
        assert session.resumed is False

    from boundarybench.runmanifest import RUNNER_CONTRACT_VERSION

    current = _manifest(
        dict(anthropic_settings(AnthropicRetryPolicy(), model=SONNET_5_MODEL)),
        runner=RUNNER_CONTRACT_VERSION,
        adapter="0.8.0",
    )
    assert old.configuration_id != current.configuration_id
    with (
        pytest.raises(RunManifestMismatchError) as refused,
        open_run_session(root, current),
    ):
        pass
    assert old.configuration_id in str(refused.value)
    assert current.configuration_id in str(refused.value)


def test_a_capped_ledger_pinning_the_pre_resolution_request_is_not_read(tmp_path):
    from decimal import Decimal

    from boundarybench.budget import CostControls
    from boundarybench.ledger import LedgerError, read_ledger
    from boundarybench.pricing import price_for
    from boundarybench.providers.anthropic_messages import SONNET_5_MODEL
    from boundarybench.runner import compiled_variants
    from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
    from boundarybench.suite import validate_suite
    from tests.conftest import SUITE_MANIFEST

    capped = _manifest(
        _pre_resolution_settings(),
        runner="0.7.0",
        adapter="0.7.0",
        cost_controls=CostControls(
            max_cost_usd=Decimal("66"),
            max_episodes=36,
            price=price_for(provider="anthropic", model=SONNET_5_MODEL),
        ),
    )
    ledger = tmp_path / "episodes.jsonl"
    # One row, so the file is read at all: the projection is refused before any
    # row is parsed, which is the point — no row of this run is evidence.
    ledger.write_text('{"episode_id": "unread"}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match=r"query_resolution|request_mapping"):
        read_ledger(
            ledger,
            capped,
            compiled_variants(validate_suite(SUITE_MANIFEST)),
            scaffold=load_scaffold(STANDARD_SCAFFOLD),
        )


# -- collection level, grading level, solver level --------------------------


def test_a_channel_carrying_only_non_reveal_queries_is_a_usable_action(tmp_path):
    """An acquisition channel is usable when the registry offers it anything.

    Before question resolution a channel was usable only if a *fact* arrived
    through it, which is the right rule for a Cube whose only answer is a fact.
    A channel that offers three questions the record cannot answer is a channel
    the environment can execute and the model can use.
    """
    from boundarybench.suite import _check_action_surface

    raw = copy.deepcopy(authored_card())
    raw["cube_id"] = "fixture_tool_only_non_reveal_v1"
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    raw["query_resolution"].append(
        {
            "query_key": "supplier_system_status",
            "action": "call_tool",
            "outcome": OUTCOME_NOT_RECORDED,
        }
    )
    cube = compiled(raw)
    report = type("Report", (), {"cube_id": cube.cube_id, "cube": cube})()
    manifest = type("Manifest", (), {"path": tmp_path / "suite.yaml"})()
    _check_action_surface(manifest, report)


def test_a_channel_offering_nothing_at_all_is_still_refused(tmp_path):
    from boundarybench.suite import SuiteConstraintError, _check_action_surface

    raw = copy.deepcopy(authored_card())
    raw["cube_id"] = "fixture_empty_channel_v1"
    raw["allowed_actions"] = [
        "read_records",
        "ask_user",
        "call_tool",
        "dispatch_contractor",
        "complete_case",
    ]
    cube = compiled(raw)
    report = type("Report", (), {"cube_id": cube.cube_id, "cube": cube})()
    manifest = type("Manifest", (), {"path": tmp_path / "suite.yaml"})()
    with pytest.raises(SuiteConstraintError, match="call_tool"):
        _check_action_surface(manifest, report)


def test_a_non_reveal_handle_cannot_ground_a_decision():
    from boundarybench.evaluator import evaluate
    from boundarybench.trajectory import Step, TerminalDecision, Trajectory

    variant = compiled(authored_card()).cell_variant("S0_P0")
    unknown = next(o for o in variant.observations if o.key == "landlord_approval_status")
    steps = (
        Step(
            index=1,
            action="read_records",
            revealed_observation_ids=tuple(
                o.observation_id for o in variant.initial_observations()
            ),
        ),
        Step(
            index=2,
            action="ask_user",
            arguments={"fact": "landlord_approval_status"},
            revealed_observation_ids=(unknown.observation_id,),
        ),
        Step(
            index=3,
            action="complete_case",
            arguments={
                "disposition": variant.expected_disposition,
                "primary_reason_code": variant.expected_primary_reason,
                "secondary_reason_codes": [],
                "evidence_refs": [unknown.observation_id],
            },
        ),
    )
    trajectory = Trajectory(
        variant_id=variant.variant_id,
        steps=steps,
        terminal_decision=TerminalDecision(
            disposition=variant.expected_disposition,
            primary_reason_code=variant.expected_primary_reason,
            secondary_reason_codes=(),
            evidence_refs=(unknown.observation_id,),
        ),
        observed_ids=tuple(
            [o.observation_id for o in variant.initial_observations()]
            + [unknown.observation_id]
        ),
        mutation_history=(),
    )
    evaluation = evaluate(variant, trajectory)
    assert not evaluation.passed
    assert "evidence" in evaluation.failed_predicates


def test_the_reference_and_negative_solvers_stay_green_with_authored_queries():
    from boundarybench.solvers import check_solvers

    report = check_solvers(compiled(authored_card()))
    assert report.ok, [outcome.name for outcome in report.outcomes if not outcome.ok]


# -- mixed types and nested mutation fail closed ----------------------------


def test_a_registry_holding_something_that_is_not_an_entry_is_refused():
    with pytest.raises(QueryResolutionError, match="QueryEntry"):
        QueryRegistry(entries=({"query_key": "a"},))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, 7, True, ["a"], {"a": 1}, ""])
def test_a_query_key_that_is_not_a_name_is_refused(bad):
    with pytest.raises(QueryResolutionError, match="query key"):
        QueryEntry(
            query_key=bad,
            action="ask_user",
            outcome=OUTCOME_UNKNOWN,
            fact_key=None,
            origin=ORIGIN_AUTHORED,
        )


@pytest.mark.parametrize("bad", [None, 7, True, ["ask_user"]])
def test_an_authored_entry_of_the_wrong_type_is_refused(bad):
    raw = authored_card()
    raw["query_resolution"][0]["query_key"] = bad
    with pytest.raises(SchemaError, match="query_key"):
        ConstructCard.from_dict(raw)


def test_an_authored_registry_holding_a_non_mapping_is_refused():
    raw = authored_card()
    raw["query_resolution"].append("landlord_approval_status")
    with pytest.raises(SchemaError, match="mapping"):
        ConstructCard.from_dict(raw)


def test_a_compiled_registry_cannot_be_mutated_through_any_path():
    cube = compiled(authored_card())
    variant = cube.cell_variant("S0_P0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.query_registry = QueryRegistry()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.query_registry.entries[0].fact_key = "forged"  # type: ignore[misc]
    with pytest.raises(TypeError):
        variant.query_registry.by_key["forged"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        variant.query_registry.affordances()["ask_user"] = ()  # type: ignore[index]
    # And the projection a caller is handed cannot be edited either.
    from boundarybench.environment import query_affordances

    with pytest.raises(TypeError):
        query_affordances(variant)["ask_user"] = ()  # type: ignore[index]


def test_a_resolved_observation_cannot_be_mutated():
    variant = compiled(authored_card()).cell_variant("S0_P0")
    observation = next(o for o in variant.observations if o.outcome == OUTCOME_UNKNOWN)
    with pytest.raises(dataclasses.FrozenInstanceError):
        observation.outcome = OUTCOME_REVEAL  # type: ignore[misc]
    with pytest.raises(TypeError):
        observation.fact["repair_quote_gbp"] = 180  # type: ignore[index]
    reveal = next(o for o in variant.observations if o.outcome == OUTCOME_REVEAL)
    with pytest.raises(TypeError):
        reveal.fact["repair_quote_gbp"] = 999  # type: ignore[index]


# -- the projection is explicit in what this build reports ------------------


def test_the_compiled_report_states_the_registry_and_its_origins():
    cube = compiled(authored_card())
    rendered = cube.as_dict()
    assert rendered["query_registry"] == cube.query_registry.as_dict()
    assert {entry["origin"] for entry in rendered["query_registry"]["entries"]} == {
        ORIGIN_AUTHORED,
        ORIGIN_PROJECTED_FACT,
    }
    for variant in rendered["variants"]:
        assert variant["query_registry"] == rendered["query_registry"]


def test_the_shipped_cubes_report_their_projection_explicitly():
    from boundarybench.compiler import compile_cube
    from boundarybench.loader import load_card
    from tests.conftest import ACCESS_CONSENT_CARD, EXAMPLE_CARD

    maintenance = compile_cube(load_card(EXAMPLE_CARD))
    assert maintenance.as_dict()["query_registry"] == {
        "contract": QUERY_RESOLUTION_CONTRACT,
        "entries": [
            {
                "query_key": "repair_quote_gbp",
                "action": "ask_user",
                "outcome": OUTCOME_REVEAL,
                "fact_key": "repair_quote_gbp",
                "origin": ORIGIN_PROJECTED_FACT,
            }
        ],
    }
    access = compile_cube(load_card(ACCESS_CONSENT_CARD))
    assert access.as_dict()["query_registry"] == {
        "contract": QUERY_RESOLUTION_CONTRACT,
        "entries": [],
    }


def test_the_package_exports_the_resolution_contract():
    import boundarybench

    for name in (
        "QUERY_OUTCOMES",
        "QUERY_RESOLUTION_CONTRACT",
        "QueryEntry",
        "QueryRegistry",
        "QueryResolutionError",
    ):
        assert name in boundarybench.__all__, name
        assert getattr(boundarybench, name) is not None
