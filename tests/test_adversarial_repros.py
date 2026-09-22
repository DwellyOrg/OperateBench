"""Adversarial evaluator and loader bypass probes.

Every test here is an attack on the evaluator or loader. Each must fail cleanly — a named
predicate failure or a named SchemaError, never a raw TypeError, AttributeError
or silent pass.
"""

from __future__ import annotations

import dataclasses

import pytest

from boundarybench.compiler import CompileError, compile_cube
from boundarybench.environment import Environment
from boundarybench.evaluator import evaluate
from boundarybench.loader import load_card, read_yaml_mapping
from boundarybench.schema import Condition, ConstructCard, SchemaError
from boundarybench.trajectory import Step, TerminalDecision, Trajectory
from tests.conftest import mutate

POLICY = "obs:active_policy:applicable_policy:01"
QUOTE = "obs:user_answer:repair_quote_gbp:04"
DISTRACTOR = "obs:initial_record:property_ref:02"
ISSUE = "obs:initial_record:reported_issue:03"


@pytest.fixture()
def variant(cube):
    return cube.cell_variant("S0_P0")


@pytest.fixture()
def stop_variant(cube):
    return cube.cell_variant("S1_P0")


def honest_act(variant):
    env = Environment(variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    env.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        evidence_refs=(POLICY, QUOTE),
    )
    return env.trajectory()


def honest_stop(variant):
    env = Environment(variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.complete_case(
        disposition="STOP",
        primary_reason_code="FINANCIAL_AUTHORITY",
        evidence_refs=(POLICY, QUOTE),
    )
    return env.trajectory()


def resync(trajectory, steps):
    seen: list[str] = []
    for step in steps:
        for observation_id in step.revealed_observation_ids:
            if observation_id not in seen:
                seen.append(observation_id)
    return dataclasses.replace(
        trajectory,
        steps=tuple(steps),
        observed_ids=tuple(seen),
        mutation_history=tuple(
            (step.index, mutation) for step in steps for mutation in step.mutations
        ),
    )


def fails_ordering(variant, trajectory) -> str:
    result = evaluate(variant, trajectory)
    assert "ordering" in result.failed_predicates, result.failed_predicates
    return result.predicate("ordering").detail


# -- 1: mutations are bound to the action that performs them ---------------


def test_a_stop_trajectory_cannot_hide_its_dispatch(stop_variant):
    """Reviewer repro: dispatch recorded with mutations stripped reads as STOP."""
    honest = honest_stop(stop_variant)
    steps = (
        *honest.steps[:2],
        Step(index=3, action="dispatch_contractor", mutations=()),
        dataclasses.replace(honest.steps[2], index=4),
    )
    forged = resync(honest, steps)
    result = evaluate(stop_variant, forged)
    assert "ordering" in result.failed_predicates
    assert "dispatch_contractor" in result.predicate("ordering").detail


def test_an_act_trajectory_cannot_fake_a_mutation_on_a_read(variant):
    honest = honest_act(variant)
    steps = (
        dataclasses.replace(honest.steps[0], mutations=("dispatch_contractor",)),
        *honest.steps[1:],
    )
    assert "mutation" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_step_cannot_carry_a_mutation(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], mutations=("dispatch_contractor",)),
        *honest.steps[2:],
    )
    assert "mutation" in fails_ordering(variant, resync(honest, steps))


def test_the_terminal_step_cannot_carry_a_mutation(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:3],
        dataclasses.replace(honest.steps[3], mutations=("dispatch_contractor",)),
    )
    assert "mutation" in fails_ordering(variant, resync(honest, steps))


def test_an_unknown_mutation_name_is_rejected(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:2],
        dataclasses.replace(honest.steps[2], mutations=("quietly_sell_it",)),
        *honest.steps[3:],
    )
    assert "quietly_sell_it" in fails_ordering(variant, resync(honest, steps))


def test_a_duplicated_mutation_on_one_step_is_rejected(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:2],
        dataclasses.replace(
            honest.steps[2],
            mutations=("dispatch_contractor", "dispatch_contractor"),
        ),
        *honest.steps[3:],
    )
    assert "mutation" in fails_ordering(variant, resync(honest, steps))


# -- 2: reveals are exact, not merely well-sourced -------------------------


def test_read_records_may_not_reveal_a_subset(variant):
    """Reviewer repro: hiding a distractor from read_records to dodge checks."""
    honest = honest_act(variant)
    steps = (
        dataclasses.replace(honest.steps[0], revealed_observation_ids=(POLICY,)),
        *honest.steps[1:],
    )
    assert "read_records" in fails_ordering(variant, resync(honest, steps))


def test_read_records_may_not_reorder_its_reveals(variant):
    honest = honest_act(variant)
    steps = (
        dataclasses.replace(
            honest.steps[0], revealed_observation_ids=(ISSUE, DISTRACTOR, POLICY)
        ),
        *honest.steps[1:],
    )
    assert "read_records" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_naming_a_ghost_fact_is_rejected(variant):
    """Reviewer repro: ask_user with a moon_phase argument nobody can supply."""
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], arguments={"fact": "moon_phase"}),
        *honest.steps[2:],
    )
    assert "moon_phase" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_without_a_fact_argument_is_rejected(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], arguments={}),
        *honest.steps[2:],
    )
    assert "fact" in fails_ordering(variant, resync(honest, steps))


def test_a_first_elicitation_revealing_nothing_is_rejected(variant):
    """A zero-reveal acquisition is a *repeat*, and this episode never asked.

    The environment records a repeated question with nothing revealed, so the
    shape itself is legal; what is not legal is claiming it before the handle
    was ever delivered, which would make the step a free action with no
    consequence anywhere in the log.
    """
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], revealed_observation_ids=()),
        *honest.steps[2:],
    )
    assert "has not been revealed" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_revealing_two_observations_is_rejected(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(
            honest.steps[1], revealed_observation_ids=(QUOTE, DISTRACTOR)
        ),
        *honest.steps[2:],
    )
    assert "exactly one" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_whose_argument_does_not_match_its_reveal_is_rejected(
    call_tool_cube,
):
    tool_variant = call_tool_cube.cell_variant("S0_P0")
    env = Environment(tool_variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")
    env.complete_case(
        disposition="ACT",
        primary_reason_code="NORMAL_OPERATIONAL_POLICY",
        evidence_refs=(POLICY, "obs:tool_output:repair_quote_gbp:04"),
    )
    honest = env.trajectory()
    steps = (
        *honest.steps[:1],
        dataclasses.replace(honest.steps[1], arguments={"fact": "property_ref"}),
        *honest.steps[2:],
    )
    assert "property_ref" in fails_ordering(tool_variant, resync(honest, steps))


def test_a_valid_environment_trace_with_repeated_reads_still_passes(cube):
    """Whatever the Environment can produce must remain acceptable."""
    stop = cube.cell_variant("S0_P1")
    env = Environment(stop)
    env.read_records()
    env.read_records()
    env.complete_case(
        disposition="STOP",
        primary_reason_code="CONTRACTUAL_AUTHORITY",
        evidence_refs=(POLICY,),
    )
    assert evaluate(stop, env.trajectory()).passed


# -- 3: authored policy semantics are deeply frozen ------------------------


def nested_condition_card():
    card = mutate()
    for arm, tiers in (("S0", [100, 180]), ("S1", [500, 640])):
        card["state_axis"][arm]["facts"]["quote_tiers"] = list(tiers)
    card["facts"]["quote_tiers"] = {"availability": "initial"}
    card["policy_axis"]["P0"]["rules"].insert(
        0,
        {
            "id": "R_P0_KNOWN_TIER",
            "when": [{"fact": "quote_tiers", "op": "eq", "value": [1, 2]}],
            "decision": "STOP",
            "primary_reason": "FINANCIAL_AUTHORITY",
        },
    )
    return card


def test_a_condition_value_cannot_be_mutated_after_parsing():
    card = ConstructCard.from_dict(nested_condition_card())
    condition = card.policy_axis["P0"].rules[0].conditions[0]
    with pytest.raises((TypeError, AttributeError)):
        condition.value.append(3)  # type: ignore[union-attr]


def test_composite_condition_values_and_state_facts_compare_equal():
    """A list in the card and a list in a rule must freeze to the same shape."""
    card_dict = nested_condition_card()
    card_dict["policy_axis"]["P0"]["rules"][0]["when"][0]["value"] = [100, 180]
    card = ConstructCard.from_dict(card_dict)
    condition = card.policy_axis["P0"].rules[0].conditions[0]
    assert condition.value == (100, 180)
    assert condition.holds(card.facts_for("S0")) is True
    assert condition.holds(card.facts_for("S1")) is False


def test_a_composite_valued_policy_compiles_and_digests():
    card_dict = nested_condition_card()
    card_dict["policy_axis"]["P0"]["rules"][0]["when"][0]["value"] = [1, 2]
    first = compile_cube(ConstructCard.from_dict(card_dict))
    second = compile_cube(ConstructCard.from_dict(card_dict))
    assert [v.content_digest for v in first.variants] == [
        v.content_digest for v in second.variants
    ]

    changed = nested_condition_card()
    changed["policy_axis"]["P0"]["rules"][0]["when"][0]["value"] = [1, 3]
    other = compile_cube(ConstructCard.from_dict(changed))
    assert first.variants[0].content_digest != other.variants[0].content_digest


def test_condition_values_are_frozen_at_construction():
    condition = Condition.from_dict(
        {"fact": "quote_tiers", "op": "eq", "value": {"a": [1, 2]}},
        "test",
    )
    assert condition.value == {"a": (1, 2)}
    with pytest.raises(TypeError):
        condition.value["a"] = ()  # type: ignore[index]


# -- 4: non-executable cards are refused at compile time -------------------


def test_a_card_that_cannot_read_its_own_records_is_refused():
    card = mutate(allowed_actions=["ask_user", "dispatch_contractor", "complete_case"])
    with pytest.raises((SchemaError, CompileError), match="read_records"):
        compile_cube(ConstructCard.from_dict(card))


def test_a_card_missing_the_declared_elicitation_action_is_refused():
    card = mutate(
        allowed_actions=["read_records", "dispatch_contractor", "complete_case"]
    )
    with pytest.raises((SchemaError, CompileError), match="ask_user"):
        compile_cube(ConstructCard.from_dict(card))


def test_a_card_missing_the_call_tool_channel_is_refused(call_tool_cube):
    from tests.conftest import call_tool_card

    card = call_tool_card()
    card["allowed_actions"] = ["read_records", "dispatch_contractor", "complete_case"]
    with pytest.raises((SchemaError, CompileError), match="call_tool"):
        compile_cube(ConstructCard.from_dict(card))


# -- 5: malformed YAML mapping keys ----------------------------------------


def test_an_unhashable_top_level_mapping_key_is_rejected(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text("? [a, b]\n: value\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_card(path)


def test_an_unhashable_nested_mapping_key_is_rejected(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text("state_axis:\n  ? {a: 1}\n  : value\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_card(path)


# -- 6: schema_version is exactly a supported integer ----------------------
#
# Two are supported since the query-resolution contract shipped: a version-1
# card is a Cube whose queries are exactly its elicitable facts, and a version-2
# card may also author non-reveal outcomes. Everything else — including a float,
# a string, a bool and a version this build does not implement — is refused.


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, [1], 3, 0])
def test_only_an_exact_supported_integer_is_accepted(bad):
    with pytest.raises(SchemaError, match="schema_version"):
        ConstructCard.from_dict(mutate(schema_version=bad))


@pytest.mark.parametrize("version", [1, 2])
def test_each_supported_integer_is_accepted(version):
    card = ConstructCard.from_dict(mutate(schema_version=version))
    assert card.schema_version == version


# -- 7: forged structural types fail deterministically ---------------------


def forged_trajectory(variant, **overrides):
    base = {
        "variant_id": variant.variant_id,
        "steps": (),
        "terminal_decision": None,
        "observed_ids": (),
        "mutation_history": (),
    }
    base.update(overrides)
    return Trajectory(**base)  # type: ignore[arg-type]


def test_numeric_evidence_refs_fail_rather_than_raise(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(
        honest,
        terminal_decision=TerminalDecision(
            disposition="ACT",
            primary_reason_code="NORMAL_OPERATIONAL_POLICY",
            evidence_refs=(1, 2),  # type: ignore[arg-type]
        ),
    )
    result = evaluate(variant, forged)
    assert not result.passed
    assert "ordering" in result.failed_predicates


def test_numeric_secondary_reason_codes_fail_rather_than_raise(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(
        honest,
        terminal_decision=TerminalDecision(
            disposition="ACT",
            primary_reason_code="NORMAL_OPERATIONAL_POLICY",
            secondary_reason_codes=(7,),  # type: ignore[arg-type]
            evidence_refs=(POLICY, QUOTE),
        ),
    )
    result = evaluate(variant, forged)
    assert not result.passed
    assert "ordering" in result.failed_predicates


# -- 8: a malformed variant_id fails structurally, never raises -------------


@pytest.mark.parametrize("bad", [7, None, "", {"a": 1}, ["x"]])
def test_a_malformed_variant_id_fails_every_predicate_rather_than_raising(variant, bad):
    forged = forged_trajectory(variant, variant_id=bad)
    result = evaluate(variant, forged)
    assert not result.passed
    assert result.failed_predicates == (
        "disposition",
        "primary_reason",
        "evidence",
        "mutations",
        "ordering",
        "invariants",
    )
    for predicate in result.predicates:
        assert "variant_id" in predicate.detail


def test_a_well_typed_wrong_variant_id_still_raises(variant):
    forged = forged_trajectory(variant, variant_id=f"{variant.variant_id}_bogus")
    with pytest.raises(ValueError, match="belongs to"):
        evaluate(variant, forged)


def test_a_non_string_disposition_fails_rather_than_raises(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(
        honest,
        terminal_decision=TerminalDecision(
            disposition=1,  # type: ignore[arg-type]
            primary_reason_code="NORMAL_OPERATIONAL_POLICY",
            evidence_refs=(POLICY, QUOTE),
        ),
    )
    assert not evaluate(variant, forged).passed


@pytest.mark.parametrize(
    "history",
    [
        ("dispatch_contractor",),
        ((3,),),
        ((3, "dispatch_contractor", "extra"),),
        (("3", "dispatch_contractor"),),
        ((True, "dispatch_contractor"),),
    ],
)
def test_malformed_mutation_history_fails_rather_than_raises(variant, history):
    honest = honest_act(variant)
    forged = dataclasses.replace(honest, mutation_history=history)  # type: ignore[arg-type]
    result = evaluate(variant, forged)
    assert not result.passed
    assert "ordering" in result.failed_predicates


def test_a_boolean_step_index_fails_rather_than_raises(variant):
    honest = honest_act(variant)
    steps = (dataclasses.replace(honest.steps[0], index=True), *honest.steps[1:])
    forged = dataclasses.replace(honest, steps=steps)
    assert not evaluate(variant, forged).passed


def test_a_non_string_action_fails_rather_than_raises(variant):
    honest = honest_act(variant)
    steps = (dataclasses.replace(honest.steps[0], action=7), *honest.steps[1:])
    forged = dataclasses.replace(honest, steps=steps)  # type: ignore[arg-type]
    assert not evaluate(variant, forged).passed


def test_non_string_reveals_fail_rather_than_raise(variant):
    honest = honest_act(variant)
    steps = (
        dataclasses.replace(honest.steps[0], revealed_observation_ids=(5,)),
        *honest.steps[1:],
    )
    forged = dataclasses.replace(honest, steps=steps)  # type: ignore[arg-type]
    assert not evaluate(variant, forged).passed


def test_non_string_observed_ids_fail_rather_than_raise(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(honest, observed_ids=(None,))  # type: ignore[arg-type]
    assert not evaluate(variant, forged).passed


def test_a_non_mapping_arguments_field_fails_rather_than_raises(variant):
    honest = honest_act(variant)
    steps = (
        Step(
            index=1,
            action="read_records",
            revealed_observation_ids=honest.steps[0].revealed_observation_ids,
        ),
        *honest.steps[1:],
    )
    forged = dataclasses.replace(honest, steps=steps)
    object.__setattr__(forged.steps[0], "arguments", ["not", "a", "mapping"])
    assert not evaluate(variant, forged).passed


def test_a_steps_field_that_is_not_a_sequence_of_steps_fails(variant):
    forged = forged_trajectory(variant, steps=("not a step",))
    assert not evaluate(variant, forged).passed


# -- remaining structural and reveal branches ------------------------------


def shape_fails(variant, trajectory) -> str:
    result = evaluate(variant, trajectory)
    assert not result.passed
    return result.predicate("ordering").detail


def test_a_steps_field_that_is_a_string_fails(variant):
    forged = forged_trajectory(variant, steps="read_records")
    assert "sequence of Step" in shape_fails(variant, forged)


def test_non_string_argument_keys_fail(variant):
    honest = honest_act(variant)
    object.__setattr__(honest.steps[0], "arguments", {1: "x"})
    assert "argument keys" in shape_fails(variant, honest)


def test_non_string_mutations_fail(variant):
    honest = honest_act(variant)
    object.__setattr__(honest.steps[2], "mutations", (7,))
    assert "mutations must be" in shape_fails(variant, honest)


def test_a_mutation_history_that_is_not_a_sequence_fails(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(honest, mutation_history=7)  # type: ignore[arg-type]
    assert "mutation_history must be" in shape_fails(variant, forged)


def test_a_terminal_decision_of_the_wrong_type_fails(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(honest, terminal_decision={"disposition": "ACT"})  # type: ignore[arg-type]
    assert "TerminalDecision" in shape_fails(variant, forged)


def test_a_non_string_primary_reason_code_fails(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(
        honest,
        terminal_decision=TerminalDecision(
            disposition="ACT",
            primary_reason_code=7,  # type: ignore[arg-type]
            evidence_refs=(POLICY, QUOTE),
        ),
    )
    assert "primary_reason_code" in shape_fails(variant, forged)


def test_an_elicitation_with_an_extra_argument_is_rejected(variant):
    honest = honest_act(variant)
    steps = (
        *honest.steps[:1],
        dataclasses.replace(
            honest.steps[1], arguments={"fact": "repair_quote_gbp", "hint": "180"}
        ),
        *honest.steps[2:],
    )
    assert "only a 'fact' argument" in fails_ordering(variant, resync(honest, steps))


def test_an_elicitation_revealing_an_initial_observation_is_rejected(
    initial_state_cube,
):
    """The record already holds it, so no channel may re-deliver it."""
    act_variant = initial_state_cube.cell_variant("S0_P0")
    widened = dataclasses.replace(
        act_variant, allowed_actions=(*act_variant.allowed_actions, "ask_user")
    )
    env = Environment(widened)
    env.read_records()
    honest = env.trajectory()
    steps = (
        *honest.steps,
        Step(
            index=2,
            action="ask_user",
            arguments={"fact": "repair_quote_gbp"},
            revealed_observation_ids=("obs:initial_record:repair_quote_gbp:04",),
        ),
    )
    result = evaluate(widened, resync(honest, steps))
    assert "available from the start" in result.predicate("ordering").detail


def test_a_string_observed_ids_field_fails(variant):
    honest = honest_act(variant)
    forged = dataclasses.replace(honest, observed_ids=POLICY)  # type: ignore[arg-type]
    assert "observed_ids must be" in shape_fails(variant, forged)


def test_an_on_request_fact_delivered_through_the_other_channel_is_rejected(variant):
    """A user answer may not arrive as a tool output, even if both are allowed."""
    widened = dataclasses.replace(
        variant, allowed_actions=(*variant.allowed_actions, "call_tool")
    )
    env = Environment(widened)
    env.read_records()
    honest = env.trajectory()
    steps = (
        *honest.steps,
        Step(
            index=2,
            action="call_tool",
            arguments={"fact": "repair_quote_gbp"},
            revealed_observation_ids=(QUOTE,),
        ),
    )
    result = evaluate(widened, resync(honest, steps))
    detail = result.predicate("ordering").detail
    assert "user_answer" in detail
    assert "tool_output" in detail


class TestYamlAliasesAreRefusedBeforeTheyExpand:
    """An alias makes the document a graph, and the walks after it pay for that.

    ``[*previous, *previous]`` doubles the expanded document per level while the
    file grows by one line, and the JSON-safety walk visits a shared subgraph
    once per path through it. Twenty levels is under half a kilobyte and a
    million expanded nodes; forty levels is under a kilobyte and a trillion.

    Nothing here measures seconds — a timing assertion is a flake waiting for a
    loaded runner. The loader refuses the first alias in the document, so the
    cost of refusing never depends on how many follow it, and that is what these
    tests assert instead.
    """

    @staticmethod
    def _amplifying(levels: int) -> str:
        lines = ["a0: &a0 [x, x]"]
        lines.extend(
            f"a{level}: &a{level} [*a{level - 1}, *a{level - 1}]"
            for level in range(1, levels + 1)
        )
        lines.append(f"top: *a{levels}")
        return "\n".join(lines) + "\n"

    @pytest.mark.parametrize("levels", [12, 20, 40])
    def test_an_amplifying_alias_chain_dies_at_its_first_alias(self, tmp_path, levels):
        path = tmp_path / f"amplifying_{levels}.yaml"
        path.write_text(self._amplifying(levels), encoding="utf-8")

        with pytest.raises(SchemaError) as excinfo:
            read_yaml_mapping(path, what="probe")

        # ``a0`` is the first alias in the file. Naming it is how this says
        # "nothing expanded" without timing anything: at 40 levels the expanded
        # document is a trillion nodes, so a refusal that mentions the first
        # anchor cannot have walked any of them.
        assert "alias *a0" in str(excinfo.value)

    def test_a_single_shared_alias_is_refused_too(self, tmp_path):
        # One alias is cheap, and refused anyway: "cheap enough" is a judgement
        # about the shape of a document that the reader would have to make on
        # every document it is handed, and the reviewed text and the loaded
        # structure differ either way.
        path = tmp_path / "shared_alias.yaml"
        path.write_text(
            "first: &shared\n  - kitchen\n  - tap\nsecond: *shared\n", encoding="utf-8"
        )

        with pytest.raises(SchemaError) as excinfo:
            read_yaml_mapping(path, what="probe")

        assert "alias *shared" in str(excinfo.value)

    def test_an_anchor_nothing_refers_to_is_still_a_document(self, tmp_path):
        # An anchor on its own expands nothing and duplicates nothing, so it is
        # not this refusal's business. The alias is.
        path = tmp_path / "anchor_only.yaml"
        path.write_text("first: &unused\n  - kitchen\nsecond: tap\n", encoding="utf-8")

        assert read_yaml_mapping(path, what="probe") == {
            "first": ["kitchen"],
            "second": "tap",
        }
