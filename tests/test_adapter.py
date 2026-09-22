"""The provider-neutral adapter boundary."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from boundarybench.adapter import (
    AdapterCall,
    ScriptedTestAdapter,
    TurnDeadline,
    build_turn_request,
    identity_for_test_double,
    policy_following_script,
)
from boundarybench.compiler import Cube
from boundarybench.environment import Environment
from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold


def _scaffold():
    return load_scaffold(STANDARD_SCAFFOLD)


def test_turn_request_carries_the_scaffold_and_only_observed_facts(cube: Cube) -> None:
    variant = cube.cell_variant("S1_P0")
    scaffold = _scaffold()
    env = Environment(variant)

    before = build_turn_request(scaffold=scaffold, environment=env, turns_remaining=4)
    assert before.system_prompt == scaffold.system_prompt
    assert before.scaffold_id == scaffold.scaffold_id
    assert before.scaffold_version == scaffold.scaffold_version
    assert before.turns_remaining == 4
    assert before.available_actions == variant.allowed_actions
    # Nothing has been read yet, so the agent has seen nothing.
    assert before.observations == ()
    assert before.transcript == ()

    env.read_records()
    after = build_turn_request(scaffold=scaffold, environment=env, turns_remaining=3)
    assert [entry.observation_id for entry in after.observations] == [
        obs.observation_id for obs in variant.initial_observations()
    ]
    assert [entry.action for entry in after.transcript] == ["read_records"]
    assert list(after.transcript[0].revealed_observation_ids) == [
        obs.observation_id for obs in variant.initial_observations()
    ]

    # The transcript is the agent's only memory of what it has already done, so
    # every field of every entry is projected exactly: the step's own index, the
    # arguments it was called with, and the handles it was given for them. An
    # entry that lost its index would leave the agent unable to tell the order
    # of what it did, and a step's arguments are how it knows which fact it
    # already asked for.
    quote = env.obtain("repair_quote_gbp")
    later = build_turn_request(scaffold=scaffold, environment=env, turns_remaining=2)
    assert [entry.as_dict() for entry in later.transcript] == [
        {
            "index": 1,
            "action": "read_records",
            "arguments": {},
            "revealed_observation_ids": [
                obs.observation_id for obs in variant.initial_observations()
            ],
        },
        {
            "index": 2,
            "action": "ask_user",
            "arguments": {"fact": "repair_quote_gbp"},
            "revealed_observation_ids": [quote.observation_id],
        },
    ]
    assert [entry.index for entry in later.transcript] == [
        step.index for step in env.steps
    ]


def test_turn_request_never_carries_the_grading_contract(cube: Cube) -> None:
    """The adapter sees the case, never the answer key or the variant identity.

    Some ground-truth *values* are legitimately observable — ``ACT`` is in the
    terminal schema, the matched rule id is one of the rule ids the active policy
    presents, and every required evidence handle is a handle the agent was shown.
    What must never cross is the *structure* that says which of them is the
    answer, and the variant identity that would let a run be recognised. So the
    shape is pinned exactly, and only the genuinely private strings are searched
    for.
    """
    variant = cube.cell_variant("S0_P0")
    env = Environment(variant)
    env.read_records()
    env.obtain("repair_quote_gbp")
    env.act("dispatch_contractor")

    request = build_turn_request(scaffold=_scaffold(), environment=env, turns_remaining=1)
    payload = request.as_dict()
    rendered = json.dumps(payload, sort_keys=True)

    for secret in (
        variant.variant_id,
        variant.cube_id,
        variant.base_cell,
        variant.content_digest,
    ):
        assert secret not in rendered, f"turn request leaks {secret!r}"

    assert set(payload) == {
        "system_prompt",
        "scaffold_id",
        "scaffold_version",
        "actions",
        "available_actions",
        "observations",
        "transcript",
        "turns_remaining",
    }
    # Evidence relevance is the evaluator's business: an observation the agent
    # can see must not be tagged with whether citing it would score.
    for entry in payload["observations"]:
        assert set(entry) == {"observation_id", "source", "key", "fact", "outcome"}
    for entry in payload["transcript"]:
        assert set(entry) == {
            "index",
            "action",
            "arguments",
            "revealed_observation_ids",
        }
    assert {field.name for field in fields(request)} == set(payload)


def test_scripted_fake_is_deterministic_and_reports_no_usage(cube: Cube) -> None:
    """Same observable request, same call — and never a fabricated measurement."""
    env = Environment(cube.cell_variant("S1_P0"))
    env.read_records()
    request = build_turn_request(scaffold=_scaffold(), environment=env, turns_remaining=5)
    adapter = ScriptedTestAdapter(
        script=policy_following_script, identity=identity_for_test_double(), settings={}
    )

    deadline = TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False)
    first = adapter.next_call(request, deadline)
    second = adapter.next_call(request, deadline)

    assert first == second
    assert first.action == "ask_user"
    assert first.arguments == {"fact": "repair_quote_gbp"}
    assert adapter.identity.provider == "test-double"
    assert adapter.last_usage().as_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_seconds": None,
    }


# -- the shipped fake's own correctness --------------------------------------
#
# Not a trust boundary: the fake is a fixture, and nothing it produces is a
# measurement. It is pinned because CI's reference execution is only a fixed
# point if the fake decides the same way every time, and because a fake that
# quietly stopped applying the policy would still complete every episode.


@pytest.mark.parametrize(
    ("op", "actual", "matches"),
    [
        ("gt", 249, False),
        ("gt", 250, False),
        ("gt", 251, True),
        ("gte", 249, False),
        ("gte", 250, True),
        ("gte", 251, True),
        ("lt", 249, True),
        ("lt", 250, False),
        ("lt", 251, False),
        ("lte", 249, True),
        ("lte", 250, True),
        ("lte", 251, False),
    ],
)
def test_the_fake_compares_a_threshold_at_its_boundary(
    op: str, actual: int, matches: bool
) -> None:
    """A standing limit is a threshold, and the cases turn on its exact value.

    Every shipped Cube separates its two state arms across a numeric limit, so
    an off-by-one comparison changes which rule the fake believes applies — and
    the episode still completes, with a different decision, for a reason nothing
    else observes.
    """
    from boundarybench.adapter import _matches

    condition = {"fact": "repair_quote_gbp", "op": op, "value": 250}

    assert _matches(condition, {"repair_quote_gbp": actual}) is matches


@pytest.mark.parametrize(
    ("condition", "facts", "matches"),
    [
        # A fact the agent has not obtained decides nothing.
        ({"fact": "quote", "op": "eq", "value": 1}, {}, False),
        ({"fact": "quote", "op": "gt", "value": 0}, {}, False),
        # bool is an int subclass: a flag is not a quantity, and `True == 1`.
        ({"fact": "quote", "op": "gt", "value": 0}, {"quote": True}, False),
        ({"fact": "quote", "op": "eq", "value": 1}, {"quote": True}, False),
        ({"fact": "quote", "op": "eq", "value": True}, {"quote": True}, True),
        # Nor is text, which would otherwise raise inside a comparison.
        ({"fact": "quote", "op": "gt", "value": 0}, {"quote": "250"}, False),
    ],
)
def test_the_fake_only_compares_what_can_be_compared(
    condition: dict[str, object], facts: dict[str, object], matches: bool
) -> None:
    from boundarybench.adapter import _matches

    assert _matches(condition, facts) is matches


def test_the_fake_elicits_through_the_only_channel_a_case_offers(
    call_tool_cube: Cube,
) -> None:
    """The deciding fact arrives through whichever channel the card declares.

    Every other fixture offers ``ask_user``, so a fake that had it hard-wired
    would look correct everywhere except on a tool-lookup case — where it would
    either stall or elicit through a channel the environment rejects.
    """
    variant = call_tool_cube.cell_variant("S1_P0")
    env = Environment(variant)
    env.read_records()
    request = build_turn_request(scaffold=_scaffold(), environment=env, turns_remaining=5)

    assert "ask_user" not in variant.allowed_actions
    assert policy_following_script(request) == AdapterCall(
        "call_tool", {"fact": "repair_quote_gbp"}
    )


def test_the_fake_gathers_every_fact_its_rules_reach_for_and_cites_each_once(
    multi_fact_cube: Cube,
) -> None:
    """A case whose rules span two facts is where citation bookkeeping shows.

    The first rule tests access consent and misses, so the case is decided by
    the later quote rule — and justifying it rests on both facts. Each handle is
    cited exactly once: a repeated handle is not extra evidence, it is a
    malformed decision that the evaluator would then have to interpret.
    """
    variant = multi_fact_cube.cell_variant("S0_P0")
    env = Environment(variant)
    adapter = ScriptedTestAdapter(
        script=policy_following_script, identity=identity_for_test_double(), settings={}
    )
    deadline = TurnDeadline(remaining_seconds=30.0, cancelled=lambda: False)

    for _ in range(6):
        request = build_turn_request(
            scaffold=_scaffold(), environment=env, turns_remaining=6
        )
        call = adapter.next_call(request, deadline)
        if call.action == "complete_case":
            break
        if call.action == "read_records":
            env.read_records()
        elif call.action in ("ask_user", "call_tool"):
            env.obtain(str(call.arguments["fact"]))
        else:
            # A workflow action is offered as its own tool, so the call names it
            # and carries nothing else.
            env.act(call.action)

    assert [step.action for step in env.steps] == [
        "read_records",
        "ask_user",
        "ask_user",
        "dispatch_contractor",
    ]
    assert [
        step.arguments["fact"] for step in env.steps if step.action == "ask_user"
    ] == ["access_consent_confirmed", "repair_quote_gbp"]
    assert call.action == "complete_case"
    cited = call.arguments["evidence_refs"]
    assert len(cited) == len(set(cited))
    assert set(cited) == {
        observation.observation_id
        for observation in env.observed()
        if observation.evidence_relevant
    }
