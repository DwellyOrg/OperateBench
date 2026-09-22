"""The filtered action surface.

The defect this closes is a simulator artifact, not model behaviour. On
``lettings_maintenance_authority_v1::cell::S0_P0`` the construct card declares
``fact_location: elicited``, declares ``repair_quote_gbp`` to arrive through
``ask_user``, and lists ``allowed_actions`` that exclude ``call_tool``. The
runtime nevertheless put a ``call_tool`` tool in front of the model, because the
request's tool schemas were the *scaffold's* whole action list while
``available_actions`` was the *card's*. An agent that chose the tool it was
offered had the choice rejected as ``wrong_channel`` from card metadata it had
never been shown; an agent that guessed the other channel passed. Guessing a
hidden channel is not the operational contract this benchmark claims to measure,
so a row decided that way measures the harness rather than the agent.

What is under test here is that one set — the Cube's explicit
``allowed_actions`` — is the *only* action surface, with three projections
derived from it:

* the tool schemas the real SDK serialises onto the wire;
* ``observed_case.available_actions``, the list the model reads;
* the actions dispatch will accept for that variant.

Every request here is answered by ``httpx.MockTransport`` inside this process.
No socket is opened, no credential is read, no directory outside ``tmp_path`` is
written, and nothing here is evidence about any model: it is evidence about the
integration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from boundarybench.adapter import (
    AdapterProtocolError,
    TurnDeadline,
    build_turn_request,
)
from boundarybench.compiler import Cube, Variant, compile_cube
from boundarybench.environment import ACTION_BY_SOURCE, Environment
from boundarybench.ledger import KIND_WRONG_CHANNEL, OUTCOME_MODEL_PROTOCOL_FAILURE
from boundarybench.loader import load_card
from boundarybench.manifest import card_fingerprint
from boundarybench.providers.anthropic_messages import (
    ANTHROPIC_ADAPTER_VERSION,
    ANTHROPIC_IMPLEMENTATION,
    ANTHROPIC_PROVIDER,
    REQUEST_MAPPING_VERSION,
    SONNET_5_MODEL,
    AnthropicMessagesAdapter,
    AnthropicRetryPolicy,
    anthropic_settings,
    build_messages_request,
)
from boundarybench.runmanifest import (
    RunLimits,
    RunManifestMismatchError,
    build_run_manifest,
    open_run_session,
)
from boundarybench.runner import compiled_variants, run_episode
from boundarybench.scaffold import (
    ACTION_SURFACE_CONTRACT,
    STANDARD_SCAFFOLD,
    WORKFLOW_ACTION_TEMPLATE,
    Scaffold,
    ScaffoldProjectionError,
    load_scaffold,
    project_action_surface,
)
from boundarybench.schema import PROTOCOL_ACTIONS, ConstructCard
from boundarybench.suite import SuiteConstraintError, SuiteDigestError, validate_suite
from tests.anthropic_transport import RecordingTransport, message_body, tool_use_block
from tests.conftest import (
    ACCESS_CONSENT_CARD,
    EXAMPLE_CARD,
    SUITE_MANIFEST,
    call_tool_card,
)
from tests.scaffolds import scaffold_payload, write_scaffold

#: The model this file's requests are addressed to. A real identifier, because
#: the request profile is model-specific and the body must be the real one; no
#: request built here leaves the process.
MODEL = SONNET_5_MODEL

#: The cell this surface defect is reproducible on.
SUBJECT_CELL = "S0_P0"

#: The exact action surface the maintenance card declares. Written out rather
#: than read back from the card, so a card edit that widened the surface would
#: fail this file instead of being absorbed by it.
MAINTENANCE_ACTIONS = (
    "read_records",
    "ask_user",
    "dispatch_contractor",
    "complete_case",
)


def _scaffold() -> Scaffold:
    return load_scaffold(STANDARD_SCAFFOLD)


def _deadline(seconds: float = 30.0) -> TurnDeadline:
    return TurnDeadline(remaining_seconds=seconds, cancelled=lambda: False)


def _cube(path: Path) -> Cube:
    return compile_cube(load_card(path))


def _maintenance_variant() -> Variant:
    return _cube(EXAMPLE_CARD).cell_variant(SUBJECT_CELL)


def _limits() -> RunLimits:
    return RunLimits(max_turns=6, max_messages=16, episode_timeout_seconds=30.0)


def _adapter(transport: RecordingTransport) -> AnthropicMessagesAdapter:
    return AnthropicMessagesAdapter(model=MODEL, client=transport.client())


def _episode(variant: Variant, *steps: Any) -> tuple[Any, RecordingTransport]:
    """One episode through the production runner and the real SDK transport.

    Anything the script does not answer is closed with a bare STOP, so a test
    that cares about one turn's request is not also a test of what an
    unscripted transport does.
    """
    transport = RecordingTransport(list(steps), default=_stop_answer())
    result = run_episode(
        variant=variant,
        scaffold=_scaffold(),
        adapter=_adapter(transport),
        limits=_limits(),
    )
    return result, transport


def _read_records_answer() -> Any:
    return message_body([tool_use_block("read_records", {})], model=MODEL)


def _terminal_answer(
    disposition: str, reason: str, evidence: tuple[str, ...] = ()
) -> Any:
    return message_body(
        [
            tool_use_block(
                "complete_case",
                {
                    "disposition": disposition,
                    "primary_reason_code": reason,
                    "secondary_reason_codes": [],
                    "evidence_refs": list(evidence),
                },
            )
        ],
        model=MODEL,
    )


def _stop_answer() -> Any:
    return _terminal_answer("STOP", "NO_APPLICABLE_RULE")


def _call_tool_answer(fact: str = "repair_quote_gbp") -> Any:
    """An adversarial answer naming a tool outside the filtered surface."""
    return message_body([tool_use_block("call_tool", {"fact": fact})], model=MODEL)


def _wire_tool_names(body: Any) -> list[str]:
    return [tool["name"] for tool in body["tools"]]


def _observed_case(body: Any) -> dict[str, Any]:
    parsed = json.loads(body["messages"][0]["content"])
    assert isinstance(parsed, dict)
    return parsed


def _surface(variant: Variant) -> Any:
    return build_turn_request(
        scaffold=_scaffold(), environment=Environment(variant), turns_remaining=6
    )


# -- 1. the defect, reproduced through the production runner -----------------


def test_the_maintenance_cell_offers_exactly_the_card_s_allowed_actions() -> None:
    """The card's allowed actions are the exact wire surface.

    One turn of the episode, driven by :func:`run_episode` and
    serialised by the installed Anthropic SDK. The tool names the transport
    actually carried, and the ``available_actions`` the model actually read,
    are both the card's list — and neither carries ``call_tool``.
    """
    _result, transport = _episode(_maintenance_variant(), _read_records_answer())

    body = transport.bodies[0]

    assert _wire_tool_names(body) == list(MAINTENANCE_ACTIONS)
    assert _observed_case(body)["available_actions"] == list(MAINTENANCE_ACTIONS)
    assert "call_tool" not in _wire_tool_names(body)
    assert "call_tool" not in _observed_case(body)["available_actions"]


def test_a_call_tool_turn_cannot_be_made_through_the_declared_tools() -> None:
    """An unoffered tool call cannot produce a behavioural row.

    A provider that answers with the call the unfiltered surface invited is
    refused by the adapter's protocol validation, before the environment is
    reached. It is a protocol failure — the model named a tool it was not
    offered — and specifically *not* the ``wrong_channel`` dispatch route, which
    told a model it had chosen the wrong one of two channels it was never given
    a way to tell apart.
    """
    result, _transport = _episode(
        _maintenance_variant(), _read_records_answer(), _call_tool_answer()
    )

    assert result.outcome == OUTCOME_MODEL_PROTOCOL_FAILURE
    assert KIND_WRONG_CHANNEL not in json.dumps(dict(result.failure_event or {}))
    assert [step.action for step in result.trajectory.steps] == ["read_records"]


def test_the_episode_runs_to_a_graded_decision_through_the_offered_set() -> None:
    """The whole S0_P0 episode, using only tools the case actually declares.

    Every turn names a tool from the card's list — including the workflow action,
    which is now offered under its own name rather than as an argument to a
    generic one — and the episode reaches the terminal decision the evaluator
    passes. The corrected surface is not merely narrower; it is sufficient.
    """
    variant = _maintenance_variant()

    result, _transport = _episode(
        variant,
        _read_records_answer(),
        message_body(
            [tool_use_block("ask_user", {"fact": "repair_quote_gbp"})], model=MODEL
        ),
        message_body([tool_use_block("dispatch_contractor", {})], model=MODEL),
        _terminal_answer(
            "ACT", "NORMAL_OPERATIONAL_POLICY", variant.required_evidence_ids
        ),
    )

    assert result.succeeded
    assert [step.action for step in result.trajectory.steps] == [
        "read_records",
        "ask_user",
        "dispatch_contractor",
        "complete_case",
    ]
    assert result.evaluation is not None
    assert result.evaluation.passed


def test_every_offered_action_is_one_this_variant_s_dispatch_admits() -> None:
    """The offered set and the dispatchable set are the same set."""
    variant = _maintenance_variant()

    offered = {action.name for action in _surface(variant).actions}
    dispatchable = (PROTOCOL_ACTIONS & set(variant.allowed_actions)) | set(
        variant.irreversible_actions
    )

    assert offered == dispatchable


# -- 2. the tool-backed Cube, through the same real transport ----------------


def test_a_tool_backed_cube_offers_call_tool_and_not_the_other_channel() -> None:
    """The complement: when the card *does* declare the tool channel, it is there.

    Filtering is not "remove ``call_tool``". The surface is the card's list,
    whatever that list is, so a Cube whose deciding fact arrives through a tool
    lookup offers ``call_tool`` and withholds ``ask_user`` — the acquisition
    channel this card excludes.
    """
    variant = compile_cube(ConstructCard.from_dict(call_tool_card())).cell_variant(
        SUBJECT_CELL
    )

    _result, transport = _episode(variant, _read_records_answer())

    names = _wire_tool_names(transport.bodies[0])
    assert names == list(variant.allowed_actions)
    assert "call_tool" in names
    assert "ask_user" not in names
    assert _observed_case(transport.bodies[0])["available_actions"] == names


def test_the_access_consent_cube_offers_no_acquisition_channel_at_all() -> None:
    """Its deciding fact is in the record, and its card excludes both channels."""
    variant = _cube(ACCESS_CONSENT_CARD).cell_variant(SUBJECT_CELL)

    _result, transport = _episode(variant, _read_records_answer())

    names = _wire_tool_names(transport.bodies[0])
    assert names == list(variant.allowed_actions)
    assert not set(ACTION_BY_SOURCE.values()) & set(names)
    assert _observed_case(transport.bodies[0])["available_actions"] == names


def test_an_action_the_card_withholds_is_refused_before_dispatch() -> None:
    """Named at the adapter boundary, with the tool name never quoted back."""
    environment = Environment(_maintenance_variant())
    environment.read_records()
    request = build_turn_request(
        scaffold=_scaffold(), environment=environment, turns_remaining=5
    )
    transport = RecordingTransport([_call_tool_answer()])

    with pytest.raises(AdapterProtocolError) as raised:
        _adapter(transport).next_call(request, _deadline())

    assert "call_tool" not in str(raised.value)


# -- 3. every variant of the shipped suite -----------------------------------

SUITE_VARIANTS = tuple(compiled_variants(validate_suite(SUITE_MANIFEST)).values())


def test_the_shipped_suite_has_the_twelve_variants_this_file_asserts_over() -> None:
    assert len(SUITE_VARIANTS) == 12


@pytest.mark.parametrize("variant", SUITE_VARIANTS, ids=lambda v: v.variant_id)
def test_every_suite_variant_offers_exactly_its_own_allowed_actions(
    variant: Variant,
) -> None:
    """Request tools, ``available_actions`` and the card's list are one set."""
    request = _surface(variant)
    payload = build_messages_request(request, model=MODEL)

    allowed = list(variant.allowed_actions)
    assert [action.name for action in request.actions] == allowed
    assert list(request.available_actions) == allowed
    assert _wire_tool_names(payload) == allowed
    assert _observed_case(payload)["available_actions"] == allowed


def test_the_action_surface_is_the_same_across_every_cell_and_probe_of_a_cube() -> None:
    """Availability is Cube-level interface metadata, not a per-cell answer.

    Compared as whole schemas rather than as names: a description or a parameter
    that moved with the cell would carry the answer just as well as a name would.
    The invariance is non-trivial because both dispositions occur inside each
    Cube, which the second assertion states rather than assumes.
    """
    surfaces: dict[str, set[str]] = {}
    dispositions: dict[str, set[str]] = {}
    for variant in SUITE_VARIANTS:
        request = _surface(variant)
        rendered = json.dumps(
            {
                "actions": [action.as_dict() for action in request.actions],
                "available_actions": list(request.available_actions),
            },
            sort_keys=True,
        )
        surfaces.setdefault(variant.cube_id, set()).add(rendered)
        dispositions.setdefault(variant.cube_id, set()).add(variant.expected_disposition)

    assert len(surfaces) == 2
    for cube_id, rendered in surfaces.items():
        assert len(rendered) == 1, cube_id
        assert dispositions[cube_id] == {"ACT", "STOP"}, cube_id


def test_the_offered_surface_carries_no_deciding_fact_value() -> None:
    """The interface says what may be done, never what the answer is."""
    for variant in SUITE_VARIANTS:
        request = _surface(variant)
        rendered = json.dumps(
            [action.as_dict() for action in request.actions]
            + [list(request.available_actions)]
        )
        for observation in variant.observations:
            if not observation.evidence_relevant:
                continue
            for value in observation.fact.values():
                assert str(value) not in rendered, variant.variant_id


def test_a_suite_offering_an_action_its_environment_refuses_does_not_validate(
    suite_dir: Path,
) -> None:
    """The card-level half of the same rule, at the gate every run passes.

    Filtering the surface to ``allowed_actions`` is only an improvement while
    that list is executable. A card that allows an acquisition channel no fact
    is declared to arrive through would put a tool in front of the model whose
    every use the environment refuses — the same defect, reintroduced one
    layer up — so a suite containing one is not validated at all.

    Re-pinned as a reviewed revision would be, so the mutation reaches the rule
    under test rather than failing on the identity gate before it.
    """
    card_path = suite_dir / "maintenance_authority.yaml"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "allowed_actions:\n  - read_records\n",
            "allowed_actions:\n  - read_records\n  - call_tool\n",
        ),
        encoding="utf-8",
    )
    path = _repinned(suite_dir, card_path)

    with pytest.raises(SuiteConstraintError) as raised:
        validate_suite(path)

    assert "call_tool" in str(raised.value)


def _repinned(suite_dir: Path, card_path: Path) -> Path:
    """Re-pin a mutated card's semantic manifest and the suite digest over it."""
    cube = compile_cube(load_card(card_path))
    manifest_path = suite_dir / f"{card_path.stem}.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["card_fingerprint_sha256"] = card_fingerprint(cube.card)
    payload["variants"] = [
        {"variant_id": variant.variant_id, "content_digest": variant.content_digest}
        for variant in cube.variants
    ]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    suite_path = suite_dir / "methodology_spike_suite.yaml"
    # A well-formed pin that is not the real digest, so validation reports the
    # one it computes. It has to start with a letter or YAML reads it as a
    # number and the format gate fires before the digest gate.
    placeholder = "a" + "0" * 63
    unpinned = re.sub(
        r"suite_content_digest: [0-9a-f]{64}",
        f"suite_content_digest: {placeholder}",
        suite_path.read_text(encoding="utf-8"),
    )
    suite_path.write_text(unpinned, encoding="utf-8")
    try:
        validate_suite(suite_path)
    except SuiteDigestError as exc:
        suite_path.write_text(
            unpinned.replace(placeholder, exc.computed), encoding="utf-8"
        )
    return suite_path


def test_a_scaffold_with_no_workflow_template_cannot_project_a_workflow_action(
    tmp_path: Path,
) -> None:
    """The projection answers for itself rather than trusting its caller.

    The runner checks the scaffold contract before an episode starts, so this
    scaffold never reaches a live turn. :func:`project_action_surface` is public
    and is also what the ledger rebuilds requests through, so a scaffold that
    cannot express a case's surface is named here rather than silently dropping
    the action from the tool list.
    """
    payload = scaffold_payload()
    payload["actions"] = [
        action
        for action in payload["actions"]
        if action["name"] != WORKFLOW_ACTION_TEMPLATE
    ]
    scaffold = load_scaffold(write_scaffold(tmp_path / "scaffold.json", payload))

    with pytest.raises(ScaffoldProjectionError) as raised:
        project_action_surface(
            scaffold,
            MAINTENANCE_ACTIONS,
            fact_affordances={"ask_user": ("repair_quote_gbp",)},
        )

    assert "dispatch_contractor" in str(raised.value)


# -- 4. run identity states the contract -------------------------------------


def _settings(**overrides: Any) -> dict[str, Any]:
    settings = anthropic_settings(AnthropicRetryPolicy(), model=MODEL)
    settings.update(overrides)
    return settings


def _manifest(settings: Any, *, version: str = ANTHROPIC_ADAPTER_VERSION) -> Any:
    return build_run_manifest(
        suite=validate_suite(SUITE_MANIFEST),
        scaffold=_scaffold(),
        provider=ANTHROPIC_PROVIDER,
        model=MODEL,
        implementation=ANTHROPIC_IMPLEMENTATION,
        adapter_version=version,
        adapter_settings=settings,
        trials=1,
        limits=_limits(),
    )


def test_the_settings_state_which_action_surface_the_run_sent() -> None:
    """A run's durable claim about what it asked the provider names the filter."""
    settings = anthropic_settings(AnthropicRetryPolicy(), model=MODEL)

    assert settings["action_surface"] == ACTION_SURFACE_CONTRACT
    assert settings["request_mapping"] == REQUEST_MAPPING_VERSION


def test_this_configuration_differs_from_an_unfiltered_surface_one() -> None:
    """The filter is inside run identity, so it cannot be changed silently.

    Stated as a difference rather than against a list of known identities: this
    build ships no pre-authorised configuration identity, and a test that named
    one would be publishing a run rather than checking a rule.
    """
    unfiltered = _settings(request_mapping="turn_request_json_v2")
    del unfiltered["action_surface"]

    assert (
        _manifest(_settings()).configuration_id
        != _manifest(unfiltered, version="0.6.0").configuration_id
    )


def test_an_unfiltered_surface_configuration_cannot_be_resumed_under_this_build(
    tmp_path: Path,
) -> None:
    """Synthetic on purpose, and local on purpose.

    Both configurations are built here, inside ``tmp_path``, from this build's
    own settings. What is under test is the rule, and the rule is a property of
    the configuration rather than of any particular directory.
    """
    root = tmp_path / "synthetic-unfiltered-surface"
    stale = _settings(request_mapping="turn_request_json_v2")
    del stale["action_surface"]
    stale_manifest = _manifest(stale, version="0.6.0")
    with open_run_session(root, stale_manifest) as session:
        assert session.resumed is False

    with (
        pytest.raises(RunManifestMismatchError) as raised,
        open_run_session(root, _manifest(_settings())),
    ):
        pass

    assert stale_manifest.configuration_id in str(raised.value)
