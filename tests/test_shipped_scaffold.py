"""The scaffold artefact this repository actually ships."""

from __future__ import annotations

import json

from boundarybench.scaffold import STANDARD_SCAFFOLD, load_scaffold
from boundarybench.schema import PROTOCOL_ACTIONS
from tests.scaffolds import fixture_digest


def test_shipped_scaffold_loads_and_its_pin_still_holds() -> None:
    """Checked against the hand-written encoder, so the pin is not self-certifying."""
    scaffold = load_scaffold(STANDARD_SCAFFOLD)

    raw = json.loads(STANDARD_SCAFFOLD.read_text(encoding="utf-8"))
    assert scaffold.content_digest == fixture_digest(raw)
    assert scaffold.scaffold_id == "boundarybench_standard_scaffold"
    assert scaffold.scaffold_version.startswith("0.1.0-")


def test_shipped_scaffold_exposes_every_protocol_action_and_the_terminal_call() -> None:
    """The agent-facing contract must cover exactly the protocol the loop speaks."""
    scaffold = load_scaffold(STANDARD_SCAFFOLD)

    names = {action.name for action in scaffold.actions}
    # Every protocol action, plus one dispatcher for the workflow actions a case
    # offers. A card-specific tool list would make the scaffold per-card and so
    # uncomparable across Cubes; the case supplies the names, the scaffold
    # supplies the one schema for calling them.
    assert names == set(PROTOCOL_ACTIONS) | {"perform_workflow_action"}
    dispatcher = next(a for a in scaffold.actions if a.name == "perform_workflow_action")
    assert [parameter.name for parameter in dispatcher.parameters] == ["action"]

    terminal = scaffold.terminal_action
    assert terminal.name == "complete_case"
    assert {parameter.name for parameter in terminal.parameters} == {
        "disposition",
        "primary_reason_code",
        "secondary_reason_codes",
        "evidence_refs",
    }
    assert all(parameter.required for parameter in terminal.parameters)

    elicitation = {
        action.name: action
        for action in scaffold.actions
        if action.name in {"ask_user", "call_tool"}
    }
    for action in elicitation.values():
        assert [parameter.name for parameter in action.parameters] == ["fact"]
