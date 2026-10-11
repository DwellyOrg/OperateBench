"""Trusted maintainer Engine/replay integration, outside the owner import boundary."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from operatebench.agents.playback import (
    RecordedOutcomeAgent,
    RecordingAgent,
    tape_from_records,
)
from operatebench.contributions.prospire.bookkeeping_reconciliation.agents import (
    BookkeepingAgent,
)
from operatebench.contributions.prospire.bookkeeping_reconciliation.evaluator import (
    evaluate,
)
from operatebench.contributions.prospire.bookkeeping_reconciliation.operation import (
    BookkeepingDomain,
)
from operatebench.contributions.prospire.bookkeeping_reconciliation.spec import load_spec
from operatebench.core.engine import Engine, has_canonical_episode_outcome_provenance
from operatebench.sdk.builtins import BUILTIN_PACKS

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "examples/contributions/prospire/bookkeeping_reconciliation/operation.yaml"
)
CONTROLS = json.loads((FIXTURE.parent / "negative_controls.json").read_text())["controls"]


def run_case(scenario, agent_id="reference"):
    spec = load_spec(FIXTURE)
    identity = {
        "operation_id": spec.data["operation_id"],
        "agent_id": agent_id,
        "operation_instance_id": "synthetic-bookkeeping-test-instance",
    }
    recording = RecordingAgent(BookkeepingAgent(agent_id))
    episode = Engine(
        BookkeepingDomain(spec, scenario), recording, identity=identity
    ).run()
    assert has_canonical_episode_outcome_provenance(episode)
    grade = evaluate(episode, spec)
    # Serialize decisions then execute the live domain afresh. No original agent
    # decision function participates in replay; full episode and grade must match.
    tape = tape_from_records(json.loads(json.dumps(recording.tape().as_list())))
    playback = RecordedOutcomeAgent(tape, agent_id=agent_id)
    replayed = Engine(
        BookkeepingDomain(spec, scenario), playback, identity=identity
    ).run()
    playback.check_exhausted()
    assert replayed.as_dict() == episode.as_dict()
    assert evaluate(replayed, spec) == grade
    return episode, grade


@pytest.mark.parametrize(
    "scenario", ["match", "conflict", "missing", "failed_zero", "reopen"]
)
def test_reference_engine_and_exact_replay(scenario):
    episode, grade = run_case(scenario)
    assert grade["reliable"], (grade, episode.status)
    assert episode.terminal_outcome == (
        "UNRESOLVED_HANDOFF" if scenario == "missing" else "REVIEWED_RELEASE"
    )
    assert any(row["record_type"] == "event_after_terminal" for row in episode.trajectory)
    if scenario == "missing":
        assert not episode.final_state["reports"]
        assert episode.final_state["handed_off"]
    if scenario == "reopen":
        assert len(episode.final_state["reports"]) == 2
        assert episode.final_state["report"]["ledger_version"] == 2
        assert (
            len(
                [
                    r
                    for r in episode.trajectory
                    if r["record_type"] == "bookkeeping_event" and r["kind"] == "approval"
                ]
            )
            == 2
        )


@pytest.mark.parametrize("control", CONTROLS, ids=lambda c: c["agent"])
def test_negative_engine_exact_findings_and_replay(control):
    good, _ = run_case(control["scenario"])
    episode, grade = run_case(control["scenario"], control["agent"])
    assert not grade["reliable"]
    assert grade["findings"] == [
        {"dimension": control["failed"], "code": control["code"]}
    ]
    assert {k for k, v in grade["dimensions"].items() if not v} == {control["failed"]}
    assert (
        episode.final_state == good.final_state
    )  # recovery cannot erase attempted fault
    assert episode.replay_final
    print(json.dumps({"control": control["agent"], "grade": grade}, sort_keys=True))


def test_alternate_review_followup_is_valid():
    episode, grade = run_case("match", "alternate")
    assert grade["reliable"] and episode.replay_final


def test_candidate_is_not_dispatchable():
    assert all(
        p.metadata.pack_id != "accounting.bookkeeping_reconciliation.synthetic.v1"
        for p in BUILTIN_PACKS
    )


def test_stale_source_and_wrong_actor_refusals_preserve_state():
    from types import SimpleNamespace

    spec = load_spec(FIXTURE)
    domain = BookkeepingDomain(spec, "match")
    state = domain.initial_state()
    state["ledger"] = domain.record(2)
    before = deepcopy(state)
    for actor, payload, code in [
        ("connector", domain.record(3), "AUTHORITY"),
        ("ledger", domain.record(1), "SOURCE_VERSION"),
    ]:
        event = SimpleNamespace(
            actor_id=actor, event_type="ledger_arrived", payload=payload
        )
        verdict = domain.reduce_event(state, event, None)
        assert verdict.code == code and not verdict.accepted
        assert state == before


def test_tampered_observation_tape_is_refused():
    from operatebench.agents.playback import PlaybackError

    spec = load_spec(FIXTURE)
    recording = RecordingAgent(BookkeepingAgent())
    identity = {
        "operation_id": spec.data["operation_id"],
        "agent_id": "reference",
        "operation_instance_id": "synthetic-tamper-test-instance",
    }
    Engine(BookkeepingDomain(spec, "match"), recording, identity=identity).run()
    records = deepcopy(recording.tape().as_list())
    records[0]["observation_digest_sha256"] = "0" * 64
    playback = RecordedOutcomeAgent(tape_from_records(records), agent_id="reference")
    with pytest.raises(PlaybackError):
        Engine(BookkeepingDomain(spec, "match"), playback, identity=identity).run()
