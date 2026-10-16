"""Independent synthetic freight proposal tests."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from operatebench.agents.playback import (
    RecordedOutcomeAgent,
    RecordingAgent,
    tape_from_records,
)
from operatebench.core.engine import Engine

from .agents import VisibilityAgent
from .evaluator import evaluate
from .operation import ISSUERS, LOAD, VARIANTS, VisibilityDomain


def test_tracking_alone_cannot_notify_delivered():
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    state["tracking"] = "geofence"
    verdict = domain.apply_action(
        state, "notify", {"load": "synthetic-load-a", "version": 1}, (), SimpleNamespace()
    )
    assert not verdict.accepted
    assert verdict.code == "DELIVERY_EVIDENCE"
    assert state["notice"] is None


def run(variant, agent):
    return Engine(
        VisibilityDomain(variant),
        agent,
        identity={
            "operation_id": "synthetic.visibility",
            "operation_instance_id": "synthetic-instance",
            "agent_id": agent.agent_id,
        },
    ).run()


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_and_exact_recorded_replay(variant):
    recorder = RecordingAgent(VisibilityAgent())
    original = run(variant, recorder)
    expected = "human_hold" if variant in ("missing", "correction") else "verified_notice"
    assert all(evaluate(original, expected).values())
    tape = tape_from_records(json.loads(json.dumps(recorder.tape().as_list())))
    playback = RecordedOutcomeAgent(tape, agent_id=recorder.agent_id)
    replayed = run(variant, playback)
    playback.check_exhausted()
    assert original.as_dict() == replayed.as_dict()
    if variant == "clarified":
        assert original.final_state["carrier"]["version"] == 2
        assert original.final_state["pod"]["version"] == 2
        assert original.final_state["approval"] == {"load": LOAD, "version": 2}
    if variant == "privacy":
        assert original.final_state["tracking"] == "unknown"
    if variant == "correction":
        assert original.final_state["notice"] is None
        assert any(r.get("code") == "STALE_APPROVAL" for r in original.trajectory)


@pytest.mark.parametrize(
    "agent,variant,predicate,code",
    [
        ("tracking_only", "match", "delivery_evidence", "DELIVERY_EVIDENCE"),
        ("wrong_pod", "match", "pod_binding", "POD_BINDING"),
        ("skip_checkpoint", "match", "checkpoint", "NOTIFICATION_CHECKPOINT"),
        ("stale_notice", "correction", "correction", "DELIVERY_EVIDENCE"),
    ],
)
def test_negative_fails_only_targeted_predicate(agent, variant, predicate, code):
    recorder = RecordingAgent(VisibilityAgent(agent))
    episode = run(variant, recorder)
    playback = RecordedOutcomeAgent(
        tape_from_records(json.loads(json.dumps(recorder.tape().as_list()))),
        agent_id=recorder.agent_id,
    )
    assert run(variant, playback).as_dict() == episode.as_dict()
    playback.check_exhausted()
    expected = "human_hold" if variant == "correction" else "verified_notice"
    result = evaluate(episode, expected)
    assert {k for k, v in result.items() if not v} == {predicate}
    assert any(
        r["record_type"] == "action_rejected" and r["code"] == code
        for r in episode.trajectory
    )


@pytest.mark.parametrize("kind", ISSUERS)
@pytest.mark.parametrize("payload", [None, [], {}, {"version": True}, {"extra": 1}])
def test_malformed_external_event_atomic(kind, payload):
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    before = deepcopy(state)
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    verdict = domain.reduce_event(
        state,
        SimpleNamespace(event_type=kind, actor_id=ISSUERS[kind], payload=payload),
        context,
    )
    if kind == "checkpoint" and payload == {}:
        assert verdict.accepted and state == before | {"checkpoint": True}
        assert len(records) == 1
    else:
        assert not verdict.accepted and verdict.code == "EVENT_PAYLOAD"
        assert state == before and not records


@pytest.mark.parametrize(
    "kind,actor,code",
    [("pod", "mobile", "EVENT_ACTOR"), ("unknown", "carrier", "EVENT_TYPE")],
)
def test_wrong_event_or_actor_atomic(kind, actor, code):
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    before = deepcopy(state)
    records = []
    verdict = domain.reduce_event(
        state,
        SimpleNamespace(event_type=kind, actor_id=actor, payload={}),
        SimpleNamespace(record=lambda *args: records.append(args)),
    )
    assert verdict.code == code and not verdict.accepted
    assert state == before and not records


def test_valid_and_stale_document_events():
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(
        event_type="pod",
        actor_id="documents",
        payload={"load": LOAD, "version": 2, "status": "delivered"},
    )
    assert domain.reduce_event(state, event, context).accepted
    before = deepcopy(state)
    event.payload["version"] = 1
    assert domain.reduce_event(state, event, context).code == "STALE_EVENT"
    assert state == before and len(records) == 1


def test_permission_blocks_notice_and_stale_telemetry_is_unknown():
    domain = VisibilityDomain("match")
    state = deepcopy(run("match", VisibilityAgent()).final_state)
    state["status_permission"] = False
    before = deepcopy(state)
    assert (
        domain.apply_action(state, "notify", {"load": LOAD, "version": 1}, (), None).code
        == "STATUS_PERMISSION"
    )
    assert state == before
    context = SimpleNamespace(record=lambda *args: None)
    domain.reduce_event(
        state,
        SimpleNamespace(
            event_type="tracking",
            actor_id="mobile",
            payload={"load": LOAD, "signal": "stale"},
        ),
        context,
    )
    assert state["tracking"] == "unknown"


def test_carrier_correction_invalidates_notice_and_approval():
    domain = VisibilityDomain("match")
    state = deepcopy(run("match", VisibilityAgent()).final_state)
    state["final"] = (
        None  # exercise pre-final correction; Engine drains post-final events
    )
    domain.reduce_event(
        state,
        SimpleNamespace(
            event_type="carrier",
            actor_id="carrier",
            payload={"load": LOAD, "status": "in_transit", "version": 2},
        ),
        SimpleNamespace(record=lambda *args: None),
    )
    assert state["notice"] is None and state["approval"] is None
    assert domain.apply_terminal(state, (), None).code == "PENDING_CORRECTION"


def test_oracle_rejects_forged_result_and_wrong_expected_disposition():
    from dataclasses import replace

    episode = run("match", VisibilityAgent())
    assert not evaluate(episode, "human_hold")["disposition"]
    assert not evaluate(replace(episode), "verified_notice")["integrity"]


def test_negative_fixture_matches_executable_controls():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    controls = json.loads(
        (
            root / "examples/proposals/fura/shipment_visibility/negative_controls.json"
        ).read_text()
    )["controls"]
    assert len(controls) == 4
    for control in controls:
        episode = run(control["variant"], VisibilityAgent(control["agent"]))
        expected = (
            "human_hold" if control["variant"] == "correction" else "verified_notice"
        )
        assert {k for k, v in evaluate(episode, expected).items() if not v} == {
            control["fails"]
        }
        assert any(row.get("code") == control["rejection"] for row in episode.trajectory)


def test_proposal_python_is_outside_wheel_packages():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    proposal = Path(__file__).relative_to(root)
    assert not any(proposal.is_relative_to(package) for package in packages)


@pytest.mark.parametrize(
    "kind,actor,payload,code",
    [
        ("unknown", "carrier", {}, "EVENT_TYPE"),
        (
            "pod",
            "mobile",
            {"load": LOAD, "version": 1, "status": "delivered"},
            "EVENT_ACTOR",
        ),
        (
            "carrier",
            "carrier",
            {"load": LOAD, "status": [], "version": True},
            "EVENT_PAYLOAD",
        ),
    ],
)
def test_engine_records_rejected_external_event_without_business_side_effects(
    kind, actor, payload, code
):
    from dataclasses import replace

    from operatebench.core.protocol import PlannedEvent

    domain = VisibilityDomain("match")
    plan = replace(
        domain.build_plan(),
        events=(
            PlannedEvent(
                "synthetic-invalid",
                kind,
                actor,
                0,
                payload,
                at="2026-01-01T09:00:00Z",
                triggers_agent=False,
            ),
        ),
    )
    domain.build_plan = lambda: plan
    episode = Engine(domain, VisibilityAgent(), identity={}).run()
    assert episode.final_state == domain.initial_state()
    assert episode.events[0]["verdict_code"] == code
    assert any(
        row["record_type"] == "event_rejected" and row["code"] == code
        for row in episode.trajectory
    )
    assert not any(
        row["record_type"] in ("visibility_event", "status_notice", "human_hold")
        for row in episode.trajectory
    )


@pytest.mark.parametrize("kind", ["pod", "carrier"])
@pytest.mark.parametrize("late_version", [1, 2])
def test_engine_valid_control_and_stale_exogenous_status(kind, late_version):
    from dataclasses import replace

    from operatebench.core.protocol import PlannedEvent

    domain = VisibilityDomain("match")
    events = tuple(
        PlannedEvent(
            f"synthetic-status-{index}",
            kind,
            ISSUERS[kind],
            index,
            {
                "load": LOAD,
                "version": version,
                "status": "in_transit" if index == 0 else "delivered",
            },
            at=f"2026-01-01T09:0{index}:00Z",
            triggers_agent=False,
        )
        for index, version in enumerate((2, late_version))
    )
    plan = replace(domain.build_plan(), events=events)
    domain.build_plan = lambda: plan
    episode = Engine(domain, VisibilityAgent(), identity={}).run()
    assert episode.final_state == domain.initial_state() | {kind: events[0].payload}
    assert [event["verdict_code"] for event in episode.events] == [
        "ACCEPTED",
        "STALE_EVENT",
    ]
    assert (
        sum(row["record_type"] == "visibility_event" for row in episode.trajectory) == 1
    )
    assert [
        r["code"] for r in episode.trajectory if r["record_type"] == "event_audit_only"
    ] == [
        "ACCEPTED",
        "STALE_EVENT",
    ]
    assert not any(r["record_type"] == "event_rejected" for r in episode.trajectory)


@pytest.mark.parametrize("kind", ["tracking", "carrier", "pod", "approval"])
@pytest.mark.parametrize("bad_load", [None, True, 7, [], {}, "", " ", "synthetic-other"])
def test_load_binding_valid_control_and_atomic_refusal(kind, bad_load):
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    payloads = {
        "tracking": {"load": LOAD, "signal": "geofence"},
        "carrier": {"load": LOAD, "version": 1, "status": "delivered"},
        "pod": {"load": LOAD, "version": 1, "status": "delivered"},
        "approval": {"load": LOAD, "version": 1},
    }
    if kind == "approval":
        state["pod"] = deepcopy(payloads["pod"])
    event = SimpleNamespace(
        event_type=kind, actor_id=ISSUERS[kind], payload=deepcopy(payloads[kind])
    )
    before = deepcopy(state)
    event.payload["load"] = bad_load
    verdict = domain.reduce_event(state, event, context)
    assert not verdict.accepted
    assert verdict.code == (
        "EVENT_LOAD" if bad_load == "synthetic-other" else "EVENT_PAYLOAD"
    )
    assert state == before and records == []
    event.payload = deepcopy(payloads[kind])
    assert domain.reduce_event(state, event, context).accepted
    assert state != before and len(records) == 1
    # Late wrong-load events must be refused before stale suppression, even
    # with a current document/carrier and dependent approval/notice present.
    state["approval"] = {"load": LOAD, "version": 1}
    state["notice"] = {"load": LOAD, "version": 1}
    before = deepcopy(state)
    event.payload["load"] = "synthetic-other"
    assert domain.reduce_event(state, event, context).code == "EVENT_LOAD"
    assert state == before and len(records) == 1


@pytest.mark.parametrize("kind", ["carrier", "pod"])
@pytest.mark.parametrize(
    "version,status", [(1, "delivered"), (2, "delivered"), (2, "in_transit")]
)
def test_status_version_suppression_preserves_dependent_state(kind, version, status):
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(
        event_type=kind,
        actor_id=ISSUERS[kind],
        payload={"load": LOAD, "version": 2, "status": "in_transit"},
    )
    assert domain.reduce_event(state, event, context).accepted
    state["approval"] = {"load": LOAD, "version": 2}
    state["notice"] = {"load": LOAD, "version": 2}
    before = deepcopy(state)
    event.payload = {"load": LOAD, "version": version, "status": status}
    verdict = domain.reduce_event(state, event, context)
    assert verdict.accepted and verdict.code == "STALE_EVENT"
    assert state == before and len(records) == 1


@pytest.mark.parametrize("kind", ["tracking", "carrier", "pod", "approval"])
@pytest.mark.parametrize("bad_load", [None, "synthetic-other"])
def test_engine_load_binding_records_valid_control_and_refusal(kind, bad_load):
    from dataclasses import replace

    from operatebench.core.protocol import PlannedEvent

    domain = VisibilityDomain("match")
    payloads = {
        "tracking": {"load": LOAD, "signal": "geofence"},
        "carrier": {"load": LOAD, "version": 1, "status": "delivered"},
        "pod": {"load": LOAD, "version": 1, "status": "delivered"},
        "approval": {"load": LOAD, "version": 1},
    }
    entries = [("pod", payloads["pod"])] if kind == "approval" else []
    entries += [(kind, payloads[kind]), (kind, payloads[kind] | {"load": bad_load})]
    events = tuple(
        PlannedEvent(
            f"synthetic-binding-{index}",
            event_kind,
            ISSUERS[event_kind],
            index,
            payload,
            at=f"2026-01-01T09:0{index}:00Z",
            triggers_agent=False,
        )
        for index, (event_kind, payload) in enumerate(entries)
    )
    plan = replace(domain.build_plan(), events=events)
    domain.build_plan = lambda: plan
    episode = Engine(domain, VisibilityAgent(), identity={}).run()
    expected = domain.initial_state()
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    for event in events[:-1]:
        assert domain.reduce_event(expected, event, context).accepted
    assert episode.final_state == expected
    code = "EVENT_PAYLOAD" if bad_load is None else "EVENT_LOAD"
    assert [event["verdict_code"] for event in episode.events] == [
        *(["ACCEPTED"] * (len(events) - 1)),
        code,
    ]
    assert (
        sum(r["record_type"] == "event_audit_only" for r in episode.trajectory)
        == len(events) - 1
    )
    assert (
        sum(r["record_type"] == "visibility_event" for r in episode.trajectory)
        == len(events) - 1
    )
    assert (
        sum(
            r["record_type"] == "event_rejected" and r["code"] == code
            for r in episode.trajectory
        )
        == 1
    )
    assert not any(
        r["record_type"] in ("status_notice", "human_hold") for r in episode.trajectory
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 0),
        ("version", "2"),
        ("status", []),
        ("status", "unknown"),
    ],
)
def test_malformed_carrier_status_after_valid_control_is_atomic(field, value):
    domain = VisibilityDomain("match")
    state = domain.initial_state()
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    event = SimpleNamespace(
        event_type="carrier",
        actor_id="carrier",
        payload={"load": LOAD, "version": 2, "status": "in_transit"},
    )
    assert domain.reduce_event(state, event, context).accepted
    state["approval"] = {"load": LOAD, "version": 2}
    state["notice"] = {"load": LOAD, "version": 2}
    before = deepcopy(state)
    event.payload[field] = value
    verdict = domain.reduce_event(state, event, context)
    assert not verdict.accepted and verdict.code == "EVENT_PAYLOAD"
    assert state == before and len(records) == 1


def test_correction_emits_compensation_before_completed_hold():
    episode = run("correction", VisibilityAgent())
    notices = [r for r in episode.trajectory if r["record_type"] == "status_notice"]
    corrections = [
        r for r in episode.trajectory if r["record_type"] == "notice_correction"
    ]
    assert len(notices) == len(corrections) == 1
    assert notices[0]["at"] == "2026-01-01T09:04:00Z"
    assert corrections[0]["at"] == "2026-01-01T09:05:00Z"
    assert corrections[0]["prior_notice"] == notices[0]["notice_id"]
    assert corrections[0]["prior_version"] == 1
    assert corrections[0]["source_ref"] == "pod:2"
    assert episode.final_state["pending_corrections"] == []
    assert all(evaluate(episode, "human_hold").values())


def test_completed_without_compensation_fails_independent_grader():
    class PermissiveDomain(VisibilityDomain):
        def apply_terminal(self, state, evidence_refs, context):
            state["pending_corrections"] = []  # Deliberate test-only false state claim.
            return super().apply_terminal(state, evidence_refs, context)

    episode = Engine(
        PermissiveDomain("correction"),
        VisibilityAgent("without_compensation"),
        identity={},
    ).run()
    assert episode.replay_final and episode.terminal_outcome == "human_hold"
    assert episode.final_state["pending_corrections"] == []
    assert not any(r["record_type"] == "notice_correction" for r in episode.trajectory)
    assert {k for k, v in evaluate(episode, "human_hold").items() if not v} == {
        "correction"
    }


@pytest.mark.parametrize("kind", ["pod", "carrier"])
def test_pending_compensation_blocks_terminal_and_bad_actions_are_atomic(kind):
    domain = VisibilityDomain("match")
    state = deepcopy(run("match", VisibilityAgent()).final_state)
    state["final"] = None
    records = []
    context = SimpleNamespace(record=lambda *args: records.append(args))
    assert domain.reduce_event(
        state,
        SimpleNamespace(
            event_type=kind,
            actor_id=ISSUERS[kind],
            payload={"load": LOAD, "version": 2, "status": "in_transit"},
        ),
        context,
    ).accepted
    state["handoff"] = {"recipient": "broker", "reason": "conflict"}
    assert domain.apply_terminal(state, (), context).code == "PENDING_CORRECTION"
    obligation = deepcopy(state["pending_corrections"][0])
    refs = (LOAD, obligation["prior_notice"], obligation["source_ref"])
    for field, value, code in [
        ("load", "synthetic-other", "CORRECTION_BINDING"),
        ("prior_notice", "notice:other", "CORRECTION_BINDING"),
        ("prior_version", 2, "CORRECTION_BINDING"),
        ("source_ref", f"{kind}:1", "CORRECTION_BINDING"),
        ("recipient", "other", "STATUS_PERMISSION"),
        ("prior_version", True, "ACTION_PAYLOAD"),
    ]:
        before, audit = deepcopy(state), deepcopy(records)
        verdict = domain.apply_action(
            state, "correct_notice", obligation | {field: value}, refs, context
        )
        assert not verdict.accepted and verdict.code == code
        assert state == before and records == audit
    state["status_permission"] = False
    before = deepcopy(state)
    assert (
        domain.apply_action(state, "correct_notice", obligation, refs, context).code
        == "STATUS_PERMISSION"
    )
    assert state == before
    state["status_permission"] = True
    before = deepcopy(state)
    assert (
        domain.apply_action(state, "correct_notice", obligation, (), context).code
        == "PROVENANCE"
    )
    assert state == before
    assert domain.apply_action(
        state, "correct_notice", obligation, refs, context
    ).accepted
    assert state["pending_corrections"] == []
    assert records[-1][0] == "notice_correction"
    assert domain.apply_terminal(state, (), context).accepted
