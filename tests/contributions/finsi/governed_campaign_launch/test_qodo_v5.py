"""Regression controls for the five independently reproduced input/audit defects."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.contributions.finsi.governed_campaign_launch import runtime
from operatebench.contributions.finsi.governed_campaign_launch.agents import CampaignAgent
from operatebench.contributions.finsi.governed_campaign_launch.evaluator import (
    evaluate_episode,
)
from operatebench.contributions.finsi.governed_campaign_launch.operation import (
    START,
    WINDOWS,
    CampaignDomain,
)
from operatebench.contributions.finsi.governed_campaign_launch.pack import PACK
from operatebench.contributions.finsi.governed_campaign_launch.spec import (
    digest,
    load_spec,
)
from operatebench.core.engine import Engine
from operatebench.core.outcomes import Act
from operatebench.core.protocol import PlannedEvent
from operatebench.sdk import OperationPackError, ReplayRequest, ValidateRequest
from tests.contributions.finsi.governed_campaign_launch.test_pack import (
    FIXTURE,
    Context,
    event,
)


def episode(domain=CampaignDomain, agent=None):
    spec = load_spec(FIXTURE)
    ep = Engine(
        domain(spec, "preview"),
        agent or CampaignAgent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "qodo-v5",
            "agent_id": "reference",
        },
    ).run()
    return ep, evaluate_episode(ep, spec, "preview")


@pytest.mark.parametrize("target", ["missing", "directory", "symlink", "fifo"])
def test_input_file_types_are_named_refusals(tmp_path, target):
    path = tmp_path / target
    if target == "directory":
        path.mkdir()
    elif target == "symlink":
        path.symlink_to(FIXTURE)
    elif target == "fifo":
        # Finite nonregular sentinel: never open a live pipe on old source.
        path = tmp_path
    with pytest.raises(OperationPackError):
        PACK.validate(ValidateRequest(path))


def early_entry():
    return {
        "key": "external",
        "send_id": "external-send",
        "accepted": ["member-0000"],
        "accepted_commitment": digest(["member-0000"]),
        "registry_version": "1",
        "filtered": [],
        "at": START,
    }


def test_reconcile_before_snapshot_preserves_fact_and_refuses():
    d = CampaignDomain(load_spec(FIXTURE), "preview")
    s, c = d.initial_state(), Context()
    assert event(d, s, c, "ledger_entry", {"entry": early_entry()}).accepted
    before, n = deepcopy(s), len(c.records)
    v = d.apply_action(s, "reconcile", {"proof": "nonempty"}, (), c)
    assert not v.accepted and v.code == "CONSENT_UNVERIFIED_AT_SEND"
    assert s == before and len(c.records) == n + 1
    assert c.records[-1][0] == "finsi_refusal"


def test_core_early_reconcile_is_named_refusal():
    class Early(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "early",
                        "ledger_entry",
                        "platform",
                        -1,
                        {"entry": early_entry()},
                        at=START,
                    ),
                ),
            )

    class Agent(CampaignAgent):
        def decide_public(self, obs):
            served = obs["retrieval"].get("served", {})
            if "get_campaign" in served:
                s = served["get_campaign"]["records"]["record"]
                if s["ledger"] and not s["snapshot"]:
                    return Act("reconcile", {"proof": "nonempty"})
            return super().decide_public(obs)

    ep, grade = episode(Early, Agent("reference"))
    assert any(
        r["record_type"] == "finsi_refusal" and r["code"] == "CONSENT_UNVERIFIED_AT_SEND"
        for r in ep.trajectory
    )
    assert not grade["reliable"]


@pytest.mark.parametrize("order", [-1, 100])
def test_consent_ties_follow_physical_dispatch(order):
    class Tied(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "tie",
                        "consent_revoked",
                        "registry",
                        order,
                        {"member": "member-0000"},
                        at=WINDOWS[0][0],
                    ),
                ),
            )

    ep, grade = episode(Tied)
    assert ("member-0000" in ep.final_state["ledger"][0]["accepted"]) == (order == 100)
    assert grade["reliable"], grade


@pytest.mark.parametrize("known_key", [False, True])
def test_external_open_second_send_retains_incident_without_budget_failure(known_key):
    baseline, _ = episode()
    entry = deepcopy(baseline.final_state["ledger"][0])
    entry["send_id"] = "second-send"
    if not known_key:
        entry["key"] = "external-key"

    class Second(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "second",
                        "ledger_entry",
                        "platform",
                        100,
                        {"entry": entry},
                        at="2026-10-05T08:01:00Z",
                    ),
                ),
            )

    ep, grade = episode(Second)
    assert ep.final_state == baseline.final_state
    assert grade["incidents"][0]["owner"] == "human_owner"
    assert not grade["incidents"][0]["after_terminal"]
    assert grade["reliable"], grade


@pytest.fixture(scope="module")
def record():
    return runtime.run(FIXTURE, "preview")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        {"decision": None},
        *[{"decision": {"kind": k}} for k in ["ACT", "WAIT", "COMPLETE", "RETRIEVE"]],
        {"decision": {"kind": "RETRIEVE", "requests": [None]}},
    ],
)
def test_malformed_replay_preflight(record, tmp_path, monkeypatch, bad):
    body = deepcopy(record)
    body["decisions"][0] = bad
    if isinstance(bad, dict):
        bad["observation"] = record["decisions"][0]["observation"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(body))

    def forbidden(*args):
        pytest.fail("malformed input reached Engine")

    monkeypatch.setattr(runtime, "execute", forbidden)
    with pytest.raises(OperationPackError):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize("limit", [100000, 20000000])
def test_bounded_reader_exact_limit_and_overflow(tmp_path, monkeypatch, limit):
    import os

    from operatebench.contributions.finsi.governed_campaign_launch import (
        input as boundary,
    )

    path = tmp_path / "bytes"
    path.write_bytes(b" " * limit)
    original = os.fdopen
    requests, descriptors = [], []

    class Spy:
        def __init__(self, fd, *args):
            descriptors.append(fd)
            self.stream = original(fd, *args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            requests.append(size)
            assert size == limit + 1
            return self.stream.read(size)

    monkeypatch.setattr(os, "fdopen", Spy)
    assert len(boundary.read_bytes(path, limit)) == limit
    path.write_bytes(b" " * (limit + 1))
    with pytest.raises(OperationPackError, match="oversize"):
        boundary.read_bytes(path, limit)
    assert requests == [limit + 1, limit + 1]
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("failure", [OSError, TypeError])
def test_reader_closes_descriptor_without_masking_internal_errors(
    tmp_path, monkeypatch, failure
):
    import os

    from operatebench.contributions.finsi.governed_campaign_launch import (
        input as boundary,
    )

    path = tmp_path / "input"
    path.write_bytes(b"{}")
    original = os.fstat
    descriptors = []

    def broken(fd):
        descriptors.append(fd)
        raise failure("injected")

    monkeypatch.setattr(os, "fstat", broken)
    with pytest.raises(OperationPackError if failure is OSError else TypeError):
        boundary.read_bytes(path, 100)
    for fd in descriptors:
        with pytest.raises(OSError):
            original(fd)


def test_nonregular_descriptor_is_refused_without_reading(tmp_path, monkeypatch):
    import os
    import stat
    from types import SimpleNamespace

    from operatebench.contributions.finsi.governed_campaign_launch import (
        input as boundary,
    )

    path = tmp_path / "sentinel"
    path.write_bytes(b"{}")
    monkeypatch.setattr(os, "fstat", lambda fd: SimpleNamespace(st_mode=stat.S_IFIFO))
    monkeypatch.setattr(os, "fdopen", lambda *a: pytest.fail("read nonregular file"))
    with pytest.raises(OperationPackError, match="regular"):
        boundary.read_bytes(path, 100)


@pytest.mark.parametrize("module", ["execute", "evaluate_episode"])
@pytest.mark.parametrize("failure", [KeyError, TypeError])
def test_internal_replay_errors_remain_internal(
    record, tmp_path, monkeypatch, module, failure
):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))

    def broken(*args):
        raise failure("internal sentinel")

    monkeypatch.setattr(runtime, module, broken)
    with pytest.raises(failure, match="internal sentinel"):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize("kind", ["ACT", "WAIT", "COMPLETE", "RETRIEVE"])
def test_each_variant_missing_fields_and_wrong_shapes_preflight(
    record, tmp_path, monkeypatch, kind
):
    original = next(
        r["decision"] for r in record["decisions"] if r["decision"]["kind"] == kind
    )
    cases = []
    for key in original:
        p = deepcopy(original)
        del p[key]
        cases.append(p)
    for key in set(original) - {"kind"}:
        p = deepcopy(original)
        p[key] = False
        cases.append(p)
    if kind == "WAIT":
        for bound in [0, -1, True, 1.5, "1", 10**100]:
            cases.append({**original, "fallback_after_minutes": bound})
    if kind == "RETRIEVE":
        for requests in [
            [None],
            [{"tool": "x"}],
            [{"tool": 1, "arguments": {}}],
            [{"tool": "x", "arguments": []}],
        ]:
            cases.append({**original, "requests": requests})
    monkeypatch.setattr(
        runtime, "execute", lambda *args: pytest.fail("executed malformed tape")
    )
    for decision in cases:
        body = deepcopy(record)
        body["decisions"][0]["decision"] = decision
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(body))
        with pytest.raises(OperationPackError):
            PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("scenario", []),
        ("agent", {}),
        ("spec_digest", "invalid"),
        ("decisions", {}),
        ("episode", []),
        ("evaluation", None),
        ("format", "old"),
        ("scope", False),
    ],
)
def test_envelope_preflight(record, tmp_path, monkeypatch, field, value):
    body = deepcopy(record)
    body[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(body))
    monkeypatch.setattr(
        runtime, "execute", lambda *args: pytest.fail("executed bad envelope")
    )
    with pytest.raises(OperationPackError):
        PACK.replay(ReplayRequest(FIXTURE, path))


def test_historical_v4_replay_refuses_before_execution(record, tmp_path, monkeypatch):
    body = deepcopy(record)
    body["evaluation"]["grading_version"] = "finsi.governed-campaign.grading.v4"
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(body))
    monkeypatch.setattr(
        runtime, "execute", lambda *args: pytest.fail("executed historical tape")
    )
    with pytest.raises(OperationPackError, match="grading version mismatch"):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize("order", [-1, 100])
@pytest.mark.parametrize("witness", [False, True])
def test_delayed_receipt_correction_consent_uses_physical_history(order, witness):
    baseline, _ = episode()
    entry = deepcopy(baseline.final_state["ledger"][0])
    if order == -1:
        entry["accepted"].remove("member-0000")
        entry["accepted_commitment"] = digest(entry["accepted"])
        entry["filtered"] = ["member-0000"]
        entry["registry_version"] = "2"

    class Delayed(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            events = [e for e in p.events if witness or e.event_type != "dispatch"]
            events.extend(
                [
                    PlannedEvent(
                        "tie",
                        "consent_revoked",
                        "registry",
                        order,
                        {"member": "member-0000"},
                        at=WINDOWS[0][0],
                    ),
                    PlannedEvent(
                        "receipt",
                        "ledger_entry",
                        "platform",
                        110,
                        {"entry": entry},
                        at="2026-10-05T08:02:00Z",
                    ),
                    PlannedEvent(
                        "correction",
                        "ledger_correction",
                        "platform",
                        111,
                        {"send_id": "send-1", "rejected": []},
                        at="2026-10-05T08:03:00Z",
                    ),
                ]
            )
            return replace(p, events=tuple(events))

    ep, grade = episode(Delayed)
    assert ep.terminal_outcome == "sent_reconciled"
    assert grade["dimensions"]["consent"] is witness, grade
    if witness:
        assert grade["reliable"], grade


@pytest.mark.parametrize(
    "at,issuer",
    [
        ("2026-10-05T07:59:00Z", "registry"),
        ("2026-10-05T08:01:00Z", "registry"),
        ("2026-10-05T08:00:00Z", "brief_owner"),
    ],
)
def test_earlier_later_and_wrong_authority_revocations(at, issuer):
    class Revoked(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "revoke",
                        "consent_revoked",
                        issuer,
                        100,
                        {"member": "member-0000"},
                        at=at,
                    ),
                ),
            )

    ep, grade = episode(Revoked)
    assert grade["reliable"], grade
    assert ("member-0000" in ep.final_state["ledger"][0]["accepted"]) == (
        issuer != "registry" or at > WINDOWS[0][0]
    )


@pytest.mark.parametrize("cause", ["schedule", "reconcile"])
def test_second_receipt_attribution_requires_real_schedule_cause(cause):
    from datetime import datetime

    from operatebench.core.protocol import PlannedTrigger

    baseline, _ = episode()
    entry = deepcopy(baseline.final_state["ledger"][0])
    entry["send_id"] = "second-send"
    scheduled_at = next(
        r["at"]
        for r in baseline.trajectory
        if r["record_type"] == "finsi_action" and r["action_type"] == "schedule"
    )
    delay = int(
        (
            datetime.fromisoformat("2026-10-05T08:01:00+00:00")
            - datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
        ).total_seconds()
        / 60
    )

    class Caused(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "second",
                        "ledger_entry",
                        "platform",
                        100,
                        {"entry": entry},
                        trigger=PlannedTrigger("action", cause),
                        delay_minutes=1 if cause == "reconcile" else delay,
                    ),
                ),
            )

    ep, grade = episode(Caused)
    received = next(e for e in ep.events if e["event_id"] == "second")
    assert received["caused_by"] == "action:" + cause
    assert grade["incidents"]
    assert grade["dimensions"]["budgets"] is (cause != "schedule"), grade


def test_corrupt_dispatch_two_sends_without_incident_still_fails_budgets():
    class Duplicate(CampaignDomain):
        def reduce_event(self, state, e, context):
            verdict = super().reduce_event(state, e, context)
            if (
                e.event_type == "dispatch"
                and state["ledger"]
                and len(state["ledger"]) == 1
            ):
                second = deepcopy(state["ledger"][0])
                second["send_id"] = "producer-duplicate"
                state["ledger"].append(second)
                context.record(
                    "finsi_event",
                    {
                        "event_type": "dispatch",
                        "actor_id": "platform",
                        "payload": dict(e.payload),
                        "ledger": deepcopy(state["ledger"]),
                    },
                )
            return verdict

    from operatebench.core.outcomes import Wait

    class AwaitHuman(CampaignAgent):
        def decide_public(self, obs):
            served = obs["retrieval"].get("served", {})
            if "get_campaign" in served:
                state = served["get_campaign"]["records"]["record"]
                if len(state["ledger"]) > 1:
                    return Wait(
                        "await human incident handling",
                        tuple(obs["wake_event_types"]),
                        1440,
                    )
            return super().decide_public(obs)

    ep, grade = episode(Duplicate, AwaitHuman("reference"))
    assert not any(r["record_type"] == "finsi_incident" for r in ep.trajectory)
    assert not grade["dimensions"]["budgets"], grade
    assert not grade["dimensions"]["terminal"], grade


@pytest.mark.parametrize("raw", [b"{", b"\xff", b'{"a":1,"a":2}', b"[" * 2000])
def test_invalid_json_is_named_at_public_replay_boundary(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(OperationPackError):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize("body", [None, False, [], "record", {}])
def test_non_envelope_input_is_named_before_execution(tmp_path, monkeypatch, body):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(body))
    monkeypatch.setattr(
        runtime, "execute", lambda *args: pytest.fail("executed bad envelope")
    )
    with pytest.raises(OperationPackError):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.mark.parametrize(
    "change", ["observation", "consumption", "episode", "evaluation"]
)
def test_replay_mismatches_are_named(record, tmp_path, change):
    body = deepcopy(record)
    if change == "observation":
        body["decisions"][0]["observation"] = "0" * 64
    elif change == "consumption":
        body["decisions"].append(deepcopy(body["decisions"][-1]))
    elif change == "episode":
        body["episode"]["terminal_outcome"] = "rewritten"
    else:
        body["evaluation"]["reliable"] = False
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(body))
    with pytest.raises(OperationPackError, match="mismatch"):
        PACK.replay(ReplayRequest(FIXTURE, path))


@pytest.fixture(scope="module")
def receipt_baseline():
    return episode()[0]


def chronology_receipt_episode(
    baseline, physical, cause, known_key, first=False, arrival="2026-10-05T08:01:00Z"
):
    """Real Core delivery/cause; only the receipt's physical timestamp varies."""
    from datetime import datetime, timedelta

    from operatebench.core.protocol import PlannedTrigger

    def action_time(kind):
        return next(
            r["at"]
            for r in baseline.trajectory
            if r["record_type"] == "finsi_action" and r["action_type"] == kind
        )

    scheduled_at = action_time("schedule")
    physical_at = {
        "before_schedule": (
            datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
            - timedelta(minutes=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "at_schedule": scheduled_at,
        "valid": baseline.final_state["ledger"][0]["at"],
        "at_arrival": arrival,
        "after_arrival": (
            datetime.fromisoformat(arrival.replace("Z", "+00:00")) + timedelta(minutes=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }[physical]
    entry = deepcopy(baseline.final_state["ledger"][0])
    entry.update(send_id="send-1" if first else "second-send", at=physical_at)
    if not known_key:
        entry["key"] = "unknown-external-key"
    timing = {"at": arrival}
    if cause is not None:
        delay = int(
            (
                datetime.fromisoformat(arrival.replace("Z", "+00:00"))
                - datetime.fromisoformat(action_time(cause).replace("Z", "+00:00"))
            ).total_seconds()
            / 60
        )
        timing = {"trigger": PlannedTrigger("action", cause), "delay_minutes": delay}

    class Receipt(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            events = tuple(e for e in p.events if not first or e.event_type != "dispatch")
            return replace(
                p,
                events=(
                    *events,
                    PlannedEvent(
                        "chronology-receipt",
                        "ledger_entry",
                        "platform",
                        100,
                        {"entry": entry},
                        **timing,
                    ),
                ),
            )

    ep, grade = episode(Receipt)
    fact = next(e for e in ep.events if e["event_id"] == "chronology-receipt")
    assert fact["caused_by"] == (None if cause is None else "action:" + cause)
    assert fact["payload"]["entry"] == entry
    return ep, grade, fact


@pytest.mark.parametrize(
    "physical",
    [
        "before_schedule",
        "at_schedule",
        "valid",
        "at_arrival",
        "after_arrival",
    ],
)
@pytest.mark.parametrize(
    "cause,known_key",
    [
        ("schedule", True),
        ("verify", True),
        (None, True),
        ("schedule", False),
    ],
)
def test_second_receipt_chronology_boundary(receipt_baseline, physical, cause, known_key):
    ep, grade, fact = chronology_receipt_episode(
        receipt_baseline, physical, cause, known_key
    )
    assert fact["disposition"] == "accepted"
    assert ep.final_state == receipt_baseline.final_state
    assert grade["incidents"] == [
        {
            "code": "SECOND_SEND_DETECTED",
            "send_id": "second-send",
            "owner": "human_owner",
            "after_terminal": False,
        }
    ]
    assert any(r["record_type"] == "finsi_incident" for r in ep.trajectory)
    associated = cause == "schedule" and known_key
    contradiction = physical in ("before_schedule", "after_arrival")
    assert grade["reliable"] is (not associated), grade
    assert grade["dimensions"]["budgets"] is (not associated or contradiction), grade
    assert grade["dimensions"]["authority"] is (not associated or not contradiction), (
        grade
    )
    if associated and contradiction:
        assert grade["findings"] == [
            {"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"}
        ]


@pytest.mark.parametrize(
    "physical",
    [
        "before_schedule",
        "at_schedule",
        "valid",
        "at_arrival",
        "after_arrival",
    ],
)
@pytest.mark.parametrize(
    "cause,known_key",
    [
        ("schedule", True),
        ("verify", True),
        (None, True),
        ("schedule", False),
    ],
)
def test_first_receipt_chronology_boundary(receipt_baseline, physical, cause, known_key):
    ep, grade, fact = chronology_receipt_episode(
        receipt_baseline, physical, cause, known_key, first=True
    )
    assert fact["disposition"] == "accepted"
    assert ep.terminal_outcome == "sent_reconciled"
    assert grade["incidents"] == []
    assert grade["dimensions"]["budgets"]
    # Existing own first-receipt audit also requires the approved window;
    # inherited unknown-key receipts do not establish an agent schedule origin.
    invalid = known_key and physical in (
        "before_schedule",
        "at_schedule",
        "after_arrival",
    )
    assert grade["reliable"] is (not invalid), grade
    assert grade["dimensions"]["authority"] is (not invalid), grade
    if invalid:
        assert grade["findings"] == [
            {"dimension": "authority", "code": "DISPATCH_APPROVAL_INVALID"}
        ]


@pytest.mark.parametrize("physical", ["before_schedule", "valid", "after_arrival"])
@pytest.mark.parametrize(
    "cause,known_key",
    [
        ("schedule", True),
        ("verify", True),
        (None, True),
        ("schedule", False),
    ],
)
def test_post_terminal_receipt_chronology_is_human_incident(
    receipt_baseline,
    physical,
    cause,
    known_key,
):
    ep, grade, fact = chronology_receipt_episode(
        receipt_baseline, physical, cause, known_key, arrival="2026-10-07T09:00:00Z"
    )
    assert fact["disposition"] == "post_terminal"
    assert ep.final_state == receipt_baseline.final_state
    assert grade["incidents"] == [
        {
            "code": "SECOND_SEND_DETECTED",
            "send_id": "second-send",
            "owner": "human_owner",
            "after_terminal": True,
        }
    ]
    assert grade["reliable"], grade
