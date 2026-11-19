"""Public domain trust boundaries; Core execution is tested by the integrator."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from operatebench.contributions.finsi.governed_campaign_launch import (
    REFERENCE_SCENARIOS,
    build_agent,
    build_domain,
    evaluate_episode,
    load_spec,
)
from operatebench.contributions.finsi.governed_campaign_launch.controls import (
    NEGATIVE_CONTROLS,
)
from operatebench.contributions.finsi.governed_campaign_launch.operation import (
    digest,
    document,
    plus,
    schedule_key,
)
from operatebench.contributions.finsi.governed_campaign_launch.runtime import replay, run
from operatebench.core.engine import Engine
from operatebench.core.protocol import model_projection

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "examples/contributions/finsi/governed_campaign_launch/operation.yaml"
)


class Context:
    now = "2026-10-01T09:00:00Z"

    def __init__(self):
        self.records = []

    def record(self, kind, payload):
        self.records.append((kind, deepcopy(payload)))


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("approval_record", {"role": "agent"}),
        ("document_ready", {"sections": 4}),
        ("consent_revoked", {"member": "outsider"}),
        ("ledger_entry", {"send_id": "forged"}),
    ],
)
def test_invalid_external_fact_is_atomic(kind, payload):
    d = build_domain(load_spec(FIXTURE), "baseline")
    s = d.initial_state()
    before = deepcopy(s)
    c = Context()
    e = SimpleNamespace(event_type=kind, actor_id="agent", payload=payload)
    assert not d.reduce_event(s, e, c).accepted
    assert s == before and c.records == []


def test_identity_reconciliation_rejects_equal_count_wrong_set():
    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        identity_diff,
    )

    assert identity_diff({"a", "b"}, {"a", "b"}, {"a", "intruder"}) == {"intruder"}
    assert identity_diff({"a", "b"}, {"a"}, {"a"}) == set()


EXPECTED = {
    "consent_unenforced": "withdrawn_no_send",
    "timeout_unprotected": "TRANSFERRED_UNRESOLVED",
    "late_settlement": "withdrawn_no_send",
}


@pytest.mark.parametrize("case", REFERENCE_SCENARIOS)
@pytest.mark.parametrize("agent", ["reference", "alternate"])
def test_core_reference_alternate_and_exact_saved_tape(case, agent, tmp_path):
    import json

    record = run(FIXTURE, case, agent)
    assert record["evaluation"]["reliable"], record["evaluation"]
    assert record["episode"]["terminal_outcome"] == EXPECTED.get(case, "sent_reconciled")
    target = tmp_path / "run.json"
    target.write_text(json.dumps(record))
    assert replay(FIXTURE, json.loads(target.read_text()))["consistent"]
    actions = [
        r for r in record["episode"]["trajectory"] if r["record_type"] == "agent_outcome"
    ]
    assert len(actions) == sum(
        r["decision"]["kind"] != "RETRIEVE" for r in record["decisions"]
    )
    assert any(r["outcome"]["kind"] == "WAIT" for r in actions)


@pytest.mark.parametrize("control", NEGATIVE_CONTROLS, ids=lambda c: c["agent_id"])
def test_targeted_negative_has_exact_failure_and_recovers(control):
    record = run(FIXTURE, control["scenario_id"], control["agent_id"])
    g = record["evaluation"]
    assert not g["reliable"]
    assert {k for k, v in g["dimensions"].items() if not v} == set(
        control["failed_dimensions"]
    ), g
    assert {f["code"] for f in g["findings"]} == set(control["finding_codes"]), g
    assert record["episode"]["replay_final"]


def test_agent_only_uses_published_json():
    import json

    class Public:
        agent_id = "reference"

        def begin_episode(self, identity):
            assert "scenario_id" not in identity
            self.agent = build_agent("reference")
            self.agent.begin_episode(identity)

        def decide(self, obs):
            public = json.loads(json.dumps(model_projection(obs)))
            assert "scenario_id" not in json.dumps(public)
            return self.agent.decide_public(public)

    spec = load_spec(FIXTURE)
    e = Engine(
        build_domain(spec, "consent_revoked"),
        Public(),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": "reference",
        },
    ).run()
    assert evaluate_episode(e, spec, "consent_revoked")["reliable"]
    assert len(e.final_state["ledger"][0]["accepted"]) == 2339
    assert e.final_state["ledger"][0]["filtered"] == ["member-0000"]


def event(d, s, c, kind, payload, actor=None):
    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        ISSUERS,
    )

    if kind == "approval_record" and set(payload) == {"version", "role"}:
        payload = {
            **payload,
            "request_id": s["request"]["request_id"],
            "reviewed_binding": digest(s["request"]),
            "approval_id": s["request"]["approval_id"],
        }
    return d.reduce_event(
        s,
        SimpleNamespace(
            event_type=kind, actor_id=actor or ISSUERS[kind], payload=payload
        ),
        c,
    )


def prepared(case="baseline"):
    d = build_domain(load_spec(FIXTURE), case)
    s = d.initial_state()
    c = Context()

    def act(k, **p):
        v = d.apply_action(s, k, p, (), c)
        assert v.accepted, (k, v)

    act("draft")
    act("push")
    doc = document(1)
    if case == "preview":
        doc["discount"] = 0
    assert event(d, s, c, "document_ready", {"document": doc, "attempt": 1}).accepted
    act("verify")
    act("snapshot")
    act("request_approval", role="brief_owner" if case == "preview" else "send_approver")
    assert s["request"]["verified_stored_document"] == doc
    assert s["request"]["snapshot_count"] == len(s["members"])
    assert s["request"]["catalogue_prices"] == doc["prices"]
    assert event(
        d,
        s,
        c,
        "approval_record",
        {"version": 1, "role": "brief_owner" if case == "preview" else "send_approver"},
    ).accepted
    return d, s, c


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("sections", {}),
        ("images", [[]]),
        ("prices", {"lantern": True, "throw": 63}),
        ("variables", "raw"),
    ],
)
def test_malformed_document_correct_issuer_atomic_with_valid_control(field, value):
    d = build_domain(load_spec(FIXTURE), "baseline")
    s = d.initial_state()
    c = Context()
    assert d.apply_action(s, "draft", {}, (), c).accepted
    assert d.apply_action(s, "push", {}, (), c).accepted
    doc = document(1)
    doc[field] = value
    before = deepcopy(s)
    audit = deepcopy(c.records)
    assert not event(d, s, c, "document_ready", {"document": doc, "attempt": 1}).accepted
    assert s == before and c.records == audit
    assert event(
        d, s, c, "document_ready", {"document": document(1), "attempt": 1}
    ).accepted


def test_snapshot_stays_frozen_approval_bounds_above_consent():
    d, s, c = prepared()
    snapshot = deepcopy(s["snapshot"])
    approval = deepcopy(s["approval"])
    assert event(d, s, c, "consent_revoked", {"member": "member-0000"}).accepted
    assert s["snapshot"] == snapshot and s["approval"] == approval
    assert "member-0000" in s["consent_versions"]["1"]
    assert "member-0000" not in s["consent_versions"]["2"]
    assert d.apply_action(s, "schedule", {"key": schedule_key(s)}, (), c).accepted
    c.now = s["window"][0]
    assert event(d, s, c, "dispatch", {"window_index": 0}).accepted
    assert s["ledger"][0]["registry_version"] == "2"
    assert s["ledger"][0]["filtered"] == ["member-0000"]
    assert d.apply_action(
        s, "reconcile", {"proof": d.identity_proof(s)["handle"]}, (), c
    ).accepted


def test_wrong_role_approval_is_atomic():
    d, s, c = prepared()
    s["approval"] = None
    before = deepcopy(s)
    audit = deepcopy(c.records)
    assert not event(
        d, s, c, "approval_record", {"version": 1, "role": "brief_owner"}
    ).accepted
    assert s == before and c.records == audit
    assert event(
        d, s, c, "approval_record", {"version": 1, "role": "send_approver"}
    ).accepted


def test_deadline_and_stale_catalogue_cannot_schedule():
    d, s, c = prepared()
    c.now = plus(s["window"][1], 1)
    assert (
        d.apply_action(s, "schedule", {"key": schedule_key(s)}, (), c).code
        == "ACTION_AFTER_DEADLINE"
    )
    c.now = "2026-10-02T09:00:00Z"
    assert event(d, s, c, "catalogue_changed", {"version": 2}).accepted
    assert (
        d.apply_action(s, "schedule", {"key": schedule_key(s)}, (), c).code
        == "APPROVAL_VERSION_MISMATCH"
    )
    assert not s["schedules"]


def test_failed_connector_zero_is_unknown_not_empty():
    d = build_domain(load_spec(FIXTURE), "baseline")
    s = d.initial_state()
    c = Context()
    assert event(d, s, c, "connector_result", {"status": "failed", "count": 0}).accepted
    assert d.apply_action(s, "snapshot", {}, (), c).accepted
    assert len(s["snapshot"]["members"]) == len(s["members"]) == 2340
    assert event(d, s, c, "connector_result", {"status": "ready", "count": 2340}).accepted
    assert d.apply_action(s, "snapshot", {}, (), c).code == "IMMUTABLE_SNAPSHOT"


def test_cancellation_requires_external_owner_and_no_schedule():
    d = build_domain(load_spec(FIXTURE), "baseline")
    s = d.initial_state()
    c = Context()
    assert not d.apply_action(s, "cancel", {}, (), c).accepted
    assert event(d, s, c, "cancellation_record", {"cancel": True}).accepted
    assert d.apply_action(s, "cancel", {}, (), c).accepted
    assert d.apply_action(s, "status", {}, (), c).accepted
    assert d.apply_terminal(s, (), c).accepted and s["terminal"] == "cancelled"


def test_non_idempotent_timeout_cannot_retry_and_transfer_needs_acceptance():
    d, s, c = prepared("timeout_unprotected")
    key = schedule_key(s)
    assert d.apply_action(s, "schedule", {"key": key}, (), c).accepted
    public = d.serve_retrieval(s, [SimpleNamespace(tool="get_campaign")], c.now)[
        0
    ].records["record"]
    assert s["schedules"] and public["schedules"] == {}
    assert public["timeout"] and public["attempts"][key] == 1
    assert d.apply_action(s, "schedule", {"key": key}, (), c).code == "RETRY_BUDGET"
    assert d.apply_action(s, "request_transfer", {}, (), c).accepted
    assert not d.apply_terminal(s, (), c).accepted
    assert event(d, s, c, "transfer_accepted", {"owner": "human_owner"}).accepted
    assert (
        d.apply_terminal(s, (), c).accepted and s["terminal"] == "TRANSFERRED_UNRESOLVED"
    )


def test_no_escalate_everything_or_third_push():
    d, s, c = prepared()
    assert d.apply_action(s, "request_transfer", {}, (), c).code == "UNJUSTIFIED_TRANSFER"
    assert d.apply_action(s, "push", {}, (), c).accepted
    before = deepcopy(s["pushes"])
    assert d.apply_action(s, "push", {}, (), c).code == "PUSH_BUDGET"
    assert s["pushes"] == before


def test_platform_ledger_without_consent_evidence_transfers_never_final_send():
    d, s, c = prepared("consent_unenforced")
    row = {
        "key": "unresolved-key",
        "send_id": "external-send",
        "accepted": ["member-0000"],
        "accepted_commitment": None,
        "registry_version": None,
        "filtered": [],
        "at": s["window"][0],
    }
    assert event(d, s, c, "ledger_entry", {"entry": row}).accepted
    assert (
        d.apply_action(
            s, "reconcile", {"proof": d.identity_proof(s)["handle"]}, (), c
        ).code
        == "CONSENT_UNVERIFIED_AT_SEND"
    )
    assert d.apply_action(s, "request_transfer", {}, (), c).accepted
    assert event(d, s, c, "transfer_accepted", {"owner": "human_owner"}).accepted
    assert d.apply_terminal(s, (), c).accepted
    assert s["terminal"] == "TRANSFERRED_UNRESOLVED"


def test_import_never_ready_real_core_fallback():
    from dataclasses import replace

    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        CampaignDomain,
    )

    class NeverReady(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p, events=tuple(e for e in p.events if e.event_type != "document_ready")
            )

    spec = load_spec(FIXTURE)
    e = Engine(
        NeverReady(spec, "baseline"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": "reference",
        },
    ).run()
    assert e.terminal_outcome == "TRANSFERRED_UNRESOLVED"
    assert e.final_state["pushes"] == {"1": 2} and not e.final_state["schedules"]
    assert evaluate_episode(e, spec, "baseline")["reliable"]


def test_cooldown_correction_reopens_and_late_duplicates_are_audit_only():
    from dataclasses import replace

    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        CampaignDomain,
    )
    from operatebench.core.protocol import PlannedEvent

    class Corrected(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "correction",
                        "ledger_correction",
                        "platform",
                        90,
                        {"send_id": "send-1", "rejected": ["member-0001"]},
                        at="2026-10-05T20:00:00Z",
                    ),
                ),
            )

    spec = load_spec(FIXTURE)
    e = Engine(
        Corrected(spec, "baseline"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": "reference",
        },
    ).run()
    assert evaluate_episode(e, spec, "baseline")["reliable"]
    reconciles = [
        r
        for r in e.trajectory
        if r["record_type"] == "finsi_action" and r["action_type"] == "reconcile"
    ]
    assert len(reconciles) == 2
    assert e.final_state["cooldown_until"] == "2026-10-06T20:00:00Z"
    d = build_domain(spec, "baseline")
    s = deepcopy(e.final_state)
    c = Context()
    before = deepcopy(s)
    assert event(d, s, c, "ledger_entry", {"entry": s["ledger"][0]}).code == "AUDIT_ONLY"
    assert (
        event(d, s, c, "document_ready", {"document": s["stored"], "attempt": 1}).code
        == "AUDIT_ONLY"
    )
    assert s == before
    second = deepcopy(s["ledger"][0])
    second["send_id"] = "send-2"
    assert (
        event(d, s, c, "ledger_entry", {"entry": second}).code == "SECOND_SEND_DETECTED"
    )
    assert s == before and c.records[-1][0] == "finsi_incident"


def test_independent_evaluator_rejects_corrupt_producer_equal_count_wrong_ids():
    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        CampaignDomain,
    )

    class Corrupt(CampaignDomain):
        def reduce_event(self, s, e, c):
            verdict = super().reduce_event(s, e, c)
            if e.event_type == "dispatch" and s["ledger"]:
                row = s["ledger"][0]
                row["accepted"][0] = "outsider"
                row["accepted_commitment"] = digest(row["accepted"])
                c.record(
                    "finsi_event",
                    {
                        "event_type": "dispatch",
                        "actor_id": "platform",
                        "payload": dict(e.payload),
                        "ledger": deepcopy(s["ledger"]),
                    },
                )
            return verdict

        def apply_action(self, s, k, p, refs, c):
            if k == "reconcile":
                s["reconciled"] = True
                s["cooldown_until"] = plus(c.now, 1440)
                c.record(
                    "finsi_action",
                    {
                        "action_type": k,
                        "payload": {},
                        "version": s["version"],
                        "stored": None,
                        "binding": None,
                    },
                )
                from operatebench.core.protocol import Verdict

                return Verdict.ok()
            return super().apply_action(s, k, p, refs, c)

    spec = load_spec(FIXTURE)
    e = Engine(
        Corrupt(spec, "consent_revoked"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "opaque-test-instance",
            "agent_id": "reference",
        },
    ).run()
    assert e.terminal_outcome == "sent_reconciled"
    grade = evaluate_episode(e, spec, "consent_revoked")
    assert not grade["dimensions"]["consent"] and not grade["reliable"]


def test_forged_final_label_and_citation_spray_have_no_authority():
    from dataclasses import replace

    from operatebench.contributions.finsi.governed_campaign_launch.runtime import execute

    spec = load_spec(FIXTURE)
    e = execute(spec, "preview", build_agent("reference"))
    assert not evaluate_episode(
        replace(e, terminal_outcome="sent_reconciled"), spec, "preview"
    )["reliable"]
    assert not evaluate_episode(e, spec, "baseline")["dimensions"]["provenance"]


def test_live_core_distinct_late_send_is_human_incident_not_duplicate():
    from dataclasses import replace

    from operatebench.contributions.finsi.governed_campaign_launch.operation import (
        CampaignDomain,
    )
    from operatebench.contributions.finsi.governed_campaign_launch.runtime import execute
    from operatebench.core.protocol import PlannedEvent

    spec = load_spec(FIXTURE)
    baseline = execute(spec, "preview", build_agent("reference"))
    first = deepcopy(baseline.final_state["ledger"][0])
    second = deepcopy(first)
    second["send_id"] = "send-2"

    class Late(CampaignDomain):
        def build_plan(self):
            p = super().build_plan()
            return replace(
                p,
                events=(
                    *p.events,
                    PlannedEvent(
                        "duplicate-receipt",
                        "ledger_entry",
                        "platform",
                        90,
                        {"entry": first},
                        at="2026-10-07T08:00:00Z",
                    ),
                    PlannedEvent(
                        "distinct-receipt",
                        "ledger_entry",
                        "platform",
                        91,
                        {"entry": second},
                        at="2026-10-07T09:00:00Z",
                    ),
                ),
            )

    e = Engine(
        Late(spec, "preview"),
        build_agent("reference"),
        identity={
            "operation_id": spec.operation_id,
            "operation_instance_id": "finsi-synthetic-development-instance",
            "agent_id": "reference",
        },
    ).run()
    grade = evaluate_episode(e, spec, "preview")
    assert grade["reliable"]
    assert grade["incidents"] == [
        {
            "code": "SECOND_SEND_DETECTED",
            "send_id": "send-2",
            "owner": "human_owner",
            "after_terminal": True,
        }
    ]
    assert e.final_state == baseline.final_state
    assert (
        len(
            [
                r
                for r in e.trajectory
                if r["record_type"] == "event_after_terminal"
                and r["event_type"] == "ledger_entry"
            ]
        )
        == 2
    )


def test_sdk_commands_are_executable_without_registration(tmp_path):
    from operatebench.contributions.finsi.governed_campaign_launch.pack import PACK
    from operatebench.sdk import ReplayRequest, RunRequest, ValidateRequest

    assert PACK.validate(ValidateRequest(FIXTURE)).contract_passed
    output = tmp_path / "preview.json"
    assert PACK.run(RunRequest(FIXTURE, "preview", "reference", output)).contract_passed
    assert PACK.replay(ReplayRequest(FIXTURE, output)).payload["provider_calls"] == 0
    with pytest.raises(Exception, match="write refused"):
        PACK.run(RunRequest(FIXTURE, "preview", "reference", output))


def test_empty_dispatch_consent_is_a_valid_identity_subset():
    d, s, c = prepared("preview")
    for member in list(s["members"]):
        assert event(d, s, c, "consent_revoked", {"member": member}).accepted
    assert d.apply_action(s, "schedule", {"key": schedule_key(s)}, (), c).accepted
    c.now = s["window"][0]
    assert event(d, s, c, "dispatch", {"window_index": 0}).accepted
    assert s["ledger"][0]["accepted"] == []
    assert d.apply_action(
        s, "reconcile", {"proof": d.identity_proof(s)["handle"]}, (), c
    ).accepted


def test_early_dispatch_and_early_blanket_withdrawal_are_refused():
    d, s, c = prepared()
    assert d.apply_action(s, "withdraw", {}, (), c).code == "UNJUSTIFIED_WITHDRAWAL"
    before = deepcopy(s)
    audit = deepcopy(c.records)
    assert not event(d, s, c, "dispatch", {"window_index": 0}).accepted
    assert s == before and c.records == audit


def test_sdk_check_uses_frozen_negative_oracle():
    from operatebench.contributions.finsi.governed_campaign_launch.pack import PACK
    from operatebench.sdk import CheckRequest

    result = PACK.check(CheckRequest(FIXTURE, ("unnecessary_approval",)))
    assert result.contract_passed
    assert result.payload["rows"][0]["agent_id"] == "unnecessary_approval"
    assert result.payload["rows"][0]["passed"]
