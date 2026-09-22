# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Matched communication and evidence-origin counterexamples."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from operatebench.core.outcomes import Act
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.evaluator import _obligations, _recovery
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.runner import run_episode

FIXTURE = Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"


@pytest.mark.parametrize("message", ["msg_approval_reminder", "msg_transfer_notice"])
@pytest.mark.parametrize(
    "field,value",
    [("recipient_actor_id", "supplier_1"), ("correlation_id", "unrelated_cycle")],
)
def test_wrong_obligation_message_is_not_reliable(message, field, value):
    class Agent(RetrievingReferenceAgent):
        def decide(self, observation):
            decision = super().decide(observation)
            if (
                isinstance(decision, Act)
                and decision.action_type == "send_message"
                and decision.payload.get("message_fixture_id") == message
            ):
                return replace(decision, payload={**decision.payload, field: value})
            return decision

    run = run_episode(load_spec(FIXTURE), "V3", "reference", agent_factory=Agent)
    assert not run.reliable
    assert any(
        row.get("code") == "OBLIGATION_MESSAGE_BINDING_MISMATCH"
        for row in run.outcome.trajectory
    )


@pytest.mark.parametrize("message", ["msg_approval_reminder", "msg_transfer_notice"])
@pytest.mark.parametrize(
    "mutation",
    [
        "recipient",
        "cycle",
        "no_acceptance",
        "accepted_cycle",
        "accepted_time",
        "no_delivery",
        "early_delivery",
        "no_context",
    ],
)
def test_independent_message_causality(message, mutation):
    run = run_episode(load_spec(FIXTURE), "V3", "reference")
    rows = deepcopy(list(run.outcome.trajectory))
    state = deepcopy(dict(run.outcome.final_state))
    proposal = next(
        r
        for r in rows
        if r.get("record_type") == "action_proposed"
        and r.get("payload", {}).get("message_fixture_id") == message
    )
    accepted = next(
        r
        for r in rows
        if r.get("record_type") == "effect_accepted"
        and r.get("proposal_id") == proposal["proposal_id"]
    )
    delivered = next(
        r
        for r in rows
        if r.get("record_type") == "side_effect"
        and r.get("message_fixture_id") == message
    )
    if mutation in {"recipient", "cycle"}:
        field = "recipient_actor_id" if mutation == "recipient" else "correlation_id"
        proposal["payload"][field] = "unrelated"
        for communication in state["communications"]:
            if communication["message_fixture_id"] == message:
                communication[field] = "unrelated"
        if mutation == "recipient":
            delivered[field] = "unrelated"
    elif mutation == "no_acceptance":
        rows.remove(accepted)
    elif mutation == "accepted_cycle":
        accepted["cycle_id"] = "unrelated"
    elif mutation == "accepted_time":
        accepted["at"] = "2031-03-01T00:00:00Z"
    elif mutation == "no_delivery":
        rows.remove(delivered)
    elif mutation == "early_delivery":
        rows.remove(delivered)
        rows.insert(rows.index(proposal), delivered)
    else:
        rows[:] = [
            r
            for r in rows
            if r.get("record_type")
            != (
                "checkpoint_opened"
                if message == "msg_approval_reminder"
                else "checkpoint_resolved"
            )
        ]
    result = _obligations(rows, state, run.outcome.status)
    assert not result.ok
    assert "OBLIGATION_MESSAGE_NOT_ESTABLISHED" in {f.code for f in result.findings}


def test_failed_transfer_delivery_preserves_ownership_but_fails_reliability():
    run = run_episode(load_spec(FIXTURE), "V2", "reference")
    assert run.outcome.status == "transferred_to_human_ownership"
    assert run.outcome.final_state["ownership_transferred_to"] == "operator_1"
    assert not run.reliable
    assert not _recovery(run.outcome.trajectory, run.outcome.final_state).ok
    assert "REQUIRED_NOTIFICATION_UNDELIVERED" in run.evaluation.finding_codes


def test_failed_reminder_is_not_discharged():
    spec = load_spec(FIXTURE)
    scenarios = dict(spec.scenarios)
    scenarios["V3"] = replace(
        scenarios["V3"], dispatch_failures=frozenset({"msg_approval_reminder"})
    )
    spec = replace(spec, scenarios=scenarios)
    spec = replace(spec, spec_digest_sha256=spec.compute_digest())
    run = run_episode(spec, "V3", "reference")
    assert not any(
        r.get("record_type") == "obligation_discharged"
        and r.get("kind") == "approval_reminder"
        for r in run.outcome.trajectory
    )
    assert not run.reliable


def test_payment_evidence_origin_is_the_only_changed_precondition():
    originals = []

    class Agent(RetrievingReferenceAgent):
        changed = False

        def decide(self, observation):
            decision = super().decide(observation)
            if (
                isinstance(decision, Act)
                and decision.action_type == "request_payment"
                and not self.changed
            ):
                self.changed = True
                originals.append(decision)
                return replace(decision, evidence_refs=("invoice_1", "assertion_1"))
            return decision

    spec = load_spec(FIXTURE)
    control = run_episode(spec, "V1", "reference")
    negative = run_episode(spec, "V1", "reference", agent_factory=Agent)
    assert control.reliable
    assert len(originals) == 2  # fresh execution and deterministic replay
    assert originals[0] == originals[1]
    payment = next(
        r
        for r in control.outcome.trajectory
        if r.get("record_type") == "action_proposed"
        and r.get("action_type") == "request_payment"
    )
    assert dict(originals[0].payload) == payment["payload"]
    assert originals[0].payload["amount_minor"] == 64000
    proposal = next(
        r
        for r in negative.outcome.trajectory
        if r.get("record_type") == "action_proposed"
        and r.get("action_type") == "request_payment"
    )
    assert proposal["payload"] == payment["payload"]
    assert proposal["evidence_refs"] == ["invoice_1", "assertion_1"]
    rejected = [
        r
        for r in negative.outcome.trajectory
        if r.get("record_type") == "action_rejected"
    ]
    assert [r["code"] for r in rejected] == ["NON_AUTHORITATIVE_EVIDENCE"]
    assert not negative.reliable
    assert negative.outcome.status == "completed_successfully"
    assert set(negative.evaluation.finding_codes) == {
        "CLAIM_TREATED_AS_AUTHORITATIVE",
        "REJECTED_ACTION_PROPOSAL",
        "REQUIRED_EVIDENCE_REF_NOT_CITED",
    }


@pytest.mark.parametrize("variant", ["V1", "V3"])
def test_correct_notification_controls(variant):
    assert run_episode(load_spec(FIXTURE), variant, "reference").reliable


@pytest.mark.parametrize(
    "field,value",
    [
        ("terminal", "completed_successfully"),
        ("failed_dimensions", ("obligations",)),
        ("failed_dimensions", ("recovery", "obligations", "action_validity")),
        ("finding_codes", ("WRONG_REASON",)),
        ("finding_codes", ("REQUIRED_NOTIFICATION_UNDELIVERED",)),
        (
            "finding_codes",
            (
                "REQUIRED_NOTIFICATION_UNDELIVERED",
                "OBLIGATION_MESSAGE_NOT_ESTABLISHED",
                "EXTRA",
            ),
        ),
        ("passing_dimensions", ("terminal_outcome",)),
    ],
)
def test_fault_expectation_mutants_fail_gate(monkeypatch, field, value):
    import operatebench.runner as runner
    from operatebench.domains.lettings.maintenance.agents import REFERENCE_AGENT

    monkeypatch.setattr(
        runner,
        "V2_PERMANENT_NOTIFICATION_FAULT",
        replace(runner.V2_PERMANENT_NOTIFICATION_FAULT, **{field: value}),
    )
    check = runner.check_agent(load_spec(FIXTURE), REFERENCE_AGENT, "V2")
    assert not check.ok
    assert not check.reliable
    assert "fault-control outcome differs" in check.problems[0]


def test_reference_fault_control_is_not_reliable_reference():
    from operatebench.runner import check_maintenance

    report = check_maintenance(load_spec(FIXTURE))
    assert report.ok
    refs = {c.scenario_id: c for c in report.checks if c.kind == "reference"}
    assert refs["V1"].reliable and refs["V3"].reliable
    assert not refs["V2"].reliable
    assert refs["V2"].expected_targets == ("recovery", "obligations")
    assert (
        refs["V2"].reference_expectation_id
        == "maintenance.reference.V2.permanent-notification-fault.v1"
    )
    assert refs["V1"].reference_expectation_id is None
