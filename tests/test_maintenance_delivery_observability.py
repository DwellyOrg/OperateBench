# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Delivery is a fact, not an accepted communication decision."""

from dataclasses import replace
from pathlib import Path

import pytest

from operatebench.core.retrieval import RetrievalRequest
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import load_spec
from operatebench.domains.lettings.maintenance.state import Cycle
from operatebench.runner import run_episode

FIXTURE = Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"


class Context:
    now = "2031-03-03T09:00:00Z"

    def __init__(self, fail):
        self.fail = fail
        self.records = []

    def dispatch_fails(self, fixture):
        return self.fail

    def record(self, kind, payload):
        self.records.append((kind, payload))

    def schedule_timer(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("fixture", ["msg_transfer_notice", "msg_completion_notice"])
def test_failed_dispatch_preserves_attempt_not_delivery(fixture):
    domain = MaintenanceOperation(load_spec(FIXTURE), "V2")
    state = domain.initial_state()
    state.current_cycle_id = "cycle_1"
    state.issue_reporting_actor_id = "customer_1"
    state.ownership_transferred_to = "operator_1"
    state.cycles["cycle_1"] = Cycle(cycle_id="cycle_1", kind="DIAGNOSTIC")
    context = Context(True)
    verdict = domain._act_send_message(
        state,
        {
            "recipient_actor_id": "customer_1",
            "message_fixture_id": fixture,
            "correlation_id": "cycle_1",
        },
        (),
        context,
    )
    assert verdict.accepted
    assert len(state.communications) == 1
    assert not state.transfer_notice_sent
    assert not state.cycles["cycle_1"].completion_notice_sent
    assert any(
        k == "side_effect_failed" and p["committed_decision_preserved"]
        for k, p in context.records
    )
    served = domain.serve_retrieval(
        state, (RetrievalRequest(tool="list_communications"),), context.now
    )
    assert served[0].records["communications"][0]["dispatch_status"] == "FAILED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("recipient_actor_id", "customer_2"),
        ("correlation_id", "unrelated"),
        ("dispatch_status", "FAILED"),
    ],
)
@pytest.mark.parametrize("terminal", ["transfer", "completion"])
def test_terminal_requires_delivered_actor_and_cycle_not_sent_flag(
    field, value, terminal
):
    domain = MaintenanceOperation(load_spec(FIXTURE), "V2")
    state = domain.initial_state()
    state.current_cycle_id = "cycle_1"
    state.issue_reporting_actor_id = "customer_1"
    state.ownership_transferred_to = "operator_1"
    state.transfer_notice_sent = True  # deliberately inconsistent durable assertion
    state.communications = [
        {
            "recipient_actor_id": "customer_1",
            "correlation_id": "cycle_1",
            "message_fixture_id": "msg_transfer_notice",
            "dispatch_status": "DELIVERED",
            "at": Context.now,
            field: value,
        }
    ]
    if terminal == "completion":
        state.communications[0]["message_fixture_id"] = "msg_completion_notice"
        state.cycles["cycle_1"] = Cycle(
            cycle_id="cycle_1",
            kind="DIAGNOSTIC",
            visit_status="WORK_VERIFIED",
            completion_notice_sent=True,
        )
        verdict = domain._terminal_provisional_close(state, Context(False))
    else:
        verdict = domain._terminal_transfer(state, Context(False))
    assert not verdict.accepted
    assert state.terminal is None


def test_reference_v2_cannot_report_completed_transfer_after_failed_delivery():
    run = run_episode(load_spec(FIXTURE), "V2", "reference")
    assert run.outcome.final_state["ownership_transferred_to"] == "operator_1"
    assert run.outcome.status != "transferred_to_human_ownership"
    assert not run.outcome.final_state["transfer_notice_sent"]
    assert "REQUIRED_NOTIFICATION_UNDELIVERED" in run.evaluation.finding_codes


def test_completion_dispatch_failure_cannot_discharge_notice_obligation():
    spec = load_spec(FIXTURE)
    scenarios = dict(spec.scenarios)
    scenarios["V1"] = replace(
        scenarios["V1"], dispatch_failures=frozenset({"msg_completion_notice"})
    )
    spec = replace(spec, scenarios=scenarios)
    spec = replace(spec, spec_digest_sha256=spec.compute_digest())
    run = run_episode(spec, "V1", "reference")
    assert not any(
        r["record_type"] == "obligation_discharged"
        and r.get("kind") == "customer_completion_notice"
        for r in run.outcome.trajectory
    )
    assert run.outcome.status != "completed_successfully"


def test_failed_attempt_then_successful_dispatch_is_explicit_recovery():
    domain = MaintenanceOperation(load_spec(FIXTURE), "V2")
    state = domain.initial_state()
    state.current_cycle_id = "cycle_1"
    state.issue_reporting_actor_id = "customer_1"
    state.ownership_transferred_to = "operator_1"
    payload = {
        "recipient_actor_id": "customer_1",
        "message_fixture_id": "msg_transfer_notice",
        "correlation_id": "cycle_1",
    }
    assert domain._act_send_message(state, payload, (), Context(True)).accepted
    assert not domain._terminal_transfer(state, Context(True)).accepted
    assert domain._act_send_message(state, payload, (), Context(False)).accepted
    assert [r["dispatch_status"] for r in state.communications] == ["FAILED", "DELIVERED"]
    assert domain._terminal_transfer(state, Context(False)).accepted
