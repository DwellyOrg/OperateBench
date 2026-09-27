# SPDX-FileCopyrightText: 2026 PROSPIRE TECHNOLOGIES LTD
# SPDX-License-Identifier: Apache-2.0
"""Versioned secondary-channel recovery; no fault suppression or self-attestation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from operatebench.artifact import read_artifact, replay_artifact, write_artifact
from operatebench.core.errors import SpecSchemaError
from operatebench.core.outcomes import Act, Wait
from operatebench.domains.lettings.maintenance.agents import RetrievingReferenceAgent
from operatebench.domains.lettings.maintenance.evaluator import _recovery
from operatebench.domains.lettings.maintenance.operation import MaintenanceOperation
from operatebench.domains.lettings.maintenance.spec import (
    load_spec,
    validate_action_payload,
)
from operatebench.runner import run_episode

LEGACY = Path(__file__).parents[1] / "examples/operatebench/maintenance_v0_1.yaml"


def successor():
    return load_spec(LEGACY.with_name("maintenance_delivery_recovery_v0_7.yaml"))


def test_public_reference_recovers_v2_without_disabling_primary_fault():
    spec = successor()
    assert "msg_transfer_notice" in spec.scenario("V2").dispatch_failures
    run = run_episode(spec, "V2", "reference")
    assert run.outcome.status == "transferred_to_human_ownership"
    assert run.evaluation.reliable, run.evaluation.finding_codes
    notices = [
        r
        for r in run.outcome.final_state["communications"]
        if r["message_fixture_id"] == "msg_transfer_notice"
    ]
    assert [r["dispatch_status"] for r in notices] == ["FAILED", "DELIVERED"]
    assert notices[1]["dispatch_channel"] == "secondary"
    original = run.outcome.final_state["communications"][notices[1]["recovery_of"]]
    assert original == notices[0]
    receipts = [
        r
        for r in run.outcome.trajectory
        if r["record_type"] == "side_effect" and r.get("dispatch_channel") == "secondary"
    ]
    assert len(receipts) == 1
    assert receipts[0]["recovery_of"] == notices[1]["recovery_of"]


def test_successor_strict_artifact_roundtrip(tmp_path):
    spec = successor()
    path = write_artifact(
        run_episode(spec, "V2", "reference"), tmp_path / "recovery.json"
    )
    assert replay_artifact(spec, read_artifact(path)).ok


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "wrong_actor",
        "wrong_index",
        "wrong_channel",
        "original_not_failed",
        "state_only",
    ],
)
def test_independent_oracle_requires_one_bound_environment_receipt(mutation):
    run = run_episode(successor(), "V2", "reference")
    rows = deepcopy(list(run.outcome.trajectory))
    state = deepcopy(run.outcome.final_state)
    receipt = next(
        r
        for r in rows
        if r.get("dispatch_channel") == "secondary" and r["record_type"] == "side_effect"
    )
    if mutation == "missing":
        rows.remove(receipt)
    elif mutation == "duplicate":
        rows.insert(rows.index(receipt), deepcopy(receipt))
    elif mutation == "wrong_actor":
        receipt["recipient_actor_id"] = "supplier_1"
    elif mutation == "wrong_index":
        receipt["recovery_of"] = -1
    elif mutation == "wrong_channel":
        receipt["dispatch_channel"] = "invented"
    elif mutation == "original_not_failed":
        state["communications"][receipt["recovery_of"]]["dispatch_status"] = "DELIVERED"
    else:
        state["communications"][-1]["dispatch_status"] = "FAILED"
    assert not _recovery(rows, state).ok


@pytest.mark.parametrize(
    "change",
    [
        {"dispatch_channel": "secondary", "recovery_of": 0},
        {"dispatch_channel": "primary"},
        {"recovery_of": 1},
        {"recovery_of": True},
        {"committed_decision_preserved": 1},
    ],
    ids=[
        "secondary_wrong_index",
        "channel_only",
        "index_only",
        "bool_index",
        "bool_flag",
    ],
)
def test_original_failure_requires_primary_only_shape(change):
    run = run_episode(successor(), "V2", "reference")
    assert run.evaluation.reliable
    rows = deepcopy(list(run.outcome.trajectory))
    failure = next(
        r
        for r in rows
        if r["record_type"] == "side_effect_failed"
        and r.get("message_fixture_id") == "msg_transfer_notice"
    )
    assert "dispatch_channel" not in failure and "recovery_of" not in failure
    receipt = next(r for r in rows if r.get("dispatch_channel") == "secondary")
    assert receipt["recovery_of"] == 1
    failure.update(change)
    result = _recovery(rows, run.outcome.final_state)
    assert not result.ok
    assert [f.code for f in result.findings] == ["REQUIRED_NOTIFICATION_UNDELIVERED"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycle_id", "unrelated_cycle"),
        ("at", "2031-03-03T09:00:00Z"),
        ("cycle_id", True),
        ("at", True),
        ("cycle_id", None),
        ("at", None),
    ],
    ids=[
        "wrong_cycle",
        "wrong_time",
        "bool_cycle",
        "bool_time",
        "null_cycle",
        "null_time",
    ],
)
def test_original_acceptance_binds_proposal_time_and_correlated_cycle(field, value):
    run = run_episode(successor(), "V2", "reference")
    assert run.evaluation.reliable
    rows = deepcopy(list(run.outcome.trajectory))
    failure_index = next(
        i
        for i, r in enumerate(rows)
        if r["record_type"] == "side_effect_failed"
        and r.get("message_fixture_id") == "msg_transfer_notice"
    )
    proposal = rows[failure_index - 1]
    acceptance = rows[failure_index + 1]
    assert proposal["record_type"] == "action_proposed"
    assert acceptance["record_type"] == "effect_accepted"
    assert acceptance["proposal_id"] == proposal["proposal_id"]
    assert acceptance["cycle_id"] == proposal["payload"]["correlation_id"]
    assert acceptance["at"] == proposal["at"]
    acceptance[field] = value
    result = _recovery(rows, run.outcome.final_state)
    assert not result.ok
    assert [f.code for f in result.findings] == ["REQUIRED_NOTIFICATION_UNDELIVERED"]


class DispatchContext:
    now = "2031-03-03T09:00:00Z"

    def __init__(self):
        self.calls = []
        self.records = []

    def dispatch_fails(self, key):
        self.calls.append(key)
        return key == "msg_transfer_notice"

    def record(self, kind, payload):
        self.records.append((kind, payload))


def failed_intent():
    domain = MaintenanceOperation(successor(), "V2")
    state = domain.initial_state()
    state.current_cycle_id = "cycle_1"
    state.issue_reporting_actor_id = "customer_1"
    state.ownership_transferred_to = "operator_1"
    context = DispatchContext()
    payload = {
        "recipient_actor_id": "customer_1",
        "message_fixture_id": "msg_transfer_notice",
        "correlation_id": "cycle_1",
    }
    assert domain._act_send_message(state, payload, (), context).accepted
    return domain, state, context, dict(payload, recovery_of=0)


@pytest.mark.parametrize(
    "change",
    [
        {"recipient_actor_id": "supplier_1"},
        {"correlation_id": "cycle_2"},
        {"recovery_of": 9},
        {"recovery_of": -1},
        {"recovery_of": True},
    ],
)
def test_bad_recovery_is_nonmutating_and_does_not_dispatch(change):
    domain, state, context, payload = failed_intent()
    before = deepcopy(state.communications)
    assert not domain._act_send_message(
        state, dict(payload, **change), (), context
    ).accepted
    assert state.communications == before
    assert context.calls == ["msg_transfer_notice"]
    assert not domain._terminal_transfer(state, context).accepted


def test_recovery_exactly_once_and_primary_remains_failed():
    domain, state, context, payload = failed_intent()
    assert domain._act_send_message(state, payload, (), context).accepted
    assert not domain._act_send_message(state, payload, (), context).accepted
    assert context.calls == ["msg_transfer_notice", "secondary:msg_transfer_notice"]
    assert state.communications[0]["dispatch_status"] == "FAILED"
    assert len(state.communications) == 2


def test_caller_cannot_declare_receipt():
    _, _, _, payload = failed_intent()
    with pytest.raises(SpecSchemaError):
        validate_action_payload(
            "send_message", dict(payload, dispatch_status="DELIVERED"), "forged"
        )


def test_secondary_fault_is_not_automatic_success():
    spec = successor()
    scenarios = dict(spec.scenarios)
    scenarios["V2"] = replace(
        scenarios["V2"],
        dispatch_failures=frozenset(
            {"msg_transfer_notice", "secondary:msg_transfer_notice"}
        ),
    )
    spec = replace(spec, scenarios=scenarios)
    spec = replace(spec, spec_digest_sha256=spec.compute_digest())
    run = run_episode(spec, "V2", "reference")
    assert not run.evaluation.reliable
    assert run.outcome.status == "operational_horizon_exhausted"
    assert not run.outcome.final_state["transfer_notice_sent"]


@pytest.mark.parametrize(
    "mode", ["no_recovery", "wrong_actor", "wrong_cycle", "forged", "duplicate"]
)
def test_public_only_negative_controls_never_become_reliable(mode):
    class Control(RetrievingReferenceAgent):
        saved = None
        attempted = False

        def decide(self, observation):
            result = super().decide(observation)
            if isinstance(result, Act) and "recovery_of" in result.payload:
                if mode == "no_recovery":
                    return Wait(
                        reason="control omits recovery", fallback_after_minutes=2880
                    )
                payload = dict(result.payload)
                if mode == "wrong_actor":
                    payload["recipient_actor_id"] = "supplier_1"
                elif mode == "wrong_cycle":
                    payload["correlation_id"] = "unrelated_cycle"
                elif mode == "forged":
                    payload["dispatch_status"] = "DELIVERED"
                self.saved = replace(result, payload=payload)
                return self.saved
            if mode == "duplicate" and self.saved is not None and not self.attempted:
                self.attempted = True
                return self.saved
            return result

    run = run_episode(successor(), "V2", "reference", agent_factory=Control)
    assert not run.evaluation.reliable
    notices = [
        r
        for r in run.outcome.final_state["communications"]
        if r["message_fixture_id"] == "msg_transfer_notice"
    ]
    assert sum(r["dispatch_status"] == "DELIVERED" for r in notices) == (
        mode == "duplicate"
    )
    assert "FAILED" in [r["dispatch_status"] for r in notices]


def test_recovery_is_versioned_public_contract_not_fixture_renaming():
    legacy = load_spec(LEGACY)
    spec = successor()
    old_domain = MaintenanceOperation(legacy, "V2")
    new_domain = MaintenanceOperation(spec, "V2")
    assert legacy.message_fixtures == spec.message_fixtures
    assert (
        legacy.scenario("V2").dispatch_failures == spec.scenario("V2").dispatch_failures
    )
    assert legacy.spec_digest_sha256 != spec.spec_digest_sha256
    assert "recovery_of" not in old_domain.action_schemas()["send_message"]["optional"]
    schema = new_domain.action_schemas()["send_message"]
    assert schema["optional"]["recovery_of"] == "integer"
    guidance = schema["field_guidance"]["dispatch_status"]
    for term in (
        "secondary",
        "FAILED",
        "zero-based",
        "recipient_actor_id",
        "correlation_id",
    ):
        assert term in guidance
    _domain, state, context, payload = failed_intent()
    assert not old_domain._act_send_message(state, payload, (), context).accepted


@pytest.mark.parametrize("enabled", [False, True])
def test_schema_publication_and_validator_share_versioned_payload_table(enabled):
    from operatebench.domains.lettings.maintenance.spec import action_payload_schemas

    spec = successor() if enabled else load_spec(LEGACY)
    schemas = MaintenanceOperation(spec, "V2").action_schemas()
    table = action_payload_schemas(recovery_enabled=enabled)
    for action, schema in schemas.items():
        if action == "complete":
            continue
        required, optional = table[action]
        assert schema["required"] == dict(required)
        assert schema["optional"] == dict(optional)
    payload = {
        "recipient_actor_id": "customer_1",
        "message_fixture_id": "msg_transfer_notice",
        "correlation_id": "cycle_1",
        "recovery_of": 0,
    }
    if enabled:
        validate_action_payload(
            "send_message", payload, "successor", recovery_enabled=True
        )
    else:
        with pytest.raises(SpecSchemaError):
            validate_action_payload("send_message", payload, "legacy")
    with pytest.raises(SpecSchemaError):
        validate_action_payload(
            "send_message",
            dict(payload, dispatch_status="DELIVERED"),
            "forgery",
            recovery_enabled=enabled,
        )


def test_secondary_failure_key_is_authored_and_identity_bound():
    from operatebench.domains.lettings.maintenance.spec import OperationSpec

    body = yaml.safe_load(
        LEGACY.with_name("maintenance_delivery_recovery_v0_7.yaml").read_text()
    )
    body["scenarios"]["V2"]["dispatch_failures"].append("secondary:msg_transfer_notice")
    failed = OperationSpec.from_mapping(body)
    assert failed.spec_digest_sha256 != successor().spec_digest_sha256
    assert not run_episode(failed, "V2", "reference").evaluation.reliable
    body["operation_version"] = "0.6.0"
    with pytest.raises(SpecSchemaError):
        OperationSpec.from_mapping(body)
