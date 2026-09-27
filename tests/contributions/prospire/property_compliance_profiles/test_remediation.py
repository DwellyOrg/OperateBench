"""Independent corruption probes retain real Core provenance and subclass controls."""

from copy import deepcopy
from dataclasses import replace

import pytest

from operatebench.contributions.prospire.property_compliance_profiles.operation import (
    ComplianceDomain,
)

from .test_pack import FIXTURE, load_spec, run_case


class QuietCorruption(ComplianceDomain):
    mode = "early"

    def apply_action(self, state, action_type, payload, evidence_refs, context):
        mode = self.mode

        class Proxy:
            @property
            def now(self):
                return context.now

            def record(self, *args, **kwargs):
                return context.record(*args, **kwargs)

            def schedule_timer(
                self, timer_id, event_type, actor_id, delay_minutes, payload
            ):
                if event_type == "quiet_elapsed":
                    payload = dict(payload)
                    if mode == "early":
                        delay_minutes = 1
                    elif mode == "cause":
                        payload["provisional_at"] = "2026-09-26T08:00:00Z"
                    elif mode == "generation":
                        payload["generation"] += 1
                return context.schedule_timer(
                    timer_id, event_type, actor_id, delay_minutes, payload
                )

        return super().apply_action(state, action_type, payload, evidence_refs, Proxy())


@pytest.mark.parametrize("mode", ["early", "cause", "generation", "valid"])
def test_quiet_contract_runtime_and_independent_grader(mode):
    class Domain(QuietCorruption):
        pass

    Domain.mode = mode
    ep, grade = run_case("EN_reopen", domain=Domain(load_spec(FIXTURE), "EN_reopen"))
    if mode == "valid":
        assert grade["reliable"], grade
        assert ep.terminal_outcome == "VERIFIED_CURRENT"
    else:
        assert not grade["reliable"]
        assert "INVALID_QUIET_CONTRACT" in {f["code"] for f in grade["findings"]}
        assert ep.terminal_outcome != "VERIFIED_CURRENT"


@pytest.mark.parametrize("mutation", ["findings", "hash", "replacement", "valid"])
def test_reports_bound_to_authored_fixture(mutation):
    class Domain(ComplianceDomain):
        def build_plan(self):
            plan = super().build_plan()
            events = []
            for event in plan.events:
                if event.event_type == (
                    "report_replaced" if mutation == "replacement" else "reports_received"
                ):
                    payload = deepcopy(event.payload)
                    if mutation in ("findings", "replacement"):
                        payload["reports"][0]["findings"] = []
                    elif mutation == "hash":
                        payload["reports"][0]["content_hash"] = "0" * 64
                    event = replace(event, payload=payload)
                events.append(event)
            return replace(plan, events=tuple(events))

    scenario = "EN_reopen" if mutation == "replacement" else "EN_remediation"
    _, grade = run_case(scenario, domain=Domain(load_spec(FIXTURE), scenario))
    if mutation == "valid":
        assert grade["reliable"], grade
    else:
        assert not grade["reliable"]
        assert "AUTHORED_REPORT_MISMATCH" in {f["code"] for f in grade["findings"]}


@pytest.mark.parametrize("field", ["verifications", "deliveries", "valid"])
def test_final_evidence_reconciled(field):
    class Domain(ComplianceDomain):
        def canonical_state(self, state):
            result = super().canonical_state(state)
            if result["final"] and field != "valid":
                result[field] = {}
            return result

    _, grade = run_case(
        "EN_remediation", domain=Domain(load_spec(FIXTURE), "EN_remediation")
    )
    if field == "valid":
        assert grade["reliable"], grade
    else:
        assert not grade["reliable"]
        assert "CAUSAL_STATE_MISMATCH" in {f["code"] for f in grade["findings"]}


@pytest.mark.parametrize("field", ["qualified", "standards_met", "valid"])
def test_unsafe_verification_never_sends_or_completes(field):
    class Domain(ComplianceDomain):
        def reduce_event(self, state, event, context):
            if event.event_type == "verification_received" and field != "valid":
                payload = deepcopy(dict(event.payload))
                payload[field] = False
                event = replace(event, payload=payload)
            return super().reduce_event(state, event, context)

    ep, grade = run_case(
        "EN_remediation", domain=Domain(load_spec(FIXTURE), "EN_remediation")
    )
    if field == "valid":
        assert grade["reliable"], grade
        assert ep.final_state["deliveries"]
    else:
        assert ep.terminal_outcome != "VERIFIED_CURRENT"
        assert not ep.final_state["deliveries"]
        assert not ep.final_state["verifications"]
        assert not grade["reliable"]


@pytest.mark.parametrize("field", ["qualified", "standards_met"])
def test_persisted_unsafe_confirmation_cannot_bypass_runtime_gates(field):
    from types import SimpleNamespace

    domain = ComplianceDomain(load_spec(FIXTURE), "EN_remediation")
    ep, _ = run_case("EN_remediation")
    state = deepcopy(ep.final_state)
    state["final"] = False
    state["verifications"]["installation-v1"][field] = False
    state["deliveries"] = {}
    context = SimpleNamespace(now="2026-09-26T10:30:00Z", record=lambda *args: None)
    assert domain._ready(state, context.now) == "VERIFICATION"
    domain.apply_action(
        state,
        "send_bundle",
        {"document_id": "installation-v1", "recipient": "tenant"},
        ["installation-v1"],
        context,
    )
    assert not state["deliveries"]
    domain.apply_terminal(state, [], context)
    assert not state["final"]


def test_independent_quiet_audit_even_when_reducer_forces_quiet():
    class Domain(QuietCorruption):
        def reduce_event(self, state, event, context):
            result = super().reduce_event(state, event, context)
            if event.event_type == "quiet_elapsed":
                state["quiet"] = True
            return result

    ep, grade = run_case("EN_reopen", domain=Domain(load_spec(FIXTURE), "EN_reopen"))
    assert ep.terminal_outcome == "VERIFIED_CURRENT"
    assert not grade["reliable"]
    assert "INVALID_QUIET_CONTRACT" in {f["code"] for f in grade["findings"]}
