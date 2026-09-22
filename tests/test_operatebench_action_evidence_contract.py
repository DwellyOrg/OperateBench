"""Canonical per-action evidence obligations (Phase 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from operatebench.core.read_contract import (
    COMPLETE_OUTCOME_KEY,
    ActionEvidenceContract,
    PayloadRegistryKey,
    QuoteForSelectedApproval,
    WorkEvidenceForInvoice,
    action_evidence_contract_problem,
)
from operatebench.domains.lettings.maintenance.evaluator import evaluate
from operatebench.domains.lettings.maintenance.retrieval import (
    MAINTENANCE_RETRIEVAL_TOOLS,
    maintenance_retrieval_catalogue,
)
from operatebench.domains.lettings.maintenance.spec import (
    MAINTENANCE_ACTION_EVIDENCE_CONTRACT,
    MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
    action_schema_view,
    load_spec,
)
from operatebench.runner import run_episode

# Deliberately handwritten: a second derivation catches accidental additions and
# omissions in the canonical declaration rather than restating it mechanically.
EXPECTED = {
    "authorise_supplier_work": (
        PayloadRegistryKey("list_checkpoints", "approvals", "approval_checkpoint_id"),
    ),
    "request_invoice_validation": (
        WorkEvidenceForInvoice(
            "list_billing",
            "invoices",
            "invoice_id",
            "get_case_record",
            "cycles",
            "list_authoritative_records",
            "authoritative_records",
        ),
    ),
    "request_payment": (
        PayloadRegistryKey("list_billing", "invoices", "invoice_id"),
        WorkEvidenceForInvoice(
            "list_billing",
            "invoices",
            "invoice_id",
            "get_case_record",
            "cycles",
            "list_authoritative_records",
            "authoritative_records",
        ),
    ),
    "request_exception_resolution": (
        PayloadRegistryKey("list_checkpoints", "approvals", "approval_checkpoint_id"),
        QuoteForSelectedApproval(
            "list_checkpoints",
            "approvals",
            "approval_checkpoint_id",
            "list_quotes",
            "quotes",
        ),
    ),
}


def _problem(contract: ActionEvidenceContract) -> str | None:
    return action_evidence_contract_problem(
        contract,
        catalogue=maintenance_retrieval_catalogue(),
        retrieval_tools=MAINTENANCE_RETRIEVAL_TOOLS,
        payload_schemas=MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
        outcome_keys=(*MAINTENANCE_ACTION_PAYLOAD_SCHEMAS, COMPLETE_OUTCOME_KEY),
    )


def test_contract_has_exactly_the_four_handwritten_obligations() -> None:
    assert isinstance(MAINTENANCE_ACTION_EVIDENCE_CONTRACT, ActionEvidenceContract)
    assert {
        key: MAINTENANCE_ACTION_EVIDENCE_CONTRACT.evidence_for(key)
        for key in MAINTENANCE_ACTION_EVIDENCE_CONTRACT.outcome_keys()
        if MAINTENANCE_ACTION_EVIDENCE_CONTRACT.evidence_for(key)
    } == EXPECTED


def test_contract_covers_every_action_and_complete() -> None:
    assert set(MAINTENANCE_ACTION_EVIDENCE_CONTRACT.outcome_keys()) == {
        *MAINTENANCE_ACTION_PAYLOAD_SCHEMAS,
        COMPLETE_OUTCOME_KEY,
    }
    assert _problem(MAINTENANCE_ACTION_EVIDENCE_CONTRACT) is None


def test_contract_and_selectors_are_immutable() -> None:
    with pytest.raises((FrozenInstanceError, TypeError)):
        MAINTENANCE_ACTION_EVIDENCE_CONTRACT.outcomes["new"] = None  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        EXPECTED["authorise_supplier_work"][0].root = "quotes"  # type: ignore[misc]


def test_projection_is_detached_static_and_contains_no_resolved_ids() -> None:
    view = action_schema_view()
    for outcome, schema in view.items():
        assert schema["evidence_refs"] == (
            MAINTENANCE_ACTION_EVIDENCE_CONTRACT.evidence_schema_part(outcome)
        )
    projected = view["request_exception_resolution"]["evidence_refs"]
    rendered = repr(projected)
    assert "checkpoint_1" not in rendered
    assert "quote_1:v1" not in rendered
    assert projected["required"] == {
        "approval_checkpoint_id": {
            "from": "list_checkpoints.approvals",
            "select": "key == payload.approval_checkpoint_id",
        },
        "quote_record_key": {
            "from": "list_quotes.quotes",
            "select": (
                "key == approval.quote_id + ':v' + approval.quote_version where "
                "approval=list_checkpoints.approvals[payload.approval_checkpoint_id]"
            ),
        },
    }
    projected["required"].clear()
    assert (
        MAINTENANCE_ACTION_EVIDENCE_CONTRACT.evidence_for("request_exception_resolution")
        == EXPECTED["request_exception_resolution"]
    )


@pytest.mark.parametrize(
    "contract,needle",
    [
        (ActionEvidenceContract({"bogus": {}}), "outcome"),
        (
            ActionEvidenceContract(
                {"request_payment": {"reads": {"read_moons": "required"}}}
            ),
            "read_moons",
        ),
        (
            ActionEvidenceContract(
                {
                    "request_payment": {
                        "evidence_refs": (
                            PayloadRegistryKey("list_billing", "moons", "invoice_id"),
                        )
                    }
                }
            ),
            "moons",
        ),
        (
            ActionEvidenceContract(
                {
                    "request_payment": {
                        "evidence_refs": (
                            PayloadRegistryKey("list_billing", "invoices", "moon_id"),
                        )
                    }
                }
            ),
            "moon_id",
        ),
        (
            ActionEvidenceContract(
                {
                    "request_payment": {
                        "evidence_refs": (("arbitrary", "field", "walker"),)
                    }
                }
            ),
            "unknown evidence selector",
        ),
    ],
)
def test_closed_contract_fails_unknown_descriptors(
    contract: ActionEvidenceContract, needle: str
) -> None:
    problem = _problem(contract)
    assert problem is not None and needle in problem


def test_linked_selector_fails_closed_when_target_is_missing() -> None:
    broken = ActionEvidenceContract(
        {
            "request_payment": {
                "evidence_refs": (
                    WorkEvidenceForInvoice(
                        "list_billing",
                        "invoices",
                        "invoice_id",
                        "get_case_record",
                        "missing_cycles",
                        "list_authoritative_records",
                        "authoritative_records",
                    ),
                )
            }
        }
    )
    assert "missing_cycles" in (_problem(broken) or "")


def test_linked_selector_fails_closed_when_target_is_ambiguous() -> None:
    selector = WorkEvidenceForInvoice(
        "list_billing",
        "invoices",
        "invoice_id",
        "get_case_record",
        "cycles",
        "list_authoritative_records",
        "authoritative_records",
    )
    broken = ActionEvidenceContract(
        {"request_payment": {"evidence_refs": (selector, selector)}}
    )
    assert "duplicate" in (_problem(broken) or "")


def test_distinct_selectors_with_the_same_projected_key_fail_closed() -> None:
    broken = ActionEvidenceContract(
        {
            "request_payment": {
                "evidence_refs": (
                    PayloadRegistryKey("list_billing", "invoices", "invoice_id"),
                    PayloadRegistryKey("list_billing", "payment", "invoice_id"),
                )
            }
        }
    )

    problem = _problem(broken)
    assert problem is not None
    assert "request_payment" in problem
    assert "invoice_id" in problem
    assert "projection" in problem


def test_malformed_selector_projection_fails_closed_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = PayloadRegistryKey("list_billing", "invoices", "invoice_id")
    broken = ActionEvidenceContract({"request_payment": {"evidence_refs": (selector,)}})

    def malformed_projection(self: PayloadRegistryKey) -> object:
        return ("invoice_id",)

    monkeypatch.setattr(PayloadRegistryKey, "projection", malformed_projection)
    problem = _problem(broken)
    assert problem is not None
    assert "request_payment" in problem
    assert "projection" in problem


@pytest.mark.parametrize(
    "outcome,selector,needle",
    [
        (
            "request_payment",
            PayloadRegistryKey("read_moons", "invoices", "invoice_id"),
            "read_moons",
        ),
        (
            "request_payment",
            PayloadRegistryKey("list_quotes", "invoices", "invoice_id"),
            "ownership",
        ),
        (
            "request_exception_resolution",
            QuoteForSelectedApproval(
                "list_checkpoints",
                "approvals",
                "approval_checkpoint_id",
                "list_billing",
                "quotes",
            ),
            "ownership",
        ),
        (
            "request_payment",
            WorkEvidenceForInvoice(
                "list_billing",
                "invoices",
                "invoice_id",
                "get_case_record",
                "cycles",
                "list_quotes",
                "authoritative_records",
            ),
            "ownership",
        ),
    ],
)
def test_selector_tool_and_root_relationships_fail_closed(
    outcome: str, selector: object, needle: str
) -> None:
    broken = ActionEvidenceContract({outcome: {"evidence_refs": (selector,)}})
    assert needle in (_problem(broken) or "")


def test_unknown_obligation_field_fails_closed() -> None:
    broken = ActionEvidenceContract(
        {"request_payment": {"evidence_refs": (), "moon_policy": "required"}}
    )
    assert "moon_policy" in (_problem(broken) or "")


def test_evaluator_independently_flags_a_missing_exact_citation() -> None:
    spec = load_spec("examples/operatebench/maintenance_v0_1.yaml")
    run = run_episode(spec, "V1", "reference")
    forged = []
    changed = False
    for row in run.outcome.trajectory:
        body = dict(row)
        if (
            not changed
            and body.get("record_type") == "action_proposed"
            and MAINTENANCE_ACTION_EVIDENCE_CONTRACT.evidence_for(
                str(body.get("action_type"))
            )
        ):
            body["evidence_refs"] = []
            changed = True
        forged.append(body)
    assert changed
    result = evaluate(
        status=run.outcome.status,
        replay_final=run.outcome.replay_final,
        simulated_minutes=run.outcome.simulated_minutes,
        invocations=run.outcome.invocations,
        final_state=run.outcome.final_state,
        trajectory=forged,
        events=run.outcome.events,
        scenario=spec.scenario("V1"),
        replay_ok=None,
    )
    discipline = next(d for d in result.dimensions if d.name == "retrieval_discipline")
    assert "REQUIRED_EVIDENCE_REF_NOT_CITED" in {
        finding.code for finding in discipline.findings
    }
