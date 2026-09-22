# SPDX-License-Identifier: Apache-2.0
"""Independent contract checks for the specified-only domain-audit budget v1."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs" / "schemas"
PACKAGED = ROOT / "src" / "operatebench" / "resources" / "identity"
PROTOCOL = "operatebench.domain-generated-audit-budget.v1"
INT64_MAX = 2**63 - 1
V1_EVENT_CAPACITY = 99998
EVIDENCE_RECORD_CAPACITY = 100000
FILES = (
    "domain-generated-audit-budget-v1.schema.json",
    "domain-generated-audit-budget-evidence-v1.schema.json",
    "domain-generated-audit-budget-registry-v1.schema.json",
    "domain-generated-audit-budget-registry-v1.json",
    "domain-generated-audit-budget-census-v1.schema.json",
    "domain-generated-audit-budget-census-v1.json",
    "domain-generated-audit-budget-v1.golden.json",
)
EXPECTED_CODES = (
    "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    "DOMAIN_AUDIT_COUNTER_INTEGRITY",
)
EXPECTED_KINDS = (
    "scheduled_audit_event_intent",
    "immediate_audit_event_intent",
)
EXPECTED_EXCLUDED_KINDS = (
    "authored_absolute_event",
    "authored_conditional_event",
    "authored_conditional_activation",
    "authored_conditional_queueing",
    "authored_audit_only_delivery",
    "authored_non_triggering_delivery",
    "authored_rejected_delivery",
    "authored_post_terminal_delivery",
    "generic_business_state_record",
    "generic_evidence_record",
    "checkpoint_lifecycle",
    "obligation_lifecycle",
    "proposal_record",
    "refusal_record",
    "accepted_effect_record",
    "retrieval_record",
    "agent_record",
    "invocation_record",
    "scaffold_record",
    "provider_record",
    "execution_ledger_record",
    "side_effect_record",
    "wait_record",
    "wake_record",
    "existing_timer_lifecycle",
    "finality_cut",
    "untriggered_at_finality",
    "transition_horizon_finding",
    "budget_integrity_finding",
    "observability_finding",
    "terminal_budget_finding",
    "terminal_budget_summary",
)
CONTRACT_REFUSALS = (
    "unsupported_or_wrong_contract",
    "non_exact_type_or_boolean",
    "negative_value",
    "duplicate_or_unauthorized_order",
    "source_or_aggregate_mismatch",
    "v1_capacity_exceeded",
    "unknown_source_or_kind",
    "invalid_batch_or_identity",
    "reservation_lifecycle_violation",
    "post_finality_attempt",
)
COUNTER_REFUSALS = (
    "checked_signed_int64_arithmetic_overflow",
    "malformed_runtime_counter",
)
BUDGET_REFUSALS = (
    "valid_reservation_above_declared_aggregate",
    "valid_reservation_above_declared_source",
)
EXPECTED_REFUSAL_MAPPING = tuple(
    [(name, EXPECTED_CODES[1]) for name in CONTRACT_REFUSALS]
    + [(name, EXPECTED_CODES[2]) for name in COUNTER_REFUSALS]
    + [(name, EXPECTED_CODES[0]) for name in BUDGET_REFUSALS]
)
EXPECTED_REFUSAL_FIELD_MATRIX = (
    {
        "refusal_class": "unsupported_or_wrong_contract",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "non_exact_type_or_boolean",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "negative_value",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "duplicate_or_unauthorized_order",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "source_or_aggregate_mismatch",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "v1_capacity_exceeded",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "pre_identity",
        "source_id_rule": "no_evidence",
        "requested_count_rule": "no_evidence",
        "transition_id_rule": "no_evidence",
        "invocation_index_rule": "no_evidence",
        "pre_operation_count_rule": "no_evidence",
        "pre_source_count_rule": "no_evidence",
        "declared_max_source": "eligibility_candidate_only_no_evidence",
        "finding_required": False,
        "finding_integrity_code": None,
    },
    {
        "refusal_class": "unknown_source_or_kind",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "assigned_transition",
        "source_id_rule": "exact_attempted_source",
        "requested_count_rule": "positive_exact_attempted_batch_size",
        "transition_id_rule": "exact_attempted_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": (
            "exact_authoritative_source_committed_before_or_null_if_unknown"
        ),
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
    {
        "refusal_class": "invalid_batch_or_identity",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "assigned_transition",
        "source_id_rule": "exact_attempted_source",
        "requested_count_rule": "zero",
        "transition_id_rule": "exact_attempted_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "exact_authoritative_source_committed_before",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
    {
        "refusal_class": "reservation_lifecycle_violation",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "assigned_transition",
        "source_id_rule": "exact_reservation_source",
        "requested_count_rule": "positive_exact_reservation_size",
        "transition_id_rule": "exact_reservation_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "exact_authoritative_source_committed_before",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
    {
        "refusal_class": "post_finality_attempt",
        "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "stage": "assigned_preexecution",
        "source_id_rule": "null",
        "requested_count_rule": "zero",
        "transition_id_rule": "null",
        "invocation_index_rule": "null",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "null",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
    {
        "refusal_class": "checked_signed_int64_arithmetic_overflow",
        "integrity_code": "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "stage": "assigned_transition",
        "source_id_rule": "exact_attempted_source",
        "requested_count_rule": "positive_exact_attempted_batch_size",
        "transition_id_rule": "exact_attempted_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "exact_authoritative_source_committed_before",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_COUNTER_INTEGRITY",
    },
    {
        "refusal_class": "malformed_runtime_counter",
        "integrity_code": "DOMAIN_AUDIT_COUNTER_INTEGRITY",
        "stage": "assigned_preexecution",
        "source_id_rule": "null",
        "requested_count_rule": "zero",
        "transition_id_rule": "null",
        "invocation_index_rule": "null",
        "pre_operation_count_rule": "zero_untrusted_counter",
        "pre_source_count_rule": "null",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_COUNTER_INTEGRITY",
    },
    {
        "refusal_class": "valid_reservation_above_declared_aggregate",
        "integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "stage": "assigned_transition",
        "source_id_rule": "exact_attempted_source",
        "requested_count_rule": "positive_exact_attempted_batch_size",
        "transition_id_rule": "exact_attempted_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "exact_authoritative_source_committed_before",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    },
    {
        "refusal_class": "valid_reservation_above_declared_source",
        "integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "stage": "assigned_transition",
        "source_id_rule": "exact_attempted_source",
        "requested_count_rule": "positive_exact_attempted_batch_size",
        "transition_id_rule": "exact_attempted_transition",
        "invocation_index_rule": "exact_current_or_null_when_not_invocation_scoped",
        "pre_operation_count_rule": "exact_authoritative_committed_before",
        "pre_source_count_rule": "exact_authoritative_source_committed_before",
        "declared_max_source": "accepted_card_value",
        "finding_required": True,
        "finding_integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    },
)
EXPECTED_ASSIGNED_DECLARATION_MISMATCH = {
    "stage": "assigned_preexecution",
    "refusal_class": "source_or_aggregate_mismatch",
    "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    "source_id": None,
    "requested_count": 0,
    "transition_id": None,
    "invocation_index": None,
    "operation_committed_before": 0,
    "source_committed_before": None,
    "declared_max_source": "accepted_card_value",
    "committed_intents": 0,
    "finding_count": 1,
    "summary_count": 1,
    "summary_result": "ERROR",
}
EXPECTED_ASSIGNED_OVER_CAPACITY_DEFECT = {
    "stage": "assigned_preexecution",
    "refusal_class": "v1_capacity_exceeded",
    "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    "source_id": None,
    "requested_count": 0,
    "transition_id": None,
    "invocation_index": None,
    "operation_committed_before": 0,
    "source_committed_before": None,
    "declared_max_source": "accepted_card_value",
    "committed_intents": 0,
    "finding_count": 1,
    "summary_count": 1,
    "summary_result": "ERROR",
}
EXPECTED_MALFORMED_CASES = {
    "boolean_declared_max": {
        "base": "exact_contract",
        "target": "declaration",
        "expected_layer": "schema",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "replace", "path": "/declared_max_events", "value": True}],
    },
    "v1_capacity_exceeded": {
        "base": "exact_contract",
        "target": "declaration",
        "expected_layer": "schema",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {"op": "replace", "path": "/declared_max_events", "value": 99999},
            {
                "op": "replace",
                "path": "/sources/0/max_accepted_occurrences",
                "value": 99999,
            },
            {"op": "replace", "path": "/sources/0/max_events_per_transition", "value": 1},
            {"op": "remove", "path": "/sources/1"},
        ],
    },
    "duplicate_source_id": {
        "base": "exact_contract",
        "target": "declaration",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {
                "op": "replace",
                "path": "/sources/1/source_id",
                "value": "maintenance.approval.audit",
            }
        ],
    },
    "unauthorized_source_order": {
        "base": "exact_contract",
        "target": "declaration",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "swap", "path": "/sources/0", "from": "/sources/1"}],
    },
    "aggregate_mismatch": {
        "base": "exact_contract",
        "target": "declaration",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "replace", "path": "/declared_max_events", "value": 2}],
    },
    "source_kind_mismatch": {
        "base": "exact_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {
                "op": "replace",
                "path": "/records/0/counted_kind",
                "value": "immediate_audit_event_intent",
            }
        ],
    },
    "duplicate_batch_ordinal": {
        "base": "exact_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "replace", "path": "/records/1/batch_ordinal", "value": 0}],
    },
    "finding_before_commit": {
        "base": "one_over_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "move", "path": "/records/0", "from": "/records/3"}],
    },
    "zero_budget_request": {
        "base": "one_over_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {"op": "replace", "path": "/records/3/requested_count", "value": 0}
        ],
    },
    "finding_declared_limit_mismatch": {
        "base": "one_over_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {"op": "replace", "path": "/records/3/declared_max_events", "value": 2}
        ],
    },
    "source_capacity_exceeded": {
        "base": "exact_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {
                "op": "replace",
                "path": "/records/2/source_id",
                "value": "maintenance.approval.audit",
            },
            {
                "op": "replace",
                "path": "/records/2/counted_kind",
                "value": "scheduled_audit_event_intent",
            },
        ],
    },
    "commit_after_finding": {
        "base": "one_over_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [{"op": "move", "path": "/records/4", "from": "/records/0"}],
    },
    "duplicate_intent_id": {
        "base": "exact_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {
                "op": "replace",
                "path": "/records/1/intent_id",
                "value": "maintenance.approval.reminder.1",
            }
        ],
    },
    "summary_count_mismatch": {
        "base": "exact_evidence",
        "target": "evidence",
        "expected_layer": "semantic",
        "expected_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        "operations": [
            {"op": "replace", "path": "/records/3/committed_count", "value": 2}
        ],
    },
}
EXPECTED_SEMANTIC_CASES = {
    "delivery_existing_identity": {
        "operations": ["delivery_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "commit_new_identity": {
        "operations": ["commit_new_identity"],
        "committed_delta": 1,
        "expected_integrity_code": None,
    },
    "cancel_existing_identity": {
        "operations": ["cancel_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "fire_existing_identity": {
        "operations": ["fire_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "disposition_existing_identity": {
        "operations": ["disposition_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "resolution_existing_identity": {
        "operations": ["resolution_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "normalization_existing_identity": {
        "operations": ["normalization_existing_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "move_same_unconsumed_identity": {
        "operations": ["move_same_unconsumed_identity"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "proposal_refused": {
        "operations": ["proposal_refused"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "transport_retry": {
        "operations": ["transport_retry"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "reducer_refused": {
        "operations": ["reducer_refused"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "failed_uncommitted_transition": {
        "operations": ["failed_uncommitted_transition"],
        "committed_delta": 0,
        "expected_integrity_code": None,
    },
    "retry_committed_same_identity": {
        "operations": ["retry_committed_same_identity"],
        "committed_delta": 0,
        "expected_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
    "cancel_and_replace_new_identity": {
        "operations": ["cancel_and_replace_new_identity"],
        "committed_delta": 1,
        "expected_integrity_code": None,
    },
    "rearm_new_identity": {
        "operations": ["rearm_new_identity"],
        "committed_delta": 1,
        "expected_integrity_code": None,
    },
    "reschedule_new_identity": {
        "operations": ["reschedule_new_identity"],
        "committed_delta": 1,
        "expected_integrity_code": None,
    },
    "reopen_new_identity": {
        "operations": ["reopen_new_identity"],
        "committed_delta": 1,
        "expected_integrity_code": None,
    },
    "post_finality_reservation": {
        "operations": ["post_finality_reservation"],
        "committed_delta": 0,
        "expected_integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
    },
}
EXPECTED_SEMANTIC_OPERATIONS = {
    "delivery_existing_identity": ("delivery_existing_identity",),
    "commit_new_identity": ("commit_new_identity",),
    "cancel_existing_identity": ("cancel_existing_identity",),
    "fire_existing_identity": ("fire_existing_identity",),
    "disposition_existing_identity": ("disposition_existing_identity",),
    "resolution_existing_identity": ("resolution_existing_identity",),
    "normalization_existing_identity": ("normalization_existing_identity",),
    "move_same_unconsumed_identity": ("move_same_unconsumed_identity",),
    "proposal_refused": ("proposal_refused",),
    "transport_retry": ("transport_retry",),
    "reducer_refused": ("reducer_refused",),
    "failed_uncommitted_transition": ("failed_uncommitted_transition",),
    "retry_committed_same_identity": ("retry_committed_same_identity",),
    "cancel_and_replace_new_identity": ("cancel_and_replace_new_identity",),
    "rearm_new_identity": ("rearm_new_identity",),
    "reschedule_new_identity": ("reschedule_new_identity",),
    "reopen_new_identity": ("reopen_new_identity",),
    "post_finality_reservation": ("post_finality_reservation",),
}
EXPECTED_TRANSITION_TRACES = {
    "delivery_existing_identity": ("disposition:existing-identity:disposed:delivery",),
    "commit_new_identity": (
        "reserve_attempt:new-identity-1",
        "reserve_commit:new-identity-1",
    ),
    "cancel_existing_identity": ("disposition:existing-identity:cancelled:cancel",),
    "fire_existing_identity": ("disposition:existing-identity:fired:fire",),
    "disposition_existing_identity": (
        "disposition:existing-identity:disposed:disposition",
    ),
    "resolution_existing_identity": (
        "disposition:existing-identity:resolved:resolution",
    ),
    "normalization_existing_identity": (
        "disposition:existing-identity:disposed:normalization",
    ),
    "move_same_unconsumed_identity": ("move:existing-identity",),
    "proposal_refused": ("uncommitted_refusal:proposal",),
    "transport_retry": ("uncommitted_refusal:transport_retry",),
    "reducer_refused": ("uncommitted_refusal:reducer",),
    "failed_uncommitted_transition": ("uncommitted_refusal:failed_transition",),
    "retry_committed_same_identity": (
        "reserve_attempt:existing-identity",
        "reserve_refused_duplicate:existing-identity",
    ),
    "cancel_and_replace_new_identity": (
        "disposition:existing-identity:cancelled:cancel",
        "reserve_attempt:new-identity-1",
        "reserve_commit:new-identity-1",
    ),
    "rearm_new_identity": (
        "disposition:existing-identity:fired:fire",
        "reserve_attempt:new-identity-1",
        "reserve_commit:new-identity-1",
    ),
    "reschedule_new_identity": (
        "disposition:existing-identity:cancelled:reschedule",
        "reserve_attempt:new-identity-1",
        "reserve_commit:new-identity-1",
    ),
    "reopen_new_identity": (
        "disposition:existing-identity:resolved:resolution",
        "reserve_attempt:new-identity-1",
        "reserve_commit:new-identity-1",
    ),
    "post_finality_reservation": (
        "enter_finality",
        "reserve_attempt:new-identity-1",
        "reserve_refused_finality:new-identity-1",
    ),
}
EXPECTED_FINAL_EXISTING_STATUS = {
    "delivery_existing_identity": "disposed",
    "commit_new_identity": "active",
    "cancel_existing_identity": "cancelled",
    "fire_existing_identity": "fired",
    "disposition_existing_identity": "disposed",
    "resolution_existing_identity": "resolved",
    "normalization_existing_identity": "disposed",
    "move_same_unconsumed_identity": "active",
    "proposal_refused": "active",
    "transport_retry": "active",
    "reducer_refused": "active",
    "failed_uncommitted_transition": "active",
    "retry_committed_same_identity": "active",
    "cancel_and_replace_new_identity": "cancelled",
    "rearm_new_identity": "fired",
    "reschedule_new_identity": "cancelled",
    "reopen_new_identity": "resolved",
    "post_finality_reservation": "active",
}
SUCCESSFUL_NEW_IDENTITY_CASES = frozenset(
    {
        "commit_new_identity",
        "cancel_and_replace_new_identity",
        "rearm_new_identity",
        "reschedule_new_identity",
        "reopen_new_identity",
    }
)
EXPECTED_OWNERS = (
    "operation_core",
    "core_runtime",
    "card",
    "semantic_scenario",
    "variant",
    "scaffold",
    "provider",
    "evaluator",
    "no_time",
)
EXPECTED_DEPENDENCIES = (
    "contract_resources_v1",
    "operation_source_declaration_and_literal_maxima",
    "core_reservation_counter_and_integrity_runtime",
    "episode_outcome_successor_with_integrity_error",
    "executed_runtime_evidence_successor",
    "runtime_identity_adapter_additive_registry",
    "evaluator_and_measurement_alignment",
    "separately_authorized_no_time_implementation",
)
EXPECTED_CENSUS = (
    (
        "/declaration/declared_max_events",
        "operation_core",
        "core_runtime_preflight",
        "authored_maximum",
    ),
    (
        "/declaration/sources",
        "operation_core",
        "core_runtime_preflight",
        "source_semantics",
    ),
    (
        "/protocol/counted_kinds",
        "core_runtime",
        "core_reservation_api",
        "closed_taxonomy",
    ),
    (
        "/protocol/excluded_kinds",
        "core_runtime",
        "core_reservation_api",
        "closed_taxonomy",
    ),
    (
        "/runtime/operation_counter",
        "core_runtime",
        "core_reservation_api",
        "authoritative_counter",
    ),
    (
        "/runtime/source_counters",
        "core_runtime",
        "core_reservation_api",
        "authoritative_counter",
    ),
    (
        "/runtime/transition_reservation",
        "core_runtime",
        "core_transition_commit",
        "transient_capability",
    ),
    (
        "/evidence/committed_intents",
        "core_runtime",
        "future_evidence_verifier",
        "derived_evidence",
    ),
    (
        "/evidence/integrity_finding",
        "core_runtime",
        "future_outcome_successor",
        "bounded_refusal",
    ),
    (
        "/evidence/terminal_summary",
        "core_runtime",
        "future_evidence_verifier",
        "derived_evidence",
    ),
    (
        "/card/declared_max_domain_generated_audit_events",
        "operation_core",
        "card_v2_projector",
        "authored_maximum",
    ),
    (
        "/no_time/domain_generated_audit_term",
        "operation_core",
        "future_no_time_verifier",
        "specified_only_consumer",
    ),
)
EXPECTED_VECTOR_PINS = {
    "zero_contract": (
        282,
        "888b0133abe6917317c7639bd19ede18aa505c42fcefd317a2181177723c6fb0",
    ),
    "exact_contract": (
        573,
        "6560fe842f0ae8b88dab9aa65750db4c1f09edd7cda742df40deb853e9b4f1cf",
    ),
    "zero_evidence": (
        491,
        "35d9a930e3666e2db356ed03583a6d6d7d482bec6bb2e94f3d5b92eaf173fdab",
    ),
    "exact_evidence": (
        1695,
        "08a2a42abb5f22d2f12eefbbcf072fdfb221f9bd82b1cc88fa9ba6569b6c8e19",
    ),
    "one_over_evidence": (
        2039,
        "8e0d741d78933437167d1c63e83dd254a9aa0ea969ed34a9fed6bbd1481c7f87",
    ),
    "partial_two_unit_refusal_evidence": (
        1669,
        "0afd64fbb343247d214a9359b1ba507ac7c2ce088b8341b6622e5ddaedea68ef",
    ),
}

ZERO_CONTRACT = {
    "schema": "operatebench.domain_generated_audit_budget.v1",
    "schema_version": 1,
    "contract_id": PROTOCOL,
    "operation_core_content_digest_sha256": "1" * 64,
    "declared_max_events": 0,
    "sources": [],
}
EXACT_CONTRACT = {
    "schema": "operatebench.domain_generated_audit_budget.v1",
    "schema_version": 1,
    "contract_id": PROTOCOL,
    "operation_core_content_digest_sha256": "2" * 64,
    "declared_max_events": 3,
    "sources": [
        {
            "source_id": "maintenance.approval.audit",
            "counted_kind": "scheduled_audit_event_intent",
            "max_accepted_occurrences": 1,
            "max_events_per_transition": 2,
        },
        {
            "source_id": "maintenance.note.audit",
            "counted_kind": "immediate_audit_event_intent",
            "max_accepted_occurrences": 1,
            "max_events_per_transition": 1,
        },
    ],
}
ZERO_EVIDENCE = {
    "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
    "schema_version": 1,
    "contract_id": PROTOCOL,
    "operation_core_content_digest_sha256": "1" * 64,
    "operation_instance_id": "opinst_zero",
    "declared_max_events": 0,
    "records": [
        {
            "record_type": "terminal_summary",
            "contract_id": PROTOCOL,
            "declared_max_events": 0,
            "committed_count": 0,
            "result": "SUCCESS",
            "integrity_code": None,
        }
    ],
}
EXACT_EVIDENCE = {
    "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
    "schema_version": 1,
    "contract_id": PROTOCOL,
    "operation_core_content_digest_sha256": "2" * 64,
    "operation_instance_id": "opinst_exact",
    "declared_max_events": 3,
    "records": [
        {
            "record_type": "committed_intent",
            "contract_id": PROTOCOL,
            "intent_digest_sha256": "3" * 64,
            "intent_id": "maintenance.approval.reminder.1",
            "source_id": "maintenance.approval.audit",
            "counted_kind": "scheduled_audit_event_intent",
            "transition_id": "transition.approval.1",
            "batch_ordinal": 0,
            "operation_committed_count": 1,
        },
        {
            "record_type": "committed_intent",
            "contract_id": PROTOCOL,
            "intent_digest_sha256": "4" * 64,
            "intent_id": "maintenance.approval.deadline.1",
            "source_id": "maintenance.approval.audit",
            "counted_kind": "scheduled_audit_event_intent",
            "transition_id": "transition.approval.1",
            "batch_ordinal": 1,
            "operation_committed_count": 2,
        },
        {
            "record_type": "committed_intent",
            "contract_id": PROTOCOL,
            "intent_digest_sha256": "5" * 64,
            "intent_id": "maintenance.note.1",
            "source_id": "maintenance.note.audit",
            "counted_kind": "immediate_audit_event_intent",
            "transition_id": "transition.note.1",
            "batch_ordinal": 0,
            "operation_committed_count": 3,
        },
        {
            "record_type": "terminal_summary",
            "contract_id": PROTOCOL,
            "declared_max_events": 3,
            "committed_count": 3,
            "result": "SUCCESS",
            "integrity_code": None,
        },
    ],
}
ONE_OVER_EVIDENCE = copy.deepcopy(EXACT_EVIDENCE)
ONE_OVER_EVIDENCE["operation_instance_id"] = "opinst_one_over"
ONE_OVER_EVIDENCE["records"].insert(
    -1,
    {
        "record_type": "integrity_finding",
        "contract_id": PROTOCOL,
        "integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "refusal_class": "valid_reservation_above_declared_aggregate",
        "source_id": "maintenance.note.audit",
        "requested_count": 1,
        "operation_committed_before": 3,
        "source_committed_before": 1,
        "declared_max_events": 3,
        "transition_id": "transition.note.2",
        "invocation_index": None,
    },
)
ONE_OVER_EVIDENCE["records"][-1]["integrity_code"] = "DOMAIN_AUDIT_BUDGET_EXCEEDED"
ONE_OVER_EVIDENCE["records"][-1]["result"] = "ERROR"
PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE = copy.deepcopy(EXACT_EVIDENCE)
PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE["operation_instance_id"] = "opinst_atomic_refusal"
PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE["records"] = [
    *PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE["records"][:2],
    {
        "record_type": "integrity_finding",
        "contract_id": PROTOCOL,
        "integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
        "refusal_class": "valid_reservation_above_declared_source",
        "source_id": "maintenance.approval.audit",
        "requested_count": 2,
        "operation_committed_before": 2,
        "source_committed_before": 2,
        "declared_max_events": 3,
        "transition_id": "transition.approval.2",
        "invocation_index": None,
    },
    {
        "record_type": "terminal_summary",
        "contract_id": PROTOCOL,
        "declared_max_events": 3,
        "committed_count": 2,
        "result": "ERROR",
        "integrity_code": "DOMAIN_AUDIT_BUDGET_EXCEEDED",
    },
]
ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE = {
    "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
    "schema_version": 1,
    "contract_id": "operatebench.domain-generated-audit-budget.v1",
    "operation_core_content_digest_sha256": (
        "6666666666666666666666666666666666666666666666666666666666666666"
    ),
    "operation_instance_id": "opinst_assigned_over_capacity_defect",
    "declared_max_events": 99999,
    "records": [
        {
            "record_type": "integrity_finding",
            "contract_id": "operatebench.domain-generated-audit-budget.v1",
            "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
            "refusal_class": "v1_capacity_exceeded",
            "source_id": None,
            "requested_count": 0,
            "operation_committed_before": 0,
            "source_committed_before": None,
            "declared_max_events": 99999,
            "transition_id": None,
            "invocation_index": None,
        },
        {
            "record_type": "terminal_summary",
            "contract_id": "operatebench.domain-generated-audit-budget.v1",
            "declared_max_events": 99999,
            "committed_count": 0,
            "result": "ERROR",
            "integrity_code": "DOMAIN_AUDIT_CONTRACT_VIOLATION",
        },
    ],
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _patterns(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pattern":
                found.append(child)
            found.extend(_patterns(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_patterns(child))
    return found


def _pointer_parent(value: Any, pointer: str) -> tuple[Any, str]:
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]
    ]
    parent = value
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    return parent, parts[-1]


def _pointer_get(value: Any, pointer: str) -> Any:
    parent, key = _pointer_parent(value, pointer)
    return parent[int(key)] if isinstance(parent, list) else parent[key]


def _apply_mutation(value: Any, operation: dict[str, Any]) -> None:
    parent, key = _pointer_parent(value, operation["path"])
    index: int | str = int(key) if isinstance(parent, list) else key
    if operation["op"] == "replace":
        parent[index] = copy.deepcopy(operation["value"])
    elif operation["op"] == "remove":
        if isinstance(parent, list):
            parent.pop(index)
        else:
            del parent[index]
    elif operation["op"] == "swap":
        other_parent, other_key = _pointer_parent(value, operation["from"])
        other_index: int | str = (
            int(other_key) if isinstance(other_parent, list) else other_key
        )
        parent[index], other_parent[other_index] = (
            other_parent[other_index],
            parent[index],
        )
    else:
        assert operation["op"] == "move"
        moved = copy.deepcopy(_pointer_get(value, operation["from"]))
        from_parent, from_key = _pointer_parent(value, operation["from"])
        if isinstance(from_parent, list):
            from_parent.pop(int(from_key))
        else:
            del from_parent[from_key]
        parent, key = _pointer_parent(value, operation["path"])
        if isinstance(parent, list):
            parent.insert(int(key), moved)
        else:
            parent[key] = moved


@dataclass
class _LifecycleState:
    """Minimal executable protocol model, independent of golden result literals."""

    known_identities: set[str] = dataclass_field(
        default_factory=lambda: {"existing-identity"}
    )
    committed_identities: set[str] = dataclass_field(
        default_factory=lambda: {"existing-identity"}
    )
    statuses: dict[str, str] = dataclass_field(
        default_factory=lambda: {"existing-identity": "active"}
    )
    committed_count: int = 1
    finality: bool = False
    integrity_code: str | None = None
    attempted_reservations: list[str] = dataclass_field(default_factory=list)
    trace: list[str] = dataclass_field(default_factory=list)
    _next_identity_number: int = dataclass_field(default=1, init=False, repr=False)
    initial_known_identities: frozenset[str] = dataclass_field(
        default=frozenset({"existing-identity"}), init=False
    )
    initial_committed_identities: frozenset[str] = dataclass_field(
        default=frozenset({"existing-identity"}), init=False
    )
    initial_committed_count: int = dataclass_field(default=1, init=False)

    @property
    def existing_identity(self) -> str:
        return "existing-identity"

    def _assert_invariants(self) -> None:
        assert self.existing_identity == "existing-identity"
        assert self.existing_identity in self.known_identities
        assert self.committed_identities <= self.known_identities
        assert set(self.statuses) == self.known_identities
        assert self.committed_count == len(self.committed_identities)
        assert set(self.statuses.values()) <= {
            "active",
            "disposed",
            "cancelled",
            "fired",
            "resolved",
        }

    def _candidate_new_identity(self) -> str:
        return f"new-identity-{self._next_identity_number}"

    def _protocol_snapshot(self) -> tuple[object, ...]:
        return (
            frozenset(self.known_identities),
            frozenset(self.committed_identities),
            tuple(sorted(self.statuses.items())),
            self.committed_count,
            self.finality,
            self._next_identity_number,
        )

    def dispose_existing(self, status: str, reason: str) -> None:
        assert self.integrity_code is None
        assert self.statuses[self.existing_identity] == "active"
        assert status in {"disposed", "cancelled", "fired", "resolved"}
        self.statuses[self.existing_identity] = status
        self.trace.append(f"disposition:{self.existing_identity}:{status}:{reason}")
        self._assert_invariants()

    def refuse_uncommitted(self, reason: str) -> None:
        assert self.integrity_code is None
        assert self.statuses[self.existing_identity] == "active"
        self.trace.append(f"uncommitted_refusal:{reason}")
        self._assert_invariants()

    def move_existing(self) -> None:
        assert self.integrity_code is None
        assert self.statuses[self.existing_identity] == "active"
        self.trace.append(f"move:{self.existing_identity}")
        self._assert_invariants()

    def reserve(self, identity: str) -> None:
        assert self.integrity_code is None
        before = self._protocol_snapshot()
        self.attempted_reservations.append(identity)
        self.trace.append(f"reserve_attempt:{identity}")
        refusal: str | None = None
        if self.finality:
            refusal = "finality"
        elif identity in self.committed_identities or identity in self.known_identities:
            refusal = "duplicate"
        if refusal is not None:
            self.trace.append(f"reserve_refused_{refusal}:{identity}")
            self.integrity_code = "DOMAIN_AUDIT_CONTRACT_VIOLATION"
            assert self._protocol_snapshot() == before
            self._assert_invariants()
            return
        assert identity == self._candidate_new_identity()
        self.known_identities.add(identity)
        self.committed_identities.add(identity)
        self.statuses[identity] = "active"
        self.committed_count += 1
        self._next_identity_number += 1
        self.trace.append(f"reserve_commit:{identity}")
        self._assert_invariants()

    def reserve_new(self) -> None:
        self.reserve(self._candidate_new_identity())

    def enter_finality(self) -> None:
        assert self.integrity_code is None
        assert not self.finality
        self.finality = True
        self.trace.append("enter_finality")
        self._assert_invariants()


def _execute_semantic_operations(operations: list[str]) -> _LifecycleState:
    state = _LifecycleState()
    state._assert_invariants()
    for operation in operations:
        if state.integrity_code is not None:
            continue
        if operation == "delivery_existing_identity":
            state.dispose_existing("disposed", "delivery")
        elif operation == "commit_new_identity":
            state.reserve_new()
        elif operation == "cancel_existing_identity":
            state.dispose_existing("cancelled", "cancel")
        elif operation == "fire_existing_identity":
            state.dispose_existing("fired", "fire")
        elif operation == "disposition_existing_identity":
            state.dispose_existing("disposed", "disposition")
        elif operation == "resolution_existing_identity":
            state.dispose_existing("resolved", "resolution")
        elif operation == "normalization_existing_identity":
            state.dispose_existing("disposed", "normalization")
        elif operation == "move_same_unconsumed_identity":
            state.move_existing()
        elif operation == "proposal_refused":
            state.refuse_uncommitted("proposal")
        elif operation == "transport_retry":
            state.refuse_uncommitted("transport_retry")
        elif operation == "reducer_refused":
            state.refuse_uncommitted("reducer")
        elif operation == "failed_uncommitted_transition":
            state.refuse_uncommitted("failed_transition")
        elif operation == "retry_committed_same_identity":
            state.reserve(state.existing_identity)
        elif operation == "cancel_and_replace_new_identity":
            state.dispose_existing("cancelled", "cancel")
            state.reserve_new()
        elif operation == "rearm_new_identity":
            state.dispose_existing("fired", "fire")
            state.reserve_new()
        elif operation == "reschedule_new_identity":
            state.dispose_existing("cancelled", "reschedule")
            state.reserve_new()
        elif operation == "reopen_new_identity":
            state.dispose_existing("resolved", "resolution")
            state.reserve_new()
        elif operation == "post_finality_reservation":
            state.enter_finality()
            state.reserve_new()
        else:
            raise AssertionError(f"unknown semantic operation: {operation}")
    state._assert_invariants()
    return state


def _assert_operation_deltas_match_execution(operation_deltas: Any) -> None:
    """Bind every closed golden delta row to one independently executed operation."""
    assert type(operation_deltas) is dict
    assert set(operation_deltas) == set(EXPECTED_SEMANTIC_OPERATIONS)
    assert len(operation_deltas) == 18
    for operation in EXPECTED_SEMANTIC_OPERATIONS:
        row = operation_deltas[operation]
        assert type(row) is dict
        assert set(row) == {"committed_delta", "integrity_code"}
        assert type(row["committed_delta"]) is int
        assert row["integrity_code"] is None or type(row["integrity_code"]) is str

        # The operation name selects the independent dispatcher path. No golden
        # value supplies setup, control flow, or an expected transition result.
        state = _execute_semantic_operations([operation])
        executed_row = {
            "committed_delta": state.committed_count - state.initial_committed_count,
            "integrity_code": state.integrity_code,
        }
        assert row == executed_row


def _execute_all_golden_mutations(golden: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(golden["mutation_operation_schema"])
    operation_validator = Draft202012Validator(golden["mutation_operation_schema"])
    declaration_validator = Draft202012Validator(_load(DOCS / FILES[0]))
    evidence_validator = Draft202012Validator(_load(DOCS / FILES[1]))
    bases = {**golden["declaration_vectors"], **golden["evidence_vectors"]}
    for case in golden["malformed_cases"]:
        candidate = copy.deepcopy(bases[case["base"]])
        for operation in case["operations"]:
            operation_validator.validate(operation)
            _apply_mutation(candidate, operation)
        if case["expected_layer"] == "schema":
            with pytest.raises(ValidationError):
                declaration_validator.validate(candidate)
        elif case["target"] == "declaration":
            declaration_validator.validate(candidate)
            with pytest.raises(AssertionError):
                _checked_contract(
                    candidate,
                    tuple(source["source_id"] for source in EXACT_CONTRACT["sources"]),
                )
        else:
            evidence_validator.validate(candidate)
            with pytest.raises(AssertionError):
                _checked_evidence(candidate, EXACT_CONTRACT)
        assert case["expected_code"] == EXPECTED_CODES[1]


def _checked_contract(
    candidate: dict[str, Any], authorized_source_ids: tuple[str, ...] | None = None
) -> None:
    """Independent declaration oracle; literals do not come from the schemas/registry."""
    assert type(candidate["declared_max_events"]) is int
    assert 0 <= candidate["declared_max_events"] <= V1_EVENT_CAPACITY
    total = 0
    source_ids: list[str] = []
    for source in candidate["sources"]:
        assert source["source_id"] not in source_ids
        source_ids.append(source["source_id"])
        occurrences = source["max_accepted_occurrences"]
        fanout = source["max_events_per_transition"]
        assert type(occurrences) is int and type(fanout) is int
        assert 0 <= occurrences <= V1_EVENT_CAPACITY
        assert 1 <= fanout <= V1_EVENT_CAPACITY
        product = occurrences * fanout
        assert product <= INT64_MAX
        total += product
        assert total <= V1_EVENT_CAPACITY
    assert total == candidate["declared_max_events"]
    if authorized_source_ids is not None:
        assert tuple(source_ids) == authorized_source_ids


def _checked_evidence(
    candidate: dict[str, Any],
    declaration: dict[str, Any] | None,
    *,
    accepted_card_declared_max: int | None = None,
) -> None:
    """Independent class-complete executable specification for cross-record semantics."""
    assert type(candidate["declared_max_events"]) is int
    assert 0 <= candidate["declared_max_events"] <= INT64_MAX
    assert candidate["contract_id"] == PROTOCOL
    if declaration is None:
        assert accepted_card_declared_max is not None
        assert type(accepted_card_declared_max) is int
        assert 0 <= accepted_card_declared_max <= INT64_MAX
        assert candidate["declared_max_events"] == accepted_card_declared_max
        sources: dict[str, dict[str, Any]] = {}
    else:
        assert candidate["contract_id"] == declaration["contract_id"]
        assert (
            candidate["operation_core_content_digest_sha256"]
            == declaration["operation_core_content_digest_sha256"]
        )
        assert candidate["declared_max_events"] == declaration["declared_max_events"]
        sources = {source["source_id"]: source for source in declaration["sources"]}
    assert 1 <= len(candidate["records"]) <= EVIDENCE_RECORD_CAPACITY
    records = candidate["records"]
    assert records[-1]["record_type"] == "terminal_summary"
    assert sum(r["record_type"] == "terminal_summary" for r in records) == 1
    findings = [r for r in records if r["record_type"] == "integrity_finding"]
    assert len(findings) <= 1
    committed: list[dict[str, Any]] = []
    phase = "commits"
    batches: dict[tuple[str, str], list[int]] = {}
    source_counts = dict.fromkeys(sources, 0)
    for record in records:
        kind = record["record_type"]
        if kind == "committed_intent":
            assert phase == "commits"
            committed.append(record)
            source = sources.get(record["source_id"])
            assert source is not None
            assert record["counted_kind"] == source["counted_kind"]
            key = (record["transition_id"], record["source_id"])
            batches.setdefault(key, []).append(record["batch_ordinal"])
            source_counts[record["source_id"]] += 1
        elif kind == "integrity_finding":
            assert phase == "commits"
            phase = "finding"
        else:
            assert kind == "terminal_summary" and phase in {"commits", "finding"}
            phase = "summary"
    assert len({r["intent_id"] for r in committed}) == len(committed)
    assert len({r["intent_digest_sha256"] for r in committed}) == len(committed)
    assert [r["operation_committed_count"] for r in committed] == list(
        range(1, len(committed) + 1)
    )
    for (transition_id, source_id), ordinals in batches.items():
        source = sources[source_id]
        assert ordinals == list(range(len(ordinals))), transition_id
        assert len(ordinals) <= source["max_events_per_transition"]
    for source_id, source in sources.items():
        distinct_batches = sum(key[1] == source_id for key in batches)
        assert distinct_batches <= source["max_accepted_occurrences"]
        assert (
            source_counts[source_id]
            <= source["max_accepted_occurrences"] * source["max_events_per_transition"]
        )
    summary = records[-1]
    assert summary["committed_count"] == len(committed)
    assert summary["declared_max_events"] == candidate["declared_max_events"]
    if candidate["declared_max_events"] > V1_EVENT_CAPACITY:
        assert not committed
    else:
        assert len(committed) <= candidate["declared_max_events"]
    if not findings:
        assert candidate["declared_max_events"] <= V1_EVENT_CAPACITY
        assert summary["result"] == "SUCCESS"
        assert summary["integrity_code"] is None
        return
    finding = findings[0]
    assert summary["result"] == "ERROR"
    assert summary["integrity_code"] == finding["integrity_code"]
    assert finding["declared_max_events"] == candidate["declared_max_events"]
    assert type(finding["requested_count"]) is int
    refusal_class = finding["refusal_class"]
    matrix = {row["refusal_class"]: row for row in EXPECTED_REFUSAL_FIELD_MATRIX}
    rule = matrix[refusal_class]
    if rule["stage"] == "pre_identity":
        assert refusal_class in {"source_or_aggregate_mismatch", "v1_capacity_exceeded"}
        expected_defect = (
            EXPECTED_ASSIGNED_OVER_CAPACITY_DEFECT
            if refusal_class == "v1_capacity_exceeded"
            else EXPECTED_ASSIGNED_DECLARATION_MISMATCH
        )
        assert len(committed) == expected_defect["committed_intents"] == 0
        assert finding["source_id"] is expected_defect["source_id"]
        assert finding["requested_count"] == expected_defect["requested_count"]
        assert finding["transition_id"] is expected_defect["transition_id"]
        assert finding["invocation_index"] is expected_defect["invocation_index"]
        assert finding["operation_committed_before"] == 0
        assert finding["source_committed_before"] is None
        assert finding["integrity_code"] == expected_defect["integrity_code"]
        if refusal_class == "v1_capacity_exceeded":
            assert candidate["declared_max_events"] > V1_EVENT_CAPACITY
        return
    assert finding["integrity_code"] == rule["finding_integrity_code"]
    if rule["source_id_rule"] == "null":
        assert finding["source_id"] is None
    else:
        assert type(finding["source_id"]) is str and finding["source_id"]
    if rule["requested_count_rule"] == "zero":
        assert finding["requested_count"] == 0
    else:
        assert finding["requested_count"] >= 1
    if rule["transition_id_rule"] == "null":
        assert finding["transition_id"] is None
    else:
        assert type(finding["transition_id"]) is str and finding["transition_id"]
        assert finding["transition_id"] not in {r["transition_id"] for r in committed}
    if rule["invocation_index_rule"] == "null":
        assert finding["invocation_index"] is None
    else:
        assert (
            finding["invocation_index"] is None
            or type(finding["invocation_index"]) is int
        )
    expected_operation_before = (
        0
        if rule["pre_operation_count_rule"] == "zero_untrusted_counter"
        else len(committed)
    )
    assert finding["operation_committed_before"] == expected_operation_before
    if rule["pre_source_count_rule"] == "null":
        assert finding["source_committed_before"] is None
    elif finding["source_id"] in source_counts:
        assert finding["source_committed_before"] == source_counts[finding["source_id"]]
    else:
        assert rule["pre_source_count_rule"].endswith("_or_null_if_unknown")
        assert finding["source_committed_before"] is None
    if finding["integrity_code"] == EXPECTED_CODES[0]:
        assert finding["source_id"] in sources
        aggregate_over = (
            len(committed) + finding["requested_count"] > candidate["declared_max_events"]
        )
        source = sources[finding["source_id"]]
        source_over = (
            source_counts[finding["source_id"]] + finding["requested_count"]
            > source["max_accepted_occurrences"] * source["max_events_per_transition"]
        )
        assert aggregate_over or source_over
        assert aggregate_over if refusal_class == BUDGET_REFUSALS[0] else source_over


def test_resources_exist_meta_validate_validate_and_match_packaged_bytes() -> None:
    for name in FILES:
        assert (DOCS / name).is_file(), name
        assert (DOCS / name).read_bytes() == (PACKAGED / name).read_bytes(), name
    for name in FILES:
        if name.endswith(".schema.json"):
            Draft202012Validator.check_schema(_load(DOCS / name))
    Draft202012Validator(_load(DOCS / FILES[2])).validate(_load(DOCS / FILES[3]))
    Draft202012Validator(_load(DOCS / FILES[4])).validate(_load(DOCS / FILES[5]))


def test_positive_contract_and_evidence_vectors_pass_shape_and_semantics() -> None:
    contract_validator = Draft202012Validator(_load(DOCS / FILES[0]))
    evidence_validator = Draft202012Validator(_load(DOCS / FILES[1]))
    for contract in (ZERO_CONTRACT, EXACT_CONTRACT):
        contract_validator.validate(contract)
        _checked_contract(contract)
    for evidence, declaration in (
        (ZERO_EVIDENCE, ZERO_CONTRACT),
        (EXACT_EVIDENCE, EXACT_CONTRACT),
        (ONE_OVER_EVIDENCE, EXACT_CONTRACT),
        (PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE, EXACT_CONTRACT),
        (ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE, None),
    ):
        evidence_validator.validate(evidence)
        _checked_evidence(
            evidence,
            declaration,
            accepted_card_declared_max=(99999 if declaration is None else None),
        )


def test_registry_freezes_exact_literals_cardinalities_and_dependency_order() -> None:
    registry = _load(DOCS / FILES[3])
    assert registry["protocol_id"] == PROTOCOL
    assert registry["evidence_record_capacity"] == EVIDENCE_RECORD_CAPACITY
    assert registry["v1_declared_event_capacity"] == V1_EVENT_CAPACITY
    assert tuple(registry["counted_kinds"]) == EXPECTED_KINDS
    assert tuple(registry["excluded_kinds"]) == EXPECTED_EXCLUDED_KINDS
    assert tuple(registry["integrity_codes"]) == EXPECTED_CODES
    assert (
        tuple(
            (row["refusal_class"], row["integrity_code"])
            for row in registry["refusal_mapping"]
        )
        == EXPECTED_REFUSAL_MAPPING
    )
    assert tuple(registry["refusal_mapping"]) == EXPECTED_REFUSAL_FIELD_MATRIX
    assert (
        registry["assigned_accepted_declaration_mismatch"]
        == EXPECTED_ASSIGNED_DECLARATION_MISMATCH
    )
    assert (
        registry["assigned_over_capacity_defect"]
        == EXPECTED_ASSIGNED_OVER_CAPACITY_DEFECT
    )
    assert tuple(registry["owners"]) == EXPECTED_OWNERS
    assert tuple(registry["dependency_order"]) == EXPECTED_DEPENDENCIES
    assert registry["runtime_status"] == "unimplemented"
    assert registry["no_time_status"] == "specified_only_unauthorized"
    assert registry["maintenance_source_declaration_status"] == "unresolved"
    assert registry["episode_outcome_successor_status"] == "hard_dependency_unimplemented"
    assert (
        registry["executed_evidence_successor_status"] == "hard_dependency_unimplemented"
    )
    assert registry["card_v2_schema_change"] is False
    assert registry["runtime_identity_adapter_v1_change"] is False


def test_registry_schema_uses_exact_tuples_and_projection_key_sets() -> None:
    registry_schema = _load(DOCS / FILES[2])
    expected_arrays = {
        "counted_kinds": EXPECTED_KINDS,
        "excluded_kinds": EXPECTED_EXCLUDED_KINDS,
        "integrity_codes": EXPECTED_CODES,
        "owners": EXPECTED_OWNERS,
        "dependency_order": EXPECTED_DEPENDENCIES,
    }
    for field, expected in expected_arrays.items():
        shape = registry_schema["properties"][field]
        assert shape["minItems"] == shape["maxItems"] == len(expected)
        assert shape["items"] is False
        assert tuple(item["const"] for item in shape["prefixItems"]) == expected
    refusal_shape = registry_schema["properties"]["refusal_mapping"]
    assert (
        refusal_shape["minItems"]
        == refusal_shape["maxItems"]
        == len(EXPECTED_REFUSAL_MAPPING)
    )
    assert refusal_shape["items"] is False
    assert (
        tuple(
            (
                item["allOf"][1]["properties"]["refusal_class"]["const"],
                item["allOf"][1]["properties"]["integrity_code"]["const"],
            )
            for item in refusal_shape["prefixItems"]
        )
        == EXPECTED_REFUSAL_MAPPING
    )
    declaration_schema = _load(DOCS / FILES[0])
    assert (
        declaration_schema["properties"]["declared_max_events"]["maximum"]
        == V1_EVENT_CAPACITY
    )
    source_properties = declaration_schema["properties"]["sources"]["items"]["properties"]
    assert source_properties["max_accepted_occurrences"]["maximum"] == V1_EVENT_CAPACITY
    assert source_properties["max_events_per_transition"]["maximum"] == V1_EVENT_CAPACITY
    evidence_schema = _load(DOCS / FILES[1])
    assert (
        evidence_schema["properties"]["records"]["maxItems"] == EVIDENCE_RECORD_CAPACITY
    )
    diagnostic_declared_max_shapes = (
        evidence_schema["properties"]["declared_max_events"],
        evidence_schema["$defs"]["integrityFinding"]["properties"]["declared_max_events"],
        evidence_schema["$defs"]["terminalSummary"]["properties"]["declared_max_events"],
        evidence_schema["$defs"]["evidenceDigestProjection"]["properties"][
            "declared_max_events"
        ],
    )
    assert all(
        shape == {"type": "integer", "minimum": 0, "maximum": INT64_MAX}
        for shape in diagnostic_declared_max_shapes
    )
    assert set(evidence_schema["$defs"]["intentDigestProjection"]["required"]) == {
        "contract_id",
        "operation_core_content_digest_sha256",
        "operation_instance_id",
        "intent_id",
        "source_id",
        "transition_id",
        "batch_ordinal",
    }
    assert set(evidence_schema["$defs"]["evidenceDigestProjection"]["required"]) == {
        "contract_id",
        "operation_core_content_digest_sha256",
        "operation_instance_id",
        "declared_max_events",
        "records",
    }


def test_census_is_an_exact_independent_owner_consumer_tuple_surface() -> None:
    census = _load(DOCS / FILES[5])
    actual = tuple(
        (row["field_path"], row["owner"], row["consumer"], row["role"])
        for row in census["entries"]
    )
    assert actual == EXPECTED_CENSUS
    assert len(actual) == len(set(actual)) == 12


def test_schema_rejects_bool_ranges_wrong_literals_kinds_codes_and_suffixes() -> None:
    contract_validator = Draft202012Validator(_load(DOCS / FILES[0]))
    evidence_validator = Draft202012Validator(_load(DOCS / FILES[1]))
    mutations: list[tuple[Draft202012Validator, dict[str, Any]]] = []
    for field, value in (
        ("declared_max_events", True),
        ("declared_max_events", -1),
        ("declared_max_events", INT64_MAX + 1),
    ):
        candidate = copy.deepcopy(ZERO_CONTRACT)
        candidate[field] = value
        mutations.append((contract_validator, candidate))
    for field, value in (
        ("contract_id", PROTOCOL + "\n"),
        ("operation_core_content_digest_sha256", "1" * 64 + "\n"),
    ):
        candidate = copy.deepcopy(ZERO_CONTRACT)
        candidate[field] = value
        mutations.append((contract_validator, candidate))
    candidate = copy.deepcopy(EXACT_CONTRACT)
    candidate["sources"][0]["counted_kind"] = "timer"
    mutations.append((contract_validator, candidate))
    for field, value in (
        ("max_accepted_occurrences", True),
        ("max_accepted_occurrences", INT64_MAX + 1),
        ("max_events_per_transition", False),
        ("max_events_per_transition", 0),
        ("max_events_per_transition", INT64_MAX + 1),
    ):
        candidate = copy.deepcopy(EXACT_CONTRACT)
        candidate["sources"][0][field] = value
        mutations.append((contract_validator, candidate))
    candidate = copy.deepcopy(EXACT_EVIDENCE)
    candidate["records"][-1]["integrity_code"] = "WRONG"
    mutations.append((evidence_validator, candidate))
    for validator, candidate in mutations:
        with pytest.raises(ValidationError):
            validator.validate(candidate)


def test_assigned_over_capacity_diagnostic_is_representable_but_strictly_bounded() -> (
    None
):
    validator = Draft202012Validator(_load(DOCS / FILES[1]))
    validator.validate(ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE)
    _checked_evidence(
        ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE, None, accepted_card_declared_max=99999
    )
    for bad in (True, -1, "99999", None, INT64_MAX + 1):
        candidate = copy.deepcopy(ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE)
        candidate["declared_max_events"] = bad
        candidate["records"][0]["declared_max_events"] = bad
        candidate["records"][1]["declared_max_events"] = bad
        with pytest.raises(ValidationError):
            validator.validate(candidate)
    for mutation in ("success", "budget", "committed"):
        candidate = copy.deepcopy(ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE)
        if mutation == "success":
            candidate["records"] = [copy.deepcopy(candidate["records"][-1])]
            candidate["records"][0].update(result="SUCCESS", integrity_code=None)
        elif mutation == "budget":
            candidate["records"][0].update(
                integrity_code=EXPECTED_CODES[0], refusal_class=BUDGET_REFUSALS[0]
            )
            candidate["records"][-1]["integrity_code"] = EXPECTED_CODES[0]
        else:
            candidate["records"].insert(0, copy.deepcopy(EXACT_EVIDENCE["records"][0]))
        with pytest.raises(AssertionError):
            _checked_evidence(candidate, None, accepted_card_declared_max=99999)


def test_pre_identity_eligibility_classes_have_no_episode_evidence_matrix() -> None:
    assert len(EXPECTED_REFUSAL_FIELD_MATRIX) == 14
    pre_identity = [
        row for row in EXPECTED_REFUSAL_FIELD_MATRIX if row["stage"] == "pre_identity"
    ]
    assert len(pre_identity) == 6
    for row in pre_identity:
        assert row["finding_required"] is False
        assert row["finding_integrity_code"] is None
        assert all(
            row[field] == "no_evidence"
            for field in (
                "source_id_rule",
                "requested_count_rule",
                "transition_id_rule",
                "invocation_index_rule",
                "pre_operation_count_rule",
                "pre_source_count_rule",
            )
        )


def test_independent_oracle_exercises_every_assigned_refusal_matrix_row() -> None:
    assigned_rows = [
        row for row in EXPECTED_REFUSAL_FIELD_MATRIX if row["stage"] != "pre_identity"
    ]
    assert len(assigned_rows) == 8
    for row in assigned_rows:
        refusal_class = row["refusal_class"]
        source_id = (
            None
            if row["source_id_rule"] == "null"
            else (
                "unknown.audit"
                if refusal_class == "unknown_source_or_kind"
                else "maintenance.approval.audit"
            )
        )
        requested_count = 0 if row["requested_count_rule"] == "zero" else 1
        if refusal_class == "valid_reservation_above_declared_aggregate":
            requested_count = 4
        elif refusal_class == "valid_reservation_above_declared_source":
            requested_count = 3
        transition_id = (
            None
            if row["transition_id_rule"] == "null"
            else f"transition.refusal.{refusal_class}"
        )
        source_before = (
            None
            if row["pre_source_count_rule"] == "null"
            or refusal_class == "unknown_source_or_kind"
            else 0
        )
        candidate = copy.deepcopy(EXACT_EVIDENCE)
        candidate["operation_instance_id"] = f"opinst.{refusal_class}"
        candidate["records"] = [
            {
                "record_type": "integrity_finding",
                "contract_id": PROTOCOL,
                "integrity_code": row["finding_integrity_code"],
                "refusal_class": refusal_class,
                "source_id": source_id,
                "requested_count": requested_count,
                "operation_committed_before": 0,
                "source_committed_before": source_before,
                "declared_max_events": 3,
                "transition_id": transition_id,
                "invocation_index": None,
            },
            {
                "record_type": "terminal_summary",
                "contract_id": PROTOCOL,
                "declared_max_events": 3,
                "committed_count": 0,
                "result": "ERROR",
                "integrity_code": row["finding_integrity_code"],
            },
        ]
        Draft202012Validator(_load(DOCS / FILES[1])).validate(candidate)
        _checked_evidence(candidate, EXACT_CONTRACT)

        field_attack = copy.deepcopy(candidate)
        if row["requested_count_rule"] == "zero":
            field_attack["records"][0]["requested_count"] = 1
        else:
            field_attack["records"][0]["requested_count"] = 0
        with pytest.raises(AssertionError):
            _checked_evidence(field_attack, EXACT_CONTRACT)


def test_every_lexical_schema_pattern_uses_the_portable_absolute_ending() -> None:
    patterns = [
        pattern
        for name in FILES
        if name.endswith(".schema.json")
        for pattern in _patterns(_load(DOCS / name))
    ]
    assert patterns
    assert all(pattern.endswith(r"(?![\s\S])") for pattern in patterns)
    assert not any("$" in pattern for pattern in patterns)


def test_independent_semantic_oracle_rejects_aggregate_and_cross_record_attacks() -> None:
    mismatch = copy.deepcopy(EXACT_CONTRACT)
    mismatch["declared_max_events"] = 2
    with pytest.raises(AssertionError):
        _checked_contract(mismatch)
    duplicate_source = copy.deepcopy(EXACT_CONTRACT)
    duplicate_source["sources"][1]["source_id"] = duplicate_source["sources"][0][
        "source_id"
    ]
    with pytest.raises(AssertionError):
        _checked_contract(duplicate_source)
    duplicate_intent = copy.deepcopy(EXACT_EVIDENCE)
    duplicate_intent["records"][1]["intent_id"] = duplicate_intent["records"][0][
        "intent_id"
    ]
    with pytest.raises(AssertionError):
        _checked_evidence(duplicate_intent, EXACT_CONTRACT)
    aggregate_mismatch = copy.deepcopy(EXACT_EVIDENCE)
    aggregate_mismatch["records"][-1]["committed_count"] = 2
    with pytest.raises(AssertionError):
        _checked_evidence(aggregate_mismatch, EXACT_CONTRACT)


def test_golden_vectors_are_independently_pinned_and_cover_required_semantics() -> None:
    golden = _load(DOCS / FILES[6])
    assert golden["declaration_vectors"] == {
        "zero_contract": ZERO_CONTRACT,
        "exact_contract": EXACT_CONTRACT,
    }
    assert golden["evidence_vectors"] == {
        "zero_evidence": ZERO_EVIDENCE,
        "exact_evidence": EXACT_EVIDENCE,
        "one_over_evidence": ONE_OVER_EVIDENCE,
        "partial_two_unit_refusal_evidence": PARTIAL_TWO_UNIT_REFUSAL_EVIDENCE,
        "assigned_over_capacity_defect_evidence": ASSIGNED_OVER_CAPACITY_DEFECT_EVIDENCE,
    }
    _assert_operation_deltas_match_execution(golden["operation_deltas"])
    assert {
        case["name"]: {
            key: case[key]
            for key in ("base", "target", "expected_layer", "expected_code", "operations")
        }
        for case in golden["malformed_cases"]
    } == EXPECTED_MALFORMED_CASES
    assert golden["semantic_cases"] == EXPECTED_SEMANTIC_CASES
    expected_hashes = {
        "intent": "de1baf2f26aef2ef576f0481b7c1856651982052f87dc96b19bdb9c78e191fd9",
        "zero_evidence": (
            "fe10d88bd66b54aa26aac603d0c860c30f6d4453f482d58fac626e03135b819b"
        ),
        "exact_evidence": (
            "86eb458b1105828780997152992c5dcdb7677d7d0b37c3821e7417617ecb6deb"
        ),
        "one_over_evidence": (
            "44ef54a2d929a982293072465015fcf948a523e23cf7e83b6dc259621b73df15"
        ),
        "partial_two_unit_refusal_evidence": (
            "527de97cf4727d61b6733c730c8d264296207af6a3eb47ea23fc5b6261c9800f"
        ),
        "assigned_over_capacity_defect_evidence": (
            "4e924d147f4157c45a0bf337f5ab2aa422c5cd4e94a40992cd7823f53357b274"
        ),
    }
    assert {
        name: value["sha256"] for name, value in golden["digest_vectors"].items()
    } == expected_hashes


def test_coherent_golden_repin_cannot_move_independent_payload_or_hash_literals() -> None:
    candidate = copy.deepcopy(_load(DOCS / FILES[6]))
    candidate["semantic_cases"]["cancel_existing_identity"]["operations"] = [
        "delivery_existing_identity"
    ]
    assert candidate["semantic_cases"] != EXPECTED_SEMANTIC_CASES
    candidate = copy.deepcopy(_load(DOCS / FILES[6]))
    malformed = {case["name"]: case for case in candidate["malformed_cases"]}
    replacement_case = copy.deepcopy(malformed["duplicate_intent_id"])
    replacement_case["name"] = "duplicate_batch_ordinal"
    candidate["malformed_cases"] = [
        replacement_case if case["name"] == "duplicate_batch_ordinal" else case
        for case in candidate["malformed_cases"]
    ]
    actual_malformed = {
        case["name"]: {
            key: case[key]
            for key in ("base", "target", "expected_layer", "expected_code", "operations")
        }
        for case in candidate["malformed_cases"]
    }
    assert actual_malformed != EXPECTED_MALFORMED_CASES
    candidate = copy.deepcopy(_load(DOCS / FILES[6]))
    candidate["semantic_cases"]["commit_new_identity"]["operations"] = [
        "delivery_existing_identity"
    ]
    with pytest.raises(AssertionError):
        case = candidate["semantic_cases"]["commit_new_identity"]
        state = _execute_semantic_operations(case["operations"])
        assert (
            state.committed_count - state.initial_committed_count
            == case["committed_delta"]
        )
    for operation, field, replacement in (
        ("commit_new_identity", "committed_delta", 0),
        ("retry_committed_same_identity", "integrity_code", None),
        ("post_finality_reservation", "integrity_code", None),
    ):
        candidate = copy.deepcopy(_load(DOCS / FILES[6]))
        candidate["operation_deltas"][operation][field] = replacement
        with pytest.raises(AssertionError):
            _assert_operation_deltas_match_execution(candidate["operation_deltas"])
    candidate = copy.deepcopy(_load(DOCS / FILES[6]))
    vector = candidate["digest_vectors"]["exact_evidence"]
    vector["content"]["declared_max_events"] = 4
    raw = _canonical({"content": vector["content"], "domain": vector["domain"]})
    vector["canonical_utf8_hex"] = raw.hex()
    vector["byte_count"] = len(raw)
    vector["sha256"] = hashlib.sha256(raw).hexdigest()
    assert (
        vector["sha256"]
        != "86eb458b1105828780997152992c5dcdb7677d7d0b37c3821e7417617ecb6deb"
    )


FROZEN_PINS = {
    "docs/schemas/operation-card-v1.schema.json": (
        16069,
        "30d0d860125e28f80176ae65eb7c4283a800f15b8123c6134d25153be03c680a",
    ),
    "docs/schemas/operation-card-v2.schema.json": (
        16481,
        "f107e4446312d2a872209e1f12444f71251f95558f060f95ccfbe27550a15156",
    ),
    "docs/schemas/identity-field-census-v1.schema.json": (
        2249,
        "ab301b186bc6ff52abac8855cbff781713fb54dcf01a77d1409eddcda64b7c33",
    ),
    "docs/schemas/identity-field-census-v1.registry.json": (
        45641,
        "056b7acf26a0fcdc01b04abe7552b3f0c35a0ad70935434d01d94ef70ef1ef61",
    ),
    "docs/schemas/identity-field-census-v2.schema.json": (
        2329,
        "f06fbbba730cdcbe73c8e8241ec7069600eb13f761772b2bea9e6584135d925c",
    ),
    "docs/schemas/identity-field-census-v2.registry.json": (
        45938,
        "278d324cb721693fe6f7cd3b59c8357d3c71c3a3ed8c5ae82e6b4bd256122046",
    ),
    "docs/schemas/episode-outcome-v2.schema.json": (
        4902,
        "09637aa8a45f22e8d9e7e409ee2529335befd3a96e98ed13ee23efcce4187d67",
    ),
    "docs/schemas/executed-runtime-evidence-v1.schema.json": (
        1623,
        "3e11d79f28687876595cc7c0028d65f038484c9adae2ccdc91818925d0b324cb",
    ),
    "docs/schemas/runtime-identity-adapter-registry-v1.schema.json": (
        13530,
        "bedbe34c063a65207751543397e34ddaa82caeb16c2adb9b65f6a8316a793de6",
    ),
    "docs/schemas/runtime-identity-adapter-registry-v1.json": (
        23947,
        "4e242d3caf82a786065e20e267c5373d5a87c6f408fe678a91a7e910aab4b421",
    ),
    "docs/schemas/runtime-identity-adapter-census-v1.schema.json": (
        8001,
        "d27993a85908134c7b3cf4307777015265b3e02a55ed1e6f00fb411409d593f4",
    ),
    "docs/schemas/runtime-identity-adapter-census-v1.json": (
        40477,
        "9090d52dae30f3550670feceaad4a5e709d28671319f114be486b4b49b2a8a32",
    ),
    "tests/fixtures/b3/artifact8-evidence-freeze-v1.json": (
        2777,
        "6ec8409ca55dc4fbb0f5d9078f25004429edd6525209d1e8080b163b36a2b8c9",
    ),
    "tests/fixtures/b3/artifact8-deterministic-reference-v1.json": (
        220582,
        "45f2f3bec377230a0fc464c5ea1b64798a053beeb588c3ec4bdb477eaf7cfab5",
    ),
    "tests/fixtures/b3/artifact8-fake-openai-reference-v1.json": (
        228801,
        "115bd5c058f4b4057cf7b4cd18d7cf0b367c0234b4367fbac92396aa26a24b3f",
    ),
    "tests/fixtures/b3/artifact8-fake-openai-reference-v1.execution-ledger-v2.ndjson": (
        88863,
        "29fc1c33d601529151b9340bc253992c3b55c28ba4c55849f070df13ead7afdb",
    ),
}


def _assert_frozen_resource(relative: str, raw: bytes) -> None:
    expected = FROZEN_PINS[relative]
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == expected


_EXPECTED_OPERATEBENCH_EXPORTS = (
    "CARD_SCHEMA_VERSION",
    "OPERATEBENCH_VERSION",
    "CardValidationError",
    "load_card_json",
    "load_field_census",
    "recompute_card_digest",
    "validate_card",
    "validate_field_census",
)
_EXPECTED_CARDS_EXPORTS = (
    "CARD_SCHEMAS",
    "CARD_SCHEMA_VERSION",
    "OPERATION_CARD_SCHEMA",
    "OPERATION_CARD_V2_SCHEMA",
    "SEMANTIC_SCENARIO_CARD_SCHEMA",
    "SEMANTIC_SCENARIO_CARD_V2_SCHEMA",
    "SUPPORTED_CARD_SCHEMA_VERSIONS",
    "VARIANT_CARD_SCHEMA",
    "VARIANT_CARD_V2_SCHEMA",
    "CardValidationError",
    "card_to_json",
    "load_card_json",
    "load_card_json_v2",
    "load_card_registry",
    "load_card_registry_v2",
    "load_field_census",
    "load_field_census_v2",
    "load_schema",
    "recompute_card_digest",
    "validate_card",
    "validate_field_census",
    "validate_field_census_v2",
)


# These history-independent tests own finite compatibility properties of the
# checked-out tree. Whether the candidate PR changed runtime, product, or CI code
# belongs to exact-diff and SHA-bound review; a shipped unit test cannot prove that
# historical scope statement.
def test_frozen_resource_manifest_is_history_independent_and_mutation_sensitive() -> None:
    for relative, expected in FROZEN_PINS.items():
        raw = (ROOT / relative).read_bytes()
        _assert_frozen_resource(relative, raw)
        mutated = bytes([raw[0] ^ 1]) + raw[1:]
        with pytest.raises(AssertionError):
            _assert_frozen_resource(relative, mutated)
        assert expected == (len(raw), hashlib.sha256(raw).hexdigest())
    cards = ROOT / "src" / "operatebench" / "resources" / "cards"
    for relative in FROZEN_PINS:
        if not relative.startswith("docs/schemas/"):
            continue
        name = Path(relative).name
        packaged_root = (
            cards if name.startswith(("operation-card", "identity-field")) else PACKAGED
        )
        assert (DOCS / name).read_bytes() == (packaged_root / name).read_bytes()


def test_contract_resources_are_docs_and_packaged_data_only() -> None:
    resources = {
        path.relative_to(ROOT).as_posix()
        for directory in (DOCS, PACKAGED)
        for path in directory.iterdir()
        if path.is_file() and "domain-generated-audit-budget" in path.name
    }
    assert resources == {
        f"{root}/{name}"
        for root in ("docs/schemas", "src/operatebench/resources/identity")
        for name in FILES
    }


def test_exact_public_exports_and_console_entrypoints_remain_compatible() -> None:
    import operatebench as package
    import operatebench.cards as cards

    assert tuple(package.__all__) == _EXPECTED_OPERATEBENCH_EXPORTS
    assert tuple(cards.__all__) == _EXPECTED_CARDS_EXPORTS

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]
    assert scripts == {
        "operatebench": "operatebench.cli:main",
        "boundarybench": "boundarybench.cli:main",
    }


def test_v1_capacity_is_exactly_representable_and_one_over_refuses() -> None:
    validator = Draft202012Validator(_load(DOCS / FILES[0]))
    evidence_validator = Draft202012Validator(_load(DOCS / FILES[1]))
    at_bound = copy.deepcopy(EXACT_CONTRACT)
    at_bound["declared_max_events"] = 99998
    at_bound["sources"] = [
        {
            "source_id": "capacity.audit",
            "counted_kind": "scheduled_audit_event_intent",
            "max_accepted_occurrences": 99998,
            "max_events_per_transition": 1,
        }
    ]
    validator.validate(at_bound)
    _checked_contract(at_bound)
    over = copy.deepcopy(at_bound)
    over["declared_max_events"] = 99999
    over["sources"][0]["max_accepted_occurrences"] = 99999
    with pytest.raises(ValidationError):
        validator.validate(over)
    records = [
        {
            "record_type": "committed_intent",
            "contract_id": PROTOCOL,
            "intent_digest_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "intent_id": f"capacity.audit.{index}",
            "source_id": "capacity.audit",
            "counted_kind": "scheduled_audit_event_intent",
            "transition_id": f"transition.{index}",
            "batch_ordinal": 0,
            "operation_committed_count": index,
        }
        for index in range(1, V1_EVENT_CAPACITY + 1)
    ]
    success = {
        "schema": "operatebench.domain_generated_audit_budget_evidence.v1",
        "schema_version": 1,
        "contract_id": PROTOCOL,
        "operation_core_content_digest_sha256": "2" * 64,
        "operation_instance_id": "opinst_capacity",
        "declared_max_events": V1_EVENT_CAPACITY,
        "records": [
            *records,
            {
                "record_type": "terminal_summary",
                "contract_id": PROTOCOL,
                "declared_max_events": V1_EVENT_CAPACITY,
                "committed_count": V1_EVENT_CAPACITY,
                "result": "SUCCESS",
                "integrity_code": None,
            },
        ],
    }
    evidence_validator.validate(success)
    _checked_evidence(success, at_bound)
    refusal = copy.deepcopy(success)
    refusal["records"][-1:-1] = [
        {
            "record_type": "integrity_finding",
            "contract_id": PROTOCOL,
            "integrity_code": EXPECTED_CODES[0],
            "refusal_class": "valid_reservation_above_declared_aggregate",
            "source_id": "capacity.audit",
            "requested_count": 1,
            "operation_committed_before": V1_EVENT_CAPACITY,
            "source_committed_before": V1_EVENT_CAPACITY,
            "declared_max_events": V1_EVENT_CAPACITY,
            "transition_id": "transition.over",
            "invocation_index": None,
        }
    ]
    refusal["records"][-1]["result"] = "ERROR"
    refusal["records"][-1]["integrity_code"] = EXPECTED_CODES[0]
    assert len(refusal["records"]) == EVIDENCE_RECORD_CAPACITY
    evidence_validator.validate(refusal)
    _checked_evidence(refusal, at_bound)


def test_cross_record_oracle_rejects_all_reviewer_attacks() -> None:
    attacks = []
    wrong_kind = copy.deepcopy(EXACT_EVIDENCE)
    wrong_kind["records"][0]["counted_kind"] = "immediate_audit_event_intent"
    attacks.append(wrong_kind)
    duplicate_ordinal = copy.deepcopy(EXACT_EVIDENCE)
    duplicate_ordinal["records"][1]["batch_ordinal"] = 0
    attacks.append(duplicate_ordinal)
    finding_before_commit = copy.deepcopy(ONE_OVER_EVIDENCE)
    finding_before_commit["records"].insert(0, finding_before_commit["records"].pop(-2))
    attacks.append(finding_before_commit)
    zero_request = copy.deepcopy(ONE_OVER_EVIDENCE)
    zero_request["records"][-2]["requested_count"] = 0
    attacks.append(zero_request)
    source_over = copy.deepcopy(EXACT_EVIDENCE)
    source_over["records"][2]["source_id"] = "maintenance.approval.audit"
    source_over["records"][2]["counted_kind"] = "scheduled_audit_event_intent"
    attacks.append(source_over)
    finding_limit = copy.deepcopy(ONE_OVER_EVIDENCE)
    finding_limit["records"][-2]["declared_max_events"] = 2
    attacks.append(finding_limit)
    for attack in attacks:
        with pytest.raises(AssertionError):
            _checked_evidence(attack, EXACT_CONTRACT)


def test_registry_and_census_independently_freeze_compatibility_and_ownership() -> None:
    registry = _load(DOCS / FILES[3])
    assert registry["card_v2_schema_change"] is False
    assert registry["runtime_identity_adapter_v1_change"] is False

    # Registration is owned only at this exact, byte-frozen adapter boundary. The
    # frozen v1 adapter registry contains no registration for this specified protocol.
    adapter_raw = (DOCS / "runtime-identity-adapter-registry-v1.json").read_bytes()
    assert (len(adapter_raw), hashlib.sha256(adapter_raw).hexdigest()) == (
        23947,
        "4e242d3caf82a786065e20e267c5373d5a87c6f408fe678a91a7e910aab4b421",
    )
    adapter_registry = json.loads(adapter_raw)
    assert (
        adapter_registry["schema"] == "operatebench.runtime_identity_adapter_registry.v1"
    )
    assert adapter_registry["schema_version"] == 1
    assert adapter_registry["current_complete_lanes"] == []
    assert PROTOCOL.encode() not in adapter_raw

    assert tuple(registry["excluded_kinds"]) == EXPECTED_EXCLUDED_KINDS
    assert (
        tuple(
            (row["refusal_class"], row["integrity_code"])
            for row in registry["refusal_mapping"]
        )
        == EXPECTED_REFUSAL_MAPPING
    )
    assert EXPECTED_CENSUS[-1] == (
        "/no_time/domain_generated_audit_term",
        "operation_core",
        "future_no_time_verifier",
        "specified_only_consumer",
    )


def test_digest_vectors_use_exact_domain_wrappers_and_closed_projections() -> None:
    golden = _load(DOCS / FILES[6])
    schema = _load(DOCS / FILES[1])
    assert golden["digest_domains"] == {
        "intent": "operatebench.digest.domain-generated-audit-intent.v1",
        "evidence": "operatebench.digest.domain-generated-audit-budget-evidence.v1",
    }
    for vector in golden["digest_vectors"].values():
        wrapped = {"content": vector["content"], "domain": vector["domain"]}
        raw = _canonical(wrapped)
        assert vector["canonical_utf8_hex"] == raw.hex()
        assert vector["byte_count"] == len(raw)
        assert vector["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (
            hashlib.sha256(
                vector["domain"].encode() + b"\0" + _canonical(vector["content"])
            ).hexdigest()
            != vector["sha256"]
        )
    intent_wrapper = {
        "domain": golden["digest_vectors"]["intent"]["domain"],
        "content": golden["digest_vectors"]["intent"]["content"],
    }
    intent_wrapper_schema = {
        "$schema": schema["$schema"],
        "$id": schema["$id"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/intentDigestWrapper",
    }
    Draft202012Validator(intent_wrapper_schema).validate(intent_wrapper)
    evidence_wrapper = {
        "domain": golden["digest_vectors"]["exact_evidence"]["domain"],
        "content": golden["digest_vectors"]["exact_evidence"]["content"],
    }
    # Resolve the local $defs through the full schema, not an online resolver.
    wrapped_schema = copy.deepcopy(schema)
    wrapped_schema["$ref"] = "#/$defs/evidenceDigestWrapper"
    for key in tuple(wrapped_schema):
        if key not in {"$schema", "$id", "$defs", "$ref"}:
            del wrapped_schema[key]
    Draft202012Validator(wrapped_schema).validate(evidence_wrapper)
    assert tuple(sorted(golden["digest_vectors"]["intent"]["content"])) == (
        "batch_ordinal",
        "contract_id",
        "intent_id",
        "operation_core_content_digest_sha256",
        "operation_instance_id",
        "source_id",
        "transition_id",
    )
    assert tuple(sorted(evidence_wrapper["content"])) == (
        "contract_id",
        "declared_max_events",
        "operation_core_content_digest_sha256",
        "operation_instance_id",
        "records",
    )


def test_refusal_precedence_and_terminal_matrix_are_exact() -> None:
    registry = _load(DOCS / FILES[3])
    assert registry["identity_accounting"] == [
        {
            "identity_state": "before_identity",
            "primary_result": None,
            "finding_required": False,
            "terminal_summary_required": False,
            "denominator_claim": False,
            "disposition": "configuration_refusal",
        },
        {
            "identity_state": "assigned",
            "primary_result": "ERROR",
            "finding_required": True,
            "terminal_summary_required": True,
            "denominator_claim": True,
            "disposition": "integrity_refusal",
        },
    ]
    for refusal_class, code in (
        ("post_finality_attempt", EXPECTED_CODES[1]),
        ("malformed_runtime_counter", EXPECTED_CODES[2]),
    ):
        candidate = copy.deepcopy(ZERO_EVIDENCE)
        candidate["operation_instance_id"] = "opinst_refusal"
        candidate["records"] = [
            {
                "record_type": "integrity_finding",
                "contract_id": PROTOCOL,
                "integrity_code": code,
                "refusal_class": refusal_class,
                "source_id": None,
                "requested_count": 0,
                "operation_committed_before": 0,
                "source_committed_before": None,
                "declared_max_events": 0,
                "transition_id": None,
                "invocation_index": None,
            },
            {
                "record_type": "terminal_summary",
                "contract_id": PROTOCOL,
                "declared_max_events": 0,
                "committed_count": 0,
                "result": "ERROR",
                "integrity_code": code,
            },
        ]
        Draft202012Validator(_load(DOCS / FILES[1])).validate(candidate)
        _checked_evidence(candidate, ZERO_CONTRACT)
    wrong_precedence = copy.deepcopy(ONE_OVER_EVIDENCE)
    wrong_precedence["records"][-2]["refusal_class"] = "malformed_runtime_counter"
    with pytest.raises(AssertionError):
        _checked_evidence(wrong_precedence, EXACT_CONTRACT)
    error_without_finding = copy.deepcopy(ZERO_EVIDENCE)
    error_without_finding["records"][-1]["result"] = "ERROR"
    error_without_finding["records"][-1]["integrity_code"] = EXPECTED_CODES[1]
    with pytest.raises(AssertionError):
        _checked_evidence(error_without_finding, ZERO_CONTRACT)


def test_semantic_lifecycle_rejects_a_duplicate_reservation_atomically() -> None:
    state = _execute_semantic_operations(["retry_committed_same_identity"])
    assert state.attempted_reservations == [state.existing_identity]
    assert state.known_identities == {state.existing_identity}
    assert state.committed_identities == {state.existing_identity}
    assert state.committed_count == 1
    assert state.integrity_code == EXPECTED_CODES[1]


def test_semantic_lifecycle_is_fail_closed_and_rejects_unknown_operations() -> None:
    state = _execute_semantic_operations(
        ["retry_committed_same_identity", "commit_new_identity"]
    )
    assert state.committed_count == state.initial_committed_count
    assert state.known_identities == set(state.initial_known_identities)
    assert state.trace == list(
        EXPECTED_TRANSITION_TRACES["retry_committed_same_identity"]
    )
    with pytest.raises(AssertionError, match="unknown semantic operation"):
        _execute_semantic_operations(["unknown_operation"])


def test_all_golden_mutations_and_semantic_operations_are_executable() -> None:
    golden = _load(DOCS / FILES[6])
    _execute_all_golden_mutations(golden)
    _assert_operation_deltas_match_execution(golden["operation_deltas"])
    assert set(golden["semantic_cases"]) == set(EXPECTED_SEMANTIC_OPERATIONS)
    for name, case in golden["semantic_cases"].items():
        assert tuple(case["operations"]) == EXPECTED_SEMANTIC_OPERATIONS[name]
        state = _execute_semantic_operations(case["operations"])

        assert state.initial_known_identities == frozenset({"existing-identity"})
        assert state.initial_committed_identities == frozenset({"existing-identity"})
        assert state.initial_committed_count == 1
        assert tuple(state.trace) == EXPECTED_TRANSITION_TRACES[name]
        assert (
            state.statuses[state.existing_identity]
            == EXPECTED_FINAL_EXISTING_STATUS[name]
        )

        expected_identities = {"existing-identity"}
        if name in SUCCESSFUL_NEW_IDENTITY_CASES:
            expected_identities.add("new-identity-1")
        assert state.known_identities == expected_identities
        assert state.committed_identities == expected_identities
        assert state.committed_count == len(expected_identities)
        assert state.finality is (name == "post_finality_reservation")

        expected_attempts: list[str] = []
        if name in SUCCESSFUL_NEW_IDENTITY_CASES or name == "post_finality_reservation":
            expected_attempts = ["new-identity-1"]
        elif name == "retry_committed_same_identity":
            expected_attempts = ["existing-identity"]
        assert state.attempted_reservations == expected_attempts

        # Golden outcomes assert over execution; they never select a transition.
        assert (
            state.committed_count - state.initial_committed_count
            == case["committed_delta"]
        )
        assert state.integrity_code == case["expected_integrity_code"]
